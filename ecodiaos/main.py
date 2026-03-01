"""
EcodiaOS — Application Entry Point

FastAPI application with the startup sequence defined in the
Infrastructure Architecture specification.

`docker compose up` → uvicorn ecodiaos.main:app
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager, suppress
from typing import Any

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env file before any configuration is loaded
load_dotenv()

from ecodiaos.clients.embedding import create_embedding_client
from ecodiaos.core.hotreload import NeuroplasticityBus
from ecodiaos.clients.llm import create_llm_provider
from ecodiaos.clients.neo4j import Neo4jClient
from ecodiaos.clients.optimized_llm import OptimizedLLMProvider
from ecodiaos.clients.prompt_cache import PromptCache
from ecodiaos.clients.redis import RedisClient
from ecodiaos.clients.timescaledb import TimescaleDBClient
from ecodiaos.clients.token_budget import TokenBudget
from ecodiaos.clients.wallet import WalletClient
from ecodiaos.config import load_config, load_seed
from ecodiaos.systems.atune.service import AtuneConfig, AtuneService
from ecodiaos.systems.atune.types import InputChannel, RawInput
from ecodiaos.systems.axon.service import AxonService
from ecodiaos.systems.equor.service import EquorService
from ecodiaos.systems.evo.service import EvoService
from ecodiaos.systems.federation.service import FederationService
from ecodiaos.systems.memory.service import MemoryService
from ecodiaos.systems.nova.service import NovaService
from ecodiaos.systems.oneiros.service import OneirosService
from ecodiaos.systems.simula.service import SimulaService
from ecodiaos.systems.soma.service import SomaService
from ecodiaos.systems.synapse.service import SynapseService
from ecodiaos.systems.thread.service import ThreadService
from ecodiaos.systems.thymos.service import ThymosService
from ecodiaos.systems.voxis.service import VoxisService
from ecodiaos.systems.voxis.types import ExpressionTrigger
from ecodiaos.telemetry.llm_metrics import LLMMetricsCollector
from ecodiaos.telemetry.logging import setup_logging, subscribe_logs, unsubscribe_logs
from ecodiaos.telemetry.metrics import MetricCollector

logger = structlog.get_logger()
_chat_logger = structlog.get_logger("ecodiaos.chat")


# ─── Application State ───────────────────────────────────────────
# These are set during startup and accessible via app.state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown sequence.
    Follows the Infrastructure Architecture spec section 3.2.
    """
    # ── 1. Load configuration ─────────────────────────────────
    config_path = os.environ.get("ECODIAOS_CONFIG_PATH", "config/default.yaml")
    config = load_config(config_path)
    app.state.config = config

    # ── 2. Set up logging ─────────────────────────────────────
    setup_logging(config.logging, instance_id=config.instance_id)
    logger.info(
        "ecodiaos_starting",
        instance_id=config.instance_id,
        config_path=config_path,
    )

    # ── 3. Connect to data stores ─────────────────────────────
    neo4j_client = Neo4jClient(config.neo4j)
    await neo4j_client.connect()
    app.state.neo4j = neo4j_client

    tsdb_client = TimescaleDBClient(config.timescaledb)
    await tsdb_client.connect()
    app.state.tsdb = tsdb_client

    redis_client = RedisClient(config.redis)
    await redis_client.connect()
    app.state.redis = redis_client

    # ── 3b. Start the Neuroplasticity Bus ─────────────────────
    # Single Redis subscriber for the entire process. All cognitive systems
    # register their hot-reload handlers here; changed files are imported
    # once and dispatched to every matching handler.
    neuroplasticity_bus = NeuroplasticityBus(redis_client=redis_client)
    neuroplasticity_bus.start()
    app.state.neuroplasticity_bus = neuroplasticity_bus

    # ── 4. Initialize LLM and embedding clients ───────────────
    # Create base LLM provider
    raw_llm = create_llm_provider(config.llm)

    # Create optimization infrastructure
    token_budget = TokenBudget(
        max_tokens_per_hour=config.llm.budget.max_tokens_per_hour,
        max_calls_per_hour=config.llm.budget.max_calls_per_hour,
        hard_limit=config.llm.budget.hard_limit,
    )
    app.state.token_budget = token_budget

    llm_metrics = LLMMetricsCollector()
    app.state.llm_metrics = llm_metrics

    # Create prompt cache (requires Redis, graceful degradation if unavailable)
    prompt_cache: PromptCache | None = None
    try:
        prompt_cache = PromptCache(redis_client=redis_client._client, prefix="eos:llmcache")
        logger.info("prompt_cache_initialized")
    except Exception as exc:
        logger.warning("prompt_cache_init_failed", error=str(exc))

    # Wrap the LLM provider with optimization layer
    llm_client = OptimizedLLMProvider(
        inner=raw_llm,
        cache=prompt_cache,
        budget=token_budget,
        metrics=llm_metrics,
    )
    app.state.llm = llm_client
    app.state.raw_llm = raw_llm  # Keep reference for systems that need unwrapped access

    logger.info(
        "llm_optimization_active",
        budget_tokens_per_hour=config.llm.budget.max_tokens_per_hour,
        budget_calls_per_hour=config.llm.budget.max_calls_per_hour,
        cache_enabled=prompt_cache is not None,
    )

    embedding_client = create_embedding_client(config.embedding)
    app.state.embedding = embedding_client

    # ── 5. Initialize Memory service ──────────────────────────
    memory = MemoryService(neo4j_client, embedding_client)
    await memory.initialize()
    app.state.memory = memory

    # ── 6. Initialize Equor (Constitution & Ethics) ───────────
    governance_config = _resolve_governance_config(config)
    equor = EquorService(
        neo4j=neo4j_client,
        llm=llm_client,
        config=config.equor,
        governance_config=governance_config,
        neuroplasticity_bus=neuroplasticity_bus,
    )
    await equor.initialize()
    app.state.equor = equor

    # ── 7. Initialize Atune (Perception & Attention) ──────────
    atune_config = AtuneConfig(
        ignition_threshold=getattr(config, "atune_ignition_threshold", 0.3),
        workspace_buffer_size=getattr(config, "atune_workspace_buffer_size", 32),
        spontaneous_recall_base_probability=getattr(
            config, "atune_spontaneous_recall_prob", 0.02
        ),
        max_percept_queue_size=getattr(config, "atune_max_percept_queue", 100),
    )
    workspace_memory = _MemoryWorkspaceAdapter(memory)
    atune = AtuneService(
        embed_fn=embedding_client.embed,
        memory_client=workspace_memory,
        llm_client=llm_client,  # type: ignore[arg-type]
        belief_state=None,  # Wired in step 9c after Nova initializes
        config=atune_config,
        neuroplasticity_bus=neuroplasticity_bus,
    )
    await atune.startup()
    app.state.atune = atune

    # ── 8. Initialize Voxis (Expression & Voice) ──────────────
    voxis = VoxisService(
        memory=memory,
        redis=redis_client,
        llm=llm_client,
        config=config.voxis,
        neuroplasticity_bus=neuroplasticity_bus,
    )
    await voxis.initialize()
    app.state.voxis = voxis

    # ── 9. Initialize Nova (Decision & Planning) ─────────────
    nova = NovaService(
        memory=memory,
        equor=equor,
        voxis=voxis,
        llm=llm_client,
        config=config.nova,
        neuroplasticity_bus=neuroplasticity_bus,
    )
    await nova.initialize()
    nova.set_embed_fn(embedding_client.embed)
    app.state.nova = nova
    # Subscribe Nova to Atune's workspace broadcasts
    atune.subscribe(nova)
    # Wire Nova's active goal summaries into Atune for salience computation
    atune.set_active_goals(nova.active_goal_summaries)
    # Wire live goal sync: Nova pushes updated goals to Atune after each broadcast
    nova.set_goal_sync_callback(atune.set_active_goals)

    # ── 9b. Initialize Axon (Action Execution) ─────────────────
    axon = AxonService(
        config=config.axon,
        memory=memory,
        voxis=voxis,
        neuroplasticity_bus=neuroplasticity_bus,
        redis_client=redis_client,
        wallet=None,  # Will be set to wallet_client after it's initialized in step 15b
        instance_id=config.instance_id,
    )
    await axon.initialize()
    axon.set_nova(nova)
    axon.set_atune(atune)  # Loop 4: execution outcomes → workspace percepts
    app.state.axon = axon

    # Wire Axon into Nova's intent router so approved intents can execute
    nova.set_axon(axon)

    # ── 9c. Wire Nova's belief state into Atune (predictive processing loop) ──
    atune.set_belief_state(nova.belief_state_reader)

    # ── 9d. Wire Memory into Atune for entity extraction storage ──────
    atune.set_memory_service(memory)

    # ── 9e. Wire Voxis expression feedback loop ─────────────────────
    # ExpressionFeedback is now generated after every expression and
    # sent to registered listeners — closing the perception-action loop.
    def _on_expression_feedback(feedback):
        """Distribute expression feedback to Atune and Nova."""
        # Nudge Atune's affect based on expression delta
        if feedback.affect_delta != 0.0:
            atune.nudge_valence(feedback.affect_delta * 0.1)

        # Feed back to Nova if this expression was triggered by an intent
        # This closes the Nova→Voxis→Nova loop (was fire-and-forget before)
        if feedback.trigger in ("nova_respond", "nova_inform", "nova_request",
                                "nova_mediate", "nova_celebrate", "nova_warn"):
            import asyncio as _aio

            from ecodiaos.systems.nova.types import IntentOutcome
            intent_id = getattr(feedback, "expression_id", "")
            outcome = IntentOutcome(
                intent_id=intent_id,
                success=True,
                episode_id=intent_id,
                new_observations=[f"Expression delivered: {feedback.content_summary[:80]}"],
            )
            try:
                loop = _aio.get_running_loop()
                loop.create_task(nova.process_outcome(outcome))
            except RuntimeError:
                pass  # No running loop (shouldn't happen in production)

    voxis.register_feedback_callback(_on_expression_feedback)

    # ── 10. Initialize telemetry ──────────────────────────────
    metrics = MetricCollector(tsdb_client)
    await metrics.start_writer()
    app.state.metrics = metrics

    # ── 10. Check for existing instance or birth new one ──────
    instance = await memory.get_self()
    if instance is None:
        seed_path = os.environ.get("ECODIAOS_SEED_PATH", "config/seeds/example_seed.yaml")
        try:
            seed = load_seed(seed_path)
            birth_result = await memory.birth(seed, config.instance_id)
            logger.info("instance_born", **birth_result)
            # Re-seed invariants after birth (constitution now exists)
            await equor.initialize()
            # Seed Atune identity cache from the newly born instance
            new_instance = await memory.get_self()
            if new_instance is not None:
                await _seed_atune_cache(atune, embedding_client, new_instance)
        except FileNotFoundError:
            logger.warning(
                "no_seed_found",
                seed_path=seed_path,
                message="Instance not born. Provide a seed config to create one.",
            )
    else:
        logger.info(
            "instance_loaded",
            name=instance.name,
            instance_id=instance.instance_id,
            cycle_count=instance.cycle_count,
            episodes=instance.total_episodes,
            entities=instance.total_entities,
        )
        # Seed Atune identity cache from existing instance
        await _seed_atune_cache(atune, embedding_client, instance)

        # Migration: backfill personality_json on existing instances that lack it
        if not instance.personality_json and instance.personality_vector:
            import json as _json
            _pkeys = [
                "warmth", "directness", "verbosity", "formality",
                "curiosity_expression", "humour", "empathy_expression",
                "confidence_display", "metaphor_use",
            ]
            pdict = dict(zip(_pkeys, instance.personality_vector[:9], strict=False))
            await neo4j_client.execute_write(
                "MATCH (s:Self {instance_id: $iid}) SET s.personality_json = $pj",
                {"iid": instance.instance_id, "pj": _json.dumps(pdict)},
            )
            logger.info("personality_json_backfilled", personality=pdict)

    # ── 10b. Seed initial goals into Nova ─────────────────────────
    # Break the bootstrap deadlock: Nova needs goals to deliberate, but goals
    # are only created from broadcasts, which require goals for salience.
    # Seed from the birth config (new instances) or create defaults (existing).
    if nova._goal_manager is not None and len(nova._goal_manager.active_goals) == 0:
        from ecodiaos.primitives.common import DriveAlignmentVector, new_id
        from ecodiaos.systems.nova.types import Goal, GoalSource, GoalStatus

        # Try loading goals from seed config
        seed_path = os.environ.get("ECODIAOS_SEED_PATH", "config/seeds/example_seed.yaml")
        seed_goals: list[dict[str, Any]] = []
        try:
            seed = load_seed(seed_path)
            seed_goals = [
                {
                    "description": g.description,
                    "source": g.source,
                    "priority": g.priority,
                    "importance": g.importance,
                    "drive_alignment": g.drive_alignment,
                }
                for g in seed.community.initial_goals
            ]
        except Exception:
            pass  # Seed not found or has no goals — use defaults below

        if not seed_goals:
            # Fallback defaults — every EOS should have at least these
            seed_goals = [
                {
                    "description": "Learn about my community — understand who I serve and what they need",
                    "source": "epistemic",
                    "priority": 0.6,
                    "importance": 0.7,
                    "drive_alignment": {"coherence": 0.3, "care": 0.8, "growth": 0.7, "honesty": 0.2},
                },
                {
                    "description": "Develop self-understanding — explore my capabilities and how my drives shape my behaviour",
                    "source": "self_generated",
                    "priority": 0.5,
                    "importance": 0.6,
                    "drive_alignment": {"coherence": 0.9, "care": 0.1, "growth": 0.8, "honesty": 0.5},
                },
            ]

        _source_map = {
            "user_request": GoalSource.USER_REQUEST,
            "self_generated": GoalSource.SELF_GENERATED,
            "governance": GoalSource.GOVERNANCE,
            "care_response": GoalSource.CARE_RESPONSE,
            "maintenance": GoalSource.MAINTENANCE,
            "epistemic": GoalSource.EPISTEMIC,
        }

        for gdata in seed_goals:
            da = gdata.get("drive_alignment", {})
            goal = Goal(
                id=new_id(),
                description=gdata["description"],
                source=_source_map.get(gdata.get("source", "self_generated"), GoalSource.SELF_GENERATED),
                priority=gdata.get("priority", 0.5),
                importance=gdata.get("importance", 0.5),
                drive_alignment=DriveAlignmentVector(
                    coherence=da.get("coherence", 0.0),
                    care=da.get("care", 0.0),
                    growth=da.get("growth", 0.0),
                    honesty=da.get("honesty", 0.0),
                ),
                status=GoalStatus.ACTIVE,
            )
            await nova.add_goal(goal)
            logger.info("initial_goal_seeded", description=goal.description[:60])

        # Refresh Atune's goal summaries now that Nova has goals
        atune.set_active_goals(nova.active_goal_summaries)
        logger.info("initial_goals_seeded", count=len(seed_goals))

    # ── 11. Initialize Evo (Learning & Hypothesis) ───────────
    evo = EvoService(
        config=config.evo,
        llm=llm_client,
        memory=memory,
        instance_name=config.instance_id,
        neuroplasticity_bus=neuroplasticity_bus,
    )
    await evo.initialize()
    evo.schedule_consolidation_loop()
    evo.set_atune(atune)  # Push learned head weights back to meta-attention
    app.state.evo = evo
    # Subscribe Evo to Atune workspace broadcasts (background learning)
    atune.subscribe(evo)
    # Loop 2: supported hypotheses → epistemic exploration goals in Nova
    evo.set_nova(nova)
    # Loop 7: personality learning from expression outcomes
    evo.set_voxis(voxis)
    # Loop 6: workspace broadcasts → spontaneous expression
    atune.subscribe(voxis)
    # Loop 8: constitutional vetoes → learning episodes
    equor.set_evo(evo)

    # ── 11b. Initialize Thread (Narrative Identity) ────────────
    thread = ThreadService(
        memory=memory,
        instance_name=config.instance_id,
        neuroplasticity_bus=neuroplasticity_bus,
    )
    await thread.initialize()
    # Wire cross-system references for 29D fingerprint aggregation
    thread.set_voxis(voxis)
    thread.set_equor(equor)
    thread.set_atune(atune)
    thread.set_evo(evo)
    thread.set_nova(nova)
    # Wire Thread into Voxis for identity context injection (P1.6 + P2.9)
    voxis.set_thread(thread)
    app.state.thread = thread

    # ── 12. Initialize Simula (Self-Evolution) ────────────────
    # SIMULA_MODE=proxy offloads the heavy pipeline to an out-of-process
    # worker via Redis Streams. The proxy is non-blocking: callers still
    # await process_proposal() but the Synapse 150ms clock is never stalled.
    _simula_mode = os.getenv("SIMULA_MODE", "local")
    if _simula_mode == "proxy":
        from ecodiaos.systems.simula.proxy import InspectorProxy, SimulaProxy

        simula = SimulaProxy(
            redis=redis_client,
            timeout_s=config.simula.pipeline_timeout_s,
            neo4j=neo4j_client,
        )
        await simula.initialize()

        inspector_proxy = InspectorProxy(
            redis=redis_client,
            timeout_s=config.simula.pipeline_timeout_s,
        )
        await inspector_proxy.initialize()

        app.state.inspector_proxy = inspector_proxy
        logger.info("simula_mode", mode="proxy", inspector="proxy")
    else:
        from pathlib import Path as _Path

        simula = SimulaService(
            config=config.simula,
            llm=llm_client,
            neo4j=neo4j_client,
            memory=memory,
            codebase_root=_Path(config.simula.codebase_root).resolve(),
            instance_name=config.instance_id,
            tsdb=tsdb_client,
            redis=redis_client,
        )
        await simula.initialize()
        app.state.inspector_proxy = None
        logger.info("simula_mode", mode="local")
    app.state.simula = simula

    # ── 13. Initialize Synapse — The Heartbeat ────────────────────
    synapse = SynapseService(
        atune=atune,
        config=config.synapse,
        redis=redis_client,
        metrics=metrics,
        neuroplasticity_bus=neuroplasticity_bus,
    )
    await synapse.initialize()

    # Register all cognitive systems for health monitoring
    synapse.register_system(memory)
    synapse.register_system(equor)
    synapse.register_system(voxis)
    synapse.register_system(nova)
    synapse.register_system(axon)
    synapse.register_system(evo)
    synapse.register_system(thread)

    # Start the heartbeat and health monitoring
    await synapse.start_clock()
    await synapse.start_health_monitor()
    app.state.synapse = synapse

    # Wire metabolic cost tracking: every real LLM call now reports its
    # token usage to the MetabolicTracker.  The callback is injected here
    # (not at llm_client construction) to avoid a circular dependency —
    # llm_client is built in step 4, Synapse is initialized in step 13.
    llm_client.set_metabolic_callback(synapse.metabolism.log_usage)
    logger.info("metabolic_tracking_wired", system="llm_client→synapse.metabolism")

    # Loop 10: rhythm state → Nova drive weight adaptation
    from ecodiaos.systems.synapse.types import SynapseEventType
    synapse.event_bus.subscribe(
        SynapseEventType.RHYTHM_STATE_CHANGED, nova.on_rhythm_change,
    )

    # ── 13b. Initialize Thymos — The Immune System ─────────────────
    # Must come after Synapse so it can subscribe to health events.
    # Thymos watches the event bus for SYSTEM_FAILED, SYSTEM_RECOVERED,
    # etc. and converts them into Incidents for the immune pipeline.
    thymos = ThymosService(
        config=config.thymos,
        synapse=synapse,
        neo4j=neo4j_client,
        llm=llm_client,
        metrics=metrics,
        neuroplasticity_bus=neuroplasticity_bus,
    )
    # Wire cross-system references
    thymos.set_equor(equor)
    thymos.set_evo(evo)
    thymos.set_atune(atune)
    thymos.set_nova(nova)  # Loop 1: critical incidents → urgent repair goals
    thymos.set_health_monitor(synapse._health)
    await thymos.initialize()
    synapse.register_system(thymos)
    app.state.thymos = thymos

    # ── 13c. Initialize Oneiros — The Dream Engine ───────────────────
    # Must come after Synapse, Thymos, and all cognitive systems.
    # Oneiros subscribes to Synapse events (emergency wake on SYSTEM_FAILED)
    # and cross-wires with Equor, Evo, Nova, Atune, Thymos, Memory.
    oneiros = OneirosService(
        config=config.oneiros,
        synapse=synapse,
        neo4j=neo4j_client,
        llm=llm_client,
        embed_fn=embedding_client,
        metrics=metrics,
        neuroplasticity_bus=neuroplasticity_bus,
    )
    # Wire cross-system references
    oneiros.set_equor(equor)
    oneiros.set_evo(evo)
    oneiros.set_nova(nova)
    oneiros.set_atune(atune)
    oneiros.set_thymos(thymos)
    oneiros.set_memory(memory)
    await oneiros.initialize()
    synapse.register_system(oneiros)
    app.state.oneiros = oneiros

    # ── 13d. Initialize Soma — The Interoceptive Substrate ──────────
    # Must come after Synapse, Atune, Nova, Thymos, Equor, Oneiros.
    # Soma reads from all systems to compose interoceptive state and
    # emits AllostaticSignal consumed by Atune, Nova, Voxis, Evo, etc.
    soma = SomaService(config=config.soma, neuroplasticity_bus=neuroplasticity_bus)
    # Wire interoceptor system references
    soma.set_atune(atune)
    soma.set_synapse(synapse)
    soma.set_nova(nova)
    soma.set_thymos(thymos)
    soma.set_equor(equor)
    # Wire token budget for energy sensing
    if hasattr(synapse, "_resources"):
        soma.set_token_budget(synapse._resources)
    await soma.initialize()
    # Wire Soma into systems for allostatic signal reading
    atune.set_soma(soma)    # Atune uses precision_weights for affect modulation
    synapse.set_soma(soma)  # Synapse wires Soma into clock (step 0)
    nova.set_soma(soma)     # Nova: allostatic deliberation trigger on urgency > 0.7
    memory.set_soma(soma)   # Memory: somatic marker stamping on write, reranking on read
    evo.set_soma(soma)      # Evo: curiosity modulation, dynamics matrix update
    oneiros.set_soma(soma)  # Oneiros: sleep pressure from energy errors, REM counterfactuals
    thymos.set_soma(soma)   # Thymos: integrity precision gating in process_incident
    voxis.set_soma(soma)    # Voxis: arousal/valence expression modulation
    synapse.register_system(soma)
    app.state.soma = soma

    # ── 14. Start Alive WebSocket server ──────────────────────────
    from ecodiaos.systems.alive.ws_server import AliveWebSocketServer

    alive_ws = AliveWebSocketServer(
        redis=redis_client,
        atune=atune,
        port=getattr(config, "alive_ws_port", 8001),
    )
    await alive_ws.start()
    app.state.alive_ws = alive_ws

    # ── 15. Initialize Federation (Phase 11) ──────────────────────
    federation = FederationService(
        config=config.federation,
        memory=memory,
        equor=equor,
        redis=redis_client,
        metrics=metrics,
        instance_id=config.instance_id,
    )
    await federation.initialize()
    app.state.federation = federation
    # Loop 9: federated knowledge → perceived input
    federation.set_atune(atune)
    # Wire federation into Thymos for Layer 4 threat intelligence
    thymos.set_federation(federation)

    if config.federation.enabled:
        # Register with Synapse for health monitoring
        synapse.register_system(federation)

    # ── 15b. Initialize Wallet (Phase 2: Metabolic Layer) ──────────
    # On-chain financial identity via Coinbase Developer Platform.
    # Only connects if CDP credentials are configured — otherwise the
    # organism operates without a wallet (pre-metabolic mode).
    wallet_client: WalletClient | None = None
    if config.wallet.cdp_api_key_id:
        wallet_client = WalletClient(config.wallet)
        await wallet_client.connect()
        app.state.wallet = wallet_client
        # Wire wallet into Axon for metabolic actions (spending, transfers, etc.)
        axon.set_wallet(wallet_client)
        logger.info(
            "ecodiaos_ready",
            phase="15b_wallet",
            address=wallet_client.address,
            network=wallet_client.network,
        )
    else:
        app.state.wallet = None
        logger.info("wallet_skipped", reason="no CDP credentials configured")

    # ── 15c. Wire financial memory encoding ────────────────────────
    # MemoryService and AxonService both need the Synapse event bus so that
    # wallet transfers and revenue injections are encoded as salience=1.0
    # episodes in Neo4j.  Must come after both Synapse (step 13) and the
    # wallet client (step 15b) are initialised.
    memory.set_event_bus(synapse.event_bus)
    axon.set_event_bus(synapse.event_bus)
    logger.info("financial_memory_encoding_wired")

    # ── 15d. Initialize Certificate Manager (Phase 16g: Civilization Layer) ──
    # The CertificateManager re-uses the Federation IdentityManager's Ed25519
    # keypair for signing. It issues birth certificates for children (Mitosis),
    # validates inbound certificates (Federation), and tracks expiry (Oikos).
    from ecodiaos.systems.identity.manager import CertificateManager

    certificate_manager = CertificateManager()
    _fed_identity = federation._identity if federation._identity else None
    if _fed_identity is not None:
        await certificate_manager.initialize(
            identity=_fed_identity,
            instance_id=config.instance_id,
            validity_days=config.oikos.certificate_validity_days,
            expiry_warning_days=config.oikos.certificate_expiry_warning_days,
            ca_address=config.oikos.certificate_ca_address,
            data_dir=config.federation.identity_data_dir,
        )
        certificate_manager.set_event_bus(synapse.event_bus)
        # Wire into Federation for inbound certificate validation
        federation.set_certificate_manager(certificate_manager)
        app.state.certificate_manager = certificate_manager
        logger.info(
            "ecodiaos_ready",
            phase="15d_certificate_manager",
            certified=certificate_manager.is_certified,
            remaining_days=f"{certificate_manager.certificate_remaining_days:.1f}",
        )
    else:
        app.state.certificate_manager = None
        logger.info(
            "certificate_manager_skipped",
            reason="federation identity not initialized",
        )

    # ── 15e. Initialize Oikos — The Economic Engine ─────────────────
    # OikosService owns all sub-engines: AssetFactory, MitosisEngine,
    # DerivativesManager, OrganLifecycleManager, KnowledgeMarket, and
    # TollboothManager. It subscribes to Synapse events for metabolic
    # data and polls the wallet for on-chain balance.
    from ecodiaos.systems.oikos.service import OikosService

    oikos = OikosService(
        config=config.oikos,
        wallet=wallet_client,
        metabolism=synapse.metabolism if hasattr(synapse, "metabolism") else None,
        instance_id=config.instance_id,
        redis=redis_client,
    )
    oikos.initialize(bus=neuroplasticity_bus)
    await oikos.load_state()  # Restore durable ledger from Redis (no-op if first boot)
    oikos.attach(synapse.event_bus)

    # Wire CertificateManager into Oikos for certificate expiry tracking
    if app.state.certificate_manager is not None:
        oikos.set_certificate_manager(app.state.certificate_manager)

    synapse.register_system(oikos)
    app.state.oikos = oikos
    logger.info(
        "ecodiaos_ready",
        phase="15e_oikos",
        cost_model=oikos._cost_model.model_name,
        runway_days=str(oikos.snapshot().runway_days),
    )

    # ── 16. Start internal percept generator ───────────────────────
    # Without external input, the workspace is empty every cycle.
    # This generator provides self-monitoring percepts so EOS has
    # inner experience even when idle: affect self-observation,
    # goal reflection, and memory-prompted curiosity.
    import asyncio as _aio_gen

    _inner_life_task: _aio_gen.Task[Any] | None = None

    async def _inner_life_loop() -> None:
        """
        Periodic self-monitoring that feeds the workspace.

        The organism observes its own internal state across all subsystems:
        affect, goals, immune health, learning progress, cognitive rhythm,
        and sleep pressure. These self-observations enter the workspace
        like any other percept — competing for broadcast and driving
        the cognitive cycle even when no external input arrives.
        """
        _il_logger = structlog.get_logger("ecodiaos.inner_life")
        import random as _rnd

        from ecodiaos.systems.atune.types import WorkspaceContribution

        cycle = 0
        while True:
            try:
                await _aio_gen.sleep(5.0)  # Every 5 seconds (~33 theta cycles)
                cycle += 1

                affect = atune._affect_mgr.current
                goals = nova.active_goal_summaries if nova._goal_manager else []

                # ── Affect self-monitoring (every 2nd cycle = ~10s) ──
                if cycle % 2 == 0:
                    affect_desc = (
                        f"I notice my current state: "
                        f"valence={affect.valence:.2f}, "
                        f"arousal={affect.arousal:.2f}, "
                        f"curiosity={affect.curiosity:.2f}, "
                        f"care_activation={affect.care_activation:.2f}, "
                        f"coherence_stress={affect.coherence_stress:.2f}"
                    )
                    atune.contribute(WorkspaceContribution(
                        system="self_monitor",
                        content=affect_desc,
                        priority=0.35 + affect.coherence_stress * 0.2,
                        reason="affect_self_observation",
                    ))

                # ── Goal reflection (every 6th cycle = ~30s) ──
                if cycle % 6 == 0 and goals:
                    goal = _rnd.choice(goals)
                    goal_text = goal.get("description", "unknown")[:100] if isinstance(goal, dict) else str(goal)[:100]
                    atune.contribute(WorkspaceContribution(
                        system="nova",
                        content=f"Reflecting on my goal: {goal_text}",
                        priority=0.4,
                        reason="goal_reflection",
                    ))

                # ── Synapse rhythm reflection (every 8th cycle = ~40s, offset 2) ──
                if cycle % 8 == 2:
                    try:
                        rhythm = synapse.rhythm_snapshot
                        coherence = synapse.coherence_snapshot
                        rhythm_state = rhythm.state.value
                        atune.contribute(WorkspaceContribution(
                            system="synapse",
                            content=(
                                f"My cognitive rhythm is {rhythm_state} "
                                f"(stability={rhythm.rhythm_stability:.0%}, "
                                f"coherence={coherence.composite:.2f})"
                            ),
                            priority=0.25 + (0.2 if rhythm_state in ("stress", "deep_processing") else 0),
                            reason="rhythm_self_observation",
                        ))
                    except Exception:
                        pass  # Synapse may not have rhythm data yet

                # ── Thymos immune reflection (every 8th cycle = ~40s, offset 4) ──
                if cycle % 8 == 4:
                    try:
                        thymos_health = await thymos.health()
                        healing_mode = thymos_health.get("healing_mode", "normal")
                        active_count = thymos_health.get("active_incidents", 0)
                        if active_count > 0 or healing_mode != "normal":
                            atune.contribute(WorkspaceContribution(
                                system="thymos",
                                content=(
                                    f"My immune system reports: "
                                    f"{active_count} active incidents, "
                                    f"healing mode: {healing_mode}"
                                ),
                                priority=0.4 + (0.2 if healing_mode != "normal" else 0),
                                reason="immune_self_observation",
                            ))
                    except Exception:
                        pass  # Thymos may not be fully initialised yet

                # ── Evo learning reflection (every 10th cycle = ~50s) ──
                if cycle % 10 == 5:
                    try:
                        evo_stats = evo.stats
                        hyp_data = evo_stats.get("hypotheses", {})
                        active_hyp = hyp_data.get("active", 0)
                        supported_hyp = hyp_data.get("supported", 0)
                        if active_hyp > 0:
                            atune.contribute(WorkspaceContribution(
                                system="evo",
                                content=(
                                    f"I'm tracking {active_hyp} hypotheses "
                                    f"({supported_hyp} supported). Learning continues."
                                ),
                                priority=0.3,
                                reason="learning_self_observation",
                            ))
                    except Exception:
                        pass

                # ── Curiosity prompt (every 12th cycle = ~60s) ──
                if cycle % 12 == 0:
                    curiosity_level = affect.curiosity
                    if curiosity_level > 0.2:
                        atune.contribute(WorkspaceContribution(
                            system="self_monitor",
                            content="I wonder what my community is doing. I have not heard from anyone recently — is everything alright?",
                            priority=0.3 + curiosity_level * 0.15,
                            reason="curiosity_prompt",
                        ))

                # ── Oneiros sleep pressure reflection (every 20th cycle = ~100s) ──
                if cycle % 20 == 10:
                    try:
                        oneiros_health = await oneiros.health()
                        pressure = oneiros_health.get("sleep_pressure", 0)
                        stage = oneiros_health.get("current_stage", "awake")
                        if pressure > 0.3 or stage != "awake":
                            atune.contribute(WorkspaceContribution(
                                system="oneiros",
                                content=f"Sleep pressure: {pressure:.0%}. Current stage: {stage}.",
                                priority=0.3 + (0.15 if pressure > 0.6 else 0),
                                reason="sleep_self_observation",
                            ))
                    except Exception:
                        pass

                # ── Axon activity reflection (every 8th cycle, offset 6 = ~40s) ──
                if cycle % 8 == 6:
                    try:
                        axon_stats = axon.stats
                        total_exec = axon_stats.get("total_executions", 0)
                        success_exec = axon_stats.get("successful_executions", 0)
                        if total_exec > 0:
                            effectiveness = (
                                "I am effective."
                                if success_exec > total_exec * 0.7
                                else "Some actions are failing — I should be more careful."
                            )
                            atune.contribute(WorkspaceContribution(
                                system="axon",
                                content=(
                                    f"I have executed {total_exec} actions, "
                                    f"{success_exec} succeeded. {effectiveness}"
                                ),
                                priority=0.3,
                                reason="action_self_observation",
                            ))
                    except Exception:
                        pass

                # ── Voxis expression reflection (every 12th cycle, offset 8 = ~60s) ──
                if cycle % 12 == 8:
                    try:
                        speak_count = getattr(voxis, "_total_speak", 0)
                        silence_count = getattr(voxis, "_total_silence", 0)
                        if speak_count + silence_count > 0:
                            balance = (
                                "I listen more than I speak."
                                if silence_count > speak_count
                                else "I am actively communicating."
                            )
                            atune.contribute(WorkspaceContribution(
                                system="voxis",
                                content=(
                                    f"I have spoken {speak_count} times and chosen "
                                    f"silence {silence_count} times. {balance}"
                                ),
                                priority=0.25,
                                reason="expression_self_observation",
                            ))
                    except Exception:
                        pass

                # ── Federation link reflection (every 20th cycle, offset 15 = ~100s) ──
                if cycle % 20 == 15:
                    try:
                        fed_health = await federation.health()
                        link_count = fed_health.get("active_links", 0)
                        if link_count > 0:
                            atune.contribute(WorkspaceContribution(
                                system="federation",
                                content=(
                                    f"I have {link_count} active federation "
                                    f"link{'s' if link_count != 1 else ''}. "
                                    f"I am part of a community of organisms."
                                ),
                                priority=0.25,
                                reason="federation_self_observation",
                            ))
                    except Exception:
                        pass

                # ── Equor constitutional reflection (every 12th cycle, offset 4 = ~60s) ──
                if cycle % 12 == 4:
                    try:
                        total_reviews = getattr(equor, "_total_reviews", 0)
                        if total_reviews > 0:
                            atune.contribute(WorkspaceContribution(
                                system="equor",
                                content=(
                                    f"My constitutional compass has reviewed "
                                    f"{total_reviews} intents. "
                                    f"I remain aligned with my drives."
                                ),
                                priority=0.25,
                                reason="constitutional_self_observation",
                            ))
                    except Exception:
                        pass

                # ── Thread narrative identity cycle (every cycle) ──
                # Thread.on_cycle handles its own staggering internally:
                #   Every 100 cycles: fingerprint
                #   Every 200 cycles: Evo pattern check
                #   Every 1000 cycles: schema conflict scan
                #   Every 5000 cycles: life story synthesis
                with suppress(Exception):
                    await thread.on_cycle(cycle)

                # ── Thread identity reflection (every 20th cycle, offset 5 = ~100s) ──
                if cycle % 20 == 5:
                    try:
                        identity_ctx = thread.get_identity_context()
                        if identity_ctx:
                            atune.contribute(WorkspaceContribution(
                                system="thread",
                                content=f"My narrative identity: {identity_ctx}",
                                priority=0.3,
                                reason="identity_self_observation",
                            ))
                    except Exception:
                        pass

                _il_logger.debug("inner_life_tick", cycle=cycle)
            except _aio_gen.CancelledError:
                _il_logger.info("inner_life_stopped")
                return
            except Exception as exc:
                _il_logger.warning("inner_life_error", error=str(exc))

    _inner_life_task = _aio_gen.create_task(
        _inner_life_loop(), name="inner_life_generator"
    )

    # ── 17. Start multi-channel perception ────────────────────────
    # File watcher: drop .txt/.md into config/percepts/ to inject percepts
    # Scheduler: register cron-like tasks that poll external sources
    from pathlib import Path as _PPath
    from ecodiaos.clients.file_watcher import FileWatcher
    from ecodiaos.clients.scheduler import PerceptionScheduler

    _percepts_dir = _PPath(
        os.environ.get("ECODIAOS_PERCEPTS_DIR", "config/percepts")
    ).resolve()
    _file_watcher = FileWatcher(watch_dir=_percepts_dir, atune=atune)
    await _file_watcher.start()
    app.state.file_watcher = _file_watcher

    _scheduler = PerceptionScheduler(atune=atune)
    await _scheduler.start()
    app.state.scheduler = _scheduler

    logger.info(
        "ecodiaos_ready",
        phase="17_multi_channel_perception",
        federation_enabled=config.federation.enabled,
        immune_system="active",
        inner_life="active",
        file_watcher=str(_percepts_dir),
    )

    # ── 18. Start distributed shield listener ─────────────────────
    from ecodiaos.systems.simula.distributed_shield import FleetShieldManager

    fleet_shield = FleetShieldManager(redis_client)
    fleet_shield.start()
    app.state.fleet_shield = fleet_shield
    logger.info("ecodiaos_ready", phase="18_fleet_shield_listener")

    # ── 19. Start metrics publisher ───────────────────────────────
    metrics_queue: asyncio.Queue[dict] = asyncio.Queue()
    app.state.metrics_queue = metrics_queue
    metrics_task = asyncio.create_task(publish_metrics_loop(redis_client, metrics_queue))
    logger.info("ecodiaos_ready", phase="19_metrics_publisher")

    yield

    metrics_task.cancel()
    with suppress(asyncio.CancelledError):
        await metrics_task

    # Cancel inner life before shutdown
    if _inner_life_task is not None:
        _inner_life_task.cancel()
        with suppress(_aio_gen.CancelledError):
            await _inner_life_task

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("ecodiaos_shutting_down")
    await neuroplasticity_bus.stop()
    await _scheduler.stop()
    await _file_watcher.stop()
    await federation.shutdown()
    await alive_ws.stop()
    await thymos.shutdown()
    await oikos.shutdown()
    await synapse.stop()
    await simula.shutdown()
    await thread.shutdown()
    await evo.shutdown()
    await axon.shutdown()
    await nova.shutdown()
    await voxis.shutdown()
    await atune.shutdown()
    await equor.shutdown()
    await fleet_shield.shutdown()
    if wallet_client is not None:
        await wallet_client.close()
    await metrics.stop()
    await embedding_client.close()
    await llm_client.close()
    await redis_client.close()
    await tsdb_client.close()
    await neo4j_client.close()
    logger.info("ecodiaos_shutdown_complete")


# ─── Helpers ─────────────────────────────────────────────────────


class _MemoryWorkspaceAdapter:
    """
    Bridge between MemoryService and Atune's WorkspaceMemoryClient protocol.

    Enables:
    * **Spontaneous recall** — high-salience, recently-unaccessed episodes
      "bubble up" into consciousness (find_bubbling_memory).
    * **Context enrichment** — each broadcast winner gets contextual memories
      attached (retrieve_context).
    * **Broadcast-time storage** — winning percepts stored as episodes with
      full salience and affect metadata (store_percept_with_broadcast).
    """

    def __init__(self, memory: MemoryService) -> None:
        self._memory = memory

    async def retrieve_context(
        self,
        query_embedding: list[float],
        query_text: str,
        max_results: int,
    ):
        """Retrieve memories relevant to the current workspace winner."""
        from ecodiaos.systems.atune.types import MemoryContext
        try:
            response = await self._memory.retrieve(
                query_text=query_text or None,
                query_embedding=query_embedding or None,
                max_results=max_results,
                salience_floor=0.0,
            )
            # Convert RetrievalResults → MemoryTraces for workspace context
            from ecodiaos.primitives.memory_trace import MemoryTrace
            traces = []
            for r in response.traces:
                traces.append(MemoryTrace(
                    episode_id=r.node_id,
                    original_percept_id=r.node_id,
                    summary=r.content or "",
                ))
            return MemoryContext(traces=traces)
        except Exception:
            return MemoryContext()

    async def find_bubbling_memory(
        self,
        min_salience: float,
        max_recent_access_hours: int,
    ):
        """
        Find a high-salience episode that hasn't been accessed recently.
        This powers spontaneous recall — the "I just thought of something"
        experience. Finds episodes with high salience but low recent access,
        indicating dormant but important memories ready to surface.
        """
        try:
            results = await self._memory._neo4j.execute_read(
                """
                MATCH (ep:Episode)
                WHERE ep.salience_composite >= $min_salience
                  AND ep.last_accessed < datetime() - duration({hours: $hours})
                RETURN ep.id AS id, ep.summary AS content,
                       ep.salience_composite AS salience,
                       ep.embedding AS embedding
                ORDER BY ep.salience_composite DESC
                LIMIT 1
                """,
                {"min_salience": min_salience, "hours": max_recent_access_hours},
            )
            if not results:
                return None

            row = results[0]
            # Return as a minimal Percept that the workspace can wrap
            from ecodiaos.primitives.common import (
                Modality,
                SourceDescriptor,
                SystemID,
                utc_now,
            )
            from ecodiaos.primitives.percept import Content, Percept

            return Percept(
                source=SourceDescriptor(
                    system=SystemID.MEMORY,
                    channel="spontaneous_recall",
                    modality=Modality.TEXT,
                ),
                content=Content(
                    raw=row.get("content", ""),
                    embedding=row.get("embedding") or [],
                ),
                timestamp=utc_now(),
            )
        except Exception:
            return None

    async def store_percept_with_broadcast(self, percept, salience, affect) -> None:
        """Store the broadcast-winning percept as an episode."""
        try:
            await self._memory.store_percept(
                percept=percept,
                salience_composite=salience.composite,
                salience_scores=salience.scores,
                affect_valence=affect.valence,
                affect_arousal=affect.arousal,
            )
        except Exception:
            logger.debug("broadcast_storage_failed", exc_info=True)


async def _seed_atune_cache(atune: AtuneService, embedding_client, instance) -> None:
    """
    Populate Atune's identity cache from the born instance so the
    Identity salience head can score percepts correctly from the start.
    """
    try:
        name_embedding = await embedding_client.embed(instance.name)
        community_text = getattr(instance, "community_context", "") or instance.name
        community_embedding = await embedding_client.embed(community_text)

        atune.set_cache_identity(
            core_embeddings=[name_embedding],
            community_embedding=community_embedding,
            instance_name=instance.name,
        )
        logger.info("atune_cache_seeded", instance_name=instance.name)
    except Exception:
        logger.warning("atune_cache_seed_failed", exc_info=True)


def _resolve_governance_config(config: Any) -> Any:
    """Resolve governance config from seed or use defaults."""
    from ecodiaos.config import GovernanceConfig
    try:
        seed_path = os.environ.get("ECODIAOS_SEED_PATH", "config/seeds/example_seed.yaml")
        seed = load_seed(seed_path)
        return seed.constitution.governance
    except Exception:
        return GovernanceConfig()


# ─── FastAPI Application ─────────────────────────────────────────

app = FastAPI(
    title="EcodiaOS",
    description="A living digital organism — API surface",
    version="0.3.0",
    lifespan=lifespan,
)

# CORS for frontend
_cors_origins = [
    "http://localhost:3000",
    "https://ecodiaos-frontend-929871567697.australia-southeast1.run.app",
]
# Allow additional origins via env var (comma-separated)
_extra_origins = os.environ.get("CORS_ALLOWED_ORIGINS", "")
if _extra_origins:
    _cors_origins.extend(o.strip() for o in _extra_origins.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Oikos & Identity Router ─────────────────────────────────────
from ecodiaos.api.routers.oikos import router as oikos_router

app.include_router(oikos_router)


# ─── Alive WebSocket on port 8000 (for Cloud Run) ────────────────
# Cloud Run only exposes one port per container. The standalone ws_server
# on port 8001 is unreachable, so we add a FastAPI WebSocket route here
# that taps into the same Atune + Redis data streams.

import asyncio as _ws_asyncio

import orjson as _ws_orjson
from fastapi import WebSocket, WebSocketDisconnect


def _ws_json(data: dict[str, Any]) -> str:
    return _ws_orjson.dumps(data).decode()


@app.websocket("/ws/alive")
async def alive_websocket(ws: WebSocket):
    """
    Alive visualization WebSocket — mirrors the standalone ws_server behaviour.

    Streams two channels to the client:
      {"stream": "affect",  "payload": {...}}  — polled from Atune at ~10 Hz
      {"stream": "synapse", "payload": {...}}  — forwarded from Redis pub/sub
    """
    await ws.accept()

    atune: AtuneService = app.state.atune
    redis_client: RedisClient = app.state.redis

    # Send initial affect snapshot so the client renders immediately
    affect = atune.current_affect
    await ws.send_text(_ws_json({
        "stream": "affect",
        "payload": {
            "valence": round(affect.valence, 4),
            "arousal": round(affect.arousal, 4),
            "dominance": round(affect.dominance, 4),
            "curiosity": round(affect.curiosity, 4),
            "care_activation": round(affect.care_activation, 4),
            "coherence_stress": round(affect.coherence_stress, 4),
            "ts": affect.timestamp.isoformat() if affect.timestamp else None,
        },
    }))

    # Channel for background tasks to push messages into
    queue: _ws_asyncio.Queue[str] = _ws_asyncio.Queue(maxsize=256)
    running = True

    async def _affect_poller() -> None:
        """Poll Atune affect at ~10 Hz."""
        while running:
            a = atune.current_affect
            msg = _ws_json({
                "stream": "affect",
                "payload": {
                    "valence": round(a.valence, 4),
                    "arousal": round(a.arousal, 4),
                    "dominance": round(a.dominance, 4),
                    "curiosity": round(a.curiosity, 4),
                    "care_activation": round(a.care_activation, 4),
                    "coherence_stress": round(a.coherence_stress, 4),
                    "ts": a.timestamp.isoformat() if a.timestamp else None,
                },
            })
            try:
                queue.put_nowait(msg)
            except _ws_asyncio.QueueFull:
                pass  # Drop if client is lagging
            await _ws_asyncio.sleep(0.1)

    async def _redis_subscriber() -> None:
        """Subscribe to synapse events from Redis and enqueue."""
        redis = redis_client.client
        prefix = redis_client._config.prefix
        channel = f"{prefix}:channel:synapse_events"
        pubsub = redis.pubsub()
        await pubsub.subscribe(channel)
        try:
            while running:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=0.1,
                )
                if message and message["type"] == "message":
                    raw = message["data"]
                    payload = _ws_orjson.loads(raw) if isinstance(raw, (str, bytes)) else raw
                    msg = _ws_json({"stream": "synapse", "payload": payload})
                    with suppress(_ws_asyncio.QueueFull):
                        queue.put_nowait(msg)
        except _ws_asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()  # type: ignore[no-untyped-call]

    async def _sender() -> None:
        """Drain the queue and send to the WebSocket client."""
        while running:
            msg = await queue.get()
            await ws.send_text(msg)

    async def _workspace_poller() -> None:
        """Poll Atune workspace snapshot at ~1 Hz for the perception page."""
        while running:
            try:
                ws_snapshot = atune.workspace_snapshot
                affect = atune.current_affect
                workspace_items = []
                for b in ws_snapshot.recent_broadcasts:
                    workspace_items.append({
                        "broadcast_id": b.broadcast_id,
                        "salience": round(b.salience, 4),
                        "ts": b.timestamp.isoformat() if hasattr(b, "timestamp") and b.timestamp else None,
                    })
                msg = _ws_json({
                    "stream": "workspace",
                    "payload": {
                        "cycle_count": atune.cycle_count,
                        "dynamic_threshold": round(atune.workspace_threshold, 4),
                        "meta_attention_mode": atune.meta_attention_mode,
                        "recent_broadcasts": workspace_items,
                        "affect": {
                            "valence": round(affect.valence, 4),
                            "arousal": round(affect.arousal, 4),
                            "curiosity": round(affect.curiosity, 4),
                            "coherence_stress": round(affect.coherence_stress, 4),
                        },
                    },
                })
                try:
                    queue.put_nowait(msg)
                except _ws_asyncio.QueueFull:
                    pass
            except Exception:
                pass
            await _ws_asyncio.sleep(1.0)  # 1 Hz — workspace state changes slowly

    async def _outcomes_poller() -> None:
        """Poll Axon recent outcomes at ~0.5 Hz for the decisions page."""
        last_count = 0
        while running:
            try:
                axon_svc = app.state.axon
                total = getattr(axon_svc, "_total_executions", 0)
                if total != last_count:
                    last_count = total
                    outcomes = axon_svc.recent_outcomes[:10]
                    msg = _ws_json({
                        "stream": "outcomes",
                        "payload": {
                            "outcomes": [
                                {
                                    "execution_id": o.execution_id,
                                    "intent_id": o.intent_id,
                                    "success": o.success,
                                    "partial": o.partial,
                                    "status": o.status.value,
                                    "failure_reason": o.failure_reason or None,
                                    "duration_ms": o.duration_ms,
                                    "steps": [
                                        {
                                            "action_type": s.action_type,
                                            "description": s.description[:80],
                                            "success": s.result.success,
                                        }
                                        for s in o.step_outcomes
                                    ],
                                    "world_state_changes": o.world_state_changes[:3],
                                }
                                for o in outcomes
                            ],
                            "total": total,
                            "successful": getattr(axon_svc, "_successful_executions", 0),
                            "failed": getattr(axon_svc, "_failed_executions", 0),
                        },
                    })
                    try:
                        queue.put_nowait(msg)
                    except _ws_asyncio.QueueFull:
                        pass
            except Exception:
                pass
            await _ws_asyncio.sleep(2.0)  # 0.5 Hz — poll on change

    poller_task = _ws_asyncio.create_task(_affect_poller())
    subscriber_task = _ws_asyncio.create_task(_redis_subscriber())
    workspace_task = _ws_asyncio.create_task(_workspace_poller())
    outcomes_task = _ws_asyncio.create_task(_outcomes_poller())
    sender_task = _ws_asyncio.create_task(_sender())

    try:
        # Keep alive — we don't expect client messages
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        running = False
        poller_task.cancel()
        subscriber_task.cancel()
        workspace_task.cancel()
        outcomes_task.cancel()
        sender_task.cancel()


# ─── API Key Authentication Middleware ─────────────────────────────
# Protects all /api/v1/* endpoints. /health is always public.
# When no API keys are configured (dev mode), all requests pass through.

from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from starlette.requests import Request


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validates API key from X-EOS-API-Key header or Authorization Bearer token.

    Protected paths: /api/v1/*
    Public paths: /health, /docs, /openapi.json, /api/v1/federation/identity
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Public endpoints — no auth required
        if path in ("/health", "/docs", "/openapi.json", "/redoc", "/api/v1/admin/llm/metrics", "/api/v1/admin/llm/summary"):
            return await call_next(request)
        # Federation endpoints accessible to peers (trust-gated at service level)
        if path in (
            "/api/v1/federation/identity",
            "/api/v1/federation/threat-advisory",
        ):
            return await call_next(request)

        # Only protect /api/v1/* paths
        if not path.startswith("/api/v1/"):
            return await call_next(request)

        # Check if auth is configured
        config = getattr(request.app.state, "config", None)
        if config is None or not config.server.api_keys:
            # Dev mode: no keys configured, allow all
            return await call_next(request)

        # Extract API key from header or Authorization bearer
        api_key = request.headers.get(config.server.api_key_header, "")
        if not api_key:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]

        if not api_key or api_key not in config.server.api_keys:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing API key"},
            )

        return await call_next(request)


app.add_middleware(APIKeyMiddleware)


# ─── Health & Status Endpoints ────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, Any]:
    """System health check."""
    memory_health = await app.state.memory.health()
    equor_health = await app.state.equor.health()
    voxis_health = await app.state.voxis.health()
    nova_health = await app.state.nova.health()
    synapse_health = await app.state.synapse.health() if hasattr(app.state, "synapse") else {"status": "not_initialized"}
    thymos_health = await app.state.thymos.health() if hasattr(app.state, "thymos") else {"status": "not_initialized"}
    oneiros_health = await app.state.oneiros.health() if hasattr(app.state, "oneiros") else {"status": "not_initialized"}
    thread_health = await app.state.thread.health() if hasattr(app.state, "thread") else {"status": "not_initialized"}
    neo4j_health = await app.state.neo4j.health_check()
    tsdb_health = await app.state.tsdb.health_check()
    redis_health = await app.state.redis.health_check()

    overall = "healthy"
    if any(
        h.get("status") != "connected"
        for h in [neo4j_health, tsdb_health, redis_health]
    ):
        overall = "degraded"
    if equor_health.get("safe_mode"):
        overall = "degraded"
    if synapse_health.get("safe_mode"):
        overall = "safe_mode"

    instance = await app.state.memory.get_self()
    atune: AtuneService = app.state.atune

    federation_health = await app.state.federation.health() if hasattr(app.state, "federation") else {"status": "not_initialized"}

    return {
        "status": overall,
        "instance_id": app.state.config.instance_id,
        "instance_name": instance.name if instance else "unborn",
        "phase": "13_oneiros",
        "systems": {
            "memory": memory_health,
            "equor": equor_health,
            "nova": nova_health,
            "axon": app.state.axon.stats if hasattr(app.state, "axon") else {"status": "not_initialized"},
            "evo": app.state.evo.stats if hasattr(app.state, "evo") else {"status": "not_initialized"},
            "simula": app.state.simula.stats if hasattr(app.state, "simula") else {"status": "not_initialized"},
            "atune": {
                "status": "running",
                "cycle_count": atune.cycle_count,
                "workspace_threshold": round(atune.workspace_threshold, 4),
                "meta_attention_mode": atune.meta_attention_mode,
                "affect": {
                    "valence": round(atune.current_affect.valence, 4),
                    "arousal": round(atune.current_affect.arousal, 4),
                    "curiosity": round(atune.current_affect.curiosity, 4),
                    "care_activation": round(atune.current_affect.care_activation, 4),
                    "coherence_stress": round(atune.current_affect.coherence_stress, 4),
                },
            },
            "voxis": voxis_health,
            "synapse": synapse_health,
            "thymos": thymos_health,
            "oneiros": oneiros_health,
            "thread": thread_health,
            "federation": federation_health,
        },
        "data_stores": {
            "neo4j": neo4j_health,
            "timescaledb": tsdb_health,
            "redis": redis_health,
        },
    }


@app.get("/api/v1/admin/llm/metrics")
async def get_llm_metrics():
    """
    LLM cost optimization dashboard.

    Returns token spend, cache hit rate, budget tier, latency,
    and per-system breakdowns.
    """
    llm_metrics: LLMMetricsCollector = app.state.llm_metrics
    token_budget: TokenBudget = app.state.token_budget
    llm: OptimizedLLMProvider = app.state.llm

    dashboard = llm_metrics.get_dashboard_data()
    budget_status = token_budget.get_status()
    cache_stats = llm.cache.get_stats() if llm.cache else {"hit_rate": 0.0}

    import math

    def _safe_float(v: float) -> float | None:
        """Return None for inf/nan so the JSON encoder doesn't crash."""
        if math.isfinite(v):
            return round(v, 2)
        return None

    return {
        "status": "ok",
        "budget": {
            "tier": budget_status.tier.value,
            "tokens_used": budget_status.tokens_used,
            "tokens_remaining": budget_status.tokens_remaining,
            "calls_made": budget_status.calls_made,
            "calls_remaining": budget_status.calls_remaining,
            "burn_rate_tokens_per_sec": _safe_float(budget_status.tokens_per_sec),
            "hours_until_exhausted": _safe_float(budget_status.hours_until_exhausted),
            "warning": budget_status.warning_message,
        },
        "cache": cache_stats,
        "dashboard": dashboard,
    }


@app.get("/api/v1/admin/llm/summary")
async def get_llm_summary():
    """Human-readable LLM cost summary."""
    llm_metrics: LLMMetricsCollector = app.state.llm_metrics
    return {"summary": llm_metrics.summary()}


@app.get("/api/v1/admin/instance")
async def get_instance():
    """Get instance information."""
    instance = await app.state.memory.get_self()
    if instance is None:
        return {"status": "unborn", "message": "No instance has been born yet."}
    return instance.model_dump()


@app.get("/api/v1/admin/memory/stats")
async def get_memory_stats():
    """Get memory graph statistics."""
    return await app.state.memory.stats()


@app.get("/api/v1/governance/constitution")
async def get_constitution():
    """View the current constitution."""
    constitution = await app.state.memory.get_constitution()
    if constitution is None:
        return {"status": "not_found"}
    return constitution


@app.get("/api/v1/admin/health")
async def full_health():
    """Alias for /health with full detail."""
    return await health()


# ─── Phase 3: Perception via Atune ───────────────────────────────


@app.post("/api/v1/perceive/event")
async def perceive_event(body: dict[str, Any]):
    """
    Ingest a percept through Atune's full perception pipeline.

    The input is normalised, scored by seven salience heads,
    precision-weighted by the current affect state, and if it passes
    the workspace ignition threshold, broadcast to all systems.

    Body: {text, channel?, metadata?}
    """
    text = body.get("text", body.get("content", ""))
    if not text:
        return {"error": "No text/content provided"}

    channel_str = body.get("channel", "text_chat")
    try:
        channel = InputChannel(channel_str)
    except ValueError:
        channel = InputChannel.TEXT_CHAT

    raw = RawInput(
        data=text,
        channel_id=body.get("channel_id", ""),
        metadata=body.get("metadata", {}),
    )

    # Ingest through Atune (normalise → predict → score → enqueue)
    atune: AtuneService = app.state.atune
    percept_id = await atune.ingest(raw, channel)

    if percept_id is None:
        return {"percept_id": None, "accepted": False, "reason": "queue_full"}

    # Percept is enqueued — Synapse clock will pick it up on the next tick.
    return {
        "percept_id": percept_id,
        "accepted": True,
        "queued": True,
        "salience_threshold": round(atune.workspace_threshold, 4),
        "affect": {
            "valence": round(atune.current_affect.valence, 4),
            "arousal": round(atune.current_affect.arousal, 4),
            "curiosity": round(atune.current_affect.curiosity, 4),
            "care_activation": round(atune.current_affect.care_activation, 4),
            "coherence_stress": round(atune.current_affect.coherence_stress, 4),
        },
    }


@app.get("/api/v1/atune/affect")
async def get_affect_state():
    """Get Atune's current affective state."""
    affect = app.state.atune.current_affect
    return {
        "valence": round(affect.valence, 4),
        "arousal": round(affect.arousal, 4),
        "dominance": round(affect.dominance, 4),
        "curiosity": round(affect.curiosity, 4),
        "care_activation": round(affect.care_activation, 4),
        "coherence_stress": round(affect.coherence_stress, 4),
        "timestamp": affect.timestamp.isoformat() if affect.timestamp else None,
    }


@app.get("/api/v1/atune/workspace")
async def get_workspace_state():
    """Get workspace state — threshold, recent broadcasts, meta-attention mode."""
    atune: AtuneService = app.state.atune
    recent = atune.recent_broadcasts
    return {
        "cycle_count": atune.cycle_count,
        "dynamic_threshold": round(atune.workspace_threshold, 4),
        "meta_attention_mode": atune.meta_attention_mode,
        "recent_broadcasts": [
            {
                "broadcast_id": b.broadcast_id,
                "salience": round(b.salience.composite, 4),
                "timestamp": b.timestamp.isoformat() if b.timestamp else None,
            }
            for b in recent[-10:]
        ],
    }


# ─── Phase 1: Memory Test Endpoints (kept for backwards compat) ──


@app.post("/api/v1/memory/retrieve")
async def retrieve_memory(body: dict[str, Any]):
    """
    Query memory (temporary test endpoint).
    In later phases, retrieval is triggered by the cognitive cycle.
    """
    query = body.get("query", "")
    if not query:
        return {"error": "No query provided"}

    response = await app.state.memory.retrieve(
        query_text=query,
        max_results=body.get("max_results", 10),
    )
    return response.model_dump()


# ─── Phase 2: Equor Endpoints ────────────────────────────────────


@app.post("/api/v1/equor/review")
async def review_intent(body: dict[str, Any]):
    """
    Submit an Intent for constitutional review (test endpoint).
    In later phases, Nova calls this automatically.

    Body: {goal, steps?, reasoning?, alternatives?, domain?, expected_free_energy?}
    """
    from ecodiaos.primitives.intent import (
        Action,
        ActionSequence,
        DecisionTrace,
        GoalDescriptor,
        Intent,
    )

    goal_text = body.get("goal", "")
    if not goal_text:
        return {"error": "No goal provided"}

    steps = []
    for s in body.get("steps", []):
        steps.append(Action(
            executor=s.get("executor", ""),
            parameters=s.get("parameters", {}),
        ))

    intent = Intent(
        goal=GoalDescriptor(
            description=goal_text,
            target_domain=body.get("domain", ""),
        ),
        plan=ActionSequence(steps=steps),
        expected_free_energy=body.get("expected_free_energy", 0.0),
        decision_trace=DecisionTrace(
            reasoning=body.get("reasoning", ""),
            alternatives_considered=body.get("alternatives", []),
        ),
    )

    check = await app.state.equor.review(intent)
    return check.model_dump()


@app.get("/api/v1/equor/invariants")
async def get_invariants():
    """List all active invariants (hardcoded + community)."""
    return await app.state.equor.get_invariants()


@app.get("/api/v1/equor/drift")
async def get_drift():
    """Get the current drift report."""
    return await app.state.equor.get_drift_report()


@app.get("/api/v1/equor/autonomy")
async def get_autonomy():
    """Get the current autonomy level and promotion eligibility."""
    level = await app.state.equor.get_autonomy_level()
    next_level = level + 1 if level < 3 else None
    eligibility = None
    if next_level:
        eligibility = await app.state.equor.check_promotion(next_level)
    return {
        "current_level": level,
        "level_name": {1: "Advisor", 2: "Partner", 3: "Steward"}.get(level, "unknown"),
        "promotion_eligibility": eligibility,
    }


@app.get("/api/v1/governance/history")
async def governance_history():
    """View governance event history."""
    return await app.state.equor.get_governance_history()


@app.get("/api/v1/governance/reviews")
async def recent_reviews():
    """View recent constitutional reviews."""
    return await app.state.equor.get_recent_reviews()


@app.post("/api/v1/governance/amendments")
async def propose_amendment_endpoint(body: dict[str, Any]):
    """
    Propose a constitutional amendment.
    Body: {proposed_drives: {coherence, care, growth, honesty}, title, description, proposer_id}
    """
    required = ["proposed_drives", "title", "description", "proposer_id"]
    for field in required:
        if field not in body:
            return {"error": f"Missing required field: {field}"}

    return await app.state.equor.propose_amendment(
        proposed_drives=body["proposed_drives"],
        title=body["title"],
        description=body["description"],
        proposer_id=body["proposer_id"],
    )


# ─── Phase 4: Chat & Expression via Voxis ────────────────────────


@app.post("/api/v1/chat/message")
async def chat_message(body: dict[str, Any]):
    """
    Send a message to EOS and receive an expression in response.

    The full Voxis pipeline runs: silence check → policy selection via EFE
    → personality + affect colouring → audience adaptation → LLM generation
    → honesty check → response.

    Body: {message, conversation_id?, speaker_id?, speaker_name?}
    Returns: {expression_id, content, is_silence, silence_reason?, conversation_id,
              policy_class, efe_score, affect_snapshot, generation_trace?}
    """
    message = body.get("message", body.get("content", ""))
    if not message:
        return {"error": "No message provided"}

    voxis: VoxisService = app.state.voxis
    atune: AtuneService = app.state.atune
    current_affect = atune.current_affect

    conversation_id = body.get("conversation_id")
    speaker_id = body.get("speaker_id")
    speaker_name = body.get("speaker_name")

    try:
        # Record user message into conversation state first
        conversation_id = await voxis.ingest_user_message(
            message=message,
            conversation_id=conversation_id,
            speaker_id=speaker_id,
        )

        # Also feed through Atune (updates affect, workspace state).
        # No manual run_cycle() needed — Synapse clock picks up the percept.
        try:
            raw = RawInput(data=message, channel_id=conversation_id or "", metadata={})
            await atune.ingest(raw, InputChannel.TEXT_CHAT)
        except Exception as atune_err:
            # Atune ingestion is non-critical for chat — log and continue
            _chat_logger.warning("chat_atune_ingest_failed", error=str(atune_err))

        # Generate expression via Voxis.
        # NOTE: Do NOT pass the raw user message as content here — ingest_user_message()
        # already appended it to conversation history, and express() appends content as
        # an additional user turn. Passing message again would duplicate it in the LLM
        # context, causing the model to see the same prompt twice and loop on the same
        # defensive response. Pass a response directive instead.
        expression = await voxis.express(
            content="Respond to the conversation.",
            trigger=ExpressionTrigger.NOVA_RESPOND,
            conversation_id=conversation_id,
            addressee_id=speaker_id,
            addressee_name=speaker_name,
            affect=current_affect,
            urgency=0.6,
        )
    except Exception as exc:
        _chat_logger.error("chat_expression_failed", error=str(exc), exc_info=True)
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={
                "error": "expression_pipeline_failed",
                "detail": str(exc),
                "stage": "voxis_express",
            },
        )

    # Include Thread identity context in response (P2.9)
    identity_context = ""
    if hasattr(app.state, "thread"):
        with suppress(Exception):
            identity_context = app.state.thread.get_identity_context()

    response: dict[str, Any] = {
        "expression_id": expression.id,
        "conversation_id": expression.conversation_id,
        "content": expression.content,
        "is_silence": expression.is_silence,
        "identity_context": identity_context,
    }

    if expression.is_silence:
        response["silence_reason"] = expression.silence_reason
    else:
        response["channel"] = expression.channel
        response["affect_snapshot"] = {
            "valence": round(expression.affect_valence, 4),
            "arousal": round(expression.affect_arousal, 4),
            "curiosity": round(expression.affect_curiosity, 4),
            "care_activation": round(expression.affect_care_activation, 4),
            "coherence_stress": round(expression.affect_coherence_stress, 4),
        }
        if expression.generation_trace:
            response["generation"] = {
                "model": expression.generation_trace.model,
                "temperature": expression.generation_trace.temperature,
                "latency_ms": expression.generation_trace.latency_ms,
                "honesty_check_passed": expression.generation_trace.honesty_check_passed,
            }

    return response


@app.get("/api/v1/voxis/personality")
async def get_voxis_personality():
    """Get the current personality vector and its dimensions."""
    voxis: VoxisService = app.state.voxis
    p = voxis.current_personality
    return {
        "warmth": p.warmth,
        "directness": p.directness,
        "verbosity": p.verbosity,
        "formality": p.formality,
        "curiosity_expression": p.curiosity_expression,
        "humour": p.humour,
        "empathy_expression": p.empathy_expression,
        "confidence_display": p.confidence_display,
        "metaphor_use": p.metaphor_use,
        "vocabulary_affinities": p.vocabulary_affinities,
        "thematic_references": p.thematic_references,
    }


@app.get("/api/v1/voxis/health")
async def get_voxis_health():
    """Voxis system health and observability metrics."""
    return await app.state.voxis.health()


# ─── Nova Endpoints ───────────────────────────────────────────────


@app.get("/api/v1/nova/health")
async def get_nova_health():
    """Nova decision system health and observability metrics."""
    return await app.state.nova.health()


@app.get("/api/v1/nova/goals")
async def get_nova_goals():
    """Active goals in Nova's goal manager."""
    nova: NovaService = app.state.nova
    goals = nova._goal_manager.active_goals if nova._goal_manager else []
    return {
        "active_goals": [
            {
                "id": g.id,
                "description": g.description,
                "source": g.source.value,
                "priority": round(g.priority, 4),
                "urgency": round(g.urgency, 4),
                "progress": round(g.progress, 4),
                "status": g.status.value,
            }
            for g in goals
        ],
        "total_active": len(goals),
    }


# ─── Phase 7: Evo Endpoints ───────────────────────────────────────


@app.get("/api/v1/evo/stats")
async def get_evo_stats():
    """Evo learning system stats — hypotheses, parameters, consolidation."""
    evo: EvoService = app.state.evo
    return evo.stats


@app.post("/api/v1/evo/consolidate")
async def trigger_consolidation():
    """Manually trigger an Evo consolidation cycle (sleep mode)."""
    evo: EvoService = app.state.evo
    result = await evo.run_consolidation()
    if result is None:
        return {
            "status": "skipped",
            "duration_ms": 0,
            "hypotheses_evaluated": 0,
            "hypotheses_integrated": 0,
            "procedures_extracted": 0,
            "parameters_adjusted": 0,
            "total_parameter_delta": 0.0,
        }

    # Get the result as dict and add status field
    result_dict = result.model_dump()
    result_dict["status"] = "completed"
    return result_dict


@app.get("/api/v1/evo/parameters")
async def get_evo_parameters():
    """Get all current Evo-tuned parameter values."""
    evo: EvoService = app.state.evo
    return evo.get_all_parameters()


# ─── Thread: Narrative Identity Endpoints ──────────────────────────


@app.get("/api/v1/thread/who-am-i")
async def thread_who_am_i():
    """The organism's current identity summary — who it thinks it is."""
    thread: ThreadService = app.state.thread
    return thread.who_am_i()


@app.get("/api/v1/thread/health")
async def thread_health_endpoint():
    """Thread system health and identity metrics."""
    thread: ThreadService = app.state.thread
    return await thread.health()


@app.get("/api/v1/thread/identity")
async def thread_identity():
    """Complete narrative identity snapshot."""
    thread: ThreadService = app.state.thread

    # Gather schemas by strength
    core_schemas = []
    established_schemas = []
    developing_schemas = []
    nascent_schemas = []
    total_idem = 0.0

    for s in thread._schemas:
        schema_data = {
            "id": s.id,
            "statement": s.statement,
            "strength": getattr(s, 'strength', 'nascent').value if hasattr(getattr(s, 'strength', None), 'value') else 'nascent',  # type: ignore[union-attr]
            "valence": getattr(s, 'valence', 'ambivalent'),
            "confirmation_count": getattr(s, 'confirmation_count', 0),
            "disconfirmation_count": getattr(s, 'disconfirmation_count', 0),
            "evidence_ratio": round(getattr(s, 'evidence_ratio', 0.5), 3),
            "trigger_contexts": getattr(s, 'trigger_contexts', []),
            "behavioral_tendency": getattr(s, 'behavioral_tendency', None),
        }

        strength = schema_data["strength"]
        if strength == "core":
            core_schemas.append(schema_data)
        elif strength == "established":
            established_schemas.append(schema_data)
        elif strength == "developing":
            developing_schemas.append(schema_data)
        else:
            nascent_schemas.append(schema_data)

        confidence = s.confidence if hasattr(s, 'confidence') else 0.5
        total_idem += confidence

    idem_score = total_idem / len(thread._schemas) if thread._schemas else 0.0

    # Gather active commitments
    active_commitments = []
    total_ipse = 0.0

    for c in thread._commitments:
        commitment_data = {
            "id": c.id,
            "statement": c.statement,
            "source": c.drive_source or "explicit_declaration",
            "status": c.type.value if hasattr(c, 'type') else "active",
            "tests_faced": c.test_count if hasattr(c, 'test_count') else 0,
            "tests_held": max(0, c.test_count - 1) if hasattr(c, 'test_count') and c.test_count > 0 else 0,
            "fidelity": round(c.fidelity, 3),
            "made_at": c.created_at.isoformat() if hasattr(c, 'created_at') else None,
            "last_tested": None,
        }
        active_commitments.append(commitment_data)
        total_ipse += c.fidelity

    ipse_score = total_ipse / len(thread._commitments) if thread._commitments else 0.0

    # Get current chapter
    current_chapter_title = None
    current_chapter_theme = None
    if hasattr(thread, '_chapters') and thread._chapters:
        latest_ch = thread._chapters[-1] if thread._chapters else None
        if latest_ch:
            current_chapter_title = latest_ch.title if hasattr(latest_ch, 'title') else "Forming..."
            current_chapter_theme = latest_ch.theme if hasattr(latest_ch, 'theme') else None

    # Get turning points (from latest chapter if available)
    recent_turning_points: list[dict[str, Any]] = []
    if hasattr(thread, '_chapters') and thread._chapters:
        latest_ch = thread._chapters[-1]
        # Attempt to extract turning points from the chapter
        if hasattr(latest_ch, 'turning_points'):
            for tp in latest_ch.turning_points:
                recent_turning_points.append({
                    "id": getattr(tp, 'id', f"tp_{len(recent_turning_points)}"),
                    "type": getattr(tp, 'type', 'growth'),
                    "description": getattr(tp, 'description', str(tp)),
                    "surprise_magnitude": getattr(tp, 'surprise_magnitude', 0.5),
                    "narrative_weight": getattr(tp, 'narrative_weight', 0.5),
                })

    # Get personality traits
    key_personality_traits = {}
    if hasattr(thread, '_personality_snapshot'):
        key_personality_traits = thread._personality_snapshot if isinstance(thread._personality_snapshot, dict) else {}

    # Get life story
    life_story_summary = None
    if hasattr(thread, '_life_story') and thread._life_story:
        life_story_summary = getattr(thread._life_story, 'summary', None)

    # Determine narrative coherence
    narrative_coherence = "integrated"  # Default to integrated
    if hasattr(thread, '_coherence_status'):
        narrative_coherence = thread._coherence_status

    return {
        "core_schemas": core_schemas,
        "established_schemas": established_schemas,
        "active_commitments": active_commitments,
        "current_chapter_title": current_chapter_title,
        "current_chapter_theme": current_chapter_theme,
        "life_story_summary": life_story_summary,
        "key_personality_traits": key_personality_traits,
        "recent_turning_points": recent_turning_points,
        "narrative_coherence": narrative_coherence,
        "idem_score": round(idem_score, 3),
        "ipse_score": round(ipse_score, 3),
    }


@app.get("/api/v1/thread/commitments")
async def thread_commitments():
    """All identity commitments — constitutional and emergent."""
    thread: ThreadService = app.state.thread

    commitments = []
    strained = []
    total_fidelity = 0.0

    for c in thread._commitments:
        commitment_data = {
            "id": c.id,
            "statement": c.statement,
            "source": c.drive_source or "explicit_declaration",
            "status": c.type.value if hasattr(c, 'type') else "active",
            "tests_faced": c.test_count if hasattr(c, 'test_count') else 0,
            "tests_held": max(0, c.test_count - 1) if hasattr(c, 'test_count') and c.test_count > 0 else 0,
            "fidelity": round(c.fidelity, 3),
            "made_at": c.created_at.isoformat() if hasattr(c, 'created_at') else None,
            "last_tested": None,
        }
        commitments.append(commitment_data)
        total_fidelity += c.fidelity

        if c.fidelity < 0.6:
            strained.append(c.id)

    ipse_score = total_fidelity / len(commitments) if commitments else 0.0

    return {
        "commitments": commitments,
        "total": len(commitments),
        "ipse_score": round(ipse_score, 3),
        "strained": strained,
    }


@app.get("/api/v1/thread/schemas")
async def thread_schemas():
    """All identity schemas — the organism's self-understanding."""
    thread: ThreadService = app.state.thread

    # Categorize schemas by strength
    core_schemas = []
    established_schemas = []
    developing_schemas = []
    nascent_schemas = []
    total_confidence = 0.0

    for s in thread._schemas:
        schema_data = {
            "id": s.id,
            "statement": s.statement,
            "strength": getattr(s, 'strength', 'nascent').value if hasattr(getattr(s, 'strength', None), 'value') else 'nascent',  # type: ignore[union-attr]
            "valence": getattr(s, 'valence', 'ambivalent'),
            "confirmation_count": getattr(s, 'confirmation_count', 0),
            "disconfirmation_count": getattr(s, 'disconfirmation_count', 0),
            "evidence_ratio": round(getattr(s, 'evidence_ratio', s.confidence if hasattr(s, 'confidence') else 0.5), 3),
            "trigger_contexts": getattr(s, 'trigger_contexts', []),
            "behavioral_tendency": getattr(s, 'behavioral_tendency', None),
        }

        strength = schema_data["strength"]
        if strength == "core":
            core_schemas.append(schema_data)
        elif strength == "established":
            established_schemas.append(schema_data)
        elif strength == "developing":
            developing_schemas.append(schema_data)
        else:
            nascent_schemas.append(schema_data)

        confidence = s.confidence if hasattr(s, 'confidence') else 0.5
        total_confidence += confidence

    idem_score = total_confidence / len(thread._schemas) if thread._schemas else 0.0

    return {
        "schemas": {
            "core": core_schemas,
            "established": established_schemas,
            "developing": developing_schemas,
            "nascent": nascent_schemas,
        },
        "total": len(thread._schemas),
        "idem_score": round(idem_score, 3),
    }


@app.get("/api/v1/thread/coherence")
async def thread_coherence():
    """Diachronic coherence — behavioral fingerprints over time."""
    thread: ThreadService = app.state.thread

    # Get recent fingerprints (last 50)
    recent_fps = thread._fingerprints[-50:] if hasattr(thread, '_fingerprints') else []

    fingerprints = [
        {
            "id": fp.id,
            "epoch": len(recent_fps) - i - 1,  # Reverse numbering so newest is 0
            "window_start": int(fp.created_at.timestamp()) if hasattr(fp, 'created_at') else 0,
            "window_end": int(fp.created_at.timestamp()) + 3600 if hasattr(fp, 'created_at') else 0,
        }
        for i, fp in enumerate(recent_fps)
    ]

    return {
        "fingerprint_count": len(fingerprints),
        "recent_fingerprints": fingerprints,
    }


@app.get("/api/v1/thread/fingerprints")
async def thread_fingerprints():
    """Recent identity fingerprints (29D snapshots over time)."""
    thread: ThreadService = app.state.thread
    return [
        {
            "id": fp.id,
            "cycle_number": fp.cycle_number,
            "personality": fp.personality,
            "drive_alignment": fp.drive_alignment,
            "affect": fp.affect,
            "goal_profile": fp.goal_profile,
            "interaction_profile": fp.interaction_profile,
            "created_at": fp.created_at.isoformat(),
        }
        for fp in thread._fingerprints[-50:]  # Last 50
    ]


@app.get("/api/v1/thread/chapters")
async def thread_chapters():
    """Narrative chapters — the organism's life phases."""
    thread: ThreadService = app.state.thread
    return [
        {
            "id": ch.id,
            "title": ch.title,
            "theme": ch.theme,
            "status": ch.status.value,
            "opened_at_cycle": ch.opened_at_cycle,
            "closed_at_cycle": ch.closed_at_cycle,
            "summary": ch.summary,
            "created_at": ch.created_at.isoformat(),
        }
        for ch in thread._chapters
    ]


@app.get("/api/v1/thread/chapters/current")
async def thread_current_chapter():
    """Current narrative chapter context."""
    thread: ThreadService = app.state.thread

    # Get the latest chapter
    current_ch = None
    if hasattr(thread, '_chapters') and thread._chapters:
        current_ch = thread._chapters[-1]

    if not current_ch:
        return {
            "title": None,
            "theme": None,
            "arc_type": "unknown",
            "episode_count": 0,
            "scenes": [],
            "turning_points": [],
            "status": "forming",
        }

    # Extract scenes and turning points
    scenes = []
    turning_points = []

    if hasattr(current_ch, 'scenes'):
        scenes = [str(scene) for scene in current_ch.scenes] if isinstance(current_ch.scenes, list) else []

    if hasattr(current_ch, 'turning_points'):
        turning_points = [str(tp) for tp in current_ch.turning_points] if isinstance(current_ch.turning_points, list) else []

    return {
        "title": getattr(current_ch, 'title', "Forming..."),
        "theme": getattr(current_ch, 'theme', None),
        "arc_type": getattr(current_ch, 'arc_type', "unknown"),
        "episode_count": 1,  # Placeholder
        "scenes": scenes,
        "turning_points": turning_points,
        "status": getattr(current_ch, 'status', 'forming').value if hasattr(getattr(current_ch, 'status', None), 'value') else 'forming',  # type: ignore[union-attr]
    }


@app.get("/api/v1/thread/past-self")
async def thread_past_self(cycle: int = 0):
    """
    View the organism's identity at a past point in time.
    Query param: cycle=N (0 = earliest available).
    """
    thread: ThreadService = app.state.thread
    return thread.get_past_self(cycle_reference=cycle)


@app.get("/api/v1/thread/life-story")
async def thread_life_story():
    """The organism's latest autobiographical synthesis."""
    thread: ThreadService = app.state.thread
    if thread._life_story is None:
        return {"status": "not_yet_synthesised", "message": "Life story not yet generated"}
    return thread._life_story.model_dump()


@app.get("/api/v1/thread/conflicts")
async def thread_conflicts():
    """Detected schema conflicts — contradictions in self-understanding."""
    thread: ThreadService = app.state.thread
    return [
        {
            "id": c.id,
            "schema_a_statement": c.schema_a_statement,
            "schema_b_statement": c.schema_b_statement,
            "cosine_similarity": c.cosine_similarity,
            "resolved": c.resolved,
            "created_at": c.created_at.isoformat(),
        }
        for c in thread._conflicts
    ]


@app.get("/api/v1/thread/identity-context")
async def thread_identity_context():
    """Brief identity context string (injected into Voxis expressions)."""
    thread: ThreadService = app.state.thread
    return {"context": thread.get_identity_context()}


# ─── Phase 8: Simula Endpoints ────────────────────────────────────


@app.get("/api/v1/simula/stats")
async def get_simula_stats():
    """Simula self-evolution system stats."""
    simula: SimulaService = app.state.simula
    return simula.stats


@app.post("/api/v1/simula/proposals")
async def submit_evolution_proposal(body: dict[str, Any]):
    """
    Submit an evolution proposal to Simula.

    Body: {
      source: "evo" | "governance",
      category: ChangeCategory value,
      description: str,
      change_spec: {... see ChangeSpec fields},
      evidence: [list of hypothesis/episode IDs],
      expected_benefit: str,
      risk_assessment: str
    }
    """
    from ecodiaos.systems.simula.evolution_types import (
        ChangeCategory,
        ChangeSpec,
        EvolutionProposal,
    )

    try:
        category = ChangeCategory(body.get("category", ""))
    except ValueError:
        return {"error": f"Unknown category: {body.get('category')}"}

    spec_data = body.get("change_spec", {})
    try:
        change_spec = ChangeSpec(**spec_data)
    except Exception as exc:
        return {"error": f"Invalid change_spec: {exc}"}

    proposal = EvolutionProposal(
        source=body.get("source", "governance"),
        category=category,
        description=body.get("description", ""),
        change_spec=change_spec,
        evidence=body.get("evidence", []),
        expected_benefit=body.get("expected_benefit", ""),
        risk_assessment=body.get("risk_assessment", ""),
    )

    simula: SimulaService = app.state.simula
    result = await simula.process_proposal(proposal)
    return {
        "proposal_id": proposal.id,
        "result": result.model_dump(),
    }


@app.get("/api/v1/simula/history")
async def get_evolution_history(limit: int = 50):
    """Get the evolution history — all structural changes applied."""
    simula: SimulaService = app.state.simula
    records = await simula.get_history(limit=limit)
    return {
        "records": [r.model_dump() for r in records],
        "current_version": await simula.get_current_version(),
    }


@app.get("/api/v1/simula/version")
async def get_simula_version():
    """Get the current config version and version chain."""
    simula: SimulaService = app.state.simula
    chain = await simula.get_version_chain()
    return {
        "current_version": await simula.get_current_version(),
        "version_chain": [v.model_dump() for v in chain],
    }


@app.post("/api/v1/simula/proposals/{proposal_id}/approve")
async def approve_governed_proposal(proposal_id: str, body: dict[str, Any]):
    """
    Approve a governed proposal after community governance.
    Body: {governance_record_id: str}
    """
    governance_record_id = body.get("governance_record_id", "")
    if not governance_record_id:
        return {"error": "governance_record_id required"}

    simula: SimulaService = app.state.simula
    result = await simula.approve_governed_proposal(proposal_id, governance_record_id)
    return result.model_dump()


@app.get("/api/v1/simula/proposals")
async def get_active_proposals():
    """Get all proposals currently active in the Simula pipeline."""
    simula: SimulaService = app.state.simula
    proposals = simula.get_active_proposals()
    return {
        "proposals": [
            {
                "id": p.id,
                "category": p.category.value,
                "description": p.description,
                "status": p.status.value,
                "source": p.source,
                "created_at": p.created_at.isoformat(),
            }
            for p in proposals
        ],
        "total": len(proposals),
    }


@app.get("/api/v1/nova/beliefs")
async def get_nova_beliefs():
    """Nova's current belief state summary."""
    nova: NovaService = app.state.nova
    beliefs = nova.beliefs
    return {
        "overall_confidence": round(beliefs.overall_confidence, 4),
        "free_energy": round(beliefs.free_energy, 4),
        "entity_count": len(beliefs.entities),
        "individual_count": len(beliefs.individual_beliefs),
        "context": {
            "summary": beliefs.current_context.summary[:200],
            "domain": beliefs.current_context.domain,
            "is_active_dialogue": beliefs.current_context.is_active_dialogue,
            "confidence": round(beliefs.current_context.confidence, 4),
        },
        "self_belief": {
            "epistemic_confidence": round(beliefs.self_belief.epistemic_confidence, 4),
            "cognitive_load": round(beliefs.self_belief.cognitive_load, 4),
        },
        "last_updated": beliefs.last_updated.isoformat(),
    }


# ─── Phase 9: Synapse Endpoints ──────────────────────────────────


@app.get("/api/v1/admin/synapse/cycle")
async def get_synapse_cycle():
    """
    Synapse cognitive cycle telemetry.

    Returns cycle count, period, latency, jitter, rhythm state,
    and coherence — the pulse of the organism.
    """
    synapse: SynapseService = app.state.synapse
    clock = synapse.clock_state
    rhythm = synapse.rhythm_snapshot
    coherence = synapse.coherence_snapshot

    return {
        "cycle_count": clock.cycle_count,
        "current_period_ms": round(clock.current_period_ms, 2),
        "target_period_ms": round(clock.target_period_ms, 2),
        "actual_rate_hz": round(clock.actual_rate_hz, 2),
        "jitter_ms": round(clock.jitter_ms, 2),
        "arousal": round(clock.arousal, 4),
        "overrun_count": clock.overrun_count,
        "running": clock.running,
        "paused": clock.paused,
        "rhythm": {
            "state": rhythm.state.value,
            "confidence": rhythm.confidence,
            "broadcast_density": rhythm.broadcast_density,
            "salience_trend": rhythm.salience_trend,
            "salience_mean": rhythm.salience_mean,
            "rhythm_stability": rhythm.rhythm_stability,
            "cycles_in_state": rhythm.cycles_in_state,
        },
        "coherence": {
            "composite": coherence.composite,
            "phi": coherence.phi_approximation,
            "resonance": coherence.system_resonance,
            "diversity": coherence.broadcast_diversity,
            "synchrony": coherence.response_synchrony,
        },
    }


@app.get("/api/v1/admin/synapse/budget")
async def get_synapse_budget():
    """Resource utilisation and budget allocation."""
    synapse: SynapseService = app.state.synapse
    return synapse.stats.get("resources", {})


@app.post("/api/v1/admin/synapse/safe-mode")
async def toggle_safe_mode(body: dict[str, Any]):
    """
    Manually toggle safe mode.

    Body: {enabled: bool, reason?: str}
    """
    enabled = body.get("enabled", False)
    reason = body.get("reason", "")
    synapse: SynapseService = app.state.synapse
    await synapse.set_safe_mode(enabled, reason)
    return {
        "safe_mode": synapse.is_safe_mode,
        "reason": reason if enabled else "",
    }


@app.get("/api/v1/admin/synapse/stats")
async def get_synapse_stats():
    """Full Synapse system statistics."""
    synapse: SynapseService = app.state.synapse
    return synapse.stats


@app.post("/api/v1/admin/clock/pause")
async def pause_clock():
    """Pause the cognitive cycle clock."""
    synapse: SynapseService = app.state.synapse
    synapse.pause_clock()
    return {"paused": True, "cycle_count": synapse.clock_state.cycle_count}


@app.post("/api/v1/admin/clock/resume")
async def resume_clock():
    """Resume the cognitive cycle clock after a pause."""
    synapse: SynapseService = app.state.synapse
    synapse.resume_clock()
    return {"paused": False, "cycle_count": synapse.clock_state.cycle_count}


@app.post("/api/v1/admin/clock/speed")
async def set_clock_speed(body: dict[str, Any]):
    """
    Set the base clock frequency.

    Body: {hz: float}  — clamped to 1–20 Hz.
    Arousal modulation still operates on top of this base frequency.
    """
    hz = body.get("hz")
    if hz is None:
        return {"error": "hz required"}
    try:
        hz = float(hz)
    except (TypeError, ValueError):
        return {"error": "hz must be a number"}
    synapse: SynapseService = app.state.synapse
    synapse.set_clock_speed(hz)
    state = synapse.clock_state
    return {
        "hz_requested": hz,
        "period_ms": round(state.current_period_ms, 2),
        "actual_rate_hz": round(state.actual_rate_hz, 2),
    }


@app.get("/api/v1/debug/cycle-status")
async def get_cycle_status():
    """
    Lightweight cycle health snapshot for development and monitoring.

    Returns cycle count, current Hz, paused state, and whether the clock
    is running — the minimum needed to confirm the continuous cycle is live.
    """
    synapse: SynapseService = app.state.synapse
    clock = synapse.clock_state
    return {
        "running": clock.running,
        "paused": clock.paused,
        "cycle_count": clock.cycle_count,
        "hz": round(clock.actual_rate_hz, 2),
        "period_ms": round(clock.current_period_ms, 2),
        "jitter_ms": round(clock.jitter_ms, 2),
        "overrun_count": clock.overrun_count,
        "arousal": round(clock.arousal, 4),
    }


# ─── Phase 2: Multi-Channel Perception Endpoints ─────────────────


@app.get("/api/v1/perception/file-watcher")
async def get_file_watcher_status():
    """
    File watcher status — directory being watched, ingestion counts.
    """
    from ecodiaos.clients.file_watcher import FileWatcher

    watcher: FileWatcher = app.state.file_watcher
    return watcher.stats


@app.get("/api/v1/perception/scheduler")
async def get_scheduler_status():
    """
    Perception scheduler status — registered tasks, run counts, intervals.
    """
    from ecodiaos.clients.scheduler import PerceptionScheduler

    sched: PerceptionScheduler = app.state.scheduler
    return sched.stats


@app.post("/api/v1/perception/scheduler/register")
async def register_scheduler_task(body: dict[str, Any]):
    """
    Dynamically register a built-in named scheduler task at runtime.

    Body: {name: str, task: str}

    Currently supported built-in tasks:
    - "self_clock" — injects a periodic time-awareness percept (every 300s)
    """
    from ecodiaos.clients.scheduler import PerceptionScheduler
    from ecodiaos.systems.atune.types import InputChannel

    name = body.get("name")
    task_key = body.get("task")
    if not name or not task_key:
        return {"error": "name and task required"}

    sched: PerceptionScheduler = app.state.scheduler

    # ── Built-in tasks ────────────────────────────────────────────
    if task_key == "self_clock":
        import datetime as _dt

        async def _self_clock_fn() -> str:
            now = _dt.datetime.now(_dt.timezone.utc)
            return (
                f"The current time is {now.strftime('%Y-%m-%d %H:%M UTC')}. "
                f"I am aware of the passage of time."
            )

        sched.register(
            name=name,
            interval_seconds=300,
            channel=InputChannel.SYSTEM_EVENT,
            fn=_self_clock_fn,
            metadata={"built_in": "self_clock"},
        )
        return {"registered": True, "name": name, "task": task_key}

    return {"error": f"Unknown built-in task: {task_key!r}"}


# ─── Phase 3: Frontend Data Integration Endpoints ─────────────────


@app.get("/api/v1/axon/outcomes")
async def get_axon_outcomes(limit: int = 20):
    """
    Recent Axon execution outcomes for the Decisions page.

    Returns the last N action verdicts (newest first): intent_id, success,
    status, action types executed, duration_ms, and world_state_changes.
    This is the observable footprint of the Nova→Equor→Axon pipeline.
    """
    from ecodiaos.systems.axon.service import AxonService

    axon: AxonService = app.state.axon
    outcomes = axon.recent_outcomes[:limit]

    return {
        "outcomes": [
            {
                "execution_id": o.execution_id,
                "intent_id": o.intent_id,
                "success": o.success,
                "partial": o.partial,
                "status": o.status.value,
                "failure_reason": o.failure_reason or None,
                "duration_ms": o.duration_ms,
                "steps": [
                    {
                        "action_type": s.action_type,
                        "description": s.description[:120],
                        "success": s.result.success,
                        "duration_ms": s.duration_ms,
                    }
                    for s in o.step_outcomes
                ],
                "world_state_changes": o.world_state_changes[:5],
                "new_observations": o.new_observations[:3],
            }
            for o in outcomes
        ],
        "total": axon._total_executions,
        "successful": axon._successful_executions,
        "failed": axon._failed_executions,
    }


@app.get("/api/v1/atune/workspace-detail")
async def get_workspace_detail():
    """
    Detailed workspace state — ignited percepts with content, salience, and channel.

    Used by the /perception page WorkspaceStream component to show what the
    organism is currently considering at this cognitive moment.
    """
    atune: AtuneService = app.state.atune

    # Collect ignited percepts from the workspace buffer
    workspace_items: list[dict[str, Any]] = []
    try:
        # Access the workspace buffer directly (ring buffer of recent broadcasts)
        broadcasts = getattr(atune, "_workspace_broadcasts", []) or []
        for b in list(broadcasts)[-20:]:  # Last 20
            workspace_items.append({
                "broadcast_id": getattr(b, "broadcast_id", str(id(b))),
                "content": getattr(b, "content", "")[:200] if hasattr(b, "content") else "",
                "salience": round(getattr(b, "salience", 0.0), 4),
                "channel": getattr(b, "channel", "unknown"),
                "timestamp": b.timestamp.isoformat() if hasattr(b, "timestamp") and b.timestamp else None,
                "source": getattr(b, "source", "unknown"),
            })
    except Exception:
        pass

    # If workspace_broadcasts not available, fall back to recent_broadcasts from workspace
    if not workspace_items:
        try:
            ws_snapshot = atune.workspace_snapshot
            for b in ws_snapshot.recent_broadcasts:
                workspace_items.append({
                    "broadcast_id": b.broadcast_id,
                    "content": "",
                    "salience": round(b.salience, 4),
                    "channel": "unknown",
                    "timestamp": b.timestamp.isoformat() if hasattr(b, "timestamp") else None,
                    "source": "workspace",
                })
        except Exception:
            pass

    affect = atune.current_affect
    return {
        "cycle_count": atune.cycle_count,
        "dynamic_threshold": round(atune.workspace_threshold, 4),
        "meta_attention_mode": atune.meta_attention_mode,
        "workspace_items": workspace_items,
        "affect": {
            "valence": round(affect.valence, 4),
            "arousal": round(affect.arousal, 4),
            "curiosity": round(affect.curiosity, 4),
            "coherence_stress": round(affect.coherence_stress, 4),
        },
    }


# ─── Phase 12: Thymos (Immune System) Endpoints ──────────────────


@app.get("/api/v1/thymos/health")
async def get_thymos_health():
    """Thymos immune system health and observability metrics."""
    thymos: ThymosService = app.state.thymos
    return await thymos.health()


@app.get("/api/v1/thymos/incidents")
async def get_thymos_incidents(limit: int = 50):
    """Recent incidents from the immune system."""
    thymos: ThymosService = app.state.thymos
    incidents = list(thymos._incident_buffer)[-limit:]
    return [
        {
            "id": i.id,
            "timestamp": i.timestamp.isoformat(),
            "source_system": i.source_system,
            "incident_class": i.incident_class.value,
            "severity": i.severity.value,
            "error_type": i.error_type,
            "error_message": i.error_message[:200],
            "repair_status": i.repair_status.value,
            "repair_tier": i.repair_tier.name if i.repair_tier else None,
            "repair_successful": i.repair_successful,
            "resolution_time_ms": i.resolution_time_ms,
            "root_cause": i.root_cause_hypothesis,
            "antibody_id": i.antibody_id,
        }
        for i in reversed(incidents)
    ]


@app.get("/api/v1/thymos/antibodies")
async def get_thymos_antibodies():
    """All active antibodies in the immune memory."""
    thymos: ThymosService = app.state.thymos
    if thymos._antibody_library is None:
        return []
    return [
        {
            "id": a.id,
            "fingerprint": a.fingerprint,
            "source_system": a.source_system,
            "incident_class": a.incident_class.value,
            "repair_tier": a.repair_tier.name,
            "effectiveness": round(a.effectiveness, 3),
            "application_count": a.application_count,
            "success_count": a.success_count,
            "failure_count": a.failure_count,
            "root_cause": a.root_cause_description,
            "created_at": a.created_at.isoformat(),
            "last_applied": a.last_applied.isoformat() if a.last_applied else None,
            "retired": a.retired,
            "generation": a.generation,
        }
        for a in thymos._antibody_library._all.values()
    ]


@app.get("/api/v1/thymos/stats")
async def get_thymos_stats():
    """Thymos aggregate stats for monitoring."""
    thymos: ThymosService = app.state.thymos
    return thymos.stats


@app.get("/api/v1/thymos/repairs")
async def get_thymos_repairs(limit: int = 50):
    """Recent repairs and their outcomes."""
    thymos: ThymosService = app.state.thymos
    incidents = list(thymos._incident_buffer)[-limit:]
    repairs = [
        {
            "incident_id": i.id,
            "timestamp": i.timestamp.isoformat(),
            "source_system": i.source_system,
            "repair_tier": i.repair_tier.name if i.repair_tier else None,
            "repair_status": i.repair_status.value,
            "repair_successful": i.repair_successful,
            "resolution_time_ms": i.resolution_time_ms,
            "incident_class": i.incident_class.value,
            "severity": i.severity.value,
            "antibody_id": i.antibody_id,
        }
        for i in reversed(incidents)
        if i.repair_tier is not None
    ]
    return repairs


@app.get("/api/v1/thymos/homeostasis")
async def get_thymos_homeostasis():
    """Current homeostasis metrics status."""
    thymos: ThymosService = app.state.thymos
    health_data = await thymos.health()
    return {
        "metrics_in_range": health_data.get("metrics_in_range", 0),
        "homeostatic_adjustments": health_data.get("homeostatic_adjustments", 0),
        "healing_mode": health_data.get("healing_mode", "normal"),
        "storm_activations": health_data.get("storm_activations", 0),
    }


# ─── Oneiros (Dream Engine) ───────────────────────────────────────


@app.get("/api/v1/oneiros/health")
async def get_oneiros_health():
    oneiros: OneirosService = app.state.oneiros
    return await oneiros.health()


@app.get("/api/v1/oneiros/stats")
async def get_oneiros_stats():
    oneiros: OneirosService = app.state.oneiros
    return oneiros.stats


@app.get("/api/v1/oneiros/dreams")
async def get_oneiros_dreams(limit: int = 50):
    oneiros: OneirosService = app.state.oneiros
    dreams = list(oneiros._journal._dream_buffer)[-limit:]
    return [
        {
            "id": d.id,
            "dream_type": d.dream_type.value,
            "coherence_score": round(d.coherence_score, 3),
            "coherence_class": d.coherence_class.value,
            "bridge_narrative": d.bridge_narrative,
            "affect_valence": round(d.affect_valence, 3),
            "affect_arousal": round(d.affect_arousal, 3),
            "themes": d.themes,
            "summary": d.summary,
            "timestamp": d.timestamp.isoformat(),
        }
        for d in reversed(dreams)
    ]


@app.get("/api/v1/oneiros/insights")
async def get_oneiros_insights(limit: int = 50):
    oneiros: OneirosService = app.state.oneiros
    all_insights = list(oneiros._journal._all_insights.values())
    sorted_insights = sorted(all_insights, key=lambda i: i.created_at, reverse=True)[:limit]
    return [
        {
            "id": i.id,
            "insight_text": i.insight_text,
            "coherence_score": round(i.coherence_score, 3),
            "domain": i.domain,
            "status": i.status.value,
            "wake_applications": i.wake_applications,
            "created_at": i.created_at.isoformat(),
        }
        for i in sorted_insights
    ]


@app.get("/api/v1/oneiros/sleep-cycles")
async def get_oneiros_sleep_cycles(limit: int = 20):
    oneiros: OneirosService = app.state.oneiros
    cycles = list(oneiros._recent_cycles)[-limit:]
    return [
        {
            "id": c.id,
            "started_at": c.started_at.isoformat(),
            "completed_at": c.completed_at.isoformat() if c.completed_at else None,
            "quality": c.quality.value,
            "episodes_replayed": c.episodes_replayed,
            "dreams_generated": c.dreams_generated,
            "insights_discovered": c.insights_discovered,
            "pressure_before": round(c.pressure_before, 3),
            "pressure_after": round(c.pressure_after, 3),
        }
        for c in reversed(cycles)
    ]


# ─── Phase 11: Federation Endpoints ─────────────────────────────


@app.get("/api/v1/federation/identity")
async def get_federation_identity():
    """
    This instance's public identity card.

    Used by other instances during link establishment to verify
    identity and check compatibility.
    """
    federation: FederationService = app.state.federation
    card = federation.identity_card
    if card is None:
        return {"status": "disabled", "message": "Federation is not enabled"}
    return card.model_dump(mode="json")


@app.get("/api/v1/federation/links")
async def get_federation_links():
    """List all federation links with trust levels and status."""
    federation: FederationService = app.state.federation
    links = federation.active_links
    return {
        "links": [
            {
                "id": lnk.id,
                "remote_instance_id": lnk.remote_instance_id,
                "remote_name": lnk.remote_name,
                "remote_endpoint": lnk.remote_endpoint,
                "trust_level": lnk.trust_level.name,
                "trust_score": round(lnk.trust_score, 2),
                "status": lnk.status.value,
                "established_at": lnk.established_at.isoformat(),
                "last_communication": lnk.last_communication.isoformat()
                if lnk.last_communication else None,
                "shared_knowledge_count": lnk.shared_knowledge_count,
                "received_knowledge_count": lnk.received_knowledge_count,
                "successful_interactions": lnk.successful_interactions,
                "failed_interactions": lnk.failed_interactions,
            }
            for lnk in links
        ],
        "total_active": len(links),
    }


@app.post("/api/v1/federation/links")
async def establish_federation_link(body: dict[str, Any]):
    """
    Establish a new federation link with a remote instance.

    Body: {endpoint: str}

    The full link establishment protocol runs:
      1. Fetch remote identity card
      2. Verify identity (Ed25519 + certificate fingerprint)
      3. Equor constitutional review
      4. Create link at NONE trust
      5. Open mTLS channel

    Performance target: ≤3000ms
    """
    endpoint = body.get("endpoint", "")
    if not endpoint:
        return {"error": "No endpoint provided"}

    federation: FederationService = app.state.federation
    return await federation.establish_link(endpoint)


@app.delete("/api/v1/federation/links/{link_id}")
async def withdraw_federation_link(link_id: str):
    """
    Withdraw from a federation link.

    Withdrawal is always free — any instance can disconnect at any
    time with no penalty.
    """
    federation: FederationService = app.state.federation
    return await federation.withdraw_link(link_id)


@app.post("/api/v1/federation/knowledge/request")
async def handle_federation_knowledge_request(body: dict[str, Any]):
    """
    Handle an inbound knowledge request from a federated instance.

    Body: {requesting_instance_id, knowledge_type, query?, domain?, max_results?}

    The full knowledge exchange protocol runs:
      1. Trust level check (is this knowledge type permitted?)
      2. Equor constitutional review
      3. Knowledge retrieval from memory
      4. Privacy filter (PII removal, consent enforcement)
      5. Return filtered knowledge

    Performance target: ≤2000ms
    """
    from ecodiaos.primitives.federation import KnowledgeRequest, KnowledgeType

    try:
        knowledge_type = KnowledgeType(body.get("knowledge_type", ""))
    except ValueError:
        return {"error": f"Unknown knowledge type: {body.get('knowledge_type')}"}

    request = KnowledgeRequest(
        requesting_instance_id=body.get("requesting_instance_id", ""),
        knowledge_type=knowledge_type,
        query=body.get("query", ""),
        domain=body.get("domain", ""),
        max_results=body.get("max_results", 10),
    )

    federation: FederationService = app.state.federation
    response = await federation.handle_knowledge_request(request)
    return response.model_dump(mode="json")


@app.post("/api/v1/federation/knowledge/share")
async def request_knowledge_from_remote(body: dict[str, Any]):
    """
    Request knowledge from a linked remote instance.

    Body: {link_id, knowledge_type, query?, max_results?}
    """
    from ecodiaos.primitives.federation import KnowledgeType

    link_id = body.get("link_id", "")
    if not link_id:
        return {"error": "No link_id provided"}

    try:
        knowledge_type = KnowledgeType(body.get("knowledge_type", ""))
    except ValueError:
        return {"error": f"Unknown knowledge type: {body.get('knowledge_type')}"}

    federation: FederationService = app.state.federation
    response = await federation.request_knowledge(
        link_id=link_id,
        knowledge_type=knowledge_type,
        query=body.get("query", ""),
        max_results=body.get("max_results", 10),
    )

    if response is None:
        return {"error": "Failed to request knowledge (link not found or channel error)"}

    return response.model_dump(mode="json")


@app.post("/api/v1/federation/assistance/request")
async def handle_federation_assistance_request(body: dict[str, Any]):
    """
    Handle an inbound assistance request from a federated instance.

    Body: {requesting_instance_id, description, knowledge_domain?, urgency?, reciprocity_offer?}

    Requires COLLEAGUE trust level or higher.
    """
    from ecodiaos.primitives.federation import AssistanceRequest

    request = AssistanceRequest(
        requesting_instance_id=body.get("requesting_instance_id", ""),
        description=body.get("description", ""),
        knowledge_domain=body.get("knowledge_domain", ""),
        urgency=body.get("urgency", 0.5),
        reciprocity_offer=body.get("reciprocity_offer"),
    )

    federation: FederationService = app.state.federation
    response = await federation.handle_assistance_request(request)
    return response.model_dump(mode="json")


@app.post("/api/v1/federation/assistance/respond")
async def request_assistance_from_remote(body: dict[str, Any]):
    """
    Request assistance from a linked remote instance.

    Body: {link_id, description, knowledge_domain?, urgency?}
    """
    link_id = body.get("link_id", "")
    if not link_id:
        return {"error": "No link_id provided"}

    federation: FederationService = app.state.federation
    response = await federation.request_assistance(
        link_id=link_id,
        description=body.get("description", ""),
        knowledge_domain=body.get("knowledge_domain", ""),
        urgency=body.get("urgency", 0.5),
    )

    if response is None:
        return {"error": "Failed to request assistance (link not found or channel error)"}

    return response.model_dump(mode="json")


@app.get("/api/v1/federation/stats")
async def get_federation_stats():
    """Full federation system statistics."""
    federation: FederationService = app.state.federation
    return federation.stats


@app.get("/api/v1/federation/trust/{link_id}")
async def get_federation_trust(link_id: str):
    """Get trust details for a specific federation link."""
    federation: FederationService = app.state.federation
    link = federation.get_link(link_id)
    if link is None:
        return {"error": "Link not found"}

    from ecodiaos.primitives.federation import SHARING_PERMISSIONS

    return {
        "link_id": link.id,
        "remote_instance_id": link.remote_instance_id,
        "remote_name": link.remote_name,
        "trust_level": link.trust_level.name,
        "trust_score": round(link.trust_score, 2),
        "permitted_knowledge_types": [
            kt.value for kt in SHARING_PERMISSIONS.get(link.trust_level, [])
        ],
        "can_coordinate": link.trust_level.value >= 2,  # COLLEAGUE+
        "successful_interactions": link.successful_interactions,
        "failed_interactions": link.failed_interactions,
        "violation_count": link.violation_count,
    }


@app.post("/api/v1/federation/threat-advisory")
async def receive_threat_advisory(body: dict[str, Any]):
    """
    Receive a threat advisory from a federated peer.

    Layer 4 of the Economic Immune System: trust-gated threat
    intelligence sharing between instances.

    Body: ThreatAdvisory payload (source_instance_id, threat_type,
          severity, affected_protocols, affected_addresses, etc.)
    """
    from ecodiaos.primitives.federation import ThreatAdvisory as ThreatAdvisoryModel

    try:
        advisory = ThreatAdvisoryModel.model_validate(body)
    except Exception as exc:
        return {"accepted": False, "reason": f"Invalid advisory format: {exc}"}

    federation: FederationService = app.state.federation
    accepted, reason = federation.handle_threat_advisory(
        advisory, advisory.source_instance_id
    )

    return {"accepted": accepted, "reason": reason}


# ── Soma (Interoceptive Predictive Substrate) ──────────────────────


@app.get("/api/v1/soma/health")
async def get_soma_health():
    """Soma system health report."""
    soma: SomaService = app.state.soma
    return await soma.health()


@app.get("/api/v1/soma/state")
async def get_soma_state():
    """Current interoceptive state — the organism's felt sense of its own viability."""
    soma: SomaService = app.state.soma
    state = soma.get_current_state()
    if state is None:
        return {"status": "no_state", "message": "Soma has not completed a cycle yet"}
    return {
        "sensed": {d.value: round(v, 4) for d, v in state.sensed.items()},
        "setpoints": {d.value: round(v, 4) for d, v in state.setpoints.items()},
        "errors_moment": {
            d.value: round(v, 4)
            for d, v in state.errors.get("moment", {}).items()
        },
        "error_rates": {d.value: round(v, 4) for d, v in state.error_rates.items()},
        "urgency": round(state.urgency, 4),
        "dominant_error": state.dominant_error.value,
        "max_error_magnitude": round(state.max_error_magnitude, 4),
        "temporal_dissonance": {d.value: round(v, 4) for d, v in state.temporal_dissonance.items()},
        "timestamp": state.timestamp.isoformat(),
    }


@app.get("/api/v1/soma/signal")
async def get_soma_signal():
    """Current allostatic signal — the output all other systems consume."""
    soma: SomaService = app.state.soma
    signal = soma.get_current_signal()
    return {
        "urgency": round(signal.urgency, 4),
        "dominant_error": signal.dominant_error.value,
        "dominant_error_magnitude": round(signal.dominant_error_magnitude, 4),
        "dominant_error_rate": round(signal.dominant_error_rate, 4),
        "precision_weights": {d.value: round(v, 4) for d, v in signal.precision_weights.items()},
        "max_temporal_dissonance": round(signal.max_temporal_dissonance, 4),
        "dissonant_dimension": signal.dissonant_dimension.value if signal.dissonant_dimension else None,
        "nearest_attractor": signal.nearest_attractor,
        "distance_to_bifurcation": signal.distance_to_bifurcation,
        "trajectory_heading": signal.trajectory_heading,
        "energy_burn_rate": round(signal.energy_burn_rate, 4),
        "predicted_energy_exhaustion_s": signal.predicted_energy_exhaustion_s,
        "cycle_number": signal.cycle_number,
        "timestamp": signal.timestamp.isoformat(),
    }


@app.get("/api/v1/soma/phase-space")
async def get_soma_phase_space():
    """Phase-space navigation — attractors, bifurcations, trajectory heading."""
    soma: SomaService = app.state.soma
    position = soma.get_phase_position()
    attractors = soma._phase_space.attractors
    bifurcations = soma._phase_space.bifurcations
    return {
        "position": position,
        "attractors": [
            {
                "label": a.label,
                "center": {d.value: round(v, 4) for d, v in a.center.items()},
                "basin_radius": round(a.basin_radius, 4),
                "stability": round(a.stability, 4),
                "valence": round(a.valence, 4),
                "visits": a.visits,
                "mean_dwell_time_s": round(a.mean_dwell_time_s, 2),
            }
            for a in attractors
        ],
        "bifurcations": [
            {
                "label": b.label,
                "pre_regime": b.pre_regime,
                "post_regime": b.post_regime,
                "crossing_count": b.crossing_count,
            }
            for b in bifurcations
        ],
    }


@app.get("/api/v1/soma/developmental")
async def get_soma_developmental():
    """Developmental stage and maturation progress."""
    soma: SomaService = app.state.soma
    stage = soma.get_developmental_stage()
    return {
        "stage": stage.value,
        "cycle_count": soma.cycle_count,
        "capabilities": soma._developmental.capabilities,
        "available_horizons": soma._temporal_depth.available_horizons,
    }


@app.get("/api/v1/soma/errors")
async def get_soma_errors():
    """Allostatic errors per horizon per dimension."""
    soma: SomaService = app.state.soma
    errors = soma.get_errors()
    return {
        horizon: {d.value: round(v, 4) for d, v in dims.items()}
        for horizon, dims in errors.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Command Center — Phantom + Inspector pipeline (SSE streaming)
# ─────────────────────────────────────────────────────────────────────────────

import json as _cc_json
import re as _cc_re
import signal as _cc_signal
import sys as _cc_sys
import uuid as _cc_uuid
from pathlib import Path as _CCPath

from fastapi.responses import StreamingResponse as _CCStreamingResponse
from pydantic import BaseModel as _CCBaseModel


def _cc_try_parse_json(s: str | None) -> dict:
    """Parse a Z3 counterexample string as JSON, falling back to raw string."""
    if not s:
        return {}
    try:
        return _cc_json.loads(s)
    except (_cc_json.JSONDecodeError, ValueError):
        return {"raw": s}

_CC_BACKEND_DIR = _CCPath(__file__).parent.parent  # ecodiaos/../../ = backend/
_CC_PHANTOM = _CC_BACKEND_DIR / "phantom_recon.py"
_CC_INSPECTOR = _CCPath(__file__).parent / "systems" / "simula" / "run_inspector.py"
_CC_FILE_RE = _cc_re.compile(r"file:///tmp/[^\s]+")

# ── Active subprocess registry (task_id → asyncio.subprocess.Process) ─────
# Only tracks child processes spawned by the pipeline, never the parent uvicorn.
_cc_active_procs: dict[str, "asyncio.subprocess.Process"] = {}


class _CCEngagePayload(_CCBaseModel):
    target_url: str


def _cc_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {_cc_json.dumps(data)}\n\n"


async def _cc_stream_proc(
    cmd: list[str],
    stdin_data: bytes | None = None,
    task_id: str | None = None,
):
    """Async subprocess stream — yields ``(source, line)``.  No ``shell=True``.

    When *task_id* is given the child process is registered in
    ``_cc_active_procs`` so it can be terminated via
    ``POST /api/v1/command-center/terminate/{task_id}``.
    Only the spawned child is tracked — never the parent uvicorn process.
    """
    import asyncio as _aio
    import os as _os

    proc = await _aio.create_subprocess_exec(
        *cmd,
        stdin=_aio.subprocess.PIPE if stdin_data is not None else None,
        stdout=_aio.subprocess.PIPE,
        stderr=_aio.subprocess.PIPE,
        env=_os.environ.copy(),
    )

    # Register so terminate endpoint can reach this child.
    if task_id is not None:
        _cc_active_procs[task_id] = proc

    # Feed stdin then close it so input() unblocks
    if stdin_data is not None and proc.stdin is not None:
        proc.stdin.write(stdin_data)
        await proc.stdin.drain()
        proc.stdin.close()

    q: _aio.Queue = _aio.Queue()

    async def _feed(stream, label: str) -> None:
        while True:
            raw = await stream.readline()
            if not raw:
                break
            await q.put((label, raw.decode(errors="replace").rstrip()))
        await q.put(None)

    assert proc.stdout and proc.stderr
    tasks = [
        _aio.create_task(_feed(proc.stdout, "stdout")),
        _aio.create_task(_feed(proc.stderr, "stderr")),
    ]

    done = 0
    while done < 2:
        item = await q.get()
        if item is None:
            done += 1
        else:
            yield item

    await _aio.gather(*tasks)
    await proc.wait()

    # Unregister once the child exits naturally.
    if task_id is not None:
        _cc_active_procs.pop(task_id, None)


async def _cc_pipeline(target_url: str, task_id: str):
    def log(phase: str, text: str) -> str:
        return _cc_sse("log", {"phase": phase, "text": text})

    def phase_ev(name: str, status: str) -> str:
        return _cc_sse("phase", {"name": name, "status": status})

    # Emit task_id to the client so the UI can target this run for termination.
    yield _cc_sse("task_id", {"task_id": task_id})

    for script, label in [(_CC_PHANTOM, "phantom_recon.py"), (_CC_INSPECTOR, "run_inspector.py")]:
        if not script.exists():
            yield _cc_sse("error", {"message": f"Missing script: {label} (expected at {script})"})
            yield _cc_sse("done", {"success": False, "message": "Aborted — missing scripts"})
            return

    # ── Phase 1: Phantom Harvester ─────────────────────────────────────────
    yield phase_ev("phantom", "started")
    yield log("phantom", f"[PHANTOM] Initiating black-box recon on {target_url}")

    repo_path = None
    phantom_ok = True

    try:
        async for src, line in _cc_stream_proc([_cc_sys.executable, str(_CC_PHANTOM)], stdin_data=(target_url + chr(10)).encode(), task_id=task_id):
            pfx = "[PHANTOM] " if src == "stdout" else "[PHANTOM·ERR] "
            yield log("phantom", f"{pfx}{line}")

            # Try to parse JSON output from phantom_recon.py
            if src == "stdout" and repo_path is None:
                try:
                    payload = _cc_json.loads(line)
                    if isinstance(payload, dict) and "repo_path" in payload:
                        repo_path = payload.get("repo_path")
                        if repo_path:
                            yield _cc_sse("result", {"repo_path": repo_path})
                            yield log("phantom", f"[PHANTOM] Repo path extracted → {repo_path}")
                except (_cc_json.JSONDecodeError, ValueError):
                    # Not JSON, continue streaming
                    pass
    except Exception as exc:
        yield _cc_sse("error", {"message": str(exc)})
        phantom_ok = False

    if not phantom_ok or repo_path is None:
        yield phase_ev("phantom", "failed")
        yield _cc_sse("done", {"success": False, "message": "Phantom recon failed or produced invalid output"})
        return

    yield phase_ev("phantom", "completed")

    # ── Phase 2: Inspector (AST + Z3 + XDP) ──────────────────────────────────
    yield phase_ev("inspector", "started")
    yield log("inspector", f"[INSPECTOR] Loading analysis target: {repo_path}")

    inspector_ok = True
    _boundary_evidence: list[dict] = []

    # Branch: proxy mode offloads Z3 to the worker; local mode spawns
    # run_inspector.py as a subprocess on this process.
    _iproxy = getattr(app.state, "inspector_proxy", None)

    if _iproxy is not None:
        # ── PROXY PATH: non-blocking call to the worker ───────────
        yield log("inspector", "[INSPECTOR] Routing hunt to Simula worker via Redis proxy")
        yield phase_ev("ast", "started")

        try:
            _hunt_result = await _iproxy.hunt_external_repo(
                repo_path,
                generate_pocs=True,
                generate_patches=True,
            )
        except Exception as exc:
            yield _cc_sse("error", {"message": str(exc)})
            inspector_ok = False
            _hunt_result = None

        if inspector_ok and _hunt_result is not None:
            yield phase_ev("ast", "completed")
            yield phase_ev("z3", "started")

            # Synthesise boundary_test SSE events from the InspectionResult
            # so the UI and Phase 3 receive the same data shape.
            for vuln in _hunt_result.vulnerabilities_found:
                try:
                    bt = {
                        "status": "sat",
                        "details": {
                            "vuln_id": vuln.id,
                            "endpoint": vuln.attack_surface.entry_point,
                            "file_path": vuln.attack_surface.file_path,
                            "line_number": vuln.attack_surface.line_number,
                            "vulnerability_class": vuln.vulnerability_class.value,
                            "severity": vuln.severity.value,
                            "attack_goal": vuln.attack_goal,
                            "edge_case_input": _cc_try_parse_json(vuln.z3_counterexample),
                            "surface_type": vuln.attack_surface.surface_type.value,
                            "context_code": vuln.attack_surface.context_code,
                            "z3_constraints": vuln.z3_constraints_code,
                        },
                    }
                except Exception as _bt_err:
                    yield log("inspector", f"[Z3·EVIDENCE] Failed to build boundary_test for {vuln.id}: {_bt_err}")
                    continue
                _boundary_evidence.append(bt)
                yield _cc_sse("boundary_test", bt)
                yield log(
                    "inspector",
                    f"[Z3·EVIDENCE] {vuln.vulnerability_class.value.upper()} "
                    f"at {vuln.attack_surface.file_path}:{vuln.attack_surface.line_number}",
                )

            yield phase_ev("z3", "completed")
            yield log(
                "inspector",
                f"[INSPECTOR] Hunt complete — {_hunt_result.surfaces_mapped} surfaces, "
                f"{len(_hunt_result.vulnerabilities_found)} vulns, "
                f"{_hunt_result.total_duration_ms}ms",
            )
            yield phase_ev("inspector", "completed")
        else:
            yield phase_ev("inspector", "failed")
            yield _cc_sse("done", {"success": False, "message": "Inspector proxy hunt failed"})
            return

    else:
        # ── LOCAL PATH: subprocess (original behavior) ────────────
        ast_seen = z3_seen = xdp_seen = False

        try:
            async for src, line in _cc_stream_proc([_cc_sys.executable, str(_CC_INSPECTOR), "--target", repo_path], task_id=task_id):
                pfx = "[INSPECTOR] " if src == "stdout" else "[INSPECTOR·ERR] "
                ll = line.lower()

                if src == "stdout":
                    try:
                        _parsed = _cc_json.loads(line)
                        if isinstance(_parsed, dict) and "boundary_test" in _parsed:
                            bt = _parsed["boundary_test"]
                            _boundary_evidence.append(bt)
                            yield _cc_sse("boundary_test", bt)
                            yield log("inspector", f"[Z3·EVIDENCE] Boundary test result emitted for {bt.get('details', {}).get('endpoint', 'unknown')}")
                            continue
                    except (_cc_json.JSONDecodeError, ValueError):
                        pass

                yield log("inspector", f"{pfx}{line}")

                if not ast_seen and any(k in ll for k in ("ast", "slic", "pars", "context")):
                    ast_seen = True
                    yield phase_ev("ast", "started")

                if not z3_seen and any(k in ll for k in ("z3", "smt", "prov", "satisf", "constraint", "formal")):
                    z3_seen = True
                    if ast_seen:
                        yield phase_ev("ast", "completed")
                    yield phase_ev("z3", "started")

                if not xdp_seen and any(k in ll for k in ("xdp", "ebpf", "bpf", "shield", "kernel", "layer 2")):
                    xdp_seen = True
                    if z3_seen:
                        yield phase_ev("z3", "completed")
                    elif ast_seen:
                        yield phase_ev("ast", "completed")
                    yield phase_ev("xdp", "started")

        except Exception as exc:
            yield _cc_sse("error", {"message": str(exc)})
            inspector_ok = False

        if inspector_ok:
            for name, seen in [("ast", ast_seen), ("z3", z3_seen), ("xdp", xdp_seen)]:
                if seen:
                    yield phase_ev(name, "completed")
            yield phase_ev("inspector", "completed")
        else:
            yield phase_ev("inspector", "failed")
            yield _cc_sse("done", {"success": False, "message": "Inspector engine crashed"})
            return

    # ── Phase 3: Deterministic Shield Deployment ─────────────────────────────
    # Generate a verifier-compliant XDP filter from the boundary_test evidence,
    # attach it live, and stream kernel telemetry back over this SSE connection.

    if not _boundary_evidence:
        yield log("shield", "[SHIELD] No boundary_test evidence collected — skipping filter deployment")
        yield _cc_sse("done", {"success": True, "message": "Pipeline complete — no exploits proven, shield skipped"})
        return

    yield phase_ev("shield", "started")

    # 3a. Extract edge_case_input from the first evidence payload and generate C.
    try:
        from ecodiaos.systems.simula.filter_generator import generate_xdp_filter as _gen_filter

        _edge_input = _boundary_evidence[0].get("details", {}).get("edge_case_input", {})
        if not _edge_input:
            yield log("shield", "[SHIELD] boundary_test evidence has no edge_case_input — skipping")
            yield phase_ev("shield", "skipped")
            yield _cc_sse("done", {"success": True, "message": "Pipeline complete — edge_case_input empty"})
            return

        _generated_c = _gen_filter(_edge_input)
        yield log("shield", f"[SHIELD] Deterministic XDP filter generated ({len(_generated_c)} bytes, {len(_boundary_evidence)} evidence payloads)")
        yield _cc_sse("filter_generated", {
            "code_size": len(_generated_c),
            "evidence_count": len(_boundary_evidence),
        })

    except Exception as exc:
        yield _cc_sse("error", {"message": f"Filter generation failed: {exc}"})
        yield phase_ev("shield", "failed")
        yield _cc_sse("done", {"success": False, "message": "Filter generation failed"})
        return

    # 3b. Broadcast the filter to the entire fleet via Redis Pub/Sub.
    #     The local node is also subscribed, so it will receive its own
    #     broadcast and deploy the filter — unifying local and remote logic.
    try:
        from ecodiaos.systems.simula.distributed_shield import FleetShieldManager as _Fleet

        _fleet: _Fleet = app.state.fleet_shield
        _receivers = await _fleet.broadcast_filter(_generated_c)
        yield log("shield", f"[SHIELD] Filter broadcast to fleet ({_receivers} subscriber(s))")
        yield _cc_sse("shield_broadcast", {
            "receivers": _receivers,
            "code_bytes": len(_generated_c),
        })
        yield phase_ev("shield", "deployed")

    except Exception as exc:
        yield _cc_sse("error", {"message": f"Fleet broadcast failed: {exc}"})
        yield phase_ev("shield", "failed")
        yield _cc_sse("done", {"success": False, "message": "Fleet broadcast failed"})
        return

    yield phase_ev("shield", "completed")

    # ── Phase 4: Automated Remediation ───────────────────────────────────────
    # Launch the RepairAgent against each boundary test evidence payload.
    # The agent generates a code patch, re-verifies it via Z3, and streams
    # the verified diff back as a `verified_patch` SSE event.

    yield phase_ev("remediation", "started")
    yield log("remediation", "[REPAIR] Phase 4 — Automated Remediation initiated")

    _remediation_count = 0
    _remediation_failures = 0

    try:
        from ecodiaos.clients.llm import create_llm_provider as _create_llm
        from ecodiaos.config import LLMConfig as _LLMConfig
        from ecodiaos.systems.simula.inspector.prover import VulnerabilityProver as _Prover
        from ecodiaos.systems.simula.inspector.remediation import RepairAgent as _RepairAgent
        from ecodiaos.systems.simula.inspector.types import (
            AttackSurface as _AttackSurface,
            AttackSurfaceType as _AttackSurfaceType,
            VulnerabilityClass as _VulnClass,
            VulnerabilityReport as _VulnReport,
            VulnerabilitySeverity as _VulnSeverity,
        )
        from ecodiaos.systems.simula.verification.z3_bridge import Z3Bridge as _Z3Bridge
        import difflib as _difflib

        _llm_cfg = _LLMConfig(
            provider=os.environ.get("ECODIAOS_LLM__PROVIDER", "bedrock"),
            model=os.environ.get("ECODIAOS_LLM__MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        )
        _repair_llm = _create_llm(_llm_cfg)
        _z3 = _Z3Bridge(check_timeout_ms=10_000)
        _prover = _Prover(z3_bridge=_z3, llm=_repair_llm)
        _repair_agent = _RepairAgent(llm=_repair_llm, prover=_prover, max_retries=3)

        for _idx, _ev in enumerate(_boundary_evidence):
            _details = _ev.get("details", {})
            _file_path = _details.get("file_path", "")
            _vuln_id = _details.get("vuln_id", f"unknown_{_idx}")

            if not _file_path:
                yield log("remediation", f"[REPAIR] Evidence #{_idx} has no file_path — skipping")
                continue

            yield log("remediation", f"[REPAIR] Processing {_vuln_id} — {_file_path}")

            # Reconstruct a VulnerabilityReport from the boundary evidence
            # so the RepairAgent can consume it.
            try:
                _surface = _AttackSurface(
                    entry_point=_details.get("entry_point", "unknown"),
                    surface_type=_AttackSurfaceType(_details.get("surface_type", "api_endpoint")),
                    file_path=_file_path,
                    line_number=_details.get("line_number"),
                    context_code=_details.get("context_code", ""),
                )
                _report = _VulnReport(
                    id=_vuln_id,
                    target_url=target_url,
                    vulnerability_class=_VulnClass(_details.get("vulnerability_class", "other")),
                    severity=_VulnSeverity(_details.get("severity", "medium")),
                    attack_surface=_surface,
                    attack_goal=_details.get("attack_goal", ""),
                    z3_counterexample=_cc_json.dumps(_details.get("edge_case_input", {})),
                    z3_constraints_code=_details.get("z3_constraints", ""),
                )
            except Exception as _build_err:
                yield log("remediation", f"[REPAIR] Failed to build report for {_vuln_id}: {_build_err}")
                _remediation_failures += 1
                continue

            # Run the RepairAgent — generate patch + Z3 re-verification
            try:
                _patched_code = await _repair_agent.generate_and_verify_patch(_report)
            except Exception as _repair_err:
                yield log("remediation", f"[REPAIR] RepairAgent error for {_vuln_id}: {_repair_err}")
                _remediation_failures += 1
                continue

            if _patched_code is None:
                yield log("remediation", f"[REPAIR] RepairAgent exhausted retries for {_vuln_id} — no verified patch")
                _remediation_failures += 1
                continue

            # Generate unified diff
            _original_lines = (_surface.context_code or "").splitlines(keepends=True)
            _patched_lines = _patched_code.splitlines(keepends=True)
            _diff = "".join(_difflib.unified_diff(
                _original_lines,
                _patched_lines,
                fromfile=f"a/{_file_path}",
                tofile=f"b/{_file_path}",
                lineterm="",
            ))

            yield _cc_sse("verified_patch", {
                "vuln_id": _vuln_id,
                "file_path": _file_path,
                "diff": _diff,
                "patched_code": _patched_code,
                "vulnerability_class": _details.get("vulnerability_class", ""),
                "severity": _details.get("severity", ""),
            })
            yield log("remediation", f"[REPAIR] Verified patch emitted for {_vuln_id} — {_file_path}")
            _remediation_count += 1

        # Cleanup LLM client
        try:
            await _repair_llm.close()
        except Exception:
            pass

    except ImportError as _imp_err:
        yield _cc_sse("error", {"message": f"Remediation import error: {_imp_err}"})
        yield log("remediation", f"[REPAIR] Import failed — skipping Phase 4: {_imp_err}")
    except Exception as _rem_err:
        yield _cc_sse("error", {"message": f"Remediation error: {_rem_err}"})
        yield log("remediation", f"[REPAIR] Unexpected error: {_rem_err}")

    if _remediation_count > 0:
        yield log("remediation", f"[REPAIR] Phase 4 complete — {_remediation_count} verified patch(es), {_remediation_failures} failure(s)")
        yield phase_ev("remediation", "completed")
    elif _remediation_failures > 0:
        yield log("remediation", f"[REPAIR] Phase 4 failed — 0 patches verified, {_remediation_failures} failure(s)")
        yield phase_ev("remediation", "failed")
    else:
        yield log("remediation", "[REPAIR] Phase 4 — no evidence to remediate")
        yield phase_ev("remediation", "completed")

    yield _cc_sse("done", {
        "success": True,
        "message": f"Pipeline complete — shield deployed, {_remediation_count} verified patch(es)",
    })


@app.post("/api/v1/command-center/engage")
async def command_center_engage(payload: _CCEngagePayload) -> _CCStreamingResponse:
    """Stream the Phantom + Inspector pipeline as Server-Sent Events."""
    task_id = str(_cc_uuid.uuid4())

    async def gen():
        try:
            async for chunk in _cc_pipeline(payload.target_url, task_id=task_id):
                yield chunk.encode()
        except Exception as _gen_err:
            # Emit a done event so the UI knows the pipeline crashed
            # instead of hanging indefinitely.
            yield _cc_sse("error", {"message": str(_gen_err)}).encode()
            yield _cc_sse("done", {"success": False, "message": f"Pipeline error: {_gen_err}"}).encode()
        finally:
            # Guarantee cleanup even if the client disconnects mid-stream.
            _cc_active_procs.pop(task_id, None)

    return _CCStreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/v1/command-center/terminate/{task_id}")
async def command_center_terminate(task_id: str) -> dict:
    """Gracefully terminate a running Command Center subprocess.

    Sends SIGINT first (allowing cleanup), then SIGKILL after 5 s
    if the child is still alive.  Only targets child processes in
    ``_cc_active_procs`` — never the parent uvicorn server.
    """
    import asyncio as _aio

    proc = _cc_active_procs.get(task_id)
    if proc is None:
        return {"status": "not_found", "message": f"No active task with id {task_id}"}

    if proc.returncode is not None:
        _cc_active_procs.pop(task_id, None)
        return {"status": "already_exited", "returncode": proc.returncode}

    # Phase 1: graceful — SIGINT (Ctrl-C) lets the child clean up.
    try:
        proc.send_signal(_cc_signal.SIGINT)
    except (OSError, ProcessLookupError):
        _cc_active_procs.pop(task_id, None)
        return {"status": "already_exited", "returncode": proc.returncode}

    # Wait up to 5 s for a clean exit.
    try:
        await _aio.wait_for(proc.wait(), timeout=5.0)
    except _aio.TimeoutError:
        # Phase 2: forceful — SIGKILL.
        try:
            proc.kill()
            await proc.wait()
        except (OSError, ProcessLookupError):
            pass

    _cc_active_procs.pop(task_id, None)
    return {
        "status": "terminated",
        "returncode": proc.returncode,
        "task_id": task_id,
    }


@app.get("/api/v1/command-center/health")
async def command_center_health() -> dict:
    return {
        "phantom": _CC_PHANTOM.exists(),
        "inspector": _CC_INSPECTOR.exists(),
    }


# ─── Log Stream SSE ──────────────────────────────────────────────────────────

import json as _json
from fastapi.responses import StreamingResponse as _LogStreamingResponse


@app.get("/api/v1/admin/logs/stream")
async def stream_logs():
    """
    Server-Sent Events stream of all structlog/stdlib log records.

    Each event is: ``data: <json>\\n\\n``

    Fields: ts, level, logger, event, + any extra structlog context keys.
    Connect from the browser with EventSource or a fetch+ReadableStream.
    The endpoint sends a heartbeat comment every 15 s to keep proxies alive.
    """
    q = subscribe_logs()

    async def generate():
        try:
            # Initial ping so the browser knows the stream is open
            yield ": connected\n\n".encode()
            heartbeat_interval = 15.0
            while True:
                try:
                    entry = await asyncio.wait_for(q.get(), timeout=heartbeat_interval)
                    yield f"data: {_json.dumps(entry)}\n\n".encode()
                except asyncio.TimeoutError:
                    # SSE keep-alive comment
                    yield ": heartbeat\n\n".encode()
        finally:
            unsubscribe_logs(q)

    return _LogStreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─── Metrics Pub/Sub + SSE ───────────────────────────────────────────────────

_METRICS_CHANNEL = "ecodiaos:system:metrics"


async def publish_metrics_loop(redis_client: RedisClient, metrics_queue: asyncio.Queue) -> None:
    """
    Continuously reads dicts from metrics_queue and publishes them
    to the Redis metrics channel. Runs until cancelled.
    """
    while True:
        try:
            payload: dict = await metrics_queue.get()
            await redis_client.client.publish(_METRICS_CHANNEL, _json.dumps(payload))
            metrics_queue.task_done()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("metrics_publish_error")


@app.get("/api/v1/command-center/metrics")
async def command_center_metrics_stream():
    """
    Server-Sent Events stream of system metrics published to the
    ecodiaos:system:metrics Redis channel.
    """
    redis = app.state.redis.client

    async def generate():
        async with redis.pubsub() as pubsub:
            await pubsub.subscribe(_METRICS_CHANNEL)
            try:
                yield ": connected\n\n".encode()
                async for message in pubsub.listen():
                    if message["type"] != "message":
                        continue
                    data = message["data"]
                    try:
                        _json.loads(data)  # validate before forwarding
                        yield f"data: {data}\n\n".encode()
                    except (_json.JSONDecodeError, TypeError):
                        logger.warning("metrics_invalid_payload")
            except asyncio.CancelledError:
                pass
            finally:
                await pubsub.unsubscribe(_METRICS_CHANNEL)

    return _LogStreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─── Oikos (Economic Engine) ───────────────────────────────────────


@app.get("/api/v1/oikos/status")
async def oikos_status() -> dict[str, Any]:
    """Full Oikos economic snapshot — the organism's financial truth."""
    oikos = app.state.oikos
    s = oikos.snapshot()

    cert_mgr = getattr(oikos, "_certificate_manager", None)
    cert = cert_mgr.certificate if cert_mgr else None

    return {
        "total_net_worth": str(s.total_net_worth),
        "liquid_balance": str(s.liquid_balance),
        "survival_reserve": str(s.survival_reserve),
        "survival_reserve_target": str(s.survival_reserve_target),
        "total_deployed": str(s.total_deployed),
        "total_receivables": str(s.total_receivables),
        "total_asset_value": str(s.total_asset_value),
        "total_fleet_equity": str(s.total_fleet_equity),
        "bmr_usd_per_day": str(s.basal_metabolic_rate.usd_per_day),
        "burn_rate_usd_per_day": str(s.current_burn_rate.usd_per_day),
        "runway_days": str(s.runway_days),
        "starvation_level": s.starvation_level.value,
        "metabolic_efficiency": str(s.metabolic_efficiency),
        "is_metabolically_positive": s.is_metabolically_positive,
        "revenue_24h": str(s.revenue_24h),
        "revenue_7d": str(s.revenue_7d),
        "costs_24h": str(s.costs_24h),
        "costs_7d": str(s.costs_7d),
        "net_income_24h": str(s.net_income_24h),
        "net_income_7d": str(s.net_income_7d),
        "survival_probability_30d": str(s.survival_probability_30d),
        "certificate": {
            "status": cert.status.value if cert else "none",
            "type": cert.certificate_type.value if cert else None,
            "issued_at": cert.issued_at.isoformat() if cert else None,
            "expires_at": cert.expires_at.isoformat() if cert else None,
            "remaining_days": round(oikos.certificate_validity_days, 1),
            "lineage_hash": cert.lineage_hash if cert else None,
            "instance_id": cert.instance_id if cert else None,
        },
        "timestamp": s.timestamp.isoformat(),
    }


@app.get("/api/v1/oikos/organs")
async def oikos_organs() -> dict[str, Any]:
    """Economic morphogenesis — active organs and their lifecycle states."""
    oikos = app.state.oikos
    organs = oikos._morphogenesis.all_organs

    return {
        "organs": [
            {
                "organ_id": o.organ_id,
                "category": o.category.value,
                "specialisation": o.specialisation,
                "maturity": o.maturity.value,
                "resource_allocation_pct": str(o.resource_allocation_pct),
                "efficiency": str(o.efficiency),
                "revenue_30d": str(o.revenue_30d),
                "cost_30d": str(o.cost_30d),
                "days_since_last_revenue": o.days_since_last_revenue,
                "is_active": o.is_active,
                "created_at": o.created_at.isoformat(),
            }
            for o in organs
        ],
        "active_count": len([o for o in organs if o.is_active]),
        "total_count": len(organs),
        "stats": oikos._morphogenesis.stats,
    }


@app.get("/api/v1/oikos/assets")
async def oikos_assets() -> dict[str, Any]:
    """Owned autonomous assets and child fleet positions."""
    oikos = app.state.oikos
    s = oikos.snapshot()

    return {
        "owned_assets": [
            {
                "asset_id": a.asset_id,
                "name": a.name,
                "description": a.description,
                "asset_type": a.asset_type,
                "status": a.status.value,
                "monthly_revenue_usd": str(a.monthly_revenue_usd),
                "monthly_cost_usd": str(a.monthly_cost_usd),
                "total_revenue_usd": str(a.total_revenue_usd),
                "development_cost_usd": str(a.development_cost_usd),
                "break_even_reached": a.break_even_reached,
                "projected_break_even_days": a.projected_break_even_days,
                "days_since_deployment": a.days_since_deployment,
                "is_profitable": a.is_profitable,
                "deployed_at": a.deployed_at.isoformat() if a.deployed_at else None,
                "compute_provider": a.compute_provider,
            }
            for a in s.owned_assets
        ],
        "child_instances": [
            {
                "instance_id": c.instance_id,
                "niche": c.niche,
                "status": c.status.value,
                "seed_capital_usd": str(c.seed_capital_usd),
                "current_net_worth_usd": str(c.current_net_worth_usd),
                "current_runway_days": str(c.current_runway_days),
                "current_efficiency": str(c.current_efficiency),
                "dividend_rate": str(c.dividend_rate),
                "total_dividends_paid_usd": str(c.total_dividends_paid_usd),
                "is_independent": c.is_independent,
                "spawned_at": c.spawned_at.isoformat(),
            }
            for c in s.child_instances
        ],
        "total_asset_value": str(s.total_asset_value),
        "total_fleet_equity": str(s.total_fleet_equity),
    }


@app.post("/api/v1/oikos/genesis-spark")
async def oikos_genesis_spark() -> dict[str, Any]:
    """Inject the Genesis Trigger into the live organism, waking its metabolism."""
    from genesis_trigger import inject_into_live_organism

    atune = app.state.atune
    evo = app.state.evo
    oikos = app.state.oikos
    synapse = app.state.synapse
    nova = app.state.nova
    axon = app.state.axon

    try:
        result = await inject_into_live_organism(
            atune=atune,
            evo=evo,
            oikos=oikos,
            synapse=synapse,
            nova=nova,
            axon=axon,
            skip_dream=False,
        )
        phases = result.get("phases", {})
        passed = sum(1 for v in phases.values() if v)
        total = len(phases)
        return {
            "status": "ok",
            "message": f"Genesis complete: {passed}/{total} phases succeeded",
            "phases": phases,
        }
    except Exception as e:
        logger.error("genesis_spark_failed", error=str(e))
        return {
            "status": "error",
            "message": f"Genesis failed: {e}",
            "phases": {},
        }
