"""
EcodiaOS — Application Entry Point

FastAPI application with the startup sequence defined in the
Infrastructure Architecture specification.

`docker compose up` → uvicorn ecodiaos.main:app
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ecodiaos.clients.embedding import create_embedding_client
from ecodiaos.clients.llm import create_llm_provider
from ecodiaos.clients.neo4j import Neo4jClient
from ecodiaos.clients.optimized_llm import OptimizedLLMProvider
from ecodiaos.clients.prompt_cache import PromptCache
from ecodiaos.clients.redis import RedisClient
from ecodiaos.clients.timescaledb import TimescaleDBClient
from ecodiaos.clients.token_budget import TokenBudget
from ecodiaos.config import EcodiaOSConfig, load_config, load_seed
from ecodiaos.telemetry.llm_metrics import LLMMetricsCollector
from ecodiaos.systems.memory.service import MemoryService
from ecodiaos.systems.equor.service import EquorService
from ecodiaos.systems.atune.service import AtuneService, AtuneConfig
from ecodiaos.systems.atune.types import InputChannel, RawInput
from ecodiaos.systems.voxis.service import VoxisService
from ecodiaos.systems.voxis.types import ExpressionTrigger
from ecodiaos.systems.nova.service import NovaService
from ecodiaos.systems.axon.service import AxonService
from ecodiaos.systems.evo.service import EvoService
from ecodiaos.systems.thread.service import ThreadService
from ecodiaos.systems.simula.service import SimulaService
from ecodiaos.systems.synapse.service import SynapseService
from ecodiaos.systems.thymos.service import ThymosService
from ecodiaos.systems.oneiros.service import OneirosService
from ecodiaos.systems.soma.service import SomaService
from ecodiaos.systems.federation.service import FederationService
from ecodiaos.telemetry.logging import setup_logging
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
        llm_client=llm_client,
        belief_state=None,  # Wired in step 9c after Nova initializes
        config=atune_config,
    )
    await atune.startup()
    app.state.atune = atune

    # ── 8. Initialize Voxis (Expression & Voice) ──────────────
    voxis = VoxisService(
        memory=memory,
        redis=redis_client,
        llm=llm_client,
        config=config.voxis,
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
        redis_client=redis_client,
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
            _PKEYS = [
                "warmth", "directness", "verbosity", "formality",
                "curiosity_expression", "humour", "empathy_expression",
                "confidence_display", "metaphor_use",
            ]
            pdict = dict(zip(_PKEYS, instance.personality_vector[:9]))
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
        from ecodiaos.systems.nova.types import Goal, GoalSource, GoalStatus
        from ecodiaos.primitives.common import DriveAlignmentVector, new_id

        # Try loading goals from seed config
        seed_path = os.environ.get("ECODIAOS_SEED_PATH", "config/seeds/example_seed.yaml")
        seed_goals: list[dict] = []
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
    import asyncio as _asyncio
    from pathlib import Path as _Path
    simula = SimulaService(
        config=config.simula,
        llm=llm_client,
        neo4j=neo4j_client,
        memory=memory,
        codebase_root=_Path(config.simula.codebase_root).resolve(),
        instance_name=config.instance_id,
    )
    await simula.initialize()
    app.state.simula = simula

    # ── 13. Initialize Synapse — The Heartbeat ────────────────────
    synapse = SynapseService(
        atune=atune,
        config=config.synapse,
        redis=redis_client,
        metrics=metrics,
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
    soma = SomaService(config=config.soma)
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

    if config.federation.enabled:
        # Register with Synapse for health monitoring
        synapse.register_system(federation)

    # ── 16. Start internal percept generator ───────────────────────
    # Without external input, the workspace is empty every cycle.
    # This generator provides self-monitoring percepts so EOS has
    # inner experience even when idle: affect self-observation,
    # goal reflection, and memory-prompted curiosity.
    import asyncio as _aio_gen

    _inner_life_task: _aio_gen.Task | None = None

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
        from ecodiaos.systems.atune.types import WorkspaceContribution
        import random as _rnd

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
                try:
                    await thread.on_cycle(cycle)
                except Exception:
                    pass

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

    logger.info(
        "ecodiaos_ready",
        phase="12_thymos",
        federation_enabled=config.federation.enabled,
        immune_system="active",
        inner_life="active",
    )

    yield

    # Cancel inner life before shutdown
    if _inner_life_task is not None:
        _inner_life_task.cancel()
        try:
            await _inner_life_task
        except _aio_gen.CancelledError:
            pass

    # ── Shutdown ──────────────────────────────────────────────
    logger.info("ecodiaos_shutting_down")
    await federation.shutdown()
    await alive_ws.stop()
    await thymos.shutdown()
    await synapse.stop()
    await simula.shutdown()
    await thread.shutdown()
    await evo.shutdown()
    await axon.shutdown()
    await nova.shutdown()
    await voxis.shutdown()
    await atune.shutdown()
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
            from ecodiaos.primitives.percept import Percept, Content
            from ecodiaos.primitives.common import (
                Modality, SourceDescriptor, SystemID, utc_now,
            )

            return Percept(
                source=SourceDescriptor(
                    system=SystemID.INTERNAL,
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
            episode_id = await self._memory.store_percept(
                percept=percept,
                salience_composite=salience.composite,
                salience_scores=salience.scores,
                affect_valence=affect.valence,
                affect_arousal=affect.arousal,
            )
            # Tell Atune so entity extraction can link to this episode
            # (Atune service picks this up via set_last_episode_id)
            return episode_id
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


def _resolve_governance_config(config):
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


# ─── Alive WebSocket on port 8000 (for Cloud Run) ────────────────
# Cloud Run only exposes one port per container. The standalone ws_server
# on port 8001 is unreachable, so we add a FastAPI WebSocket route here
# that taps into the same Atune + Redis data streams.

import asyncio as _ws_asyncio

import orjson as _ws_orjson
from fastapi import WebSocket, WebSocketDisconnect


def _ws_json(data: dict) -> str:
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
                    try:
                        queue.put_nowait(msg)
                    except _ws_asyncio.QueueFull:
                        pass
        except _ws_asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    async def _sender() -> None:
        """Drain the queue and send to the WebSocket client."""
        while running:
            msg = await queue.get()
            await ws.send_text(msg)

    poller_task = _ws_asyncio.create_task(_affect_poller())
    subscriber_task = _ws_asyncio.create_task(_redis_subscriber())
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
        sender_task.cancel()


# ─── API Key Authentication Middleware ─────────────────────────────
# Protects all /api/v1/* endpoints. /health is always public.
# When no API keys are configured (dev mode), all requests pass through.

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


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
        # Federation identity is public (used during link establishment)
        if path == "/api/v1/federation/identity":
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
async def health():
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

    return {
        "status": "ok",
        "budget": {
            "tier": budget_status.tier.value,
            "tokens_used": budget_status.tokens_used,
            "tokens_remaining": budget_status.tokens_remaining,
            "calls_made": budget_status.calls_made,
            "calls_remaining": budget_status.calls_remaining,
            "burn_rate_tokens_per_sec": round(budget_status.tokens_per_sec, 2),
            "hours_until_exhausted": round(budget_status.hours_until_exhausted, 2),
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
async def perceive_event(body: dict):
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

    # Run a workspace cycle immediately (until Synapse drives the clock)
    broadcast = await atune.run_cycle()

    return {
        "percept_id": percept_id,
        "accepted": True,
        "broadcast": broadcast.broadcast_id if broadcast else None,
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
async def retrieve_memory(body: dict):
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
async def review_intent(body: dict):
    """
    Submit an Intent for constitutional review (test endpoint).
    In later phases, Nova calls this automatically.

    Body: {goal, steps?, reasoning?, alternatives?, domain?, expected_free_energy?}
    """
    from ecodiaos.primitives.intent import (
        Intent, GoalDescriptor, ActionSequence, Action, DecisionTrace,
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
async def propose_amendment_endpoint(body: dict):
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
async def chat_message(body: dict):
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

        # Also feed through Atune (updates affect, workspace state)
        try:
            raw = RawInput(data=message, channel_id=conversation_id or "", metadata={})
            percept_id = await atune.ingest(raw, InputChannel.TEXT_CHAT)
            if percept_id:
                await atune.run_cycle()
        except Exception as atune_err:
            # Atune ingestion is non-critical for chat — log and continue
            _chat_logger.warning("chat_atune_ingest_failed", error=str(atune_err))

        # Generate expression via Voxis
        expression = await voxis.express(
            content=message,
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
        try:
            identity_context = app.state.thread.get_identity_context()
        except Exception:
            pass

    response: dict = {
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
        return {"status": "skipped", "reason": "Already running or not initialized"}
    return result.model_dump()


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


@app.get("/api/v1/thread/commitments")
async def thread_commitments():
    """All identity commitments — constitutional and emergent."""
    thread: ThreadService = app.state.thread
    return [
        {
            "id": c.id,
            "type": c.type.value,
            "statement": c.statement,
            "strength": c.strength.value,
            "drive_source": c.drive_source,
            "fidelity": round(c.fidelity, 3),
            "test_count": c.test_count,
            "created_at": c.created_at.isoformat(),
        }
        for c in thread._commitments
    ]


@app.get("/api/v1/thread/schemas")
async def thread_schemas():
    """All identity schemas — the organism's self-understanding."""
    thread: ThreadService = app.state.thread
    return [
        {
            "id": s.id,
            "statement": s.statement,
            "status": s.status.value,
            "confidence": round(s.confidence, 3),
            "evidence_ratio": round(s.evidence_ratio, 3),
            "created_at": s.created_at.isoformat(),
        }
        for s in thread._schemas
    ]


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
async def submit_evolution_proposal(body: dict):
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
    from ecodiaos.systems.simula.types import (
        ChangeCategory, ChangeSpec, EvolutionProposal,
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
async def approve_governed_proposal(proposal_id: str, body: dict):
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
async def toggle_safe_mode(body: dict):
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
                "id": l.id,
                "remote_instance_id": l.remote_instance_id,
                "remote_name": l.remote_name,
                "remote_endpoint": l.remote_endpoint,
                "trust_level": l.trust_level.name,
                "trust_score": round(l.trust_score, 2),
                "status": l.status.value,
                "established_at": l.established_at.isoformat(),
                "last_communication": l.last_communication.isoformat()
                if l.last_communication else None,
                "shared_knowledge_count": l.shared_knowledge_count,
                "received_knowledge_count": l.received_knowledge_count,
                "successful_interactions": l.successful_interactions,
                "failed_interactions": l.failed_interactions,
            }
            for l in links
        ],
        "total_active": len(links),
    }


@app.post("/api/v1/federation/links")
async def establish_federation_link(body: dict):
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
async def handle_federation_knowledge_request(body: dict):
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
async def request_knowledge_from_remote(body: dict):
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
async def handle_federation_assistance_request(body: dict):
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
async def request_assistance_from_remote(body: dict):
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
