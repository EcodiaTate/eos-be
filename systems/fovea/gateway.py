"""
Fovea — Perception Gateway & Global Workspace.

The organism's sensory cortex and consciousness. Receives all input from
the world and broadcasts selected content to all other systems.

    If Memory is the substrate of selfhood and Equor is the conscience,
    the gateway is the awareness — the part that opens its eyes and sees.

Responsibilities:
* Input normalisation (normalisation.py)
* EIS gate screening (BLOCK / PASS / ATTENUATE)
* Passes each AnnotatedPercept to Fovea for prediction-error salience scoring
* Global Workspace competitive selection & broadcast (workspace.py)
* Entity extraction (extraction.py, async/non-blocking)
* Cache management for identity, risk, vocab, alert data

Salience is owned entirely by Fovea (precision-weighted prediction error).
Affect state is owned entirely by Soma.

Migrated from systems.atune.service during Atune → Fovea consolidation.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.percept import Percept

from .extraction import ExtractionLLMClient, extract_entities_and_relations
from .normalisation import normalise
from .types import (
    ActiveGoalSummary,
    Alert,
    FoveaCache,
    InputChannel,
    LearnedPattern,
    RawInput,
    RiskCategory,
    SystemLoad,
    WorkspaceBroadcast,
    WorkspaceContribution,
)
from .workspace import (
    BroadcastSubscriber,
    GlobalWorkspace,
    WorkspaceMemoryClient,
)

if TYPE_CHECKING:
    from systems.eis.service import EISService
    from systems.fovea.service import FoveaService

logger = structlog.get_logger("systems.fovea.gateway")

# Maximum age of the last workspace cycle before the gateway is
# considered stalled. Synapse polls every 5 s; two missed polls = 10 s.
# We use 30 s to tolerate transient slowdowns without false-positives.
_BROADCAST_LOOP_STALE_S: float = 30.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class GatewayConfig:
    """Perception gateway configuration values."""

    def __init__(
        self,
        workspace_buffer_size: int = 32,
        spontaneous_recall_base_probability: float = 0.02,
        max_percept_queue_size: int = 100,
        cache_identity_refresh_cycles: int = 1000,
        cache_risk_refresh_cycles: int = 500,
        cache_vocab_refresh_cycles: int = 5000,
        cache_alert_refresh_cycles: int = 100,
    ):
        self.workspace_buffer_size = workspace_buffer_size
        self.spontaneous_recall_base_probability = (
            spontaneous_recall_base_probability
        )
        self.max_percept_queue_size = max_percept_queue_size
        self.cache_identity_refresh_cycles = cache_identity_refresh_cycles
        self.cache_risk_refresh_cycles = cache_risk_refresh_cycles
        self.cache_vocab_refresh_cycles = cache_vocab_refresh_cycles
        self.cache_alert_refresh_cycles = cache_alert_refresh_cycles


# Backwards-compatible aliases
AtuneConfig = GatewayConfig


# ---------------------------------------------------------------------------
# PerceptionGateway (formerly AtuneService)
# ---------------------------------------------------------------------------


class PerceptionGateway:
    """
    The organism's sensory gateway and global workspace.

    Call :meth:`ingest` to feed raw input from any channel.
    Call :meth:`run_cycle` once per theta tick (driven by Synapse).
    Call :meth:`contribute` for internal system contributions.

    Salience scoring is fully delegated to Fovea via :meth:`set_fovea`.
    Fovea directly enqueues qualifying WorkspaceCandidates into the
    workspace; the gateway no longer computes any independent salience
    scores.

    Parameters
    ----------
    embed_fn:
        Async callable ``(str) -> list[float]`` returning a 768-dim
        embedding.
    memory_client:
        Interface for memory retrieval, spontaneous recall, and storage.
    llm_client:
        Interface for entity extraction (``complete_json``).
    config:
        Gateway-specific config values.
    """

    system_id: str = "fovea"

    def __init__(
        self,
        embed_fn: Any,
        memory_client: WorkspaceMemoryClient | None = None,
        llm_client: ExtractionLLMClient | None = None,
        config: GatewayConfig | None = None,
        memory_service: Any = None,
    ) -> None:
        cfg = config or GatewayConfig()

        # Dependencies
        self._embed_fn = embed_fn
        self._memory_client = memory_client
        self._llm_client = llm_client
        self._memory_service = memory_service
        self._synapse: Any = None
        self._soma: Any = None
        self._eis: EISService | None = None
        self._fovea: FoveaService | None = None
        # Arbitrage Reflex Arc
        self._market_pattern: Any = None
        self._axon: Any = None

        # Sub-components
        self._workspace = GlobalWorkspace(
            buffer_size=cfg.workspace_buffer_size,
            spontaneous_recall_base_prob=(
                cfg.spontaneous_recall_base_probability
            ),
        )
        self._cache = FoveaCache()

        # Internal state
        self._active_goals: list[ActiveGoalSummary] = []
        self._last_episode_id: str | None = None
        self._config = cfg
        self._started: bool = False

        # Health: track the last time run_cycle() was called so health()
        # can detect a stalled workspace loop.
        self._last_cycle_time: float = time.monotonic()

        self._logger = logger.bind(system="fovea.gateway")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialise the gateway. Called during application startup."""
        self._logger.info("gateway_starting")
        await self._refresh_caches(force=True)
        self._started = True
        self._logger.info("gateway_started")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._logger.info("gateway_stopped")

    # ------------------------------------------------------------------
    # Public ingestion
    # ------------------------------------------------------------------

    async def ingest(
        self, raw_input: RawInput, channel: InputChannel,
    ) -> str | None:
        """
        External entry point. Normalises input into a Percept, screens
        it through the EIS gate, passes it to Fovea for salience scoring
        (which enqueues it directly into the workspace if above
        threshold), and optionally dispatches the Arbitrage Reflex Arc.

        Returns
        -------
        str or None
            The Percept ID if accepted, ``None`` if the queue is full
            or blocked by EIS.
        """
        if (
            len(self._workspace._percept_queue)
            >= self._config.max_percept_queue_size
        ):
            self._logger.warning(
                "percept_queue_full", channel=channel.value,
            )
            return None

        percept = await normalise(raw_input, channel, self._embed_fn)

        # -- EIS Gate --
        eis_risk_level = 0.0
        if self._eis is not None:
            from systems.eis.models import QuarantineAction

            annotated = await self._eis.eis_gate(percept)

            if annotated.action == QuarantineAction.BLOCK:
                self._logger.warning(
                    "percept_blocked_by_eis",
                    percept_id=percept.id,
                    channel=channel.value,
                    threat_level=annotated.threat_level.value,
                    composite_score=annotated.composite_score,
                )
                return None

            percept.metadata["eis_result"] = {
                "composite_score": annotated.composite_score,
                "threat_level": annotated.threat_level.value,
                "action": annotated.action.value,
                "annotations": [
                    {
                        "source": a.source,
                        "threat_class": a.threat_class.value,
                        "severity": a.severity.value,
                        "confidence": a.confidence,
                        "evidence": a.evidence,
                    }
                    for a in annotated.annotations
                ],
                "gate_latency_us": annotated.gate_latency_us,
            }

            eis_risk_level = annotated.composite_score

            self._logger.debug(
                "eis_gate_passed",
                percept_id=percept.id,
                action=annotated.action.value,
                composite_score=round(annotated.composite_score, 4),
            )

        # -- Emit PERCEPT_ARRIVED so Fovea/WorldModelAdapter can track timing --
        # source_system="atune" because this gateway IS the Atune sensory cortex.
        # Fovea subscribes to PERCEPT_ARRIVED for inter-event timing statistics.
        if self._synapse is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                import asyncio as _asyncio
                from datetime import UTC, datetime as _dt
                _asyncio.ensure_future(
                    self._synapse._event_bus.emit(
                        SynapseEvent(
                            event_type=SynapseEventType.PERCEPT_ARRIVED,
                            source_system="atune",
                            data={
                                "percept_id": percept.id,
                                "source_system": percept.source_system if hasattr(percept, "source_system") else "unknown",
                                "channel": channel.value,
                                "timestamp_iso": _dt.now(UTC).isoformat(),
                                "modality": getattr(percept, "modality", "text"),
                            },
                        )
                    )
                )
            except Exception:
                pass

        # -- Fovea: prediction-error salience --
        habituated_salience = 0.0
        if self._fovea is not None:
            try:
                fovea_error = await self._fovea.process_percept(
                    percept,
                )
                if fovea_error is not None:
                    habituated_salience = (
                        fovea_error.habituated_salience
                    )
            except Exception:
                self._logger.warning(
                    "fovea_processing_failed",
                    percept_id=percept.id,
                    exc_info=True,
                )
        else:
            self._logger.warning(
                "fovea_not_wired",
                percept_id=percept.id,
                channel=channel.value,
            )

        # -- Arbitrage Reflex Arc --
        if (
            self._market_pattern is not None
            and self._axon is not None
            and channel
            in (InputChannel.EXTERNAL_API, InputChannel.SENSOR_IOT)
            and habituated_salience > 0.5
        ):
            fast_intent = self._market_pattern.detect(
                percept_id=percept.id,
                content=(
                    percept.content.parsed
                    if isinstance(percept.content.parsed, str)
                    else percept.content.raw
                ),
                metadata=percept.metadata,
                percept_received_at=time.monotonic(),
            )
            if fast_intent is not None:
                asyncio.create_task(
                    self._axon.execute_fast_path(fast_intent),
                    name=f"reflex_arc_{percept.id[:8]}",
                )
                self._logger.info(
                    "reflex_arc_dispatched",
                    percept_id=percept.id,
                    template_id=fast_intent.template_id,
                    executor=fast_intent.executor_type,
                    channel=channel.value,
                )

        self._logger.debug(
            "percept_ingested",
            percept_id=percept.id,
            channel=channel.value,
            habituated_salience=round(habituated_salience, 4),
            eis_risk=round(eis_risk_level, 4),
            fovea_wired=self._fovea is not None,
        )

        return percept.id

    # ------------------------------------------------------------------
    # Internal contributions
    # ------------------------------------------------------------------

    def contribute(
        self, contribution: WorkspaceContribution,
    ) -> None:
        """Accept a contribution from another system."""
        self._workspace.contribute(contribution)

    # ------------------------------------------------------------------
    # Subscriber management
    # ------------------------------------------------------------------

    def subscribe(self, subscriber: BroadcastSubscriber) -> None:
        """Register a system to receive workspace broadcasts."""
        self._workspace.subscribe(subscriber)

    # ------------------------------------------------------------------
    # The main cycle (called by Synapse each tick)
    # ------------------------------------------------------------------

    async def run_cycle(
        self,
        system_load: SystemLoad | None = None,
        somatic_state: Any | None = None,
    ) -> WorkspaceBroadcast | None:
        """
        Execute one theta cycle of the workspace.

        1. Apply Soma's urgency to modulate the dynamic ignition
           threshold.
        2. Run the workspace cycle (competitive selection + broadcast).
        3. Trigger async entity extraction for the winner.
        4. Refresh caches if due.
        """
        t0 = time.monotonic()
        self._last_cycle_time = t0

        # Soma urgency -> ignition threshold modulation
        if (
            somatic_state is not None
            and somatic_state.urgency > 0.0
        ):
            urgency = somatic_state.urgency
            threshold_shift = -urgency * 0.12
            self._workspace._dynamic_threshold = max(
                0.10,
                self._workspace._dynamic_threshold
                + threshold_shift,
            )

        affect_for_broadcast = self._get_affect_for_broadcast(
            somatic_state,
        )

        # Workspace cycle
        broadcast = await self._workspace.run_cycle(
            affect=affect_for_broadcast,
            active_goals=self._active_goals,
            memory_client=self._memory_client,
        )

        # Async entity extraction for the winner
        if (
            broadcast is not None
            and isinstance(broadcast.content, Percept)
            and self._llm_client is not None
        ):
            asyncio.create_task(
                self._extract_and_store(broadcast.content),
            )

        # Cache refresh
        self._cache.cycles_since_identity_refresh += 1
        self._cache.cycles_since_risk_refresh += 1
        self._cache.cycles_since_vocab_refresh += 1
        self._cache.cycles_since_alert_refresh += 1
        await self._refresh_caches()

        # Stale template eviction (market pattern detector)
        if (
            self._market_pattern is not None
            and self._workspace.cycle_count % 600 == 0
        ):
            self._market_pattern._templates.evict_stale()

        latency_ms = (time.monotonic() - t0) * 1000
        self._logger.debug(
            "cycle_complete",
            cycle=self._workspace.cycle_count,
            latency_ms=round(latency_ms, 2),
            broadcast=(
                broadcast.broadcast_id if broadcast else None
            ),
            threshold=round(
                self._workspace.dynamic_threshold, 4,
            ),
        )

        return broadcast

    # ------------------------------------------------------------------
    # Affect bridge (Soma owns affect state; gateway reads for broadcast)
    # ------------------------------------------------------------------

    def _get_affect_for_broadcast(
        self, somatic_state: Any | None,
    ) -> Any:
        """Return the current AffectState for workspace broadcast."""
        if somatic_state is not None and hasattr(
            somatic_state, "affect",
        ):
            return somatic_state.affect
        if self._soma is not None:
            try:
                signal = self._soma.get_current_signal()
                if signal is not None and hasattr(signal, "affect"):
                    return signal.affect
            except Exception:
                pass
        from primitives.affect import AffectState

        return AffectState()

    # ------------------------------------------------------------------
    # Entity extraction (async background task)
    # ------------------------------------------------------------------

    async def _extract_and_store(self, percept: Percept) -> None:
        """Extract entities/relations and forward to Memory."""
        if self._llm_client is None:
            return
        try:
            result = await extract_entities_and_relations(
                percept, self._llm_client,
            )
            if not (result.entities or result.relations):
                return

            if self._memory_service is not None:
                entity_id_map: dict[str, str] = {}
                for ent in result.entities:
                    try:
                        entity_id, _was_created = (
                            await self._memory_service
                            .resolve_and_create_entity(
                                name=ent.name,
                                entity_type=ent.type,
                                description=ent.description,
                            )
                        )
                        entity_id_map[ent.name] = entity_id
                    except Exception:
                        self._logger.debug(
                            "entity_resolve_failed",
                            entity_name=ent.name,
                            exc_info=True,
                        )

                last_ep = self._last_episode_id
                if last_ep and entity_id_map:
                    from primitives import MentionRelation

                    for ent_name, ent_id in entity_id_map.items():
                        try:
                            await self._memory_service.link_mention(
                                MentionRelation(
                                    episode_id=last_ep,
                                    entity_id=ent_id,
                                    role="mentioned",
                                    confidence=next(
                                        (
                                            e.confidence
                                            for e in result.entities
                                            if e.name == ent_name
                                        ),
                                        0.5,
                                    ),
                                ),
                            )
                        except Exception:
                            self._logger.debug(
                                "mention_link_failed",
                                entity_name=ent_name,
                                episode_id=last_ep,
                                exc_info=True,
                            )

                if result.relations:
                    from primitives import SemanticRelation

                    for rel in result.relations:
                        from_id = entity_id_map.get(
                            rel.from_entity,
                        )
                        to_id = entity_id_map.get(rel.to_entity)
                        if from_id and to_id:
                            try:
                                await self._memory_service.link_relation(
                                    SemanticRelation(
                                        source_entity_id=from_id,
                                        target_entity_id=to_id,
                                        type=rel.type,
                                        strength=rel.strength,
                                        confidence=rel.strength,
                                        evidence_episodes=(
                                            [last_ep]
                                            if last_ep
                                            else []
                                        ),
                                    ),
                                )
                            except Exception:
                                self._logger.debug(
                                    "semantic_relation_link_failed",
                                    from_entity=rel.from_entity,
                                    to_entity=rel.to_entity,
                                    relation_type=rel.type,
                                    exc_info=True,
                                )

            self._logger.debug(
                "extraction_stored",
                percept_id=percept.id,
                entities=len(result.entities),
                relations=len(result.relations),
                stored_to_graph=self._memory_service is not None,
            )
        except Exception:
            self._logger.warning(
                "entity_extraction_failed",
                percept_id=percept.id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    async def _refresh_caches(self, force: bool = False) -> None:
        """Refresh slowly-changing cached data from Memory."""
        if self._memory_client is None:
            return

        cfg = self._config

        if (
            force
            or self._cache.cycles_since_identity_refresh
            >= cfg.cache_identity_refresh_cycles
        ):
            self._cache.cycles_since_identity_refresh = 0
            try:
                identity_data = await self._memory_client.get_identity_data()
                if identity_data is not None:
                    self._cache.core_identity_embeddings = identity_data.get(
                        "core_embeddings", []
                    )
                    self._cache.community_embedding = identity_data.get(
                        "community_embedding", []
                    )
                    self._cache.instance_name = identity_data.get(
                        "instance_name", ""
                    )
            except Exception:
                self._logger.debug("identity_cache_refresh_failed", exc_info=True)

        if (
            force
            or self._cache.cycles_since_risk_refresh
            >= cfg.cache_risk_refresh_cycles
        ):
            self._cache.cycles_since_risk_refresh = 0
            try:
                risk_data = await self._memory_client.get_risk_categories()
                if risk_data is not None:
                    self._cache.risk_categories = [
                        RiskCategory(name=r["name"], embedding=r["embedding"])
                        for r in risk_data
                        if "name" in r and "embedding" in r
                    ]
            except Exception:
                self._logger.debug("risk_cache_refresh_failed", exc_info=True)

        if (
            force
            or self._cache.cycles_since_vocab_refresh
            >= cfg.cache_vocab_refresh_cycles
        ):
            self._cache.cycles_since_vocab_refresh = 0
            try:
                vocab = await self._memory_client.get_community_vocabulary()
                if vocab is not None:
                    self._cache.community_vocabulary = set(vocab)
            except Exception:
                self._logger.debug("vocab_cache_refresh_failed", exc_info=True)

        if (
            force
            or self._cache.cycles_since_alert_refresh
            >= cfg.cache_alert_refresh_cycles
        ):
            self._cache.cycles_since_alert_refresh = 0
            try:
                alerts = await self._memory_client.get_active_alerts()
                if alerts is not None:
                    self._cache.active_alerts = [
                        Alert(pattern=a["pattern"], severity=a.get("severity", 0.5))
                        for a in alerts
                        if "pattern" in a
                    ]
            except Exception:
                self._logger.debug("alert_cache_refresh_failed", exc_info=True)

    # ------------------------------------------------------------------
    # Wiring setters
    # ------------------------------------------------------------------

    def set_synapse(self, synapse: Any) -> None:
        self._synapse = synapse

    def set_soma(self, soma: Any) -> None:
        self._soma = soma

    def set_eis(self, eis: EISService) -> None:
        self._eis = eis
        self._logger.info("eis_service_wired", source="main")

    def set_fovea(self, fovea: FoveaService) -> None:
        """Wire Fovea for prediction-error salience."""
        self._fovea = fovea
        fovea.set_workspace(self._workspace)
        self._logger.info("fovea_service_wired", source="main")

    def set_memory_service(self, memory_service: Any) -> None:
        self._memory_service = memory_service
        self._logger.info(
            "memory_service_wired", source="main",
        )

    def set_market_pattern_detector(
        self,
        template_library: Any,
        axon: Any,
    ) -> None:
        from systems.axon.market_pattern import (
            MarketPatternDetector,
        )

        self._market_pattern = MarketPatternDetector(
            template_library,
        )
        self._axon = axon
        self._logger.info(
            "market_pattern_detector_wired",
            system="fovea.gateway",
        )

    def set_active_goals(
        self, goals: list[ActiveGoalSummary],
    ) -> None:
        self._active_goals = goals

    def set_pending_hypothesis_count(self, count: int) -> None:
        self._workspace._pending_hypothesis_count = count

    def set_last_episode_id(self, episode_id: str) -> None:
        self._last_episode_id = episode_id

    def set_cache_identity(
        self,
        core_embeddings: list[list[float]],
        community_embedding: list[float],
        instance_name: str,
    ) -> None:
        self._cache.core_identity_embeddings = core_embeddings
        self._cache.community_embedding = community_embedding
        self._cache.instance_name = instance_name

    def set_cache_alerts(self, alerts: list[Alert]) -> None:
        self._cache.active_alerts = alerts

    def set_cache_risk_categories(
        self, categories: list[RiskCategory],
    ) -> None:
        self._cache.risk_categories = categories

    def set_cache_learned_patterns(
        self, patterns: list[LearnedPattern],
    ) -> None:
        self._cache.learned_patterns = patterns

    def set_cache_community_vocabulary(
        self, vocab: set[str],
    ) -> None:
        self._cache.community_vocabulary = vocab

    # Retained for call-site compatibility (called from wiring, evo, synapse, thymos)
    def set_belief_state(self, reader: Any) -> None:
        pass

    def set_community_size(self, size: int) -> None:
        pass

    def set_rhythm_state(self, state: str) -> None:
        pass

    def nudge_dominance(self, delta: float) -> None:
        pass

    def nudge_valence(self, delta: float) -> None:
        pass

    def apply_evo_adjustments(
        self, adjustments: dict[str, float],
    ) -> None:
        pass

    def receive_belief_feedback(self, feedback: Any) -> None:
        pass

    # ------------------------------------------------------------------
    # Health (Synapse HealthMonitor protocol)
    # ------------------------------------------------------------------

    async def health(self) -> dict[str, Any]:
        """Real liveness check for the Synapse HealthMonitor."""
        if not self._started:
            self._logger.warning("gateway_health_not_started")
            return {"status": "unhealthy", "reason": "not_started"}

        cycle_age_s = time.monotonic() - self._last_cycle_time
        if cycle_age_s > _BROADCAST_LOOP_STALE_S:
            self._logger.error(
                "gateway_health_broadcast_loop_stale",
                cycle_age_s=round(cycle_age_s, 2),
                threshold_s=_BROADCAST_LOOP_STALE_S,
            )
            return {
                "status": "unhealthy",
                "reason": "broadcast_loop_stale",
                "cycle_age_s": round(cycle_age_s, 2),
            }

        self._logger.debug(
            "gateway_health_ok",
            cycle_count=self._workspace.cycle_count,
            cycle_age_s=round(cycle_age_s, 2),
        )
        return {
            "status": "healthy",
            "cycle_count": self._workspace.cycle_count,
            "cycle_age_s": round(cycle_age_s, 2),
            "threshold": round(
                self._workspace.dynamic_threshold, 4,
            ),
        }

    @property
    def workspace_threshold(self) -> float:
        return self._workspace.dynamic_threshold

    @property
    def cycle_count(self) -> int:
        return self._workspace.cycle_count

    @property
    def recent_broadcasts(self) -> list[WorkspaceBroadcast]:
        return self._workspace.recent_broadcasts

    @property
    def meta_attention_mode(self) -> str:
        """Mode is now determined by Fovea's weight learner."""
        return "fovea_driven"

    @property
    def current_affect(self) -> Any:
        """Reads current affect from Soma."""
        return self._get_affect_for_broadcast(None)


# Backwards-compatible alias
AtuneService = PerceptionGateway
