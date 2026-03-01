"""
Atune — Perception, Attention & Global Workspace.

Atune is EOS's sensory cortex and its consciousness.  It receives all
input from the world, determines what matters, and broadcasts selected
content to all other systems.

    If Memory is the substrate of selfhood and Equor is the conscience,
    Atune is the awareness — the part that opens its eyes and sees.

This service class orchestrates:
* Percept normalisation (normalisation.py)
* Prediction error computation (prediction.py)
* Seven-head salience scoring (salience.py)
* Global Workspace competitive selection & broadcast (workspace.py)
* Affective state management (affect.py)
* Meta-attention (meta.py)
* Entity extraction (extraction.py, async/non-blocking)
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.core.hotreload import NeuroplasticityBus
from ecodiaos.primitives.percept import Percept

from .affect import AffectManager
from .extraction import ExtractionLLMClient, extract_entities_and_relations
from .meta import MetaAttentionController
from .normalisation import normalise
from .prediction import BeliefStateReader, compute_prediction_error
from .salience import ALL_HEADS, SalienceHead, compute_salience
from .types import (
    ActiveGoalSummary,
    Alert,
    AttentionContext,
    AtuneCache,
    InputChannel,
    LearnedPattern,
    MetaContext,
    RawInput,
    RiskCategory,
    SystemLoad,
    WorkspaceBroadcast,
    WorkspaceCandidate,
    WorkspaceContribution,
)
from .workspace import BroadcastSubscriber, GlobalWorkspace, WorkspaceMemoryClient

if TYPE_CHECKING:
    from ecodiaos.primitives.affect import AffectState

logger = structlog.get_logger("ecodiaos.systems.atune")


# ---------------------------------------------------------------------------
# Configuration (matches config.yaml schema)
# ---------------------------------------------------------------------------


class AtuneConfig:
    """Atune-specific configuration values."""

    def __init__(
        self,
        ignition_threshold: float = 0.3,
        workspace_buffer_size: int = 32,
        spontaneous_recall_base_probability: float = 0.02,
        max_percept_queue_size: int = 100,
        affect_persist_interval: int = 10,
        cache_identity_refresh_cycles: int = 1000,
        cache_risk_refresh_cycles: int = 500,
        cache_vocab_refresh_cycles: int = 5000,
        cache_alert_refresh_cycles: int = 100,
    ):
        self.ignition_threshold = ignition_threshold
        self.workspace_buffer_size = workspace_buffer_size
        self.spontaneous_recall_base_probability = spontaneous_recall_base_probability
        self.max_percept_queue_size = max_percept_queue_size
        self.affect_persist_interval = affect_persist_interval
        self.cache_identity_refresh_cycles = cache_identity_refresh_cycles
        self.cache_risk_refresh_cycles = cache_risk_refresh_cycles
        self.cache_vocab_refresh_cycles = cache_vocab_refresh_cycles
        self.cache_alert_refresh_cycles = cache_alert_refresh_cycles


# ---------------------------------------------------------------------------
# AtuneService
# ---------------------------------------------------------------------------


class AtuneService:
    """
    The organism's sensory cortex and consciousness.

    Call :meth:`ingest` to feed raw input from any channel.
    Call :meth:`run_cycle` once per theta tick (driven by Synapse).
    Call :meth:`contribute` for internal system contributions.

    Parameters
    ----------
    embed_fn:
        Async callable ``(str) -> list[float]`` that returns a 768-dim
        embedding.
    memory_client:
        Interface for memory retrieval, spontaneous recall, and storage.
    llm_client:
        Interface for entity extraction (``complete_json``).
    belief_state:
        Interface for prediction error computation.
    config:
        Atune-specific config values.
    """

    system_id: str = "atune"

    def __init__(
        self,
        embed_fn: Any,
        memory_client: WorkspaceMemoryClient | None = None,
        llm_client: ExtractionLLMClient | None = None,
        belief_state: BeliefStateReader | None = None,
        config: AtuneConfig | None = None,
        memory_service: Any = None,
        neuroplasticity_bus: NeuroplasticityBus | None = None,
    ) -> None:
        cfg = config or AtuneConfig()

        # Dependencies
        self._embed_fn = embed_fn
        self._memory_client = memory_client
        self._llm_client = llm_client
        self._belief_state = belief_state
        self._memory_service = memory_service  # Full MemoryService for entity storage
        self._bus = neuroplasticity_bus
        self._soma = None  # Soma service for allostatic signal reading

        # Sub-components
        self._workspace = GlobalWorkspace(
            ignition_threshold=cfg.ignition_threshold,
            buffer_size=cfg.workspace_buffer_size,
            spontaneous_recall_base_prob=cfg.spontaneous_recall_base_probability,
        )
        self._affect_mgr = AffectManager(persist_interval=cfg.affect_persist_interval)
        self._meta = MetaAttentionController()
        self._cache = AtuneCache()

        # Internal state
        self._heads: list[SalienceHead] = list(ALL_HEADS)
        self._active_goals: list[ActiveGoalSummary] = []
        self._pending_hypothesis_count: int = 0
        self._community_size: int = 0
        self._rhythm_state: str = "normal"
        self._last_episode_id: str | None = None  # For entity→episode linking
        self._config = cfg

        self._logger = logger.bind(system="atune")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        """Initialise Atune. Called during application startup."""
        self._logger.info("atune_starting")
        await self._refresh_caches(force=True)

        # Register with the NeuroplasticityBus for hot-reload of SalienceHead subclasses.
        if self._bus is not None:
            self._bus.register(
                base_class=SalienceHead,
                registration_callback=self._on_salience_head_evolved,
                system_id="atune",
            )

        self._logger.info("atune_started")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if self._bus is not None:
            self._bus.deregister(SalienceHead)

        self._logger.info("atune_shutting_down")
        # Persist final affect state
        if self._memory_service is not None:
            try:
                await self._memory_service.update_affect(self._affect_mgr.current)
            except Exception:
                self._logger.warning("affect_final_persist_failed", exc_info=True)
        self._logger.info("atune_stopped")

    # ------------------------------------------------------------------
    # Public ingestion
    # ------------------------------------------------------------------

    async def ingest(self, raw_input: RawInput, channel: InputChannel) -> str | None:
        """
        External entry point.  Normalises input into a Percept and enqueues
        it for the next workspace cycle.

        Returns
        -------
        str or None
            The Percept ID if accepted, ``None`` if the queue is full.
        """
        if len(self._workspace._percept_queue) >= self._config.max_percept_queue_size:
            self._logger.warning("percept_queue_full", channel=channel.value)
            return None

        percept = await normalise(raw_input, channel, self._embed_fn)

        # Pre-compute prediction error and salience so the workspace cycle
        # only needs to do selection + broadcast.
        pe = await compute_prediction_error(percept, self._belief_state, self._embed_fn)

        # Build attention context
        meta_ctx = MetaContext(
            risk_level=0.0,
            recent_broadcast_count=len(self._workspace.recent_broadcasts),
            cycles_since_last_broadcast=0,
            active_goal_count=len(self._active_goals),
            pending_hypothesis_count=self._pending_hypothesis_count,
            rhythm_state=self._rhythm_state,
        )
        head_weights = await self._meta.compute_head_weights(
            self._affect_mgr.current, meta_ctx,
        )

        context = AttentionContext(
            prediction_error=pe,
            affect_state=self._affect_mgr.current,
            active_goals=self._active_goals,
            core_identity_embeddings=self._cache.core_identity_embeddings,
            community_embedding=self._cache.community_embedding,
            source_habituation=self._workspace.habituation_map,
            risk_categories=self._cache.risk_categories,
            learned_patterns=self._cache.learned_patterns,
            community_vocabulary=self._cache.community_vocabulary,
            active_alerts=self._cache.active_alerts,
            pending_decisions=[],
            community_size=self._community_size,
            instance_name=self._cache.instance_name,
        )

        salience = await compute_salience(
            percept, context, self._affect_mgr.current, self._heads,
            head_weights=head_weights,
        )

        if salience.composite >= self._workspace._dynamic_threshold:
            self._workspace.enqueue_scored_percept(
                WorkspaceCandidate(
                    content=percept,
                    salience=salience,
                    source=f"external:{channel.value}",
                    prediction_error=pe,
                )
            )

        self._logger.debug(
            "percept_ingested",
            percept_id=percept.id,
            channel=channel.value,
            salience=round(salience.composite, 4),
            above_threshold=salience.composite >= self._workspace._dynamic_threshold,
        )

        return percept.id

    # ------------------------------------------------------------------
    # Internal contributions
    # ------------------------------------------------------------------

    def contribute(self, contribution: WorkspaceContribution) -> None:
        """Accept a contribution from another system for the next cycle."""
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
    ) -> WorkspaceBroadcast | None:
        """
        Execute one theta cycle of the workspace.

        1. Update affect state.
        2. Run the workspace cycle (selection + broadcast).
        3. Trigger async entity extraction for the winner.
        4. Refresh caches if due.

        Returns the broadcast if ignition occurred, else ``None``.
        """
        t0 = time.monotonic()
        load = system_load or SystemLoad()

        # ── Update affect ────────────────────────────────────────────
        # Peek at the top candidate for affect input (but don't drain queue)
        peek_percept: Percept | None = None
        peek_pe = None
        if self._workspace._percept_queue:
            top = self._workspace._percept_queue[0]
            if isinstance(top.content, Percept):
                peek_percept = top.content
                peek_pe = top.prediction_error

        # Get precision weights from Soma (if available)
        precision_weights: dict[str, float] = {}
        if self._soma is not None:
            try:
                signal = self._soma.get_current_signal()
                if signal is not None:
                    precision_weights = signal.precision_weights
            except Exception:
                pass  # Fall back to uniform if Soma unavailable

        await self._affect_mgr.update(peek_percept, peek_pe, load, precision_weights)

        # ── Coherence stress → threshold modulation ────────────────────
        # High coherence stress means beliefs conflict with percepts.
        # Lower the threshold to let more information in for resolution.
        if self._affect_mgr.current.coherence_stress > 0.7:
            stress_excess = self._affect_mgr.current.coherence_stress - 0.7
            # Up to -0.06 threshold reduction at max stress
            self._workspace._dynamic_threshold = max(
                0.15,
                self._workspace._dynamic_threshold - stress_excess * 0.2,
            )

        # ── Workspace cycle ──────────────────────────────────────────
        broadcast = await self._workspace.run_cycle(
            affect=self._affect_mgr.current,
            active_goals=self._active_goals,
            memory_client=self._memory_client,
        )

        # ── Async entity extraction for the winner ───────────────────
        if (
            broadcast is not None
            and isinstance(broadcast.content, Percept)
            and self._llm_client is not None
        ):
            asyncio.create_task(
                self._extract_and_store(broadcast.content)
            )

        # ── Affect persistence ───────────────────────────────────────
        if self._affect_mgr.needs_persist:
            self._affect_mgr.mark_persisted()
            if self._memory_service is not None:
                asyncio.create_task(
                    self._persist_affect_safe(),
                    name="atune_affect_persist",
                )

        # ── Cache refresh ────────────────────────────────────────────
        self._cache.cycles_since_identity_refresh += 1
        self._cache.cycles_since_risk_refresh += 1
        self._cache.cycles_since_vocab_refresh += 1
        self._cache.cycles_since_alert_refresh += 1
        await self._refresh_caches()

        # ── Observability ────────────────────────────────────────────
        latency_ms = (time.monotonic() - t0) * 1000
        self._logger.debug(
            "cycle_complete",
            cycle=self._workspace.cycle_count,
            latency_ms=round(latency_ms, 2),
            broadcast=broadcast.broadcast_id if broadcast else None,
            threshold=round(self._workspace.dynamic_threshold, 4),
            mode=self._meta.current_mode,
        )

        return broadcast

    # ------------------------------------------------------------------
    # Entity extraction (async background task)
    # ------------------------------------------------------------------

    async def _extract_and_store(self, percept: Percept) -> None:
        """Extract entities/relations and forward to Memory for graph storage."""
        if self._llm_client is None:
            return
        try:
            result = await extract_entities_and_relations(percept, self._llm_client)
            if not (result.entities or result.relations):
                return

            # Store to Memory if the memory client exposes entity storage
            if self._memory_service is not None:
                # Resolve and create each entity (handles deduplication)
                entity_id_map: dict[str, str] = {}  # name → entity_id
                for ent in result.entities:
                    try:
                        entity_id, was_created = await self._memory_service.resolve_and_create_entity(
                            name=ent.name,
                            entity_type=ent.type,
                            description=ent.description,
                        )
                        entity_id_map[ent.name] = entity_id
                    except Exception:
                        self._logger.debug(
                            "entity_resolve_failed",
                            entity_name=ent.name,
                            exc_info=True,
                        )

                # Link entities to the episode (if a last_episode_id is known)
                last_ep = getattr(self, "_last_episode_id", None)
                if last_ep and entity_id_map:
                    from ecodiaos.primitives import MentionRelation
                    for ent_name, ent_id in entity_id_map.items():
                        try:
                            await self._memory_service.link_mention(
                                MentionRelation(
                                    episode_id=last_ep,
                                    entity_id=ent_id,
                                    role="mentioned",
                                    confidence=next(
                                        (e.confidence for e in result.entities if e.name == ent_name),
                                        0.5,
                                    ),
                                )
                            )
                        except Exception:
                            pass  # Non-critical

                # Create semantic relations between extracted entities
                if result.relations:
                    from ecodiaos.primitives import SemanticRelation
                    for rel in result.relations:
                        from_id = entity_id_map.get(rel.from_entity)
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
                                        evidence_episodes=[last_ep] if last_ep else [],
                                    )
                                )
                            except Exception:
                                pass  # Non-critical

            self._logger.debug(
                "extraction_stored",
                percept_id=percept.id,
                entities=len(result.entities),
                relations=len(result.relations),
                stored_to_graph=self._memory_service is not None,
            )
        except Exception:
            self._logger.warning("entity_extraction_failed", percept_id=percept.id, exc_info=True)

    # ------------------------------------------------------------------
    # Affect persistence
    # ------------------------------------------------------------------

    async def _persist_affect_safe(self) -> None:
        """Persist current affect to the Self node for restart durability."""
        try:
            await self._memory_service.update_affect(self._affect_mgr.current)
        except Exception:
            self._logger.debug("affect_persist_failed", exc_info=True)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    async def _refresh_caches(self, force: bool = False) -> None:
        """Refresh slowly-changing cached data from Memory."""
        if self._memory_client is None:
            return

        cfg = self._config

        if force or self._cache.cycles_since_identity_refresh >= cfg.cache_identity_refresh_cycles:
            # Refresh core identity embeddings and community embedding
            # In production: fetch from memory_client
            self._cache.cycles_since_identity_refresh = 0

        if force or self._cache.cycles_since_risk_refresh >= cfg.cache_risk_refresh_cycles:
            self._cache.cycles_since_risk_refresh = 0

        if force or self._cache.cycles_since_vocab_refresh >= cfg.cache_vocab_refresh_cycles:
            self._cache.cycles_since_vocab_refresh = 0

        if force or self._cache.cycles_since_alert_refresh >= cfg.cache_alert_refresh_cycles:
            self._cache.cycles_since_alert_refresh = 0

    # ------------------------------------------------------------------
    # State setters (called by other services or API)
    # ------------------------------------------------------------------

    def set_active_goals(self, goals: list[ActiveGoalSummary]) -> None:
        """Update the active goal list (called by Nova)."""
        self._active_goals = goals

    def set_belief_state(self, reader: BeliefStateReader) -> None:
        """
        Wire in Nova's belief state for top-down prediction.

        This closes the predictive processing loop — the single most
        important architectural connection in EOS. With this wired:
          Nova beliefs → Atune prediction → prediction error → salience → broadcast → Nova
        Without it, Atune treats every percept as equally novel (magnitude 0.5).
        """
        self._belief_state = reader
        self._logger.info("belief_state_wired", source="nova")

    def set_pending_hypothesis_count(self, count: int) -> None:
        """Update pending hypothesis count (called by Evo)."""
        self._pending_hypothesis_count = count
        self._workspace._pending_hypothesis_count = count

    def set_community_size(self, size: int) -> None:
        self._community_size = size

    def set_rhythm_state(self, state: str) -> None:
        """
        Update the current cognitive rhythm state (called by Synapse).
        This feeds into meta-attention to modulate salience head weights
        based on the organism's emergent cognitive state.
        """
        self._rhythm_state = state

    def set_memory_service(self, memory_service: Any) -> None:
        """Wire the full MemoryService for entity extraction storage."""
        self._memory_service = memory_service
        self._logger.info("memory_service_wired", source="main")

    def set_last_episode_id(self, episode_id: str) -> None:
        """Track the most recent episode ID for entity→episode linking."""
        self._last_episode_id = episode_id

    def set_cache_identity(
        self,
        core_embeddings: list[list[float]],
        community_embedding: list[float],
        instance_name: str,
    ) -> None:
        """Manually set identity cache (for testing or initial boot)."""
        self._cache.core_identity_embeddings = core_embeddings
        self._cache.community_embedding = community_embedding
        self._cache.instance_name = instance_name

    def set_cache_alerts(self, alerts: list[Alert]) -> None:
        self._cache.active_alerts = alerts

    def set_cache_risk_categories(self, categories: list[RiskCategory]) -> None:
        self._cache.risk_categories = categories

    def set_cache_learned_patterns(self, patterns: list[LearnedPattern]) -> None:
        self._cache.learned_patterns = patterns

    def set_cache_community_vocabulary(self, vocab: set[str]) -> None:
        self._cache.community_vocabulary = vocab

    def set_soma(self, soma: Any) -> None:
        """Wire Soma service for allostatic signal reading."""
        self._soma = soma

    # ------------------------------------------------------------------
    # Hot-reload callbacks
    # ------------------------------------------------------------------

    def _on_salience_head_evolved(self, head: SalienceHead) -> None:
        """
        Registration callback for NeuroplasticityBus.

        Called once per SalienceHead subclass found in a hot-reloaded file.
        Replaces any existing head with the same name, or appends if new.
        """
        existing_names = [h.name for h in self._heads]
        if head.name in existing_names:
            self._heads = [
                head if h.name == head.name else h
                for h in self._heads
            ]
            self._logger.info(
                "salience_head_replaced",
                name=head.name,
                total_heads=len(self._heads),
            )
        else:
            self._heads.append(head)
            self._logger.info(
                "salience_head_added",
                name=head.name,
                total_heads=len(self._heads),
            )

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def current_affect(self) -> AffectState:
        return self._affect_mgr.current

    @property
    def meta_attention_mode(self) -> str:
        return self._meta.current_mode

    @property
    def workspace_threshold(self) -> float:
        return self._workspace.dynamic_threshold

    @property
    def cycle_count(self) -> int:
        return self._workspace.cycle_count

    @property
    def recent_broadcasts(self) -> list[WorkspaceBroadcast]:
        return self._workspace.recent_broadcasts

    def apply_evo_adjustments(self, adjustments: dict[str, float]) -> None:
        """Forward Evo's head-weight adjustments to meta-attention."""
        self._meta.apply_evo_adjustments(adjustments)

    def nudge_dominance(self, delta: float) -> None:
        """Forward Axon's dominance feedback to affect manager."""
        self._affect_mgr.nudge_dominance(delta)

    def nudge_valence(self, delta: float) -> None:
        """Forward valence feedback to affect manager."""
        self._affect_mgr.nudge_valence(delta)
