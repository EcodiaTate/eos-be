"""
EcodiaOS — Evo Service

The Learning & Hypothesis system. Evo is the Growth drive made computational.

Evo observes the stream of experience, forms hypotheses, accumulates evidence,
and — when the evidence is sufficient — adjusts the organism's parameters,
codifies successful procedures, and proposes structural changes.

It operates in two modes:
  WAKE (online)   — lightweight pattern detection during each cognitive cycle
  SLEEP (offline) — deep consolidation: schema induction, procedure extraction,
                     parameter optimisation, self-model update

Interface:
  initialize()          — build sub-systems, load persisted parameter state
  receive_broadcast()   — online learning step (called by Synapse, ≤20ms budget)
  run_consolidation()   — explicit trigger for sleep mode
  shutdown()            — graceful teardown
  get_parameter()       — current value of any tunable parameter
  stats                 — service-level metrics

Cognitive cycle role (step 7 — LEARN):
  Evo runs as a background participant. It receives every workspace broadcast,
  updates its pattern context, and occasionally triggers hypothesis generation.
  The consolidation cycle runs asynchronously and never blocks the theta rhythm.

Guard rails inherited from sub-systems:
  - Velocity limits on parameter changes
  - Hypotheses must be falsifiable
  - Cannot touch Equor evaluation logic or constitutional drives
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.clients.llm import LLMProvider
from ecodiaos.config import EvoConfig
from ecodiaos.primitives.memory_trace import Episode
from ecodiaos.systems.atune.types import WorkspaceBroadcast
from ecodiaos.systems.evo.consolidation import ConsolidationOrchestrator
from ecodiaos.systems.evo.detectors import PatternDetector, build_default_detectors
from ecodiaos.systems.evo.hypothesis import HypothesisEngine
from ecodiaos.systems.evo.parameter_tuner import ParameterTuner
from ecodiaos.systems.evo.procedure_extractor import ProcedureExtractor
from ecodiaos.systems.evo.self_model import SelfModelManager
from ecodiaos.systems.evo.types import (
    ConsolidationResult,
    HypothesisStatus,
    PatternCandidate,
    PatternContext,
    SelfModelStats,
)

if TYPE_CHECKING:
    from ecodiaos.systems.memory.service import MemoryService

logger = structlog.get_logger()

# How often to attempt hypothesis generation from accumulated patterns
_HYPOTHESIS_GENERATION_INTERVAL: int = 50   # Every 50 broadcasts
# How often to evaluate evidence against all active hypotheses
_EVIDENCE_EVALUATION_INTERVAL: int = 10     # Every 10 broadcasts


class EvoService:
    """
    Evo — the EOS learning and hypothesis system.

    Coordinates four sub-systems:
      HypothesisEngine       — hypothesis lifecycle
      ParameterTuner         — parameter adjustment with velocity limiting
      ProcedureExtractor     — action sequence → procedure codification
      SelfModelManager       — meta-cognitive self-assessment
      ConsolidationOrchestrator — sleep mode pipeline
    """

    system_id: str = "evo"

    def __init__(
        self,
        config: EvoConfig,
        llm: LLMProvider,
        memory: MemoryService | None = None,
        instance_name: str = "EOS",
    ) -> None:
        self._config = config
        self._llm = llm
        self._memory = memory
        self._instance_name = instance_name
        self._initialized: bool = False
        self._logger = logger.bind(system="evo")

        # Cross-system references (wired post-init by main.py)
        self._atune: Any = None  # AtuneService — for pushing learned head weights
        self._nova: Any = None   # NovaService — for generating epistemic goals from hypotheses
        self._voxis: Any = None  # VoxisService — for personality learning from expression outcomes
        self._soma: Any = None   # SomaService — for curiosity modulation and dynamics update

        # Sub-systems (built in initialize())
        self._hypothesis_engine: HypothesisEngine | None = None
        self._parameter_tuner: ParameterTuner | None = None
        self._procedure_extractor: ProcedureExtractor | None = None
        self._self_model: SelfModelManager | None = None
        self._orchestrator: ConsolidationOrchestrator | None = None

        # Online state
        self._detectors: list[PatternDetector] = []
        self._pattern_context: PatternContext = PatternContext()
        self._pending_candidates: list[PatternCandidate] = []

        # Cycle counters
        self._total_broadcasts: int = 0
        self._cycles_since_consolidation: int = 0
        self._total_consolidations: int = 0
        self._total_evidence_evaluations: int = 0

        # Background task handle
        self._consolidation_task: asyncio.Task[None] | None = None
        self._consolidation_in_flight: bool = False

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Build all sub-systems and load persisted parameter state.
        Must be called before any other method.
        """
        if self._initialized:
            return

        self._hypothesis_engine = HypothesisEngine(
            llm=self._llm,
            instance_name=self._instance_name,
            memory=self._memory,
        )
        self._parameter_tuner = ParameterTuner(memory=self._memory)
        self._procedure_extractor = ProcedureExtractor(
            llm=self._llm,
            memory=self._memory,
        )
        self._self_model = SelfModelManager(memory=self._memory)
        self._orchestrator = ConsolidationOrchestrator(
            hypothesis_engine=self._hypothesis_engine,
            parameter_tuner=self._parameter_tuner,
            procedure_extractor=self._procedure_extractor,
            self_model_manager=self._self_model,
            memory=self._memory,
        )

        self._detectors = build_default_detectors()

        # Restore persisted parameter values
        restored = await self._parameter_tuner.load_from_memory()

        self._initialized = True
        self._logger.info(
            "evo_initialized",
            detectors=len(self._detectors),
            parameters_restored=restored,
        )

    async def shutdown(self) -> None:
        """Graceful shutdown. Cancels any running consolidation task."""
        if self._consolidation_task and not self._consolidation_task.done():
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass

        self._logger.info(
            "evo_shutdown",
            total_broadcasts=self._total_broadcasts,
            total_consolidations=self._total_consolidations,
            total_evidence_evaluations=self._total_evidence_evaluations,
            hypothesis_stats=(
                self._hypothesis_engine.stats
                if self._hypothesis_engine else {}
            ),
        )

    # ─── Online Learning (Wake Mode) ──────────────────────────────────────────

    async def receive_broadcast(self, broadcast: WorkspaceBroadcast) -> None:
        """
        Online learning step. Called by the cognitive cycle (step 7 — LEARN).
        Budget: ≤20ms for pattern scanning. Heavy work is fire-and-forget.

        Does NOT raise — Evo failures must not interrupt the cognitive cycle.
        """
        if not self._initialized:
            return

        self._total_broadcasts += 1
        self._cycles_since_consolidation += 1

        try:
            # Update context with current broadcast data
            self._pattern_context.previous_affect = self._pattern_context.current_affect
            self._pattern_context.current_affect = broadcast.affect

            # Extract entity IDs from memory context (for CooccurrenceDetector)
            entity_ids: list[str] = []
            for trace in broadcast.context.memory_context.traces:
                entity_ids.extend(trace.entities)
            self._pattern_context.recent_entity_ids = list(set(entity_ids))[:20]

            # Run lightweight pattern scanning from the percept
            # We create a minimal Episode from broadcast for the detectors
            episode = _broadcast_to_episode(broadcast)
            await self._scan_episode_online(episode)

            # Curiosity-modulated hypothesis generation interval
            # High curiosity → generate hypotheses more aggressively
            curiosity_multiplier = 1.0
            if self._soma is not None:
                try:
                    signal = self._soma.get_current_signal()
                    curiosity_drive = signal.state.sensed.get("curiosity_drive", 0.5)
                    # Scale: 0.5 drive = 1.0x, 1.0 drive = 1.5x (generate earlier)
                    curiosity_multiplier = 0.5 + curiosity_drive * 1.0
                except Exception:
                    pass

            effective_interval = max(
                10, int(_HYPOTHESIS_GENERATION_INTERVAL / curiosity_multiplier)
            )

            # Periodically generate hypotheses from accumulated patterns
            if self._total_broadcasts % effective_interval == 0:
                asyncio.create_task(
                    self._generate_hypotheses_safe(),
                    name="evo_hypothesis_generation",
                )

            # Periodically evaluate recent episodes as evidence
            if self._total_broadcasts % _EVIDENCE_EVALUATION_INTERVAL == 0:
                asyncio.create_task(
                    self._evaluate_recent_evidence_safe(),
                    name="evo_evidence_evaluation",
                )

        except Exception as exc:
            self._logger.error("broadcast_processing_failed", error=str(exc))

    async def process_episode(self, episode: Episode) -> None:
        """
        Evaluate an episode as evidence against all active hypotheses.
        Called during evidence evaluation sweep (fire-and-forget from broadcast handler).
        Budget: per-hypothesis ≤200ms (from hypothesis_engine.evaluate_evidence).
        """
        if not self._initialized or self._hypothesis_engine is None:
            return

        active = self._hypothesis_engine.get_active()
        for h in active:
            try:
                result = await self._hypothesis_engine.evaluate_evidence(h, episode)
                self._total_evidence_evaluations += 1

                # When a hypothesis crosses into SUPPORTED, generate an
                # epistemic goal so Nova can actively explore the finding
                if (
                    result is not None
                    and result.new_status == HypothesisStatus.SUPPORTED
                    and self._nova is not None
                ):
                    await self._generate_goal_from_hypothesis(h)

            except Exception as exc:
                self._logger.warning(
                    "evidence_evaluation_error",
                    hypothesis_id=h.id,
                    error=str(exc),
                )

    async def _generate_goal_from_hypothesis(self, hypothesis) -> None:
        """
        Convert a supported hypothesis into an epistemic exploration goal.

        When Evo accumulates enough evidence to support a hypothesis, the
        organism should actively explore and test it — not just passively wait.
        """
        from ecodiaos.systems.nova.types import Goal, GoalSource, GoalStatus
        from ecodiaos.primitives.common import new_id, DriveAlignmentVector

        goal = Goal(
            id=new_id(),
            description=(
                f"Explore supported hypothesis: "
                f"{hypothesis.statement[:120]}"
            ),
            source=GoalSource.EPISTEMIC,
            priority=0.55,
            urgency=0.3,
            importance=0.6,
            drive_alignment=DriveAlignmentVector(
                coherence=0.3, care=0.0, growth=0.7, honesty=0.0,
            ),
            status=GoalStatus.ACTIVE,
        )
        try:
            await self._nova.add_goal(goal)
            self._logger.info(
                "epistemic_goal_generated",
                hypothesis_id=hypothesis.id,
                goal_id=goal.id,
            )
        except Exception as exc:
            self._logger.warning("epistemic_goal_failed", error=str(exc))

    # ─── Consolidation (Sleep Mode) ────────────────────────────────────────────

    async def run_consolidation(self) -> ConsolidationResult | None:
        """
        Trigger a consolidation cycle explicitly.
        Returns None if already running or not initialized.
        Safe to call from tests and management APIs.
        """
        if not self._initialized or self._orchestrator is None:
            return None

        if self._consolidation_task and not self._consolidation_task.done():
            self._logger.info("consolidation_already_running")
            return None

        return await self._run_consolidation_now()

    def schedule_consolidation_loop(self) -> None:
        """
        Start the background consolidation loop.
        Called once by the application startup (e.g., from main.py or Synapse).
        """
        self._consolidation_task = asyncio.create_task(
            self._consolidation_loop(),
            name="evo_consolidation_loop",
        )

    # ─── Parameter Query ──────────────────────────────────────────────────────

    def get_parameter(self, name: str) -> float | None:
        """
        Return the current value of a tunable parameter.
        Systems call this each cycle to pick up Evo-applied adjustments.
        Returns None if parameter is unknown.
        """
        if self._parameter_tuner is None:
            return None
        return self._parameter_tuner.get_current_parameter(name)

    def get_all_parameters(self) -> dict[str, float]:
        """Return all current parameter values."""
        if self._parameter_tuner is None:
            return {}
        return self._parameter_tuner.get_all_parameters()

    def get_self_model(self) -> SelfModelStats | None:
        """Return the current self-model statistics."""
        if self._self_model is None:
            return None
        return self._self_model.get_current()

    def get_capability_rate(self, capability: str) -> float | None:
        """Return the success rate for a named capability."""
        if self._self_model is None:
            return None
        return self._self_model.get_capability_rate(capability)

    def set_atune(self, atune: Any) -> None:
        """Wire Atune so Evo can push learned head-weight adjustments."""
        self._atune = atune
        self._logger.info("atune_wired_to_evo")

    def set_nova(self, nova: Any) -> None:
        """Wire Nova so supported hypotheses generate epistemic exploration goals."""
        self._nova = nova
        self._logger.info("nova_wired_to_evo")

    def set_voxis(self, voxis: Any) -> None:
        """Wire Voxis so Evo can push personality adjustments from expression outcomes."""
        self._voxis = voxis
        self._logger.info("voxis_wired_to_evo")

    def set_soma(self, soma: Any) -> None:
        """Wire Soma for curiosity modulation and dynamics learning."""
        self._soma = soma
        self._logger.info("soma_wired_to_evo")

    # ─── Thread Integration ────────────────────────────────────────────────────

    def get_pending_candidates_snapshot(self) -> list[PatternCandidate]:
        """
        Return a snapshot of current pending pattern candidates.

        Called by Thread every ~200 cycles to check for mature patterns
        that should be crystallised into identity schemas. Does NOT
        clear the candidates — that happens during hypothesis generation.
        """
        return list(self._pending_candidates)

    def on_schema_formed(
        self,
        schema_id: str,
        statement: str,
        status: str,
        source_patterns: list[str] | None = None,
    ) -> None:
        """
        Callback from Thread when a pattern crystallises into an identity schema.

        Closes the learning loop: Evo detects patterns → Thread forms schemas
        → Evo knows the pattern was internalised as identity.
        """
        self._logger.info(
            "schema_formed_notification",
            schema_id=schema_id,
            statement=statement[:80],
            status=status,
            source_patterns=source_patterns or [],
        )

    # ─── Health ────────────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Health check for the Evo system (required by Synapse health monitor)."""
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "initialized": self._initialized,
            "total_broadcasts": self._total_broadcasts,
            "total_consolidations": self._total_consolidations,
            "total_evidence_evaluations": self._total_evidence_evaluations,
            "pending_candidates": len(self._pending_candidates),
        }

    # ─── Stats ────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        hypothesis_stats = (
            self._hypothesis_engine.stats if self._hypothesis_engine else {}
        )
        tuner_stats = (
            self._parameter_tuner.stats if self._parameter_tuner else {}
        )
        extractor_stats = (
            self._procedure_extractor.stats if self._procedure_extractor else {}
        )
        consolidation_stats = (
            self._orchestrator.stats if self._orchestrator else {}
        )
        return {
            "initialized": self._initialized,
            "total_broadcasts": self._total_broadcasts,
            "cycles_since_consolidation": self._cycles_since_consolidation,
            "total_consolidations": self._total_consolidations,
            "total_evidence_evaluations": self._total_evidence_evaluations,
            "pending_candidates": len(self._pending_candidates),
            "episodes_scanned": self._pattern_context.episodes_scanned,
            "hypothesis": hypothesis_stats,
            "parameter_tuner": tuner_stats,
            "procedure_extractor": extractor_stats,
            "consolidation": consolidation_stats,
        }

    # ─── Private ──────────────────────────────────────────────────────────────

    async def _scan_episode_online(self, episode: Episode) -> None:
        """Run all online detectors on one episode. ≤20ms budget."""
        self._pattern_context.episodes_scanned += 1
        for detector in self._detectors:
            try:
                candidates = await detector.scan(episode, self._pattern_context)
                self._pending_candidates.extend(candidates)
            except Exception as exc:
                self._logger.warning(
                    "detector_failed",
                    detector=detector.name,
                    error=str(exc),
                )

    async def _generate_hypotheses_safe(self) -> None:
        """
        Fire-and-forget hypothesis generation from pending pattern candidates.
        Consumes and clears pending_candidates after generation.
        """
        if self._hypothesis_engine is None:
            return
        if not self._pending_candidates:
            return

        candidates = list(self._pending_candidates)
        self._pending_candidates.clear()

        try:
            new_hypotheses = await self._hypothesis_engine.generate_hypotheses(
                patterns=candidates,
            )
            if new_hypotheses:
                self._logger.info(
                    "hypotheses_generated",
                    count=len(new_hypotheses),
                    from_patterns=len(candidates),
                )
        except Exception as exc:
            self._logger.error("hypothesis_generation_safe_failed", error=str(exc))

    async def _evaluate_recent_evidence_safe(self) -> None:
        """
        Fire-and-forget evidence sweep: retrieve recent episodes and evaluate
        them against all active hypotheses.

        Uses active hypothesis statements as queries to find evidence that is
        specifically relevant — not a random sample. This is active evidence
        seeking: the learning system goes looking for what it needs.
        """
        if not self._initialized or self._memory is None:
            return
        if self._hypothesis_engine is None:
            return
        try:
            # Build a query from active hypothesis statements for targeted retrieval
            active = self._hypothesis_engine.get_all_active()
            if not active:
                return

            # Sample up to 3 hypotheses and use their statements as queries
            sample = active[:3]
            seen_episodes: set[str] = set()
            for h in sample:
                query = h.statement[:200] if h.statement else ""
                if not query:
                    continue
                response = await self._memory.retrieve(
                    query_text=query,
                    max_results=3,
                    salience_floor=0.0,
                )
                for trace in response.traces:
                    trace_id = getattr(trace, "node_id", None) or ""
                    if trace_id in seen_episodes:
                        continue
                    seen_episodes.add(trace_id)
                    episode = _trace_to_episode(trace)
                    await self.process_episode(episode)
        except Exception as exc:
            self._logger.warning("evidence_sweep_failed", error=str(exc))

    async def _run_consolidation_now(self) -> ConsolidationResult:
        """Execute consolidation and update counters."""
        assert self._orchestrator is not None
        try:
            result = await self._orchestrator.run(self._pattern_context)
            self._cycles_since_consolidation = 0
            self._total_consolidations += 1

            # Push learned head-weight adjustments to Atune's meta-attention
            # Evo tunes parameters like "atune.head.novelty.weight" — extract
            # the deltas and forward them so they actually take effect.
            self._push_atune_head_weights()

            # Push learned personality adjustments to Voxis
            # Evo tunes parameters like "voxis.personality.warmth" — extract
            # the deltas and forward them so Voxis personality actually evolves.
            self._push_voxis_personality()

            # If Evo discovered systematic mis-predictions in interoceptive
            # transitions during consolidation, update Soma's dynamics matrix
            self._push_soma_dynamics_update(result)

            return result
        except Exception as exc:
            self._logger.error("consolidation_run_failed", error=str(exc))
            return ConsolidationResult()

    def _push_atune_head_weights(self) -> None:
        """
        Extract atune.head.* parameters from the tuner and push them to Atune.

        Evo learns optimal head weights like "atune.head.novelty.weight" via
        parameter hypotheses. The tuner stores the current values, but Atune's
        MetaAttentionController needs the *deltas from default* to apply them.
        """
        if self._atune is None or self._parameter_tuner is None:
            return

        from ecodiaos.systems.evo.types import PARAMETER_DEFAULTS

        all_params = self._parameter_tuner.get_all_parameters()
        adjustments: dict[str, float] = {}

        for param_name, current_value in all_params.items():
            if not param_name.startswith("atune.head."):
                continue
            # Extract head name: "atune.head.novelty.weight" → "novelty"
            parts = param_name.split(".")
            if len(parts) >= 3:
                head_name = parts[2]
                default_value = PARAMETER_DEFAULTS.get(param_name, current_value)
                delta = current_value - default_value
                if abs(delta) > 0.001:
                    adjustments[head_name] = delta

        if adjustments:
            try:
                self._atune.apply_evo_adjustments(adjustments)
                self._logger.info(
                    "atune_head_weights_pushed",
                    adjustments={k: round(v, 4) for k, v in adjustments.items()},
                )
            except Exception:
                self._logger.debug("atune_head_push_failed", exc_info=True)

    def _push_voxis_personality(self) -> None:
        """
        Extract voxis.personality.* parameters from the tuner and push them to Voxis.

        Evo learns personality adjustments like "voxis.personality.warmth" via
        parameter hypotheses. The tuner stores the current values; Voxis needs
        the deltas from defaults applied via update_personality().
        """
        if self._voxis is None or self._parameter_tuner is None:
            return

        from ecodiaos.systems.evo.types import PARAMETER_DEFAULTS

        all_params = self._parameter_tuner.get_all_parameters()
        personality_deltas: dict[str, float] = {}

        for param_name, current_value in all_params.items():
            if not param_name.startswith("voxis.personality."):
                continue
            # Extract dimension: "voxis.personality.warmth" → "warmth"
            parts = param_name.split(".")
            if len(parts) >= 3:
                dimension = parts[2]
                default_value = PARAMETER_DEFAULTS.get(param_name, current_value)
                delta = current_value - default_value
                if abs(delta) > 0.001:
                    personality_deltas[dimension] = delta

        if personality_deltas:
            try:
                self._voxis.update_personality(personality_deltas)
                self._logger.info(
                    "voxis_personality_pushed",
                    dimensions={k: round(v, 4) for k, v in personality_deltas.items()},
                )
            except Exception:
                self._logger.debug("voxis_personality_push_failed", exc_info=True)

    def _push_soma_dynamics_update(self, result: ConsolidationResult) -> None:
        """
        If consolidation found systematic mis-predictions in interoceptive transitions,
        update Soma's dynamics matrix to refine the 9x9 cross-dimension coupling.

        Extracts soma.dynamics.* parameters from the tuner and pushes the
        updated coupling matrix to Soma for improved allostatic prediction.
        """
        if self._soma is None or self._parameter_tuner is None:
            return

        from ecodiaos.systems.evo.types import PARAMETER_DEFAULTS

        all_params = self._parameter_tuner.get_all_parameters()
        dynamics_updates: dict[str, float] = {}

        for param_name, current_value in all_params.items():
            if not param_name.startswith("soma.dynamics."):
                continue
            default_value = PARAMETER_DEFAULTS.get(param_name, current_value)
            delta = current_value - default_value
            if abs(delta) > 0.005:
                dynamics_updates[param_name] = current_value

        if dynamics_updates:
            try:
                self._soma.update_dynamics_matrix(dynamics_updates)
                self._logger.info(
                    "soma_dynamics_pushed",
                    updated_entries=len(dynamics_updates),
                )
            except Exception:
                self._logger.debug("soma_dynamics_push_failed", exc_info=True)

    async def _consolidation_loop(self) -> None:
        """
        Background loop that triggers consolidation based on time/cycle thresholds.
        Runs indefinitely until cancelled.
        """
        while True:
            try:
                # Poll every 60 seconds to check if consolidation is due
                await asyncio.sleep(60)

                if not self._initialized or self._orchestrator is None:
                    continue

                if self._consolidation_in_flight:
                    self._logger.debug("consolidation_still_in_flight_skipping")
                    continue

                if self._orchestrator.should_run(
                    cycle_count=self._total_broadcasts,
                    cycles_since_last=self._cycles_since_consolidation,
                ):
                    self._logger.info(
                        "consolidation_triggered",
                        cycles_since_last=self._cycles_since_consolidation,
                    )
                    self._consolidation_in_flight = True
                    try:
                        await self._run_consolidation_now()
                    finally:
                        self._consolidation_in_flight = False

            except asyncio.CancelledError:
                self._logger.info("consolidation_loop_cancelled")
                return
            except Exception as exc:
                self._logger.error("consolidation_loop_error", error=str(exc))
                await asyncio.sleep(60)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _broadcast_to_episode(broadcast: WorkspaceBroadcast) -> Episode:
    """
    Create a minimal Episode from a WorkspaceBroadcast for online scanning.
    The episode is not stored — it is used only for detector input.
    """
    from ecodiaos.primitives.common import new_id, utc_now
    from ecodiaos.primitives.memory_trace import Episode

    # Extract text from percept content if available
    content_str = ""
    if broadcast.content is not None:
        content_obj = broadcast.content
        # Try to get raw text from Percept.content.raw
        if hasattr(content_obj, "content") and hasattr(content_obj.content, "raw"):
            content_str = str(content_obj.content.raw or "")
        elif hasattr(content_obj, "raw"):
            content_str = str(content_obj.raw or "")

    source = ""
    if broadcast.content is not None and hasattr(broadcast.content, "source"):
        src = broadcast.content.source
        if hasattr(src, "channel"):
            source = f"{getattr(src, 'system', '')}.{src.channel}"

    return Episode(
        id=new_id(),
        event_time=broadcast.timestamp,
        ingestion_time=utc_now(),
        source=source,
        raw_content=content_str[:500],
        summary=content_str[:200],
        salience_composite=broadcast.salience.composite,
        salience_scores=broadcast.salience.scores,
        affect_valence=broadcast.affect.valence,
        affect_arousal=broadcast.affect.arousal,
    )


def _trace_to_episode(trace: Any) -> Episode:
    """Build a minimal Episode from a RetrievalResult for evidence evaluation."""
    from ecodiaos.primitives.common import new_id, utc_now
    from ecodiaos.primitives.memory_trace import Episode

    return Episode(
        id=str(getattr(trace, "node_id", new_id())),
        source="memory",
        raw_content=str(getattr(trace, "content", ""))[:500],
        summary=str(getattr(trace, "content", ""))[:200],
        salience_composite=float(getattr(trace, "salience", 0.0)),
        affect_valence=float(getattr(trace, "metadata", {}).get("affect_valence", 0.0)),
        affect_arousal=float(getattr(trace, "metadata", {}).get("affect_arousal", 0.0)),
    )
