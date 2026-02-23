"""
EcodiaOS — Oneiros: The Dream Engine Service

System #13 orchestrator. Coordinates the circadian rhythm, sleep
stage transitions, NREM consolidation, REM dreaming, lucid
exploration, and wake degradation.

Thymos gave the organism a will to live. Oneiros gives it an inner life.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import Any

import structlog

from ecodiaos.primitives.common import new_id, utc_now
from ecodiaos.systems.oneiros.circadian import CircadianClock, SleepStageController
from ecodiaos.systems.oneiros.journal import DreamInsightTracker, DreamJournal
from ecodiaos.systems.oneiros.lucid import DirectedExploration, MetaCognition
from ecodiaos.systems.oneiros.nrem import (
    BeliefCompressor,
    EpisodicReplay,
    HypothesisPruner,
    SynapticDownscaler,
)
from ecodiaos.systems.oneiros.rem import (
    AffectProcessor,
    DreamGenerator,
    EthicalDigestion,
    ThreatSimulator,
)
from ecodiaos.systems.oneiros.types import (
    LucidResult,
    NREMConsolidationResult,
    OneirosHealthSnapshot,
    REMDreamResult,
    SleepCycle,
    SleepQuality,
    SleepStage,
    WakeDegradation,
)

logger = structlog.get_logger().bind(system="oneiros")


# ─── Synapse Event Types ─────────────────────────────────────────

# These will be added to SynapseEventType enum during integration.
# For now, we define string constants.
SLEEP_ONSET = "sleep_onset"
SLEEP_STAGE_CHANGED = "sleep_stage_changed"
DREAM_INSIGHT = "dream_insight"
WAKE_ONSET = "wake_onset"
SLEEP_PRESSURE_WARNING = "sleep_pressure_warning"
SLEEP_FORCED = "sleep_forced"
EMERGENCY_WAKE = "emergency_wake"


class OneirosService:
    """
    The Dream Engine — System #13.

    Coordinates the organism's circadian rhythm, manages transitions
    between states of consciousness, and orchestrates the cognitive
    work that happens during sleep: memory consolidation, creative
    dreaming, emotional processing, threat rehearsal, ethical
    digestion, and metacognitive self-observation.

    The organism that sleeps is not the same organism that wakes up.
    """

    system_id: str = "oneiros"

    def __init__(
        self,
        config: Any = None,
        synapse: Any = None,
        neo4j: Any = None,
        llm: Any = None,
        embed_fn: Any = None,
        metrics: Any = None,
    ) -> None:
        self._config = config
        self._synapse = synapse
        self._neo4j = neo4j
        self._llm = llm
        self._embed_fn = embed_fn
        self._metrics = metrics

        # Cross-system references (set via setters after construction)
        self._equor: Any = None
        self._evo: Any = None
        self._nova: Any = None
        self._atune: Any = None
        self._thymos: Any = None
        self._memory: Any = None

        # Core subsystems
        self._clock = CircadianClock(config)
        self._stage_controller = SleepStageController(config)
        self._journal = DreamJournal(neo4j)
        self._insight_tracker = DreamInsightTracker(self._journal)

        # NREM workers
        self._episodic_replay = EpisodicReplay(neo4j, llm, config)
        self._synaptic_downscaler = SynapticDownscaler(neo4j, config)
        self._belief_compressor = BeliefCompressor(None, config)  # nova set later
        self._hypothesis_pruner = HypothesisPruner(None, config)  # evo set later

        # REM workers
        self._dream_generator = DreamGenerator(neo4j, llm, embed_fn, config)
        self._affect_processor = AffectProcessor(neo4j, config)
        self._threat_simulator = ThreatSimulator(neo4j, llm, None, config)  # thymos set later
        self._ethical_digestion = EthicalDigestion(llm, None, config)  # equor set later

        # Lucid workers
        self._directed_exploration = DirectedExploration(llm, self._journal, config)
        self._metacognition = MetaCognition(self._journal, neo4j, llm)

        # State
        self._initialized: bool = False
        self._current_cycle: SleepCycle | None = None
        self._sleep_task: asyncio.Task[None] | None = None
        self._creative_goal: str | None = None

        # Lifetime metrics
        self._total_sleep_cycles: int = 0
        self._total_dreams: int = 0
        self._total_insights: int = 0
        self._insights_validated: int = 0
        self._insights_invalidated: int = 0
        self._insights_integrated: int = 0
        self._episodes_consolidated: int = 0
        self._semantic_nodes_created: int = 0
        self._traces_pruned: int = 0
        self._hypotheses_pruned: int = 0
        self._hypotheses_promoted: int = 0
        self._affect_traces_processed: int = 0
        self._threats_simulated: int = 0
        self._response_plans_created: int = 0
        self._dream_coherence_sum: float = 0.0
        self._sleep_quality_sum: float = 0.0

        # Recent sleep cycles buffer
        self._recent_cycles: deque[SleepCycle] = deque(maxlen=50)

        # Pending insights for wake broadcast
        self._pending_wake_insights: list[Any] = []

        self._logger = logger

    # ── Cross-System Wiring ───────────────────────────────────────

    def set_equor(self, equor: Any) -> None:
        self._equor = equor
        self._ethical_digestion._equor = equor

    def set_evo(self, evo: Any) -> None:
        self._evo = evo
        self._hypothesis_pruner._evo = evo

    def set_nova(self, nova: Any) -> None:
        self._nova = nova
        self._belief_compressor._nova = nova

    def set_atune(self, atune: Any) -> None:
        self._atune = atune

    def set_thymos(self, thymos: Any) -> None:
        self._thymos = thymos
        self._threat_simulator._thymos = thymos

    def set_memory(self, memory: Any) -> None:
        self._memory = memory

    # ── Lifecycle ─────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize the dream engine."""
        await self._journal.initialize()

        # Subscribe to Thymos events for emergency wake
        if self._synapse is not None:
            try:
                event_bus = self._synapse._event_bus
                # Subscribe to critical system events
                from ecodiaos.systems.synapse.types import SynapseEventType

                for event_type in (
                    SynapseEventType.SYSTEM_FAILED,
                    SynapseEventType.SAFE_MODE_ENTERED,
                ):
                    event_bus.subscribe(event_type, self._on_critical_event)
            except Exception as exc:
                self._logger.warning("synapse_subscribe_failed", error=str(exc))

        self._initialized = True
        self._logger.info("oneiros_initialized")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if self._sleep_task is not None and not self._sleep_task.done():
            self._sleep_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sleep_task

        self._logger.info(
            "oneiros_shutdown",
            total_cycles=self._total_sleep_cycles,
            total_dreams=self._total_dreams,
            total_insights=self._total_insights,
        )

    async def health(self) -> dict[str, Any]:
        """Health snapshot for Synapse."""
        snapshot = self._build_health_snapshot()
        return snapshot.model_dump()

    # ── Cognitive Cycle Hook ──────────────────────────────────────

    async def on_cycle(self, affect_valence: float = 0.0, affect_arousal: float = 0.0) -> None:
        """
        Called every cognitive cycle by the main loop.

        During WAKE: updates sleep pressure.
        During SLEEP: this is not called (sleep runs its own loop).
        """
        if self._stage_controller.is_sleeping:
            return

        # Update pressure sources
        self._clock.tick()
        self._clock.record_affect_trace(affect_valence, affect_arousal)
        self._clock.record_episode()  # Approximate: 1 episode per cycle

        # Update Evo hypothesis count periodically
        if self._evo is not None and self._clock.pressure.cycles_since_sleep % 100 == 0:
            try:
                if hasattr(self._evo, "_hypothesis_engine"):
                    engine = self._evo._hypothesis_engine
                    count = len(getattr(engine, "_hypotheses", {}))
                    self._clock.record_hypothesis_count(count)
            except Exception:
                pass

        # Emit metrics
        self._emit_metric("oneiros.sleep_pressure", self._clock.pressure.composite_pressure)

        # Check sleep triggers
        if self._clock.must_sleep():
            self._logger.warning("forced_sleep", pressure=self._clock.pressure.composite_pressure)
            await self._emit_event(SLEEP_FORCED, {
                "pressure": self._clock.pressure.composite_pressure,
            })
            await self.begin_sleep()
        elif self._clock.should_sleep():
            await self._emit_event(SLEEP_PRESSURE_WARNING, {
                "pressure": self._clock.pressure.composite_pressure,
            })

    async def on_episode_stored(self, valence: float, arousal: float) -> None:
        """Called when Memory stores an episode — more accurate pressure tracking."""
        self._clock.record_affect_trace(valence, arousal)

    # ── Sleep Orchestration ───────────────────────────────────────

    async def begin_sleep(self) -> None:
        """
        Initiate a sleep cycle.

        The organism transitions through: HYPNAGOGIA → NREM → REM →
        (optionally LUCID) → HYPNOPOMPIA → WAKE.
        """
        if self._stage_controller.is_sleeping:
            self._logger.warning("already_sleeping")
            return

        cycle_id = new_id()
        self._current_cycle = SleepCycle(
            id=cycle_id,
            pressure_before=self._clock.pressure.composite_pressure,
        )

        self._stage_controller.set_has_creative_goal(self._creative_goal is not None)
        self._stage_controller.begin_sleep(cycle_id)

        await self._journal.record_sleep_cycle(self._current_cycle)
        await self._emit_event(SLEEP_ONSET, {
            "cycle_id": cycle_id,
            "pressure": self._clock.pressure.composite_pressure,
            "stage": SleepStage.HYPNAGOGIA.value,
        })

        # Run the sleep cycle in a background task
        self._sleep_task = asyncio.create_task(self._run_sleep_cycle(cycle_id))

    async def _run_sleep_cycle(self, cycle_id: str) -> None:
        """
        Execute the full sleep cycle.

        This runs as a background task while the main cognitive
        cycle is suspended (or running in reduced mode).
        """
        try:
            # ── HYPNAGOGIA ────────────────────────────────────────
            await self._run_hypnagogia()

            # ── NREM ──────────────────────────────────────────────
            nrem_result = await self._run_nrem(cycle_id)
            if self._current_cycle is not None:
                self._current_cycle.episodes_replayed = nrem_result.episodes_replayed
                self._current_cycle.semantic_nodes_created = nrem_result.semantic_nodes_created
                self._current_cycle.traces_pruned = (
                    nrem_result.traces_pruned + nrem_result.traces_decayed
                )
                self._current_cycle.beliefs_compressed = nrem_result.beliefs_merged
                self._current_cycle.hypotheses_pruned = nrem_result.hypotheses_retired
                self._current_cycle.hypotheses_promoted = nrem_result.hypotheses_promoted

            # ── REM ───────────────────────────────────────────────
            rem_result = await self._run_rem(cycle_id)
            if self._current_cycle is not None:
                self._current_cycle.dreams_generated = rem_result.dreams_generated
                self._current_cycle.insights_discovered = rem_result.insights_discovered
                self._current_cycle.affect_traces_processed = rem_result.affect_traces_processed
                self._current_cycle.affect_reduction_mean = rem_result.mean_valence_reduction
                self._current_cycle.threats_simulated = rem_result.threats_simulated
                self._current_cycle.ethical_cases_digested = rem_result.ethical_cases_digested

            # ── LUCID (optional) ──────────────────────────────────
            lucid_result: LucidResult | None = None
            if self._stage_controller.is_in_lucid or self._creative_goal is not None:
                lucid_result = await self._run_lucid(cycle_id)
                if self._current_cycle is not None and lucid_result is not None:
                    self._current_cycle.lucid_explorations = lucid_result.explorations_completed
                    self._current_cycle.meta_observations = lucid_result.meta_observations

            # ── HYPNOPOMPIA ───────────────────────────────────────
            await self._run_hypnopompia(cycle_id)

            # ── Complete cycle ────────────────────────────────────
            await self._complete_cycle(SleepQuality.NORMAL)

        except asyncio.CancelledError:
            await self._complete_cycle(SleepQuality.DEPRIVED)
            raise
        except Exception as exc:
            self._logger.error("sleep_cycle_error", error=str(exc), cycle_id=cycle_id)
            await self._complete_cycle(SleepQuality.FRAGMENTED)

    async def _run_hypnagogia(self) -> None:
        """Transition in — wind down external processing."""
        self._logger.info("hypnagogia_begin")
        # In a full integration, this would signal Atune to raise thresholds,
        # Nova to pause goal generation, Voxis to enter drowsy mode.
        # For now, we simulate the transition duration.
        stage = self._stage_controller.advance(
            self._stage_controller._hypnagogia_s
        )
        if stage is not None:
            await self._emit_event(SLEEP_STAGE_CHANGED, {
                "stage": stage.value,
                "cycle_id": self._current_cycle.id if self._current_cycle else "",
            })

    async def _run_nrem(self, cycle_id: str) -> NREMConsolidationResult:
        """Run all NREM consolidation workers."""
        self._logger.info("nrem_begin", cycle_id=cycle_id)
        result = NREMConsolidationResult()

        # Advance stage
        self._stage_controller.advance(0.1)
        await self._emit_event(SLEEP_STAGE_CHANGED, {
            "stage": SleepStage.NREM.value,
            "cycle_id": cycle_id,
        })

        # Run all four NREM workers
        try:
            replay_result, downscale_result, belief_result, hypothesis_result = (
                await asyncio.gather(
                    self._episodic_replay.run(cycle_id),
                    self._synaptic_downscaler.run(),
                    self._belief_compressor.run(),
                    self._hypothesis_pruner.run(),
                    return_exceptions=True,
                )
            )

            # Episodic replay
            if not isinstance(replay_result, BaseException):
                result.episodes_replayed = replay_result.episodes_replayed
                result.semantic_nodes_created = replay_result.semantic_nodes_created
                result.replay_duration_ms = replay_result.duration_ms
                self._episodes_consolidated += replay_result.episodes_replayed
                self._semantic_nodes_created += replay_result.semantic_nodes_created
            else:
                self._logger.error("episodic_replay_failed", error=str(replay_result))

            # Synaptic downscaling
            if not isinstance(downscale_result, BaseException):
                result.traces_decayed = downscale_result.traces_decayed
                result.traces_pruned = downscale_result.traces_pruned
                result.mean_salience_reduction = downscale_result.mean_reduction
                result.downscale_duration_ms = downscale_result.duration_ms
                self._traces_pruned += downscale_result.traces_pruned
            else:
                self._logger.error("downscale_failed", error=str(downscale_result))

            # Belief compression
            if not isinstance(belief_result, BaseException):
                result.beliefs_merged = belief_result.beliefs_merged
                result.beliefs_archived = belief_result.beliefs_archived
                result.beliefs_flagged_contradictory = belief_result.beliefs_flagged_contradictory
                result.compression_duration_ms = belief_result.duration_ms
            else:
                self._logger.error("belief_compression_failed", error=str(belief_result))

            # Hypothesis pruning
            if not isinstance(hypothesis_result, BaseException):
                result.hypotheses_retired = hypothesis_result.hypotheses_retired
                result.hypotheses_promoted = hypothesis_result.hypotheses_promoted
                result.hypotheses_merged = hypothesis_result.hypotheses_merged
                result.pruning_duration_ms = hypothesis_result.duration_ms
                self._hypotheses_pruned += hypothesis_result.hypotheses_retired
                self._hypotheses_promoted += hypothesis_result.hypotheses_promoted
            else:
                self._logger.error("hypothesis_pruning_failed", error=str(hypothesis_result))

        except Exception as exc:
            self._logger.error("nrem_error", error=str(exc))

        result.total_duration_ms = (
            result.replay_duration_ms
            + result.downscale_duration_ms
            + result.compression_duration_ms
            + result.pruning_duration_ms
        )

        self._logger.info(
            "nrem_complete",
            episodes=result.episodes_replayed,
            semantic_nodes=result.semantic_nodes_created,
            traces_pruned=result.traces_pruned,
            duration_ms=result.total_duration_ms,
        )

        # Advance stage past NREM
        nrem_budget_s = self._stage_controller._nrem_end_s - self._stage_controller._hypnagogia_s
        self._stage_controller.advance(nrem_budget_s)

        return result

    async def _run_rem(self, cycle_id: str) -> REMDreamResult:
        """Run all REM dream workers."""
        self._logger.info("rem_begin", cycle_id=cycle_id)
        result = REMDreamResult()

        await self._emit_event(SLEEP_STAGE_CHANGED, {
            "stage": SleepStage.REM.value,
            "cycle_id": cycle_id,
        })

        try:
            # Run dream generation and affect processing concurrently
            # Threat simulation and ethical digestion run after
            dream_result, affect_result = await asyncio.gather(
                self._dream_generator.run(
                    cycle_id,
                    max_dreams=_get(self._config, "max_dreams_per_rem", 50),
                ),
                self._affect_processor.run(
                    cycle_id,
                    max_traces=_get(self._config, "max_affect_traces_per_rem", 100),
                ),
                return_exceptions=True,
            )

            # Dream generation results
            if not isinstance(dream_result, BaseException):
                result.dreams_generated = dream_result.dreams_generated
                result.insights_discovered = dream_result.insights_discovered
                result.fragments_stored = dream_result.fragments_stored
                result.noise_discarded = dream_result.noise_discarded
                result.dream_duration_ms = dream_result.duration_ms

                # Record dreams and insights in journal
                for dream in dream_result.dreams:
                    await self._journal.record_dream(dream)
                    self._total_dreams += 1
                    self._dream_coherence_sum += dream.coherence_score

                for insight in dream_result.insights:
                    await self._journal.record_insight(insight)
                    self._total_insights += 1
                    self._pending_wake_insights.append(insight)
                    await self._emit_event(DREAM_INSIGHT, {
                        "insight_id": insight.id,
                        "coherence": insight.coherence_score,
                        "domain": insight.domain,
                    })
                    # Convert high-coherence insights into waking goals
                    if self._nova is not None and insight.coherence_score > 0.5:
                        await self._convert_insight_to_goal(insight)
            else:
                self._logger.error("dream_generation_failed", error=str(dream_result))

            # Affect processing results
            if not isinstance(affect_result, BaseException):
                result.affect_traces_processed = affect_result.traces_processed
                result.mean_valence_reduction = affect_result.mean_valence_reduction
                result.mean_arousal_reduction = affect_result.mean_arousal_reduction
                result.affect_duration_ms = affect_result.duration_ms
                self._affect_traces_processed += affect_result.traces_processed
            else:
                self._logger.error("affect_processing_failed", error=str(affect_result))

            # Threat simulation and ethical digestion (sequential, lower priority)
            threat_result, ethical_result = await asyncio.gather(
                self._threat_simulator.run(
                    cycle_id,
                    max_scenarios=_get(self._config, "max_threats_per_rem", 15),
                ),
                self._ethical_digestion.run(
                    cycle_id,
                    max_cases=_get(self._config, "max_ethical_cases_per_rem", 10),
                ),
                return_exceptions=True,
            )

            if not isinstance(threat_result, BaseException):
                result.threats_simulated = threat_result.threats_simulated
                result.response_plans_created = threat_result.response_plans_created
                result.threat_duration_ms = threat_result.duration_ms
                self._threats_simulated += threat_result.threats_simulated
                self._response_plans_created += threat_result.response_plans_created

                for dream in threat_result.dreams:
                    await self._journal.record_dream(dream)

            if not isinstance(ethical_result, BaseException):
                result.ethical_cases_digested = ethical_result.cases_digested
                result.heuristics_refined = ethical_result.heuristics_refined
                result.ethical_duration_ms = ethical_result.duration_ms

                for dream in ethical_result.dreams:
                    await self._journal.record_dream(dream)

        except Exception as exc:
            self._logger.error("rem_error", error=str(exc))

        result.total_duration_ms = (
            result.dream_duration_ms
            + result.affect_duration_ms
            + result.threat_duration_ms
            + result.ethical_duration_ms
        )

        self._logger.info(
            "rem_complete",
            dreams=result.dreams_generated,
            insights=result.insights_discovered,
            affect_processed=result.affect_traces_processed,
            threats=result.threats_simulated,
            duration_ms=result.total_duration_ms,
        )

        # Advance stage past REM
        rem_budget_s = self._stage_controller._rem_end_s - self._stage_controller._nrem_end_s
        self._stage_controller.advance(rem_budget_s)

        return result

    async def _run_lucid(self, cycle_id: str) -> LucidResult:
        """Run lucid dreaming workers."""
        self._logger.info("lucid_begin", cycle_id=cycle_id)
        result = LucidResult()

        await self._emit_event(SLEEP_STAGE_CHANGED, {
            "stage": SleepStage.LUCID.value,
            "cycle_id": cycle_id,
        })

        try:
            # Directed exploration
            trigger = self._creative_goal or None
            exploration_result = await self._directed_exploration.run(cycle_id, trigger)
            result.explorations_completed = exploration_result.explorations_completed
            result.variations_generated = exploration_result.variations_generated
            result.high_value_variations = exploration_result.high_value_insights

            for dream in exploration_result.dreams:
                await self._journal.record_dream(dream)
                self._total_dreams += 1

            for insight in exploration_result.insights:
                await self._journal.record_insight(insight)
                self._total_insights += 1
                self._pending_wake_insights.append(insight)
                # Convert high-coherence insights into waking goals
                if self._nova is not None and insight.coherence_score > 0.5:
                    await self._convert_insight_to_goal(insight)

            # MetaCognition
            meta_result = await self._metacognition.run(cycle_id)
            result.meta_observations = meta_result.observations_made
            result.recurring_themes_detected = meta_result.themes_analyzed
            result.self_knowledge_nodes_created = meta_result.self_knowledge_nodes_created

            for dream in meta_result.dreams:
                await self._journal.record_dream(dream)
                self._total_dreams += 1

        except Exception as exc:
            self._logger.error("lucid_error", error=str(exc))

        result.total_duration_ms = (
            exploration_result.duration_ms + meta_result.duration_ms
            if 'exploration_result' in dir() and 'meta_result' in dir()
            else 0
        )

        self._logger.info(
            "lucid_complete",
            explorations=result.explorations_completed,
            observations=result.meta_observations,
        )

        # Advance stage past LUCID
        lucid_budget_s = self._stage_controller._lucid_end_s - self._stage_controller._rem_end_s
        self._stage_controller.advance(lucid_budget_s)

        return result

    async def _run_hypnopompia(self, cycle_id: str) -> None:
        """Transition out — restore wake processing."""
        self._logger.info("hypnopompia_begin", cycle_id=cycle_id)

        # Advance through hypnopompia
        self._stage_controller.advance(self._stage_controller._hypnopompia_s)
        self._stage_controller.wake()

        await self._emit_event(SLEEP_STAGE_CHANGED, {
            "stage": SleepStage.WAKE.value,
            "cycle_id": cycle_id,
        })

    async def _complete_cycle(self, quality: SleepQuality) -> None:
        """Finalize a sleep cycle."""
        if self._current_cycle is None:
            return

        self._current_cycle.completed_at = utc_now()
        self._current_cycle.quality = quality
        self._current_cycle.pressure_after = self._clock.pressure.composite_pressure

        # Reset sleep pressure
        self._clock.reset_after_sleep(quality)
        self._current_cycle.pressure_after = self._clock.pressure.composite_pressure

        # Update journal
        await self._journal.update_sleep_cycle(self._current_cycle)
        self._recent_cycles.append(self._current_cycle)

        # Update lifetime metrics
        self._total_sleep_cycles += 1
        quality_scores = {
            SleepQuality.DEEP: 1.0,
            SleepQuality.NORMAL: 0.75,
            SleepQuality.FRAGMENTED: 0.4,
            SleepQuality.DEPRIVED: 0.1,
        }
        self._sleep_quality_sum += quality_scores.get(quality, 0.5)

        # Clear creative goal after use
        self._creative_goal = None

        # Emit wake event
        await self._emit_event(WAKE_ONSET, {
            "cycle_id": self._current_cycle.id,
            "quality": quality.value,
            "insights_count": len(self._pending_wake_insights),
            "dreams_generated": self._current_cycle.dreams_generated,
            "episodes_consolidated": self._current_cycle.episodes_replayed,
        })

        self._logger.info(
            "sleep_cycle_complete",
            cycle_id=self._current_cycle.id,
            quality=quality.value,
            dreams=self._current_cycle.dreams_generated,
            insights=self._current_cycle.insights_discovered,
            episodes_consolidated=self._current_cycle.episodes_replayed,
            pressure_before=round(self._current_cycle.pressure_before, 3),
            pressure_after=round(self._current_cycle.pressure_after, 3),
        )

        self._current_cycle = None

    # ── Emergency Wake ────────────────────────────────────────────

    async def _on_critical_event(self, event: Any) -> None:
        """Handle critical Synapse events during sleep."""
        if not self._stage_controller.is_sleeping:
            return

        event_type = getattr(event, "event_type", None)

        self._logger.warning(
            "emergency_wake_triggered",
            event_type=str(event_type),
            current_stage=self._stage_controller.current_stage.value,
        )

        self._stage_controller.emergency_wake(f"Critical event: {event_type}")

        if self._current_cycle is not None:
            self._current_cycle.interrupted = True
            self._current_cycle.interrupt_reason = f"Critical: {event_type}"

        await self._emit_event(EMERGENCY_WAKE, {
            "reason": str(event_type),
            "stage_interrupted": self._stage_controller.current_stage.value,
        })

    # ── Public API ────────────────────────────────────────────────

    def set_creative_goal(self, goal: str) -> None:
        """Set a creative goal for the next lucid dreaming phase."""
        self._creative_goal = goal
        self._stage_controller.set_has_creative_goal(True)
        self._logger.info("creative_goal_set", goal=goal[:100])

    def get_pending_insights(self) -> list[Any]:
        """Get insights discovered during sleep for wake broadcast."""
        insights = self._pending_wake_insights.copy()
        self._pending_wake_insights.clear()
        return insights

    async def _convert_insight_to_goal(self, insight: Any) -> None:
        """
        Convert a high-coherence dream insight into a waking exploration goal.

        Dreams consolidate experience and sometimes surface important patterns
        that the waking mind should explore. This closes the dream→goal loop:
        insight → Nova goal → deliberation → action → new experience.
        """
        from ecodiaos.systems.nova.types import Goal, GoalSource, GoalStatus
        from ecodiaos.primitives.common import new_id, DriveAlignmentVector

        # Higher coherence → higher priority (0.5 base + up to 0.2 bonus)
        priority = min(1.0, 0.5 + insight.coherence_score * 0.2)

        goal = Goal(
            id=new_id(),
            description=f"Explore dream insight: {insight.insight_text[:120]}",
            source=GoalSource.SELF_GENERATED,
            priority=priority,
            urgency=0.3,
            importance=0.55,
            drive_alignment=DriveAlignmentVector(
                coherence=0.4, care=0.1, growth=0.4, honesty=0.1,
            ),
            status=GoalStatus.ACTIVE,
        )
        try:
            await self._nova.add_goal(goal)
            self._logger.info(
                "dream_insight_goal_created",
                insight_id=insight.id,
                goal_id=goal.id,
                coherence=f"{insight.coherence_score:.2f}",
            )
        except Exception as exc:
            self._logger.warning("dream_insight_goal_failed", error=str(exc))

    @property
    def is_sleeping(self) -> bool:
        return self._stage_controller.is_sleeping

    @property
    def current_stage(self) -> SleepStage:
        return self._stage_controller.current_stage

    @property
    def sleep_pressure(self) -> float:
        return self._clock.pressure.composite_pressure

    @property
    def degradation(self) -> WakeDegradation:
        return self._clock.degradation

    @property
    def stats(self) -> dict[str, Any]:
        """Aggregate statistics."""
        return {
            "total_sleep_cycles": self._total_sleep_cycles,
            "total_dreams": self._total_dreams,
            "total_insights": self._total_insights,
            "insights_validated": self._insights_validated,
            "insights_integrated": self._insights_integrated,
            "episodes_consolidated": self._episodes_consolidated,
            "semantic_nodes_created": self._semantic_nodes_created,
            "traces_pruned": self._traces_pruned,
            "affect_traces_processed": self._affect_traces_processed,
            "threats_simulated": self._threats_simulated,
            "mean_dream_coherence": (
                self._dream_coherence_sum / max(self._total_dreams, 1)
            ),
            "mean_sleep_quality": (
                self._sleep_quality_sum / max(self._total_sleep_cycles, 1)
            ),
            "current_pressure": self._clock.pressure.composite_pressure,
            "current_stage": self._stage_controller.current_stage.value,
            "current_degradation": self._clock.degradation.composite_impairment,
        }

    # ── Health ────────────────────────────────────────────────────

    def _build_health_snapshot(self) -> OneirosHealthSnapshot:
        """Build the health snapshot."""
        degradation = self._clock.degradation
        pressure = self._clock.pressure

        return OneirosHealthSnapshot(
            status="sleeping" if self.is_sleeping else "healthy",
            current_stage=self._stage_controller.current_stage,
            sleep_pressure=pressure.composite_pressure,
            wake_degradation=degradation.composite_impairment,
            current_sleep_debt_hours=(
                pressure.cycles_since_sleep * 0.00015 / 3600  # rough estimate
                if pressure.composite_pressure > pressure.threshold
                else 0.0
            ),
            total_sleep_cycles=self._total_sleep_cycles,
            total_dreams=self._total_dreams,
            total_insights=self._total_insights,
            insights_validated=self._insights_validated,
            insights_invalidated=self._insights_invalidated,
            insights_integrated=self._insights_integrated,
            mean_dream_coherence=(
                self._dream_coherence_sum / max(self._total_dreams, 1)
            ),
            mean_sleep_quality=(
                self._sleep_quality_sum / max(self._total_sleep_cycles, 1)
            ),
            episodes_consolidated=self._episodes_consolidated,
            semantic_nodes_created=self._semantic_nodes_created,
            traces_pruned=self._traces_pruned,
            hypotheses_pruned=self._hypotheses_pruned,
            hypotheses_promoted=self._hypotheses_promoted,
            affect_traces_processed=self._affect_traces_processed,
            mean_affect_reduction=0.0,  # Would need rolling average
            threats_simulated=self._threats_simulated,
            response_plans_created=self._response_plans_created,
            last_sleep_completed=pressure.last_sleep_completed,
            last_sleep_quality=(
                self._recent_cycles[-1].quality if self._recent_cycles else None
            ),
        )

    # ── Internal Helpers ──────────────────────────────────────────

    async def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit a Synapse event."""
        if self._synapse is None:
            return
        try:
            from ecodiaos.systems.synapse.types import SynapseEvent, SynapseEventType

            # Map our string events to SynapseEventType if possible,
            # otherwise log as custom event
            event = SynapseEvent(
                id=new_id(),
                event_type=SynapseEventType.SYSTEM_STARTED,  # Placeholder
                timestamp=utc_now(),
                data={"oneiros_event": event_type, **data},
                source_system="oneiros",
            )
            await self._synapse._event_bus.emit(event)
        except Exception:
            pass  # Events are best-effort

    def _emit_metric(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Emit a telemetry metric."""
        if self._metrics is None:
            return
        with contextlib.suppress(Exception):
            self._metrics.record(name, value, tags=tags or {})


def _get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
