"""
EcodiaOS — Soma Service (The Interoceptive Predictive Substrate)

The organism's felt sense of being alive. Soma predicts internal states,
computes the gap between where the organism is and where it needs to be,
and emits the allostatic signals that make every other system care about
staying viable.

Cognitive cycle role (step 0 — SENSE):
  Soma runs FIRST in every theta cycle, before Atune. It reads from all
  systems, predicts multi-horizon states, computes allostatic errors,
  and emits an AllostaticSignal that downstream systems consume.

Iron Rules:
  - Total cycle budget: 5ms. Soma is the fastest system in the organism.
  - No LLM calls. No database calls. No network calls during cycle.
  - All reads are in-memory from system references.
  - If Soma fails, the organism degrades gracefully to pre-Soma behaviour.
  - Soma is advisory, not commanding — systems MAY ignore the signal.

Interface:
  initialize()              — wire system refs, load config, seed attractors
  run_cycle()               — main theta cycle entry (sense → predict → emit)
  get_current_state()       — last computed interoceptive state
  get_current_signal()      — last emitted allostatic signal
  get_somatic_marker()      — snapshot for memory stamping
  get_errors()              — allostatic errors per horizon per dimension
  get_phase_position()      — attractor, bifurcation, trajectory info
  get_developmental_stage() — current maturation stage
  create_somatic_marker()   — create marker from current state
  somatic_rerank()          — boost memory candidates by somatic similarity
  update_dynamics_matrix()  — Evo updates cross-dimension coupling
  update_emotion_regions()  — Evo refines emotion boundaries
  shutdown()                — graceful teardown
  health()                  — self-health report for Synapse
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.systems.soma.allostatic_controller import AllostaticController
from ecodiaos.systems.soma.base import BaseAllostaticRegulator, BaseSomaPredictor
from ecodiaos.systems.soma.counterfactual import CounterfactualEngine
from ecodiaos.systems.soma.developmental import DevelopmentalManager
from ecodiaos.systems.soma.interoceptor import Interoceptor
from ecodiaos.systems.soma.phase_space import PhaseSpaceModel
from ecodiaos.systems.soma.predictor import InteroceptivePredictor
from ecodiaos.systems.soma.somatic_memory import SomaticMemoryIntegration
from ecodiaos.systems.soma.temporal_depth import TemporalDepthManager
from ecodiaos.systems.soma.types import (
    ALL_DIMENSIONS,
    EMOTION_REGIONS,
    AllostaticSignal,
    CounterfactualTrace,
    DevelopmentalStage,
    InteroceptiveDimension,
    InteroceptiveState,
    SomaticMarker,
)

if TYPE_CHECKING:
    from ecodiaos.config import SomaConfig
    from ecodiaos.core.hotreload import NeuroplasticityBus

logger = structlog.get_logger("ecodiaos.systems.soma")


class SomaService:
    """
    Soma — the EOS interoceptive predictive substrate.

    Coordinates seven sub-systems:
      Interoceptor       — reads 9D sensed state from all systems (2ms)
      Predictor          — multi-horizon generative model (1ms)
      AllostaticCtl      — setpoint management, urgency, signal construction (0.5ms)
      PhaseSpace         — attractor detection, bifurcation mapping (2ms every 100 cycles)
      SomaticMemory      — marker creation, embodied retrieval reranking (1ms)
      TemporalDepth      — multi-scale prediction, temporal dissonance
      Counterfactual     — Oneiros REM counterfactual replay
      Developmental      — stage gating, maturation triggers
    """

    system_id: str = "soma"

    def __init__(
        self,
        config: SomaConfig,
        neuroplasticity_bus: NeuroplasticityBus | None = None,
    ) -> None:
        self._config = config
        self._bus = neuroplasticity_bus

        # Sub-systems
        self._interoceptor = Interoceptor()
        self._predictor: BaseSomaPredictor = InteroceptivePredictor(
            buffer_size=config.trajectory_buffer_size,
            ewm_span=config.prediction_ewm_span,
        )
        self._controller: BaseAllostaticRegulator = AllostaticController(
            adaptation_alpha=config.setpoint_adaptation_alpha,
            urgency_threshold=config.urgency_threshold,
        )
        self._phase_space = PhaseSpaceModel(
            max_attractors=config.max_attractors,
            min_dwell_cycles=config.attractor_min_dwell_cycles,
            detection_enabled=config.bifurcation_detection_enabled,
        )
        self._somatic_memory = SomaticMemoryIntegration(
            rerank_boost=config.somatic_rerank_boost,
        )
        self._temporal_depth = TemporalDepthManager()
        self._counterfactual = CounterfactualEngine()
        self._developmental = DevelopmentalManager(
            initial_stage=DevelopmentalStage(config.initial_stage),
        )

        # State
        self._synapse_ref: Any = None  # Cached for hot-swap forwarding
        self._cycle_count: int = 0
        self._current_state: InteroceptiveState | None = None
        self._current_signal: AllostaticSignal = AllostaticSignal.default()
        self._last_cycle_duration_ms: float = 0.0
        self._cycle_durations: deque[float] = deque(maxlen=100)
        self._phase_space_update_counter: int = 0
        self._enabled: bool = config.cycle_enabled

        # Emotion regions (Evo can update)
        self._emotion_regions: dict[str, dict[str, Any]] = dict(EMOTION_REGIONS)

    async def initialize(self) -> None:
        """Initialize sub-systems and register with NeuroplasticityBus."""
        self._temporal_depth.set_stage(self._developmental.stage)
        self._counterfactual.set_dynamics(self._predictor.dynamics_matrix)

        # Register hot-reloadable strategies with the NeuroplasticityBus
        if self._bus is not None:
            self._bus.register(
                base_class=BaseSomaPredictor,
                registration_callback=self._on_predictor_evolved,
                system_id="soma",
            )
            self._bus.register(
                base_class=BaseAllostaticRegulator,
                registration_callback=self._on_regulator_evolved,
                system_id="soma",
            )

        logger.info(
            "soma_initialized",
            stage=self._developmental.stage.value,
            attractors=self._phase_space.attractor_count,
        )

    async def shutdown(self) -> None:
        """Graceful teardown — deregister from NeuroplasticityBus."""
        if self._bus is not None:
            self._bus.deregister(BaseSomaPredictor)
            self._bus.deregister(BaseAllostaticRegulator)
        logger.info("soma_shutdown", cycle_count=self._cycle_count)

    async def health(self) -> dict[str, Any]:
        """Health check for Synapse monitoring."""
        mean_duration = (
            sum(self._cycle_durations) / len(self._cycle_durations)
            if self._cycle_durations
            else 0.0
        )
        return {
            "status": "healthy" if self._enabled else "degraded",
            "cycle_count": self._cycle_count,
            "last_cycle_ms": round(self._last_cycle_duration_ms, 3),
            "mean_cycle_ms": round(mean_duration, 3),
            "stage": self._developmental.stage.value,
            "attractors": self._phase_space.attractor_count,
            "urgency": round(self._current_signal.urgency, 3),
            "dominant_error": self._current_signal.dominant_error.value,
        }

    # ─── System Wiring ──────────────────────────────────────────

    def set_atune(self, atune: Any) -> None:
        self._interoceptor.set_atune(atune)

    def set_synapse(self, synapse: Any) -> None:
        self._interoceptor.set_synapse(synapse)
        # Forward to controller if it supports metabolic sensing (MetabolicAllostaticRegulator)
        if hasattr(self._controller, "set_synapse"):
            self._controller.set_synapse(synapse)  # type: ignore[union-attr]
        self._synapse_ref = synapse

    def set_nova(self, nova: Any) -> None:
        self._interoceptor.set_nova(nova)

    def set_thymos(self, thymos: Any) -> None:
        self._interoceptor.set_thymos(thymos)

    def set_equor(self, equor: Any) -> None:
        self._interoceptor.set_equor(equor)

    def set_token_budget(self, budget: Any) -> None:
        self._interoceptor.set_token_budget(budget)

    # ─── NeuroplasticityBus Callbacks ─────────────────────────────

    def _on_predictor_evolved(self, predictor: BaseSomaPredictor) -> None:
        """
        Hot-swap the interoceptive predictor in the live service.

        Called by NeuroplasticityBus when a new BaseSomaPredictor subclass
        is discovered. The swap is atomic — any in-flight cycle that already
        captured a reference to the old predictor completes normally.
        """
        self._predictor = predictor
        self._counterfactual.set_dynamics(predictor.dynamics_matrix)
        logger.info(
            "soma_predictor_hot_reloaded",
            predictor=type(predictor).__name__,
        )

    def _on_regulator_evolved(self, regulator: BaseAllostaticRegulator) -> None:
        """
        Hot-swap the allostatic controller in the live service.

        Called by NeuroplasticityBus when a new BaseAllostaticRegulator subclass
        is discovered. Preserves current context by re-applying it, and forwards
        the cached Synapse reference so MetabolicAllostaticRegulator can read
        MetabolicSnapshot immediately after the swap.
        """
        self._controller = regulator
        # Re-wire Synapse if the new regulator supports metabolic sensing
        synapse = getattr(self, "_synapse_ref", None)
        if synapse is not None and hasattr(regulator, "set_synapse"):
            regulator.set_synapse(synapse)  # type: ignore[union-attr]
        logger.info(
            "soma_regulator_hot_reloaded",
            regulator=type(regulator).__name__,
        )

    # ─── Core Cycle ──────────────────────────────────────────────

    async def run_cycle(self) -> AllostaticSignal:
        """
        Main theta cycle entry. Called by Synapse BEFORE Atune.

        Pipeline:
          1. Sense — read 9D state from interoceptors (<=2ms)
          2. Buffer — push into trajectory ring buffer
          3. Predict — multi-horizon forecasts (<=1ms)
          4. Compute errors — predicted - setpoint per horizon per dim
          5. Compute error rates — d(error)/dt
          6. Compute temporal dissonance — moment vs session divergence
          7. Compute urgency — max(|errors|) * max(|error_rates|)
          8. Update phase space — every N cycles only (<=2ms)
          9. Build and emit AllostaticSignal
          10. Evaluate developmental transition

        Total budget: <=5ms.
        """
        if not self._enabled:
            return self._current_signal

        start = time.perf_counter()
        self._cycle_count += 1

        try:
            # 1. Sense
            sensed = self._interoceptor.sense()

            # 2. Buffer
            self._predictor.push_state(sensed)

            # 3. Tick setpoints (EMA toward targets)
            if self._developmental.setpoint_adaptation_enabled():
                self._controller.tick_setpoints()

            # 4. Predict
            available_horizons = self._temporal_depth.available_horizons
            predictions = self._predictor.predict_all_horizons(sensed, available_horizons)

            # 5. Compute allostatic errors
            setpoints = self._controller.setpoints
            errors = self._predictor.compute_allostatic_errors(predictions, setpoints)

            # 6. Compute error rates
            moment_errors = errors.get("moment", {d: 0.0 for d in ALL_DIMENSIONS})
            error_rates = self._predictor.compute_error_rates(moment_errors)

            # 7. Compute temporal dissonance
            dissonance = self._temporal_depth.compute_dissonance(predictions)

            # 8. Compute urgency
            urgency = self._controller.compute_urgency(errors, error_rates)

            # 9. Find dominant error
            dominant_dim, dominant_mag = self._controller.find_dominant_error(errors)

            # 10. Compute precision weights
            # Build intermediate state for precision computation
            state = InteroceptiveState(
                sensed=sensed,
                setpoints=setpoints,
                predicted=predictions,
                errors=errors,
                error_rates=error_rates,
                precision={d: 1.0 for d in ALL_DIMENSIONS},
                max_error_magnitude=dominant_mag,
                dominant_error=dominant_dim,
                temporal_dissonance=dissonance,
                urgency=urgency,
            )

            # 11. Update phase space (every N cycles)
            self._phase_space_update_counter += 1
            if (
                self._developmental.phase_space_enabled()
                and self._phase_space_update_counter >= self._config.phase_space_update_interval
            ):
                velocity = self._predictor.compute_velocity()
                self._phase_space.update(self._predictor.raw_trajectory, velocity)
                self._phase_space_update_counter = 0

            # 12. Build signal
            phase_dict = self._phase_space.snapshot_dict()
            signal = self._controller.build_signal(state, phase_dict, self._cycle_count)

            # Update current state reference on the signal's state
            signal.state.precision = signal.precision_weights
            self._current_state = signal.state
            self._current_signal = signal

            # 13. Evaluate developmental transitions (every 1000 cycles)
            if self._cycle_count % 1000 == 0:
                mean_conf = sensed.get(InteroceptiveDimension.CONFIDENCE, 0.5)
                promoted = self._developmental.evaluate_transition(
                    cycle_count=self._cycle_count,
                    mean_confidence=mean_conf,
                    attractor_count=self._phase_space.attractor_count,
                    bifurcation_count=self._phase_space.bifurcation_count,
                )
                if promoted:
                    self._temporal_depth.set_stage(self._developmental.stage)

        except Exception as exc:
            logger.error("soma_cycle_error", error=str(exc), cycle=self._cycle_count)
            # Emit default signal on failure — graceful degradation
            self._current_signal = AllostaticSignal.default()
            self._current_signal.cycle_number = self._cycle_count

        # Track timing
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._last_cycle_duration_ms = elapsed_ms
        self._cycle_durations.append(elapsed_ms)

        if elapsed_ms > 50.0:
            logger.warning("soma_cycle_slow", elapsed_ms=round(elapsed_ms, 2))

        return self._current_signal

    # ─── Query Methods ───────────────────────────────────────────

    def get_current_state(self) -> InteroceptiveState | None:
        """Returns the last computed interoceptive state."""
        return self._current_state

    def get_current_signal(self) -> AllostaticSignal:
        """Returns the last emitted allostatic signal."""
        return self._current_signal

    def get_somatic_marker(self) -> SomaticMarker:
        """Returns a somatic marker from the current state for memory stamping."""
        return self.create_somatic_marker()

    def get_errors(self) -> dict[str, dict[InteroceptiveDimension, float]]:
        """Returns allostatic errors per horizon per dimension."""
        if self._current_state is not None:
            return self._current_state.errors
        return {}

    def get_phase_position(self) -> dict[str, Any]:
        """Returns nearest attractor, distance to bifurcation, trajectory heading."""
        return self._phase_space.snapshot_dict()

    def get_developmental_stage(self) -> DevelopmentalStage:
        """Returns the current maturation stage."""
        return self._developmental.stage

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def urgency(self) -> float:
        return self._current_signal.urgency

    @property
    def urgency_threshold(self) -> float:
        return self._controller.urgency_threshold

    @property
    def dominant_error(self) -> InteroceptiveDimension:
        return self._current_signal.dominant_error

    @property
    def precision_weights(self) -> dict[InteroceptiveDimension, float]:
        return self._current_signal.precision_weights

    # ─── Memory Integration ──────────────────────────────────────

    def create_somatic_marker(self) -> SomaticMarker:
        """
        Snapshot current state as a somatic marker. Budget: 1ms.
        Called by Memory when storing traces.
        """
        if self._current_state is None:
            return SomaticMarker()

        attractor_label = self._phase_space.get_nearest_attractor_label(
            self._current_state.sensed,
        )
        return self._somatic_memory.create_marker(
            self._current_state,
            attractor_label=attractor_label,
        )

    def somatic_rerank(
        self,
        candidates: list[Any],
        current_state: InteroceptiveState | None = None,
    ) -> list[Any]:
        """
        Boost memories with similar somatic markers.
        Up to +30% salience boost based on somatic similarity.
        """
        state = current_state or self._current_state
        if state is None:
            return candidates
        return self._somatic_memory.somatic_rerank(candidates, state)

    # ─── Evo Integration ─────────────────────────────────────────

    def update_dynamics_matrix(self, new_dynamics: list[list[float]]) -> None:
        """Evo updates the 9x9 cross-dimension coupling matrix."""
        self._predictor.update_dynamics(new_dynamics)
        self._counterfactual.set_dynamics(self._predictor.dynamics_matrix)

    def update_emotion_regions(self, updated_regions: dict[str, dict[str, Any]]) -> None:
        """Evo refines emotion region boundaries."""
        self._emotion_regions.update(updated_regions)

    # ─── Counterfactual (Oneiros Integration) ────────────────────

    def generate_counterfactual(
        self,
        decision_id: str,
        actual_trajectory: list[dict[InteroceptiveDimension, float]],
        alternative_description: str,
        alternative_initial_impact: dict[InteroceptiveDimension, float],
        num_steps: int = 10,
    ) -> CounterfactualTrace:
        """
        Generate a counterfactual interoceptive trajectory.
        Only available at REFLECTIVE stage and above.
        """
        if not self._developmental.counterfactual_enabled():
            return CounterfactualTrace(
                decision_id=decision_id,
                lesson="Counterfactual not available at current developmental stage",
            )

        return self._counterfactual.generate_counterfactual(
            decision_id=decision_id,
            actual_trajectory=actual_trajectory,
            alternative_description=alternative_description,
            alternative_initial_impact=alternative_initial_impact,
            setpoints=self._controller.setpoints,
            num_steps=num_steps,
        )

    # ─── Context Management ──────────────────────────────────────

    def set_context(self, context: str) -> None:
        """
        Switch allostatic context. Called when the organism's activity changes.

        Contexts: "conversation", "deep_processing", "recovery", "exploration"
        """
        if self._developmental.setpoint_adaptation_enabled():
            self._controller.set_context(context)
