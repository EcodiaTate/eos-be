"""
EcodiaOS — Soma Interoceptor

Reads from all cognitive systems every theta cycle to compose the
9-dimensional sensed interoceptive state. All reads are in-memory —
no database, no LLM, no network calls. Total sensing budget: 2ms.

Each interoceptor maps a system reference to one dimension with a
transform function and fallback value for when the source is unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.systems.soma.types import (
    ALL_DIMENSIONS,
    InteroceptiveDimension,
    _clamp_dimension,
)

if TYPE_CHECKING:
    pass

logger = structlog.get_logger("ecodiaos.systems.soma.interoceptor")


# ─── Fallback Values ──────────────────────────────────────────────

FALLBACK_VALUES: dict[InteroceptiveDimension, float] = {
    InteroceptiveDimension.ENERGY: 0.5,
    InteroceptiveDimension.AROUSAL: 0.4,
    InteroceptiveDimension.VALENCE: 0.0,
    InteroceptiveDimension.CONFIDENCE: 0.5,
    InteroceptiveDimension.COHERENCE: 0.5,
    InteroceptiveDimension.SOCIAL_CHARGE: 0.3,
    InteroceptiveDimension.CURIOSITY_DRIVE: 0.5,
    InteroceptiveDimension.INTEGRITY: 0.8,
    InteroceptiveDimension.TEMPORAL_PRESSURE: 0.15,
}


class Interoceptor:
    """
    Composes the 9D sensed interoceptive state by reading from system references.

    All system references are set via set_*() methods during initialization.
    Missing references gracefully degrade to fallback values.
    """

    def __init__(self) -> None:
        # System references — set during wiring
        self._atune: Any = None
        self._synapse: Any = None
        self._nova: Any = None
        self._thymos: Any = None
        self._equor: Any = None
        self._token_budget: Any = None  # Synapse resource manager

    def set_atune(self, atune: Any) -> None:
        self._atune = atune

    def set_synapse(self, synapse: Any) -> None:
        self._synapse = synapse

    def set_nova(self, nova: Any) -> None:
        self._nova = nova

    def set_thymos(self, thymos: Any) -> None:
        self._thymos = thymos

    def set_equor(self, equor: Any) -> None:
        self._equor = equor

    def set_token_budget(self, budget: Any) -> None:
        self._token_budget = budget

    def sense(self) -> dict[InteroceptiveDimension, float]:
        """
        Read all 9 interoceptive dimensions from system references.

        Returns a dict mapping each dimension to its current sensed value,
        clamped to valid ranges. Total budget: <=2ms.
        """
        state: dict[InteroceptiveDimension, float] = {}

        state[InteroceptiveDimension.ENERGY] = self._sense_energy()
        state[InteroceptiveDimension.AROUSAL] = self._sense_arousal()
        state[InteroceptiveDimension.VALENCE] = self._sense_valence()
        state[InteroceptiveDimension.CONFIDENCE] = self._sense_confidence()
        state[InteroceptiveDimension.COHERENCE] = self._sense_coherence()
        state[InteroceptiveDimension.SOCIAL_CHARGE] = self._sense_social_charge()
        state[InteroceptiveDimension.CURIOSITY_DRIVE] = self._sense_curiosity_drive()
        state[InteroceptiveDimension.INTEGRITY] = self._sense_integrity()
        state[InteroceptiveDimension.TEMPORAL_PRESSURE] = self._sense_temporal_pressure()

        # Ensure all dimensions clamped to valid ranges
        for dim in ALL_DIMENSIONS:
            state[dim] = _clamp_dimension(dim, state[dim])

        return state

    # ─── Per-Dimension Readers ────────────────────────────────────

    def _sense_energy(self) -> float:
        """ENERGY: 1.0 - token_budget.utilization. Metabolic availability."""
        try:
            if self._token_budget is not None:
                status = self._token_budget.get_status()
                if hasattr(status, "utilization"):
                    return 1.0 - float(status.utilization)
                if isinstance(status, dict):
                    return 1.0 - float(status.get("utilization", 0.5))
            # Fallback: try synapse resource manager
            if self._synapse is not None and hasattr(self._synapse, "_resources"):
                resources = self._synapse._resources
                if hasattr(resources, "get_status"):
                    status = resources.get_status()
                    if hasattr(status, "utilization"):
                        return 1.0 - float(status.utilization)
        except Exception:
            pass
        return FALLBACK_VALUES[InteroceptiveDimension.ENERGY]

    def _sense_arousal(self) -> float:
        """AROUSAL: direct from Atune affect_manager.current_affect.arousal."""
        try:
            if self._atune is not None:
                affect = self._get_current_affect()
                if affect is not None:
                    return float(affect.arousal)
        except Exception:
            pass
        return FALLBACK_VALUES[InteroceptiveDimension.AROUSAL]

    def _sense_valence(self) -> float:
        """VALENCE: direct from Atune affect_manager.current_affect.valence."""
        try:
            if self._atune is not None:
                affect = self._get_current_affect()
                if affect is not None:
                    return float(affect.valence)
        except Exception:
            pass
        return FALLBACK_VALUES[InteroceptiveDimension.VALENCE]

    def _sense_confidence(self) -> float:
        """CONFIDENCE: 1.0 - clamp(mean_prediction_error, 0, 1)."""
        try:
            if self._atune is not None:
                # Try to read mean prediction error from recent cycles
                if hasattr(self._atune, "mean_prediction_error"):
                    mpe = float(self._atune.mean_prediction_error)
                    return 1.0 - max(0.0, min(1.0, mpe))
                # Fallback: derive from coherence_stress
                affect = self._get_current_affect()
                if affect is not None:
                    return 1.0 - max(0.0, min(1.0, float(affect.coherence_stress)))
        except Exception:
            pass
        return FALLBACK_VALUES[InteroceptiveDimension.CONFIDENCE]

    def _sense_coherence(self) -> float:
        """COHERENCE: synapse.coherence_monitor.current_phi (already 0-1)."""
        try:
            if self._synapse is not None:
                if hasattr(self._synapse, "_coherence"):
                    coherence = self._synapse._coherence
                    if hasattr(coherence, "current_phi"):
                        phi = coherence.current_phi
                        if phi is not None:
                            return float(phi)
                # Try via snapshot
                if hasattr(self._synapse, "coherence_snapshot"):
                    snap = self._synapse.coherence_snapshot
                    if hasattr(snap, "phi"):
                        return float(snap.phi)
        except Exception:
            pass
        return FALLBACK_VALUES[InteroceptiveDimension.COHERENCE]

    def _sense_social_charge(self) -> float:
        """SOCIAL_CHARGE: atune.affect_manager.current_affect.care_activation."""
        try:
            if self._atune is not None:
                affect = self._get_current_affect()
                if affect is not None:
                    return float(affect.care_activation)
        except Exception:
            pass
        return FALLBACK_VALUES[InteroceptiveDimension.SOCIAL_CHARGE]

    def _sense_curiosity_drive(self) -> float:
        """CURIOSITY_DRIVE: atune.affect_manager.current_affect.curiosity."""
        try:
            if self._atune is not None:
                affect = self._get_current_affect()
                if affect is not None:
                    return float(affect.curiosity)
        except Exception:
            pass
        return FALLBACK_VALUES[InteroceptiveDimension.CURIOSITY_DRIVE]

    def _sense_integrity(self) -> float:
        """INTEGRITY: min(thymos_health, 1.0 - equor_drift)."""
        thymos_health = 1.0
        equor_component = 1.0

        try:
            if self._thymos is not None:
                if hasattr(self._thymos, "current_health_score"):
                    score = self._thymos.current_health_score
                    if score is not None:
                        thymos_health = float(score)
                elif hasattr(self._thymos, "_governor"):
                    gov = self._thymos._governor
                    if hasattr(gov, "health_score"):
                        thymos_health = float(gov.health_score)
        except Exception:
            pass

        try:
            if self._equor is not None:
                if hasattr(self._equor, "constitutional_drift"):
                    drift = self._equor.constitutional_drift
                    if drift is not None:
                        equor_component = 1.0 - max(0.0, min(1.0, float(drift)))
                elif hasattr(self._equor, "_drift_monitor"):
                    monitor = self._equor._drift_monitor
                    if hasattr(monitor, "current_drift_magnitude"):
                        drift = monitor.current_drift_magnitude
                        if drift is not None:
                            equor_component = 1.0 - max(0.0, min(1.0, float(drift)))
        except Exception:
            pass

        return min(thymos_health, equor_component)

    def _sense_temporal_pressure(self) -> float:
        """TEMPORAL_PRESSURE: nova.goal_urgency_max + arousal * 0.3, clamped 0-1."""
        goal_urgency = 0.0
        arousal_boost = 0.0

        try:
            if self._nova is not None:
                if hasattr(self._nova, "goal_urgency_max"):
                    gu = self._nova.goal_urgency_max
                    if gu is not None:
                        goal_urgency = float(gu)
                elif hasattr(self._nova, "_goal_manager"):
                    gm = self._nova._goal_manager
                    if gm is not None and hasattr(gm, "max_urgency"):
                        goal_urgency = float(gm.max_urgency or 0.0)
        except Exception:
            pass

        try:
            if self._atune is not None:
                affect = self._get_current_affect()
                if affect is not None:
                    arousal_boost = affect.arousal * 0.3
        except Exception:
            pass

        return max(0.0, min(1.0, goal_urgency + arousal_boost))

    # ─── Helpers ──────────────────────────────────────────────────

    def _get_current_affect(self) -> Any:
        """Get current AffectState from Atune, trying known attribute paths."""
        if self._atune is None:
            return None
        # Path 1: direct attribute
        if hasattr(self._atune, "_affect_mgr"):
            mgr = self._atune._affect_mgr
            if hasattr(mgr, "current"):
                return mgr.current
        # Path 2: public method
        if hasattr(self._atune, "current_affect"):
            ca = self._atune.current_affect
            if callable(ca):
                return ca()
            return ca
        return None
