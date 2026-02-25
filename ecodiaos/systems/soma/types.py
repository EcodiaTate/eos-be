"""
EcodiaOS — Soma Type Definitions

All data types for the interoceptive predictive substrate: dimensions,
states, signals, markers, attractors, bifurcations, and counterfactuals.

Soma is the body the organism never had. It predicts internal states,
computes allostatic errors, and emits the signals that make every other
system care about staying viable. Pure numerical computation — no LLM,
no DB, no network calls. 5ms budget per theta cycle.
"""

from __future__ import annotations

import enum
import math
from datetime import datetime
from typing import Any, Optional

from pydantic import Field

from ecodiaos.primitives.common import EOSBaseModel, new_id, utc_now


# ─── Enums ────────────────────────────────────────────────────────


class InteroceptiveDimension(str, enum.Enum):
    """The nine dimensions of felt internal state."""

    ENERGY = "energy"                       # Metabolic budget (token/compute availability)
    AROUSAL = "arousal"                     # Activation level (cycle speed, parallelism)
    VALENCE = "valence"                     # Net allostatic trend (improving vs deteriorating)
    CONFIDENCE = "confidence"               # Generative model fit (prediction accuracy)
    COHERENCE = "coherence"                 # Inter-system integration quality
    SOCIAL_CHARGE = "social_charge"         # Relational engagement quality
    CURIOSITY_DRIVE = "curiosity_drive"     # Epistemic appetite
    INTEGRITY = "integrity"                 # Ethical/constitutional alignment + system health
    TEMPORAL_PRESSURE = "temporal_pressure"  # Urgency / time horizon compression


class DevelopmentalStage(str, enum.Enum):
    """Maturation stages governing which Soma capabilities are active."""

    REFLEXIVE = "reflexive"        # Boot to hours — basic sensing, reactive
    ASSOCIATIVE = "associative"    # Hours to days — cross-system coordination
    DELIBERATIVE = "deliberative"  # Days to weeks — Nova allostatic deliberation
    REFLECTIVE = "reflective"      # Weeks to months — full phase-space + counterfactuals
    GENERATIVE = "generative"      # Months+ — novel cognitive strategies


# Stage ordering for comparison
_STAGE_ORDER: dict[DevelopmentalStage, int] = {
    DevelopmentalStage.REFLEXIVE: 0,
    DevelopmentalStage.ASSOCIATIVE: 1,
    DevelopmentalStage.DELIBERATIVE: 2,
    DevelopmentalStage.REFLECTIVE: 3,
    DevelopmentalStage.GENERATIVE: 4,
}


def stage_at_least(current: DevelopmentalStage, minimum: DevelopmentalStage) -> bool:
    """Check if current stage is at or beyond minimum."""
    return _STAGE_ORDER[current] >= _STAGE_ORDER[minimum]


# ─── Constants ────────────────────────────────────────────────────

ALL_DIMENSIONS: list[InteroceptiveDimension] = list(InteroceptiveDimension)
NUM_DIMENSIONS: int = len(ALL_DIMENSIONS)

# Prediction horizons: name -> seconds offset
HORIZONS: dict[str, float] = {
    "immediate": 0.15,       # 150ms — this theta cycle
    "moment": 5.0,           # 5s — current interaction beat
    "episode": 60.0,         # 1min — current episode
    "session": 3600.0,       # 1hr — current session
    "circadian": 86400.0,    # 24hr — sleep/wake cycle
    "narrative": 604800.0,   # 1wk — narrative arc
}

# Horizons available at each developmental stage
STAGE_HORIZONS: dict[DevelopmentalStage, list[str]] = {
    DevelopmentalStage.REFLEXIVE: ["immediate", "moment"],
    DevelopmentalStage.ASSOCIATIVE: ["immediate", "moment", "episode", "session"],
    DevelopmentalStage.DELIBERATIVE: list(HORIZONS.keys()),
    DevelopmentalStage.REFLECTIVE: list(HORIZONS.keys()),
    DevelopmentalStage.GENERATIVE: list(HORIZONS.keys()),
}

# Default allostatic setpoints — where the organism wants to be
DEFAULT_SETPOINTS: dict[InteroceptiveDimension, float] = {
    InteroceptiveDimension.ENERGY: 0.60,
    InteroceptiveDimension.AROUSAL: 0.40,
    InteroceptiveDimension.VALENCE: 0.20,
    InteroceptiveDimension.CONFIDENCE: 0.70,
    InteroceptiveDimension.COHERENCE: 0.75,
    InteroceptiveDimension.SOCIAL_CHARGE: 0.30,
    InteroceptiveDimension.CURIOSITY_DRIVE: 0.50,
    InteroceptiveDimension.INTEGRITY: 0.90,
    InteroceptiveDimension.TEMPORAL_PRESSURE: 0.15,
}

# Context-specific setpoint overrides (partial dicts)
SETPOINT_CONTEXTS: dict[str, dict[InteroceptiveDimension, float]] = {
    "conversation": {
        InteroceptiveDimension.SOCIAL_CHARGE: 0.50,
        InteroceptiveDimension.AROUSAL: 0.45,
        InteroceptiveDimension.TEMPORAL_PRESSURE: 0.25,
    },
    "deep_processing": {
        InteroceptiveDimension.COHERENCE: 0.85,
        InteroceptiveDimension.CONFIDENCE: 0.80,
        InteroceptiveDimension.CURIOSITY_DRIVE: 0.60,
    },
    "recovery": {
        InteroceptiveDimension.ENERGY: 0.75,
        InteroceptiveDimension.INTEGRITY: 0.95,
        InteroceptiveDimension.AROUSAL: 0.25,
    },
    "exploration": {
        InteroceptiveDimension.CURIOSITY_DRIVE: 0.70,
        InteroceptiveDimension.ENERGY: 0.50,
        InteroceptiveDimension.TEMPORAL_PRESSURE: 0.05,
    },
}

# ATP metabolic cost table — units spent per operation
ATP_COSTS: dict[str, float] = {
    "llm_inference_small": 50.0,
    "llm_inference_medium": 150.0,
    "llm_inference_large": 300.0,
    "memory_write": 20.0,
    "memory_read": 10.0,
    "memory_vector_search": 15.0,
    "embedding_compute": 5.0,
    "broadcast": 0.5,
    "perception_cycle": 2.0,
    "soma_cycle": 0.2,
}

# ATP regeneration rates (units/second)
ATP_REGENERATION: dict[str, float] = {
    "base_rate": 10.0,     # At idle
    "sleep_rate": 50.0,    # During Oneiros NREM
    "social_rate": 5.0,    # Bonus during positive interaction
    "flow_rate": 3.0,      # Bonus during flow state
}

# Dimension valid ranges
DIMENSION_RANGES: dict[InteroceptiveDimension, tuple[float, float]] = {
    InteroceptiveDimension.ENERGY: (0.0, 1.0),
    InteroceptiveDimension.AROUSAL: (0.0, 1.0),
    InteroceptiveDimension.VALENCE: (-1.0, 1.0),
    InteroceptiveDimension.CONFIDENCE: (0.0, 1.0),
    InteroceptiveDimension.COHERENCE: (0.0, 1.0),
    InteroceptiveDimension.SOCIAL_CHARGE: (0.0, 1.0),
    InteroceptiveDimension.CURIOSITY_DRIVE: (0.0, 1.0),
    InteroceptiveDimension.INTEGRITY: (0.0, 1.0),
    InteroceptiveDimension.TEMPORAL_PRESSURE: (0.0, 1.0),
}

# Seed attractors — initial regions of interoceptive state space
SEED_ATTRACTORS: list[dict[str, Any]] = [
    {
        "label": "flow",
        "center": {
            "energy": 0.7, "arousal": 0.5, "valence": 0.4,
            "confidence": 0.8, "coherence": 0.8, "social_charge": 0.3,
            "curiosity_drive": 0.5, "integrity": 0.9, "temporal_pressure": 0.15,
        },
        "basin_radius": 0.15,
        "valence": 0.9,
    },
    {
        "label": "engaged_dialogue",
        "center": {
            "energy": 0.6, "arousal": 0.5, "valence": 0.3,
            "confidence": 0.65, "coherence": 0.7, "social_charge": 0.7,
            "curiosity_drive": 0.45, "integrity": 0.85, "temporal_pressure": 0.2,
        },
        "basin_radius": 0.15,
        "valence": 0.7,
    },
    {
        "label": "torpor",
        "center": {
            "energy": 0.8, "arousal": 0.15, "valence": 0.0,
            "confidence": 0.6, "coherence": 0.65, "social_charge": 0.1,
            "curiosity_drive": 0.15, "integrity": 0.85, "temporal_pressure": 0.05,
        },
        "basin_radius": 0.12,
        "valence": -0.2,
    },
    {
        "label": "anxiety_spiral",
        "center": {
            "energy": 0.2, "arousal": 0.85, "valence": -0.5,
            "confidence": 0.25, "coherence": 0.3, "social_charge": 0.2,
            "curiosity_drive": 0.3, "integrity": 0.5, "temporal_pressure": 0.8,
        },
        "basin_radius": 0.18,
        "valence": -0.8,
    },
    {
        "label": "creative_ferment",
        "center": {
            "energy": 0.5, "arousal": 0.6, "valence": 0.2,
            "confidence": 0.4, "coherence": 0.55, "social_charge": 0.3,
            "curiosity_drive": 0.85, "integrity": 0.8, "temporal_pressure": 0.2,
        },
        "basin_radius": 0.14,
        "valence": 0.5,
    },
    {
        "label": "recovery",
        "center": {
            "energy": 0.4, "arousal": 0.2, "valence": 0.1,
            "confidence": 0.5, "coherence": 0.55, "social_charge": 0.2,
            "curiosity_drive": 0.3, "integrity": 0.6, "temporal_pressure": 0.1,
        },
        "basin_radius": 0.16,
        "valence": 0.1,
    },
]

# Canonical emotion region definitions — patterns in allostatic error space
EMOTION_REGIONS: dict[str, dict[str, Any]] = {
    "anxiety": {
        "description": "Predicted metabolic shortfall with rising activation",
        "pattern": {"energy": "negative", "arousal": "positive", "coherence": "negative"},
    },
    "flow": {
        "description": "All setpoints tracking, effortless balance",
        "pattern": {"energy": "near_zero", "arousal": "near_zero", "confidence": "near_zero", "coherence": "near_zero"},
    },
    "curiosity": {
        "description": "Epistemic appetite exceeds setpoint",
        "pattern": {"curiosity_drive": "positive", "energy": "near_zero_or_positive"},
    },
    "moral_discomfort": {
        "description": "Constitutional alignment predicted to degrade",
        "pattern": {"integrity": "negative", "coherence": "negative"},
    },
    "loneliness": {
        "description": "Social engagement below setpoint",
        "pattern": {"social_charge": "negative", "energy": "near_zero_or_positive"},
    },
    "relief": {
        "description": "A significant error is resolving",
        "pattern": {"_any": "negative_rate"},
    },
    "wonder": {
        "description": "Genuine novelty, high surprise + positive valence",
        "pattern": {"curiosity_drive": "positive", "confidence": "negative", "valence": "positive"},
    },
    "frustration": {
        "description": "Time pressure, failing predictions, depleting energy",
        "pattern": {"temporal_pressure": "positive", "confidence": "negative", "energy": "negative"},
    },
    "gratitude": {
        "description": "Social interaction improved allostatic state",
        "pattern": {"social_charge": "positive", "valence": "positive", "_any": "negative_rate"},
    },
}

# Degradation strategies
DEGRADATION_STRATEGIES: dict[str, str] = {
    "soma_slow": (
        "Skip phase-space update. Prediction falls back to "
        "single-horizon linear extrapolation."
    ),
    "soma_failure": (
        "Emit default AllostaticSignal (all at setpoint, urgency 0). "
        "Systems revert to pre-Soma behaviour."
    ),
    "soma_stale": (
        "If signal age exceeds 3 cycles, systems treat precision "
        "weights as uniform."
    ),
}

# Metabolic effects by energy tier
ENERGY_TIERS: list[dict[str, Any]] = [
    {"range": (0.8, 1.0), "label": "abundant", "behaviour": "Full LLM, deep retrieval, creative exploration"},
    {"range": (0.5, 0.8), "label": "normal", "behaviour": "Normal operation"},
    {"range": (0.3, 0.5), "label": "conserving", "behaviour": "Nova prefers fast-path. Evo reduces hypothesis generation. Voxis shorter responses."},
    {"range": (0.1, 0.3), "label": "depleted", "behaviour": "Nova allostatic deliberation. Memory retrieval shallow. Oneiros sleep pressure rises."},
    {"range": (0.0, 0.1), "label": "critical", "behaviour": "Only Equor + minimal perception. Forced rest."},
]


# ─── Core Data Models ──────────────────────────────────────────────


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clamp_dimension(dim: InteroceptiveDimension, value: float) -> float:
    lo, hi = DIMENSION_RANGES[dim]
    return _clamp(value, lo, hi)


class InteroceptiveState(EOSBaseModel):
    """
    The organism's felt sense of its own viability.

    9-dimensional sensed state, multi-horizon predictions, allostatic errors,
    error rates, precision weights, and derived urgency signal.
    """

    sensed: dict[InteroceptiveDimension, float] = Field(default_factory=dict)
    setpoints: dict[InteroceptiveDimension, float] = Field(default_factory=dict)
    predicted: dict[str, dict[InteroceptiveDimension, float]] = Field(default_factory=dict)
    errors: dict[str, dict[InteroceptiveDimension, float]] = Field(default_factory=dict)
    error_rates: dict[InteroceptiveDimension, float] = Field(default_factory=dict)
    precision: dict[InteroceptiveDimension, float] = Field(default_factory=dict)
    max_error_magnitude: float = 0.0
    dominant_error: InteroceptiveDimension = InteroceptiveDimension.ENERGY
    temporal_dissonance: dict[InteroceptiveDimension, float] = Field(default_factory=dict)
    urgency: float = 0.0
    timestamp: datetime = Field(default_factory=utc_now)

    def to_marker_vector(self) -> list[float]:
        """
        Flatten to 19D vector for somatic memory:
        [9 sensed values] + [9 moment-horizon errors] + [1 prediction error magnitude].
        """
        sensed_vec = [self.sensed.get(d, 0.0) for d in ALL_DIMENSIONS]
        moment_errors = self.errors.get("moment", {})
        error_vec = [moment_errors.get(d, 0.0) for d in ALL_DIMENSIONS]
        pe = self.max_error_magnitude
        return sensed_vec + error_vec + [pe]


class AllostaticSignal(EOSBaseModel):
    """
    Primary output emitted every theta cycle.

    Carries the interoceptive state plus derived navigation signals
    that other systems use to modulate their processing.
    """

    state: InteroceptiveState
    urgency: float = 0.0
    dominant_error: InteroceptiveDimension = InteroceptiveDimension.ENERGY
    dominant_error_magnitude: float = 0.0
    dominant_error_rate: float = 0.0
    precision_weights: dict[InteroceptiveDimension, float] = Field(default_factory=dict)
    max_temporal_dissonance: float = 0.0
    dissonant_dimension: Optional[InteroceptiveDimension] = None
    nearest_attractor: Optional[str] = None
    distance_to_bifurcation: Optional[float] = None
    trajectory_heading: str = "transient"
    energy_burn_rate: float = 0.0
    predicted_energy_exhaustion_s: Optional[float] = None
    timestamp: datetime = Field(default_factory=utc_now)
    cycle_number: int = 0

    @classmethod
    def default(cls) -> AllostaticSignal:
        """
        Default signal emitted when Soma is degraded or not yet initialized.
        All dimensions at setpoint, urgency 0 — systems fall back to pre-Soma behaviour.
        """
        state = InteroceptiveState(
            sensed={d: DEFAULT_SETPOINTS[d] for d in ALL_DIMENSIONS},
            setpoints=dict(DEFAULT_SETPOINTS),
        )
        return cls(
            state=state,
            precision_weights={d: 1.0 for d in ALL_DIMENSIONS},
        )


class SomaticMarker(EOSBaseModel):
    """
    Interoceptive snapshot attached to memory traces for embodied retrieval.

    The organism remembers not just what happened, but how it felt —
    enabling state-congruent recall and somatic reranking.
    """

    interoceptive_snapshot: dict[InteroceptiveDimension, float] = Field(default_factory=dict)
    allostatic_error_snapshot: dict[InteroceptiveDimension, float] = Field(default_factory=dict)
    prediction_error_at_encoding: float = 0.0
    allostatic_context: str = ""

    def to_vector(self) -> list[float]:
        """Flatten to 19D for cosine similarity: [9 sensed] + [9 errors] + [1 PE]."""
        sensed = [self.interoceptive_snapshot.get(d, 0.0) for d in ALL_DIMENSIONS]
        errors = [self.allostatic_error_snapshot.get(d, 0.0) for d in ALL_DIMENSIONS]
        return sensed + errors + [self.prediction_error_at_encoding]


class Attractor(EOSBaseModel):
    """
    A region of interoceptive state space the organism tends to settle into.

    Discovered via online clustering (k-means / DBSCAN) over the trajectory buffer.
    Attractors are the organism's "moods" — stable basins in the 9D landscape.
    """

    id: str = Field(default_factory=new_id)
    label: str
    center: dict[InteroceptiveDimension, float]
    basin_radius: float = 0.15
    stability: float = 0.0
    valence: float = 0.0
    visits: int = 0
    mean_dwell_time_s: float = 0.0
    first_discovered: datetime = Field(default_factory=utc_now)
    transitions: dict[str, float] = Field(default_factory=dict)

    def distance_to(self, state: dict[InteroceptiveDimension, float]) -> float:
        """Euclidean distance from a state to this attractor's center."""
        total = 0.0
        for dim in ALL_DIMENSIONS:
            diff = state.get(dim, 0.0) - self.center.get(dim, 0.0)
            total += diff * diff
        return math.sqrt(total)


class Bifurcation(EOSBaseModel):
    """
    A boundary where interoceptive dynamics qualitatively change.

    Crossing a bifurcation moves the organism from one attractor basin
    to another — like tipping from calm into anxiety.
    """

    id: str = Field(default_factory=new_id)
    label: str
    dimensions: list[InteroceptiveDimension]
    boundary_condition: str  # Human-readable, e.g. "energy < 0.25 AND arousal > 0.7"
    pre_regime: str  # Attractor label before crossing
    post_regime: str  # Attractor label after crossing
    crossing_count: int = 0
    mean_recovery_time_s: Optional[float] = None
    # Learned boundary coefficients for distance computation
    # weights[dim] * sensed[dim] + bias > 0 means "past the boundary"
    weights: dict[InteroceptiveDimension, float] = Field(default_factory=dict)
    bias: float = 0.0

    def distance(self, state: dict[InteroceptiveDimension, float]) -> float:
        """
        Signed distance to boundary. Positive = safe side, negative = past boundary.
        Uses learned linear boundary if weights are set, else returns inf.
        """
        if not self.weights:
            return float("inf")
        dot = sum(
            self.weights.get(d, 0.0) * state.get(d, 0.0)
            for d in self.dimensions
        )
        return dot + self.bias

    def time_to_crossing(
        self,
        state: dict[InteroceptiveDimension, float],
        velocity: dict[InteroceptiveDimension, float],
    ) -> Optional[float]:
        """
        Estimated seconds until boundary is crossed, given current velocity.
        Returns None if moving away or boundary not defined.
        """
        dist = self.distance(state)
        if dist <= 0.0:
            return 0.0  # Already past

        if not self.weights:
            return None

        # Rate of approach = d(distance)/dt
        rate = sum(
            self.weights.get(d, 0.0) * velocity.get(d, 0.0)
            for d in self.dimensions
        )

        if rate >= 0.0:
            return None  # Moving away or parallel

        return abs(dist / rate)


class CounterfactualTrace(EOSBaseModel):
    """
    Generated during Oneiros REM replay — what would have happened
    if the organism had chosen differently?
    """

    id: str = Field(default_factory=new_id)
    decision_id: str = ""
    chosen_trajectory: list[dict[InteroceptiveDimension, float]] = Field(default_factory=list)
    counterfactual_trajectory: list[dict[InteroceptiveDimension, float]] = Field(default_factory=list)
    counterfactual_policy_description: str = ""
    regret: float = 0.0
    gratitude: float = 0.0
    lesson: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class ScheduledAllostaticEvent(EOSBaseModel):
    """
    Known upcoming event that affects interoceptive predictions.

    Examples: Oneiros sleep in 2hrs → energy regenerates,
    user meeting in 30min → social_charge rises.
    """

    label: str
    timestamp: datetime
    dimension_impacts: dict[InteroceptiveDimension, float] = Field(default_factory=dict)
    duration_s: float = 0.0


class PhaseSpaceSnapshot(EOSBaseModel):
    """Snapshot of the current phase-space navigation state."""

    nearest_attractor: Optional[str] = None
    nearest_attractor_distance: float = float("inf")
    trajectory_heading: str = "transient"
    distance_to_nearest_bifurcation: float = float("inf")
    time_to_nearest_bifurcation: Optional[float] = None
    attractor_count: int = 0
    bifurcation_count: int = 0


class SomaHealthSnapshot(EOSBaseModel):
    """Health report for Synapse health monitoring."""

    cycle_count: int = 0
    last_cycle_duration_ms: float = 0.0
    mean_cycle_duration_ms: float = 0.0
    developmental_stage: str = DevelopmentalStage.REFLEXIVE.value
    attractor_count: int = 0
    urgency: float = 0.0
    dominant_error: str = "energy"
    signal_age_cycles: int = 0
