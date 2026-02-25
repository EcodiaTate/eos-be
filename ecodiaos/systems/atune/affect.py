"""
Atune — Affective State Management.

Maintains and updates the global :class:`AffectState` that modulates all
processing across every cognitive system.  This is NOT emotion simulation —
it is a **functional analog**: the system's processing state that
modulates attention and decision-making.

An organism under high care-activation *literally perceives differently*
than one in a learning state.  This is architectural, not cosmetic.

Mood dynamics (v2):
    The affect system now distinguishes between fast-changing *reactive*
    affect (percept-driven, decays quickly) and slow-changing *mood*
    (baseline that drifts over many cycles). This mirrors the neuroscience
    distinction between emotion (seconds) and mood (hours/days).

    Additionally:
    * **Negativity bias** — negative valence persists ~2x longer than positive.
    * **Emotional memory** — high-arousal events leave a stronger imprint on mood.
    * **Mood floor** — mood can't swing as widely as reactive affect.
"""

from __future__ import annotations

import structlog

from ecodiaos.primitives.affect import AffectState
from ecodiaos.primitives.common import utc_now
from ecodiaos.primitives.percept import Percept

from .helpers import clamp, detect_distress
from .salience import analyse_sentiment
from .types import PredictionError, PredictionErrorDirection, SystemLoad

logger = structlog.get_logger("ecodiaos.systems.atune.affect")

# ---------------------------------------------------------------------------
# Default resting values
# ---------------------------------------------------------------------------

_RESTING_AROUSAL = 0.3
_RESTING_CURIOSITY = 0.4
_RESTING_DOMINANCE = 0.5

# Mood dynamics constants
_MOOD_INERTIA = 0.995         # Mood changes very slowly (0.5% per cycle)
_MOOD_IMPACT_RATE = 0.005     # How much reactive affect shifts mood per cycle
_MOOD_AROUSAL_BOOST = 2.0     # High-arousal events impact mood more strongly
_NEGATIVITY_BIAS = 1.8        # Negative affect persists ~1.8x longer than positive
_MOOD_VALENCE_RANGE = 0.4     # Mood valence clamped to [-0.4, 0.4] (narrower than affect)


# ---------------------------------------------------------------------------
# Affect manager
# ---------------------------------------------------------------------------


class AffectManager:
    """
    Tracks and updates the global AffectState each workspace cycle.

    Maintains two layers:
    * **Reactive affect** — fast-changing, percept-driven, decays quickly.
    * **Mood baseline** — slow-changing, reflects cumulative experience.

    The output AffectState blends both: 70% reactive + 30% mood baseline.
    This means even in the absence of new percepts, the organism's affect
    state reflects its recent emotional history.
    """

    def __init__(
        self,
        initial_affect: AffectState | None = None,
        persist_interval: int = 10,
    ) -> None:
        self._current = initial_affect or AffectState(
            valence=0.0,
            arousal=_RESTING_AROUSAL,
            dominance=_RESTING_DOMINANCE,
            curiosity=_RESTING_CURIOSITY,
            care_activation=0.0,
            coherence_stress=0.0,
            source_events=[],
            timestamp=utc_now(),
        )
        self._persist_interval = persist_interval
        self._cycles_since_persist: int = 0
        self._logger = logger.bind(component="affect_manager")

        # ── Mood baseline (slow-changing) ──
        self._mood_valence: float = 0.0
        self._mood_arousal: float = _RESTING_AROUSAL
        self._mood_stress: float = 0.0

        # ── Emotional memory (recent high-arousal events) ──
        # Ring buffer of (valence, arousal) from recent high-impact events.
        # Influences mood drift direction.
        self._emotional_memory: list[tuple[float, float]] = []
        self._emotional_memory_max: int = 20

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------

    @property
    def current(self) -> AffectState:
        return self._current

    @property
    def mood_valence(self) -> float:
        """The slow-changing mood baseline valence."""
        return self._mood_valence

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    async def update(
        self,
        percept: Percept | None,
        prediction_error: PredictionError | None,
        system_load: SystemLoad,
        precision_weights: dict[str, float] | None = None,
    ) -> AffectState:
        """
        Produce a new :class:`AffectState` from the current inputs.

        Each dimension is updated with an inertia coefficient so the state
        cannot jump wildly between cycles. The mood baseline is updated
        separately at a much slower rate.

        Soma's precision_weights modulate inertia: higher precision → trust current
        value more (increase inertia), lower precision → adapt faster (decrease inertia).
        """
        cur = self._current
        precision_weights = precision_weights or {}

        # ── Valence (positive / negative) ────────────────────────────
        if percept is not None:
            text = percept.content.parsed if isinstance(percept.content.parsed, str) else ""
            sentiment = await analyse_sentiment(text)
            reactive_valence = sentiment.valence

            # Negativity bias: negative valence has higher inertia
            if reactive_valence < 0:
                # Negative: 80% inertia (persists longer)
                inertia = 0.80
            else:
                # Positive: 88% inertia (decays faster toward neutral)
                inertia = 0.88

            # Apply Soma precision modulation: higher precision → more inertia
            # precision_weights maps InteroceptiveDimension (e.g. "valence") to 0-1 confidence
            valence_precision = precision_weights.get("valence", 1.0)
            # Adjust inertia: 1.0 precision = base inertia, 0.0 = fully adaptive
            inertia = inertia * valence_precision + (1.0 - valence_precision) * 0.5

            new_valence = cur.valence * inertia + reactive_valence * (1.0 - inertia)

            # Emotional memory: record high-arousal events
            if sentiment.arousal > 0.5:
                self._emotional_memory.append((reactive_valence, sentiment.arousal))
                if len(self._emotional_memory) > self._emotional_memory_max:
                    self._emotional_memory = self._emotional_memory[-self._emotional_memory_max:]
        else:
            # Decay toward mood baseline (not neutral zero!)
            # This is the key change: without input, affect drifts to mood, not zero
            decay_target = self._mood_valence
            if cur.valence < 0:
                # Negativity bias: negative decays slower
                new_valence = cur.valence * 0.99 + decay_target * 0.01
            else:
                new_valence = cur.valence * 0.97 + decay_target * 0.03

        # ── Arousal (activation level) ───────────────────────────────
        if prediction_error is not None and prediction_error.magnitude > 0.5:
            new_arousal = min(1.0, cur.arousal + prediction_error.magnitude * 0.1)
        else:
            # Decay toward mood baseline arousal
            arousal_precision = precision_weights.get("arousal", 1.0)
            # Higher precision → more inertia (trust current arousal more)
            arousal_inertia = 0.94 * arousal_precision + 0.6 * (1.0 - arousal_precision)
            new_arousal = cur.arousal * arousal_inertia + self._mood_arousal * (1.0 - arousal_inertia)

        # System load increases arousal
        new_arousal = min(1.0, new_arousal + system_load.cpu_utilisation * 0.05)

        # ── Dominance (sense of control) ─────────────────────────────
        # Mainly updated by Axon feedback; gentle drift toward resting here
        new_dominance = cur.dominance * 0.98 + _RESTING_DOMINANCE * 0.02

        # ── Curiosity (epistemic drive) ──────────────────────────────
        if prediction_error is not None:
            if prediction_error.direction == PredictionErrorDirection.NOVEL:
                new_curiosity = min(1.0, cur.curiosity + 0.05)
            elif prediction_error.direction == PredictionErrorDirection.CONTRADICTS_BELIEF:
                new_curiosity = min(1.0, cur.curiosity + 0.08)
            else:
                new_curiosity = cur.curiosity * 0.97 + _RESTING_CURIOSITY * 0.03
        else:
            new_curiosity = cur.curiosity * 0.97 + _RESTING_CURIOSITY * 0.03

        # ── Care activation ──────────────────────────────────────────
        if percept is not None:
            text = percept.content.parsed if isinstance(percept.content.parsed, str) else ""
            distress = detect_distress(text)
            if distress > 0.3:
                new_care = min(1.0, cur.care_activation + distress * 0.2)
            else:
                new_care = cur.care_activation * 0.95
        else:
            new_care = cur.care_activation * 0.95

        # ── Coherence stress (prediction error accumulation) ─────────
        if prediction_error is not None:
            new_stress = cur.coherence_stress * 0.9 + prediction_error.magnitude * 0.1
        else:
            # Decay toward mood stress baseline
            new_stress = cur.coherence_stress * 0.94 + self._mood_stress * 0.06

        # ── Update mood baseline (very slow drift) ───────────────────
        self._update_mood(new_valence, new_arousal, new_stress)

        # ── Assemble ─────────────────────────────────────────────────
        new_affect = AffectState(
            valence=clamp(new_valence, -1.0, 1.0),
            arousal=clamp(new_arousal, 0.0, 1.0),
            dominance=clamp(new_dominance, 0.0, 1.0),
            curiosity=clamp(new_curiosity, 0.0, 1.0),
            care_activation=clamp(new_care, 0.0, 1.0),
            coherence_stress=clamp(new_stress, 0.0, 1.0),
            source_events=[percept.id] if percept else [],
            timestamp=utc_now(),
        )

        self._current = new_affect
        self._cycles_since_persist += 1

        return new_affect

    # ------------------------------------------------------------------
    # Mood dynamics
    # ------------------------------------------------------------------

    def _update_mood(
        self,
        reactive_valence: float,
        reactive_arousal: float,
        reactive_stress: float,
    ) -> None:
        """
        Drift the mood baseline toward the current reactive affect.

        Higher-arousal events impact mood more strongly (emotional memory).
        Negative events shift mood more than positive ones (negativity bias).
        """
        # Arousal amplification: high-arousal states impact mood more
        arousal_factor = 1.0 + (reactive_arousal - 0.3) * _MOOD_AROUSAL_BOOST
        arousal_factor = max(0.5, min(3.0, arousal_factor))

        # Valence mood drift
        valence_delta = reactive_valence - self._mood_valence
        if valence_delta < 0:
            # Negativity bias: negative shifts have more impact
            impact = _MOOD_IMPACT_RATE * _NEGATIVITY_BIAS * arousal_factor
        else:
            impact = _MOOD_IMPACT_RATE * arousal_factor

        self._mood_valence = clamp(
            self._mood_valence * _MOOD_INERTIA + valence_delta * impact,
            -_MOOD_VALENCE_RANGE,
            _MOOD_VALENCE_RANGE,
        )

        # Arousal mood drift (simpler — just slow average)
        self._mood_arousal = (
            self._mood_arousal * _MOOD_INERTIA
            + reactive_arousal * (1.0 - _MOOD_INERTIA)
        )

        # Stress mood drift
        self._mood_stress = (
            self._mood_stress * _MOOD_INERTIA
            + reactive_stress * (1.0 - _MOOD_INERTIA)
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @property
    def needs_persist(self) -> bool:
        """True when the affect state should be written to Memory."""
        return self._cycles_since_persist >= self._persist_interval

    def mark_persisted(self) -> None:
        self._cycles_since_persist = 0

    # ------------------------------------------------------------------
    # External overrides (e.g. Axon feedback on dominance)
    # ------------------------------------------------------------------

    def nudge_dominance(self, delta: float) -> None:
        """Apply a small delta to dominance (used by Axon success/failure feedback)."""
        self._current = self._current.model_copy(
            update={"dominance": clamp(self._current.dominance + delta, 0.0, 1.0)}
        )

    def nudge_valence(self, delta: float) -> None:
        """Apply a small delta to valence."""
        self._current = self._current.model_copy(
            update={"valence": clamp(self._current.valence + delta, -1.0, 1.0)}
        )
