"""
Atune — Meta-Attention Controller.

Dynamically adjusts salience head weights based on the current situation.
This is what makes Atune **adaptive** rather than static: the attention
system can reconfigure itself moment-to-moment.

Over time, Evo observes which head-weight configurations lead to better
outcomes and proposes adjustments to the base weights, personalising the
organism's attention patterns through experience.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from .helpers import clamp
from .salience import ALL_HEADS

if TYPE_CHECKING:
    from ecodiaos.primitives.affect import AffectState

    from .types import MetaContext

logger = structlog.get_logger("ecodiaos.systems.atune.meta")


# ---------------------------------------------------------------------------
# Detected attention mode (for observability)
# ---------------------------------------------------------------------------


class AttentionMode:
    """Label for the current meta-attention state."""

    CRISIS = "crisis"
    CARE = "care"
    LEARNING = "learning"
    COHERENCE_REPAIR = "coherence_repair"
    ROUTINE = "routine"


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class MetaAttentionController:
    """
    Dynamically adjusts salience head weights based on situation.

    Examples
    --------
    * During a crisis: Risk head weight increases, Keyword head decreases.
    * During learning: Novelty and Causal heads increase.
    * During caregiving: Emotional head increases, Goal head may decrease.
    * During routine: all heads near baseline.
    """

    def __init__(self) -> None:
        self._current_mode: str = AttentionMode.ROUTINE
        self._evo_adjustments: dict[str, float] = {}
        self._logger = logger.bind(component="meta_attention")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def compute_head_weights(
        self,
        affect: AffectState,
        context: MetaContext,
    ) -> dict[str, float]:
        """
        Return a normalised weight dict ``{head_name: weight}`` for the
        current situation.
        """
        # Start from base weights (+ any Evo adjustments)
        weights: dict[str, float] = {}
        for head in ALL_HEADS:
            base = head.base_weight + self._evo_adjustments.get(head.name, 0.0)
            weights[head.name] = max(0.01, base)

        mode = AttentionMode.ROUTINE

        # ── Crisis mode: arousal > 0.8 AND risk detected ─────────────
        if affect.arousal > 0.8 and context.risk_level > 0.6:
            weights["risk"] *= 1.5
            weights["emotional"] *= 1.3
            weights["keyword"] *= 0.7
            weights["novelty"] *= 0.8
            mode = AttentionMode.CRISIS

        # ── Care mode: care_activation > 0.7 ─────────────────────────
        elif affect.care_activation > 0.7:
            weights["emotional"] *= 1.4
            weights["identity"] *= 1.2
            weights["causal"] *= 0.8
            mode = AttentionMode.CARE

        # ── Learning mode: high curiosity AND low stress ─────────────
        elif affect.curiosity > 0.7 and affect.coherence_stress < 0.4:
            weights["novelty"] *= 1.4
            weights["causal"] *= 1.3
            weights["risk"] *= 0.8
            mode = AttentionMode.LEARNING

        # ── Coherence repair: high coherence stress ──────────────────
        elif affect.coherence_stress > 0.7:
            weights["novelty"] *= 1.3
            weights["causal"] *= 1.2
            weights["keyword"] *= 0.7
            mode = AttentionMode.COHERENCE_REPAIR

        # ── Rhythm-state modulation from Synapse ──────────────────
        # The rhythm state is an emergent metacognitive signal. When the
        # organism detects it is in a particular cognitive state, attention
        # shifts accordingly — protecting flow, responding to stress,
        # combating boredom.
        rhythm = context.rhythm_state
        if rhythm == "stress":
            # Filter more noise, focus on critical signals
            weights["risk"] *= 1.3
            weights["novelty"] *= 0.7
        elif rhythm == "flow":
            # Protect the current focus — increase habituation decay rate
            # and suppress low-priority novelty
            weights["goal"] *= 1.3
            weights["novelty"] *= 0.8
        elif rhythm == "boredom":
            # Seek stimulation — boost novelty, increase curiosity-driven attention
            weights["novelty"] *= 1.4
            weights["causal"] *= 1.2
            weights["risk"] *= 0.8
        elif rhythm == "deep_processing":
            # Deep deliberation — boost causal understanding and identity relevance
            weights["causal"] *= 1.3
            weights["identity"] *= 1.2
            weights["keyword"] *= 0.8

        # Normalise so weights sum to 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        if mode != self._current_mode:
            self._logger.info(
                "attention_mode_changed",
                old_mode=self._current_mode,
                new_mode=mode,
            )
            self._current_mode = mode

        return weights

    # ------------------------------------------------------------------
    # Evo integration
    # ------------------------------------------------------------------

    def apply_evo_adjustments(self, adjustments: dict[str, float]) -> None:
        """
        Evo proposes small changes to base head weights after observing
        which configurations lead to better outcomes.

        Parameters
        ----------
        adjustments:
            ``{head_name: delta}`` — deltas are *added* to the base weight.
            Deltas are clamped to ±0.05 per application to prevent sudden
            shifts.
        """
        for name, delta in adjustments.items():
            clamped = clamp(delta, -0.05, 0.05)
            current = self._evo_adjustments.get(name, 0.0)
            # Total accumulated adjustment capped at ±0.15
            self._evo_adjustments[name] = clamp(current + clamped, -0.15, 0.15)

        self._logger.info(
            "evo_adjustments_applied",
            adjustments=adjustments,
            accumulated=dict(self._evo_adjustments),
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def current_mode(self) -> str:
        return self._current_mode

    @property
    def evo_adjustments(self) -> dict[str, float]:
        return dict(self._evo_adjustments)
