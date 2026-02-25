"""
EcodiaOS — Soma Temporal Depth

Multi-scale prediction coordination and temporal dissonance computation.
Works with the predictor to manage forecasts at all 6 horizons and
compute the organism's sense of temporal coherence — whether its
short-term and long-term trajectories are aligned or in conflict.

High temporal dissonance signals Nova to deliberate time-horizon tradeoffs:
  Positive dissonance = feels good now, heading bad (temptation)
  Negative dissonance = feels bad now, heading good (perseverance)
"""

from __future__ import annotations

from typing import Optional

import structlog

from ecodiaos.systems.soma.types import (
    ALL_DIMENSIONS,
    HORIZONS,
    STAGE_HORIZONS,
    DevelopmentalStage,
    InteroceptiveDimension,
)

logger = structlog.get_logger("ecodiaos.systems.soma.temporal_depth")


class TemporalDepthManager:
    """
    Manages multi-scale temporal prediction and dissonance computation.

    Governs which horizons are active based on developmental stage
    and provides the dissonance signal that Nova uses for
    time-horizon deliberation.
    """

    def __init__(self) -> None:
        self._current_stage = DevelopmentalStage.REFLEXIVE
        self._dissonance_threshold = 0.2  # |dissonance| > this triggers Nova

    def set_stage(self, stage: DevelopmentalStage) -> None:
        self._current_stage = stage

    @property
    def available_horizons(self) -> list[str]:
        """Return horizon names available at the current developmental stage."""
        return STAGE_HORIZONS.get(self._current_stage, ["immediate", "moment"])

    def compute_dissonance(
        self,
        predictions: dict[str, dict[InteroceptiveDimension, float]],
    ) -> dict[InteroceptiveDimension, float]:
        """
        Temporal dissonance = moment prediction - session prediction.

        Positive = feels good now, heading bad later (temptation)
        Negative = feels bad now, heading good later (perseverance)
        """
        moment = predictions.get("moment", {})
        session = predictions.get("session", {})

        if not moment or not session:
            return {dim: 0.0 for dim in ALL_DIMENSIONS}

        return {
            dim: moment.get(dim, 0.0) - session.get(dim, 0.0)
            for dim in ALL_DIMENSIONS
        }

    def max_dissonance(
        self,
        dissonance: dict[InteroceptiveDimension, float],
    ) -> tuple[float, Optional[InteroceptiveDimension]]:
        """
        Find the dimension with maximum absolute temporal dissonance.
        Returns (max_value, dimension) or (0.0, None) if no dissonance.
        """
        if not dissonance:
            return 0.0, None

        max_val = 0.0
        max_dim: Optional[InteroceptiveDimension] = None

        for dim, val in dissonance.items():
            if abs(val) > abs(max_val):
                max_val = val
                max_dim = dim

        return max_val, max_dim

    def should_nova_deliberate(
        self,
        dissonance: dict[InteroceptiveDimension, float],
    ) -> bool:
        """
        Check if temporal dissonance exceeds threshold, warranting
        Nova deliberation on time-horizon tradeoffs.
        """
        max_val, _ = self.max_dissonance(dissonance)
        return abs(max_val) > self._dissonance_threshold

    def get_horizon_weights(self) -> dict[str, float]:
        """
        Return attention weights for each horizon based on stage.

        Earlier stages weight near-term horizons more heavily.
        Later stages distribute attention more evenly across scales.
        """
        horizons = self.available_horizons
        n = len(horizons)
        if n == 0:
            return {}

        if self._current_stage == DevelopmentalStage.REFLEXIVE:
            # Heavy near-term weighting
            weights = {h: 1.0 / (i + 1) for i, h in enumerate(horizons)}
        elif self._current_stage == DevelopmentalStage.ASSOCIATIVE:
            # Moderate near-term bias
            weights = {h: 1.0 / (i * 0.5 + 1) for i, h in enumerate(horizons)}
        else:
            # Even distribution with slight near-term preference
            weights = {h: 1.0 / (i * 0.3 + 1) for i, h in enumerate(horizons)}

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {h: w / total for h, w in weights.items()}

        return weights
