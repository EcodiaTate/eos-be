"""
EcodiaOS — Thread Chapter Detector

Detects narrative chapter boundaries in the stream of experience.
A chapter boundary means: the story has shifted. The organism is now
in a different phase of its life.

Boundaries emerge from genuine changes in what the organism is experiencing,
pursuing, and feeling — not arbitrary segmentation.

Algorithm: 5-factor weighted Bayesian surprise approximation with
spike detection, sustained shift detection, and goal resolution triggers.

Performance: boundary check ≤10ms per episode (pure computation, no LLM).
Chapter closure ≤5s (includes LLM narrative composition).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.primitives.common import utc_now

if TYPE_CHECKING:
    from ecodiaos.primitives.affect import AffectState
from ecodiaos.systems.thread.types import (
    ChapterStatus,
    NarrativeChapter,
    NarrativeSurpriseAccumulator,
    ThreadConfig,
)

logger = structlog.get_logger()


class ChapterDetector:
    """
    Detects chapter boundaries via multi-factor Bayesian surprise.

    Runs per-episode. Boundary check is pure computation (≤10ms).
    When a boundary is detected, returns True and the caller (ThreadService)
    triggers the async closure process.
    """

    def __init__(self, config: ThreadConfig) -> None:
        self._config = config
        self._acc = NarrativeSurpriseAccumulator()
        self._logger = logger.bind(system="thread.chapter_detector")
        self._last_schema_formation: float | None = None

    @property
    def accumulator(self) -> NarrativeSurpriseAccumulator:
        return self._acc

    def reset_for_new_chapter(self, chapter_id: str) -> None:
        """Reset the accumulator for a newly opened chapter."""
        self._acc.reset(chapter_id)
        self._logger.debug("accumulator_reset", chapter_id=chapter_id)

    def check_boundary(
        self,
        episode_data: dict[str, Any],
        affect: AffectState,
        schema_challenged: bool = False,
    ) -> bool:
        """
        Evaluate whether the current episode marks a chapter boundary.
        Runs per-episode. Must complete in ≤10ms (no LLM calls).

        Args:
            episode_data: Dict with keys like 'affect_valence', 'affect_arousal',
                          'has_goal_completion', 'has_goal_failure', 'has_goal_creation',
                          'has_new_core_entity', 'context_domain'.
            affect: Current organism affect state.
            schema_challenged: Whether this episode challenged an ESTABLISHED+ schema.

        Returns:
            True if a chapter boundary is detected.
        """
        surprise = self._compute_episode_surprise(episode_data, affect, schema_challenged)

        # Update exponential moving average
        alpha = self._config.surprise_ema_alpha
        self._acc.surprise_ema = alpha * surprise + (1 - alpha) * self._acc.surprise_ema
        self._acc.cumulative_surprise += surprise
        self._acc.episodes_in_chapter += 1

        # Update affect EMAs
        ep_valence = float(episode_data.get("affect_valence", 0.0))
        ep_arousal = float(episode_data.get("affect_arousal", affect.arousal))
        self._acc.affect_ema_valence = (
            alpha * ep_valence + (1 - alpha) * self._acc.affect_ema_valence
        )
        self._acc.affect_ema_arousal = (
            alpha * ep_arousal + (1 - alpha) * self._acc.affect_ema_arousal
        )

        # Track goal events
        if episode_data.get("has_goal_completion"):
            self._acc.goal_completions_in_window += 1
        if episode_data.get("has_goal_failure"):
            self._acc.goal_failures_in_window += 1
        if schema_challenged:
            self._acc.schema_challenges_in_window += 1

        # --- Boundary conditions ---

        # 1. Surprise spike: current surprise > N × EMA
        spike = surprise > self._config.surprise_spike_multiplier * max(self._acc.surprise_ema, 0.1)

        # 2. Sustained shift: EMA has risen > N × chapter-start baseline
        sustained = self._acc.surprise_ema > self._config.surprise_sustained_multiplier * max(
            self._acc.surprise_ema_baseline, 0.1
        )

        # 3. Goal resolution: major goal completed or failed
        goal_resolution = (
            self._acc.goal_completions_in_window > 0
            or self._acc.goal_failures_in_window > 0
        )

        # 4. Temporal guards
        min_length_met = self._acc.episodes_in_chapter >= self._config.chapter_min_episodes
        max_length_exceeded = self._acc.episodes_in_chapter >= self._config.chapter_max_episodes

        # Force boundary at max length regardless
        if max_length_exceeded:
            self._logger.info(
                "chapter_boundary_forced",
                reason="max_length",
                episodes=self._acc.episodes_in_chapter,
            )
            return True

        # Require minimum length before evaluating
        if min_length_met and (spike or sustained or goal_resolution):
            # Confirm with secondary check: has affect trajectory meaningfully shifted?
            affect_shifted = (
                abs(affect.valence - self._acc.affect_ema_valence)
                > self._config.affect_shift_threshold
            )

            if spike or sustained or (goal_resolution and affect_shifted):
                reason = "spike" if spike else ("sustained" if sustained else "goal_resolution")
                self._logger.info(
                    "chapter_boundary_detected",
                    reason=reason,
                    episodes=self._acc.episodes_in_chapter,
                    surprise=round(surprise, 4),
                    ema=round(self._acc.surprise_ema, 4),
                )
                return True

        return False

    def _compute_episode_surprise(
        self,
        episode_data: dict[str, Any],
        affect: AffectState,
        schema_challenged: bool,
    ) -> float:
        """
        Compute 5-factor weighted surprise for a single episode.
        Pure computation, no I/O.
        """
        cfg = self._config

        # Factor 1: Affect delta
        affect_signal = self._compute_affect_delta(episode_data)

        # Factor 2: Goal event
        goal_signal = self._compute_goal_event(episode_data)

        # Factor 3: Context shift
        context_signal = self._compute_context_shift(episode_data)

        # Factor 4: New core entity
        entity_signal = 1.0 if episode_data.get("has_new_core_entity") else 0.0

        # Factor 5: Schema challenge
        schema_signal = 0.0
        if schema_challenged:
            challenge_strength = float(episode_data.get("schema_challenge_strength", 0.5))
            schema_signal = 0.8 + 0.2 * challenge_strength

        surprise = (
            cfg.surprise_weight_affect * affect_signal
            + cfg.surprise_weight_goal * goal_signal
            + cfg.surprise_weight_context * context_signal
            + cfg.surprise_weight_entity * entity_signal
            + cfg.surprise_weight_schema * schema_signal
        )

        return float(surprise)

    def _compute_affect_delta(self, episode_data: dict[str, Any]) -> float:
        """Affect delta relative to running chapter mean. Returns 0.0-1.0."""
        ep_valence = float(episode_data.get("affect_valence", 0.0))
        ep_arousal = float(episode_data.get("affect_arousal", 0.0))
        valence_delta = abs(ep_valence - self._acc.affect_ema_valence)
        arousal_delta = abs(ep_arousal - self._acc.affect_ema_arousal)
        return float(min(1.0, max(valence_delta, arousal_delta) / 0.5))

    def _compute_goal_event(self, episode_data: dict[str, Any]) -> float:
        """Check if episode contains goal resolution event."""
        if episode_data.get("has_goal_failure"):
            return 1.0
        if episode_data.get("has_goal_completion"):
            return 0.8
        if episode_data.get("has_goal_creation"):
            return 0.3
        return 0.0

    def _compute_context_shift(self, episode_data: dict[str, Any]) -> float:
        """Heuristic for context/domain change. Returns 0.0-1.0."""
        if episode_data.get("is_new_domain"):
            return 0.8
        if episode_data.get("is_new_community_interaction"):
            return 0.6
        return 0.0

    def create_new_chapter(
        self,
        previous_chapter: NarrativeChapter | None = None,
        personality_snapshot: dict[str, float] | None = None,
        active_schema_ids: list[str] | None = None,
    ) -> NarrativeChapter:
        """
        Create a new FORMING chapter. Called after boundary detection
        and closure of the previous chapter.
        """
        chapter = NarrativeChapter(
            status=ChapterStatus.FORMING,
            started_at=utc_now(),
            personality_snapshot_start=personality_snapshot or {},
            active_schema_ids=active_schema_ids or [],
        )
        self.reset_for_new_chapter(chapter.id)
        self._logger.info("chapter_opened", chapter_id=chapter.id)
        return chapter
