"""
Atune — Prediction Error Computation.

Implements the predictive processing framework (Clark 2013, Friston 2010).
Every Percept is compared against the instance's current expectations to
compute a **prediction error** — the "surprise" signal that drives the
Free Energy Principle.

    F = E_q[ln q(s) - ln p(o,s)]

We approximate this with embedding distance plus optional semantic
divergence, producing a :class:`PredictionError` that feeds into salience
scoring.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol

import structlog

from .helpers import clamp, cosine_similarity
from .types import PredictionError, PredictionErrorDirection

if TYPE_CHECKING:
    from ecodiaos.primitives.percept import Percept

logger = structlog.get_logger("ecodiaos.systems.atune.prediction")


# ---------------------------------------------------------------------------
# Belief state protocol
# ---------------------------------------------------------------------------


class BeliefPrediction:
    """What the belief model predicted for a given source."""

    embedding: list[float]
    predicted_content: str | None

    def __init__(self, embedding: list[float], predicted_content: str | None = None):
        self.embedding = embedding
        self.predicted_content = predicted_content


class BeliefStateReader(Protocol):
    """Minimal interface Atune needs from the belief / memory layer."""

    async def predict_for_source(self, source_system: str) -> BeliefPrediction | None:
        """Return the expected embedding for the next Percept from *source_system*."""
        ...


# ---------------------------------------------------------------------------
# Surprise classification
# ---------------------------------------------------------------------------


def _classify_surprise(
    magnitude: float,
    percept_embedding: list[float],
    expected_embedding: list[float] | None,
) -> PredictionErrorDirection:
    """
    Decide the *direction* of surprise:

    * ``contradicts_belief`` — high distance AND content that opposes
      existing belief.
    * ``novel`` — high distance, no prior expectation, or unfamiliar domain.
    * ``confirms_unexpected`` — moderate distance; the content was possible
      but the model didn't weight it highly.
    * ``expected`` — low distance; prediction was accurate.
    """
    if expected_embedding is None:
        return PredictionErrorDirection.NOVEL

    if magnitude > 0.65:
        return PredictionErrorDirection.CONTRADICTS_BELIEF
    if magnitude > 0.35:
        return PredictionErrorDirection.NOVEL
    if magnitude > 0.15:
        return PredictionErrorDirection.CONFIRMS_UNEXPECTED
    return PredictionErrorDirection.EXPECTED


# ---------------------------------------------------------------------------
# Semantic divergence (lightweight)
# ---------------------------------------------------------------------------


async def _compute_semantic_divergence(
    parsed_text: str | None,
    predicted_content: str | None,
    embed_fn: object,
) -> float:
    """
    Estimate semantic divergence between observed content and predicted
    content.  Falls back to a moderate surprise if either side is missing.
    """
    if not parsed_text or not predicted_content:
        return 0.3  # Moderate default when we can't compare

    # Embed both and measure distance
    observed_emb: list[float] = await embed_fn(parsed_text)  # type: ignore[operator]
    predicted_emb: list[float] = await embed_fn(predicted_content)  # type: ignore[operator]
    return clamp(1.0 - cosine_similarity(observed_emb, predicted_emb), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def compute_prediction_error(
    percept: Percept,
    belief_state: BeliefStateReader | None,
    embed_fn: object,
) -> PredictionError:
    """
    Compute how surprising *percept* is given the current belief state.

    The result combines:
    * **Embedding distance** between actual and expected (0.6 weight).
    * **Semantic divergence** via a second embedding comparison (0.4 weight).

    When no belief prediction exists the Percept is treated as moderately
    novel (magnitude 0.5).
    """
    source_system = percept.source.system

    # --- Retrieve expectation -------------------------------------------------
    expected: BeliefPrediction | None = None
    if belief_state is not None:
        try:
            expected = await belief_state.predict_for_source(source_system)
        except Exception:
            logger.debug("belief_prediction_failed", source=source_system)

    if expected is None:
        return PredictionError(
            magnitude=0.5,
            direction=PredictionErrorDirection.NOVEL,
            domain=source_system,
        )

    # --- Embedding distance ---------------------------------------------------
    # When the belief prediction has a valid embedding, use both embedding
    # distance and semantic divergence. When only predicted_content is
    # available (embedding=[]), rely entirely on semantic divergence.
    percept_embedding = percept.content.embedding
    has_embedding = (
        bool(expected.embedding)
        and percept_embedding is not None
        and len(expected.embedding) == len(percept_embedding)
    )

    if has_embedding and percept_embedding is not None:
        embedding_distance: float | None = 1.0 - cosine_similarity(
            percept_embedding,
            expected.embedding,
        )
    else:
        embedding_distance = None  # Will use semantic divergence only

    # --- Semantic divergence --------------------------------------------------
    parsed_text = percept.content.parsed if isinstance(percept.content.parsed, str) else None
    semantic_surprise = await _compute_semantic_divergence(
        parsed_text,
        expected.predicted_content,
        embed_fn,
    )

    # --- Combine --------------------------------------------------------------
    if embedding_distance is not None:
        magnitude = clamp(0.6 * embedding_distance + 0.4 * semantic_surprise, 0.0, 1.0)
    else:
        # Semantic divergence only — still meaningful prediction error
        magnitude = clamp(semantic_surprise, 0.0, 1.0)

    direction = _classify_surprise(
        magnitude,
        percept_embedding or [],
        expected.embedding,
    )

    logger.debug(
        "prediction_error_computed",
        percept_id=percept.id,
        magnitude=round(magnitude, 4),
        direction=direction.value,
        source=source_system,
    )

    return PredictionError(
        magnitude=magnitude,
        direction=direction,
        domain=source_system,
        expected_embedding=expected.embedding,
        actual_embedding=percept.content.embedding,
    )
