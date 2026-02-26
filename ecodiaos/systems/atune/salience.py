"""
Atune — Multi-Head Salience Engine.

Seven specialised attention heads, each tuned to detect a different
dimension of importance.  Architecturally inspired by transformer
multi-head attention, applied to the problem of determining
"what matters in the world right now."

Each head returns a score in [0, 1].  Scores are precision-weighted by
the current :class:`AffectState` (Fristonian precision = attention),
then combined using learned base-weights into a composite
:class:`SalienceVector`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

import structlog

from .helpers import (
    clamp,
    cosine_similarity,
    detect_causal_language,
    detect_conflict,
    detect_direct_address,
    detect_distress,
    detect_positive_emotion,
    detect_risk_patterns,
    detect_urgency,
    estimate_consequence_scope,
    estimate_temporal_proximity,
    match_keyword_set,
)
from .types import AttentionContext, SalienceVector

if TYPE_CHECKING:
    from ecodiaos.primitives.affect import AffectState
    from ecodiaos.primitives.percept import Percept

logger = structlog.get_logger("ecodiaos.systems.atune.salience")


# ---------------------------------------------------------------------------
# Memory interface (minimal read protocol for the Risk head)
# ---------------------------------------------------------------------------


class BadOutcomeLookup(Protocol):
    async def find_similar_bad_outcomes(
        self, embedding: list[float], threshold: float
    ) -> float:
        """Return a 0-1 similarity score to known bad outcomes."""
        ...


# ---------------------------------------------------------------------------
# Sentiment analysis stub
# ---------------------------------------------------------------------------


class SentimentResult:
    __slots__ = ("valence", "arousal")

    def __init__(self, valence: float = 0.0, arousal: float = 0.0):
        self.valence = valence
        self.arousal = arousal


async def analyse_sentiment(text: str | None) -> SentimentResult:
    """
    Lightweight heuristic sentiment.

    In production this would call the LLM or a dedicated model.  The
    heuristic version lets the pipeline run without an LLM call on every
    cycle, keeping us within the ≤40 ms budget for all heads.
    """
    if not text:
        return SentimentResult()

    positive = detect_positive_emotion(text)
    distress = detect_distress(text)
    conflict = detect_conflict(text)
    negative = max(distress, conflict)

    valence = clamp(positive - negative, -1.0, 1.0)
    arousal = clamp(max(positive, negative), 0.0, 1.0)
    return SentimentResult(valence=valence, arousal=arousal)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class SalienceHead:
    """Base class for all salience scoring heads."""

    name: str = "base"
    base_weight: float = 0.0
    precision_sensitivity: dict[str, float] = {}

    async def score(self, percept: Percept, context: AttentionContext) -> float:
        """Return salience in [0.0, 1.0] for this dimension."""
        raise NotImplementedError

    def _text(self, percept: Percept) -> str:
        """Convenience: extract plain-text from a Percept."""
        if isinstance(percept.content.parsed, str):
            return percept.content.parsed
        return percept.content.raw if isinstance(percept.content.raw, str) else ""


# ---------------------------------------------------------------------------
# Head 1: Novelty
# ---------------------------------------------------------------------------


class NoveltyHead(SalienceHead):
    """
    Detects new, unexpected, or contradictory information.
    Neuroscience analog: hippocampal novelty detection.
    """

    name = "novelty"
    base_weight = 0.20
    precision_sensitivity = {"curiosity": 0.4, "coherence_stress": 0.2}

    async def score(self, percept: Percept, context: AttentionContext) -> float:
        pe = context.prediction_error
        raw_novelty = pe.magnitude

        # Contradiction bonus
        if pe.direction.value == "contradicts_belief":
            raw_novelty = min(1.0, raw_novelty * 1.3)

        # Habituation decay
        habituation = context.source_habituation.get(percept.source.system, 0.0)

        return clamp(raw_novelty * (1.0 - habituation * 0.5), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Head 2: Risk
# ---------------------------------------------------------------------------


class RiskHead(SalienceHead):
    """
    Detects potential threats, dangers, or negative developments.
    Neuroscience analog: amygdala threat detection.

    Maintains a small in-memory cache of embeddings from episodes that
    had negative outcomes (low affect valence, failed actions). When a
    new percept resembles a past bad outcome, the risk score rises.
    """

    name = "risk"
    base_weight = 0.18
    precision_sensitivity = {"arousal": 0.5, "care_activation": 0.3}

    # Class-level shared cache of bad-outcome embeddings (populated by Synapse/Evo)
    _bad_outcome_embeddings: list[list[float]] = []
    _max_bad_outcomes: int = 50

    @classmethod
    def record_bad_outcome(cls, embedding: list[float]) -> None:
        """
        Record an embedding associated with a negative outcome.

        Called by Nova/Axon when an intent fails, or by Atune when a
        high-distress percept is processed. The Ring buffer keeps the
        most recent bad outcomes for amygdala-like threat matching.
        """
        cls._bad_outcome_embeddings.append(embedding)
        if len(cls._bad_outcome_embeddings) > cls._max_bad_outcomes:
            cls._bad_outcome_embeddings = cls._bad_outcome_embeddings[-cls._max_bad_outcomes:]

    async def score(self, percept: Percept, context: AttentionContext) -> float:
        text = self._text(percept)

        risk_patterns = detect_risk_patterns(text)

        # Semantic similarity to known risk categories
        risk_similarity = 0.0
        if context.risk_categories and percept.content.embedding:
            risk_similarity = max(
                cosine_similarity(percept.content.embedding, cat.embedding)
                for cat in context.risk_categories
            )

        urgency = detect_urgency(text)

        # Historical bad-outcome similarity — compare against cached bad outcomes
        bad_outcome = 0.0
        if self._bad_outcome_embeddings and percept.content.embedding:
            # Find max similarity to any recorded bad outcome
            bad_outcome = max(
                cosine_similarity(percept.content.embedding, bad_emb)
                for bad_emb in self._bad_outcome_embeddings
            )
            # Only count strong matches (>0.6 similarity)
            bad_outcome = max(0.0, (bad_outcome - 0.6) / 0.4) if bad_outcome > 0.6 else 0.0

        composite = (
            0.25 * risk_patterns
            + 0.25 * risk_similarity
            + 0.25 * urgency
            + 0.25 * bad_outcome
        )
        return clamp(composite, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Head 3: Identity Relevance
# ---------------------------------------------------------------------------


class IdentityHead(SalienceHead):
    """
    Detects information related to EOS's core identity or community.
    Neuroscience analog: self-referential processing (DMN).
    """

    name = "identity"
    base_weight = 0.12
    precision_sensitivity = {"coherence_stress": 0.3, "valence": -0.2}

    async def score(self, percept: Percept, context: AttentionContext) -> float:
        text = self._text(percept)

        # Similarity to core identity entities
        max_identity_sim = 0.0
        if context.core_identity_embeddings and percept.content.embedding:
            max_identity_sim = max(
                cosine_similarity(percept.content.embedding, e)
                for e in context.core_identity_embeddings
            )

        # Direct address
        direct = detect_direct_address(text)

        # Community relevance
        community_rel = 0.0
        if context.community_embedding and percept.content.embedding:
            community_rel = cosine_similarity(
                percept.content.embedding, context.community_embedding
            )

        # Name mention
        name_mention = 0.0
        if context.instance_name and context.instance_name.lower() in text.lower():
            name_mention = 1.0

        composite = (
            0.25 * max_identity_sim
            + 0.30 * direct
            + 0.25 * community_rel
            + 0.20 * name_mention
        )
        return clamp(composite, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Head 4: Goal Relevance
# ---------------------------------------------------------------------------


class GoalHead(SalienceHead):
    """
    Detects information relevant to active goals or pending decisions.
    Neuroscience analog: prefrontal goal maintenance (dlPFC).
    """

    name = "goal"
    base_weight = 0.15
    precision_sensitivity = {"arousal": 0.2, "dominance": 0.3}

    async def score(self, percept: Percept, context: AttentionContext) -> float:
        if not context.active_goals:
            return 0.0

        goal_scores = (
            [
                cosine_similarity(percept.content.embedding, g.target_embedding) * g.priority
                for g in context.active_goals
            ]
            if percept.content.embedding
            else []
        )
        max_goal = max(goal_scores) if goal_scores else 0.0

        # Bonus: resolves a pending decision?
        resolves = 0.0
        if context.pending_decisions and percept.content.embedding:
            for dec in context.pending_decisions:
                if dec.embedding:
                    sim = cosine_similarity(percept.content.embedding, dec.embedding)
                    if sim > 0.7:
                        resolves = max(resolves, sim)

        return clamp(max_goal + resolves * 0.3, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Head 5: Emotional / Affective
# ---------------------------------------------------------------------------


class EmotionalHead(SalienceHead):
    """
    Detects emotionally charged content — distress, joy, conflict, gratitude.
    Neuroscience analog: limbic system, ACC.
    """

    name = "emotional"
    base_weight = 0.15
    precision_sensitivity = {"care_activation": 0.5, "valence": 0.2}

    async def score(self, percept: Percept, context: AttentionContext) -> float:
        text = self._text(percept)

        sentiment = await analyse_sentiment(text)
        emotional_intensity = abs(sentiment.valence) * max(sentiment.arousal, 0.1)

        distress_signals = detect_distress(text)
        conflict_signals = detect_conflict(text)
        positive_signals = detect_positive_emotion(text)

        care_boost = context.affect_state.care_activation * 0.2

        composite = (
            0.25 * emotional_intensity
            + 0.30 * distress_signals
            + 0.20 * conflict_signals
            + 0.15 * positive_signals
            + 0.10 * care_boost
        )
        return clamp(composite, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Head 6: Causal / Consequential
# ---------------------------------------------------------------------------


class CausalHead(SalienceHead):
    """
    Detects cause-effect relationships and significant consequences.
    Neuroscience analog: temporal-parietal junction.
    """

    name = "causal"
    base_weight = 0.10
    precision_sensitivity = {"coherence_stress": 0.4}

    async def score(self, percept: Percept, context: AttentionContext) -> float:
        text = self._text(percept)

        causal_lang = detect_causal_language(text)
        consequence = estimate_consequence_scope(text, context.community_size)
        temporal_prox = estimate_temporal_proximity(text)

        composite = (
            0.30 * causal_lang
            + 0.40 * consequence
            + 0.30 * temporal_prox
        )
        return clamp(composite, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Head 7: Keyword / Pattern
# ---------------------------------------------------------------------------


class KeywordHead(SalienceHead):
    """
    Detects specific keywords, learned patterns, and alert signatures.
    Neuroscience analog: early cortex feature detection.
    """

    name = "keyword"
    base_weight = 0.10
    precision_sensitivity: dict[str, float] = {}  # Minimally affected by affect

    async def score(self, percept: Percept, context: AttentionContext) -> float:
        text = self._text(percept)

        # Learned patterns
        keyword_matches = 0.0
        if context.learned_patterns:
            keyword_set = {p.pattern for p in context.learned_patterns}
            keyword_matches = match_keyword_set(text, keyword_set)

        # Community vocabulary
        community_terms = match_keyword_set(text, context.community_vocabulary)

        # Active alerts
        alert_matches = 0.0
        if context.active_alerts:
            alert_set = {a.pattern for a in context.active_alerts}
            alert_matches = match_keyword_set(text, alert_set)

        composite = (
            0.30 * keyword_matches
            + 0.30 * community_terms
            + 0.40 * alert_matches
        )
        return clamp(composite, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Head registry
# ---------------------------------------------------------------------------

ALL_HEADS: list[SalienceHead] = [
    NoveltyHead(),
    RiskHead(),
    IdentityHead(),
    GoalHead(),
    EmotionalHead(),
    CausalHead(),
    KeywordHead(),
]


# ---------------------------------------------------------------------------
# Precision weighting
# ---------------------------------------------------------------------------


def compute_precision(head: SalienceHead, affect: AffectState) -> float:
    """
    Compute precision (gain) for *head* given current *affect*.

    Precision IS attention in predictive processing terms.
    High precision = more gain on this channel = paying more attention.
    """
    precision = 1.0
    for affect_dim, sensitivity in head.precision_sensitivity.items():
        affect_value = getattr(affect, affect_dim, 0.0)
        precision += sensitivity * affect_value
    return clamp(precision, 0.3, 2.0)


# ---------------------------------------------------------------------------
# Composite salience
# ---------------------------------------------------------------------------


async def compute_salience(
    percept: Percept,
    context: AttentionContext,
    affect: AffectState,
    heads: list[SalienceHead] | None = None,
    head_weights: dict[str, float] | None = None,
) -> SalienceVector:
    """
    Run all seven heads in parallel, apply precision weighting, then
    combine into a composite score.

    When ``head_weights`` is provided (from MetaAttentionController), those
    dynamic weights override each head's static ``base_weight``. This allows
    the meta-attention system to shift focus — e.g. boosting the GoalHead
    during active goal pursuit or the RiskHead during high coherence stress.
    """
    heads = heads or ALL_HEADS

    # Run all heads concurrently
    raw_scores = await asyncio.gather(
        *(head.score(percept, context) for head in heads)
    )

    # Precision-weight each score
    precision_weighted: dict[str, float] = {}
    for head, raw in zip(heads, raw_scores, strict=False):
        p = compute_precision(head, affect)
        precision_weighted[head.name] = raw * p

    # Use dynamic meta-attention weights when available, else static base_weight
    weights = {
        h.name: head_weights.get(h.name, h.base_weight) if head_weights else h.base_weight
        for h in heads
    }
    total_weight = sum(weights.values())
    if total_weight == 0.0:
        total_weight = 1.0

    composite = sum(
        weights[h.name] * precision_weighted[h.name] for h in heads
    ) / total_weight

    return SalienceVector(
        scores=precision_weighted,
        composite=clamp(composite, 0.0, 1.0),
        prediction_error=context.prediction_error,
    )
