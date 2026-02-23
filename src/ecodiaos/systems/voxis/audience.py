"""
EcodiaOS — Voxis Audience Profiler

Builds an AudienceProfile for each expression from Memory retrieval,
and adapts the StrategyParams to match the audience.

The audience is always a person (or people), never a 'user'. The profiler
builds real context from memory — relationship history, communication
preferences, technical level, emotional state — and uses it to shape
how the organism speaks.
"""

from __future__ import annotations

import structlog

from ecodiaos.systems.voxis.types import (
    AffectEstimate,
    AudienceProfile,
    StrategyParams,
)

logger = structlog.get_logger()


class AudienceProfiler:
    """
    Builds and applies audience profiles.

    In the current phase, profiles are built from whatever context is
    available (conversation state + any memory facts passed in).
    When Nova is wired in, it will supply richer belief-state context.
    """

    def __init__(self) -> None:
        self._logger = logger.bind(system="voxis.audience")

    def build_profile(
        self,
        addressee_id: str | None,
        addressee_name: str | None,
        interaction_count: int,
        memory_facts: list[dict],  # Simplified entity/relation data from Memory
        audience_type: str = "individual",
        group_size: int | None = None,
        group_context: str | None = None,
    ) -> AudienceProfile:
        """
        Build an AudienceProfile from available context.

        memory_facts is a list of entity/relation dicts from Memory retrieval.
        We extract: technical_level, preferred_register, communication_preferences,
        and emotional_state_estimate from these facts if present.
        """
        tech_level = 0.5  # Default: unknown
        preferred_register = "neutral"
        comm_prefs: dict = {}
        affect_est = AffectEstimate()
        relationship_strength = self._estimate_relationship_strength(interaction_count)
        language = "en"

        for fact in memory_facts:
            ftype = fact.get("type", "")
            value = fact.get("value")

            if ftype == "technical_level" and isinstance(value, (int, float)):
                tech_level = float(max(0.0, min(1.0, value)))
            elif ftype == "preferred_register" and isinstance(value, str):
                preferred_register = value
            elif ftype == "prefers_bullet_points" and value:
                comm_prefs["prefers_bullet_points"] = True
            elif ftype == "prefers_brief" and value:
                comm_prefs["prefers_brief"] = True
            elif ftype == "language" and isinstance(value, str):
                language = value
            elif ftype == "emotional_distress" and isinstance(value, (int, float)):
                affect_est = affect_est.model_copy(update={"distress": float(value)})
            elif ftype == "emotional_frustration" and isinstance(value, (int, float)):
                affect_est = affect_est.model_copy(update={"frustration": float(value)})

        profile = AudienceProfile(
            audience_type=audience_type,
            individual_id=addressee_id,
            name=addressee_name,
            interaction_count=interaction_count,
            preferred_register=preferred_register,
            technical_level=tech_level,
            emotional_state_estimate=affect_est,
            communication_preferences=comm_prefs,
            relationship_strength=relationship_strength,
            group_size=group_size,
            group_context=group_context,
            language=language,
        )

        self._logger.debug(
            "audience_profile_built",
            audience_type=audience_type,
            interaction_count=interaction_count,
            relationship_strength=round(relationship_strength, 3),
            technical_level=round(tech_level, 3),
        )

        return profile

    def adapt(self, strategy: StrategyParams, audience: AudienceProfile) -> StrategyParams:
        """
        Return a new StrategyParams adapted to the audience profile.

        Applied after personality and affect colouring. Audience adaptation
        can override certain strategy parameters when the audience context
        is strong enough (e.g., distress overrides information density).
        """
        s = strategy.model_copy(deep=True)

        # ── Technical level ───────────────────────────────────────
        if audience.technical_level < 0.25:
            s.jargon_level = "none"
            s.explanation_depth = "thorough"
            s.analogy_encouraged = True
            s.assume_knowledge = False
        elif audience.technical_level > 0.75:
            s.jargon_level = "domain_appropriate"
            s.explanation_depth = "concise"
            s.assume_knowledge = True

        # ── Relationship depth ────────────────────────────────────
        if audience.relationship_strength > 0.7:
            s.formality_override = "relaxed"
            s.reference_shared_history = True
        elif audience.relationship_strength < 0.15 and audience.interaction_count == 0:
            s.introduce_self_if_first = True
            s.formality_override = "polite"
        elif audience.relationship_strength < 0.15:
            s.formality_override = "polite"

        # ── Emotional state ───────────────────────────────────────
        est = audience.emotional_state_estimate
        if est.distress > 0.5:
            s.empathy_first = True
            s.information_density = "low"
            s.emotional_acknowledgment = "explicit"
        elif est.distress > 0.3:
            s.emotional_acknowledgment = "explicit"

        if est.frustration > 0.5:
            s.directness_override = "high"
            s.target_length = max(50, int(s.target_length * 0.7))
        elif est.frustration > 0.3:
            s.target_length = max(50, int(s.target_length * 0.85))

        if est.curiosity > 0.6:
            # Engaged and curious — match that energy
            s.exploratory_tangents_allowed = True

        # ── Communication preferences ─────────────────────────────
        prefs = audience.communication_preferences
        if prefs.get("prefers_bullet_points"):
            s.formatting = "structured"
        if prefs.get("prefers_brief"):
            s.target_length = min(s.target_length, 200)

        # ── Group adaptation ──────────────────────────────────────
        if audience.audience_type == "group":
            s.address_style = "collective"
            if s.formality_override is None:
                s.formality_override = "professional"
            s.avoid_singling_out = True
        elif audience.audience_type == "community":
            s.address_style = "collective"
            s.avoid_singling_out = True

        # ── Language ──────────────────────────────────────────────
        s.language = audience.language

        self._logger.debug(
            "audience_adapted",
            audience_type=audience.audience_type,
            relationship=round(audience.relationship_strength, 3),
            distress=round(est.distress, 3),
            target_length=s.target_length,
        )

        return s

    @staticmethod
    def _estimate_relationship_strength(interaction_count: int) -> float:
        """
        Estimate relationship strength from number of past interactions.

        Uses a diminishing-returns curve: strength grows quickly at first,
        then plateaus. 0 interactions = stranger (0.0), 100+ = established (0.8+).
        """
        if interaction_count <= 0:
            return 0.0
        # Sigmoid-like growth: fast early, plateaus around 0.85
        import math
        return min(0.9, 0.85 * (1.0 - math.exp(-interaction_count / 30.0)))
