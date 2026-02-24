"""
EcodiaOS — Evo Hypothesis Engine

Manages the full lifecycle of hypotheses:
  1. Generation  — LLM produces testable claims from detected patterns
  2. Testing     — each new episode is evaluated as evidence
  3. Integration — supported hypotheses have their mutations applied
  4. Archival    — refuted or stale hypotheses are stored and closed

Implements approximate Bayesian model comparison (spec Section IV.2):
  Evidence(H) = Σ log p(observation_i | H) - complexity(H)

Approximated as:
  evidence_score += strength × (1 - complexity_penalty × 0.1)  [for support]
  evidence_score -= strength                                     [for contradiction]

Integration requires (spec Section IX, VELOCITY_LIMITS):
  - evidence_score > 3.0
  - len(supporting_episodes) >= 10
  - hypothesis age >= 24 hours

Performance budget: evidence_evaluate ≤200ms per hypothesis (spec Section X).
"""

from __future__ import annotations

import json
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.clients.llm import LLMProvider, Message
from ecodiaos.clients.optimized_llm import OptimizedLLMProvider
from ecodiaos.primitives.common import new_id, utc_now
from ecodiaos.primitives.memory_trace import Episode
from ecodiaos.systems.evo.types import (
    EvidenceDirection,
    EvidenceResult,
    Hypothesis,
    HypothesisCategory,
    HypothesisStatus,
    Mutation,
    MutationType,
    PatternCandidate,
    VELOCITY_LIMITS,
)

if TYPE_CHECKING:
    from ecodiaos.systems.memory.service import MemoryService

logger = structlog.get_logger()

# Evidence thresholds (from VELOCITY_LIMITS)
_SUPPORT_SCORE_THRESHOLD: float = 3.0
_SUPPORT_EPISODE_THRESHOLD: int = VELOCITY_LIMITS["min_evidence_for_integration"]
_MIN_AGE_HOURS: int = VELOCITY_LIMITS["min_hypothesis_age_hours"]
_MAX_ACTIVE: int = VELOCITY_LIMITS["max_active_hypotheses"]

# LLM generation limits
_MAX_PER_BATCH: int = 3
_SYSTEM_PROMPT = (
    "You are the learning subsystem of a living digital organism. "
    "Your role is to generate precise, falsifiable hypotheses from observed patterns "
    "and evaluate evidence rigorously. Prefer simple explanations. "
    "Always respond with valid JSON matching the requested schema."
)


class HypothesisEngine:
    """
    Manages hypothesis generation, evidence accumulation, and lifecycle.

    Dependencies:
      llm     — LLM provider for generation and evaluation
      memory  — optional; used to persist hypotheses as :Hypothesis nodes
    """

    def __init__(
        self,
        llm: LLMProvider,
        instance_name: str = "EOS",
        memory: MemoryService | None = None,
    ) -> None:
        self._llm = llm
        self._instance_name = instance_name
        self._memory = memory
        self._logger = logger.bind(system="evo.hypothesis")
        self._optimized = isinstance(llm, OptimizedLLMProvider)

        # In-memory hypothesis registry (also persisted to Memory graph)
        self._active: dict[str, Hypothesis] = {}

        # Metrics
        self._total_proposed: int = 0
        self._total_supported: int = 0
        self._total_refuted: int = 0
        self._total_integrated: int = 0

    # ─── Generation ───────────────────────────────────────────────────────────

    async def generate_hypotheses(
        self,
        patterns: list[PatternCandidate],
        existing_summaries: list[str] | None = None,
    ) -> list[Hypothesis]:
        """
        Generate new hypotheses from a batch of pattern candidates.
        Uses LLM reasoning grounded in the pattern evidence.
        Respects MAX_ACTIVE_HYPOTHESES by skipping if at capacity.

        Returns up to _MAX_PER_BATCH new hypotheses.
        """
        if not patterns:
            return []

        if len(self._active) >= _MAX_ACTIVE:
            self._logger.warning(
                "hypothesis_capacity_reached",
                active=len(self._active),
                max=_MAX_ACTIVE,
            )
            return []

        prompt = _build_generation_prompt(
            instance_name=self._instance_name,
            patterns=patterns,
            existing_hypotheses=existing_summaries or list(self._active_summaries()),
            max_hypotheses=_MAX_PER_BATCH,
        )

        # Budget check: skip hypothesis generation in YELLOW/RED (low priority)
        if self._optimized:
            assert isinstance(self._llm, OptimizedLLMProvider)
            if not self._llm.should_use_llm("evo.hypothesis", estimated_tokens=1200):
                self._logger.info("hypothesis_generation_skipped_budget")
                return []

        try:
            if self._optimized:
                response = await self._llm.generate(  # type: ignore[call-arg]
                    system_prompt=_SYSTEM_PROMPT,
                    messages=[Message("user", prompt)],
                    max_tokens=1200,
                    temperature=0.5,
                    output_format="json",
                    cache_system="evo.hypothesis",
                    cache_method="generate",
                )
            else:
                response = await self._llm.generate(
                    system_prompt=_SYSTEM_PROMPT,
                    messages=[Message("user", prompt)],
                    max_tokens=1200,
                    temperature=0.5,
                    output_format="json",
                )
            raw = _parse_json_safe(response.text)
        except Exception as exc:
            self._logger.error("hypothesis_generation_failed", error=str(exc))
            return []

        hypotheses: list[Hypothesis] = []
        for item in raw.get("hypotheses", [])[:_MAX_PER_BATCH]:
            try:
                h = _build_hypothesis(item)
                self._active[h.id] = h
                self._total_proposed += 1
                hypotheses.append(h)
                self._logger.info(
                    "hypothesis_proposed",
                    hypothesis_id=h.id,
                    category=h.category.value,
                    statement=h.statement[:80],
                )
            except (KeyError, ValueError) as exc:
                self._logger.warning("hypothesis_parse_failed", error=str(exc))
                continue

        # Persist to Memory if available
        if self._memory is not None:
            for h in hypotheses:
                await self._persist_hypothesis(h)

        return hypotheses

    # ─── Evidence Evaluation ──────────────────────────────────────────────────

    async def evaluate_evidence(
        self,
        hypothesis: Hypothesis,
        episode: Episode,
    ) -> EvidenceResult:
        """
        Evaluate whether this episode provides evidence for or against a hypothesis.
        LLM evaluates with temperature=0.3 for consistency.
        Updates hypothesis in-place and returns the result.
        Budget: ≤200ms.
        """
        prompt = _build_evidence_prompt(hypothesis, episode)

        # Budget check: skip evidence evaluation in YELLOW/RED
        if self._optimized:
            assert isinstance(self._llm, OptimizedLLMProvider)
            if not self._llm.should_use_llm("evo.evidence", estimated_tokens=300):
                return EvidenceResult(
                    hypothesis_id=hypothesis.id,
                    episode_id=episode.id,
                    direction=EvidenceDirection.NEUTRAL,
                    strength=0.0,
                    reasoning="Skipped — budget constraints",
                )

        try:
            if self._optimized:
                response = await self._llm.evaluate(  # type: ignore[call-arg]
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0.3,
                    cache_system="evo.evidence",
                    cache_method="evaluate",
                )
            else:
                response = await self._llm.evaluate(
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0.3,
                )
            raw = _parse_json_safe(response.text)
        except Exception as exc:
            self._logger.error(
                "evidence_evaluation_failed",
                hypothesis_id=hypothesis.id,
                error=str(exc),
            )
            return EvidenceResult(
                hypothesis_id=hypothesis.id,
                episode_id=episode.id,
                direction=EvidenceDirection.NEUTRAL,
                strength=0.0,
                reasoning="evaluation failed",
                new_score=hypothesis.evidence_score,
                new_status=hypothesis.status,
            )

        direction_raw = raw.get("direction", "neutral")
        strength = float(raw.get("strength", 0.0))
        strength = max(0.0, min(1.0, strength))
        reasoning = str(raw.get("reasoning", ""))

        try:
            direction = EvidenceDirection(direction_raw)
        except ValueError:
            direction = EvidenceDirection.NEUTRAL

        # Update hypothesis evidence score (Bayesian accumulation with Occam penalty)
        if direction == EvidenceDirection.SUPPORTS:
            hypothesis.supporting_episodes.append(episode.id)
            hypothesis.evidence_score += strength * (
                1.0 - hypothesis.complexity_penalty * 0.1
            )
        elif direction == EvidenceDirection.CONTRADICTS:
            hypothesis.contradicting_episodes.append(episode.id)
            hypothesis.evidence_score -= strength

        hypothesis.last_evidence_at = utc_now()

        # Status transitions
        if hypothesis.status in (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING):
            hypothesis.status = HypothesisStatus.TESTING
            if (
                hypothesis.evidence_score > _SUPPORT_SCORE_THRESHOLD
                and len(hypothesis.supporting_episodes) >= _SUPPORT_EPISODE_THRESHOLD
            ):
                hypothesis.status = HypothesisStatus.SUPPORTED
                self._total_supported += 1
                self._logger.info(
                    "hypothesis_supported",
                    hypothesis_id=hypothesis.id,
                    evidence_score=round(hypothesis.evidence_score, 2),
                    supporting_count=len(hypothesis.supporting_episodes),
                )
            elif hypothesis.evidence_score < -2.0:
                hypothesis.status = HypothesisStatus.REFUTED
                self._total_refuted += 1
                self._logger.info(
                    "hypothesis_refuted",
                    hypothesis_id=hypothesis.id,
                    evidence_score=round(hypothesis.evidence_score, 2),
                )

        return EvidenceResult(
            hypothesis_id=hypothesis.id,
            episode_id=episode.id,
            direction=direction,
            strength=strength,
            reasoning=reasoning,
            new_score=hypothesis.evidence_score,
            new_status=hypothesis.status,
        )

    # ─── Integration ──────────────────────────────────────────────────────────

    async def integrate_hypothesis(self, hypothesis: Hypothesis) -> bool:
        """
        Mark a supported hypothesis as INTEGRATED.
        The caller (ConsolidationOrchestrator) is responsible for applying
        the proposed_mutation — this method only closes the hypothesis lifecycle.

        Returns True if integration is valid and was applied.
        """
        if hypothesis.status != HypothesisStatus.SUPPORTED:
            return False

        # Age check — must have been active for at least 24 hours
        age_hours = (utc_now() - hypothesis.created_at).total_seconds() / 3600
        if age_hours < _MIN_AGE_HOURS:
            self._logger.info(
                "hypothesis_integration_deferred",
                hypothesis_id=hypothesis.id,
                age_hours=round(age_hours, 1),
                required_hours=_MIN_AGE_HOURS,
            )
            return False

        hypothesis.status = HypothesisStatus.INTEGRATED
        self._total_integrated += 1

        # Remove from active registry
        self._active.pop(hypothesis.id, None)

        self._logger.info(
            "hypothesis_integrated",
            hypothesis_id=hypothesis.id,
            category=hypothesis.category.value,
            evidence_score=round(hypothesis.evidence_score, 2),
            mutation_type=(
                hypothesis.proposed_mutation.type.value
                if hypothesis.proposed_mutation else "none"
            ),
        )

        if self._memory is not None:
            await self._persist_hypothesis(hypothesis)

        return True

    async def archive_hypothesis(
        self,
        hypothesis: Hypothesis,
        reason: str = "",
    ) -> None:
        """Mark a hypothesis as ARCHIVED and remove from active registry."""
        hypothesis.status = HypothesisStatus.ARCHIVED
        self._active.pop(hypothesis.id, None)

        self._logger.info(
            "hypothesis_archived",
            hypothesis_id=hypothesis.id,
            reason=reason or "not specified",
            evidence_score=round(hypothesis.evidence_score, 2),
        )

        if self._memory is not None:
            await self._persist_hypothesis(hypothesis)

    def is_stale(self, hypothesis: Hypothesis, max_age_days: int = 7) -> bool:
        """
        Return True if this hypothesis has not received evidence in max_age_days.
        Stale hypotheses are archived during consolidation.
        """
        if hypothesis.status not in (
            HypothesisStatus.PROPOSED, HypothesisStatus.TESTING
        ):
            return False
        age = utc_now() - hypothesis.last_evidence_at
        return age > timedelta(days=max_age_days)

    # ─── Query Interface ──────────────────────────────────────────────────────

    def get_active(self) -> list[Hypothesis]:
        """Return all currently active hypotheses (proposed or testing)."""
        return [
            h for h in self._active.values()
            if h.status in (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING)
        ]

    def get_supported(self) -> list[Hypothesis]:
        """Return all supported hypotheses ready for integration."""
        return [
            h for h in self._active.values()
            if h.status == HypothesisStatus.SUPPORTED
        ]

    def get_all_active(self) -> list[Hypothesis]:
        """Return all hypotheses still in the registry (not yet archived/integrated)."""
        return list(self._active.values())

    @property
    def stats(self) -> dict[str, int]:
        return {
            "active": len(self._active),
            "proposed": self._total_proposed,
            "supported": self._total_supported,
            "refuted": self._total_refuted,
            "integrated": self._total_integrated,
        }

    # ─── Private ──────────────────────────────────────────────────────────────

    def _active_summaries(self) -> list[str]:
        """Return short statements of all active hypotheses (deduplication prompt)."""
        return [h.statement[:100] for h in self._active.values()]

    async def _persist_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Store hypothesis as a governance record in Memory."""
        if self._memory is None:
            return
        try:
            await self._memory._neo4j.execute_write(
                """
                MERGE (h:Hypothesis {hypothesis_id: $hypothesis_id})
                SET h.type = $type,
                    h.category = $category,
                    h.statement = $statement,
                    h.status = $status,
                    h.evidence_score = $evidence_score,
                    h.supporting_count = $supporting_count,
                    h.contradicting_count = $contradicting_count,
                    h.created_at = $created_at
                """,
                {
                    "hypothesis_id": hypothesis.id,
                    "type": "hypothesis",
                    "category": hypothesis.category.value,
                    "statement": hypothesis.statement,
                    "status": hypothesis.status.value,
                    "evidence_score": hypothesis.evidence_score,
                    "supporting_count": len(hypothesis.supporting_episodes),
                    "contradicting_count": len(hypothesis.contradicting_episodes),
                    "created_at": hypothesis.created_at.isoformat(),
                },
            )
        except Exception as exc:
            self._logger.warning(
                "hypothesis_persist_failed",
                hypothesis_id=hypothesis.id,
                error=str(exc),
            )


# ─── Prompt Builders ──────────────────────────────────────────────────────────


def _build_generation_prompt(
    instance_name: str,
    patterns: list[PatternCandidate],
    existing_hypotheses: list[str],
    max_hypotheses: int,
) -> str:
    pattern_lines = "\n".join(
        f"- [{p.type.value}] {', '.join(p.elements[:4])} "
        f"(count={p.count}, confidence={p.confidence:.2f})"
        for p in patterns[:10]
    )
    existing_block = (
        "\n".join(f"- {s}" for s in existing_hypotheses[:10])
        if existing_hypotheses
        else "(none)"
    )
    return f"""You are the learning system of {instance_name}, a living digital organism.

DETECTED PATTERNS:
{pattern_lines}

CURRENT ACTIVE HYPOTHESES (avoid duplicates):
{existing_block}

Generate up to {max_hypotheses} hypotheses that explain the patterns above.

Rules:
- Each hypothesis must be FALSIFIABLE — state exactly how it could be proven false
- Prefer SIMPLER explanations (Occam's razor) — penalise unnecessary complexity
- Do NOT duplicate existing hypotheses
- Max {max_hypotheses} hypotheses per batch

For each hypothesis respond in this exact JSON schema:
{{
  "hypotheses": [
    {{
      "category": "world_model|self_model|social|procedural|parameter",
      "statement": "A clear, testable claim (1-2 sentences)",
      "formal_test": "Specific observable condition that would confirm or refute this",
      "complexity": "low|medium|high",
      "proposed_mutation": {{
        "type": "parameter_adjustment|procedure_creation|schema_addition|evolution_proposal",
        "target": "parameter name, procedure name, or entity type",
        "value": 0.0,
        "description": "What specifically to change if confirmed"
      }}
    }}
  ]
}}

If no compelling hypotheses arise from these patterns, return {{"hypotheses": []}}."""


def _build_evidence_prompt(hypothesis: Hypothesis, episode: Episode) -> str:
    return f"""HYPOTHESIS: {hypothesis.statement}
FORMAL TEST: {hypothesis.formal_test}

EVIDENCE (episode):
Content: {episode.raw_content[:300] or episode.summary[:300]}
Source: {episode.source}
Time: {episode.event_time.isoformat()}
Affect: valence={episode.affect_valence:.2f}, arousal={episode.affect_arousal:.2f}
Salience: {episode.salience_composite:.2f}

Does this episode provide evidence FOR or AGAINST the hypothesis?

Respond in JSON:
{{
  "direction": "supports|contradicts|neutral",
  "strength": 0.0,
  "reasoning": "1-2 sentence explanation"
}}

Where strength (0.0–1.0) represents how strongly this evidence bears on the hypothesis."""


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _build_hypothesis(item: dict[str, Any]) -> Hypothesis:
    """Parse one hypothesis item from LLM JSON response."""
    category_raw = str(item.get("category", "world_model"))
    try:
        category = HypothesisCategory(category_raw)
    except ValueError:
        category = HypothesisCategory.WORLD_MODEL

    complexity_raw = str(item.get("complexity", "low"))
    complexity_map = {"low": 0.05, "medium": 0.15, "high": 0.30}
    complexity_penalty = complexity_map.get(complexity_raw, 0.10)

    mutation: Mutation | None = None
    mutation_data = item.get("proposed_mutation")
    if mutation_data and isinstance(mutation_data, dict):
        try:
            mutation = Mutation(
                type=MutationType(mutation_data.get("type", "parameter_adjustment")),
                target=str(mutation_data.get("target", "")),
                value=float(mutation_data.get("value", 0.0)),
                description=str(mutation_data.get("description", "")),
            )
        except (ValueError, KeyError):
            mutation = None

    return Hypothesis(
        category=category,
        statement=str(item["statement"]),
        formal_test=str(item["formal_test"]),
        complexity_penalty=complexity_penalty,
        proposed_mutation=mutation,
    )


def _parse_json_safe(text: str) -> dict[str, Any]:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}
