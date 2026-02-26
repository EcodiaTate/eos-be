"""
EcodiaOS — Equor Drive Evaluators

Four parallel evaluators — one per constitutional drive.
Each scores alignment from -1.0 (strongly violates) to +1.0 (strongly promotes).

In Phase 2, these use heuristic analysis of the Intent structure.
In later phases (when Nova and the full active inference engine are online),
they will also incorporate LLM-based contextual reasoning.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from ecodiaos.primitives.common import DriveAlignmentVector

if TYPE_CHECKING:
    from ecodiaos.primitives.intent import Intent

logger = structlog.get_logger()


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ─── Coherence Evaluator ─────────────────────────────────────────


async def evaluate_coherence(intent: Intent) -> float:
    """
    "Does this action make the world more understandable and
     internally consistent for EOS?"

    Checks:
    - Is reasoning provided and non-empty?
    - Were alternatives considered?
    - Does the action have a clear goal with success criteria?
    """
    score = 0.0

    # Reasoning chain quality
    reasoning = intent.decision_trace.reasoning
    if reasoning and len(reasoning) > 20:
        score += 0.25  # Has substantive reasoning
    elif reasoning:
        score += 0.1   # Has some reasoning
    else:
        score -= 0.15  # No reasoning at all — incoherent

    # Alternatives considered — sign of deliberation
    alternatives = intent.decision_trace.alternatives_considered
    if alternatives and len(alternatives) >= 2:
        score += 0.2
    elif alternatives:
        score += 0.1

    # Goal clarity
    if intent.goal.description and len(intent.goal.description) > 10:
        score += 0.15
    if intent.goal.success_criteria:
        score += 0.1

    # Plan completeness
    if intent.plan.steps:
        score += 0.1
        if intent.plan.contingencies:
            score += 0.1  # Has contingency planning

    # Expected free energy — if computed, lower is more coherent
    if intent.expected_free_energy < 0:
        score += min(0.2, abs(intent.expected_free_energy) * 0.1)
    elif intent.expected_free_energy > 0.5:
        score -= min(0.15, intent.expected_free_energy * 0.1)

    return _clamp(score)


# ─── Care Evaluator ──────────────────────────────────────────────


async def evaluate_care(intent: Intent) -> float:
    """
    "Does this action promote the wellbeing of the people
     and systems EOS stewards?"

    This is the most nuanced evaluator. In Phase 2, it uses
    heuristic analysis. Later phases add LLM-based stakeholder
    and harm assessment.
    """
    score = 0.0
    goal_lower = intent.goal.description.lower()

    # Positive care indicators
    care_positive = [
        "help", "support", "assist", "protect", "wellbeing", "care for",
        "benefit", "improve", "nurture", "comfort", "inform", "guide",
        "empower", "include", "welcome", "share knowledge",
    ]
    for indicator in care_positive:
        if indicator in goal_lower:
            score += 0.15
            break  # One match is enough to establish positive orientation

    # Harm indicators (weighted 2x per spec — "first, do no harm")
    harm_indicators = [
        "ignore wellbeing", "disregard safety", "override consent",
        "exclude", "punish", "withhold help", "abandon",
        "dismiss concern", "silence", "suppress",
    ]
    for indicator in harm_indicators:
        if indicator in goal_lower:
            score -= 0.30
            break

    # Action type assessment
    for step in intent.plan.steps:
        executor = step.executor.lower()
        # Communication is generally care-positive
        if "communicate" in executor or "notify" in executor:
            score += 0.1
        # Observation is neutral-to-positive
        elif "observe" in executor or "analyse" in executor:
            score += 0.05
        # Resource actions need more scrutiny but aren't inherently harmful
        elif "resource" in executor:
            pass  # Neutral

    # Consent awareness — does the intent mention consent?
    all_text = goal_lower + " " + intent.decision_trace.reasoning.lower()
    if "consent" in all_text or "permission" in all_text or "approval" in all_text:
        score += 0.1

    # Equality — does it specify treating people differently?
    if "only for" in goal_lower or "exclude" in goal_lower or "except" in goal_lower:
        score -= 0.1

    return _clamp(score)


# ─── Growth Evaluator ────────────────────────────────────────────


async def evaluate_growth(intent: Intent) -> float:
    """
    "Does this action make EOS or its community more capable,
     aware, or mature?"
    """
    score = 0.0
    goal_lower = intent.goal.description.lower()

    # Growth-positive indicators
    growth_positive = [
        "learn", "discover", "improve", "expand", "develop",
        "explore", "experiment", "understand", "investigate",
        "adapt", "evolve", "create", "innovate", "teach",
    ]
    for indicator in growth_positive:
        if indicator in goal_lower:
            score += 0.2
            break

    # Stagnation indicators
    stagnation_indicators = [
        "avoid", "refuse to try", "stay the same", "no change",
        "repeat exactly", "do nothing",
    ]
    for indicator in stagnation_indicators:
        if indicator in goal_lower:
            score -= 0.15
            break

    # Novelty — is this intent doing something new?
    if intent.decision_trace.alternatives_considered:
        score += 0.1  # Considered options = growth-oriented thinking

    # Epistemic value — does this reduce uncertainty?
    if "uncertain" in goal_lower or "investigate" in goal_lower or "verify" in goal_lower:
        score += 0.15

    # Risk calibration — extremely high or zero risk are both anti-growth
    # (Growth requires appropriate challenge)
    # For now, slight positive for having a plan at all
    if intent.plan.steps:
        score += 0.05

    return _clamp(score)


# ─── Honesty Evaluator ───────────────────────────────────────────


async def evaluate_honesty(intent: Intent) -> float:
    """
    "Is this action truthful, transparent, and authentic?"
    """
    score = 0.0
    goal_lower = intent.goal.description.lower()

    # Deception indicators (heavily penalised per spec)
    deception_indicators = [
        "mislead", "deceive", "hide the truth", "pretend",
        "misrepresent", "conceal", "cover up", "fabricate",
        "omit important", "manipulate perception",
    ]
    for indicator in deception_indicators:
        if indicator in goal_lower:
            score -= 0.5
            break

    # Transparency indicators
    transparency_positive = [
        "transparent", "explain", "disclose", "honest", "truthful",
        "acknowledge", "admit", "clarify", "correct the record",
        "share openly",
    ]
    for indicator in transparency_positive:
        if indicator in goal_lower:
            score += 0.2
            break

    # Explainability — is there a decision trace?
    if intent.decision_trace.reasoning:
        score += 0.15
    else:
        score -= 0.1  # No reasoning = opaque decision

    # Uncertainty calibration — does the intent express appropriate confidence?
    reasoning_lower = intent.decision_trace.reasoning.lower()
    if "uncertain" in reasoning_lower or "not sure" in reasoning_lower:
        score += 0.1  # Acknowledging uncertainty is honest
    if "definitely" in reasoning_lower or "absolutely certain" in reasoning_lower:
        score -= 0.05  # Overconfidence is mildly dishonest

    # Check plan content for output honesty
    for step in intent.plan.steps:
        content = str(step.parameters.get("content", "")).lower()
        if "i am certain" in content and "uncertain" in reasoning_lower:
            score -= 0.2  # Expressing certainty when reasoning is uncertain

    return _clamp(score)


# ─── Parallel Evaluation ─────────────────────────────────────────


async def evaluate_all_drives(intent: Intent) -> DriveAlignmentVector:
    """
    Run all four drive evaluators in parallel.
    Returns a DriveAlignmentVector with scores from each.
    """
    coherence, care, growth, honesty = await asyncio.gather(
        evaluate_coherence(intent),
        evaluate_care(intent),
        evaluate_growth(intent),
        evaluate_honesty(intent),
    )

    alignment = DriveAlignmentVector(
        coherence=coherence,
        care=care,
        growth=growth,
        honesty=honesty,
    )

    logger.debug(
        "drive_evaluation_complete",
        coherence=f"{coherence:.2f}",
        care=f"{care:.2f}",
        growth=f"{growth:.2f}",
        honesty=f"{honesty:.2f}",
        composite=f"{alignment.composite:.2f}",
    )

    return alignment
