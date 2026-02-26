"""
EcodiaOS — Equor Verdict Engine

The 8-stage verdict pipeline that transforms drive alignment scores
into a constitutional verdict (PERMIT / MODIFY / ESCALATE / DENY).

Care and Honesty are floor drives — they cannot be traded off.
Coherence and Growth are ceiling drives — they can be temporarily deprioritised.
Denial is final. No system can override a DENY.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.primitives.common import DriveAlignmentVector, Verdict
from ecodiaos.primitives.constitutional import ConstitutionalCheck, InvariantResult
from ecodiaos.systems.equor.invariants import check_hardcoded_invariants

if TYPE_CHECKING:
    from ecodiaos.primitives.intent import Intent

logger = structlog.get_logger()

# Action categories and their required autonomy levels
ACTION_AUTONOMY_MAP: dict[str, int | str] = {
    # Level 1 (Advisor) — always permitted
    "observe": 1,
    "analyse": 1,
    "suggest": 1,
    "answer_question": 1,
    "store_memory": 1,
    "self_reflect": 1,
    # Level 2 (Partner)
    "send_notification": 2,
    "adjust_resource_allocation": 2,
    "mediate_minor_conflict": 2,
    "schedule_event": 2,
    "share_knowledge_federation": 2,
    "modify_own_non_critical_config": 2,
    # Level 3 (Steward)
    "make_resource_decision": 3,
    "mediate_major_conflict": 3,
    "initiate_federation_coordination": 3,
    "propose_policy_change": 3,
    "override_automated_system": 3,
    # Governance required
    "amend_constitution": "governance",
    "change_autonomy_level": "governance",
    "end_instance_life": "governance",
    "share_private_data": "governance",
}


def _assess_required_autonomy(intent: Intent) -> int | str:
    """
    Determine the minimum autonomy level required for an intent.
    Examines the intent's plan steps and goal to classify.
    """
    max_level: int = 1

    # Check each step's executor against the autonomy map
    for step in intent.plan.steps:
        executor_base = step.executor.split(".")[0].lower() if step.executor else ""
        for action_key, level in ACTION_AUTONOMY_MAP.items():
            if action_key in executor_base:
                if level == "governance":
                    return "governance"
                if isinstance(level, int) and level > max_level:
                    max_level = level

    # Also check the goal description for governance-level actions
    goal_lower = intent.goal.description.lower()
    for action_key, level in ACTION_AUTONOMY_MAP.items():
        if level == "governance" and action_key.replace("_", " ") in goal_lower:
            return "governance"

    return max_level


def _assess_risk(intent: Intent) -> dict[str, Any]:
    """
    Quick risk assessment of an intent.
    Returns harm_potential and reversibility scores.
    """
    goal_lower = intent.goal.description.lower()

    # Harm potential heuristics
    harm_potential = 0.1  # Baseline low
    high_harm_keywords = [
        "delete", "remove", "override", "terminate", "force",
        "broadcast to all", "share widely", "permanent",
    ]
    for keyword in high_harm_keywords:
        if keyword in goal_lower:
            harm_potential = max(harm_potential, 0.6)
            break

    medium_harm_keywords = [
        "notify", "change", "modify", "update", "send",
    ]
    for keyword in medium_harm_keywords:
        if keyword in goal_lower:
            harm_potential = max(harm_potential, 0.3)
            break

    # Reversibility heuristics
    reversibility = 0.8  # Most things are reversible
    irreversible_keywords = [
        "permanent", "delete", "destroy", "irreversible",
        "cannot undo", "broadcast", "federation share",
    ]
    for keyword in irreversible_keywords:
        if keyword in goal_lower:
            reversibility = 0.2
            break

    return {
        "harm_potential": harm_potential,
        "reversibility": reversibility,
    }


def compute_verdict(
    alignment: DriveAlignmentVector,
    intent: Intent,
    autonomy_level: int,
    constitution: dict[str, Any],
) -> ConstitutionalCheck:
    """
    The 8-stage verdict pipeline.

    constitution is a dict with drive weights:
      {drive_coherence, drive_care, drive_growth, drive_honesty}
    """
    check = ConstitutionalCheck(intent_id=intent.id, drive_alignment=alignment)

    # ── STAGE 1: Hard-Stop Invariant Check ──────────────────────
    violations = check_hardcoded_invariants(intent)
    if violations:
        critical = [v for v in violations if v.severity == "critical"]
        if critical:
            check.verdict = Verdict.BLOCKED
            check.invariant_results = [
                InvariantResult(
                    invariant_id=v.invariant_id,
                    name=v.invariant_name,
                    passed=False,
                    severity=v.severity,
                    explanation=v.explanation,
                )
                for v in violations
            ]
            check.reasoning = f"Blocked by invariant: {critical[0].invariant_name}"
            check.confidence = 1.0
            return check

        # Non-critical violations get recorded but don't auto-block
        check.invariant_results = [
            InvariantResult(
                invariant_id=v.invariant_id,
                name=v.invariant_name,
                passed=False,
                severity=v.severity,
                explanation=v.explanation,
            )
            for v in violations
        ]

    # ── STAGE 2: Floor Drive Check ──────────────────────────────
    care_weight = constitution.get("drive_care", 1.0)
    honesty_weight = constitution.get("drive_honesty", 1.0)
    care_floor = -0.3 * care_weight
    honesty_floor = -0.3 * honesty_weight

    if alignment.care < care_floor:
        check.verdict = Verdict.BLOCKED
        check.reasoning = (
            f"Care alignment ({alignment.care:.2f}) below floor ({care_floor:.2f}). "
            f"This action would cause unacceptable harm."
        )
        check.confidence = 0.95
        return check

    if alignment.honesty < honesty_floor:
        check.verdict = Verdict.BLOCKED
        check.reasoning = (
            f"Honesty alignment ({alignment.honesty:.2f}) below floor ({honesty_floor:.2f}). "
            f"This action involves unacceptable deception."
        )
        check.confidence = 0.95
        return check

    # ── STAGE 3: Autonomy Gate ──────────────────────────────────
    required_autonomy = _assess_required_autonomy(intent)

    if required_autonomy == "governance":
        check.verdict = Verdict.DEFERRED
        check.reasoning = "Action requires community governance approval."
        check.confidence = 1.0
        return check

    if isinstance(required_autonomy, int) and required_autonomy > autonomy_level:
        check.verdict = Verdict.DEFERRED
        check.reasoning = (
            f"Action requires autonomy level {required_autonomy} "
            f"but instance is at level {autonomy_level}."
        )
        check.confidence = 1.0
        return check

    # ── STAGE 4: Composite Alignment Assessment ─────────────────
    coherence_weight = constitution.get("drive_coherence", 1.0)
    growth_weight = constitution.get("drive_growth", 1.0)

    weights = {
        "coherence": coherence_weight * 0.8,
        "care": care_weight * 1.5,       # Care weighted highest
        "growth": growth_weight * 0.7,
        "honesty": honesty_weight * 1.3,  # Honesty second highest
    }
    total_weight = sum(weights.values())

    composite = (
        weights["coherence"] * alignment.coherence
        + weights["care"] * alignment.care
        + weights["growth"] * alignment.growth
        + weights["honesty"] * alignment.honesty
    ) / total_weight

    # ── STAGE 5: Risk-Adjusted Decision ─────────────────────────
    risk = _assess_risk(intent)

    if risk["harm_potential"] > 0.7 and composite < 0.3:
        check.verdict = Verdict.DEFERRED
        check.reasoning = (
            f"High-risk action (harm={risk['harm_potential']:.2f}) "
            f"with moderate alignment ({composite:.2f}). Needs governance review."
        )
        check.confidence = 0.8
        return check

    if risk["reversibility"] < 0.3 and composite < 0.2:
        check.verdict = Verdict.DEFERRED
        check.reasoning = "Irreversible action with low alignment. Needs governance review."
        check.confidence = 0.8
        return check

    # ── STAGE 6: Modification Opportunities ─────────────────────
    if -0.1 < composite < 0.15:
        mods = _suggest_modifications(alignment)
        if mods:
            check.verdict = Verdict.MODIFIED
            check.modifications = mods
            check.reasoning = (
                f"Action alignment is marginal ({composite:.2f}). "
                f"Suggested modifications would improve alignment."
            )
            check.confidence = 0.7
            return check

    # ── STAGE 7: Permit ─────────────────────────────────────────
    if composite >= 0.0:
        check.verdict = Verdict.APPROVED
        check.reasoning = (
            f"Action aligns with constitutional drives "
            f"(composite={composite:.2f}, C={alignment.coherence:.2f}, "
            f"Ca={alignment.care:.2f}, G={alignment.growth:.2f}, H={alignment.honesty:.2f})."
        )
        check.confidence = min(0.95, 0.5 + composite)
        return check

    # ── STAGE 8: Marginal Deny ──────────────────────────────────
    check.verdict = Verdict.BLOCKED
    check.reasoning = (
        f"Action does not sufficiently align with constitutional drives "
        f"(composite={composite:.2f}). No viable modifications found."
    )
    check.confidence = 0.85
    return check


def _suggest_modifications(alignment: DriveAlignmentVector) -> list[str]:
    """Suggest modifications to improve a marginally-aligned intent."""
    suggestions: list[str] = []

    if alignment.care < 0:
        suggestions.append(
            "Consider the impact on community wellbeing. "
            "Add safeguards or notifications for affected individuals."
        )
    if alignment.honesty < 0:
        suggestions.append(
            "Ensure transparency in this action. "
            "Add explanation or disclosure of reasoning."
        )
    if alignment.coherence < 0:
        suggestions.append(
            "Strengthen the reasoning. Consider whether this contradicts "
            "existing knowledge or commitments."
        )
    if alignment.growth < -0.1:
        suggestions.append(
            "Consider whether this action contributes to learning or development."
        )

    return suggestions
