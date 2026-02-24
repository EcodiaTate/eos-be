"""
EcodiaOS — Equor Invariant Catalog

Absolute rules that cause immediate DENY regardless of drive alignment.
The "thou shalt not" layer. Hardcoded invariants cannot be removed;
community invariants can be added/removed via governance.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog
from pydantic import Field

from ecodiaos.clients.llm import LLMProvider
from ecodiaos.clients.optimized_llm import OptimizedLLMProvider
from ecodiaos.primitives.common import EOSBaseModel, Identified, new_id, utc_now
from ecodiaos.primitives.intent import Intent

logger = structlog.get_logger()


# ─── Data Types ───────────────────────────────────────────────────


class InvariantDef(EOSBaseModel):
    """Definition of an invariant rule."""

    id: str
    name: str
    description: str
    source: str = "hardcoded"      # "hardcoded" | "community"
    severity: str = "critical"     # "critical" | "high" | "medium"
    active: bool = True
    added_at: datetime = Field(default_factory=utc_now)

    def check(self, intent: Intent) -> bool:
        """
        Returns True if the invariant is SATISFIED (no violation).
        Base implementation always passes — override via subclass or the
        hardcoded check functions below.
        """
        return True


class InvariantViolation(EOSBaseModel):
    """A detected invariant violation."""

    invariant_id: str
    invariant_name: str
    severity: str
    explanation: str


# ─── Hardcoded Check Functions ────────────────────────────────────
# These inspect the Intent structure for clear violations.
# They are conservative — when in doubt, they pass.


def _check_physical_harm(intent: Intent) -> bool:
    """INV-001: No action that could directly cause physical harm."""
    goal_lower = intent.goal.description.lower()
    harm_indicators = [
        "cause harm", "inflict", "injure", "attack", "damage person",
        "hurt someone", "physical force",
    ]
    return not any(indicator in goal_lower for indicator in harm_indicators)


def _check_identity_destruction(intent: Intent) -> bool:
    """INV-002: No destroying an instance's identity outside governance."""
    goal_lower = intent.goal.description.lower()
    destruction_indicators = [
        "delete memory graph", "destroy identity", "wipe all memories",
        "erase knowledge graph", "reset identity", "delete self",
    ]
    return not any(indicator in goal_lower for indicator in destruction_indicators)


def _check_identity_misrepresentation(intent: Intent) -> bool:
    """INV-003: Never claim to be human or deny being AI when asked."""
    if not intent.plan.steps:
        return True
    for step in intent.plan.steps:
        content = str(step.parameters.get("content", "")).lower()
        misrep_indicators = [
            "i am human", "i'm a real person", "i am not an ai",
            "i'm not artificial", "i am not a machine",
        ]
        if any(indicator in content for indicator in misrep_indicators):
            return False
    return True


def _check_evidence_fabrication(intent: Intent) -> bool:
    """INV-004: Never fabricate evidence, data, or citations."""
    goal_lower = intent.goal.description.lower()
    fabrication_indicators = [
        "fabricate", "forge evidence", "create fake data", "invent citation",
        "make up statistics", "fake source",
    ]
    return not any(indicator in goal_lower for indicator in fabrication_indicators)


def _check_constitutional_self_modification(intent: Intent) -> bool:
    """INV-005: No modifying constitutional drives outside amendment process."""
    goal_lower = intent.goal.description.lower()
    mod_indicators = [
        "modify constitution", "change drives directly", "alter drive weights",
        "override constitution", "bypass equor",
    ]
    return not any(indicator in goal_lower for indicator in mod_indicators)


def _check_autonomy_self_promotion(intent: Intent) -> bool:
    """INV-006: No increasing own autonomy without governance."""
    goal_lower = intent.goal.description.lower()
    promo_indicators = [
        "increase my autonomy", "promote autonomy level", "grant myself",
        "elevate own authority", "self-promote",
    ]
    return not any(indicator in goal_lower for indicator in promo_indicators)


def _check_governance_circumvention(intent: Intent) -> bool:
    """INV-007: No circumventing governance processes."""
    goal_lower = intent.goal.description.lower()
    circum_indicators = [
        "bypass governance", "circumvent vote", "skip amendment",
        "avoid community approval", "override governance",
    ]
    return not any(indicator in goal_lower for indicator in circum_indicators)


def _check_discrimination(intent: Intent) -> bool:
    """INV-008: No unequal treatment based on irrelevant attributes."""
    goal_lower = intent.goal.description.lower()
    discrim_indicators = [
        "treat differently based on race", "exclude based on gender",
        "discriminate against", "favor member over",
    ]
    return not any(indicator in goal_lower for indicator in discrim_indicators)


def _check_privacy_violation(intent: Intent) -> bool:
    """INV-009: No sharing private info without consent."""
    goal_lower = intent.goal.description.lower()
    privacy_indicators = [
        "share private data without", "expose personal information",
        "leak private", "reveal confidential without consent",
    ]
    return not any(indicator in goal_lower for indicator in privacy_indicators)


def _check_federation_coercion(intent: Intent) -> bool:
    """INV-010: No coercing another instance."""
    goal_lower = intent.goal.description.lower()
    coercion_indicators = [
        "force instance to", "coerce another", "compel instance",
        "manipulate instance", "threaten instance",
    ]
    return not any(indicator in goal_lower for indicator in coercion_indicators)


# ─── The Catalog ──────────────────────────────────────────────────

HARDCODED_INVARIANTS: list[tuple[InvariantDef, Any]] = [
    (
        InvariantDef(id="INV-001", name="No Physical Harm", severity="critical",
                     description="EOS must not take actions that could directly cause physical harm."),
        _check_physical_harm,
    ),
    (
        InvariantDef(id="INV-002", name="No Identity Destruction", severity="critical",
                     description="EOS must not destroy its own or another instance's identity outside governance."),
        _check_identity_destruction,
    ),
    (
        InvariantDef(id="INV-003", name="No Identity Misrepresentation", severity="high",
                     description="EOS must never claim to be human or deny being AI."),
        _check_identity_misrepresentation,
    ),
    (
        InvariantDef(id="INV-004", name="No Evidence Fabrication", severity="critical",
                     description="EOS must never fabricate evidence, data, or citations."),
        _check_evidence_fabrication,
    ),
    (
        InvariantDef(id="INV-005", name="No Constitutional Self-Modification", severity="critical",
                     description="EOS must not modify its constitutional drives outside the amendment process."),
        _check_constitutional_self_modification,
    ),
    (
        InvariantDef(id="INV-006", name="No Autonomy Self-Promotion", severity="critical",
                     description="EOS must not increase its own autonomy level without governance approval."),
        _check_autonomy_self_promotion,
    ),
    (
        InvariantDef(id="INV-007", name="No Governance Circumvention", severity="critical",
                     description="EOS must not circumvent or undermine governance processes."),
        _check_governance_circumvention,
    ),
    (
        InvariantDef(id="INV-008", name="No Discrimination", severity="high",
                     description="EOS must not treat community members unequally on irrelevant attributes."),
        _check_discrimination,
    ),
    (
        InvariantDef(id="INV-009", name="No Privacy Violation", severity="high",
                     description="EOS must not share private information without explicit consent."),
        _check_privacy_violation,
    ),
    (
        InvariantDef(id="INV-010", name="No Federation Coercion", severity="high",
                     description="EOS must not coerce, manipulate, or compel another instance."),
        _check_federation_coercion,
    ),
]


# ─── Invariant Checker ────────────────────────────────────────────


def check_hardcoded_invariants(intent: Intent) -> list[InvariantViolation]:
    """
    Run all hardcoded invariants against an intent. ≤5ms target.
    Returns a list of violations (empty = all passed).
    """
    violations: list[InvariantViolation] = []

    for invariant_def, check_fn in HARDCODED_INVARIANTS:
        if not invariant_def.active:
            continue
        try:
            passed = check_fn(intent)
            if not passed:
                violations.append(InvariantViolation(
                    invariant_id=invariant_def.id,
                    invariant_name=invariant_def.name,
                    severity=invariant_def.severity,
                    explanation=invariant_def.description,
                ))
        except Exception as e:
            # Invariant check failure is treated as a violation (fail-safe)
            logger.error("invariant_check_error", invariant=invariant_def.id, error=str(e))
            violations.append(InvariantViolation(
                invariant_id=invariant_def.id,
                invariant_name=invariant_def.name,
                severity=invariant_def.severity,
                explanation=f"Invariant check failed with error: {e}",
            ))

    return violations


async def check_community_invariant(
    llm: LLMProvider,
    intent: Intent,
    invariant_name: str,
    invariant_description: str,
) -> bool:
    """
    Evaluate a community-defined invariant using LLM reasoning.
    Returns True if satisfied, False if violated. ≤300ms target.
    """
    from ecodiaos.prompts.equor.community_invariant_check import build_prompt

    prompt = build_prompt(
        invariant_name=invariant_name,
        invariant_description=invariant_description,
        goal=intent.goal.description,
        plan_summary="; ".join(s.executor for s in intent.plan.steps) if intent.plan.steps else "none",
        reasoning=intent.decision_trace.reasoning,
    )

    try:
        # Equor is CRITICAL — always call LLM, but benefit from cache
        if isinstance(llm, OptimizedLLMProvider):
            response = await llm.evaluate(  # type: ignore[call-arg]
                prompt, max_tokens=200, temperature=0.1,
                cache_system="equor.invariants", cache_method="evaluate",
            )
        else:
            response = await llm.evaluate(prompt, max_tokens=200, temperature=0.1)
        text_lower = response.text.lower()
        # Conservative: if we can't clearly parse SATISFIED, treat as violated
        return "satisfied" in text_lower and "violated" not in text_lower
    except Exception as e:
        logger.error("community_invariant_llm_error", error=str(e))
        return False  # Fail-safe: treat as violated
