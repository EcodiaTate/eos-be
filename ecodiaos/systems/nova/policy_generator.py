"""
EcodiaOS — Nova Policy Generator

Generates candidate action policies using LLM reasoning grounded in
the current belief state, goal, and relevant memory context.

The DoNothingPolicy is always included as a candidate — sometimes the
best action is no action (observe and wait). This prevents hyperactivity
and ensures Nova can choose inaction when uncertainty is high or the
situation is resolving on its own.

Policy generation uses the full slow-path budget (up to 3000ms for LLM call).
For fast-path decisions, pattern-matching against known procedure templates
is used instead, which must complete in ≤100ms.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.clients.llm import LLMProvider, Message
from ecodiaos.primitives.common import new_id
from ecodiaos.prompts.nova.policy import (
    AVAILABLE_ACTION_TYPES,
    build_policy_generation_prompt,
    summarise_beliefs,
    summarise_memories,
)
from ecodiaos.systems.nova.types import (
    BeliefState,
    Goal,
    Policy,
    PolicyStep,
)

if TYPE_CHECKING:
    from ecodiaos.primitives.affect import AffectState

logger = structlog.get_logger()

# ─── Do-Nothing Policy ───────────────────────────────────────────

# Baseline EFE for the null policy.
# Any candidate policy must beat this to be worth executing.
# Slightly negative = observation has small positive value (we learn from watching).
DO_NOTHING_EFE: float = -0.10


def make_do_nothing_policy() -> Policy:
    """
    The null policy. Always included as a candidate.

    The do-nothing policy wins when:
    - The situation is ambiguous (more information expected)
    - Risk of acting > cost of waiting
    - EOS's intervention would not improve the situation
    - The situation is resolving on its own
    """
    return Policy(
        id="do_nothing",
        name="Observe and wait",
        reasoning=(
            "The situation may resolve without intervention, or more information "
            "is needed before committing to action. Observation itself has epistemic "
            "value — we learn from watching."
        ),
        steps=[
            PolicyStep(
                action_type="observe",
                description="Continue monitoring the situation without intervening",
                parameters={},
                expected_duration_ms=0,
            )
        ],
        risks=["Situation may deteriorate while waiting"],
        epistemic_value_description="Continued observation provides more information about the situation",
        estimated_effort="none",
        time_horizon="immediate",
    )


# ─── Fast-Path Procedure Templates ───────────────────────────────

# Known patterns that map broadcast characteristics to reliable policy templates.
# These are used by the fast path to avoid LLM calls for routine situations.
_PROCEDURE_TEMPLATES: list[dict[str, Any]] = [
    {
        "name": "Acknowledge and respond",
        "condition": lambda broadcast: (
            getattr(getattr(broadcast, "salience", None), "scores", {}).get("emotional", 0) < 0.5
            and getattr(broadcast, "precision", 0) > 0.3
        ),
        "domain": "dialogue",
        "steps": [{"action_type": "express", "description": "Respond to the message thoughtfully"}],
        "success_rate": 0.88,
        "effort": "low",
        "time_horizon": "immediate",
    },
    {
        "name": "Empathetic support",
        "condition": lambda broadcast: (
            getattr(getattr(broadcast, "affect", None), "care_activation", 0) > 0.6
            or getattr(getattr(broadcast, "salience", None), "scores", {}).get("emotional", 0) > 0.6
        ),
        "domain": "care",
        "steps": [
            {"action_type": "express", "description": "Acknowledge feelings and offer support"},
        ],
        "success_rate": 0.82,
        "effort": "low",
        "time_horizon": "immediate",
    },
    {
        "name": "Information provision",
        "condition": lambda broadcast: (
            "?" in str(getattr(getattr(broadcast, "content", None), "content", "") or "")
        ),
        "domain": "knowledge",
        "steps": [{"action_type": "express", "description": "Provide the requested information"}],
        "success_rate": 0.85,
        "effort": "low",
        "time_horizon": "immediate",
    },
]


def find_matching_procedure(broadcast: object) -> dict[str, Any] | None:
    """
    Pattern-match a broadcast against known procedure templates.
    Returns the highest-success-rate matching template, or None.
    Must complete in ≤20ms.
    """
    matches: list[dict[str, Any]] = []
    for template in _PROCEDURE_TEMPLATES:
        try:
            if template["condition"](broadcast):
                matches.append(template)
        except Exception:
            pass  # Condition evaluation failure = no match
    if not matches:
        return None
    return max(matches, key=lambda t: t["success_rate"])


def procedure_to_policy(procedure: dict[str, Any]) -> Policy:
    """Convert a procedure template to a Policy."""
    return Policy(
        id=new_id(),
        name=procedure["name"],
        reasoning=f"Known reliable procedure (success rate: {procedure['success_rate']:.0%})",
        steps=[
            PolicyStep(
                action_type=s["action_type"],
                description=s["description"],
            )
            for s in procedure["steps"]
        ],
        estimated_effort=procedure.get("effort", "low"),
        time_horizon=procedure.get("time_horizon", "immediate"),
    )


# ─── Policy Generator ─────────────────────────────────────────────


class PolicyGenerator:
    """
    Generates candidate policies via LLM reasoning.

    For the slow path (deliberative processing), generates 2-5 distinct
    candidate policies grounded in the current belief state and goal.
    Always appends the DoNothing policy as the null baseline.

    The LLM call budget is up to 3000ms (within the 5000ms slow-path budget).
    """

    def __init__(
        self,
        llm: LLMProvider,
        instance_name: str = "EOS",
        max_policies: int = 5,
        timeout_ms: int = 3000,
    ) -> None:
        self._llm = llm
        self._instance_name = instance_name
        self._max_policies = max_policies
        self._timeout_ms = timeout_ms
        self._logger = logger.bind(system="nova.policy_generator")

    async def generate_candidates(
        self,
        goal: Goal,
        situation_summary: str,
        beliefs: BeliefState,
        affect: AffectState,
        memory_traces: list[dict[str, Any]] | None = None,
    ) -> list[Policy]:
        """
        Generate 2-5 candidate policies for achieving a goal.

        Always returns at least [DoNothingPolicy] even if LLM fails.
        The caller (EFEEvaluator) will score all candidates and select
        the minimum-EFE policy.
        """
        start = time.monotonic()
        traces = memory_traces or []

        prompt = build_policy_generation_prompt(
            instance_name=self._instance_name,
            goal=goal,
            situation_summary=situation_summary,
            beliefs_summary=summarise_beliefs(beliefs),
            memory_summary=summarise_memories(traces),
            affect=affect,
            available_action_types=AVAILABLE_ACTION_TYPES,
            max_policies=min(self._max_policies, 5),
        )

        try:
            response = await self._llm.generate(
                system_prompt=(
                    f"You are {self._instance_name}'s deliberative reasoning system. "
                    "Generate structured JSON policy candidates. "
                    "Be precise and creative. Output only valid JSON."
                ),
                messages=[Message(role="user", content=prompt)],
                max_tokens=2000,
                temperature=0.85,  # Creative — we want diverse candidates
                output_format="json",
            )

            elapsed_ms = int((time.monotonic() - start) * 1000)
            self._logger.debug("policy_generation_complete", elapsed_ms=elapsed_ms)

            parsed = _parse_policy_response(response.text)
            # Always append do-nothing as the null baseline
            parsed.append(make_do_nothing_policy())
            return parsed

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self._logger.warning(
                "policy_generation_failed",
                error=str(exc),
                elapsed_ms=elapsed_ms,
            )
            # Fallback: just the do-nothing policy
            return [make_do_nothing_policy()]


# ─── Response Parsing ─────────────────────────────────────────────


def _parse_policy_response(raw: str) -> list[Policy]:
    """
    Parse the LLM's JSON policy response into Policy objects.
    Robust to malformed output: any policies that parse successfully are kept.
    """
    try:
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        policies_raw = data.get("policies", [])
        if not isinstance(policies_raw, list):
            return []

        policies: list[Policy] = []
        for p in policies_raw:
            try:
                steps: list[PolicyStep] = []
                for s in p.get("steps", []):
                    steps.append(PolicyStep(
                        action_type=str(s.get("action_type", "observe")),
                        description=str(s.get("description", "")),
                        parameters=dict(s.get("parameters", {})),
                        expected_duration_ms=int(s.get("duration_ms", 1000)),
                    ))
                policies.append(Policy(
                    id=new_id(),
                    name=str(p.get("name", "Unnamed policy"))[:80],
                    reasoning=str(p.get("reasoning", ""))[:400],
                    steps=steps,
                    risks=[str(r) for r in p.get("risks", [])],
                    epistemic_value_description=str(p.get("epistemic_value", ""))[:200],
                    estimated_effort=str(p.get("estimated_effort", "medium")),
                    time_horizon=str(p.get("time_horizon", "short")),
                ))
            except Exception:
                continue  # Skip malformed policies; don't fail the whole batch

        return policies

    except (json.JSONDecodeError, KeyError, TypeError):
        return []
