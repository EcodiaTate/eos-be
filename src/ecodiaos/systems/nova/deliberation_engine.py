"""
EcodiaOS — Nova Deliberation Engine

The dual-process decision engine. Implements the System 1 / System 2 split
from cognitive architecture — not as a performance trick, but as a genuine
model of how deliberation works.

Fast path (System 1, ≤150ms total):
  - Pattern-match against known procedure templates
  - Build intent directly from matched procedure
  - Submit to Equor for critical-path review (≤50ms)
  - If denied, escalate to slow path

Slow path (System 2, ≤5000ms total):
  - Generate 2-5 candidate policies via LLM (≤3000ms)
  - Evaluate EFE for each candidate in parallel (≤200ms per policy)
  - Select minimum-EFE policy
  - Formulate Intent from selected policy
  - Submit to Equor for standard review (≤500ms)
  - If denied, retry with next-best policy

Routing decision:
  The choice between fast and slow path is itself a decision.
  Novelty, risk, emotional intensity, and belief conflict drive it.
  Over time (via Evo), more situations shift from slow to fast as
  reliable patterns are learned.

The null outcome: if no active goal matches and the broadcast doesn't
warrant creating one, deliberation returns None — no action taken.
This is the correct outcome, not a failure.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import structlog

from ecodiaos.primitives.affect import AffectState
from ecodiaos.primitives.common import Verdict, new_id, utc_now
from ecodiaos.primitives.constitutional import ConstitutionalCheck
from ecodiaos.primitives.intent import (
    Action,
    ActionSequence,
    DecisionTrace,
    GoalDescriptor,
    Intent,
)
from ecodiaos.systems.atune.types import WorkspaceBroadcast
from ecodiaos.systems.nova.efe_evaluator import EFEEvaluator
from ecodiaos.systems.nova.goal_manager import GoalManager
from ecodiaos.systems.nova.policy_generator import (
    PolicyGenerator,
    find_matching_procedure,
    make_do_nothing_policy,
    procedure_to_policy,
)
from ecodiaos.systems.nova.types import (
    BeliefState,
    DecisionRecord,
    EFEScore,
    Goal,
    Policy,
    PriorityContext,
    SituationAssessment,
)

if TYPE_CHECKING:
    from ecodiaos.systems.equor.service import EquorService

logger = structlog.get_logger()

# Thresholds that trigger slow-path deliberation (from spec)
_NOVELTY_THRESHOLD = 0.6
_RISK_THRESHOLD = 0.5
_EMOTIONAL_THRESHOLD = 0.7
_PRECISION_THRESHOLD = 0.8


class DeliberationEngine:
    """
    The dual-process cognitive architecture.

    Receives workspace broadcasts, assesses situation, routes to fast or slow
    deliberation, and returns a constitutional Intent (or None for do-nothing).

    Performance targets:
      - Fast path: ≤150ms total (≤100ms procedure + ≤50ms Equor critical)
      - Slow path: ≤5000ms total (≤3000ms generation + ≤500ms Equor + overhead)
    """

    def __init__(
        self,
        goal_manager: GoalManager,
        policy_generator: PolicyGenerator,
        efe_evaluator: EFEEvaluator,
        equor: "EquorService",
        drive_weights: dict[str, float] | None = None,
        fast_path_timeout_ms: int = 100,
        slow_path_timeout_ms: int = 5000,
    ) -> None:
        self._goals = goal_manager
        self._policy_gen = policy_generator
        self._efe = efe_evaluator
        self._equor = equor
        self._drive_weights = drive_weights or {
            "coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0
        }
        self._fast_timeout = fast_path_timeout_ms / 1000.0
        self._slow_timeout = slow_path_timeout_ms / 1000.0
        self._last_equor_check: ConstitutionalCheck | None = None
        self._logger = logger.bind(system="nova.deliberation_engine")

    def update_drive_weights(self, weights: dict[str, float]) -> None:
        """Called by NovaService when constitution changes."""
        self._drive_weights = weights

    @property
    def last_equor_check(self) -> ConstitutionalCheck | None:
        """The Equor check from the most recent approved intent."""
        return self._last_equor_check

    async def deliberate(
        self,
        broadcast: WorkspaceBroadcast,
        belief_state: BeliefState,
        affect: AffectState,
        belief_delta_is_conflicting: bool = False,
        memory_traces: list[dict] | None = None,
    ) -> tuple[Intent | None, DecisionRecord]:
        """
        Main deliberation entry point.

        Returns (Intent | None, DecisionRecord).
        Returns None when the best action is no action.
        DecisionRecord is always returned for observability.
        """
        start = time.monotonic()
        self._last_equor_check = None  # Reset per-deliberation

        try:
            # End-to-end timeout: the entire deliberation (including possible
            # fast→slow escalation) must complete within the slow-path budget.
            async with asyncio.timeout(self._slow_timeout):
                return await self._deliberate_inner(
                    broadcast, belief_state, affect,
                    belief_delta_is_conflicting, memory_traces, start,
                )
        except asyncio.TimeoutError:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self._logger.warning("deliberation_end_to_end_timeout", elapsed_ms=elapsed_ms)
            record = DecisionRecord(
                broadcast_id=broadcast.broadcast_id,
                path="timeout",
                latency_ms=elapsed_ms,
            )
            return None, record

    async def _deliberate_inner(
        self,
        broadcast: WorkspaceBroadcast,
        belief_state: BeliefState,
        affect: AffectState,
        belief_delta_is_conflicting: bool,
        memory_traces: list[dict] | None,
        start: float,
    ) -> tuple[Intent | None, DecisionRecord]:
        """Inner deliberation logic, called within the end-to-end timeout."""
        # Recompute goal priorities before deliberating
        priority_ctx = PriorityContext(
            current_affect=affect,
            drive_weights=self._drive_weights,
        )
        self._goals.recompute_priorities(priority_ctx)

        # Assess situation to determine path
        assessment = self._assess_situation(
            broadcast=broadcast,
            belief_conflict=belief_delta_is_conflicting,
        )

        record = DecisionRecord(
            broadcast_id=broadcast.broadcast_id,
            situation_assessment=assessment,
        )

        # Route to appropriate path
        if assessment.requires_deliberation:
            self._logger.debug("deliberation_slow_path", broadcast_id=broadcast.broadcast_id)
            intent = await self._slow_path(broadcast, assessment, belief_state, affect, memory_traces)
            path = "slow"
        else:
            self._logger.debug("deliberation_fast_path", broadcast_id=broadcast.broadcast_id)
            intent, escalated = await self._fast_path(broadcast, assessment, belief_state, affect)
            path = "slow" if escalated else "fast"
            if escalated and intent is None:
                # Fast path escalated and slow path succeeded
                intent = await self._slow_path(broadcast, assessment, belief_state, affect, memory_traces)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        record = record.model_copy(
            update={
                "path": path if intent is not None else ("do_nothing" if not assessment.requires_deliberation else "no_goal"),
                "intent_dispatched": intent is not None,
                "latency_ms": elapsed_ms,
            }
        )

        self._logger.info(
            "deliberation_complete",
            path=record.path,
            intent_dispatched=intent is not None,
            latency_ms=elapsed_ms,
        )
        return intent, record

    # ─── Situation Assessment ─────────────────────────────────────

    def _assess_situation(
        self,
        broadcast: WorkspaceBroadcast,
        belief_conflict: bool,
    ) -> SituationAssessment:
        """
        Determine if deliberative (slow) or habitual (fast) processing is needed.
        Must complete in ≤20ms.
        """
        salience_scores = broadcast.salience.scores if broadcast.salience.scores else {}
        novelty = salience_scores.get("novelty", 0.0)
        risk = salience_scores.get("risk", 0.0)
        emotional = salience_scores.get("emotional", 0.0)
        precision = broadcast.precision

        requires_deliberation = (
            novelty > _NOVELTY_THRESHOLD
            or risk > _RISK_THRESHOLD
            or emotional > _EMOTIONAL_THRESHOLD
            or belief_conflict
            or precision > _PRECISION_THRESHOLD
        )

        has_procedure = find_matching_procedure(broadcast) is not None

        return SituationAssessment(
            novelty=novelty,
            risk=risk,
            emotional_intensity=emotional,
            belief_conflict=belief_conflict,
            requires_deliberation=requires_deliberation,
            has_matching_procedure=has_procedure,
            broadcast_precision=precision,
        )

    # ─── Fast Path ────────────────────────────────────────────────

    async def _fast_path(
        self,
        broadcast: WorkspaceBroadcast,
        assessment: SituationAssessment,
        belief_state: BeliefState,
        affect: AffectState,
    ) -> tuple[Intent | None, bool]:
        """
        System 1: Pattern-match → build intent → Equor critical review.
        Returns (intent | None, escalated_to_slow).
        """
        try:
            async with asyncio.timeout(self._fast_timeout):
                procedure = find_matching_procedure(broadcast)
                if procedure is None:
                    return None, True  # No matching procedure → escalate

                policy = procedure_to_policy(procedure)

                # Find or create a goal for this intent
                goal = self._goals.find_relevant_goal(broadcast)
                if goal is None:
                    goal = self._goals.create_from_broadcast(broadcast)
                if goal is None:
                    return None, False  # No goal → do nothing

                intent = _policy_to_intent(policy, goal, path="fast", confidence=procedure["success_rate"])

                # Equor critical-path review (≤50ms budget within fast path)
                check = await self._equor.review(intent)

                if check.verdict == Verdict.APPROVED:
                    self._last_equor_check = check
                    return intent, False
                elif check.verdict == Verdict.MODIFIED:
                    self._last_equor_check = check
                    intent = _apply_modifications(intent, check.modifications)
                    return intent, False
                else:
                    # Denied by Equor → escalate to slow path
                    self._logger.info("fast_path_equor_denied_escalating", intent_id=intent.id)
                    return None, True

        except asyncio.TimeoutError:
            self._logger.warning("fast_path_timeout_escalating")
            return None, True
        except Exception as exc:
            self._logger.error("fast_path_error", error=str(exc))
            return None, True

    # ─── Slow Path ────────────────────────────────────────────────

    async def _slow_path(
        self,
        broadcast: WorkspaceBroadcast,
        assessment: SituationAssessment,
        belief_state: BeliefState,
        affect: AffectState,
        memory_traces: list[dict] | None,
    ) -> Intent | None:
        """
        System 2: Generate → EFE score → select → Equor standard review.
        """
        try:
            async with asyncio.timeout(self._slow_timeout):
                # ── Find or create goal ──
                goal = self._goals.find_relevant_goal(broadcast)
                if goal is None:
                    goal = self._goals.create_from_broadcast(broadcast)
                if goal is None:
                    return None  # No goal warrants no action

                # ── Extract situation summary for policy generation ──
                situation = _extract_situation_summary(broadcast)

                # ── Generate candidate policies (up to 3000ms) ──
                candidates = await self._policy_gen.generate_candidates(
                    goal=goal,
                    situation_summary=situation,
                    beliefs=belief_state,
                    affect=affect,
                    memory_traces=memory_traces,
                )

                if not candidates:
                    return None

                # ── Evaluate EFE for all candidates (parallelised) ──
                scored = await self._efe.evaluate_all(
                    policies=candidates,
                    goal=goal,
                    beliefs=belief_state,
                    affect=affect,
                    drive_weights=self._drive_weights,
                )

                # scored is sorted: lowest EFE first
                # If do-nothing wins, return None
                if scored and scored[0][0].id == "do_nothing":
                    self._logger.info(
                        "do_nothing_policy_selected",
                        goal=goal.description[:60],
                        do_nothing_efe=scored[0][1].total,
                    )
                    return None

                # ── Equor review with retry on denial ──
                for policy, efe_score in scored:
                    if policy.id == "do_nothing":
                        continue  # Skip do-nothing — if we're here, we want to act

                    intent = _policy_to_intent(
                        policy,
                        goal,
                        path="slow",
                        confidence=efe_score.confidence,
                        efe_score=efe_score,
                        all_efe_scores={p.name: e.total for p, e in scored},
                    )

                    check = await self._equor.review(intent)

                    if check.verdict == Verdict.APPROVED:
                        self._last_equor_check = check
                        return intent
                    elif check.verdict == Verdict.MODIFIED:
                        self._last_equor_check = check
                        return _apply_modifications(intent, check.modifications)
                    elif check.verdict == Verdict.DEFERRED:
                        self._logger.info("intent_deferred_by_equor", intent_id=intent.id)
                        return None  # Governance will handle
                    # BLOCKED → try next policy
                    self._logger.info(
                        "policy_blocked_by_equor_trying_next",
                        policy=policy.name,
                        reasoning=check.reasoning[:80],
                    )

                return None  # All policies blocked

        except asyncio.TimeoutError:
            self._logger.warning("slow_path_timeout")
            return None
        except Exception as exc:
            self._logger.error("slow_path_error", error=str(exc))
            return None


# ─── Intent Construction ─────────────────────────────────────────


def _policy_to_intent(
    policy: Policy,
    goal: Goal,
    path: str,
    confidence: float = 0.7,
    efe_score: EFEScore | None = None,
    all_efe_scores: dict[str, float] | None = None,
) -> Intent:
    """
    Convert a selected Policy into a formal Intent primitive.
    Intents are the cross-system communication unit; Policies are Nova-internal.
    """
    actions = [
        Action(
            executor=f"executor.{step.action_type}",
            parameters={
                "description": step.description,
                **step.parameters,
            },
            timeout_ms=step.expected_duration_ms * 2,
        )
        for step in policy.steps
    ]

    efe_reasoning = ""
    if efe_score:
        efe_reasoning = efe_score.reasoning

    trace_alternatives = []
    if all_efe_scores:
        trace_alternatives = [{"policy": name, "efe": efe} for name, efe in all_efe_scores.items()]

    return Intent(
        id=new_id(),
        goal=GoalDescriptor(
            description=goal.description,
            target_domain=goal.target_domain,
            success_criteria={"criteria": goal.success_criteria} if goal.success_criteria else {},
        ),
        plan=ActionSequence(steps=actions),
        expected_free_energy=efe_score.total if efe_score else 0.0,
        priority=goal.priority,
        decision_trace=DecisionTrace(
            reasoning=(
                f"Path: {path}. Policy: {policy.name}. {policy.reasoning[:200]}. "
                f"EFE evaluation: {efe_reasoning}"
            ),
            alternatives_considered=trace_alternatives,
            free_energy_scores=all_efe_scores or {},
        ),
    )


def _apply_modifications(intent: Intent, modifications: list[str]) -> Intent:
    """Apply Equor's suggested modifications to an intent."""
    existing_reasoning = intent.decision_trace.reasoning
    modification_notes = "; ".join(modifications[:5])
    updated_trace = intent.decision_trace.model_copy(
        update={
            "reasoning": f"{existing_reasoning} | Equor modifications: {modification_notes}"
        }
    )
    return intent.model_copy(update={"decision_trace": updated_trace})


def _extract_situation_summary(broadcast: WorkspaceBroadcast) -> str:
    """Extract a brief situation summary from a broadcast for policy generation."""
    content = broadcast.content
    paths = [
        ("content", "content"),
        ("content",),
    ]
    for path in paths:
        obj: object = content
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, str) and obj:
            return obj[:400]
    return f"Broadcast from workspace (salience: {broadcast.salience.composite:.2f})"
