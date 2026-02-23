"""
EcodiaOS — Thymos Prescription Layer (Repair Strategy)

Based on diagnosis, Thymos prescribes the least invasive effective repair.
This follows the principle of minimal intervention: try rest before
antibiotics before surgery.

Two components:
  1. RepairPrescriber   — selects the repair tier and generates RepairSpec
  2. RepairValidator    — gates repairs through constitutional review + safety
"""

from __future__ import annotations

from typing import Any

import structlog

from ecodiaos.primitives.common import utc_now
from ecodiaos.systems.thymos.types import (
    Diagnosis,
    Incident,
    IncidentSeverity,
    ParameterFix,
    RepairSpec,
    RepairTier,
    ValidationResult,
)

logger = structlog.get_logger()


# ─── Parameter Fix Registry ─────────────────────────────────────


# Root cause → parameter adjustments that might resolve it
PARAMETER_FIXES: dict[str, list[ParameterFix]] = {
    "memory_pressure": [
        ParameterFix(
            parameter_path="synapse.resources.memory.evo",
            delta=-128,
            reason="Reduce Evo memory to relieve pressure",
        ),
        ParameterFix(
            parameter_path="synapse.resources.memory.simula",
            delta=-64,
            reason="Reduce Simula memory",
        ),
    ],
    "retrieval_timeout": [
        ParameterFix(
            parameter_path="memory.retrieval.timeout_ms",
            delta=50,
            reason="Give Memory more time for retrieval",
        ),
    ],
    "workspace_contention": [
        ParameterFix(
            parameter_path="synapse.clock.current_period_ms",
            delta=20,
            reason="Slow the cycle to reduce workspace contention",
        ),
    ],
    "llm_rate_limit": [
        ParameterFix(
            parameter_path="voxis.generation.max_concurrent",
            delta=-1,
            reason="Reduce concurrent LLM calls",
        ),
        ParameterFix(
            parameter_path="evo.hypothesis.batch_size",
            delta=-1,
            reason="Reduce Evo LLM usage",
        ),
    ],
    "Resource pressure causing latency increase": [
        ParameterFix(
            parameter_path="synapse.clock.current_period_ms",
            delta=30,
            reason="Slow cycle to reduce resource pressure",
        ),
    ],
}


# ─── Repair Prescriber ──────────────────────────────────────────


class RepairPrescriber:
    """
    Prescribes repairs following the principle of minimal intervention:
    the least invasive fix that resolves the issue.

    Tier 0: No-op — transient, already resolved
    Tier 1: Parameter tweak — adjustable without code changes
    Tier 2: System restart — bad state but code is fine
    Tier 3: Known fix — apply antibody from the library
    Tier 4: Novel fix — generate via Simula Code Agent
    Tier 5: Human escalation — cannot auto-resolve
    """

    def __init__(self) -> None:
        self._logger = logger.bind(system="thymos", component="prescriber")

    async def prescribe(
        self,
        incident: Incident,
        diagnosis: Diagnosis,
    ) -> RepairSpec:
        """Generate a repair specification based on diagnosis."""

        # ── TIER 0: No-op ──
        if incident.occurrence_count == 1 and self._is_likely_transient(incident):
            return RepairSpec(
                tier=RepairTier.NOOP,
                action="log_and_monitor",
                reason="Transient single occurrence — monitoring",
            )

        # ── TIER 3: Known Fix (Antibody) ── (check before parameter/restart)
        if diagnosis.antibody_id is not None:
            return RepairSpec(
                tier=RepairTier.KNOWN_FIX,
                action="apply_antibody",
                antibody_id=diagnosis.antibody_id,
                reason=f"Antibody match: {diagnosis.root_cause}",
            )

        # ── TIER 1: Parameter Tweak ──
        param_fix = self._check_parameter_fixes(diagnosis)
        if param_fix is not None:
            return param_fix

        # ── TIER 2: System Restart ──
        restart_causes = {
            "state_corruption",
            "resource_leak",
            "deadlock",
            "memory_leak",
            "unbounded growth",
        }
        if any(cause in diagnosis.root_cause.lower() for cause in restart_causes):
            return RepairSpec(
                tier=RepairTier.RESTART,
                action="restart_system",
                target_system=incident.source_system,
                reason=f"State issue: {diagnosis.root_cause}",
            )

        # ── TIER 4: Novel Fix (Codegen) ──
        if diagnosis.confidence > 0.6 and self._is_codegen_appropriate(incident):
            return RepairSpec(
                tier=RepairTier.NOVEL_FIX,
                action="simula_codegen",
                target_system=incident.source_system,
                reason=f"Novel repair needed: {diagnosis.root_cause}",
            )

        # ── TIER 2: Restart as fallback ──
        if incident.severity in (IncidentSeverity.CRITICAL, IncidentSeverity.HIGH):
            return RepairSpec(
                tier=RepairTier.RESTART,
                action="restart_system",
                target_system=incident.source_system,
                reason=f"High severity, no specific fix: {diagnosis.root_cause}",
            )

        # ── TIER 5: Human Escalation ──
        return RepairSpec(
            tier=RepairTier.ESCALATE,
            action="alert_operator",
            reason=(
                f"Cannot auto-resolve: {diagnosis.root_cause} "
                f"(confidence: {diagnosis.confidence:.2f})"
            ),
        )

    def _is_likely_transient(self, incident: Incident) -> bool:
        """Check if an incident is likely transient (network hiccup, etc.)."""
        transient_types = {
            "TimeoutError",
            "ConnectionError",
            "ConnectionResetError",
            "ConnectionRefusedError",
        }
        return incident.error_type in transient_types

    def _check_parameter_fixes(self, diagnosis: Diagnosis) -> RepairSpec | None:
        """Check if a parameter adjustment can resolve the issue."""
        # Check root cause against known parameter fix patterns
        for pattern, fixes in PARAMETER_FIXES.items():
            if pattern.lower() in diagnosis.root_cause.lower():
                return RepairSpec(
                    tier=RepairTier.PARAMETER,
                    action="adjust_parameters",
                    parameter_changes=[f.model_dump() for f in fixes],
                    reason=f"Parameter adjustment for: {pattern}",
                )
        return None

    def _is_codegen_appropriate(self, incident: Incident) -> bool:
        """Should we attempt codegen repair?"""
        # Don't codegen for transient or low-severity issues
        if incident.severity in (IncidentSeverity.LOW, IncidentSeverity.INFO):
            return False
        # Don't codegen for system-wide issues
        if incident.blast_radius > 0.5:
            return False
        # Must have a stack trace for codegen to work on
        return incident.stack_trace is not None


# ─── Repair Validator ────────────────────────────────────────────


class RepairValidator:
    """
    Validates a proposed repair before application.

    Gate 1: Equor constitutional review (Tier 3+)
    Gate 2: Blast radius check (reject > 0.5 for auto-repair)
    Gate 3: Rate limit check (prevent healing storms)

    Simula sandbox validation (Gate 3 from spec) is handled by
    Simula's own simulation pipeline when Tier 4 repairs route through it.
    """

    MAX_REPAIRS_PER_HOUR = 5
    MAX_NOVEL_REPAIRS_PER_DAY = 3

    def __init__(self, equor: Any = None) -> None:
        """
        Args:
            equor: The EquorService for constitutional review.
        """
        self._equor = equor
        self._recent_repairs: list[float] = []  # timestamps of recent repairs
        self._recent_novel: list[float] = []  # timestamps of recent novel repairs
        self._logger = logger.bind(system="thymos", component="repair_validator")

    async def validate(
        self,
        incident: Incident,
        repair: RepairSpec,
    ) -> ValidationResult:
        """Run the full validation gate on a proposed repair."""

        # Gate 1: Constitutional review for Tier 3+
        if repair.tier >= RepairTier.KNOWN_FIX and self._equor is not None:
            try:
                review = await self._constitutional_review(incident, repair)
                if not review.approved:
                    return review
            except Exception as exc:
                self._logger.warning(
                    "equor_review_failed",
                    error=str(exc),
                    tier=repair.tier.name,
                )
                # Equor failure for high-tier repairs → escalate
                if repair.tier >= RepairTier.NOVEL_FIX:
                    return ValidationResult(
                        approved=False,
                        reason=f"Equor review failed: {exc}",
                        escalate_to=RepairTier.ESCALATE,
                    )

        # Gate 2: Blast radius for Tier 3+
        if repair.tier >= RepairTier.KNOWN_FIX:
            if incident.blast_radius > 0.5:
                return ValidationResult(
                    approved=False,
                    reason=(
                        f"Blast radius too high ({incident.blast_radius:.2f}) "
                        f"for automated repair"
                    ),
                    escalate_to=RepairTier.ESCALATE,
                )

        # Gate 3: Rate limiting
        now_ts = utc_now().timestamp()
        hour_ago = now_ts - 3600.0
        day_ago = now_ts - 86400.0

        recent_count = sum(1 for ts in self._recent_repairs if ts > hour_ago)
        if recent_count >= self.MAX_REPAIRS_PER_HOUR:
            return ValidationResult(
                approved=False,
                reason=(
                    f"Healing budget exceeded: {recent_count} repairs in last hour "
                    f"(max: {self.MAX_REPAIRS_PER_HOUR})"
                ),
                escalate_to=RepairTier.ESCALATE,
            )

        if repair.tier == RepairTier.NOVEL_FIX:
            novel_count = sum(1 for ts in self._recent_novel if ts > day_ago)
            if novel_count >= self.MAX_NOVEL_REPAIRS_PER_DAY:
                return ValidationResult(
                    approved=False,
                    reason=(
                        f"Novel repair budget exceeded: {novel_count} in 24h "
                        f"(max: {self.MAX_NOVEL_REPAIRS_PER_DAY})"
                    ),
                    escalate_to=RepairTier.ESCALATE,
                )

        return ValidationResult(approved=True)

    def record_repair(self, repair: RepairSpec) -> None:
        """Record that a repair was applied (for rate limiting)."""
        now_ts = utc_now().timestamp()
        self._recent_repairs.append(now_ts)
        if repair.tier == RepairTier.NOVEL_FIX:
            self._recent_novel.append(now_ts)

        # Prune old entries
        hour_ago = now_ts - 3600.0
        self._recent_repairs = [ts for ts in self._recent_repairs if ts > hour_ago]
        day_ago = now_ts - 86400.0
        self._recent_novel = [ts for ts in self._recent_novel if ts > day_ago]

    async def _constitutional_review(
        self,
        incident: Incident,
        repair: RepairSpec,
    ) -> ValidationResult:
        """Submit repair to Equor as an Intent for constitutional review."""
        from ecodiaos.primitives.intent import (
            Action,
            ActionSequence,
            DecisionTrace,
            GoalDescriptor,
            Intent,
        )
        from ecodiaos.primitives.common import SystemID

        intent = Intent(
            goal=GoalDescriptor(
                description=f"Immune repair: {repair.reason}",
                target_domain=repair.target_system or incident.source_system,
            ),
            plan=ActionSequence(
                steps=[
                    Action(
                        executor=f"thymos.{repair.action}",
                        parameters={
                            "tier": repair.tier.name,
                            "target": repair.target_system or incident.source_system,
                            "incident_id": incident.id,
                        },
                    )
                ]
            ),
            expected_free_energy=0.0,
            created_by=SystemID.THYMOS,
            priority=0.8 if incident.severity == IncidentSeverity.CRITICAL else 0.6,
            decision_trace=DecisionTrace(
                reasoning=f"Thymos immune repair: {repair.reason}",
                alternatives_considered=[],
            ),
        )

        review = await self._equor.review(intent)

        if review.verdict.value == "approved":
            return ValidationResult(approved=True)
        elif review.verdict.value == "modified":
            return ValidationResult(
                approved=True,
                modifications={"equor_modifications": review.reasoning},
            )
        else:
            return ValidationResult(
                approved=False,
                reason=f"Equor denied repair: {review.reasoning}",
                escalate_to=RepairTier.ESCALATE,
            )

    @property
    def repairs_this_hour(self) -> int:
        now_ts = utc_now().timestamp()
        hour_ago = now_ts - 3600.0
        return sum(1 for ts in self._recent_repairs if ts > hour_ago)

    @property
    def novel_repairs_today(self) -> int:
        now_ts = utc_now().timestamp()
        day_ago = now_ts - 86400.0
        return sum(1 for ts in self._recent_novel if ts > day_ago)
