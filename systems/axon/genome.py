"""
EcodiaOS -- Axon Genome Extraction & Seeding

Implements GenomeExtractionProtocol for the Axon system, enabling Mitosis
to snapshot and restore Axon's heritable operational state:

  - Executor reliability scores (per-action-type success/failure ratios)
  - Timeout calibration data (per-action-type observed execution durations)
  - Circuit breaker configuration (thresholds, recovery windows)

This gives child instances a warm start: they inherit the parent's learned
knowledge about which executors are reliable, how long operations typically
take, and how aggressive circuit breaker protection should be.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID
from primitives.genome import OrganGenomeSegment
from systems.genome_helpers import build_segment, check_schema_version, verify_segment

if TYPE_CHECKING:
    from systems.axon.service import AxonService

logger = structlog.get_logger()

_SCHEMA_VERSION_AXON = 1
_SYSTEM_ID: SystemID = "axon"


class AxonGenomeExtractor:
    """
    Extracts and seeds Axon's heritable state for Mitosis inheritance.

    Extract: Computes per-executor reliability from recent outcomes and
    aggregate execution counts. Captures circuit breaker configuration
    and timeout calibration derived from observed step durations.

    Seed: Applies inherited reliability scores and circuit breaker
    config to a fresh AxonService instance, giving the child a warm start.
    """

    def __init__(self, service: AxonService) -> None:
        self._service = service
        self._log = logger.bind(system="axon", subsystem="genome")

    # ── Extract ──────────────────────────────────────────────────────

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        """
        Serialise Axon's heritable state into an OrganGenomeSegment.

        Returns an empty segment (version=0) if no executions have occurred,
        since there is nothing meaningful to inherit.
        """
        try:
            total = self._service._total_executions
            if total == 0:
                self._log.debug("axon_genome_extract_empty", reason="no_executions")
                return build_segment(
                    system_id=_SYSTEM_ID,
                    payload={},
                    version=0,
                )

            executor_reliability = self._compute_executor_reliability()
            timeout_calibration = self._compute_timeout_calibration()
            circuit_breaker_config = self._extract_circuit_breaker_config()
            template_snapshot = self._extract_template_snapshot()

            payload: dict[str, Any] = {
                "executor_reliability": executor_reliability,
                "timeout_calibration": timeout_calibration,
                "circuit_breaker_config": circuit_breaker_config,
                "template_snapshot": template_snapshot,
            }

            segment = build_segment(
                system_id=_SYSTEM_ID,
                payload=payload,
                version=_SCHEMA_VERSION_AXON,
            )

            self._log.info(
                "axon_genome_extracted",
                total_executions=total,
                executor_count=len(executor_reliability),
                timeout_entries=len(timeout_calibration),
                template_count=len(template_snapshot.get("templates", [])),
            )

            return segment

        except Exception as exc:
            self._log.error("axon_genome_extract_failed", error=str(exc))
            return build_segment(
                system_id=_SYSTEM_ID,
                payload={},
                version=0,
            )

    # ── Seed ─────────────────────────────────────────────────────────

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        """
        Apply inherited Axon state from a parent's genome segment.

        Verifies payload integrity and schema compatibility before applying.
        Returns True on success, False on any failure.
        """
        try:
            if segment.version == 0:
                self._log.debug("axon_genome_seed_skip", reason="empty_segment")
                return True

            if not verify_segment(segment):
                self._log.error("axon_genome_seed_hash_mismatch")
                return False

            if not check_schema_version(segment):
                self._log.error(
                    "axon_genome_seed_schema_incompatible",
                    schema_version=segment.schema_version,
                )
                return False

            payload = segment.payload

            # ── Apply executor reliability ───────────────────────────
            reliability: dict[str, Any] = payload.get("executor_reliability", {})
            if reliability:
                self._apply_executor_reliability(reliability)

            # ── Apply timeout calibration ────────────────────────────
            timeout_cal: dict[str, Any] = payload.get("timeout_calibration", {})
            if timeout_cal:
                self._apply_timeout_calibration(timeout_cal)

            # ── Apply circuit breaker config ─────────────────────────
            cb_config: dict[str, Any] = payload.get("circuit_breaker_config", {})
            if cb_config:
                self._apply_circuit_breaker_config(cb_config)

            # ── Apply template snapshot ──────────────────────────────
            template_snapshot: dict[str, Any] = payload.get("template_snapshot", {})
            templates_applied = 0
            if template_snapshot:
                templates_applied = self._apply_template_snapshot(template_snapshot)

            self._log.info(
                "axon_genome_seeded",
                reliability_entries=len(reliability),
                timeout_entries=len(timeout_cal),
                cb_configured=bool(cb_config),
                templates_applied=templates_applied,
            )

            return True

        except Exception as exc:
            self._log.error("axon_genome_seed_failed", error=str(exc))
            return False

    # ── Private: Extraction Helpers ──────────────────────────────────

    def _compute_executor_reliability(self) -> dict[str, dict[str, Any]]:
        """
        Compute per-executor reliability from recent outcomes.

        Scans the service's recent_outcomes deque and tallies success/failure
        per action_type. Combines with aggregate counters for a global rate.
        """
        per_executor: dict[str, dict[str, int]] = defaultdict(
            lambda: {"success": 0, "failure": 0, "total": 0}
        )

        for outcome in self._service._recent_outcomes:
            # Each AxonOutcome has step_outcomes with action_type
            for step in outcome.step_outcomes:
                action_type = step.action_type
                per_executor[action_type]["total"] += 1
                if step.result.success:
                    per_executor[action_type]["success"] += 1
                else:
                    per_executor[action_type]["failure"] += 1

        reliability: dict[str, dict[str, Any]] = {}
        for action_type, counts in per_executor.items():
            total = counts["total"]
            success_rate = counts["success"] / total if total > 0 else 0.0
            reliability[action_type] = {
                "success_rate": round(success_rate, 4),
                "total_observed": total,
                "successes": counts["success"],
                "failures": counts["failure"],
            }

        # Include aggregate counters as a summary entry
        reliability["__aggregate__"] = {
            "total_executions": self._service._total_executions,
            "successful_executions": self._service._successful_executions,
            "failed_executions": self._service._failed_executions,
            "success_rate": round(
                self._service._successful_executions / self._service._total_executions,
                4,
            )
            if self._service._total_executions > 0
            else 0.0,
        }

        return reliability

    def _compute_timeout_calibration(self) -> dict[str, dict[str, Any]]:
        """
        Derive timeout calibration from observed step durations in recent outcomes.

        For each executor, computes min/max/mean duration in milliseconds.
        Child instances can use these to set tighter or more generous timeouts.
        """
        durations: dict[str, list[int]] = defaultdict(list)

        for outcome in self._service._recent_outcomes:
            for step in outcome.step_outcomes:
                durations[step.action_type].append(step.duration_ms)

        calibration: dict[str, dict[str, Any]] = {}
        for action_type, times in durations.items():
            if not times:
                continue
            sorted_times = sorted(times)
            count = len(sorted_times)
            mean_ms = sum(sorted_times) / count
            # p95 approximation
            p95_idx = min(int(count * 0.95), count - 1)
            calibration[action_type] = {
                "min_ms": sorted_times[0],
                "max_ms": sorted_times[-1],
                "mean_ms": round(mean_ms, 1),
                "p95_ms": sorted_times[p95_idx],
                "sample_count": count,
            }

        return calibration

    def _extract_circuit_breaker_config(self) -> dict[str, Any]:
        """
        Extract the current circuit breaker configuration and per-executor state.

        Captures the global thresholds plus any executor-specific state that
        indicates known-problematic executors a child should be cautious with.
        """
        cb = self._service._circuit_breaker
        if cb is None:
            return {}

        config: dict[str, Any] = {
            "failure_threshold": cb.failure_threshold,
            "recovery_timeout_s": cb.recovery_timeout_s,
            "half_open_max_calls": cb.half_open_max_calls,
        }

        # Capture per-executor circuit state so children inherit knowledge
        # of which executors have been problematic
        executor_states: dict[str, dict[str, Any]] = {}
        for action_type, state in cb._states.items():
            executor_states[action_type] = {
                "status": state.status.value,
                "consecutive_failures": state.consecutive_failures,
            }

        if executor_states:
            config["executor_states"] = executor_states

        return config

    # ── Private: Seeding Helpers ─────────────────────────────────────

    def _apply_executor_reliability(self, reliability: dict[str, Any]) -> None:
        """
        Warm-start the service's execution counters from inherited reliability.

        Uses the aggregate entry to set the global counters, giving the child
        a sense of the parent's execution history without inflating metrics.
        """
        aggregate = reliability.get("__aggregate__")
        if aggregate is None:
            return

        # Seed global counters from the parent's aggregate.
        # These are informational warm-start values -- they will be
        # incremented naturally as the child executes.
        total = int(aggregate.get("total_executions", 0))
        successes = int(aggregate.get("successful_executions", 0))
        failures = int(aggregate.get("failed_executions", 0))

        # Only seed if the child has not yet executed anything (fresh instance)
        if self._service._total_executions == 0:
            self._service._total_executions = total
            self._service._successful_executions = successes
            self._service._failed_executions = failures

            self._log.debug(
                "axon_genome_reliability_applied",
                total=total,
                successes=successes,
                failures=failures,
            )

    def _apply_timeout_calibration(self, timeout_cal: dict[str, Any]) -> None:
        """
        Store inherited timeout calibration on the service's introspector.

        The introspector uses this data to recommend timeouts for executors
        the child has not yet observed directly.
        """
        introspector = self._service._introspector
        if introspector is None:
            self._log.debug("axon_genome_timeout_skip", reason="no_introspector")
            return

        # If the introspector supports seeding calibration, apply it.
        # Otherwise store as a simple attribute for later consumption.
        if hasattr(introspector, "seed_timeout_calibration"):
            introspector.seed_timeout_calibration(timeout_cal)
            self._log.debug(
                "axon_genome_timeout_applied",
                entries=len(timeout_cal),
            )
        elif hasattr(introspector, "_timeout_calibration"):
            introspector._timeout_calibration = timeout_cal
            self._log.debug(
                "axon_genome_timeout_stored",
                entries=len(timeout_cal),
            )
        else:
            self._log.debug(
                "axon_genome_timeout_skip",
                reason="introspector_no_calibration_interface",
            )

    def _apply_circuit_breaker_config(self, cb_config: dict[str, Any]) -> None:
        """
        Apply inherited circuit breaker thresholds and known-problematic executor state.

        Updates the global thresholds (failure_threshold, recovery_timeout_s,
        half_open_max_calls) and pre-loads executor states so the child is
        cautious with executors the parent found unreliable.
        """
        cb = self._service._circuit_breaker
        if cb is None:
            self._log.debug("axon_genome_cb_skip", reason="no_circuit_breaker")
            return

        # Apply global thresholds if present
        if "failure_threshold" in cb_config:
            cb.failure_threshold = int(cb_config["failure_threshold"])
        if "recovery_timeout_s" in cb_config:
            cb.recovery_timeout_s = int(cb_config["recovery_timeout_s"])
        if "half_open_max_calls" in cb_config:
            cb.half_open_max_calls = int(cb_config["half_open_max_calls"])

        # Pre-load known-problematic executor states.
        # Only seed executors that had consecutive failures -- don't seed
        # OPEN state directly since the child should observe fresh failures.
        executor_states: dict[str, Any] = cb_config.get("executor_states", {})
        seeded_count = 0
        for action_type, state_data in executor_states.items():
            consecutive_failures = int(state_data.get("consecutive_failures", 0))
            if consecutive_failures > 0:
                from systems.axon.types import CircuitState, CircuitStatus

                # Start the child's circuit in CLOSED but with inherited
                # failure count, so it trips faster if the problem persists
                cb._states[action_type] = CircuitState(
                    status=CircuitStatus.CLOSED,
                    consecutive_failures=min(
                        consecutive_failures, cb.failure_threshold - 1
                    ),
                    tripped_at=0.0,
                    half_open_calls=0,
                )
                seeded_count += 1

        self._log.debug(
            "axon_genome_cb_applied",
            failure_threshold=cb.failure_threshold,
            recovery_timeout_s=cb.recovery_timeout_s,
            executor_states_seeded=seeded_count,
        )

    def _extract_template_snapshot(self) -> dict[str, Any]:
        """
        Capture the top-10 executor patterns by success_rate for template inheritance.

        Each entry includes action_pattern, success_rate, expected_cost_mean,
        expected_cost_variance, and template_confidence so child instances can
        warm-start their execution strategy without any prior execution history.
        """
        per_executor: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"success": 0, "failure": 0, "total": 0, "total_ms": 0}
        )

        for outcome in self._service._recent_outcomes:
            for step in outcome.step_outcomes:
                action_type = step.action_type
                per_executor[action_type]["total"] += 1
                per_executor[action_type]["total_ms"] += step.duration_ms
                if step.result.success:
                    per_executor[action_type]["success"] += 1
                else:
                    per_executor[action_type]["failure"] += 1

        raw_templates = []
        for action_type, counts in per_executor.items():
            total = counts["total"]
            if total < 5:
                continue  # Skip executors with too few observations
            success_rate = counts["success"] / total if total > 0 else 0.0
            mean_ms = counts["total_ms"] / total if total > 0 else 0.0
            raw_templates.append({
                "action_pattern": action_type,
                "success_rate": round(success_rate, 4),
                "expected_cost_mean": round(mean_ms, 2),
                "expected_cost_variance": 0.0,  # variance not tracked at this layer
                "cached_approvals": [],
                "template_confidence": round(max(0.5, success_rate), 4),
            })

        raw_templates.sort(key=lambda t: t["success_rate"], reverse=True)
        top_templates = raw_templates[:10]

        return {
            "templates": top_templates,
            "circuit_breaker_thresholds": {
                "__default__": self._service._circuit_breaker.failure_threshold
                if self._service._circuit_breaker else 5,
            },
        }

    def _apply_template_snapshot(self, template_snapshot: dict[str, Any]) -> int:
        """
        Apply inherited template snapshot to the child AxonService.

        Seeds the introspector with inherited execution priors so the child
        starts with realistic success rate estimates for known action patterns.
        Returns the count of templates applied.
        """
        templates: list[dict[str, Any]] = template_snapshot.get("templates", [])
        if not templates:
            return 0

        introspector = self._service._introspector
        applied = 0

        for template_data in templates:
            action_type = str(template_data.get("action_pattern", ""))
            success_rate = float(template_data.get("success_rate", 0.0))
            confidence = float(template_data.get("template_confidence", max(0.5, success_rate)))

            if not action_type:
                continue

            if introspector is not None and hasattr(introspector, "seed_inherited_template"):
                introspector.seed_inherited_template(
                    action_type=action_type,
                    success_rate=success_rate,
                    confidence=confidence,
                    inherited_from_parent=True,
                )
            applied += 1

        # Apply circuit breaker thresholds from template snapshot
        cb_thresholds: dict[str, Any] = template_snapshot.get("circuit_breaker_thresholds", {})
        if cb_thresholds and self._service._circuit_breaker is not None:
            default_threshold = int(cb_thresholds.get(
                "__default__", self._service._circuit_breaker.failure_threshold
            ))
            if default_threshold != self._service._circuit_breaker.failure_threshold:
                self._service._circuit_breaker.failure_threshold = default_threshold
                self._log.debug(
                    "axon_genome_template_cb_threshold",
                    threshold=default_threshold,
                )

        self._log.info(
            "axon_genome_templates_applied",
            templates_applied=applied,
        )
        return applied
