"""
EcodiaOS — Thymos Sentinel Layer (Detection)

Sentinels are the sensory organs of the immune system. They instrument
every system boundary to capture failures, anomalies, contract violations,
and degradation trends.

Five sentinel classes:
  1. ExceptionSentinel  — unhandled exceptions with full context
  2. ContractSentinel   — inter-system SLA violations
  3. FeedbackLoopSentinel — severed cognitive feedback loops
  4. DriftSentinel      — statistical process control for gradual degradation
  5. CognitiveStallSentinel — workspace cycle producing nothing
"""

from __future__ import annotations

import hashlib
import math
import traceback
from collections import deque
from typing import Any

import structlog

from ecodiaos.primitives.common import utc_now
from ecodiaos.systems.thymos.types import (
    ContractSLA,
    DriftConfig,
    FeedbackLoop,
    Incident,
    IncidentClass,
    IncidentSeverity,
    StallConfig,
)

logger = structlog.get_logger()


# ─── System Dependency Graph ────────────────────────────────────


# Which systems directly affect the user
_USER_FACING_SYSTEMS = frozenset({"voxis", "alive", "atune"})

# Downstream impact map: if system X fails, these are affected
_DOWNSTREAM: dict[str, list[str]] = {
    "memory": ["atune", "nova", "evo", "voxis", "simula", "federation"],
    "equor": ["nova", "axon", "simula", "federation"],
    "atune": ["nova", "evo", "voxis"],
    "nova": ["axon", "voxis"],
    "voxis": [],
    "axon": [],
    "evo": ["simula"],
    "simula": [],
    "synapse": ["atune", "nova", "evo", "memory"],
    "federation": [],
}

# Total number of cognitive systems
_TOTAL_SYSTEMS = 11


# ─── Exception Sentinel ─────────────────────────────────────────


class ExceptionSentinel:
    """
    Intercepts unhandled exceptions from system methods.
    Creates Incidents with full diagnostic context.

    Does NOT suppress the exception — it propagates normally after capture.
    The goal is observation, not intervention.
    """

    def __init__(self) -> None:
        self._logger = logger.bind(system="thymos", component="exception_sentinel")

    def intercept(
        self,
        system_id: str,
        method_name: str,
        exception: BaseException,
        context: dict[str, Any] | None = None,
    ) -> Incident:
        """Create an Incident from an unhandled exception."""
        fp = self.fingerprint(system_id, method_name, exception)
        affected = _DOWNSTREAM.get(system_id, [])
        blast = len(affected) / _TOTAL_SYSTEMS

        return Incident(
            timestamp=utc_now(),
            incident_class=IncidentClass.CRASH,
            severity=self._assess_severity(system_id, exception),
            fingerprint=fp,
            source_system=system_id,
            error_type=type(exception).__name__,
            error_message=str(exception)[:500],
            stack_trace=traceback.format_exc()[:2000],
            context={
                "method": method_name,
                **(context or {}),
            },
            affected_systems=affected,
            blast_radius=blast,
            user_visible=system_id in _USER_FACING_SYSTEMS,
            constitutional_impact=self._assess_constitutional_impact(system_id),
        )

    def fingerprint(
        self,
        system_id: str,
        method: str,
        exc: BaseException,
    ) -> str:
        """
        Create a stable fingerprint for deduplication.

        Hash of: system_id + exception type + first frame in our code.
        Groups "same bug, different call path" together while
        distinguishing genuinely different errors.
        """
        first_frame = self._extract_first_local_frame(exc)
        raw = f"{system_id}:{type(exc).__name__}:{first_frame}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _extract_first_local_frame(self, exc: BaseException) -> str:
        """Extract the first stack frame from our code (not library)."""
        tb = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
        for frame in reversed(tb):
            if "ecodiaos" in frame.filename:
                return f"{frame.filename}:{frame.lineno}:{frame.name}"
        # Fallback: last frame
        if tb:
            f = tb[-1]
            return f"{f.filename}:{f.lineno}:{f.name}"
        return "unknown"

    def _assess_severity(
        self,
        system_id: str,
        exception: BaseException,
    ) -> IncidentSeverity:
        """Initial severity based on system criticality and exception type."""
        # Critical systems crashing is always HIGH+
        if system_id in ("equor", "memory", "atune", "synapse"):
            return IncidentSeverity.CRITICAL
        if system_id in ("nova", "voxis", "axon"):
            return IncidentSeverity.HIGH
        # Non-critical systems
        if isinstance(exception, (TimeoutError, ConnectionError)):
            return IncidentSeverity.MEDIUM
        return IncidentSeverity.MEDIUM

    def _assess_constitutional_impact(self, system_id: str) -> dict[str, float]:
        """Estimate impact on each drive when a system fails."""
        impacts: dict[str, dict[str, float]] = {
            "equor": {"coherence": 0.9, "care": 0.5, "growth": 0.3, "honesty": 0.9},
            "memory": {"coherence": 0.8, "care": 0.3, "growth": 0.7, "honesty": 0.4},
            "atune": {"coherence": 0.7, "care": 0.4, "growth": 0.5, "honesty": 0.3},
            "nova": {"coherence": 0.6, "care": 0.3, "growth": 0.5, "honesty": 0.2},
            "voxis": {"coherence": 0.2, "care": 0.7, "growth": 0.1, "honesty": 0.6},
            "axon": {"coherence": 0.3, "care": 0.5, "growth": 0.2, "honesty": 0.1},
            "evo": {"coherence": 0.2, "care": 0.1, "growth": 0.8, "honesty": 0.1},
            "simula": {"coherence": 0.1, "care": 0.1, "growth": 0.7, "honesty": 0.1},
            "synapse": {"coherence": 0.8, "care": 0.3, "growth": 0.3, "honesty": 0.2},
        }
        return impacts.get(
            system_id,
            {"coherence": 0.1, "care": 0.1, "growth": 0.1, "honesty": 0.1},
        )


# ─── Contract Sentinel ──────────────────────────────────────────


# Inter-system contract SLAs from the Architecture Spec §IV
DEFAULT_CONTRACT_SLAS: list[ContractSLA] = [
    ContractSLA(source="atune", target="memory", operation="store_percept", max_latency_ms=100),
    ContractSLA(source="memory", target="atune", operation="retrieval", max_latency_ms=200),
    ContractSLA(source="atune", target="all", operation="broadcast", max_latency_ms=50),
    ContractSLA(source="nova", target="equor", operation="review", max_latency_ms=500),
    ContractSLA(source="nova", target="equor", operation="review_critical", max_latency_ms=50),
]


class ContractSentinel:
    """
    Instruments inter-system calls to verify SLA compliance.

    Does NOT add latency to the call itself — measurements are taken
    around the existing call, not inserted into it. The sentinel
    observes the event bus, not the call stack.
    """

    def __init__(self, slas: list[ContractSLA] | None = None) -> None:
        self._slas: dict[tuple[str, str, str], ContractSLA] = {}
        for sla in slas or DEFAULT_CONTRACT_SLAS:
            self._slas[(sla.source, sla.target, sla.operation)] = sla
        self._logger = logger.bind(system="thymos", component="contract_sentinel")

    def check_contract(
        self,
        source: str,
        target: str,
        operation: str,
        latency_ms: float,
    ) -> Incident | None:
        """Check if an inter-system call violated its SLA."""
        sla = self._slas.get((source, target, operation))
        if sla is None:
            return None

        if latency_ms <= sla.max_latency_ms:
            return None

        overshoot = latency_ms / sla.max_latency_ms
        fp = hashlib.sha256(
            f"contract:{source}:{target}:{operation}".encode()
        ).hexdigest()[:16]

        return Incident(
            timestamp=utc_now(),
            incident_class=IncidentClass.CONTRACT_VIOLATION,
            severity=IncidentSeverity.MEDIUM,
            fingerprint=fp,
            source_system=source,
            error_type="ContractViolation",
            error_message=(
                f"Contract violation: {source}→{target}.{operation} "
                f"took {latency_ms:.0f}ms (SLA: {sla.max_latency_ms}ms)"
            ),
            context={
                "expected_ms": sla.max_latency_ms,
                "actual_ms": latency_ms,
                "overshoot_factor": overshoot,
                "target_system": target,
                "operation": operation,
            },
            affected_systems=[source, target],
            blast_radius=2 / _TOTAL_SYSTEMS,
            user_visible=target in _USER_FACING_SYSTEMS,
        )


# ─── Feedback Loop Sentinel ─────────────────────────────────────


# The feedback loops identified in the architecture audit
DEFAULT_FEEDBACK_LOOPS: list[FeedbackLoop] = [
    FeedbackLoop(
        name="top_down_prediction",
        source="nova",
        target="atune",
        signal="belief_state",
        check="atune.has_received_beliefs_in_last_n_cycles(10)",
        description="Nova beliefs → Atune prediction error modeling",
    ),
    FeedbackLoop(
        name="goal_guided_attention",
        source="nova",
        target="atune",
        signal="active_goals",
        check="atune.salience_head_weights_include_goal_component()",
        description="Nova goals → Atune salience weighting",
    ),
    FeedbackLoop(
        name="expression_feedback",
        source="voxis",
        target="atune",
        signal="expression_feedback",
        check="atune.has_received_expression_feedback_in_last_n_cycles(100)",
        description="Voxis expression → Atune learning signal",
    ),
    FeedbackLoop(
        name="evo_head_weights",
        source="evo",
        target="atune",
        signal="head_weight_adjustments",
        check="atune.has_received_evo_adjustments()",
        description="Evo learned weights → Atune meta-attention",
    ),
    FeedbackLoop(
        name="axon_outcome_beliefs",
        source="axon",
        target="nova",
        signal="action_outcomes",
        check="nova.has_received_outcomes_in_last_n_cycles(100)",
        description="Axon action outcomes → Nova belief updates",
    ),
    FeedbackLoop(
        name="memory_salience_decay",
        source="memory",
        target="memory",
        signal="salience_decay",
        check="memory.salience_decay_running()",
        description="Memory salience decay over time",
    ),
    FeedbackLoop(
        name="personality_evolution",
        source="voxis",
        target="evo",
        signal="expression_effectiveness",
        check="evo.has_personality_evidence()",
        description="Voxis expression → Evo personality tuning",
    ),
    FeedbackLoop(
        name="rhythm_modulation",
        source="synapse",
        target="nova",
        signal="rhythm_state",
        check="nova.receives_rhythm_updates()",
        description="Synapse rhythm → Nova decision thresholds",
    ),
    FeedbackLoop(
        name="consolidation_weights",
        source="evo",
        target="atune",
        signal="parameter_adjustments",
        check="atune.has_evo_parameters()",
        description="Evo consolidation → Atune salience head weights",
    ),
    FeedbackLoop(
        name="drive_weight_modulation",
        source="equor",
        target="equor",
        signal="contextual_drive_weights",
        check="equor.drive_weights_modulated()",
        description="Context → dynamic drive weighting",
    ),
    FeedbackLoop(
        name="affect_expression",
        source="atune",
        target="voxis",
        signal="affect_state",
        check="voxis.uses_affect_for_style()",
        description="Atune affect → Voxis expression style",
    ),
    FeedbackLoop(
        name="federation_trust_access",
        source="federation",
        target="federation",
        signal="trust_level_changes",
        check="federation.trust_updates_permissions()",
        description="Trust level changes → knowledge exchange permissions",
    ),
    FeedbackLoop(
        name="simula_version_params",
        source="simula",
        target="synapse",
        signal="config_version",
        check="synapse.uses_current_config_version()",
        description="Simula config changes → system parameter propagation",
    ),
    FeedbackLoop(
        name="coherence_safe_mode",
        source="synapse",
        target="synapse",
        signal="coherence_level",
        check="synapse.coherence_triggers_safe_mode()",
        description="Low coherence → safe mode consideration",
    ),
    FeedbackLoop(
        name="community_schema",
        source="memory",
        target="evo",
        signal="community_detection",
        check="evo.uses_community_structure()",
        description="Neo4j community detection → Evo schema induction",
    ),
]


class FeedbackLoopSentinel:
    """
    Periodically verifies that each feedback loop is actively transmitting.

    Unlike heartbeats (which check "is the system alive?"), this checks
    "is the system CONNECTED?" A system can be alive but disconnected
    from the cognitive cycle — like a nerve that's intact but severed
    from the brain.
    """

    def __init__(self, loops: list[FeedbackLoop] | None = None) -> None:
        self._loops = loops or DEFAULT_FEEDBACK_LOOPS
        # Track which loops have been verified as connected
        self._loop_status: dict[str, bool] = {
            loop.name: False for loop in self._loops
        }
        self._last_check: dict[str, float] = {}  # loop_name → timestamp
        self._logger = logger.bind(system="thymos", component="feedback_loop_sentinel")

    def report_loop_active(self, loop_name: str) -> None:
        """Called when evidence of a loop transmitting is observed."""
        self._loop_status[loop_name] = True
        self._last_check[loop_name] = utc_now().timestamp()

    def check_loops(self, max_staleness_s: float = 30.0) -> list[Incident]:
        """
        Check all loops. Returns incidents for any that aren't transmitting.

        A loop is considered severed if:
        - It has never been observed as active, OR
        - It was last seen active more than max_staleness_s seconds ago
        """
        now = utc_now()
        incidents: list[Incident] = []

        for loop in self._loops:
            connected = self._loop_status.get(loop.name, False)
            last_ts = self._last_check.get(loop.name)

            if connected and last_ts is not None:
                age_s = now.timestamp() - last_ts
                if age_s <= max_staleness_s:
                    continue  # Loop is fresh

            fp = hashlib.sha256(f"loop:{loop.name}".encode()).hexdigest()[:16]
            incidents.append(
                Incident(
                    timestamp=now,
                    incident_class=IncidentClass.LOOP_SEVERANCE,
                    severity=IncidentSeverity.HIGH,
                    fingerprint=fp,
                    source_system=loop.source,
                    error_type="FeedbackLoopSevered",
                    error_message=(
                        f"Feedback loop '{loop.name}' is not transmitting: "
                        f"{loop.description}"
                    ),
                    context={
                        "loop_name": loop.name,
                        "source": loop.source,
                        "target": loop.target,
                        "signal": loop.signal,
                    },
                    affected_systems=[loop.source, loop.target],
                    blast_radius=2 / _TOTAL_SYSTEMS,
                    user_visible=False,
                )
            )

        return incidents

    @property
    def loop_statuses(self) -> dict[str, bool]:
        """Current status of all feedback loops."""
        return dict(self._loop_status)


# ─── Drift Sentinel ─────────────────────────────────────────────


class _RollingBaseline:
    """Exponential moving average + standard deviation tracker."""

    def __init__(self, window: int) -> None:
        self._window = window
        self._values: deque[float] = deque(maxlen=window)
        self._ema: float = 0.0
        self._ema_sq: float = 0.0
        self._alpha: float = 2.0 / (window + 1)
        self._count: int = 0

    @property
    def is_warmed_up(self) -> bool:
        return self._count >= self._window // 4  # 25% of window

    def update(self, value: float) -> None:
        self._values.append(value)
        self._count += 1
        if self._count == 1:
            self._ema = value
            self._ema_sq = value * value
        else:
            self._ema = self._alpha * value + (1 - self._alpha) * self._ema
            self._ema_sq = (
                self._alpha * (value * value)
                + (1 - self._alpha) * self._ema_sq
            )

    @property
    def mean(self) -> float:
        return self._ema

    @property
    def std(self) -> float:
        variance = max(0.0, self._ema_sq - self._ema * self._ema)
        return math.sqrt(variance)

    def z_score(self, value: float) -> float:
        s = self.std
        if s < 1e-9:
            return 0.0
        return (value - self._ema) / s


# Default metrics to monitor for drift
DEFAULT_DRIFT_METRICS: dict[str, DriftConfig] = {
    "synapse.cycle.latency_ms": DriftConfig(window=1000, sigma_threshold=2.5),
    "memory.retrieval.latency_ms": DriftConfig(window=500, sigma_threshold=2.0),
    "atune.salience.processing_ms": DriftConfig(window=500, sigma_threshold=2.0),
    "nova.efe.computation_ms": DriftConfig(window=500, sigma_threshold=2.5),
    "voxis.generation.latency_ms": DriftConfig(window=300, sigma_threshold=2.0),
    "synapse.resources.memory_mb": DriftConfig(
        window=200, sigma_threshold=3.0, direction="above"
    ),
    "evo.self_model.success_rate": DriftConfig(
        window=100, sigma_threshold=2.0, direction="below"
    ),
    "atune.coherence.phi": DriftConfig(
        window=200, sigma_threshold=2.0, direction="below"
    ),
}


class DriftSentinel:
    """
    Statistical process control for system metrics.

    Maintains a rolling baseline (EMA + std dev) for each metric.
    When a metric deviates beyond the control limits, it's flagged.

    This catches:
    - Memory leaks (gradual increase in memory_mb)
    - Latency creep (gradually slower responses)
    - Accuracy decay (prediction errors gradually increasing)
    - Throughput degradation (cycles/second declining)

    Adapts to the organism's actual operating characteristics.
    """

    def __init__(
        self,
        metrics: dict[str, DriftConfig] | None = None,
    ) -> None:
        self._metrics = metrics or DEFAULT_DRIFT_METRICS
        self._baselines: dict[str, _RollingBaseline] = {}
        for name, cfg in self._metrics.items():
            self._baselines[name] = _RollingBaseline(cfg.window)
        self._logger = logger.bind(system="thymos", component="drift_sentinel")

    def record_metric(self, metric_name: str, value: float) -> Incident | None:
        """
        Record a metric value and check for drift.
        Returns an Incident if drift is detected, None otherwise.
        """
        config = self._metrics.get(metric_name)
        if config is None:
            return None

        baseline = self._baselines[metric_name]
        baseline.update(value)

        if not baseline.is_warmed_up:
            return None

        z = baseline.z_score(value)

        is_drift = False
        if config.direction == "above" and z > config.sigma_threshold or config.direction == "below" and z < -config.sigma_threshold or config.direction is None and abs(z) > config.sigma_threshold:
            is_drift = True

        if not is_drift:
            return None

        fp = hashlib.sha256(f"drift:{metric_name}".encode()).hexdigest()[:16]
        system_id = metric_name.split(".")[0] if "." in metric_name else "unknown"

        return Incident(
            timestamp=utc_now(),
            incident_class=IncidentClass.DRIFT,
            severity=IncidentSeverity.MEDIUM,
            fingerprint=fp,
            source_system=system_id,
            error_type="MetricDrift",
            error_message=(
                f"Metric '{metric_name}' drifting: "
                f"value={value:.2f}, baseline={baseline.mean:.2f}, "
                f"z-score={z:.2f} (threshold: ±{config.sigma_threshold})"
            ),
            context={
                "metric_name": metric_name,
                "current_value": value,
                "baseline_mean": baseline.mean,
                "baseline_std": baseline.std,
                "z_score": z,
                "direction": config.direction or "both",
            },
            blast_radius=0.1,
            user_visible=False,
        )

    @property
    def baselines(self) -> dict[str, dict[str, float]]:
        """Current baseline statistics for all monitored metrics."""
        return {
            name: {
                "mean": b.mean,
                "std": b.std,
                "warmed_up": b.is_warmed_up,
                "samples": b._count,
            }
            for name, b in self._baselines.items()
        }


# ─── Cognitive Stall Sentinel ────────────────────────────────────


# Default stall thresholds
DEFAULT_STALL_THRESHOLDS: dict[str, StallConfig] = {
    "broadcast_ack_rate": StallConfig(min_value=0.3, window_cycles=50),
    "nova_intent_rate": StallConfig(min_value=0.01, window_cycles=200),
    "evo_evidence_rate": StallConfig(min_value=0.001, window_cycles=500),
    "atune_percept_rate": StallConfig(min_value=0.1, window_cycles=50),
}


class CognitiveStallSentinel:
    """
    Detects when the cognitive cycle is running but accomplishing nothing.

    The heartbeat is fine. The systems are "healthy." But nothing is
    happening. The organism is not thinking.

    This is the equivalent of a person who is conscious but catatonic.
    """

    def __init__(
        self,
        thresholds: dict[str, StallConfig] | None = None,
    ) -> None:
        # Merge custom thresholds with defaults to ensure all expected keys exist
        self._thresholds = DEFAULT_STALL_THRESHOLDS.copy()
        if thresholds:
            self._thresholds.update(thresholds)

        self._counters: dict[str, deque[float]] = {
            name: deque(maxlen=cfg.window_cycles)
            for name, cfg in self._thresholds.items()
        }
        self._logger = logger.bind(system="thymos", component="stall_sentinel")

    def record_cycle(
        self,
        had_broadcast: bool,
        nova_had_intent: bool,
        evo_had_evidence: bool,
        atune_had_percept: bool,
    ) -> list[Incident]:
        """Record one cognitive cycle's activity and check for stalls."""
        self._counters["broadcast_ack_rate"].append(1.0 if had_broadcast else 0.0)
        self._counters["nova_intent_rate"].append(1.0 if nova_had_intent else 0.0)
        self._counters["evo_evidence_rate"].append(1.0 if evo_had_evidence else 0.0)
        self._counters["atune_percept_rate"].append(1.0 if atune_had_percept else 0.0)

        incidents: list[Incident] = []
        now = utc_now()

        for name, cfg in self._thresholds.items():
            window = self._counters[name]
            if len(window) < cfg.window_cycles:
                continue  # Not enough data yet

            rate = sum(window) / len(window)
            if rate >= cfg.min_value:
                continue  # Above threshold, no stall

            fp = hashlib.sha256(f"stall:{name}".encode()).hexdigest()[:16]
            incidents.append(
                Incident(
                    timestamp=now,
                    incident_class=IncidentClass.COGNITIVE_STALL,
                    severity=IncidentSeverity.HIGH,
                    fingerprint=fp,
                    source_system="synapse",
                    error_type="CognitiveStall",
                    error_message=(
                        f"Cognitive stall: '{name}' rate is {rate:.4f} "
                        f"(minimum: {cfg.min_value}) over {cfg.window_cycles} cycles"
                    ),
                    context={
                        "metric_name": name,
                        "rate": rate,
                        "threshold": cfg.min_value,
                        "window_cycles": cfg.window_cycles,
                    },
                    blast_radius=0.5,
                    user_visible=True,  # Catatonic organism affects users
                )
            )

        return incidents
