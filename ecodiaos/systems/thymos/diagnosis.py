"""
EcodiaOS — Thymos Diagnostic Layer (Root Cause Analysis)

Diagnosis is where Thymos becomes genuinely intelligent. It doesn't just
look at the failing system — it reasons about causality across the organism.

Three diagnostic strategies:
  1. CausalAnalyzer       — trace error causality through the dependency graph
  2. TemporalCorrelator   — what changed in the window before the incident?
  3. DiagnosticEngine      — LLM-backed hypothesis generation and testing
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import structlog

from ecodiaos.primitives.common import utc_now
from ecodiaos.systems.thymos.types import (
    Antibody,
    CausalChain,
    Diagnosis,
    DiagnosticEvidence,
    DiagnosticHypothesis,
    DiagnosticTestResult,
    Incident,
    RepairTier,
    TemporalCorrelation,
)

logger = structlog.get_logger()


# ─── System Dependency Graph ────────────────────────────────────


# Upstream dependencies: if system X depends on Y, a failure in Y
# may be the root cause of a failure in X.
_UPSTREAM_DEPS: dict[str, list[str]] = {
    "nova": ["memory", "equor", "atune"],
    "voxis": ["memory", "nova", "atune"],
    "axon": ["nova", "equor"],
    "evo": ["memory", "atune"],
    "simula": ["evo", "memory", "equor"],
    "atune": ["memory", "synapse"],
    "federation": ["memory", "equor"],
    # Core systems have no upstream deps within the cognitive layer
    "memory": [],
    "equor": [],
    "synapse": [],
}


# ─── Causal Analyzer ────────────────────────────────────────────


class CausalAnalyzer:
    """
    Traces error causality through the system dependency graph.

    When Nova fails, traverse upstream:
    1. Is Memory responding within SLA?
    2. Is Atune sending broadcasts?
    3. Is Equor processing reviews?

    The first unhealthy upstream system is the likely root cause.
    If all upstream systems are healthy, the failure is local.
    """

    def __init__(self, health_provider: Any = None) -> None:
        """
        Args:
            health_provider: Object with get_record(system_id) -> health dict.
                             Typically the Synapse HealthMonitor.
        """
        self._health = health_provider
        self._recent_incidents: dict[str, list[Incident]] = {}  # system_id → recent
        self._logger = logger.bind(system="thymos", component="causal_analyzer")

    def record_incident(self, incident: Incident) -> None:
        """Record an incident for cross-system correlation."""
        system = incident.source_system
        if system not in self._recent_incidents:
            self._recent_incidents[system] = []
        self._recent_incidents[system].append(incident)
        # Keep only last 50 per system
        if len(self._recent_incidents[system]) > 50:
            self._recent_incidents[system] = self._recent_incidents[system][-50:]

    async def trace_root_cause(self, incident: Incident) -> CausalChain:
        """
        Trace the root cause of an incident through upstream dependencies.
        """
        upstream = _UPSTREAM_DEPS.get(incident.source_system, [])

        if not upstream:
            return CausalChain(
                root_system=incident.source_system,
                chain=[incident.source_system],
                confidence=0.6,
                reasoning=(
                    f"No upstream dependencies — failure is local to "
                    f"{incident.source_system}"
                ),
            )

        # Check each upstream system for recent issues
        unhealthy_upstream: list[str] = []

        for system_id in upstream:
            is_unhealthy = False

            # Check health monitor if available
            if self._health is not None:
                try:
                    record = self._health.get_record(system_id)
                    if record and record.status not in ("healthy",):
                        is_unhealthy = True
                except Exception:
                    pass

            # Check recent incidents from that system
            recent = self._recent_incidents.get(system_id, [])
            now = utc_now()
            recent_window = [
                i for i in recent
                if (now - i.timestamp).total_seconds() < 60.0
            ]
            if recent_window:
                is_unhealthy = True

            if is_unhealthy:
                unhealthy_upstream.append(system_id)

        if unhealthy_upstream:
            root = unhealthy_upstream[0]
            chain = [root, incident.source_system]

            # Recurse one level deeper — is the upstream's upstream also failing?
            deeper_upstream = _UPSTREAM_DEPS.get(root, [])
            for deeper in deeper_upstream:
                deeper_recent = self._recent_incidents.get(deeper, [])
                now = utc_now()
                if any(
                    (now - i.timestamp).total_seconds() < 60.0
                    for i in deeper_recent
                ):
                    chain.insert(0, deeper)
                    root = deeper
                    break

            return CausalChain(
                root_system=root,
                chain=chain,
                confidence=0.8,
                reasoning=(
                    f"{incident.source_system} failure likely caused by "
                    f"upstream {root} issue"
                ),
            )

        return CausalChain(
            root_system=incident.source_system,
            chain=[incident.source_system],
            confidence=0.6,
            reasoning=(
                f"All upstream systems healthy — failure is local to "
                f"{incident.source_system}"
            ),
        )

    def find_common_upstream(self, incidents: list[Incident]) -> str | None:
        """
        Given multiple concurrent incidents, find the common upstream cause.

        Used during cytokine storms to focus diagnostic effort.
        """
        if not incidents:
            return None

        # Count upstream mentions
        upstream_counts: dict[str, int] = {}
        for inc in incidents:
            for up in _UPSTREAM_DEPS.get(inc.source_system, []):
                upstream_counts[up] = upstream_counts.get(up, 0) + 1
            # Also count the source itself
            upstream_counts[inc.source_system] = (
                upstream_counts.get(inc.source_system, 0) + 1
            )

        if not upstream_counts:
            return None

        # Most common upstream system
        return max(upstream_counts, key=upstream_counts.get)  # type: ignore[arg-type]


# ─── Temporal Correlator ────────────────────────────────────────


class TemporalCorrelator:
    """
    Queries what changed in the window before the incident.

    "What happened in the 30 seconds before Nova crashed?"
    - Memory latency spiked to 450ms (SLA: 200ms)
    - Synapse resource allocation shifted (Evo consolidation started)
    - A new code deployment was applied by Simula

    This surfaces the TRUE root cause when the proximate cause is misleading.
    """

    def __init__(self) -> None:
        # Ring buffer of system events for temporal queries
        self._events: list[dict[str, Any]] = []
        self._max_events = 1000
        self._logger = logger.bind(system="thymos", component="temporal_correlator")

    def record_event(
        self,
        event_type: str,
        details: str,
        system_id: str = "unknown",
    ) -> None:
        """Record a system event for later correlation."""
        self._events.append({
            "event_type": event_type,
            "details": details,
            "system_id": system_id,
            "timestamp": utc_now(),
        })
        # Trim to max
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

    def record_metric_anomaly(
        self,
        metric_name: str,
        value: float,
        baseline: float,
        z_score: float,
    ) -> None:
        """Record a metric anomaly for later correlation."""
        self._events.append({
            "event_type": "metric_anomaly",
            "details": f"{metric_name}={value:.2f} (baseline={baseline:.2f}, z={z_score:.2f})",
            "system_id": metric_name.split(".")[0] if "." in metric_name else "unknown",
            "timestamp": utc_now(),
            "metric_name": metric_name,
            "value": value,
            "z_score": z_score,
        })
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

    def correlate(
        self,
        incident: Incident,
        window_s: float = 30.0,
    ) -> list[TemporalCorrelation]:
        """
        Find events that occurred in the window before the incident.
        """
        start = incident.timestamp - timedelta(seconds=window_s)
        end = incident.timestamp

        correlations: list[TemporalCorrelation] = []

        for event in self._events:
            ts = event["timestamp"]
            if start <= ts <= end:
                delta_ms = int((incident.timestamp - ts).total_seconds() * 1000)
                correlations.append(
                    TemporalCorrelation(
                        type=event["event_type"],
                        timestamp=ts,
                        description=event["details"],
                        time_delta_ms=delta_ms,
                    )
                )

        return sorted(correlations, key=lambda c: c.time_delta_ms)


# ─── Diagnostic Engine ──────────────────────────────────────────


class DiagnosticEngine:
    """
    For complex or novel errors, generates and tests diagnostic hypotheses
    using LLM-backed reasoning focused on error causality.

    If an antibody matches, skip this step entirely.
    """

    def __init__(self, llm_client: Any = None) -> None:
        self._llm = llm_client
        self._logger = logger.bind(system="thymos", component="diagnostic_engine")

    async def diagnose(
        self,
        incident: Incident,
        causal_chain: CausalChain,
        correlations: list[TemporalCorrelation],
        antibody_match: Antibody | None = None,
    ) -> Diagnosis:
        """
        Generate and evaluate diagnostic hypotheses for an incident.

        If an antibody matches with high effectiveness, skip diagnosis.
        """
        # Fast path: known fix
        if antibody_match is not None and antibody_match.effectiveness > 0.8:
            return Diagnosis(
                root_cause=antibody_match.root_cause_description,
                confidence=antibody_match.effectiveness,
                repair_tier=RepairTier.KNOWN_FIX,
                antibody_id=antibody_match.id,
                reasoning="Known pattern — antibody match with high effectiveness",
            )

        # Gather evidence
        evidence = DiagnosticEvidence(
            incident=incident,
            causal_chain=causal_chain,
            temporal_correlations=correlations,
        )

        # Generate hypotheses
        hypotheses = await self._generate_hypotheses(evidence)
        if not hypotheses:
            return Diagnosis(
                root_cause=f"Unknown error in {incident.source_system}: {incident.error_type}",
                confidence=0.3,
                repair_tier=RepairTier.RESTART,
                reasoning="Could not generate diagnostic hypotheses",
            )

        # Test each hypothesis
        tested: list[tuple[DiagnosticHypothesis, DiagnosticTestResult]] = []
        for hypothesis in hypotheses:
            result = await self._run_diagnostic_test(hypothesis, incident)
            tested.append((hypothesis, result))

        # Select best hypothesis
        best_hyp, best_result = max(tested, key=lambda t: t[1].confidence)

        return Diagnosis(
            root_cause=best_hyp.statement,
            confidence=best_result.confidence,
            repair_tier=best_hyp.suggested_repair_tier,
            all_hypotheses=hypotheses,
            test_results=[t[1] for t in tested],
            reasoning=best_result.reasoning,
        )

    async def _generate_hypotheses(
        self,
        evidence: DiagnosticEvidence,
    ) -> list[DiagnosticHypothesis]:
        """
        Generate diagnostic hypotheses — LLM-backed if available,
        rule-based fallback otherwise.
        """
        # Try LLM-backed hypothesis generation
        if self._llm is not None:
            try:
                return await self._generate_hypotheses_llm(evidence)
            except Exception as exc:
                self._logger.warning(
                    "llm_diagnosis_failed",
                    error=str(exc),
                    fallback="rule_based",
                )

        # Rule-based fallback
        return self._generate_hypotheses_rules(evidence)

    async def _generate_hypotheses_llm(
        self,
        evidence: DiagnosticEvidence,
    ) -> list[DiagnosticHypothesis]:
        """Use LLM to generate diagnostic hypotheses."""
        incident = evidence.incident
        correlations_text = "\n".join(
            f"  - [{c.time_delta_ms}ms before] {c.description}"
            for c in evidence.temporal_correlations[:10]
        ) or "  (none recorded)"

        prompt = f"""You are the diagnostic engine of a living digital organism.

INCIDENT:
  System: {incident.source_system}
  Class: {incident.incident_class.value}
  Error: {incident.error_type}: {incident.error_message}
  Stack trace: {(incident.stack_trace or 'N/A')[:500]}

CAUSAL CHAIN: {' → '.join(evidence.causal_chain.chain)}
Confidence: {evidence.causal_chain.confidence:.2f}
Reasoning: {evidence.causal_chain.reasoning}

TEMPORAL CORRELATIONS (what changed before the incident):
{correlations_text}

Generate exactly 3 diagnostic hypotheses. For each:
- statement: concise root cause claim
- diagnostic_test: a specific check name from: check_memory_pressure, check_upstream_latency, check_event_bus_backlog, check_belief_staleness, check_workspace_contention, check_consolidation_active, check_llm_availability, check_resource_exhaustion
- suggested_repair_tier: "parameter" | "restart" | "known_fix" | "novel_fix" | "escalate"
- confidence_prior: 0.0 to 1.0

Rules:
- Prefer simpler explanations (Occam's razor)
- Consider upstream causes, not just local symptoms
- At least one hypothesis should consider a non-obvious cause

Respond in JSON array format:
[{{"statement": "...", "diagnostic_test": "...", "suggested_repair_tier": "...", "confidence_prior": 0.0}}]"""

        response = await self._llm.generate(
            prompt=prompt,
            system="You are a fault diagnosis engine. Respond only in valid JSON.",
            max_tokens=1000,
            temperature=0.3,
        )

        # Parse LLM response
        content = response.content if hasattr(response, "content") else str(response)
        # Extract JSON from response
        start = content.find("[")
        end = content.rfind("]") + 1
        if start < 0 or end <= start:
            return self._generate_hypotheses_rules(evidence)

        raw = json.loads(content[start:end])
        tier_map = {
            "parameter": RepairTier.PARAMETER,
            "restart": RepairTier.RESTART,
            "known_fix": RepairTier.KNOWN_FIX,
            "novel_fix": RepairTier.NOVEL_FIX,
            "escalate": RepairTier.ESCALATE,
        }

        hypotheses: list[DiagnosticHypothesis] = []
        for item in raw[:3]:
            tier_str = item.get("suggested_repair_tier", "restart")
            hypotheses.append(
                DiagnosticHypothesis(
                    statement=item.get("statement", "Unknown"),
                    diagnostic_test=item.get("diagnostic_test", "check_upstream_latency"),
                    suggested_repair_tier=tier_map.get(tier_str, RepairTier.RESTART),
                    confidence_prior=min(1.0, max(0.0, item.get("confidence_prior", 0.5))),
                )
            )

        return hypotheses

    def _generate_hypotheses_rules(
        self,
        evidence: DiagnosticEvidence,
    ) -> list[DiagnosticHypothesis]:
        """Rule-based hypothesis generation — always available."""
        incident = evidence.incident
        hypotheses: list[DiagnosticHypothesis] = []

        # Hypothesis 1: Based on causal chain
        if len(evidence.causal_chain.chain) > 1:
            root = evidence.causal_chain.chain[0]
            hypotheses.append(
                DiagnosticHypothesis(
                    statement=f"Upstream failure in {root} cascading to {incident.source_system}",
                    diagnostic_test="check_upstream_latency",
                    diagnostic_test_params={"system_id": root},
                    suggested_repair_tier=RepairTier.RESTART,
                    confidence_prior=evidence.causal_chain.confidence,
                )
            )

        # Hypothesis 2: Based on incident class
        class_hypotheses: dict[str, tuple[str, str, RepairTier]] = {
            "crash": (
                "Unhandled edge case in {system} — possibly a null/None reference",
                "check_upstream_latency",
                RepairTier.NOVEL_FIX,
            ),
            "degradation": (
                "Resource pressure causing latency increase in {system}",
                "check_memory_pressure",
                RepairTier.PARAMETER,
            ),
            "contract_violation": (
                "Temporary overload causing SLA breach",
                "check_upstream_latency",
                RepairTier.PARAMETER,
            ),
            "resource_exhaustion": (
                "Memory leak or unbounded growth in {system}",
                "check_resource_exhaustion",
                RepairTier.RESTART,
            ),
            "cognitive_stall": (
                "Feedback loop disconnection preventing cognitive processing",
                "check_event_bus_backlog",
                RepairTier.RESTART,
            ),
        }

        class_entry = class_hypotheses.get(incident.incident_class.value)
        if class_entry:
            stmt, test, tier = class_entry
            hypotheses.append(
                DiagnosticHypothesis(
                    statement=stmt.format(system=incident.source_system),
                    diagnostic_test=test,
                    suggested_repair_tier=tier,
                    confidence_prior=0.5,
                )
            )

        # Hypothesis 3: Temporal correlation-based
        if evidence.temporal_correlations:
            most_recent = evidence.temporal_correlations[0]
            hypotheses.append(
                DiagnosticHypothesis(
                    statement=f"Triggered by recent event: {most_recent.description}",
                    diagnostic_test="check_upstream_latency",
                    suggested_repair_tier=RepairTier.PARAMETER,
                    confidence_prior=0.4,
                )
            )

        # Ensure at least one hypothesis
        if not hypotheses:
            hypotheses.append(
                DiagnosticHypothesis(
                    statement=f"Unknown error in {incident.source_system}",
                    diagnostic_test="check_upstream_latency",
                    suggested_repair_tier=RepairTier.RESTART,
                    confidence_prior=0.3,
                )
            )

        return hypotheses[:3]

    async def _run_diagnostic_test(
        self,
        hypothesis: DiagnosticHypothesis,
        incident: Incident,
    ) -> DiagnosticTestResult:
        """
        Run a diagnostic test for a hypothesis.

        In the initial implementation, tests are heuristic-based
        rather than executing actual system queries (which requires
        deeper system introspection APIs).
        """
        # Map hypothesis confidence to test result
        # The diagnostic test adjusts confidence based on available evidence
        test_name = hypothesis.diagnostic_test

        # Adjust confidence based on incident context
        confidence = hypothesis.confidence_prior

        # If the incident has a stack trace and the hypothesis mentions code,
        # boost confidence
        if incident.stack_trace and "edge case" in hypothesis.statement.lower():
            confidence = min(1.0, confidence + 0.1)

        # If recurrence is high, structural hypotheses are more likely
        if incident.occurrence_count > 10:
            if hypothesis.suggested_repair_tier in (
                RepairTier.NOVEL_FIX,
                RepairTier.RESTART,
            ):
                confidence = min(1.0, confidence + 0.15)

        # If causal chain is long, upstream hypotheses are more likely
        if "upstream" in hypothesis.statement.lower():
            confidence = min(1.0, confidence + 0.1)

        return DiagnosticTestResult(
            test_name=test_name,
            passed=confidence > 0.5,
            confidence=confidence,
            reasoning=(
                f"Hypothesis '{hypothesis.statement[:60]}' "
                f"evaluated with confidence {confidence:.2f}"
            ),
        )
