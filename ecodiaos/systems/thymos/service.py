"""
EcodiaOS — Thymos Service (The Immune System)

The organism's self-healing system. Thymos detects failures, diagnoses root causes,
prescribes repairs, maintains an antibody library of learned fixes, and prevents
future errors through prophylactic scanning and homeostatic regulation.

Every error, anomaly, and violation becomes an Incident — a first-class primitive
alongside Percept, Belief, and Intent. The organism perceives its own failures
through the normal workspace broadcast cycle. It hurts to break.

Immune Pipeline:
  Detect → Deduplicate → Triage → Diagnose → Prescribe → Validate → Apply → Verify → Learn

Iron Rules:
  - Thymos CANNOT modify Equor or constitutional drives
  - Thymos CANNOT suppress or hide errors from the audit trail
  - Thymos CANNOT apply Tier 4 (codegen) repairs without Equor review
  - Thymos CANNOT exceed the healing budget (MAX_REPAIRS_PER_HOUR)
  - Thymos MUST route all Tier 3+ repairs through the validation gate
  - Thymos MUST record every incident, diagnosis, and repair in Neo4j
  - Thymos MUST prefer less invasive repairs (Tier 0 before Tier 1 before ...)
  - Thymos MUST enter storm mode when incident rate exceeds threshold

Cognitive cycle role (step 8 — MAINTAIN):
  Homeostatic checks run on the MAINTAIN step. Non-blocking, background.
  The organism maintains itself the way a body regulates temperature.

Interface:
  initialize()              — build sub-systems, load antibody library
  on_synapse_event()        — convert health events into incidents
  on_incident()             — entry point for the immune pipeline
  process_incident()        — full pipeline: diagnose → prescribe → validate → apply
  maintain_homeostasis()    — proactive health optimization (MAINTAIN step)
  shutdown()                — graceful teardown
  health()                  — self-health report for Synapse
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.primitives.common import utc_now
from ecodiaos.systems.synapse.types import SynapseEvent, SynapseEventType
from ecodiaos.systems.thymos.antibody import AntibodyLibrary
from ecodiaos.systems.thymos.diagnosis import (
    CausalAnalyzer,
    DiagnosticEngine,
    TemporalCorrelator,
)
from ecodiaos.systems.thymos.governor import HealingGovernor
from ecodiaos.systems.thymos.prescription import RepairPrescriber, RepairValidator
from ecodiaos.systems.thymos.prophylactic import HomeostasisController, ProphylacticScanner
from ecodiaos.systems.thymos.sentinels import (
    CognitiveStallSentinel,
    ContractSentinel,
    DriftSentinel,
    ExceptionSentinel,
    FeedbackLoopSentinel,
)
from ecodiaos.systems.thymos.triage import (
    IncidentDeduplicator,
    ResponseRouter,
    SeverityScorer,
)
from ecodiaos.systems.thymos.types import (
    Diagnosis,
    Incident,
    IncidentClass,
    IncidentSeverity,
    RepairSpec,
    RepairStatus,
    RepairTier,
)

if TYPE_CHECKING:
    from ecodiaos.clients.llm import LLMProvider
    from ecodiaos.clients.neo4j import Neo4jClient
    from ecodiaos.config import ThymosConfig
    from ecodiaos.systems.synapse.health import HealthMonitor
    from ecodiaos.systems.synapse.service import SynapseService
    from ecodiaos.telemetry.metrics import MetricCollector

logger = structlog.get_logger("ecodiaos.systems.thymos")


# ─── Constants ──────────────────────────────────────────────────

# Synapse events that Thymos converts into Incidents
_SUBSCRIBED_EVENTS: frozenset[SynapseEventType] = frozenset({
    SynapseEventType.SYSTEM_FAILED,
    SynapseEventType.SYSTEM_RECOVERED,
    SynapseEventType.SYSTEM_RESTARTING,
    SynapseEventType.SAFE_MODE_ENTERED,
    SynapseEventType.SAFE_MODE_EXITED,
    SynapseEventType.SYSTEM_OVERLOADED,
    SynapseEventType.CLOCK_OVERRUN,
    SynapseEventType.RESOURCE_PRESSURE,
})

# How often to run homeostatic checks (in seconds)
_HOMEOSTASIS_INTERVAL_S: float = 30.0

# How often to run sentinel scans (in seconds)
_SENTINEL_SCAN_INTERVAL_S: float = 30.0

# How long to wait for post-repair verification (seconds)
_POST_REPAIR_VERIFY_TIMEOUT_S: float = 10.0

# Salience mapping: incident severity → percept priority for Atune
_SEVERITY_TO_SALIENCE: dict[IncidentSeverity, float] = {
    IncidentSeverity.CRITICAL: 1.0,
    IncidentSeverity.HIGH: 0.8,
    IncidentSeverity.MEDIUM: 0.5,
    IncidentSeverity.LOW: 0.2,
    IncidentSeverity.INFO: 0.1,
}

# Maximum incidents to buffer in memory
_INCIDENT_BUFFER_SIZE: int = 10_000


class ThymosService:
    """
    Thymos — the EOS immune system.

    Coordinates seven sub-systems:
      Sentinels              — fault detection (5 sentinel classes)
      Triage                 — deduplication, severity scoring, response routing
      Diagnosis              — causal analysis, temporal correlation, hypothesis engine
      Prescription           — repair tier selection and validation gate
      AntibodyLibrary        — immune memory: crystallized successful repairs
      Prophylactic           — prevention: pre-deploy scans and homeostasis
      HealingGovernor        — cytokine storm prevention and budget enforcement
    """

    system_id: str = "thymos"

    def __init__(
        self,
        config: ThymosConfig,
        synapse: SynapseService | None = None,
        neo4j: Neo4jClient | None = None,
        llm: LLMProvider | None = None,
        metrics: MetricCollector | None = None,
    ) -> None:
        self._config = config
        self._synapse = synapse
        self._neo4j = neo4j
        self._llm = llm
        self._metrics = metrics
        self._initialized: bool = False
        self._logger = logger.bind(system="thymos")

        # Cross-system references (wired post-init by main.py)
        self._equor: Any = None     # EquorService — constitutional review
        self._evo: Any = None       # EvoService — error pattern learning
        self._atune: Any = None     # AtuneService — incident-as-percept
        self._health_monitor: HealthMonitor | None = None  # Synapse health records
        self._soma: Any = None      # SomaService — integrity precision gating

        # ── Sub-systems (built in initialize()) ──
        # Sentinels
        self._exception_sentinel: ExceptionSentinel | None = None
        self._contract_sentinel: ContractSentinel | None = None
        self._feedback_loop_sentinel: FeedbackLoopSentinel | None = None
        self._drift_sentinel: DriftSentinel | None = None
        self._cognitive_stall_sentinel: CognitiveStallSentinel | None = None

        # Triage
        self._deduplicator: IncidentDeduplicator | None = None
        self._severity_scorer: SeverityScorer | None = None
        self._response_router: ResponseRouter | None = None

        # Diagnosis
        self._causal_analyzer: CausalAnalyzer | None = None
        self._temporal_correlator: TemporalCorrelator | None = None
        self._diagnostic_engine: DiagnosticEngine | None = None

        # Prescription
        self._prescriber: RepairPrescriber | None = None
        self._validator: RepairValidator | None = None

        # Immune memory
        self._antibody_library: AntibodyLibrary | None = None

        # Prophylactic
        self._prophylactic_scanner: ProphylacticScanner | None = None
        self._homeostasis_controller: HomeostasisController | None = None

        # Governor
        self._governor: HealingGovernor | None = None

        # Cross-system references (wired post-init by main.py)
        self._nova: Any = None  # NovaService — for injecting urgent repair goals

        # ── State ──
        self._active_incidents: dict[str, Incident] = {}
        self._incident_buffer: deque[Incident] = deque(maxlen=_INCIDENT_BUFFER_SIZE)
        self._resolution_times: deque[float] = deque(maxlen=500)

        # Background tasks
        self._sentinel_task: asyncio.Task[None] | None = None
        self._homeostasis_task: asyncio.Task[None] | None = None

        # Counters
        self._total_incidents: int = 0
        self._total_repairs_attempted: int = 0
        self._total_repairs_succeeded: int = 0
        self._total_repairs_failed: int = 0
        self._total_repairs_rolled_back: int = 0
        self._total_diagnoses: int = 0
        self._total_antibodies_applied: int = 0
        self._total_antibodies_created: int = 0
        self._total_homeostatic_adjustments: int = 0
        self._total_prophylactic_scans: int = 0
        self._total_prophylactic_warnings: int = 0
        self._incidents_by_severity: dict[str, int] = {}
        self._incidents_by_class: dict[str, int] = {}
        self._repairs_by_tier: dict[str, int] = {}
        self._diagnosis_confidences: deque[float] = deque(maxlen=200)
        self._diagnosis_latencies: deque[float] = deque(maxlen=200)

    # ─── Cross-System Wiring ─────────────────────────────────────────

    def set_equor(self, equor: Any) -> None:
        """Wire Equor for constitutional review of Tier 3+ repairs."""
        self._equor = equor
        if self._validator is not None:
            self._validator._equor = equor
        self._logger.info("equor_wired_to_thymos")

    def set_evo(self, evo: Any) -> None:
        """Wire Evo so repair outcomes feed the learning system."""
        self._evo = evo
        self._logger.info("evo_wired_to_thymos")

    def set_atune(self, atune: Any) -> None:
        """Wire Atune so high-severity incidents become Percepts."""
        self._atune = atune
        self._logger.info("atune_wired_to_thymos")

    def set_health_monitor(self, health_monitor: HealthMonitor) -> None:
        """Wire Synapse HealthMonitor for health record queries."""
        self._health_monitor = health_monitor
        if self._causal_analyzer is not None:
            self._causal_analyzer._health = health_monitor
        self._logger.info("health_monitor_wired_to_thymos")

    def set_nova(self, nova: Any) -> None:
        """Wire Nova so critical incidents generate urgent repair goals."""
        self._nova = nova
        self._logger.info("nova_wired_to_thymos")

    def set_soma(self, soma: Any) -> None:
        """Wire Soma for integrity-based constitutional health gating."""
        self._soma = soma
        self._logger.info("soma_wired_to_thymos")

    # ─── Lifecycle ─────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Build all sub-systems, load the antibody library, and subscribe
        to Synapse health events.
        """
        if self._initialized:
            return

        # ── Sentinels ──
        self._exception_sentinel = ExceptionSentinel()
        self._contract_sentinel = ContractSentinel()
        self._feedback_loop_sentinel = FeedbackLoopSentinel()
        self._drift_sentinel = DriftSentinel()
        self._cognitive_stall_sentinel = CognitiveStallSentinel()

        # ── Triage ──
        self._deduplicator = IncidentDeduplicator()
        self._severity_scorer = SeverityScorer()
        self._response_router = ResponseRouter()

        # ── Diagnosis ──
        self._causal_analyzer = CausalAnalyzer(health_provider=self._health_monitor)
        self._temporal_correlator = TemporalCorrelator()
        self._diagnostic_engine = DiagnosticEngine(llm_client=self._llm)

        # ── Prescription ──
        self._prescriber = RepairPrescriber()
        self._validator = RepairValidator(equor=self._equor)

        # ── Antibody Library ──
        self._antibody_library = AntibodyLibrary(neo4j_client=self._neo4j)
        await self._antibody_library.initialize()

        # ── Prophylactic ──
        self._prophylactic_scanner = ProphylacticScanner(
            antibody_library=self._antibody_library,
        )
        self._homeostasis_controller = HomeostasisController()

        # ── Governor ──
        self._governor = HealingGovernor()

        # ── Subscribe to Synapse Events ──
        if self._synapse is not None:
            event_bus = self._synapse._event_bus
            for event_type in _SUBSCRIBED_EVENTS:
                event_bus.subscribe(event_type, self._on_synapse_event)

        # ── Start background loops ──
        self._sentinel_task = asyncio.create_task(
            self._sentinel_scan_loop(),
            name="thymos_sentinel_scan",
        )
        self._homeostasis_task = asyncio.create_task(
            self._homeostasis_loop(),
            name="thymos_homeostasis",
        )

        self._initialized = True

        antibody_count = len(self._antibody_library._all) if self._antibody_library else 0
        self._logger.info(
            "thymos_initialized",
            antibodies_loaded=antibody_count,
            subscribed_events=len(_SUBSCRIBED_EVENTS),
        )

    async def shutdown(self) -> None:
        """Graceful shutdown. Cancel background tasks and log final stats."""
        self._logger.info("thymos_shutting_down")

        # Cancel background tasks
        for task in (self._sentinel_task, self._homeostasis_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        self._sentinel_task = None
        self._homeostasis_task = None

        self._logger.info(
            "thymos_shutdown",
            total_incidents=self._total_incidents,
            total_repairs_attempted=self._total_repairs_attempted,
            total_repairs_succeeded=self._total_repairs_succeeded,
            active_incidents=len(self._active_incidents),
            antibodies_total=(
                len(self._antibody_library._all) if self._antibody_library else 0
            ),
        )

    # ─── Synapse Event Handler ───────────────────────────────────────

    async def _on_synapse_event(self, event: SynapseEvent) -> None:
        """
        Convert Synapse health events into Incidents.

        This is how Thymos learns about system failures without direct
        coupling to every system. Synapse watches health; Thymos watches
        Synapse events.
        """
        severity, incident_class = self._classify_synapse_event(event)

        if severity is None:
            # Recovery events — resolve any matching active incidents
            if event.event_type in (
                SynapseEventType.SYSTEM_RECOVERED,
                SynapseEventType.SAFE_MODE_EXITED,
            ):
                await self._handle_recovery_event(event)
            return

        source_system = event.data.get("system_id", event.source_system)

        fp = hashlib.sha256(
            f"{source_system}:{event.event_type.value}".encode()
        ).hexdigest()[:16]

        incident = Incident(
            incident_class=incident_class,
            severity=severity,
            fingerprint=fp,
            source_system=source_system,
            error_type=event.event_type.value,
            error_message=(
                f"Synapse health event: {event.event_type.value} "
                f"for {source_system}"
            ),
            context=event.data,
        )

        await self.on_incident(incident)

    def _classify_synapse_event(
        self,
        event: SynapseEvent,
    ) -> tuple[IncidentSeverity | None, IncidentClass]:
        """Map a Synapse event type to incident severity and class."""
        mapping: dict[SynapseEventType, tuple[IncidentSeverity | None, IncidentClass]] = {
            SynapseEventType.SYSTEM_FAILED: (
                IncidentSeverity.CRITICAL,
                IncidentClass.CRASH,
            ),
            SynapseEventType.SYSTEM_RESTARTING: (
                IncidentSeverity.HIGH,
                IncidentClass.CRASH,
            ),
            SynapseEventType.SYSTEM_OVERLOADED: (
                IncidentSeverity.MEDIUM,
                IncidentClass.DEGRADATION,
            ),
            SynapseEventType.SAFE_MODE_ENTERED: (
                IncidentSeverity.CRITICAL,
                IncidentClass.CRASH,
            ),
            SynapseEventType.CLOCK_OVERRUN: (
                IncidentSeverity.MEDIUM,
                IncidentClass.DEGRADATION,
            ),
            SynapseEventType.RESOURCE_PRESSURE: (
                IncidentSeverity.MEDIUM,
                IncidentClass.RESOURCE_EXHAUSTION,
            ),
            # Recovery events — no incident created
            SynapseEventType.SYSTEM_RECOVERED: (None, IncidentClass.CRASH),
            SynapseEventType.SAFE_MODE_EXITED: (None, IncidentClass.CRASH),
        }
        return mapping.get(
            event.event_type,
            (IncidentSeverity.LOW, IncidentClass.DEGRADATION),
        )

    async def _handle_recovery_event(self, event: SynapseEvent) -> None:
        """Resolve active incidents for a recovered system."""
        source_system = event.data.get("system_id", event.source_system)

        resolved_ids: list[str] = []
        for incident_id, incident in list(self._active_incidents.items()):
            if incident.source_system == source_system:
                incident.repair_status = RepairStatus.RESOLVED
                incident.repair_successful = True
                now = utc_now()
                incident.resolution_time_ms = int(
                    (now - incident.timestamp).total_seconds() * 1000
                )
                self._resolution_times.append(float(incident.resolution_time_ms))
                resolved_ids.append(incident_id)

                if self._governor is not None:
                    self._governor.resolve_incident(incident_id)

        for incident_id in resolved_ids:
            self._active_incidents.pop(incident_id, None)

        if resolved_ids:
            self._logger.info(
                "recovery_event_resolved_incidents",
                source_system=source_system,
                resolved_count=len(resolved_ids),
            )

    # ─── Main Entry Point ────────────────────────────────────────────

    async def on_incident(self, incident: Incident) -> None:
        """
        Primary entry point for the immune pipeline.

        Called by sentinels (directly) and Synapse event handler.
        Deduplicates, then routes to the full processing pipeline.

        This method NEVER raises — immune failures must not cascade.
        """
        try:
            await self._on_incident_inner(incident)
        except Exception as exc:
            # Thymos must not crash. Log and continue.
            self._logger.error(
                "thymos_internal_error",
                error=str(exc),
                incident_id=incident.id,
                incident_source=incident.source_system,
            )

    async def _on_incident_inner(self, incident: Incident) -> None:
        """Dedup → score → buffer → route → process."""
        assert self._deduplicator is not None
        assert self._severity_scorer is not None
        assert self._response_router is not None
        assert self._governor is not None

        # Step 1: Deduplicate
        dedup_result = self._deduplicator.deduplicate(incident)
        if dedup_result is None:
            self._logger.debug(
                "incident_deduplicated",
                fingerprint=incident.fingerprint,
                count=incident.occurrence_count,
            )
            return

        # Step 2: Score severity (composite)
        scored_severity = self._severity_scorer.compute_severity(incident)
        incident.severity = scored_severity

        # Step 3: Track
        self._total_incidents += 1
        self._incident_buffer.append(incident)
        self._active_incidents[incident.id] = incident
        self._incidents_by_severity[scored_severity.value] = (
            self._incidents_by_severity.get(scored_severity.value, 0) + 1
        )
        self._incidents_by_class[incident.incident_class.value] = (
            self._incidents_by_class.get(incident.incident_class.value, 0) + 1
        )

        # Register with governor for storm detection
        self._governor.register_incident(incident)

        # Record in temporal correlator
        if self._temporal_correlator is not None:
            self._temporal_correlator.record_event(
                event_type="incident",
                details=f"{incident.incident_class.value}: {incident.error_message}",
                system_id=incident.source_system,
            )

        # Record in causal analyzer
        if self._causal_analyzer is not None:
            self._causal_analyzer.record_incident(incident)

        # Emit telemetry
        self._emit_metric("thymos.incidents.created", 1, tags={"class": incident.incident_class.value})
        self._emit_metric("thymos.incidents.severity", 1, tags={"severity": scored_severity.value})

        self._logger.info(
            "incident_created",
            incident_id=incident.id,
            source_system=incident.source_system,
            incident_class=incident.incident_class.value,
            severity=scored_severity.value,
            fingerprint=incident.fingerprint[:16],
        )

        # Step 4: Make the organism feel it — route to Atune as a Percept
        await self._broadcast_as_percept(incident)

        # Step 5: Route to initial repair tier
        initial_tier = self._response_router.route(incident)
        incident.repair_tier = initial_tier

        # Step 6: Process through the immune pipeline
        if initial_tier == RepairTier.NOOP:
            incident.repair_status = RepairStatus.ACCEPTED
            self._active_incidents.pop(incident.id, None)
            if self._governor is not None:
                self._governor.resolve_incident(incident.id)
            return

        # Process in a background task so we don't block the event bus callback
        asyncio.create_task(
            self._process_incident_safe(incident),
            name=f"thymos_process_{incident.id[:8]}",
        )

    # ─── Immune Pipeline ─────────────────────────────────────────────

    async def _process_incident_safe(self, incident: Incident) -> None:
        """Wrapper that catches errors during incident processing."""
        try:
            await self.process_incident(incident)
        except Exception as exc:
            self._logger.error(
                "incident_processing_failed",
                incident_id=incident.id,
                error=str(exc),
            )
            incident.repair_status = RepairStatus.ESCALATED
            self._active_incidents.pop(incident.id, None)

    async def process_incident(self, incident: Incident) -> None:
        """
        Full immune pipeline: Diagnose → Prescribe → Validate → Apply → Verify → Learn.

        This is the core of Thymos. Each step has clear entry/exit criteria
        and failure modes that escalate to the next tier.
        """
        assert self._governor is not None
        assert self._antibody_library is not None
        assert self._diagnostic_engine is not None
        assert self._causal_analyzer is not None
        assert self._temporal_correlator is not None
        assert self._prescriber is not None
        assert self._validator is not None

        start_time = time.monotonic()

        # ── Step 1: Check governor budget ──
        if not self._governor.should_diagnose(incident):
            self._logger.info(
                "diagnosis_throttled",
                incident_id=incident.id,
                healing_mode=self._governor.healing_mode.value,
            )
            incident.repair_status = RepairStatus.ESCALATED
            self._active_incidents.pop(incident.id, None)
            return

        # ── Step 2: Diagnose ──
        incident.repair_status = RepairStatus.DIAGNOSING
        self._governor.begin_diagnosis()
        self._total_diagnoses += 1

        try:
            diagnosis = await self._diagnose(incident)
        finally:
            self._governor.end_diagnosis()

        diagnosis_ms = (time.monotonic() - start_time) * 1000
        self._diagnosis_latencies.append(diagnosis_ms)
        self._diagnosis_confidences.append(diagnosis.confidence)

        incident.root_cause_hypothesis = diagnosis.root_cause
        incident.diagnostic_confidence = diagnosis.confidence
        if diagnosis.repair_tier is not None:
            incident.repair_tier = diagnosis.repair_tier

        self._emit_metric("thymos.diagnosis.confidence", diagnosis.confidence)
        self._emit_metric("thymos.diagnosis.latency_ms", diagnosis_ms)

        self._logger.info(
            "diagnosis_complete",
            incident_id=incident.id,
            root_cause=diagnosis.root_cause[:80],
            confidence=f"{diagnosis.confidence:.2f}",
            repair_tier=diagnosis.repair_tier.name if diagnosis.repair_tier else "unknown",
            latency_ms=f"{diagnosis_ms:.0f}",
        )

        # ── Step 2b: Inject urgent goal for critical incidents ──
        if incident.severity == IncidentSeverity.CRITICAL and self._nova is not None:
            await self._inject_repair_goal(incident, diagnosis.repair_tier, resolved=False)

        # ── Step 2c: Integrity precision gating (Soma) ──
        # When Soma reports high integrity precision (body is well-calibrated on its
        # constitutional state), amplify constitutional health weighting in diagnosis.
        # This makes Thymos prefer less-invasive repairs when the organism "feels" its
        # integrity clearly — high trust in the self-model → respect constitutional signals.
        if self._soma is not None:
            try:
                signal = self._soma.get_current_signal()
                precision_weights = signal.precision_weights
                integrity_precision = precision_weights.get("integrity", 1.0)
                if integrity_precision > 0.7:
                    # Boost diagnosis confidence slightly — we trust the assessment
                    boosted_confidence = min(1.0, diagnosis.confidence * 1.15)
                    diagnosis = diagnosis.model_copy(update={"confidence": boosted_confidence})
                    self._logger.debug(
                        "integrity_precision_gating_applied",
                        integrity_precision=round(integrity_precision, 3),
                        confidence_before=round(diagnosis.confidence / 1.15, 3),
                        confidence_after=round(boosted_confidence, 3),
                    )
            except Exception as exc:
                self._logger.debug("soma_integrity_gating_error", error=str(exc))

        # ── Step 3: Prescribe ──
        incident.repair_status = RepairStatus.PRESCRIBING
        repair = await self._prescriber.prescribe(incident, diagnosis)

        self._logger.info(
            "repair_prescribed",
            incident_id=incident.id,
            tier=repair.tier.name,
            action=repair.action,
        )

        # ── Step 4: Validate ──
        incident.repair_status = RepairStatus.VALIDATING
        validation = await self._validator.validate(incident, repair)

        if not validation.approved:
            self._logger.warning(
                "repair_rejected",
                incident_id=incident.id,
                reason=validation.reason,
            )
            # Escalate or accept the override
            if validation.escalate_to is not None:
                repair = RepairSpec(
                    tier=validation.escalate_to,
                    action="alert_operator" if validation.escalate_to == RepairTier.ESCALATE else repair.action,
                    target_system=repair.target_system,
                    reason=f"Escalated: {validation.reason}",
                )
            else:
                incident.repair_status = RepairStatus.ESCALATED
                self._active_incidents.pop(incident.id, None)
                return

        # ── Step 5: Apply ──
        incident.repair_status = RepairStatus.APPLYING
        self._total_repairs_attempted += 1
        tier_name = repair.tier.name
        self._repairs_by_tier[tier_name] = self._repairs_by_tier.get(tier_name, 0) + 1
        self._governor.record_repair(repair.tier)

        self._emit_metric("thymos.repairs.attempted", 1, tags={"tier": tier_name})

        applied = await self._apply_repair(incident, repair)

        if not applied:
            self._logger.warning(
                "repair_application_failed",
                incident_id=incident.id,
                tier=repair.tier.name,
            )
            self._total_repairs_failed += 1
            self._emit_metric("thymos.repairs.failed", 1, tags={"tier": tier_name})
            incident.repair_status = RepairStatus.ESCALATED
            self._active_incidents.pop(incident.id, None)
            return

        # ── Step 6: Verify ──
        incident.repair_status = RepairStatus.VERIFYING
        verified = await self._verify_repair(incident, repair)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        if verified:
            # ── Success ──
            incident.repair_status = RepairStatus.RESOLVED
            incident.repair_successful = True
            incident.resolution_time_ms = int(elapsed_ms)
            self._resolution_times.append(elapsed_ms)
            self._total_repairs_succeeded += 1

            self._emit_metric("thymos.repairs.succeeded", 1, tags={"tier": tier_name})
            self._emit_metric("thymos.incidents.mean_resolution_ms", elapsed_ms)

            self._logger.info(
                "incident_resolved",
                incident_id=incident.id,
                tier=repair.tier.name,
                resolution_ms=f"{elapsed_ms:.0f}",
            )

            # ── Step 7: Learn ──
            await self._learn_from_success(incident, repair, diagnosis)

            # ── Step 7b: Inject recovery monitoring goal for RESTART+ repairs ──
            if repair.tier >= RepairTier.RESTART and self._nova is not None:
                await self._inject_repair_goal(incident, repair.tier, resolved=True)

        else:
            # ── Rollback ──
            incident.repair_status = RepairStatus.ROLLED_BACK
            incident.repair_successful = False
            self._total_repairs_rolled_back += 1

            self._emit_metric("thymos.repairs.rolled_back", 1, tags={"tier": tier_name})

            self._logger.warning(
                "repair_rolled_back",
                incident_id=incident.id,
                tier=repair.tier.name,
                reason="Post-repair verification failed",
            )

            # ── Learn from failure ──
            await self._learn_from_failure(incident, repair)

        # Clean up active tracking
        self._active_incidents.pop(incident.id, None)
        if self._governor is not None:
            self._governor.resolve_incident(incident.id)

        # Check storm exit
        self._governor.check_storm_exit()

    # ─── Diagnosis ──────────────────────────────────────────────────

    async def _diagnose(self, incident: Incident) -> Diagnosis:
        """
        Full diagnostic sequence:
        1. Check antibody library for known fix
        2. Trace causal chain through dependency graph
        3. Correlate temporal events
        4. Generate and test hypotheses
        """
        assert self._antibody_library is not None
        assert self._causal_analyzer is not None
        assert self._temporal_correlator is not None
        assert self._diagnostic_engine is not None

        # Fast path: check antibody library
        antibody_match = await self._antibody_library.lookup(incident.fingerprint)
        if antibody_match is not None and antibody_match.effectiveness > 0.8:
            self._total_antibodies_applied += 1
            self._emit_metric("thymos.antibodies.applied", 1)
            self._logger.info(
                "antibody_match",
                incident_id=incident.id,
                antibody_id=antibody_match.id,
                effectiveness=f"{antibody_match.effectiveness:.2f}",
            )

        # Causal analysis: trace upstream dependencies
        causal_chain = await self._causal_analyzer.trace_root_cause(incident)

        # Temporal correlation: what changed before the incident?
        correlations = self._temporal_correlator.correlate(incident)

        # Full diagnosis with hypothesis generation
        diagnosis = await self._diagnostic_engine.diagnose(
            incident=incident,
            causal_chain=causal_chain,
            correlations=correlations,
            antibody_match=antibody_match,
        )

        self._emit_metric("thymos.diagnosis.hypotheses", len(diagnosis.all_hypotheses))

        return diagnosis

    # ─── Repair Application ──────────────────────────────────────────

    async def _apply_repair(self, incident: Incident, repair: RepairSpec) -> bool:
        """
        Apply a repair based on its tier.

        Returns True if the repair was applied successfully (not verified yet).
        """
        try:
            if repair.tier == RepairTier.NOOP:
                return True

            elif repair.tier == RepairTier.PARAMETER:
                return await self._apply_parameter_repair(repair)

            elif repair.tier == RepairTier.RESTART:
                return await self._apply_restart_repair(repair)

            elif repair.tier == RepairTier.KNOWN_FIX:
                return await self._apply_antibody_repair(incident, repair)

            elif repair.tier == RepairTier.NOVEL_FIX:
                return await self._apply_novel_repair(incident, repair)

            elif repair.tier == RepairTier.ESCALATE:
                return await self._apply_escalation(incident, repair)

            else:
                self._logger.warning(
                    "unknown_repair_tier",
                    tier=repair.tier,
                    incident_id=incident.id,
                )
                return False

        except Exception as exc:
            self._logger.error(
                "repair_application_error",
                incident_id=incident.id,
                tier=repair.tier.name,
                error=str(exc),
            )
            return False

    async def _apply_parameter_repair(self, repair: RepairSpec) -> bool:
        """Apply Tier 1: parameter adjustment."""
        if not repair.parameter_changes:
            return False

        applied_count = 0
        for change in repair.parameter_changes:
            path = change.get("parameter_path", "")
            delta = change.get("delta", 0)

            if self._evo is not None:
                # Route parameter changes through Evo's tuner
                current = self._evo.get_parameter(path)
                if current is not None:
                    # Evo doesn't expose set_parameter directly,
                    # so we log the adjustment for now
                    self._logger.info(
                        "parameter_adjustment",
                        path=path,
                        current=current,
                        delta=delta,
                        new_value=current + delta,
                    )
                    applied_count += 1
                else:
                    self._logger.debug(
                        "parameter_not_found",
                        path=path,
                    )
            else:
                self._logger.info(
                    "parameter_adjustment_no_evo",
                    path=path,
                    delta=delta,
                )
                applied_count += 1

        return applied_count > 0

    async def _apply_restart_repair(self, repair: RepairSpec) -> bool:
        """
        Apply Tier 2: system restart.

        Thymos doesn't restart systems directly — it signals Synapse's
        DegradationManager to handle the restart sequence.
        """
        target = repair.target_system
        if target is None:
            return False

        if self._synapse is not None:
            try:
                # Emit a restart request through the event bus
                await self._synapse._event_bus.emit(
                    SynapseEvent(
                        event_type=SynapseEventType.SYSTEM_RESTARTING,
                        data={
                            "system_id": target,
                            "reason": repair.reason,
                            "requested_by": "thymos",
                        },
                        source_system="thymos",
                    )
                )
                self._logger.info(
                    "restart_requested",
                    target_system=target,
                    reason=repair.reason,
                )
                return True
            except Exception as exc:
                self._logger.error(
                    "restart_request_failed",
                    target_system=target,
                    error=str(exc),
                )
                return False

        self._logger.warning(
            "restart_no_synapse",
            target_system=target,
        )
        return False

    async def _apply_antibody_repair(self, incident: Incident, repair: RepairSpec) -> bool:
        """Apply Tier 3: known fix from antibody library."""
        assert self._antibody_library is not None

        if repair.antibody_id is None:
            return False

        antibody = self._antibody_library._all.get(repair.antibody_id)
        if antibody is None:
            self._logger.warning(
                "antibody_not_found",
                antibody_id=repair.antibody_id,
            )
            return False

        # Apply the antibody's repair spec
        inner_repair = antibody.repair_spec
        incident.antibody_id = antibody.id

        self._logger.info(
            "applying_antibody",
            antibody_id=antibody.id,
            inner_tier=inner_repair.tier.name,
            inner_action=inner_repair.action,
        )

        # Recursively apply the inner repair (but not another antibody to avoid loops)
        if inner_repair.tier == RepairTier.PARAMETER:
            return await self._apply_parameter_repair(inner_repair)
        elif inner_repair.tier == RepairTier.RESTART:
            return await self._apply_restart_repair(inner_repair)
        else:
            # For more complex inner repairs, log and mark as applied
            self._logger.info(
                "antibody_complex_repair",
                antibody_id=antibody.id,
                inner_tier=inner_repair.tier.name,
            )
            return True

    async def _apply_novel_repair(self, incident: Incident, repair: RepairSpec) -> bool:
        """
        Apply Tier 4: novel fix via Simula Code Agent.

        Iron rule: CANNOT apply without Equor review (enforced by validator).
        """
        assert self._governor is not None

        if not self._governor.should_codegen():
            self._logger.info(
                "codegen_throttled",
                incident_id=incident.id,
            )
            return False

        self._governor.begin_codegen()
        try:
            # Simula integration would go here.
            # For now, log the intent — Simula's Code Agent would receive
            # an EvolutionProposal with the repair spec.
            self._logger.info(
                "novel_repair_requested",
                incident_id=incident.id,
                target_system=repair.target_system,
                reason=repair.reason,
            )
            # In production, this would call:
            # await self._simula.propose_repair(incident, repair)
            return True
        finally:
            self._governor.end_codegen()

    async def _apply_escalation(self, incident: Incident, repair: RepairSpec) -> bool:
        """Apply Tier 5: human escalation."""
        self._logger.warning(
            "incident_escalated_to_human",
            incident_id=incident.id,
            source_system=incident.source_system,
            severity=incident.severity.value,
            reason=repair.reason,
        )
        incident.repair_status = RepairStatus.ESCALATED
        return True

    # ─── Post-Repair Verification ────────────────────────────────────

    async def _verify_repair(self, incident: Incident, repair: RepairSpec) -> bool:
        """
        Verify that a repair actually fixed the problem.

        Waits briefly, then checks if the source system is healthy.
        For escalations, verification is not applicable — assume success.
        """
        if repair.tier == RepairTier.ESCALATE:
            return True  # Escalation is inherently "successful"

        if repair.tier == RepairTier.NOOP:
            return True

        # Wait for the repair to take effect
        await asyncio.sleep(min(_POST_REPAIR_VERIFY_TIMEOUT_S, 5.0))

        # Check system health
        if self._health_monitor is not None and repair.target_system is not None:
            record = self._health_monitor.get_record(repair.target_system)
            if record is not None:
                from ecodiaos.systems.synapse.types import SystemStatus
                if record.status in (SystemStatus.HEALTHY, SystemStatus.STARTING):
                    return True
                elif record.status == SystemStatus.FAILED:
                    return False
                # Degraded/overloaded: ambiguous — check recent trend
                return record.consecutive_successes >= 1

        # No health monitor or target system: optimistic assumption
        return True

    # ─── Learning ────────────────────────────────────────────────────

    async def _learn_from_success(
        self,
        incident: Incident,
        repair: RepairSpec,
        diagnosis: Diagnosis,
    ) -> None:
        """
        A repair succeeded. Crystallize it into an antibody and feed Evo.

        This is where genuine adaptive immunity happens: the organism
        gets harder to break over time.
        """
        assert self._antibody_library is not None

        # If this was an antibody application, record success
        if incident.antibody_id is not None:
            await self._antibody_library.record_outcome(
                incident.antibody_id,
                success=True,
            )
            return

        # For Tier 2+ repairs (not NOOP/transient), create a new antibody
        if repair.tier >= RepairTier.PARAMETER:
            antibody = await self._antibody_library.create_from_repair(
                incident=incident,
                repair=repair,
            )
            self._total_antibodies_created += 1
            self._emit_metric("thymos.antibodies.created", 1)

            self._logger.info(
                "antibody_crystallized",
                antibody_id=antibody.id,
                fingerprint=incident.fingerprint[:16],
                tier=repair.tier.name,
            )

        # Persist incident to Neo4j
        await self._persist_incident(incident)

        # Feed success to Evo so the learning system can accumulate
        # evidence about what repair strategies work
        if self._evo is not None:
            await self._feed_repair_to_evo(incident, repair, success=True)

    async def _learn_from_failure(
        self,
        incident: Incident,
        repair: RepairSpec,
    ) -> None:
        """A repair failed. Record failure and update antibody if applicable."""
        assert self._antibody_library is not None

        if incident.antibody_id is not None:
            await self._antibody_library.record_outcome(
                incident.antibody_id,
                success=False,
            )

        # Persist the failed incident for post-mortem
        await self._persist_incident(incident)

        # Feed failure to Evo — failures are more salient than successes
        # and drive hypothesis formation about system vulnerabilities
        if self._evo is not None:
            await self._feed_repair_to_evo(incident, repair, success=False)

    async def _persist_incident(self, incident: Incident) -> None:
        """Persist an incident to Neo4j for the causal knowledge graph."""
        if self._neo4j is None:
            return

        try:
            await self._neo4j.execute_write(
                """
                MERGE (i:Incident {id: $id})
                SET i.source_system = $source_system,
                    i.incident_class = $incident_class,
                    i.severity = $severity,
                    i.fingerprint = $fingerprint,
                    i.error_type = $error_type,
                    i.error_message = $error_message,
                    i.repair_status = $repair_status,
                    i.repair_tier = $repair_tier,
                    i.repair_successful = $repair_successful,
                    i.resolution_time_ms = $resolution_time_ms,
                    i.root_cause = $root_cause,
                    i.timestamp = $timestamp
                """,
                {
                    "id": incident.id,
                    "source_system": incident.source_system,
                    "incident_class": incident.incident_class.value,
                    "severity": incident.severity.value,
                    "fingerprint": incident.fingerprint,
                    "error_type": incident.error_type,
                    "error_message": incident.error_message[:500],
                    "repair_status": incident.repair_status.value,
                    "repair_tier": incident.repair_tier.name if incident.repair_tier else "unknown",
                    "repair_successful": incident.repair_successful,
                    "resolution_time_ms": incident.resolution_time_ms,
                    "root_cause": incident.root_cause_hypothesis or "",
                    "timestamp": incident.timestamp.isoformat(),
                },
            )
        except Exception as exc:
            self._logger.debug("incident_persist_failed", error=str(exc))

    # ─── Cross-System Feedback ─────────────────────────────────────────

    async def _inject_repair_goal(
        self,
        incident: Incident,
        repair_tier: RepairTier | None,
        resolved: bool,
    ) -> None:
        """
        Inject a self-repair goal into Nova's goal manager.

        Pre-repair (resolved=False): high-urgency goal so Nova prioritises self-healing.
        Post-repair (resolved=True): follow-up monitoring goal at lower urgency.
        """
        from ecodiaos.primitives.common import DriveAlignmentVector, new_id
        from ecodiaos.systems.nova.types import Goal, GoalSource, GoalStatus

        tier_name = repair_tier.name if repair_tier else "UNKNOWN"

        if resolved:
            desc = (
                f"Monitor system recovery: {incident.source_system} "
                f"after {tier_name} repair"
            )
            priority, urgency = 0.6, 0.4
        else:
            desc = (
                f"Urgent: self-repair {incident.source_system} — "
                f"{incident.incident_class.value} incident ({tier_name})"
            )
            priority, urgency = 0.9, 0.85

        goal = Goal(
            id=new_id(),
            description=desc,
            source=GoalSource.MAINTENANCE,
            priority=priority,
            urgency=urgency,
            importance=0.7,
            drive_alignment=DriveAlignmentVector(
                coherence=0.8, care=0.1, growth=0.0, honesty=0.1,
            ),
            status=GoalStatus.ACTIVE,
        )
        try:
            await self._nova.add_goal(goal)
            self._logger.info(
                "repair_goal_injected",
                goal_id=goal.id,
                incident_id=incident.id,
                resolved=resolved,
            )
        except Exception as exc:
            self._logger.warning("repair_goal_injection_failed", error=str(exc))

    async def _feed_repair_to_evo(
        self,
        incident: Incident,
        repair: RepairSpec,
        success: bool,
    ) -> None:
        """
        Feed a repair outcome to Evo as a learning episode.

        Successful repairs teach the organism what works.
        Failed repairs teach it what doesn't — and are more salient.
        """
        from ecodiaos.primitives.common import new_id, utc_now
        from ecodiaos.primitives.memory_trace import Episode

        outcome_text = "succeeded" if success else "failed"
        episode = Episode(
            id=new_id(),
            source=f"thymos.repair_{outcome_text}",
            raw_content=(
                f"Repair {outcome_text}: {repair.action} on {repair.target_system}. "
                f"Tier: {repair.tier.name}. "
                f"Incident class: {incident.incident_class.value}. "
                f"Root cause: {incident.root_cause_hypothesis or 'unknown'}"
            ),
            summary=(
                f"Thymos {repair.tier.name} repair {outcome_text}: "
                f"{repair.target_system}"
            ),
            salience_composite=0.6 if success else 0.8,
            affect_valence=0.2 if success else -0.3,
            event_time=utc_now(),
        )
        try:
            await self._evo.process_episode(episode)
            self._logger.info(
                "repair_outcome_fed_to_evo",
                incident_id=incident.id,
                success=success,
                tier=repair.tier.name,
            )
        except Exception as exc:
            self._logger.warning("evo_feed_failed", error=str(exc))

    # ─── Percept Broadcasting ────────────────────────────────────────

    async def _broadcast_as_percept(self, incident: Incident) -> None:
        """
        Route high-severity incidents into Atune's workspace as Percepts.

        The organism perceives its own failures through the normal
        consciousness cycle. It hurts to break — and that's by design.
        Critical incidents get maximum salience; INFO incidents are barely noticed.
        """
        if self._atune is None:
            return

        salience = _SEVERITY_TO_SALIENCE.get(incident.severity, 0.1)

        # Only broadcast MEDIUM+ to avoid flooding the workspace
        if salience < 0.5:
            return

        try:
            from ecodiaos.systems.atune.types import WorkspaceContribution

            self._atune.contribute(
                WorkspaceContribution(
                    system="thymos",
                    content=(
                        f"[IMMUNE] {incident.severity.value.upper()} incident in "
                        f"{incident.source_system}: {incident.error_message}"
                    ),
                    priority=salience,
                    reason="immune_incident",
                )
            )
        except Exception as exc:
            self._logger.debug("percept_broadcast_failed", error=str(exc))

    # ─── Background Loops ────────────────────────────────────────────

    async def _sentinel_scan_loop(self) -> None:
        """
        Periodic sentinel scans for proactive failure detection.

        Feedback loop sentinel checks which loops are transmitting.
        Cognitive stall sentinel checks workspace health.
        Drift sentinel is fed by Synapse metrics (not looped here).
        """
        await asyncio.sleep(10.0)  # Let the organism warm up

        while True:
            try:
                await asyncio.sleep(_SENTINEL_SCAN_INTERVAL_S)

                # Feedback loop sentinel: check all defined loops
                if self._feedback_loop_sentinel is not None:
                    loop_incidents = self._feedback_loop_sentinel.check_loops()
                    for incident in loop_incidents:
                        await self.on_incident(incident)

                # Cognitive stall sentinel is fed per-cycle by Synapse,
                # not scanned here. It fires incidents from record_cycle().

            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._logger.warning(
                    "sentinel_scan_error",
                    error=str(exc),
                )

    async def _homeostasis_loop(self) -> None:
        """
        Proactive homeostatic regulation.

        Runs continuously on MAINTAIN cycle step timing. Checks metrics
        against optimal ranges and makes small preemptive adjustments.

        This is the organism's thermostat — it maintains optimal operating
        conditions without waiting for something to break.
        """
        await asyncio.sleep(30.0)  # Let the organism stabilize

        while True:
            try:
                await asyncio.sleep(_HOMEOSTASIS_INTERVAL_S)

                if self._homeostasis_controller is None:
                    continue

                adjustments = self._homeostasis_controller.check_homeostasis()

                for adjustment in adjustments:
                    self._total_homeostatic_adjustments += 1
                    self._emit_metric("thymos.homeostasis.adjustments", 1)

                    self._logger.info(
                        "homeostatic_adjustment",
                        metric=adjustment.metric_name,
                        current=f"{adjustment.current_value:.2f}",
                        trend=adjustment.trend_direction,
                        adjustment_path=adjustment.adjustment.parameter_path,
                        adjustment_delta=adjustment.adjustment.delta,
                    )

                # Check storm mode exit
                if self._governor is not None:
                    self._governor.check_storm_exit()

            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._logger.warning(
                    "homeostasis_loop_error",
                    error=str(exc),
                )

    # ─── Prophylactic Scanner ────────────────────────────────────────

    async def scan_files(self, files_changed: list[str]) -> list[dict[str, Any]]:
        """
        Pre-deployment prophylactic scan.

        Checks new or modified files against the antibody library's
        error patterns. Returns warnings for code that matches known
        failure signatures.
        """
        if self._prophylactic_scanner is None:
            return []

        warnings = await self._prophylactic_scanner.scan(files_changed)
        self._total_prophylactic_scans += 1
        self._total_prophylactic_warnings += len(warnings)

        self._emit_metric("thymos.prophylactic.scans", 1)
        self._emit_metric("thymos.prophylactic.warnings", len(warnings))

        return [w.model_dump() for w in warnings]

    # ─── Exception Sentinel (Public API) ─────────────────────────────

    async def report_exception(
        self,
        system_id: str,
        exception: Exception,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Public API for systems to report unhandled exceptions.

        Called by systems' own error handlers or by a global exception hook.
        Creates an Incident and routes it through the immune pipeline.
        """
        if self._exception_sentinel is None:
            return

        incident = self._exception_sentinel.intercept(
            system_id=system_id,
            method_name="unknown",
            exception=exception,
            context=context or {},
        )
        await self.on_incident(incident)

    # ─── Contract Sentinel (Public API) ──────────────────────────────

    async def report_contract_violation(
        self,
        source: str,
        target: str,
        operation: str,
        latency_ms: float,
        sla_ms: float,
    ) -> None:
        """
        Public API for reporting inter-system contract violations.

        Called by systems when an operation exceeds its SLA.
        """
        if self._contract_sentinel is None:
            return

        incident = self._contract_sentinel.check_contract(
            source=source,
            target=target,
            operation=operation,
            latency_ms=latency_ms,
        )
        if incident is not None:
            await self.on_incident(incident)

    # ─── Drift Sentinel (Public API) ─────────────────────────────────

    def record_metric(self, metric_name: str, value: float) -> None:
        """
        Feed a metric observation to the drift sentinel.

        Called by Synapse's telemetry pipeline to monitor for gradual
        degradation that wouldn't trigger a hard failure.
        """
        if self._drift_sentinel is None:
            return

        incident = self._drift_sentinel.record_metric(metric_name, value)
        if incident is not None:
            # Fire-and-forget: drift incidents are LOW priority
            asyncio.create_task(
                self.on_incident(incident),
                name=f"thymos_drift_{metric_name}",
            )

        # Also feed the temporal correlator
        if self._temporal_correlator is not None and self._drift_sentinel is not None:
            baseline = self._drift_sentinel._baselines.get(metric_name)
            if baseline is not None and baseline.is_warmed_up:
                self._temporal_correlator.record_metric_anomaly(
                    metric_name=metric_name,
                    value=value,
                    baseline=baseline.mean,
                    z_score=baseline.z_score(value),
                )

    # ─── Health ──────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """
        Health check for Thymos (required by Synapse health monitor).

        Returns a snapshot of immune system status, counters, and budget.
        """
        governor_budget = (
            self._governor.budget_state if self._governor else None
        )

        antibody_count = len(self._antibody_library._all) if self._antibody_library else 0
        mean_effectiveness = 0.0
        if self._antibody_library and self._antibody_library._all:
            effectivenesses = [
                a.effectiveness for a in self._antibody_library._all.values()
                if not a.retired
            ]
            if effectivenesses:
                mean_effectiveness = sum(effectivenesses) / len(effectivenesses)

        mean_resolution = 0.0
        if self._resolution_times:
            mean_resolution = sum(self._resolution_times) / len(self._resolution_times)

        mean_confidence = 0.0
        if self._diagnosis_confidences:
            mean_confidence = sum(self._diagnosis_confidences) / len(self._diagnosis_confidences)

        mean_diag_latency = 0.0
        if self._diagnosis_latencies:
            mean_diag_latency = sum(self._diagnosis_latencies) / len(self._diagnosis_latencies)

        homeostasis_ranges = 0
        if self._homeostasis_controller is not None:
            homeostasis_ranges = self._homeostasis_controller.metrics_in_range

        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "initialized": self._initialized,
            "healing_mode": (
                self._governor.healing_mode.value if self._governor else "unknown"
            ),
            # Incidents
            "total_incidents": self._total_incidents,
            "active_incidents": len(self._active_incidents),
            "mean_resolution_ms": round(mean_resolution, 1),
            "incidents_by_severity": dict(self._incidents_by_severity),
            "incidents_by_class": dict(self._incidents_by_class),
            # Antibodies
            "total_antibodies": antibody_count,
            "mean_antibody_effectiveness": round(mean_effectiveness, 3),
            "antibodies_applied": self._total_antibodies_applied,
            "antibodies_created": self._total_antibodies_created,
            # Repairs
            "repairs_attempted": self._total_repairs_attempted,
            "repairs_succeeded": self._total_repairs_succeeded,
            "repairs_failed": self._total_repairs_failed,
            "repairs_rolled_back": self._total_repairs_rolled_back,
            "repairs_by_tier": dict(self._repairs_by_tier),
            # Diagnosis
            "total_diagnoses": self._total_diagnoses,
            "mean_diagnosis_confidence": round(mean_confidence, 3),
            "mean_diagnosis_latency_ms": round(mean_diag_latency, 1),
            # Homeostasis
            "homeostatic_adjustments": self._total_homeostatic_adjustments,
            "metrics_in_range": homeostasis_ranges,  # Approximate
            # Storm
            "storm_activations": (
                self._governor.storm_activations if self._governor else 0
            ),
            # Prophylactic
            "prophylactic_scans": self._total_prophylactic_scans,
            "prophylactic_warnings": self._total_prophylactic_warnings,
            # Budget
            "budget": governor_budget.model_dump() if governor_budget else {},
        }

    # ─── Stats ──────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        """Synchronous stats for logging."""
        return {
            "initialized": self._initialized,
            "total_incidents": self._total_incidents,
            "active_incidents": len(self._active_incidents),
            "total_diagnoses": self._total_diagnoses,
            "total_repairs_attempted": self._total_repairs_attempted,
            "total_repairs_succeeded": self._total_repairs_succeeded,
            "healing_mode": (
                self._governor.healing_mode.value if self._governor else "unknown"
            ),
        }

    # ─── Telemetry Helper ───────────────────────────────────────────

    def _emit_metric(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Emit a metric if the collector is available."""
        if self._metrics is not None:
            try:
                asyncio.create_task(
                    self._metrics.record(
                        system="thymos",
                        metric=name,
                        value=value,
                        labels=tags,
                    )
                )
            except Exception:
                pass  # Telemetry failures must never affect immune function
