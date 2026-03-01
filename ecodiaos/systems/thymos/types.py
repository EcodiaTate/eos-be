"""
EcodiaOS — Thymos Type Definitions

All data types for the immune system: incidents, antibodies, repairs,
diagnoses, sentinels, and healing governance.

Every error, anomaly, and violation in EOS becomes an Incident — a
first-class primitive alongside Percept, Belief, and Intent.
"""

from __future__ import annotations

from datetime import datetime
import enum
from typing import Any

from pydantic import Field

from ecodiaos.primitives.common import EOSBaseModel, new_id, utc_now


# ─── Enums ────────────────────────────────────────────────────────


class IncidentSeverity(enum.StrEnum):
    """How bad is it?"""

    CRITICAL = "critical"  # System down, user impact, drives affected
    HIGH = "high"  # System degraded, partial user impact
    MEDIUM = "medium"  # Anomaly detected, no immediate user impact
    LOW = "low"  # Cosmetic, informational, or transient
    INFO = "info"  # Normal variance logged for pattern detection


class IncidentClass(enum.StrEnum):
    """What kind of failure is this?"""

    CRASH = "crash"  # Unhandled exception, system death
    DEGRADATION = "degradation"  # Slow or incorrect responses
    CONTRACT_VIOLATION = "contract_violation"  # Inter-system SLA breach
    LOOP_SEVERANCE = "loop_severance"  # Feedback loop not transmitting
    DRIFT = "drift"  # Gradual metric deviation from baseline
    PREDICTION_FAILURE = "prediction_failure"  # Active inference errors elevated
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Budget exceeded
    COGNITIVE_STALL = "cognitive_stall"  # Workspace cycle blocked or empty
    ECONOMIC_THREAT = "economic_threat"  # Malicious on-chain activity detected
    PROTOCOL_DEGRADATION = "protocol_degradation"  # DeFi protocol health declining


class RepairTier(int, enum.Enum):
    """Escalation ladder — least invasive first."""

    NOOP = 0  # Transient, already resolved
    PARAMETER = 1  # Adjust a configuration value
    RESTART = 2  # Restart the affected system
    KNOWN_FIX = 3  # Apply an antibody from the library
    NOVEL_FIX = 4  # Generate a new fix via Simula Code Agent
    ESCALATE = 5  # Human operator intervention required


class RepairStatus(enum.StrEnum):
    """Lifecycle of an incident repair."""

    PENDING = "pending"
    DIAGNOSING = "diagnosing"
    PRESCRIBING = "prescribing"
    VALIDATING = "validating"
    APPLYING = "applying"
    VERIFYING = "verifying"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    ACCEPTED = "accepted"  # Transient or INFO, no repair needed
    ROLLED_BACK = "rolled_back"


class HealingMode(enum.StrEnum):
    """Organism-wide healing state."""

    NOMINAL = "nominal"  # Normal operation
    HEALING = "healing"  # Active repair in progress
    STORM = "storm"  # Cytokine storm — focus on root cause only


# ─── Sentinel Types ──────────────────────────────────────────────


class ContractSLA(EOSBaseModel):
    """SLA definition for an inter-system contract."""

    source: str
    target: str
    operation: str
    max_latency_ms: float


class FeedbackLoop(EOSBaseModel):
    """Definition of a feedback loop that should be actively transmitting."""

    name: str
    source: str
    target: str
    signal: str
    check: str  # Descriptive check expression
    description: str


class DriftConfig(EOSBaseModel):
    """Configuration for statistical drift detection on a metric."""

    window: int = 500  # Number of samples in the rolling baseline
    sigma_threshold: float = 2.5  # Standard deviations before flagging
    direction: str | None = None  # "above", "below", or None (both)


class StallConfig(EOSBaseModel):
    """Threshold for cognitive stall detection."""

    min_value: float  # Rate must be above this
    window_cycles: int  # Number of cycles to observe


# ─── Economic Immune Types ───────────────────────────────────────


class ThreatType(enum.StrEnum):
    """Categories of economic threats the immune system detects."""

    FLASH_LOAN_ATTACK = "flash_loan_attack"
    PRICE_MANIPULATION = "price_manipulation"
    SUSPICIOUS_CONTRACT = "suspicious_contract"
    MEMPOOL_POISONING = "mempool_poisoning"
    RUG_PULL = "rug_pull"
    ORACLE_MANIPULATION = "oracle_manipulation"
    GOVERNANCE_ATTACK = "governance_attack"


class ThreatSeverity(enum.StrEnum):
    """Severity classification for economic threats."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProtocolAlert(EOSBaseModel):
    """Alert raised when a DeFi protocol's health degrades."""

    protocol: str
    alert_type: str  # "tvl_drop" | "oracle_deviation" | "contract_paused" | "governance_anomaly"
    current_value: float
    threshold_value: float
    deviation_percent: float
    requires_withdrawal: bool = False


class ThreatPattern(EOSBaseModel):
    """A detection pattern for recognising on-chain threats."""

    pattern_id: str = Field(default_factory=new_id)
    threat_type: ThreatType
    description: str
    detection_rule: str  # Human-readable rule description
    severity: ThreatSeverity
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    false_positive_rate: float = Field(0.05, ge=0.0, le=1.0)


class AddressBlacklistEntry(EOSBaseModel):
    """A blacklisted on-chain address with provenance tracking."""

    address: str
    chain_id: int = 8453  # Base L2
    reason: str
    threat_type: ThreatType
    source: str = "local"  # "local" | "federation" | "external"
    source_instance_id: str = ""
    confirmed: bool = False


class SimulationResult(EOSBaseModel):
    """Result of pre-simulating a transaction before broadcast."""

    passed: bool = True
    revert_reason: str = ""
    gas_used: int = 0
    value_delta_usd: float = 0.0
    slippage_bps: int = 0
    mev_risk_detected: bool = False
    warnings: list[str] = Field(default_factory=list)


# ─── Incident ────────────────────────────────────────────────────


class Incident(EOSBaseModel):
    """
    The fundamental immune primitive.

    Every error, anomaly, and violation becomes an Incident.
    Incidents are also Percepts — the organism perceives its own
    failures through the normal workspace broadcast cycle.
    """

    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)

    # ── Classification ──
    incident_class: IncidentClass
    severity: IncidentSeverity
    fingerprint: str  # Hash of (class, system, error_signature)

    # ── Source ──
    source_system: str
    error_type: str  # Exception class name or anomaly type
    error_message: str
    stack_trace: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    # ── Impact Assessment ──
    affected_systems: list[str] = Field(default_factory=list)
    blast_radius: float = Field(default=0.0, ge=0.0, le=1.0)
    user_visible: bool = False
    constitutional_impact: dict[str, float] = Field(
        default_factory=lambda: {
            "coherence": 0.0,
            "care": 0.0,
            "growth": 0.0,
            "honesty": 0.0,
        }
    )

    # ── Deduplication ──
    occurrence_count: int = 1
    first_seen: datetime | None = None

    # ── Diagnosis ──
    root_cause_hypothesis: str | None = None
    diagnostic_confidence: float = 0.0
    causal_chain: list[str] | None = None

    # ── Repair ──
    repair_tier: RepairTier | None = None
    repair_status: RepairStatus = RepairStatus.PENDING
    antibody_id: str | None = None

    # ── Learning ──
    resolution_time_ms: int | None = None
    repair_successful: bool | None = None


# ─── Diagnosis Types ─────────────────────────────────────────────


class CausalChain(EOSBaseModel):
    """Result of tracing error causality through the system graph."""

    root_system: str
    chain: list[str]  # System A → System B → failure
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: str = ""


class TemporalCorrelation(EOSBaseModel):
    """Something that changed in the window before the incident."""

    type: str  # "metric_anomaly" | "system_event"
    timestamp: datetime
    description: str
    time_delta_ms: int  # How many ms before the incident


class DiagnosticHypothesis(EOSBaseModel):
    """A testable hypothesis about what caused an incident."""

    id: str = Field(default_factory=new_id)
    statement: str
    diagnostic_test: str  # Name of the test to run
    diagnostic_test_params: dict[str, Any] = Field(default_factory=dict)
    suggested_repair_tier: RepairTier = RepairTier.PARAMETER
    confidence_prior: float = Field(0.5, ge=0.0, le=1.0)


class DiagnosticTestResult(EOSBaseModel):
    """Result of running a diagnostic test."""

    test_name: str
    passed: bool
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    raw_value: Any = None


class DiagnosticEvidence(EOSBaseModel):
    """All evidence gathered for diagnosing an incident."""

    incident: Incident
    causal_chain: CausalChain
    temporal_correlations: list[TemporalCorrelation] = Field(default_factory=list)
    recent_similar: list[Incident] = Field(default_factory=list)
    system_health_history: dict[str, Any] = Field(default_factory=dict)


class Diagnosis(EOSBaseModel):
    """Final diagnosis of an incident's root cause."""

    root_cause: str
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    repair_tier: RepairTier = RepairTier.PARAMETER
    antibody_id: str | None = None
    all_hypotheses: list[DiagnosticHypothesis] = Field(default_factory=list)
    test_results: list[DiagnosticTestResult] = Field(default_factory=list)
    reasoning: str = ""


# ─── Repair Types ────────────────────────────────────────────────


class ParameterFix(EOSBaseModel):
    """A single parameter adjustment."""

    parameter_path: str  # e.g., "synapse.clock.current_period_ms"
    delta: float  # Change amount (can be negative)
    reason: str = ""


class RepairSpec(EOSBaseModel):
    """Specification for a repair action."""

    tier: RepairTier
    action: str  # e.g., "log_and_monitor", "restart_system", "apply_antibody"
    target_system: str | None = None
    antibody_id: str | None = None
    parameter_changes: list[dict[str, Any]] = Field(default_factory=list)
    code_changes: dict[str, Any] | None = None
    evolution_proposal_id: str | None = None
    reason: str = ""


class ValidationResult(EOSBaseModel):
    """Result of the repair validation gate."""

    approved: bool
    reason: str = ""
    escalate_to: RepairTier | None = None
    modifications: dict[str, Any] | None = None


# ─── Antibody Types ──────────────────────────────────────────────


class Antibody(EOSBaseModel):
    """
    A crystallized successful repair.

    When a repair succeeds, it becomes an Antibody. The next time an
    incident with the same fingerprint appears, the antibody is applied
    instantly — no diagnosis needed.

    This is genuine adaptive immunity: the organism gets harder to break
    over time.
    """

    id: str = Field(default_factory=new_id)

    # ── Matching ──
    fingerprint: str
    incident_class: IncidentClass
    source_system: str
    error_pattern: str  # Regex or fragment for matching error_message

    # ── Repair ──
    repair_tier: RepairTier
    repair_spec: RepairSpec
    root_cause_description: str

    # ── Effectiveness ──
    application_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    effectiveness: float = 1.0  # success / (success + failure)

    # ── Lifecycle ──
    created_at: datetime = Field(default_factory=utc_now)
    last_applied: datetime | None = None
    source_incident_id: str = ""
    retired: bool = False

    # ── Lineage ──
    generation: int = 1
    parent_antibody_id: str | None = None


# ─── Prophylactic Types ──────────────────────────────────────────


class ProphylacticWarning(EOSBaseModel):
    """Warning issued by the prophylactic scanner."""

    filepath: str
    antibody_id: str
    warning: str
    suggestion: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ParameterAdjustment(EOSBaseModel):
    """A homeostatic parameter nudge — Tier 1, no governance."""

    metric_name: str
    current_value: float
    optimal_min: float
    optimal_max: float
    adjustment: ParameterFix
    trend_direction: str  # "rising" | "falling"


# ─── Governor Types ──────────────────────────────────────────────


class HealingBudgetState(EOSBaseModel):
    """Current state of the healing budget."""

    repairs_this_hour: int = 0
    novel_repairs_today: int = 0
    max_repairs_per_hour: int = 5
    max_novel_repairs_per_day: int = 3
    active_diagnoses: int = 0
    max_concurrent_diagnoses: int = 3
    active_codegen: int = 0
    max_concurrent_codegen: int = 1
    storm_mode: bool = False
    storm_focus_system: str | None = None
    cpu_budget_fraction: float = 0.10


# ─── Health Snapshot ─────────────────────────────────────────────


class ThymosHealthSnapshot(EOSBaseModel):
    """Thymos system health and observability."""

    status: str = "healthy"
    healing_mode: HealingMode = HealingMode.NOMINAL

    # Incident metrics
    total_incidents_created: int = 0
    active_incidents: int = 0
    mean_resolution_ms: float = 0.0
    incidents_by_severity: dict[str, int] = Field(default_factory=dict)
    incidents_by_class: dict[str, int] = Field(default_factory=dict)

    # Antibody metrics
    total_antibodies: int = 0
    mean_antibody_effectiveness: float = 0.0
    antibodies_applied: int = 0
    antibodies_created: int = 0
    antibodies_retired: int = 0

    # Repair metrics
    repairs_attempted: int = 0
    repairs_succeeded: int = 0
    repairs_failed: int = 0
    repairs_rolled_back: int = 0
    repairs_by_tier: dict[str, int] = Field(default_factory=dict)

    # Diagnosis metrics
    diagnoses_run: int = 0
    mean_diagnosis_confidence: float = 0.0
    mean_diagnosis_latency_ms: float = 0.0

    # Homeostasis metrics
    homeostatic_adjustments: int = 0
    metrics_in_range: int = 0
    metrics_total: int = 0

    # Storm metrics
    storm_activations: int = 0

    # Prophylactic metrics
    prophylactic_scans: int = 0
    prophylactic_warnings: int = 0

    # Budget
    budget: HealingBudgetState = Field(default_factory=HealingBudgetState)

    timestamp: datetime = Field(default_factory=utc_now)
