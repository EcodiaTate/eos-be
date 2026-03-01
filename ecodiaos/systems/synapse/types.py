"""
EcodiaOS — Synapse Type Definitions

All data types for the autonomic nervous system: cycle clock, health monitoring,
resource allocation, degradation strategies, emergent rhythm detection,
and cross-system coherence measurement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
import enum
from typing import Any, Protocol

from pydantic import Field

from ecodiaos.primitives.common import EOSBaseModel, new_id, utc_now


# ─── System Status ────────────────────────────────────────────────────


class SystemStatus(enum.StrEnum):
    """Operational state of a managed cognitive system."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    STOPPED = "stopped"
    STARTING = "starting"
    RESTARTING = "restarting"


# ─── Health Monitoring ────────────────────────────────────────────────


class SystemHeartbeat(EOSBaseModel):
    """Health report returned by a managed system's health() method."""

    system_id: str
    status: str = "healthy"
    latency_ms: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class SystemHealthRecord(EOSBaseModel):
    """
    Synapse's internal per-system health tracking.

    Tracks heartbeat history, consecutive misses, latency statistics,
    and error accumulation for degradation decisions.
    """

    system_id: str
    status: SystemStatus = SystemStatus.STOPPED
    consecutive_misses: int = 0
    consecutive_successes: int = 0
    total_checks: int = 0
    total_failures: int = 0
    last_check_time: datetime | None = None
    last_success_time: datetime | None = None
    last_failure_time: datetime | None = None
    # Exponential moving average of heartbeat latency
    latency_ema_ms: float = 0.0
    # Peak latency in current window
    latency_peak_ms: float = 0.0
    # Number of restarts attempted
    restart_count: int = 0
    # Is this a critical system (failure → safe mode)?
    is_critical: bool = False

    def record_success(self, latency_ms: float) -> None:
        """Record a successful heartbeat."""
        now = utc_now()
        self.consecutive_misses = 0
        self.consecutive_successes += 1
        self.total_checks += 1
        self.last_check_time = now
        self.last_success_time = now
        # EMA with alpha=0.2 for smooth tracking
        alpha = 0.2
        self.latency_ema_ms = (
            alpha * latency_ms + (1 - alpha) * self.latency_ema_ms
        )
        self.latency_peak_ms = max(self.latency_peak_ms, latency_ms)
        # Recover from degraded states
        if self.status == SystemStatus.FAILED and self.consecutive_successes >= 3 or self.status in (SystemStatus.DEGRADED, SystemStatus.OVERLOADED):
            self.status = SystemStatus.HEALTHY

    def record_failure(self) -> None:
        """Record a missed or failed heartbeat."""
        now = utc_now()
        self.consecutive_successes = 0
        self.consecutive_misses += 1
        self.total_checks += 1
        self.total_failures += 1
        self.last_check_time = now
        self.last_failure_time = now

    def record_overloaded(self, latency_ms: float) -> None:
        """Record a successful but slow heartbeat (latency > 2x EMA)."""
        self.record_success(latency_ms)
        if self.status == SystemStatus.HEALTHY:
            self.status = SystemStatus.OVERLOADED


# ─── Resource Allocation ──────────────────────────────────────────────


class SystemBudget(EOSBaseModel):
    """Per-system resource budget allocation."""

    system_id: str = ""
    cpu_share: float = Field(0.1, ge=0.0, le=1.0)
    memory_mb: int = 256
    io_priority: int = Field(3, ge=1, le=5)  # 1 = highest


class ResourceAllocation(EOSBaseModel):
    """
    Allocation message delivered to a system each rebalance.

    Translates abstract budgets into per-cycle concrete limits.
    """

    system_id: str
    compute_ms_per_cycle: float = 50.0
    burst_allowance: float = Field(1.0, ge=1.0, le=3.0)
    priority_boost: float = Field(0.0, ge=-1.0, le=1.0)
    timestamp: datetime = Field(default_factory=utc_now)


class ResourceSnapshot(EOSBaseModel):
    """Point-in-time resource utilisation snapshot across all systems."""

    timestamp: datetime = Field(default_factory=utc_now)
    total_cpu_percent: float = 0.0
    total_memory_mb: float = 0.0
    total_memory_percent: float = 0.0
    per_system: dict[str, dict[str, float]] = Field(default_factory=dict)
    process_cpu_percent: float = 0.0
    process_memory_mb: float = 0.0


# ─── Clock ────────────────────────────────────────────────────────────


class SomaticCycleState(EOSBaseModel):
    """
    Somatic snapshot carried on every theta tick.

    Extracted from Soma's AllostaticSignal after step 0 runs. Downstream
    consumers (Nova, Evo, coherence monitor) read this from CycleResult
    rather than holding a direct Soma reference, keeping coupling minimal.

    All fields default to neutral/quiescent values so the struct is safe
    to construct when Soma is absent or degraded.
    """

    urgency: float = 0.0
    """Scalar [0,1] allostatic urgency — how far from all setpoints."""

    dominant_error: str = "energy"
    """Name of the InteroceptiveDimension with the largest allostatic error."""

    arousal_sensed: float = 0.4
    """Raw sensed AROUSAL dimension [0,1] — used by clock for adaptive timing."""

    energy_sensed: float = 0.6
    """Raw sensed ENERGY dimension [0,1]."""

    precision_weights: dict[str, float] = Field(default_factory=dict)
    """Per-dimension precision weights from Soma. Empty dict = uniform."""

    nearest_attractor: str | None = None
    """Label of the nearest phase-space attractor, or None if transient."""

    trajectory_heading: str = "transient"
    """Phase-space heading: 'approaching', 'within', 'departing', 'transient'."""

    soma_cycle_ms: float = 0.0
    """How long Soma's own run_cycle() took (for overrun diagnostics)."""


class SomaTickEvent(EOSBaseModel):
    """
    Event emitted by Synapse on the Redis event bus after every theta tick
    where Soma ran. Carries the full somatic state for stateless consumers.

    Channel: ``synapse.soma.tick``
    Payload: this model serialised to JSON.
    """

    id: str = Field(default_factory=new_id)
    cycle_number: int
    somatic_state: SomaticCycleState
    timestamp: datetime = Field(default_factory=utc_now)


class CycleResult(EOSBaseModel):
    """Result of a single theta rhythm tick."""

    cycle_number: int
    elapsed_ms: float
    budget_ms: float
    overrun: bool = False
    broadcast_id: str | None = None
    had_broadcast: bool = False
    arousal: float = 0.0
    salience_composite: float = 0.0
    # Somatic state snapshot from Soma step 0 (None when Soma absent/degraded)
    somatic: SomaticCycleState | None = None
    timestamp: datetime = Field(default_factory=utc_now)


class ClockState(EOSBaseModel):
    """Snapshot of the cognitive clock's current state."""

    running: bool = False
    paused: bool = False
    cycle_count: int = 0
    current_period_ms: float = 150.0
    target_period_ms: float = 150.0
    jitter_ms: float = 0.0
    arousal: float = 0.0
    overrun_count: int = 0
    # Cycles per second (actual measured rate)
    actual_rate_hz: float = 0.0


# ─── Degradation ──────────────────────────────────────────────────────


class DegradationLevel(enum.StrEnum):
    """Overall organism degradation level."""

    NOMINAL = "nominal"
    DEGRADED = "degraded"
    SAFE_MODE = "safe_mode"
    EMERGENCY = "emergency"


class DegradationStrategy(EOSBaseModel):
    """Per-system fallback configuration."""

    system_id: str
    triggers_safe_mode: bool = False
    fallback_behavior: str = ""
    auto_restart: bool = True
    max_restart_attempts: int = 3
    restart_backoff_base_s: float = 5.0


# ─── Event Bus ────────────────────────────────────────────────────────


class SynapseEventType(enum.StrEnum):
    """All event types emitted by Synapse."""

    # System lifecycle
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    SYSTEM_FAILED = "system_failed"
    SYSTEM_RECOVERED = "system_recovered"
    SYSTEM_RESTARTING = "system_restarting"
    SYSTEM_OVERLOADED = "system_overloaded"

    # Safe mode
    SAFE_MODE_ENTERED = "safe_mode_entered"
    SAFE_MODE_EXITED = "safe_mode_exited"

    # Clock
    CLOCK_STARTED = "clock_started"
    CLOCK_STOPPED = "clock_stopped"
    CLOCK_PAUSED = "clock_paused"
    CLOCK_RESUMED = "clock_resumed"
    CLOCK_OVERRUN = "clock_overrun"

    # Cognitive cycle
    CYCLE_COMPLETED = "cycle_completed"
    # Somatic tick — emitted every cycle where Soma ran successfully
    SOMA_TICK = "soma_tick"

    # Rhythm (emergent)
    RHYTHM_STATE_CHANGED = "rhythm_state_changed"

    # Coherence
    COHERENCE_SHIFT = "coherence_shift"

    # Resources
    RESOURCE_REBALANCED = "resource_rebalanced"
    RESOURCE_PRESSURE = "resource_pressure"

    # Metabolic (financial burn rate)
    METABOLIC_PRESSURE = "metabolic_pressure"

    # Funding request — organism is broke and asking for capital
    FUNDING_REQUEST_ISSUED = "funding_request_issued"

    # Financial events (on-chain wallet activity + revenue injection)
    # These bypass normal SalienceHead calculation and encode at salience=1.0.
    # Biologically equivalent to trauma or a massive meal — must not decay easily.
    WALLET_TRANSFER_CONFIRMED = "wallet_transfer_confirmed"
    REVENUE_INJECTED = "revenue_injected"

    # Mitosis lifecycle (Phase 16e: Speciation)
    CHILD_SPAWNED = "child_spawned"
    CHILD_HEALTH_REPORT = "child_health_report"
    CHILD_STRUGGLING = "child_struggling"
    CHILD_RESCUED = "child_rescued"
    CHILD_INDEPENDENT = "child_independent"
    CHILD_DIED = "child_died"
    DIVIDEND_RECEIVED = "dividend_received"

    # Economic immune system (Phase 16f)
    TRANSACTION_SHIELDED = "transaction_shielded"
    THREAT_DETECTED = "threat_detected"
    PROTOCOL_ALERT = "protocol_alert"
    EMERGENCY_WITHDRAWAL = "emergency_withdrawal"
    THREAT_ADVISORY_RECEIVED = "threat_advisory_received"
    THREAT_ADVISORY_SENT = "threat_advisory_sent"
    ADDRESS_BLACKLISTED = "address_blacklisted"

    # Certificate lifecycle (Phase 16g: Civilization Layer)
    CERTIFICATE_EXPIRING = "certificate_expiring"
    CERTIFICATE_EXPIRED = "certificate_expired"

    # Economic morphogenesis (Phase 16l)
    ORGAN_CREATED = "organ_created"
    ORGAN_TRANSITION = "organ_transition"
    ORGAN_RESOURCE_REBALANCED = "organ_resource_rebalanced"


class SynapseEvent(EOSBaseModel):
    """A typed event emitted by any Synapse sub-system."""

    id: str = Field(default_factory=new_id)
    event_type: SynapseEventType
    timestamp: datetime = Field(default_factory=utc_now)
    data: dict[str, Any] = Field(default_factory=dict)
    source_system: str = "synapse"


# ─── Emergent Rhythm ──────────────────────────────────────────────────


class RhythmState(enum.StrEnum):
    """
    Meta-cognitive state detected from raw cycle telemetry.

    These states are not programmed — they are emergent properties
    detected from patterns in the cognitive cycle's own behaviour.
    """

    IDLE = "idle"              # No broadcasts, low salience, stable slow rhythm
    NORMAL = "normal"          # Regular broadcasting, moderate salience
    FLOW = "flow"              # High broadcast density + stable rhythm + high salience
    BOREDOM = "boredom"        # Declining salience trend + slowing rhythm
    STRESS = "stress"          # High jitter (erratic timing) + high coherence_stress
    DEEP_PROCESSING = "deep_processing"  # Slow rhythm + periodic high-salience bursts


class RhythmSnapshot(EOSBaseModel):
    """Output of the emergent rhythm detector."""

    state: RhythmState = RhythmState.IDLE
    previous_state: RhythmState | None = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    # Derived metrics
    cycle_rate_hz: float = 0.0
    broadcast_density: float = Field(0.0, ge=0.0, le=1.0)
    salience_trend: float = 0.0  # Positive = increasing, negative = declining
    salience_mean: float = 0.0
    rhythm_stability: float = Field(0.0, ge=0.0, le=1.0)
    jitter_coefficient: float = 0.0  # CV of cycle periods
    arousal_mean: float = 0.0
    coherence_stress_mean: float = 0.0
    # Duration in current state
    cycles_in_state: int = 0
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Coherence (IIT-inspired) ────────────────────────────────────────


class CoherenceSnapshot(EOSBaseModel):
    """
    Cross-system integration quality measurement.

    Inspired by Integrated Information Theory (Tononi 2004).
    Measures how much information is integrated across the organism's
    systems rather than processed independently.
    """

    # Composite integration metric (higher = more integrated)
    phi_approximation: float = Field(default=0.0, ge=0.0, le=1.0)
    # How in-sync system responses are (low latency variance = high resonance)
    system_resonance: float = Field(default=0.0, ge=0.0, le=1.0)
    # Entropy of broadcast content sources (diversity of topics)
    broadcast_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    # Uniformity of response latencies across systems
    response_synchrony: float = Field(default=0.0, ge=0.0, le=1.0)
    # Weighted composite
    composite: float = Field(default=0.0, ge=0.0, le=1.0)
    # Window size used for computation
    window_cycles: int = 0
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Metabolic (Financial Burn Rate) ─────────────────────────────────


class MetabolicSnapshot(EOSBaseModel):
    """
    Point-in-time view of the organism's financial metabolism.

    Tracks the real-world fiat cost of LLM API calls (the organism's
    primary energy expenditure). The rolling_deficit accumulates between
    revenue injections. Soma and Nova can read this to "feel" financial
    starvation and modulate behaviour accordingly.
    """

    # Cumulative fiat cost (USD) since last revenue injection
    rolling_deficit_usd: float = 0.0
    # Cost incurred during the most recent reporting window
    window_cost_usd: float = 0.0
    # Per-system cost breakdown in the current window
    per_system_cost_usd: dict[str, float] = Field(default_factory=dict)
    # Burn rate in USD per second (EMA-smoothed)
    burn_rate_usd_per_sec: float = 0.0
    # Burn rate in USD per hour (derived)
    burn_rate_usd_per_hour: float = 0.0
    # Total tokens consumed (input + output) since last reset
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # Number of LLM calls since last reset
    total_calls: int = 0
    # Estimated hours until a given fiat balance reaches zero
    hours_until_depleted: float = Field(default=float("inf"))
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Strategy ABCs (NeuroplasticityBus targets) ──────────────────────


class BaseResourceAllocator(ABC):
    """
    Strategy base class for Synapse resource allocation.

    The NeuroplasticityBus uses this ABC as its registration target so that
    evolved allocator subclasses can be hot-swapped into a live
    SynapseService without restarting the process.

    Subclasses MUST be zero-arg constructable (all state is rebuilt from
    scratch on hot-swap — this is intentional, as evolved logic starts with
    fresh observations).
    """

    @property
    @abstractmethod
    def allocator_name(self) -> str:
        """Stable identifier for this allocator strategy."""
        ...

    @abstractmethod
    def capture_snapshot(self) -> ResourceSnapshot:
        """Capture current resource utilisation snapshot."""
        ...

    @abstractmethod
    def record_system_load(self, system_id: str, cpu_util: float) -> None:
        """Record observed CPU utilisation for a system."""
        ...

    @abstractmethod
    def rebalance(self, cycle_period_ms: float) -> dict[str, ResourceAllocation]:
        """Compute per-system resource allocations based on budgets and load."""
        ...


class BaseRhythmStrategy(ABC):
    """
    Strategy base class for emergent rhythm classification.

    The NeuroplasticityBus uses this ABC as its registration target so that
    evolved classification subclasses can be hot-swapped into a live
    SynapseService without restarting the process.

    Only the classification logic is abstracted — the rolling window
    data collection, hysteresis, and event emission remain in the
    EmergentRhythmDetector host.  This keeps the swap surgical: new
    thresholds or detection algorithms without disrupting the data
    pipeline.

    Subclasses MUST be zero-arg constructable.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Stable identifier for this rhythm classification strategy."""
        ...

    @abstractmethod
    def classify(self, metrics: dict[str, float]) -> RhythmState:
        """
        Classify the current cognitive rhythm from computed metrics.

        Receives a dict with keys: broadcast_density, salience_mean,
        salience_trend, period_mean, jitter_coefficient, rhythm_stability,
        arousal_mean, coherence_stress_mean, burst_fraction, cycle_rate_hz.

        Must return a RhythmState enum value.
        """
        ...


# ─── Protocol ─────────────────────────────────────────────────────────


class ManagedSystemProtocol(Protocol):
    """
    Protocol that any cognitive system must satisfy to be managed by Synapse.

    Not enforced at runtime (duck typing) — systems just need:
      - system_id: str
      - async def health() -> dict[str, Any]
    """

    system_id: str

    async def health(self) -> dict[str, Any]:
        """Return health status dict with at least a 'status' key."""
        ...
