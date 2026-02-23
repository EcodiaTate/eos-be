"""
EcodiaOS — Synapse Type Definitions

All data types for the autonomic nervous system: cycle clock, health monitoring,
resource allocation, degradation strategies, emergent rhythm detection,
and cross-system coherence measurement.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from ecodiaos.primitives.common import EOSBaseModel, new_id, utc_now


# ─── System Status ────────────────────────────────────────────────────


class SystemStatus(str, enum.Enum):
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
        if self.status == SystemStatus.FAILED and self.consecutive_successes >= 3:
            self.status = SystemStatus.HEALTHY
        elif self.status in (SystemStatus.DEGRADED, SystemStatus.OVERLOADED):
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


class DegradationLevel(str, enum.Enum):
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


class SynapseEventType(str, enum.Enum):
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

    # Rhythm (emergent)
    RHYTHM_STATE_CHANGED = "rhythm_state_changed"

    # Coherence
    COHERENCE_SHIFT = "coherence_shift"

    # Resources
    RESOURCE_REBALANCED = "resource_rebalanced"
    RESOURCE_PRESSURE = "resource_pressure"


class SynapseEvent(EOSBaseModel):
    """A typed event emitted by any Synapse sub-system."""

    id: str = Field(default_factory=new_id)
    event_type: SynapseEventType
    timestamp: datetime = Field(default_factory=utc_now)
    data: dict[str, Any] = Field(default_factory=dict)
    source_system: str = "synapse"


# ─── Emergent Rhythm ──────────────────────────────────────────────────


class RhythmState(str, enum.Enum):
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
    phi_approximation: float = Field(0.0, ge=0.0, le=1.0)
    # How in-sync system responses are (low latency variance = high resonance)
    system_resonance: float = Field(0.0, ge=0.0, le=1.0)
    # Entropy of broadcast content sources (diversity of topics)
    broadcast_diversity: float = Field(0.0, ge=0.0, le=1.0)
    # Uniformity of response latencies across systems
    response_synchrony: float = Field(0.0, ge=0.0, le=1.0)
    # Weighted composite
    composite: float = Field(0.0, ge=0.0, le=1.0)
    # Window size used for computation
    window_cycles: int = 0
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Protocol ─────────────────────────────────────────────────────────


class ManagedSystemProtocol:
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
