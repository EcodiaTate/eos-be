"""
EcodiaOS — Synapse Service

The autonomic nervous system. Synapse is the heartbeat of EOS — it drives
the cognitive cycle clock, monitors system health, allocates resources,
detects emergent cognitive rhythms, and measures cross-system coherence.

Synapse is invisible when it works. It is the heartbeat, the circulation,
the autonomic regulation that keeps everything alive. You don't notice
your nervous system until it fails — and Synapse is designed never to fail.

Zero LLM tokens consumed. Pure computation, monitoring, coordination.

Lifecycle:
  initialize()          — build all sub-systems
  register_system()     — register a cognitive system for management
  start_clock()         — start the theta rhythm
  start_health_monitor()— start background health polling
  stop()                — graceful shutdown
  health()              — self-health report

The _on_cycle callback (called by CognitiveClock after every tick):
  1. Feed CycleResult into EmergentRhythmDetector (every cycle)
  2. Feed broadcast data into CoherenceMonitor (every cycle)
  3. Trigger CoherenceMonitor.compute() (every 50 cycles)
  4. Trigger ResourceAllocator.capture_snapshot() (every 33 cycles)
  5. Record telemetry to MetricCollector
  6. Emit CYCLE_COMPLETED event to Redis for Alive
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.systems.synapse.clock import CognitiveClock
from ecodiaos.systems.synapse.coherence import CoherenceMonitor
from ecodiaos.systems.synapse.degradation import DegradationManager
from ecodiaos.systems.synapse.event_bus import EventBus
from ecodiaos.systems.synapse.health import HealthMonitor
from ecodiaos.systems.synapse.resources import ResourceAllocator
from ecodiaos.systems.synapse.rhythm import EmergentRhythmDetector
from ecodiaos.systems.synapse.types import (
    ClockState,
    CoherenceSnapshot,
    CycleResult,
    RhythmSnapshot,
    SynapseEvent,
    SynapseEventType,
)

if TYPE_CHECKING:
    from ecodiaos.clients.redis import RedisClient
    from ecodiaos.config import SynapseConfig
    from ecodiaos.systems.atune.service import AtuneService
    from ecodiaos.telemetry.metrics import MetricCollector

logger = structlog.get_logger("ecodiaos.systems.synapse")

# How often to compute coherence (in cycles)
_COHERENCE_INTERVAL: int = 50

# How often to capture a resource snapshot (in cycles)
_RESOURCE_SNAPSHOT_INTERVAL: int = 33

# How often to rebalance resource allocations (in cycles)
_REBALANCE_INTERVAL: int = 100


class SynapseService:
    """
    Synapse — the EOS autonomic nervous system.

    Coordinates six sub-systems:
      CognitiveClock          — theta rhythm driving Atune
      HealthMonitor           — background health polling
      ResourceAllocator       — adaptive resource budgets
      DegradationManager      — graceful fallback on failure
      EmergentRhythmDetector  — meta-cognitive state detection
      CoherenceMonitor        — cross-system integration quality
      EventBus                — dual-output event publication
    """

    system_id: str = "synapse"

    def __init__(
        self,
        atune: AtuneService,
        config: SynapseConfig,
        redis: RedisClient | None = None,
        metrics: MetricCollector | None = None,
    ) -> None:
        self._atune = atune
        self._config = config
        self._redis = redis
        self._metrics = metrics
        self._logger = logger.bind(system="synapse")
        self._initialized: bool = False

        # Sub-systems
        self._event_bus = EventBus(redis=redis)
        self._clock = CognitiveClock(atune=atune, config=config)
        self._health = HealthMonitor(config=config, event_bus=self._event_bus)
        self._resources = ResourceAllocator()
        self._degradation = DegradationManager(
            event_bus=self._event_bus,
            health_monitor=self._health,
        )
        self._rhythm = EmergentRhythmDetector(event_bus=self._event_bus)
        self._coherence = CoherenceMonitor(event_bus=self._event_bus)

        # Cycle counter for periodic sub-system triggers
        self._cycle_count: int = 0

    # ─── Lifecycle ───────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Build all sub-systems and wire inter-dependencies."""
        if self._initialized:
            return

        # Wire health → degradation
        self._health.set_degradation_manager(self._degradation)

        # Set the per-cycle callback on the clock
        self._clock.set_on_cycle(self._on_cycle)

        self._initialized = True
        self._logger.info("synapse_initialized")

    def set_soma(self, soma: Any) -> None:
        """Wire Soma service into the clock (step 0 of theta cycle)."""
        self._clock.set_soma(soma)

    def register_system(self, system: Any) -> None:
        """
        Register a cognitive system for health monitoring and degradation.

        The system must have:
          - system_id: str
          - async health() -> dict[str, Any]
        """
        self._health.register(system)
        self._degradation.register_system(system)
        # Update coherence monitor with total system count
        self._coherence.set_total_systems(len(self._health.get_all_records()))

    async def start_clock(self) -> None:
        """Start the cognitive cycle clock (the heartbeat)."""
        if not self._initialized:
            raise RuntimeError("SynapseService.initialize() must be called first")

        self._clock.start()

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.CLOCK_STARTED,
            data={"period_ms": self._config.cycle_period_ms},
        ))

        self._logger.info(
            "clock_started",
            period_ms=self._config.cycle_period_ms,
            min_ms=self._config.min_cycle_period_ms,
            max_ms=self._config.max_cycle_period_ms,
        )

    async def start_health_monitor(self) -> None:
        """Start the background health monitoring loop."""
        if not self._initialized:
            raise RuntimeError("SynapseService.initialize() must be called first")

        self._health.start()
        self._logger.info(
            "health_monitor_started",
            interval_ms=self._config.health_check_interval_ms,
        )

    async def stop(self) -> None:
        """Graceful shutdown of all sub-systems."""
        self._logger.info("synapse_stopping")

        await self._clock.stop()
        await self._health.stop()

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.CLOCK_STOPPED,
            data={"total_cycles": self._cycle_count},
        ))

        self._logger.info(
            "synapse_stopped",
            total_cycles=self._cycle_count,
            rhythm_state=self._rhythm.current_state.value,
            coherence=self._coherence.latest.composite,
        )

    # ─── Health ──────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Self-health report (implements ManagedSystem protocol)."""
        return {
            "status": "healthy" if self._initialized else "starting",
            "cycle_count": self._cycle_count,
            "safe_mode": self._health.is_safe_mode,
            "rhythm_state": self._rhythm.current_state.value,
            "coherence_composite": self._coherence.latest.composite,
        }

    # ─── Safe Mode ───────────────────────────────────────────────────

    @property
    def is_safe_mode(self) -> bool:
        return self._health.is_safe_mode

    async def set_safe_mode(self, enabled: bool, reason: str = "") -> None:
        """Manually toggle safe mode (admin API)."""
        await self._health.set_safe_mode(enabled, reason)
        if enabled:
            self._clock.pause()
        else:
            self._clock.resume()

    # ─── Accessors ───────────────────────────────────────────────────

    @property
    def clock_state(self) -> ClockState:
        return self._clock.state

    @property
    def rhythm_snapshot(self) -> RhythmSnapshot:
        return self._rhythm._build_snapshot()

    @property
    def coherence_snapshot(self) -> CoherenceSnapshot:
        return self._coherence.latest

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    # ─── Stats ───────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "cycle_count": self._cycle_count,
            "clock": self._clock.state.model_dump(),
            "health": self._health.stats,
            "degradation": self._degradation.stats,
            "resources": self._resources.stats,
            "rhythm": self._rhythm.stats,
            "coherence": self._coherence.stats,
            "event_bus": self._event_bus.stats,
        }

    # ─── Per-Cycle Callback ──────────────────────────────────────────

    async def _on_cycle(self, result: CycleResult) -> None:
        """
        Called by CognitiveClock after every theta tick.

        This is the central integration point where all sub-systems
        are fed cycle telemetry. The work here must be lightweight —
        heavy computation is deferred to periodic triggers.
        """
        self._cycle_count = result.cycle_number

        # ── 1. Feed rhythm detector (every cycle) ──
        coherence_stress = 0.0
        with contextlib.suppress(Exception):
            coherence_stress = self._atune.current_affect.coherence_stress

        await self._rhythm.update(result, coherence_stress=coherence_stress)

        # Push rhythm state to Atune so meta-attention can modulate salience
        # weights based on the organism's emergent cognitive state.
        try:
            self._atune.set_rhythm_state(self._rhythm.current_state.value)
        except Exception:
            pass  # Non-critical — meta-attention falls back to "normal"

        # ── 2. Feed coherence monitor (every cycle) ──
        source = ""
        if result.had_broadcast and result.broadcast_id:
            source = result.broadcast_id[:8]  # Use broadcast ID prefix as source proxy

        self._coherence.record_broadcast(
            source=source,
            salience=result.salience_composite,
            had_content=result.had_broadcast,
        )

        # ── 3. Periodic: compute coherence → adapt clock ──
        if self._cycle_count % _COHERENCE_INTERVAL == 0:
            snapshot = await self._coherence.compute()

            # Use coherence to modulate clock speed: low coherence → slow
            # down to give systems time to resynchronize.
            if snapshot is not None:
                # Activate drag when composite drops below 0.4
                if snapshot.composite < 0.4:
                    drag = (0.4 - snapshot.composite) / 0.4  # 0→1 as coherence drops
                    self._clock.set_coherence_drag(drag)
                else:
                    self._clock.set_coherence_drag(0.0)

        # ── 4. Periodic: resource snapshot ──
        if self._cycle_count % _RESOURCE_SNAPSHOT_INTERVAL == 0:
            self._resources.capture_snapshot()

        # ── 5. Periodic: rebalance resources ──
        if self._cycle_count % _REBALANCE_INTERVAL == 0:
            self._resources.rebalance(self._clock.state.current_period_ms)

        # ── 6. Record telemetry ──
        if self._metrics is not None:
            try:
                await self._metrics.record(
                    "synapse", "cycle.latency_ms", result.elapsed_ms,
                )
                await self._metrics.record(
                    "synapse", "cycle.period_ms", result.budget_ms,
                )
                await self._metrics.record(
                    "synapse", "cycle.arousal", result.arousal,
                )
                if result.had_broadcast:
                    await self._metrics.record(
                        "synapse", "cycle.salience", result.salience_composite,
                    )
            except Exception:
                pass  # Telemetry failures must never block the cycle

        # ── 7. Emit cycle event to Redis for Alive ──
        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.CYCLE_COMPLETED,
            data={
                "cycle": result.cycle_number,
                "elapsed_ms": result.elapsed_ms,
                "period_ms": result.budget_ms,
                "arousal": result.arousal,
                "had_broadcast": result.had_broadcast,
                "salience": result.salience_composite,
                "rhythm": self._rhythm.current_state.value,
            },
        ))
