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
from ecodiaos.systems.synapse.metabolism import MetabolicTracker
from ecodiaos.systems.synapse.resources import ResourceAllocator
from ecodiaos.systems.synapse.rhythm import EmergentRhythmDetector
from ecodiaos.systems.synapse.types import (
    BaseResourceAllocator,
    BaseRhythmStrategy,
    ClockState,
    CoherenceSnapshot,
    CycleResult,
    MetabolicSnapshot,
    RhythmSnapshot,
    SomaTickEvent,
    SynapseEvent,
    SynapseEventType,
)

if TYPE_CHECKING:
    from ecodiaos.clients.redis import RedisClient
    from ecodiaos.config import SynapseConfig
    from ecodiaos.core.hotreload import NeuroplasticityBus
    from ecodiaos.systems.atune.service import AtuneService
    from ecodiaos.telemetry.metrics import MetricCollector

logger = structlog.get_logger("ecodiaos.systems.synapse")

# How often to compute coherence (in cycles)
_COHERENCE_INTERVAL: int = 50

# How often to capture a resource snapshot (in cycles)
_RESOURCE_SNAPSHOT_INTERVAL: int = 33

# How often to rebalance resource allocations (in cycles)
_REBALANCE_INTERVAL: int = 100

# How often to snapshot metabolic state and emit pressure events (in cycles)
_METABOLIC_INTERVAL: int = 50

# Burn rate threshold (USD/hour) above which METABOLIC_PRESSURE fires
_METABOLIC_PRESSURE_THRESHOLD_USD_HR: float = 1.0


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
        neuroplasticity_bus: NeuroplasticityBus | None = None,
    ) -> None:
        self._atune = atune
        self._config = config
        self._redis = redis
        self._metrics = metrics
        self._neuroplasticity_bus = neuroplasticity_bus
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
        self._metabolism = MetabolicTracker()

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

        # Register with NeuroplasticityBus for hot-reload of allocators & rhythm strategies
        if self._neuroplasticity_bus is not None:
            self._neuroplasticity_bus.register(
                base_class=BaseResourceAllocator,
                registration_callback=self._on_allocator_evolved,
                system_id="synapse",
            )
            self._neuroplasticity_bus.register(
                base_class=BaseRhythmStrategy,
                registration_callback=self._on_rhythm_strategy_evolved,
                system_id="synapse",
            )

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

        # Deregister from NeuroplasticityBus
        if self._neuroplasticity_bus is not None:
            self._neuroplasticity_bus.deregister(BaseResourceAllocator)
            self._neuroplasticity_bus.deregister(BaseRhythmStrategy)

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
            "metabolic_deficit_usd": round(self._metabolism.rolling_deficit_usd, 6),
            "burn_rate_usd_per_hour": round(self._metabolism.burn_rate_usd_per_hour, 4),
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

    # ─── Clock Control (admin API) ───────────────────────────────────

    def pause_clock(self) -> None:
        """Pause the cognitive cycle clock (e.g., for maintenance)."""
        self._clock.pause()

    def resume_clock(self) -> None:
        """Resume the cognitive cycle clock."""
        self._clock.resume()

    def set_clock_speed(self, hz: float) -> None:
        """Override base clock frequency (1–20 Hz)."""
        self._clock.set_speed(hz)

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

    @property
    def metabolic_snapshot(self) -> MetabolicSnapshot:
        return self._metabolism.snapshot()

    @property
    def metabolic_deficit(self) -> float:
        """Current rolling deficit in USD — how much the organism owes."""
        return self._metabolism.rolling_deficit_usd

    @property
    def metabolism(self) -> MetabolicTracker:
        """Direct access for callers that need log_usage or inject_revenue."""
        return self._metabolism


    async def inject_revenue(
        self,
        amount_usd: float,
        source: str = "external",
    ) -> None:
        """
        Record incoming revenue and emit REVENUE_INJECTED on the event bus.

        This is the preferred entry point for revenue injections. It keeps
        the MetabolicTracker updated AND ensures Memory encodes the event as
        a salience=1.0 episode so the organism learns what actions lead to income.

        Args:
            amount_usd: Revenue amount in USD.
            source: Human-readable label for the revenue origin
                    (e.g. "stripe", "on-chain-fee", "client-payment").
        """
        from ecodiaos.systems.synapse.types import SynapseEvent, SynapseEventType

        self._metabolism.inject_revenue(amount_usd)

        event = SynapseEvent(
            event_type=SynapseEventType.REVENUE_INJECTED,
            data={
                "amount_usd": round(amount_usd, 8),
                "source": source,
                "new_deficit_usd": round(self._metabolism.rolling_deficit_usd, 6),
            },
            source_system="synapse",
        )
        try:
            await self._event_bus.emit(event)
        except Exception as exc:
            logger.error("revenue_injected_event_emit_failed", error=str(exc))
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
            "metabolism": self._metabolism.stats,
            "event_bus": self._event_bus.stats,
        }

    # ─── NeuroplasticityBus callbacks ────────────────────────────────

    def _on_allocator_evolved(self, new_allocator: BaseResourceAllocator) -> None:
        """
        Hot-swap the resource allocator when Simula evolves a new one.

        The old allocator's accumulated load observations are intentionally
        discarded — evolved logic starts with fresh observations.  The swap
        happens between cycles so the active theta tick is never disrupted.
        """
        old_name = self._resources.allocator_name if isinstance(self._resources, BaseResourceAllocator) else "unknown"
        self._resources = new_allocator
        self._logger.info(
            "resource_allocator_evolved",
            old=old_name,
            new=new_allocator.allocator_name,
        )

    def _on_rhythm_strategy_evolved(self, new_strategy: BaseRhythmStrategy) -> None:
        """
        Hot-swap the rhythm classification strategy.

        The EmergentRhythmDetector's rolling window and hysteresis state
        are preserved — only the classification algorithm changes.  This
        means the new strategy immediately has a full 100-cycle window
        of data to classify against, rather than starting cold.
        """
        self._rhythm.set_strategy(new_strategy)
        self._logger.info(
            "rhythm_strategy_evolved",
            new=new_strategy.strategy_name,
        )

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

        # ── 5b. Periodic: metabolic snapshot + pressure event ──
        if self._cycle_count % _METABOLIC_INTERVAL == 0:
            meta_snap = self._metabolism.snapshot()
            if meta_snap.burn_rate_usd_per_hour > _METABOLIC_PRESSURE_THRESHOLD_USD_HR:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.METABOLIC_PRESSURE,
                    data={
                        "rolling_deficit_usd": meta_snap.rolling_deficit_usd,
                        "burn_rate_usd_per_hour": meta_snap.burn_rate_usd_per_hour,
                        "total_calls": meta_snap.total_calls,
                        "per_system_cost_usd": meta_snap.per_system_cost_usd,
                    },
                ))
            # Reset window accumulator so next interval is a clean delta
            self._metabolism.reset_window()

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
        cycle_data: dict[str, Any] = {
            "cycle": result.cycle_number,
            "elapsed_ms": result.elapsed_ms,
            "period_ms": result.budget_ms,
            "arousal": result.arousal,
            "had_broadcast": result.had_broadcast,
            "salience": result.salience_composite,
            "rhythm": self._rhythm.current_state.value,
            "metabolic_deficit_usd": self._metabolism.rolling_deficit_usd,
            "burn_rate_usd_per_hour": round(
                self._metabolism.burn_rate_usd_per_hour, 6,
            ),
        }
        if result.somatic is not None:
            cycle_data["soma"] = {
                "urgency": result.somatic.urgency,
                "dominant_error": result.somatic.dominant_error,
                "arousal_sensed": result.somatic.arousal_sensed,
                "energy_sensed": result.somatic.energy_sensed,
                "nearest_attractor": result.somatic.nearest_attractor,
                "trajectory_heading": result.somatic.trajectory_heading,
            }

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.CYCLE_COMPLETED,
            data=cycle_data,
        ))

        # ── 8. Emit SomaTickEvent for stateless consumers ──
        if result.somatic is not None:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.SOMA_TICK,
                data=SomaTickEvent(
                    cycle_number=result.cycle_number,
                    somatic_state=result.somatic,
                ).model_dump(),
            ))
