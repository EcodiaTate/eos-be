"""
EcodiaOS — Synapse Health Monitor

Background 5-second polling of all managed cognitive systems.
Three consecutive missed heartbeats → system declared failed.
Critical system failure (equor, memory, atune) → safe mode.

Health monitoring is the immune system of the organism. It detects failures,
triggers degradation strategies, and coordinates recovery.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.systems.synapse.types import (
    SynapseEvent,
    SynapseEventType,
    SystemHealthRecord,
    SystemStatus,
)

if TYPE_CHECKING:
    from ecodiaos.config import SynapseConfig
    from ecodiaos.systems.synapse.degradation import DegradationManager
    from ecodiaos.systems.synapse.event_bus import EventBus

logger = structlog.get_logger("ecodiaos.systems.synapse.health")

# Health check timeout per system (seconds)
_HEALTH_CHECK_TIMEOUT_S: float = 2.0

# Critical systems — failure triggers safe mode
_CRITICAL_SYSTEMS: frozenset[str] = frozenset({"equor", "memory", "atune"})


class HealthMonitor:
    """
    Monitors the health of all registered cognitive systems via periodic
    heartbeat polling. Detects failures, triggers degradation, and
    coordinates recovery.

    The monitor runs as a background asyncio task, polling every
    health_check_interval_ms (default 5000ms).
    """

    def __init__(
        self,
        config: SynapseConfig,
        event_bus: EventBus,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._logger = logger.bind(component="health_monitor")

        # Managed systems (duck-typed: system_id + async health())
        self._systems: dict[str, Any] = {}
        # Per-system health records
        self._records: dict[str, SystemHealthRecord] = {}

        # Safe mode state
        self._safe_mode: bool = False
        self._safe_mode_reason: str = ""

        # Degradation manager (wired after construction)
        self._degradation: DegradationManager | None = None

        # Background task
        self._task: asyncio.Task[None] | None = None
        self._running: bool = False

        # Metrics
        self._total_checks: int = 0
        self._total_failures_detected: int = 0
        self._total_recoveries: int = 0
        self._restart_count: int = 0

    # ─── Registration ────────────────────────────────────────────────

    def register(self, system: Any) -> None:
        """
        Register a cognitive system for health monitoring.

        The system must have:
          - system_id: str
          - async health() -> dict[str, Any]
        """
        sid = getattr(system, "system_id", None)
        if sid is None:
            raise ValueError(f"System {system} has no system_id attribute")

        self._systems[sid] = system
        self._records[sid] = SystemHealthRecord(
            system_id=sid,
            status=SystemStatus.STARTING,
            is_critical=sid in _CRITICAL_SYSTEMS,
        )
        self._logger.info("system_registered", system_id=sid, is_critical=sid in _CRITICAL_SYSTEMS)

    def set_degradation_manager(self, degradation: DegradationManager) -> None:
        """Wire the degradation manager after construction."""
        self._degradation = degradation

    # ─── Control ─────────────────────────────────────────────────────

    def start(self) -> asyncio.Task[None]:
        """Start the background health monitoring loop."""
        if self._running:
            raise RuntimeError("HealthMonitor is already running")
        self._running = True
        self._task = asyncio.create_task(
            self._monitor_loop(),
            name="synapse_health_monitor",
        )
        self._logger.info(
            "health_monitor_started",
            interval_ms=self._config.health_check_interval_ms,
            systems=list(self._systems.keys()),
        )
        return self._task

    async def stop(self) -> None:
        """Stop the health monitoring loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._logger.info(
            "health_monitor_stopped",
            total_checks=self._total_checks,
            failures_detected=self._total_failures_detected,
            recoveries=self._total_recoveries,
        )

    # ─── Safe Mode ───────────────────────────────────────────────────

    @property
    def is_safe_mode(self) -> bool:
        return self._safe_mode

    @property
    def safe_mode_reason(self) -> str:
        return self._safe_mode_reason

    async def set_safe_mode(self, enabled: bool, reason: str = "") -> None:
        """Manually toggle safe mode (for admin API)."""
        if enabled and not self._safe_mode:
            await self._enter_safe_mode(reason or "manual_admin_toggle")
        elif not enabled and self._safe_mode:
            await self._exit_safe_mode()

    # ─── State ───────────────────────────────────────────────────────

    def get_record(self, system_id: str) -> SystemHealthRecord | None:
        return self._records.get(system_id)

    def get_all_records(self) -> dict[str, SystemHealthRecord]:
        return dict(self._records)

    @property
    def healthy_count(self) -> int:
        return sum(
            1 for r in self._records.values()
            if r.status == SystemStatus.HEALTHY
        )

    @property
    def failed_systems(self) -> list[str]:
        return [
            r.system_id for r in self._records.values()
            if r.status == SystemStatus.FAILED
        ]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "safe_mode": self._safe_mode,
            "safe_mode_reason": self._safe_mode_reason,
            "total_checks": self._total_checks,
            "failures_detected": self._total_failures_detected,
            "recoveries": self._total_recoveries,
            "restarts": self._restart_count,
            "systems_healthy": self.healthy_count,
            "systems_failed": len(self.failed_systems),
            "per_system": {
                sid: {
                    "status": r.status.value,
                    "consecutive_misses": r.consecutive_misses,
                    "latency_ema_ms": round(r.latency_ema_ms, 2),
                    "total_failures": r.total_failures,
                    "restart_count": r.restart_count,
                }
                for sid, r in self._records.items()
            },
        }

    # ─── Monitor Loop ────────────────────────────────────────────────

    async def _monitor_loop(self) -> None:
        """Background polling loop. Runs until stopped."""
        interval_s = self._config.health_check_interval_ms / 1000.0

        while self._running:
            try:
                await self._check_all_systems()
                await asyncio.sleep(interval_s)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._logger.error("health_monitor_error", error=str(exc))
                await asyncio.sleep(interval_s)

    async def _check_all_systems(self) -> None:
        """Run health checks on all registered systems in parallel."""
        if not self._systems:
            return

        # Launch all health checks concurrently
        tasks = {
            sid: asyncio.create_task(
                self._check_system(sid, system),
                name=f"health_check_{sid}",
            )
            for sid, system in self._systems.items()
        }

        # Wait for all to complete (each has its own timeout)
        await asyncio.gather(*tasks.values(), return_exceptions=True)

    async def _check_system(self, system_id: str, system: Any) -> None:
        """Check a single system's health."""
        record = self._records[system_id]
        self._total_checks += 1

        t0 = time.monotonic()
        try:
            health_result = await asyncio.wait_for(
                system.health(),
                timeout=_HEALTH_CHECK_TIMEOUT_S,
            )
            latency_ms = (time.monotonic() - t0) * 1000.0

            status = health_result.get("status", "healthy") if isinstance(health_result, dict) else "healthy"

            if status == "healthy":
                was_failed = record.status == SystemStatus.FAILED
                # Detect overloaded: latency > 2x the EMA (if we have history)
                if record.latency_ema_ms > 0 and latency_ms > record.latency_ema_ms * 2:
                    record.record_overloaded(latency_ms)
                else:
                    record.record_success(latency_ms)

                # Recovery detection
                if was_failed and record.status == SystemStatus.HEALTHY:
                    await self._handle_recovery(system_id)
            else:
                # System reported non-healthy status
                record.record_failure()
                if record.consecutive_misses >= self._config.health_failure_threshold:
                    await self._handle_failure(system_id)

        except TimeoutError:
            record.record_failure()
            self._logger.warning(
                "health_check_timeout",
                system_id=system_id,
                consecutive_misses=record.consecutive_misses,
            )
            if record.consecutive_misses >= self._config.health_failure_threshold:
                await self._handle_failure(system_id)

        except Exception as exc:
            record.record_failure()
            self._logger.warning(
                "health_check_error",
                system_id=system_id,
                error=str(exc),
                consecutive_misses=record.consecutive_misses,
            )
            if record.consecutive_misses >= self._config.health_failure_threshold:
                await self._handle_failure(system_id)

    # ─── Failure & Recovery ──────────────────────────────────────────

    async def _handle_failure(self, system_id: str) -> None:
        """Handle a confirmed system failure."""
        record = self._records[system_id]
        if record.status == SystemStatus.FAILED:
            return  # Already handling this failure

        record.status = SystemStatus.FAILED
        self._total_failures_detected += 1

        self._logger.error(
            "system_declared_failed",
            system_id=system_id,
            consecutive_misses=record.consecutive_misses,
            is_critical=record.is_critical,
        )

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.SYSTEM_FAILED,
            data={
                "system_id": system_id,
                "consecutive_misses": record.consecutive_misses,
                "is_critical": record.is_critical,
            },
        ))

        # Critical system failure → safe mode
        if record.is_critical:
            await self._enter_safe_mode(f"{system_id}_failure")

        # Attempt restart via degradation manager
        if self._degradation is not None:
            await self._degradation.handle_failure(system_id)

    async def _handle_recovery(self, system_id: str) -> None:
        """Handle a system recovery after failure."""
        self._total_recoveries += 1

        self._logger.info(
            "system_recovered",
            system_id=system_id,
        )

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.SYSTEM_RECOVERED,
            data={"system_id": system_id},
        ))

        if self._degradation is not None:
            await self._degradation.record_recovery(system_id)

        # Check if we can exit safe mode
        if self._safe_mode:
            await self._check_safe_mode_exit()

    async def _enter_safe_mode(self, reason: str) -> None:
        """Enter safe mode — no autonomous actions permitted."""
        if self._safe_mode:
            return

        self._safe_mode = True
        self._safe_mode_reason = reason
        self._logger.critical("safe_mode_entered", reason=reason)

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.SAFE_MODE_ENTERED,
            data={"reason": reason},
        ))

    async def _exit_safe_mode(self) -> None:
        """Exit safe mode — all critical systems are healthy again."""
        if not self._safe_mode:
            return

        self._safe_mode = False
        reason = self._safe_mode_reason
        self._safe_mode_reason = ""
        self._logger.info("safe_mode_exited", previous_reason=reason)

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.SAFE_MODE_EXITED,
            data={"previous_reason": reason},
        ))

    async def _check_safe_mode_exit(self) -> None:
        """Check if all critical systems are healthy and we can exit safe mode."""
        for sid in _CRITICAL_SYSTEMS:
            record = self._records.get(sid)
            if record and record.status != SystemStatus.HEALTHY:
                return  # At least one critical system is still unhealthy
        await self._exit_safe_mode()
