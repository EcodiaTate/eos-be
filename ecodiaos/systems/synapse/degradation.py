"""
EcodiaOS — Synapse Degradation Manager

Per-system graceful fallback strategies. When a system fails, Synapse
applies the appropriate degradation strategy: safe mode for critical
systems, queuing for Axon, raw fallback for Voxis, etc.

Auto-restart with exponential backoff ensures failed systems get
multiple recovery attempts before giving up.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.systems.synapse.types import (
    DegradationLevel,
    DegradationStrategy,
    SynapseEvent,
    SynapseEventType,
)

if TYPE_CHECKING:
    from ecodiaos.systems.synapse.event_bus import EventBus
    from ecodiaos.systems.synapse.health import HealthMonitor

logger = structlog.get_logger("ecodiaos.systems.synapse.degradation")


# ─── Per-System Strategies (from spec) ─────────────────────────────

_STRATEGIES: dict[str, DegradationStrategy] = {
    "equor": DegradationStrategy(
        system_id="equor",
        triggers_safe_mode=True,
        fallback_behavior="Enter safe mode. No actions without ethics.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "memory": DegradationStrategy(
        system_id="memory",
        triggers_safe_mode=True,
        fallback_behavior="Enter safe mode. Use in-context memory only.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "atune": DegradationStrategy(
        system_id="atune",
        triggers_safe_mode=True,
        fallback_behavior="Enter safe mode. Direct input passthrough, no salience scoring.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "nova": DegradationStrategy(
        system_id="nova",
        triggers_safe_mode=False,
        fallback_behavior="Voxis responds with 'I'm having difficulty thinking right now.'",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "voxis": DegradationStrategy(
        system_id="voxis",
        triggers_safe_mode=False,
        fallback_behavior="Use raw LLM output without personality rendering.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "axon": DegradationStrategy(
        system_id="axon",
        triggers_safe_mode=False,
        fallback_behavior="Queue intents, retry when Axon recovers.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "evo": DegradationStrategy(
        system_id="evo",
        triggers_safe_mode=False,
        fallback_behavior="Skip consolidation. No learning, but core function preserved.",
        auto_restart=True,
        max_restart_attempts=2,
    ),
    "simula": DegradationStrategy(
        system_id="simula",
        triggers_safe_mode=False,
        fallback_behavior="No evolution. Fully functional otherwise.",
        auto_restart=True,
        max_restart_attempts=2,
    ),
}


class DegradationManager:
    """
    Manages graceful degradation when cognitive systems fail.

    Each system has a defined fallback strategy. Critical systems
    (equor, memory, atune) trigger safe mode. Non-critical failures
    apply specific fallback behaviours and attempt auto-restart
    with exponential backoff.
    """

    def __init__(
        self,
        event_bus: EventBus,
        health_monitor: HealthMonitor,
    ) -> None:
        self._event_bus = event_bus
        self._health = health_monitor
        self._logger = logger.bind(component="degradation_manager")

        # Managed systems (for restart)
        self._systems: dict[str, Any] = {}

        # Restart tracking
        self._restart_attempts: dict[str, int] = {}
        self._restart_tasks: dict[str, asyncio.Task[None]] = {}

        # Current degradation level
        self._level: DegradationLevel = DegradationLevel.NOMINAL

    # ─── System Registration ─────────────────────────────────────────

    def register_system(self, system: Any) -> None:
        """Register a system for potential restart management."""
        sid = getattr(system, "system_id", None)
        if sid:
            self._systems[sid] = system

    # ─── Failure Handling ────────────────────────────────────────────

    async def handle_failure(self, system_id: str) -> None:
        """
        Apply the degradation strategy for a failed system.

        1. Look up the strategy
        2. Log the fallback behaviour
        3. Attempt auto-restart with exponential backoff
        4. Update the overall degradation level
        """
        strategy = _STRATEGIES.get(system_id)
        if strategy is None:
            self._logger.warning(
                "no_degradation_strategy",
                system_id=system_id,
            )
            return

        self._logger.warning(
            "applying_degradation_strategy",
            system_id=system_id,
            fallback=strategy.fallback_behavior,
            triggers_safe_mode=strategy.triggers_safe_mode,
        )

        # Update level
        self._update_level()

        # Auto-restart if configured
        if strategy.auto_restart:
            attempts = self._restart_attempts.get(system_id, 0)
            if attempts < strategy.max_restart_attempts:
                self._schedule_restart(system_id, strategy, attempts)
            else:
                self._logger.error(
                    "max_restart_attempts_reached",
                    system_id=system_id,
                    attempts=attempts,
                )

    async def record_recovery(self, system_id: str) -> None:
        """Record that a system has recovered. Reset restart counter."""
        self._restart_attempts[system_id] = 0

        # Cancel any pending restart task
        task = self._restart_tasks.pop(system_id, None)
        if task and not task.done():
            task.cancel()

        self._update_level()
        self._logger.info("recovery_recorded", system_id=system_id)

    # ─── Auto-Restart ────────────────────────────────────────────────

    def _schedule_restart(
        self,
        system_id: str,
        strategy: DegradationStrategy,
        attempt: int,
    ) -> None:
        """Schedule an auto-restart with exponential backoff."""
        # Exponential backoff: base * 2^attempt
        delay_s = strategy.restart_backoff_base_s * (2 ** attempt)

        self._logger.info(
            "scheduling_restart",
            system_id=system_id,
            attempt=attempt + 1,
            max_attempts=strategy.max_restart_attempts,
            delay_s=delay_s,
        )

        task = asyncio.create_task(
            self._restart_system(system_id, attempt, delay_s),
            name=f"restart_{system_id}_{attempt}",
        )
        self._restart_tasks[system_id] = task

    async def _restart_system(
        self,
        system_id: str,
        attempt: int,
        delay_s: float,
    ) -> None:
        """Wait for backoff, then attempt to restart the system."""
        try:
            await asyncio.sleep(delay_s)

            system = self._systems.get(system_id)
            if system is None:
                self._logger.warning("restart_no_system_ref", system_id=system_id)
                return

            self._restart_attempts[system_id] = attempt + 1

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.SYSTEM_RESTARTING,
                data={
                    "system_id": system_id,
                    "attempt": attempt + 1,
                },
            ))

            # Attempt shutdown then re-initialize
            try:
                if hasattr(system, "shutdown"):
                    await system.shutdown()
            except Exception:
                pass  # Shutdown may fail on a broken system

            if hasattr(system, "initialize"):
                await system.initialize()
                self._logger.info(
                    "system_restarted",
                    system_id=system_id,
                    attempt=attempt + 1,
                )
            else:
                self._logger.warning(
                    "system_no_initialize",
                    system_id=system_id,
                )

        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._logger.error(
                "restart_failed",
                system_id=system_id,
                attempt=attempt + 1,
                error=str(exc),
            )

    # ─── Level Computation ───────────────────────────────────────────

    def _update_level(self) -> None:
        """Recompute the overall degradation level from system health."""
        if self._health.is_safe_mode:
            self._level = DegradationLevel.SAFE_MODE
        elif len(self._health.failed_systems) > 0:
            self._level = DegradationLevel.DEGRADED
        else:
            self._level = DegradationLevel.NOMINAL

    @property
    def level(self) -> DegradationLevel:
        self._update_level()
        return self._level

    def get_strategy(self, system_id: str) -> DegradationStrategy | None:
        return _STRATEGIES.get(system_id)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "level": self._level.value,
            "restart_attempts": dict(self._restart_attempts),
            "active_restart_tasks": [
                sid for sid, t in self._restart_tasks.items()
                if not t.done()
            ],
        }
