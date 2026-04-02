"""
Tests for HealthMonitor - verify that degraded systems are not declared failed.

Regression test for: Equor reporting 'degraded' status (drift_severity >= 0.5)
was being treated as a health check failure, which after 3 consecutive checks
caused equor to be declared FAILED → safe_mode → organism unresponsive.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from systems.synapse.health import HealthMonitor, _ALIVE_STATUSES
from systems.synapse.types import SystemStatus


class FakeSystem:
    """Minimal system stub for health monitor tests."""

    def __init__(self, system_id: str, health_response: dict[str, Any] | None = None):
        self.system_id = system_id
        self._health_response = health_response or {"status": "healthy"}

    async def health(self) -> dict[str, Any]:
        return self._health_response

    def set_health(self, response: dict[str, Any]) -> None:
        self._health_response = response


@pytest.fixture
def config():
    cfg = MagicMock()
    cfg.health_check_interval_ms = 5000
    cfg.health_failure_threshold = 3
    return cfg


@pytest.fixture
def event_bus():
    bus = MagicMock()
    bus.emit = AsyncMock()
    return bus


@pytest.fixture
def monitor(config, event_bus):
    return HealthMonitor(config, event_bus)


@pytest.mark.asyncio
async def test_degraded_status_does_not_trigger_failure(monitor, config):
    """A system reporting 'degraded' should NOT accumulate consecutive misses."""
    equor = FakeSystem("equor", {"status": "degraded", "drift_severity": 0.6})
    monitor.register(equor)

    # Run 10 health checks - all returning "degraded"
    for _ in range(10):
        await monitor._check_system("equor", equor)

    record = monitor.get_record("equor")
    # Must NOT be FAILED - degraded is an alive status
    assert record.status != SystemStatus.FAILED
    assert record.consecutive_misses == 0
    assert not monitor.is_safe_mode


@pytest.mark.asyncio
async def test_healthy_status_records_success(monitor):
    sys = FakeSystem("memory", {"status": "healthy"})
    monitor.register(sys)

    await monitor._check_system("memory", sys)

    record = monitor.get_record("memory")
    assert record.consecutive_misses == 0
    assert record.consecutive_successes >= 1


@pytest.mark.asyncio
async def test_error_status_triggers_failure_after_threshold(monitor, config):
    """A system reporting 'error' should be declared FAILED after threshold misses."""
    equor = FakeSystem("equor", {"status": "error", "error": "something broke"})
    monitor.register(equor)

    for _ in range(config.health_failure_threshold):
        await monitor._check_system("equor", equor)

    record = monitor.get_record("equor")
    assert record.status == SystemStatus.FAILED
    # equor is critical, so safe mode should be entered
    assert monitor.is_safe_mode


@pytest.mark.asyncio
async def test_running_status_treated_as_alive(monitor):
    """Systems reporting 'running' should be treated as alive."""
    sys = FakeSystem("voxis", {"status": "running"})
    monitor.register(sys)

    for _ in range(5):
        await monitor._check_system("voxis", sys)

    record = monitor.get_record("voxis")
    assert record.status != SystemStatus.FAILED
    assert record.consecutive_misses == 0


@pytest.mark.asyncio
async def test_degraded_then_healthy_recovers(monitor):
    """Degraded systems should not escalate to FAILED, and should recover."""
    sys = FakeSystem("equor", {"status": "degraded"})
    monitor.register(sys)

    # Run degraded checks - should NOT escalate to FAILED
    for _ in range(5):
        await monitor._check_system("equor", sys)
    record = monitor.get_record("equor")
    assert record.status != SystemStatus.FAILED
    assert record.consecutive_misses == 0

    # Now report healthy - should stay non-failed
    sys.set_health({"status": "healthy"})
    await monitor._check_system("equor", sys)
    record = monitor.get_record("equor")
    assert record.status != SystemStatus.FAILED
    assert record.consecutive_misses == 0


@pytest.mark.asyncio
async def test_safe_mode_status_treated_as_alive(monitor):
    """Systems reporting 'safe_mode' should NOT be declared failed."""
    sys = FakeSystem("equor", {"status": "safe_mode"})
    monitor.register(sys)

    for _ in range(5):
        await monitor._check_system("equor", sys)

    record = monitor.get_record("equor")
    assert record.status != SystemStatus.FAILED
    assert record.consecutive_misses == 0


def test_alive_statuses_constant():
    """Verify all expected alive statuses are in the set."""
    assert "healthy" in _ALIVE_STATUSES
    assert "degraded" in _ALIVE_STATUSES
    assert "running" in _ALIVE_STATUSES
    assert "safe_mode" in _ALIVE_STATUSES
    assert "error" not in _ALIVE_STATUSES
    assert "failed" not in _ALIVE_STATUSES
