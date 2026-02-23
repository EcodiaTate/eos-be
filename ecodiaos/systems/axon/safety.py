"""
EcodiaOS — Axon Safety Mechanisms

Three interlocking safety systems protect against runaway action loops,
cascading failures, and excessive external calls:

1. RateLimiter — sliding-window counters per executor type
   Prevents any single executor from flooding an external service or
   spamming notifications. Counters are in-memory (per-process).
   For distributed deployments, back this with Redis (future Synapse work).

2. CircuitBreaker — per-executor open/half-open/closed state machine
   If an executor repeatedly fails, it is disabled for a recovery window.
   After recovery_timeout_s, a single probe execution is allowed (half-open).
   Success → closed (normal). Failure → re-opens. Prevents cascading failures
   from a degraded external service.

3. BudgetTracker — per-cycle execution budget enforcement
   Limits the total number and type of actions EOS can take in a single
   cognitive cycle. This is the non-negotiable safety valve — it exists
   to prevent EOS from acting obsessively or exhausting shared resources.
   Budget limits come from AxonConfig and cannot be raised at runtime.

These are defence-in-depth. Nova's EFE evaluation and Equor's constitutional
review are the first two lines. The safety mechanisms are the third.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

import structlog

from ecodiaos.config import AxonConfig
from ecodiaos.systems.axon.types import CircuitState, CircuitStatus, ExecutionBudget, RateLimit

logger = structlog.get_logger()


# ─── Rate Limiter ─────────────────────────────────────────────────


class RateLimiter:
    """
    Sliding-window rate limiter.

    Each action type gets its own window. Calls are timestamped; any
    call outside the window_seconds boundary is evicted before checking.

    Thread-safe for single-process use (asyncio). For multi-process,
    integrate with Redis counters via Synapse.
    """

    def __init__(self) -> None:
        # action_type → deque of call timestamps (monotonic)
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        self._logger = logger.bind(system="axon.rate_limiter")

    def check(self, action_type: str, rate_limit: RateLimit) -> bool:
        """
        Return True if the action is within its rate limit.

        Evicts expired entries from the window, then checks count.
        Does NOT record the call — call record() after a successful check.
        """
        window = self._windows[action_type]
        now = time.monotonic()
        cutoff = now - rate_limit.window_seconds

        # Evict expired timestamps
        while window and window[0] < cutoff:
            window.popleft()

        allowed = len(window) < rate_limit.max_calls
        if not allowed:
            self._logger.warning(
                "rate_limit_exceeded",
                action_type=action_type,
                current_count=len(window),
                max_calls=rate_limit.max_calls,
                window_seconds=rate_limit.window_seconds,
            )
        return allowed

    def record(self, action_type: str) -> None:
        """Record a call for rate-limit accounting."""
        self._windows[action_type].append(time.monotonic())

    def reset(self, action_type: str) -> None:
        """Reset the window for a specific action type (testing / governance)."""
        self._windows[action_type].clear()

    def current_count(self, action_type: str, window_seconds: float) -> int:
        """Return the number of calls within the last window_seconds."""
        window = self._windows[action_type]
        cutoff = time.monotonic() - window_seconds
        return sum(1 for ts in window if ts >= cutoff)


# ─── Circuit Breaker ──────────────────────────────────────────────


class CircuitBreaker:
    """
    Per-executor circuit breaker using a three-state finite state machine.

    States:
      CLOSED — normal operation; all executions allowed
      OPEN — tripped; all executions blocked for recovery_timeout_s
      HALF_OPEN — probing; allows exactly half_open_max_calls attempts

    Transitions:
      CLOSED → OPEN: failure_threshold consecutive failures
      OPEN → HALF_OPEN: after recovery_timeout_s
      HALF_OPEN → CLOSED: probe succeeds
      HALF_OPEN → OPEN: probe fails
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: int = 300,
        half_open_max_calls: int = 1,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.half_open_max_calls = half_open_max_calls
        self._states: dict[str, CircuitState] = {}
        self._logger = logger.bind(system="axon.circuit_breaker")

    def can_execute(self, action_type: str) -> bool:
        """Return True if the circuit is closed or in a half-open probe window."""
        state = self._states.get(action_type)
        if state is None:
            return True

        if state.status == CircuitStatus.CLOSED:
            return True

        if state.status == CircuitStatus.OPEN:
            elapsed = time.monotonic() - state.tripped_at
            if elapsed >= self.recovery_timeout_s:
                # Transition to half-open for probing
                state.status = CircuitStatus.HALF_OPEN
                state.half_open_calls = 0
                self._logger.info(
                    "circuit_half_open",
                    action_type=action_type,
                    elapsed_s=int(elapsed),
                )
                return True
            return False

        if state.status == CircuitStatus.HALF_OPEN:
            if state.half_open_calls < self.half_open_max_calls:
                state.half_open_calls += 1
                return True
            return False

        return False  # Unreachable, but safe

    def record_result(self, action_type: str, success: bool) -> None:
        """Update circuit state after an execution attempt."""
        state = self._states.setdefault(action_type, CircuitState())

        if success:
            if state.status == CircuitStatus.HALF_OPEN:
                state.status = CircuitStatus.CLOSED
                state.consecutive_failures = 0
                self._logger.info("circuit_closed", action_type=action_type)
            elif state.status == CircuitStatus.CLOSED:
                state.consecutive_failures = 0
        else:
            state.consecutive_failures += 1
            if state.consecutive_failures >= self.failure_threshold:
                if state.status != CircuitStatus.OPEN:
                    state.status = CircuitStatus.OPEN
                    state.tripped_at = time.monotonic()
                    self._logger.warning(
                        "circuit_tripped",
                        action_type=action_type,
                        consecutive_failures=state.consecutive_failures,
                    )

    def status(self, action_type: str) -> CircuitStatus:
        """Return current circuit status for an executor."""
        state = self._states.get(action_type)
        return state.status if state else CircuitStatus.CLOSED

    def reset(self, action_type: str) -> None:
        """Manually reset a circuit (governance action)."""
        if action_type in self._states:
            del self._states[action_type]
            self._logger.info("circuit_manually_reset", action_type=action_type)

    def trip_count(self) -> int:
        """Total circuits currently open."""
        return sum(
            1 for s in self._states.values() if s.status == CircuitStatus.OPEN
        )


# ─── Budget Tracker ───────────────────────────────────────────────


class BudgetTracker:
    """
    Per-cycle execution budget enforcement.

    The budget is replenished at the start of each cognitive cycle by calling
    begin_cycle(). Checks are cumulative within the cycle — once a limit is
    hit, it blocks for the remainder of the cycle.

    Limits are sourced from AxonConfig and cannot be changed at runtime.
    """

    def __init__(self, config: AxonConfig) -> None:
        self._budget = ExecutionBudget(
            max_actions_per_cycle=config.max_actions_per_cycle,
            max_api_calls_per_minute=config.max_api_calls_per_minute,
            max_notifications_per_hour=config.max_notifications_per_hour,
            max_concurrent_executions=config.max_concurrent_executions,
            total_timeout_per_cycle_ms=config.total_timeout_per_cycle_ms,
        )
        self._logger = logger.bind(system="axon.budget_tracker")
        self._reset_counters()

    def _reset_counters(self) -> None:
        self._actions_this_cycle: int = 0
        self._concurrent_executions: int = 0
        self._cycle_start: float = time.monotonic()

    def begin_cycle(self) -> None:
        """Called at the start of each cognitive cycle to reset per-cycle counters."""
        self._reset_counters()

    def can_execute(self) -> tuple[bool, str]:
        """
        Check if the budget allows another execution.
        Returns (allowed, reason) — reason is empty string if allowed.
        """
        if self._actions_this_cycle >= self._budget.max_actions_per_cycle:
            return False, (
                f"Budget: max actions per cycle reached "
                f"({self._budget.max_actions_per_cycle})"
            )
        if self._concurrent_executions >= self._budget.max_concurrent_executions:
            return False, (
                f"Budget: max concurrent executions reached "
                f"({self._budget.max_concurrent_executions})"
            )
        elapsed_ms = int((time.monotonic() - self._cycle_start) * 1000)
        if elapsed_ms >= self._budget.total_timeout_per_cycle_ms:
            return False, (
                f"Budget: cycle timeout exceeded "
                f"({self._budget.total_timeout_per_cycle_ms}ms)"
            )
        return True, ""

    def begin_execution(self) -> None:
        """Called when an execution starts (tracks concurrency)."""
        self._concurrent_executions += 1
        self._actions_this_cycle += 1

    def end_execution(self) -> None:
        """Called when an execution completes or fails (releases concurrency slot)."""
        self._concurrent_executions = max(0, self._concurrent_executions - 1)

    @property
    def utilisation(self) -> float:
        """Fraction of cycle action budget consumed (0.0 to 1.0+)."""
        return self._actions_this_cycle / max(1, self._budget.max_actions_per_cycle)

    @property
    def budget(self) -> ExecutionBudget:
        return self._budget
