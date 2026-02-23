"""
EcodiaOS — Axon Executor Registry

The registry maps action type names to their Executor instances.

Registration happens at AxonService initialisation. The registry is then
immutable at runtime — no executor can be added or removed during a cognitive
cycle. This makes executor lookup O(1) and side-effect-free.

Lookup normalisation:
  Intents use dot-notation executor names (e.g. "executor.observe", "data.store").
  The registry normalises these to the canonical action_type used at registration.
  Normalisation strips a leading "executor." prefix and maps common aliases.

This means executors register under their canonical name ("observe", "call_api")
and the registry handles the dotted forms that Nova's PolicyGenerator may produce.
"""

from __future__ import annotations

import structlog

from ecodiaos.systems.axon.executor import Executor

logger = structlog.get_logger()

# Map dotted/aliased executor names → canonical action_type
_ALIAS_MAP: dict[str, str] = {
    # Observation
    "executor.observe": "observe",
    "executor.query_memory": "query_memory",
    "executor.analyse": "analyse",
    "executor.search": "search",
    # Communication
    "executor.respond": "respond_text",
    "respond": "respond_text",
    "executor.respond_text": "respond_text",
    "executor.notify": "send_notification",
    "notify": "send_notification",
    "executor.notification": "send_notification",
    "executor.post": "post_message",
    "post": "post_message",
    "executor.post_message": "post_message",
    "executor.email": "send_email",
    # Data operations
    "executor.create": "create_record",
    "create": "create_record",
    "executor.create_record": "create_record",
    "executor.update": "update_record",
    "update": "update_record",
    "executor.update_record": "update_record",
    "executor.schedule": "schedule_event",
    "schedule": "schedule_event",
    "executor.reminder": "set_reminder",
    "reminder": "set_reminder",
    # Integration
    "executor.api": "call_api",
    "api": "call_api",
    "executor.call_api": "call_api",
    "executor.webhook": "webhook_trigger",
    "webhook": "webhook_trigger",
    # Internal
    "executor.store": "store_insight",
    "store": "store_insight",
    "executor.store_insight": "store_insight",
    "executor.update_goal": "update_goal",
    "executor.consolidate": "trigger_consolidation",
    "consolidate": "trigger_consolidation",
    # Resource & config
    "executor.allocate": "allocate_resource",
    "executor.config": "adjust_config",
    # Federation
    "executor.federate": "federation_send",
    "federate": "federation_send",
    "executor.federation_send": "federation_send",
    "executor.federation_share": "federation_share",
}


def _normalise(action_type: str) -> str:
    """Normalise an action type string to its canonical registry key."""
    normalised = _ALIAS_MAP.get(action_type)
    if normalised:
        return normalised
    # Strip "executor." prefix if present
    if action_type.startswith("executor."):
        return action_type[len("executor."):]
    return action_type


class ExecutorRegistry:
    """
    Immutable-at-runtime registry of available action executors.

    Built at startup, queried at execution time.
    """

    def __init__(self) -> None:
        self._executors: dict[str, Executor] = {}
        self._logger = logger.bind(system="axon.registry")

    def register(self, executor: Executor) -> None:
        """
        Register an executor under its canonical action_type.

        Raises ValueError if the action_type is already registered.
        Call during initialisation only — not during a cognitive cycle.
        """
        key = executor.action_type
        if not key:
            raise ValueError(f"Executor {executor!r} has no action_type set")
        if key in self._executors:
            raise ValueError(
                f"Executor for action_type {key!r} already registered — "
                f"existing: {self._executors[key]!r}, new: {executor!r}"
            )
        self._executors[key] = executor
        self._logger.debug("executor_registered", action_type=key)

    def get(self, action_type: str) -> Executor | None:
        """
        Look up an executor by action type, with alias normalisation.

        Returns None if no executor is registered for the given type.
        """
        canonical = _normalise(action_type)
        return self._executors.get(canonical)

    def get_strict(self, action_type: str) -> Executor:
        """
        Look up an executor; raise KeyError if not found.
        """
        executor = self.get(action_type)
        if executor is None:
            canonical = _normalise(action_type)
            raise KeyError(
                f"No executor registered for action_type {action_type!r} "
                f"(normalised: {canonical!r}). "
                f"Available: {sorted(self._executors.keys())}"
            )
        return executor

    def list_types(self) -> list[str]:
        """Return all registered canonical action type names."""
        return sorted(self._executors.keys())

    def __contains__(self, action_type: str) -> bool:
        return self.get(action_type) is not None

    def __len__(self) -> int:
        return len(self._executors)

    def __repr__(self) -> str:
        return f"<ExecutorRegistry executors={self.list_types()}>"
