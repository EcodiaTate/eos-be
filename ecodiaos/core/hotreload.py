"""
EcodiaOS — Neuroplasticity Bus

A single central dispatcher that subscribes once to ``eos:events:code_evolved``,
imports each changed file once, and fans out discovered classes to every system
that has registered a handler for a given base class.

Architecture
============

Before (broken):
  - N systems × 1 pubsub connection = N Redis subscriptions to the same channel
  - Each system independently imports every changed file
  - N redundant ``importlib`` calls per event regardless of relevance

After (this module):
  - 1 pubsub connection, lifetime of the process
  - Each changed file imported exactly once
  - All classes in the module walked once
  - Each class dispatched only to the handler(s) whose base_class it matches

Usage
=====

At application startup (main.py)::

    bus = NeuroplasticityBus(redis_client=redis_client)
    bus.start()
    app.state.neuroplasticity_bus = bus

In each service's ``initialize()``::

    bus.register(
        base_class=Executor,
        registration_callback=self._on_executor_evolved,
        system_id="axon",
        instance_qualifier=lambda cls: bool(cls.action_type),
        instance_factory=None,   # Executor is zero-arg constructable
    )

In each service's ``shutdown()``::

    bus.deregister(base_class=Executor)

At application shutdown::

    await bus.stop()

Handler descriptor
==================

Each registered handler is a ``_Handler`` dataclass that carries:

- ``base_class``             — the ABC to match against
- ``registration_callback`` — called with the instantiated component
- ``system_id``             — for log attribution
- ``instance_qualifier``    — optional ``(cls) → bool`` gate
- ``instance_factory``      — optional ``(cls) → T``; defaults to ``cls()``

Design invariants
=================

- Never raises into the event loop.
- Each file is imported at most once per event, even if N handlers could match it.
- Poisoned modules are evicted from sys.modules so a later retry starts clean.
- Handlers can be registered/deregistered at any time (before or after start()).
- Graceful cancellation via asyncio.CancelledError on stop().
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import orjson
import structlog

_CODE_EVOLVED_CHANNEL = "eos:events:code_evolved"

logger = structlog.get_logger()


@dataclass
class _Handler:
    """Internal descriptor for one system's hot-reload registration."""
    base_class: type
    registration_callback: Callable[[Any], None]
    system_id: str
    instance_qualifier: Callable[[type], bool] | None = None
    instance_factory: Callable[[type], Any] | None = None


class NeuroplasticityBus:
    """
    Single-subscriber, multi-handler hot-reload dispatcher.

    One instance lives for the lifetime of the process and is shared by all
    cognitive systems.  Each system calls ``register()`` to declare which base
    class it cares about and what to do when a new subclass is discovered.
    """

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._handlers: list[_Handler] = []
        self._task: asyncio.Task[None] | None = None
        self._logger = logger.bind(system="neuroplasticity_bus")

    # ─── Handler registry ─────────────────────────────────────────

    def register(
        self,
        base_class: type,
        registration_callback: Callable[[Any], None],
        system_id: str,
        instance_qualifier: Callable[[type], bool] | None = None,
        instance_factory: Callable[[type], Any] | None = None,
    ) -> None:
        """
        Register a handler for subclasses of *base_class*.

        Safe to call before or after ``start()``.  Replaces any existing
        handler for the same *base_class* so re-initialization is idempotent.

        Args:
            base_class:             The ABC whose concrete subclasses trigger
                                    *registration_callback*.
            registration_callback:  Called once per discovered instance.
            system_id:              Log attribution label (e.g. "axon").
            instance_qualifier:     Optional ``(cls) → bool`` — return False
                                    to skip a class even if it passes the
                                    subclass check.
            instance_factory:       Optional ``(cls) → T`` — use when the
                                    component needs constructor args.  Defaults
                                    to zero-arg ``cls()``.
        """
        # Remove any existing handler for this base_class (idempotent re-register)
        self._handlers = [h for h in self._handlers if h.base_class is not base_class]
        self._handlers.append(_Handler(
            base_class=base_class,
            registration_callback=registration_callback,
            system_id=system_id,
            instance_qualifier=instance_qualifier,
            instance_factory=instance_factory,
        ))
        self._logger.info(
            "handler_registered",
            system=system_id,
            base_class=base_class.__name__,
            total_handlers=len(self._handlers),
        )

    def deregister(self, base_class: type) -> None:
        """
        Remove the handler for *base_class*.

        Safe to call at any time.  No-op if the base class was never registered.
        """
        before = len(self._handlers)
        self._handlers = [h for h in self._handlers if h.base_class is not base_class]
        if len(self._handlers) < before:
            self._logger.info(
                "handler_deregistered",
                base_class=base_class.__name__,
            )

    # ─── Lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        """
        Attach the single background subscriber to the running event loop.

        Safe to call multiple times — subsequent calls are no-ops if the task
        is already alive.  Must be called from inside an async context.
        """
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(
            self._subscriber_loop(),
            name="neuroplasticity_bus_subscriber",
        )
        self._logger.info(
            "neuroplasticity_bus_started",
            channel=_CODE_EVOLVED_CHANNEL,
        )

    async def stop(self) -> None:
        """
        Cancel the background subscriber and wait for clean teardown.

        Safe to call even if ``start()`` was never called.
        """
        if self._task is None or self._task.done():
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._logger.info("neuroplasticity_bus_stopped")

    # ─── Subscriber loop ──────────────────────────────────────────

    async def _subscriber_loop(self) -> None:
        """
        Subscribe once to ``eos:events:code_evolved``, parse each payload,
        import each changed file exactly once, and fan out to all handlers.
        """
        pubsub = None
        try:
            raw_redis = self._redis.client  # type: ignore[union-attr]
            pubsub = raw_redis.pubsub()
            await pubsub.subscribe(_CODE_EVOLVED_CHANNEL)
            self._logger.info(
                "neuroplasticity_bus_listening",
                channel=_CODE_EVOLVED_CHANNEL,
            )

            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue

                raw_data = message["data"]
                if isinstance(raw_data, (bytes, bytearray)):
                    raw_data = raw_data.decode("utf-8")

                self._logger.info(
                    "code_evolved_signal_received",
                    data=raw_data[:200],
                )

                try:
                    payload = orjson.loads(raw_data)
                    files_changed: list[str] = payload.get("files_changed", [])
                except Exception as exc:
                    self._logger.warning(
                        "hotreload_payload_parse_error",
                        error=str(exc),
                    )
                    continue

                for rel_path in files_changed:
                    await self._process_file(rel_path)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._logger.error(
                "neuroplasticity_bus_fatal",
                error=str(exc),
                exc_info=True,
            )
        finally:
            if pubsub is not None:
                try:
                    await pubsub.unsubscribe(_CODE_EVOLVED_CHANNEL)
                    await pubsub.close()
                except Exception:
                    pass

    # ─── File processing ──────────────────────────────────────────

    async def _process_file(self, rel_path: str) -> None:
        """
        Import *rel_path* once, then dispatch every discovered class to
        whichever handler(s) match it.

        A single evolved file may contain subclasses for multiple systems
        (e.g. a combined executor + policy generator), and all matching
        handlers receive their respective subclasses from the same import.
        """
        if not self._handlers:
            return  # No systems registered yet — nothing to dispatch to

        abs_path = self._resolve_abs_path(rel_path)
        if abs_path is None:
            self._logger.warning(
                "hotreload_file_not_found",
                rel_path=rel_path,
                sys_path_roots=sys.path[:5],
            )
            return

        module_name = (
            rel_path.replace(os.sep, ".").replace("/", ".").removesuffix(".py")
        )

        self._logger.info(
            "hotreload_importing",
            rel_path=rel_path,
            abs_path=abs_path,
            module=module_name,
        )

        module = self._import_module(rel_path, abs_path, module_name)
        if module is None:
            return

        # Walk module members once; dispatch each class to all matching handlers
        all_classes = inspect.getmembers(module, inspect.isclass)
        dispatched: dict[str, list[str]] = {}  # system_id → [class names]

        for _name, cls in all_classes:
            if cls.__module__ != module_name:
                continue  # Imported dependency, not defined in this file

            for handler in self._handlers:
                if not self._matches_handler(cls, handler):
                    continue

                try:
                    instance = (
                        handler.instance_factory(cls)
                        if handler.instance_factory is not None
                        else cls()
                    )
                    handler.registration_callback(instance)
                    dispatched.setdefault(handler.system_id, []).append(cls.__name__)
                    self._logger.info(
                        "hotreload_component_registered",
                        class_name=cls.__name__,
                        base_class=handler.base_class.__name__,
                        system=handler.system_id,
                        rel_path=rel_path,
                    )
                except Exception as exc:
                    self._logger.error(
                        "hotreload_register_failed",
                        class_name=cls.__name__,
                        system=handler.system_id,
                        error=str(exc),
                    )

        if dispatched:
            self._logger.info(
                "hotreload_complete",
                rel_path=rel_path,
                dispatched=dispatched,
            )

    # ─── Helpers ──────────────────────────────────────────────────

    def _resolve_abs_path(self, rel_path: str) -> str | None:
        """Walk sys.path roots and return the first absolute path that exists."""
        for base in sys.path:
            candidate = os.path.join(base, rel_path)
            if os.path.isfile(candidate):
                return candidate
        return None

    def _import_module(self, rel_path: str, abs_path: str, module_name: str) -> Any:
        """
        Dynamically load *abs_path* under *module_name*.

        Returns the module object, or None on failure.
        Evicts poisoned sys.modules entries on import error so later retries
        start clean.
        """
        try:
            spec = importlib.util.spec_from_file_location(module_name, abs_path)
            if spec is None or spec.loader is None:
                self._logger.warning("hotreload_no_spec", rel_path=rel_path)
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            return module

        except Exception as exc:
            self._logger.error(
                "hotreload_import_failed",
                rel_path=rel_path,
                error=str(exc),
                exc_info=True,
            )
            sys.modules.pop(module_name, None)
            return None

    def _matches_handler(self, cls: type, handler: _Handler) -> bool:
        """
        Return True if *cls* should be dispatched to *handler*.

        Criteria:
          1. Not the base class itself.
          2. Is a subclass of handler.base_class.
          3. Passes the optional instance_qualifier predicate.
        """
        if cls is handler.base_class:
            return False
        try:
            if not issubclass(cls, handler.base_class):
                return False
        except TypeError:
            return False
        if handler.instance_qualifier is not None:
            return handler.instance_qualifier(cls)
        return True


# ─── Thin per-system facade ───────────────────────────────────────
#
# Systems call bus.register() directly — no separate HotReloader object needed.
# This alias exists only so existing code that imports HotReloader still works
# during the transition period.  New code should use NeuroplasticityBus directly.

class HotReloader:
    """
    Deprecated facade — wraps a shared NeuroplasticityBus.

    Preserved for backward compatibility with the three services already wired
    (Axon, Nova, Voxis).  New systems should call ``bus.register()`` directly
    instead of instantiating this class.

    The ``start()`` and ``stop()`` methods are intentional no-ops here because
    the bus manages its own lifecycle.  The handler is registered immediately
    at construction time.
    """

    def __init__(
        self,
        redis_client: Any,
        base_class: type,
        registration_callback: Callable[[Any], None],
        system_id: str,
        instance_qualifier: Callable[[type], bool] | None = None,
        instance_factory: Callable[[type], Any] | None = None,
    ) -> None:
        # Lazily create or reuse the process-wide bus stored on the redis client
        # so the transition works without touching main.py yet.
        if not hasattr(redis_client, "_neuroplasticity_bus"):
            bus = NeuroplasticityBus(redis_client=redis_client)
            redis_client._neuroplasticity_bus = bus  # type: ignore[attr-defined]
        self._bus: NeuroplasticityBus = redis_client._neuroplasticity_bus
        self._base_class = base_class
        self._bus.register(
            base_class=base_class,
            registration_callback=registration_callback,
            system_id=system_id,
            instance_qualifier=instance_qualifier,
            instance_factory=instance_factory,
        )

    def start(self) -> None:
        """Start the shared bus (idempotent)."""
        self._bus.start()

    async def stop(self) -> None:
        """Deregister this handler from the bus. Does not stop the bus itself."""
        self._bus.deregister(self._base_class)
