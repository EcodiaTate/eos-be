"""
EcodiaOS — Axon Built-in Executors

All built-in executors for the EOS action system.

Executors are organised by capability category:
  observation   — ObserveExecutor, QueryMemoryExecutor, AnalyseExecutor, SearchExecutor
  communication — RespondTextExecutor, NotificationExecutor, PostMessageExecutor
  data          — CreateRecordExecutor, UpdateRecordExecutor, ScheduleExecutor, ReminderExecutor
  integration   — APICallExecutor, WebhookExecutor
  internal      — StoreInsightExecutor, UpdateGoalExecutor, ConsolidationExecutor

Import build_default_registry() to get a fully-populated ExecutorRegistry.
"""

from __future__ import annotations

from typing import Any

from ecodiaos.systems.axon.executors.communication import (
    NotificationExecutor,
    PostMessageExecutor,
    RespondTextExecutor,
)
from ecodiaos.systems.axon.executors.data import (
    CreateRecordExecutor,
    ReminderExecutor,
    ScheduleExecutor,
    UpdateRecordExecutor,
)
from ecodiaos.systems.axon.executors.integration import (
    APICallExecutor,
    WebhookExecutor,
)
from ecodiaos.systems.axon.executors.internal import (
    ConsolidationExecutor,
    StoreInsightExecutor,
    UpdateGoalExecutor,
)
from ecodiaos.systems.axon.executors.observation import (
    AnalyseExecutor,
    ObserveExecutor,
    QueryMemoryExecutor,
    SearchExecutor,
)
from ecodiaos.systems.axon.registry import ExecutorRegistry

__all__ = [
    "build_default_registry",
    # Observation
    "ObserveExecutor",
    "QueryMemoryExecutor",
    "AnalyseExecutor",
    "SearchExecutor",
    # Communication
    "RespondTextExecutor",
    "NotificationExecutor",
    "PostMessageExecutor",
    # Data
    "CreateRecordExecutor",
    "UpdateRecordExecutor",
    "ScheduleExecutor",
    "ReminderExecutor",
    # Integration
    "APICallExecutor",
    "WebhookExecutor",
    # Internal
    "StoreInsightExecutor",
    "UpdateGoalExecutor",
    "ConsolidationExecutor",
]


def build_default_registry(
    memory: Any = None,
    voxis: Any = None,
    redis_client: Any = None,
) -> ExecutorRegistry:
    """
    Build and return a fully-populated ExecutorRegistry with all built-in executors.

    Args:
        memory: MemoryService instance (for memory-backed executors)
        voxis: VoxisService instance (for RespondTextExecutor)
        redis_client: Redis client (for scheduled tasks and reminders)
    """
    registry = ExecutorRegistry()

    # ── Observation (Level 1) ──────────────────────────────────────
    registry.register(ObserveExecutor(memory=memory))
    registry.register(QueryMemoryExecutor(memory=memory))
    registry.register(AnalyseExecutor(memory=memory))
    registry.register(SearchExecutor())

    # ── Communication (Level 1-2) ─────────────────────────────────
    registry.register(RespondTextExecutor(voxis=voxis))
    registry.register(NotificationExecutor(redis_client=redis_client))
    registry.register(PostMessageExecutor(memory=memory))

    # ── Data Operations (Level 2) ─────────────────────────────────
    registry.register(CreateRecordExecutor(memory=memory))
    registry.register(UpdateRecordExecutor(memory=memory))
    registry.register(ScheduleExecutor(redis_client=redis_client))
    registry.register(ReminderExecutor(redis_client=redis_client))

    # ── Integration (Level 2-3) ───────────────────────────────────
    registry.register(APICallExecutor())
    registry.register(WebhookExecutor())

    # ── Internal (Level 1) ───────────────────────────────────────
    registry.register(StoreInsightExecutor(memory=memory))
    registry.register(UpdateGoalExecutor())
    registry.register(ConsolidationExecutor(memory=memory))

    return registry
