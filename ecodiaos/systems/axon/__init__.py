"""
EcodiaOS — Axon (Action Execution System)

Axon is the motor cortex. It takes Intents approved by Equor and turns them
into real-world effects — API calls, data operations, scheduled tasks,
notifications, and federated messages.

If Nova decides what to do, Axon does it.
The gap between intention and action is where trust lives.

Public interface:
  AxonService         — main service class
  ExecutionRequest    — input to AxonService.execute()
  AxonOutcome         — output of AxonService.execute()
  Executor            — ABC for custom executors
  ExecutorRegistry    — registry of available executors
"""

from ecodiaos.systems.axon.executor import Executor
from ecodiaos.systems.axon.registry import ExecutorRegistry
from ecodiaos.systems.axon.service import AxonService
from ecodiaos.systems.axon.types import AxonOutcome, ExecutionRequest

__all__ = [
    "AxonService",
    "ExecutionRequest",
    "AxonOutcome",
    "Executor",
    "ExecutorRegistry",
]
