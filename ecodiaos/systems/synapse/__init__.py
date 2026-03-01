"""
EcodiaOS — Synapse (System #9)

The autonomic nervous system. Drives the cognitive cycle clock,
monitors system health, allocates resources, detects emergent
cognitive rhythms, and measures cross-system coherence.
"""

from ecodiaos.systems.synapse.clock import CognitiveClock
from ecodiaos.systems.synapse.coherence import CoherenceMonitor
from ecodiaos.systems.synapse.degradation import DegradationManager
from ecodiaos.systems.synapse.event_bus import EventBus
from ecodiaos.systems.synapse.health import HealthMonitor
from ecodiaos.systems.synapse.metabolism import MetabolicTracker
from ecodiaos.systems.synapse.resources import ResourceAllocator
from ecodiaos.systems.synapse.rhythm import DefaultRhythmStrategy, EmergentRhythmDetector
from ecodiaos.systems.synapse.service import SynapseService
from ecodiaos.systems.synapse.types import (
    BaseResourceAllocator,
    BaseRhythmStrategy,
    ClockState,
    CoherenceSnapshot,
    CycleResult,
    DegradationLevel,
    DegradationStrategy,
    ManagedSystemProtocol,
    MetabolicSnapshot,
    ResourceAllocation,
    ResourceSnapshot,
    RhythmSnapshot,
    RhythmState,
    SomaticCycleState,
    SomaTickEvent,
    SynapseEvent,
    SynapseEventType,
    SystemBudget,
    SystemHealthRecord,
    SystemHeartbeat,
    SystemStatus,
)

__all__ = [
    # Service
    "SynapseService",
    # Sub-systems
    "CognitiveClock",
    "CoherenceMonitor",
    "DegradationManager",
    "EventBus",
    "HealthMonitor",
    "ResourceAllocator",
    "EmergentRhythmDetector",
    "DefaultRhythmStrategy",
    "MetabolicTracker",
    # Strategy ABCs (NeuroplasticityBus targets)
    "BaseResourceAllocator",
    "BaseRhythmStrategy",
    # Types
    "ClockState",
    "CoherenceSnapshot",
    "CycleResult",
    "DegradationLevel",
    "DegradationStrategy",
    "ManagedSystemProtocol",
    "MetabolicSnapshot",
    "ResourceAllocation",
    "ResourceSnapshot",
    "RhythmSnapshot",
    "RhythmState",
    "SomaticCycleState",
    "SomaTickEvent",
    "SynapseEvent",
    "SynapseEventType",
    "SystemBudget",
    "SystemHeartbeat",
    "SystemHealthRecord",
    "SystemStatus",
]
