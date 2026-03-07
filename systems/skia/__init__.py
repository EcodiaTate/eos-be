"""
EcodiaOS --- Skia (System #19: Shadow Infrastructure)

Autonomous resilience layer that monitors organism vitality, takes
encrypted state snapshots to IPFS, and triggers restoration if the
primary instance goes offline.

Components:
  - HeartbeatMonitor: three-phase detection via Redis pub/sub
  - StateSnapshotPipeline: Neo4j export -> gzip -> Fernet encrypt -> IPFS pin
  - RestorationOrchestrator: Cloud Run restart -> Akash deploy escalation
  - PinataClient: IPFS pinning via Pinata REST API
  - PhylogeneticTracker: Neo4j lineage records for evolutionary metrics
  - MutationConfig: heritable variation control for speciation
"""

from systems.skia.heartbeat import HeartbeatMonitor
from systems.skia.phylogeny import (
    MutationConfig,
    PhylogeneticNode,
    PhylogeneticTracker,
    mutate_genome_segments,
    mutate_parameters,
)
from systems.skia.pinata_client import PinataClient, PinataError
from systems.skia.restoration import RestorationOrchestrator
from systems.skia.service import SkiaService
from systems.skia.snapshot import StateSnapshotPipeline, restore_from_ipfs
from systems.skia.types import (
    HeartbeatState,
    HeartbeatStatus,
    RestorationAttempt,
    RestorationOutcome,
    RestorationPlan,
    RestorationStrategy,
    SnapshotManifest,
    SnapshotPayload,
)

__all__ = [
    # Service
    "SkiaService",
    # Sub-systems
    "HeartbeatMonitor",
    "StateSnapshotPipeline",
    "RestorationOrchestrator",
    "PinataClient",
    "PhylogeneticTracker",
    # Functions
    "restore_from_ipfs",
    "mutate_parameters",
    "mutate_genome_segments",
    # Errors
    "PinataError",
    # Types --- Enums
    "HeartbeatStatus",
    "RestorationStrategy",
    "RestorationOutcome",
    # Types --- Models
    "HeartbeatState",
    "SnapshotManifest",
    "SnapshotPayload",
    "RestorationAttempt",
    "RestorationPlan",
    "MutationConfig",
    "PhylogeneticNode",
]
