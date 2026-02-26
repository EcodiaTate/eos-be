"""
EcodiaOS — Simula Hunter: Zero-Day Discovery Engine

Hunter inverts Simula's internal verification logic — instead of proving
code is *correct*, it proves code is *exploitable* by translating Z3 SAT
counterexamples into weaponized exploit proofs-of-concept.

Public API:
  HunterService             — full pipeline orchestrator (Phase 7)
  TargetWorkspace           — abstraction for internal or external codebases
  TargetIngestor            — clones repos, builds graphs, maps attack surfaces
  VulnerabilityProver       — proves vulnerabilities via Z3 constraint inversion
  HunterRepairOrchestrator  — autonomous patch generation + Z3 re-verification
  HunterAnalyticsEmitter    — structured event logging + TSDB persistence (Phase 9)
  HunterAnalyticsView       — aggregate vulnerability analytics with time-windowed trends
  HunterAnalyticsStore      — durable TimescaleDB event storage + historical queries
  HunterEvent               — structured analytics event data model
  TargetType                — INTERNAL_EOS | EXTERNAL_REPO
  AttackSurface             — discovered exploitable entry point
  VulnerabilityReport       — proven vulnerability with Z3 counterexample + PoC
  HuntResult                — aggregated results from a full hunt
  HunterConfig              — authorization and resource limits
  RemediationResult         — outcome of autonomous vulnerability remediation
"""

from ecodiaos.systems.simula.hunter.analytics import (
    HunterAnalyticsEmitter,
    HunterAnalyticsStore,
    HunterAnalyticsView,
    HunterEvent,
)
from ecodiaos.systems.simula.hunter.ingestor import TargetIngestor
from ecodiaos.systems.simula.hunter.prover import VulnerabilityProver
from ecodiaos.systems.simula.hunter.remediation import HunterRepairOrchestrator
from ecodiaos.systems.simula.hunter.service import HunterService
from ecodiaos.systems.simula.hunter.types import (
    AttackSurface,
    AttackSurfaceType,
    HunterConfig,
    HuntResult,
    RemediationAttempt,
    RemediationResult,
    RemediationStatus,
    TargetType,
    VulnerabilityClass,
    VulnerabilityReport,
    VulnerabilitySeverity,
)
from ecodiaos.systems.simula.hunter.workspace import TargetWorkspace

__all__ = [
    "HunterService",
    "TargetWorkspace",
    "TargetIngestor",
    "VulnerabilityProver",
    "HunterRepairOrchestrator",
    "HunterAnalyticsEmitter",
    "HunterAnalyticsView",
    "HunterAnalyticsStore",
    "HunterEvent",
    "TargetType",
    "AttackSurface",
    "AttackSurfaceType",
    "VulnerabilityClass",
    "VulnerabilityReport",
    "VulnerabilitySeverity",
    "HuntResult",
    "HunterConfig",
    "RemediationStatus",
    "RemediationAttempt",
    "RemediationResult",
]
