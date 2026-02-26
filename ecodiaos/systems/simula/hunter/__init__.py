"""
EcodiaOS — Simula Hunter: Zero-Day Discovery Engine

Hunter inverts Simula's internal verification logic — instead of proving
code is *correct*, it proves code is *exploitable* by translating Z3 SAT
counterexamples into weaponized exploit proofs-of-concept.

Public API (Phases 1-11):
  HunterService             — full pipeline orchestrator (Phase 7)
  TargetWorkspace           — abstraction for internal or external codebases
  TargetIngestor            — clones repos, builds graphs, maps attack surfaces
  VulnerabilityProver       — proves vulnerabilities via Z3 constraint inversion
  HunterRepairOrchestrator  — autonomous patch generation + Z3 re-verification
  HunterSafetyGates         — PoC execution, workspace isolation, config validation (Phase 11)
  SafetyResult              — outcome of a safety gate check
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

Advanced Features (Phase 12):
  MultiLanguageSurfaceDetector   — Go, Rust, TypeScript attack surface detection
  ExploitChainAnalyzer           — multi-vulnerability chain discovery
  ExploitChain                   — discovered chain of exploitable vulnerabilities
  ZeroDayMarketplace             — cryptographic peer-to-peer vulnerability trading
  MarketplaceVulnerabilityListing — marketplace listing with encrypted PoC
  MarketplacePurchaseAgreement    — atomic transaction for zero-day sale
  AutonomousPatchingOrchestrator  — auto-generate GitHub pull requests with patches
  GitHubPRConfig                 — GitHub API configuration for PR submission
  GitHubPRResult                 — result of PR submission attempt
  ContinuousHuntingScheduler     — recurring hunts with cron-based scheduling
  ScheduledHuntConfig            — configuration for a scheduled hunt
  HuntScheduleRun                — single execution of a scheduled hunt
"""

from ecodiaos.systems.simula.hunter.advanced import (
    AutonomousPatchingOrchestrator,
    ContinuousHuntingScheduler,
    ExploitChain,
    ExploitChainAnalyzer,
    GitHubPRConfig,
    GitHubPRResult,
    HuntScheduleRun,
    LanguageType,
    MarketplacePurchaseAgreement,
    MarketplaceVulnerabilityListing,
    MultiLanguageSurfaceDetector,
    ScheduledHuntConfig,
    ZeroDayMarketplace,
)
from ecodiaos.systems.simula.hunter.analytics import (
    HunterAnalyticsEmitter,
    HunterAnalyticsStore,
    HunterAnalyticsView,
    HunterEvent,
)
from ecodiaos.systems.simula.hunter.ingestor import TargetIngestor
from ecodiaos.systems.simula.hunter.prover import VulnerabilityProver
from ecodiaos.systems.simula.hunter.remediation import HunterRepairOrchestrator
from ecodiaos.systems.simula.hunter.safety import HunterSafetyGates, SafetyResult
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
    # Core (Phases 1-11)
    "HunterService",
    "HunterSafetyGates",
    "SafetyResult",
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
    # Advanced (Phase 12)
    "LanguageType",
    "MultiLanguageSurfaceDetector",
    "ExploitChain",
    "ExploitChainAnalyzer",
    "ZeroDayMarketplace",
    "MarketplaceVulnerabilityListing",
    "MarketplacePurchaseAgreement",
    "AutonomousPatchingOrchestrator",
    "GitHubPRConfig",
    "GitHubPRResult",
    "ContinuousHuntingScheduler",
    "ScheduledHuntConfig",
    "HuntScheduleRun",
]
