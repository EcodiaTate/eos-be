"""
EcodiaOS — Simula: Self-Evolution System

The organism's capacity for metamorphosis. Where Evo adjusts the knobs,
Simula redesigns the dashboard.

Public API:
  SimulaService              — main service, wired in main.py
  EvoSimulaBridge            — translates Evo proposals to Simula format
  EvolutionAnalyticsEngine   — evolution quality tracking
  ProposalIntelligence       — dedup, prioritize, dependency analysis
  SimulaCodeAgent            — agentic code generation engine
  EvolutionHistoryManager    — immutable evolution history in Neo4j
  EvolutionProposal          — submitted by Evo when a hypothesis reaches SUPPORTED
  ProposalResult             — outcome of process_proposal()
  CodeChangeResult           — output of the code agent
  ChangeCategory             — taxonomy of allowed (and forbidden) change types
  ChangeSpec                 — formal specification of what to change
  EnrichedSimulationResult   — deep multi-strategy simulation output

Stage 1 enhancements:
  1A: Extended-thinking model routing for governance/high-risk proposals
  1B: Voyage-code-3 embeddings for semantic dedup + find_similar + Neo4j vector index
  1C: KVzip-inspired context compression for agentic tool loops

Stage 2 enhancements (Formal Verification Core):
  2A: Dafny proof-carrying code with Clover pattern
  2B: LLM + Z3 invariant discovery loop
  2C: Static analysis gates (Bandit / Semgrep)
  2D: AgentCoder pattern — test/code separation pipeline

Stage 3 enhancements (Incremental & Learning):
  3A: Salsa incremental verification — dependency-aware memoization
  3B: SWE-grep agentic retrieval — multi-hop code search
  3C: LILO library learning — abstraction extraction from successful proposals

Hunter — Zero-Day Discovery Engine:
  TargetWorkspace      — workspace abstraction (internal/external)
  AttackSurface        — discovered entry point
  VulnerabilityReport  — proven vulnerability + PoC
  HuntResult           — aggregated hunt results
  HunterConfig         — authorization and resource limits
"""

# Stage 2D: AgentCoder agents
from ecodiaos.systems.simula.agents.test_designer import TestDesignerAgent
from ecodiaos.systems.simula.agents.test_executor import TestExecutorAgent
from ecodiaos.systems.simula.analytics import EvolutionAnalyticsEngine
from ecodiaos.systems.simula.bridge import EvoSimulaBridge
from ecodiaos.systems.simula.code_agent import SimulaCodeAgent
from ecodiaos.systems.simula.history import EvolutionHistoryManager

# Hunter: Zero-Day Discovery Engine
from ecodiaos.systems.simula.hunter import (
    AttackSurface,
    AttackSurfaceType,
    HunterConfig,
    HuntResult,
    TargetType,
    TargetWorkspace,
    VulnerabilityReport,
    VulnerabilitySeverity,
)
from ecodiaos.systems.simula.learning.lilo import LiloLibraryEngine
from ecodiaos.systems.simula.proposal_intelligence import ProposalIntelligence
from ecodiaos.systems.simula.retrieval.swe_grep import SweGrepRetriever
from ecodiaos.systems.simula.service import SimulaService
from ecodiaos.systems.simula.types import (
    FORBIDDEN,
    GOVERNANCE_REQUIRED,
    SELF_APPLICABLE,
    SIMULA_IRON_RULES,
    CategorySuccessRate,
    CautionAdjustment,
    ChangeCategory,
    ChangeSpec,
    CodeChangeResult,
    ConfigVersion,
    CounterfactualResult,
    DependencyImpact,
    EnrichedSimulationResult,
    EvolutionAnalytics,
    EvolutionProposal,
    EvolutionRecord,
    EvoProposalEnriched,
    ProposalCluster,
    ProposalPriority,
    ProposalResult,
    ProposalStatus,
    ResourceCostEstimate,
    RiskLevel,
    SimulationResult,
    TriageResult,
    TriageStatus,
)

# Stage 2: Verification bridges
from ecodiaos.systems.simula.verification.dafny_bridge import DafnyBridge

# Stage 3: Engines
from ecodiaos.systems.simula.verification.incremental import IncrementalVerificationEngine
from ecodiaos.systems.simula.verification.static_analysis import StaticAnalysisBridge

# Stages 2 + 3: Verification types
from ecodiaos.systems.simula.verification.types import (
    DAFNY_TRIGGERABLE_CATEGORIES,
    AbstractionExtractionResult,
    # Stage 3C: LILO Library Learning
    AbstractionKind,
    # Stage 2A: Dafny
    AgentCoderIterationResult,
    AgentCoderResult,
    CachedVerificationResult,
    CloverRoundResult,
    DafnyVerificationResult,
    DafnyVerificationStatus,
    DiscoveredInvariant,
    FormalVerificationResult,
    FunctionSignature,
    IncrementalVerificationResult,
    InvariantKind,
    InvariantVerificationResult,
    InvariantVerificationStatus,
    LibraryAbstraction,
    LibraryStats,
    RetrievalHop,
    # Stage 3B: SWE-grep Retrieval
    RetrievalToolKind,
    RetrievedContext,
    StaticAnalysisFinding,
    StaticAnalysisResult,
    StaticAnalysisSeverity,
    SweGrepResult,
    TestDesignResult,
    TestExecutionResult,
    # Stage 3A: Incremental Verification
    VerificationCacheStatus,
    VerificationCacheTier,
)
from ecodiaos.systems.simula.verification.z3_bridge import Z3Bridge

__all__ = [
    # Services
    "SimulaService",
    "SimulaCodeAgent",
    "EvolutionHistoryManager",
    "EvoSimulaBridge",
    "EvolutionAnalyticsEngine",
    "ProposalIntelligence",
    # Core types
    "ChangeCategory",
    "ChangeSpec",
    "CodeChangeResult",
    "ConfigVersion",
    "EvolutionProposal",
    "EvolutionRecord",
    "ProposalResult",
    "ProposalStatus",
    "RiskLevel",
    "SimulationResult",
    # Enriched types
    "EnrichedSimulationResult",
    "CautionAdjustment",
    "CounterfactualResult",
    "DependencyImpact",
    "ResourceCostEstimate",
    "EvoProposalEnriched",
    "ProposalPriority",
    "ProposalCluster",
    "CategorySuccessRate",
    "EvolutionAnalytics",
    "TriageStatus",
    "TriageResult",
    # Constants
    "FORBIDDEN",
    "GOVERNANCE_REQUIRED",
    "SELF_APPLICABLE",
    "SIMULA_IRON_RULES",
    # Stage 2: Verification types
    "DafnyVerificationStatus",
    "CloverRoundResult",
    "DafnyVerificationResult",
    "InvariantKind",
    "InvariantVerificationStatus",
    "DiscoveredInvariant",
    "InvariantVerificationResult",
    "StaticAnalysisSeverity",
    "StaticAnalysisFinding",
    "StaticAnalysisResult",
    "TestDesignResult",
    "TestExecutionResult",
    "AgentCoderIterationResult",
    "AgentCoderResult",
    "FormalVerificationResult",
    "DAFNY_TRIGGERABLE_CATEGORIES",
    # Stage 2: Bridges
    "DafnyBridge",
    "Z3Bridge",
    "StaticAnalysisBridge",
    # Stage 2D: Agents
    "TestDesignerAgent",
    "TestExecutorAgent",
    # Stage 3A: Incremental Verification
    "VerificationCacheStatus",
    "VerificationCacheTier",
    "FunctionSignature",
    "CachedVerificationResult",
    "IncrementalVerificationResult",
    "IncrementalVerificationEngine",
    # Stage 3B: SWE-grep Retrieval
    "RetrievalToolKind",
    "RetrievalHop",
    "RetrievedContext",
    "SweGrepResult",
    "SweGrepRetriever",
    # Stage 3C: LILO Library Learning
    "AbstractionKind",
    "LibraryAbstraction",
    "AbstractionExtractionResult",
    "LibraryStats",
    "LiloLibraryEngine",
    # Hunter: Zero-Day Discovery Engine
    "TargetWorkspace",
    "TargetType",
    "AttackSurface",
    "AttackSurfaceType",
    "VulnerabilityReport",
    "VulnerabilitySeverity",
    "HuntResult",
    "HunterConfig",
]
