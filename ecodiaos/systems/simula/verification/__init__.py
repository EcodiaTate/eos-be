"""
EcodiaOS -- Simula Verification Subsystem (Stages 2 + 3 + 4)

Formal verification core: Dafny proof-carrying code, Z3 invariant
discovery, static analysis gates, incremental verification, and
Lean 4 proof generation.

Verification boundary: Tests → Static analysis → Z3 invariants → Dafny proofs → Lean 4 proofs
Stage 3A adds: Salsa-style incremental verification with dependency-aware caching.
Stage 4A adds: Lean 4 proof generation with DeepSeek-Prover-V2 pattern.
Stage 4B adds: GRPO domain fine-tuning (types only — engine in learning/).
Stage 4C adds: Diffusion-based code repair (types only — agent in agents/).
"""

from ecodiaos.systems.simula.verification.dafny_bridge import DafnyBridge
from ecodiaos.systems.simula.verification.incremental import IncrementalVerificationEngine
from ecodiaos.systems.simula.verification.lean_bridge import LeanBridge
from ecodiaos.systems.simula.verification.static_analysis import StaticAnalysisBridge
from ecodiaos.systems.simula.verification.types import (
    DAFNY_TRIGGERABLE_CATEGORIES,
    LEAN_PROOF_CATEGORIES,
    LEAN_PROOF_DOMAINS,
    AbstractionExtractionResult,
    AbstractionKind,
    AgentCoderIterationResult,
    AgentCoderResult,
    CachedVerificationResult,
    CloverRoundResult,
    DafnyVerificationResult,
    DafnyVerificationStatus,
    DiffusionDenoiseStep,
    DiffusionRepairResult,
    DiffusionRepairStatus,
    DiscoveredInvariant,
    FormalVerificationResult,
    FunctionSignature,
    GRPOEvaluationResult,
    GRPORollout,
    GRPOTrainingBatch,
    GRPOTrainingRun,
    GRPOTrainingStatus,
    IncrementalVerificationResult,
    InvariantKind,
    InvariantVerificationResult,
    InvariantVerificationStatus,
    LeanProofAttempt,
    LeanProofStatus,
    LeanSubgoal,
    LeanTacticKind,
    LeanVerificationResult,
    LibraryAbstraction,
    LibraryStats,
    ProofLibraryStats,
    ProvenLemma,
    RetrievalHop,
    RetrievalToolKind,
    RetrievedContext,
    StaticAnalysisFinding,
    StaticAnalysisResult,
    StaticAnalysisSeverity,
    SweGrepResult,
    TestDesignResult,
    TestExecutionResult,
    TrainingExample,
    VerificationCacheStatus,
    VerificationCacheTier,
    Z3RoundResult,
)
from ecodiaos.systems.simula.verification.z3_bridge import Z3Bridge

__all__ = [
    # Dafny (Stage 2A)
    "DafnyVerificationStatus",
    "CloverRoundResult",
    "DafnyVerificationResult",
    "DafnyBridge",
    # Z3 (Stage 2B)
    "InvariantKind",
    "InvariantVerificationStatus",
    "DiscoveredInvariant",
    "Z3RoundResult",
    "InvariantVerificationResult",
    "Z3Bridge",
    # Static Analysis (Stage 2C)
    "StaticAnalysisSeverity",
    "StaticAnalysisFinding",
    "StaticAnalysisResult",
    "StaticAnalysisBridge",
    # AgentCoder (Stage 2D)
    "TestDesignResult",
    "TestExecutionResult",
    "AgentCoderIterationResult",
    "AgentCoderResult",
    # Combined
    "FormalVerificationResult",
    # Constants
    "DAFNY_TRIGGERABLE_CATEGORIES",
    "LEAN_PROOF_CATEGORIES",
    "LEAN_PROOF_DOMAINS",
    # Stage 3A: Incremental Verification
    "VerificationCacheStatus",
    "VerificationCacheTier",
    "FunctionSignature",
    "CachedVerificationResult",
    "IncrementalVerificationResult",
    "IncrementalVerificationEngine",
    # Stage 3B: SWE-grep Retrieval (types only — engine in retrieval/)
    "RetrievalToolKind",
    "RetrievalHop",
    "RetrievedContext",
    "SweGrepResult",
    # Stage 3C: LILO Library Learning (types only — engine in learning/)
    "AbstractionKind",
    "LibraryAbstraction",
    "AbstractionExtractionResult",
    "LibraryStats",
    # Stage 4A: Lean 4 Proof Generation
    "LeanProofStatus",
    "LeanTacticKind",
    "LeanSubgoal",
    "LeanProofAttempt",
    "ProvenLemma",
    "LeanVerificationResult",
    "ProofLibraryStats",
    "LeanBridge",
    # Stage 4B: GRPO Domain Fine-Tuning (types only — engine in learning/)
    "GRPOTrainingStatus",
    "TrainingExample",
    "GRPORollout",
    "GRPOTrainingBatch",
    "GRPOEvaluationResult",
    "GRPOTrainingRun",
    # Stage 4C: Diffusion-Based Code Repair (types only — agent in agents/)
    "DiffusionRepairStatus",
    "DiffusionDenoiseStep",
    "DiffusionRepairResult",
]
