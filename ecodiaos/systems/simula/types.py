"""
EcodiaOS -- Simula Internal Types

All data types internal to the Simula self-evolution system.
Simula is the organism's capacity for metamorphosis: structural change
beyond parameter tuning. These types model the full lifecycle of an
evolution proposal -- from reception through simulation, governance,
application, and immutable history.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from ecodiaos.primitives.common import (
    EOSBaseModel,
    Identified,
    Timestamped,
    new_id,
    utc_now,
)


# --- Enums -------------------------------------------------------------------


class ChangeCategory(str, enum.Enum):
    ADD_EXECUTOR = "add_executor"
    ADD_INPUT_CHANNEL = "add_input_channel"
    ADD_PATTERN_DETECTOR = "add_pattern_detector"
    ADJUST_BUDGET = "adjust_budget"
    MODIFY_CONTRACT = "modify_contract"
    ADD_SYSTEM_CAPABILITY = "add_system_capability"
    MODIFY_CYCLE_TIMING = "modify_cycle_timing"
    CHANGE_CONSOLIDATION = "change_consolidation"
    MODIFY_EQUOR = "modify_equor"
    MODIFY_CONSTITUTION = "modify_constitution"
    MODIFY_INVARIANTS = "modify_invariants"
    MODIFY_SELF_EVOLUTION = "modify_self_evolution"


class ProposalStatus(str, enum.Enum):
    PROPOSED = "proposed"
    SIMULATING = "simulating"
    AWAITING_GOVERNANCE = "awaiting_governance"
    APPROVED = "approved"
    APPLYING = "applying"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNACCEPTABLE = "unacceptable"


class ImpactType(str, enum.Enum):
    IMPROVEMENT = "improvement"
    REGRESSION = "regression"
    NEUTRAL = "neutral"


# --- Models -----------------------------------------------------------------


class ChangeSpec(EOSBaseModel):
    """
    Formal specification of what to change.
    One model covers every ChangeCategory -- fields are optional by category.
    """

    # ADD_EXECUTOR
    executor_name: str | None = None
    executor_description: str | None = None
    executor_action_type: str | None = None
    executor_input_schema: dict[str, Any] | None = None

    # ADD_INPUT_CHANNEL
    channel_name: str | None = None
    channel_type: str | None = None
    channel_description: str | None = None

    # ADD_PATTERN_DETECTOR
    detector_name: str | None = None
    detector_description: str | None = None
    detector_pattern_type: str | None = None

    # ADJUST_BUDGET
    budget_parameter: str | None = None
    budget_old_value: float | None = None
    budget_new_value: float | None = None

    # MODIFY_CONTRACT
    contract_changes: list[str] = Field(default_factory=list)

    # ADD_SYSTEM_CAPABILITY
    capability_description: str | None = None

    # MODIFY_CYCLE_TIMING
    timing_parameter: str | None = None
    timing_old_value: float | None = None
    timing_new_value: float | None = None

    # CHANGE_CONSOLIDATION
    consolidation_schedule: str | None = None

    # Cross-cutting
    affected_systems: list[str] = Field(default_factory=list)
    additional_context: str = ""
    code_hint: str = ""  # optional hint of what the code should look like


class SimulationDifference(EOSBaseModel):
    """Describes how one episode's outcome would differ under the proposed change."""

    episode_id: str
    original_outcome: str
    simulated_outcome: str
    impact: ImpactType
    reasoning: str = ""


class SimulationResult(EOSBaseModel):
    """Aggregate outcome of simulating a proposal against recent episodes."""

    episodes_tested: int = 0
    differences: int = 0
    improvements: int = 0
    regressions: int = 0
    neutral_changes: int = 0
    risk_level: RiskLevel = RiskLevel.LOW
    risk_summary: str = ""
    benefit_summary: str = ""
    simulated_at: datetime = Field(default_factory=utc_now)


class ProposalResult(EOSBaseModel):
    """Final outcome recorded once a proposal reaches a terminal state."""

    status: ProposalStatus
    reason: str = ""
    version: int | None = None
    governance_record_id: str | None = None
    files_changed: list[str] = Field(default_factory=list)


class EvolutionProposal(Identified, Timestamped):
    """
    The full proposal lifecycle object -- richer than Evo's simplified version.
    Owns the proposal from receipt through simulation, governance, and application.
    """

    source: str  # "evo" | "governance"
    category: ChangeCategory
    description: str
    change_spec: ChangeSpec
    evidence: list[str] = Field(default_factory=list)  # hypothesis IDs / episode IDs
    expected_benefit: str = ""
    risk_assessment: str = ""
    status: ProposalStatus = ProposalStatus.PROPOSED
    simulation: SimulationResult | None = None
    governance_record_id: str | None = None
    result: ProposalResult | None = None


class FileSnapshot(EOSBaseModel):
    """
    One file's state immediately before a change was applied, enabling rollback.
    content is None when the file did not previously exist -- rollback deletes it.
    """

    path: str  # absolute path
    content: str | None  # None means file did not exist before
    existed: bool = True


class ConfigSnapshot(Identified, Timestamped):
    """Full snapshot of all affected files captured before applying a change."""

    proposal_id: str
    files: list[FileSnapshot] = Field(default_factory=list)
    config_version: int  # the version at snapshot time


class ConfigVersion(EOSBaseModel):
    """Tracks one step in the config version chain."""

    version: int
    timestamp: datetime = Field(default_factory=utc_now)
    proposal_ids: list[str] = Field(default_factory=list)  # evolution proposal IDs
    config_hash: str  # SHA256 hash of the canonical config state


class EvolutionRecord(Identified, Timestamped):
    """Immutable history entry written to Neo4j after each successful application."""

    proposal_id: str
    category: ChangeCategory
    description: str
    from_version: int
    to_version: int
    files_changed: list[str] = Field(default_factory=list)
    simulation_risk: RiskLevel
    applied_at: datetime = Field(default_factory=utc_now)
    rolled_back: bool = False
    rollback_reason: str = ""


class CodeChangeResult(EOSBaseModel):
    """What the code agent returns after implementing a structural change."""

    success: bool
    files_written: list[str] = Field(default_factory=list)
    summary: str = ""
    error: str = ""
    lint_passed: bool = True
    tests_passed: bool = True
    test_output: str = ""


class HealthCheckResult(EOSBaseModel):
    """Result of a post-apply codebase health check."""

    healthy: bool
    issues: list[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=utc_now)


# --- Enriched Simulation Models ----------------------------------------------


class CounterfactualResult(EOSBaseModel):
    """
    Result of asking: 'If this change had existed during episode X,
    what would have been different?'

    Batched into a single LLM call across multiple episodes for
    token efficiency (~800 tokens per 30-episode batch).
    """

    episode_id: str
    would_have_triggered: bool = False
    predicted_outcome: str = ""
    impact: ImpactType = ImpactType.NEUTRAL
    confidence: float = 0.5
    reasoning: str = ""


class DependencyImpact(EOSBaseModel):
    """
    A file or module affected by a proposed change, discovered
    via static import-graph analysis (zero LLM tokens).
    """

    file_path: str
    impact_type: str = "import_dependency"  # "direct_modification" | "import_dependency" | "test_coverage"
    risk_contribution: float = 0.0


class ResourceCostEstimate(EOSBaseModel):
    """
    Heuristic estimation of the ongoing resource cost a change
    would add to the running system. Computed without LLM calls.
    """

    estimated_additional_llm_tokens_per_hour: int = 0
    estimated_additional_compute_ms_per_cycle: int = 0
    estimated_memory_mb: float = 0.0
    budget_headroom_percent: float = 100.0


class EnrichedSimulationResult(SimulationResult):
    """
    Extended simulation result with deep multi-strategy analysis.
    Produced by the upgraded ChangeSimulator, consumed by SimulaService
    for richer risk/benefit decision-making.
    """

    counterfactuals: list[CounterfactualResult] = Field(default_factory=list)
    dependency_impacts: list[DependencyImpact] = Field(default_factory=list)
    resource_cost_estimate: ResourceCostEstimate | None = None
    constitutional_alignment: float = 0.0
    counterfactual_regression_rate: float = 0.0
    dependency_blast_radius: int = 0


# --- Bridge Models -----------------------------------------------------------


class EvoProposalEnriched(EOSBaseModel):
    """
    Evo proposal enriched with hypothesis evidence and inferred context.
    Produced by EvoSimulaBridge, consumed by SimulaService.translate().
    """

    evo_description: str
    evo_rationale: str
    hypothesis_ids: list[str] = Field(default_factory=list)
    hypothesis_statements: list[str] = Field(default_factory=list)
    evidence_scores: list[float] = Field(default_factory=list)
    supporting_episode_ids: list[str] = Field(default_factory=list)
    mutation_target: str = ""
    mutation_type: str = ""
    inferred_category: ChangeCategory | None = None
    inferred_change_spec: ChangeSpec | None = None


# --- Proposal Intelligence Models --------------------------------------------


class ProposalPriority(EOSBaseModel):
    """
    Priority score for a proposal, enabling intelligent processing order.
    Higher priority_score = process first.

    Formula: evidence_strength * expected_impact / max(0.1, estimated_risk * estimated_cost)
    """

    proposal_id: str
    priority_score: float = 0.0
    evidence_strength: float = 0.0
    expected_impact: float = 0.0
    estimated_risk: float = 0.0
    estimated_cost: float = 0.0
    reasoning: str = ""


class ProposalCluster(EOSBaseModel):
    """
    Group of semantically similar proposals that could be deduplicated.
    Detected via cheap heuristics first, LLM only for ambiguous cases.
    """

    representative_id: str
    member_ids: list[str] = Field(default_factory=list)
    similarity_scores: list[float] = Field(default_factory=list)
    merge_recommendation: str = ""


# --- Analytics Models --------------------------------------------------------


class CategorySuccessRate(EOSBaseModel):
    """Success rate tracking for a specific change category."""

    category: ChangeCategory
    total: int = 0
    approved: int = 0
    rejected: int = 0
    rolled_back: int = 0

    @property
    def success_rate(self) -> float:
        return self.approved / max(1, self.total)

    @property
    def rollback_rate(self) -> float:
        return self.rolled_back / max(1, self.total)


class EvolutionAnalytics(EOSBaseModel):
    """
    Aggregate evolution quality metrics computed from Neo4j history.
    Enables Simula to learn from its own performance over time.
    Zero LLM tokens -- pure computation from stored records.
    """

    category_rates: dict[str, CategorySuccessRate] = Field(default_factory=dict)
    total_proposals: int = 0
    evolution_velocity: float = 0.0  # proposals per day
    mean_simulation_risk: float = 0.0
    rollback_rate: float = 0.0
    last_updated: datetime = Field(default_factory=utc_now)


# --- Constants ---------------------------------------------------------------

SELF_APPLICABLE: frozenset[ChangeCategory] = frozenset({
    ChangeCategory.ADD_EXECUTOR,
    ChangeCategory.ADD_INPUT_CHANNEL,
    ChangeCategory.ADD_PATTERN_DETECTOR,
    ChangeCategory.ADJUST_BUDGET,
})

GOVERNANCE_REQUIRED: frozenset[ChangeCategory] = frozenset({
    ChangeCategory.MODIFY_CONTRACT,
    ChangeCategory.ADD_SYSTEM_CAPABILITY,
    ChangeCategory.MODIFY_CYCLE_TIMING,
    ChangeCategory.CHANGE_CONSOLIDATION,
})

FORBIDDEN: frozenset[ChangeCategory] = frozenset({
    ChangeCategory.MODIFY_EQUOR,
    ChangeCategory.MODIFY_CONSTITUTION,
    ChangeCategory.MODIFY_INVARIANTS,
    ChangeCategory.MODIFY_SELF_EVOLUTION,
})

SIMULA_IRON_RULES: list[str] = [
    "Simula CANNOT modify Equor in any way.",
    "Simula CANNOT modify constitutional drives.",
    "Simula CANNOT modify invariants.",
    "Simula CANNOT modify its own logic (no self-modifying code).",
    "Simula CANNOT bypass governance for governed changes.",
    "Simula CANNOT apply changes without rollback capability.",
    "Simula CANNOT delete evolution history records.",
    "Simula MUST simulate before applying any change.",
    "Simula MUST maintain version continuity -- no identity-breaking changes.",
]

# Paths the code agent is NEVER allowed to write to
FORBIDDEN_WRITE_PATHS: list[str] = [
    "src/ecodiaos/systems/equor",
    "src/ecodiaos/systems/simula",
    "src/ecodiaos/primitives/constitutional.py",
    "src/ecodiaos/primitives/common.py",
    "src/ecodiaos/config.py",
]
