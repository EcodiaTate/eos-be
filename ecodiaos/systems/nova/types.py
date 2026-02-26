"""
EcodiaOS — Nova Internal Types

All types internal to Nova's decision and planning system.
These are richer than the shared primitives — they carry the full
cognitive context needed for deliberation, goal tracking, and EFE scoring.

Design notes:
- BeliefState is Nova's internal model of the world. It is NOT the shared
  Belief primitive (which represents a single probability distribution).
- Goal is Nova's rich internal goal structure. When an Intent is dispatched,
  it carries a GoalDescriptor (from primitives/intent.py), which is a lean
  summary suitable for cross-system communication.
- Policy is Nova's internal candidate action plan, distinct from Intent.
  Intents are finalised, Equor-reviewed plans; Policies are candidates.
"""

from __future__ import annotations

from datetime import datetime
import enum
from typing import Any

from pydantic import Field

from ecodiaos.primitives.affect import AffectState
from ecodiaos.primitives.common import (
    DriveAlignmentVector,
    EOSBaseModel,
    Identified,
    Timestamped,
    new_id,
    utc_now,
)


# ─── Belief State ─────────────────────────────────────────────────


class EntityBelief(EOSBaseModel):
    """Nova's belief about a single entity in the world."""

    entity_id: str
    name: str = ""
    entity_type: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)
    # 0.0 = completely uncertain, 1.0 = certain
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    last_observed: datetime = Field(default_factory=utc_now)
    # Percept/episode IDs that support this belief
    source_episodes: list[str] = Field(default_factory=list)


class ContextBelief(EOSBaseModel):
    """Nova's belief about the current conversational/situational context."""

    summary: str = ""
    domain: str = ""           # e.g., "technical", "emotional", "social"
    is_active_dialogue: bool = False
    user_intent_estimate: str = ""
    # Surprise level — how different is this from predictions?
    prediction_error_magnitude: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SelfBelief(EOSBaseModel):
    """Nova's beliefs about EOS's own state and capabilities."""

    # Map of capability name → confidence (0-1)
    capabilities: dict[str, float] = Field(default_factory=dict)
    # Estimated cognitive load (0-1)
    cognitive_load: float = Field(default=0.0, ge=0.0, le=1.0)
    # Confidence in own current beliefs overall
    epistemic_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Estimated goal completion capacity (can we take on more?)
    goal_capacity_remaining: float = Field(default=1.0, ge=0.0, le=1.0)


class IndividualBelief(EOSBaseModel):
    """Nova's beliefs about a specific individual."""

    individual_id: str
    name: str = ""
    # Estimated emotional state (valence estimate)
    estimated_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    # Confidence in that estimate
    valence_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    # Estimated engagement level (0-1)
    engagement_level: float = Field(default=0.5, ge=0.0, le=1.0)
    # General trust in the interaction
    relationship_trust: float = Field(default=0.5, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=utc_now)


class BeliefState(EOSBaseModel):
    """
    Nova's complete world model.

    This is the cognitive map — the best current estimate of world state.
    Updated continuously from workspace broadcasts and Memory retrieval.
    Drives all deliberation: which goals are relevant, which policies can work,
    what the expected free energy of each action is.

    Variational free energy (VFE) field tracks the aggregate prediction error:
        VFE ≈ Σ_i (1 - confidence_i) × salience_i
    Lower VFE = beliefs are well-supported = less surprise = better organism state.
    """

    # ── World model ──
    entities: dict[str, EntityBelief] = Field(default_factory=dict)

    # ── Situation model ──
    current_context: ContextBelief = Field(default_factory=ContextBelief)
    active_individual_ids: list[str] = Field(default_factory=list)
    individual_beliefs: dict[str, IndividualBelief] = Field(default_factory=dict)

    # ── Self model ──
    self_belief: SelfBelief = Field(default_factory=SelfBelief)

    # ── Metadata ──
    last_updated: datetime = Field(default_factory=utc_now)
    # Overall belief confidence (mean precision across all beliefs)
    overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Current variational free energy estimate
    free_energy: float = Field(default=0.5, ge=0.0, le=1.0)

    def compute_free_energy(self) -> float:
        """
        Compute variational free energy as the precision-weighted prediction error.

        VFE ≈ 1 - mean(confidence) across salient beliefs.
        Lower is better (well-supported beliefs = low surprise).
        This is a tractable approximation of the full VFE functional.
        """
        confidences: list[float] = [self.overall_confidence]
        confidences.extend(e.confidence for e in self.entities.values())
        confidences.append(self.current_context.confidence)
        if self.individual_beliefs:
            confidences.extend(b.valence_confidence for b in self.individual_beliefs.values())
        mean_confidence = sum(confidences) / max(1, len(confidences))
        return 1.0 - mean_confidence


class BeliefDelta(EOSBaseModel):
    """
    A structured change to the belief state.
    Produced by belief update operations and used for goal progress assessment.
    """

    entity_updates: dict[str, EntityBelief] = Field(default_factory=dict)
    entity_additions: dict[str, EntityBelief] = Field(default_factory=dict)
    entity_removals: list[str] = Field(default_factory=list)
    context_update: ContextBelief | None = None
    individual_updates: dict[str, IndividualBelief] = Field(default_factory=dict)
    prediction_error_magnitude: float = Field(default=0.0, ge=0.0, le=1.0)
    contradicted_belief_ids: list[str] = Field(default_factory=list)

    def involves_belief_conflict(self) -> bool:
        """True if this delta contains contradictions with existing beliefs."""
        return len(self.contradicted_belief_ids) > 0 or self.prediction_error_magnitude > 0.6

    def is_empty(self) -> bool:
        return (
            not self.entity_updates
            and not self.entity_additions
            and not self.entity_removals
            and self.context_update is None
            and not self.individual_updates
        )


# ─── Goal Types ───────────────────────────────────────────────────


class GoalStatus(enum.StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"


class GoalSource(enum.StrEnum):
    USER_REQUEST = "user_request"
    SELF_GENERATED = "self_generated"
    GOVERNANCE = "governance"
    CARE_RESPONSE = "care_response"
    MAINTENANCE = "maintenance"
    EPISTEMIC = "epistemic"


class Goal(Identified, Timestamped):
    """
    A living goal structure. Goals are not tasks — they are desires.
    Priority, urgency, and importance shift with context.
    """

    description: str
    target_domain: str = ""
    # The specific success state we want to reach (natural language)
    success_criteria: str = ""

    # ── Priority ──
    priority: float = Field(default=0.5, ge=0.0, le=1.0)   # Dynamic, recomputed each cycle
    urgency: float = Field(default=0.3, ge=0.0, le=1.0)    # Time sensitivity
    importance: float = Field(default=0.5, ge=0.0, le=1.0) # Constitutional weight

    # ── Drive alignment ──
    drive_alignment: DriveAlignmentVector = Field(default_factory=DriveAlignmentVector)

    # ── Source & Lifecycle ──
    source: GoalSource = GoalSource.USER_REQUEST
    status: GoalStatus = GoalStatus.ACTIVE
    deadline: datetime | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)

    # ── Dependencies ──
    depends_on: list[str] = Field(default_factory=list)  # Goal IDs
    blocks: list[str] = Field(default_factory=list)

    # ── Tracking ──
    intents_issued: list[str] = Field(default_factory=list)
    evidence_of_progress: list[str] = Field(default_factory=list)  # Episode IDs


class PriorityContext(EOSBaseModel):
    """Context needed for dynamic goal priority computation."""

    current_affect: AffectState = Field(default_factory=AffectState.neutral)
    drive_weights: dict[str, float] = Field(
        default_factory=lambda: {"coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0}
    )
    goal_statuses: dict[str, str] = Field(default_factory=dict)  # goal_id → status
    # Episode timestamps for staleness computation
    episode_timestamps: dict[str, datetime] = Field(default_factory=dict)


# ─── Policy Types ─────────────────────────────────────────────────


class PolicyStep(EOSBaseModel):
    """A single step in a policy's execution plan."""

    action_type: str  # "express" | "observe" | "request_info" | "store" | "wait" | "federate"
    parameters: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    expected_duration_ms: int = 1000


class Policy(Identified):
    """
    A candidate course of action.
    Generated by the PolicyGenerator and scored by the EFEEvaluator.
    Policies are Nova-internal; they become Intents after Equor review.
    """

    name: str
    type: str = "deliberate"            # "deliberate" | "express" | "observe" | "defer" | "do_nothing" | etc.
    description: str = ""              # Human-readable description of the policy
    reasoning: str = ""
    steps: list[PolicyStep] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    epistemic_value_description: str = ""
    estimated_effort: str = "medium"    # "none" | "low" | "medium" | "high"
    time_horizon: str = "short"         # "immediate" | "short" | "medium" | "long"
    # Set by EFEEvaluator
    efe_score: float | None = None


# ─── EFE Scoring ──────────────────────────────────────────────────


class PragmaticEstimate(EOSBaseModel):
    """How well a policy achieves the goal."""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Estimated probability of goal achievement
    success_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""


class EpistemicEstimate(EOSBaseModel):
    """How much uncertainty a policy reduces."""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    # How many uncertain beliefs would this policy test?
    uncertainties_addressed: int = 0
    expected_info_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    # Is this genuinely exploring new territory?
    novelty: float = Field(default=0.0, ge=0.0, le=1.0)


class RiskEstimate(EOSBaseModel):
    """Expected harm from executing a policy."""

    expected_harm: float = Field(default=0.0, ge=0.0, le=1.0)
    reversibility: float = Field(default=1.0, ge=0.0, le=1.0)  # 1.0 = fully reversible
    identified_risks: list[str] = Field(default_factory=list)


class EFEScore(EOSBaseModel):
    """
    The complete Expected Free Energy decomposition for a policy.

    G(π) = -[pragmatic_value + epistemic_value + constitutional_alignment + feasibility]
           + risk_penalty

    Lower total = more preferred policy (active inference convention).
    """

    # Component scores (all 0-1, higher = better)
    pragmatic: PragmaticEstimate = Field(default_factory=PragmaticEstimate)
    epistemic: EpistemicEstimate = Field(default_factory=EpistemicEstimate)
    constitutional_alignment: float = Field(default=0.5, ge=0.0, le=1.0)
    feasibility: float = Field(default=0.5, ge=0.0, le=1.0)
    risk: RiskEstimate = Field(default_factory=RiskEstimate)

    # Weighted total (lower = preferred)
    total: float = 0.0
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = ""


class EFEWeights(EOSBaseModel):
    """
    Weights for EFE components.
    Starting defaults match spec; Evo adjusts them over time.
    """

    pragmatic: float = 0.35
    epistemic: float = 0.20
    constitutional: float = 0.20
    feasibility: float = 0.15
    risk: float = 0.10


# ─── Situation Assessment ─────────────────────────────────────────


class SituationAssessment(EOSBaseModel):
    """
    The output of the fast/slow routing decision.
    Determines which deliberation path to take.
    """

    novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    risk: float = Field(default=0.0, ge=0.0, le=1.0)
    emotional_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    belief_conflict: bool = False
    requires_deliberation: bool = False
    # Was a matching fast-path procedure found in memory?
    has_matching_procedure: bool = False
    # The broadcast precision (importance signal from Atune)
    broadcast_precision: float = Field(default=0.5, ge=0.0, le=1.0)


# ─── Pending Intent & Outcome Tracking ───────────────────────────


class PendingIntent(EOSBaseModel):
    """Tracks an intent that has been dispatched and is awaiting outcome."""

    intent_id: str
    goal_id: str
    routed_to: str  # "voxis" | "axon"
    dispatched_at: datetime = Field(default_factory=utc_now)
    policy_name: str = ""


class IntentOutcome(EOSBaseModel):
    """The result of executing an intent."""

    intent_id: str
    success: bool
    episode_id: str = ""
    failure_reason: str = ""
    # Any new information revealed by execution
    new_observations: list[str] = Field(default_factory=list)


# ─── Decision Record (Observability) ─────────────────────────────


class DecisionRecord(EOSBaseModel):
    """
    Full record of a deliberation cycle for observability and Evo learning.
    """

    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)
    broadcast_id: str = ""
    path: str = ""  # "fast" | "slow" | "do_nothing" | "no_goal"
    situation_assessment: SituationAssessment = Field(default_factory=SituationAssessment)
    goal_id: str | None = None
    goal_description: str = ""
    policies_generated: int = 0
    selected_policy_name: str = ""
    efe_scores: dict[str, float] = Field(default_factory=dict)
    equor_verdict: str = ""
    intent_dispatched: bool = False
    latency_ms: int = 0
