"""
EcodiaOS — Evo Internal Types

All data types internal to the Evo learning system.
These are NOT shared primitives — they model Evo's cognitive structures:
hypotheses, pattern candidates, parameter adjustments, procedures,
consolidation state, and self-model statistics.
"""

from __future__ import annotations

import enum
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import Field

from ecodiaos.primitives.affect import AffectState
from ecodiaos.primitives.common import (
    EOSBaseModel,
    Identified,
    Timestamped,
    new_id,
    utc_now,
)


# ─── Enums ────────────────────────────────────────────────────────────────────


class PatternType(str, enum.Enum):
    """Categories of patterns Evo can detect."""

    COOCCURRENCE = "cooccurrence"
    ACTION_SEQUENCE = "action_sequence"
    TEMPORAL = "temporal"
    AFFECT_PATTERN = "affect_pattern"


class HypothesisCategory(str, enum.Enum):
    """What kind of claim does this hypothesis make?"""

    WORLD_MODEL = "world_model"    # Claim about external world structure
    SELF_MODEL = "self_model"      # Claim about EOS's own capabilities
    SOCIAL = "social"              # Claim about community member patterns
    PROCEDURAL = "procedural"      # Claim about action sequence effectiveness
    PARAMETER = "parameter"        # Claim about optimal system parameters


class HypothesisStatus(str, enum.Enum):
    """Lifecycle states for a hypothesis."""

    PROPOSED = "proposed"      # Just generated, not yet tested
    TESTING = "testing"        # Accumulating evidence
    SUPPORTED = "supported"    # Evidence_score > threshold AND enough episodes
    REFUTED = "refuted"        # Evidence_score below threshold
    INTEGRATED = "integrated"  # Mutation applied; hypothesis closed
    ARCHIVED = "archived"      # Stale or superseded


class MutationType(str, enum.Enum):
    """What kind of change does a confirmed hypothesis propose?"""

    PARAMETER_ADJUSTMENT = "parameter_adjustment"  # Nudge a system parameter
    PROCEDURE_CREATION = "procedure_creation"       # Codify a successful sequence
    SCHEMA_ADDITION = "schema_addition"             # Add entity/relation type
    EVOLUTION_PROPOSAL = "evolution_proposal"       # Structural change → Simula


class EvidenceDirection(str, enum.Enum):
    """How does a piece of evidence relate to a hypothesis?"""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


# ─── Pattern Candidate ────────────────────────────────────────────────────────


class PatternCandidate(EOSBaseModel):
    """
    A pattern candidate detected during online or offline processing.
    Candidates accumulate into hypotheses when they reach the min_occurrences
    threshold. Raw signal, not yet a claim.
    """

    type: PatternType
    elements: list[str]                                # What was detected
    count: int                                          # How many times seen
    confidence: float = 0.5                             # Detector confidence
    examples: list[str] = Field(default_factory=list)  # Episode IDs (evidence)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Pattern Context (mutable accumulator) ────────────────────────────────────


@dataclass
class PatternContext:
    """
    Mutable state accumulated across episodes during wake mode.
    Holds sliding-window counters for all four detector types.
    Reset after each consolidation cycle.

    Not a Pydantic model because it is mutated in-place continuously.
    """

    # CooccurrenceDetector: canonical_pair_key → count
    # Key format: "{entity_a}::{entity_b}" (sorted for stability)
    cooccurrence_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    # SequenceDetector: sequence_hash → count
    sequence_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    # SequenceDetector: sequence_hash → [episode_id, ...]
    sequence_examples: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # TemporalDetector: "{source}::h{hour}" or "{source}::d{weekday}" → count
    temporal_bins: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    # Temporal baselines: source_type → expected count per bin
    temporal_baselines: dict[str, float] = field(default_factory=dict)

    # AffectPatternDetector: stimulus_type → [(valence_delta, arousal_delta), ...]
    affect_responses: dict[str, list[tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Current affect (set by service before each scan, used by affect detector)
    previous_affect: AffectState | None = None
    current_affect: AffectState | None = None

    # Recent entity IDs from the last workspace broadcast (CooccurrenceDetector)
    recent_entity_ids: list[str] = field(default_factory=list)

    # Running episode counter since last reset
    episodes_scanned: int = 0

    def get_mature_sequences(self, min_occurrences: int = 3) -> list[PatternCandidate]:
        """Return action sequence candidates that have met the threshold."""
        candidates: list[PatternCandidate] = []
        for seq_hash, count in self.sequence_counts.items():
            if count >= min_occurrences:
                candidates.append(
                    PatternCandidate(
                        type=PatternType.ACTION_SEQUENCE,
                        elements=[seq_hash],
                        count=count,
                        confidence=min(0.9, 0.5 + count * 0.05),
                        examples=self.sequence_examples.get(seq_hash, [])[:10],
                        metadata={"sequence_hash": seq_hash},
                    )
                )
        return candidates

    def reset(self) -> None:
        """Reset all counters. Called after each consolidation cycle."""
        self.cooccurrence_counts.clear()
        self.sequence_counts.clear()
        self.sequence_examples.clear()
        self.temporal_bins.clear()
        self.temporal_baselines.clear()
        self.affect_responses.clear()
        self.recent_entity_ids.clear()
        self.episodes_scanned = 0
        self.previous_affect = None
        self.current_affect = None


# ─── Mutation ─────────────────────────────────────────────────────────────────


class Mutation(EOSBaseModel):
    """
    A proposed change to the organism's model, parameters, or structure.
    Attached to a Hypothesis; applied only when hypothesis status = SUPPORTED.
    """

    type: MutationType
    target: str          # Param name, procedure name, or schema element
    value: float = 0.0   # Delta for parameter adjustments; ignored for others
    description: str = ""


# ─── Hypothesis ───────────────────────────────────────────────────────────────


class Hypothesis(Identified, Timestamped):
    """
    A testable hypothesis about the world, self, or processing parameters.
    Stored as a :Hypothesis node in the Memory graph.

    Lifecycle:
      proposed → testing → supported | refuted → integrated | archived

    Evidence scoring follows approximate Bayesian model comparison:
      evidence_score += strength * (1 - complexity_penalty * 0.1)  [for support]
      evidence_score -= strength                                     [for contradiction]

    Integration thresholds (from VELOCITY_LIMITS):
      - evidence_score > 3.0
      - len(supporting_episodes) >= 10
      - hypothesis age >= 24 hours
    """

    category: HypothesisCategory
    statement: str                 # Natural language claim
    formal_test: str               # How we would falsify this

    # Evidence tracking
    supporting_episodes: list[str] = Field(default_factory=list)
    contradicting_episodes: list[str] = Field(default_factory=list)
    evidence_score: float = 0.0
    last_evidence_at: datetime = Field(default_factory=utc_now)

    # Lifecycle
    status: HypothesisStatus = HypothesisStatus.PROPOSED

    # Occam's razor — simpler hypotheses are preferred
    complexity_penalty: float = 0.1

    # What to apply if hypothesis reaches SUPPORTED
    proposed_mutation: Mutation | None = None


# ─── Evidence Result ──────────────────────────────────────────────────────────


class EvidenceResult(EOSBaseModel):
    """Result of evaluating a single episode against a hypothesis."""

    hypothesis_id: str
    episode_id: str
    direction: EvidenceDirection
    strength: float = 0.0
    reasoning: str = ""
    new_score: float = 0.0
    new_status: HypothesisStatus = HypothesisStatus.TESTING


# ─── Parameter Tuning ─────────────────────────────────────────────────────────


class ParameterSpec(EOSBaseModel):
    """Defines the valid range and step size for a tunable parameter."""

    min_val: float
    max_val: float
    step: float


class ParameterAdjustment(EOSBaseModel):
    """A proposed or applied adjustment to a system parameter."""

    parameter: str
    old_value: float
    new_value: float
    delta: float = 0.0
    hypothesis_id: str
    evidence_score: float
    supporting_count: int
    applied_at: datetime = Field(default_factory=utc_now)


# ─── Procedures ───────────────────────────────────────────────────────────────


class ProcedureStep(EOSBaseModel):
    """One step in a procedural memory."""

    action_type: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_duration_ms: int = 1000


class Procedure(Identified, Timestamped):
    """
    A reusable action sequence extracted from successful episodes.
    Stored as :Procedure nodes in the Memory graph.
    These become the "habits" Nova's fast path can use.
    """

    name: str
    preconditions: list[str] = Field(default_factory=list)
    steps: list[ProcedureStep] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    success_rate: float = 1.0          # Updated as procedure is used
    source_episodes: list[str] = Field(default_factory=list)
    usage_count: int = 0


# ─── Schema Induction ─────────────────────────────────────────────────────────


class SchemaInduction(EOSBaseModel):
    """
    A proposed structural change to the Memory graph's schema.
    New entity types, relation types, or community patterns from regularities.
    """

    entities: list[dict[str, str]] = Field(default_factory=list)
    relations: list[dict[str, str]] = Field(default_factory=list)
    communities: list[dict[str, str]] = Field(default_factory=list)
    source_hypothesis: str = ""


# ─── Evolution Proposals ──────────────────────────────────────────────────────


class EvolutionProposal(EOSBaseModel):
    """
    A structural change proposal submitted to Simula.
    Evo can propose; Simula gates the actual change.
    """

    description: str
    rationale: str
    supporting_hypotheses: list[str] = Field(default_factory=list)
    proposed_at: datetime = Field(default_factory=utc_now)


# ─── Self-Model ───────────────────────────────────────────────────────────────


class CapabilityScore(EOSBaseModel):
    """Success rate for a specific named capability."""

    capability: str
    success_count: int = 0
    total_count: int = 0

    @property
    def rate(self) -> float:
        return self.success_count / max(1, self.total_count)


class SelfModelStats(EOSBaseModel):
    """
    What EOS knows about itself: overall effectiveness and per-capability scores.
    Updated during each consolidation cycle.
    """

    success_rate: float = 0.5
    mean_alignment: float = 0.5
    total_outcomes_evaluated: int = 0
    capability_scores: dict[str, CapabilityScore] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)


# ─── Consolidation ────────────────────────────────────────────────────────────


class ConsolidationResult(EOSBaseModel):
    """Summary of what happened during one consolidation cycle."""

    duration_ms: int = 0
    hypotheses_evaluated: int = 0
    hypotheses_integrated: int = 0
    hypotheses_archived: int = 0
    procedures_extracted: int = 0
    schemas_induced: int = 0
    parameters_adjusted: int = 0
    total_parameter_delta: float = 0.0
    self_model_updated: bool = False
    triggered_at: datetime = Field(default_factory=utc_now)


# ─── Constants ────────────────────────────────────────────────────────────────


# All parameters Evo is permitted to adjust (spec Section V)
TUNABLE_PARAMETERS: dict[str, ParameterSpec] = {
    # Atune — salience head weights
    "atune.head.novelty.weight":     ParameterSpec(min_val=0.05, max_val=0.40, step=0.01),
    "atune.head.risk.weight":        ParameterSpec(min_val=0.05, max_val=0.40, step=0.01),
    "atune.head.identity.weight":    ParameterSpec(min_val=0.05, max_val=0.30, step=0.01),
    "atune.head.goal.weight":        ParameterSpec(min_val=0.05, max_val=0.30, step=0.01),
    "atune.head.emotional.weight":   ParameterSpec(min_val=0.05, max_val=0.30, step=0.01),
    "atune.head.causal.weight":      ParameterSpec(min_val=0.05, max_val=0.25, step=0.01),
    "atune.head.keyword.weight":     ParameterSpec(min_val=0.05, max_val=0.25, step=0.01),
    # Nova — EFE weights
    "nova.efe.pragmatic":            ParameterSpec(min_val=0.15, max_val=0.55, step=0.02),
    "nova.efe.epistemic":            ParameterSpec(min_val=0.05, max_val=0.40, step=0.02),
    "nova.efe.constitutional":       ParameterSpec(min_val=0.10, max_val=0.40, step=0.02),
    "nova.efe.feasibility":          ParameterSpec(min_val=0.05, max_val=0.30, step=0.02),
    "nova.efe.risk":                 ParameterSpec(min_val=0.05, max_val=0.25, step=0.02),
    # Voxis — personality vector
    "voxis.personality.warmth":      ParameterSpec(min_val=-1.0, max_val=1.0, step=0.03),
    "voxis.personality.directness":  ParameterSpec(min_val=-1.0, max_val=1.0, step=0.03),
    "voxis.personality.verbosity":   ParameterSpec(min_val=-1.0, max_val=1.0, step=0.03),
    "voxis.personality.formality":   ParameterSpec(min_val=-1.0, max_val=1.0, step=0.03),
    "voxis.personality.humour":      ParameterSpec(min_val=0.0,  max_val=1.0, step=0.03),
    # Memory — salience model weights
    "memory.salience.recency":       ParameterSpec(min_val=0.10, max_val=0.40, step=0.02),
    "memory.salience.frequency":     ParameterSpec(min_val=0.05, max_val=0.25, step=0.02),
    "memory.salience.affect":        ParameterSpec(min_val=0.05, max_val=0.30, step=0.02),
    "memory.salience.surprise":      ParameterSpec(min_val=0.05, max_val=0.25, step=0.02),
    "memory.salience.relevance":     ParameterSpec(min_val=0.10, max_val=0.40, step=0.02),
}

# Default initial values (mid-range or from spec defaults)
PARAMETER_DEFAULTS: dict[str, float] = {
    "atune.head.novelty.weight":     0.20,
    "atune.head.risk.weight":        0.20,
    "atune.head.identity.weight":    0.15,
    "atune.head.goal.weight":        0.15,
    "atune.head.emotional.weight":   0.15,
    "atune.head.causal.weight":      0.10,
    "atune.head.keyword.weight":     0.05,
    "nova.efe.pragmatic":            0.35,
    "nova.efe.epistemic":            0.20,
    "nova.efe.constitutional":       0.20,
    "nova.efe.feasibility":          0.15,
    "nova.efe.risk":                 0.10,
    "voxis.personality.warmth":      0.0,
    "voxis.personality.directness":  0.0,
    "voxis.personality.verbosity":   0.0,
    "voxis.personality.formality":   0.0,
    "voxis.personality.humour":      0.0,
    "memory.salience.recency":       0.25,
    "memory.salience.frequency":     0.15,
    "memory.salience.affect":        0.20,
    "memory.salience.surprise":      0.15,
    "memory.salience.relevance":     0.25,
}

# Change velocity limits (spec Section IX)
VELOCITY_LIMITS: dict[str, Any] = {
    "max_total_parameter_delta_per_cycle": 0.15,
    "max_single_parameter_delta":          0.03,
    "min_evidence_for_integration":        10,
    "min_hypothesis_age_hours":            24,
    "max_active_hypotheses":               50,
    "max_new_procedures_per_cycle":        3,
}

# What Evo cannot touch (spec Section IX)
EVO_CONSTRAINTS: dict[str, str] = {
    "equor_evaluation":          "forbidden",
    "constitutional_drives":     "forbidden",
    "invariants":                "forbidden",
    "self_evaluation_criteria":  "forbidden",
    "parameters":                "permitted_within_range",
    "knowledge_structures":      "permitted",
    "evolution_proposals":       "permitted_as_proposal",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────


def hash_sequence(sequence: list[str]) -> str:
    """Stable, deterministic hash of an action sequence."""
    canonical = json.dumps(sequence, sort_keys=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
