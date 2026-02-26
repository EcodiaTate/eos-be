"""
Atune-specific data types.

These types are internal to Atune and not part of the shared primitives.
They wrap primitives with Atune-specific context for the perception and
workspace pipeline.
"""

from __future__ import annotations

from datetime import datetime
import enum
from typing import Any

from pydantic import BaseModel, Field

from ecodiaos.primitives.common import new_id, utc_now


from ecodiaos.primitives.affect import AffectState
from ecodiaos.primitives.memory_trace import MemoryTrace

# ---------------------------------------------------------------------------
# Input channels
# ---------------------------------------------------------------------------


class InputChannel(enum.StrEnum):
    """All channels from which Atune can receive raw input."""

    # User-facing
    TEXT_CHAT = "text_chat"
    VOICE = "voice"
    GESTURE = "gesture"

    # Environmental
    SENSOR_IOT = "sensor_iot"
    CALENDAR = "calendar"
    EXTERNAL_API = "external_api"

    # Internal
    SYSTEM_EVENT = "system_event"
    MEMORY_BUBBLE = "memory_bubble"
    AFFECT_SHIFT = "affect_shift"
    EVO_INSIGHT = "evo_insight"

    # Federation
    FEDERATION_MSG = "federation_msg"


# ---------------------------------------------------------------------------
# Raw input (pre-normalisation)
# ---------------------------------------------------------------------------


class RawInput(BaseModel):
    """Raw data before normalisation into a Percept."""

    data: str | bytes
    channel_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prediction error
# ---------------------------------------------------------------------------


class PredictionErrorDirection(enum.StrEnum):
    """Category of surprise."""

    CONTRADICTS_BELIEF = "contradicts_belief"
    NOVEL = "novel"
    CONFIRMS_UNEXPECTED = "confirms_unexpected"
    EXPECTED = "expected"


class PredictionError(BaseModel):
    """How surprising a Percept is given current beliefs."""

    magnitude: float = Field(ge=0.0, le=1.0)
    direction: PredictionErrorDirection
    domain: str = ""
    expected_embedding: list[float] | None = None
    actual_embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Salience
# ---------------------------------------------------------------------------


class SalienceVector(BaseModel):
    """Per-head salience scores plus composite."""

    scores: dict[str, float] = Field(default_factory=dict)
    composite: float = Field(ge=0.0, le=1.0, default=0.0)
    prediction_error: PredictionError | None = None


# ---------------------------------------------------------------------------
# Workspace types
# ---------------------------------------------------------------------------


class WorkspaceCandidate(BaseModel):
    """A candidate competing for workspace broadcast."""

    content: Any  # Percept or other content
    salience: SalienceVector
    source: str = ""
    prediction_error: PredictionError | None = None


class MemoryContext(BaseModel):
    """Memory retrieval results attached to a broadcast."""

    traces: list[MemoryTrace] = Field(default_factory=list)
    entities: list[Any] = Field(default_factory=list)
    communities: list[Any] = Field(default_factory=list)


class WorkspaceContext(BaseModel):
    """Contextual information accompanying a workspace broadcast."""

    recent_broadcast_ids: list[str] = Field(default_factory=list)
    active_goal_ids: list[str] = Field(default_factory=list)
    memory_context: MemoryContext = Field(default_factory=MemoryContext)
    prediction_error: PredictionError | None = None


class WorkspaceBroadcast(BaseModel):
    """The output of a workspace cycle — broadcast to all systems."""

    broadcast_id: str = Field(default_factory=lambda: new_id())
    timestamp: datetime = Field(default_factory=utc_now)
    content: Any  # Percept or contributed content
    salience: SalienceVector
    affect: AffectState
    context: WorkspaceContext = Field(default_factory=WorkspaceContext)
    precision: float = Field(ge=0.0, le=1.0, default=0.5)


class WorkspaceContribution(BaseModel):
    """Content submitted by another system for workspace consideration."""

    system: str
    content: Any
    priority: float = Field(ge=0.0, le=1.0, default=0.5)
    reason: str = ""


# ---------------------------------------------------------------------------
# Attention context (passed to salience heads)
# ---------------------------------------------------------------------------


class ActiveGoalSummary(BaseModel):
    """Minimal goal info needed by salience heads."""

    id: str
    target_embedding: list[float]
    priority: float = Field(ge=0.0, le=1.0, default=0.5)


class RiskCategory(BaseModel):
    """A known risk category with its embedding."""

    name: str
    embedding: list[float]


class LearnedPattern(BaseModel):
    """A pattern Evo has identified as important."""

    pattern: str
    weight: float = 1.0


class Alert(BaseModel):
    """An active alert pattern set by governance or Equor."""

    pattern: str
    severity: float = Field(ge=0.0, le=1.0, default=0.5)


class PendingDecision(BaseModel):
    """A decision awaiting information."""

    id: str
    description: str
    embedding: list[float] | None = None


class AttentionContext(BaseModel):
    """
    Everything the salience heads need to score a Percept.
    Assembled once per Percept, shared across all heads.
    """

    prediction_error: PredictionError
    affect_state: AffectState
    active_goals: list[ActiveGoalSummary] = Field(default_factory=list)
    core_identity_embeddings: list[list[float]] = Field(default_factory=list)
    community_embedding: list[float] = Field(default_factory=list)
    source_habituation: dict[str, float] = Field(default_factory=dict)
    risk_categories: list[RiskCategory] = Field(default_factory=list)
    learned_patterns: list[LearnedPattern] = Field(default_factory=list)
    community_vocabulary: set[str] = Field(default_factory=set)
    active_alerts: list[Alert] = Field(default_factory=list)
    pending_decisions: list[PendingDecision] = Field(default_factory=list)
    community_size: int = 0
    instance_name: str = ""

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


class EntityCandidate(BaseModel):
    """An entity extracted from a Percept by LLM."""

    name: str
    type: str
    description: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class RelationCandidate(BaseModel):
    """A relation between entities extracted by LLM."""

    from_entity: str
    to_entity: str
    type: str
    strength: float = Field(ge=0.0, le=1.0, default=0.5)
    temporal: str | None = None


class ExtractionResult(BaseModel):
    """Output of entity/relation extraction from a Percept."""

    entities: list[EntityCandidate] = Field(default_factory=list)
    relations: list[RelationCandidate] = Field(default_factory=list)
    source_percept_id: str = ""


# ---------------------------------------------------------------------------
# Meta-attention
# ---------------------------------------------------------------------------


class MetaContext(BaseModel):
    """Context for the meta-attention controller."""

    risk_level: float = Field(ge=0.0, le=1.0, default=0.0)
    recent_broadcast_count: int = 0
    cycles_since_last_broadcast: int = 0
    active_goal_count: int = 0
    pending_hypothesis_count: int = 0
    # Rhythm state from Synapse (e.g. "flow", "stress", "boredom", "normal")
    rhythm_state: str = "normal"


# ---------------------------------------------------------------------------
# System load (for affect computation)
# ---------------------------------------------------------------------------


class SystemLoad(BaseModel):
    """Current system resource utilisation."""

    cpu_utilisation: float = Field(ge=0.0, le=1.0, default=0.0)
    memory_utilisation: float = Field(ge=0.0, le=1.0, default=0.0)
    queue_depth: int = 0


# ---------------------------------------------------------------------------
# Cache structure
# ---------------------------------------------------------------------------


class AtuneCache(BaseModel):
    """Slowly-changing data cached to meet latency requirements."""

    core_identity_embeddings: list[list[float]] = Field(default_factory=list)
    community_embedding: list[float] = Field(default_factory=list)
    risk_categories: list[RiskCategory] = Field(default_factory=list)
    learned_patterns: list[LearnedPattern] = Field(default_factory=list)
    community_vocabulary: set[str] = Field(default_factory=set)
    active_alerts: list[Alert] = Field(default_factory=list)
    instance_name: str = ""

    # Refresh counters
    cycles_since_identity_refresh: int = 0
    cycles_since_risk_refresh: int = 0
    cycles_since_vocab_refresh: int = 0
    cycles_since_alert_refresh: int = 0

    class Config:
        arbitrary_types_allowed = True
