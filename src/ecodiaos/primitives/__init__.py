"""
EcodiaOS — Shared Primitives

The lingua franca of the organism. Every system communicates through these types.
"""

from ecodiaos.primitives.affect import AffectDelta, AffectState
from ecodiaos.primitives.belief import Belief
from ecodiaos.primitives.common import (
    AutonomyLevel,
    ConsolidationLevel,
    DriveAlignmentVector,
    EntityType,
    HealthStatus,
    Modality,
    ResourceBudget,
    SalienceVector,
    SourceDescriptor,
    SystemID,
    Verdict,
    new_id,
    utc_now,
)
from ecodiaos.primitives.constitutional import ConstitutionalCheck, InvariantResult
from ecodiaos.primitives.expression import Expression, ExpressionStrategy, PersonalityVector
from ecodiaos.primitives.federation import (
    AssistanceRequest,
    AssistanceResponse,
    FederationInteraction,
    FederationLink,
    FederationLinkStatus,
    FilteredKnowledge,
    InstanceIdentityCard,
    InteractionOutcome,
    KnowledgeItem,
    KnowledgeRequest,
    KnowledgeResponse,
    KnowledgeType,
    PrivacyLevel,
    SHARING_PERMISSIONS,
    TRUST_THRESHOLDS,
    TrustLevel,
    TrustPolicy,
    VIOLATION_MULTIPLIER,
    ViolationType,
)
from ecodiaos.primitives.governance import AmendmentProposal, GovernanceRecord
from ecodiaos.primitives.intent import (
    Action,
    ActionSequence,
    DecisionTrace,
    EthicalClearance,
    GoalDescriptor,
    Intent,
)
from ecodiaos.primitives.memory_trace import (
    Community,
    ConstitutionNode,
    Entity,
    Episode,
    MemoryRetrievalRequest,
    MemoryRetrievalResponse,
    MemoryTrace,
    MentionRelation,
    RetrievalResult,
    SelfNode,
    SemanticRelation,
)
from ecodiaos.primitives.percept import Content, Percept, Provenance
from ecodiaos.primitives.telemetry import InstanceHealth, MetricPoint, SystemHealth

__all__ = [
    # Common
    "SystemID", "Modality", "EntityType", "ConsolidationLevel", "AutonomyLevel",
    "Verdict", "HealthStatus", "DriveAlignmentVector", "ResourceBudget",
    "SalienceVector", "SourceDescriptor", "new_id", "utc_now",
    # Percept
    "Percept", "Content", "Provenance",
    # Affect
    "AffectState", "AffectDelta",
    # Memory
    "Episode", "Entity", "Community", "SelfNode", "ConstitutionNode",
    "MentionRelation", "SemanticRelation", "MemoryTrace",
    "MemoryRetrievalRequest", "MemoryRetrievalResponse", "RetrievalResult",
    # Belief
    "Belief",
    # Intent
    "Intent", "GoalDescriptor", "Action", "ActionSequence",
    "EthicalClearance", "DecisionTrace",
    # Constitutional
    "ConstitutionalCheck", "InvariantResult",
    # Expression
    "Expression", "ExpressionStrategy", "PersonalityVector",
    # Governance
    "AmendmentProposal", "GovernanceRecord",
    # Telemetry
    "MetricPoint", "SystemHealth", "InstanceHealth",
    # Federation
    "InstanceIdentityCard", "FederationLink", "FederationLinkStatus",
    "TrustLevel", "TrustPolicy", "ViolationType", "InteractionOutcome",
    "FederationInteraction", "KnowledgeType", "PrivacyLevel",
    "KnowledgeItem", "KnowledgeRequest", "KnowledgeResponse",
    "FilteredKnowledge", "AssistanceRequest", "AssistanceResponse",
    "SHARING_PERMISSIONS", "TRUST_THRESHOLDS", "VIOLATION_MULTIPLIER",
]
