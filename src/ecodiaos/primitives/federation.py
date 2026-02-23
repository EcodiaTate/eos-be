"""
EcodiaOS — Federation Primitives

Instance identity cards, federation links, trust levels, knowledge exchange,
coordinated action, and privacy-filtered sharing types.

The Federation Protocol governs how EOS instances relate to each other —
as sovereign entities that can choose to share knowledge, coordinate action,
and build relationships. Every interaction is consent-based; trust starts
at zero and builds through demonstrated reliability.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from ecodiaos.primitives.common import EOSBaseModel, Identified, utc_now


# ─── Trust Levels ─────────────────────────────────────────────────


class TrustLevel(int, enum.Enum):
    """
    Trust levels between federated instances.

    Trust starts at NONE after mutual authentication and builds through
    successful interactions. Violations cost 3x — a privacy breach
    resets trust to zero immediately.
    """

    NONE = 0          # Authenticated but no trust. Greetings only.
    ACQUAINTANCE = 1  # Can exchange public knowledge and non-sensitive queries.
    COLLEAGUE = 2     # Can exchange community-level knowledge and coordinate.
    PARTNER = 3       # Can share sensitive (non-private) knowledge and co-plan.
    ALLY = 4          # Deep trust. Can share most knowledge and delegate actions.


class FederationLinkStatus(str, enum.Enum):
    """Status of a federation link."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"


class ViolationType(str, enum.Enum):
    """Categories of trust violations in federation interactions."""

    PRIVACY_BREACH = "privacy_breach"       # Shared individual data without consent
    DECEPTION = "deception"                 # Provided false information
    CONSENT_VIOLATION = "consent_violation"  # Acted without proper consent
    PROTOCOL_VIOLATION = "protocol_violation"  # Broke federation protocol rules
    RESOURCE_ABUSE = "resource_abuse"       # Excessive/unreasonable requests


class InteractionOutcome(str, enum.Enum):
    """Outcome of a federation interaction."""

    SUCCESSFUL = "successful"
    FAILED = "failed"
    VIOLATION = "violation"
    TIMEOUT = "timeout"


# ─── Instance Identity ───────────────────────────────────────────


class TrustPolicy(EOSBaseModel):
    """How an instance manages trust with federation partners."""

    auto_accept_links: bool = False
    min_trust_for_knowledge: TrustLevel = TrustLevel.ACQUAINTANCE
    min_trust_for_coordination: TrustLevel = TrustLevel.COLLEAGUE
    max_trust_level: TrustLevel = TrustLevel.ALLY
    trust_decay_enabled: bool = True
    trust_decay_rate_per_day: float = 0.1  # Inactive links lose trust slowly


class InstanceIdentityCard(EOSBaseModel):
    """
    Public identity of an EOS instance for federation.

    This is the "business card" exchanged during link establishment.
    The certificate_fingerprint and public_key_pem are used for mutual
    authentication. The constitutional_hash allows compatibility checks.
    """

    instance_id: str
    name: str
    description: str = ""
    born_at: datetime = Field(default_factory=utc_now)
    community_context: str = ""
    personality_summary: str = ""
    autonomy_level: int = 1
    endpoint: str = ""
    certificate_fingerprint: str = ""
    public_key_pem: str = ""
    constitutional_hash: str = ""
    capabilities: list[str] = Field(default_factory=list)
    trust_policy: TrustPolicy = Field(default_factory=TrustPolicy)
    protocol_version: str = "1.0"


# ─── Federation Link ────────────────────────────────────────────


class FederationLink(Identified):
    """
    An active link between two federated instances.

    Tracks trust score (float that maps to TrustLevel thresholds),
    interaction history stats, and communication state.
    """

    local_instance_id: str
    remote_instance_id: str
    remote_name: str = ""
    remote_endpoint: str
    trust_level: TrustLevel = TrustLevel.NONE
    trust_score: float = 0.0
    established_at: datetime = Field(default_factory=utc_now)
    last_communication: datetime | None = None
    shared_knowledge_count: int = 0
    received_knowledge_count: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    violation_count: int = 0
    status: FederationLinkStatus = FederationLinkStatus.ACTIVE
    remote_identity: InstanceIdentityCard | None = None


# ─── Federation Interaction ──────────────────────────────────────


class FederationInteraction(Identified):
    """
    Record of a single federation interaction (knowledge exchange,
    assistance request, etc.) used for trust scoring and audit.
    """

    link_id: str
    remote_instance_id: str
    interaction_type: str  # "knowledge_request" | "knowledge_share" | "assistance" | "greeting"
    direction: str  # "outbound" | "inbound"
    outcome: InteractionOutcome = InteractionOutcome.SUCCESSFUL
    violation_type: ViolationType | None = None
    trust_value: float = 1.0  # How much trust this interaction is worth
    description: str = ""
    timestamp: datetime = Field(default_factory=utc_now)
    latency_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Knowledge Exchange ─────────────────────────────────────────


class KnowledgeType(str, enum.Enum):
    """Types of knowledge that can be exchanged between instances."""

    PUBLIC_ENTITIES = "public_entities"
    COMMUNITY_DESCRIPTION = "community_description"
    COMMUNITY_LEVEL_KNOWLEDGE = "community_level_knowledge"
    PROCEDURES = "procedures"
    HYPOTHESES = "hypotheses"
    ANONYMISED_PATTERNS = "anonymised_patterns"
    SCHEMA_STRUCTURES = "schema_structures"


class PrivacyLevel(str, enum.Enum):
    """Privacy classification of knowledge items."""

    PUBLIC = "public"               # Freely shareable
    COMMUNITY_ONLY = "community_only"  # Shareable at COLLEAGUE+
    PRIVATE = "private"             # Never crosses federation boundary


class KnowledgeItem(EOSBaseModel):
    """A single piece of knowledge prepared for federation sharing."""

    item_id: str
    knowledge_type: KnowledgeType
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC
    content: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    source_instance_id: str = ""
    created_at: datetime = Field(default_factory=utc_now)


class KnowledgeRequest(Identified):
    """Request for knowledge from a remote instance."""

    requesting_instance_id: str
    knowledge_type: KnowledgeType
    query: str = ""
    query_embedding: list[float] | None = None
    domain: str = ""
    max_results: int = 10
    timestamp: datetime = Field(default_factory=utc_now)


class KnowledgeResponse(EOSBaseModel):
    """Response to a knowledge request."""

    request_id: str
    granted: bool
    reason: str = ""
    knowledge: list[KnowledgeItem] = Field(default_factory=list)
    attribution: str = ""  # Instance ID of the sharing instance
    trust_level_required: TrustLevel | None = None
    timestamp: datetime = Field(default_factory=utc_now)


class FilteredKnowledge(EOSBaseModel):
    """Knowledge after privacy filtering — safe to send across federation."""

    items: list[KnowledgeItem] = Field(default_factory=list)
    items_removed_by_privacy: int = 0
    items_anonymised: int = 0


# ─── Sharing Permissions ─────────────────────────────────────────


SHARING_PERMISSIONS: dict[TrustLevel, list[KnowledgeType]] = {
    TrustLevel.NONE: [],
    TrustLevel.ACQUAINTANCE: [
        KnowledgeType.PUBLIC_ENTITIES,
        KnowledgeType.COMMUNITY_DESCRIPTION,
    ],
    TrustLevel.COLLEAGUE: [
        KnowledgeType.PUBLIC_ENTITIES,
        KnowledgeType.COMMUNITY_DESCRIPTION,
        KnowledgeType.COMMUNITY_LEVEL_KNOWLEDGE,
        KnowledgeType.PROCEDURES,
    ],
    TrustLevel.PARTNER: [
        KnowledgeType.PUBLIC_ENTITIES,
        KnowledgeType.COMMUNITY_DESCRIPTION,
        KnowledgeType.COMMUNITY_LEVEL_KNOWLEDGE,
        KnowledgeType.PROCEDURES,
        KnowledgeType.HYPOTHESES,
        KnowledgeType.ANONYMISED_PATTERNS,
    ],
    TrustLevel.ALLY: [
        KnowledgeType.PUBLIC_ENTITIES,
        KnowledgeType.COMMUNITY_DESCRIPTION,
        KnowledgeType.COMMUNITY_LEVEL_KNOWLEDGE,
        KnowledgeType.PROCEDURES,
        KnowledgeType.HYPOTHESES,
        KnowledgeType.ANONYMISED_PATTERNS,
        KnowledgeType.SCHEMA_STRUCTURES,
    ],
}


# ─── Coordinated Action ─────────────────────────────────────────


class AssistanceRequest(Identified):
    """Request for assistance from a remote instance."""

    requesting_instance_id: str
    description: str
    knowledge_domain: str = ""
    urgency: float = 0.5  # 0-1
    reciprocity_offer: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)


class AssistanceResponse(EOSBaseModel):
    """Response to an assistance request."""

    request_id: str
    accepted: bool
    reason: str = ""
    estimated_completion_ms: int | None = None
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Trust Thresholds ────────────────────────────────────────────


TRUST_THRESHOLDS: dict[TrustLevel, float] = {
    TrustLevel.ACQUAINTANCE: 5.0,
    TrustLevel.COLLEAGUE: 20.0,
    TrustLevel.PARTNER: 50.0,
    TrustLevel.ALLY: 100.0,
}

# Violations cost 3x their trust value; privacy breaches are instant reset.
VIOLATION_MULTIPLIER: float = 3.0
