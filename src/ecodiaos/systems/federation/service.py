"""
EcodiaOS — Federation Service

The Federation Protocol governs how EOS instances relate to each other —
as sovereign entities that can choose to share knowledge, coordinate action,
and build relationships. Not a hive mind. Not isolated. A community of
individuals, each with their own identity, personality, and community,
choosing to help each other grow.

The FederationService orchestrates five sub-systems:
  IdentityManager   — Instance identity cards, Ed25519 signing, verification
  TrustManager      — Trust scoring, level transitions, decay
  PrivacyFilter     — PII removal, consent enforcement
  KnowledgeExchange — Knowledge request/response protocol
  CoordinationMgr   — Assistance requests and coordinated action
  ChannelManager    — Mutual TLS channels to remote instances

Lifecycle:
  initialize()             — build all sub-systems, load keys
  establish_link()         — connect to a remote instance
  withdraw_link()          — disconnect from a remote instance
  handle_knowledge_req()   — handle inbound knowledge request
  request_knowledge()      — request knowledge from remote
  handle_assistance_req()  — handle inbound assistance request
  request_assistance()     — request assistance from remote
  shutdown()               — graceful shutdown

Performance targets:
  Identity verification: ≤500ms
  Knowledge request handling: ≤2000ms
  Trust update: ≤50ms
  Link establishment: ≤3000ms
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.config import FederationConfig
from ecodiaos.primitives.common import new_id, utc_now
from ecodiaos.primitives.federation import (
    AssistanceRequest,
    AssistanceResponse,
    FederationInteraction,
    FederationLink,
    FederationLinkStatus,
    InstanceIdentityCard,
    InteractionOutcome,
    KnowledgeRequest,
    KnowledgeResponse,
    KnowledgeType,
    TrustLevel,
    TrustPolicy,
)
from ecodiaos.systems.federation.channel import ChannelManager
from ecodiaos.systems.federation.coordination import CoordinationManager
from ecodiaos.systems.federation.identity import IdentityManager
from ecodiaos.systems.federation.knowledge import KnowledgeExchangeManager
from ecodiaos.systems.federation.privacy import PrivacyFilter
from ecodiaos.systems.federation.trust import TrustManager

if TYPE_CHECKING:
    from ecodiaos.clients.redis import RedisClient
    from ecodiaos.systems.equor.service import EquorService
    from ecodiaos.systems.memory.service import MemoryService
    from ecodiaos.telemetry.metrics import MetricCollector

logger = structlog.get_logger("ecodiaos.systems.federation")


class FederationService:
    """
    Federation — the EOS diplomatic system.

    Coordinates identity, trust, knowledge exchange, coordinated action,
    and privacy filtering across federation links with other EOS instances.

    Every federation action goes through Equor. The constitutional drives
    apply to inter-instance relations just as they do to individual
    interactions. An instance cannot be helpful to a federation partner
    at the expense of its own community (Care drive).
    """

    system_id: str = "federation"

    def __init__(
        self,
        config: FederationConfig,
        memory: MemoryService | None = None,
        equor: EquorService | None = None,
        redis: RedisClient | None = None,
        metrics: MetricCollector | None = None,
        instance_id: str = "",
    ) -> None:
        self._config = config
        self._memory = memory
        self._equor = equor
        self._redis = redis
        self._metrics = metrics
        self._instance_id = instance_id
        self._logger = logger.bind(system="federation")
        self._initialized: bool = False

        # Sub-systems (built in initialize())
        self._identity: IdentityManager | None = None
        self._trust: TrustManager | None = None
        self._privacy: PrivacyFilter | None = None
        self._knowledge: KnowledgeExchangeManager | None = None
        self._coordination: CoordinationManager | None = None
        self._channels: ChannelManager | None = None

        # Active federation links (link_id → FederationLink)
        self._links: dict[str, FederationLink] = {}

        # Interaction history (for audit, limited ring buffer)
        self._interaction_history: list[FederationInteraction] = []
        self._max_history: int = 1000

    # ─── Lifecycle ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Build all sub-systems and load identity keys."""
        if self._initialized:
            return

        if not self._config.enabled:
            self._logger.info("federation_disabled")
            self._initialized = True
            return

        # Build sub-systems
        trust_max = TrustLevel(min(self._config.max_trust_level, 4))
        self._trust = TrustManager(
            trust_decay_enabled=self._config.trust_decay_enabled,
            trust_decay_rate_per_day=self._config.trust_decay_rate_per_day,
            max_trust_level=trust_max,
        )

        self._privacy = PrivacyFilter()

        self._knowledge = KnowledgeExchangeManager(
            memory=self._memory,
            privacy_filter=self._privacy,
            max_items_per_request=self._config.max_knowledge_items_per_request,
        )

        self._coordination = CoordinationManager()

        # Channel manager with TLS configuration
        tls_cert = Path(self._config.tls_cert_path) if self._config.tls_cert_path else None
        tls_key = Path(self._config.tls_key_path) if self._config.tls_key_path else None
        ca_cert = Path(self._config.ca_cert_path) if self._config.ca_cert_path else None

        self._channels = ChannelManager(
            tls_cert_path=tls_cert,
            tls_key_path=tls_key,
            ca_cert_path=ca_cert,
        )

        # Identity manager — load/generate keys and build identity card
        self._identity = IdentityManager()

        # Gather instance info from memory for the identity card
        instance_name = "EOS"
        community_context = ""
        personality_summary = ""
        autonomy_level = 1

        if self._memory:
            try:
                self_node = await self._memory.get_self()
                if self_node:
                    instance_name = self_node.name
                    community_context = getattr(self_node, "community_context", "")
                    autonomy_level = getattr(self_node, "autonomy_level", 1)
            except Exception:
                pass

        private_key_path = (
            Path(self._config.private_key_path) if self._config.private_key_path else None
        )

        trust_policy = TrustPolicy(
            auto_accept_links=self._config.auto_accept_links,
            trust_decay_enabled=self._config.trust_decay_enabled,
            trust_decay_rate_per_day=self._config.trust_decay_rate_per_day,
            max_trust_level=trust_max,
        )

        await self._identity.initialize(
            instance_id=self._instance_id,
            instance_name=instance_name,
            community_context=community_context,
            personality_summary=personality_summary,
            autonomy_level=autonomy_level,
            endpoint=self._config.endpoint or "",
            capabilities=["knowledge_exchange", "coordinated_action"],
            trust_policy=trust_policy,
            private_key_path=private_key_path,
            tls_cert_path=tls_cert,
        )

        # Load persisted links from Redis
        await self._load_links()

        self._initialized = True
        self._logger.info(
            "federation_initialized",
            instance_id=self._instance_id,
            enabled=True,
            endpoint=self._config.endpoint,
            active_links=len(self._links),
        )

    async def shutdown(self) -> None:
        """Graceful shutdown — close all channels, persist link state."""
        self._logger.info("federation_shutting_down")

        # Persist link state
        await self._persist_links()

        # Close all channels
        if self._channels:
            await self._channels.close_all()

        self._logger.info(
            "federation_shutdown_complete",
            active_links=len(self._links),
            total_interactions=len(self._interaction_history),
        )

    # ─── Link Management ────────────────────────────────────────────

    async def establish_link(
        self, remote_endpoint: str
    ) -> dict[str, Any]:
        """
        Establish a new federation link with a remote instance.

        Steps:
          1. Fetch remote identity card
          2. Verify remote identity
          3. Equor constitutional review
          4. Create link with NONE trust
          5. Open communication channel
          6. Exchange greetings

        Performance target: ≤3000ms
        """
        if not self._config.enabled:
            return {"error": "Federation is disabled"}

        if not self._identity or not self._channels or not self._trust:
            return {"error": "Federation not initialized"}

        start = utc_now()

        # Check max links
        active_count = sum(
            1 for l in self._links.values()
            if l.status == FederationLinkStatus.ACTIVE
        )
        if active_count >= self._config.max_concurrent_links:
            return {"error": f"Maximum concurrent links ({self._config.max_concurrent_links}) reached"}

        # Step 1: Create a temporary link for the channel
        temp_link = FederationLink(
            local_instance_id=self._instance_id,
            remote_instance_id="unknown",
            remote_endpoint=remote_endpoint,
            status=FederationLinkStatus.PENDING,
        )

        # Step 2: Open channel and fetch remote identity
        try:
            channel = await self._channels.open_channel(temp_link)
            remote_identity = await channel.get_identity()

            if remote_identity is None:
                await self._channels.close_channel(temp_link.id)
                return {"error": "Could not fetch remote identity card"}
        except Exception as exc:
            return {"error": f"Connection failed: {exc}"}

        # Step 3: Verify identity
        verification = self._identity.verify_identity(remote_identity)
        if not verification.verified:
            await self._channels.close_channel(temp_link.id)
            return {"error": f"Identity verification failed: {verification.errors}"}

        # Step 4: Check for duplicate link
        for existing in self._links.values():
            if (
                existing.remote_instance_id == remote_identity.instance_id
                and existing.status == FederationLinkStatus.ACTIVE
            ):
                await self._channels.close_channel(temp_link.id)
                return {
                    "error": "Already linked to this instance",
                    "existing_link_id": existing.id,
                }

        # Step 5: Equor constitutional review
        equor_permitted = True
        if self._equor:
            try:
                # Build a lightweight intent for federation link review
                from ecodiaos.primitives.intent import (
                    Intent, GoalDescriptor, ActionSequence, DecisionTrace,
                )
                intent = Intent(
                    goal=GoalDescriptor(
                        description=f"Establish federation link with {remote_identity.name}",
                        target_domain="federation",
                    ),
                    plan=ActionSequence(steps=[]),
                    decision_trace=DecisionTrace(
                        reasoning=f"Federation link request from {remote_identity.name} "
                                  f"({remote_identity.community_context[:100]})",
                    ),
                )
                check = await self._equor.review(intent)
                from ecodiaos.primitives.common import Verdict
                equor_permitted = check.verdict in (Verdict.APPROVED, Verdict.MODIFIED)
            except Exception as exc:
                self._logger.warning("equor_review_failed", error=str(exc))

        if not equor_permitted:
            await self._channels.close_channel(temp_link.id)
            return {"error": "Constitutional review denied this federation link"}

        # Step 6: Create the official link
        link = FederationLink(
            local_instance_id=self._instance_id,
            remote_instance_id=remote_identity.instance_id,
            remote_name=remote_identity.name,
            remote_endpoint=remote_endpoint,
            trust_level=TrustLevel.NONE,
            trust_score=0.0,
            status=FederationLinkStatus.ACTIVE,
            remote_identity=remote_identity,
        )

        # Re-open channel with the real link
        await self._channels.close_channel(temp_link.id)
        await self._channels.open_channel(link)

        self._links[link.id] = link
        await self._persist_links()

        elapsed_ms = int((utc_now() - start).total_seconds() * 1000)

        # Record interaction
        interaction = FederationInteraction(
            link_id=link.id,
            remote_instance_id=link.remote_instance_id,
            interaction_type="link_establishment",
            direction="outbound",
            outcome=InteractionOutcome.SUCCESSFUL,
            description=f"Link established with {remote_identity.name}",
            trust_value=1.0,
            latency_ms=elapsed_ms,
        )
        self._record_interaction(interaction)

        # Update trust for the successful establishment
        self._trust.update_trust(link, interaction)

        # Record metric
        if self._metrics:
            await self._metrics.record("federation", "links.established", 1.0)

        self._logger.info(
            "link_established",
            link_id=link.id,
            remote_id=remote_identity.instance_id,
            remote_name=remote_identity.name,
            elapsed_ms=elapsed_ms,
        )

        return {
            "link_id": link.id,
            "remote_instance_id": remote_identity.instance_id,
            "remote_name": remote_identity.name,
            "trust_level": link.trust_level.name,
            "status": link.status.value,
            "elapsed_ms": elapsed_ms,
        }

    async def withdraw_link(self, link_id: str) -> dict[str, Any]:
        """
        Withdraw from a federation link.

        Withdrawal is always free — any instance can disconnect at any
        time with no penalty. This ensures federation is always
        voluntary, never coerced.
        """
        link = self._links.get(link_id)
        if not link:
            return {"error": "Link not found"}

        link.status = FederationLinkStatus.WITHDRAWN

        # Close the channel
        if self._channels:
            await self._channels.close_channel(link_id)

        await self._persist_links()

        if self._metrics:
            await self._metrics.record("federation", "links.dropped", 1.0)

        self._logger.info(
            "link_withdrawn",
            link_id=link_id,
            remote_id=link.remote_instance_id,
        )

        return {
            "link_id": link_id,
            "status": "withdrawn",
            "remote_instance_id": link.remote_instance_id,
        }

    # ─── Knowledge Exchange ─────────────────────────────────────────

    async def handle_knowledge_request(
        self,
        request: KnowledgeRequest,
    ) -> KnowledgeResponse:
        """
        Handle an inbound knowledge request from a remote instance.

        This is called by the API router when a federated instance
        sends a knowledge request to us.
        """
        if not self._knowledge or not self._trust:
            return KnowledgeResponse(
                request_id=request.id,
                granted=False,
                reason="Federation not initialized",
            )

        # Find the link for this requesting instance
        link = self._find_link_by_instance(request.requesting_instance_id)
        if not link:
            return KnowledgeResponse(
                request_id=request.id,
                granted=False,
                reason="No active federation link with this instance",
            )

        # Equor review
        equor_permitted = True
        if self._equor:
            try:
                from ecodiaos.primitives.intent import (
                    Intent, GoalDescriptor, ActionSequence, DecisionTrace,
                )
                intent = Intent(
                    goal=GoalDescriptor(
                        description=f"Share {request.knowledge_type.value} with {link.remote_name}",
                        target_domain="federation.knowledge",
                    ),
                    plan=ActionSequence(steps=[]),
                    decision_trace=DecisionTrace(
                        reasoning=f"Knowledge request from {link.remote_name}: {request.query[:100]}",
                    ),
                )
                check = await self._equor.review(intent)
                from ecodiaos.primitives.common import Verdict
                equor_permitted = check.verdict in (Verdict.APPROVED, Verdict.MODIFIED)
            except Exception:
                pass

        response, interaction = await self._knowledge.handle_request(
            request=request,
            link=link,
            equor_permitted=equor_permitted,
        )

        # Update trust
        self._trust.update_trust(link, interaction)
        self._record_interaction(interaction)

        # Metrics
        if self._metrics:
            metric_name = "knowledge.shared" if response.granted else "knowledge.denied"
            await self._metrics.record("federation", metric_name, 1.0)
            if response.granted:
                await self._metrics.record(
                    "federation", "privacy.items_filtered",
                    float(len(response.knowledge)),
                )

        return response

    async def request_knowledge(
        self,
        link_id: str,
        knowledge_type: KnowledgeType,
        query: str = "",
        max_results: int = 10,
    ) -> KnowledgeResponse | None:
        """
        Request knowledge from a remote federated instance.
        """
        if not self._knowledge or not self._channels or not self._identity:
            return None

        link = self._links.get(link_id)
        if not link or link.status != FederationLinkStatus.ACTIVE:
            return None

        channel = self._channels.get_channel(link_id)
        if not channel:
            return None

        request = self._knowledge.build_request(
            knowledge_type=knowledge_type,
            query=query,
            max_results=max_results,
            local_instance_id=self._instance_id,
        )

        response = await channel.request_knowledge(request)

        if response:
            interaction = await self._knowledge.ingest_response(response, link)
            if self._trust:
                self._trust.update_trust(link, interaction)
            self._record_interaction(interaction)

        return response

    # ─── Coordinated Action ─────────────────────────────────────────

    async def handle_assistance_request(
        self,
        request: AssistanceRequest,
    ) -> AssistanceResponse:
        """
        Handle an inbound assistance request from a remote instance.
        """
        if not self._coordination or not self._trust:
            return AssistanceResponse(
                request_id=request.id,
                accepted=False,
                reason="Federation not initialized",
            )

        link = self._find_link_by_instance(request.requesting_instance_id)
        if not link:
            return AssistanceResponse(
                request_id=request.id,
                accepted=False,
                reason="No active federation link with this instance",
            )

        # Equor review
        equor_permitted = True
        if self._equor:
            try:
                from ecodiaos.primitives.intent import (
                    Intent, GoalDescriptor, ActionSequence, DecisionTrace,
                )
                intent = Intent(
                    goal=GoalDescriptor(
                        description=f"Assist {link.remote_name}: {request.description[:100]}",
                        target_domain="federation.assistance",
                    ),
                    plan=ActionSequence(steps=[]),
                    decision_trace=DecisionTrace(
                        reasoning=f"Assistance request from {link.remote_name}",
                    ),
                )
                check = await self._equor.review(intent)
                from ecodiaos.primitives.common import Verdict
                equor_permitted = check.verdict in (Verdict.APPROVED, Verdict.MODIFIED)
            except Exception:
                pass

        response, interaction = await self._coordination.handle_request(
            request=request,
            link=link,
            equor_permitted=equor_permitted,
        )

        self._trust.update_trust(link, interaction)
        self._record_interaction(interaction)

        if self._metrics:
            metric = "assistance.accepted" if response.accepted else "assistance.requested"
            await self._metrics.record("federation", metric, 1.0)

        return response

    async def request_assistance(
        self,
        link_id: str,
        description: str,
        knowledge_domain: str = "",
        urgency: float = 0.5,
    ) -> AssistanceResponse | None:
        """
        Request assistance from a remote federated instance.
        """
        if not self._coordination or not self._channels or not self._identity:
            return None

        link = self._links.get(link_id)
        if not link or link.status != FederationLinkStatus.ACTIVE:
            return None

        channel = self._channels.get_channel(link_id)
        if not channel:
            return None

        request = self._coordination.build_request(
            description=description,
            knowledge_domain=knowledge_domain,
            urgency=urgency,
            local_instance_id=self._instance_id,
        )

        return await channel.request_assistance(request)

    # ─── Identity ───────────────────────────────────────────────────

    @property
    def identity_card(self) -> InstanceIdentityCard | None:
        """This instance's public identity card."""
        if self._identity:
            return self._identity.identity_card
        return None

    # ─── Link Queries ───────────────────────────────────────────────

    @property
    def active_links(self) -> list[FederationLink]:
        """All active federation links."""
        return [
            l for l in self._links.values()
            if l.status == FederationLinkStatus.ACTIVE
        ]

    def get_link(self, link_id: str) -> FederationLink | None:
        return self._links.get(link_id)

    def get_link_by_instance(self, instance_id: str) -> FederationLink | None:
        return self._find_link_by_instance(instance_id)

    # ─── Trust Decay (called periodically) ──────────────────────────

    async def apply_trust_decay(self) -> None:
        """Apply trust decay to all active links (call periodically)."""
        if not self._trust:
            return
        for link in self.active_links:
            self._trust.apply_decay(link)

    # ─── Health ─────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Self-health report (implements ManagedSystem protocol)."""
        if not self._config.enabled:
            return {
                "status": "disabled",
                "enabled": False,
                "active_links": 0,
                "mean_trust": 0.0,
                "total_interactions": 0,
            }

        active = self.active_links
        return {
            "status": "healthy" if self._initialized else "starting",
            "enabled": True,
            "active_links": len(active),
            "mean_trust": self._trust.mean_trust(active) if self._trust and active else 0.0,
            "total_interactions": len(self._interaction_history),
        }

    @property
    def stats(self) -> dict[str, Any]:
        active = self.active_links
        return {
            "initialized": self._initialized,
            "enabled": self._config.enabled,
            "instance_id": self._instance_id,
            "active_links": len(active),
            "total_links": len(self._links),
            "mean_trust": round(
                self._trust.mean_trust(active) if self._trust and active else 0.0, 2
            ),
            "identity": self._identity.stats if self._identity else {},
            "trust": self._trust.stats if self._trust else {},
            "knowledge": self._knowledge.stats if self._knowledge else {},
            "coordination": self._coordination.stats if self._coordination else {},
            "channels": self._channels.stats if self._channels else {},
            "privacy": self._privacy.stats if self._privacy else {},
            "interaction_history_size": len(self._interaction_history),
            "links": [
                {
                    "id": l.id,
                    "remote_id": l.remote_instance_id,
                    "remote_name": l.remote_name,
                    "trust_level": l.trust_level.name,
                    "trust_score": round(l.trust_score, 2),
                    "status": l.status.value,
                    "shared_count": l.shared_knowledge_count,
                    "received_count": l.received_knowledge_count,
                    "successful": l.successful_interactions,
                    "failed": l.failed_interactions,
                    "violations": l.violation_count,
                }
                for l in self._links.values()
            ],
        }

    # ─── Internal ───────────────────────────────────────────────────

    def _find_link_by_instance(self, instance_id: str) -> FederationLink | None:
        """Find an active link for a given remote instance ID."""
        for link in self._links.values():
            if (
                link.remote_instance_id == instance_id
                and link.status == FederationLinkStatus.ACTIVE
            ):
                return link
        return None

    def _record_interaction(self, interaction: FederationInteraction) -> None:
        """Record an interaction in the history ring buffer."""
        self._interaction_history.append(interaction)
        if len(self._interaction_history) > self._max_history:
            self._interaction_history = self._interaction_history[-self._max_history:]

    async def _persist_links(self) -> None:
        """Persist link state to Redis (primary) with local file fallback."""
        # Always write local backup regardless of Redis availability
        self._persist_links_to_file()

        if not self._redis:
            return
        try:
            links_data = {
                link_id: link.model_dump_json()
                for link_id, link in self._links.items()
            }
            for link_id, data in links_data.items():
                await self._redis.set(
                    f"fed:links:{link_id}",
                    data,
                    ttl=None,  # Persistent
                )
            # Store the link ID index
            await self._redis.set(
                "fed:link_ids",
                ",".join(self._links.keys()),
                ttl=None,
            )
        except Exception as exc:
            self._logger.warning(
                "link_persist_redis_failed_local_backup_written",
                error=str(exc),
            )

    async def _load_links(self) -> None:
        """Load persisted links: try Redis first, fall back to local file."""
        loaded = False

        if self._redis:
            try:
                link_ids_raw = await self._redis.get("fed:link_ids")
                if link_ids_raw:
                    link_ids = link_ids_raw.split(",") if link_ids_raw else []
                    for link_id in link_ids:
                        if not link_id:
                            continue
                        data = await self._redis.get(f"fed:links:{link_id}")
                        if data:
                            link = FederationLink.model_validate_json(data)
                            self._links[link.id] = link
                    loaded = bool(self._links)
            except Exception as exc:
                self._logger.warning("link_load_redis_failed", error=str(exc))

        # Fall back to local file if Redis was empty or unavailable
        if not loaded:
            self._load_links_from_file()

        self._logger.info("links_loaded", count=len(self._links), source="redis" if loaded else "file")

    def _persist_links_to_file(self) -> None:
        """Write link state to a local JSON file as a backup."""
        try:
            backup_path = Path(self._config.data_dir or ".") / "federation_links.json"
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                link_id: link.model_dump_json()
                for link_id, link in self._links.items()
            }
            backup_path.write_text(json.dumps(data), encoding="utf-8")
        except Exception as exc:
            self._logger.debug("link_file_backup_failed", error=str(exc))

    def _load_links_from_file(self) -> None:
        """Restore link state from local JSON backup file."""
        try:
            backup_path = Path(self._config.data_dir or ".") / "federation_links.json"
            if not backup_path.exists():
                return
            raw = json.loads(backup_path.read_text(encoding="utf-8"))
            for link_id, link_json in raw.items():
                link = FederationLink.model_validate_json(link_json)
                self._links[link.id] = link
        except Exception as exc:
            self._logger.debug("link_file_restore_failed", error=str(exc))
