"""
EcodiaOS — Memory Service

The single interface to the knowledge graph. Every other system
reads from and writes to memory through this service.

Memory is the substrate of selfhood.
"""

from __future__ import annotations

from typing import Any

import structlog

from ecodiaos.clients.embedding import EmbeddingClient
from ecodiaos.clients.neo4j import Neo4jClient
from ecodiaos.config import SeedConfig
from ecodiaos.primitives import (
    AffectState,
    Entity,
    Episode,
    MemoryRetrievalRequest,
    MemoryRetrievalResponse,
    MentionRelation,
    Percept,
    SelfNode,
    SemanticRelation,
    new_id,
    utc_now,
)
from ecodiaos.systems.memory.birth import birth_instance
from ecodiaos.systems.memory.consolidation import run_consolidation
from ecodiaos.systems.memory.episodic import (
    count_episodes,
    get_episode,
    get_recent_episodes,
    link_episode_sequence,
    store_episode,
    update_access,
)
from ecodiaos.systems.memory.retrieval import hybrid_retrieve
from ecodiaos.systems.memory.schema import ensure_schema
from ecodiaos.systems.memory.semantic import (
    count_entities,
    create_entity,
    create_or_strengthen_relation,
    find_similar_entity,
    get_entity,
    get_entity_neighbours,
    link_episode_to_entity,
    merge_into_entity,
)

logger = structlog.get_logger()


class MemoryService:
    """
    The Memory & Identity Core.

    This is the ground truth of identity. Every other system reads from
    and writes to this substrate. If you lose the graph, you lose the self.
    """

    system_id: str = "memory"

    def __init__(
        self,
        neo4j: Neo4jClient,
        embedding_client: EmbeddingClient,
    ) -> None:
        self._neo4j = neo4j
        self._embedding = embedding_client
        self._instance_id: str | None = None
        # Episode sequence tracking — links consecutive episodes with FOLLOWED_BY
        self._last_episode_id: str | None = None
        self._last_episode_time: float | None = None  # monotonic seconds
        # Soma for somatic marker stamping and reranking
        self._soma: Any = None

    # ─── Lifecycle ────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Ensure schema exists (idempotent)."""
        await ensure_schema(self._neo4j)
        logger.info("memory_service_initialised")

    def set_soma(self, soma: Any) -> None:
        """Wire Soma for somatic marker stamping and reranking."""
        self._soma = soma
        logger.info("soma_wired_to_memory")

    async def get_self(self) -> SelfNode | None:
        """Get the current instance's Self node, or None if not yet born."""
        results = await self._neo4j.execute_read(
            "MATCH (s:Self) RETURN s LIMIT 1"
        )
        if not results:
            return None

        data = results[0]["s"]
        self._instance_id = data.get("instance_id")
        # Reconstruct current_affect from flattened scalar properties
        current_affect = {
            "valence": data.get("affect_valence", 0.0),
            "arousal": data.get("affect_arousal", 0.0),
            "dominance": data.get("affect_dominance", 0.5),
            "curiosity": data.get("affect_curiosity", 0.0),
            "care_activation": data.get("affect_care_activation", 0.0),
            "coherence_stress": data.get("affect_coherence_stress", 0.0),
        }
        # Convert neo4j.time.DateTime to Python datetime
        born_at = data.get("born_at", utc_now())
        if hasattr(born_at, "to_native"):
            born_at = born_at.to_native()
        # Parse personality_json if stored (Neo4j stores as string, needs json.loads)
        raw_pjson = data.get("personality_json", None)
        personality_json: dict[str, float] = {}
        if raw_pjson:
            if isinstance(raw_pjson, str):
                import json
                try:
                    personality_json = json.loads(raw_pjson)
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(raw_pjson, dict):
                personality_json = raw_pjson

        return SelfNode(
            instance_id=data.get("instance_id", ""),
            name=data.get("name", ""),
            born_at=born_at,
            current_affect=current_affect,
            autonomy_level=data.get("autonomy_level", 1),
            personality_vector=data.get("personality_vector", []),
            personality_json=personality_json,
            cycle_count=data.get("cycle_count", 0),
            total_episodes=data.get("total_episodes", 0),
            total_entities=data.get("total_entities", 0),
            total_communities=data.get("total_communities", 0),
        )

    async def birth(self, seed: SeedConfig, instance_id: str) -> dict:
        """Birth a new instance from a seed configuration."""
        result = await birth_instance(
            self._neo4j, self._embedding, seed, instance_id,
        )
        self._instance_id = instance_id
        return result

    # ─── Episode Storage (from Atune) ─────────────────────────────

    async def store_percept(
        self,
        percept: Percept,
        salience_composite: float = 0.0,
        salience_scores: dict[str, float] | None = None,
        affect_valence: float = 0.0,
        affect_arousal: float = 0.0,
        free_energy: float = 0.0,
    ) -> str:
        """
        Store a percept as an episode in the knowledge graph.

        Returns the episode ID.
        Entity extraction should be done asynchronously after this returns.

        Automatically links the new episode to the previous one via
        FOLLOWED_BY relationship, creating temporal causal chains that
        enable sequential memory and "what happened before/after" queries.
        """
        import time as _time

        # Compute embedding if not already present
        embedding = percept.content.embedding
        if not embedding and percept.content.raw:
            embedding = await self._embedding.embed(percept.content.raw)

        # Stamp somatic marker at encoding time
        somatic_marker = None
        if self._soma is not None:
            try:
                somatic_marker = self._soma.get_somatic_marker()
            except Exception as exc:
                logger.debug("soma_marker_error", error=str(exc))

        episode = Episode(
            event_time=percept.timestamp,
            ingestion_time=utc_now(),
            source=f"{percept.source.system.value}:{percept.source.channel}",
            modality=percept.source.modality.value,
            raw_content=percept.content.raw,
            summary=percept.content.raw[:200],  # Placeholder; LLM summary done async
            embedding=embedding,
            salience_composite=salience_composite,
            salience_scores=salience_scores or {},
            affect_valence=affect_valence,
            affect_arousal=affect_arousal,
            free_energy=free_energy,
            somatic_marker=somatic_marker,
        )

        episode_id = await store_episode(self._neo4j, episode)

        # Link to previous episode (temporal sequence)
        now = _time.monotonic()
        if self._last_episode_id is not None:
            gap = now - self._last_episode_time if self._last_episode_time else 0.0
            # Only link if episodes are within 1 hour of each other
            if gap < 3600.0:
                try:
                    # Higher causal strength for closely spaced episodes
                    causal = max(0.05, min(0.8, 1.0 - (gap / 300.0)))
                    await link_episode_sequence(
                        self._neo4j,
                        previous_episode_id=self._last_episode_id,
                        current_episode_id=episode_id,
                        gap_seconds=gap,
                        causal_strength=causal,
                    )
                except Exception:
                    logger.debug("episode_link_failed", exc_info=True)

        self._last_episode_id = episode_id
        self._last_episode_time = now

        return episode_id

    # ─── Retrieval (from any system) ──────────────────────────────

    async def retrieve(
        self,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        max_results: int = 10,
        salience_floor: float = 0.0,
        include_communities: bool = False,
    ) -> MemoryRetrievalResponse:
        """
        Hybrid retrieval. Must respond within 200ms.
        """
        # Compute embedding from text if not provided
        if query_text and not query_embedding:
            query_embedding = await self._embedding.embed(query_text)

        request = MemoryRetrievalRequest(
            query_text=query_text,
            query_embedding=query_embedding,
            max_results=max_results,
            salience_floor=salience_floor,
            include_communities=include_communities,
        )

        response = await hybrid_retrieve(self._neo4j, request)

        # Somatic reranking: boost candidates with somatic similarity to current state
        if self._soma is not None and response.traces:
            try:
                response.traces = self._soma.somatic_rerank(response.traces)
            except Exception as exc:
                logger.debug("somatic_rerank_error", error=str(exc))

        # Update access timestamps for retrieved episodes (salience boost)
        retrieved_ids = [r.node_id for r in response.traces]
        await update_access(self._neo4j, retrieved_ids)

        return response

    # ─── Entity Management (from extraction pipeline) ─────────────

    async def resolve_and_create_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
    ) -> tuple[str, bool]:
        """
        Find or create an entity. Handles deduplication.

        Returns: (entity_id, was_created)
        """
        # Compute embedding
        embed_text = f"{name}: {description}"
        embedding = await self._embedding.embed(embed_text)

        # Try to find existing
        existing = await find_similar_entity(self._neo4j, name, embedding)
        if existing:
            await merge_into_entity(self._neo4j, existing["id"], description)
            return existing["id"], False

        # Create new
        entity = Entity(
            name=name,
            type=entity_type,
            description=description,
            embedding=embedding,
        )
        entity_id = await create_entity(self._neo4j, entity)
        return entity_id, True

    async def link_mention(self, mention: MentionRelation) -> None:
        """Link an episode to an entity."""
        await link_episode_to_entity(self._neo4j, mention)

    async def link_relation(self, relation: SemanticRelation) -> None:
        """Create or strengthen a semantic relation."""
        await create_or_strengthen_relation(self._neo4j, relation)

    # ─── State Queries ────────────────────────────────────────────

    async def get_constitution(self) -> dict | None:
        """Get the current constitutional state."""
        results = await self._neo4j.execute_read(
            """
            MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
            RETURN c
            """
        )
        return results[0]["c"] if results else None

    async def get_core_entities(self) -> list[dict]:
        """Get all entities marked as core to the instance's identity."""
        return await self._neo4j.execute_read(
            """
            MATCH (e:Entity {is_core_identity: true})
            RETURN e
            ORDER BY e.salience_score DESC
            """
        )

    async def update_affect(self, affect: AffectState) -> None:
        """Update the current affect state on the Self node (flattened scalars)."""
        affect_map = affect.to_map()
        await self._neo4j.execute_write(
            """
            MATCH (s:Self)
            SET s.affect_valence = $affect_valence,
                s.affect_arousal = $affect_arousal,
                s.affect_dominance = $affect_dominance,
                s.affect_curiosity = $affect_curiosity,
                s.affect_care_activation = $affect_care_activation,
                s.affect_coherence_stress = $affect_coherence_stress
            """,
            {
                "affect_valence": affect_map.get("valence", 0.0),
                "affect_arousal": affect_map.get("arousal", 0.0),
                "affect_dominance": affect_map.get("dominance", 0.0),
                "affect_curiosity": affect_map.get("curiosity", 0.0),
                "affect_care_activation": affect_map.get("care_activation", 0.0),
                "affect_coherence_stress": affect_map.get("coherence_stress", 0.0),
            },
        )

    async def increment_cycle_count(self) -> None:
        """Increment the cognitive cycle counter."""
        await self._neo4j.execute_write(
            "MATCH (s:Self) SET s.cycle_count = s.cycle_count + 1"
        )

    # ─── Consolidation ────────────────────────────────────────────

    async def consolidate(self) -> dict:
        """Run the consolidation pipeline (memory sleep cycle)."""
        return await run_consolidation(self._neo4j)

    # ─── Stats ────────────────────────────────────────────────────

    async def stats(self) -> dict:
        """Get graph statistics."""
        ep_count = await count_episodes(self._neo4j)
        ent_count = await count_entities(self._neo4j)
        return {
            "episode_count": ep_count,
            "entity_count": ent_count,
        }

    # ─── Health ───────────────────────────────────────────────────

    async def health(self) -> dict:
        """Health check for the memory system (must complete within 2s)."""
        neo4j_health = await self._neo4j.health_check()
        return {
            "status": "healthy" if neo4j_health["status"] == "connected" else "degraded",
            "neo4j": neo4j_health,
            "instance_id": self._instance_id,
        }
