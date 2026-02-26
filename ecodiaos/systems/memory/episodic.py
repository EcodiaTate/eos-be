"""
EcodiaOS — Episodic Memory

Storage and retrieval of discrete experience records (Episodes).
This is the "what happened" layer of memory.
"""

from __future__ import annotations

import json
from datetime import datetime

import structlog

from ecodiaos.clients.neo4j import Neo4jClient
from ecodiaos.primitives import Episode, new_id, utc_now

logger = structlog.get_logger()


async def store_episode(
    neo4j: Neo4jClient,
    episode: Episode,
) -> str:
    """
    Store a new episode in the knowledge graph.
    Target: ≤50ms (just node creation; extraction is async).

    If a somatic_marker is present (from Soma §0.5), its 19D vector
    is stored as ``somatic_vector`` for cosine-similarity reranking
    and the full marker dict is stored as ``somatic_marker_json``.
    """
    # Flatten somatic marker to 19D vector + JSON for persistence
    somatic_vector: list[float] | None = None
    somatic_marker_json: str | None = None
    if episode.somatic_marker is not None:
        try:
            if hasattr(episode.somatic_marker, "to_vector"):
                somatic_vector = episode.somatic_marker.to_vector()
            elif episode.somatic_vector is not None:
                somatic_vector = episode.somatic_vector
            somatic_marker_json = json.dumps(
                episode.somatic_marker.model_dump()
                if hasattr(episode.somatic_marker, "model_dump")
                else str(episode.somatic_marker)
            )
        except Exception:
            logger.debug("somatic_marker_serialise_failed", exc_info=True)

    await neo4j.execute_write(
        """
        CREATE (e:Episode {
            id: $id,
            event_time: datetime($event_time),
            ingestion_time: datetime($ingestion_time),
            valid_from: datetime($valid_from),
            valid_until: $valid_until,
            source: $source,
            modality: $modality,
            raw_content: $raw_content,
            summary: $summary,
            embedding: $embedding,
            salience_composite: $salience_composite,
            salience_scores_json: $salience_scores_json,
            affect_valence: $affect_valence,
            affect_arousal: $affect_arousal,
            consolidation_level: $consolidation_level,
            last_accessed: datetime($last_accessed),
            access_count: 0,
            free_energy: $free_energy,
            somatic_vector: $somatic_vector,
            somatic_marker_json: $somatic_marker_json
        })
        """,
        {
            "id": episode.id,
            "event_time": episode.event_time.isoformat(),
            "ingestion_time": episode.ingestion_time.isoformat(),
            "valid_from": episode.valid_from.isoformat(),
            "valid_until": episode.valid_until.isoformat() if episode.valid_until else None,
            "source": episode.source,
            "modality": episode.modality,
            "raw_content": episode.raw_content,
            "summary": episode.summary,
            "embedding": episode.embedding,
            "salience_composite": episode.salience_composite,
            "salience_scores_json": json.dumps(episode.salience_scores),
            "affect_valence": episode.affect_valence,
            "affect_arousal": episode.affect_arousal,
            "consolidation_level": episode.consolidation_level,
            "last_accessed": episode.last_accessed.isoformat(),
            "free_energy": episode.free_energy,
            "somatic_vector": somatic_vector,
            "somatic_marker_json": somatic_marker_json,
        },
    )

    # Increment Self counter
    await neo4j.execute_write(
        "MATCH (s:Self) SET s.total_episodes = s.total_episodes + 1"
    )

    logger.debug(
        "episode_stored",
        episode_id=episode.id,
        source=episode.source,
        salience=episode.salience_composite,
    )
    return episode.id


async def link_episode_sequence(
    neo4j: Neo4jClient,
    previous_episode_id: str,
    current_episode_id: str,
    gap_seconds: float = 0.0,
    causal_strength: float = 0.1,
) -> None:
    """Link two episodes in temporal sequence."""
    await neo4j.execute_write(
        """
        MATCH (prev:Episode {id: $prev_id})
        MATCH (curr:Episode {id: $curr_id})
        CREATE (prev)-[:FOLLOWED_BY {
            gap_seconds: $gap,
            causal_strength: $causal
        }]->(curr)
        """,
        {
            "prev_id": previous_episode_id,
            "curr_id": current_episode_id,
            "gap": gap_seconds,
            "causal": causal_strength,
        },
    )


async def get_episode(neo4j: Neo4jClient, episode_id: str) -> dict | None:
    """Retrieve a single episode by ID."""
    results = await neo4j.execute_read(
        "MATCH (e:Episode {id: $id}) RETURN e",
        {"id": episode_id},
    )
    if results:
        return results[0]["e"]
    return None


async def get_recent_episodes(
    neo4j: Neo4jClient,
    limit: int = 20,
    min_salience: float = 0.0,
) -> list[dict]:
    """Get the most recent episodes, optionally filtered by salience."""
    return await neo4j.execute_read(
        """
        MATCH (e:Episode)
        WHERE e.salience_composite >= $min_salience
        RETURN e
        ORDER BY e.ingestion_time DESC
        LIMIT $limit
        """,
        {"min_salience": min_salience, "limit": limit},
    )


async def update_access(neo4j: Neo4jClient, episode_ids: list[str]) -> None:
    """Update access timestamps and counts for retrieved episodes (salience boost)."""
    if not episode_ids:
        return
    await neo4j.execute_write(
        """
        UNWIND $ids AS eid
        MATCH (e:Episode {id: eid})
        SET e.last_accessed = datetime(),
            e.access_count = e.access_count + 1
        """,
        {"ids": episode_ids},
    )


async def count_episodes(neo4j: Neo4jClient) -> int:
    """Get total episode count."""
    results = await neo4j.execute_read("MATCH (e:Episode) RETURN count(e) AS cnt")
    return results[0]["cnt"] if results else 0
