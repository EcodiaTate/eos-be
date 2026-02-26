"""
EcodiaOS — Memory Consolidation

Periodic "sleep" process that:
1. Decays salience scores
2. Re-runs community detection (Leiden)
3. Promotes high-confidence extracted knowledge
4. Merges near-duplicate entities

Target: ≤60 seconds, non-blocking.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.systems.memory.salience import decay_all_salience

if TYPE_CHECKING:
    from ecodiaos.clients.neo4j import Neo4jClient

logger = structlog.get_logger()


async def run_consolidation(neo4j: Neo4jClient) -> dict[str, Any]:
    """
    Run the full consolidation pipeline.
    Called on a timer by Evo (every consolidation_interval_hours).
    """
    start = time.monotonic()
    report: dict[str, Any] = {"steps": {}}

    # Step 1: Salience decay
    try:
        decay_result = await decay_all_salience(neo4j)
        report["steps"]["salience_decay"] = decay_result
    except Exception as e:
        logger.error("consolidation_salience_decay_failed", error=str(e))
        report["steps"]["salience_decay"] = {"error": str(e)}

    # Step 2: Community detection via Leiden (requires Neo4j GDS)
    try:
        community_result = await _run_community_detection(neo4j)
        report["steps"]["community_detection"] = community_result
    except Exception as e:
        logger.warning("consolidation_community_detection_skipped", error=str(e))
        report["steps"]["community_detection"] = {"skipped": True, "reason": str(e)}

    # Step 3: Entity deduplication scan
    try:
        dedup_result = await _scan_near_duplicate_entities(neo4j)
        report["steps"]["entity_dedup"] = dedup_result
    except Exception as e:
        logger.error("consolidation_entity_dedup_failed", error=str(e))
        report["steps"]["entity_dedup"] = {"error": str(e)}

    elapsed_ms = int((time.monotonic() - start) * 1000)
    report["duration_ms"] = elapsed_ms

    logger.info("consolidation_complete", duration_ms=elapsed_ms, report=report)
    return report


async def _run_community_detection(neo4j: Neo4jClient) -> dict[str, Any]:
    """
    Run hierarchical Leiden community detection on the entity graph.
    Requires Neo4j Graph Data Science plugin.
    """
    # Project the entity graph into GDS
    try:
        await neo4j.execute_write(
            """
            CALL gds.graph.project(
                'entity_graph',
                'Entity',
                {
                    RELATES_TO: {
                        properties: ['strength'],
                        orientation: 'UNDIRECTED'
                    }
                }
            )
            """
        )
    except Exception as e:
        if "already exists" in str(e).lower():
            # Drop and recreate
            await neo4j.execute_write("CALL gds.graph.drop('entity_graph', false)")
            await neo4j.execute_write(
                """
                CALL gds.graph.project(
                    'entity_graph',
                    'Entity',
                    {
                        RELATES_TO: {
                            properties: ['strength'],
                            orientation: 'UNDIRECTED'
                        }
                    }
                )
                """
            )
        else:
            raise

    # Run Leiden
    results = await neo4j.execute_write(
        """
        CALL gds.leiden.write('entity_graph', {
            writeProperty: 'leiden_community',
            includeIntermediateCommunities: true,
            maxLevels: 5,
            gamma: 1.0,
            theta: 0.01
        })
        YIELD communityCount, modularity, ranLevels
        RETURN communityCount, modularity, ranLevels
        """
    )

    # Clean up projection
    await neo4j.execute_write("CALL gds.graph.drop('entity_graph', false)")

    community_count = 0
    modularity = 0.0
    levels = 0
    if results:
        community_count = results[0].get("communityCount", 0)
        modularity = results[0].get("modularity", 0.0)
        levels = results[0].get("ranLevels", 0)

    # Materialize Community nodes from Leiden results
    materialized = await _materialize_community_nodes(neo4j)

    return {
        "community_count": community_count,
        "modularity": modularity,
        "levels": levels,
        "communities_materialized": materialized,
    }


async def _materialize_community_nodes(neo4j: Neo4jClient) -> int:
    """
    Create Community nodes from Leiden community IDs written to entities.

    For each unique leiden_community value:
    1. MERGE a Community node with that ID
    2. Compute the community label from its top-3 member entity names
    3. Create BELONGS_TO relationships from entities to their community
    4. Compute a community embedding (mean of member embeddings)

    Returns the number of communities materialized.
    """
    try:
        # Step 1: Create/merge Community nodes for each unique community ID
        # and link entities to them
        result = await neo4j.execute_write(
            """
            MATCH (e:Entity)
            WHERE e.leiden_community IS NOT NULL
            WITH e.leiden_community AS cid, collect(e) AS members
            // Merge the Community node
            MERGE (c:Community {community_id: cid})
            SET c.member_count = size(members),
                c.updated_at = datetime()
            // Compute a label from top-3 highest-salience members
            WITH c, cid, members
            UNWIND members AS m
            WITH c, cid, m ORDER BY m.salience_score DESC
            WITH c, cid, collect(m.name)[0..3] AS top_names, collect(m) AS all_members
            SET c.label = reduce(s = '', n IN top_names | s + CASE WHEN s = '' THEN '' ELSE ', ' END + n)
            // Create BELONGS_TO relationships
            WITH c, all_members
            UNWIND all_members AS member
            MERGE (member)-[:BELONGS_TO]->(c)
            RETURN count(DISTINCT c) AS materialized
            """
        )
        materialized = result[0]["materialized"] if result else 0

        # Step 2: Compute community embeddings (mean of member embeddings)
        await neo4j.execute_write(
            """
            MATCH (c:Community)<-[:BELONGS_TO]-(e:Entity)
            WHERE e.embedding IS NOT NULL
            WITH c, collect(e.embedding) AS embeddings
            WHERE size(embeddings) > 0
            // Compute element-wise mean embedding
            WITH c, embeddings,
                 range(0, size(embeddings[0])-1) AS dims
            SET c.embedding = [i IN dims |
                reduce(s = 0.0, emb IN embeddings | s + emb[i]) / size(embeddings)
            ]
            """
        )

        logger.info("community_nodes_materialized", count=materialized)
        return materialized

    except Exception as e:
        logger.warning("community_materialization_failed", error=str(e))
        return 0


async def _scan_near_duplicate_entities(neo4j: Neo4jClient) -> dict[str, Any]:
    """
    Find and flag near-duplicate entities for potential merging.
    Full merge requires LLM confirmation (done by Evo), so we just flag here.
    """
    # Find entity pairs with very high embedding similarity
    results = await neo4j.execute_read(
        """
        MATCH (a:Entity), (b:Entity)
        WHERE a.id < b.id
          AND a.embedding IS NOT NULL
          AND b.embedding IS NOT NULL
          AND vector.similarity.cosine(a.embedding, b.embedding) > 0.92
        RETURN a.id AS id_a, a.name AS name_a,
               b.id AS id_b, b.name AS name_b,
               vector.similarity.cosine(a.embedding, b.embedding) AS similarity
        LIMIT 20
        """
    )

    return {
        "near_duplicates_found": len(results),
        "pairs": [
            {
                "a": r.get("name_a", ""),
                "b": r.get("name_b", ""),
                "similarity": r.get("similarity", 0.0),
            }
            for r in results
        ],
    }
