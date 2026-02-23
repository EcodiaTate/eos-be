"""
EcodiaOS -- Simula Evolution History

Maintains the complete, immutable record of all structural changes
applied to this EOS instance. Records are (:EvolutionRecord) nodes
in the Memory graph, linked as:
  (:ConfigVersion)-[:EVOLVED_FROM]->(:ConfigVersion)

The Honesty drive demands this record be permanent and complete.
No record can be deleted or modified after writing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from ecodiaos.systems.simula.types import ConfigVersion, EvolutionRecord

if TYPE_CHECKING:
    from ecodiaos.clients.neo4j import Neo4jClient

logger = structlog.get_logger().bind(system="simula.history")


class EvolutionHistoryManager:
    """
    Writes and queries the immutable evolution history stored in Neo4j.
    Every applied structural change produces exactly one EvolutionRecord node.
    Records are never deleted or updated.
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j
        self._log = logger

    async def record(self, record: EvolutionRecord) -> None:
        """""""""
        Write an immutable EvolutionRecord node to Neo4j.
        This is the permanent history of every structural change.
        """""""""
        await self._neo4j.execute_write(
            """
            CREATE (:EvolutionRecord {
                id: $id,
                proposal_id: $proposal_id,
                category: $category,
                description: $description,
                from_version: $from_version,
                to_version: $to_version,
                files_changed: $files_changed,
                simulation_risk: $simulation_risk,
                applied_at: $applied_at,
                rolled_back: $rolled_back,
                rollback_reason: $rollback_reason,
                simulation_episodes_tested: $simulation_episodes_tested,
                counterfactual_regression_rate: $counterfactual_regression_rate,
                dependency_blast_radius: $dependency_blast_radius,
                constitutional_alignment: $constitutional_alignment,
                resource_tokens_per_hour: $resource_tokens_per_hour,
                caution_reasoning: $caution_reasoning,
                created_at: $created_at
            })
            """,
            {
                "id": record.id,
                "proposal_id": record.proposal_id,
                "category": record.category.value,
                "description": record.description,
                "from_version": record.from_version,
                "to_version": record.to_version,
                "files_changed": record.files_changed,
                "simulation_risk": record.simulation_risk.value,
                "applied_at": record.applied_at.isoformat(),
                "rolled_back": record.rolled_back,
                "rollback_reason": record.rollback_reason,
                "simulation_episodes_tested": record.simulation_episodes_tested,
                "counterfactual_regression_rate": record.counterfactual_regression_rate,
                "dependency_blast_radius": record.dependency_blast_radius,
                "constitutional_alignment": record.constitutional_alignment,
                "resource_tokens_per_hour": record.resource_tokens_per_hour,
                "caution_reasoning": record.caution_reasoning,
                "created_at": record.created_at.isoformat(),
            },
        )
        self._log.info(
            "evolution_recorded",
            record_id=record.id,
            proposal_id=record.proposal_id,
            category=record.category.value,
            from_version=record.from_version,
            to_version=record.to_version,
        )

    async def record_version(self, version: ConfigVersion, previous_version: int | None) -> None:
        """""""""
        Write a ConfigVersion node and optionally chain it to the previous version.
        """""""""
        await self._neo4j.execute_write(
            """
            MERGE (:ConfigVersion {
                version: $version,
                timestamp: $timestamp,
                proposal_ids: $proposal_ids,
                config_hash: $config_hash
            })
            """,
            {
                "version": version.version,
                "timestamp": version.timestamp.isoformat(),
                "proposal_ids": version.proposal_ids,
                "config_hash": version.config_hash,
            },
        )
        if previous_version is not None:
            await self._neo4j.execute_write(
                """
                MATCH (new:ConfigVersion {version: $new_v})
                MATCH (prev:ConfigVersion {version: $prev_v})
                MERGE (new)-[:EVOLVED_FROM]->(prev)
                """,
                {"new_v": version.version, "prev_v": previous_version},
            )
        self._log.info(
            "config_version_recorded",
            version=version.version,
            previous_version=previous_version,
        )

    async def get_history(self, limit: int = 50) -> list[EvolutionRecord]:
        """""""""
        Retrieve the most recent N evolution records, newest first.
        """""""""
        rows = await self._neo4j.execute_read(
            """
            MATCH (r:EvolutionRecord)
            RETURN r
            ORDER BY r.created_at DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        records: list[EvolutionRecord] = []
        for row in rows:
            data = row["r"]
            records.append(EvolutionRecord(**data))
        return records

    async def get_version_chain(self) -> list[ConfigVersion]:
        """""""""
        Retrieve all ConfigVersion nodes ordered by version ASC.
        """""""""
        rows = await self._neo4j.execute_read(
            """
            MATCH (v:ConfigVersion)
            RETURN v
            ORDER BY v.version ASC
            """
        )
        versions: list[ConfigVersion] = []
        for row in rows:
            data = row["v"]
            versions.append(ConfigVersion(**data))
        return versions

    async def get_current_version(self) -> int:
        """""""""
        Return the highest config version number, or 0 if none exists.
        """""""""
        rows = await self._neo4j.execute_read(
            """
            MATCH (v:ConfigVersion)
            RETURN max(v.version) AS max_version
            """
        )
        if not rows or rows[0]["max_version"] is None:
            return 0
        return int(rows[0]["max_version"])
