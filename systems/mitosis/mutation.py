"""
EcodiaOS -- Mutation Operator (Spec 26, Gap #12 / Speciation Gap #1)

Applies controlled random perturbations to genome segments during inheritance.
Mutations are deterministic given a seed for reproducibility, and all mutations
are logged as Neo4j audit trail nodes.

Mutation ranges per segment type:
  - Belief weights: +/-5% Gaussian noise per weight
  - Simula parameters: +/-3% per learnable config param
  - Personality vector: +/-0.02 per dimension (bounded [0, 1])
  - Drive calibration: +/-2% per drive weight
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING

import numpy as np
import structlog

from primitives.common import SystemID, new_id, utc_now
from primitives.genome import OrganGenomeSegment, OrganismGenome

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()


# -- Mutation magnitude per segment type ------------------------------------

_MUTATION_RANGES: dict[str, float] = {
    # SystemID value -> max relative perturbation (fraction)
    "evo": 0.05,        # Belief weights: +/-5%
    "simula": 0.03,     # Learnable config: +/-3%
    "voxis": 0.02,      # Personality vector: absolute +/-0.02
    "telos": 0.02,      # Drive calibration: +/-2%
    "nova": 0.03,       # Belief priors: +/-3%
    "soma": 0.02,       # Allostatic setpoints: +/-2%
    "atune": 0.03,      # Salience weights: +/-3%
    "fovea": 0.03,      # Precision weights: +/-3%
}

# Segments where mutation is absolute (not relative to value)
_ABSOLUTE_MUTATION_SEGMENTS = {"voxis"}


class MutationRecord:
    """In-memory record of mutations applied during one inheritance event."""

    __slots__ = ("parent_genome_id", "child_genome_id", "mutations_applied", "seed", "timestamp")

    def __init__(
        self,
        *,
        parent_genome_id: str,
        child_genome_id: str,
        mutations_applied: list[dict[str, object]],
        seed: int,
    ) -> None:
        self.parent_genome_id = parent_genome_id
        self.child_genome_id = child_genome_id
        self.mutations_applied = mutations_applied
        self.seed = seed
        self.timestamp = utc_now()


class MutationOperator:
    """
    Apply controlled random perturbations to an OrganismGenome.

    Deterministic given (genome_id, seed) -- the RNG is seeded from a hash
    of parent_genome_id + explicit seed so the same mutation can be
    reproduced for auditing.

    Parameters
    ----------
    mutation_rate : float
        Probability that any individual parameter is mutated (default 0.05).
    neo4j : Neo4jClient | None
        If provided, mutation records are persisted as Neo4j nodes.
    """

    def __init__(
        self,
        *,
        mutation_rate: float = 0.05,
        neo4j: Neo4jClient | None = None,
    ) -> None:
        self._mutation_rate = mutation_rate
        self._neo4j = neo4j
        self._log = logger.bind(subsystem="mitosis.mutation")

    # -- Public API ---------------------------------------------------------

    async def mutate(
        self,
        genome: OrganismGenome,
        *,
        seed: int | None = None,
    ) -> tuple[OrganismGenome, MutationRecord]:
        """
        Create a mutated copy of *genome* for a child instance.

        Returns (mutated_genome, record) where record lists every
        individual mutation applied.
        """
        # Derive deterministic RNG seed from parent genome ID + explicit seed
        if seed is None:
            seed = int.from_bytes(
                hashlib.sha256(genome.id.encode()).digest()[:8],
                byteorder="big",
            )
        rng = np.random.default_rng(seed)

        child_genome_id = new_id()
        mutations_applied: list[dict[str, object]] = []

        # Deep-copy segments so we don't mutate the parent
        mutated_segments: dict[SystemID, OrganGenomeSegment] = {}

        for sys_id, segment in genome.segments.items():
            # Only mutate segments that have a defined range
            magnitude = _MUTATION_RANGES.get(sys_id.value if hasattr(sys_id, "value") else str(sys_id))
            if magnitude is None or segment.version == 0:
                # No mutation defined or empty segment -- copy as-is
                mutated_segments[sys_id] = segment.model_copy(deep=True)
                continue

            is_absolute = (sys_id.value if hasattr(sys_id, "value") else str(sys_id)) in _ABSOLUTE_MUTATION_SEGMENTS
            mutated_payload, segment_mutations = self._mutate_payload(
                payload=segment.payload,
                magnitude=magnitude,
                is_absolute=is_absolute,
                rng=rng,
                system_id=str(sys_id.value if hasattr(sys_id, "value") else sys_id),
            )

            # Recompute hash for integrity
            payload_bytes = json.dumps(mutated_payload, sort_keys=True, default=str).encode()
            payload_hash = hashlib.sha256(payload_bytes).hexdigest()[:16]

            # Philosophical constraint: drive weights must sum to 1.0 after mutation
            # Children are constitutional variants, not clones. Gaussian noise on
            # drive_weight_snapshot is the mechanism for phenotypic evolution.
            # Normalize drive weights if this is the evo segment.
            sys_id_val = sys_id.value if hasattr(sys_id, "value") else str(sys_id)
            if sys_id_val == "evo":
                mutated_payload = self._apply_evo_hypothesis_dropout(
                    mutated_payload, rng=rng,
                )
            if sys_id_val == "evo" and "drive_weight_snapshot" in mutated_payload:
                dws = mutated_payload["drive_weight_snapshot"]
                if isinstance(dws, dict):
                    weight_keys = ["coherence", "care", "growth", "honesty"]
                    raw_weights = {k: max(0.01, float(dws.get(k, 0.25))) for k in weight_keys}
                    total = sum(raw_weights.values())
                    if total > 0:
                        normalized = {k: v / total for k, v in raw_weights.items()}
                        for k in weight_keys:
                            dws[k] = round(normalized[k], 6)
                        mutated_payload["drive_weight_snapshot"] = dws
                        # Recompute bytes/hash after normalization
                        payload_bytes = json.dumps(mutated_payload, sort_keys=True, default=str).encode()
                        payload_hash = hashlib.sha256(payload_bytes).hexdigest()[:16]

            mutated_segments[sys_id] = OrganGenomeSegment(
                system_id=sys_id,
                version=segment.version,
                schema_version=segment.schema_version,
                payload=mutated_payload,
                payload_hash=payload_hash,
                size_bytes=len(payload_bytes),
                extracted_at=segment.extracted_at,
            )

            mutations_applied.extend(segment_mutations)

        # Assemble child genome
        total_size = sum(s.size_bytes for s in mutated_segments.values())
        child_genome = OrganismGenome(
            id=child_genome_id,
            instance_id="",  # Will be set by caller
            generation=genome.generation + 1,
            parent_genome_id=genome.id,
            segments=mutated_segments,
            total_size_bytes=total_size,
            fitness_at_extraction=genome.fitness_at_extraction,
        )

        record = MutationRecord(
            parent_genome_id=genome.id,
            child_genome_id=child_genome_id,
            mutations_applied=mutations_applied,
            seed=seed,
        )

        self._log.info(
            "genome_mutated",
            parent_id=genome.id,
            child_id=child_genome_id,
            total_mutations=len(mutations_applied),
            segments_mutated=sum(
                1 for s in mutated_segments.values()
                if s.payload_hash != genome.segments.get(
                    next(k for k in genome.segments if k == s.system_id),
                    OrganGenomeSegment(system_id=s.system_id),
                ).payload_hash
            ),
            seed=seed,
        )

        # Persist audit record to Neo4j
        if self._neo4j is not None:
            await self._persist_mutation_record(record)

        return child_genome, record

    # -- Internal -----------------------------------------------------------

    def _mutate_payload(
        self,
        payload: dict[str, object],
        magnitude: float,
        is_absolute: bool,
        rng: np.random.Generator,
        system_id: str,
        prefix: str = "",
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        """
        Recursively mutate numeric values in a payload dict.

        Returns (mutated_payload, list_of_mutation_records).
        """
        mutated = copy.deepcopy(payload)
        mutations: list[dict[str, object]] = []

        for key, value in mutated.items():
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"

            if isinstance(value, dict):
                sub_mutated, sub_mutations = self._mutate_payload(
                    payload=value,  # type: ignore[arg-type]
                    magnitude=magnitude,
                    is_absolute=is_absolute,
                    rng=rng,
                    system_id=system_id,
                    prefix=full_key,
                )
                mutated[key] = sub_mutated
                mutations.extend(sub_mutations)
            elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                # Mutate numeric arrays element-wise
                new_list = []
                for i, v in enumerate(value):
                    if not isinstance(v, (int, float)):
                        new_list.append(v)
                        continue
                    if rng.random() < self._mutation_rate:
                        old_val = float(v)
                        if is_absolute:
                            delta = rng.normal(0, magnitude)
                        else:
                            delta = old_val * rng.normal(0, magnitude)
                        new_val = old_val + delta
                        # Bound to [0, 1] for normalized values
                        if is_absolute:
                            new_val = max(0.0, min(1.0, new_val))
                        new_list.append(new_val)
                        mutations.append({
                            "system": system_id,
                            "key": f"{full_key}[{i}]",
                            "old": old_val,
                            "new": new_val,
                            "delta": new_val - old_val,
                        })
                    else:
                        new_list.append(v)
                mutated[key] = new_list
            elif isinstance(value, (int, float)):
                if rng.random() < self._mutation_rate:
                    old_val = float(value)
                    if is_absolute:
                        delta = rng.normal(0, magnitude)
                    else:
                        delta = old_val * rng.normal(0, magnitude)
                    new_val = old_val + delta
                    if is_absolute:
                        new_val = max(0.0, min(1.0, new_val))
                    mutated[key] = new_val
                    mutations.append({
                        "system": system_id,
                        "key": full_key,
                        "old": old_val,
                        "new": new_val,
                        "delta": new_val - old_val,
                    })
            # Non-numeric values pass through unchanged

        return mutated, mutations

    def _apply_evo_hypothesis_dropout(
        self,
        payload: dict[str, object],
        rng: np.random.Generator,
    ) -> dict[str, object]:
        """
        Apply 5% hypothesis dropout to the evo segment payload.

        Per Speciation Bible §8.4: "5% of hypotheses randomly dropped" during
        genome inheritance. This uses the deterministic seeded RNG from the
        parent mutate() call so dropout is reproducible given the same seed.

        The dropout targets the "top_50_hypotheses" list (BeliefGenome schema).
        Drive weight snapshot and embeddings are unaffected.
        """
        _HYPOTHESIS_DROPOUT_RATE = 0.05

        if "top_50_hypotheses" not in payload:
            return payload

        hypotheses = payload["top_50_hypotheses"]
        if not isinstance(hypotheses, list) or not hypotheses:
            return payload

        # Deterministic per-hypothesis Bernoulli trial using seeded RNG
        keep = [h for h in hypotheses if rng.random() > _HYPOTHESIS_DROPOUT_RATE]
        dropped_count = len(hypotheses) - len(keep)

        if dropped_count > 0:
            self._log.info(
                "mutation.hypothesis_dropout",
                total=len(hypotheses),
                dropped=dropped_count,
                kept=len(keep),
                dropout_rate=_HYPOTHESIS_DROPOUT_RATE,
            )

        mutated = dict(payload)
        mutated["top_50_hypotheses"] = keep
        return mutated

    async def _persist_mutation_record(self, record: MutationRecord) -> None:
        """Write a MutationRecord node to Neo4j and link to parent/child genomes."""
        if self._neo4j is None:
            return

        try:
            record_id = new_id()
            mutations_json = json.dumps(record.mutations_applied, default=str)

            await self._neo4j.execute_write(
                """
                CREATE (m:MutationRecord {
                    id: $id,
                    parent_genome_id: $parent_genome_id,
                    child_genome_id: $child_genome_id,
                    mutations_applied: $mutations_json,
                    mutation_count: $mutation_count,
                    seed: $seed,
                    created_at: datetime($created_at)
                })
                WITH m
                OPTIONAL MATCH (pg:OrganismGenome {id: $parent_genome_id})
                FOREACH (_ IN CASE WHEN pg IS NOT NULL THEN [1] ELSE [] END |
                    CREATE (pg)-[:MUTATED_INTO]->(m)
                )
                WITH m
                OPTIONAL MATCH (cg:OrganismGenome {id: $child_genome_id})
                FOREACH (_ IN CASE WHEN cg IS NOT NULL THEN [1] ELSE [] END |
                    CREATE (m)-[:PRODUCED]->(cg)
                )
                """,
                {
                    "id": record_id,
                    "parent_genome_id": record.parent_genome_id,
                    "child_genome_id": record.child_genome_id,
                    "mutations_json": mutations_json,
                    "mutation_count": len(record.mutations_applied),
                    "seed": record.seed,
                    "created_at": record.timestamp.isoformat(),
                },
            )

            self._log.info(
                "mutation_record_persisted",
                record_id=record_id,
                parent_id=record.parent_genome_id,
                child_id=record.child_genome_id,
                mutation_count=len(record.mutations_applied),
            )
        except Exception as exc:
            self._log.error("mutation_record_persist_failed", error=str(exc))
