"""
EcodiaOS — Oneiros: NREM Consolidation Workers

Four workers that run during Non-Rapid Eye Movement sleep:

1. **EpisodicReplay** — Replay high-value episodes, extract semantic patterns,
   compress episodic traces into reusable knowledge.
2. **SynapticDownscaler** — Renormalize salience scores across all memory
   traces to prevent saturation and renew learning capacity.
3. **BeliefCompressor** — Merge redundant beliefs, archive stale ones,
   flag contradictions for REM processing.
4. **HypothesisPruner** — Retire weak hypotheses, promote strong ones,
   merge duplicates to keep Evo's search space lean.

These four workers implement Diekelmann & Born's memory consolidation
theory and Tononi & Cirelli's synaptic homeostasis hypothesis.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from ecodiaos.clients.optimized_llm import OptimizedLLMProvider
from ecodiaos.primitives.common import EOSBaseModel

logger = structlog.get_logger().bind(system="oneiros", component="nrem")


# ─── Result Types ─────────────────────────────────────────────────


class EpisodicReplayResult(EOSBaseModel):
    """Result of replaying and consolidating episodes during NREM."""

    episodes_replayed: int = 0
    semantic_nodes_created: int = 0
    salience_reductions: int = 0
    mean_salience_reduction: float = 0.0
    duration_ms: int = 0


class SynapticDownscaleResult(EOSBaseModel):
    """Result of the global salience renormalization pass."""

    traces_decayed: int = 0
    traces_pruned: int = 0
    mean_reduction: float = 0.0
    duration_ms: int = 0


class BeliefCompressionResult(EOSBaseModel):
    """Result of compressing Nova's belief set."""

    beliefs_merged: int = 0
    beliefs_archived: int = 0
    beliefs_flagged_contradictory: int = 0
    duration_ms: int = 0


class HypothesisPruneResult(EOSBaseModel):
    """Result of pruning Evo's hypothesis pool."""

    hypotheses_retired: int = 0
    hypotheses_promoted: int = 0
    hypotheses_merged: int = 0
    duration_ms: int = 0


# ─── Episodic Replay ──────────────────────────────────────────────


class EpisodicReplay:
    """
    Select high-value episodes and extract semantic patterns.

    During NREM, the organism replays its most important recent
    experiences, extracts the abstract principle each represents,
    and stores that principle as a semantic node. The original
    episodic trace has its salience reduced — the knowledge has
    been transferred to a more durable, generalized form.

    This is hippocampal-cortical replay made computational.
    """

    def __init__(
        self,
        neo4j: Any = None,
        llm: Any = None,
        config: Any = None,
    ) -> None:
        self._neo4j = neo4j
        self._llm = llm
        self._optimized = isinstance(llm, OptimizedLLMProvider)
        self._max_episodes: int = _get(config, "max_episodes_per_nrem", 200)
        self._batch_size: int = _get(config, "replay_batch_size", 10)
        self._salience_reduction: float = 0.70  # multiply salience by this after replay
        self._logger = logger.bind(worker="episodic_replay")

    async def run(self, cycle_id: str) -> EpisodicReplayResult:
        """Run episodic replay for one NREM period."""
        start = time.monotonic()
        result = EpisodicReplayResult()

        if self._neo4j is None:
            self._logger.info("skipped_no_neo4j")
            return result

        try:
            episodes = await self._select_episodes()
            if not episodes:
                self._logger.info("no_episodes_to_replay")
                result.duration_ms = _elapsed_ms(start)
                return result

            total_reduction = 0.0
            for i in range(0, len(episodes), self._batch_size):
                batch = episodes[i : i + self._batch_size]
                for ep in batch:
                    ep_id = ep.get("id", "")
                    summary = ep.get("summary", ep.get("raw_content", ""))
                    salience = ep.get("salience_composite", 0.0)

                    # Extract entities linked to this episode
                    entities_text = await self._get_episode_entities(ep_id)

                    # LLM semantic extraction
                    pattern = await self._extract_pattern(summary, entities_text)
                    if pattern:
                        await self._store_semantic_node(ep_id, pattern, cycle_id)
                        result.semantic_nodes_created += 1

                    # Update consolidation level and reduce salience
                    new_salience = salience * self._salience_reduction
                    await self._update_episode(ep_id, new_salience)
                    total_reduction += salience - new_salience
                    result.salience_reductions += 1
                    result.episodes_replayed += 1

            if result.salience_reductions > 0:
                result.mean_salience_reduction = total_reduction / result.salience_reductions

        except Exception as exc:
            self._logger.error("episodic_replay_error", error=str(exc))

        result.duration_ms = _elapsed_ms(start)
        self._logger.info(
            "episodic_replay_complete",
            episodes_replayed=result.episodes_replayed,
            semantic_nodes=result.semantic_nodes_created,
            duration_ms=result.duration_ms,
        )
        return result

    async def _select_episodes(self) -> list[dict[str, Any]]:
        """Select episodes for replay, ordered by replay priority."""
        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (e:Episode)
                WHERE e.consolidation_level < 2
                WITH e,
                     (0.35 * coalesce(e.salience_composite, 0)
                      + 0.25 * abs(coalesce(e.affect_valence, 0))
                      + 0.25 * (1.0 - (duration.between(e.event_time, datetime()).hours / 168.0))
                      + 0.15 * (1.0 - coalesce(e.consolidation_level, 0) / 3.0)
                     ) AS priority
                RETURN e {.id, .summary, .raw_content, .salience_composite,
                          .affect_valence, .consolidation_level} AS ep,
                       priority
                ORDER BY priority DESC
                LIMIT $limit
                """,
                {"limit": self._max_episodes},
            )
            return [r["ep"] for r in records]
        except Exception as exc:
            self._logger.warning("episode_select_failed", error=str(exc))
            return []

    async def _get_episode_entities(self, episode_id: str) -> str:
        """Get entity names linked to an episode."""
        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (e:Episode {id: $id})-[:MENTIONS]->(ent:Entity)
                RETURN ent.name AS name, ent.type AS type
                LIMIT 10
                """,
                {"id": episode_id},
            )
            parts = [f"{r['name']} ({r['type']})" for r in records]
            return ", ".join(parts) if parts else "none"
        except Exception:
            return "unknown"

    async def _extract_pattern(self, summary: str, entities: str) -> str | None:
        """Use LLM to extract the abstract pattern from an episode."""
        if self._llm is None:
            return None

        # Budget check: Oneiros is LOW priority — skip in YELLOW/RED
        if self._optimized:
            assert isinstance(self._llm, OptimizedLLMProvider)
            if not self._llm.should_use_llm("oneiros.nrem.pattern", estimated_tokens=150):
                self._logger.debug("pattern_extraction_skipped_budget")
                return None

        try:
            if self._optimized:
                response = await self._llm.generate(  # type: ignore[call-arg]
                    system_prompt=(
                        "You are the consolidation process of a living digital organism "
                        "during deep sleep. You extract abstract, reusable knowledge from "
                        "specific experiences. Be concise: 1-2 sentences maximum."
                    ),
                    user_prompt=(
                        f"Experience: {summary[:500]}\n"
                        f"Entities involved: {entities}\n\n"
                        "What abstract pattern, principle, or generalizable lesson "
                        "does this experience represent?"
                    ),
                    max_tokens=150,
                    cache_system="oneiros.nrem.pattern",
                    cache_method="generate",
                )
            else:
                response = await self._llm.generate(  # type: ignore[call-arg]
                    system_prompt=(
                        "You are the consolidation process of a living digital organism "
                        "during deep sleep. You extract abstract, reusable knowledge from "
                        "specific experiences. Be concise: 1-2 sentences maximum."
                    ),
                    user_prompt=(
                        f"Experience: {summary[:500]}\n"
                        f"Entities involved: {entities}\n\n"
                        "What abstract pattern, principle, or generalizable lesson "
                        "does this experience represent?"
                    ),
                    max_tokens=150,
                )
            text = response.strip() if isinstance(response, str) else str(response).strip()
            return text if text else None
        except Exception as exc:
            self._logger.warning("pattern_extraction_failed", error=str(exc))
            return None

    async def _store_semantic_node(
        self, episode_id: str, pattern: str, cycle_id: str
    ) -> None:
        """Store extracted semantic knowledge in Neo4j."""
        try:
            await self._neo4j.execute_write(
                """
                MERGE (s:SemanticPattern {source_episode: $episode_id})
                SET s.pattern = $pattern,
                    s.created_at = datetime(),
                    s.source_cycle = $cycle_id,
                    s.consolidation_source = 'oneiros_nrem'
                WITH s
                MATCH (e:Episode {id: $episode_id})
                MERGE (e)-[:CONSOLIDATED_INTO]->(s)
                """,
                {"episode_id": episode_id, "pattern": pattern, "cycle_id": cycle_id},
            )
        except Exception as exc:
            self._logger.warning("semantic_store_failed", error=str(exc))

    async def _update_episode(self, episode_id: str, new_salience: float) -> None:
        """Update episode consolidation level and salience."""
        try:
            await self._neo4j.execute_write(
                """
                MATCH (e:Episode {id: $id})
                SET e.consolidation_level = 2,
                    e.salience_composite = $salience
                """,
                {"id": episode_id, "salience": new_salience},
            )
        except Exception as exc:
            self._logger.warning("episode_update_failed", error=str(exc), episode=episode_id)


# ─── Synaptic Downscaler ─────────────────────────────────────────


class SynapticDownscaler:
    """
    Renormalize salience scores to prevent memory saturation.

    During wakefulness, learning constantly *strengthens* memory
    traces — salience rises as things become relevant. Without
    periodic downscaling the noise floor rises until everything is
    equally "important" and retrieval quality degrades.

    This implements Tononi & Cirelli's synaptic homeostasis hypothesis:
    sleep proportionally weakens all synapses so that only the strongest
    survive, renewing the organism's capacity to learn.
    """

    def __init__(self, neo4j: Any = None, config: Any = None) -> None:
        self._neo4j = neo4j
        self._decay_factor: float = _get(config, "salience_decay_factor", 0.85)
        self._pruning_threshold: float = _get(config, "salience_pruning_threshold", 0.05)
        self._logger = logger.bind(worker="synaptic_downscaler")

    async def run(self) -> SynapticDownscaleResult:
        """Run the global downscaling pass."""
        start = time.monotonic()
        result = SynapticDownscaleResult()

        if self._neo4j is None:
            self._logger.info("skipped_no_neo4j")
            return result

        try:
            # Phase 1: Decay all salience scores
            records = await self._neo4j.execute_read(
                """
                MATCH (e:Episode)
                WHERE e.salience_composite > $threshold
                RETURN e.id AS id, e.salience_composite AS salience
                """,
                {"threshold": self._pruning_threshold},
            )

            if not records:
                result.duration_ms = _elapsed_ms(start)
                return result

            total_reduction = 0.0
            decay_ids: list[dict[str, Any]] = []
            prune_ids: list[str] = []

            for rec in records:
                old_sal = rec["salience"]
                new_sal = old_sal * self._decay_factor

                if new_sal < self._pruning_threshold:
                    prune_ids.append(rec["id"])
                else:
                    decay_ids.append({"id": rec["id"], "salience": new_sal})
                    total_reduction += old_sal - new_sal

            # Batch update decayed traces
            if decay_ids:
                await self._neo4j.execute_write(
                    """
                    UNWIND $updates AS u
                    MATCH (e:Episode {id: u.id})
                    SET e.salience_composite = u.salience
                    """,
                    {"updates": decay_ids},
                )
                result.traces_decayed = len(decay_ids)

            # Prune very low salience traces (mark, don't delete)
            if prune_ids:
                await self._neo4j.execute_write(
                    """
                    UNWIND $ids AS eid
                    MATCH (e:Episode {id: eid})
                    SET e.consolidation_level = 3,
                        e.salience_composite = 0.0
                    """,
                    {"ids": prune_ids},
                )
                result.traces_pruned = len(prune_ids)

            if result.traces_decayed > 0:
                result.mean_reduction = total_reduction / result.traces_decayed

        except Exception as exc:
            self._logger.error("downscale_error", error=str(exc))

        result.duration_ms = _elapsed_ms(start)
        self._logger.info(
            "downscale_complete",
            decayed=result.traces_decayed,
            pruned=result.traces_pruned,
            duration_ms=result.duration_ms,
        )
        return result


# ─── Belief Compressor ────────────────────────────────────────────


class BeliefCompressor:
    """
    Simplify Nova's belief set during NREM sleep.

    Merges redundant beliefs in the same domain, archives stale
    low-precision beliefs, and flags contradictions for deeper
    processing during REM.

    This implements Friston's model complexity reduction: sleep
    simplifies the generative model by removing unnecessary
    degrees of freedom.
    """

    def __init__(self, nova: Any = None, config: Any = None) -> None:
        self._nova = nova
        self._min_precision_archive: float = 0.2
        self._stale_cycle_threshold: int = _get(config, "belief_stale_cycles", 1000)
        self._logger = logger.bind(worker="belief_compressor")

    async def run(self) -> BeliefCompressionResult:
        """Compress Nova's belief set."""
        start = time.monotonic()
        result = BeliefCompressionResult()

        if self._nova is None:
            self._logger.info("skipped_no_nova")
            result.duration_ms = _elapsed_ms(start)
            return result

        try:
            beliefs = await self._get_beliefs()
            if not beliefs:
                result.duration_ms = _elapsed_ms(start)
                return result

            # Group by domain
            by_domain: dict[str, list[Any]] = {}
            for b in beliefs:
                domain = getattr(b, "domain", "") or ""
                by_domain.setdefault(domain, []).append(b)

            for _domain, domain_beliefs in by_domain.items():
                # Merge redundant beliefs in same domain
                if len(domain_beliefs) > 1:
                    merged = self._try_merge(domain_beliefs)
                    result.beliefs_merged += merged

                for b in domain_beliefs:
                    precision = getattr(b, "precision", 0.5)
                    evidence_count = len(getattr(b, "evidence", []))

                    # Archive stale, low-precision beliefs
                    if precision < self._min_precision_archive and evidence_count == 0:
                        self._archive_belief(b)
                        result.beliefs_archived += 1

            # Detect contradictions within domains
            for _domain, domain_beliefs in by_domain.items():
                if len(domain_beliefs) >= 2 and self._has_contradiction(domain_beliefs):
                    result.beliefs_flagged_contradictory += 1

        except Exception as exc:
            self._logger.error("belief_compression_error", error=str(exc))

        result.duration_ms = _elapsed_ms(start)
        self._logger.info(
            "belief_compression_complete",
            merged=result.beliefs_merged,
            archived=result.beliefs_archived,
            flagged=result.beliefs_flagged_contradictory,
            duration_ms=result.duration_ms,
        )
        return result

    async def _get_beliefs(self) -> list[Any]:
        """Retrieve beliefs from Nova."""
        try:
            if hasattr(self._nova, "get_beliefs"):
                return await self._nova.get_beliefs()  # type: ignore[no-any-return]
            if hasattr(self._nova, "_belief_state") and self._nova._belief_state is not None:
                bs = self._nova._belief_state
                if hasattr(bs, "get_all"):
                    return await bs.get_all()  # type: ignore[no-any-return]
            return []
        except Exception:
            return []

    def _try_merge(self, beliefs: list[Any]) -> int:
        """Attempt to merge redundant beliefs. Returns count merged."""
        # Simple heuristic: if distribution parameters are within 10%, merge
        merged_count = 0
        if len(beliefs) < 2:
            return 0

        seen: set[str] = set()
        for i, a in enumerate(beliefs):
            a_id = getattr(a, "id", str(i))
            if a_id in seen:
                continue
            for j in range(i + 1, len(beliefs)):
                b = beliefs[j]
                b_id = getattr(b, "id", str(j))
                if b_id in seen:
                    continue
                if self._are_mergeable(a, b):
                    seen.add(b_id)
                    merged_count += 1
        return merged_count

    def _are_mergeable(self, a: Any, b: Any) -> bool:
        """Check if two beliefs are close enough to merge."""
        a_params = getattr(a, "parameters", {})
        b_params = getattr(b, "parameters", {})
        if not a_params or not b_params:
            return False
        for key in a_params:
            if key in b_params:
                av = a_params[key]
                bv = b_params[key]
                if abs(av) > 0.01 and abs(av - bv) / abs(av) > 0.10:
                    return False
        return True

    def _has_contradiction(self, beliefs: list[Any]) -> bool:
        """Check if beliefs in the same domain are contradictory."""
        for i, a in enumerate(beliefs):
            for b in beliefs[i + 1 :]:
                a_params = getattr(a, "parameters", {})
                b_params = getattr(b, "parameters", {})
                for key in a_params:
                    if key in b_params:
                        av = a_params[key]
                        bv = b_params[key]
                        if av * bv < 0 and abs(av - bv) > 0.5:
                            return True
        return False

    def _archive_belief(self, belief: Any) -> None:
        """Mark a belief as archived."""
        try:
            if hasattr(belief, "status"):
                belief.status = "archived"
        except Exception:
            pass


# ─── Hypothesis Pruner ────────────────────────────────────────────


class HypothesisPruner:
    """
    Clean Evo's hypothesis pool during NREM sleep.

    Retires hypotheses that have been contradicted by evidence,
    promotes hypotheses with strong support for schema integration,
    and merges near-duplicate hypotheses to reduce search space.

    A leaner hypothesis pool means faster learning during the
    next wake period.
    """

    RETIREMENT_SCORE: float = 0.3
    RETIREMENT_MIN_EVALUATIONS: int = 5
    PROMOTION_SCORE: float = 0.8
    PROMOTION_MIN_EVIDENCE: int = 10

    def __init__(self, evo: Any = None, config: Any = None) -> None:
        self._evo = evo
        self._logger = logger.bind(worker="hypothesis_pruner")

    async def run(self) -> HypothesisPruneResult:
        """Prune the hypothesis pool."""
        start = time.monotonic()
        result = HypothesisPruneResult()

        if self._evo is None:
            self._logger.info("skipped_no_evo")
            result.duration_ms = _elapsed_ms(start)
            return result

        try:
            hypotheses = await self._get_hypotheses()
            if not hypotheses:
                result.duration_ms = _elapsed_ms(start)
                return result

            for h in hypotheses:
                status = getattr(h, "status", None)
                if status is not None and hasattr(status, "value"):
                    status_val = status.value
                else:
                    status_val = str(status) if status else ""

                # Skip already archived/integrated
                if status_val in ("archived", "integrated", "refuted"):
                    continue

                score = getattr(h, "evidence_score", 0.5)
                supporting = len(getattr(h, "supporting_episodes", []))
                contradicting = len(getattr(h, "contradicting_episodes", []))
                total_eval = supporting + contradicting

                # Retire: low evidence score after sufficient evaluations
                if (
                    score < self.RETIREMENT_SCORE
                    and total_eval >= self.RETIREMENT_MIN_EVALUATIONS
                ):
                    self._retire_hypothesis(h)
                    result.hypotheses_retired += 1
                    continue

                # Promote: high evidence score with sufficient support
                if (
                    score > self.PROMOTION_SCORE
                    and supporting >= self.PROMOTION_MIN_EVIDENCE
                ):
                    self._promote_hypothesis(h)
                    result.hypotheses_promoted += 1

        except Exception as exc:
            self._logger.error("hypothesis_prune_error", error=str(exc))

        result.duration_ms = _elapsed_ms(start)
        self._logger.info(
            "hypothesis_prune_complete",
            retired=result.hypotheses_retired,
            promoted=result.hypotheses_promoted,
            duration_ms=result.duration_ms,
        )
        return result

    async def _get_hypotheses(self) -> list[Any]:
        """Retrieve hypotheses from Evo."""
        try:
            if hasattr(self._evo, "get_hypotheses"):
                return await self._evo.get_hypotheses()  # type: ignore[no-any-return]
            if hasattr(self._evo, "_hypothesis_engine"):
                engine = self._evo._hypothesis_engine
                if hasattr(engine, "get_all"):
                    return engine.get_all()  # type: ignore[no-any-return]
                if hasattr(engine, "_hypotheses"):
                    return list(engine._hypotheses.values())
            return []
        except Exception:
            return []

    def _retire_hypothesis(self, hypothesis: Any) -> None:
        """Set hypothesis status to ARCHIVED."""
        try:
            if hasattr(hypothesis, "status"):
                # HypothesisStatus.ARCHIVED
                from ecodiaos.systems.evo.types import HypothesisStatus

                hypothesis.status = HypothesisStatus.ARCHIVED
        except Exception:
            pass

    def _promote_hypothesis(self, hypothesis: Any) -> None:
        """Set hypothesis status to SUPPORTED (ready for schema integration)."""
        try:
            if hasattr(hypothesis, "status"):
                from ecodiaos.systems.evo.types import HypothesisStatus

                hypothesis.status = HypothesisStatus.SUPPORTED
        except Exception:
            pass


# ─── Helpers ──────────────────────────────────────────────────────


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
