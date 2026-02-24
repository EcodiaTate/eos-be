"""
Tests for Oneiros NREM Consolidation Workers.

Covers:
  - EpisodicReplay: episode selection, replay, pattern extraction
  - SynapticDownscaler: salience decay, pruning
  - BeliefCompressor: archiving, merging, contradiction detection
  - HypothesisPruner: retirement, promotion, status filtering
"""

from __future__ import annotations

from typing import Any

import pytest

from ecodiaos.systems.oneiros.nrem import (
    BeliefCompressor,
    EpisodicReplay,
    HypothesisPruner,
    SynapticDownscaler,
)


# ─── Mock Stubs ──────────────────────────────────────────────────


class FakeNeo4j:
    """Minimal Neo4j mock that stores queries and returns preconfigured data."""

    def __init__(self, read_returns: list[Any] | None = None) -> None:
        self._read_returns = read_returns or []
        self._read_call_count = 0
        self._write_calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_read(self, query: str, params: dict[str, Any] | None = None) -> list[Any]:
        idx = min(self._read_call_count, len(self._read_returns) - 1)
        self._read_call_count += 1
        if idx < 0:
            return []
        return self._read_returns[idx]

    async def execute_write(self, query: str, params: dict[str, Any] | None = None) -> None:
        self._write_calls.append((query, params or {}))


class FakeLLM:
    """Minimal LLM mock that returns a fixed response."""

    def __init__(self, response: str = "A reusable pattern.") -> None:
        self._response = response
        self.call_count = 0

    async def generate(self, **kwargs: Any) -> str:
        self.call_count += 1
        return self._response


class FakeBelief:
    """Stub for a Nova belief."""

    def __init__(
        self,
        belief_id: str = "b1",
        domain: str = "social",
        precision: float = 0.5,
        evidence: list[str] | None = None,
        parameters: dict[str, float] | None = None,
        status: str = "active",
    ) -> None:
        self.id = belief_id
        self.domain = domain
        self.precision = precision
        self.evidence = evidence or []
        self.parameters = parameters or {}
        self.status = status


class FakeNova:
    """Stub for the Nova system."""

    def __init__(self, beliefs: list[FakeBelief] | None = None) -> None:
        self._beliefs = beliefs or []

    async def get_beliefs(self) -> list[FakeBelief]:
        return self._beliefs


class FakeHypothesis:
    """Stub for an Evo hypothesis."""

    def __init__(
        self,
        hyp_id: str = "h1",
        evidence_score: float = 0.5,
        supporting: int = 0,
        contradicting: int = 0,
        status: str = "active",
    ) -> None:
        self.id = hyp_id
        self.evidence_score = evidence_score
        self.supporting_episodes = ["s"] * supporting
        self.contradicting_episodes = ["c"] * contradicting
        self.status = status


class FakeEvo:
    """Stub for the Evo system."""

    def __init__(self, hypotheses: list[FakeHypothesis] | None = None) -> None:
        self._hypotheses = hypotheses or []

    async def get_hypotheses(self) -> list[FakeHypothesis]:
        return self._hypotheses


# ─── EpisodicReplay ──────────────────────────────────────────────


class TestEpisodicReplay:
    @pytest.mark.asyncio
    async def test_run_without_neo4j_returns_empty(self):
        replay = EpisodicReplay(neo4j=None, llm=None)
        result = await replay.run("cycle_1")
        assert result.episodes_replayed == 0
        assert result.semantic_nodes_created == 0

    @pytest.mark.asyncio
    async def test_run_with_no_episodes_returns_empty(self):
        neo4j = FakeNeo4j(read_returns=[[]])  # No episodes found
        replay = EpisodicReplay(neo4j=neo4j, llm=None)
        result = await replay.run("cycle_1")
        assert result.episodes_replayed == 0

    @pytest.mark.asyncio
    async def test_run_replays_episodes(self):
        episodes = [
            {"ep": {"id": "ep1", "summary": "A community gathering.", "salience_composite": 0.8}},
            {"ep": {"id": "ep2", "summary": "A system error occurred.", "salience_composite": 0.6}},
        ]
        neo4j = FakeNeo4j(read_returns=[
            episodes,  # _select_episodes (code does r["ep"] for r in records)
            [],  # _get_episode_entities for ep1
            [],  # _get_episode_entities for ep2
        ])
        llm = FakeLLM(response="Pattern: community resilience in the face of errors.")
        replay = EpisodicReplay(neo4j=neo4j, llm=llm, config={"replay_batch_size": 10})
        result = await replay.run("cycle_1")

        assert result.episodes_replayed == 2
        assert result.semantic_nodes_created == 2
        assert result.mean_salience_reduction > 0.0

    @pytest.mark.asyncio
    async def test_run_updates_consolidation_level(self):
        episodes = [{"ep": {"id": "ep1", "summary": "Test episode.", "salience_composite": 0.9}}]
        neo4j = FakeNeo4j(read_returns=[episodes, []])
        replay = EpisodicReplay(neo4j=neo4j, llm=None)
        result = await replay.run("cycle_1")

        assert result.episodes_replayed == 1
        # No LLM means no semantic node, but episode still replayed + salience updated
        assert result.semantic_nodes_created == 0
        # Check that a write was issued for the episode update
        assert len(neo4j._write_calls) >= 1

    @pytest.mark.asyncio
    async def test_run_reduces_salience(self):
        episodes = [{"ep": {"id": "ep1", "summary": "Important event.", "salience_composite": 1.0}}]
        neo4j = FakeNeo4j(read_returns=[episodes, []])
        replay = EpisodicReplay(neo4j=neo4j, llm=None)
        result = await replay.run("cycle_1")

        assert result.salience_reductions == 1
        # With salience_reduction factor of 0.70:
        # old=1.0, new=0.70, reduction=0.30
        assert abs(result.mean_salience_reduction - 0.30) < 0.01

    @pytest.mark.asyncio
    async def test_pattern_extraction_with_llm(self):
        episodes = [{"ep": {"id": "ep1", "summary": "Learning from mistakes.", "salience_composite": 0.5}}]
        neo4j = FakeNeo4j(read_returns=[episodes, []])
        llm = FakeLLM(response="Iterative improvement through failure analysis.")
        replay = EpisodicReplay(neo4j=neo4j, llm=llm)
        result = await replay.run("cycle_1")

        assert result.semantic_nodes_created == 1
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_pattern_extraction_without_llm(self):
        episodes = [{"ep": {"id": "ep1", "summary": "An event.", "salience_composite": 0.5}}]
        neo4j = FakeNeo4j(read_returns=[episodes, []])
        replay = EpisodicReplay(neo4j=neo4j, llm=None)
        result = await replay.run("cycle_1")

        # Without LLM, no semantic nodes are created
        assert result.semantic_nodes_created == 0
        assert result.episodes_replayed == 1


# ─── SynapticDownscaler ─────────────────────────────────────────


class TestSynapticDownscaler:
    @pytest.mark.asyncio
    async def test_run_without_neo4j_returns_empty(self):
        ds = SynapticDownscaler(neo4j=None)
        result = await ds.run()
        assert result.traces_decayed == 0
        assert result.traces_pruned == 0

    @pytest.mark.asyncio
    async def test_run_decays_traces(self):
        records = [
            {"id": "t1", "salience": 0.80},
            {"id": "t2", "salience": 0.60},
        ]
        neo4j = FakeNeo4j(read_returns=[records])
        ds = SynapticDownscaler(neo4j=neo4j, config={"salience_decay_factor": 0.85, "salience_pruning_threshold": 0.05})
        result = await ds.run()

        assert result.traces_decayed == 2
        assert result.traces_pruned == 0
        assert result.mean_reduction > 0.0

    @pytest.mark.asyncio
    async def test_run_prunes_below_threshold(self):
        records = [
            {"id": "t1", "salience": 0.04},  # 0.04 * 0.85 = 0.034 < 0.05 threshold
        ]
        neo4j = FakeNeo4j(read_returns=[records])
        ds = SynapticDownscaler(neo4j=neo4j, config={"salience_decay_factor": 0.85, "salience_pruning_threshold": 0.05})
        result = await ds.run()

        assert result.traces_pruned == 1
        assert result.traces_decayed == 0

    @pytest.mark.asyncio
    async def test_decay_factor_applied_correctly(self):
        records = [
            {"id": "t1", "salience": 1.0},
        ]
        neo4j = FakeNeo4j(read_returns=[records])
        ds = SynapticDownscaler(neo4j=neo4j, config={"salience_decay_factor": 0.50, "salience_pruning_threshold": 0.01})
        result = await ds.run()

        assert result.traces_decayed == 1
        # old=1.0, new=0.5, reduction=0.5
        assert abs(result.mean_reduction - 0.50) < 0.01

    @pytest.mark.asyncio
    async def test_mean_reduction_computed(self):
        records = [
            {"id": "t1", "salience": 1.0},
            {"id": "t2", "salience": 0.5},
        ]
        neo4j = FakeNeo4j(read_returns=[records])
        ds = SynapticDownscaler(neo4j=neo4j, config={"salience_decay_factor": 0.80, "salience_pruning_threshold": 0.01})
        result = await ds.run()

        # t1: 1.0 -> 0.8, reduction=0.2
        # t2: 0.5 -> 0.4, reduction=0.1
        # mean = (0.2 + 0.1) / 2 = 0.15
        assert abs(result.mean_reduction - 0.15) < 0.01


# ─── BeliefCompressor ───────────────────────────────────────────


class TestBeliefCompressor:
    @pytest.mark.asyncio
    async def test_run_without_nova_returns_empty(self):
        bc = BeliefCompressor(nova=None)
        result = await bc.run()
        assert result.beliefs_merged == 0
        assert result.beliefs_archived == 0

    @pytest.mark.asyncio
    async def test_run_with_empty_beliefs(self):
        nova = FakeNova(beliefs=[])
        bc = BeliefCompressor(nova=nova)
        result = await bc.run()
        assert result.beliefs_merged == 0
        assert result.beliefs_archived == 0

    @pytest.mark.asyncio
    async def test_archives_low_precision_beliefs(self):
        beliefs = [
            FakeBelief(belief_id="b1", domain="social", precision=0.1, evidence=[]),
            FakeBelief(belief_id="b2", domain="learning", precision=0.05, evidence=[]),
        ]
        nova = FakeNova(beliefs=beliefs)
        bc = BeliefCompressor(nova=nova)
        result = await bc.run()

        assert result.beliefs_archived == 2
        assert beliefs[0].status == "archived"
        assert beliefs[1].status == "archived"

    @pytest.mark.asyncio
    async def test_detects_contradictions(self):
        # Two beliefs in the same domain with contradictory parameters
        beliefs = [
            FakeBelief(belief_id="b1", domain="social", precision=0.8, evidence=["e1"], parameters={"trust": 0.9}),
            FakeBelief(belief_id="b2", domain="social", precision=0.8, evidence=["e2"], parameters={"trust": -0.9}),
        ]
        nova = FakeNova(beliefs=beliefs)
        bc = BeliefCompressor(nova=nova)
        result = await bc.run()

        assert result.beliefs_flagged_contradictory >= 1

    @pytest.mark.asyncio
    async def test_merges_similar_beliefs(self):
        # Two beliefs in the same domain with very similar parameters
        beliefs = [
            FakeBelief(belief_id="b1", domain="social", precision=0.8, evidence=["e1"], parameters={"trust": 0.80}),
            FakeBelief(belief_id="b2", domain="social", precision=0.8, evidence=["e2"], parameters={"trust": 0.81}),
        ]
        nova = FakeNova(beliefs=beliefs)
        bc = BeliefCompressor(nova=nova)
        result = await bc.run()

        assert result.beliefs_merged >= 1


# ─── HypothesisPruner ───────────────────────────────────────────


class TestHypothesisPruner:
    @pytest.mark.asyncio
    async def test_run_without_evo_returns_empty(self):
        pruner = HypothesisPruner(evo=None)
        result = await pruner.run()
        assert result.hypotheses_retired == 0
        assert result.hypotheses_promoted == 0

    @pytest.mark.asyncio
    async def test_retires_low_evidence_hypotheses(self):
        hypotheses = [
            FakeHypothesis(hyp_id="h1", evidence_score=0.1, supporting=1, contradicting=5),
        ]
        evo = FakeEvo(hypotheses=hypotheses)
        pruner = HypothesisPruner(evo=evo)
        result = await pruner.run()

        assert result.hypotheses_retired == 1

    @pytest.mark.asyncio
    async def test_promotes_high_evidence_hypotheses(self):
        hypotheses = [
            FakeHypothesis(hyp_id="h1", evidence_score=0.9, supporting=15, contradicting=0),
        ]
        evo = FakeEvo(hypotheses=hypotheses)
        pruner = HypothesisPruner(evo=evo)
        result = await pruner.run()

        assert result.hypotheses_promoted == 1

    @pytest.mark.asyncio
    async def test_skips_already_archived(self):
        hypotheses = [
            FakeHypothesis(hyp_id="h1", evidence_score=0.1, supporting=0, contradicting=10, status="archived"),
        ]
        evo = FakeEvo(hypotheses=hypotheses)
        pruner = HypothesisPruner(evo=evo)
        result = await pruner.run()

        assert result.hypotheses_retired == 0
        assert result.hypotheses_promoted == 0

    @pytest.mark.asyncio
    async def test_requires_minimum_evaluations_for_retirement(self):
        # Score is low but total evaluations below RETIREMENT_MIN_EVALUATIONS (5)
        hypotheses = [
            FakeHypothesis(hyp_id="h1", evidence_score=0.1, supporting=1, contradicting=2),
        ]
        evo = FakeEvo(hypotheses=hypotheses)
        pruner = HypothesisPruner(evo=evo)
        result = await pruner.run()

        # Only 3 evaluations < 5 minimum, so should NOT retire
        assert result.hypotheses_retired == 0
