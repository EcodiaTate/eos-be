"""Tests for REM dream workers: DreamGenerator, AffectProcessor, ThreatSimulator, EthicalDigestion."""
import pytest
from ecodiaos.systems.oneiros.rem import (
    DreamGenerator, AffectProcessor, ThreatSimulator, EthicalDigestion,
    _score_coherence, _classify_coherence,
)
from ecodiaos.systems.oneiros.types import Dream, DreamType, DreamCoherence, DreamInsight

# Mock Neo4j that returns canned data
class MockNeo4j:
    def __init__(self, read_results=None, write_results=None):
        self._read = read_results or []
        self._writes = []
    async def execute_read(self, query, params=None):
        return self._read
    async def execute_write(self, query, params=None):
        self._writes.append((query, params))

# Mock LLM
class MockLLM:
    def __init__(self, response="A test bridge narrative with deep pattern and insight."):
        self._response = response
    async def generate(self, system_prompt, user_prompt, max_tokens=200):
        return self._response

class TestCoherenceScoring:
    def test_no_connection(self):
        assert _score_coherence("NO_CONNECTION") == 0.0
    def test_empty_string(self):
        assert _score_coherence("") == 0.0
    def test_high_coherence_keywords(self):
        score = _score_coherence("This reveals a fundamental pattern that links both experiences through a common underlying principle")
        assert score >= 0.7
    def test_medium_coherence(self):
        score = _score_coherence("There might be some connection between these two things in a general sense")
        assert 0.3 <= score <= 0.7
    def test_short_text_penalized(self):
        score = _score_coherence("yes")
        assert score < 0.3
    def test_classification_insight(self):
        assert _classify_coherence(0.8) == DreamCoherence.INSIGHT
    def test_classification_fragment(self):
        assert _classify_coherence(0.5) == DreamCoherence.FRAGMENT
    def test_classification_noise(self):
        assert _classify_coherence(0.2) == DreamCoherence.NOISE
    def test_classification_at_insight_boundary(self):
        assert _classify_coherence(0.70) == DreamCoherence.INSIGHT
    def test_classification_at_fragment_boundary(self):
        assert _classify_coherence(0.40) == DreamCoherence.FRAGMENT
    def test_classification_below_fragment(self):
        assert _classify_coherence(0.39) == DreamCoherence.NOISE

class TestDreamGenerator:
    @pytest.mark.asyncio
    async def test_generate_without_neo4j(self):
        gen = DreamGenerator(neo4j=None, llm=None)
        result = await gen.generate_dream("c1")
        assert result.dream is not None
        assert result.dream.coherence_class == DreamCoherence.NOISE
    @pytest.mark.asyncio
    async def test_generate_with_mocked_neo4j(self):
        episodes = [{"e": {"id": "ep1", "summary": "Test episode", "raw_content": "test", "salience_composite": 0.8, "affect_valence": 0.9, "affect_arousal": 0.7, "embedding": None}}]
        random_eps = [{"e": {"id": "ep2", "summary": "Random episode", "raw_content": "random", "affect_valence": 0.1, "affect_arousal": 0.2}}]
        neo4j = MockNeo4j(read_results=episodes)  # Simplified
        llm = MockLLM("This reveals a fundamental pattern linking both experiences.")
        gen = DreamGenerator(neo4j=neo4j, llm=llm)
        result = await gen.generate_dream("c1")
        assert result.dream.dream_type == DreamType.RECOMBINATION
    @pytest.mark.asyncio
    async def test_run_generates_multiple(self):
        gen = DreamGenerator(neo4j=None, llm=None)
        result = await gen.run("c1", max_dreams=5)
        assert result.dreams_generated == 5
        assert len(result.dreams) == 5
    @pytest.mark.asyncio
    async def test_run_accumulates_counts(self):
        gen = DreamGenerator(neo4j=None, llm=None)
        result = await gen.run("c1", max_dreams=3)
        assert result.dreams_generated == result.insights_discovered + result.fragments_stored + result.noise_discarded
    def test_domain_inference_social(self):
        gen = DreamGenerator()
        result = gen._infer_domain({"summary": "community meeting with people"}, [])
        assert result == "social"
    def test_domain_inference_system(self):
        gen = DreamGenerator()
        result = gen._infer_domain({"summary": "system crash error failure"}, [])
        assert result == "system_health"
    def test_domain_inference_general(self):
        gen = DreamGenerator()
        result = gen._infer_domain({"summary": "something happened"}, [])
        assert result == "general"

class TestAffectProcessor:
    @pytest.mark.asyncio
    async def test_run_without_neo4j(self):
        proc = AffectProcessor(neo4j=None)
        result = await proc.run("c1")
        assert result.traces_processed == 0
    @pytest.mark.asyncio
    async def test_run_with_traces(self):
        records = [
            {"id": "e1", "valence": 0.9, "arousal": 0.8},
            {"id": "e2", "valence": -0.8, "arousal": 0.9},
        ]
        neo4j = MockNeo4j(read_results=records)
        proc = AffectProcessor(neo4j=neo4j, config={"affect_dampening_factor": 0.5})
        result = await proc.run("c1")
        assert result.traces_processed == 2
        assert result.mean_valence_reduction > 0
    @pytest.mark.asyncio
    async def test_dampening_factor(self):
        records = [{"id": "e1", "valence": 1.0, "arousal": 1.0}]
        neo4j = MockNeo4j(read_results=records)
        proc = AffectProcessor(neo4j=neo4j, config={"affect_dampening_factor": 0.5})
        result = await proc.run("c1", max_traces=1)
        assert result.traces_processed == 1
        # Check write was called
        assert len(neo4j._writes) > 0

class TestThreatSimulator:
    @pytest.mark.asyncio
    async def test_run_without_dependencies(self):
        sim = ThreatSimulator()
        result = await sim.run("c1")
        assert result.threats_simulated == 0
    @pytest.mark.asyncio
    async def test_run_with_neo4j(self):
        records = [{"summary": "A high-stress dangerous situation"}]
        neo4j = MockNeo4j(read_results=records)
        llm = MockLLM("A cascade failure scenario where the memory system overloads.")
        sim = ThreatSimulator(neo4j=neo4j, llm=llm)
        result = await sim.run("c1", max_scenarios=2)
        assert result.threats_simulated > 0

class TestEthicalDigestion:
    @pytest.mark.asyncio
    async def test_run_without_equor(self):
        dig = EthicalDigestion()
        result = await dig.run("c1")
        assert result.cases_digested == 0
    @pytest.mark.asyncio
    async def test_run_with_edge_cases(self):
        class MockAlignment:
            coherence = 0.05
            care = 0.8
            growth = 0.7
            honesty = 0.6
        class MockCheck:
            drive_alignment = MockAlignment()
            verdict = "approved"
            reasoning = "Borderline coherence"
            intent_id = "i1"
        class MockEquor:
            _recent_checks = [MockCheck()]
        llm = MockLLM("A more nuanced approach would consider the contextual factors.")
        dig = EthicalDigestion(llm=llm, equor=MockEquor())
        result = await dig.run("c1")
        assert result.cases_digested == 1
        assert result.heuristics_refined == 1
        assert len(result.dreams) == 1
