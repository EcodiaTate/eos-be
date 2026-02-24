"""Tests for Lucid dreaming workers: DirectedExploration and MetaCognition."""
import pytest
from ecodiaos.systems.oneiros.lucid import DirectedExploration, MetaCognition
from ecodiaos.systems.oneiros.types import DreamInsight, DreamCoherence
from ecodiaos.systems.oneiros.journal import DreamJournal

class MockLLM:
    def __init__(self, response="[DOMAIN] Applied to economics | novelty=0.8 | utility=0.7\n[INVERSE] The opposite reveals | novelty=0.6 | utility=0.5\n[EXTREME] Taken to extreme | novelty=0.9 | utility=0.4"):
        self._response = response
    async def generate(self, system_prompt, user_prompt, max_tokens=200):
        return self._response

class TestDirectedExploration:
    @pytest.mark.asyncio
    async def test_run_without_trigger(self):
        exp = DirectedExploration(llm=None, journal=None)
        result = await exp.run("c1")
        assert result.explorations_completed == 0

    @pytest.mark.asyncio
    async def test_run_with_string_trigger(self):
        llm = MockLLM()
        exp = DirectedExploration(llm=llm, journal=None)
        result = await exp.run("c1", trigger="A test insight about patterns")
        assert result.explorations_completed > 0
        assert result.variations_generated > 0

    @pytest.mark.asyncio
    async def test_run_with_insight_trigger(self):
        llm = MockLLM()
        insight = DreamInsight(dream_id="d1", sleep_cycle_id="c1", insight_text="Test insight", coherence_score=0.9)
        exp = DirectedExploration(llm=llm, journal=None)
        result = await exp.run("c1", trigger=insight)
        assert result.explorations_completed > 0

    def test_variation_parsing(self):
        exp = DirectedExploration()
        variations = exp._parse_variations(
            "[DOMAIN] Economic application of pattern | novelty=0.8 | utility=0.7\n"
            "[INVERSE] The opposite reveals | novelty=0.6 | utility=0.5\n"
            "[EXTREME] Taken to extreme | novelty=0.9 | utility=0.4"
        )
        assert len(variations) == 3
        assert variations[0]["label"] == "DOMAIN"
        assert variations[0]["novelty"] == pytest.approx(0.8)

    def test_variation_parsing_empty(self):
        exp = DirectedExploration()
        variations = exp._parse_variations("")
        assert len(variations) == 0

    @pytest.mark.asyncio
    async def test_high_value_filter(self):
        llm = MockLLM("[DOMAIN] Great insight about connections | novelty=0.9 | utility=0.9")
        exp = DirectedExploration(llm=llm)
        result = await exp.run("c1", trigger="test")
        assert result.high_value_insights > 0

class TestMetaCognition:
    @pytest.mark.asyncio
    async def test_run_with_no_data(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        meta = MetaCognition(journal=journal)
        result = await meta.run("c1")
        assert result.observations_made == 0

    @pytest.mark.asyncio
    async def test_run_with_llm_and_themes(self):
        journal = DreamJournal(neo4j=None)
        await journal.initialize()
        # Add some dreams with themes
        from ecodiaos.systems.oneiros.types import Dream, DreamType, DreamCoherence
        for i in range(5):
            d = Dream(dream_type=DreamType.RECOMBINATION, sleep_cycle_id="c1",
                     themes=["learning", "growth"], coherence_score=0.6,
                     coherence_class=DreamCoherence.FRAGMENT)
            await journal.record_dream(d)
        llm = MockLLM("I notice recurring themes of learning and growth in my dreams.\nThis suggests ongoing cognitive development.\nThe pattern indicates curiosity-driven exploration.")
        meta = MetaCognition(journal=journal, llm=llm)
        result = await meta.run("c1")
        assert result.observations_made > 0
