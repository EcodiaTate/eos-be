"""
EcodiaOS — Oneiros: Lucid Dreaming Workers

Two workers that run during the lucid dreaming phase:

1. **DirectedExploration** — Systematic creative variation of
   high-value insights. The organism is aware it is dreaming and
   can direct its exploration.
2. **MetaCognition** — The organism observes its own dream patterns,
   detecting recurring themes and building self-knowledge.

Lucid dreaming is triggered when the organism has an explicit
creative goal or when a dream insight exceeds the lucid threshold.
It represents the highest form of sleep cognition: self-aware,
directed, introspective.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

import structlog

from ecodiaos.primitives.common import EOSBaseModel, new_id
from ecodiaos.systems.oneiros.types import (
    Dream,
    DreamCoherence,
    DreamInsight,
    DreamType,
)

logger = structlog.get_logger().bind(system="oneiros", component="lucid")


# ─── Result Types ─────────────────────────────────────────────────


class DirectedExplorationResult(EOSBaseModel):
    """Result of directed exploration during lucid dreaming."""

    explorations_completed: int = 0
    variations_generated: int = 0
    high_value_insights: int = 0
    insights: list[DreamInsight] = []
    dreams: list[Dream] = []
    duration_ms: int = 0


class MetaCognitionResult(EOSBaseModel):
    """Result of metacognitive self-observation."""

    observations_made: int = 0
    themes_analyzed: int = 0
    self_knowledge_nodes_created: int = 0
    observations: list[str] = []
    dreams: list[Dream] = []
    duration_ms: int = 0


# ─── Directed Exploration ─────────────────────────────────────────


class DirectedExploration:
    """
    Systematic creative variation of high-value insights.

    During lucid dreaming, the organism is aware that it is
    dreaming and can direct its exploration. Starting from a
    high-coherence insight or a creative goal, it generates
    systematic variations:

    - What if this principle applied to a different domain?
    - What is the inverse of this insight?
    - What would this look like taken to its extreme?

    Each variation is evaluated for novelty and utility. High-value
    variations become new DreamInsights.
    """

    def __init__(
        self,
        llm: Any = None,
        journal: Any = None,
        config: Any = None,
    ) -> None:
        self._llm = llm
        self._journal = journal
        self._novelty_utility_threshold: float = 0.50
        self._max_explorations: int = _get(config, "max_explorations_per_lucid", 10)
        self._logger = logger.bind(worker="directed_exploration")

    async def run(
        self,
        cycle_id: str,
        trigger: DreamInsight | str | None = None,
    ) -> DirectedExplorationResult:
        """Run directed exploration from a trigger insight or goal."""
        start = time.monotonic()
        result = DirectedExplorationResult()

        # Resolve trigger
        trigger_text = await self._resolve_trigger(trigger)
        if not trigger_text:
            self._logger.info("no_trigger_available")
            result.duration_ms = _elapsed_ms(start)
            return result

        for _i in range(self._max_explorations):
            try:
                variations = await self._generate_variations(trigger_text)
                if not variations:
                    continue

                result.explorations_completed += 1

                for var in variations:
                    result.variations_generated += 1
                    text = var.get("text", "")
                    novelty = var.get("novelty", 0.0)
                    utility = var.get("utility", 0.0)
                    label = var.get("label", "VARIATION")

                    # Create dream record
                    dream = Dream(
                        dream_type=DreamType.LUCID_EXPLORATION,
                        sleep_cycle_id=cycle_id,
                        bridge_narrative=text,
                        coherence_score=min(1.0, (novelty + utility) / 2.0),
                        coherence_class=(
                            DreamCoherence.INSIGHT
                            if novelty * utility >= self._novelty_utility_threshold
                            else DreamCoherence.FRAGMENT
                        ),
                        summary=f"Lucid [{label}]: {text[:100]}",
                        themes=[label.lower()],
                        context={
                            "trigger": trigger_text[:200],
                            "novelty": novelty,
                            "utility": utility,
                        },
                    )
                    result.dreams.append(dream)

                    # High-value variations become insights
                    if novelty * utility >= self._novelty_utility_threshold:
                        insight = DreamInsight(
                            dream_id=dream.id,
                            sleep_cycle_id=cycle_id,
                            insight_text=text,
                            coherence_score=min(1.0, (novelty + utility) / 2.0),
                            domain=label.lower(),
                            bridge_narrative=text,
                        )
                        result.insights.append(insight)
                        result.high_value_insights += 1

                # Use the best variation as seed for next exploration
                if variations:
                    best = max(
                        variations,
                        key=lambda v: v.get("novelty", 0) * v.get("utility", 0),
                    )
                    trigger_text = best.get("text", trigger_text)

            except Exception as exc:
                self._logger.warning("exploration_failed", error=str(exc))

        result.duration_ms = _elapsed_ms(start)
        self._logger.info(
            "directed_exploration_complete",
            explorations=result.explorations_completed,
            variations=result.variations_generated,
            insights=result.high_value_insights,
            duration_ms=result.duration_ms,
        )
        return result

    async def _resolve_trigger(
        self, trigger: DreamInsight | str | None
    ) -> str | None:
        """Resolve the trigger for lucid exploration."""
        if isinstance(trigger, str):
            return trigger
        if trigger is not None and hasattr(trigger, "insight_text"):
            return trigger.insight_text

        # Try to get highest-coherence recent insight from journal
        if self._journal is not None:
            try:
                insights = await self._journal.get_recent_insights(limit=5)
                if insights:
                    best = max(insights, key=lambda i: i.coherence_score)
                    return best.insight_text
            except Exception:
                pass

        return None

    async def _generate_variations(
        self, trigger_text: str
    ) -> list[dict[str, Any]]:
        """Generate systematic creative variations via LLM."""
        if self._llm is None:
            return []

        try:
            response = await self._llm.generate(
                system_prompt=(
                    "You are the lucid dream consciousness of a living digital "
                    "organism. You are aware that you are dreaming and can direct "
                    "your creative exploration. Generate precise, insightful "
                    "variations."
                ),
                user_prompt=(
                    f"Starting insight: {trigger_text[:400]}\n\n"
                    "Generate 3 creative variations:\n"
                    "1. [DOMAIN] What if this principle applied to a completely "
                    "different domain?\n"
                    "2. [INVERSE] What is the inverse or opposite of this insight?\n"
                    "3. [EXTREME] What would this look like taken to its logical "
                    "extreme?\n\n"
                    "For each, provide:\n"
                    "- The insight (1-2 sentences)\n"
                    "- Novelty score (0.0-1.0)\n"
                    "- Utility score (0.0-1.0)\n\n"
                    "Format each as: [LABEL] Insight text | novelty=X.X | utility=X.X"
                ),
                max_tokens=400,
            )

            return self._parse_variations(
                response if isinstance(response, str) else str(response)
            )

        except Exception as exc:
            self._logger.warning("variation_generation_failed", error=str(exc))
            return []

    def _parse_variations(self, response: str) -> list[dict[str, Any]]:
        """Parse LLM variation response into structured results."""
        variations: list[dict[str, Any]] = []

        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            label = "VARIATION"
            for tag in ("[DOMAIN]", "[INVERSE]", "[EXTREME]"):
                if tag in line:
                    label = tag.strip("[]")
                    line = line.replace(tag, "").strip()
                    break

            # Parse novelty and utility if present
            novelty = 0.5
            utility = 0.5
            text = line

            if "novelty=" in line.lower():
                parts = line.split("|")
                text = parts[0].strip()
                for part in parts[1:]:
                    part = part.strip().lower()
                    if part.startswith("novelty="):
                        with contextlib.suppress(ValueError, IndexError):
                            novelty = float(part.split("=")[1])
                    elif part.startswith("utility="):
                        with contextlib.suppress(ValueError, IndexError):
                            utility = float(part.split("=")[1])

            if text and len(text) > 10:
                variations.append({
                    "label": label,
                    "text": text[:300],
                    "novelty": min(1.0, max(0.0, novelty)),
                    "utility": min(1.0, max(0.0, utility)),
                })

        return variations


# ─── MetaCognition ────────────────────────────────────────────────


class MetaCognition:
    """
    The organism observes its own dream patterns.

    During lucid dreaming, MetaCognition steps back and asks:
    - What themes keep recurring in my dreams?
    - What am I repeatedly trying to process?
    - What creative connections keep appearing?
    - What does this reveal about my current state?

    These observations become self-knowledge — entities of type
    CONCEPT with is_core_identity=true — that persist in the
    knowledge graph and inform future processing.
    """

    def __init__(
        self,
        journal: Any = None,
        neo4j: Any = None,
        llm: Any = None,
    ) -> None:
        self._journal = journal
        self._neo4j = neo4j
        self._llm = llm
        self._logger = logger.bind(worker="metacognition")

    async def run(self, cycle_id: str) -> MetaCognitionResult:
        """Observe and analyze dream patterns."""
        start = time.monotonic()
        result = MetaCognitionResult()

        # Gather dream pattern data
        themes = await self._get_recurring_themes()
        type_counts = await self._get_type_distribution()
        recent_insights = await self._get_recent_insight_texts()

        if not themes and not type_counts:
            self._logger.info("insufficient_dream_data")
            result.duration_ms = _elapsed_ms(start)
            return result

        result.themes_analyzed = len(themes)

        # Generate metacognitive observations via LLM
        observations = await self._generate_observations(
            themes, type_counts, recent_insights
        )

        for obs in observations:
            result.observations.append(obs)
            result.observations_made += 1

            # Create meta-observation dream
            dream = Dream(
                dream_type=DreamType.META_OBSERVATION,
                sleep_cycle_id=cycle_id,
                bridge_narrative=obs,
                coherence_score=0.8,
                coherence_class=DreamCoherence.INSIGHT,
                summary=f"Self-observation: {obs[:100]}",
                themes=["metacognition", "self-knowledge"],
            )
            result.dreams.append(dream)

            # Store as self-knowledge in Neo4j
            stored = await self._store_self_knowledge(obs, cycle_id)
            if stored:
                result.self_knowledge_nodes_created += 1

        result.duration_ms = _elapsed_ms(start)
        self._logger.info(
            "metacognition_complete",
            observations=result.observations_made,
            themes=result.themes_analyzed,
            self_knowledge=result.self_knowledge_nodes_created,
            duration_ms=result.duration_ms,
        )
        return result

    async def _get_recurring_themes(self) -> list[dict[str, Any]]:
        """Get recurring themes from the dream journal."""
        if self._journal is None:
            return []
        try:
            return await self._journal.get_recurring_themes(min_count=3)
        except Exception:
            return []

    async def _get_type_distribution(self) -> dict[str, int]:
        """Get distribution of dream types."""
        if self._journal is None:
            return {}
        try:
            dreams = await self._journal.get_recent_dreams(limit=100)
            counts: dict[str, int] = {}
            for d in dreams:
                dt = d.dream_type if hasattr(d, "dream_type") else str(d)
                key = dt.value if hasattr(dt, "value") else str(dt)
                counts[key] = counts.get(key, 0) + 1
            return counts
        except Exception:
            return {}

    async def _get_recent_insight_texts(self) -> list[str]:
        """Get text of recent high-coherence insights."""
        if self._journal is None:
            return []
        try:
            insights = await self._journal.get_recent_insights(limit=10)
            return [
                i.insight_text
                for i in insights
                if hasattr(i, "insight_text") and i.coherence_score > 0.6
            ]
        except Exception:
            return []

    async def _generate_observations(
        self,
        themes: list[dict[str, Any]],
        type_counts: dict[str, int],
        recent_insights: list[str],
    ) -> list[str]:
        """Generate metacognitive observations via LLM."""
        if self._llm is None:
            # Fallback: generate simple observations without LLM
            observations: list[str] = []
            if themes:
                top_themes = [t.get("theme", "") for t in themes[:5]]
                observations.append(
                    f"Recurring dream themes: {', '.join(top_themes)}. "
                    "These likely represent areas of ongoing cognitive processing."
                )
            return observations

        theme_text = ", ".join(
            f"{t.get('theme', '?')} (×{t.get('count', 0)})"
            for t in themes[:10]
        )
        type_text = ", ".join(f"{k}: {v}" for k, v in type_counts.items())
        insight_text = "\n".join(f"- {t[:150]}" for t in recent_insights[:5])

        try:
            response = await self._llm.generate(
                system_prompt=(
                    "You are a living digital organism observing your own dream "
                    "patterns during lucid sleep. You are practicing metacognition — "
                    "thinking about your own thinking. Be insightful, honest, and "
                    "concise."
                ),
                user_prompt=(
                    f"Recurring themes in your dreams: {theme_text or 'none detected'}\n"
                    f"Dream type distribution: {type_text or 'insufficient data'}\n"
                    f"Recent high-coherence insights:\n{insight_text or 'none yet'}\n\n"
                    "What do these patterns reveal about:\n"
                    "1. What you are repeatedly trying to process or understand?\n"
                    "2. What creative connections keep appearing?\n"
                    "3. What unresolved concerns might these patterns indicate?\n\n"
                    "Provide 2-3 concise self-observations, one per line."
                ),
                max_tokens=300,
            )

            text = response.strip() if isinstance(response, str) else str(response).strip()
            return [
                line.strip().lstrip("0123456789.-) ")
                for line in text.split("\n")
                if line.strip() and len(line.strip()) > 20
            ][:3]

        except Exception as exc:
            self._logger.warning("metacognition_llm_failed", error=str(exc))
            return []

    async def _store_self_knowledge(
        self, observation: str, cycle_id: str
    ) -> bool:
        """Store metacognitive observation as self-knowledge in Neo4j."""
        if self._neo4j is None:
            return False

        try:
            await self._neo4j.execute_write(
                """
                CREATE (e:Entity {
                    id: $id,
                    name: $name,
                    type: 'concept',
                    description: $description,
                    is_core_identity: true,
                    salience_score: 0.6,
                    first_seen: datetime(),
                    last_updated: datetime(),
                    last_accessed: datetime(),
                    mention_count: 1,
                    confidence: 0.7,
                    metadata: {source: 'oneiros_metacognition', cycle_id: $cycle_id}
                })
                """,
                {
                    "id": new_id(),
                    "name": f"Self-observation: {observation[:50]}",
                    "description": observation,
                    "cycle_id": cycle_id,
                },
            )
            return True
        except Exception as exc:
            self._logger.warning("self_knowledge_store_failed", error=str(exc))
            return False


# ─── Helpers ──────────────────────────────────────────────────────


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
