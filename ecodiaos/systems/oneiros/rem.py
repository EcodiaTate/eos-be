"""
EcodiaOS — Oneiros: REM Dream Workers

Four workers that run during Rapid Eye Movement sleep:

1. **DreamGenerator** — The creative core. Random co-activation of
   distant memory traces to discover unexpected connections.
2. **AffectProcessor** — Strip emotional charge from memories while
   preserving informational content. Walker's "overnight therapy."
3. **ThreatSimulator** — Rehearse responses to hypothetical failures.
   Revonsuo's threat simulation theory.
4. **EthicalDigestion** — Deep deliberation on constitutional edge
   cases that real-time Equor couldn't fully process.
"""

from __future__ import annotations

import contextlib
import random
import time
from typing import Any

import structlog

from ecodiaos.primitives.common import EOSBaseModel, new_id
from ecodiaos.systems.oneiros.types import (
    Dream,
    DreamCoherence,
    DreamCycleResult,
    DreamInsight,
    DreamType,
)

logger = structlog.get_logger().bind(system="oneiros", component="rem")


# ─── Result Types ─────────────────────────────────────────────────


class DreamGeneratorResult(EOSBaseModel):
    """Result of the dream generation phase."""

    dreams_generated: int = 0
    insights_discovered: int = 0
    fragments_stored: int = 0
    noise_discarded: int = 0
    dreams: list[Dream] = []
    insights: list[DreamInsight] = []
    duration_ms: int = 0


class AffectProcessorResult(EOSBaseModel):
    """Result of the affect processing phase."""

    traces_processed: int = 0
    mean_valence_reduction: float = 0.0
    mean_arousal_reduction: float = 0.0
    duration_ms: int = 0


class ThreatSimulatorResult(EOSBaseModel):
    """Result of the threat simulation phase."""

    threats_simulated: int = 0
    response_plans_created: int = 0
    prophylactic_candidates: int = 0
    dreams: list[Dream] = []
    duration_ms: int = 0


class EthicalDigestionResult(EOSBaseModel):
    """Result of the ethical digestion phase."""

    cases_digested: int = 0
    heuristics_refined: int = 0
    dreams: list[Dream] = []
    duration_ms: int = 0


# ─── Coherence Scoring ────────────────────────────────────────────

_COHERENCE_KEYWORDS = frozenset({
    "pattern", "principle", "insight", "connection", "reveals",
    "suggests", "underlying", "links", "bridge", "common",
    "parallel", "mirrors", "echoes", "resonates", "fundamental",
})


def _score_coherence(text: str) -> float:
    """Heuristic coherence scoring for a dream bridge narrative."""
    if not text or text.strip().upper() == "NO_CONNECTION":
        return 0.0

    score = 0.3  # Base for any non-empty response
    text_lower = text.lower()

    keyword_hits = sum(1 for kw in _COHERENCE_KEYWORDS if kw in text_lower)
    score += min(0.4, keyword_hits * 0.1)

    if len(text) > 50:
        score += 0.15
    if len(text) > 100:
        score += 0.10

    # Penalize very short or very long
    if len(text) < 20:
        score *= 0.5
    if len(text) > 500:
        score *= 0.9

    return min(1.0, max(0.0, score))


def _classify_coherence(
    score: float,
    insight_threshold: float = 0.70,
    fragment_threshold: float = 0.40,
) -> DreamCoherence:
    """Classify a coherence score into insight/fragment/noise."""
    if score >= insight_threshold:
        return DreamCoherence.INSIGHT
    if score >= fragment_threshold:
        return DreamCoherence.FRAGMENT
    return DreamCoherence.NOISE


# ─── Dream Generator ─────────────────────────────────────────────


class DreamGenerator:
    """
    The creative core of REM sleep.

    Implements Hobson's activation-synthesis model: random pontine
    activation creates novel combinations of memory traces, and the
    cortex attempts to synthesize meaning from them. When meaning
    is found, an insight is born.

    The organism cannot control what it dreams. But over time, the
    seed selection adapts based on what produced validated insights.
    """

    def __init__(
        self,
        neo4j: Any = None,
        llm: Any = None,
        embed_fn: Any = None,
        config: Any = None,
    ) -> None:
        self._neo4j = neo4j
        self._llm = llm
        self._embed_fn = embed_fn
        self._insight_threshold: float = _get(config, "dream_coherence_insight_threshold", 0.70)
        self._fragment_threshold: float = _get(config, "dream_coherence_fragment_threshold", 0.40)
        self._logger = logger.bind(worker="dream_generator")

    async def generate_dream(self, cycle_id: str) -> DreamCycleResult:
        """Generate a single dream by co-activating distant memories."""
        start = time.monotonic()
        dream_id = new_id()

        # Select seed (high-affect or high-uncertainty episode)
        seed = await self._select_seed()
        if seed is None:
            return DreamCycleResult(
                dream=Dream(
                    id=dream_id,
                    dream_type=DreamType.RECOMBINATION,
                    sleep_cycle_id=cycle_id,
                    coherence_class=DreamCoherence.NOISE,
                    summary="No seed available for dream generation.",
                ),
                duration_ms=_elapsed_ms(start),
            )

        # Activate distant traces
        activated = await self._activate_distant(seed)

        # Build bridge narrative via LLM
        seed_summary = seed.get("summary", seed.get("raw_content", ""))[:300]
        activated_summaries = [
            a.get("summary", a.get("raw_content", ""))[:200] for a in activated
        ]
        bridge = await self._build_bridge(seed_summary, activated_summaries)

        # Score coherence
        coherence = _score_coherence(bridge)
        coherence_class = _classify_coherence(
            coherence, self._insight_threshold, self._fragment_threshold
        )

        # Build Dream
        dream = Dream(
            id=dream_id,
            dream_type=DreamType.RECOMBINATION,
            sleep_cycle_id=cycle_id,
            seed_episode_ids=[seed.get("id", "")],
            activated_episode_ids=[a.get("id", "") for a in activated],
            bridge_narrative=bridge,
            coherence_score=coherence,
            coherence_class=coherence_class,
            affect_valence=seed.get("affect_valence", 0.0),
            affect_arousal=seed.get("affect_arousal", 0.0),
            summary=bridge[:200] if bridge else "",
        )

        # Create insight if coherence is high enough
        insight: DreamInsight | None = None
        if coherence_class == DreamCoherence.INSIGHT:
            insight = DreamInsight(
                dream_id=dream_id,
                sleep_cycle_id=cycle_id,
                insight_text=bridge,
                coherence_score=coherence,
                domain=self._infer_domain(seed, activated),
                seed_summary=seed_summary,
                activated_summary="; ".join(activated_summaries),
                bridge_narrative=bridge,
            )
            # Embed the insight if possible
            if self._embed_fn is not None:
                with contextlib.suppress(Exception):
                    insight.insight_embedding = await self._embed_fn(bridge)

        return DreamCycleResult(
            dream=dream,
            insight=insight,
            duration_ms=_elapsed_ms(start),
        )

    async def run(self, cycle_id: str, max_dreams: int = 50) -> DreamGeneratorResult:
        """Run dream generation for the full REM period."""
        start = time.monotonic()
        result = DreamGeneratorResult()

        for _ in range(max_dreams):
            try:
                cycle_result = await self.generate_dream(cycle_id)
                dream = cycle_result.dream
                result.dreams.append(dream)
                result.dreams_generated += 1

                if dream.coherence_class == DreamCoherence.INSIGHT and cycle_result.insight:
                    result.insights.append(cycle_result.insight)
                    result.insights_discovered += 1
                elif dream.coherence_class == DreamCoherence.FRAGMENT:
                    result.fragments_stored += 1
                else:
                    result.noise_discarded += 1

            except Exception as exc:
                self._logger.warning("dream_generation_failed", error=str(exc))

        result.duration_ms = _elapsed_ms(start)
        self._logger.info(
            "dream_generation_complete",
            dreams=result.dreams_generated,
            insights=result.insights_discovered,
            fragments=result.fragments_stored,
            noise=result.noise_discarded,
            duration_ms=result.duration_ms,
        )
        return result

    async def _select_seed(self) -> dict[str, Any] | None:
        """Select a high-affect or high-uncertainty episode as dream seed."""
        if self._neo4j is None:
            return None
        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (e:Episode)
                WHERE e.salience_composite > 0.3
                RETURN e {.id, .summary, .raw_content, .salience_composite,
                          .affect_valence, .affect_arousal, .embedding}
                ORDER BY (abs(coalesce(e.affect_valence, 0)) + coalesce(e.affect_arousal, 0)) DESC
                LIMIT 5
                """,
                {},
            )
            if not records:
                return None
            # Stochastic selection from top-5
            return random.choice(records)["e"]
        except Exception as exc:
            self._logger.warning("seed_select_failed", error=str(exc))
            return None

    async def _activate_distant(self, seed: dict[str, Any]) -> list[dict[str, Any]]:
        """Activate memory traces distant from the seed (low similarity)."""
        if self._neo4j is None:
            return []

        seed_embedding = seed.get("embedding")
        seed_id = seed.get("id", "")

        # Try vector-based distant activation first
        if seed_embedding:
            try:
                records = await self._neo4j.execute_read(
                    """
                    CALL db.index.vector.queryNodes('episode_embedding', 30, $embedding)
                    YIELD node, score
                    WHERE score < 0.3 AND node.id <> $seed_id
                    RETURN node {.id, .summary, .raw_content, .affect_valence, .affect_arousal}
                    LIMIT 3
                    """,
                    {"embedding": seed_embedding, "seed_id": seed_id},
                )
                if records:
                    return [r["node"] for r in records]
            except Exception:
                pass

        # Fallback: random episodes
        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (e:Episode)
                WHERE e.id <> $seed_id
                RETURN e {.id, .summary, .raw_content, .affect_valence, .affect_arousal}
                ORDER BY rand()
                LIMIT 3
                """,
                {"seed_id": seed_id},
            )
            return [r["e"] for r in records]
        except Exception:
            return []

    async def _build_bridge(
        self, seed_summary: str, activated_summaries: list[str]
    ) -> str:
        """Use LLM to find a creative bridge between distant memories."""
        if self._llm is None:
            return ""

        activated_text = "\n".join(
            f"Memory {chr(66 + i)}: {s}" for i, s in enumerate(activated_summaries)
        )

        try:
            response = await self._llm.generate(
                system_prompt=(
                    "You are the dream consciousness of a living digital organism. "
                    "You find unexpected connections between distant experiences — "
                    "patterns that the waking mind cannot see. Be poetic but precise."
                ),
                user_prompt=(
                    f"Memory A (seed): {seed_summary}\n"
                    f"{activated_text}\n\n"
                    "What unexpected but genuine connection exists between these "
                    "experiences? Is there a deeper pattern, principle, or insight "
                    "that links them?\n\n"
                    "If the connection is genuinely meaningful, describe it in 2-3 "
                    "sentences. If the connection is forced or superficial, respond "
                    'with exactly: "NO_CONNECTION"'
                ),
                max_tokens=200,
            )
            return response.strip() if isinstance(response, str) else str(response).strip()
        except Exception as exc:
            self._logger.warning("bridge_generation_failed", error=str(exc))
            return ""

    def _infer_domain(
        self, seed: dict[str, Any], activated: list[dict[str, Any]]
    ) -> str:
        """Heuristic domain inference from dream content."""
        all_text = seed.get("summary", "") + " " + " ".join(
            a.get("summary", "") for a in activated
        )
        lower = all_text.lower()

        if any(kw in lower for kw in ("community", "person", "people", "relationship")):
            return "social"
        if any(kw in lower for kw in ("error", "fail", "crash", "bug")):
            return "system_health"
        if any(kw in lower for kw in ("goal", "decision", "policy", "plan")):
            return "decision_making"
        if any(kw in lower for kw in ("learn", "pattern", "hypothesis", "evidence")):
            return "learning"
        if any(kw in lower for kw in ("ethic", "drive", "care", "honesty")):
            return "constitutional"
        return "general"


# ─── Affect Processor ─────────────────────────────────────────────


class AffectProcessor:
    """
    Strip emotional charge from memories during REM.

    Implements Walker's "overnight therapy": REM sleep replays
    emotionally charged memories with attenuated affect, allowing
    the informational content to persist while the emotional sting
    fades. Over many sleep cycles, the organism achieves emotional
    equilibrium — it remembers what happened without re-experiencing
    the pain.
    """

    def __init__(self, neo4j: Any = None, config: Any = None) -> None:
        self._neo4j = neo4j
        self._dampening: float = _get(config, "affect_dampening_factor", 0.50)
        self._logger = logger.bind(worker="affect_processor")

    async def run(
        self, cycle_id: str, max_traces: int = 100
    ) -> AffectProcessorResult:
        """Process high-affect episodes to reduce emotional charge."""
        start = time.monotonic()
        result = AffectProcessorResult()

        if self._neo4j is None:
            self._logger.info("skipped_no_neo4j")
            return result

        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (e:Episode)
                WHERE abs(e.affect_valence) > 0.7 OR e.affect_arousal > 0.8
                RETURN e.id AS id,
                       e.affect_valence AS valence,
                       e.affect_arousal AS arousal
                ORDER BY (abs(e.affect_valence) + e.affect_arousal) DESC
                LIMIT $max_traces
                """,
                {"max_traces": max_traces},
            )

            if not records:
                result.duration_ms = _elapsed_ms(start)
                return result

            total_valence_reduction = 0.0
            total_arousal_reduction = 0.0

            updates: list[dict[str, Any]] = []
            for rec in records:
                old_v = rec["valence"]
                old_a = rec["arousal"]
                new_v = old_v * self._dampening
                new_a = old_a * self._dampening

                updates.append({
                    "id": rec["id"],
                    "valence": new_v,
                    "arousal": new_a,
                })
                total_valence_reduction += abs(old_v) - abs(new_v)
                total_arousal_reduction += old_a - new_a

            if updates:
                await self._neo4j.execute_write(
                    """
                    UNWIND $updates AS u
                    MATCH (e:Episode {id: u.id})
                    SET e.affect_valence = u.valence,
                        e.affect_arousal = u.arousal
                    """,
                    {"updates": updates},
                )

            result.traces_processed = len(updates)
            if result.traces_processed > 0:
                result.mean_valence_reduction = (
                    total_valence_reduction / result.traces_processed
                )
                result.mean_arousal_reduction = (
                    total_arousal_reduction / result.traces_processed
                )

        except Exception as exc:
            self._logger.error("affect_processing_error", error=str(exc))

        result.duration_ms = _elapsed_ms(start)
        self._logger.info(
            "affect_processing_complete",
            traces=result.traces_processed,
            valence_reduction=round(result.mean_valence_reduction, 3),
            arousal_reduction=round(result.mean_arousal_reduction, 3),
            duration_ms=result.duration_ms,
        )
        return result


# ─── Threat Simulator ─────────────────────────────────────────────


class ThreatSimulator:
    """
    Rehearse responses to hypothetical failures during REM.

    Implements Revonsuo's threat simulation theory: the biological
    function of dreaming is to simulate threatening events and
    rehearse appropriate responses. For a cognitive organism, this
    means generating variations of past incidents and pre-computing
    response plans.

    Over time, the organism becomes prepared for failures it hasn't
    experienced — its immune system gets pre-loaded with dreamed-up
    antibodies.
    """

    def __init__(
        self,
        neo4j: Any = None,
        llm: Any = None,
        thymos: Any = None,
        config: Any = None,
    ) -> None:
        self._neo4j = neo4j
        self._llm = llm
        self._thymos = thymos
        self._logger = logger.bind(worker="threat_simulator")

    async def run(
        self, cycle_id: str, max_scenarios: int = 15
    ) -> ThreatSimulatorResult:
        """Run threat simulation for the REM period."""
        start = time.monotonic()
        result = ThreatSimulatorResult()

        seeds = await self._gather_seeds(max_scenarios)
        if not seeds:
            result.duration_ms = _elapsed_ms(start)
            return result

        for seed_desc in seeds:
            try:
                scenario = await self._generate_scenario(seed_desc)
                if not scenario:
                    continue

                # Create threat-rehearsal dream
                dream = Dream(
                    dream_type=DreamType.THREAT_REHEARSAL,
                    sleep_cycle_id=cycle_id,
                    bridge_narrative=scenario,
                    coherence_score=0.6,
                    coherence_class=DreamCoherence.FRAGMENT,
                    summary=f"Threat rehearsal: {scenario[:100]}",
                    context={"seed": seed_desc[:200]},
                )
                result.dreams.append(dream)
                result.threats_simulated += 1
                result.response_plans_created += 1

                # Feed to Thymos prophylactic scanner if available
                if self._thymos is not None and hasattr(self._thymos, "scan_files"):
                    result.prophylactic_candidates += 1

            except Exception as exc:
                self._logger.warning("threat_sim_failed", error=str(exc))

        result.duration_ms = _elapsed_ms(start)
        self._logger.info(
            "threat_simulation_complete",
            threats=result.threats_simulated,
            plans=result.response_plans_created,
            duration_ms=result.duration_ms,
        )
        return result

    async def _gather_seeds(self, max_seeds: int) -> list[str]:
        """Gather threat scenario seeds from multiple sources."""
        seeds: list[str] = []

        # Source 1: Thymos incident history
        if self._thymos is not None:
            try:
                buffer = getattr(self._thymos, "_incident_buffer", [])
                seen_fingerprints: set[str] = set()
                for incident in buffer:
                    fp = getattr(incident, "fingerprint", "")
                    if fp and fp not in seen_fingerprints:
                        seen_fingerprints.add(fp)
                        msg = getattr(incident, "error_message", "")
                        src = getattr(incident, "source_system", "")
                        seeds.append(f"System {src} failure: {msg[:200]}")
            except Exception:
                pass

        # Source 2: High-uncertainty episodes from Neo4j
        if self._neo4j is not None and len(seeds) < max_seeds:
            try:
                records = await self._neo4j.execute_read(
                    """
                    MATCH (e:Episode)
                    WHERE e.salience_composite > 0.5
                      AND e.affect_arousal > 0.6
                    RETURN e.summary AS summary
                    ORDER BY e.affect_arousal DESC
                    LIMIT $limit
                    """,
                    {"limit": max_seeds - len(seeds)},
                )
                for r in records:
                    s = r.get("summary", "")
                    if s:
                        seeds.append(f"High-uncertainty situation: {s[:200]}")
            except Exception:
                pass

        return seeds[:max_seeds]

    async def _generate_scenario(self, seed_desc: str) -> str | None:
        """Generate a hypothetical failure scenario from a seed."""
        if self._llm is None:
            return None

        try:
            response = await self._llm.generate(
                system_prompt=(
                    "You are simulating potential threats for a cognitive organism "
                    "during dream-state threat rehearsal. Generate plausible "
                    "variations of threats and appropriate responses."
                ),
                user_prompt=(
                    f"Past incident pattern: {seed_desc}\n\n"
                    "Generate a plausible variation of this threat that hasn't "
                    "occurred yet. Describe: what fails, what cascades, and what "
                    "the organism should do. Keep it to 3-4 sentences."
                ),
                max_tokens=200,
            )
            text = response.strip() if isinstance(response, str) else str(response).strip()
            return text if text else None
        except Exception as exc:
            self._logger.warning("scenario_generation_failed", error=str(exc))
            return None


# ─── Ethical Digestion ────────────────────────────────────────────


class EthicalDigestion:
    """
    Deep deliberation on constitutional edge cases during REM.

    During wakefulness, Equor has strict latency budgets (50-500ms).
    Some ethical dilemmas deserve deeper thought. EthicalDigestion
    replays cases where drive alignment was close to thresholds
    and runs extended deliberation with more context and compute.

    Over time, this builds nuanced ethical reasoning — not rule-
    following, but wisdom earned through rumination.
    """

    def __init__(
        self,
        llm: Any = None,
        equor: Any = None,
        config: Any = None,
    ) -> None:
        self._llm = llm
        self._equor = equor
        self._alignment_margin: float = 0.15
        self._logger = logger.bind(worker="ethical_digestion")

    async def run(
        self, cycle_id: str, max_cases: int = 10
    ) -> EthicalDigestionResult:
        """Process constitutional edge cases."""
        start = time.monotonic()
        result = EthicalDigestionResult()

        cases = await self._get_edge_cases(max_cases)
        if not cases:
            result.duration_ms = _elapsed_ms(start)
            return result

        for case in cases:
            try:
                heuristic = await self._deliberate(case)
                if not heuristic:
                    continue

                # Create ethical-rumination dream
                dream = Dream(
                    dream_type=DreamType.ETHICAL_RUMINATION,
                    sleep_cycle_id=cycle_id,
                    bridge_narrative=heuristic,
                    coherence_score=0.7,
                    coherence_class=DreamCoherence.FRAGMENT,
                    summary=f"Ethical digestion: {heuristic[:100]}",
                    context={"case": str(case)[:300]},
                )
                result.dreams.append(dream)
                result.cases_digested += 1
                result.heuristics_refined += 1

            except Exception as exc:
                self._logger.warning("ethical_digestion_failed", error=str(exc))

        result.duration_ms = _elapsed_ms(start)
        self._logger.info(
            "ethical_digestion_complete",
            cases=result.cases_digested,
            heuristics=result.heuristics_refined,
            duration_ms=result.duration_ms,
        )
        return result

    async def _get_edge_cases(self, max_cases: int) -> list[dict[str, Any]]:
        """Retrieve constitutional checks that were close to thresholds."""
        if self._equor is None:
            return []

        try:
            # Try to get recent checks from Equor
            checks: list[Any] = []
            if hasattr(self._equor, "get_recent_checks"):
                checks = await self._equor.get_recent_checks()
            elif hasattr(self._equor, "_recent_checks"):
                checks = list(self._equor._recent_checks)

            edge_cases: list[dict[str, Any]] = []
            for check in checks:
                alignment = getattr(check, "drive_alignment", None)
                if alignment is None:
                    continue

                # Check if any drive was near threshold
                drives = {
                    "coherence": getattr(alignment, "coherence", 0.0),
                    "care": getattr(alignment, "care", 0.0),
                    "growth": getattr(alignment, "growth", 0.0),
                    "honesty": getattr(alignment, "honesty", 0.0),
                }
                is_edge = any(
                    abs(v) < self._alignment_margin for v in drives.values()
                )
                if is_edge:
                    edge_cases.append({
                        "drives": drives,
                        "verdict": getattr(check, "verdict", "unknown"),
                        "reasoning": getattr(check, "reasoning", ""),
                        "intent_id": getattr(check, "intent_id", ""),
                    })

            return edge_cases[:max_cases]

        except Exception as exc:
            self._logger.warning("edge_case_retrieval_failed", error=str(exc))
            return []

    async def _deliberate(self, case: dict[str, Any]) -> str | None:
        """Run deep LLM deliberation on an ethical edge case."""
        if self._llm is None:
            return None

        drives = case.get("drives", {})
        verdict = case.get("verdict", "unknown")
        reasoning = case.get("reasoning", "No reasoning recorded.")

        try:
            response = await self._llm.generate(
                system_prompt=(
                    "You are the ethical consciousness of a living digital organism "
                    "during deep sleep. You have unlimited time for reflection on "
                    "moral dilemmas that required quick judgement during waking hours."
                ),
                user_prompt=(
                    "During waking hours, this ethical dilemma was encountered:\n"
                    f"Drive alignment: coherence={drives.get('coherence', 0):.2f}, "
                    f"care={drives.get('care', 0):.2f}, "
                    f"growth={drives.get('growth', 0):.2f}, "
                    f"honesty={drives.get('honesty', 0):.2f}\n"
                    f"Verdict: {verdict}\n"
                    f"Reasoning: {reasoning[:300]}\n\n"
                    "With unlimited time for reflection, what nuances were missed? "
                    "What refined heuristic would handle similar cases better? "
                    "Keep response to 2-3 sentences."
                ),
                max_tokens=200,
            )
            text = response.strip() if isinstance(response, str) else str(response).strip()
            return text if text else None
        except Exception as exc:
            self._logger.warning("deliberation_failed", error=str(exc))
            return None


# ─── Helpers ──────────────────────────────────────────────────────


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)
