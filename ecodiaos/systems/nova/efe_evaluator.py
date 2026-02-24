"""
EcodiaOS — Nova EFE Evaluator

Computes Expected Free Energy for each candidate policy.

The full EFE decomposition (from the spec and Friston et al.):

    G(π) = -[pragmatic_value + epistemic_value + constitutional + feasibility]
           + risk_penalty

All positive components are negated (lower total = preferred).

The five components:
1. Pragmatic value  (0.35) — probability of achieving the goal state
2. Epistemic value  (0.20) — expected information gain (Growth drive)
3. Constitutional   (0.20) — alignment with drive weights (constitutional fit)
4. Feasibility      (0.15) — can we actually do this given our capabilities?
5. Risk             (0.10) — expected harm if we're wrong (positive = increases EFE)

Weights start at spec defaults; Evo adjusts them over time as evidence accumulates.

Pragmatic and epistemic components require LLM evaluation for slow-path policies.
Constitutional alignment is computed analytically from the DriveAlignmentVector.
Feasibility and risk are computed heuristically (fast, no LLM needed).
"""

from __future__ import annotations

import json
import time

import structlog

from ecodiaos.clients.llm import LLMProvider, Message
from ecodiaos.clients.optimized_llm import OptimizedLLMProvider
from ecodiaos.clients.output_validator import OutputValidator
from ecodiaos.primitives.affect import AffectState
from ecodiaos.prompts.nova.policy import (
    build_epistemic_value_prompt,
    build_pragmatic_value_prompt,
    summarise_beliefs,
)
from ecodiaos.systems.nova.efe_heuristics import EFEHeuristics
from ecodiaos.systems.nova.policy_generator import DO_NOTHING_EFE, make_do_nothing_policy
from ecodiaos.systems.nova.types import (
    BeliefState,
    EFEScore,
    EFEWeights,
    EpistemicEstimate,
    Goal,
    Policy,
    PragmaticEstimate,
    RiskEstimate,
)

logger = structlog.get_logger()

# LLM evaluation timeout per policy (must fit within slow-path 5000ms budget)
_EVAL_TIMEOUT_MS = 200

# High-risk action types that increase EFE
_HIGH_RISK_ACTIONS = {"federate", "external_api", "irreversible"}
# Conservative action types that reduce risk
_LOW_RISK_ACTIONS = {"observe", "wait", "express"}


class EFEEvaluator:
    """
    Evaluates Expected Free Energy for candidate policies.

    For the do-nothing policy, EFE is fixed at DO_NOTHING_EFE (-0.1).
    For all other policies, EFE is computed from the five components above.

    LLM calls are made for pragmatic and epistemic estimation when in slow path.
    All other components are computed analytically.
    """

    def __init__(
        self,
        llm: LLMProvider,
        weights: EFEWeights | None = None,
        use_llm_estimation: bool = True,
    ) -> None:
        self._llm = llm
        self._weights = weights or EFEWeights()
        self._use_llm = use_llm_estimation
        self._logger = logger.bind(system="nova.efe_evaluator")
        # Optimization: detect if we have the optimized provider for budget/cache
        self._optimized = isinstance(llm, OptimizedLLMProvider)
        self._heuristics = EFEHeuristics()
        self._validator = OutputValidator()

    @property
    def weights(self) -> EFEWeights:
        return self._weights

    def update_weights(self, new_weights: EFEWeights) -> None:
        """Called by Evo after learning that certain EFE components predict outcomes better."""
        self._weights = new_weights

    async def evaluate(
        self,
        policy: Policy,
        goal: Goal,
        beliefs: BeliefState,
        affect: AffectState,
        drive_weights: dict[str, float],
    ) -> EFEScore:
        """
        Compute full EFE score for a policy.

        G(π) = -[p*pragmatic + e*epistemic + c*constitutional + f*feasibility] + r*risk

        Lower G = more preferred policy.
        """
        start = time.monotonic()

        # ── Do-nothing gets the fixed baseline ──
        if policy.id == "do_nothing":
            score = EFEScore(
                total=DO_NOTHING_EFE,
                pragmatic=PragmaticEstimate(score=0.1, success_probability=0.05, confidence=0.9),
                epistemic=EpistemicEstimate(score=0.3, uncertainties_addressed=0, expected_info_gain=0.2, novelty=0.1),
                constitutional_alignment=0.3,
                feasibility=1.0,  # Always feasible
                risk=RiskEstimate(expected_harm=0.0, reversibility=1.0),
                confidence=0.95,
                reasoning="Do-nothing policy: fixed EFE baseline. Any active policy must beat -0.10.",
            )
            score = score.model_copy(update={"total": DO_NOTHING_EFE})
            return score

        # ── Pragmatic value ──
        # Budget-aware: check if we should use LLM or fall back to heuristics
        use_llm_pragmatic = self._use_llm
        use_llm_epistemic = self._use_llm
        if self._optimized:
            assert isinstance(self._llm, OptimizedLLMProvider)
            if not self._llm.should_use_llm("nova.efe.pragmatic", estimated_tokens=200):
                use_llm_pragmatic = False
                self._heuristics.log_heuristic_fallback(
                    "nova.efe.pragmatic", "budget_exhausted", policy.type if hasattr(policy, 'type') else "unknown"
                )
            if not self._llm.should_use_llm("nova.efe.epistemic", estimated_tokens=150):
                use_llm_epistemic = False
                self._heuristics.log_heuristic_fallback(
                    "nova.efe.epistemic", "budget_exhausted", policy.type if hasattr(policy, 'type') else "unknown"
                )

        if use_llm_pragmatic:
            pragmatic = await self._estimate_pragmatic_llm(policy, goal, beliefs)
        else:
            pragmatic = _estimate_pragmatic_heuristic(policy, goal)

        # ── Epistemic value ──
        if use_llm_epistemic:
            epistemic = await self._estimate_epistemic_llm(policy, beliefs)
        else:
            epistemic = _estimate_epistemic_heuristic(policy, beliefs)

        # ── Constitutional alignment (analytical) ──
        constitutional = _compute_constitutional_alignment(
            policy=policy,
            drive_weights=drive_weights,
            goal=goal,
        )

        # ── Feasibility (heuristic) ──
        feasibility = _estimate_feasibility(policy, beliefs)

        # ── Risk (heuristic) ──
        risk = _estimate_risk(policy)

        # ── Weighted EFE total ──
        w = self._weights
        total = (
            - w.pragmatic * pragmatic.score
            - w.epistemic * epistemic.score
            - w.constitutional * constitutional
            - w.feasibility * feasibility
            + w.risk * risk.expected_harm
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        self._logger.debug(
            "efe_evaluated",
            policy=policy.name,
            total=round(total, 4),
            pragmatic=round(pragmatic.score, 3),
            epistemic=round(epistemic.score, 3),
            constitutional=round(constitutional, 3),
            feasibility=round(feasibility, 3),
            risk=round(risk.expected_harm, 3),
            elapsed_ms=elapsed_ms,
        )

        reasoning = (
            f"Pragmatic={pragmatic.score:.2f} (prob={pragmatic.success_probability:.2f}), "
            f"Epistemic={epistemic.score:.2f}, "
            f"Constitutional={constitutional:.2f}, "
            f"Feasibility={feasibility:.2f}, "
            f"Risk={risk.expected_harm:.2f} → "
            f"G(π)={total:.3f}"
        )

        return EFEScore(
            pragmatic=pragmatic,
            epistemic=epistemic,
            constitutional_alignment=constitutional,
            feasibility=feasibility,
            risk=risk,
            total=total,
            confidence=min(pragmatic.confidence, feasibility),
            reasoning=reasoning,
        )

    async def evaluate_all(
        self,
        policies: list[Policy],
        goal: Goal,
        beliefs: BeliefState,
        affect: AffectState,
        drive_weights: dict[str, float],
    ) -> list[tuple[Policy, EFEScore]]:
        """
        Evaluate EFE for all candidate policies.
        Results are sorted by EFE (lowest = most preferred).
        """
        import asyncio

        # Evaluate all policies in parallel (within the 5000ms budget)
        tasks = [
            self.evaluate(policy, goal, beliefs, affect, drive_weights)
            for policy in policies
        ]
        scores = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[tuple[Policy, EFEScore]] = []
        for policy, score in zip(policies, scores):
            if isinstance(score, Exception):
                self._logger.warning(
                    "efe_evaluation_error",
                    policy=policy.name,
                    error=str(score),
                )
                # Assign a neutral EFE on failure so it doesn't block selection
                score = EFEScore(total=0.0, reasoning="Evaluation failed — using neutral score")
            results.append((policy, score))

        # Sort: minimum EFE first
        results.sort(key=lambda x: x[1].total)
        return results

    # ─── Private LLM Estimators ───────────────────────────────────

    async def _estimate_pragmatic_llm(
        self,
        policy: Policy,
        goal: Goal,
        beliefs: BeliefState,
    ) -> PragmaticEstimate:
        """LLM-based pragmatic value estimation with caching and validation."""
        steps_desc = " → ".join(s.description for s in policy.steps) or "No steps specified"
        prompt = build_pragmatic_value_prompt(
            policy_name=policy.name,
            policy_reasoning=policy.reasoning,
            policy_steps_desc=steps_desc,
            goal_description=goal.description,
            goal_success_criteria=goal.success_criteria,
            beliefs_summary=summarise_beliefs(beliefs, max_entities=3),
        )
        try:
            # Use cache-tagged evaluate if optimized provider is available
            if self._optimized:
                response = await self._llm.evaluate(  # type: ignore[call-arg]
                    prompt, max_tokens=200, temperature=0.2,
                    cache_system="nova.efe.pragmatic", cache_method="evaluate",
                )
            else:
                response = await self._llm.evaluate(prompt, max_tokens=200, temperature=0.2)

            # Use output validator for robust JSON extraction
            data = self._validator.extract_json(response.text)
            if data and isinstance(data, dict):
                data = self._validator.auto_fix_dict(
                    data,
                    required_keys=["success_probability", "confidence", "reasoning"],
                    defaults={"success_probability": 0.5, "confidence": 0.5, "reasoning": ""},
                )
                return PragmaticEstimate(
                    score=float(data.get("success_probability", 0.5)),
                    success_probability=float(data.get("success_probability", 0.5)),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=str(data.get("reasoning", ""))[:200],
                )
        except Exception:
            pass
        return _estimate_pragmatic_heuristic(policy, goal)

    async def _estimate_epistemic_llm(
        self,
        policy: Policy,
        beliefs: BeliefState,
    ) -> EpistemicEstimate:
        """LLM-based epistemic value estimation with caching and validation."""
        steps_desc = " → ".join(s.description for s in policy.steps)
        known_uncertainties = _identify_uncertain_domains(beliefs)
        prompt = build_epistemic_value_prompt(
            policy_name=policy.name,
            policy_steps_desc=steps_desc,
            beliefs_summary=summarise_beliefs(beliefs, max_entities=3),
            known_uncertainties=known_uncertainties,
        )
        try:
            # Use cache-tagged evaluate if optimized provider is available
            if self._optimized:
                response = await self._llm.evaluate(  # type: ignore[call-arg]
                    prompt, max_tokens=150, temperature=0.2,
                    cache_system="nova.efe.epistemic", cache_method="evaluate",
                )
            else:
                response = await self._llm.evaluate(prompt, max_tokens=150, temperature=0.2)

            # Use output validator for robust JSON extraction
            data = self._validator.extract_json(response.text)
            if data and isinstance(data, dict):
                data = self._validator.auto_fix_dict(
                    data,
                    required_keys=["info_gain", "uncertainties_addressed", "novelty"],
                    defaults={"info_gain": 0.3, "uncertainties_addressed": 0, "novelty": 0.2},
                )
                return EpistemicEstimate(
                    score=float(data.get("info_gain", 0.3)),
                    uncertainties_addressed=int(data.get("uncertainties_addressed", 0)),
                    expected_info_gain=float(data.get("info_gain", 0.3)),
                    novelty=float(data.get("novelty", 0.2)),
                )
        except Exception:
            pass
        return _estimate_epistemic_heuristic(policy, beliefs)


# ─── Analytical / Heuristic Estimators ───────────────────────────


def _estimate_pragmatic_heuristic(policy: Policy, goal: Goal) -> PragmaticEstimate:
    """
    Heuristic pragmatic estimate when LLM is unavailable.
    Based on effort level and action type alignment with the goal.
    """
    effort_map = {"none": 0.05, "low": 0.5, "medium": 0.65, "high": 0.75}
    base = effort_map.get(policy.estimated_effort, 0.5)

    # "express" actions are always somewhat pragmatic for dialogue goals
    has_express = any(s.action_type == "express" for s in policy.steps)
    if has_express and "dialogue" in goal.target_domain.lower():
        base = min(1.0, base + 0.1)

    return PragmaticEstimate(
        score=base,
        success_probability=base,
        confidence=0.4,  # Low confidence in heuristic
        reasoning="Heuristic estimate based on effort level",
    )


def _estimate_epistemic_heuristic(policy: Policy, beliefs: BeliefState) -> EpistemicEstimate:
    """
    Heuristic epistemic estimate.
    Policies that observe or ask questions have higher epistemic value.
    """
    epistemic_action_types = {"observe", "request_info"}
    epistemic_steps = sum(1 for s in policy.steps if s.action_type in epistemic_action_types)
    n = max(1, len(policy.steps))
    epistemic_ratio = epistemic_steps / n

    # Low overall confidence → more to learn → higher epistemic value
    uncertainty_bonus = (1.0 - beliefs.overall_confidence) * 0.3

    score = min(1.0, epistemic_ratio * 0.6 + uncertainty_bonus)
    return EpistemicEstimate(
        score=score,
        uncertainties_addressed=epistemic_steps,
        expected_info_gain=score,
        novelty=0.0,
    )


def _compute_constitutional_alignment(
    policy: Policy,
    drive_weights: dict[str, float],
    goal: Goal,
) -> float:
    """
    Compute constitutional alignment analytically.

    This is the inner product of the goal's drive alignment vector
    with the current constitutional drive weights, normalised to [0,1].

    A policy serving a goal that aligns with strong drives gets a
    constitutional boost. This is how the constitution shapes behaviour
    without explicit rule-following.
    """
    alignment = goal.drive_alignment

    w_coherence = drive_weights.get("coherence", 1.0)
    w_care = drive_weights.get("care", 1.0)
    w_growth = drive_weights.get("growth", 1.0)
    w_honesty = drive_weights.get("honesty", 1.0)

    weighted = (
        alignment.coherence * w_coherence
        + alignment.care * w_care
        + alignment.growth * w_growth
        + alignment.honesty * w_honesty
    )
    weight_sum = w_coherence + w_care + w_growth + w_honesty or 4.0
    return min(1.0, max(0.0, weighted / weight_sum))


def _estimate_feasibility(policy: Policy, beliefs: BeliefState) -> float:
    """
    Heuristic feasibility: can we actually execute this policy?

    Based on capability beliefs and cognitive load.
    """
    # Do-nothing and observe are always feasible
    if all(s.action_type in {"observe", "wait"} for s in policy.steps):
        return 1.0

    # Base feasibility from epistemic confidence (we know what we're doing)
    base = beliefs.self_belief.epistemic_confidence

    # Penalise high cognitive load
    load_penalty = beliefs.self_belief.cognitive_load * 0.2

    # High-effort policies are less feasible
    effort_penalty = {"none": 0.0, "low": 0.0, "medium": 0.05, "high": 0.15}.get(
        policy.estimated_effort, 0.05
    )

    return max(0.1, min(1.0, base - load_penalty - effort_penalty))


def _estimate_risk(policy: Policy) -> RiskEstimate:
    """
    Heuristic risk estimation from action types.

    High-risk action types increase EFE (make the policy less preferred).
    Low-risk action types are neutral or reduce risk.
    """
    risk_score = 0.0
    identified_risks: list[str] = list(policy.risks)

    high_risk_steps = [s for s in policy.steps if s.action_type in _HIGH_RISK_ACTIONS]
    if high_risk_steps:
        risk_score += 0.3 * len(high_risk_steps) / max(1, len(policy.steps))
        identified_risks.extend(f"High-risk action type: {s.action_type}" for s in high_risk_steps)

    # Time horizon: long-horizon policies are riskier (more uncertainty)
    horizon_risk = {"immediate": 0.0, "short": 0.05, "medium": 0.1, "long": 0.2}.get(
        policy.time_horizon, 0.05
    )
    risk_score += horizon_risk

    # Reversibility: express actions are fully reversible; actions are not
    all_expressive = all(s.action_type in _LOW_RISK_ACTIONS for s in policy.steps)
    reversibility = 1.0 if all_expressive else 0.7

    return RiskEstimate(
        expected_harm=min(1.0, risk_score),
        reversibility=reversibility,
        identified_risks=identified_risks[:5],
    )


def _identify_uncertain_domains(beliefs: BeliefState) -> str:
    """Identify where belief confidence is lowest for epistemic prompting."""
    uncertain: list[str] = []
    if beliefs.overall_confidence < 0.4:
        uncertain.append("overall world model (low confidence)")
    if beliefs.current_context.confidence < 0.4:
        uncertain.append("current situation context")
    if beliefs.self_belief.epistemic_confidence < 0.4:
        uncertain.append("own capabilities")
    for iid, ib in beliefs.individual_beliefs.items():
        if ib.valence_confidence < 0.3:
            uncertain.append(f"emotional state of {iid}")
    return "; ".join(uncertain) if uncertain else "no specific uncertainties identified"


def _parse_json_response(raw: str) -> dict | None:
    """Safely parse a JSON response from the LLM."""
    try:
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return None
