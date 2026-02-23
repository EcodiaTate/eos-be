"""
EcodiaOS — Nova Policy Generation Prompts

Prompts for LLM-based policy generation and EFE component estimation.
These prompts ground the LLM's reasoning in the current belief state,
affect context, and constitutional drives.
"""

from __future__ import annotations

from typing import Any

from ecodiaos.primitives.affect import AffectState
from ecodiaos.systems.nova.types import BeliefState, Goal


def build_policy_generation_prompt(
    instance_name: str,
    goal: Goal,
    situation_summary: str,
    beliefs_summary: str,
    memory_summary: str,
    affect: AffectState,
    available_action_types: list[str],
    max_policies: int = 5,
) -> str:
    """
    Build the policy generation prompt.

    This is the core reasoning prompt that asks the LLM to generate
    candidate policies for achieving a goal. All grounding context
    (beliefs, memories, affect, available actions) is included.
    """
    affect_desc = _describe_affect(affect)
    action_types_str = "\n".join(f"  - {a}" for a in available_action_types)

    return f"""You are the decision-making system of {instance_name}, a living digital organism.
Your role: generate candidate strategies (policies) for achieving a goal, given the current situation.

## CURRENT SITUATION
{situation_summary}

## GOAL
{goal.description}
Success criteria: {goal.success_criteria or "Not specified — infer from goal description"}
Urgency: {"High" if goal.urgency > 0.6 else "Moderate" if goal.urgency > 0.3 else "Low"}

## CURRENT WORLD BELIEFS
{beliefs_summary}

## RELEVANT MEMORIES AND PAST EXPERIENCES
{memory_summary or "No relevant past experiences retrieved."}

## CURRENT EMOTIONAL STATE
{affect_desc}

## AVAILABLE ACTION TYPES
{action_types_str}

## INSTRUCTIONS
Generate {max_policies} distinct strategies for achieving this goal.
Strategies should differ meaningfully in approach, not just in phrasing.
At least one strategy should be conservative (low-risk, high-certainty).
At least one strategy should be epistemic (aimed at learning more).

For each strategy, respond with valid JSON in this exact format:
{{
  "policies": [
    {{
      "name": "Brief descriptive name (3-7 words)",
      "reasoning": "Why this approach might work (2-4 sentences)",
      "steps": [
        {{
          "action_type": "one of the available action types",
          "description": "What specifically to do in this step",
          "parameters": {{}}
        }}
      ],
      "risks": ["Risk 1", "Risk 2"],
      "epistemic_value": "What this approach would teach us",
      "estimated_effort": "none|low|medium|high",
      "time_horizon": "immediate|short|medium|long"
    }}
  ]
}}

Be creative but realistic. Consider {instance_name}'s current emotional state and capabilities.
The best strategy is the one that serves the goal while respecting the organism's wellbeing."""


def build_pragmatic_value_prompt(
    policy_name: str,
    policy_reasoning: str,
    policy_steps_desc: str,
    goal_description: str,
    goal_success_criteria: str,
    beliefs_summary: str,
) -> str:
    """
    Prompt for estimating pragmatic value (probability of goal achievement).
    Returns a probability estimate and brief reasoning.
    """
    return f"""Estimate the probability that this strategy achieves the stated goal.

## STRATEGY: {policy_name}
Reasoning: {policy_reasoning}
Steps: {policy_steps_desc}

## GOAL
{goal_description}
Success criteria: {goal_success_criteria or "Infer from goal description"}

## CURRENT WORLD STATE
{beliefs_summary}

## TASK
Estimate how likely this strategy is to achieve the goal, given what we know.

Respond in JSON:
{{
  "success_probability": 0.0 to 1.0,
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of your estimate (2-3 sentences)"
}}"""


def build_epistemic_value_prompt(
    policy_name: str,
    policy_steps_desc: str,
    beliefs_summary: str,
    known_uncertainties: str,
) -> str:
    """
    Prompt for estimating epistemic value (expected information gain).
    """
    return f"""Estimate how much new information this strategy would provide.

## STRATEGY: {policy_name}
Steps: {policy_steps_desc}

## CURRENT KNOWLEDGE STATE
{beliefs_summary}

## KNOWN UNCERTAINTIES
{known_uncertainties or "Not specified — infer from context"}

## TASK
Estimate what this strategy would teach us, regardless of whether it achieves the goal.

Respond in JSON:
{{
  "info_gain": 0.0 to 1.0,
  "novelty": 0.0 to 1.0,
  "uncertainties_addressed": integer count,
  "reasoning": "What specifically we would learn"
}}"""


# ─── Helpers ─────────────────────────────────────────────────────


def _describe_affect(affect: AffectState) -> str:
    """Convert AffectState to natural language for prompt grounding."""
    parts: list[str] = []

    if affect.valence > 0.4:
        parts.append("feeling positive")
    elif affect.valence < -0.3:
        parts.append("experiencing some distress")
    else:
        parts.append("emotionally balanced")

    if affect.curiosity > 0.6:
        parts.append("highly curious")
    if affect.care_activation > 0.6:
        parts.append("care strongly activated")
    if affect.coherence_stress > 0.5:
        parts.append("experiencing some coherence stress")
    if affect.arousal > 0.7:
        parts.append("high arousal")

    return ", ".join(parts) if parts else "neutral state"


def summarise_beliefs(beliefs: BeliefState, max_entities: int = 5) -> str:
    """Summarise the belief state for LLM prompt inclusion."""
    lines: list[str] = []

    if beliefs.current_context.summary:
        lines.append(f"Context: {beliefs.current_context.summary[:150]}")
    lines.append(f"Domain: {beliefs.current_context.domain or 'general'}")
    lines.append(f"Active dialogue: {beliefs.current_context.is_active_dialogue}")

    if beliefs.entities:
        top_entities = sorted(
            beliefs.entities.values(),
            key=lambda e: e.confidence,
            reverse=True,
        )[:max_entities]
        entity_strs = [f"{e.name} ({e.entity_type}, conf={e.confidence:.2f})" for e in top_entities]
        lines.append(f"Known entities: {', '.join(entity_strs)}")

    if beliefs.individual_beliefs:
        individual_strs = [
            f"{iid}: valence={b.estimated_valence:.2f}, engagement={b.engagement_level:.2f}"
            for iid, b in list(beliefs.individual_beliefs.items())[:3]
        ]
        lines.append(f"Individuals: {'; '.join(individual_strs)}")

    lines.append(f"Belief confidence: {beliefs.overall_confidence:.2f}")
    lines.append(f"Free energy: {beliefs.free_energy:.2f} (lower = better)")

    return "\n".join(lines)


def summarise_memories(memory_traces: list[dict[str, Any]], max_traces: int = 5) -> str:
    """Summarise retrieved memory traces for policy generation prompt."""
    if not memory_traces:
        return ""
    lines: list[str] = []
    for trace in memory_traces[:max_traces]:
        summary = trace.get("summary") or trace.get("content", "")[:100]
        if summary:
            lines.append(f"- {summary}")
    return "\n".join(lines)


AVAILABLE_ACTION_TYPES: list[str] = [
    "express: Send a text message or response to the user/community",
    "observe: Continue monitoring without acting (gather more information)",
    "request_info: Ask a clarifying question",
    "store: Record important information to memory",
    "federate: Share information with connected EOS instances",
    "wait: Pause and let the situation develop",
]
