"""
EcodiaOS -- Simula Evo↔Simula Bridge

Translates Evo's lightweight evolution proposals into Simula's rich
EvolutionProposal format, enriched with hypothesis evidence, episode
context, and LLM-inferred change specifications.

This completes the learning→evolution loop: Evo detects patterns,
forms hypotheses, and when one reaches SUPPORTED status with an
EVOLUTION_PROPOSAL mutation, this bridge translates it into a fully
specified change that Simula can simulate, gate, and apply.

Translation pipeline:
  1. Collect evidence from supporting hypotheses
  2. Infer ChangeCategory from mutation type + target (rule-based, LLM fallback)
  3. Build formal ChangeSpec via LLM reasoning (single structured output call)
  4. Construct the rich SimulaEvolutionProposal

Budget: ~500 tokens per translation (1 LLM call for ChangeSpec construction).
Rule-based category inference uses zero LLM tokens.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.primitives.common import new_id, utc_now
from ecodiaos.systems.simula.types import (
    ChangeCategory,
    ChangeSpec,
    EvoProposalEnriched,
    EvolutionProposal,
    ProposalStatus,
)

if TYPE_CHECKING:
    from ecodiaos.clients.llm import LLMProvider
    from ecodiaos.systems.memory.service import MemoryService

logger = structlog.get_logger().bind(system="simula.bridge")

# Rule-based keyword → category mapping for zero-token inference
_CATEGORY_KEYWORDS: list[tuple[list[str], ChangeCategory]] = [
    (["executor", "action_type", "action type", "axon"], ChangeCategory.ADD_EXECUTOR),
    (["input_channel", "input channel", "channel", "atune", "sensor"], ChangeCategory.ADD_INPUT_CHANNEL),
    (["detector", "pattern_detector", "pattern detector", "scan"], ChangeCategory.ADD_PATTERN_DETECTOR),
    (["budget", "parameter", "tunable", "weight", "threshold"], ChangeCategory.ADJUST_BUDGET),
    (["contract", "interface", "inter-system", "protocol"], ChangeCategory.MODIFY_CONTRACT),
    (["capability", "system capability", "new capability"], ChangeCategory.ADD_SYSTEM_CAPABILITY),
    (["cycle", "timing", "theta", "rhythm"], ChangeCategory.MODIFY_CYCLE_TIMING),
    (["consolidation", "sleep", "schedule"], ChangeCategory.CHANGE_CONSOLIDATION),
]


class EvoSimulaBridge:
    """
    Translates Evo evolution proposals into Simula's rich format.
    Enriches with hypothesis evidence, infers change categories,
    and builds formal change specifications.

    Used by:
      - Evo's ConsolidationOrchestrator (Phase 8) via callback
      - SimulaService.receive_evo_proposal()
    """

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryService | None = None,
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._log = logger

    async def translate_proposal(
        self,
        evo_description: str,
        evo_rationale: str,
        hypothesis_ids: list[str],
        hypothesis_statements: list[str],
        evidence_scores: list[float],
        supporting_episode_ids: list[str],
        mutation_target: str = "",
        mutation_type: str = "",
    ) -> EvolutionProposal:
        """
        Full translation pipeline: Evo proposal → Simula EvolutionProposal.

        Steps:
          1. Collect and structure evidence
          2. Infer ChangeCategory (rule-based, LLM fallback)
          3. Build formal ChangeSpec (LLM-assisted)
          4. Construct rich proposal
        """
        self._log.info(
            "bridge_translating",
            description=evo_description[:80],
            hypotheses=len(hypothesis_ids),
            mutation_target=mutation_target,
            mutation_type=mutation_type,
        )

        # 1. Structure the enriched evidence
        enriched = EvoProposalEnriched(
            evo_description=evo_description,
            evo_rationale=evo_rationale,
            hypothesis_ids=hypothesis_ids,
            hypothesis_statements=hypothesis_statements,
            evidence_scores=evidence_scores,
            supporting_episode_ids=supporting_episode_ids,
            mutation_target=mutation_target,
            mutation_type=mutation_type,
        )

        # 2. Infer ChangeCategory
        category = await self._infer_category(
            mutation_target=mutation_target,
            mutation_type=mutation_type,
            description=evo_description,
        )
        enriched.inferred_category = category

        # 3. Build formal ChangeSpec
        change_spec = await self._build_change_spec(
            category=category,
            description=evo_description,
            mutation_target=mutation_target,
            evidence_summaries=hypothesis_statements[:5],
        )
        enriched.inferred_change_spec = change_spec

        # 4. Construct the rich Simula proposal
        proposal = EvolutionProposal(
            source="evo",
            category=category,
            description=evo_description,
            change_spec=change_spec,
            evidence=hypothesis_ids,
            expected_benefit=evo_rationale,
            risk_assessment="",
            status=ProposalStatus.PROPOSED,
        )

        self._log.info(
            "bridge_translated",
            proposal_id=proposal.id,
            inferred_category=category.value,
            evidence_count=len(hypothesis_ids),
        )
        return proposal

    async def _infer_category(
        self,
        mutation_target: str,
        mutation_type: str,
        description: str,
    ) -> ChangeCategory:
        """
        Infer the ChangeCategory from mutation metadata.

        Step 1 (zero tokens): Rule-based keyword matching on target + description.
        Step 2 (LLM fallback): If no rule matches, ask LLM to classify (~200 tokens).
        """
        # Combine all text for keyword matching
        combined = f"{mutation_target} {mutation_type} {description}".lower()

        # Rule-based matching
        for keywords, category in _CATEGORY_KEYWORDS:
            for keyword in keywords:
                if keyword in combined:
                    self._log.debug(
                        "category_inferred_rule",
                        keyword=keyword,
                        category=category.value,
                    )
                    return category

        # LLM fallback for ambiguous cases
        return await self._infer_category_llm(description, mutation_target)

    async def _infer_category_llm(
        self, description: str, mutation_target: str,
    ) -> ChangeCategory:
        """LLM-based category classification. ~200 tokens."""
        categories = [
            f"- {c.value}: {c.name}"
            for c in ChangeCategory
            if c not in {
                ChangeCategory.MODIFY_EQUOR,
                ChangeCategory.MODIFY_CONSTITUTION,
                ChangeCategory.MODIFY_INVARIANTS,
                ChangeCategory.MODIFY_SELF_EVOLUTION,
            }
        ]

        prompt = (
            "Classify this proposed EcodiaOS structural change into one category.\n\n"
            f"Description: {description[:300]}\n"
            f"Target: {mutation_target}\n\n"
            "Categories:\n" + "\n".join(categories) + "\n\n"
            "Reply with the category value only (e.g., 'add_executor')."
        )

        try:
            response = await asyncio.wait_for(
                self._llm.evaluate(prompt=prompt, max_tokens=30, temperature=0.1),
                timeout=5.0,
            )
            text = response.text.strip().lower().strip("'\"")
            try:
                return ChangeCategory(text)
            except ValueError:
                # Try partial matching
                for cat in ChangeCategory:
                    if cat.value in text:
                        return cat
        except Exception as exc:
            self._log.warning("category_llm_inference_failed", error=str(exc))

        # Ultimate fallback
        self._log.warning(
            "category_fallback",
            description=description[:50],
            defaulting_to="add_system_capability",
        )
        return ChangeCategory.ADD_SYSTEM_CAPABILITY

    async def _build_change_spec(
        self,
        category: ChangeCategory,
        description: str,
        mutation_target: str,
        evidence_summaries: list[str],
    ) -> ChangeSpec:
        """
        Build a formal ChangeSpec via LLM-assisted reasoning.
        Single call with structured output. ~500 tokens.
        """
        evidence_text = "\n".join(f"- {s[:150]}" for s in evidence_summaries) or "none"

        # Category-specific field instructions
        field_instructions = self._get_field_instructions(category)

        prompt = (
            "You are constructing a formal change specification for EcodiaOS.\n\n"
            f"Category: {category.value}\n"
            f"Description: {description[:300]}\n"
            f"Target: {mutation_target}\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Required fields for {category.value}:\n{field_instructions}\n\n"
            "Reply as key=value pairs, one per line. Example:\n"
            "executor_name=email_sender\n"
            "executor_action_type=send_email\n"
            "executor_description=Sends email notifications via SMTP\n"
            "affected_systems=axon\n"
            "additional_context=Triggered by notification intents"
        )

        try:
            response = await asyncio.wait_for(
                self._llm.evaluate(prompt=prompt, max_tokens=300, temperature=0.2),
                timeout=8.0,
            )
            return self._parse_change_spec(response.text, category, description)
        except Exception as exc:
            self._log.warning("change_spec_build_failed", error=str(exc))
            # Return a minimal spec based on what we know
            return self._fallback_change_spec(category, description, mutation_target)

    def _get_field_instructions(self, category: ChangeCategory) -> str:
        """Return field-specific instructions for each category."""
        instructions: dict[ChangeCategory, str] = {
            ChangeCategory.ADD_EXECUTOR: (
                "executor_name (snake_case module name)\n"
                "executor_action_type (unique string identifier)\n"
                "executor_description (what it does)\n"
                "affected_systems (always includes 'axon')"
            ),
            ChangeCategory.ADD_INPUT_CHANNEL: (
                "channel_name (snake_case module name)\n"
                "channel_type (unique string identifier)\n"
                "channel_description (what it ingests)\n"
                "affected_systems (always includes 'atune')"
            ),
            ChangeCategory.ADD_PATTERN_DETECTOR: (
                "detector_name (PascalCase class name)\n"
                "detector_pattern_type (unique string identifier)\n"
                "detector_description (what patterns it detects)\n"
                "affected_systems (always includes 'evo')"
            ),
            ChangeCategory.ADJUST_BUDGET: (
                "budget_parameter (dotted path, e.g., 'nova.efe.pragmatic')\n"
                "budget_old_value (current value)\n"
                "budget_new_value (proposed value)\n"
                "affected_systems (which system this parameter belongs to)"
            ),
            ChangeCategory.MODIFY_CONTRACT: (
                "contract_changes (list of changes)\n"
                "affected_systems (which systems are involved)\n"
                "additional_context (why this contract change is needed)"
            ),
            ChangeCategory.ADD_SYSTEM_CAPABILITY: (
                "capability_description (what the new capability does)\n"
                "affected_systems (which systems are involved)\n"
                "additional_context (design rationale)"
            ),
            ChangeCategory.MODIFY_CYCLE_TIMING: (
                "timing_parameter (which timing to change)\n"
                "timing_old_value (current value in ms)\n"
                "timing_new_value (proposed value in ms)\n"
                "affected_systems (always includes 'synapse')"
            ),
            ChangeCategory.CHANGE_CONSOLIDATION: (
                "consolidation_schedule (new schedule description)\n"
                "affected_systems (always includes 'evo')\n"
                "additional_context (why the schedule should change)"
            ),
        }
        return instructions.get(category, "additional_context (describe the change)")

    def _parse_change_spec(
        self, text: str, category: ChangeCategory, description: str,
    ) -> ChangeSpec:
        """Parse LLM key=value output into a ChangeSpec."""
        fields: dict[str, Any] = {}

        for line in text.strip().splitlines():
            line = line.strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().lower()
            value = value.strip()

            if key == "affected_systems":
                fields[key] = [s.strip() for s in value.split(",")]
            elif key == "contract_changes":
                fields[key] = [s.strip() for s in value.split(",")]
            elif key in ("budget_old_value", "budget_new_value", "timing_old_value", "timing_new_value"):
                try:
                    fields[key] = float(value)
                except ValueError:
                    pass
            else:
                fields[key] = value

        # Ensure additional_context includes the original description
        if "additional_context" not in fields:
            fields["additional_context"] = description[:200]

        try:
            return ChangeSpec(**fields)
        except Exception:
            # If parsing fails, return a minimal spec
            return self._fallback_change_spec(category, description, "")

    def _fallback_change_spec(
        self, category: ChangeCategory, description: str, mutation_target: str,
    ) -> ChangeSpec:
        """Build a minimal ChangeSpec when LLM parsing fails."""
        spec = ChangeSpec(additional_context=description[:300])

        if category == ChangeCategory.ADD_EXECUTOR:
            name = mutation_target or "new_executor"
            spec.executor_name = name.replace(" ", "_").lower()
            spec.executor_action_type = spec.executor_name
            spec.executor_description = description[:200]
            spec.affected_systems = ["axon"]
        elif category == ChangeCategory.ADD_INPUT_CHANNEL:
            name = mutation_target or "new_channel"
            spec.channel_name = name.replace(" ", "_").lower()
            spec.channel_type = spec.channel_name
            spec.channel_description = description[:200]
            spec.affected_systems = ["atune"]
        elif category == ChangeCategory.ADD_PATTERN_DETECTOR:
            name = mutation_target or "NewDetector"
            spec.detector_name = "".join(w.capitalize() for w in name.split("_"))
            spec.detector_pattern_type = name.replace(" ", "_").lower()
            spec.detector_description = description[:200]
            spec.affected_systems = ["evo"]
        elif category == ChangeCategory.ADJUST_BUDGET:
            spec.budget_parameter = mutation_target
            spec.affected_systems = []
        else:
            spec.capability_description = description[:200]
            spec.affected_systems = []

        return spec
