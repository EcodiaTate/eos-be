"""
EcodiaOS -- Simula Deep Simulation Engine

Before any change is applied, the simulator performs multi-strategy
impact prediction. This is the brain of Simula's decision-making.

Strategy stack (per proposal):
  1. Category-specific validation (static analysis / budget check / LLM reasoning)
  2. Counterfactual episode replay — "What if this existed during episode X?"
  3. Dependency graph analysis — blast radius via import-graph traversal
  4. Resource cost estimation — heuristic compute/memory/token impact
  5. Constitutional alignment prediction — drive alignment scoring
  6. Risk synthesis — combine all signals into a unified assessment

Budget efficiency:
  - Counterfactual replay: batches 30 episodes into ONE LLM call (~800 tokens)
  - Constitutional alignment: single call, 100 tokens max output
  - Dependency analysis: pure Python ast module, zero LLM tokens
  - Resource cost: heuristic lookup table, zero LLM tokens
  - Analytics-informed caution: uses cached history, zero LLM tokens

Target latency: <=30s for full simulation (spec requirement).
"""

from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.config import SimulaConfig
from ecodiaos.systems.simula.types import (
    ChangeCategory,
    CounterfactualResult,
    DependencyImpact,
    EnrichedSimulationResult,
    EvolutionProposal,
    GOVERNANCE_REQUIRED,
    ImpactType,
    ResourceCostEstimate,
    RiskLevel,
    SELF_APPLICABLE,
    SimulationResult,
)

if TYPE_CHECKING:
    from ecodiaos.clients.llm import LLMProvider
    from ecodiaos.systems.memory.service import MemoryService
    from ecodiaos.systems.simula.analytics import EvolutionAnalyticsEngine

logger = structlog.get_logger().bind(system="simula.simulation")

# Valid Python identifier pattern for names
_VALID_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")
_PASCAL_CASE = re.compile(r"^[A-Z][A-Za-z0-9]*$")

# Resource cost heuristics per category (zero LLM tokens)
_RESOURCE_COST_HEURISTICS: dict[ChangeCategory, dict[str, int | float]] = {
    ChangeCategory.ADD_EXECUTOR: {
        "llm_tokens_per_hour": 500,
        "compute_ms_per_cycle": 5,
        "memory_mb": 2.0,
    },
    ChangeCategory.ADD_INPUT_CHANNEL: {
        "llm_tokens_per_hour": 200,
        "compute_ms_per_cycle": 10,
        "memory_mb": 5.0,
    },
    ChangeCategory.ADD_PATTERN_DETECTOR: {
        "llm_tokens_per_hour": 0,
        "compute_ms_per_cycle": 3,
        "memory_mb": 1.0,
    },
    ChangeCategory.ADJUST_BUDGET: {
        "llm_tokens_per_hour": 0,
        "compute_ms_per_cycle": 0,
        "memory_mb": 0.0,
    },
}

# System directories for dependency analysis
_SYSTEM_DIRS: dict[str, str] = {
    "memory": "src/ecodiaos/systems/memory",
    "equor": "src/ecodiaos/systems/equor",
    "atune": "src/ecodiaos/systems/atune",
    "voxis": "src/ecodiaos/systems/voxis",
    "nova": "src/ecodiaos/systems/nova",
    "axon": "src/ecodiaos/systems/axon",
    "evo": "src/ecodiaos/systems/evo",
    "simula": "src/ecodiaos/systems/simula",
}


class ChangeSimulator:
    """
    Deep multi-strategy impact simulator. Combines category-specific
    validation with counterfactual replay, dependency analysis, resource
    estimation, and constitutional alignment prediction.

    All strategies run concurrently where possible (asyncio.gather)
    to stay within the 30s latency target.
    """

    def __init__(
        self,
        config: SimulaConfig,
        llm: LLMProvider,
        memory: MemoryService | None = None,
        analytics: EvolutionAnalyticsEngine | None = None,
        codebase_root: Path | None = None,
    ) -> None:
        self._config = config
        self._llm = llm
        self._memory = memory
        self._analytics = analytics
        self._root = codebase_root or Path(config.codebase_root).resolve()
        self._log = logger

    async def simulate(self, proposal: EvolutionProposal) -> EnrichedSimulationResult:
        """
        Main simulation entry point. Runs category-specific validation
        plus cross-cutting deep analysis, then synthesizes a unified
        risk assessment.

        All independent analyses run concurrently via asyncio.gather.
        """
        self._log.info(
            "deep_simulation_started",
            proposal_id=proposal.id,
            category=proposal.category.value,
        )

        # Forbidden categories are rejected before reaching simulation,
        # but defend in depth
        from ecodiaos.systems.simula.types import FORBIDDEN
        if proposal.category in FORBIDDEN:
            return EnrichedSimulationResult(
                risk_level=RiskLevel.UNACCEPTABLE,
                risk_summary=f"Category {proposal.category.value} is forbidden.",
            )

        # Run all strategies concurrently
        base_task = self._simulate_by_category(proposal)
        counterfactual_task = self._counterfactual_replay(proposal)
        dependency_task = self._analyze_dependencies(proposal)
        alignment_task = self._predict_constitutional_alignment(proposal)

        base_result, counterfactuals, dependency_impacts, alignment = await asyncio.gather(
            base_task,
            counterfactual_task,
            dependency_task,
            alignment_task,
            return_exceptions=True,
        )

        # Handle exceptions gracefully -- individual strategy failures
        # should not prevent the simulation from completing
        if isinstance(base_result, BaseException):
            self._log.error("base_simulation_failed", error=str(base_result))
            base_result = SimulationResult(
                risk_level=RiskLevel.HIGH,
                risk_summary=f"Base simulation failed: {base_result}",
            )
        if isinstance(counterfactuals, BaseException):
            self._log.warning("counterfactual_replay_failed", error=str(counterfactuals))
            counterfactuals = []
        if isinstance(dependency_impacts, BaseException):
            self._log.warning("dependency_analysis_failed", error=str(dependency_impacts))
            dependency_impacts = []
        if isinstance(alignment, BaseException):
            self._log.warning("alignment_prediction_failed", error=str(alignment))
            alignment = 0.0

        # Resource cost estimation (pure heuristic, synchronous)
        cost_estimate = self._estimate_resource_cost(proposal)

        # Synthesize all signals
        result = self._synthesize_risk(
            base_result=base_result,
            counterfactuals=counterfactuals,
            dependency_impacts=dependency_impacts,
            cost_estimate=cost_estimate,
            constitutional_alignment=alignment,
            proposal=proposal,
        )

        self._log.info(
            "deep_simulation_complete",
            proposal_id=proposal.id,
            risk_level=result.risk_level.value,
            counterfactuals=len(result.counterfactuals),
            dependency_blast_radius=result.dependency_blast_radius,
            constitutional_alignment=round(result.constitutional_alignment, 2),
            episodes_tested=result.episodes_tested,
        )
        return result

    # ─── Category-Specific Simulation ────────────────────────────────────────

    async def _simulate_by_category(self, proposal: EvolutionProposal) -> SimulationResult:
        """Dispatch to the right category-specific strategy."""
        if proposal.category in SELF_APPLICABLE:
            if proposal.category == ChangeCategory.ADJUST_BUDGET:
                return await self._simulate_budget(proposal)
            else:
                return await self._simulate_additive(proposal)
        elif proposal.category in GOVERNANCE_REQUIRED:
            return await self._simulate_governance(proposal)
        else:
            return SimulationResult(
                risk_level=RiskLevel.UNACCEPTABLE,
                risk_summary=f"Category {proposal.category.value} is forbidden.",
            )

    async def _simulate_additive(self, proposal: EvolutionProposal) -> SimulationResult:
        """
        Enhanced static analysis for additive changes.
        Beyond name validation: checks naming conventions, system existence,
        existing overlap detection, and spec completeness.
        """
        spec = proposal.change_spec
        issues: list[str] = []

        # Determine the relevant name and validate by category
        name: str | None = None
        class_name: str | None = None

        if proposal.category == ChangeCategory.ADD_EXECUTOR:
            name = spec.executor_name
            if not spec.executor_action_type:
                issues.append("executor_action_type is required")
            if name and not _SNAKE_CASE.match(name):
                issues.append(
                    f"Executor module name {name!r} should be snake_case "
                    f"(e.g., 'email_sender', not 'EmailSender')"
                )
            # Check if executor with this action_type already exists
            if spec.executor_action_type:
                existing = await self._check_existing_executor(spec.executor_action_type)
                if existing:
                    issues.append(
                        f"Executor for action_type {spec.executor_action_type!r} "
                        f"already exists: {existing}"
                    )
            # Verify the axon executors directory exists
            executors_dir = self._root / "src" / "ecodiaos" / "systems" / "axon" / "executors"
            if not executors_dir.exists():
                issues.append("Axon executors directory not found -- system may not be built yet")

        elif proposal.category == ChangeCategory.ADD_INPUT_CHANNEL:
            name = spec.channel_name
            if not spec.channel_type:
                issues.append("channel_type is required")
            if name and not _SNAKE_CASE.match(name):
                issues.append(f"Channel module name {name!r} should be snake_case")

        elif proposal.category == ChangeCategory.ADD_PATTERN_DETECTOR:
            name = spec.detector_name
            if not spec.detector_pattern_type:
                issues.append("detector_pattern_type is required")
            if name and not _PASCAL_CASE.match(name):
                issues.append(
                    f"Detector class name {name!r} should be PascalCase "
                    f"(e.g., 'FrequencyDetector')"
                )

        if name is None:
            issues.append("No name provided for additive change")
        elif not _VALID_NAME.match(name):
            issues.append(f"Name {name!r} is not a valid Python identifier")

        if issues:
            return SimulationResult(
                risk_level=RiskLevel.HIGH,
                risk_summary="Spec validation failed: " + "; ".join(issues),
                benefit_summary=proposal.expected_benefit,
            )

        return SimulationResult(
            episodes_tested=0,
            risk_level=RiskLevel.LOW,
            risk_summary="Additive change passes enhanced static analysis.",
            benefit_summary=proposal.expected_benefit,
        )

    async def _simulate_budget(self, proposal: EvolutionProposal) -> SimulationResult:
        """Validate budget parameter range and assess risk magnitude."""
        from ecodiaos.systems.evo.types import TUNABLE_PARAMETERS

        spec = proposal.change_spec
        if not spec.budget_parameter:
            return SimulationResult(
                risk_level=RiskLevel.HIGH,
                risk_summary="ADJUST_BUDGET proposal missing budget_parameter.",
            )
        if spec.budget_parameter not in TUNABLE_PARAMETERS:
            return SimulationResult(
                risk_level=RiskLevel.UNACCEPTABLE,
                risk_summary=f"Parameter {spec.budget_parameter!r} is not in TUNABLE_PARAMETERS.",
            )

        param_spec = TUNABLE_PARAMETERS[spec.budget_parameter]
        new_val = spec.budget_new_value
        old_val = spec.budget_old_value

        if new_val is None:
            return SimulationResult(
                risk_level=RiskLevel.HIGH,
                risk_summary="ADJUST_BUDGET proposal missing budget_new_value.",
            )
        if new_val < param_spec.min_val or new_val > param_spec.max_val:
            return SimulationResult(
                risk_level=RiskLevel.UNACCEPTABLE,
                risk_summary=(
                    f"{spec.budget_parameter} new value {new_val} is outside allowed range "
                    f"[{param_spec.min_val}, {param_spec.max_val}]."
                ),
            )

        delta = abs(new_val - (old_val or 0.0))
        risk = RiskLevel.MODERATE if delta > 0.05 else RiskLevel.LOW
        return SimulationResult(
            episodes_tested=0,
            risk_level=risk,
            risk_summary=(
                f"{spec.budget_parameter}: {old_val} -> {new_val} "
                f"(delta={delta:.4f}). Risk: {risk.value}."
            ),
            benefit_summary=proposal.expected_benefit,
        )

    async def _simulate_governance(self, proposal: EvolutionProposal) -> SimulationResult:
        """
        LLM-based impact assessment for governance-required changes.
        Retrieves up to 30 recent episode summaries and asks the LLM
        to reason about the impact with explicit risk dimensions.
        """
        episode_context = ""
        episodes_count = 0

        if self._memory is not None:
            try:
                episodes = await asyncio.wait_for(
                    self._memory.retrieve_recent_episodes(limit=30),
                    timeout=5.0,
                )
                episodes_count = len(episodes)
                episode_context = self._build_episode_context(episodes)
            except Exception as exc:
                self._log.warning("episode_fetch_failed", error=str(exc))

        # Build an explicit multi-dimension assessment prompt
        prompt = (
            "You are evaluating a proposed structural change to EcodiaOS, "
            "a computational cognitive architecture.\n\n"
            f"PROPOSAL\n"
            f"Category: {proposal.category.value}\n"
            f"Description: {proposal.description}\n"
            f"Expected benefit: {proposal.expected_benefit}\n"
            f"Affected systems: {', '.join(proposal.change_spec.affected_systems) or 'unspecified'}\n\n"
            f"RECENT EPISODE CONTEXT ({episodes_count} episodes):\n{episode_context}\n\n"
            "Assess this change across four dimensions:\n"
            "1. BEHAVIORAL_RISK: Would existing behaviors regress? (LOW/MODERATE/HIGH)\n"
            "2. INTEGRATION_RISK: Could this break inter-system contracts? (LOW/MODERATE/HIGH)\n"
            "3. RESOURCE_RISK: Would this significantly increase resource consumption? (LOW/MODERATE/HIGH)\n"
            "4. REVERSIBILITY: How easy is rollback? (EASY/MODERATE/HARD)\n\n"
            "Reply with:\n"
            "RISK: <overall level: LOW|MODERATE|HIGH>\n"
            "BEHAVIORAL: <level>\n"
            "INTEGRATION: <level>\n"
            "RESOURCE: <level>\n"
            "REVERSIBILITY: <level>\n"
            "REASONING: <2-3 sentences>\n"
            "BENEFIT: <1 sentence>"
        )

        try:
            response = await asyncio.wait_for(
                self._llm.evaluate(prompt=prompt, max_tokens=400, temperature=0.2),
                timeout=10.0,
            )
            risk_level, risk_summary, benefit_summary = self._parse_llm_risk(response.text)
        except asyncio.TimeoutError:
            self._log.warning("simulation_llm_timeout", proposal_id=proposal.id)
            risk_level = RiskLevel.HIGH
            risk_summary = "LLM assessment timed out; defaulting to HIGH risk."
            benefit_summary = proposal.expected_benefit
        except Exception as exc:
            self._log.error("simulation_llm_error", error=str(exc))
            risk_level = RiskLevel.HIGH
            risk_summary = f"LLM assessment failed: {exc}"
            benefit_summary = proposal.expected_benefit

        return SimulationResult(
            episodes_tested=episodes_count,
            risk_level=risk_level,
            risk_summary=risk_summary,
            benefit_summary=benefit_summary,
        )

    # ─── Counterfactual Episode Replay ───────────────────────────────────────

    async def _counterfactual_replay(
        self, proposal: EvolutionProposal,
    ) -> list[CounterfactualResult]:
        """
        For additive changes, ask: 'If this had existed during recent episodes,
        when would it have been invoked? What would have changed?'

        Token-efficient: batches up to 30 episodes into a single LLM call
        with structured output (~800 tokens total).

        Returns empty list for non-additive changes or when Memory is unavailable.
        """
        # Only meaningful for additive changes
        if proposal.category not in {
            ChangeCategory.ADD_EXECUTOR,
            ChangeCategory.ADD_INPUT_CHANNEL,
            ChangeCategory.ADD_PATTERN_DETECTOR,
        }:
            return []

        if self._memory is None:
            return []

        # Retrieve recent episodes
        try:
            episodes = await asyncio.wait_for(
                self._memory.retrieve_recent_episodes(limit=30),
                timeout=5.0,
            )
        except Exception as exc:
            self._log.warning("counterfactual_episode_fetch_failed", error=str(exc))
            return []

        if not episodes:
            return []

        # Build the batch counterfactual prompt
        episode_summaries = []
        for i, ep in enumerate(episodes[:30], start=1):
            summary = getattr(ep, "summary", "") or getattr(ep, "raw_content", "")[:150]
            source = getattr(ep, "source", "unknown")
            episode_summaries.append(f"{i}. [{source}] {summary[:150]}")

        spec = proposal.change_spec
        change_desc = self._describe_additive_change(proposal)

        prompt = (
            f"EcodiaOS is considering adding a new capability:\n{change_desc}\n\n"
            f"Below are {len(episode_summaries)} recent episodes. For each, determine:\n"
            f"- Would this new capability have been triggered/relevant? (yes/no)\n"
            f"- If yes, what would have been different? (improvement/regression/neutral)\n\n"
            f"EPISODES:\n" + "\n".join(episode_summaries) + "\n\n"
            f"Reply as a numbered list matching the episode numbers:\n"
            f"<number>. <yes|no> | <improvement|regression|neutral> | <1 sentence reason>\n"
            f"Only include episodes where the answer is 'yes'."
        )

        try:
            response = await asyncio.wait_for(
                self._llm.evaluate(prompt=prompt, max_tokens=500, temperature=0.2),
                timeout=15.0,
            )
            return self._parse_counterfactual_response(response.text, episodes)
        except Exception as exc:
            self._log.warning("counterfactual_llm_failed", error=str(exc))
            return []

    def _describe_additive_change(self, proposal: EvolutionProposal) -> str:
        """Human-readable description of an additive change for counterfactual prompt."""
        spec = proposal.change_spec
        if proposal.category == ChangeCategory.ADD_EXECUTOR:
            return (
                f"New Axon Executor: {spec.executor_name or 'unnamed'}\n"
                f"Action type: {spec.executor_action_type or 'unspecified'}\n"
                f"Description: {spec.executor_description or proposal.description}"
            )
        elif proposal.category == ChangeCategory.ADD_INPUT_CHANNEL:
            return (
                f"New Atune Input Channel: {spec.channel_name or 'unnamed'}\n"
                f"Channel type: {spec.channel_type or 'unspecified'}\n"
                f"Description: {spec.channel_description or proposal.description}"
            )
        elif proposal.category == ChangeCategory.ADD_PATTERN_DETECTOR:
            return (
                f"New Evo Pattern Detector: {spec.detector_name or 'unnamed'}\n"
                f"Pattern type: {spec.detector_pattern_type or 'unspecified'}\n"
                f"Description: {spec.detector_description or proposal.description}"
            )
        return proposal.description

    def _parse_counterfactual_response(
        self, text: str, episodes: list,
    ) -> list[CounterfactualResult]:
        """Parse the LLM's batch counterfactual response into structured results."""
        results: list[CounterfactualResult] = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            try:
                # Expected: "3. yes | improvement | Would have handled email notifications"
                num_part, rest = line.split(".", 1)
                idx = int(num_part.strip()) - 1
                if idx < 0 or idx >= len(episodes):
                    continue

                parts = [p.strip() for p in rest.split("|")]
                if len(parts) < 2:
                    continue

                triggered = parts[0].lower().strip() in ("yes", "y", "true")
                if not triggered:
                    continue

                impact_str = parts[1].lower().strip() if len(parts) > 1 else "neutral"
                if "improvement" in impact_str:
                    impact = ImpactType.IMPROVEMENT
                elif "regression" in impact_str:
                    impact = ImpactType.REGRESSION
                else:
                    impact = ImpactType.NEUTRAL

                reasoning = parts[2].strip() if len(parts) > 2 else ""

                ep = episodes[idx]
                results.append(CounterfactualResult(
                    episode_id=getattr(ep, "id", f"ep_{idx}"),
                    would_have_triggered=True,
                    predicted_outcome=reasoning[:200],
                    impact=impact,
                    confidence=0.6,
                    reasoning=reasoning[:300],
                ))
            except (ValueError, IndexError):
                continue

        return results

    # ─── Dependency Graph Analysis ───────────────────────────────────────────

    async def _analyze_dependencies(
        self, proposal: EvolutionProposal,
    ) -> list[DependencyImpact]:
        """
        Static analysis of the affected system's import graph.
        Uses the ast module to parse Python files and trace imports.
        Zero LLM tokens -- pure computation.
        """
        affected_systems = proposal.change_spec.affected_systems
        if not affected_systems:
            affected_systems = self._infer_affected_systems(proposal)

        impacts: list[DependencyImpact] = []

        for sys_name in affected_systems:
            sys_dir = self._root / _SYSTEM_DIRS.get(sys_name, f"src/ecodiaos/systems/{sys_name}")
            if not sys_dir.exists():
                continue

            # Find all Python files in the affected system
            py_files = list(sys_dir.rglob("*.py"))

            # For each file, find what imports it from other systems
            for py_file in py_files:
                rel_path = str(py_file.relative_to(self._root))
                module_name = self._path_to_module(rel_path)
                if not module_name:
                    continue

                # Check how many other files import this module
                importers = await self._find_importers(module_name)
                if importers:
                    impacts.append(DependencyImpact(
                        file_path=rel_path,
                        impact_type="import_dependency",
                        risk_contribution=min(1.0, len(importers) * 0.1),
                    ))

            # Check for test coverage
            test_dir = self._root / "tests" / "unit" / "systems" / sys_name
            if test_dir.exists():
                test_files = list(test_dir.rglob("*.py"))
                impacts.append(DependencyImpact(
                    file_path=str(test_dir.relative_to(self._root)),
                    impact_type="test_coverage",
                    risk_contribution=0.0 if test_files else 0.3,
                ))

        return impacts

    def _infer_affected_systems(self, proposal: EvolutionProposal) -> list[str]:
        """Infer which systems a change affects from the category."""
        category_to_systems: dict[ChangeCategory, list[str]] = {
            ChangeCategory.ADD_EXECUTOR: ["axon"],
            ChangeCategory.ADD_INPUT_CHANNEL: ["atune"],
            ChangeCategory.ADD_PATTERN_DETECTOR: ["evo"],
            ChangeCategory.ADJUST_BUDGET: [],
            ChangeCategory.MODIFY_CONTRACT: [],
            ChangeCategory.ADD_SYSTEM_CAPABILITY: [],
            ChangeCategory.MODIFY_CYCLE_TIMING: ["synapse"],
            ChangeCategory.CHANGE_CONSOLIDATION: ["evo"],
        }
        return category_to_systems.get(proposal.category, [])

    async def _find_importers(self, module_name: str) -> list[str]:
        """Find files that import the given module. Scans src/ directory."""
        importers: list[str] = []
        src_dir = self._root / "src"
        if not src_dir.exists():
            return importers

        # Extract the short module name for import matching
        parts = module_name.split(".")
        short_name = parts[-1] if parts else module_name

        for py_file in src_dir.rglob("*.py"):
            try:
                source = py_file.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(py_file))
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if module_name in alias.name:
                            importers.append(str(py_file.relative_to(self._root)))
                            break
                elif isinstance(node, ast.ImportFrom):
                    if node.module and module_name in node.module:
                        importers.append(str(py_file.relative_to(self._root)))

        return importers

    def _path_to_module(self, rel_path: str) -> str | None:
        """Convert a relative file path to a dotted module name."""
        parts = rel_path.replace("\\", "/").split("/")
        if parts and parts[0] == "src":
            parts = parts[1:]
        if not parts:
            return None
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts) if parts else None

    # ─── Resource Cost Estimation ────────────────────────────────────────────

    def _estimate_resource_cost(
        self, proposal: EvolutionProposal,
    ) -> ResourceCostEstimate:
        """
        Heuristic estimation of ongoing resource impact.
        Zero LLM tokens -- pure lookup + arithmetic.
        """
        heuristics = _RESOURCE_COST_HEURISTICS.get(proposal.category)
        if heuristics is None:
            # Governance-required changes: estimate moderate cost
            return ResourceCostEstimate(
                estimated_additional_llm_tokens_per_hour=1000,
                estimated_additional_compute_ms_per_cycle=10,
                estimated_memory_mb=5.0,
                budget_headroom_percent=90.0,
            )

        tokens = int(heuristics.get("llm_tokens_per_hour", 0))
        compute = int(heuristics.get("compute_ms_per_cycle", 0))
        memory = float(heuristics.get("memory_mb", 0.0))

        # Budget headroom: what percent of the relevant system's budget remains
        # after adding this cost
        system_budget = self._get_system_budget(proposal)
        headroom = 100.0
        if system_budget > 0 and tokens > 0:
            headroom = max(0.0, 100.0 * (1.0 - tokens / system_budget))

        return ResourceCostEstimate(
            estimated_additional_llm_tokens_per_hour=tokens,
            estimated_additional_compute_ms_per_cycle=compute,
            estimated_memory_mb=memory,
            budget_headroom_percent=round(headroom, 1),
        )

    def _get_system_budget(self, proposal: EvolutionProposal) -> int:
        """Get the affected system's hourly token budget."""
        category_to_system: dict[ChangeCategory, str] = {
            ChangeCategory.ADD_EXECUTOR: "axon",
            ChangeCategory.ADD_INPUT_CHANNEL: "atune",
            ChangeCategory.ADD_PATTERN_DETECTOR: "evo",
        }
        sys_name = category_to_system.get(proposal.category, "")
        # Default system budgets (from config/default.yaml)
        default_budgets: dict[str, int] = {
            "atune": 60000,
            "equor": 30000,
            "nova": 120000,
            "voxis": 120000,
            "evo": 60000,
            "axon": 60000,
            "simula": 10000,
        }
        return default_budgets.get(sys_name, 60000)

    # ─── Constitutional Alignment Prediction ─────────────────────────────────

    async def _predict_constitutional_alignment(
        self, proposal: EvolutionProposal,
    ) -> float:
        """
        Predict how well this change aligns with the four constitutional drives.
        Single LLM call, 100 tokens max output. Returns -1.0 to 1.0.

        Budget: ~200 tokens total (prompt + response).
        """
        prompt = (
            "EcodiaOS has four constitutional drives: "
            "coherence (make sense), care (orient toward wellbeing), "
            "growth (become more capable), honesty (represent reality truthfully).\n\n"
            f"Proposed change: {proposal.description[:200]}\n"
            f"Category: {proposal.category.value}\n"
            f"Expected benefit: {proposal.expected_benefit[:100]}\n\n"
            "Score the alignment of this change with the drives from -1.0 to 1.0.\n"
            "Reply with a single number only (e.g., 0.7)."
        )

        try:
            response = await asyncio.wait_for(
                self._llm.evaluate(prompt=prompt, max_tokens=20, temperature=0.1),
                timeout=5.0,
            )
            # Extract the float from the response
            text = response.text.strip()
            # Handle potential formatting like "0.7" or "Score: 0.7"
            for token in text.split():
                try:
                    score = float(token.strip(".,;:"))
                    return max(-1.0, min(1.0, score))
                except ValueError:
                    continue
            return 0.0
        except Exception as exc:
            self._log.warning("alignment_prediction_failed", error=str(exc))
            return 0.0

    # ─── Risk Synthesis ──────────────────────────────────────────────────────

    def _synthesize_risk(
        self,
        base_result: SimulationResult,
        counterfactuals: list[CounterfactualResult],
        dependency_impacts: list[DependencyImpact],
        cost_estimate: ResourceCostEstimate,
        constitutional_alignment: float,
        proposal: EvolutionProposal,
    ) -> EnrichedSimulationResult:
        """
        Combine all simulation signals into a unified risk assessment.

        Risk factors (weighted):
          - Base category simulation: 40%
          - Counterfactual regression rate: 20%
          - Dependency blast radius: 15%
          - Resource cost: 10%
          - Constitutional alignment: 15% (negative alignment increases risk)

        Dynamic adjustment: if analytics show high rollback rate for this
        category, bump the risk level up one notch.
        """
        # Counterfactual regression rate
        cf_regressions = sum(1 for cf in counterfactuals if cf.impact == ImpactType.REGRESSION)
        cf_total = len(counterfactuals) if counterfactuals else 1
        cf_regression_rate = cf_regressions / max(1, cf_total)

        # Dependency blast radius
        blast_radius = len(dependency_impacts)
        total_risk_contribution = sum(d.risk_contribution for d in dependency_impacts)

        # Resource risk (0-1 scale based on budget consumption)
        resource_risk = 1.0 - (cost_estimate.budget_headroom_percent / 100.0) if cost_estimate else 0.0

        # Constitutional risk (alignment < 0 adds risk)
        alignment_risk = max(0.0, -constitutional_alignment)

        # Base risk as numeric
        base_risk_numeric = {
            RiskLevel.LOW: 0.1,
            RiskLevel.MODERATE: 0.4,
            RiskLevel.HIGH: 0.7,
            RiskLevel.UNACCEPTABLE: 1.0,
        }.get(base_result.risk_level, 0.4)

        # Weighted composite risk score (0.0 - 1.0)
        composite_risk = (
            0.40 * base_risk_numeric
            + 0.20 * cf_regression_rate
            + 0.15 * min(1.0, total_risk_contribution)
            + 0.10 * resource_risk
            + 0.15 * alignment_risk
        )

        # Dynamic caution adjustment from analytics history
        if self._analytics is not None and self._analytics.should_increase_caution(proposal.category):
            composite_risk = min(1.0, composite_risk + 0.15)
            self._log.info(
                "caution_increased",
                category=proposal.category.value,
                composite_risk=round(composite_risk, 3),
            )

        # Map composite score to RiskLevel
        if composite_risk >= 0.75:
            final_risk = RiskLevel.UNACCEPTABLE
        elif composite_risk >= 0.50:
            final_risk = RiskLevel.HIGH
        elif composite_risk >= 0.25:
            final_risk = RiskLevel.MODERATE
        else:
            final_risk = RiskLevel.LOW

        # Build summary
        summary_parts = [base_result.risk_summary]
        if counterfactuals:
            cf_improvements = sum(1 for cf in counterfactuals if cf.impact == ImpactType.IMPROVEMENT)
            summary_parts.append(
                f"Counterfactual: {cf_improvements} improvements, "
                f"{cf_regressions} regressions across {len(counterfactuals)} triggered episodes."
            )
        if blast_radius > 0:
            summary_parts.append(f"Blast radius: {blast_radius} affected files/modules.")
        if constitutional_alignment != 0.0:
            summary_parts.append(f"Constitutional alignment: {constitutional_alignment:+.2f}.")

        return EnrichedSimulationResult(
            episodes_tested=base_result.episodes_tested,
            differences=base_result.differences,
            improvements=base_result.improvements + sum(
                1 for cf in counterfactuals if cf.impact == ImpactType.IMPROVEMENT
            ),
            regressions=base_result.regressions + cf_regressions,
            neutral_changes=base_result.neutral_changes + sum(
                1 for cf in counterfactuals if cf.impact == ImpactType.NEUTRAL
            ),
            risk_level=final_risk,
            risk_summary=" ".join(summary_parts),
            benefit_summary=base_result.benefit_summary,
            counterfactuals=counterfactuals,
            dependency_impacts=dependency_impacts,
            resource_cost_estimate=cost_estimate,
            constitutional_alignment=constitutional_alignment,
            counterfactual_regression_rate=round(cf_regression_rate, 3),
            dependency_blast_radius=blast_radius,
        )

    # ─── Helpers ─────────────────────────────────────────────────────────────

    async def _check_existing_executor(self, action_type: str) -> str | None:
        """Check if an executor for this action_type already exists."""
        executors_dir = self._root / "src" / "ecodiaos" / "systems" / "axon" / "executors"
        if not executors_dir.exists():
            return None

        for py_file in executors_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            try:
                source = py_file.read_text(encoding="utf-8")
                if f'"{action_type}"' in source or f"'{action_type}'" in source:
                    return str(py_file.name)
            except Exception:
                continue
        return None

    async def _check_name_conflict(self, name: str, category: ChangeCategory) -> bool:
        """Returns True if the name would cause a conflict."""
        if not _VALID_NAME.match(name):
            return True
        return False

    def _build_episode_context(self, episodes: list) -> str:
        """Build concise context string from episode objects."""
        lines: list[str] = []
        for i, ep in enumerate(episodes[:30], start=1):
            summary = getattr(ep, "summary", "") or getattr(ep, "raw_content", "")[:200]
            source = getattr(ep, "source", "")
            lines.append(f"{i}. [{source}] {summary[:200]}")
        return "\n".join(lines)

    def _parse_llm_risk(self, text: str) -> tuple[RiskLevel, str, str]:
        """Parse the LLM response to extract risk level, reasoning, and benefit."""
        risk_level = RiskLevel.MODERATE
        risk_summary = text[:500]
        benefit_summary = ""

        for line in text.splitlines():
            line = line.strip()
            upper = line.upper()
            if upper.startswith("RISK:"):
                level_str = line.split(":", 1)[-1].strip().upper()
                if level_str == "LOW":
                    risk_level = RiskLevel.LOW
                elif level_str == "MODERATE":
                    risk_level = RiskLevel.MODERATE
                elif level_str == "HIGH":
                    risk_level = RiskLevel.HIGH
                elif level_str == "UNACCEPTABLE":
                    risk_level = RiskLevel.UNACCEPTABLE
            elif upper.startswith("REASONING:"):
                risk_summary = line.split(":", 1)[-1].strip()
            elif upper.startswith("BENEFIT:"):
                benefit_summary = line.split(":", 1)[-1].strip()

        return risk_level, risk_summary, benefit_summary
