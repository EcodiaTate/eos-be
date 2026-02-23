"""
EcodiaOS — Simula Change Applicator

Routes approved evolution proposals to the appropriate application
strategy and coordinates with RollbackManager for safety.

Application strategies by category:

  ADJUST_BUDGET → direct config update (no code generation needed)
  ADD_EXECUTOR, ADD_INPUT_CHANNEL, ADD_PATTERN_DETECTOR → SimulaCodeAgent
  MODIFY_CONTRACT, ADD_SYSTEM_CAPABILITY, etc. → SimulaCodeAgent (post governance)

All strategies:
  1. Snapshot affected files via RollbackManager
  2. Apply change
  3. On failure → rollback immediately
  4. On success → return CodeChangeResult + snapshot (for caller health check)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import yaml

from ecodiaos.systems.simula.types import (
    ChangeCategory,
    CodeChangeResult,
    ConfigSnapshot,
    EvolutionProposal,
)

if TYPE_CHECKING:
    from ecodiaos.systems.simula.code_agent import SimulaCodeAgent
    from ecodiaos.systems.simula.health import HealthChecker
    from ecodiaos.systems.simula.rollback import RollbackManager

logger = structlog.get_logger()


class ApplicationError(RuntimeError):
    """Raised when a change application fails unrecoverably."""


class ChangeApplicator:
    """
    Routes approved evolution proposals to the right application strategy.

    For code-level changes: delegates to SimulaCodeAgent.
    For budget changes: updates the YAML config directly.
    Always snapshots before applying, so rollback is always possible.
    """

    def __init__(
        self,
        code_agent: SimulaCodeAgent,
        rollback_manager: RollbackManager,
        health_checker: HealthChecker,
        codebase_root: Path,
    ) -> None:
        self._agent = code_agent
        self._rollback = rollback_manager
        self._health = health_checker
        self._root = codebase_root
        self._logger = logger.bind(system="simula.applicator")

    async def apply(
        self, proposal: EvolutionProposal
    ) -> tuple[CodeChangeResult, ConfigSnapshot]:
        """
        Apply an evolution proposal. Returns (result, snapshot).

        The snapshot is needed by SimulaService for rollback if the
        post-application health check fails.
        """
        self._logger.info(
            "applying_change",
            proposal_id=proposal.id,
            category=proposal.category.value,
        )

        if proposal.category == ChangeCategory.ADJUST_BUDGET:
            return await self._apply_budget(proposal)
        else:
            return await self._apply_via_code_agent(proposal)

    # ── Budget Adjustment (direct config update) ──────────────────────────────

    async def _apply_budget(
        self, proposal: EvolutionProposal
    ) -> tuple[CodeChangeResult, ConfigSnapshot]:
        """Direct config update for budget changes — no code generation."""
        spec = proposal.change_spec
        if not spec.budget_parameter or spec.budget_new_value is None:
            result = CodeChangeResult(
                success=False,
                error="Budget change spec missing parameter or new_value",
            )
            return result, ConfigSnapshot(
                proposal_id=proposal.id,
                config_version=0,
            )

        config_path = self._root / "config" / "default.yaml"
        snapshot = await self._rollback.snapshot(
            proposal_id=proposal.id,
            paths=[config_path],
        )

        try:
            data: dict = {}
            if config_path.exists():
                with open(config_path) as f:
                    data = yaml.safe_load(f) or {}

            # Navigate the dotted parameter path (e.g. "nova.efe.pragmatic")
            parts = spec.budget_parameter.split(".")
            node = data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = spec.budget_new_value

            with open(config_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)

            rel_path = str(config_path.relative_to(self._root))
            self._logger.info(
                "budget_updated",
                parameter=spec.budget_parameter,
                old_value=spec.budget_old_value,
                new_value=spec.budget_new_value,
            )
            return CodeChangeResult(
                success=True,
                files_written=[rel_path],
                summary=(
                    f"Updated {spec.budget_parameter} "
                    f"from {spec.budget_old_value} to {spec.budget_new_value}"
                ),
            ), snapshot

        except Exception as exc:
            await self._rollback.restore(snapshot)
            return CodeChangeResult(
                success=False,
                error=f"Budget update failed: {exc}",
            ), snapshot

    # ── Code Agent Application ────────────────────────────────────────────────

    async def _apply_via_code_agent(
        self, proposal: EvolutionProposal
    ) -> tuple[CodeChangeResult, ConfigSnapshot]:
        """Use SimulaCodeAgent to generate and write the implementation."""
        affected_dirs = _infer_affected_paths(proposal, self._root)
        snapshot = await self._rollback.snapshot(
            proposal_id=proposal.id,
            paths=affected_dirs,
        )

        result = await self._agent.implement(proposal)

        if not result.success:
            self._logger.warning(
                "code_agent_failed",
                proposal_id=proposal.id,
                error=result.error,
            )
            await self._rollback.restore(snapshot)

        return result, snapshot


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _infer_affected_paths(proposal: EvolutionProposal, root: Path) -> list[Path]:
    """Infer which existing paths will likely be affected by this change."""
    paths: list[Path] = []
    category = proposal.category
    spec = proposal.change_spec

    if category == ChangeCategory.ADD_EXECUTOR:
        paths.append(root / "src" / "ecodiaos" / "systems" / "axon" / "registry.py")
        executors_dir = root / "src" / "ecodiaos" / "systems" / "axon" / "executors"
        if executors_dir.exists():
            paths.append(executors_dir)
    elif category == ChangeCategory.ADD_INPUT_CHANNEL:
        paths.append(root / "src" / "ecodiaos" / "systems" / "atune")
    elif category == ChangeCategory.ADD_PATTERN_DETECTOR:
        paths.append(root / "src" / "ecodiaos" / "systems" / "evo" / "detectors.py")
    elif category in {
        ChangeCategory.MODIFY_CONTRACT,
        ChangeCategory.ADD_SYSTEM_CAPABILITY,
    }:
        for sys_name in (spec.affected_systems or []):
            sys_path = root / "src" / "ecodiaos" / "systems" / sys_name
            if sys_path.exists():
                paths.append(sys_path)

    return [p for p in paths if p.exists()]
