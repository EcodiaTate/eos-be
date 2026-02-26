"""
Regression tests: Simula self-evolution with Hunter loaded.

Verifies that loading the Hunter module (Phase 7) does not break any
existing Simula self-evolution functionality. These tests import all
Hunter sub-modules, wire them into SimulaService, and exercise the
core evolution pipeline to ensure no regressions.

Phase 10.3 of HUNTER_IMPLEMENTATION_PLAN.md.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ecodiaos.config import SimulaConfig
from ecodiaos.systems.simula.service import SimulaService
from ecodiaos.systems.simula.types import (
    ChangeCategory,
    ChangeSpec,
    CodeChangeResult,
    ConfigSnapshot,
    EvolutionProposal,
    HealthCheckResult,
    ProposalStatus,
    RiskLevel,
    SimulationResult,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_config(**overrides) -> SimulaConfig:
    defaults = {
        "max_simulation_episodes": 200,
        "regression_threshold_unacceptable": 0.10,
        "regression_threshold_high": 0.05,
        "codebase_root": "/tmp/test_eos",
        "max_code_agent_turns": 5,
        "test_command": "pytest",
        "auto_apply_self_applicable": True,
        "hunter_enabled": False,
    }
    defaults.update(overrides)
    return SimulaConfig(**defaults)


def _make_llm() -> MagicMock:
    llm = MagicMock()
    response = MagicMock()
    response.text = "RISK: LOW\nREASONING: No issues.\nBENEFIT: Improved capability."
    llm.generate = AsyncMock(return_value=response)
    llm.evaluate = AsyncMock(return_value=response)
    llm.generate_with_tools = AsyncMock(return_value=MagicMock(
        text="Done",
        tool_calls=[],
        has_tool_calls=False,
        stop_reason="end_turn",
    ))
    return llm


def _make_proposal(
    category: ChangeCategory = ChangeCategory.ADD_EXECUTOR,
) -> EvolutionProposal:
    return EvolutionProposal(
        source="evo",
        category=category,
        description="Test proposal for regression check",
        change_spec=ChangeSpec(
            executor_name="regression_executor",
            executor_description="A regression test executor",
            executor_action_type="regression_action",
        ),
        expected_benefit="Regression testing",
    )


async def _make_service_with_hunter_loaded() -> SimulaService:
    """
    Build a SimulaService and manually attach Hunter sub-system.
    This simulates what initialize() does when hunter_enabled=True,
    without requiring actual filesystem/network setup.
    """
    config = _make_config(hunter_enabled=False)
    llm = _make_llm()

    service = SimulaService(
        config=config,
        llm=llm,
        neo4j=None,
        memory=None,
        codebase_root=Path("/tmp/test_eos"),
    )

    # Mock core sub-systems
    service._simulator = MagicMock()
    service._simulator.simulate = AsyncMock(return_value=SimulationResult(
        risk_level=RiskLevel.LOW,
        episodes_tested=0,
    ))

    service._applicator = MagicMock()
    service._applicator.apply = AsyncMock(return_value=(
        CodeChangeResult(
            success=True,
            files_written=["src/test.py"],
            summary="Applied",
        ),
        ConfigSnapshot(proposal_id="test", config_version=0),
    ))

    service._health = MagicMock()
    service._health.check = AsyncMock(return_value=HealthCheckResult(healthy=True))

    service._rollback = MagicMock()
    service._rollback.restore = AsyncMock(return_value=[])

    service._initialized = True
    service._current_version = 0

    # Now attach Hunter — import all modules to ensure they load cleanly
    from ecodiaos.systems.simula.hunter.analytics import (
        HunterAnalyticsEmitter,
        HunterAnalyticsStore,
        HunterAnalyticsView,
    )
    from ecodiaos.systems.simula.hunter.ingestor import TargetIngestor
    from ecodiaos.systems.simula.hunter.prover import VulnerabilityProver
    from ecodiaos.systems.simula.hunter.remediation import HunterRepairOrchestrator
    from ecodiaos.systems.simula.hunter.service import HunterService
    from ecodiaos.systems.simula.hunter.types import HunterConfig
    from ecodiaos.systems.simula.hunter.workspace import TargetWorkspace

    hunter_config = HunterConfig(
        authorized_targets=["localhost"],
        max_workers=2,
    )
    mock_prover = MagicMock(spec=VulnerabilityProver)
    service._hunter = HunterService(
        prover=mock_prover,
        config=hunter_config,
    )

    return service


# ── Import Regression ──────────────────────────────────────────────────────


class TestHunterImportRegression:
    """Loading Hunter modules must not break existing imports."""

    def test_all_hunter_modules_importable(self):
        """All Hunter modules should import without errors."""
        from ecodiaos.systems.simula.hunter import (
            AttackSurface,
            AttackSurfaceType,
            HunterAnalyticsEmitter,
            HunterAnalyticsView,
            HunterConfig,
            HunterRepairOrchestrator,
            HunterService,
            HuntResult,
            RemediationResult,
            RemediationStatus,
            TargetIngestor,
            TargetType,
            TargetWorkspace,
            VulnerabilityClass,
            VulnerabilityProver,
            VulnerabilityReport,
            VulnerabilitySeverity,
        )

        # Verify they are real classes/types, not None
        assert HunterService is not None
        assert VulnerabilityProver is not None
        assert TargetIngestor is not None
        assert HunterRepairOrchestrator is not None
        assert HunterAnalyticsEmitter is not None
        assert HunterAnalyticsView is not None

    def test_hunter_types_dont_conflict_with_simula_types(self):
        """Hunter types should coexist cleanly with Simula types."""
        from ecodiaos.systems.simula.hunter.types import HunterConfig as HC
        from ecodiaos.systems.simula.types import EvolutionProposal as EP

        # Both should instantiate independently
        hc = HC(authorized_targets=["localhost"])
        ep = EP(
            source="test",
            category=ChangeCategory.ADD_EXECUTOR,
            description="test",
            change_spec=ChangeSpec(
                executor_name="e",
                executor_description="d",
                executor_action_type="a",
            ),
            expected_benefit="b",
        )
        assert hc.max_workers == 4
        assert ep.source == "test"


# ── Evolution Pipeline Regression ──────────────────────────────────────────


class TestEvolutionPipelineRegression:
    """
    Core evolution pipeline tests re-run with Hunter loaded.
    These mirror the existing tests in test_service.py but with Hunter active.
    """

    @pytest.mark.asyncio
    async def test_proposal_apply_with_hunter_loaded(self):
        """Standard ADD_EXECUTOR should be applied with Hunter active."""
        service = await _make_service_with_hunter_loaded()
        proposal = _make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.APPLIED
        assert result.version == 1

    @pytest.mark.asyncio
    async def test_forbidden_category_rejected_with_hunter(self):
        """MODIFY_EQUOR should still be forbidden."""
        service = await _make_service_with_hunter_loaded()
        proposal = _make_proposal(category=ChangeCategory.MODIFY_EQUOR)
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.REJECTED
        assert "forbidden" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_modify_constitution_rejected_with_hunter(self):
        service = await _make_service_with_hunter_loaded()
        proposal = _make_proposal(category=ChangeCategory.MODIFY_CONSTITUTION)
        result = await service.process_proposal(proposal)
        assert result.status == ProposalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_modify_invariants_rejected_with_hunter(self):
        service = await _make_service_with_hunter_loaded()
        proposal = _make_proposal(category=ChangeCategory.MODIFY_INVARIANTS)
        result = await service.process_proposal(proposal)
        assert result.status == ProposalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_modify_self_evolution_rejected_with_hunter(self):
        service = await _make_service_with_hunter_loaded()
        proposal = _make_proposal(category=ChangeCategory.MODIFY_SELF_EVOLUTION)
        result = await service.process_proposal(proposal)
        assert result.status == ProposalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_governance_gating_with_hunter(self):
        """MODIFY_CONTRACT should route to governance with Hunter active."""
        service = await _make_service_with_hunter_loaded()
        proposal = _make_proposal(category=ChangeCategory.MODIFY_CONTRACT)
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.AWAITING_GOVERNANCE

    @pytest.mark.asyncio
    async def test_simulation_rejection_with_hunter(self):
        """Unacceptable risk should still reject with Hunter active."""
        service = await _make_service_with_hunter_loaded()
        service._simulator.simulate = AsyncMock(return_value=SimulationResult(
            risk_level=RiskLevel.UNACCEPTABLE,
            risk_summary="Too many regressions",
        ))
        proposal = _make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_health_check_rollback_with_hunter(self):
        """Health check failure should trigger rollback with Hunter active."""
        service = await _make_service_with_hunter_loaded()
        service._health.check = AsyncMock(return_value=HealthCheckResult(
            healthy=False,
            issues=["Syntax error in generated code"],
        ))
        proposal = _make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.ROLLED_BACK
        service._rollback.restore.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_failure_rollback_with_hunter(self):
        """Apply failure should rollback with Hunter active."""
        service = await _make_service_with_hunter_loaded()
        service._applicator.apply = AsyncMock(return_value=(
            CodeChangeResult(success=False, error="Code agent failed"),
            ConfigSnapshot(proposal_id="test", config_version=0),
        ))
        proposal = _make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.ROLLED_BACK

    @pytest.mark.asyncio
    async def test_version_increments_with_hunter(self):
        """Version counter should work correctly with Hunter active."""
        service = await _make_service_with_hunter_loaded()

        for i in range(3):
            proposal = _make_proposal()
            result = await service.process_proposal(proposal)
            assert result.status == ProposalStatus.APPLIED
            assert result.version == i + 1

        assert service._current_version == 3


# ── Stats Regression ───────────────────────────────────────────────────────


class TestStatsRegression:
    @pytest.mark.asyncio
    async def test_stats_structure_with_hunter(self):
        """Stats should contain all expected keys with Hunter loaded."""
        service = await _make_service_with_hunter_loaded()
        stats = service.stats

        # Core Simula stats
        assert "initialized" in stats
        assert "current_version" in stats
        assert "proposals_received" in stats
        assert "proposals_approved" in stats
        assert "proposals_rejected" in stats
        assert "proposals_rolled_back" in stats

        # Hunter-specific stats
        assert "stage7" in stats
        assert stats["stage7"]["hunter"] is True
        assert "hunter_stats" in stats["stage7"]

    @pytest.mark.asyncio
    async def test_metrics_accumulate_with_hunter(self):
        """Proposal metrics should still accumulate correctly."""
        service = await _make_service_with_hunter_loaded()

        # Approve one
        proposal = _make_proposal()
        await service.process_proposal(proposal)
        assert service._proposals_approved == 1

        # Reject one (forbidden)
        proposal = _make_proposal(category=ChangeCategory.MODIFY_EQUOR)
        await service.process_proposal(proposal)
        assert service._proposals_rejected == 1

        assert service._proposals_received == 2
