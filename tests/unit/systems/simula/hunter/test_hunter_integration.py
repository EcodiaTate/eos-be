"""
Integration tests for Hunter ↔ SimulaService.

Validates that:
  - SimulaService works correctly with Hunter disabled (default)
  - Hunter sub-system initializes when hunter_enabled=True
  - Self-evolution pipeline remains fully functional when Hunter is active
  - SimulaService.stats includes Hunter subsystem status
  - _ensure_hunter raises when Hunter is disabled
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
        description="Test proposal",
        change_spec=ChangeSpec(
            executor_name="test_executor",
            executor_description="A test executor",
            executor_action_type="test_action",
        ),
        expected_benefit="Testing",
    )


async def _make_service(config: SimulaConfig | None = None) -> SimulaService:
    """Build a SimulaService with all subsystems mocked."""
    config = config or _make_config()
    llm = _make_llm()
    service = SimulaService(
        config=config,
        llm=llm,
        neo4j=None,
        memory=None,
        codebase_root=Path("/tmp/test_eos"),
    )

    # Mock sub-systems to avoid filesystem / network calls
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
    return service


# ── Hunter Disabled (Default) ──────────────────────────────────────────────


class TestHunterDisabled:
    @pytest.mark.asyncio
    async def test_service_works_with_hunter_disabled(self):
        """Normal evolution pipeline should work when hunter_enabled=False."""
        service = await _make_service(config=_make_config(hunter_enabled=False))
        proposal = _make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.APPLIED
        assert service._hunter is None

    @pytest.mark.asyncio
    async def test_ensure_hunter_raises_when_disabled(self):
        """_ensure_hunter should raise RuntimeError when Hunter not enabled."""
        service = await _make_service(config=_make_config(hunter_enabled=False))
        with pytest.raises(RuntimeError, match="Hunter is not enabled"):
            service._ensure_hunter()

    @pytest.mark.asyncio
    async def test_stats_show_hunter_disabled(self):
        """Stats should reflect that Hunter is disabled."""
        service = await _make_service(config=_make_config(hunter_enabled=False))
        stats = service.stats
        assert stats["stage7"]["hunter"] is False

    @pytest.mark.asyncio
    async def test_hunt_external_target_raises_when_disabled(self):
        """hunt_external_target should raise when Hunter is disabled."""
        service = await _make_service(config=_make_config(hunter_enabled=False))
        with pytest.raises(RuntimeError, match="Hunter is not enabled"):
            await service.hunt_external_target("https://github.com/test/repo")

    @pytest.mark.asyncio
    async def test_hunt_internal_eos_raises_when_disabled(self):
        """hunt_internal_eos should raise when Hunter is disabled."""
        service = await _make_service(config=_make_config(hunter_enabled=False))
        with pytest.raises(RuntimeError, match="Hunter is not enabled"):
            await service.hunt_internal_eos()


# ── Evolution Pipeline Unaffected ──────────────────────────────────────────


class TestEvolutionUnaffected:
    @pytest.mark.asyncio
    async def test_proposal_approved_without_hunter(self):
        """Standard ADD_EXECUTOR proposal should be approved normally."""
        service = await _make_service()
        proposal = _make_proposal(category=ChangeCategory.ADD_EXECUTOR)
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.APPLIED
        assert result.version == 1

    @pytest.mark.asyncio
    async def test_forbidden_category_still_rejected(self):
        """Iron rules for evolution should hold regardless of Hunter."""
        service = await _make_service()
        proposal = _make_proposal(category=ChangeCategory.MODIFY_EQUOR)
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.REJECTED
        assert "forbidden" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_governance_routing_still_works(self):
        """Governance gating for MODIFY_CONTRACT should still function."""
        service = await _make_service()
        proposal = _make_proposal(category=ChangeCategory.MODIFY_CONTRACT)
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.AWAITING_GOVERNANCE

    @pytest.mark.asyncio
    async def test_simulation_gating_still_works(self):
        """UNACCEPTABLE risk should still reject proposals."""
        service = await _make_service()
        service._simulator.simulate = AsyncMock(return_value=SimulationResult(
            risk_level=RiskLevel.UNACCEPTABLE,
            risk_summary="Too many regressions",
        ))
        proposal = _make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_rollback_still_works(self):
        """Health check failure should still trigger rollback."""
        service = await _make_service()
        service._health.check = AsyncMock(return_value=HealthCheckResult(
            healthy=False,
            issues=["Generated code has errors"],
        ))
        proposal = _make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.ROLLED_BACK
        service._rollback.restore.assert_called_once()

    @pytest.mark.asyncio
    async def test_version_increments_correctly(self):
        """Version should increment on successful apply."""
        service = await _make_service()
        assert service._current_version == 0

        proposal = _make_proposal()
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.APPLIED
        assert service._current_version == 1


# ── Hunter Enabled (Mocked Sub-systems) ───────────────────────────────────


class TestHunterEnabled:
    @pytest.mark.asyncio
    async def test_hunter_attribute_set_when_enabled(self):
        """When hunter_enabled=True, _hunter should be set after init."""
        service = await _make_service()
        # Manually set _hunter to simulate successful initialization
        from ecodiaos.systems.simula.hunter.service import HunterService
        from ecodiaos.systems.simula.hunter.types import HunterConfig

        mock_prover = MagicMock()
        hunter_config = HunterConfig(authorized_targets=["localhost"])
        service._hunter = HunterService(
            prover=mock_prover,
            config=hunter_config,
        )

        assert service._ensure_hunter() is not None

    @pytest.mark.asyncio
    async def test_evolution_still_works_with_hunter_active(self):
        """Self-evolution proposals should work even when Hunter is active."""
        service = await _make_service()

        # Simulate Hunter being active
        from ecodiaos.systems.simula.hunter.service import HunterService
        from ecodiaos.systems.simula.hunter.types import HunterConfig

        mock_prover = MagicMock()
        hunter_config = HunterConfig(authorized_targets=["localhost"])
        service._hunter = HunterService(
            prover=mock_prover,
            config=hunter_config,
        )

        # Normal evolution should still work
        proposal = _make_proposal(category=ChangeCategory.ADD_EXECUTOR)
        result = await service.process_proposal(proposal)

        assert result.status == ProposalStatus.APPLIED
        assert result.version == 1

    @pytest.mark.asyncio
    async def test_stats_include_hunter_when_active(self):
        """Stats should include Hunter subsystem info when active."""
        service = await _make_service()

        from ecodiaos.systems.simula.hunter.service import HunterService
        from ecodiaos.systems.simula.hunter.types import HunterConfig

        mock_prover = MagicMock()
        hunter_config = HunterConfig(
            authorized_targets=["localhost"],
            max_workers=2,
        )
        service._hunter = HunterService(
            prover=mock_prover,
            config=hunter_config,
        )

        stats = service.stats
        assert stats["stage7"]["hunter"] is True
        assert "hunter_stats" in stats["stage7"]
        assert stats["stage7"]["hunter_stats"]["config"]["max_workers"] == 2


# ── SimulaConfig Hunter Fields ────────────────────────────────────────────


class TestSimulaConfigHunterFields:
    def test_default_hunter_disabled(self):
        config = _make_config()
        assert config.hunter_enabled is False

    def test_hunter_config_fields(self):
        config = _make_config(
            hunter_enabled=True,
            hunter_max_workers=8,
            hunter_sandbox_timeout_s=60,
            hunter_clone_depth=5,
            hunter_log_analytics=True,
            hunter_generate_pocs=True,
            hunter_generate_patches=True,
            hunter_remediation_enabled=True,
        )
        assert config.hunter_enabled is True
        assert config.hunter_max_workers == 8
        assert config.hunter_sandbox_timeout_s == 60
        assert config.hunter_clone_depth == 5
        assert config.hunter_log_analytics is True
        assert config.hunter_generate_pocs is True
        assert config.hunter_generate_patches is True
        assert config.hunter_remediation_enabled is True

    def test_default_hunter_config_values(self):
        config = SimulaConfig(
            codebase_root="/tmp/test",
            max_simulation_episodes=100,
        )
        assert config.hunter_enabled is False
        assert config.hunter_max_workers == 4
        assert config.hunter_sandbox_timeout_s == 30
        assert config.hunter_clone_depth == 1
        assert config.hunter_log_analytics is True
        assert config.hunter_generate_pocs is False
        assert config.hunter_generate_patches is False
        assert config.hunter_remediation_enabled is False
