"""
EcodiaOS — Simula Service

The self-evolution system. Simula is the organism's capacity for
metamorphosis: structural change beyond parameter tuning.

Where Evo adjusts the knobs, Simula redesigns the dashboard.

Simula coordinates the full evolution proposal pipeline:
  1. DEDUPLICATE — check for duplicate/similar active proposals
  2. VALIDATE    — reject forbidden categories immediately
  3. SIMULATE    — deep multi-strategy impact prediction
  4. GATE        — route governed changes through community governance
  5. APPLY       — invoke the code agent or config updater with rollback
  6. VERIFY      — health check post-application
  7. RECORD      — write immutable history, increment version, update analytics

Interfaces:
  initialize()            — build sub-systems, load current version
  process_proposal()      — main entry point for rich proposals
  receive_evo_proposal()  — receive from Evo via bridge translation
  get_history()           — recent evolution records
  get_current_version()   — current config version number
  get_analytics()         — evolution quality metrics
  shutdown()              — graceful teardown
  stats                   — service-level metrics

Iron Rules (never violated — see SIMULA_IRON_RULES in types.py):
  - Cannot modify Equor, constitutional drives, invariants
  - Cannot modify its own logic
  - Must simulate before applying any change
  - Must maintain rollback capability
  - Evolution history is immutable
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.clients.llm import LLMProvider
from ecodiaos.config import SimulaConfig
from ecodiaos.systems.simula.analytics import EvolutionAnalyticsEngine
from ecodiaos.systems.simula.applicator import ChangeApplicator
from ecodiaos.systems.simula.bridge import EvoSimulaBridge
from ecodiaos.systems.simula.code_agent import SimulaCodeAgent
from ecodiaos.systems.simula.health import HealthChecker
from ecodiaos.systems.simula.history import EvolutionHistoryManager
from ecodiaos.systems.simula.proposal_intelligence import ProposalIntelligence
from ecodiaos.systems.simula.rollback import RollbackManager
from ecodiaos.systems.simula.simulation import ChangeSimulator
from ecodiaos.systems.simula.types import (
    FORBIDDEN,
    GOVERNANCE_REQUIRED,
    ConfigVersion,
    EnrichedSimulationResult,
    EvolutionAnalytics,
    EvolutionRecord,
    EvolutionProposal,
    ProposalResult,
    ProposalStatus,
    RiskLevel,
    SimulationResult,
    TriageResult,
    TriageStatus,
)
from ecodiaos.primitives.common import new_id, utc_now

if TYPE_CHECKING:
    from ecodiaos.clients.neo4j import Neo4jClient
    from ecodiaos.systems.memory.service import MemoryService

logger = structlog.get_logger()


class SimulaService:
    """
    Simula — the EOS self-evolution system.

    Coordinates eight sub-systems:
      ChangeSimulator           — deep multi-strategy impact prediction
      SimulaCodeAgent           — Claude-backed code generation with 11 tools
      ChangeApplicator          — routes proposals to the right application strategy
      RollbackManager           — file snapshots and restore
      EvolutionHistoryManager   — immutable Neo4j history
      EvoSimulaBridge           — Evo→Simula proposal translation
      ProposalIntelligence      — deduplication, prioritization, dependency analysis
      EvolutionAnalyticsEngine  — evolution quality tracking
    """

    system_id: str = "simula"

    def __init__(
        self,
        config: SimulaConfig,
        llm: LLMProvider,
        neo4j: Neo4jClient | None = None,
        memory: MemoryService | None = None,
        codebase_root: Path | None = None,
        instance_name: str = "EOS",
    ) -> None:
        self._config = config
        self._llm = llm
        self._neo4j = neo4j
        self._memory = memory
        self._root = codebase_root or Path(config.codebase_root).resolve()
        self._instance_name = instance_name
        self._initialized: bool = False
        self._logger = logger.bind(system="simula")

        # Sub-systems (built in initialize())
        self._simulator: ChangeSimulator | None = None
        self._code_agent: SimulaCodeAgent | None = None
        self._applicator: ChangeApplicator | None = None
        self._rollback: RollbackManager | None = None
        self._history: EvolutionHistoryManager | None = None
        self._health: HealthChecker | None = None
        self._bridge: EvoSimulaBridge | None = None
        self._intelligence: ProposalIntelligence | None = None
        self._analytics: EvolutionAnalyticsEngine | None = None

        # State
        self._current_version: int = 0
        self._active_proposals: dict[str, EvolutionProposal] = {}

        # Metrics
        self._proposals_received: int = 0
        self._proposals_approved: int = 0
        self._proposals_rejected: int = 0
        self._proposals_rolled_back: int = 0
        self._proposals_awaiting_governance: int = 0
        self._proposals_deduplicated: int = 0

    # ─── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Build all sub-systems and load current config version from history.
        Must be called before any other method.
        """
        if self._initialized:
            return

        # Build the rollback manager
        self._rollback = RollbackManager(codebase_root=self._root)

        # Build the health checker
        self._health = HealthChecker(
            codebase_root=self._root,
            test_command=self._config.test_command,
        )

        # Build the code agent (uses the configured model, not the shared LLM)
        code_agent_llm = self._llm
        self._code_agent = SimulaCodeAgent(
            llm=code_agent_llm,
            codebase_root=self._root,
            max_turns=self._config.max_code_agent_turns,
        )

        # Build the applicator
        self._applicator = ChangeApplicator(
            code_agent=self._code_agent,
            rollback_manager=self._rollback,
            health_checker=self._health,
            codebase_root=self._root,
        )

        # Build the history manager (requires Neo4j)
        if self._neo4j is not None:
            self._history = EvolutionHistoryManager(neo4j=self._neo4j)
            self._current_version = await self._history.get_current_version()
        else:
            self._logger.warning(
                "simula_no_neo4j",
                message="Evolution history will not be persisted (no Neo4j client)",
            )
            self._current_version = 0

        # Build the analytics engine (depends on history)
        self._analytics = EvolutionAnalyticsEngine(history=self._history)

        # Build the deep simulator (depends on analytics for dynamic caution)
        self._simulator = ChangeSimulator(
            config=self._config,
            llm=self._llm,
            memory=self._memory,
            analytics=self._analytics,
            codebase_root=self._root,
        )

        # Build the Evo↔Simula bridge
        self._bridge = EvoSimulaBridge(
            llm=self._llm,
            memory=self._memory,
        )

        # Build the proposal intelligence layer
        self._intelligence = ProposalIntelligence(
            llm=self._llm,
            analytics=self._analytics,
        )

        # Pre-compute analytics from history
        if self._history is not None:
            try:
                await self._analytics.compute_analytics()
            except Exception as exc:
                self._logger.warning("initial_analytics_failed", error=str(exc))

        self._initialized = True
        self._logger.info(
            "simula_initialized",
            current_version=self._current_version,
            codebase_root=str(self._root),
            max_code_agent_turns=self._config.max_code_agent_turns,
            subsystems=[
                "simulator", "code_agent", "applicator", "rollback",
                "health", "bridge", "intelligence", "analytics",
                "history" if self._history else "history(disabled)",
            ],
        )

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._logger.info(
            "simula_shutdown",
            proposals_received=self._proposals_received,
            proposals_approved=self._proposals_approved,
            proposals_rejected=self._proposals_rejected,
            proposals_rolled_back=self._proposals_rolled_back,
            proposals_deduplicated=self._proposals_deduplicated,
            current_version=self._current_version,
        )

    # ─── Triage (Fast-Path Pre-Simulation) ─────────────────────────────────────

    def _triage_proposal(self, proposal: EvolutionProposal) -> TriageResult:
        """
        Fast-path proposal check. If trivial, skip expensive simulation.
        Trivial = budget tweaks <5% with sufficient data.

        Returns TriageResult with skip_simulation=True for trivial cases.
        """
        if proposal.category.value != "adjust_budget":
            return TriageResult(
                status=TriageStatus.REQUIRES_SIMULATION,
                skip_simulation=False,
            )

        spec = proposal.change_spec
        if not spec.budget_new_value or spec.budget_old_value is None:
            return TriageResult(
                status=TriageStatus.REQUIRES_SIMULATION,
                skip_simulation=False,
            )

        # Check delta < 5%
        old_val = spec.budget_old_value
        new_val = spec.budget_new_value
        if old_val == 0.0:
            delta_pct = 1.0  # Treat zero as 100% change
        else:
            delta_pct = abs(new_val - old_val) / abs(old_val)

        if delta_pct < 0.05:
            self._logger.info(
                "proposal_triaged",
                proposal_id=proposal.id,
                status="trivial",
                reason=f"Budget delta {delta_pct:.1%} < 5%",
            )
            return TriageResult(
                status=TriageStatus.TRIVIAL,
                assumed_risk=RiskLevel.LOW,
                reason=f"Budget delta {delta_pct:.1%} < 5%",
                skip_simulation=True,
            )

        return TriageResult(
            status=TriageStatus.REQUIRES_SIMULATION,
            skip_simulation=False,
        )

    # ─── Main Pipeline ─────────────────────────────────────────────────────────

    async def process_proposal(self, proposal: EvolutionProposal) -> ProposalResult:
        """
        Main entry point for evolution proposals.

        Pipeline:
          DEDUP → VALIDATE → SIMULATE → [GOVERNANCE GATE] → APPLY → VERIFY → RECORD

        Spec reference: Section III.3.2
        Performance target: validation ≤50ms, simulation ≤30s, apply ≤5s
        """
        self._proposals_received += 1
        log = self._logger.bind(proposal_id=proposal.id, category=proposal.category.value)
        log.info("proposal_received", source=proposal.source, description=proposal.description[:100])

        # ── STEP 0: Deduplication ────────────────────────────────────────────
        if self._intelligence is not None and self._active_proposals:
            try:
                all_proposals = [proposal] + list(self._active_proposals.values())
                clusters = await self._intelligence.deduplicate(all_proposals)
                if self._intelligence.is_duplicate(proposal, clusters):
                    self._proposals_deduplicated += 1
                    log.info("proposal_deduplicated")
                    return ProposalResult(
                        status=ProposalStatus.REJECTED,
                        reason="Duplicate of an active proposal",
                    )
            except Exception as exc:
                log.warning("dedup_check_failed", error=str(exc))

        self._active_proposals[proposal.id] = proposal

        # ── STEP 1: Validate ────────────────────────────────────────────────
        if proposal.category in FORBIDDEN:
            proposal.status = ProposalStatus.REJECTED
            self._proposals_rejected += 1
            reason = (
                f"Category '{proposal.category.value}' is forbidden for Simula. "
                f"Iron rule: {self._get_iron_rule_for(proposal)}"
            )
            log.warning("proposal_rejected_forbidden", reason=reason)
            self._active_proposals.pop(proposal.id, None)
            return ProposalResult(status=ProposalStatus.REJECTED, reason=reason)

        # ── STEP 1.5: Triage (fast-path for trivial cases) ──────────────────
        triage = self._triage_proposal(proposal)
        if triage.skip_simulation:
            # Build synthetic simulation result
            proposal.simulation = SimulationResult(
                episodes_tested=0,
                risk_level=triage.assumed_risk or RiskLevel.LOW,
                risk_summary=f"Triaged as trivial: {triage.reason}",
                benefit_summary=proposal.expected_benefit,
            )
            log.info("proposal_triaged_skipping_simulation", reason=triage.reason)
            # Skip STEP 2 (Simulate) and proceed directly to governance/apply

        # ── STEP 2: Simulate (deep multi-strategy) ─────────────────────────
        # Skip if already triaged (has synthetic simulation)
        if proposal.simulation is None:
            proposal.status = ProposalStatus.SIMULATING
            log.info("proposal_simulating")

            try:
                simulation = await self._simulate_change(proposal)
                proposal.simulation = simulation
            except Exception as exc:
                proposal.status = ProposalStatus.REJECTED
                self._proposals_rejected += 1
                reason = f"Simulation failed: {exc}"
                log.error("simulation_error", error=str(exc))
                self._active_proposals.pop(proposal.id, None)
                return ProposalResult(status=ProposalStatus.REJECTED, reason=reason)
        else:
            simulation = proposal.simulation

        if simulation.risk_level == RiskLevel.UNACCEPTABLE:
            proposal.status = ProposalStatus.REJECTED
            self._proposals_rejected += 1
            reason = f"Simulation shows unacceptable risk: {simulation.risk_summary}"
            log.warning("proposal_rejected_risk", risk_level=simulation.risk_level.value)
            self._active_proposals.pop(proposal.id, None)
            return ProposalResult(status=ProposalStatus.REJECTED, reason=reason)

        # ── STEP 3: Governance gate ─────────────────────────────────────────
        if self.requires_governance(proposal):
            proposal.status = ProposalStatus.AWAITING_GOVERNANCE
            self._proposals_awaiting_governance += 1
            try:
                governance_id = await self._submit_to_governance(proposal, simulation)
                proposal.governance_record_id = governance_id
            except Exception as exc:
                log.error("governance_submission_error", error=str(exc))
                governance_id = f"gov_{new_id()}"
                proposal.governance_record_id = governance_id

            log.info("proposal_awaiting_governance", governance_id=governance_id)
            return ProposalResult(
                status=ProposalStatus.AWAITING_GOVERNANCE,
                governance_record_id=governance_id,
            )

        # ── STEP 4: Apply (self-applicable changes only) ───────────────────
        return await self._apply_change(proposal)

    async def receive_evo_proposal(
        self,
        evo_description: str,
        evo_rationale: str,
        hypothesis_ids: list[str],
        hypothesis_statements: list[str],
        evidence_scores: list[float],
        supporting_episode_ids: list[str],
        mutation_target: str = "",
        mutation_type: str = "",
    ) -> ProposalResult:
        """
        Receive a proposal from Evo via the bridge.
        Translates the lightweight Evo proposal into Simula's rich format,
        then feeds it into the main pipeline.

        This is the public API that Evo's ConsolidationOrchestrator calls.
        """
        if self._bridge is None:
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason="Simula bridge not initialized",
            )

        self._logger.info(
            "evo_proposal_received",
            description=evo_description[:80],
            hypotheses=len(hypothesis_ids),
        )

        try:
            translated = await self._bridge.translate_proposal(
                evo_description=evo_description,
                evo_rationale=evo_rationale,
                hypothesis_ids=hypothesis_ids,
                hypothesis_statements=hypothesis_statements,
                evidence_scores=evidence_scores,
                supporting_episode_ids=supporting_episode_ids,
                mutation_target=mutation_target,
                mutation_type=mutation_type,
            )
        except Exception as exc:
            self._logger.error("bridge_translation_failed", error=str(exc))
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason=f"Bridge translation failed: {exc}",
            )

        return await self.process_proposal(translated)

    async def approve_governed_proposal(
        self, proposal_id: str, governance_record_id: str
    ) -> ProposalResult:
        """
        Called when a governed proposal receives community approval.
        Resumes the pipeline from the application step.
        """
        proposal = self._active_proposals.get(proposal_id)
        if proposal is None:
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason=f"Proposal {proposal_id} not found in active proposals",
            )
        if proposal.status != ProposalStatus.AWAITING_GOVERNANCE:
            return ProposalResult(
                status=ProposalStatus.REJECTED,
                reason=f"Proposal {proposal_id} is not awaiting governance (status: {proposal.status})",
            )

        proposal.status = ProposalStatus.APPROVED
        self._proposals_awaiting_governance = max(0, self._proposals_awaiting_governance - 1)
        self._logger.info("governed_proposal_approved", proposal_id=proposal_id)
        return await self._apply_change(proposal)

    def requires_governance(self, proposal: EvolutionProposal) -> bool:
        """Changes in the GOVERNANCE_REQUIRED category always need governance."""
        return proposal.category in GOVERNANCE_REQUIRED

    # ─── Query Interface ───────────────────────────────────────────────────────

    async def get_history(self, limit: int = 50) -> list[EvolutionRecord]:
        """Return the most recent evolution records."""
        if self._history is None:
            return []
        return await self._history.get_history(limit=limit)

    async def get_current_version(self) -> int:
        """Return the current config version number."""
        return self._current_version

    async def get_version_chain(self) -> list[ConfigVersion]:
        """Return the full version history chain."""
        if self._history is None:
            return []
        return await self._history.get_version_chain()

    def get_active_proposals(self) -> list[EvolutionProposal]:
        """Return all proposals currently in the pipeline."""
        return list(self._active_proposals.values())

    async def get_analytics(self) -> EvolutionAnalytics:
        """Return current evolution quality analytics."""
        if self._analytics is None:
            return EvolutionAnalytics()
        return await self._analytics.compute_analytics()

    async def get_prioritized_proposals(self) -> list[dict[str, Any]]:
        """Return active proposals ranked by priority score."""
        if self._intelligence is None or not self._active_proposals:
            return []
        priorities = await self._intelligence.prioritize(list(self._active_proposals.values()))
        return [p.model_dump() for p in priorities]

    # ─── Stats ────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "initialized": self._initialized,
            "current_version": self._current_version,
            "proposals_received": self._proposals_received,
            "proposals_approved": self._proposals_approved,
            "proposals_rejected": self._proposals_rejected,
            "proposals_rolled_back": self._proposals_rolled_back,
            "proposals_deduplicated": self._proposals_deduplicated,
            "proposals_awaiting_governance": self._proposals_awaiting_governance,
            "active_proposals": len(self._active_proposals),
        }

        # Include cached analytics summary if available
        if self._analytics is not None and self._analytics._cached_analytics is not None:
            analytics = self._analytics._cached_analytics
            base["analytics"] = {
                "total_proposals": analytics.total_proposals,
                "evolution_velocity": analytics.evolution_velocity,
                "rollback_rate": analytics.rollback_rate,
                "mean_simulation_risk": analytics.mean_simulation_risk,
            }

        return base

    # ─── Evo Bridge Callback ──────────────────────────────────────────────────

    def get_evo_callback(self) -> Any:
        """
        Return a callback function for Evo's ConsolidationOrchestrator.
        This is wired during system initialization in main.py.

        The callback signature matches what Evo Phase 8 expects:
          async def callback(evo_proposal, hypotheses) -> ProposalResult
        """
        async def _evo_callback(evo_proposal: Any, hypotheses: list[Any]) -> ProposalResult:
            # Extract fields from Evo's lightweight types
            hypothesis_ids = [getattr(h, "id", "") for h in hypotheses]
            hypothesis_statements = [getattr(h, "statement", "") for h in hypotheses]
            evidence_scores = [getattr(h, "evidence_score", 0.0) for h in hypotheses]

            # Collect all supporting episode IDs across hypotheses
            episode_ids: list[str] = []
            for h in hypotheses:
                episode_ids.extend(getattr(h, "supporting_episodes", []))

            # Extract mutation info if available
            mutation_target = ""
            mutation_type = ""
            for h in hypotheses:
                mutation = getattr(h, "proposed_mutation", None)
                if mutation is not None:
                    mutation_target = getattr(mutation, "target", "")
                    mutation_type = getattr(mutation, "type", "")
                    if hasattr(mutation_type, "value"):
                        mutation_type = mutation_type.value
                    break

            return await self.receive_evo_proposal(
                evo_description=getattr(evo_proposal, "description", ""),
                evo_rationale=getattr(evo_proposal, "rationale", ""),
                hypothesis_ids=hypothesis_ids,
                hypothesis_statements=hypothesis_statements,
                evidence_scores=evidence_scores,
                supporting_episode_ids=episode_ids,
                mutation_target=mutation_target,
                mutation_type=mutation_type,
            )

        return _evo_callback

    # ─── Private: Application ──────────────────────────────────────────────────

    async def _apply_change(self, proposal: EvolutionProposal) -> ProposalResult:
        """
        Apply a validated, simulated, approved proposal.
        Includes health check and automatic rollback on failure.
        """
        assert self._applicator is not None
        assert self._health is not None
        assert self._rollback is not None

        proposal.status = ProposalStatus.APPLYING
        log = self._logger.bind(proposal_id=proposal.id, category=proposal.category.value)
        log.info("applying_change")

        code_result, snapshot = await self._applicator.apply(proposal)

        if not code_result.success:
            proposal.status = ProposalStatus.ROLLED_BACK
            self._proposals_rolled_back += 1
            log.warning("apply_failed_no_success", error=code_result.error)
            self._active_proposals.pop(proposal.id, None)
            self._invalidate_analytics()
            return ProposalResult(
                status=ProposalStatus.ROLLED_BACK,
                reason=f"Application failed: {code_result.error}",
            )

        # ── Health check ──────────────────────────────────────────────────────
        health = await self._health.check(code_result.files_written)

        if not health.healthy:
            # Rollback
            log.warning("health_check_failed_rolling_back", issues=health.issues)
            await self._rollback.restore(snapshot)
            proposal.status = ProposalStatus.ROLLED_BACK
            self._proposals_rolled_back += 1

            # Record the rollback in history
            await self._record_evolution(proposal, code_result.files_written, rolled_back=True,
                                          rollback_reason="; ".join(health.issues))

            self._active_proposals.pop(proposal.id, None)
            self._invalidate_analytics()
            return ProposalResult(
                status=ProposalStatus.ROLLED_BACK,
                reason=f"Post-apply health check failed: {'; '.join(health.issues)}",
            )

        # ── Success ───────────────────────────────────────────────────────────
        proposal.status = ProposalStatus.APPLIED
        self._proposals_approved += 1

        from_version = self._current_version
        self._current_version += 1

        await self._record_evolution(
            proposal,
            code_result.files_written,
            rolled_back=False,
        )

        # Clean up active proposals
        self._active_proposals.pop(proposal.id, None)
        self._invalidate_analytics()

        log.info(
            "change_applied",
            from_version=from_version,
            to_version=self._current_version,
            files_changed=len(code_result.files_written),
        )

        return ProposalResult(
            status=ProposalStatus.APPLIED,
            version=self._current_version,
            files_changed=code_result.files_written,
        )

    async def _simulate_change(self, proposal: EvolutionProposal) -> SimulationResult:
        """Delegate to the deep ChangeSimulator."""
        if self._simulator is None:
            return SimulationResult(risk_level=RiskLevel.LOW, risk_summary="Simulator not initialized")
        return await self._simulator.simulate(proposal)

    async def _submit_to_governance(
        self, proposal: EvolutionProposal, simulation: SimulationResult
    ) -> str:
        """
        Submit a governed proposal to the community governance system.
        Returns a governance record ID. Enriches the governance record
        with deep simulation data for community review.
        """
        record_id = f"gov_{new_id()}"

        if self._neo4j is not None:
            try:
                # Include enriched simulation data for governance reviewers
                risk_summary = simulation.risk_summary
                benefit_summary = simulation.benefit_summary

                # Add counterfactual and alignment data if available (enriched simulation)
                enrichment = []
                if isinstance(simulation, EnrichedSimulationResult):
                    if simulation.constitutional_alignment != 0.0:
                        enrichment.append(f"Constitutional alignment: {simulation.constitutional_alignment:+.2f}")
                    if simulation.dependency_blast_radius > 0:
                        enrichment.append(f"Blast radius: {simulation.dependency_blast_radius} files")
                if enrichment:
                    risk_summary = f"{risk_summary} [{'; '.join(enrichment)}]"

                await self._neo4j.execute(
                    """
                    CREATE (:GovernanceProposal {
                        id: $id,
                        proposal_id: $proposal_id,
                        category: $category,
                        description: $description,
                        risk_level: $risk_level,
                        risk_summary: $risk_summary,
                        benefit_summary: $benefit_summary,
                        submitted_at: $submitted_at,
                        status: 'pending'
                    })
                    """,
                    {
                        "id": record_id,
                        "proposal_id": proposal.id,
                        "category": proposal.category.value,
                        "description": proposal.description,
                        "risk_level": simulation.risk_level.value,
                        "risk_summary": risk_summary,
                        "benefit_summary": benefit_summary,
                        "submitted_at": utc_now().isoformat(),
                    },
                )
            except Exception as exc:
                self._logger.warning("governance_neo4j_write_failed", error=str(exc))

        return record_id

    async def _record_evolution(
        self,
        proposal: EvolutionProposal,
        files_changed: list[str],
        rolled_back: bool = False,
        rollback_reason: str = "",
    ) -> None:
        """Write an immutable evolution record and update the version chain."""
        if self._history is None:
            return

        from_version = self._current_version - (0 if rolled_back else 1)
        to_version = self._current_version

        risk_level = (
            proposal.simulation.risk_level
            if proposal.simulation
            else RiskLevel.LOW
        )

        # Extract simulation detail fields if enriched simulation was performed
        sim_detail = {
            "simulation_episodes_tested": 0,
            "counterfactual_regression_rate": 0.0,
            "dependency_blast_radius": 0,
            "constitutional_alignment": 0.0,
            "resource_tokens_per_hour": 0,
            "caution_reasoning": "",
        }
        if isinstance(proposal.simulation, EnrichedSimulationResult):
            sim_detail["simulation_episodes_tested"] = proposal.simulation.episodes_tested
            sim_detail["counterfactual_regression_rate"] = proposal.simulation.counterfactual_regression_rate
            sim_detail["dependency_blast_radius"] = proposal.simulation.dependency_blast_radius
            sim_detail["constitutional_alignment"] = proposal.simulation.constitutional_alignment
            if proposal.simulation.resource_cost_estimate:
                sim_detail["resource_tokens_per_hour"] = (
                    proposal.simulation.resource_cost_estimate.estimated_additional_llm_tokens_per_hour
                )
            if proposal.simulation.caution_adjustment:
                sim_detail["caution_reasoning"] = proposal.simulation.caution_adjustment.reasoning

        record = EvolutionRecord(
            proposal_id=proposal.id,
            category=proposal.category,
            description=proposal.description,
            from_version=from_version,
            to_version=to_version,
            files_changed=files_changed,
            simulation_risk=risk_level,
            rolled_back=rolled_back,
            rollback_reason=rollback_reason,
            **sim_detail,
        )

        try:
            await self._history.record(record)
        except Exception as exc:
            self._logger.error("history_write_failed", error=str(exc))
            return

        if not rolled_back:
            config_hash = self._compute_config_hash(files_changed)
            version = ConfigVersion(
                version=self._current_version,
                proposal_ids=[proposal.id],
                config_hash=config_hash,
            )
            try:
                await self._history.record_version(version, previous_version=from_version)
            except Exception as exc:
                self._logger.error("version_write_failed", error=str(exc))

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _compute_config_hash(self, files_changed: list[str]) -> str:
        """Compute a stable hash of the current config state."""
        hasher = hashlib.sha256()
        for rel_path in sorted(files_changed):
            full_path = self._root / rel_path
            hasher.update(rel_path.encode())
            if full_path.exists():
                hasher.update(str(full_path.stat().st_mtime).encode())
        return hasher.hexdigest()[:16]

    def _get_iron_rule_for(self, proposal: EvolutionProposal) -> str:
        """Return the relevant iron rule for a forbidden category."""
        rule_map = {
            "modify_equor": "Simula CANNOT modify Equor in any way.",
            "modify_constitution": "Simula CANNOT modify constitutional drives.",
            "modify_invariants": "Simula CANNOT modify invariants.",
            "modify_self_evolution": "Simula CANNOT modify its own logic (no self-modifying code).",
        }
        return rule_map.get(proposal.category.value, "Category is forbidden.")

    def _invalidate_analytics(self) -> None:
        """Invalidate analytics cache after a proposal completes."""
        if self._analytics is not None:
            self._analytics.invalidate_cache()
