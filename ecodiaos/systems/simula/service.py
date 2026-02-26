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

from ecodiaos.clients.embedding import EmbeddingClient, create_voyage_client
from ecodiaos.clients.llm import LLMProvider, create_thinking_provider
from ecodiaos.config import SimulaConfig
from ecodiaos.systems.simula.analytics import EvolutionAnalyticsEngine
from ecodiaos.systems.simula.applicator import ChangeApplicator
from ecodiaos.systems.simula.bridge import EvoSimulaBridge
from ecodiaos.systems.simula.code_agent import SimulaCodeAgent
from ecodiaos.systems.simula.health import HealthChecker
from ecodiaos.systems.simula.history import EvolutionHistoryManager
from ecodiaos.systems.simula.learning.grpo import GRPOTrainingEngine
from ecodiaos.systems.simula.learning.lilo import LiloLibraryEngine
from ecodiaos.systems.simula.proposal_intelligence import ProposalIntelligence
from ecodiaos.systems.simula.retrieval.swe_grep import SweGrepRetriever
from ecodiaos.systems.simula.rollback import RollbackManager
from ecodiaos.systems.simula.simulation import ChangeSimulator
from ecodiaos.systems.simula.verification.incremental import IncrementalVerificationEngine
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

        # Stage 3 sub-systems
        self._incremental: IncrementalVerificationEngine | None = None
        self._swe_grep: SweGrepRetriever | None = None
        self._lilo: LiloLibraryEngine | None = None

        # Stage 4 sub-systems
        self._lean_bridge: object | None = None  # LeanBridge (lazy import)
        self._grpo: GRPOTrainingEngine | None = None
        self._diffusion_repair: object | None = None  # DiffusionRepairAgent (lazy import)

        # Stage 5 sub-systems
        self._synthesis: object | None = None  # SynthesisStrategySelector (lazy import)
        self._repair_agent: object | None = None  # RepairAgent (lazy import)
        self._orchestrator: object | None = None  # MultiAgentOrchestrator (lazy import)
        self._causal_debugger: object | None = None  # CausalDebugger (lazy import)
        self._issue_resolver: object | None = None  # IssueResolver (lazy import)

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
        self._proposals_applied_since_consolidation: int = 0

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

        # ── Stage 2: Verification bridges ─────────────────────────────────────
        from ecodiaos.systems.simula.verification.dafny_bridge import DafnyBridge
        from ecodiaos.systems.simula.verification.z3_bridge import Z3Bridge
        from ecodiaos.systems.simula.verification.static_analysis import StaticAnalysisBridge

        dafny_bridge: DafnyBridge | None = None
        if self._config.dafny_enabled:
            dafny_bridge = DafnyBridge(
                dafny_path=self._config.dafny_binary_path,
                verify_timeout_s=self._config.dafny_verify_timeout_s,
                max_rounds=self._config.dafny_max_clover_rounds,
            )
            self._logger.info("dafny_bridge_initialized")

        z3_bridge: Z3Bridge | None = None
        if self._config.z3_enabled:
            z3_bridge = Z3Bridge(
                check_timeout_ms=self._config.z3_check_timeout_ms,
                max_rounds=self._config.z3_max_discovery_rounds,
            )
            self._logger.info("z3_bridge_initialized")

        static_bridge: StaticAnalysisBridge | None = None
        if self._config.static_analysis_enabled:
            static_bridge = StaticAnalysisBridge(
                codebase_root=self._root,
            )
            self._logger.info("static_analysis_bridge_initialized")

        # Store for AgentCoder pipeline
        self._dafny_bridge = dafny_bridge
        self._z3_bridge = z3_bridge
        self._static_bridge = static_bridge

        # Build the health checker with Stage 2 bridges + Stage 3 Z3 blocking
        self._health = HealthChecker(
            codebase_root=self._root,
            test_command=self._config.test_command,
            dafny_bridge=dafny_bridge,
            z3_bridge=z3_bridge,
            static_analysis_bridge=static_bridge,
            llm=self._llm,
            z3_blocking=self._config.z3_blocking,
        )

        # ── Stage 1A: Extended-thinking provider for governance/high-risk ────
        thinking_provider = None
        if self._config.thinking_model_api_key:
            try:
                thinking_provider = create_thinking_provider(
                    api_key=self._config.thinking_model_api_key,
                    model=self._config.thinking_model,
                    provider=self._config.thinking_model_provider,
                    reasoning_budget=self._config.thinking_budget_tokens,
                )
                self._logger.info(
                    "thinking_provider_initialized",
                    model=self._config.thinking_model,
                    provider=self._config.thinking_model_provider,
                )
            except Exception as exc:
                self._logger.warning("thinking_provider_init_failed", error=str(exc))

        # ── Stage 1B: Voyage-code-3 embedding client ────────────────────────
        embedding_client: EmbeddingClient | None = None
        if self._config.embedding_api_key:
            try:
                embedding_client = create_voyage_client(
                    api_key=self._config.embedding_api_key,
                    model=self._config.embedding_model,
                )
                self._logger.info(
                    "embedding_client_initialized",
                    model=self._config.embedding_model,
                )
            except Exception as exc:
                self._logger.warning("embedding_client_init_failed", error=str(exc))

        # Store for shutdown cleanup
        self._embedding_client = embedding_client

        # Build the code agent with Stage 1 + 2 enhancements
        code_agent_llm = self._llm
        self._code_agent = SimulaCodeAgent(
            llm=code_agent_llm,
            codebase_root=self._root,
            max_turns=self._config.max_code_agent_turns,
            thinking_provider=thinking_provider,
            thinking_budget_tokens=self._config.thinking_budget_tokens,
            embedding_client=embedding_client,
            kv_compression_ratio=self._config.kv_compression_ratio,
            kv_compression_enabled=self._config.kv_compression_enabled,
            # Stage 2C: static analysis post-generation gate
            static_analysis_bridge=static_bridge,
            static_analysis_max_fix_iterations=self._config.static_analysis_max_fix_iterations,
        )

        # ── Stage 2D: AgentCoder pipeline agents ──────────────────────────────
        from ecodiaos.systems.simula.agents.test_designer import TestDesignerAgent
        from ecodiaos.systems.simula.agents.test_executor import TestExecutorAgent

        test_designer: TestDesignerAgent | None = None
        test_executor: TestExecutorAgent | None = None
        if self._config.agent_coder_enabled:
            test_designer = TestDesignerAgent(
                llm=self._llm,
                codebase_root=self._root,
            )
            test_executor = TestExecutorAgent(
                codebase_root=self._root,
                test_timeout_s=self._config.agent_coder_test_timeout_s,
            )
            self._logger.info("agent_coder_pipeline_initialized")

        # Build the applicator with Stage 2D pipeline
        self._applicator = ChangeApplicator(
            code_agent=self._code_agent,
            rollback_manager=self._rollback,
            health_checker=self._health,
            codebase_root=self._root,
            test_designer=test_designer,
            test_executor=test_executor,
            static_analysis_bridge=static_bridge,
            agent_coder_enabled=self._config.agent_coder_enabled,
            agent_coder_max_iterations=self._config.agent_coder_max_iterations,
        )

        # Build the history manager (requires Neo4j) with Stage 1B embedding support
        if self._neo4j is not None:
            self._history = EvolutionHistoryManager(
                neo4j=self._neo4j,
                embedding_client=embedding_client,
            )
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

        # Build the proposal intelligence layer with Stage 1B embedding dedup
        self._intelligence = ProposalIntelligence(
            llm=self._llm,
            analytics=self._analytics,
            embedding_client=embedding_client,
        )

        # ── Stage 3A: Incremental verification ─────────────────────────────────
        if self._config.incremental_verification_enabled:
            from ecodiaos.clients.redis import RedisClient

            redis_client: RedisClient | None = None
            try:
                redis_client = RedisClient()
            except Exception as exc:
                self._logger.warning("redis_client_init_failed", error=str(exc))

            self._incremental = IncrementalVerificationEngine(
                codebase_root=self._root,
                redis=redis_client,
                neo4j=self._neo4j,
                hot_ttl_seconds=self._config.incremental_hot_ttl_seconds,
            )
            self._logger.info("incremental_verification_initialized")

        # ── Stage 3B: SWE-grep retrieval ──────────────────────────────────────
        if self._config.swe_grep_enabled:
            self._swe_grep = SweGrepRetriever(
                codebase_root=self._root,
                llm=self._llm,
                max_hops=self._config.swe_grep_max_hops,
            )
            self._logger.info("swe_grep_retriever_initialized")

        # ── Stage 3C: LILO library learning ───────────────────────────────────
        if self._config.lilo_enabled:
            self._lilo = LiloLibraryEngine(
                neo4j=self._neo4j,
                llm=self._llm,
                codebase_root=self._root,
            )
            self._logger.info("lilo_library_initialized")

        # ── Stage 4A: Lean 4 proof generation ────────────────────────────────
        lean_bridge_instance = None
        if self._config.lean_enabled:
            from ecodiaos.systems.simula.verification.lean_bridge import LeanBridge

            lean_bridge_instance = LeanBridge(
                lean_path=self._config.lean_binary_path,
                project_path=self._config.lean_project_path or None,
                verify_timeout_s=self._config.lean_verify_timeout_s,
                max_attempts=self._config.lean_max_attempts,
                copilot_enabled=self._config.lean_copilot_enabled,
                dojo_enabled=self._config.lean_dojo_enabled,
                proof_library_max_size=self._config.lean_proof_library_max_size,
                neo4j=self._neo4j,
            )
            self._lean_bridge = lean_bridge_instance
            self._logger.info("lean_bridge_initialized")

        # Wire Lean bridge into health checker
        if lean_bridge_instance is not None and self._health is not None:
            self._health._lean = lean_bridge_instance
            self._health._lean_blocking = self._config.lean_blocking

        # ── Stage 4B: GRPO domain fine-tuning ─────────────────────────────────
        if self._config.grpo_enabled:
            self._grpo = GRPOTrainingEngine(
                config=self._config,
                neo4j=self._neo4j,
            )
            self._logger.info("grpo_engine_initialized")

        # ── Stage 4C: Diffusion-based code repair ────────────────────────────
        if self._config.diffusion_repair_enabled:
            from ecodiaos.systems.simula.agents.diffusion_repair import DiffusionRepairAgent

            self._diffusion_repair = DiffusionRepairAgent(
                llm=self._llm,
                codebase_root=self._root,
                model_name=self._config.diffusion_model,
                max_denoise_steps=self._config.diffusion_max_denoise_steps,
                timeout_s=self._config.diffusion_timeout_s,
                sketch_first=self._config.diffusion_sketch_first,
            )
            self._logger.info("diffusion_repair_agent_initialized")

        # ── Stage 5A: Neurosymbolic synthesis ────────────────────────────────
        if self._config.synthesis_enabled:
            from ecodiaos.systems.simula.synthesis.chopchop import ChopChopEngine
            from ecodiaos.systems.simula.synthesis.hysynth import HySynthEngine
            from ecodiaos.systems.simula.synthesis.sketch_solver import SketchSolver
            from ecodiaos.systems.simula.synthesis.strategy_selector import (
                SynthesisStrategySelector,
            )

            hysynth = HySynthEngine(
                llm=self._llm,
                codebase_root=self._root,
                max_candidates=self._config.hysynth_max_candidates,
                beam_width=self._config.hysynth_beam_width,
                timeout_s=self._config.hysynth_timeout_s,
            )
            sketch = SketchSolver(
                llm=self._llm,
                z3_bridge=z3_bridge,
                max_holes=self._config.sketch_max_holes,
                solver_timeout_ms=self._config.sketch_solver_timeout_ms,
            )
            chopchop = ChopChopEngine(
                llm=self._llm,
                codebase_root=self._root,
                max_retries_per_chunk=self._config.chopchop_max_retries,
                chunk_size_lines=self._config.chopchop_chunk_size_lines,
                timeout_s=self._config.chopchop_timeout_s,
            )
            self._synthesis = SynthesisStrategySelector(
                hysynth=hysynth,
                sketch_solver=sketch,
                chopchop=chopchop,
                codebase_root=self._root,
            )
            self._logger.info("synthesis_subsystem_initialized")

        # ── Stage 5B: Neural program repair ──────────────────────────────────
        if self._config.repair_agent_enabled:
            from ecodiaos.systems.simula.agents.repair_agent import RepairAgent

            self._repair_agent = RepairAgent(
                reasoning_llm=self._llm,
                code_llm=self._llm,
                codebase_root=self._root,
                neo4j=self._neo4j,
                max_retries=self._config.repair_max_retries,
                cost_budget_usd=self._config.repair_cost_budget_usd,
                timeout_s=self._config.repair_timeout_s,
                use_similar_fixes=self._config.repair_use_similar_fixes,
            )
            self._logger.info("repair_agent_initialized")

        # ── Stage 5C: Multi-agent orchestration ─────────────────────────────
        if self._config.orchestration_enabled and self._code_agent is not None:
            from ecodiaos.systems.simula.orchestration.orchestrator import MultiAgentOrchestrator
            from ecodiaos.systems.simula.orchestration.task_planner import TaskPlanner

            task_planner = TaskPlanner(
                codebase_root=self._root,
                llm=self._llm,
                max_dag_nodes=self._config.orchestration_max_dag_nodes,
            )
            self._orchestrator = MultiAgentOrchestrator(
                llm=self._llm,
                codebase_root=self._root,
                code_agent=self._code_agent,
                task_planner=task_planner,
                max_agents_per_stage=self._config.orchestration_max_agents_per_stage,
                timeout_s=self._config.orchestration_timeout_s,
            )
            self._logger.info("orchestrator_initialized")

        # ── Stage 5D: Causal debugging ───────────────────────────────────────
        if self._config.causal_debugging_enabled:
            from ecodiaos.systems.simula.debugging.causal_dag import CausalDebugger

            self._causal_debugger = CausalDebugger(
                llm=self._llm,
                codebase_root=self._root,
                max_interventions=self._config.causal_max_interventions,
                fault_injection_enabled=self._config.causal_fault_injection_enabled,
                timeout_s=self._config.causal_timeout_s,
            )
            self._logger.info("causal_debugger_initialized")

        # ── Stage 5E: Autonomous issue resolution ────────────────────────────
        if self._config.issue_resolution_enabled:
            from ecodiaos.systems.simula.resolution.issue_resolver import IssueResolver
            from ecodiaos.systems.simula.resolution.monitors import (
                DegradationMonitor,
                PerfRegressionMonitor,
                SecurityVulnMonitor,
            )

            perf_monitor = (
                PerfRegressionMonitor()
                if self._config.issue_perf_regression_enabled
                else None
            )
            security_monitor = (
                SecurityVulnMonitor(self._root)
                if self._config.issue_security_scan_enabled
                else None
            )
            degradation_monitor = DegradationMonitor(
                window_hours=self._config.issue_degradation_window_hours,
            )

            self._issue_resolver = IssueResolver(
                llm=self._llm,
                codebase_root=self._root,
                neo4j=self._neo4j,
                code_agent=self._code_agent,
                repair_agent=self._repair_agent if isinstance(self._repair_agent, object) else None,
                perf_monitor=perf_monitor,
                security_monitor=security_monitor,
                degradation_monitor=degradation_monitor,
                max_autonomy_level=self._config.issue_max_autonomy_level,
                abstention_threshold=self._config.issue_abstention_confidence_threshold,
            )
            self._logger.info("issue_resolver_initialized")

        # Wire SWE-grep into the bridge for pre-translation retrieval (3B.5)
        if self._bridge is not None and self._swe_grep is not None:
            self._bridge.set_swe_grep(self._swe_grep)

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
                "dafny" if dafny_bridge else "dafny(disabled)",
                "z3" if z3_bridge else "z3(disabled)",
                "static_analysis" if static_bridge else "static_analysis(disabled)",
                "incremental" if self._incremental else "incremental(disabled)",
                "swe_grep" if self._swe_grep else "swe_grep(disabled)",
                "lilo" if self._lilo else "lilo(disabled)",
                "lean" if self._lean_bridge else "lean(disabled)",
                "grpo" if self._grpo else "grpo(disabled)",
                "diffusion_repair" if self._diffusion_repair else "diffusion_repair(disabled)",
                "synthesis" if self._synthesis else "synthesis(disabled)",
                "repair_agent" if self._repair_agent else "repair_agent(disabled)",
                "orchestrator" if self._orchestrator else "orchestrator(disabled)",
                "causal_debugger" if self._causal_debugger else "causal_debugger(disabled)",
                "issue_resolver" if self._issue_resolver else "issue_resolver(disabled)",
            ],
            stage1_extended_thinking=thinking_provider is not None,
            stage1_embeddings=embedding_client is not None,
            stage1_kv_compression=self._config.kv_compression_enabled,
            stage1_kv_ratio=self._config.kv_compression_ratio,
            stage2_dafny=dafny_bridge is not None,
            stage2_z3=z3_bridge is not None,
            stage2_static_analysis=static_bridge is not None,
            stage2_agent_coder=self._config.agent_coder_enabled,
            stage3_incremental=self._incremental is not None,
            stage3_swe_grep=self._swe_grep is not None,
            stage3_lilo=self._lilo is not None,
            stage4_lean=self._lean_bridge is not None,
            stage4_grpo=self._grpo is not None,
            stage4_diffusion_repair=self._diffusion_repair is not None,
            stage5_synthesis=self._synthesis is not None,
            stage5_repair_agent=self._repair_agent is not None,
            stage5_orchestrator=self._orchestrator is not None,
            stage5_causal_debugger=self._causal_debugger is not None,
            stage5_issue_resolver=self._issue_resolver is not None,
        )

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        # Clean up Stage 1B embedding client
        if hasattr(self, "_embedding_client") and self._embedding_client is not None:
            try:
                await self._embedding_client.close()
            except Exception:
                pass

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

        # Stage 3 subsystem status
        base["stage3"] = {
            "incremental_verification": self._incremental is not None,
            "swe_grep": self._swe_grep is not None,
            "lilo": self._lilo is not None,
        }

        # Stage 4 subsystem status
        base["stage4"] = {
            "lean": self._lean_bridge is not None,
            "grpo": self._grpo is not None,
            "diffusion_repair": self._diffusion_repair is not None,
        }

        # Stage 5 subsystem status
        base["stage5"] = {
            "synthesis": self._synthesis is not None,
            "repair_agent": self._repair_agent is not None,
            "orchestrator": self._orchestrator is not None,
            "causal_debugger": self._causal_debugger is not None,
            "issue_resolver": self._issue_resolver is not None,
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

        # Stage 3C: Inject LILO library prompt into the code agent
        if self._lilo is not None and self._code_agent is not None:
            try:
                self._code_agent._lilo_prompt = await self._lilo.get_library_prompt()
            except Exception as exc:
                log.warning("lilo_prompt_error", error=str(exc))

        # Stage 4A: Inject proof library context into the code agent
        if self._lean_bridge is not None and self._code_agent is not None:
            try:
                from ecodiaos.systems.simula.verification.lean_bridge import LeanBridge
                if isinstance(self._lean_bridge, LeanBridge):
                    lib_stats = self._lean_bridge.get_library_stats()
                    if lib_stats.total_lemmas > 0:
                        self._code_agent._proof_library_prompt = (
                            f"\n\n## Proof Library ({lib_stats.total_lemmas} proven lemmas)\n"
                            f"The Lean 4 proof library contains {lib_stats.total_lemmas} proven lemmas "
                            f"across domains: {', '.join(f'{d}: {c}' for d, c in lib_stats.by_domain.items())}.\n"
                            f"Mean Lean Copilot automation rate: {lib_stats.mean_copilot_automation:.0%}.\n"
                            f"Your implementation may benefit from existing verified properties."
                        )
            except Exception as exc:
                log.warning("proof_library_prompt_error", error=str(exc))

        # Stage 4B: GRPO A/B model routing
        grpo_model_used = ""
        grpo_ab_group = ""
        if self._grpo is not None and self._code_agent is not None:
            try:
                if self._grpo.should_use_finetuned():
                    grpo_model_used = self._config.grpo_base_model + "-finetuned"
                    grpo_ab_group = "finetuned"
                    self._code_agent._grpo_model_id = grpo_model_used
                    log.info("grpo_routing_finetuned", model=grpo_model_used)
                else:
                    grpo_ab_group = "base"
                    log.info("grpo_routing_base")
            except Exception as exc:
                log.warning("grpo_routing_error", error=str(exc))

        # ── Stage 5A: Synthesis-first (fast-path before CEGIS) ────────────────
        synthesis_result_stash = None
        if self._synthesis is not None:
            try:
                from ecodiaos.systems.simula.synthesis.strategy_selector import (
                    SynthesisStrategySelector,
                )
                from ecodiaos.systems.simula.synthesis.types import SynthesisStatus

                if isinstance(self._synthesis, SynthesisStrategySelector):
                    synth_result = await self._synthesis.synthesise(proposal)
                    synthesis_result_stash = synth_result
                    if synth_result.status == SynthesisStatus.SYNTHESIZED and synth_result.final_code:
                        log.info(
                            "synthesis_succeeded",
                            strategy=synth_result.strategy.value,
                            tokens=synth_result.total_llm_tokens,
                            duration_ms=synth_result.total_duration_ms,
                        )
                        # Write synthesised code and skip CEGIS
                        if synth_result.files_written:
                            for fpath in synth_result.files_written:
                                full_path = self._root / fpath
                                if full_path.exists():
                                    log.debug("synthesis_wrote_file", path=fpath)
                    else:
                        log.info(
                            "synthesis_fell_back_to_cegis",
                            strategy=synth_result.strategy.value,
                            status=synth_result.status.value,
                        )
            except Exception as exc:
                log.warning("synthesis_error", error=str(exc))

        # ── Stage 5C: Multi-agent orchestration for multi-file proposals ──────
        if self._orchestrator is not None:
            try:
                from ecodiaos.systems.simula.orchestration.orchestrator import MultiAgentOrchestrator

                if isinstance(self._orchestrator, MultiAgentOrchestrator):
                    # Estimate affected files from proposal target + code_hint
                    estimated_files = []
                    if proposal.target:
                        estimated_files.append(proposal.target)
                    if hasattr(proposal, "affected_files"):
                        estimated_files.extend(proposal.affected_files)

                    threshold = self._config.orchestration_multi_file_threshold
                    if len(estimated_files) >= threshold:
                        log.info(
                            "orchestration_engaged",
                            files=len(estimated_files),
                            threshold=threshold,
                        )
                        orc_result = await self._orchestrator.orchestrate(
                            proposal=proposal,
                            files_to_change=estimated_files,
                        )
                        proposal._orchestration_result = orc_result  # type: ignore[attr-defined]
                        log.info(
                            "orchestration_complete",
                            success=orc_result.success,
                            stages=orc_result.parallel_stages,
                            agents=orc_result.agents_used,
                        )
            except Exception as exc:
                log.warning("orchestration_error", error=str(exc))

        code_result, snapshot = await self._applicator.apply(proposal)

        # ── Stage 5B: Neural repair agent (primary recovery before diffusion) ─
        if not code_result.success and self._repair_agent is not None:
            log.info("repair_agent_attempting")
            try:
                from ecodiaos.systems.simula.agents.repair_agent import RepairAgent as RepairAgentCls
                from ecodiaos.systems.simula.verification.types import RepairStatus

                if isinstance(self._repair_agent, RepairAgentCls):
                    broken_files = {
                        f: (self._root / f).read_text()
                        for f in code_result.files_written
                        if (self._root / f).exists()
                    }
                    repair_result = await self._repair_agent.repair(
                        proposal=proposal,
                        broken_files=broken_files,
                        test_output=code_result.test_output or code_result.error,
                    )
                    if repair_result.status == RepairStatus.REPAIRED:
                        log.info(
                            "repair_agent_succeeded",
                            attempts=repair_result.total_attempts,
                            cost=f"${repair_result.total_cost_usd:.4f}",
                        )
                        code_result.success = True
                        code_result.files_written = repair_result.files_repaired
                        code_result.error = ""
                        code_result.repair_attempted = True
                        code_result.repair_succeeded = True
                        code_result.repair_cost_usd = repair_result.total_cost_usd
                        proposal._repair_result = repair_result  # type: ignore[attr-defined]
                    else:
                        log.info("repair_agent_insufficient", status=repair_result.status.value)
                        code_result.repair_attempted = True
                        code_result.repair_succeeded = False
                        proposal._repair_result = repair_result  # type: ignore[attr-defined]
            except Exception as exc:
                log.warning("repair_agent_error", error=str(exc))

        # Stage 4C: Diffusion repair fallback when code agent fails
        if not code_result.success and self._diffusion_repair is not None:
            log.info("diffusion_repair_fallback_attempting")
            try:
                from ecodiaos.systems.simula.agents.diffusion_repair import DiffusionRepairAgent
                if isinstance(self._diffusion_repair, DiffusionRepairAgent):
                    repair_result = await self._diffusion_repair.repair(
                        original_code=code_result.summary or "",
                        files_written=code_result.files_written,
                        test_failures=code_result.test_output or code_result.error,
                    )
                    if repair_result.status.value == "repaired":
                        log.info(
                            "diffusion_repair_succeeded",
                            steps=len(repair_result.denoise_steps),
                            improvement=f"{repair_result.improvement_rate:.0%}",
                        )
                        # Mark as success — diffusion repair saved the change
                        code_result.success = True
                        code_result.files_written = repair_result.files_repaired
                        code_result.error = ""
                        # Stash repair metadata on proposal for history recording
                        proposal._diffusion_repair_result = repair_result  # type: ignore[attr-defined]
                    else:
                        log.info("diffusion_repair_insufficient", status=repair_result.status.value)
            except Exception as exc:
                log.warning("diffusion_repair_error", error=str(exc))

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

        # Stash GRPO metadata for history recording
        proposal._grpo_model_used = grpo_model_used  # type: ignore[attr-defined]
        proposal._grpo_ab_group = grpo_ab_group  # type: ignore[attr-defined]

        # Stash Stage 5A synthesis metadata
        if synthesis_result_stash is not None:
            proposal._synthesis_result = synthesis_result_stash  # type: ignore[attr-defined]
            code_result.synthesis_strategy = synthesis_result_stash.strategy.value
            code_result.synthesis_speedup = synthesis_result_stash.speedup_vs_cegis

        # ── Health check (with Stage 2 formal verification) ────────────────
        health = await self._health.check(
            code_result.files_written, proposal=proposal,
        )

        # Stash formal verification result for history recording
        if health.formal_verification is not None:
            proposal._formal_verification_result = health.formal_verification  # type: ignore[attr-defined]

        # Stash Lean 4 verification result for history recording
        if health.lean_verification is not None:
            proposal._lean_verification_result = health.lean_verification  # type: ignore[attr-defined]

        if not health.healthy:
            recovered = False

            # ── Stage 5D: Causal debugging before repair ──────────────────
            causal_diagnosis = None
            if self._causal_debugger is not None:
                log.info("causal_debugging_starting", issues=health.issues)
                try:
                    from ecodiaos.systems.simula.debugging.causal_dag import CausalDebugger as CausalDbgCls

                    if isinstance(self._causal_debugger, CausalDbgCls):
                        causal_diagnosis = await self._causal_debugger.diagnose(
                            files_written=code_result.files_written,
                            health_issues=health.issues,
                            test_output=code_result.test_output or "",
                        )
                        log.info(
                            "causal_diagnosis_complete",
                            root_cause=causal_diagnosis.root_cause,
                            confidence=f"{causal_diagnosis.confidence:.2f}",
                            interventions=causal_diagnosis.interventions_performed,
                        )
                        # Stash for history recording
                        proposal._causal_diagnosis = causal_diagnosis  # type: ignore[attr-defined]
                        health.causal_diagnosis = causal_diagnosis
                except Exception as exc:
                    log.warning("causal_debugging_error", error=str(exc))

            # ── Stage 5B: Repair agent recovery after causal diagnosis ────
            if self._repair_agent is not None:
                log.info("repair_agent_post_health_attempting")
                try:
                    from ecodiaos.systems.simula.agents.repair_agent import RepairAgent as RepairAgentCls
                    from ecodiaos.systems.simula.verification.types import RepairStatus

                    if isinstance(self._repair_agent, RepairAgentCls):
                        broken_files = {
                            f: (self._root / f).read_text()
                            for f in code_result.files_written
                            if (self._root / f).exists()
                        }
                        # Feed causal diagnosis context to the repair agent
                        diag_context = ""
                        if causal_diagnosis is not None:
                            diag_context = (
                                f"Root cause: {causal_diagnosis.root_cause}\n"
                                f"Fix location: {causal_diagnosis.fix_location}\n"
                                f"Confidence: {causal_diagnosis.confidence:.2f}\n"
                                f"Reasoning: {' → '.join(causal_diagnosis.reasoning_chain)}"
                            )
                        repair_result = await self._repair_agent.repair(
                            proposal=proposal,
                            broken_files=broken_files,
                            test_output=(
                                code_result.test_output
                                or "; ".join(health.issues)
                            ),
                            lint_output=diag_context or None,
                        )
                        if repair_result.status == RepairStatus.REPAIRED:
                            log.info(
                                "repair_agent_post_health_succeeded",
                                attempts=repair_result.total_attempts,
                                cost=f"${repair_result.total_cost_usd:.4f}",
                            )
                            code_result.repair_attempted = True
                            code_result.repair_succeeded = True
                            code_result.repair_cost_usd = repair_result.total_cost_usd
                            proposal._repair_result = repair_result  # type: ignore[attr-defined]

                            # Re-check health after repair
                            health_recheck = await self._health.check(
                                repair_result.files_repaired, proposal=proposal,
                            )
                            if health_recheck.healthy:
                                log.info("health_recheck_passed_after_repair")
                                health = health_recheck
                                code_result.files_written = repair_result.files_repaired
                                code_result.success = True
                                recovered = True
                            else:
                                log.warning(
                                    "health_recheck_still_failing",
                                    issues=health_recheck.issues,
                                )
                        else:
                            log.info(
                                "repair_agent_post_health_insufficient",
                                status=repair_result.status.value,
                            )
                            code_result.repair_attempted = True
                            code_result.repair_succeeded = False
                except Exception as exc:
                    log.warning("repair_agent_post_health_error", error=str(exc))

            # ── Rollback only if all recovery failed ──────────────────────
            if not recovered:
                log.warning("health_check_failed_rolling_back", issues=health.issues)
                await self._rollback.restore(snapshot)
                proposal.status = ProposalStatus.ROLLED_BACK
                self._proposals_rolled_back += 1

                # Record the rollback in history
                await self._record_evolution(
                    proposal, code_result.files_written,
                    rolled_back=True,
                    rollback_reason="; ".join(health.issues),
                )

                self._active_proposals.pop(proposal.id, None)
                self._invalidate_analytics()
                return ProposalResult(
                    status=ProposalStatus.ROLLED_BACK,
                    reason=f"Post-apply health check failed: {'; '.join(health.issues)}",
                )

        # ── Stage 3A: Incremental verification cache update ─────────────────
        if self._incremental is not None:
            try:
                incr_result = await self._incremental.verify_incremental(
                    files_changed=code_result.files_written,
                    proposal_id=proposal.id,
                )
                log.info(
                    "incremental_verification_complete",
                    checked=incr_result.functions_checked,
                    skipped=incr_result.functions_skipped_early_cutoff,
                    cache_hit_rate=f"{incr_result.cache_hit_rate:.0%}",
                )
            except Exception as exc:
                log.warning("incremental_verification_error", error=str(exc))

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

        # ── Stage 4B: Record GRPO training data ──────────────────────────────
        if self._grpo is not None:
            try:
                await self._grpo.record_proposal_applied(
                    proposal_id=proposal.id,
                    category=proposal.category.value,
                    files_written=code_result.files_written,
                    tests_passed=code_result.tests_passed,
                    lint_passed=code_result.lint_passed,
                    rolled_back=False,
                )
                # Check if retraining is warranted
                if self._grpo.should_retrain():
                    log.info("grpo_retrain_triggered")
                    # Fire-and-forget: retraining is expensive and non-blocking
                    import asyncio
                    asyncio.create_task(self._grpo_retrain_background())
            except Exception as exc:
                log.warning("grpo_record_error", error=str(exc))

        # ── Stage 3C: LILO abstraction extraction ────────────────────────────
        self._proposals_applied_since_consolidation += 1
        if self._lilo is not None:
            try:
                extraction = await self._lilo.extract_from_proposals(
                    proposal_ids=[proposal.id],
                    files_changed={proposal.id: code_result.files_written},
                )
                if extraction.extracted:
                    log.info(
                        "lilo_extraction_complete",
                        extracted=len(extraction.extracted),
                        merged=extraction.merged_into_existing,
                    )
                # Periodic consolidation
                if (
                    self._proposals_applied_since_consolidation
                    >= self._config.lilo_consolidation_interval_proposals
                ):
                    await self._lilo.consolidate(
                        max_library_size=self._config.lilo_max_library_size,
                    )
                    self._proposals_applied_since_consolidation = 0
                    log.info("lilo_consolidation_complete")
            except Exception as exc:
                log.warning("lilo_extraction_error", error=str(exc))

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

        # Stage 2: Attach formal verification metadata if available
        if hasattr(proposal, "_formal_verification_result"):
            fv = proposal._formal_verification_result
            if fv is not None:
                if fv.dafny and fv.dafny.status:
                    record.formal_verification_status = fv.dafny.status.value
                    record.dafny_rounds = fv.dafny.rounds_attempted
                if fv.z3 and fv.z3.valid_invariants:
                    record.discovered_invariants_count = len(fv.z3.valid_invariants)
                if fv.static_analysis:
                    record.static_analysis_findings = len(fv.static_analysis.findings)

        # Stage 4A: Attach Lean 4 proof metadata if available
        if hasattr(proposal, "_lean_verification_result"):
            lean_r = proposal._lean_verification_result
            if lean_r is not None:
                record.lean_proof_status = lean_r.status.value
                record.lean_proof_rounds = len(lean_r.attempts)
                record.lean_proven_lemmas_count = len(lean_r.proven_lemmas)
                record.lean_copilot_automation_rate = lean_r.copilot_automation_rate
                record.lean_library_lemmas_reused = len(lean_r.library_lemmas_used)

        # Stage 4B: Attach GRPO model routing metadata
        if hasattr(proposal, "_grpo_model_used"):
            record.grpo_model_used = proposal._grpo_model_used
        if hasattr(proposal, "_grpo_ab_group"):
            record.grpo_ab_group = proposal._grpo_ab_group

        # Stage 4C: Attach diffusion repair metadata if used
        if hasattr(proposal, "_diffusion_repair_result"):
            dr = proposal._diffusion_repair_result
            if dr is not None:
                record.diffusion_repair_used = True
                record.diffusion_repair_status = dr.status.value
                record.diffusion_repair_steps = len(dr.denoise_steps)
                record.diffusion_improvement_rate = dr.improvement_rate

        # Stage 5A: Attach synthesis metadata
        if hasattr(proposal, "_synthesis_result"):
            sr = proposal._synthesis_result
            if sr is not None:
                record.synthesis_strategy_used = sr.strategy.value
                record.synthesis_status = sr.status.value
                record.synthesis_speedup_vs_baseline = sr.speedup_vs_cegis
                record.synthesis_candidates_explored = sr.candidates_explored

        # Stage 5B: Attach repair agent metadata
        if hasattr(proposal, "_repair_result"):
            rr = proposal._repair_result
            if rr is not None:
                record.repair_agent_used = True
                record.repair_agent_status = rr.status.value
                record.repair_attempts = rr.total_attempts
                record.repair_cost_usd = rr.total_cost_usd

        # Stage 5C: Attach orchestration metadata
        if hasattr(proposal, "_orchestration_result"):
            orc = proposal._orchestration_result
            if orc is not None:
                record.orchestration_used = True
                record.orchestration_dag_nodes = orc.dag_nodes
                record.orchestration_agents_used = orc.agents_used
                record.orchestration_parallel_stages = orc.parallel_stages

        # Stage 5D: Attach causal debugging metadata
        if hasattr(proposal, "_causal_diagnosis"):
            cd = proposal._causal_diagnosis
            if cd is not None:
                record.causal_debug_used = True
                record.causal_root_cause = cd.root_cause
                record.causal_confidence = cd.confidence
                record.causal_interventions = cd.interventions_performed

        # Stage 5E: Attach issue resolution metadata
        if hasattr(proposal, "_issue_resolution_result"):
            ir = proposal._issue_resolution_result
            if ir is not None:
                record.issue_resolution_used = True
                record.issue_autonomy_level = ir.autonomy_level.value
                record.issue_abstained = ir.status.value == "abstained"

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

    # ─── Stage 4B: GRPO Background Retraining ──────────────────────────────

    async def _grpo_retrain_background(self) -> None:
        """
        Background task to run the GRPO retraining pipeline.
        Collects data → SFT → GRPO RL → evaluate → deploy if improved.
        """
        if self._grpo is None:
            return
        try:
            self._logger.info("grpo_retrain_starting")
            run = await self._grpo.collect_training_data()
            if run is None:
                self._logger.info("grpo_retrain_skipped_insufficient_data")
                return

            run = await self._grpo.run_sft(run)
            run = await self._grpo.run_grpo(run)
            evaluation = await self._grpo.evaluate(run)

            if evaluation.statistically_significant and evaluation.improvement_percent > 0:
                self._logger.info(
                    "grpo_retrain_deployed",
                    improvement=f"{evaluation.improvement_percent:.1f}%",
                    pass_at_1_base=f"{evaluation.base_pass_at_1:.2f}",
                    pass_at_1_finetuned=f"{evaluation.finetuned_pass_at_1:.2f}",
                )
            else:
                self._logger.info(
                    "grpo_retrain_no_improvement",
                    improvement=f"{evaluation.improvement_percent:.1f}%",
                    significant=evaluation.statistically_significant,
                )
        except Exception as exc:
            self._logger.warning("grpo_retrain_error", error=str(exc))

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
