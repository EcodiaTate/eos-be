"""
EcodiaOS — Hunter Service Orchestrator (Phase 7)

Coordinates the full hunting pipeline from target to vulnerability reports:

  1. INGEST   — clone external repo or point at internal EOS workspace
  2. MAP      — discover exploitable attack surfaces via AST/regex scanning
  3. PROVE    — encode attacker goals as Z3 constraints, check satisfiability
  4. EXPLOIT  — translate Z3 SAT counterexamples into proof-of-concept scripts
  5. PATCH    — optionally generate + verify patches via RepairAgent
  6. REPORT   — aggregate findings into a HuntResult with full analytics

Two entry points:
  hunt_external_repo(github_url) — clone + full pipeline
  hunt_internal_eos()            — scan the EOS codebase itself

Iron Rules (non-negotiable):
  - Hunter NEVER writes to EOS source files (only temp workspace files)
  - PoC execution is scoped to authorized_targets only
  - All vulnerabilities logged via structlog with event="vulnerability_discovered"
  - Z3 SAT counterexamples validated before PoC generation
  - Internal Simula self-evolution must remain fully functional
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.primitives.common import new_id, utc_now
from ecodiaos.systems.simula.hunter.analytics import HunterAnalyticsView
from ecodiaos.systems.simula.hunter.ingestor import TargetIngestor
from ecodiaos.systems.simula.hunter.safety import HunterSafetyGates
from ecodiaos.systems.simula.hunter.types import (
    AttackSurface,
    HunterConfig,
    HuntResult,
    RemediationStatus,
    TargetType,
    VulnerabilityReport,
    VulnerabilitySeverity,
)
from ecodiaos.systems.simula.hunter.workspace import TargetWorkspace

if TYPE_CHECKING:
    from ecodiaos.systems.simula.hunter.analytics import HunterAnalyticsEmitter
    from ecodiaos.systems.simula.hunter.prover import VulnerabilityProver
    from ecodiaos.systems.simula.hunter.remediation import HunterRepairOrchestrator

logger = structlog.get_logger().bind(system="simula.hunter.service")


# ── Predefined Attack Goals ──────────────────────────────────────────────────
# Focused on realistic attack vectors, not fuzzing noise.

PREDEFINED_ATTACK_GOALS: list[str] = [
    "Unauthenticated access to protected resource",
    "SQL injection in user input",
    "Broken access control: user A can access user B's data",
    "Privilege escalation: regular user can call admin function",
    "Reentrancy: contract can call itself recursively",
    "Unvalidated redirect",
    "Race condition: concurrent access violates invariant",
]

# Modules that a PoC script must never import — these allow host-level
# side effects beyond HTTP requests.
_FORBIDDEN_POC_MODULES = frozenset({
    "subprocess", "socket", "ctypes", "pickle", "shelve",
    "marshal", "shutil", "tempfile", "multiprocessing",
})


class HunterService:
    """
    Coordinates the full hunting pipeline from target to vulnerability reports.

    Integrates:
      - TargetIngestor     (Phase 3) — clone repos, map attack surfaces
      - VulnerabilityProver (Phase 4+5) — Z3 constraint inversion + PoC gen
      - HunterRepairOrchestrator (Phase 6) — autonomous patch generation
      - HunterAnalyticsEmitter (Phase 9) — structlog event instrumentation
      - HunterAnalyticsView (Phase 9) — aggregate vulnerability statistics
    """

    def __init__(
        self,
        prover: VulnerabilityProver,
        config: HunterConfig,
        *,
        eos_root: Path | None = None,
        analytics: HunterAnalyticsEmitter | None = None,
        remediation: HunterRepairOrchestrator | None = None,
    ) -> None:
        """
        Args:
            prover: The Z3-backed vulnerability prover.
            config: Hunter authorization and resource configuration.
            eos_root: Path to the internal EOS codebase (for internal hunts).
            analytics: Optional analytics emitter for structured event logging.
            remediation: Optional remediation orchestrator for patch generation.
        """
        self._prover = prover
        self._config = config
        self._eos_root = eos_root
        self._analytics = analytics
        self._remediation = remediation
        self._safety = HunterSafetyGates()
        self._log = logger.bind(
            max_workers=config.max_workers,
            authorized_targets=len(config.authorized_targets),
        )

        # Pre-validate config via safety gates
        config_check = self._safety.validate_hunter_config(config)
        if not config_check:
            self._log.warning(
                "hunter_config_safety_warning",
                reason=config_check.reason,
            )

        # Aggregate analytics view — ingests every HuntResult automatically
        self._analytics_view = HunterAnalyticsView()

        # Metrics
        self._hunts_completed: int = 0
        self._total_surfaces_mapped: int = 0
        self._total_vulnerabilities_found: int = 0
        self._total_patches_generated: int = 0

        # Hunt history (in-memory, capped)
        self._hunt_history: list[HuntResult] = []
        self._max_history: int = 50

        # Vulnerability templates loaded once at construction
        self._templates: list[dict[str, Any]] = self._load_templates()

    # ── Public API ────────────────────────────────────────────────────────────

    async def hunt_external_repo(
        self,
        github_url: str,
        *,
        attack_goals: list[str] | None = None,
        generate_pocs: bool = True,
        generate_patches: bool = False,
    ) -> HuntResult:
        """
        Clone an external GitHub repository and run the full hunting pipeline.

        Pipeline:
          1. Clone repo → TargetWorkspace
          2. Map attack surfaces via AST/regex scanning
          3. For each surface × attack goal, run VulnerabilityProver
          4. Optionally generate PoCs and patches

        Args:
            github_url: HTTPS URL of the repository to hunt.
            attack_goals: Custom attack goals (defaults to PREDEFINED_ATTACK_GOALS).
            generate_pocs: Whether to generate proof-of-concept exploit scripts.
            generate_patches: Whether to generate patches for found vulnerabilities.

        Returns:
            HuntResult with all discovered vulnerabilities and optional patches.
        """
        # Step A: Authorization gate — target must be in authorized_targets
        if not any(
            github_url.startswith(t) or t in github_url
            for t in self._config.authorized_targets
        ):
            self._log.error(
                "hunt_target_not_authorized",
                url=github_url,
                authorized_targets=self._config.authorized_targets,
            )
            start = time.monotonic()
            return self._build_empty_result(
                new_id(), github_url, TargetType.EXTERNAL_REPO,
                start, utc_now(),
            )

        goals = attack_goals or PREDEFINED_ATTACK_GOALS
        start = time.monotonic()
        started_at = utc_now()
        hunt_id = new_id()

        log = self._log.bind(
            hunt_id=hunt_id,
            target_url=github_url,
            target_type="external_repo",
            attack_goals=len(goals),
        )

        if self._analytics:
            self._analytics.emit_hunt_started(
                github_url, "external_repo", hunt_id=hunt_id,
            )

        log.info("hunt_started", url=github_url)

        # Step 1: Clone and ingest — workspace is an async context manager;
        # the temp directory is nuked from orbit on exit regardless of outcome.
        try:
            ingestor = await TargetIngestor.ingest_from_github(
                github_url, clone_depth=self._config.clone_depth,
            )
        except Exception as exc:
            log.error("hunt_clone_failed", error=str(exc))
            if self._analytics:
                self._analytics.emit_hunt_error(
                    target_url=github_url,
                    hunt_id=hunt_id,
                    pipeline_stage="ingest",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            return self._build_empty_result(
                hunt_id, github_url, TargetType.EXTERNAL_REPO,
                start, started_at,
            )

        async with ingestor.workspace as workspace:
            # Safety gate: validate workspace isolation before proceeding
            ws_check = self._safety.validate_workspace_isolation(
                workspace, eos_root=self._eos_root,
            )
            if not ws_check:
                log.error("safety_gate_workspace_failed", reason=ws_check.reason)
                return self._build_empty_result(
                    hunt_id, github_url, TargetType.EXTERNAL_REPO,
                    start, started_at,
                )

            return await self._run_hunt_pipeline(
                hunt_id=hunt_id,
                ingestor=ingestor,
                workspace=workspace,
                target_url=github_url,
                target_type=TargetType.EXTERNAL_REPO,
                goals=goals,
                generate_pocs=generate_pocs,
                generate_patches=generate_patches,
                start=start,
                started_at=started_at,
                log=log,
            )

    async def hunt_internal_eos(
        self,
        *,
        attack_goals: list[str] | None = None,
        generate_pocs: bool = False,
        generate_patches: bool = False,
    ) -> HuntResult:
        """
        Run the hunting pipeline against the internal EOS codebase.

        Useful for continuous automated security testing of EOS itself.
        The workspace is read-only — no temp files are created.

        Args:
            attack_goals: Custom attack goals (defaults to PREDEFINED_ATTACK_GOALS).
            generate_pocs: Whether to generate proof-of-concept exploit scripts.
            generate_patches: Whether to generate patches for found vulnerabilities.

        Returns:
            HuntResult with discovered vulnerabilities.

        Raises:
            RuntimeError: If eos_root was not provided at construction time.
        """
        if self._eos_root is None:
            raise RuntimeError(
                "Cannot hunt internal EOS: eos_root was not provided. "
                "Pass eos_root= to HunterService constructor."
            )

        goals = attack_goals or PREDEFINED_ATTACK_GOALS
        start = time.monotonic()
        started_at = utc_now()
        hunt_id = new_id()

        log = self._log.bind(
            hunt_id=hunt_id,
            target_url="internal_eos",
            target_type="internal_eos",
            attack_goals=len(goals),
        )

        if self._analytics:
            self._analytics.emit_hunt_started(
                "internal_eos", "internal_eos", hunt_id=hunt_id,
            )

        log.info("hunt_started", target="internal_eos")

        workspace = TargetWorkspace.internal(self._eos_root)
        ingestor = TargetIngestor(workspace=workspace)

        return await self._run_hunt_pipeline(
            hunt_id=hunt_id,
            ingestor=ingestor,
            workspace=workspace,
            target_url="internal_eos",
            target_type=TargetType.INTERNAL_EOS,
            goals=goals,
            generate_pocs=generate_pocs,
            generate_patches=generate_patches,
            start=start,
            started_at=started_at,
            log=log,
        )

    async def generate_patches(
        self,
        hunt_result: HuntResult,
        workspace: TargetWorkspace | None = None,
    ) -> dict[str, str]:
        """
        Generate patches for all vulnerabilities in a completed HuntResult.

        For each vulnerability, calls HunterRepairOrchestrator.generate_patch()
        and returns a mapping of vulnerability_id → patch diff.

        Args:
            hunt_result: A completed HuntResult with vulnerabilities.
            workspace: Target workspace to patch in. Required if the
                       hunt's workspace has already been cleaned up.

        Returns:
            Dict mapping vulnerability ID → patch diff string.

        Raises:
            RuntimeError: If remediation orchestrator is not available.
        """
        if self._remediation is None:
            raise RuntimeError(
                "Remediation is not available. Provide a HunterRepairOrchestrator "
                "when constructing HunterService."
            )

        if not hunt_result.vulnerabilities_found:
            return {}

        log = self._log.bind(
            hunt_id=hunt_result.id,
            vulnerabilities=len(hunt_result.vulnerabilities_found),
        )
        log.info("patch_generation_started")

        # Swap workspace for the remediation run via public API
        if workspace is not None:
            self._remediation.set_workspace(workspace)

        remediation_results = await self._remediation.generate_patches_batch(
            hunt_result.vulnerabilities_found,
        )

        patches: dict[str, str] = {}
        patched_count = 0

        for vuln_id, result in remediation_results.items():
            if result.status == RemediationStatus.PATCHED and result.final_patch_diff:
                patches[vuln_id] = result.final_patch_diff
                patched_count += 1

                if self._analytics:
                    self._analytics.emit_patch_generated(
                        vuln_id=vuln_id,
                        repair_time_ms=result.total_duration_ms,
                        patch_size_bytes=len(result.final_patch_diff.encode("utf-8")),
                        hunt_id=hunt_result.id,
                        target_url=hunt_result.target_url,
                    )

        self._total_patches_generated += patched_count
        log.info(
            "patch_generation_complete",
            total=len(remediation_results),
            patched=patched_count,
            failed=len(remediation_results) - patched_count,
        )

        return patches

    def validate_poc(
        self,
        poc_code: str,
        authorized_target: str | None = None,
    ) -> bool:
        """
        Validate that a proof-of-concept script does not reach unauthorized targets.

        Delegates to HunterSafetyGates.validate_poc_execution() for deep
        validation (syntax, forbidden imports, dangerous calls, URL domain
        authorization) plus a pre-check on the explicit authorized_target.

        Args:
            poc_code: The Python PoC script to validate.
            authorized_target: The target domain the PoC should hit (if any).

        Returns:
            True if the PoC passes safety validation, False otherwise.
        """
        # Check authorized target is in config
        if authorized_target is not None:
            if authorized_target not in self._config.authorized_targets:
                self._log.warning(
                    "poc_unauthorized_target",
                    target=authorized_target,
                    authorized=self._config.authorized_targets,
                )
                return False

        # Delegate full validation to safety gates
        result = self._safety.validate_poc_execution(
            poc_code,
            self._config.authorized_targets,
            sandbox_timeout_seconds=self._config.sandbox_timeout_seconds,
        )
        if not result:
            self._log.warning(
                "poc_safety_gate_failed",
                gate=result.gate,
                reason=result.reason,
            )
        return result.passed

    def get_hunt_history(self, limit: int = 20) -> list[HuntResult]:
        """Return recent hunt results (newest first)."""
        return list(reversed(self._hunt_history[-limit:]))

    @property
    def analytics_view(self) -> HunterAnalyticsView:
        """Aggregate analytics across all completed hunts."""
        return self._analytics_view

    @property
    def stats(self) -> dict[str, Any]:
        """Service-level metrics for observability."""
        result: dict[str, Any] = {
            "hunts_completed": self._hunts_completed,
            "total_surfaces_mapped": self._total_surfaces_mapped,
            "total_vulnerabilities_found": self._total_vulnerabilities_found,
            "total_patches_generated": self._total_patches_generated,
            "hunt_history_size": len(self._hunt_history),
            "config": {
                "max_workers": self._config.max_workers,
                "sandbox_timeout_seconds": self._config.sandbox_timeout_seconds,
                "authorized_targets": len(self._config.authorized_targets),
                "log_analytics": self._config.log_vulnerability_analytics,
                "clone_depth": self._config.clone_depth,
            },
            "remediation_available": self._remediation is not None,
            "analytics_available": self._analytics is not None,
            "analytics_summary": self._analytics_view.summary,
        }
        # Include emitter health metrics when available
        if self._analytics is not None:
            result["emitter_stats"] = self._analytics.stats
        return result

    # ── Core Pipeline ─────────────────────────────────────────────────────────

    async def _run_hunt_pipeline(
        self,
        *,
        hunt_id: str,
        ingestor: TargetIngestor,
        workspace: TargetWorkspace,
        target_url: str,
        target_type: TargetType,
        goals: list[str],
        generate_pocs: bool,
        generate_patches: bool,
        start: float,
        started_at: Any,
        log: Any,
    ) -> HuntResult:
        """
        Execute the full hunt pipeline: map → prove → (poc) → (patch) → report.

        This is the shared core between hunt_external_repo and hunt_internal_eos.
        """
        # Step 2: Map attack surfaces
        log.info("mapping_attack_surfaces")
        surfaces: list[AttackSurface] = []
        try:
            surfaces = await ingestor.map_attack_surfaces()
        except Exception as exc:
            log.error("surface_mapping_failed", error=str(exc))
            if self._analytics:
                self._analytics.emit_surface_mapping_failed(
                    target_url=target_url,
                    hunt_id=hunt_id,
                    error_message=str(exc),
                )

        for surface in surfaces:
            if self._analytics:
                self._analytics.emit_attack_surface_discovered(
                    surface_type=surface.surface_type.value,
                    entry_point=surface.entry_point,
                    file_path=surface.file_path,
                    target_url=target_url,
                    hunt_id=hunt_id,
                    line_number=surface.line_number,
                )

        log.info("surfaces_mapped", total=len(surfaces))

        if not surfaces:
            log.info("no_surfaces_found")
            return self._build_empty_result(
                hunt_id, target_url, target_type, start, started_at,
            )

        # Step 3: Extract context code for surfaces that don't have it
        for surface in surfaces:
            if not surface.context_code:
                try:
                    context = await ingestor.extract_context_code(surface)
                    surface.context_code = context
                except Exception:
                    pass  # best-effort

        # Step D: Augment goals with template-derived attack descriptions
        # for surfaces whose type matches a loaded template.
        template_goals = self._template_goals_for_surfaces(surfaces)
        effective_goals = goals + [g for g in template_goals if g not in goals]

        if template_goals:
            log.info(
                "template_goals_applied",
                template_count=len(self._templates),
                extra_goals=len(effective_goals) - len(goals),
            )

        # Step 4: Prove vulnerabilities across surfaces × goals
        vulnerabilities = await self._prove_all(
            surfaces=surfaces,
            goals=effective_goals,
            target_url=target_url,
            generate_pocs=generate_pocs,
            hunt_id=hunt_id,
            log=log,
        )

        log.info(
            "proving_complete",
            total_vulnerabilities=len(vulnerabilities),
            critical=sum(
                1 for v in vulnerabilities
                if v.severity == VulnerabilitySeverity.CRITICAL
            ),
            high=sum(
                1 for v in vulnerabilities
                if v.severity == VulnerabilitySeverity.HIGH
            ),
        )

        # Step 5: Generate patches if requested and remediation is available
        patches: dict[str, str] = {}
        if generate_patches and vulnerabilities and self._remediation is not None:
            log.info("generating_patches", vulnerabilities=len(vulnerabilities))
            try:
                # Set remediation workspace via public API (not private attribute)
                self._remediation.set_workspace(workspace)

                remediation_results = await self._remediation.generate_patches_batch(
                    vulnerabilities,
                )
                for vuln_id, rem_result in remediation_results.items():
                    if (
                        rem_result.status == RemediationStatus.PATCHED
                        and rem_result.final_patch_diff
                    ):
                        patches[vuln_id] = rem_result.final_patch_diff
                        self._total_patches_generated += 1

                        if self._analytics:
                            self._analytics.emit_patch_generated(
                                vuln_id=vuln_id,
                                repair_time_ms=rem_result.total_duration_ms,
                                patch_size_bytes=len(
                                    rem_result.final_patch_diff.encode("utf-8")
                                ),
                                target_url=target_url,
                                hunt_id=hunt_id,
                            )
            except Exception as exc:
                log.error("patch_generation_failed", error=str(exc))
                if self._analytics:
                    self._analytics.emit_hunt_error(
                        target_url=target_url,
                        hunt_id=hunt_id,
                        pipeline_stage="remediation",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )

        # Step 6: Build result — timestamps captured at correct points
        elapsed_ms = int((time.monotonic() - start) * 1000)

        result = HuntResult(
            id=hunt_id,
            target_url=target_url,
            target_type=target_type,
            surfaces_mapped=len(surfaces),
            attack_surfaces=surfaces,
            vulnerabilities_found=vulnerabilities,
            generated_patches=patches,
            total_duration_ms=elapsed_ms,
            started_at=started_at,
            completed_at=utc_now(),
        )

        # Update metrics
        self._hunts_completed += 1
        self._total_surfaces_mapped += len(surfaces)
        self._total_vulnerabilities_found += len(vulnerabilities)

        # Store in history (capped at _max_history)
        self._hunt_history.append(result)
        if len(self._hunt_history) > self._max_history:
            self._hunt_history = self._hunt_history[-self._max_history:]

        # Ingest into aggregate analytics view
        self._analytics_view.ingest_hunt_result(result)

        # Compute severity counts for analytics
        critical = sum(
            1 for v in vulnerabilities
            if v.severity == VulnerabilitySeverity.CRITICAL
        )
        high = sum(
            1 for v in vulnerabilities
            if v.severity == VulnerabilitySeverity.HIGH
        )

        # Emit analytics + flush buffer
        if self._analytics:
            self._analytics.emit_hunt_completed(
                target_url=target_url,
                hunt_id=hunt_id,
                total_surfaces=len(surfaces),
                total_vulnerabilities=len(vulnerabilities),
                total_time_ms=elapsed_ms,
                total_pocs=sum(1 for v in vulnerabilities if v.proof_of_concept_code),
                total_patches=len(patches),
                critical_count=critical,
                high_count=high,
            )
            # Flush any remaining buffered events to TSDB
            await self._analytics.flush()

        # Log each vulnerability as a distinct event (Iron Rule #4)
        for vuln in vulnerabilities:
            log.info(
                "vulnerability_discovered",
                vulnerability_id=vuln.id,
                vulnerability_class=vuln.vulnerability_class.value,
                severity=vuln.severity.value,
                attack_surface=vuln.attack_surface.entry_point,
                file_path=vuln.attack_surface.file_path,
                z3_counterexample=vuln.z3_counterexample[:200],
                has_poc=bool(vuln.proof_of_concept_code),
                has_patch=vuln.id in patches,
            )

        log.info(
            "hunt_completed",
            hunt_id=hunt_id,
            surfaces=len(surfaces),
            vulnerabilities=len(vulnerabilities),
            patches=len(patches),
            duration_ms=elapsed_ms,
        )

        return result

    async def _prove_all(
        self,
        *,
        surfaces: list[AttackSurface],
        goals: list[str],
        target_url: str,
        generate_pocs: bool,
        hunt_id: str = "",
        log: Any,
    ) -> list[VulnerabilityReport]:
        """
        Prove vulnerabilities across all surfaces × attack goals.

        Uses bounded concurrency (config.max_workers) to limit parallel
        Z3 + LLM calls. Each (surface, goal) pair is independently provable.
        Individual proof attempts are wrapped in asyncio.wait_for to enforce
        the configured sandbox_timeout_seconds.
        """
        vulnerabilities: list[VulnerabilityReport] = []
        semaphore = asyncio.Semaphore(self._config.max_workers)
        timeout = self._config.sandbox_timeout_seconds

        async def prove_one(
            surface: AttackSurface,
            goal: str,
        ) -> VulnerabilityReport | None:
            async with semaphore:
                try:
                    report = await asyncio.wait_for(
                        self._prover.prove_vulnerability(
                            surface=surface,
                            attack_goal=goal,
                            target_url=target_url,
                            generate_poc=generate_pocs,
                            config=self._config,
                        ),
                        timeout=timeout,
                    )
                    if report is not None:
                        if self._analytics:
                            self._analytics.emit_vulnerability_proved(
                                vulnerability_class=report.vulnerability_class.value,
                                severity=report.severity.value,
                                z3_time_ms=0,  # prover tracks internally
                                target_url=target_url,
                                hunt_id=hunt_id,
                                vuln_id=report.id,
                                attack_goal=goal,
                                entry_point=surface.entry_point,
                            )
                        # Emit PoC analytics if one was generated
                        if report.proof_of_concept_code and self._analytics:
                            self._analytics.emit_poc_generated(
                                vuln_id=report.id,
                                poc_size_bytes=len(
                                    report.proof_of_concept_code.encode("utf-8")
                                ),
                                target_url=target_url,
                                hunt_id=hunt_id,
                            )
                    return report
                except TimeoutError:
                    log.warning(
                        "prove_vulnerability_timeout",
                        surface=surface.entry_point,
                        goal=goal[:80],
                        timeout_s=timeout,
                    )
                    if self._analytics:
                        self._analytics.emit_proof_timeout(
                            target_url=target_url,
                            hunt_id=hunt_id,
                            entry_point=surface.entry_point,
                            attack_goal=goal,
                            timeout_s=timeout,
                        )
                    return None
                except Exception as exc:
                    log.warning(
                        "prove_vulnerability_error",
                        surface=surface.entry_point,
                        goal=goal[:80],
                        error=str(exc),
                    )
                    if self._analytics:
                        self._analytics.emit_hunt_error(
                            target_url=target_url,
                            hunt_id=hunt_id,
                            pipeline_stage="prove",
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                    return None

        # Build task matrix: surface × goal
        tasks = [
            prove_one(surface, goal)
            for surface in surfaces
            for goal in goals
        ]

        log.info(
            "proving_started",
            total_tasks=len(tasks),
            surfaces=len(surfaces),
            goals=len(goals),
            max_workers=self._config.max_workers,
            timeout_s=timeout,
        )

        # Execute with bounded concurrency; exceptions already handled
        # inside prove_one, so return_exceptions=False is safe here.
        results = await asyncio.gather(*tasks)

        for result in results:
            if isinstance(result, VulnerabilityReport):
                vulnerabilities.append(result)

        return vulnerabilities

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_templates() -> list[dict[str, Any]]:
        """
        Load vulnerability templates from the templates/ directory.

        Each JSON file describes a framework-specific vulnerability pattern
        with Z3 encoding instructions and reproduction guidance. Templates
        are loaded once at construction and cached on the instance.

        Returns:
            List of parsed template dicts. Empty list if the directory
            does not exist or no JSON files are found.
        """
        templates_dir = Path(__file__).parent / "templates"
        if not templates_dir.is_dir():
            return []

        loaded: list[dict[str, Any]] = []
        for path in sorted(templates_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    loaded.append(data)
            except (json.JSONDecodeError, OSError):
                pass  # skip malformed/unreadable templates
        return loaded

    def _template_goals_for_surfaces(
        self,
        surfaces: list[AttackSurface],
    ) -> list[str]:
        """
        Derive additional attack goals from loaded templates for the given surfaces.

        A template contributes a goal when at least one surface's type appears
        in the template's ``target_surface_types`` list.  The goal string is
        built from the template's ``description`` field so the Prover LLM
        receives human-readable intent rather than raw JSON.

        Returns:
            Deduplicated list of extra goal strings derived from templates.
        """
        surface_types = {s.surface_type.value for s in surfaces}
        extra: list[str] = []
        seen: set[str] = set()

        for tmpl in self._templates:
            target_types: list[str] = tmpl.get("target_surface_types", [])
            if not any(t in surface_types for t in target_types):
                continue

            description: str = tmpl.get("description", "").strip()
            if not description or description in seen:
                continue

            # Attach Z3 encoding hints inline so the Prover LLM has full
            # context without needing to re-load the template itself.
            instructions: list[str] = tmpl.get("z3_encoding_instructions", [])
            if instructions:
                hint = " | ".join(instructions)
                goal = f"{description} [Z3 hints: {hint}]"
            else:
                goal = description

            seen.add(description)
            extra.append(goal)

        return extra

    @staticmethod
    def _build_empty_result(
        hunt_id: str,
        target_url: str,
        target_type: TargetType,
        start: float,
        started_at: Any,
    ) -> HuntResult:
        """Build an empty HuntResult (clone failed or no surfaces found)."""
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return HuntResult(
            id=hunt_id,
            target_url=target_url,
            target_type=target_type,
            total_duration_ms=elapsed_ms,
            started_at=started_at,
            completed_at=utc_now(),
        )
