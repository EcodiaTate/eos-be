"""
EcodiaOS — Evo Consolidation Orchestrator

The "sleep mode" of the learning system. Runs every 6 hours or 10,000
cognitive cycles — whichever comes first.

Eight phases (spec Section VII):
  1. Memory consolidation   — delegate to MemoryService
  2. Hypothesis review      — integrate supported, archive refuted/stale
  3. Schema induction       — propose new entity/relation types from clusters
  4. Procedure extraction   — codify mature action sequences as Procedures
  5. Parameter optimisation — apply supported parameter hypotheses
  6. Self-model update      — recompute capability and effectiveness metrics
  7. Drift data feed        — send effectiveness data to Equor's drift detector
  8. Evolution proposals    — flag structural changes that warrant Simula review

Performance budget: ≤60 seconds end-to-end (spec Section X).

Guard rails:
  - Velocity limits enforced by ParameterTuner
  - Evo cannot touch Equor evaluation logic (EVO_CONSTRAINTS)
  - Evolution proposals are submitted to Simula, not applied directly
"""

from __future__ import annotations

import time
from datetime import timedelta
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.primitives.common import utc_now
from ecodiaos.systems.evo.hypothesis import HypothesisEngine
from ecodiaos.systems.evo.parameter_tuner import ParameterTuner
from ecodiaos.systems.evo.procedure_extractor import ProcedureExtractor
from ecodiaos.systems.evo.self_model import SelfModelManager
from ecodiaos.systems.evo.types import (
    ConsolidationResult,
    EvolutionProposal,
    Hypothesis,
    HypothesisCategory,
    HypothesisStatus,
    MutationType,
    PatternContext,
    SchemaInduction,
    VELOCITY_LIMITS,
)

if TYPE_CHECKING:
    from ecodiaos.systems.memory.service import MemoryService

logger = structlog.get_logger()

_CONSOLIDATION_INTERVAL_HOURS: int = 6
_CONSOLIDATION_CYCLE_THRESHOLD: int = 10_000


class ConsolidationOrchestrator:
    """
    Drives the full consolidation pipeline — the organism dreaming.

    Receives the active hypothesis list and pattern context from EvoService.
    Coordinates all sub-systems through the 8-phase pipeline.
    """

    def __init__(
        self,
        hypothesis_engine: HypothesisEngine,
        parameter_tuner: ParameterTuner,
        procedure_extractor: ProcedureExtractor,
        self_model_manager: SelfModelManager,
        memory: MemoryService | None = None,
        simula_callback: Callable[..., Any] | None = None,
    ) -> None:
        self._hypotheses = hypothesis_engine
        self._tuner = parameter_tuner
        self._extractor = procedure_extractor
        self._self_model = self_model_manager
        self._memory = memory
        self._simula_callback = simula_callback
        self._logger = logger.bind(system="evo.consolidation")

        self._last_run_at = utc_now() - timedelta(hours=_CONSOLIDATION_INTERVAL_HOURS)
        self._total_runs: int = 0

    def should_run(self, cycle_count: int, cycles_since_last: int) -> bool:
        """
        Return True if consolidation is due.
        Triggers on:
          - 6 hours elapsed since last run
          - 10,000 cognitive cycles since last run
        """
        hours_elapsed = (utc_now() - self._last_run_at).total_seconds() / 3600
        if hours_elapsed >= _CONSOLIDATION_INTERVAL_HOURS:
            return True
        if cycles_since_last >= _CONSOLIDATION_CYCLE_THRESHOLD:
            return True
        return False

    async def run(self, pattern_context: PatternContext) -> ConsolidationResult:
        """
        Execute the full 8-phase consolidation pipeline.
        Returns a ConsolidationResult summary.

        Never raises — all phases handle their own exceptions.
        """
        self._logger.info("consolidation_starting")
        start = time.monotonic()
        result = ConsolidationResult(triggered_at=utc_now())

        # ── Phase 1: Memory Consolidation ────────────────────────────────────
        await self._phase_memory_consolidation()

        # ── Snapshot supported hypotheses BEFORE Phase 2 removes them ─────────
        # Phase 2 integrates/archives supported hypotheses which removes them
        # from the active list. Phases 3 and 5 need these hypotheses, so we
        # snapshot them first.
        self._supported_snapshot = list(self._hypotheses.get_supported())

        # ── Phase 2: Hypothesis Review ────────────────────────────────────────
        integrated, archived = await self._phase_hypothesis_review()
        result.hypotheses_evaluated = integrated + archived
        result.hypotheses_integrated = integrated
        result.hypotheses_archived = archived

        # ── Phase 3: Schema Induction ─────────────────────────────────────────
        schemas_induced = await self._phase_schema_induction()
        result.schemas_induced = schemas_induced

        # ── Phase 4: Procedure Extraction ─────────────────────────────────────
        self._extractor.begin_cycle()
        procedures_extracted = await self._phase_procedure_extraction(pattern_context)
        result.procedures_extracted = procedures_extracted

        # ── Phase 5: Parameter Optimisation ───────────────────────────────────
        self._tuner.begin_cycle()
        adj_count, total_delta = await self._phase_parameter_optimisation()
        result.parameters_adjusted = adj_count
        result.total_parameter_delta = total_delta

        # ── Phase 6: Self-Model Update ────────────────────────────────────────
        await self._phase_self_model_update()
        result.self_model_updated = True

        # ── Phase 7: Drift Data Feed ──────────────────────────────────────────
        await self._phase_drift_feed()

        # ── Phase 8: Evolution Proposals ──────────────────────────────────────
        await self._phase_evolution_proposals()

        # ── Housekeeping ──────────────────────────────────────────────────────
        pattern_context.reset()
        self._last_run_at = utc_now()
        self._total_runs += 1

        result.duration_ms = int((time.monotonic() - start) * 1000)
        self._logger.info(
            "consolidation_complete",
            duration_ms=result.duration_ms,
            hypotheses_integrated=result.hypotheses_integrated,
            hypotheses_archived=result.hypotheses_archived,
            procedures_extracted=result.procedures_extracted,
            parameters_adjusted=result.parameters_adjusted,
            total_parameter_delta=round(result.total_parameter_delta, 4),
        )
        return result

    # ─── Phases ───────────────────────────────────────────────────────────────

    async def _phase_memory_consolidation(self) -> None:
        """Phase 1: Delegate memory consolidation to MemoryService."""
        if self._memory is None:
            return
        try:
            await self._memory.consolidate()
            self._logger.info("memory_consolidation_complete")
        except Exception as exc:
            self._logger.error("memory_consolidation_failed", error=str(exc))

    async def _phase_hypothesis_review(self) -> tuple[int, int]:
        """
        Phase 2: Review all active hypotheses.
          - SUPPORTED → attempt integration (calls HypothesisEngine.integrate_hypothesis)
          - REFUTED   → archive
          - Stale     → archive
        Returns (integrated_count, archived_count).
        """
        integrated = 0
        archived = 0

        all_hypotheses = self._hypotheses.get_all_active()

        for h in all_hypotheses:
            try:
                if h.status == HypothesisStatus.SUPPORTED:
                    success = await self._hypotheses.integrate_hypothesis(h)
                    if success:
                        integrated += 1
                elif h.status == HypothesisStatus.REFUTED:
                    await self._hypotheses.archive_hypothesis(h, reason="refuted")
                    archived += 1
                elif self._hypotheses.is_stale(h):
                    await self._hypotheses.archive_hypothesis(h, reason="stale")
                    archived += 1
            except Exception as exc:
                self._logger.warning(
                    "hypothesis_review_failed",
                    hypothesis_id=h.id,
                    error=str(exc),
                )

        return integrated, archived

    async def _phase_schema_induction(self) -> int:
        """
        Phase 3: Induce new schema elements from supported world-model hypotheses.
        Uses the pre-Phase-2 snapshot so hypotheses are available after integration.
        Returns the count of schemas induced.
        """
        schemas_induced = 0
        supported = getattr(self, "_supported_snapshot", [])

        for h in supported:
            if h.category != HypothesisCategory.WORLD_MODEL:
                continue
            if h.proposed_mutation is None:
                continue
            if h.proposed_mutation.type != MutationType.SCHEMA_ADDITION:
                continue

            schema = SchemaInduction(
                entities=[{"name": h.proposed_mutation.target, "description": h.statement}],
                source_hypothesis=h.id,
            )
            success = await self._apply_schema_induction(schema)
            if success:
                schemas_induced += 1

        return schemas_induced

    async def _phase_procedure_extraction(self, context: PatternContext) -> int:
        """
        Phase 4: Extract procedures from mature action-sequence patterns.
        Returns the count of new procedures extracted.
        """
        mature_patterns = context.get_mature_sequences(
            min_occurrences=VELOCITY_LIMITS["max_new_procedures_per_cycle"]
        )
        # Get_mature_sequences returns patterns >= min, use ≥3 (spec threshold)
        all_patterns = context.get_mature_sequences(min_occurrences=3)

        extracted = 0
        for pattern in all_patterns:
            procedure = await self._extractor.extract_procedure(pattern)
            if procedure is not None:
                extracted += 1

        return extracted

    async def _phase_parameter_optimisation(self) -> tuple[int, float]:
        """
        Phase 5: Apply supported parameter hypotheses.
        Uses the pre-Phase-2 snapshot so hypotheses are available after integration.
        Velocity-limited to prevent lurching changes.
        Returns (adjustment_count, total_absolute_delta).
        """
        supported = getattr(self, "_supported_snapshot", [])
        candidates: list[Any] = []

        for h in supported:
            if h.category != HypothesisCategory.PARAMETER:
                continue
            adj = self._tuner.propose_adjustment(h)
            if adj is not None:
                candidates.append(adj)

        if not candidates:
            return 0, 0.0

        # Check velocity limit for the batch
        allowed, reason = self._tuner.check_velocity_limit(candidates)
        if not allowed:
            self._logger.warning("parameter_velocity_limit", reason=reason)
            # Apply as many as we can without exceeding total limit
            limit = VELOCITY_LIMITS["max_total_parameter_delta_per_cycle"]
            running_delta = 0.0
            filtered = []
            for adj in sorted(candidates, key=lambda a: abs(a.delta), reverse=False):
                if running_delta + abs(adj.delta) <= limit:
                    filtered.append(adj)
                    running_delta += abs(adj.delta)
            candidates = filtered

        applied = 0
        total_delta = 0.0
        for adj in candidates:
            await self._tuner.apply_adjustment(adj)
            applied += 1
            total_delta += abs(adj.delta)

        return applied, total_delta

    async def _phase_self_model_update(self) -> None:
        """Phase 6: Recompute self-model from recent outcome episodes."""
        try:
            await self._self_model.update()
        except Exception as exc:
            self._logger.error("self_model_update_failed", error=str(exc))

    async def _phase_drift_feed(self) -> None:
        """
        Phase 7: Feed effectiveness data to Equor's drift detector.
        Drift detection is handled by Equor; we just update the data it reads.
        The self-model stats written to the Self node are what Equor reads.
        """
        stats = self._self_model.get_current()
        self._logger.debug(
            "drift_data_available",
            success_rate=round(stats.success_rate, 3),
            mean_alignment=round(stats.mean_alignment, 3),
        )
        # Equor reads from Self node directly; no active push needed in Phase 7.
        # Future: could publish a Synapse event for Equor to act on immediately.

    async def _phase_evolution_proposals(self) -> None:
        """
        Phase 8: Submit structural change proposals to Simula for warranted cases.
        Only generates proposals when we have clustered, high-evidence hypotheses
        that point to architectural change. Simula gates the actual change.
        """
        supported = self._hypotheses.get_supported()
        evolution_candidates = [
            h for h in supported
            if (
                h.proposed_mutation is not None
                and h.proposed_mutation.type == MutationType.EVOLUTION_PROPOSAL
                and h.evidence_score > 5.0
            )
        ]
        for h in evolution_candidates:
            if h.proposed_mutation is None:
                continue
            proposal = EvolutionProposal(
                description=h.proposed_mutation.description or h.statement,
                rationale=h.statement,
                supporting_hypotheses=[h.id],
            )
            self._logger.info(
                "evolution_proposal_generated",
                hypothesis_id=h.id,
                description=proposal.description[:80],
            )

            # Submit to Simula via bridge callback
            if self._simula_callback is not None:
                try:
                    result = await self._simula_callback(proposal, [h])
                    self._logger.info(
                        "evolution_proposal_submitted_to_simula",
                        hypothesis_id=h.id,
                        result_status=getattr(result, "status", "unknown"),
                    )
                except Exception as exc:
                    self._logger.error(
                        "simula_submission_failed",
                        hypothesis_id=h.id,
                        error=str(exc),
                    )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    async def _apply_schema_induction(self, schema: SchemaInduction) -> bool:
        """Apply schema induction to the Memory graph."""
        if self._memory is None or not schema.entities:
            return False
        try:
            for entity_spec in schema.entities:
                name = entity_spec.get("name", "")
                description = entity_spec.get("description", "")
                if name:
                    await self._memory._neo4j.execute_write(
                        """
                        MERGE (et:EvoEntityType {name: $name})
                        SET et.description = $description,
                            et.source_hypothesis = $source_hypothesis,
                            et.induced_at = datetime()
                        """,
                        {
                            "name": name,
                            "description": description,
                            "source_hypothesis": schema.source_hypothesis,
                        },
                    )
            return True
        except Exception as exc:
            self._logger.warning("schema_induction_failed", error=str(exc))
            return False

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_runs": self._total_runs,
            "last_run_at": self._last_run_at.isoformat(),
        }
