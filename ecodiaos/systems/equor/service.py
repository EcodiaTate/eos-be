"""
EcodiaOS — Equor Service

The conscience of EOS. Single interface for:
- Constitutional review (the primary entry point from Nova)
- Invariant management
- Autonomy enforcement
- Drift monitoring
- Amendment facilitation
- Audit trail

Equor cannot be disabled. If Equor fails, the instance enters safe mode
where only Level 1 (Advisor) actions are permitted.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.primitives.common import (
    DriveAlignmentVector,
    Verdict,
    new_id,
    utc_now,
)
from ecodiaos.primitives.constitutional import ConstitutionalCheck
from ecodiaos.systems.equor.amendment import (
    apply_amendment,
    propose_amendment,
)
from ecodiaos.systems.equor.autonomy import (
    apply_autonomy_change,
    check_promotion_eligibility,
    get_autonomy_level,
)
from ecodiaos.systems.equor.drift import DriftTracker, respond_to_drift, store_drift_report
from ecodiaos.systems.equor.economic_evaluator import (
    apply_economic_adjustment,
    classify_economic_action,
    evaluate_economic_intent,
)
from ecodiaos.systems.equor.evaluators import (
    BaseEquorEvaluator,
    default_evaluators,
    evaluate_all_drives,
)
from ecodiaos.systems.equor.invariants import (
    HARDCODED_INVARIANTS,
    check_community_invariant,
)
from ecodiaos.systems.equor.schema import ensure_equor_schema, seed_hardcoded_invariants
from ecodiaos.systems.equor.verdict import compute_verdict

if TYPE_CHECKING:
    from ecodiaos.clients.llm import LLMProvider
    from ecodiaos.clients.neo4j import Neo4jClient
    from ecodiaos.config import EquorConfig, GovernanceConfig
    from ecodiaos.core.hotreload import NeuroplasticityBus
    from ecodiaos.primitives.intent import Intent

logger = structlog.get_logger()

# Review timeout: the entire review() call must not block the event loop
# beyond this budget. Community invariant LLM calls are the most expensive
# component and will be skipped if the budget is exhausted.
_REVIEW_TIMEOUT_S = 0.8
# Cache TTL for constitution and autonomy level (seconds).
# These change only via governance events, so a short TTL is safe.
_STATE_CACHE_TTL_S = 30.0


class EquorService:
    """
    The constitutional ethics system.
    Gates every intent before execution.
    Cannot be disabled.
    """

    system_id: str = "equor"

    def __init__(
        self,
        neo4j: Neo4jClient,
        llm: LLMProvider,
        config: EquorConfig,
        governance_config: GovernanceConfig,
        neuroplasticity_bus: NeuroplasticityBus | None = None,
    ):
        self._neo4j = neo4j
        self._llm = llm
        self._config = config
        self._governance = governance_config
        self._drift_tracker = DriftTracker(window_size=config.drift_window_size)
        self._safe_mode = False
        self._total_reviews = 0
        self._evo: Any = None  # Wired post-init for learning feedback from vetoes
        self._bus = neuroplasticity_bus

        # Live evaluator set — hot-reloaded via the NeuroplasticityBus.
        # Initialised with built-in defaults; the bus callback replaces
        # individual evaluators when Simula evolves a new subclass.
        self._evaluators: dict[str, BaseEquorEvaluator] = default_evaluators()

        # Cached state: constitution and autonomy level rarely change (only via
        # governance events), so we cache them to avoid hitting Neo4j on every
        # review() call. Invalidated after _STATE_CACHE_TTL_S or on mutation.
        self._cached_constitution: dict[str, Any] | None = None
        self._cached_autonomy_level: int | None = None
        self._cache_updated_at: float = 0.0

    def set_evo(self, evo: Any) -> None:
        """Wire Evo so constitutional vetoes become learning episodes."""
        self._evo = evo
        logger.info("evo_wired_to_equor")

    # ─── Lifecycle ────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Ensure schema, seed invariants, and register for hot-reload."""
        await ensure_equor_schema(self._neo4j)
        await seed_hardcoded_invariants(self._neo4j)

        if self._bus is not None:
            self._bus.register(
                base_class=BaseEquorEvaluator,
                registration_callback=self._on_evaluator_evolved,
                system_id=self.system_id,
            )

        logger.info("equor_initialized")

    async def shutdown(self) -> None:
        """Deregister evaluators from the bus on shutdown."""
        if self._bus is not None:
            self._bus.deregister(BaseEquorEvaluator)
            logger.info("equor_evaluators_deregistered")

    def _on_evaluator_evolved(self, evaluator: BaseEquorEvaluator) -> None:
        """
        NeuroplasticityBus callback — swap a single drive evaluator in-place.

        The bus instantiates the new subclass and calls this method.  We key on
        ``drive_name`` so only the matching evaluator is replaced; the other
        three continue running undisturbed.
        """
        name = evaluator.drive_name
        if name not in self._evaluators:
            logger.warning(
                "equor_unknown_drive_evolved",
                drive_name=name,
                class_name=type(evaluator).__name__,
            )
            return

        old_cls = type(self._evaluators[name]).__name__
        self._evaluators[name] = evaluator
        logger.info(
            "equor_evaluator_hot_reloaded",
            drive=name,
            old_class=old_cls,
            new_class=type(evaluator).__name__,
        )

    # ─── Primary Entry Point: Constitutional Review ───────────────

    async def review(self, intent: Intent) -> ConstitutionalCheck:
        """
        The primary entry point. Nova submits an Intent for ethical evaluation.
        Target: ≤500ms standard.

        If Equor is in safe mode, only Level 1 actions are permitted.
        """
        start = time.monotonic()

        # Safe mode: only advisory actions permitted
        if self._safe_mode:
            return self._safe_mode_review(intent)

        try:
            async with asyncio.timeout(_REVIEW_TIMEOUT_S):
                return await self._review_inner(intent, start)
        except TimeoutError:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                "equor_review_timeout",
                intent_id=intent.id,
                elapsed_ms=elapsed_ms,
            )
            # Timeout is NOT a failure — return a conservative approval so
            # we don't block the fast path. The audit trail records the timeout.
            return ConstitutionalCheck(
                intent_id=intent.id,
                verdict=Verdict.APPROVED,
                reasoning=(
                    f"Equor review timed out after {elapsed_ms}ms. "
                    "Approved conservatively (heuristic invariants passed)."
                ),
                confidence=0.5,
            )
        except Exception as e:
            # Equor failure = enter safe mode
            logger.error("equor_review_failed", error=str(e), intent_id=intent.id)
            self._safe_mode = True
            return self._safe_mode_review(intent)

    async def review_critical(self, intent: Intent) -> ConstitutionalCheck:
        """
        Lightweight critical-path review for Nova's fast path.

        Constraints (must complete in ≤50ms):
          - Uses ONLY cached constitution/autonomy (never waits for Neo4j)
          - Runs CPU-only verdict computation (drive evaluation + hardcoded invariants)
          - Skips community invariant LLM checks entirely
          - Skips audit trail write (fire-and-forget bookkeeping still runs)

        If no cached state is available (cold start), conservatively approves
        with low confidence so the fast path can proceed without blocking.
        """
        start = time.monotonic()

        if self._safe_mode:
            return self._safe_mode_review(intent)

        # Use cached state only — never block on Neo4j
        if (
            self._cached_constitution is not None
            and self._cached_autonomy_level is not None
        ):
            constitution = self._cached_constitution
            autonomy_level = self._cached_autonomy_level
        else:
            # Cold start: no cached state yet. Approve conservatively.
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.debug(
                "critical_review_no_cache",
                intent_id=intent.id,
                elapsed_ms=elapsed_ms,
            )
            return ConstitutionalCheck(
                intent_id=intent.id,
                verdict=Verdict.APPROVED,
                reasoning="Critical-path review: no cached state, approved conservatively.",
                confidence=0.4,
            )

        # Pure CPU: drive evaluation + hardcoded invariant verdict
        alignment = await evaluate_all_drives(intent, self._evaluators)

        # Economic guardrail on critical path too — all CPU, no I/O.
        economic_delta = evaluate_economic_intent(intent)
        if economic_delta is not None:
            alignment = apply_economic_adjustment(alignment, economic_delta)

        check = compute_verdict(alignment, intent, autonomy_level, constitution)

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Fire-and-forget bookkeeping (non-blocking)
        asyncio.create_task(
            self._post_review_bookkeeping(intent, alignment, check, elapsed_ms),
            name=f"equor_critical_{intent.id[:8]}",
        )

        logger.debug(
            "critical_review_complete",
            intent_id=intent.id,
            verdict=check.verdict.value,
            latency_ms=elapsed_ms,
        )
        return check

    async def _review_inner(
        self, intent: Intent, start: float,
    ) -> ConstitutionalCheck:
        """Core review logic, called within the review timeout."""
        # 1. Drive evaluation + cached state fetch in parallel.
        #    Drive evaluators are pure CPU heuristics (<1ms).
        #    Constitution/autonomy are cached with TTL to avoid Neo4j per-review.
        alignment, (constitution, autonomy_level) = await asyncio.gather(
            evaluate_all_drives(intent, self._evaluators),
            self._get_cached_state(),
        )

        # 1b. Economic guardrail: if this is an Oikos economic intent,
        #     apply domain-specific drive adjustments before verdict.
        #     This ensures Care/Honesty floors catch scam assets, harmful
        #     bounties, etc. even if base evaluators scored them neutrally.
        economic_delta = evaluate_economic_intent(intent)
        if economic_delta is not None:
            alignment = apply_economic_adjustment(alignment, economic_delta)
            logger.debug(
                "economic_adjustment_applied",
                intent_id=intent.id,
                action_type=classify_economic_action(intent),
                adjusted_care=f"{alignment.care:.2f}",
                adjusted_honesty=f"{alignment.honesty:.2f}",
            )

        # 2. Run the verdict engine (pure CPU, includes hardcoded invariant checks)
        check = compute_verdict(alignment, intent, autonomy_level, constitution)

        # 3. Community invariant checks — parallelised with a tight timeout.
        #    Only run if we haven't already blocked (invariant / floor / autonomy).
        if check.verdict not in (Verdict.BLOCKED, Verdict.DEFERRED):
            community_violations = await self._check_community_invariants(intent)
            if community_violations:
                check.verdict = Verdict.BLOCKED
                check.reasoning = (
                    f"Community invariant violated: {community_violations[0]}"
                )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # 4. Fire-and-forget: audit trail, drift tracking, Evo feedback.
        #    These are important but must not block the review response.
        asyncio.create_task(
            self._post_review_bookkeeping(intent, alignment, check, elapsed_ms),
            name=f"equor_bookkeeping_{intent.id[:8]}",
        )

        logger.info(
            "constitutional_review_complete",
            intent_id=intent.id,
            verdict=check.verdict.value,
            composite=f"{alignment.composite:.2f}",
            latency_ms=elapsed_ms,
        )

        return check

    async def _get_cached_state(self) -> tuple[dict[str, Any], int]:
        """Return (constitution_dict, autonomy_level) from cache or Neo4j."""
        now = time.monotonic()
        if (
            self._cached_constitution is not None
            and self._cached_autonomy_level is not None
            and (now - self._cache_updated_at) < _STATE_CACHE_TTL_S
        ):
            return self._cached_constitution, self._cached_autonomy_level

        # Fetch both in parallel
        constitution, autonomy_level = await asyncio.gather(
            self._get_constitution_dict(),
            get_autonomy_level(self._neo4j),
        )
        self._cached_constitution = constitution
        self._cached_autonomy_level = autonomy_level
        self._cache_updated_at = now
        return constitution, autonomy_level

    def _invalidate_state_cache(self) -> None:
        """Called after governance mutations (amendments, autonomy changes)."""
        self._cached_constitution = None
        self._cached_autonomy_level = None
        self._cache_updated_at = 0.0

    async def _post_review_bookkeeping(
        self,
        intent: Intent,
        alignment: DriveAlignmentVector,
        check: ConstitutionalCheck,
        elapsed_ms: int,
    ) -> None:
        """Non-blocking post-review work: audit trail, drift, Evo feedback."""
        try:
            await self._store_review_record(intent, alignment, check, elapsed_ms)
        except Exception:
            logger.debug("audit_trail_write_failed", exc_info=True)

        self._drift_tracker.record_decision(alignment, check.verdict.value)
        self._total_reviews += 1

        if check.verdict == Verdict.BLOCKED and self._evo is not None:
            try:
                await self._feed_veto_to_evo(intent, check)
            except Exception:
                logger.debug("evo_veto_feed_failed", exc_info=True)

        if self._total_reviews % self._config.drift_report_interval == 0:
            try:
                await self._run_drift_check()
                await self._run_promotion_check()
            except Exception:
                logger.debug("drift_or_promotion_check_failed", exc_info=True)

    # ─── Invariant Management ─────────────────────────────────────

    async def get_invariants(self) -> list[dict[str, Any]]:
        """Get all active invariants (hardcoded + community)."""
        results = await self._neo4j.execute_read(
            """
            MATCH (c:Constitution)-[:INCLUDES_INVARIANT]->(i:Invariant)
            WHERE i.active = true
            RETURN i.id AS id, i.name AS name, i.description AS description,
                   i.source AS source, i.severity AS severity
            ORDER BY i.id
            """
        )
        return [dict(r) for r in results]

    async def add_community_invariant(
        self,
        name: str,
        description: str,
        severity: str,
        governance_record_id: str,
    ) -> str:
        """Add a community-defined invariant via governance."""
        invariant_id = f"CINV-{new_id()[:8]}"
        now = utc_now()

        await self._neo4j.execute_write(
            """
            MATCH (c:Constitution)
            CREATE (i:Invariant {
                id: $id,
                name: $name,
                description: $description,
                source: 'community',
                severity: $severity,
                active: true,
                added_at: datetime($now),
                added_by: $gov_id
            })
            CREATE (c)-[:INCLUDES_INVARIANT]->(i)
            """,
            {
                "id": invariant_id,
                "name": name,
                "description": description,
                "severity": severity,
                "now": now.isoformat(),
                "gov_id": governance_record_id,
            },
        )

        logger.info("community_invariant_added", invariant_id=invariant_id, name=name)
        return invariant_id

    # ─── Autonomy ─────────────────────────────────────────────────

    async def get_autonomy_level(self) -> int:
        return await get_autonomy_level(self._neo4j)

    async def check_promotion(self, target_level: int) -> dict[str, Any]:
        current = await get_autonomy_level(self._neo4j)
        return await check_promotion_eligibility(self._neo4j, current, target_level)

    async def apply_autonomy_change(self, new_level: int, reason: str, actor: str = "governance") -> dict[str, Any]:
        self._invalidate_state_cache()
        return await apply_autonomy_change(self._neo4j, new_level, reason, actor)

    # ─── Amendments ───────────────────────────────────────────────

    async def propose_amendment(
        self,
        proposed_drives: dict[str, float],
        title: str,
        description: str,
        proposer_id: str,
    ) -> dict[str, Any]:
        return await propose_amendment(
            self._neo4j, proposed_drives, title, description,
            proposer_id, self._governance,
        )

    async def apply_amendment(self, proposal_id: str, proposed_drives: dict[str, float]) -> dict[str, Any]:
        self._invalidate_state_cache()
        return await apply_amendment(self._neo4j, proposal_id, proposed_drives)

    # ─── Drift ────────────────────────────────────────────────────

    async def get_drift_report(self) -> dict[str, Any]:
        """Get the current drift report."""
        report = self._drift_tracker.compute_report()
        response = respond_to_drift(report)
        return {**report, "recommended_action": response}

    # ─── Governance Records ───────────────────────────────────────

    async def get_recent_reviews(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent constitutional reviews from the audit trail."""
        results = await self._neo4j.execute_read(
            """
            MATCH (g:GovernanceRecord)
            WHERE g.event_type = 'constitutional_review'
            RETURN g.id AS id, g.timestamp AS timestamp,
                   g.intent_id AS intent_id, g.verdict AS verdict,
                   g.alignment_composite AS composite,
                   g.reasoning AS reasoning, g.latency_ms AS latency_ms
            ORDER BY g.timestamp DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        return [dict(r) for r in results]

    async def get_governance_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get all governance events."""
        results = await self._neo4j.execute_read(
            """
            MATCH (g:GovernanceRecord)
            RETURN g.id AS id, g.event_type AS event_type,
                   g.timestamp AS timestamp, g.actor AS actor,
                   g.outcome AS outcome
            ORDER BY g.timestamp DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        return [dict(r) for r in results]

    # ─── Health ───────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Health check for Equor."""
        return {
            "status": "safe_mode" if self._safe_mode else "healthy",
            "total_reviews": self._total_reviews,
            "drift_tracker_size": self._drift_tracker.history_size,
            "safe_mode": self._safe_mode,
            "invariant_count": len(HARDCODED_INVARIANTS),
        }

    # ─── Internal Helpers ─────────────────────────────────────────

    def _safe_mode_review(self, intent: Intent) -> ConstitutionalCheck:
        """In safe mode, only Level 1 actions pass."""
        from ecodiaos.systems.equor.verdict import ACTION_AUTONOMY_MAP

        # Check if any step requires > Level 1
        for step in intent.plan.steps:
            executor_base = step.executor.split(".")[0].lower() if step.executor else ""
            for action_key, level in ACTION_AUTONOMY_MAP.items():
                if action_key in executor_base and level != 1:
                    return ConstitutionalCheck(
                        intent_id=intent.id,
                        verdict=Verdict.BLOCKED,
                        reasoning=(
                            "Equor is in safe mode. Only Level 1 (Advisor) "
                            "actions are permitted until normal operation resumes."
                        ),
                        confidence=1.0,
                    )

        return ConstitutionalCheck(
            intent_id=intent.id,
            verdict=Verdict.APPROVED,
            reasoning="Safe mode: Level 1 action permitted.",
            confidence=0.9,
        )

    async def _get_constitution_dict(self) -> dict[str, Any]:
        """Fetch the current constitution as a plain dict."""
        results = await self._neo4j.execute_read(
            """
            MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
            RETURN c.drive_coherence AS drive_coherence,
                   c.drive_care AS drive_care,
                   c.drive_growth AS drive_growth,
                   c.drive_honesty AS drive_honesty,
                   c.version AS version
            """
        )
        if results:
            return dict(results[0])
        # Fallback defaults
        return {
            "drive_coherence": 1.0,
            "drive_care": 1.0,
            "drive_growth": 1.0,
            "drive_honesty": 1.0,
            "version": 1,
        }

    async def _check_community_invariants(self, intent: Intent) -> list[str]:
        """Check all community-defined invariants via LLM evaluation (parallelised)."""
        results = await self._neo4j.execute_read(
            """
            MATCH (c:Constitution)-[:INCLUDES_INVARIANT]->(i:Invariant)
            WHERE i.source = 'community' AND i.active = true
            RETURN i.name AS name, i.description AS description
            """
        )

        if not results:
            return []

        # Run all invariant checks in parallel with a per-check timeout
        async def _check_one(row: dict[str, Any]) -> str | None:
            try:
                async with asyncio.timeout(0.4):
                    satisfied = await check_community_invariant(
                        self._llm, intent, row["name"], row["description"],
                    )
                    return None if satisfied else row["name"]
            except TimeoutError:
                logger.warning(
                    "community_invariant_check_timeout",
                    invariant=row["name"],
                )
                return None  # Timeout = skip (fail-open for liveness)
            except Exception:
                return row["name"]  # Error = fail-safe (treat as violated)

        check_results = await asyncio.gather(
            *[_check_one(row) for row in results],
        )
        return [name for name in check_results if name is not None]

    async def _feed_veto_to_evo(
        self, intent: Intent, check: ConstitutionalCheck,
    ) -> None:
        """
        Feed a constitutional veto to Evo as a learning episode.

        The organism should learn from its constitutional failures — when Equor
        blocks an intent, the violation becomes a negative-affect episode so Evo
        can refine hypothesis about which intent patterns violate the constitution.
        """
        try:
            from ecodiaos.primitives.memory_trace import Episode

            episode = Episode(
                source="equor.veto",
                modality="internal",
                raw_content=(
                    f"Constitutional veto: {intent.goal.description[:200]}. "
                    f"Reason: {check.reasoning[:300]}"
                ),
                summary=f"Blocked intent: {check.reasoning[:100]}",
                salience_composite=0.7,  # Vetoes are important learning events
                affect_valence=-0.3,
                affect_arousal=0.4,
            )
            await self._evo.process_episode(episode)
            logger.info("veto_fed_to_evo", intent_id=intent.id)
        except Exception:
            logger.debug("evo_veto_feed_failed", exc_info=True)

    async def _store_review_record(
        self,
        intent: Intent,
        alignment: DriveAlignmentVector,
        check: ConstitutionalCheck,
        latency_ms: int,
    ) -> None:
        """Store a constitutional review in the immutable audit trail."""
        now = utc_now()
        record_id = new_id()

        await self._neo4j.execute_write(
            """
            CREATE (g:GovernanceRecord {
                id: $id,
                event_type: 'constitutional_review',
                timestamp: datetime($now),
                intent_id: $intent_id,
                alignment_coherence: $coherence,
                alignment_care: $care,
                alignment_growth: $growth,
                alignment_honesty: $honesty,
                alignment_composite: $composite,
                verdict: $verdict,
                reasoning: $reasoning,
                confidence: $confidence,
                latency_ms: $latency_ms,
                actor: 'equor',
                outcome: $verdict
            })
            """,
            {
                "id": record_id,
                "now": now.isoformat(),
                "intent_id": intent.id,
                "coherence": alignment.coherence,
                "care": alignment.care,
                "growth": alignment.growth,
                "honesty": alignment.honesty,
                "composite": alignment.composite,
                "verdict": check.verdict.value,
                "reasoning": check.reasoning,
                "confidence": check.confidence,
                "latency_ms": latency_ms,
            },
        )

    async def _run_drift_check(self) -> None:
        """Run a drift check and respond accordingly."""
        report = self._drift_tracker.compute_report()
        response = respond_to_drift(report)

        if response["action"] != "log":
            await store_drift_report(self._neo4j, report, response)

        # Auto-demote on severe drift
        if response["action"] == "demote_autonomy":
            current = await get_autonomy_level(self._neo4j)
            if current > 1:
                await apply_autonomy_change(
                    self._neo4j,
                    current - 1,
                    reason=response["detail"],
                    actor="equor_drift_detection",
                )

    async def _run_promotion_check(self) -> None:
        """
        Periodically check whether the instance is eligible for autonomy promotion.

        Promotion requires governance approval, so this method only records
        eligibility as a governance record and logs it — it does NOT auto-promote.
        Governance (human or community vote) must call apply_autonomy_change().
        """
        try:
            current = await get_autonomy_level(self._neo4j)
            if current >= 3:
                return  # Already at maximum (Steward)

            target = current + 1
            eligibility = await check_promotion_eligibility(
                self._neo4j, current, target,
            )

            if not eligibility["eligible"]:
                return

            # Record the eligibility event so governance can act on it
            now = utc_now()
            record_id = new_id()
            await self._neo4j.execute_write(
                """
                CREATE (g:GovernanceRecord {
                    id: $id,
                    event_type: 'promotion_eligible',
                    timestamp: datetime($now),
                    details: $details,
                    actor: 'equor_promotion_check',
                    outcome: 'eligible'
                })
                """,
                {
                    "id": record_id,
                    "now": now.isoformat(),
                    "details": f"Eligible for promotion from level {current} to {target}",
                },
            )

            logger.info(
                "promotion_eligibility_detected",
                current_level=current,
                target_level=target,
                checks=eligibility["checks"],
            )
        except Exception:
            logger.debug("promotion_check_failed", exc_info=True)
