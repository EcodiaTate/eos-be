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

import time
from typing import Any

import structlog

from ecodiaos.clients.neo4j import Neo4jClient
from ecodiaos.clients.llm import LLMProvider
from ecodiaos.config import EquorConfig, GovernanceConfig
from ecodiaos.primitives.common import (
    DriveAlignmentVector,
    HealthStatus,
    Verdict,
    new_id,
    utc_now,
)
from ecodiaos.primitives.constitutional import ConstitutionalCheck
from ecodiaos.primitives.intent import Intent

from ecodiaos.systems.equor.evaluators import evaluate_all_drives
from ecodiaos.systems.equor.verdict import compute_verdict
from ecodiaos.systems.equor.invariants import (
    check_hardcoded_invariants,
    check_community_invariant,
    HARDCODED_INVARIANTS,
)
from ecodiaos.systems.equor.autonomy import get_autonomy_level, check_promotion_eligibility, apply_autonomy_change
from ecodiaos.systems.equor.drift import DriftTracker, respond_to_drift, store_drift_report
from ecodiaos.systems.equor.amendment import (
    propose_amendment,
    apply_amendment,
    validate_amendment_proposal,
)
from ecodiaos.systems.equor.schema import ensure_equor_schema, seed_hardcoded_invariants

logger = structlog.get_logger()


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
    ):
        self._neo4j = neo4j
        self._llm = llm
        self._config = config
        self._governance = governance_config
        self._drift_tracker = DriftTracker(window_size=config.drift_window_size)
        self._safe_mode = False
        self._total_reviews = 0
        self._evo: Any = None  # Wired post-init for learning feedback from vetoes

    def set_evo(self, evo: Any) -> None:
        """Wire Evo so constitutional vetoes become learning episodes."""
        self._evo = evo
        logger.info("evo_wired_to_equor")

    # ─── Lifecycle ────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Ensure schema and seed invariants."""
        await ensure_equor_schema(self._neo4j)
        await seed_hardcoded_invariants(self._neo4j)
        logger.info("equor_initialized")

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
            # 1. Run all four drive evaluators in parallel
            alignment = await evaluate_all_drives(intent)

            # 2. Get current constitution and autonomy level
            constitution = await self._get_constitution_dict()
            autonomy_level = await get_autonomy_level(self._neo4j)

            # 3. Run the verdict engine (includes invariant checks)
            check = compute_verdict(alignment, intent, autonomy_level, constitution)

            # 4. Check community invariants if we haven't already blocked
            if check.verdict not in (Verdict.BLOCKED,):
                community_violations = await self._check_community_invariants(intent)
                if community_violations:
                    check.verdict = Verdict.BLOCKED
                    check.reasoning = (
                        f"Community invariant violated: {community_violations[0]}"
                    )

            # 5. Store audit trail
            elapsed_ms = int((time.monotonic() - start) * 1000)
            await self._store_review_record(intent, alignment, check, elapsed_ms)

            # 6. Update drift tracking
            self._drift_tracker.record_decision(alignment, check.verdict.value)
            self._total_reviews += 1

            # 6b. Feed blocked verdicts to Evo as negative learning episodes
            if check.verdict == Verdict.BLOCKED and self._evo is not None:
                await self._feed_veto_to_evo(intent, check)

            # 7. Check if drift report is due + promotion eligibility
            if self._total_reviews % self._config.drift_report_interval == 0:
                await self._run_drift_check()
                await self._run_promotion_check()

            logger.info(
                "constitutional_review_complete",
                intent_id=intent.id,
                verdict=check.verdict.value,
                composite=f"{alignment.composite:.2f}",
                latency_ms=elapsed_ms,
            )

            return check

        except Exception as e:
            # Equor failure = enter safe mode
            logger.error("equor_review_failed", error=str(e), intent_id=intent.id)
            self._safe_mode = True
            return self._safe_mode_review(intent)

    # ─── Invariant Management ─────────────────────────────────────

    async def get_invariants(self) -> list[dict]:
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

    async def check_promotion(self, target_level: int) -> dict:
        current = await get_autonomy_level(self._neo4j)
        return await check_promotion_eligibility(self._neo4j, current, target_level)

    async def apply_autonomy_change(self, new_level: int, reason: str, actor: str = "governance") -> dict:
        return await apply_autonomy_change(self._neo4j, new_level, reason, actor)

    # ─── Amendments ───────────────────────────────────────────────

    async def propose_amendment(
        self,
        proposed_drives: dict[str, float],
        title: str,
        description: str,
        proposer_id: str,
    ) -> dict:
        return await propose_amendment(
            self._neo4j, proposed_drives, title, description,
            proposer_id, self._governance,
        )

    async def apply_amendment(self, proposal_id: str, proposed_drives: dict[str, float]) -> dict:
        return await apply_amendment(self._neo4j, proposal_id, proposed_drives)

    # ─── Drift ────────────────────────────────────────────────────

    async def get_drift_report(self) -> dict:
        """Get the current drift report."""
        report = self._drift_tracker.compute_report()
        response = respond_to_drift(report)
        return {**report, "recommended_action": response}

    # ─── Governance Records ───────────────────────────────────────

    async def get_recent_reviews(self, limit: int = 20) -> list[dict]:
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

    async def get_governance_history(self, limit: int = 50) -> list[dict]:
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

    async def health(self) -> dict:
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

    async def _get_constitution_dict(self) -> dict:
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
        """Check all community-defined invariants via LLM evaluation."""
        results = await self._neo4j.execute_read(
            """
            MATCH (c:Constitution)-[:INCLUDES_INVARIANT]->(i:Invariant)
            WHERE i.source = 'community' AND i.active = true
            RETURN i.name AS name, i.description AS description
            """
        )

        violations = []
        for row in results:
            satisfied = await check_community_invariant(
                self._llm, intent, row["name"], row["description"],
            )
            if not satisfied:
                violations.append(row["name"])

        return violations

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
