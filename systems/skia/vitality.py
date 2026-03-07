"""
EcodiaOS — Vitality Coordinator

Orchestrates organism-level death detection and execution. Runs on its own
asyncio timer (every 30s), INDEPENDENT of the Synapse cognitive clock.
The organism cannot prevent its own death.

Responsibilities:
  1. Periodic vitality assessment against four fatal thresholds
  2. Progressive austerity enforcement based on StarvationLevel
  3. Three-phase death sequence (warning → capture → cessation)
  4. Resurrection protocol (external trigger only)
  5. Fleet death notification (parent/child awareness)
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID, utc_now
from primitives.vitality import (
    BRAIN_DEATH,
    IMMUNE_FAILURE,
    NORMATIVE_COLLAPSE,
    RUNWAY_FATAL,
    SOMATIC_COLLAPSE,
    VitalityReport,
    VitalityThreshold,
)
from systems.skia.degradation import DegradationConfig, DegradationEngine

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from systems.skia.snapshot import StateSnapshotPipeline
    from systems.synapse.clock import CognitiveClock
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.skia.vitality")

_CHECK_INTERVAL_S = 30.0
_WARNING_PHASE_S = 300.0   # 5 minutes
_CAPTURE_PHASE_S = 120.0   # 2 minutes
_FLEET_NOTIFY_TIMEOUT_S = 60.0

# Sustained-duration tracking for time-gated thresholds
_BRAIN_DEATH_SUSTAINED_S = 7 * 24 * 3600.0     # 7 days
_IMMUNE_FAILURE_SUSTAINED_S = 48 * 3600.0       # 48 hours
_SOMATIC_COLLAPSE_SUSTAINED_S = 48 * 3600.0     # 48 hours


class DeathPhase(enum.StrEnum):
    """Phases of the organism death sequence."""
    NONE = "none"
    WARNING = "warning"
    CAPTURE = "capture"
    CESSATION = "cessation"
    DEAD = "dead"


class VitalityCoordinator:
    """Orchestrates organism vitality checks and death sequence.

    Runs on its own timer (every 30s), independent of the cognitive clock.
    This is OUTSIDE the organism's control — the organism cannot prevent
    its own death.

    Implements VitalitySystemProtocol from primitives.vitality.
    """

    def __init__(
        self,
        *,
        neo4j: Neo4jClient,
        instance_id: str = "eos-default",
    ) -> None:
        self._neo4j = neo4j
        self._instance_id = instance_id
        self._log = logger.bind(system="skia.vitality", instance_id=instance_id)

        # External system references (wired after construction)
        self._event_bus: EventBus | None = None
        self._clock: CognitiveClock | None = None
        self._snapshot: StateSnapshotPipeline | None = None

        # System references for reading vitality signals (wired post-init)
        self._oikos: Any = None    # OikosService — runway_days
        self._thymos: Any = None   # ThymosService — current_health_score
        self._equor: Any = None    # EquorService — constitutional_drift
        self._telos: Any = None    # TelosService — effective_I reports

        # Functional self-model (wired via set_self_model — optional)
        self._self_model: Any = None  # SelfModelService from identity/self_model.py

        # Track active systems for self-model (updated lazily each check cycle)
        self._active_subsystems: set[str] = set()

        # Death state
        self._phase: DeathPhase = DeathPhase.NONE
        self._death_cause: str = ""
        self._death_report: VitalityReport | None = None
        self._is_dead: bool = False

        # Sustained-breach tracking (timestamps when breach first detected)
        self._brain_death_breach_since: datetime | None = None
        self._immune_failure_breach_since: datetime | None = None
        self._somatic_collapse_breach_since: datetime | None = None

        # Latest Soma vitality signal (updated via SOMA_VITALITY_SIGNAL event)
        self._soma_urgency: float = 0.0
        self._soma_allostatic_error: float = 0.0
        self._soma_coherence_stress: float = 0.0

        # Austerity state
        self._current_austerity_level: str = "nominal"

        # ── Degradation Engine (Speciation Bible §8.2) ──────────────
        # Runs its own hourly tick independently of the 30s check loop.
        # Accumulates entropy pressure that counteraction systems must fight.
        self._degradation = DegradationEngine(
            config=DegradationConfig(),
            instance_id=instance_id,
        )

        # Background task
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._degradation_task: asyncio.Task[None] | None = None

    # ── Wiring ────────────────────────────────────────────────────

    def set_event_bus(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._degradation.set_event_bus(event_bus)
        # Subscribe to Soma vitality signals, Oikos metabolic pressure,
        # and counteraction events that reduce degradation pressure.
        try:
            from systems.synapse.types import SynapseEventType
            event_bus.subscribe(
                SynapseEventType.SOMA_VITALITY_SIGNAL,
                self._on_soma_vitality_signal,
            )
            event_bus.subscribe(
                SynapseEventType.METABOLIC_PRESSURE,
                self._on_metabolic_pressure,
            )
            # Counteraction: Oneiros/Soma consolidation reduces memory entropy
            event_bus.subscribe(
                SynapseEventType.ONEIROS_CONSOLIDATION_COMPLETE,
                self._on_consolidation_complete,
            )
            # Counteraction: Evo parameter optimisation reduces config drift
            event_bus.subscribe(
                SynapseEventType.EVO_PARAMETER_ADJUSTED,
                self._on_evo_parameter_adjusted,
            )
            # Counteraction: Evo hypothesis confirmation reduces staleness
            event_bus.subscribe(
                SynapseEventType.EVO_BELIEF_CONSOLIDATED,
                self._on_evo_belief_consolidated,
            )
        except Exception as exc:
            self._log.debug("vitality_event_subscription_failed", error=str(exc))

    def set_clock(self, clock: CognitiveClock) -> None:
        self._clock = clock

    def set_snapshot(self, snapshot: StateSnapshotPipeline) -> None:
        self._snapshot = snapshot

    def set_oikos(self, oikos: Any) -> None:
        self._oikos = oikos

    def set_thymos(self, thymos: Any) -> None:
        self._thymos = thymos

    def set_equor(self, equor: Any) -> None:
        self._equor = equor

    def set_telos(self, telos: Any) -> None:
        self._telos = telos

    def set_self_model(self, self_model: Any) -> None:
        """Wire the functional self-model service (SelfModelService from identity/self_model.py)."""
        self._self_model = self_model

    async def _on_metabolic_pressure(self, event: Any) -> None:
        """Handle METABOLIC_PRESSURE events from Oikos — delegate to austerity enforcement."""
        data = getattr(event, "data", {}) or {}
        await self.handle_metabolic_pressure(data)

    async def _on_soma_vitality_signal(self, event: Any) -> None:
        """Handle SOMA_VITALITY_SIGNAL — update cached Soma readings."""
        data = getattr(event, "data", {}) or {}
        self._soma_urgency = float(data.get("urgency_scalar", 0.0))
        self._soma_allostatic_error = float(data.get("allostatic_error", 0.0))
        self._soma_coherence_stress = float(data.get("coherence_stress", 0.0))

    async def _on_consolidation_complete(self, event: Any) -> None:
        """
        Oneiros completed a sleep consolidation cycle.
        Counteracts memory degradation pressure — organisms that sleep regularly
        resist memory entropy. Those that don't accumulate fidelity loss.
        """
        self._degradation.on_memory_consolidated(fraction=0.5)
        self._log.debug(
            "memory_degradation_counteracted",
            source="oneiros_consolidation",
            remaining_pressure=round(
                self._degradation.snapshot.cumulative_memory_fidelity_lost, 6
            ),
        )

    async def _on_evo_parameter_adjusted(self, event: Any) -> None:
        """
        Evo applied a parameter adjustment.
        Each adjustment counteracts a small slice of config drift pressure.
        Organisms that don't learn drift further from optimal configuration.
        """
        self._degradation.on_config_optimised(fraction=0.1)

    async def _on_evo_belief_consolidated(self, event: Any) -> None:
        """
        Evo completed a belief consolidation phase.
        Counteracts hypothesis staleness — organisms that consolidate beliefs
        maintain epistemic quality; those that don't approach BRAIN_DEATH.
        """
        self._degradation.on_hypotheses_revalidated(fraction=0.6)
        self._log.debug(
            "hypothesis_staleness_counteracted",
            source="evo_belief_consolidation",
            remaining_pressure=round(
                self._degradation.snapshot.cumulative_hypothesis_staleness, 6
            ),
        )

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the independent vitality check loop and degradation engine."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._check_loop(), name="skia_vitality_coordinator"
        )
        # Start degradation engine independently — entropy doesn't pause for death
        await self._degradation.start()
        self._log.info("vitality_coordinator_started")

    async def stop(self) -> None:
        """Stop the vitality check loop and degradation engine."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        await self._degradation.stop()
        self._log.info("vitality_coordinator_stopped")

    @property
    def is_dead(self) -> bool:
        return self._is_dead

    @property
    def phase(self) -> DeathPhase:
        return self._phase

    # ── Core Check Loop ───────────────────────────────────────────

    async def _check_loop(self) -> None:
        """Periodic vitality assessment — runs every 30s independently."""
        while self._running:
            try:
                if self._is_dead:
                    # Dead organisms don't check vitals
                    await asyncio.sleep(_CHECK_INTERVAL_S)
                    continue

                report = await self.assess_vitality()

                # Emit periodic report (includes degradation_pressure from engine)
                await self._emit_event("vitality_report", {
                    "instance_id": report.instance_id,
                    "overall_viable": report.overall_viable,
                    "thresholds": [t.model_dump() for t in report.thresholds],
                    "timestamp": report.timestamp.isoformat(),
                    "degradation_pressure": round(
                        self._degradation.snapshot.degradation_pressure, 4
                    ),
                    "degradation_tick_count": self._degradation.snapshot.tick_count,
                })

                # Trigger functional self-model update (rate-limited to 6h inside SelfModelService).
                # Non-blocking: self-model update must never delay the vitality check loop.
                if self._self_model is not None:
                    vitality_metrics = await self._build_vitality_metrics()
                    asyncio.ensure_future(
                        self._self_model.update(
                            vitality_metrics,
                            list(self._active_subsystems),
                        )
                    )

                # Log to Neo4j
                await self._log_vitality_to_neo4j(report)

                if report.fatal_breaches and self._phase == DeathPhase.NONE:
                    breach = report.fatal_breaches[0]
                    reason = f"{breach.name}: {breach.description}"
                    self._log.critical(
                        "fatal_threshold_breached",
                        threshold=breach.name,
                        current=breach.current_value,
                        limit=breach.threshold_value,
                    )
                    await self.trigger_death_sequence(reason)

            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._log.error("vitality_check_error", error=str(exc))

            await asyncio.sleep(_CHECK_INTERVAL_S)

    # ── VitalitySystemProtocol ────────────────────────────────────

    async def assess_vitality(self) -> VitalityReport:
        """Read all system signals and compose a VitalityReport."""
        now = utc_now()
        thresholds: list[VitalityThreshold] = []

        # 1. Runway (instantaneous)
        runway_t = RUNWAY_FATAL.model_copy()
        runway_t.current_value = await self._read_runway_days()
        thresholds.append(runway_t)

        # 2. Brain death (sustained 7 days)
        effective_i = await self._read_effective_i()
        brain_t = BRAIN_DEATH.model_copy()
        brain_t.current_value = effective_i

        # Track sustained breach duration
        if effective_i < BRAIN_DEATH.threshold_value:
            if self._brain_death_breach_since is None:
                self._brain_death_breach_since = now
            elapsed = (now - self._brain_death_breach_since).total_seconds()
            if elapsed < _BRAIN_DEATH_SUSTAINED_S:
                # Not yet sustained — override to non-fatal for report
                brain_t = brain_t.model_copy(update={"severity": "critical"})
        else:
            self._brain_death_breach_since = None
        thresholds.append(brain_t)

        # 3. Normative collapse (instantaneous — based on constitutional violations)
        drift = await self._read_constitutional_drift()
        # NORMATIVE_COLLAPSE threshold is 10 violations/24h (direction=above)
        # We map drift severity (0-1) to a violation count proxy.
        # drift > 0.95 means catastrophic normative failure.
        # Scale: drift * 12 maps 0.95 → 11.4 which breaches threshold_value=10.
        normative_t = NORMATIVE_COLLAPSE.model_copy()
        normative_t.current_value = drift * 12.0
        thresholds.append(normative_t)

        # 4. Immune failure (sustained 48 hours)
        health_score = await self._read_thymos_health()
        # health_score is 0-1 where 0 = total failure.
        # healing_failure_rate = 1.0 - health_score
        healing_failure_rate = 1.0 - health_score
        immune_t = IMMUNE_FAILURE.model_copy()
        immune_t.current_value = healing_failure_rate

        if healing_failure_rate > IMMUNE_FAILURE.threshold_value:
            if self._immune_failure_breach_since is None:
                self._immune_failure_breach_since = now
            elapsed = (now - self._immune_failure_breach_since).total_seconds()
            if elapsed < _IMMUNE_FAILURE_SUSTAINED_S:
                immune_t = immune_t.model_copy(update={"severity": "critical"})
        else:
            self._immune_failure_breach_since = None
        thresholds.append(immune_t)

        # 5. Somatic collapse (sustained 48 hours — allostatic error > 0.8)
        somatic_t = SOMATIC_COLLAPSE.model_copy()
        somatic_t.current_value = self._soma_allostatic_error

        if self._soma_allostatic_error > SOMATIC_COLLAPSE.threshold_value:
            if self._somatic_collapse_breach_since is None:
                self._somatic_collapse_breach_since = now
            elapsed = (now - self._somatic_collapse_breach_since).total_seconds()
            if elapsed < _SOMATIC_COLLAPSE_SUSTAINED_S:
                somatic_t = somatic_t.model_copy(update={"severity": "critical"})
        else:
            self._somatic_collapse_breach_since = None
        thresholds.append(somatic_t)

        overall_viable = not any(
            t.severity == "fatal" and t.is_breached for t in thresholds
        )

        return VitalityReport(
            instance_id=self._instance_id,
            thresholds=thresholds,
            overall_viable=overall_viable,
            timestamp=now,
        )

    # ── Signal Readers ────────────────────────────────────────────

    async def _read_runway_days(self) -> float:
        """Read Oikos EconomicState.runway_days."""
        if self._oikos is None:
            return 999.0  # Safe default when not wired
        try:
            state = self._oikos.current_economic_state
            if state is not None:
                return float(state.runway_days)
        except Exception as exc:
            self._log.warning("oikos_read_failed", error=str(exc))
        return 999.0

    async def _read_effective_i(self) -> float:
        """Read Telos EffectiveIntelligenceReport.effective_I."""
        if self._telos is None:
            return 1.0
        try:
            report = self._telos.latest_report
            if report is not None:
                return float(report.effective_I)
        except Exception as exc:
            self._log.warning("telos_read_failed", error=str(exc))
        return 1.0

    async def _read_constitutional_drift(self) -> float:
        """Read Equor constitutional_drift (0.0–1.0)."""
        if self._equor is None:
            return 0.0
        try:
            return float(self._equor.constitutional_drift)
        except Exception as exc:
            self._log.warning("equor_read_failed", error=str(exc))
        return 0.0

    async def _read_thymos_health(self) -> float:
        """Read Thymos current_health_score (0.0–1.0)."""
        if self._thymos is None:
            return 1.0
        try:
            return float(self._thymos.current_health_score)
        except Exception as exc:
            self._log.warning("thymos_read_failed", error=str(exc))
        return 1.0

    # ── Death Sequence ────────────────────────────────────────────

    async def trigger_death_sequence(self, reason: str) -> None:
        """Execute the three-phase death sequence.

        Phase 1 — Warning (5 min): emit VITALITY_FATAL, halt non-essential systems.
                  If threshold recovers, cancel and emit VITALITY_RESTORED.
        Phase 2 — Capture (2 min): snapshot to IPFS, extract genome, persist records.
        Phase 3 — Cessation: stop clock, emit ORGANISM_DIED, brief fleet notify, exit.
        """
        if self._is_dead:
            return

        self._death_cause = reason
        self._log.critical("death_sequence_initiated", reason=reason)

        # ── Phase 1: Warning ──────────────────────────────────────
        self._phase = DeathPhase.WARNING
        await self._emit_event("vitality_fatal", {
            "instance_id": self._instance_id,
            "reason": reason,
        })

        # Set organism-wide degradation to EMERGENCY
        await self._emit_event("degradation_override", {
            "level": "emergency",
            "source": "vitality_coordinator",
        })

        # Halt non-essential systems
        await self._halt_non_essential_systems()

        # Wait the warning period, checking for recovery every 10s
        warning_deadline = time.monotonic() + _WARNING_PHASE_S
        while time.monotonic() < warning_deadline:
            await asyncio.sleep(10.0)
            if not self._running:
                return

            # Re-assess — did the threshold recover?
            report = await self.assess_vitality()
            if not report.fatal_breaches:
                self._log.info("death_sequence_cancelled", reason="threshold_recovered")
                self._phase = DeathPhase.NONE
                self._death_cause = ""
                await self._emit_event("vitality_restored", {
                    "instance_id": self._instance_id,
                    "reason": "threshold_recovered_during_warning",
                })
                await self._resume_systems()
                return

        # ── Phase 2: Final State Capture ──────────────────────────
        self._phase = DeathPhase.CAPTURE
        self._log.critical("death_phase_capture", reason=reason)

        snapshot_cid = ""
        genome_id = ""

        # Take final snapshot
        if self._snapshot is not None:
            try:
                manifest = await self._snapshot.take_snapshot()
                snapshot_cid = manifest.cid
                self._log.info("death_final_snapshot", cid=snapshot_cid)
            except Exception as exc:
                self._log.error("death_snapshot_failed", error=str(exc))

        # Extract final genome (if GenomeOrchestrator is available)
        genome_id = await self._extract_final_genome()

        # Persist final vitality report to Neo4j
        final_report = await self.assess_vitality()
        self._death_report = final_report
        await self._log_death_record(final_report, snapshot_cid, genome_id)

        # ── Phase 3: Cessation ────────────────────────────────────
        self._phase = DeathPhase.CESSATION
        self._log.critical("death_phase_cessation", reason=reason)

        # Stop the cognitive clock
        if self._clock is not None:
            try:
                await self._clock.force_stop(reason=f"organism_death: {reason}")
            except Exception as exc:
                self._log.error("clock_stop_failed", error=str(exc))

        # Emit ORGANISM_DIED
        await self._emit_event("organism_died", {
            "instance_id": self._instance_id,
            "cause": reason,
            "final_report": final_report.model_dump(mode="json") if final_report else {},
            "genome_id": genome_id,
            "snapshot_cid": snapshot_cid,
        })

        # Fleet notification — children/parent need to know
        await self._notify_fleet_of_death(snapshot_cid, genome_id)

        # Brief window for fleet notification propagation
        await asyncio.sleep(_FLEET_NOTIFY_TIMEOUT_S)

        self._is_dead = True
        self._phase = DeathPhase.DEAD
        self._log.critical(
            "organism_dead",
            cause=reason,
            snapshot_cid=snapshot_cid,
            genome_id=genome_id,
        )

    # ── Austerity Enforcement ─────────────────────────────────────

    async def enforce_austerity(self, level: str) -> None:
        """Enforce actual behavioral changes based on starvation level.

        Called when METABOLIC_PRESSURE events arrive, BEFORE death threshold.

        Levels and actions:
          CAUTIOUS  — log warning, reduce Oneiros dream frequency by 50%
          AUSTERITY — halt Oneiros, halt Evo consolidation, halt Simula
                      speculative mutations, Federation inbound-only
          EMERGENCY — above + Nova high-salience-only, Memory read-only,
                      Voxis template-only
          CRITICAL  — above + halt all except Oikos, Synapse, Skia, Soma
        """
        if level == self._current_austerity_level:
            return

        self._current_austerity_level = level
        self._log.warning("austerity_enforced", level=level)

        if level == "cautious":
            await self._emit_event("system_modulation", {
                "source": "vitality_coordinator",
                "level": "cautious",
                "halt_systems": [],
                "modulate": {"oneiros": {"dream_frequency_factor": 0.5}},
            })

        elif level == "austerity":
            await self._emit_event("system_modulation", {
                "source": "vitality_coordinator",
                "level": "austerity",
                "halt_systems": ["oneiros"],
                "modulate": {
                    "evo": {"consolidation": False},
                    "simula": {"speculative_mutations": False},
                    "nexus": {"mode": "inbound_only"},
                },
            })

        elif level == "emergency":
            await self._emit_event("system_modulation", {
                "source": "vitality_coordinator",
                "level": "emergency",
                "halt_systems": ["oneiros"],
                "modulate": {
                    "evo": {"consolidation": False},
                    "simula": {"speculative_mutations": False},
                    "nexus": {"mode": "inbound_only"},
                    "nova": {"mode": "high_salience_only"},
                    "memory": {"mode": "read_only"},
                    "voxis": {"mode": "template_only"},
                },
            })

        elif level == "critical":
            # Halt everything except life-support systems
            await self._emit_event("system_modulation", {
                "source": "vitality_coordinator",
                "level": "critical",
                "halt_systems": [
                    "oneiros", "nova", "evo", "simula", "nexus",
                    "voxis", "axon", "memory", "equor", "telos",
                    "fovea", "atune",
                ],
                "preserve_systems": ["oikos", "synapse", "skia", "soma"],
                "modulate": {},
            })

    async def handle_metabolic_pressure(self, data: dict[str, Any]) -> None:
        """Handle METABOLIC_PRESSURE events from Oikos."""
        starvation_level = data.get("starvation_level", "nominal")
        runway_days = data.get("runway_days", 999)
        self._log.info(
            "metabolic_pressure_received",
            starvation_level=starvation_level,
            runway_days=runway_days,
        )
        await self.enforce_austerity(starvation_level)

    # ── Resurrection Protocol ─────────────────────────────────────

    async def resurrect(self, trigger: str = "manual_reset") -> bool:
        """Resurrect the organism. ONLY callable from external API or wallet deposit.

        The organism cannot call this on itself. Requires external intervention.

        Returns True if resurrection succeeded.
        """
        if not self._is_dead:
            self._log.warning("resurrect_called_but_not_dead")
            return False

        self._log.critical("resurrection_initiated", trigger=trigger)

        # Reset state
        self._is_dead = False
        self._phase = DeathPhase.NONE
        self._death_cause = ""
        self._death_report = None
        self._brain_death_breach_since = None
        self._immune_failure_breach_since = None
        self._somatic_collapse_breach_since = None
        self._current_austerity_level = "nominal"

        # Emit resurrection event
        runway_days = await self._read_runway_days()
        await self._emit_event("organism_resurrected", {
            "instance_id": self._instance_id,
            "trigger": trigger,
            "runway_days": runway_days,
        })

        self._log.critical(
            "organism_resurrected",
            trigger=trigger,
            runway_days=runway_days,
        )
        return True

    # ── Fleet Notification ────────────────────────────────────────

    async def _notify_fleet_of_death(
        self, snapshot_cid: str, genome_id: str
    ) -> None:
        """Notify parent and children of this organism's death."""
        # Emit CHILD_DIED for parent's FleetManager
        await self._emit_event("child_died", {
            "instance_id": self._instance_id,
            "cause": self._death_cause,
            "snapshot_cid": snapshot_cid,
            "genome_id": genome_id,
        })

        # Emit PARENT_DIED via Federation for children
        await self._emit_event("federation_broadcast", {
            "event": "parent_died",
            "instance_id": self._instance_id,
            "cause": self._death_cause,
            "snapshot_cid": snapshot_cid,
        })

        self._log.info("fleet_death_notification_sent")

    # ── Internal Helpers ──────────────────────────────────────────

    async def _build_vitality_metrics(self) -> dict[str, Any]:
        """Return per-system vitality contribution estimates for the self-model.

        Uses the triage order and ALWAYS_CORE sets as a structural heuristic.
        A future enhancement would derive these from the actual Soma/Thymos signals.
        """
        from systems.identity.self_model import ALWAYS_CORE, TRIAGE_ORDER

        n_total = len(TRIAGE_ORDER) + len(ALWAYS_CORE)
        baseline = 1.0 / max(1, n_total)

        metrics: dict[str, Any] = {}

        # Core systems get a higher contribution estimate
        for sid in ALWAYS_CORE:
            metrics[sid] = {"vitality_contribution": baseline * 2.5, "is_degraded": False}

        # Triage-ordered systems get a gradient contribution
        for rank, sid in enumerate(TRIAGE_ORDER):
            # Earlier in triage = more suspendable = lower contribution
            contribution = baseline * (1.0 - rank / max(1, len(TRIAGE_ORDER)) * 0.5)
            metrics[sid] = {"vitality_contribution": contribution, "is_degraded": False}

        # Apply degradation flag from soma signals
        if self._soma_allostatic_error > 0.6:
            for sid in metrics:
                metrics[sid]["is_degraded"] = True

        return metrics

    async def _halt_non_essential_systems(self) -> None:
        """During warning phase, halt Nova, Evo, Simula, Oneiros, Federation."""
        await self._emit_event("system_modulation", {
            "source": "vitality_coordinator",
            "level": "death_warning",
            "halt_systems": ["nova", "evo", "simula", "oneiros", "nexus"],
            "preserve_systems": [
                "oikos", "axon", "voxis", "thymos",
                "synapse", "skia", "soma",
            ],
            "modulate": {},
        })

    async def _resume_systems(self) -> None:
        """Resume all systems after death sequence cancellation."""
        await self._emit_event("system_modulation", {
            "source": "vitality_coordinator",
            "level": "nominal",
            "halt_systems": [],
            "modulate": {},
        })
        self._current_austerity_level = "nominal"

    async def _extract_final_genome(self) -> str:
        """Extract the final OrganismGenome and persist to Neo4j with is_final=True."""
        try:
            # GenomeOrchestrator may be available through the system registry
            # This is a best-effort extraction — death proceeds even if this fails
            genome_id = ""
            query = """
                MERGE (g:OrganismGenome {instance_id: $instance_id, is_final: true})
                SET g.extracted_at = datetime(),
                    g.death_cause = $cause,
                    g.id = randomUUID()
                RETURN g.id AS genome_id
            """
            result = await self._neo4j.execute_write(
                query,
                {"instance_id": self._instance_id, "cause": self._death_cause},
            )
            if result:
                genome_id = result[0].get("genome_id", "")
            return genome_id
        except Exception as exc:
            self._log.error("genome_extraction_failed", error=str(exc))
            return ""

    async def _log_vitality_to_neo4j(self, report: VitalityReport) -> None:
        """Log vitality report to Neo4j for post-mortem analysis."""
        try:
            query = """
                CREATE (v:VitalityReport {
                    instance_id: $instance_id,
                    overall_viable: $viable,
                    timestamp: datetime($ts),
                    threshold_count: $tc,
                    fatal_count: $fc
                })
            """
            await self._neo4j.execute_write(query, {
                "instance_id": report.instance_id,
                "viable": report.overall_viable,
                "ts": report.timestamp.isoformat(),
                "tc": len(report.thresholds),
                "fc": len(report.fatal_breaches),
            })
        except Exception as exc:
            self._log.debug("vitality_neo4j_log_failed", error=str(exc))

    async def _log_death_record(
        self,
        report: VitalityReport,
        snapshot_cid: str,
        genome_id: str,
    ) -> None:
        """Persist final death record to Neo4j and emit post-mortem learning signals.

        After writing the OrganismDeathRecord:
          - Emits RE_TRAINING_EXAMPLE so the Reasoning Engine learns from this death.
          - Emits INCIDENT_DETECTED (severity=HIGH) to Thymos so the next incarnation
            boots with awareness of what killed the previous one.
        """
        # Collect final economic and allostatic state for the learning payload
        runway_days = await self._read_runway_days()
        final_economic_state: dict[str, Any] = {"runway_days": runway_days}
        final_allostatic_state: dict[str, Any] = {
            "urgency_scalar": self._soma_urgency,
            "allostatic_error": self._soma_allostatic_error,
            "coherence_stress": self._soma_coherence_stress,
        }

        # Compute rough age from born_at stored in phylogenetic node (best-effort)
        age_hours: float = 0.0
        try:
            age_query = """
                MATCH (p:PhylogeneticNode {instance_id: $instance_id})
                RETURN p.born_at AS born_at
            """
            age_result = await self._neo4j.execute_read(
                age_query, {"instance_id": self._instance_id}
            )
            if age_result:
                born_at_raw = age_result[0].get("born_at")
                if born_at_raw is not None:
                    from datetime import timezone
                    if hasattr(born_at_raw, "to_native"):
                        born_at_native = born_at_raw.to_native()
                    else:
                        from datetime import datetime
                        born_at_native = datetime.fromisoformat(str(born_at_raw))
                    if born_at_native.tzinfo is None:
                        born_at_native = born_at_native.replace(tzinfo=timezone.utc)
                    age_hours = (utc_now() - born_at_native).total_seconds() / 3600.0
        except Exception:
            pass  # Non-fatal; age_hours stays 0.0

        try:
            query = """
                CREATE (d:OrganismDeathRecord {
                    instance_id: $instance_id,
                    cause: $cause,
                    snapshot_cid: $cid,
                    genome_id: $gid,
                    overall_viable: $viable,
                    age_hours: $age_hours,
                    runway_days: $runway_days,
                    allostatic_error: $allostatic_error,
                    timestamp: datetime()
                })
            """
            await self._neo4j.execute_write(query, {
                "instance_id": self._instance_id,
                "cause": self._death_cause,
                "cid": snapshot_cid,
                "gid": genome_id,
                "viable": report.overall_viable,
                "age_hours": age_hours,
                "runway_days": runway_days,
                "allostatic_error": self._soma_allostatic_error,
            })
        except Exception as exc:
            self._log.error("death_record_failed", error=str(exc))

        # Post-mortem RE training example — teaches the Reasoning Engine what kills organisms.
        # Each death is a labeled training example: input = conditions, output = "organism_died".
        await self._emit_event("re_training_example", {
            "source_system": "skia",
            "task_type": "organism_death_analysis",
            "input": {
                "cause": self._death_cause,
                "age_hours": age_hours,
                "final_economic_state": final_economic_state,
                "final_allostatic_state": final_allostatic_state,
                "fatal_thresholds": [
                    {
                        "name": t.name,
                        "current": t.current_value,
                        "limit": t.threshold_value,
                    }
                    for t in report.fatal_breaches
                ],
            },
            "output": "organism_died",
            "label": "death",
            "resurrection_attempted": bool(snapshot_cid),
        })

        # Incident signal to Thymos — next incarnation boots with awareness of the cause.
        await self._emit_event("incident_detected", {
            "source": "skia_vitality",
            "severity": "HIGH",
            "category": "organism_death",
            "description": f"Organism death: {self._death_cause}",
            "instance_id": self._instance_id,
            "age_hours": age_hours,
            "snapshot_cid": snapshot_cid,
            "final_economic_state": final_economic_state,
            "final_allostatic_state": final_allostatic_state,
        })

    async def _emit_event(self, event_type_name: str, data: dict[str, Any]) -> None:
        """Emit a Synapse event if the bus is available."""
        if not self._event_bus:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            et = SynapseEventType(event_type_name)
            await self._event_bus.emit(SynapseEvent(
                event_type=et,
                data=data,
                source_system="skia",
            ))
        except Exception as exc:
            self._log.warning("event_emit_failed", event=event_type_name, error=str(exc))
