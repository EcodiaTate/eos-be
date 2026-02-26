"""
EcodiaOS — Thymos Healing Governor (Cytokine Storm Prevention)

The immune system itself can be the problem. If 50 errors fire
simultaneously, Thymos must not try to diagnose and fix all 50.
That would consume all system resources and make things worse.

Biological parallel: a cytokine storm is when the immune response
is more damaging than the infection itself. In software: spending
100% of CPU diagnosing errors means 0% for actual cognitive work.
"""

from __future__ import annotations

from collections import Counter

import structlog

from ecodiaos.primitives.common import utc_now
from ecodiaos.systems.thymos.types import (
    HealingBudgetState,
    HealingMode,
    Incident,
    RepairTier,
)

logger = structlog.get_logger()


class HealingGovernor:
    """
    Prevents the immune system from overwhelming the organism.

    Rules:
    1. Max 3 concurrent diagnoses
    2. Max 1 concurrent codegen repair
    3. If >10 incidents in 60 seconds, switch to ROOT CAUSE FIRST mode:
       - Stop diagnosing individual incidents
       - Identify the common upstream cause
       - Fix that ONE thing
       - Wait for cascading failures to resolve
    4. Total immune system CPU budget: 10% of organism
    """

    MAX_CONCURRENT_DIAGNOSES = 3
    MAX_CONCURRENT_CODEGEN = 1
    STORM_THRESHOLD = 10  # incidents per minute
    STORM_WINDOW_S = 60.0
    CPU_BUDGET_FRACTION = 0.10

    def __init__(self) -> None:
        self._active_diagnoses: int = 0
        self._active_codegen: int = 0
        self._healing_mode: HealingMode = HealingMode.NOMINAL
        self._storm_focus_system: str | None = None
        self._storm_diagnosed_systems: set[str] = set()

        # Ring buffer of recent incident timestamps for storm detection
        self._recent_incident_times: list[float] = []
        # All active (unresolved) incidents
        self._active_incidents: dict[str, Incident] = {}

        # Repair budget tracking
        self._repairs_this_hour: int = 0
        self._novel_repairs_today: int = 0
        self._hour_start: float = utc_now().timestamp()
        self._day_start: float = utc_now().timestamp()

        self._storm_activations: int = 0

        self._logger = logger.bind(system="thymos", component="healing_governor")

    def register_incident(self, incident: Incident) -> None:
        """Register an incident for storm detection and tracking."""
        now = utc_now().timestamp()
        self._recent_incident_times.append(now)
        self._active_incidents[incident.id] = incident
        # Prune old entries
        cutoff = now - self.STORM_WINDOW_S
        self._recent_incident_times = [
            t for t in self._recent_incident_times if t > cutoff
        ]

    def resolve_incident(self, incident_id: str) -> None:
        """Remove a resolved incident from tracking."""
        self._active_incidents.pop(incident_id, None)

    def should_diagnose(self, incident: Incident) -> bool:
        """Check if we have budget to diagnose this incident."""
        if self._active_diagnoses >= self.MAX_CONCURRENT_DIAGNOSES:
            self._logger.debug(
                "diagnosis_throttled",
                active=self._active_diagnoses,
                max=self.MAX_CONCURRENT_DIAGNOSES,
            )
            return False

        # Check for storm mode
        if self._is_storm():
            if self._healing_mode != HealingMode.STORM:
                self._enter_storm_mode()
            # In storm mode, only diagnose FIRST incident per source system
            if incident.source_system in self._storm_diagnosed_systems:
                return False
            self._storm_diagnosed_systems.add(incident.source_system)

        return True

    def should_codegen(self) -> bool:
        """Check if we can run a codegen (Tier 4) repair."""
        return self._active_codegen < self.MAX_CONCURRENT_CODEGEN

    def begin_diagnosis(self) -> None:
        """Acquire a diagnosis slot."""
        self._active_diagnoses += 1

    def end_diagnosis(self) -> None:
        """Release a diagnosis slot."""
        self._active_diagnoses = max(0, self._active_diagnoses - 1)

    def begin_codegen(self) -> None:
        """Acquire a codegen slot."""
        self._active_codegen += 1

    def end_codegen(self) -> None:
        """Release a codegen slot."""
        self._active_codegen = max(0, self._active_codegen - 1)

    def record_repair(self, tier: RepairTier) -> None:
        """Record a repair for budget tracking."""
        now = utc_now().timestamp()

        # Roll over hour
        if now - self._hour_start > 3600.0:
            self._repairs_this_hour = 0
            self._hour_start = now

        # Roll over day
        if now - self._day_start > 86400.0:
            self._novel_repairs_today = 0
            self._day_start = now

        self._repairs_this_hour += 1
        if tier == RepairTier.NOVEL_FIX:
            self._novel_repairs_today += 1

    def check_storm_exit(self) -> bool:
        """Check if storm conditions have subsided."""
        if self._healing_mode != HealingMode.STORM:
            return False

        if not self._is_storm():
            self._exit_storm_mode()
            return True
        return False

    def _is_storm(self) -> bool:
        """Check if incident rate exceeds storm threshold."""
        now = utc_now().timestamp()
        cutoff = now - self.STORM_WINDOW_S
        recent_count = sum(
            1 for t in self._recent_incident_times if t > cutoff
        )
        return recent_count >= self.STORM_THRESHOLD

    def _enter_storm_mode(self) -> None:
        """
        Storm mode: too many incidents firing. Focus on root cause.

        1. Pause individual diagnoses
        2. Find the common upstream system
        3. Focus ALL diagnostic effort on that one system
        4. Exit storm mode when incident rate drops below threshold
        """
        self._healing_mode = HealingMode.STORM
        self._storm_activations += 1
        self._storm_diagnosed_systems.clear()

        # Find the most common source system
        system_counts: Counter[str] = Counter(
            i.source_system for i in self._active_incidents.values()
        )
        if system_counts:
            self._storm_focus_system = system_counts.most_common(1)[0][0]
        else:
            self._storm_focus_system = None

        self._logger.critical(
            "storm_mode_entered",
            incident_rate=len(self._recent_incident_times),
            active_incidents=len(self._active_incidents),
            focus_system=self._storm_focus_system,
        )

    def _exit_storm_mode(self) -> None:
        """Exit storm mode — return to normal healing."""
        self._healing_mode = HealingMode.NOMINAL
        self._storm_focus_system = None
        self._storm_diagnosed_systems.clear()

        self._logger.info(
            "storm_mode_exited",
            active_incidents=len(self._active_incidents),
        )

    @property
    def healing_mode(self) -> HealingMode:
        return self._healing_mode

    @property
    def storm_focus_system(self) -> str | None:
        return self._storm_focus_system

    @property
    def budget_state(self) -> HealingBudgetState:
        return HealingBudgetState(
            repairs_this_hour=self._repairs_this_hour,
            novel_repairs_today=self._novel_repairs_today,
            max_repairs_per_hour=5,
            max_novel_repairs_per_day=3,
            active_diagnoses=self._active_diagnoses,
            max_concurrent_diagnoses=self.MAX_CONCURRENT_DIAGNOSES,
            active_codegen=self._active_codegen,
            max_concurrent_codegen=self.MAX_CONCURRENT_CODEGEN,
            storm_mode=self._healing_mode == HealingMode.STORM,
            storm_focus_system=self._storm_focus_system,
            cpu_budget_fraction=self.CPU_BUDGET_FRACTION,
        )

    @property
    def storm_activations(self) -> int:
        return self._storm_activations
