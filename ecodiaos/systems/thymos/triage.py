"""
EcodiaOS — Thymos Triage Layer (Classification & Prioritization)

When sentinels produce Incidents, Triage determines what to do with them:
  1. Deduplicate — same fingerprint within a window → increment, don't duplicate
  2. Score severity — composite scoring using blast radius, recurrence, impact
  3. Route response — severity → initial repair tier
"""

from __future__ import annotations

from datetime import datetime
from collections import defaultdict

import structlog

from ecodiaos.primitives.common import utc_now
from ecodiaos.systems.thymos.types import (
    Incident,
    IncidentClass,
    IncidentSeverity,
    RepairTier,
)


logger = structlog.get_logger()


# ─── Deduplication ───────────────────────────────────────────────


# Per-class dedup windows: how long before the same fingerprint
# creates a new incident instead of incrementing the existing one
_DEDUP_WINDOWS: dict[IncidentClass, float] = {
    IncidentClass.CRASH: 60.0,  # 1 minute — crashes are loud
    IncidentClass.DEGRADATION: 300.0,  # 5 minutes — slow issues need observation
    IncidentClass.CONTRACT_VIOLATION: 30.0,  # 30 seconds — may be transient
    IncidentClass.LOOP_SEVERANCE: 3600.0,  # 1 hour — structural, slow to resolve
    IncidentClass.DRIFT: 3600.0,  # 1 hour — drift is slow by definition
    IncidentClass.PREDICTION_FAILURE: 300.0,  # 5 minutes
    IncidentClass.RESOURCE_EXHAUSTION: 120.0,  # 2 minutes
    IncidentClass.COGNITIVE_STALL: 600.0,  # 10 minutes
}


class IncidentDeduplicator:
    """
    Same fingerprint within a time window → increment count, don't create
    a new incident. This prevents a single bug from flooding the immune
    system with thousands of identical incidents.
    """

    def __init__(self) -> None:
        # fingerprint → (incident, last_seen_timestamp)
        self._active: dict[str, tuple[Incident, datetime]] = {}
        self._logger = logger.bind(system="thymos", component="deduplicator")

    def deduplicate(self, incident: Incident) -> Incident | None:
        """
        Check if this incident is a duplicate.

        Returns:
          - None if it's a duplicate (the existing incident was incremented)
          - The incident if it's new (caller should process it)
        """
        window_s = _DEDUP_WINDOWS.get(incident.incident_class, 60.0)
        existing = self._active.get(incident.fingerprint)

        if existing is not None:
            existing_incident, last_seen = existing
            age_s = (incident.timestamp - last_seen).total_seconds()

            if age_s <= window_s:
                # Duplicate — increment existing
                existing_incident.occurrence_count += 1
                self._active[incident.fingerprint] = (
                    existing_incident,
                    incident.timestamp,
                )
                self._logger.debug(
                    "incident_deduplicated",
                    fingerprint=incident.fingerprint,
                    occurrences=existing_incident.occurrence_count,
                )
                return None
            else:
                # Window expired — treat as new, but carry forward the count
                incident.first_seen = existing_incident.first_seen or existing_incident.timestamp

        # New incident
        if incident.first_seen is None:
            incident.first_seen = incident.timestamp
        self._active[incident.fingerprint] = (incident, incident.timestamp)
        return incident

    def get_active_incident(self, fingerprint: str) -> Incident | None:
        """Get the active incident for a fingerprint, if any."""
        entry = self._active.get(fingerprint)
        return entry[0] if entry else None

    def resolve(self, fingerprint: str) -> Incident | None:
        """Remove a resolved incident from the active set."""
        entry = self._active.pop(fingerprint, None)
        return entry[0] if entry else None

    def prune_stale(self, max_age_s: float = 7200.0) -> int:
        """Remove incidents older than max_age_s. Returns count removed."""
        now = utc_now()
        to_remove: list[str] = []
        for fp, (_, last_seen) in self._active.items():
            if (now - last_seen).total_seconds() > max_age_s:
                to_remove.append(fp)
        for fp in to_remove:
            del self._active[fp]
        return len(to_remove)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def active_incidents(self) -> list[Incident]:
        return [inc for inc, _ in self._active.values()]


# ─── Severity Scoring ────────────────────────────────────────────


class SeverityScorer:
    """
    Composite severity scoring that refines the sentinel's initial assessment.

    Considers:
    1. Blast radius (0.25) — fraction of systems affected
    2. Recurrence velocity (0.20) — how fast is occurrence_count growing?
    3. Constitutional impact (0.25) — max impact across all four drives
    4. User visibility (0.15) — binary but high weight
    5. Self-healing potential (0.15) — inverse of how likely auto-fix is
    """

    def __init__(self) -> None:
        # Track recurrence velocity per fingerprint
        self._recurrence_history: dict[str, list[tuple[datetime, int]]] = defaultdict(list)
        self._logger = logger.bind(system="thymos", component="severity_scorer")

    def compute_severity(self, incident: Incident) -> IncidentSeverity:
        """Compute composite severity score."""
        # Record recurrence for velocity calculation
        self._recurrence_history[incident.fingerprint].append(
            (incident.timestamp, incident.occurrence_count)
        )
        # Keep only last 20 entries per fingerprint
        if len(self._recurrence_history[incident.fingerprint]) > 20:
            self._recurrence_history[incident.fingerprint] = (
                self._recurrence_history[incident.fingerprint][-20:]
            )

        blast = incident.blast_radius
        recurrence = self._recurrence_velocity(incident)
        constitutional = max(incident.constitutional_impact.values()) if incident.constitutional_impact else 0.0
        user_vis = 1.0 if incident.user_visible else 0.0
        healing_potential = self._healing_potential(incident)

        score = (
            blast * 0.25
            + recurrence * 0.20
            + constitutional * 0.25
            + user_vis * 0.15
            + (1.0 - healing_potential) * 0.15
        )

        if score > 0.8:
            return IncidentSeverity.CRITICAL
        if score > 0.6:
            return IncidentSeverity.HIGH
        if score > 0.3:
            return IncidentSeverity.MEDIUM
        if score > 0.1:
            return IncidentSeverity.LOW
        return IncidentSeverity.INFO

    def _recurrence_velocity(self, incident: Incident) -> float:
        """
        How fast is occurrence_count growing? 0.0 = single event, 1.0 = rapid fire.

        Uses the rate of count increase over the last observation window.
        """
        history = self._recurrence_history.get(incident.fingerprint, [])
        if len(history) < 2:
            # Single event — low velocity unless high count
            return min(1.0, incident.occurrence_count / 100.0)

        first_ts, first_count = history[0]
        last_ts, last_count = history[-1]
        elapsed_s = (last_ts - first_ts).total_seconds()

        if elapsed_s < 1.0:
            return min(1.0, (last_count - first_count) / 10.0)

        rate_per_minute = ((last_count - first_count) / elapsed_s) * 60.0
        # Normalize: 10+ per minute = 1.0
        return min(1.0, rate_per_minute / 10.0)

    def _healing_potential(self, incident: Incident) -> float:
        """
        How likely can we auto-fix this? Higher = easier to heal.

        Based on incident class — some classes are inherently easier
        to fix automatically than others.
        """
        potentials: dict[IncidentClass, float] = {
            IncidentClass.CRASH: 0.3,  # Need diagnosis, often novel
            IncidentClass.DEGRADATION: 0.5,  # Parameter tweaks often work
            IncidentClass.CONTRACT_VIOLATION: 0.6,  # Often transient
            IncidentClass.LOOP_SEVERANCE: 0.2,  # Structural — hard to auto-fix
            IncidentClass.DRIFT: 0.7,  # Parameter adjustments
            IncidentClass.PREDICTION_FAILURE: 0.4,  # Needs model update
            IncidentClass.RESOURCE_EXHAUSTION: 0.8,  # Rebalance resources
            IncidentClass.COGNITIVE_STALL: 0.3,  # Complex, may need restart
        }
        return potentials.get(incident.incident_class, 0.5)


# ─── Response Routing ────────────────────────────────────────────


# Severity → initial repair tier
RESPONSE_ROUTES: dict[IncidentSeverity, RepairTier] = {
    IncidentSeverity.CRITICAL: RepairTier.RESTART,  # Stabilize first
    IncidentSeverity.HIGH: RepairTier.KNOWN_FIX,  # Check antibody library
    IncidentSeverity.MEDIUM: RepairTier.PARAMETER,  # Try parameter adjustment
    IncidentSeverity.LOW: RepairTier.NOOP,  # Log and observe
    IncidentSeverity.INFO: RepairTier.NOOP,  # Pattern detection only
}


class ResponseRouter:
    """
    Routes incidents to the appropriate response tier.

    Critical incidents get immediate stabilization (restart) AND
    diagnosis in parallel. The restart buys time while the diagnostic
    system figures out the root cause.
    """

    def __init__(self) -> None:
        self._logger = logger.bind(system="thymos", component="response_router")

    def route(self, incident: Incident) -> RepairTier:
        """Determine the initial repair tier for an incident."""
        tier = RESPONSE_ROUTES.get(incident.severity, RepairTier.NOOP)

        # Override: if we have a high-count recurrence, escalate
        if incident.occurrence_count > 50 and tier.value < RepairTier.KNOWN_FIX.value:
            tier = RepairTier.KNOWN_FIX
            self._logger.info(
                "tier_escalated_by_recurrence",
                fingerprint=incident.fingerprint,
                occurrences=incident.occurrence_count,
                new_tier=tier.name,
            )

        incident.repair_tier = tier
        return tier
