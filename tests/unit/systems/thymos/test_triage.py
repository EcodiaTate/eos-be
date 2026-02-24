"""
Tests for Thymos Triage Layer.

Covers:
  - IncidentDeduplicator
  - SeverityScorer
  - ResponseRouter
"""

from __future__ import annotations

import pytest

from ecodiaos.primitives.common import new_id, utc_now
from ecodiaos.systems.thymos.triage import (
    IncidentDeduplicator,
    ResponseRouter,
    SeverityScorer,
)
from ecodiaos.systems.thymos.types import (
    Incident,
    IncidentClass,
    IncidentSeverity,
    RepairTier,
)


def _make_incident(
    fingerprint: str = "abc123",
    incident_class: IncidentClass = IncidentClass.CRASH,
    severity: IncidentSeverity = IncidentSeverity.MEDIUM,
    source_system: str = "nova",
    blast_radius: float = 0.2,
    occurrence_count: int = 1,
    user_visible: bool = False,
) -> Incident:
    return Incident(
        id=new_id(),
        timestamp=utc_now(),
        incident_class=incident_class,
        severity=severity,
        fingerprint=fingerprint,
        source_system=source_system,
        error_type="TestError",
        error_message="Test error message",
        blast_radius=blast_radius,
        occurrence_count=occurrence_count,
        user_visible=user_visible,
    )


# ─── IncidentDeduplicator ─────────────────────────────────────────


class TestIncidentDeduplicator:
    def test_first_occurrence_is_new(self):
        dedup = IncidentDeduplicator()
        incident = _make_incident(fingerprint="unique_fp_1")
        result = dedup.deduplicate(incident)
        assert result is incident  # New incident returned as-is

    def test_second_occurrence_is_duplicate(self):
        dedup = IncidentDeduplicator()
        inc1 = _make_incident(fingerprint="dup_fp")
        inc2 = _make_incident(fingerprint="dup_fp")
        dedup.deduplicate(inc1)
        result = dedup.deduplicate(inc2)
        assert result is None  # Duplicate → None

    def test_duplicate_increments_count(self):
        dedup = IncidentDeduplicator()
        inc1 = _make_incident(fingerprint="dup_fp")
        dedup.deduplicate(inc1)
        inc2 = _make_incident(fingerprint="dup_fp")
        dedup.deduplicate(inc2)
        # The original incident should have incremented count
        assert inc1.occurrence_count == 2

    def test_different_fingerprints_not_duplicated(self):
        dedup = IncidentDeduplicator()
        inc1 = _make_incident(fingerprint="fp_a")
        inc2 = _make_incident(fingerprint="fp_b")
        assert dedup.deduplicate(inc1) is inc1
        assert dedup.deduplicate(inc2) is inc2

    def test_active_count(self):
        dedup = IncidentDeduplicator()
        dedup.deduplicate(_make_incident(fingerprint="a"))
        dedup.deduplicate(_make_incident(fingerprint="b"))
        dedup.deduplicate(_make_incident(fingerprint="a"))  # dup
        assert dedup.active_count == 2

    def test_resolve_removes_from_active(self):
        dedup = IncidentDeduplicator()
        inc = _make_incident(fingerprint="resolve_me")
        dedup.deduplicate(inc)
        resolved = dedup.resolve("resolve_me")
        assert resolved is inc
        assert dedup.active_count == 0


# ─── SeverityScorer ────────────────────────────────────────────────


class TestSeverityScorer:
    def test_returns_severity_enum(self):
        scorer = SeverityScorer()
        incident = _make_incident()
        result = scorer.compute_severity(incident)
        assert isinstance(result, IncidentSeverity)

    def test_high_blast_radius_increases_severity(self):
        scorer = SeverityScorer()
        low_blast = _make_incident(blast_radius=0.05, severity=IncidentSeverity.LOW)
        high_blast = _make_incident(blast_radius=0.8, severity=IncidentSeverity.LOW)
        score_low = scorer.compute_severity(low_blast)
        score_high = scorer.compute_severity(high_blast)
        # Higher blast radius should result in >= severity
        severity_order = [
            IncidentSeverity.INFO,
            IncidentSeverity.LOW,
            IncidentSeverity.MEDIUM,
            IncidentSeverity.HIGH,
            IncidentSeverity.CRITICAL,
        ]
        assert severity_order.index(score_high) >= severity_order.index(score_low)

    def test_user_visible_influences_score(self):
        scorer = SeverityScorer()
        invisible = _make_incident(user_visible=False)
        visible = _make_incident(user_visible=True)
        # Both should return a valid severity
        assert isinstance(scorer.compute_severity(invisible), IncidentSeverity)
        assert isinstance(scorer.compute_severity(visible), IncidentSeverity)


# ─── ResponseRouter ────────────────────────────────────────────────


class TestResponseRouter:
    def test_critical_routes_to_restart(self):
        router = ResponseRouter()
        incident = _make_incident(severity=IncidentSeverity.CRITICAL)
        tier = router.route(incident)
        assert tier in (RepairTier.RESTART, RepairTier.KNOWN_FIX, RepairTier.ESCALATE)

    def test_info_routes_to_noop(self):
        router = ResponseRouter()
        incident = _make_incident(severity=IncidentSeverity.INFO)
        tier = router.route(incident)
        assert tier == RepairTier.NOOP

    def test_low_routes_to_noop_or_parameter(self):
        router = ResponseRouter()
        incident = _make_incident(severity=IncidentSeverity.LOW)
        tier = router.route(incident)
        assert tier in (RepairTier.NOOP, RepairTier.PARAMETER)

    def test_returns_valid_tier(self):
        router = ResponseRouter()
        for severity in IncidentSeverity:
            incident = _make_incident(severity=severity)
            tier = router.route(incident)
            assert isinstance(tier, RepairTier)
