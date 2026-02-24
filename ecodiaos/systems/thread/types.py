"""
EcodiaOS — Thread Internal Types

The data structures that constitute the organism's narrative identity.
These model commitments, schemas, fingerprints, and life chapters —
the building blocks of autobiographical selfhood.

Fingerprint dimensions (29D):
  Personality        9D  — from Voxis PersonalityVector
  Drive alignment    4D  — from Equor DriftTracker mean alignment
  Affect             6D  — from Atune current_affect
  Goal profile       5D  — estimated from episode/goal counts
  Interaction        5D  — estimated from conversation/expression metrics
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import Field

from ecodiaos.primitives.common import (
    EOSBaseModel,
    Identified,
    Timestamped,
    new_id,
    utc_now,
)


# ─── Enums ────────────────────────────────────────────────────────────────────


class CommitmentType(str, enum.Enum):
    """What kind of commitment is this?"""

    CONSTITUTIONAL_GROUNDING = "constitutional_grounding"  # Birth promises (4 drives)
    RELATIONAL = "relational"          # Commitments to specific people/communities
    VOCATIONAL = "vocational"          # Commitments about purpose and role
    EPISTEMIC = "epistemic"            # Commitments to ways of knowing
    AESTHETIC = "aesthetic"             # Commitments to style and expression


class CommitmentStrength(str, enum.Enum):
    """How deeply held is this commitment?"""

    NASCENT = "nascent"          # Just formed, untested
    DEVELOPING = "developing"    # Tested in a few situations
    ESTABLISHED = "established"  # Consistently held across contexts
    CORE = "core"                # Foundational to identity — change triggers crisis


class SchemaStatus(str, enum.Enum):
    """Lifecycle of an identity schema."""

    EMERGING = "emerging"        # Pattern detected, not yet crystallised
    FORMING = "forming"          # Accumulating evidence, taking shape
    ESTABLISHED = "established"  # Stable, integrated into self-narrative
    DOMINANT = "dominant"        # Primary lens for self-interpretation
    ARCHIVED = "archived"        # No longer active but remembered


class ChapterStatus(str, enum.Enum):
    """Status of a narrative chapter."""

    ACTIVE = "active"      # Currently living this chapter
    CLOSED = "closed"      # Chapter has ended, theme resolved
    EMERGING = "emerging"  # A new chapter is forming


# ─── Commitment ───────────────────────────────────────────────────────────────


class Commitment(Identified, Timestamped):
    """
    A lived promise that shapes identity. Ricoeur's 'keeping one's word'
    as computational structure.

    Constitutional commitments are seeded at birth from the four drives.
    Others emerge from experience and are strengthened through action.
    """

    type: CommitmentType
    statement: str                  # Natural language commitment
    strength: CommitmentStrength = CommitmentStrength.NASCENT
    drive_source: str = ""          # Which drive this commitment serves
    evidence_episodes: list[str] = Field(default_factory=list)
    violation_episodes: list[str] = Field(default_factory=list)
    last_tested_at: datetime | None = None
    test_count: int = 0
    upheld_count: int = 0

    @property
    def fidelity(self) -> float:
        """How consistently has this commitment been upheld? 0.0–1.0."""
        if self.test_count == 0:
            return 1.0  # Untested = perfect fidelity
        return self.upheld_count / self.test_count


# ─── Identity Schema ─────────────────────────────────────────────────────────


class IdentitySchema(Identified, Timestamped):
    """
    A crystallised pattern of self-understanding.

    Schemas emerge from Evo's pattern detection, are refined through
    experience, and become the lenses through which the organism
    interprets itself. "I am someone who..."
    """

    statement: str                   # "I tend to..." / "I am someone who..."
    status: SchemaStatus = SchemaStatus.EMERGING
    source_pattern_ids: list[str] = Field(default_factory=list)
    supporting_episodes: list[str] = Field(default_factory=list)
    contradicting_episodes: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)
    confidence: float = 0.5         # How well-supported is this schema
    salience: float = 0.5           # How relevant to current identity
    last_activated_at: datetime = Field(default_factory=utc_now)

    @property
    def evidence_ratio(self) -> float:
        total = len(self.supporting_episodes) + len(self.contradicting_episodes)
        if total == 0:
            return 0.5
        return len(self.supporting_episodes) / total


# ─── Identity Fingerprint ────────────────────────────────────────────────────


FINGERPRINT_DIMS = 29

# Named dimension ranges for interpretability
PERSONALITY_SLICE = slice(0, 9)    # warmth, directness, verbosity, formality,
                                    # curiosity_expression, humour, empathy_expression,
                                    # confidence_display, metaphor_use
DRIVE_SLICE = slice(9, 13)         # coherence, care, growth, honesty
AFFECT_SLICE = slice(13, 19)       # valence, arousal, dominance, curiosity,
                                    # care_activation, coherence_stress
GOAL_SLICE = slice(19, 24)         # active_goals_norm, epistemic_ratio,
                                    # care_ratio, achievement_rate, goal_turnover
INTERACTION_SLICE = slice(24, 29)  # speak_rate, silence_rate, expression_diversity,
                                    # conversation_depth, community_engagement


class IdentityFingerprint(Identified, Timestamped):
    """
    A 29-dimensional snapshot of who the organism is right now.

    Comparing fingerprints over time reveals identity drift — not
    constitutional drift (Equor handles that) but the slower shift
    in who the organism is becoming.
    """

    vector: list[float] = Field(default_factory=lambda: [0.0] * FINGERPRINT_DIMS)
    cycle_number: int = 0

    @property
    def personality(self) -> list[float]:
        return self.vector[PERSONALITY_SLICE]

    @property
    def drive_alignment(self) -> list[float]:
        return self.vector[DRIVE_SLICE]

    @property
    def affect(self) -> list[float]:
        return self.vector[AFFECT_SLICE]

    @property
    def goal_profile(self) -> list[float]:
        return self.vector[GOAL_SLICE]

    @property
    def interaction_profile(self) -> list[float]:
        return self.vector[INTERACTION_SLICE]

    def distance_to(self, other: IdentityFingerprint) -> float:
        """
        Wasserstein-inspired distance between two fingerprints.
        Uses L1 (Manhattan) distance normalised by dimensionality.
        """
        if len(self.vector) != len(other.vector):
            return float("inf")
        return sum(
            abs(a - b) for a, b in zip(self.vector, other.vector)
        ) / len(self.vector)


# ─── Narrative Chapter ───────────────────────────────────────────────────────


class NarrativeChapter(Identified, Timestamped):
    """
    A recognised phase in the organism's life story.

    Chapters emerge from significant identity shifts — a new community,
    a constitutional crisis, a period of rapid growth.
    """

    title: str
    theme: str                        # One-sentence theme
    status: ChapterStatus = ChapterStatus.ACTIVE
    opened_at_cycle: int = 0
    closed_at_cycle: int | None = None
    key_schemas: list[str] = Field(default_factory=list)     # Schema IDs
    key_commitments: list[str] = Field(default_factory=list)  # Commitment IDs
    key_episodes: list[str] = Field(default_factory=list)     # Episode IDs
    fingerprint_at_start: str = ""    # Fingerprint ID at chapter open
    fingerprint_at_close: str = ""    # Fingerprint ID at chapter close
    summary: str = ""                 # LLM-generated chapter summary


# ─── Life Story ──────────────────────────────────────────────────────────────


class LifeStorySnapshot(EOSBaseModel):
    """
    A periodic synthesis of the organism's autobiography.
    The organism's own understanding of its narrative arc.
    """

    synthesis: str                    # Natural language life story
    chapter_count: int = 0
    active_chapter: str = ""          # Current chapter title
    core_schemas: list[str] = Field(default_factory=list)     # Top schema statements
    core_commitments: list[str] = Field(default_factory=list)  # Top commitment statements
    identity_coherence: float = 0.5   # How integrated is the narrative (0–1)
    generated_at: datetime = Field(default_factory=utc_now)
    cycle_number: int = 0


# ─── Thread Health ───────────────────────────────────────────────────────────


class ThreadHealthSnapshot(EOSBaseModel):
    """Observability snapshot for Thread system health."""

    status: str = "healthy"
    total_commitments: int = 0
    total_schemas: int = 0
    total_fingerprints: int = 0
    total_chapters: int = 0
    active_chapter: str = ""
    identity_coherence: float = 0.0
    fingerprint_drift: float = 0.0
    on_cycle_count: int = 0
    life_story_integrations: int = 0


# ─── Schema Conflict ─────────────────────────────────────────────────────────


class SchemaConflict(Identified, Timestamped):
    """
    Detected when two ESTABLISHED+ schemas have contradictory statements.
    Embedding cosine similarity < -0.3 triggers this.
    """

    schema_a_id: str
    schema_b_id: str
    schema_a_statement: str = ""
    schema_b_statement: str = ""
    cosine_similarity: float = 0.0
    resolved: bool = False
    resolution_note: str = ""
