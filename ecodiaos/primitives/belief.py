"""
EcodiaOS — Belief Primitive

The fundamental unit of internal state — what EOS "thinks".
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from ecodiaos.primitives.common import Identified, utc_now


class Belief(Identified):
    """A probability distribution over possible states."""

    domain: str = ""                          # e.g., "user.emotional_state"
    distribution_type: str = "categorical"    # "categorical" | "gaussian" | "point"
    parameters: dict[str, float] = Field(default_factory=dict)
    precision: float = 0.5                    # Inverse variance — confidence
    evidence: list[str] = Field(default_factory=list)  # Percept/Belief IDs
    updated_at: datetime = Field(default_factory=utc_now)
    free_energy: float = 0.0                  # Current prediction error
