"""
EcodiaOS — Evo (Learning & Hypothesis System)

Evo is the Growth drive made computational. It observes patterns across
experience, forms hypotheses, accumulates evidence, and when evidence is
sufficient, adjusts the organism's parameters, codifies procedures, and
proposes structural evolution.

Evo operates in two modes:
  WAKE — lightweight online pattern detection during each cognitive cycle
  SLEEP — deep offline consolidation: schema induction, procedure extraction,
           parameter optimisation, self-model update, drift monitoring

Guard rails:
  - Cannot touch Equor evaluation logic or constitutional drives
  - All parameter changes are small (velocity-limited)
  - Hypotheses must be falsifiable
  - Evolution proposals go to Simula for gating, not applied directly

Public interface:
  EvoService          — main service class
  ConsolidationResult — result of a consolidation cycle
  SelfModelStats      — self-assessment metrics
  ParameterTuner      — tunable parameter management (for direct access)
"""

from ecodiaos.systems.evo.service import EvoService
from ecodiaos.systems.evo.types import (
    TUNABLE_PARAMETERS,
    VELOCITY_LIMITS,
    ConsolidationResult,
    Hypothesis,
    HypothesisCategory,
    HypothesisStatus,
    ParameterAdjustment,
    PatternCandidate,
    PatternType,
    Procedure,
    SelfModelStats,
)

__all__ = [
    "EvoService",
    "ConsolidationResult",
    "Hypothesis",
    "HypothesisCategory",
    "HypothesisStatus",
    "ParameterAdjustment",
    "PatternCandidate",
    "PatternType",
    "Procedure",
    "SelfModelStats",
    "TUNABLE_PARAMETERS",
    "VELOCITY_LIMITS",
]
