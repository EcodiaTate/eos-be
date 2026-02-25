"""
EcodiaOS — Soma (Interoceptive Predictive Substrate)

The organism's felt sense of being alive. Predicts internal states,
computes allostatic errors, and emits signals that drive regulation.
"""

from ecodiaos.systems.soma.service import SomaService
from ecodiaos.systems.soma.types import (
    AllostaticSignal,
    Attractor,
    Bifurcation,
    CounterfactualTrace,
    DevelopmentalStage,
    InteroceptiveDimension,
    InteroceptiveState,
    SomaticMarker,
)

__all__ = [
    "SomaService",
    "AllostaticSignal",
    "Attractor",
    "Bifurcation",
    "CounterfactualTrace",
    "DevelopmentalStage",
    "InteroceptiveDimension",
    "InteroceptiveState",
    "SomaticMarker",
]
