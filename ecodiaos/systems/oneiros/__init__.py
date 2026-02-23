"""
EcodiaOS — Oneiros: The Dream Engine & Circadian Architecture

System #13. The organism's capacity to sleep, dream, and wake up changed.

Thymos gave the organism a will to live. Oneiros gives it an inner life.
"""

from ecodiaos.systems.oneiros.service import OneirosService
from ecodiaos.systems.oneiros.types import (
    CircadianPhase,
    Dream,
    DreamCoherence,
    DreamCycleResult,
    DreamInsight,
    DreamType,
    InsightStatus,
    LucidResult,
    NREMConsolidationResult,
    OneirosHealthSnapshot,
    REMDreamResult,
    SleepCycle,
    SleepPressure,
    SleepQuality,
    SleepStage,
    WakeDegradation,
)

__all__ = [
    "OneirosService",
    "SleepStage",
    "DreamType",
    "DreamCoherence",
    "InsightStatus",
    "SleepQuality",
    "SleepPressure",
    "CircadianPhase",
    "Dream",
    "DreamInsight",
    "SleepCycle",
    "NREMConsolidationResult",
    "REMDreamResult",
    "LucidResult",
    "DreamCycleResult",
    "WakeDegradation",
    "OneirosHealthSnapshot",
]
