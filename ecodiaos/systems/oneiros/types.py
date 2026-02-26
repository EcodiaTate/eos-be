"""
EcodiaOS — Oneiros Type Definitions

All data types for the dream engine: sleep stages, dreams, insights,
consolidation results, sleep debt, and circadian phases.

Every dream, every insight, and every sleep cycle is a first-class
primitive — the organism's inner life made observable.
"""

from __future__ import annotations

from datetime import datetime
import enum
from typing import Any

from pydantic import Field

from ecodiaos.primitives.common import EOSBaseModel, new_id, utc_now


# ─── Enums ────────────────────────────────────────────────────────


class SleepStage(enum.StrEnum):
    """States of consciousness in the circadian cycle."""

    WAKE = "wake"                   # Normal cognitive cycle
    HYPNAGOGIA = "hypnagogia"       # Transition in (~30s)
    NREM = "nrem"                   # Consolidation (40% of sleep)
    REM = "rem"                     # Creative dreaming (40%)
    LUCID = "lucid"                 # Self-directed dreaming (10%)
    HYPNOPOMPIA = "hypnopompia"    # Transition out (~30s)


class DreamType(enum.StrEnum):
    """What kind of dream is this?"""

    RECOMBINATION = "recombination"             # Random co-activation bridge
    THREAT_REHEARSAL = "threat_rehearsal"        # Hypothetical failure simulation
    AFFECT_PROCESSING = "affect_processing"     # Emotional charge dampening
    ETHICAL_RUMINATION = "ethical_rumination"    # Constitutional edge case
    LUCID_EXPLORATION = "lucid_exploration"      # Directed creative variation
    META_OBSERVATION = "meta_observation"        # Self-observing dream patterns


class DreamCoherence(enum.StrEnum):
    """How meaningful was the dream's creative bridge?"""

    INSIGHT = "insight"       # High coherence — genuine creative discovery
    FRAGMENT = "fragment"     # Medium — store for future recombination
    NOISE = "noise"           # Low — random noise, discard


class InsightStatus(enum.StrEnum):
    """Lifecycle of a dream insight in the waking world."""

    PENDING = "pending"           # Not yet validated in wake
    VALIDATED = "validated"       # Confirmed useful in wake state
    INVALIDATED = "invalidated"   # Turned out to be noise
    INTEGRATED = "integrated"     # Became permanent semantic knowledge


class SleepQuality(enum.StrEnum):
    """How restful was this sleep cycle?"""

    DEEP = "deep"               # Full cycle, all stages completed
    NORMAL = "normal"           # Standard quality
    FRAGMENTED = "fragmented"   # Interrupted, partial consolidation
    DEPRIVED = "deprived"       # Emergency wake, minimal benefit


# ─── Sleep Pressure ───────────────────────────────────────────────


class SleepPressure(EOSBaseModel):
    """
    Homeostatic sleep drive — rises with wake time and cognitive load.

    Like adenosine accumulation in biological brains, sleep pressure
    builds during wakefulness from four independent sources. When it
    crosses the threshold, the organism must sleep.
    """

    # Raw counters
    cycles_since_sleep: int = 0
    unprocessed_affect_residue: float = 0.0     # Sum of high-affect traces
    unconsolidated_episode_count: int = 0
    hypothesis_backlog: int = 0

    # Computed
    composite_pressure: float = 0.0             # 0.0 (rested) → 1.0+ (exhausted)

    # Thresholds
    threshold: float = 0.70                     # Triggers DROWSY signal
    critical_threshold: float = 0.95            # Forces sleep unconditionally

    # Tracking
    last_sleep_completed: datetime | None = None
    last_computation: datetime = Field(default_factory=utc_now)


class CircadianPhase(EOSBaseModel):
    """Current position in the circadian cycle."""

    wake_duration_target_s: float = 79200.0     # 22 hours default
    sleep_duration_target_s: float = 7200.0     # 2 hours default
    current_phase: SleepStage = SleepStage.WAKE
    phase_elapsed_s: float = 0.0
    total_cycles_completed: int = 0


# ─── Dreams ───────────────────────────────────────────────────────


class Dream(EOSBaseModel):
    """
    A single dream experience.

    Dreams emerge from the intersection of what happened recently
    (episodic replay), what's emotionally charged (affect residue),
    random activation (noise → creativity), and what the organism
    is uncertain about (predictive model gaps).

    Every dream is recorded. The organism can see its own dream
    patterns over time — a therapist for its own psyche.
    """

    id: str = Field(default_factory=new_id)
    dream_type: DreamType
    sleep_cycle_id: str
    timestamp: datetime = Field(default_factory=utc_now)

    # Source traces
    seed_episode_ids: list[str] = Field(default_factory=list)
    activated_episode_ids: list[str] = Field(default_factory=list)

    # Creative bridge
    bridge_narrative: str = ""                  # LLM-generated connection text
    coherence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coherence_class: DreamCoherence = DreamCoherence.NOISE

    # Affect
    affect_valence: float = 0.0
    affect_arousal: float = 0.0

    # Semantics
    themes: list[str] = Field(default_factory=list)
    summary: str = ""

    # Context
    context: dict[str, Any] = Field(default_factory=dict)


class DreamInsight(EOSBaseModel):
    """
    A high-coherence dream discovery.

    When a dream produces a genuinely meaningful connection between
    distant memories, that connection becomes a DreamInsight. Insights
    are queued for broadcast on the first wake cycle, where they enter
    the Global Workspace like any other percept.

    Over time, validated insights become part of the organism's
    semantic memory — creative knowledge that emerges from sleep.
    """

    id: str = Field(default_factory=new_id)
    dream_id: str
    sleep_cycle_id: str

    # Content
    insight_text: str
    insight_embedding: list[float] | None = None
    coherence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    domain: str = ""                            # What area this concerns

    # Lifecycle
    status: InsightStatus = InsightStatus.PENDING
    validated_at: datetime | None = None
    validation_context: str = ""                # How it was validated
    wake_applications: int = 0                  # Times used in wake decisions

    # Source context
    seed_summary: str = ""
    activated_summary: str = ""
    bridge_narrative: str = ""

    # Timestamps
    created_at: datetime = Field(default_factory=utc_now)


# ─── Sleep Cycles ─────────────────────────────────────────────────


class SleepCycle(EOSBaseModel):
    """
    Record of a complete sleep cycle.

    Each cycle is a journey through NREM (consolidation), REM
    (creative dreaming), and optionally LUCID (self-directed
    exploration). The metrics accumulated here are the organism's
    sleep diary — observable, queryable, learnable.
    """

    id: str = Field(default_factory=new_id)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    quality: SleepQuality = SleepQuality.NORMAL
    interrupted: bool = False
    interrupt_reason: str = ""

    # ── NREM Metrics ──
    episodes_replayed: int = 0
    semantic_nodes_created: int = 0
    traces_pruned: int = 0
    salience_reduction_mean: float = 0.0
    beliefs_compressed: int = 0
    hypotheses_pruned: int = 0
    hypotheses_promoted: int = 0

    # ── REM Metrics ──
    dreams_generated: int = 0
    insights_discovered: int = 0
    affect_traces_processed: int = 0
    affect_reduction_mean: float = 0.0
    threats_simulated: int = 0
    ethical_cases_digested: int = 0

    # ── Lucid Metrics ──
    lucid_explorations: int = 0
    meta_observations: int = 0

    # ── Pressure ──
    pressure_before: float = 0.0
    pressure_after: float = 0.0


# ─── Consolidation Results ────────────────────────────────────────


class NREMConsolidationResult(EOSBaseModel):
    """Result of the complete NREM consolidation phase."""

    # Episodic replay
    episodes_replayed: int = 0
    semantic_nodes_created: int = 0
    replay_duration_ms: int = 0

    # Synaptic downscaling
    traces_decayed: int = 0
    traces_pruned: int = 0
    mean_salience_reduction: float = 0.0
    downscale_duration_ms: int = 0

    # Belief compression
    beliefs_merged: int = 0
    beliefs_archived: int = 0
    beliefs_flagged_contradictory: int = 0
    compression_duration_ms: int = 0

    # Hypothesis pruning
    hypotheses_retired: int = 0
    hypotheses_promoted: int = 0
    hypotheses_merged: int = 0
    pruning_duration_ms: int = 0

    total_duration_ms: int = 0


class REMDreamResult(EOSBaseModel):
    """Result of the complete REM dreaming phase."""

    # Dream generation
    dreams_generated: int = 0
    insights_discovered: int = 0
    fragments_stored: int = 0
    noise_discarded: int = 0
    dream_duration_ms: int = 0

    # Affect processing
    affect_traces_processed: int = 0
    mean_valence_reduction: float = 0.0
    mean_arousal_reduction: float = 0.0
    coherence_stress_reduction: float = 0.0
    affect_duration_ms: int = 0

    # Threat simulation
    threats_simulated: int = 0
    response_plans_created: int = 0
    prophylactic_antibodies: int = 0
    threat_duration_ms: int = 0

    # Ethical digestion
    ethical_cases_digested: int = 0
    heuristics_refined: int = 0
    ethical_duration_ms: int = 0

    total_duration_ms: int = 0


class LucidResult(EOSBaseModel):
    """Result of the lucid dreaming phase."""

    explorations_completed: int = 0
    variations_generated: int = 0
    high_value_variations: int = 0
    meta_observations: int = 0
    recurring_themes_detected: int = 0
    self_knowledge_nodes_created: int = 0
    total_duration_ms: int = 0


class DreamCycleResult(EOSBaseModel):
    """Result of a single dream within REM."""

    dream: Dream
    insight: DreamInsight | None = None
    affect_delta: float = 0.0       # Change in coherence_stress
    duration_ms: int = 0


# ─── Wake Degradation ─────────────────────────────────────────────


class WakeDegradation(EOSBaseModel):
    """
    Current degradation effects from sleep deprivation.

    These are not simulated penalties — they are actual multipliers
    applied to the respective systems. The organism genuinely
    thinks worse when sleep-deprived.
    """

    salience_noise: float = 0.0             # Added noise to salience scoring (0.0-0.15)
    efe_precision_loss: float = 0.0         # Reduced Nova EFE precision (0.0-0.20)
    expression_flatness: float = 0.0        # Reduced Voxis personality (0.0-0.25)
    learning_rate_reduction: float = 0.0    # Reduced Evo learning rate (0.0-0.30)
    composite_impairment: float = 0.0       # Overall impairment (0.0-1.0)

    @classmethod
    def from_pressure(
        cls,
        pressure: float,
        threshold: float,
        critical: float,
        *,
        noise_max: float = 0.15,
        efe_max: float = 0.20,
        flatness_max: float = 0.25,
        learning_max: float = 0.30,
    ) -> WakeDegradation:
        """Compute degradation from current sleep pressure."""
        if pressure <= threshold:
            return cls()

        impairment = min(1.0, max(0.0, (pressure - threshold) / (critical - threshold)))
        return cls(
            salience_noise=impairment * noise_max,
            efe_precision_loss=impairment * efe_max,
            expression_flatness=impairment * flatness_max,
            learning_rate_reduction=impairment * learning_max,
            composite_impairment=impairment,
        )


# ─── Health Snapshot ──────────────────────────────────────────────


class OneirosHealthSnapshot(EOSBaseModel):
    """Oneiros system health and observability."""

    status: str = "healthy"
    current_stage: SleepStage = SleepStage.WAKE

    # Sleep pressure
    sleep_pressure: float = 0.0
    wake_degradation: float = 0.0
    current_sleep_debt_hours: float = 0.0

    # Lifetime metrics
    total_sleep_cycles: int = 0
    total_dreams: int = 0
    total_insights: int = 0
    insights_validated: int = 0
    insights_invalidated: int = 0
    insights_integrated: int = 0
    mean_dream_coherence: float = 0.0
    mean_sleep_quality: float = 0.0

    # Consolidation metrics
    episodes_consolidated: int = 0
    semantic_nodes_created: int = 0
    traces_pruned: int = 0
    hypotheses_pruned: int = 0
    hypotheses_promoted: int = 0

    # Affect processing
    affect_traces_processed: int = 0
    mean_affect_reduction: float = 0.0

    # Threat simulation
    threats_simulated: int = 0
    response_plans_created: int = 0

    # Last sleep
    last_sleep_completed: datetime | None = None
    last_sleep_quality: SleepQuality | None = None

    timestamp: datetime = Field(default_factory=utc_now)
