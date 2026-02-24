"""
Tests for Oneiros Circadian Clock and Sleep Stage Controller.

Covers:
  - CircadianClock: pressure computation, thresholds, reset, degradation
  - SleepStageController: state transitions, full cycles, emergency wake
"""

from __future__ import annotations

import pytest

from ecodiaos.systems.oneiros.circadian import (
    CircadianClock,
    SleepStageController,
    _DEFAULT_WEIGHT_AFFECT,
    _DEFAULT_WEIGHT_CYCLES,
    _DEFAULT_WEIGHT_EPISODES,
    _DEFAULT_WEIGHT_HYPOTHESES,
)
from ecodiaos.systems.oneiros.types import (
    SleepQuality,
    SleepStage,
    WakeDegradation,
)


# ─── Helpers ─────────────────────────────────────────────────────


def _make_clock(**overrides) -> CircadianClock:
    """Create a CircadianClock with optional config overrides."""
    config = {
        "pressure_threshold": 0.70,
        "pressure_critical": 0.95,
        "max_wake_cycles": 528000,
    }
    config.update(overrides)
    return CircadianClock(config)


def _make_controller(**overrides) -> SleepStageController:
    """Create a SleepStageController with optional config overrides."""
    config = {
        "sleep_duration_target_s": 7200.0,
        "hypnagogia_duration_s": 30.0,
        "hypnopompia_duration_s": 30.0,
        "nrem_fraction": 0.40,
        "rem_fraction": 0.40,
        "lucid_fraction": 0.10,
    }
    config.update(overrides)
    return SleepStageController(config)


# ─── CircadianClock ──────────────────────────────────────────────


class TestCircadianClock:
    def test_initial_pressure_zero(self):
        clock = _make_clock()
        assert clock.pressure.composite_pressure == 0.0
        assert clock.pressure.cycles_since_sleep == 0
        assert clock.pressure.unprocessed_affect_residue == 0.0
        assert clock.pressure.unconsolidated_episode_count == 0
        assert clock.pressure.hypothesis_backlog == 0

    def test_tick_increments_cycles(self):
        clock = _make_clock()
        clock.tick()
        assert clock.pressure.cycles_since_sleep == 1
        clock.tick()
        assert clock.pressure.cycles_since_sleep == 2

    def test_pressure_rises_with_cycles(self):
        clock = _make_clock(max_wake_cycles=100)
        for _ in range(50):
            clock.tick()
        pressure = clock.pressure.composite_pressure
        assert pressure > 0.0
        # 50/100 = 0.5 cycle ratio, weighted by 0.40 = 0.20
        assert pressure >= 0.15

    def test_pressure_rises_with_affect(self):
        clock = _make_clock()
        # Record a high-valence trace (above 0.7 threshold)
        clock.record_affect_trace(valence=0.9, arousal=0.5)
        pressure = clock.compute_pressure()
        assert pressure > 0.0

    def test_pressure_rises_with_episodes(self):
        clock = _make_clock(episode_capacity=10)
        for _ in range(5):
            clock.record_episode()
        pressure = clock.compute_pressure()
        # 5/10 = 0.5, weighted by 0.20 = 0.10
        assert pressure > 0.0

    def test_pressure_rises_with_hypotheses(self):
        clock = _make_clock(hypothesis_capacity=10)
        clock.record_hypothesis_count(5)
        pressure = clock.compute_pressure()
        # 5/10 = 0.5, weighted by 0.15 = 0.075
        assert pressure > 0.0

    def test_pressure_clamped_at_max(self):
        clock = _make_clock(max_wake_cycles=10, episode_capacity=1, hypothesis_capacity=1)
        # Saturate all pressure sources
        for _ in range(100):
            clock.tick()
            clock.record_episode()
        clock.record_hypothesis_count(100)
        clock.record_affect_trace(valence=1.0, arousal=1.0)
        pressure = clock.compute_pressure()
        assert pressure <= 1.5

    def test_should_sleep_threshold(self):
        clock = _make_clock(max_wake_cycles=100, pressure_threshold=0.30)
        # Push cycles to just above threshold
        for _ in range(80):
            clock.tick()
        assert clock.should_sleep() is True

    def test_must_sleep_critical(self):
        clock = _make_clock(
            max_wake_cycles=100,
            pressure_threshold=0.20,
            pressure_critical=0.35,
        )
        for _ in range(100):
            clock.tick()
        assert clock.must_sleep() is True

    def test_not_sleepy_below_threshold(self):
        clock = _make_clock(max_wake_cycles=528000)
        clock.tick()
        assert clock.should_sleep() is False
        assert clock.must_sleep() is False

    def test_reset_deep_quality(self):
        clock = _make_clock(max_wake_cycles=100)
        for _ in range(100):
            clock.tick()
        pressure_before = clock.pressure.composite_pressure
        assert pressure_before > 0.0

        clock.reset_after_sleep(SleepQuality.DEEP)
        # 100% reset means cycles_since_sleep goes to 0
        assert clock.pressure.cycles_since_sleep == 0
        assert clock.pressure.composite_pressure < pressure_before

    def test_reset_normal_quality(self):
        clock = _make_clock(max_wake_cycles=100)
        for _ in range(100):
            clock.tick()
        clock.reset_after_sleep(SleepQuality.NORMAL)
        # 90% reset: cycles should be ~10% of original
        # int truncation of float: int(100 * (1.0 - 0.9)) may be 9 or 10
        assert clock.pressure.cycles_since_sleep <= 10
        assert clock.pressure.cycles_since_sleep >= 9

    def test_reset_fragmented_quality(self):
        clock = _make_clock(max_wake_cycles=100)
        for _ in range(100):
            clock.tick()
        clock.reset_after_sleep(SleepQuality.FRAGMENTED)
        # 60% reset: cycles should be 40% of original = 40
        assert clock.pressure.cycles_since_sleep == 40

    def test_reset_deprived_quality(self):
        clock = _make_clock(max_wake_cycles=100)
        for _ in range(100):
            clock.tick()
        clock.reset_after_sleep(SleepQuality.DEPRIVED)
        # 30% reset: cycles should be 70% of original = 70
        assert clock.pressure.cycles_since_sleep == 70

    def test_degradation_zero_when_rested(self):
        clock = _make_clock()
        deg = clock.degradation
        assert deg.composite_impairment == 0.0
        assert deg.salience_noise == 0.0
        assert deg.efe_precision_loss == 0.0
        assert deg.expression_flatness == 0.0
        assert deg.learning_rate_reduction == 0.0

    def test_degradation_rises_with_pressure(self):
        clock = _make_clock(max_wake_cycles=100, pressure_threshold=0.30, pressure_critical=0.40)
        for _ in range(90):
            clock.tick()
        deg = clock.degradation
        assert deg.composite_impairment > 0.0
        assert deg.salience_noise > 0.0

    def test_degradation_max_at_critical(self):
        clock = _make_clock(max_wake_cycles=100, pressure_threshold=0.20, pressure_critical=0.40)
        for _ in range(100):
            clock.tick()
        deg = clock.degradation
        assert deg.composite_impairment == 1.0
        assert deg.salience_noise == 0.15
        assert deg.efe_precision_loss == 0.20
        assert deg.expression_flatness == 0.25
        assert deg.learning_rate_reduction == 0.30

    def test_affect_below_threshold_ignored(self):
        clock = _make_clock()
        # Valence below 0.7, arousal below 0.8 — should not register
        clock.record_affect_trace(valence=0.3, arousal=0.3)
        assert clock.pressure.unprocessed_affect_residue == 0.0

    def test_pressure_weights_sum_to_one(self):
        total = (
            _DEFAULT_WEIGHT_CYCLES
            + _DEFAULT_WEIGHT_AFFECT
            + _DEFAULT_WEIGHT_EPISODES
            + _DEFAULT_WEIGHT_HYPOTHESES
        )
        assert abs(total - 1.0) < 1e-9


# ─── SleepStageController ───────────────────────────────────────


class TestSleepStageController:
    def test_initial_stage_wake(self):
        ctrl = _make_controller()
        assert ctrl.current_stage == SleepStage.WAKE
        assert ctrl.is_sleeping is False

    def test_begin_sleep_transitions_to_hypnagogia(self):
        ctrl = _make_controller()
        ctrl.begin_sleep("cycle_1")
        assert ctrl.current_stage == SleepStage.HYPNAGOGIA
        assert ctrl.is_sleeping is True

    def test_advance_hypnagogia_to_nrem(self):
        ctrl = _make_controller(hypnagogia_duration_s=10.0)
        ctrl.begin_sleep("cycle_1")
        assert ctrl.current_stage == SleepStage.HYPNAGOGIA

        new_stage = ctrl.advance(11.0)
        assert new_stage == SleepStage.NREM
        assert ctrl.current_stage == SleepStage.NREM

    def test_advance_nrem_to_rem(self):
        ctrl = _make_controller(
            hypnagogia_duration_s=1.0,
            hypnopompia_duration_s=1.0,
            sleep_duration_target_s=100.0,
            nrem_fraction=0.40,
            rem_fraction=0.40,
            lucid_fraction=0.10,
        )
        ctrl.begin_sleep("cycle_1")
        # Advance past hypnagogia
        ctrl.advance(2.0)
        assert ctrl.current_stage == SleepStage.NREM

        # effective_sleep = 100 - 1 - 1 = 98
        # nrem_end_s = 1 + 98 * 0.40 = 40.2
        # sleep_elapsed is 2.0 after first advance; need >= 40.2 total
        ctrl.advance(39.0)  # sleep_elapsed = 41.0 > 40.2
        assert ctrl.current_stage == SleepStage.REM

    def test_advance_rem_to_hypnopompia_no_creative_goal(self):
        ctrl = _make_controller(
            hypnagogia_duration_s=1.0,
            hypnopompia_duration_s=1.0,
            sleep_duration_target_s=100.0,
        )
        ctrl.set_has_creative_goal(False)
        ctrl.begin_sleep("cycle_1")

        # effective = 98, nrem_end = 1 + 39.2 = 40.2, rem_end = 40.2 + 39.2 = 79.4
        ctrl.advance(2.0)   # past hypnagogia -> NREM
        ctrl.advance(40.0)  # past NREM -> REM
        assert ctrl.current_stage == SleepStage.REM

        ctrl.advance(40.0)  # past REM -> HYPNOPOMPIA (no creative goal)
        assert ctrl.current_stage == SleepStage.HYPNOPOMPIA

    def test_advance_rem_to_lucid_with_creative_goal(self):
        ctrl = _make_controller(
            hypnagogia_duration_s=1.0,
            hypnopompia_duration_s=1.0,
            sleep_duration_target_s=100.0,
        )
        ctrl.set_has_creative_goal(True)
        ctrl.begin_sleep("cycle_1")

        ctrl.advance(2.0)   # past hypnagogia -> NREM
        ctrl.advance(40.0)  # past NREM -> REM
        assert ctrl.current_stage == SleepStage.REM

        ctrl.advance(40.0)  # past REM -> LUCID (creative goal set)
        assert ctrl.current_stage == SleepStage.LUCID

    def test_advance_lucid_to_hypnopompia(self):
        ctrl = _make_controller(
            hypnagogia_duration_s=1.0,
            hypnopompia_duration_s=1.0,
            sleep_duration_target_s=100.0,
        )
        ctrl.set_has_creative_goal(True)
        ctrl.begin_sleep("cycle_1")

        ctrl.advance(2.0)   # -> NREM
        ctrl.advance(40.0)  # -> REM
        ctrl.advance(40.0)  # -> LUCID
        assert ctrl.current_stage == SleepStage.LUCID

        # lucid_end_s = rem_end_s + 98 * 0.10 = 79.4 + 9.8 = 89.2
        ctrl.advance(10.0)  # -> HYPNOPOMPIA
        assert ctrl.current_stage == SleepStage.HYPNOPOMPIA

    def test_advance_hypnopompia_to_wake(self):
        ctrl = _make_controller(
            hypnagogia_duration_s=1.0,
            hypnopompia_duration_s=5.0,
            sleep_duration_target_s=100.0,
        )
        ctrl.set_has_creative_goal(False)
        ctrl.begin_sleep("cycle_1")

        ctrl.advance(2.0)   # -> NREM
        ctrl.advance(40.0)  # -> REM
        ctrl.advance(40.0)  # -> HYPNOPOMPIA
        assert ctrl.current_stage == SleepStage.HYPNOPOMPIA

        ctrl.advance(6.0)   # -> WAKE
        assert ctrl.current_stage == SleepStage.WAKE
        assert ctrl.is_sleeping is False

    def test_full_cycle_without_lucid(self):
        ctrl = _make_controller(
            hypnagogia_duration_s=1.0,
            hypnopompia_duration_s=1.0,
            sleep_duration_target_s=100.0,
        )
        ctrl.set_has_creative_goal(False)
        ctrl.begin_sleep("full_cycle_1")

        stages_visited = [SleepStage.HYPNAGOGIA]

        ctrl.advance(2.0)
        stages_visited.append(ctrl.current_stage)

        ctrl.advance(40.0)
        stages_visited.append(ctrl.current_stage)

        ctrl.advance(40.0)
        stages_visited.append(ctrl.current_stage)

        ctrl.advance(2.0)
        stages_visited.append(ctrl.current_stage)

        assert stages_visited == [
            SleepStage.HYPNAGOGIA,
            SleepStage.NREM,
            SleepStage.REM,
            SleepStage.HYPNOPOMPIA,
            SleepStage.WAKE,
        ]

    def test_full_cycle_with_lucid(self):
        ctrl = _make_controller(
            hypnagogia_duration_s=1.0,
            hypnopompia_duration_s=1.0,
            sleep_duration_target_s=100.0,
        )
        ctrl.set_has_creative_goal(True)
        ctrl.begin_sleep("full_cycle_2")

        stages_visited = [SleepStage.HYPNAGOGIA]

        ctrl.advance(2.0)
        stages_visited.append(ctrl.current_stage)

        ctrl.advance(40.0)
        stages_visited.append(ctrl.current_stage)

        ctrl.advance(40.0)
        stages_visited.append(ctrl.current_stage)

        ctrl.advance(10.0)
        stages_visited.append(ctrl.current_stage)

        ctrl.advance(2.0)
        stages_visited.append(ctrl.current_stage)

        assert stages_visited == [
            SleepStage.HYPNAGOGIA,
            SleepStage.NREM,
            SleepStage.REM,
            SleepStage.LUCID,
            SleepStage.HYPNOPOMPIA,
            SleepStage.WAKE,
        ]

    def test_emergency_wake_during_nrem(self):
        ctrl = _make_controller(hypnagogia_duration_s=1.0)
        ctrl.begin_sleep("cycle_1")
        ctrl.advance(2.0)  # -> NREM
        assert ctrl.current_stage == SleepStage.NREM

        ctrl.emergency_wake("critical_incident")
        assert ctrl.current_stage == SleepStage.HYPNOPOMPIA

    def test_emergency_wake_during_rem(self):
        ctrl = _make_controller(
            hypnagogia_duration_s=1.0,
            sleep_duration_target_s=100.0,
        )
        ctrl.begin_sleep("cycle_1")
        ctrl.advance(2.0)   # -> NREM
        ctrl.advance(40.0)  # -> REM
        assert ctrl.current_stage == SleepStage.REM

        ctrl.emergency_wake("system_failure")
        assert ctrl.current_stage == SleepStage.HYPNOPOMPIA

    def test_is_sleeping_property(self):
        ctrl = _make_controller()
        assert ctrl.is_sleeping is False

        ctrl.begin_sleep("cycle_1")
        assert ctrl.is_sleeping is True

    def test_begin_sleep_when_already_sleeping_warns(self):
        ctrl = _make_controller()
        ctrl.begin_sleep("cycle_1")
        assert ctrl.current_stage == SleepStage.HYPNAGOGIA

        # Calling begin_sleep again should not change state (logged warning)
        ctrl.begin_sleep("cycle_2")
        assert ctrl.current_stage == SleepStage.HYPNAGOGIA

    def test_stage_elapsed_resets_on_transition(self):
        ctrl = _make_controller(hypnagogia_duration_s=10.0)
        ctrl.begin_sleep("cycle_1")
        assert ctrl.stage_elapsed_s == 0.0

        ctrl.advance(5.0)
        assert ctrl.stage_elapsed_s == 5.0

        # Transition resets elapsed
        ctrl.advance(6.0)  # total 11 > 10 hypnagogia duration -> NREM
        assert ctrl.current_stage == SleepStage.NREM
        assert ctrl.stage_elapsed_s == 0.0
