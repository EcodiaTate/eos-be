"""
EcodiaOS - Learning Coherence Monitor

Correlates learning signals from four subsystems to determine whether the
organism is learning coherently (all systems improving together) or
incoherently (local optimization masking global regression).

Signal sources:
  - EVO_CONSOLIDATION_QUALITY  -> hypothesis promotion/pruning + KPI deltas
  - ONEIROS_SLEEP_OUTCOME      -> post-sleep verdict (beneficial/neutral/harmful)
  - BENCHMARK_REGRESSION       -> KPI regression detected (negative signal)
  - NEXUS_EPISTEMIC_VALUE      -> epistemic triangulation score trend

Emits ORGANISM_LEARNING_COHERENCE every 5 minutes with a composite
coherence_score (0-1) and direction (ascending/stable/declining).

Safety: Pure event aggregation. No parameter changes. Non-fatal throughout.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from systems.synapse.event_bus import EventBus
from systems.synapse.types import SynapseEvent, SynapseEventType

logger = logging.getLogger("ecodiaos.observatory.learning_coherence")

# Observation window: 30 minutes
_WINDOW_S = 30 * 60

# Assessment interval: 5 minutes
_ASSESS_INTERVAL_S = 5 * 60

# Direction thresholds
_ASCENDING_THRESHOLD = 0.3   # >30% positive signals -> ascending
_DECLINING_THRESHOLD = -0.3  # <-30% -> declining


@dataclass
class _LearningSignal:
    """A single directional learning signal from a subsystem."""

    timestamp: float  # monotonic
    event_type: str
    direction: float  # +1.0 positive, -1.0 negative, 0.0 neutral
    source_system: str
    detail: str = ""


class LearningCoherenceMonitor:
    """
    Subscribes to four learning-related event streams and periodically
    emits an ORGANISM_LEARNING_COHERENCE event summarising whether
    the organism's learning subsystems agree on direction.

    Attach via `monitor.attach(event_bus)`. Non-fatal: all handlers
    are wrapped in contextlib.suppress.
    """

    def __init__(self) -> None:
        self._signals: deque[_LearningSignal] = deque()
        self._event_bus: EventBus | None = None
        self._task: asyncio.Task[None] | None = None
        self._attached = False
        # Track Nexus triangulation scores for trend detection
        self._nexus_scores: deque[tuple[float, float]] = deque(maxlen=20)

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to learning events and start the assessment loop."""
        if self._attached:
            return
        self._event_bus = event_bus
        self._attached = True

        # Subscribe to the four learning signal sources
        with contextlib.suppress(Exception):
            event_bus.subscribe(
                SynapseEventType.EVO_CONSOLIDATION_QUALITY,
                self._on_evo_consolidation,
            )
        with contextlib.suppress(Exception):
            event_bus.subscribe(
                SynapseEventType.ONEIROS_SLEEP_OUTCOME,
                self._on_oneiros_outcome,
            )
        with contextlib.suppress(Exception):
            event_bus.subscribe(
                SynapseEventType.BENCHMARK_REGRESSION,
                self._on_benchmark_regression,
            )
        with contextlib.suppress(Exception):
            event_bus.subscribe(
                SynapseEventType.NEXUS_EPISTEMIC_VALUE,
                self._on_nexus_epistemic,
            )

        # Start periodic assessment
        self._task = asyncio.ensure_future(self._assessment_loop())
        logger.info("learning_coherence_monitor_attached")

    async def stop(self) -> None:
        """Cancel the assessment loop."""
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    # ── Event handlers ──────────────────────────────────────────────

    def _on_evo_consolidation(self, event: SynapseEvent) -> None:
        """EVO_CONSOLIDATION_QUALITY: positive if net KPI improvement."""
        with contextlib.suppress(Exception):
            payload = event.payload or {}
            deltas = payload.get("improvement_delta", {})
            promoted = payload.get("hypotheses_promoted", 0)
            pruned = payload.get("hypotheses_pruned", 0)

            # Net direction from KPI deltas
            if deltas:
                net = sum(deltas.values())
                direction = 1.0 if net > 0 else (-1.0 if net < 0 else 0.0)
            elif promoted > pruned:
                direction = 0.5  # Mild positive: more promoted than pruned
            else:
                direction = 0.0

            self._signals.append(_LearningSignal(
                timestamp=time.monotonic(),
                event_type=SynapseEventType.EVO_CONSOLIDATION_QUALITY,
                direction=direction,
                source_system="evo",
                detail=f"deltas={deltas}, promoted={promoted}, pruned={pruned}",
            ))

    def _on_oneiros_outcome(self, event: SynapseEvent) -> None:
        """ONEIROS_SLEEP_OUTCOME: verdict maps to direction."""
        with contextlib.suppress(Exception):
            payload = event.payload or {}
            verdict = payload.get("verdict", "neutral")
            direction_map = {"beneficial": 1.0, "neutral": 0.0, "harmful": -1.0}
            direction = direction_map.get(verdict, 0.0)

            self._signals.append(_LearningSignal(
                timestamp=time.monotonic(),
                event_type=SynapseEventType.ONEIROS_SLEEP_OUTCOME,
                direction=direction,
                source_system="oneiros",
                detail=f"verdict={verdict}, net_improvement={payload.get('net_improvement', 0)}",
            ))

    def _on_benchmark_regression(self, event: SynapseEvent) -> None:
        """BENCHMARK_REGRESSION: always a negative learning signal."""
        with contextlib.suppress(Exception):
            payload = event.payload or {}
            metric = payload.get("metric", "unknown")
            regression_pct = payload.get("regression_pct", 0)

            self._signals.append(_LearningSignal(
                timestamp=time.monotonic(),
                event_type=SynapseEventType.BENCHMARK_REGRESSION,
                direction=-1.0,
                source_system="benchmarks",
                detail=f"metric={metric}, regression_pct={regression_pct}",
            ))

    def _on_nexus_epistemic(self, event: SynapseEvent) -> None:
        """NEXUS_EPISTEMIC_VALUE: track trend of triangulation scores."""
        with contextlib.suppress(Exception):
            payload = event.payload or {}
            score = payload.get("triangulation_score", 0.0)
            now = time.monotonic()
            self._nexus_scores.append((now, score))

            # Compute trend from recent scores
            if len(self._nexus_scores) >= 3:
                recent = list(self._nexus_scores)[-5:]
                first_avg = sum(s for _, s in recent[: len(recent) // 2]) / max(len(recent) // 2, 1)
                last_avg = sum(s for _, s in recent[len(recent) // 2 :]) / max(len(recent) - len(recent) // 2, 1)
                delta = last_avg - first_avg
                if abs(delta) < 0.02:
                    direction = 0.0
                else:
                    direction = 1.0 if delta > 0 else -1.0
            else:
                # Not enough data for trend
                direction = 0.0

            self._signals.append(_LearningSignal(
                timestamp=time.monotonic(),
                event_type=SynapseEventType.NEXUS_EPISTEMIC_VALUE,
                direction=direction,
                source_system="nexus",
                detail=f"score={score:.3f}",
            ))

    # ── Assessment loop ─────────────────────────────────────────────

    async def _assessment_loop(self) -> None:
        """Periodically assess learning coherence and emit results."""
        # Wait one full window before first assessment
        await asyncio.sleep(_ASSESS_INTERVAL_S)

        while True:
            with contextlib.suppress(Exception):
                self._assess_and_emit()
            await asyncio.sleep(_ASSESS_INTERVAL_S)

    def _assess_and_emit(self) -> None:
        """Compute coherence score from recent signals and emit event."""
        now = time.monotonic()
        cutoff = now - _WINDOW_S

        # Evict old signals
        while self._signals and self._signals[0].timestamp < cutoff:
            self._signals.popleft()

        signals = list(self._signals)

        if not signals:
            # No learning activity in the window - emit stable with zero
            self._emit_coherence(
                coherence_score=0.5,
                direction="stable",
                signals_aligned=0,
                signals_total=0,
                supporting_events=[],
            )
            return

        # Group signals by source system, take most recent per source
        latest_by_source: dict[str, _LearningSignal] = {}
        for sig in signals:
            if sig.source_system not in latest_by_source or sig.timestamp > latest_by_source[sig.source_system].timestamp:
                latest_by_source[sig.source_system] = sig

        # Compute weighted mean direction
        total_direction = sum(s.direction for s in signals)
        mean_direction = total_direction / len(signals) if signals else 0.0

        # Count agreement with majority direction
        if mean_direction > 0:
            majority = 1.0
        elif mean_direction < 0:
            majority = -1.0
        else:
            majority = 0.0

        aligned = sum(
            1 for s in latest_by_source.values()
            if (majority == 0.0) or (s.direction * majority > 0)
        )
        total_sources = len(latest_by_source)

        # Coherence score: fraction of sources that agree on direction
        if total_sources > 0:
            coherence_score = aligned / total_sources
        else:
            coherence_score = 0.5

        # Determine direction label
        if mean_direction > _ASCENDING_THRESHOLD:
            direction = "ascending"
        elif mean_direction < _DECLINING_THRESHOLD:
            direction = "declining"
        else:
            direction = "stable"

        # Collect supporting event types
        supporting = [
            s.event_type
            for s in latest_by_source.values()
            if (majority == 0.0) or (s.direction * majority > 0)
        ]

        self._emit_coherence(
            coherence_score=coherence_score,
            direction=direction,
            signals_aligned=aligned,
            signals_total=total_sources,
            supporting_events=supporting,
        )

    def _emit_coherence(
        self,
        coherence_score: float,
        direction: str,
        signals_aligned: int,
        signals_total: int,
        supporting_events: list[str],
    ) -> None:
        """Emit the ORGANISM_LEARNING_COHERENCE event."""
        if not self._event_bus:
            return

        payload = {
            "coherence_score": round(coherence_score, 3),
            "direction": direction,
            "signals_aligned": signals_aligned,
            "signals_total": signals_total,
            "supporting_events": supporting_events,
            "window_seconds": _WINDOW_S,
            "assessed_at": datetime.now(timezone.utc).isoformat(),
        }

        with contextlib.suppress(Exception):
            self._event_bus.emit(
                SynapseEventType.ORGANISM_LEARNING_COHERENCE,
                payload,
                source_system="observatory",
            )
            logger.info(
                "learning_coherence_assessed",
                extra={
                    "coherence_score": payload["coherence_score"],
                    "direction": direction,
                    "aligned": f"{signals_aligned}/{signals_total}",
                },
            )

    # ── Introspection ───────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return current state for the /health or /observatory API."""
        now = time.monotonic()
        cutoff = now - _WINDOW_S
        active = [s for s in self._signals if s.timestamp >= cutoff]

        sources = set(s.source_system for s in active)
        return {
            "attached": self._attached,
            "signals_in_window": len(active),
            "active_sources": sorted(sources),
            "window_seconds": _WINDOW_S,
            "assess_interval_seconds": _ASSESS_INTERVAL_S,
        }
