"""
EcodiaOS — RE Training Data Exporter

Collects RE_TRAINING_EXAMPLE events from the Synapse bus, assembles them
into hourly RETrainingExportBatch objects, and ships them to:
  1. S3 / local filesystem — JSON lines for the offline CLoRA fine-tuning pipeline
  2. Neo4j — (:RETrainingBatch) nodes with [:CONTAINS]→(:RETrainingDatapoint) edges
              for audit lineage and Benchmarks tracking

The exporter runs as a supervised background task started in Phase 11 of
registry.py.  It is intentionally decoupled from every cognitive system —
it only reads from the event bus ring buffer + subscribes to
RE_TRAINING_EXAMPLE events.  No direct imports from any system module.

Export cadence: every 3600s (1 hour).  Batches with 0 datapoints are skipped.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from primitives.re_training import RETrainingDatapoint, RETrainingExportBatch
from primitives.common import new_id

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from clients.redis import RedisClient
    from systems.synapse.event_bus import EventBus
    from systems.synapse.types import SynapseEvent

logger = structlog.get_logger("core.re_training_exporter")

# How many seconds in each collection window
_EXPORT_INTERVAL_S: int = 3600  # 1 hour

# Local fallback export path (used when S3 is unavailable)
_LOCAL_EXPORT_DIR: str = os.environ.get(
    "RE_TRAINING_EXPORT_DIR", "data/re_training_batches"
)

# S3 config from environment
_S3_BUCKET: str = os.environ.get("RE_TRAINING_S3_BUCKET", "ecodiaos-re-training")
_S3_PREFIX: str = os.environ.get("RE_TRAINING_S3_PREFIX", "batches/")


def _outcome_from_quality(quality: float) -> str:
    """Map [0,1] outcome_quality → human-readable outcome label."""
    if quality >= 0.85:
        return "success"
    if quality >= 0.4:
        return "partial"
    return "failure"


def _datapoint_from_event(event: SynapseEvent) -> RETrainingDatapoint | None:
    """
    Convert a raw RE_TRAINING_EXAMPLE SynapseEvent into a RETrainingDatapoint.

    Returns None if the payload is malformed or missing required fields.
    The conversion is best-effort — never raises.
    """
    try:
        data = event.data
        quality = float(data.get("outcome_quality", 0.0))
        return RETrainingDatapoint(
            source_system=str(data.get("source_system", event.source_system)),
            example_type=str(data.get("category", "unknown")),
            instruction=str(data.get("instruction", ""))[:2000],
            input_context=str(data.get("input_context", ""))[:4000],
            output_action=str(data.get("output", ""))[:2000],
            outcome=_outcome_from_quality(quality),
            confidence=min(max(quality, 0.0), 1.0),
            timestamp=event.timestamp,
            reasoning_trace=str(data.get("reasoning_trace", ""))[:1000],
            alternatives_considered=list(data.get("alternatives_considered", [])),
            cost_usd=Decimal(str(data.get("cost_usd", "0"))),
            latency_ms=int(data.get("latency_ms", 0)),
            episode_id=str(data.get("episode_id", "")),
        )
    except Exception:
        logger.debug("re_datapoint_conversion_failed", exc_info=True)
        return None


class RETrainingExporter:
    """
    Subscribes to RE_TRAINING_EXAMPLE events, assembles hourly export batches,
    and ships them to S3 + Neo4j.

    Lifecycle:
        exporter = RETrainingExporter(event_bus, neo4j, redis)
        exporter.attach()          # subscribe to event bus
        await exporter.run_loop()  # blocking; call via supervised_task
        exporter.detach()          # unsubscribe (called on shutdown)

    Thread safety: all state is accessed only from the asyncio event loop.
    """

    def __init__(
        self,
        event_bus: EventBus,
        neo4j: Neo4jClient | None = None,
        redis: RedisClient | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._neo4j = neo4j
        self._redis = redis
        # In-memory accumulator for the current window
        self._pending: list[RETrainingDatapoint] = []
        # Dedup: (source_system, episode_id) — episode_id="" means no dedup
        self._seen_episode_ids: set[str] = set()
        self._window_start: datetime = datetime.now(UTC)
        self._total_exported = 0
        self._total_batches = 0
        self._attached = False

    # ─── Event Bus Integration ────────────────────────────────────────

    def attach(self) -> None:
        """Subscribe to RE_TRAINING_EXAMPLE events on the bus."""
        from systems.synapse.types import SynapseEventType

        self._event_bus.subscribe(
            SynapseEventType.RE_TRAINING_EXAMPLE,
            self._on_re_training_example,
        )
        self._attached = True
        logger.info("re_training_exporter_attached")

    def detach(self) -> None:
        """Unsubscribe (best-effort — EventBus may not support unsubscribe)."""
        self._attached = False
        logger.info("re_training_exporter_detached")

    async def _on_re_training_example(self, event: SynapseEvent) -> None:
        """Hot-path handler — called by EventBus on every RE_TRAINING_EXAMPLE event."""
        dp = _datapoint_from_event(event)
        if dp is None:
            return

        # Dedup episodes: skip if we've already captured this episode in this window
        if dp.episode_id:
            dedup_key = f"{dp.source_system}:{dp.episode_id}"
            if dedup_key in self._seen_episode_ids:
                return
            self._seen_episode_ids.add(dedup_key)

        self._pending.append(dp)

    # ─── Batch Collection ─────────────────────────────────────────────

    def collect_batch(self) -> RETrainingExportBatch:
        """
        Drain the current accumulator into an RETrainingExportBatch.

        Resets the accumulator and dedup set for the next window.
        Returns an empty batch if no examples were collected.
        """
        now = datetime.now(UTC)
        hour_window = self._window_start.strftime("%Y-%m-%dT%H:00:00Z")

        datapoints = list(self._pending)
        source_systems = sorted({dp.source_system for dp in datapoints})

        # Reset for next window
        self._pending = []
        self._seen_episode_ids = set()
        self._window_start = now

        return RETrainingExportBatch(
            id=new_id(),
            datapoints=datapoints,
            hour_window=hour_window,
            source_systems=source_systems,
        )

    # ─── Export Destinations ──────────────────────────────────────────

    async def export_to_s3(self, batch: RETrainingExportBatch) -> bool:
        """
        Export batch as JSON lines to S3 (or local filesystem fallback).

        Returns True if the export succeeded, False otherwise.
        S3 upload uses boto3 if available; falls back to local file write.
        """
        if not batch.datapoints:
            return True

        lines = "\n".join(
            json.dumps(dp.model_dump(mode="json"), default=str)
            for dp in batch.datapoints
        )
        filename = f"{batch.hour_window.replace(':', '-')}_{batch.id}.jsonl"

        # Try S3 first
        try:
            import boto3  # type: ignore[import]

            s3 = boto3.client("s3")
            key = f"{_S3_PREFIX}{filename}"
            s3.put_object(
                Bucket=_S3_BUCKET,
                Key=key,
                Body=lines.encode("utf-8"),
                ContentType="application/x-ndjson",
            )
            batch.export_destinations.append(f"s3://{_S3_BUCKET}/{key}")
            logger.info(
                "re_training_exported_s3",
                batch_id=batch.id,
                examples=batch.total_examples,
                key=key,
            )
            return True
        except ImportError:
            logger.debug("boto3_not_available_falling_back_to_local")
        except Exception:
            logger.warning("re_training_s3_export_failed", exc_info=True)

        # Local filesystem fallback
        try:
            export_dir = Path(_LOCAL_EXPORT_DIR)
            export_dir.mkdir(parents=True, exist_ok=True)
            dest = export_dir / filename
            dest.write_text(lines, encoding="utf-8")
            batch.export_destinations.append(f"local://{dest}")
            logger.info(
                "re_training_exported_local",
                batch_id=batch.id,
                examples=batch.total_examples,
                path=str(dest),
            )
            return True
        except Exception:
            logger.error("re_training_local_export_failed", exc_info=True)
            return False

    async def sync_to_memory(self, batch: RETrainingExportBatch) -> None:
        """
        Write batch lineage to Neo4j as (:RETrainingBatch) + (:RETrainingDatapoint)
        nodes.  Non-fatal — failures are logged and swallowed.
        """
        if self._neo4j is None or not batch.datapoints:
            return
        try:
            await self._neo4j.execute_read(
                """
                MERGE (b:RETrainingBatch {id: $batch_id})
                SET b.hour_window     = $hour_window,
                    b.total_examples  = $total_examples,
                    b.mean_quality    = $mean_quality,
                    b.source_systems  = $source_systems,
                    b.destinations    = $destinations,
                    b.created_at      = $created_at
                """,
                {
                    "batch_id": batch.id,
                    "hour_window": batch.hour_window,
                    "total_examples": batch.total_examples,
                    "mean_quality": round(batch.mean_quality, 4),
                    "source_systems": batch.source_systems,
                    "destinations": batch.export_destinations,
                    "created_at": batch.created_at.isoformat(),
                },
            )
            # Write a lightweight summary per source system (not every datapoint —
            # that would be 100+ nodes/hour; batch-level is sufficient for lineage).
            for system in batch.source_systems:
                system_dps = [dp for dp in batch.datapoints if dp.source_system == system]
                mean_q = sum(dp.confidence for dp in system_dps) / len(system_dps)
                await self._neo4j.execute_read(
                    """
                    MATCH (b:RETrainingBatch {id: $batch_id})
                    MERGE (s:RETrainingSource {system: $system, batch_id: $batch_id})
                    SET s.count      = $count,
                        s.mean_quality = $mean_quality
                    MERGE (b)-[:CONTAINS_SOURCE]->(s)
                    """,
                    {
                        "batch_id": batch.id,
                        "system": system,
                        "count": len(system_dps),
                        "mean_quality": round(mean_q, 4),
                    },
                )
            logger.info(
                "re_training_synced_to_neo4j",
                batch_id=batch.id,
                examples=batch.total_examples,
            )
        except Exception:
            logger.warning("re_training_neo4j_sync_failed", exc_info=True)

    # ─── Export Broadcast ────────────────────────────────────────────

    async def _emit_export_complete(self, batch: RETrainingExportBatch) -> None:
        """Emit RE_TRAINING_EXPORT_COMPLETE on the Synapse bus."""
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXPORT_COMPLETE,
                source_system="re_training_exporter",
                data={
                    "batch_id": batch.id,
                    "total_examples": batch.total_examples,
                    "source_systems": batch.source_systems,
                    "mean_quality": round(batch.mean_quality, 4),
                    "export_destinations": batch.export_destinations,
                    "export_duration_ms": batch.export_duration_ms,
                    "hour_window": batch.hour_window,
                },
            ))
        except Exception:
            logger.debug("re_training_export_emit_failed", exc_info=True)

    # ─── Main Export Cycle ────────────────────────────────────────────

    async def export_cycle(self) -> RETrainingExportBatch:
        """
        Run a single collect → export → sync cycle.

        Called once per hour by run_loop(). Safe to call directly for testing.
        """
        t0 = time.monotonic()
        batch = self.collect_batch()

        if batch.total_examples == 0:
            logger.debug("re_training_export_skipped_empty_window", hour=batch.hour_window)
            return batch

        await self.export_to_s3(batch)
        await self.sync_to_memory(batch)

        batch.export_duration_ms = int((time.monotonic() - t0) * 1000)
        self._total_exported += batch.total_examples
        self._total_batches += 1

        await self._emit_export_complete(batch)

        logger.info(
            "re_training_export_cycle_complete",
            batch_id=batch.id,
            examples=batch.total_examples,
            systems=batch.source_systems,
            mean_quality=round(batch.mean_quality, 4),
            duration_ms=batch.export_duration_ms,
            total_exported=self._total_exported,
        )
        return batch

    # ─── Background Loop ──────────────────────────────────────────────

    async def run_loop(self) -> None:
        """
        Supervised background coroutine.  Runs indefinitely, exporting once per hour.

        To be wrapped in utils.supervision.supervised_task() by registry.py.
        """
        logger.info(
            "re_training_export_loop_started",
            interval_s=_EXPORT_INTERVAL_S,
            s3_bucket=_S3_BUCKET,
        )
        while True:
            await asyncio.sleep(_EXPORT_INTERVAL_S)
            await self.export_cycle()

    # ─── Stats ───────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "pending_examples": len(self._pending),
            "total_exported": self._total_exported,
            "total_batches": self._total_batches,
            "window_start": self._window_start.isoformat(),
            "seen_episode_ids": len(self._seen_episode_ids),
            "attached": self._attached,
        }
