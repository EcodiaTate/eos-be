"""
EcodiaOS - Corpus Knowledge Ingestion Router

Ingests the local knowledge corpus (CC session logs, spec docs, CLAUDE.md files,
memory files) into the organism's Neo4j knowledge graph as Episodes.

Endpoints:
  POST /api/v1/corpus/ingest               - Batch ingest corpus records
  POST /api/v1/corpus/ingest/session-log   - Ingest a single CC session log
  GET  /api/v1/corpus/status               - Corpus ingestion stats from Neo4j
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request

from primitives.common import EOSBaseModel, Modality, SourceDescriptor, SystemID
from primitives.percept import Content, Percept

logger = structlog.get_logger("api.corpus_ingestion")

router = APIRouter(prefix="/api/v1/corpus", tags=["corpus"])

# ─── Salience table ──────────────────────────────────────────────

_SALIENCE: dict[str, float] = {
    "spec": 0.85,
    "claude_md": 0.80,
    "memory": 0.75,
    "philosophy": 0.75,
    "skill": 0.65,
    "session_log": 0.60,
    "doc": 0.55,
}

_BATCH_SIZE = 20
_CONTENT_CHAR_LIMIT = 8000

# ─── Request / Response models ───────────────────────────────────


class CorpusRecord(EOSBaseModel):
    id: str
    source: str
    type: str
    content: str
    chars: int
    collected_at: str


class CorpusIngestRequest(EOSBaseModel):
    records: list[CorpusRecord]
    source_label: str = "local_corpus"
    deduplicate: bool = True


class SessionLogRecord(EOSBaseModel):
    ts: str
    tool: str
    input_summary: str
    output_summary: str
    session_id: str


class SessionLogIngestRequest(EOSBaseModel):
    records: list[SessionLogRecord]
    session_id: str
    session_date: str


class CorpusIngestResponse(EOSBaseModel):
    ingested: int
    skipped_duplicate: int
    skipped_starvation: int
    failed: int
    episode_ids: list[str]
    duration_ms: int


class CorpusSourceCount(EOSBaseModel):
    source: str
    count: int


class CorpusStatusResponse(EOSBaseModel):
    total_corpus_episodes: int
    latest_ingest: str | None
    corpus_types: list[str]
    top_sources: list[CorpusSourceCount]


# ─── Helpers ─────────────────────────────────────────────────────


def _build_percept(
    raw: str,
    corpus_id: str,
    corpus_source: str,
    corpus_type: str,
    source_label: str,
    chars: int,
) -> Percept:
    return Percept(
        source=SourceDescriptor(
            system=SystemID.API,
            channel="corpus_ingestion",
            modality=Modality.TEXT,
        ),
        content=Content(raw=raw[:_CONTENT_CHAR_LIMIT]),
        metadata={
            "corpus_id": corpus_id,
            "corpus_source": corpus_source,
            "corpus_type": corpus_type,
            "source_label": source_label,
            "chars": chars,
        },
    )


async def _tag_episode(
    memory: Any,
    episode_id: str,
    corpus_id: str,
    corpus_source: str,
    corpus_type: str,
) -> None:
    """Stamp corpus_id / corpus_source / corpus_type onto the Episode node for dedup."""
    await memory.execute_write(
        """
        MATCH (ep:Episode {id: $episode_id})
        SET ep.corpus_id = $corpus_id,
            ep.corpus_source = $corpus_source,
            ep.corpus_type = $corpus_type
        """,
        {
            "episode_id": episode_id,
            "corpus_id": corpus_id,
            "corpus_source": corpus_source,
            "corpus_type": corpus_type,
        },
    )


async def _find_existing_corpus_ids(memory: Any, ids: list[str]) -> set[str]:
    """Return the subset of corpus IDs already present in the graph."""
    rows = await memory.execute_read(
        """
        UNWIND $ids AS cid
        MATCH (ep:Episode {corpus_id: cid})
        RETURN cid
        """,
        {"ids": ids},
    )
    return {row["cid"] for row in rows}


# ─── Routes ──────────────────────────────────────────────────────


@router.post("/ingest", response_model=CorpusIngestResponse)
async def ingest_corpus(
    body: CorpusIngestRequest,
    request: Request,
) -> CorpusIngestResponse:
    """
    Ingest a batch of corpus records into the organism's Neo4j knowledge graph.
    Each record becomes an Episode tagged with corpus metadata.
    """
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialised")

    t0 = time.monotonic()

    ingested = 0
    skipped_duplicate = 0
    skipped_starvation = 0
    failed = 0
    episode_ids: list[str] = []

    records = body.records

    # Batch deduplication check
    existing_ids: set[str] = set()
    if body.deduplicate and records:
        try:
            existing_ids = await _find_existing_corpus_ids(
                memory, [r.id for r in records]
            )
        except Exception as exc:
            logger.warning("corpus_dedup_check_failed", error=str(exc))

    # Process in batches of _BATCH_SIZE
    for batch_start in range(0, len(records), _BATCH_SIZE):
        batch = records[batch_start : batch_start + _BATCH_SIZE]

        for record in batch:
            if body.deduplicate and record.id in existing_ids:
                skipped_duplicate += 1
                continue

            salience = _SALIENCE.get(record.type, 0.55)
            percept = _build_percept(
                raw=record.content,
                corpus_id=record.id,
                corpus_source=record.source,
                corpus_type=record.type,
                source_label=body.source_label,
                chars=record.chars,
            )

            try:
                episode_id = await memory.store_percept(
                    percept=percept,
                    salience_composite=salience,
                    context_summary=f"corpus:{record.type}:{record.source}",
                )
            except Exception as exc:
                logger.warning(
                    "corpus_store_percept_failed",
                    corpus_id=record.id,
                    error=str(exc),
                )
                failed += 1
                continue

            if not episode_id:
                # Empty string → starvation gate blocked the write
                skipped_starvation += 1
                continue

            # Tag the episode node for dedup on future ingestions
            try:
                await _tag_episode(
                    memory,
                    episode_id,
                    record.id,
                    record.source,
                    record.type,
                )
            except Exception as exc:
                logger.warning(
                    "corpus_tag_episode_failed",
                    episode_id=episode_id,
                    corpus_id=record.id,
                    error=str(exc),
                )

            episode_ids.append(episode_id)
            ingested += 1

        # Yield the event loop between batches
        await asyncio.sleep(0)

    duration_ms = round((time.monotonic() - t0) * 1000)

    logger.info(
        "corpus_ingest_complete",
        ingested=ingested,
        skipped_duplicate=skipped_duplicate,
        skipped_starvation=skipped_starvation,
        failed=failed,
        source_label=body.source_label,
        duration_ms=duration_ms,
    )

    return CorpusIngestResponse(
        ingested=ingested,
        skipped_duplicate=skipped_duplicate,
        skipped_starvation=skipped_starvation,
        failed=failed,
        episode_ids=episode_ids,
        duration_ms=duration_ms,
    )


@router.post("/ingest/session-log", response_model=CorpusIngestResponse)
async def ingest_session_log(
    body: SessionLogIngestRequest,
    request: Request,
) -> CorpusIngestResponse:
    """
    Ingest a CC session log into the knowledge graph.
    Each log line becomes an Episode tagged as training data.
    """
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialised")

    t0 = time.monotonic()

    ingested = 0
    skipped_starvation = 0
    failed = 0
    episode_ids: list[str] = []

    for batch_start in range(0, len(body.records), _BATCH_SIZE):
        batch = body.records[batch_start : batch_start + _BATCH_SIZE]

        for log_rec in batch:
            content = f"{log_rec.tool}: {log_rec.input_summary} → {log_rec.output_summary}"
            corpus_id = f"sl:{body.session_id}:{log_rec.ts}"

            percept = Percept(
                source=SourceDescriptor(
                    system=SystemID.API,
                    channel="corpus_ingestion",
                    modality=Modality.TEXT,
                ),
                content=Content(raw=content[:_CONTENT_CHAR_LIMIT]),
                metadata={
                    "corpus_id": corpus_id,
                    "corpus_source": f"session_logs/{body.session_id}.jsonl",
                    "corpus_type": "session_log",
                    "source_label": "local_corpus",
                    "session_id": body.session_id,
                    "session_date": body.session_date,
                    "tool": log_rec.tool,
                },
            )

            try:
                episode_id = await memory.store_percept(
                    percept=percept,
                    salience_composite=_SALIENCE["session_log"],
                    context_summary=f"corpus:session_log:{body.session_id}",
                )
            except Exception as exc:
                logger.warning(
                    "session_log_store_percept_failed",
                    session_id=body.session_id,
                    error=str(exc),
                )
                failed += 1
                continue

            if not episode_id:
                skipped_starvation += 1
                continue

            try:
                await _tag_episode(
                    memory,
                    episode_id,
                    corpus_id,
                    f"session_logs/{body.session_id}.jsonl",
                    "session_log",
                )
            except Exception as exc:
                logger.warning(
                    "session_log_tag_episode_failed",
                    episode_id=episode_id,
                    error=str(exc),
                )

            episode_ids.append(episode_id)
            ingested += 1

        await asyncio.sleep(0)

    duration_ms = round((time.monotonic() - t0) * 1000)

    logger.info(
        "session_log_ingest_complete",
        session_id=body.session_id,
        ingested=ingested,
        skipped_starvation=skipped_starvation,
        failed=failed,
        duration_ms=duration_ms,
    )

    return CorpusIngestResponse(
        ingested=ingested,
        skipped_duplicate=0,
        skipped_starvation=skipped_starvation,
        failed=failed,
        episode_ids=episode_ids,
        duration_ms=duration_ms,
    )


@router.get("/status", response_model=CorpusStatusResponse)
async def corpus_status(request: Request) -> CorpusStatusResponse:
    """Return corpus ingestion stats: episode count, latest ingest time, top sources."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialised")

    try:
        summary_rows = await memory.execute_read(
            """
            MATCH (ep:Episode) WHERE ep.corpus_id IS NOT NULL
            RETURN count(ep) AS total,
                   max(ep.event_time) AS latest_ingest,
                   collect(DISTINCT ep.corpus_type) AS types
            """,
        )

        total = 0
        latest_ingest: str | None = None
        corpus_types: list[str] = []

        if summary_rows:
            row = summary_rows[0]
            total = row.get("total", 0) or 0
            raw_latest = row.get("latest_ingest")
            latest_ingest = str(raw_latest) if raw_latest is not None else None
            corpus_types = [t for t in (row.get("types") or []) if t]

        source_rows = await memory.execute_read(
            """
            MATCH (ep:Episode) WHERE ep.corpus_source IS NOT NULL
            RETURN ep.corpus_source AS source, count(ep) AS count
            ORDER BY count DESC LIMIT 20
            """,
        )

        top_sources = [
            CorpusSourceCount(
                source=r.get("source", ""),
                count=r.get("count", 0),
            )
            for r in (source_rows or [])
            if r.get("source")
        ]

        return CorpusStatusResponse(
            total_corpus_episodes=total,
            latest_ingest=latest_ingest,
            corpus_types=corpus_types,
            top_sources=top_sources,
        )

    except Exception as exc:
        logger.warning("corpus_status_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
