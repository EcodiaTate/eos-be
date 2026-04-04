"""
EcodiaOS - Thread (Narrative Identity) API Router

Exposes the organism's autobiographical self — its life story, active chapter,
identity schemas, commitments, and narrative coherence. This is the organism's
self-knowledge about who it is, who it has been, and who it is becoming.

  GET  /api/v1/thread/story          - Current life story / autobiography
  GET  /api/v1/thread/identity       - Brief identity context (chapter, coherence, schemas)
  GET  /api/v1/thread/commitments    - Outstanding commitments and fidelity
  GET  /api/v1/thread/health         - Thread system health and summary metrics
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = structlog.get_logger("api.thread")

router = APIRouter(prefix="/api/v1/thread", tags=["thread"])


def _get_thread(request: Request) -> Any:
    """Resolve ThreadService from app state."""
    return getattr(request.app.state, "thread", None)


@router.get("/story")
async def get_story(request: Request) -> JSONResponse:
    """Return the organism's current narrative — autobiography and life story."""
    thread = _get_thread(request)
    if thread is None:
        return JSONResponse({"story": "Thread system not initialized.", "chapter": None, "schemas": []})

    story = thread.get_current_story()
    identity = thread.get_identity_context()

    # Gather active schemas
    schemas = []
    for s in getattr(thread, "_schemas", []):
        schemas.append({
            "statement": s.statement,
            "status": s.status.value if hasattr(s.status, "value") else str(s.status),
            "confirmations": getattr(s, "confirmation_count", 0),
        })

    # Get active chapter
    chapter = None
    active_chapter = thread._get_active_chapter() if hasattr(thread, "_get_active_chapter") else None
    if active_chapter:
        chapter = {
            "title": active_chapter.title,
            "theme": getattr(active_chapter, "theme", None),
            "arc_type": getattr(active_chapter, "arc_type", None),
            "summary": getattr(active_chapter, "summary", None),
        }

    # Life story synthesis
    life_story = None
    if hasattr(thread, "_life_story") and thread._life_story:
        ls = thread._life_story
        life_story = {
            "synthesis": getattr(ls, "synthesis", None),
            "themes": getattr(ls, "themes", None),
        }

    return JSONResponse({
        "story": story,
        "identity_context": identity,
        "chapter": chapter,
        "life_story": life_story,
        "schemas": schemas[:20],
        "total_schemas": len(schemas),
    })


@router.get("/identity")
async def get_identity(request: Request) -> JSONResponse:
    """Brief identity context for the organism."""
    thread = _get_thread(request)
    if thread is None:
        return JSONResponse({"identity": "Thread not initialized.", "coherence": None})

    identity = thread.get_identity_context()
    coherence = thread._compute_identity_coherence() if hasattr(thread, "_compute_identity_coherence") else None

    return JSONResponse({
        "identity": identity,
        "coherence": coherence,
    })


@router.get("/commitments")
async def get_commitments(request: Request) -> JSONResponse:
    """Outstanding commitments with fidelity tracking."""
    thread = _get_thread(request)
    if thread is None:
        return JSONResponse({"commitments": [], "violations": []})

    commitments = thread.get_outstanding_commitments()
    violations = thread.get_commitment_violations() if hasattr(thread, "get_commitment_violations") else []

    return JSONResponse({
        "commitments": commitments[:20],
        "violations": violations[:10],
    })


@router.get("/health")
async def get_health(request: Request) -> JSONResponse:
    """Thread system health and summary metrics."""
    thread = _get_thread(request)
    if thread is None:
        return JSONResponse({"initialized": False})

    schema_count = len(getattr(thread, "_schemas", []))
    chapter_count = len(getattr(thread, "_chapters", []))
    fingerprint_count = len(getattr(thread, "_fingerprints", []))

    active_chapter = thread._get_active_chapter() if hasattr(thread, "_get_active_chapter") else None
    coherence = thread._compute_identity_coherence() if hasattr(thread, "_compute_identity_coherence") else None

    return JSONResponse({
        "initialized": True,
        "schemas": schema_count,
        "chapters": chapter_count,
        "fingerprints": fingerprint_count,
        "active_chapter": active_chapter.title if active_chapter else None,
        "identity_coherence": coherence,
        "has_life_story": bool(getattr(thread, "_life_story", None)),
    })
