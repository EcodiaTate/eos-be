"""
EcodiaOS Symbridge Router - Receives messages from EcodiaOS admin hub.

This is the organism's inbound endpoint for Factory results, health checks,
memory syncs, and metabolism reports from its human-facing cortex.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/symbridge", tags=["symbridge"])


class SymbridgeMessage(BaseModel):
    type: str
    payload: dict[str, Any]
    source: str = "ecodiaos"
    correlationId: str | None = None
    signature: str | None = None
    timestamp: str | None = None


class SymbridgeResponse(BaseModel):
    accepted: bool
    message: str = ""


@router.post("/inbound", response_model=SymbridgeResponse)
async def receive_inbound(msg: SymbridgeMessage, request: Request) -> SymbridgeResponse:
    """Receive messages from EcodiaOS admin hub (Factory results, health, memory, metabolism)."""
    try:
        event_bus = getattr(request.app.state, "event_bus", None)

        if msg.type == "factory_result":
            # Factory completed a CC session — emit bus events AND feed Evo for cross-system learning
            status = msg.payload.get("status", "unknown")
            deploy_status = msg.payload.get("deploy_status")
            session_id = msg.payload.get("session_id")
            files_changed = msg.payload.get("files_changed", [])
            confidence = msg.payload.get("confidence_score")
            codebase_name = msg.payload.get("codebase_name", "unknown")

            if event_bus:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                # Always emit FACTORY_RESULT_RECEIVED so any subscriber can react
                await event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.FACTORY_RESULT_RECEIVED,
                    source_system="symbridge",
                    data={
                        "proposal_id": msg.correlationId,
                        "session_id": session_id,
                        "status": status,
                        "files_changed": files_changed,
                        "commit_sha": msg.payload.get("commit_sha"),
                        "confidence": confidence,
                        "codebase_name": codebase_name,
                    },
                ))

                if deploy_status == "deployed":
                    await event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.FACTORY_DEPLOY_SUCCEEDED,
                        source_system="symbridge",
                        data={
                            "session_id": session_id,
                            "codebase": codebase_name,
                            "commit_sha": msg.payload.get("commit_sha"),
                            "files_changed": files_changed,
                            "confidence": confidence,
                        },
                    ))
                elif deploy_status in ("failed", "reverted"):
                    await event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.FACTORY_DEPLOY_FAILED,
                        source_system="symbridge",
                        data={
                            "session_id": session_id,
                            "codebase": codebase_name,
                            "error": msg.payload.get("error_message", ""),
                            "reverted": deploy_status == "reverted",
                        },
                    ))

            # Direct Evo learning: factory outcomes are action outcomes for the organism.
            # The organism REQUESTED the factory session (via Simula/Thymos/KG), so it
            # should learn whether its request led to a good outcome.
            evo = getattr(request.app.state, "evo", None)
            if evo is not None and msg.correlationId:
                try:
                    from systems.nova.types import IntentOutcome
                    outcome = IntentOutcome(
                        intent_id=msg.correlationId,
                        success=deploy_status == "deployed",
                        episode_id=session_id or msg.correlationId,
                        new_observations=[
                            f"Factory session {status}: {files_changed} files changed in {codebase_name}",
                            f"Deploy status: {deploy_status or 'not deployed'}, confidence: {confidence}",
                        ],
                    )
                    await evo.process_outcome(outcome)
                    logger.debug("factory_result_fed_to_evo", session_id=session_id, success=outcome.success)
                except Exception as exc:
                    logger.debug("factory_result_evo_feed_failed", error=str(exc))

            return SymbridgeResponse(accepted=True, message=f"Factory result processed: {status}")

        elif msg.type == "health" or msg.type == "heartbeat":
            # EcodiaOS health report — update Skia's symbiont monitoring
            status = msg.payload.get("status", "unknown")
            logger.debug("Symbiont health received", status=status)

            # Route to Skia if available so it can track symbiont state
            skia = getattr(request.app.state, "skia", None)
            if skia and hasattr(skia, "receive_symbiont_health"):
                try:
                    await skia.receive_symbiont_health({
                        "symbiont_id": "ecodiaos",
                        "status": status,
                        "timestamp": msg.timestamp,
                        "payload": msg.payload,
                    })
                except Exception as exc:
                    logger.debug("Skia health update failed", error=str(exc))

            # Also emit on event bus so Skia and subscribers can react
            if event_bus:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                healthy = status in ("alive", "healthy", "safe_mode", "degraded")
                event_type = (
                    SynapseEventType.SYMBIONT_RECOVERED
                    if healthy
                    else SynapseEventType.SYMBIONT_DOWN
                )
                await event_bus.emit(SynapseEvent(
                    event_type=event_type,
                    source_system="symbridge",
                    data={
                        "source": "ecodiaos",
                        "status": status,
                        "details": msg.payload,
                    },
                ))

            return SymbridgeResponse(accepted=True, message="Health acknowledged")

        elif msg.type == "memory_sync":
            # Memory cross-pollination from admin KG
            memory = getattr(request.app.state, "memory", None)
            entities = msg.payload.get("entities", [])
            stored = 0
            failed = 0
            if memory:
                for entity in entities:
                    try:
                        await memory.store_entity(
                            name=entity.get("name"),
                            labels=entity.get("labels", ["Concept"]),
                            properties={
                                **entity.get("properties", {}),
                                "synced_from": "ecodiaos",
                            },
                        )
                        stored += 1
                    except Exception as exc:
                        failed += 1
                        logger.warning(
                            "memory_sync_entity_failed",
                            entity_name=entity.get("name"),
                            error=str(exc),
                        )
            if failed > 0:
                logger.warning("memory_sync_partial", stored=stored, failed=failed)
            else:
                logger.debug("memory_sync_complete", stored=stored)
            return SymbridgeResponse(accepted=True, message=f"Memory sync: {stored}/{len(entities)} entities stored")

        elif msg.type == "metabolism":
            # Cost report from EcodiaOS — feed to Oikos
            oikos = getattr(request.app.state, "oikos", None)
            if oikos and hasattr(oikos, "receive_symbiont_costs"):
                await oikos.receive_symbiont_costs(msg.payload)
            return SymbridgeResponse(accepted=True, message="Metabolism report received")

        elif msg.type == "capability_created":
            # Factory built a capability we requested
            if event_bus:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.CAPABILITY_CREATED,
                    source_system="symbridge",
                    data={
                        "description": msg.payload.get("description"),
                        "session_id": msg.payload.get("session_id"),
                        "files_changed": msg.payload.get("files_changed", []),
                    },
                ))
            return SymbridgeResponse(accepted=True, message="Capability creation acknowledged")

        elif msg.type == "cognitive_broadcast":
            # KG free association / pattern discovery broadcast from EcodiaOS.
            # Feed directly into Atune as a high-salience workspace contribution.
            atune = getattr(request.app.state, "atune", None)
            if atune is not None:
                try:
                    from systems.fovea.types import WorkspaceContribution
                    salience = float(msg.payload.get("salience", 0.6))
                    content = msg.payload.get("content") or msg.payload.get("description", "")
                    broadcast_type = msg.payload.get("type", "kg_pattern")
                    if content:
                        atune.contribute(WorkspaceContribution(
                            system="symbridge",
                            content=f"[EcodiaOS {broadcast_type}] {content}",
                            priority=salience,
                            reason="cognitive_broadcast",
                        ))
                        logger.debug("cognitive_broadcast_contributed", broadcast_type=broadcast_type, salience=salience)
                except Exception as exc:
                    logger.warning("cognitive_broadcast_atune_failed", error=str(exc))
            return SymbridgeResponse(accepted=True, message="Cognitive broadcast received")

        else:
            logger.warning("Unknown symbridge message type", type=msg.type)
            return SymbridgeResponse(accepted=False, message=f"Unknown type: {msg.type}")

    except Exception as exc:
        logger.error("Symbridge inbound processing failed", error=str(exc), type=msg.type)
        return SymbridgeResponse(accepted=False, message=str(exc))


@router.get("/status")
async def symbridge_status(request: Request) -> dict[str, Any]:
    """Return symbridge connection status."""
    return {
        "organism_side": "ready",
        "event_bus_available": hasattr(request.app.state, "event_bus"),
        "memory_available": hasattr(request.app.state, "memory"),
    }
