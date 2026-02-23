"""
EcodiaOS — Alive WebSocket Server

Standalone WebSocket server on port 8001 that bridges two data streams
to connected browser clients:

1. Synapse events (via Redis pub/sub) — forwarded as-is
2. Affect state snapshots (polled from Atune at ~10Hz)

Messages are JSON-encoded with an envelope:
  {"stream": "synapse" | "affect", "payload": {...}}
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import orjson
import structlog
import websockets
from websockets.asyncio.server import ServerConnection

if TYPE_CHECKING:
    from ecodiaos.clients.redis import RedisClient
    from ecodiaos.systems.atune.service import AtuneService

logger = structlog.get_logger("ecodiaos.systems.alive.ws_server")

# Affect polling interval (seconds) — ~10 Hz
_AFFECT_POLL_INTERVAL: float = 0.1

# How often to send health snapshots (seconds) — ~0.2 Hz
_HEALTH_POLL_INTERVAL: float = 5.0


def _json(data: dict[str, Any]) -> str:
    """Fast JSON serialization via orjson."""
    return orjson.dumps(data).decode()


class AliveWebSocketServer:
    """
    WebSocket server for the Alive visualization layer.

    Multiplexes Synapse telemetry events and Atune affect state
    onto a single WebSocket connection per client.
    """

    system_id: str = "alive"

    def __init__(
        self,
        redis: RedisClient,
        atune: AtuneService,
        port: int = 8001,
    ) -> None:
        self._redis = redis
        self._atune = atune
        self._port = port
        self._clients: set[ServerConnection] = set()
        self._server: Any = None
        self._running: bool = False
        self._tasks: list[asyncio.Task[None]] = []
        self._logger = logger.bind(component="alive_ws")

    async def start(self) -> None:
        """Start the WebSocket server and background stream tasks."""
        self._running = True
        self._server = await websockets.serve(
            self._handler,
            "0.0.0.0",
            self._port,
        )
        self._tasks.append(asyncio.create_task(self._redis_subscriber()))
        self._tasks.append(asyncio.create_task(self._affect_poller()))
        self._logger.info("alive_ws_started", port=self._port)

    async def stop(self) -> None:
        """Shut down the server and cancel background tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self._logger.info("alive_ws_stopped")

    # ─── Connection Handler ────────────────────────────────────────────

    async def _handler(self, ws: ServerConnection) -> None:
        """Handle a new WebSocket connection."""
        self._clients.add(ws)
        remote = ws.remote_address
        self._logger.info(
            "alive_client_connected",
            remote=str(remote),
            total_clients=len(self._clients),
        )
        try:
            # Send initial state so the client can render immediately
            await self._send_initial_state(ws)
            # Keep connection alive; we don't expect client messages
            async for _ in ws:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            self._logger.info(
                "alive_client_disconnected",
                remote=str(remote),
                total_clients=len(self._clients),
            )

    async def _send_initial_state(self, ws: ServerConnection) -> None:
        """Send an initial affect snapshot so the client doesn't start blank."""
        affect = self._atune.current_affect
        msg = _json({
            "stream": "affect",
            "payload": {
                "valence": round(affect.valence, 4),
                "arousal": round(affect.arousal, 4),
                "dominance": round(affect.dominance, 4),
                "curiosity": round(affect.curiosity, 4),
                "care_activation": round(affect.care_activation, 4),
                "coherence_stress": round(affect.coherence_stress, 4),
                "ts": affect.timestamp.isoformat() if affect.timestamp else None,
            },
        })
        await ws.send(msg)

    # ─── Redis Subscriber (Synapse Events) ─────────────────────────────

    async def _redis_subscriber(self) -> None:
        """Subscribe to Redis synapse_events and forward to all clients."""
        redis = self._redis.client
        prefix = self._redis._config.prefix
        channel = f"{prefix}:channel:synapse_events"

        pubsub = redis.pubsub()
        await pubsub.subscribe(channel)
        self._logger.info("alive_redis_subscribed", channel=channel)

        try:
            while self._running:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=0.1,
                )
                if message and message["type"] == "message":
                    raw = message["data"]
                    # Redis data is already a JSON string (orjson-encoded by EventBus)
                    payload = orjson.loads(raw) if isinstance(raw, (str, bytes)) else raw
                    msg = _json({"stream": "synapse", "payload": payload})
                    await self._broadcast(msg)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._logger.error("alive_redis_subscriber_error", error=str(exc))
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    # ─── Affect Poller ─────────────────────────────────────────────────

    async def _affect_poller(self) -> None:
        """Poll Atune affect state at ~10Hz and send to all clients."""
        try:
            while self._running:
                affect = self._atune.current_affect
                msg = _json({
                    "stream": "affect",
                    "payload": {
                        "valence": round(affect.valence, 4),
                        "arousal": round(affect.arousal, 4),
                        "dominance": round(affect.dominance, 4),
                        "curiosity": round(affect.curiosity, 4),
                        "care_activation": round(affect.care_activation, 4),
                        "coherence_stress": round(affect.coherence_stress, 4),
                        "ts": (
                            affect.timestamp.isoformat()
                            if affect.timestamp
                            else None
                        ),
                    },
                })
                await self._broadcast(msg)
                await asyncio.sleep(_AFFECT_POLL_INTERVAL)
        except asyncio.CancelledError:
            pass

    # ─── Broadcast ─────────────────────────────────────────────────────

    async def _broadcast(self, message: str) -> None:
        """Send a message to all connected clients. Remove dead ones."""
        if not self._clients:
            return
        dead: set[ServerConnection] = set()
        for ws in self._clients:
            try:
                await ws.send(message)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    # ─── Health ────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Health status for Alive WebSocket server."""
        return {
            "status": "running" if self._running else "stopped",
            "port": self._port,
            "connected_clients": len(self._clients),
        }
