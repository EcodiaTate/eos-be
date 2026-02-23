"""
EcodiaOS — Alive: Visualization & Embodiment

WebSocket server that bridges Synapse telemetry and Atune affect state
to the browser-based Three.js visualization.

Public API:
  AliveWebSocketServer — standalone WS server on port 8001
"""

from ecodiaos.systems.alive.ws_server import AliveWebSocketServer

__all__ = [
    "AliveWebSocketServer",
]
