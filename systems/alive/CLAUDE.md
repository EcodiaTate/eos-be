# Alive — Consciousness Visualization & Telemetry Bridge (Spec 11a)

**Spec:** `.claude/EcodiaOS_Spec_11_Alive.md`
**System ID:** `alive`
**Role:** Real-time WebSocket bridge exposing the organism's full cognitive state. Makes EOS observable to itself and external dashboards.

---

## What's Implemented

### Core WebSocket Server (`ws_server.py`)
- `AliveWebSocketServer` on port 8001 — websockets library, async I/O
- 3 independent background tasks:
  1. **Redis Subscriber** (`_redis_subscriber`) — forwards all Synapse events from Redis pub/sub channel `{prefix}:channel:synapse_events`
  2. **Affect Poller** (`_affect_poller`) — polls `SomaService.get_current_state()` at ~10Hz (100ms)
  3. **System State Poller** (`_system_state_poller`) — aggregates all subsystem snapshots at ~1Hz (1000ms)
- **Connection Handler** (`_handler`) — sends initial state (affect + system_state) on connect, then holds open
- **Broadcast Dispatcher** (`_broadcast`) — fan-out to all clients; dead connections silently pruned

### Three Data Streams (unified envelope `{ "stream": ..., "payload": ... }`)
1. **`synapse`** — raw cognitive cycle telemetry (CYCLE_COMPLETED ~6.7Hz, RHYTHM_CHANGED, COHERENCE_ALERT, etc.)
2. **`affect`** — 9D Soma interoceptive state + urgency + dominant_error at ~10Hz
3. **`system_state`** — aggregated snapshot at ~1Hz: 13 sections (see below)

### System State Sections (13 total)

| Section | System | Access Pattern |
|---------|--------|----------------|
| `cycle` | Synapse | `clock_state` + `rhythm_snapshot` (sync) |
| `drives` | Telos (primary) + Thymos (rejections) | `last_report` (sync) + `drive_state` (sync) |
| `interoceptive` | Soma | `get_current_state()` (sync) |
| `attention` | Fovea | `get_current_attention_profile()` (sync) |
| `immune` | Thymos | `health()` (async, 0.8s timeout) |
| `goals` | Nova | `health()` + `active_goal_summaries` (async, 0.8s timeout) |
| `actions` | Axon | `recent_outcomes[:10]` (sync) |
| `economics` | Oikos | `health()` (async, 0.8s timeout) |
| `mutations` | Simula | `stats` dict (sync) |
| `benchmarks` | BenchmarkProvider protocol | `stats` property (sync) |
| `causal` | Kairos | `health()` (async, 0.8s timeout) — tier counts, I-ratio |
| `compression` | Logos | `health()` (async, 0.8s timeout) — pressure, intelligence ratio, Schwarzschild |
| `sleep` | Oneiros | `stats` + `is_sleeping` (sync) — stage, pressure, cycle metrics |

### Affect Payload — `dominance` field (Spec §4.3)
The standalone affect stream now includes `dominance`:
- **Source**: `AtuneService.current_affect.dominance` if Atune is injected (authoritative)
- **Fallback**: `Soma.SOCIAL_CHARGE` when Atune is not wired (0→1 proxy)
- Atune is passed as optional `atune=` kwarg to `AliveWebSocketServer.__init__`

### WebSocket Authentication (Spec §14.3)
Port 8001 now supports token-based auth:
- Configure via `config.alive_ws.auth_tokens` (list of secret strings)
- Client passes `?token=<secret>` in the WebSocket URL
- Invalid/missing token → close code 4401 "Unauthorized"
- No tokens configured → open access (dev / internal LAN mode)

### Timeout Protection (M2 fix)
All async gathers wrapped with `asyncio.wait_for(..., timeout=0.8s)` via `_gather_*_safe()` wrappers. A hung subsystem returns `{"available": False, "error": "timeout"}` and does not stall the poller.

### RE Training Pipeline (M6 — Spec §21 Path 1)
Every system_state snapshot is written to Redis Stream `{prefix}:stream:alive_snapshots` (maxlen=10,000, rolling ~2.7h at 1Hz). The RE training pipeline consumes this stream. Fire-and-forget; errors are non-fatal.

### Graceful Degradation
Each gatherer wraps exceptions independently — failure in one section returns `{"available": false, "error": ...}` without blocking others.

---

## Key Files

| File | Role |
|------|------|
| `ws_server.py` | Full Alive implementation — `AliveWebSocketServer` |
| `__init__.py` | Module exports |
| `tests/unit/systems/alive/test_ws_server.py` | Unit tests — 25 cases covering auth, affect payload, timeout, stream write, health |

---

## Integration Surface

**Emits (WebSocket):** All three streams to connected clients.
**Emits (Redis Stream):** `{prefix}:stream:alive_snapshots` — system_state snapshots for RE training.
**Consumes (Redis Pub/Sub):** `{prefix}:channel:synapse_events` — all Synapse-published events.
**Direct system refs (injected):** Synapse, Soma, Telos, Thymos, Nova, Axon, Oikos, Simula, Fovea, Kairos, Logos, Oneiros — passed as constructor dependencies. Read-only, no side effects.

---

## Performance Targets

| Stream | Rate | End-to-End Latency |
|--------|------|--------------------|
| Affect | ~10Hz | ~50–70ms |
| Synapse events | event-driven | ~10–30ms |
| System state | ~1Hz | ~80–120ms |

---

## Known Issues / Remaining Work

- **Benchmarks section**: returns `{"available": false}` until Benchmarks system is wired — no `"stub": true` marker (minor: dashboards cannot distinguish unwired from empty)
- **No client-specific rate limiting**: slow clients get their WebSocket buffer filled then dropped — intentional but may surprise consumers
- **`_redis_subscriber` reconnect**: relies on Redis client wrapper auto-reconnect; if wrapper doesn't retry, subscriber silently stops
- **No Prometheus metrics**: spec §14.1 metrics (connected client gauge, per-stream counters, gather latency histograms) not emitted
- **No historical ring buffer**: each session starts from zero context; spec §18.3 replay not implemented
- **FastAPI `/ws/alive` divergence**: formally documented as two distinct protocols (see protocol note in `ws_server.py` and docstring in `main.py`). Not a bug — intentional for Cloud Run single-port deployments.
- **Population-level telemetry absent**: `fleet_children` count only; no per-child fitness, heritable variation, or Bedau-Packard stats (requires Federation + Mitosis to be operational)
- **RE status section absent**: no `re_status` section; Thompson sampling weights, Claude vs RE routing, RE forgetting alarm are invisible
- **`registry.py` atune wiring**: `_init_alive_ws` now accepts `atune=` kwarg; call site must pass `atune=app.state.atune` when calling `_init_alive_ws`
