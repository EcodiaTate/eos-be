# Atune — CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_03_DISCONTINUED.md`
**System ID:** `atune`
**Role:** Sensory cortex & Global Workspace. Receives all input (text, voice, sensor, events), scores percepts via 7-head salience, and broadcasts the winner to all cognitive systems. If Memory is selfhood and Equor is conscience, Atune is awareness.

---

## Architecture

**Pipeline (per theta cycle, ≤150ms):**
```
RawInput → Normalise → EIS Gate → Prediction Error → 7-Head Salience
→ Workspace Competition → Winner Broadcast → Memory Enrichment
→ Async Entity Extraction → Affect Persistence
```

**Core modules:** `service.py` (orchestrator), `normalisation.py`, `salience.py` (7 heads), `workspace.py` (Global Workspace), `prediction.py`, `momentum.py`, `affect.py`, `meta.py`, `extraction.py`, `market_pattern.py`

**Input channels:** `TEXT_CHAT`, `VOICE`, `GESTURE`, `SENSOR_IOT`, `CALENDAR`, `EXTERNAL_API`, `SYSTEM_EVENT`, `MEMORY_BUBBLE`, `AFFECT_SHIFT`, `EVO_INSIGHT`, `FEDERATION_MSG`

---

## Seven Salience Heads

| Head | Weight | Basis |
|------|--------|-------|
| Novelty | 0.20 | Prediction error × (1 − habituation × 0.5); contradiction bonus ×1.3 |
| Risk | 0.18 | Embedding similarity to known threats (Memory-backed) |
| Goal | 0.15 | Cosine similarity to active goal embeddings |
| Identity | 0.15 | Relevance to core self entities |
| Consequence | 0.12 | Temporal proximity × consequence scope |
| Social | 0.10 | Agent mentions + sentiment + conflict |
| Economic | 0.10 | Market keyword matching via Evo patterns |

All scores precision-weighted by `AffectState` (Friston 2010: `precision ~ 1/uncertainty`) then meta-weighted by `MetaAttentionController`. Momentum tracking adds first/second derivatives per head; ACCELERATING heads trigger arousal nudge.

**Ignition threshold:** 0.3 (configurable). Winner broadcast to all Synapse subscribers.

---

## Key Types

```python
class AtuneConfig:
    ignition_threshold: float = 0.3
    workspace_buffer_size: int = 32
    spontaneous_recall_base_probability: float = 0.02
    max_percept_queue_size: int = 100
    affect_persist_interval: int = 10
    cache_identity_refresh_cycles: int = 1000
    cache_risk_refresh_cycles: int = 500

class SalienceVector:
    scores: dict[str, float]       # per-head
    composite: float
    prediction_error: PredictionError
    gradient_attention: dict[str, GradientAttentionVector]  # token-level attribution
    momentum: dict[str, HeadMomentum]
    threat_trajectory: ThreatTrajectory
```

---

## EIS Integration

Every percept passes through `eis_service.eis_gate()` before salience scoring:
- `BLOCK` → reject, log, return `None`
- `ATTENUATE` → accept with reduced salience
- `PASS` → continue

EIS result stored in `percept.metadata["eis_result"]` — RiskHead reads this directly.

---

## Memory Integration

**Retrieval** (before broadcast): `memory_client.retrieve_context(embedding, text, max_results=10)`
**Storage** (after broadcast): `memory_client.store_percept_with_broadcast(percept, salience, affect)` — stores Episode, emits `EPISODE_STORED`
**Entity extraction** (async, background): LLM extract → resolve entities → `MENTIONED_IN` edges on Neo4j

Temporal causality: episodes linked via `FOLLOWED_BY` edge if gap ≤1h.

---

## Wiring (Startup Order)

```python
atune.set_eis(eis_service)           # before startup
atune.set_memory_service(memory)     # before startup
atune.set_synapse(synapse)           # during startup
atune.set_belief_state(nova_reader)  # after startup (fallback available)
atune.set_soma(soma_service)
atune.set_market_pattern_detector(templates, axon)  # fast-path reflex arc
```

**Fast-path:** `MarketPatternDetector` detects pre-approved patterns → `FastPathIntent` → `axon.execute_fast_path()` directly (bypasses Nova/Equor, ≤50ms budget).

---

## What's Implemented

### PERCEPT_ARRIVED (2026-03-07)
`PerceptionGateway.ingest()` (`fovea/gateway.py`) emits `PERCEPT_ARRIVED` with `source_system="atune"` immediately after a percept clears the EIS gate. Fovea's `WorldModelAdapter` subscribes to this event for inter-event timing statistics. The spec_checker credits this event to the "atune" system. Payload: `percept_id`, `source_system`, `channel`, `timestamp_iso`, `modality`.

Note: `PerceptionGateway` (in `fovea/`) is the live implementation of what Spec 03 calls "Atune". It is aliased as `AtuneService` for backward compatibility.

### ATUNE_REPAIR_VALIDATION
Emitted by Thymos (`service.py:_broadcast_repair_completed()`) with `source_system="thymos"` 60 cycles after a repair, checking whether the incident fingerprint re-fires. Thymos also subscribes to it. The event type exists in `SynapseEventType` and fires as part of the repair validation pipeline.

**Status (Atune standalone): Not yet implemented** — the Atune system directory contains only this CLAUDE.md. No `.py` files exist.

The spec (Spec 03) fully defines the interface. Key things to implement:
- `AtuneService` with `ingest()`, `run_cycle()`, `contribute()`, `receive_belief_feedback()`
- All 7 salience heads as separate scoring functions
- `GlobalWorkspace` with ignition + broadcast
- `MarketPatternDetector` + `AtuneCache` with staggered refresh cycles

---

## What's Missing (All of it — system unimplemented)

1. No `ingest()` / `run_cycle()` implementation
2. No 7-head salience engine
3. No Global Workspace competition logic
4. No momentum tracking or gradient attention vectors
5. No EIS integration
6. No `MarketPatternDetector` fast-path trigger
7. No async entity extraction pipeline
8. No `AtuneCache` with refresh cycles

---

## Key Constraints

- Total cycle: ≤150ms; normalisation ≤5ms; EIS ≤30ms; 7-head scoring ≤40ms
- Workspace queue bounded at 100 percepts; overflow drops oldest
- Feedback loop protection: bias clamp ±0.40, inertia decay 0.95, per-source history 20
- Entity extraction is non-blocking — `asyncio.create_task()`, loop continues immediately
- All inter-system communication via Synapse bus — no direct system imports

## Integration Surface

| System | Direction | Purpose |
|--------|-----------|---------|
| EIS | ← | Epistemic threat screening per percept |
| Memory | ↔ | Context retrieval (pre-broadcast) + episode storage (post-broadcast) |
| Soma | ← | Precision weights for affect-coupled salience |
| Nova | ← | Belief state for prediction error |
| Evo | ← | Patterns and hypothesis counts for EconomicHead |
| Axon | → | Fast-path dispatch via MarketPatternDetector |
| All systems | → | Workspace broadcast (winner percept) |
