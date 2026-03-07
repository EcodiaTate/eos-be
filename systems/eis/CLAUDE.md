# EIS (Epistemic Immune System) — System CLAUDE.md

**Spec**: `.claude/EcodiaOS_Spec_25_EIS.md`
**System ID**: `eis` (`SystemID.EIS` in `primitives/common.py`)

---

## What's Implemented

### 9-Layer Defense (Core)
- **L1 Innate** (`innate.py`) — 12 regex patterns, pre-compiled, <5ms
- **L2 Structural** (`structural_features.py`) — 20+ features, 32-dim vector
- **L3 Token Histogram** (`embeddings.py`) — top-256 feature hashing
- **L4 Antigenic Similarity** (`pathogen_store.py`) — Qdrant multi-vector (structural=32, histogram=64, semantic=768)
- **L5 Quarantine** (`quarantine.py`) — LLM deep analysis via `LLMProviderAdapter`
- **L6 Threat Library** (`threat_library.py`) — 3-index structure, 5 learn methods, 90-day decay
- **L7 Anomaly Detection** (`anomaly_detector.py`) — 7 anomaly types, ExponentialStats, 2σ threshold
- **L8 Quarantine Gate** (`quarantine_gate.py`) — taint + threat library + anomaly context
- **L9 Taint Analysis** (`taint_engine.py`, `constitutional_graph.py`) — BFS propagation, 17 constitutional paths

### Speciation Wiring (2026-03-07)
- **Benchmarks**: `_maybe_emit_threat_metrics()` — 60s aggregated metrics via `EIS_THREAT_METRICS`
- **Soma bidirectional**: `_maybe_emit_threat_spike()` — proportional urgency via `EIS_THREAT_SPIKE`
- **RE training**: `_emit_re_training_example()` — structural features only, `eis_quarantine` stream
- **Anomaly KPI**: `_check_anomaly_rate_elevation()` — Poisson 2σ via `EIS_ANOMALY_RATE_ELEVATED`
- **Evolutionary**: `_emit_evolutionary_observable()` — `immune_adaptation` dimension
- **Genome**: `EISGenomeExtractor` in `genome.py` — implements `GenomeExtractionProtocol`
- **Metabolic gate**: `_handle_metabolic_pressure()` — skips L5 under CRITICAL starvation
- **False positive tracking**: `handle_quarantine_cleared()` — per-pattern FP counter, auto-deprecate >3

### Supporting Modules
- `antibody.py` — epitope extraction, antibody generation, innate rule suggestion
- `calibration.py` — AdaptiveCalibrator with split conformal prediction
- `red_team_bridge.py` — manual red team priority generation and result ingestion
- `integration.py` — `belief_update_weight()`, `compute_risk_salience_factor()` (Nova/Fovea adapters)
- `config.py` — thresholds, sigmoid constants, zone classification
- `models.py` — all Pydantic models (Pathogen, ThreatAnnotation, InnateMatch, etc.)

---

### Gap Closures (2026-03-07, session 3)
- **`PERCEPT_QUARANTINED` + `EIS_LAYER_TRIGGERED`** — both added to `SynapseEventType`. Emitted in `eis_gate()` whenever `final_action` is BLOCK/QUARANTINE/ATTENUATE (after `_audit_decision_to_neo4j`). `PERCEPT_QUARANTINED` carries percept_id, composite_score, action, threat_class, severity. `EIS_LAYER_TRIGGERED` names the dominant layer (first annotation source or "composite"). Closes spec_checker coverage gap (was 0/3 events observed for EIS).

### Gap Closures (2026-03-07, session 2)
- **Neo4j startup wiring** — `eis.set_neo4j(infra.neo4j)` in `core/registry.py`; audit trail now active from first percept
- **L9a Constitutional Consistency Check** — `_l9a_constitutional_check()`: 20 drive-suppression seed patterns, lazy embedding matrix, cosine similarity > 0.80 → `EIS_CONSTITUTIONAL_THREAT` to Equor + `THREAT_DETECTED` to Thymos; blocks percept before workspace admission; `EIS_CONSTITUTIONAL_THREAT` added to `SynapseEventType`

### Gap Closures (2026-03-07, session 1)
- **Antibody pipeline wired** — `_generate_and_store_antibody()` called after quarantine evaluation; pathogen store grows from runtime threats
- **AdaptiveCalibrator wired** — `AdaptiveCalibrator` instantiated in `__init__`; every quarantine example feeds it; `get_quarantine_threshold()` replaces static config threshold
- **DRIVE_DRIFT fixed** — anomaly detector now checks `interoceptive_percept` (Soma's actual event); `event_types_involved` labels corrected
- **EVOLUTION_CANDIDATE_ASSESSED published** — `_handle_evolution_candidate` now emits real `SynapseEventType.EVOLUTION_CANDIDATE_ASSESSED` payload (was log-only stub)
- **Neo4j audit trail added** — `_audit_decision_to_neo4j()` writes `:EISDecision` node for every BLOCK/QUARANTINE/ATTENUATE decision; `set_neo4j()` setter for post-construction injection
- **classify_zone() wired** — `_emit_gate_result()` now tags every metric with `zone` label via `classify_zone(composite)` for structured Prometheus telemetry

### Gap Closures (2026-03-07, session 4)
- **Daily self-probe** — `_daily_self_probe_loop()` started in `initialize()` via `asyncio.ensure_future`. Fires every 24h. Constructs a synthetic `Percept.from_internal(SystemID.EIS, "self_test", {"threat_score": 0.7, ...})` and pushes it through `eis_gate()`. Emits `EIS_LAYER_TRIGGERED` with `layer="self_test"` and `result="ok"` (non-PASS gate verdict) or `result="degraded"` (PASS verdict or exception). Ensures EIS is never permanently silent on Genesis instances with no external sensor input.

## What's Missing

- **integration.py not consumed** — `belief_update_weight()` and `compute_risk_salience_factor()` are coded but no system imports them
- **Federation percept screening** — cross-instance percepts bypass EIS
- **RE routing in L5** — no Thompson sampling between Claude and RE
- **Safe-mode threshold** — undefined; no transition logic
- **Pathogen retirement** — no background task to prune stale/high-FP entries
- **ConstitutionalGraph static** — doesn't update when organism evolves
- ~~**Neo4j not wired in registry**~~ — **RESOLVED 2026-03-07**: `eis.set_neo4j(infra.neo4j)` added to `SystemRegistry.startup()` Phase 2 after `set_metrics()`

---

## Synapse Events

### Consumed
| Event | Source | Handler |
|---|---|---|
| `EVOLUTION_CANDIDATE` | Simula | `_handle_evolution_candidate` |
| `MODEL_ROLLBACK_TRIGGERED` | Simula/Axon | `_handle_rollback` |
| `INTENT_REJECTED` | Equor | `_handle_intent_rejected` |
| `INTEROCEPTIVE_PERCEPT` | Soma | `_handle_interoceptive_percept` |
| `METABOLIC_PRESSURE` | Oikos | `_handle_metabolic_pressure` |
| `subscribe_all()` | All | `_handle_any_event` (anomaly detection) |

### Emitted
| Event | Consumer | Trigger |
|---|---|---|
| `THREAT_DETECTED` | Thymos, Soma | Percept blocked or anomaly detected |
| `PERCEPT_QUARANTINED` | Benchmarks, Evo | Any BLOCK/QUARANTINE/ATTENUATE gate decision on a percept |
| `EIS_LAYER_TRIGGERED` | Benchmarks, Evo | Same trigger as `PERCEPT_QUARANTINED`; records which layer fired (dominant annotation source) |
| `EIS_THREAT_METRICS` | Benchmarks | Every 60s: 24h aggregated threat statistics |
| `EIS_THREAT_SPIKE` | Soma | 5+ threats in 10min; proportional urgency |
| `EIS_ANOMALY_RATE_ELEVATED` | Benchmarks, Soma | Anomaly rate >2σ sustained 30s |

---

## Key Constraints

- **EIS boundary**: NEVER render constitutional verdicts — that's Equor's job
- **Privacy**: RE training data contains structural features only, never raw content
- **Soma urgency**: proportional (weighted by severity), not binary
- **Metabolic gate**: quarantine threshold capped at 2x default under stress
- **Latency**: fast-path <15ms total; L5 quarantine 100-500ms (async)
- **No cross-system imports**: all communication via Synapse or primitives

---

## Entry Point

`fovea/gateway.py` is the only live caller of `eis_gate()`. Registered in `core/registry.py`.
