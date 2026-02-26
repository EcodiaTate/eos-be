# Soma Follow-Up Work Status (System 15)

## ✅ COMPLETED — All Stage 0 Tasks

### 0.1: SomaClock Types
- ✅ All types defined in `ecodiaos/systems/soma/types.py`:
  - `InteroceptiveState` — 9D sensed, multi-horizon predictions, errors, precision
  - `AllostaticSignal` — primary output per theta cycle (urgency, precision_weights, phase space)
  - `SomaticMarker` — 19D snapshot (9 sensed + 9 errors + 1 PE) for memory stamping
  - `Attractor`, `Bifurcation`, `CounterfactualTrace`, `PhaseSpaceSnapshot`
  - 9 `InteroceptiveDimension` enums, 5 `DevelopmentalStage` enums

### 0.2: Soma Wired as Step 0 in Synapse Cognitive Cycle
- ✅ `synapse/clock.py` — Soma runs as step 0 before Atune extraction
- ✅ `synapse/service.py` — `set_soma()` method
- ✅ `main.py` — `synapse.set_soma(soma)` wiring

### 0.3: Atune precision_weights Modulated by Soma
- ✅ `atune/service.py` — reads `soma.get_current_signal().precision_weights`
- ✅ `atune/affect.py` — precision modulates valence/arousal inertia coefficients
- ✅ `main.py` — `atune.set_soma(soma)` wiring

### 0.4: Nova Allostatic Triggers
- ✅ `nova/service.py` — `set_soma()` method, urgency check before deliberation (lines 265-276)
- ✅ `nova/deliberation_engine.py` — `allostatic_mode` + `allostatic_error_dim` params, `_reorder_goals_for_allostatic_mode()` with dimension→keyword mapping
- ✅ `main.py` — `nova.set_soma(soma)` wiring

### 0.5: Memory Somatic Markers
- ✅ `primitives/memory_trace.py` — `somatic_marker` + `somatic_vector` fields on `Episode`, `MemoryTrace`, and `RetrievalResult`
- ✅ `memory/service.py` — `set_soma()`, somatic marker stamping in `store_percept()`, somatic reranking in `retrieve()` with unified_score sync
- ✅ `memory/episodic.py` — `store_episode()` persists `somatic_vector` (19D) and `somatic_marker_json` to Neo4j
- ✅ `memory/retrieval.py` — all 6 query paths (vector, BM25, 4 graph traverse hops) load `somatic_vector` from Neo4j; merge/dedup carries somatic data forward; `salience_score` synced for reranking
- ✅ `memory/schema.py` — 19D cosine vector index `episode_somatic` on `Episode.somatic_vector`
- ✅ `main.py` — `memory.set_soma(soma)` wiring

### 0.5+ Consumer Systems (Beyond Core Memory)
- ✅ **Evo** — `set_soma()`, curiosity-modulated hypothesis interval, dynamics matrix push
- ✅ **Oneiros** — `set_soma()`, sleep pressure from energy errors, REM counterfactuals
- ✅ **Thymos** — `set_soma()`, integrity precision gating on incident diagnosis
- ✅ **Voxis** — `set_soma()`, arousal/valence modulation of expression style

### 0.6: Database Migrations
- ✅ `database/migrations.py` — `migrate_timescaledb_interoceptive_state()` (hypertable with compression 7d, retention 90d)
- ✅ `database/migrations.py` — `migrate_neo4j_somatic_vector_index()` (19D cosine index on SomaticMarker nodes)
- ✅ `memory/schema.py` — `episode_somatic` vector index (19D cosine on Episode.somatic_vector)
- ✅ `database/migrations.py` — `run_all_migrations()` orchestrator

### 0.7: Integration Tests
- ✅ `tests/integration/systems/soma/test_soma_integration.py` — 28 tests covering:
  - Types integrity (dimensions, marker vectors, signal defaults)
  - Soma cycle execution and signal emission
  - Atune precision weights consumption
  - Nova allostatic trigger readiness
  - Memory somatic marker creation, storage, and round-trip
  - Somatic reranking (boost similar, mixed candidates, empty list)
  - RetrievalResult somatic fields
  - Migration function importability
  - Full Soma → Memory pathway (marker roundtrip, mixed candidates)
  - Graceful degradation (all systems work without Soma)

### main.py Wiring
- ✅ Full Soma wiring (lines 501-526):
  ```python
  soma = SomaService(config=config.soma)
  soma.set_atune(atune)
  soma.set_synapse(synapse)
  soma.set_nova(nova)
  soma.set_thymos(thymos)
  soma.set_equor(equor)
  await soma.initialize()
  atune.set_soma(soma)
  synapse.set_soma(soma)
  nova.set_soma(soma)
  memory.set_soma(soma)
  evo.set_soma(soma)
  oneiros.set_soma(soma)
  thymos.set_soma(soma)
  voxis.set_soma(soma)
  synapse.register_system(soma)
  ```

## Test Results

```
tests/unit/systems/soma/         — 127 passed ✅
tests/integration/systems/soma/  —  28 passed ✅
                                    155 total
```

## Integration Checklist

- [x] Soma step 0 in clock (soma runs before atune)
- [x] Atune precision modulation (inertia changes with precision)
- [x] Nova allostatic trigger (high urgency → goal reordering)
- [x] Memory somatic markers (traces stamped, reranking works)
- [x] Memory retrieval loads somatic_vector from Neo4j
- [x] Memory schema has 19D somatic vector index
- [x] Evo curiosity modulation
- [x] Oneiros sleep pressure from energy errors
- [x] Thymos constitutional gating
- [x] Voxis expression modulation
- [x] TimescaleDB migration defined
- [x] Neo4j vector index defined (both migrations.py + schema.py)
- [x] All consumer reads handle Soma unavailable gracefully

## Architecture Notes

1. **Graceful Degradation**: All consumers check `if self._soma is not None` + `try/except` with safe defaults.
2. **Precision Weights**: `dict[InteroceptiveDimension, float]` where 0-1. Higher precision → more inertia (trust current state).
3. **5ms Budget**: Soma stays under 5ms/cycle. All consumers read from cache.
4. **Somatic Memory Pipeline**: `Soma.get_somatic_marker()` → `Episode.somatic_marker/somatic_vector` → Neo4j `somatic_vector` property → `RetrievalResult.somatic_vector` → `Soma.somatic_rerank()` → unified_score sync.
5. **19D Vector**: 9 sensed + 9 moment errors + 1 prediction error magnitude. Stored as list[float] in Neo4j with cosine similarity index.
