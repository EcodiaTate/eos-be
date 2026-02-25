# Soma Follow-Up Work Status (System 15)

## ✅ COMPLETED

### 1. Synapse Clock Integration (Step 0)
- ✅ Modified `ecodiaos/systems/synapse/clock.py`:
  - Added `_soma` reference in `__init__`
  - Added `set_soma()` wiring method
  - Inserted `soma.run_cycle()` as step 0 in `_run_loop()` before `atune.run_cycle()`
  - Graceful error handling if Soma unavailable
- ✅ Modified `ecodiaos/systems/synapse/service.py`:
  - Added `set_soma()` to SynapseService
- ✅ Modified `ecodiaos/main.py`:
  - Call `synapse.set_soma(soma)` after soma initialization

**Result**: Soma now runs first every theta cycle, emitting AllostaticSignal before Atune reads it.

### 2. Atune Consumer Integration
- ✅ Modified `ecodiaos/systems/atune/service.py`:
  - Added `_soma` reference
  - Added `set_soma()` method
  - Extractprecision_weights from `soma.get_current_signal()` in `run_cycle()`
  - Pass precision_weights to affect manager
- ✅ Modified `ecodiaos/systems/atune/affect.py`:
  - Updated `update()` signature to accept `precision_weights: dict`
  - Applied precision modulation to valence inertia coefficient
  - Applied precision modulation to arousal inertia coefficient
  - Higher precision → more inertia (trust current), lower → adapt faster
- ✅ Wired in main.py: `atune.set_soma(soma)`

**Result**: Atune now applies precision_weights to modulate affect state adaptation speed.

## 🔄 TODO (Remaining Consumer-Side Modifications)

### 3. Nova — Allostatic Deliberation Trigger
**Location**: `ecodiaos/systems/nova/service.py` + `ecodiaos/systems/nova/deliberation_engine.py`

**What to do**:
1. Add `_soma` reference to NovaService.__init__
2. Add `set_soma()` method
3. Before calling `deliberation_engine.deliberate()`, check `soma.get_current_signal().urgency`
4. If urgency > `soma.urgency_threshold` (default 0.7), set a flag on deliberation context
5. In DeliberationEngine, when allostatic urgency is high, prioritize goals that address dominant_error dimension
6. Wire in main.py: `nova.set_soma(soma)` after nova.initialize()

**Key insight**: When urgency spikes, Nova should switch to allostatic deliberation mode — reordering goals by how well they address the body's current stressors.

### 4. Memory — Somatic Markers & Reranking
**Locations**: `ecodiaos/systems/memory/service.py` + `ecodiaos/systems/memory/trace_writer.py` + `ecodiaos/systems/memory/retrieval.py`

**What to do**:
1. Add `_soma` reference
2. When writing traces (`trace_writer.write_trace()`), call `soma.get_somatic_marker()` and attach to MemoryTrace
3. When retrieving memories (`retrieval.search()`), after getting candidates, call `soma.somatic_rerank(candidates)` to boost somatic similarity matches
4. Wire: `memory.set_soma(soma)` after memory.initialize()

**Key insight**: Memory stamps each trace with the organism's felt state at encoding time. On retrieval, it reranks candidates by somatic distance from current state, enabling state-congruent recall.

### 5. Evo — Curiosity Modulation & Dynamics Update
**Location**: `ecodiaos/systems/evo/service.py`

**What to do**:
1. Add `_soma` reference
2. In Evo's main loop, read `soma.get_current_signal().curiosity_drive`
3. Use curiosity_drive to modulate exploration vs exploitation (high curiosity → more hypothesis generation)
4. When Evo discovers systematic mis-predictions (certain interoceptive transitions are poorly predicted), call `soma.update_dynamics_matrix(new_dynamics)` to refine the 9x9 cross-dimension coupling matrix
5. Wire: `evo.set_soma(soma)`

### 6. Oneiros — Sleep Pressure & REM Counterfactuals
**Location**: `ecodiaos/systems/oneiros/service.py`

**What to do**:
1. Add `_soma` reference
2. Read `soma.get_current_signal().state.errors["immediate"][InteroceptiveDimension.ENERGY]` (energy error)
3. High energy error (large negative) = metabolic depletion → increase sleep pressure
4. During REM: call `soma.generate_counterfactual(decision_id, trajectory, description, initial_impact, num_steps)` to replay near-miss episodes
5. Wire: `oneiros.set_soma(soma)`

### 7. Thymos — Constitutional Health Gating
**Location**: `ecodiaos/systems/thymos/service.py`

**What to do**:
1. Add `_soma` reference
2. Read `soma.get_current_signal().precision_weights[InteroceptiveDimension.INTEGRITY]`
3. When integrity precision is high, weight constitutional health signals higher in risk assessment
4. Wire: `thymos.set_soma(soma)`

### 8. Voxis — Somatic Expression Modulation
**Location**: `ecodiaos/systems/voxis/service.py`

**What to do**:
1. Add `_soma` reference
2. Read `arousal` and `valence` from `soma.get_current_signal().state`
3. High arousal → more urgent/active language; low arousal → measured pace
4. High valence → warm tone; low valence → careful hedging
5. Wire: `voxis.set_soma(soma)`

## 🗄️ DATABASE MIGRATIONS

### TimescaleDB: interoceptive_state Hypertable
**File to create**: `ecodiaos/database/migrations/00012_soma_interoceptive_hypertable.py`

```sql
CREATE TABLE IF NOT EXISTS interoceptive_state (
    time             TIMESTAMPTZ NOT NULL,
    tenant_id        UUID NOT NULL,
    cycle_number     BIGINT,

    -- 9D interoceptive state (sensed)
    energy           FLOAT NOT NULL,
    arousal          FLOAT NOT NULL,
    valence          FLOAT NOT NULL,
    confidence       FLOAT NOT NULL,
    coherence        FLOAT NOT NULL,
    social_charge    FLOAT NOT NULL,
    curiosity_drive  FLOAT NOT NULL,
    integrity        FLOAT NOT NULL,
    temporal_pressure FLOAT NOT NULL,

    -- 9D allostatic errors (moment horizon)
    energy_error           FLOAT,
    arousal_error          FLOAT,
    valence_error          FLOAT,
    confidence_error       FLOAT,
    coherence_error        FLOAT,
    social_charge_error    FLOAT,
    curiosity_drive_error  FLOAT,
    integrity_error        FLOAT,
    temporal_pressure_error FLOAT,

    -- Urgency signal
    urgency              FLOAT,
    dominant_error_dim   TEXT,
    dominant_error_val   FLOAT,

    -- Phase space
    nearest_attractor    TEXT,
    distance_to_bifurcation FLOAT,
    stage                TEXT,

    PRIMARY KEY (time, tenant_id)
);

SELECT create_hypertable(
    'interoceptive_state',
    'time',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_interoceptive_tenant_time
    ON interoceptive_state (tenant_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_interoceptive_urgency
    ON interoceptive_state (tenant_id, urgency DESC)
    WHERE urgency > 0.7;
```

### Neo4j: SomaticMarker Vector Index
**File to create**: `ecodiaos/database/migrations/neo4j_00003_somatic_vector_index.cypher`

```cypher
CREATE VECTOR INDEX somatic_marker_idx
FOR (m:SomaticMarker)
ON m.embedding
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 19,
        `vector.similarity_function`: 'cosine'
    }
};

// Add constraint on SomaticMarker id if not exists
CREATE CONSTRAINT somatic_marker_id IF NOT EXISTS
FOR (m:SomaticMarker) REQUIRE m.id IS UNIQUE;
```

## Integration Checklist

- [ ] Test Soma step 0 in clock (verify soma runs before atune)
- [ ] Test Atune precision modulation (check inertia changes with precision)
- [ ] Test Nova allostatic trigger (high urgency → goal reordering)
- [ ] Test Memory somatic markers (traces stamped, reranking works)
- [ ] Test Evo curiosity modulation
- [ ] Test Oneiros sleep pressure from energy errors
- [ ] Test Thymos constitutional gating
- [ ] Test Voxis expression modulation
- [ ] Run TimescaleDB migration
- [ ] Run Neo4j vector index creation
- [ ] Verify telemetry in interoceptive_state table
- [ ] Load test: verify all consumer reads handle Soma unavailable gracefully

## Notes

1. **Graceful Degradation**: All consumer systems check `if soma is not None` and `try/except` around signal reads. If Soma fails, systems fall back to pre-Soma behavior.

2. **Precision Weights Format**: Soma emits `precision_weights: dict[InteroceptiveDimension, float]` where values are 0-1. Consumers map these to dimension names (e.g., "arousal", "valence", "integrity").

3. **5ms Budget**: Soma must stay under 5ms per cycle. All consumers read from cache (no network/DB calls during theta cycle).

4. **Phase Space Snapshot**: Phase space update is every 100 cycles only (not every cycle) to stay within budget.

5. **Developmental Gating**: Some Soma features (e.g., counterfactuals) are gated on developmental stage. Consumers should respect these gates.
