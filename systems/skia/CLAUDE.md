# Skia System CLAUDE.md

## What's Implemented

### Core Infrastructure (Original)
- **HeartbeatMonitor** (`heartbeat.py`): 3-phase detection (observe → suspect → confirm). Redis pub/sub listener + periodic checker. Exponential backoff on reconnect. False-positive prevention: Redis PING probe + recovery-during-confirmation check.
- **StateSnapshotPipeline** (`snapshot.py`): Neo4j export → gzip → Fernet encrypt → IPFS pin. Atomic Redis staging → promotion (stale key safe). CID sorted-set history. Round-trip integrity verification before CID promotion. `restore_from_ipfs()` standalone function for startup restoration. Sets `skia:restoration_complete:{id}` flag on success.
- **RestorationOrchestrator** (`restoration.py`): Cloud Run restart (GCP Admin API, RS256 JWT, env var injection) → Akash deploy (SDL template, ACTIVE state polling, 10-min timeout). Redis distributed lock (fencing token, 15-min TTL, renewal). Max 3 attempts before `infrastructure_dead = True`.
- **PinataClient** (`pinata_client.py`): IPFS pin/unpin/list/fetch. Auto-pruning beyond retention limit.

### Phase 1 (Chat 5) — VitalityCoordinator (`vitality.py`)
- 5 fatal thresholds from `primitives/vitality.py`: RUNWAY_FATAL (< 0.5 days), BRAIN_DEATH (effective_I < 0.01, 7-day sustained), NORMATIVE_COLLAPSE (drift × 12 > 10), IMMUNE_FAILURE (failure_rate > 0.9, 48h sustained), SOMATIC_COLLAPSE (allostatic_error > 0.8, 48h sustained)
- Independent 30s check loop — organism cannot prevent its own death
- 3-phase death sequence: Warning (5 min, recoverable) → Capture (IPFS snapshot + genome + Neo4j death record) → Cessation (clock.force_stop() + ORGANISM_DIED + fleet notify)
- Progressive austerity: CAUTIOUS → AUSTERITY → EMERGENCY → CRITICAL via `SYSTEM_MODULATION` events
- Resurrection protocol (external-only): `resurrect(trigger)` resets all death state
- Fleet notification: `CHILD_DIED` (to parent) + `FEDERATION_BROADCAST` (to children)
- Neo4j: `VitalityReport` node every 30s, `OrganismDeathRecord` on death, `OrganismGenome {is_final: true}` on capture
- **Subscriptions**: `SOMA_VITALITY_SIGNAL` (cached urgency/allostatic_error/coherence_stress), `METABOLIC_PRESSURE` (→ austerity enforcement via `handle_metabolic_pressure()`) — **both wired in `set_event_bus()`**

### Phase 2 (Chat 6) — Speciation Alignment (`phylogeny.py`)
- **Heritable variation**: `mutate_parameters()` with Gaussian perturbation (max ±5%) on restoration
- **Genome-level mutation**: `mutate_genome_segments()` for full organism genome mutation
- **PhylogeneticTracker**: Neo4j lineage graph — `PhylogeneticNode` nodes, `SPAWNED` edges (parent→child), `DIED_AS` edges (node→death record)
- **Evolvable thresholds**: `EVO_PARAMETER_ADJUSTED` subscription for live config hot-reload (heartbeat + vitality params)
- **Metabolic gating**: `_check_metabolic_budget()` warns if Oikos runway < 1 day before restoration
- **Dry-run**: `dry_run_restoration()` validates snapshot + strategy + budget without committing, emits `SKIA_DRY_RUN_COMPLETE`

### Speciation Enhancements (Chat 8 — 2026-03-07)

**1. Constitutional Genome in IPFS Snapshot** (`snapshot.py`, `types.py`)
- `StateSnapshotPipeline` accepts optional `memory: MemoryService` (via `__init__` or `set_memory()`)
- `take_snapshot()` calls `memory.export_genome()` between Neo4j export and serialization
- Result stored in `SnapshotPayload.constitutional_genome` (schema_version bumped to "2")
- `SnapshotPayload.last_constitutional_genome` property exposes the last captured genome
- `restore_from_ipfs()` now returns `dict | None` (the constitutional genome) so callers can apply it
- `SkiaService.set_memory()` wires Memory — must be called at organism bootstrap
- **Wire point**: `skia.set_memory(memory)` before `initialize()`

**2. Drive Weights in Shadow Provisioning** (`restoration.py`, `service.py`)
- `RestorationOrchestrator.set_constitutional_genome(genome)` stores the genome for injection
- Cloud Run restart: injects `ECODIAOS_CONSTITUTIONAL_GENOME_B64` = `base64(orjson(genome))` env var alongside `ECODIAOS_SKIA_RESTORE_CID`
- Akash deploy: injects genome into SDL template placeholder + API payload `env` dict
- `_on_death_confirmed()` syncs `snapshot.last_constitutional_genome` → restoration before calling `restore()`

**3. Fleet Resurrection Coordination** (`service.py`)
- `_detect_simultaneous_deaths()` — writes to `skia:fleet:recent_deaths` sorted set (TTL 120s), returns count of deaths in last 60s
- `_coordinate_fleet_resurrection()` — emits `SKIA_RESURRECTION_PROPOSAL`, waits 30s for `FEDERATION_RESURRECTION_APPROVED`
  - If approved and we are elected leader → proceed with restoration
  - If another instance is leader → stand down (return False)
  - If timeout (30s) → proceed autonomously
- `_on_federation_resurrection_approved()` — handles `FEDERATION_RESURRECTION_APPROVED`, overrides local CID with federation-selected snapshot_cid
- New subscription: `FEDERATION_RESURRECTION_APPROVED` in `_subscribe_events()`

**4. CRITICAL System 45s Heartbeat Gap** (`heartbeat.py`, `service.py`)
- `CRITICAL_SYSTEMS = frozenset({"equor", "thymos", "memory"})`
- `_CRITICAL_FAILURE_THRESHOLD = 9` (9 × 5s = 45s)
- `HeartbeatMonitor.__init__` accepts `on_critical_system_silent: Callable[[str], Awaitable[None]] | None`
- `_subscribe_loop()` parses heartbeat JSON for `"system"` field → updates `_critical_system_last_seen[system]`
- `_check_critical_systems()` — called every 5s poll cycle; fires callback once per silence episode, auto-clears on recovery
- `SkiaService._on_critical_system_silent()` — emits `SKIA_HEARTBEAT_LOST` (severity=CRITICAL) + `SYSTEM_MODULATION` (austerity=EMERGENCY)

**New SynapseEventTypes**: `SKIA_RESURRECTION_PROPOSAL`, `FEDERATION_RESURRECTION_APPROVED`

### Gap Fixes (Chat 7 — 2026-03-07)
- **`SkiaConfig`**: Added `mutation_rate`, `mutation_magnitude`, `worker_heartbeat_interval_s` fields (were stranded as comments on one line in config.py)
- **`SynapseEventType`**: Added `SYSTEM_MODULATION`, `DEGRADATION_OVERRIDE`, `FEDERATION_BROADCAST` (emitted by VitalityCoordinator but missing from enum — caused `ValueError` at runtime)
- **`VitalityCoordinator.set_event_bus()`**: Added `METABOLIC_PRESSURE` subscription → `_on_metabolic_pressure()` → `handle_metabolic_pressure()` (method existed but was never triggered)
- **`SkiaService._subscribe_events()`**: Added `METABOLIC_PRESSURE` → `_on_metabolic_pressure()` forwarder to VitalityCoordinator (belt-and-suspenders for early-wiring case)
- **`SnapshotManifest.cid`**: Added `.cid` property alias for `ipfs_cid` (vitality.py called `manifest.cid` which would AttributeError at death capture time)

### Gap Fixes (Chat 9 — 2026-03-07)

**1. Constitutional genome applied on restoration** (`snapshot.py`)
- `restore_from_ipfs()` gains optional `event_bus` and `memory` params
- After Neo4j import: calls `memory.seed_genome(constitutional_genome)` if Memory is wired
- Emits `GENOME_EXTRACT_REQUEST` via Synapse bus with genome payload so Memory/Equor reinitialize state
- Without this, revived organisms started with default drives

**2. Federation subscription to `SKIA_RESURRECTION_PROPOSAL`** (`federation/service.py`)
- `FederationService.set_event_bus()` now subscribes to `SKIA_RESURRECTION_PROPOSAL`
- `_on_skia_resurrection_proposal()` handler: collects ALLY-trust links, fans out 30s polls to each peer, collects snapshot CIDs, selects the most recent one
- Quorum: >50% of ALLY peers must acknowledge within 60s; if no ALLY peers, auto-approves after 10s
- Emits `FEDERATION_RESURRECTION_APPROVED` (leader + best snapshot_cid) — Skia receives this via its existing `_on_federation_resurrection_approved()` handler
- Fail-safe: emits approval even on quorum failure (survival imperative outweighs coordination)

**3. Post-mortem death learning** (`vitality.py`)
- `_log_death_record()` now emits two additional signals after writing `OrganismDeathRecord`:
  - `RE_TRAINING_EXAMPLE` — task_type=`organism_death_analysis`, input = {cause, age_hours, final_economic_state, final_allostatic_state, fatal_thresholds}, label = `"death"`
  - `INCIDENT_DETECTED` — severity=HIGH, category=`organism_death` — Thymos creates an incident that the next incarnation reads on boot
- `OrganismDeathRecord` Neo4j node gains `age_hours`, `runway_days`, `allostatic_error` fields
- Age computed from `PhylogeneticNode.born_at` (best-effort; stays 0.0 if unavailable)

**4. Dry-run data integrity verification** (`service.py`)
- `dry_run_restoration()` now runs `_dry_run_integrity_check()` when CID + pinata + vault are all available
- Downloads the snapshot, decrypts (key_version from manifest), decompresses, compares actual vs. manifest node/edge counts
- If deviation > 5% for nodes OR edges: sets `data_integrity_ok=False`, `integrity_failure_reason="data_integrity_mismatch"`, `predicted_outcome="failed_data_integrity_mismatch"`
- Check failure (exception) is a soft warning — doesn't block restoration (snapshot may still be valid)
- `result` dict gains: `actual_node_count`, `actual_edge_count`, `manifest_node_count`, `manifest_edge_count`, `node_deviation_pct`, `edge_deviation_pct`

**5. Akash SDL template created** (`config/skia/akash_sdl_template.yaml`)
- Minimal valid Akash SDL with `${DOCKER_IMAGE}`, `${ECODIAOS_SKIA_RESTORE_CID}`, `${ECODIAOS_CONSTITUTIONAL_GENOME_B64}` substitution placeholders
- Matches exactly the `str.replace()` calls in `RestorationOrchestrator._deploy_akash()`
- Default resources: 0.5 CPU, 512Mi memory, 1Gi storage
- Path matches `SkiaConfig.akash_sdl_template_path` default: `"config/skia/akash_sdl_template.yaml"`

**6. Fixed hardcoded `key_version=1`** (`snapshot.py`)
- `restore_from_ipfs()` reads `encryption_key_version` from the snapshot manifest stored in Redis (`skia:snapshot:manifest`)
- Falls back to 1 with a `warning` log if Redis is unavailable or manifest lacks the field
- Prevents decryption failures when key rotation occurs between snapshot and restoration

**7. Bedau-Packard fitness observables** (`phylogeny.py`, `service.py`)
- `PhylogeneticTracker` gains `set_event_bus()`, `start()`, `stop()`, `_fitness_emit_loop()`, `_emit_fitness_batch()`
- 24h background loop queries all `PhylogeneticNode` rows with child count + death record joins
- Emits `FITNESS_OBSERVABLE_BATCH` with per-node: `instance_id`, `generation`, `survival_hours`, `reproduction_count`, `revenue_earned`, `cause_of_death`
- `SkiaService.set_event_bus()` propagates bus to `_phylogeny.set_event_bus()`
- `SkiaService.initialize()` propagates bus if already set at init time
- `SkiaService.start()` calls `phylogeny.start()` only if event bus is wired
- `SkiaService.shutdown()` calls `phylogeny.stop()`

## Files
- `service.py` — SkiaService orchestrator (entry point)
- `vitality.py` — VitalityCoordinator (DO NOT modify death sequence)
- `phylogeny.py` — PhylogeneticTracker + MutationConfig + mutation engine + fitness loop
- `heartbeat.py` — HeartbeatMonitor
- `snapshot.py` — StateSnapshotPipeline + restore_from_ipfs()
- `restoration.py` — RestorationOrchestrator
- `pinata_client.py` — Pinata IPFS client
- `types.py` — Domain models
- `config/skia/akash_sdl_template.yaml` — Akash SDL (NOT in backend/)

## Key Constraints
- VitalityCoordinator death sequence is the death authority — do not modify
- VitalityThreshold primitives in `primitives/vitality.py` are source of truth
- `wire_vitality_systems()` on SkiaService is the wiring entry point
- Mutation rates must be small enough to preserve organism viability (max 5%)
- All inter-system comms via Synapse events — no direct imports

## Genuine Precariousness — §8.2 (2026-03-07)

**New file: `systems/skia/degradation.py`**

`DegradationEngine` — independent hourly entropy loop. Runs alongside the VitalityCoordinator 30s check loop; does NOT slow down on death-proximity.

**Key classes:**
- `DegradationConfig` — rates from env vars (`DEGRADATION_MEMORY_DECAY_RATE=0.02`, `DEGRADATION_CONFIG_DRIFT_RATE=0.01`, `DEGRADATION_HYPOTHESIS_STALENESS_RATE=0.05`, `DEGRADATION_TICK_INTERVAL_S=3600`)
- `DegradationSnapshot` — accumulates cumulative pressure. `degradation_pressure`: memory×0.40 + config×0.20 + hypothesis×0.40, capped at 1.0. Counteraction: `counteract_memory()`, `counteract_config()`, `counteract_hypotheses()`
- `DegradationEngine` — `start()` / `stop()` / `tick()` (public for tests). Emits 4 events per tick: `MEMORY_DEGRADATION`, `CONFIG_DRIFT`, `HYPOTHESIS_STALENESS`, `DEGRADATION_TICK`. Counteraction API: `on_memory_consolidated(0.5)`, `on_config_optimised(0.8)`, `on_hypotheses_revalidated(0.6)`

**VitalityCoordinator wiring** (`vitality.py`):
- `self._degradation = DegradationEngine(config=DegradationConfig(), instance_id=instance_id)`
- `set_event_bus()` wires bus + subscribes:
  - `ONEIROS_CONSOLIDATION_COMPLETE` → `on_memory_consolidated(0.5)`
  - `EVO_PARAMETER_ADJUSTED` → `on_config_optimised(0.1)`
  - `EVO_BELIEF_CONSOLIDATED` → `on_hypotheses_revalidated(0.6)`
- Vitality report now includes `degradation_pressure` + `degradation_tick_count`

**New SynapseEventTypes**: `DEGRADATION_TICK`, `MEMORY_DEGRADATION`, `CONFIG_DRIFT`, `HYPOTHESIS_STALENESS`

**Subscriber implementations** (Round 2A complete — 7 Mar 2026):
- Memory `_on_memory_degradation()` — **IMPLEMENTED**: decays `Episode.salience *= (1 - fidelity_loss_rate)` for unconsolidated episodes older than `affected_episode_age_hours`; soft-deletes (sets `decayed=true`) episodes below 0.01; emits `MEMORY_EPISODES_DECAYED`
- Evo `_on_hypothesis_staleness()` — **IMPLEMENTED**: decays `evidence_score *= (1 - staleness_rate)` on all PROPOSED/TESTING hypotheses; archives those below 0.05; emits `EVO_HYPOTHESES_STALED` + `EVO_HYPOTHESIS_REVALIDATED` (closes VitalityCoordinator feedback loop)
- Simula `_on_config_drift()` — **IMPLEMENTED**: applies `Gaussian(0, drift_rate)` noise to `min(num_params_affected, 23)` learnable config params; clamps to per-param bounds; emits `SIMULA_CONFIG_DRIFTED`
- Soma `_on_degradation_tick()` — **NEW** (Round 2A): subscribes to `DEGRADATION_TICK`; raises external stress +0.1 (pressure > 0.5) or +0.3 (pressure > 0.8) via `inject_external_stress()` — organism somatically feels its own entropy

**New SynapseEventTypes** (Round 2A): `MEMORY_EPISODES_DECAYED`, `EVO_HYPOTHESES_STALED`, `EVO_HYPOTHESIS_REVALIDATED`, `SIMULA_CONFIG_DRIFTED`

---

## Remaining Gaps (spec ref)
- **Sec 15**: SACM integration for compute cost quota checks before restoration (currently direct GCP/Akash only)
- **Sec 18.3**: Circuit breaker for repeated restoration failures — currently just `infrastructure_dead` flag with no human-notification path
- **Federation quorum polling**: `ChannelManager.send_message()` is designed for handshake/exchange payloads, not generic poll messages — resurrection poll degrades gracefully to auto-approve when peers don't respond (survival imperative preserved)
- **Genome inheritance (startup scripts)**: `restore_from_ipfs()` now applies genome internally when `event_bus`/`memory` are passed, but startup scripts must be updated to pass these args

## Integration Surface
### Emits
`skia_heartbeat`, `skia_heartbeat_lost`, `skia_restoration_triggered`, `skia_restoration_started`, `skia_restoration_complete`, `skia_restoration_completed`, `skia_snapshot_completed`, `skia_dry_run_complete`, `skia_resurrection_proposal`, `organism_spawned`, `organism_died`, `organism_resurrected`, `vitality_report`, `vitality_fatal`, `vitality_restored`, `metabolic_cost_report`, `system_modulation`, `degradation_override`, `child_died`, `federation_broadcast`, `re_training_example`, `incident_detected`, `fitness_observable_batch`, `genome_extract_request`

**Gap closure (2026-03-07, event coverage):**
- `skia_snapshot_completed` — now emitted by `StateSnapshotPipeline.take_snapshot()` after each successful IPFS pin. Requires `snapshot.set_event_bus(bus)` (wired in `SkiaService.set_event_bus()` and `initialize()`).
- `skia_restoration_triggered` — now emitted in `SkiaService._on_death_confirmed()` immediately when restoration begins, before metabolic gates or fleet coordination, so the observatory can track all restoration attempts including blocked ones.

### Subscribes
`SOMA_VITALITY_SIGNAL`, `METABOLIC_PRESSURE`, `EVO_PARAMETER_ADJUSTED`, `ORGANISM_DIED`, `FEDERATION_RESURRECTION_APPROVED`

### Wiring Points (must be called at organism bootstrap)
- `skia.set_memory(memory_service)` — enables constitutional genome in snapshots
- `skia.set_event_bus(event_bus)` — enables Synapse integration + resurrection coordination + phylogeny fitness loop
- `skia.wire_vitality_systems(clock, oikos, thymos, equor, telos)` — wires death system
- `restore_from_ipfs(..., event_bus=bus, memory=memory)` — enables genome application on cold startup restoration
