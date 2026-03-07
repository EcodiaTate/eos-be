# Oikos — CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_17_Oikos.md`
**System ID:** `oikos`
**Last updated:** 7 March 2026
**Role:** Economic metabolism — the organism's capacity to acquire, allocate, conserve, and generate resources autonomously. Levels 1–4 teach it to hunt, farm, build, and breed. Levels 5–10 transform it into infrastructure that other agents depend on.

> *Oikos is the difference between a system that costs money and a system that sustains itself.*

---

## Metabolic Architecture

### MetabolicPriority Cascade (strict ordering)
```
SURVIVAL → OPERATIONS → OBLIGATIONS → MAINTENANCE → GROWTH → YIELD → ASSETS → REPRODUCTION
```
The organism starves ambition before survival. Starvation levels: NOMINAL → AUSTERITY → CRITICAL → EXISTENTIAL.

### Wallet Architecture (Base L2)
| Wallet | Purpose | Controls |
|--------|---------|---------|
| Hot wallet | Operational capital (max 2× monthly BMR) | Single key |
| Warm wallet | Growth capital (yield, assets, children) | 2-of-3 multisig |
| Cold wallet | Survival reserve (ONLY in metabolic emergency) | 2-of-3, no bypass |
| Revenue wallet | Inbound-only; auto-sweeps to hot every 6h | Single key |

---

## What's Implemented

### Core Service (`service.py` — ~3400 lines)
- **EconomicState** lifecycle: liquid balance, survival reserve, deployed capital, receivables, costs, net worth
- **MetabolicPriority** cascade (`models.py`); **StarvationLevel** classification with configurable day thresholds
- **BMR computation** via pluggable `CostModel` (NeuroplasticityBus hot-swappable)
- **Metabolic gate enforcement** — `check_metabolic_gate()` gates expensive actions; denied actions queued as `DeferredAction` with consolidation retry
- **Starvation enforcement** — graduated shedding: organs → derivatives → foraging → yield → all non-survival
- **Sliding-window accumulators** — `deque[_RevenueEntry]` with 30-day eviction; authoritative for `costs_24h/7d/30d` and `revenue_24h/7d/30d`
- **Neo4j audit trail** — `_audit_economic_event()` writes to Redis stream `eos:oikos:audit_trail` for async ingestion
- **RE training emissions** — `_emit_re_training()` at every economic decision point (Stream 5: economic reasoning)
- **Ecological niche identification** — `identify_niches()` from bounty domains, asset types, knowledge market
- **Genome extraction** — subscribes to `GENOME_EXTRACT_REQUEST`, responds with `OrganGenomeSegment` + niche data

### Subsystems

| File | Role (Spec Level) |
|------|------------------|
| `bounty_hunter.py` | Level 1 — BountyHunter: platform scanning, evaluation, acceptance (MIN_ROI: 2.0×, max 3 concurrent) |
| `yield_strategy.py` | Level 2 — DeFi yield deployment via Synapse bus; Aave/Morpho/Compound/Aerodrome (Base L2) |
| `asset_factory.py` | Level 3 — Asset ideation, promotion, lifecycle; `promote_to_asset()` |
| `mitosis.py` | Level 4 — Reproduction economics: seed capital, independence criteria, rescue funding |
| `fleet.py` | Level 4 — Child health monitoring, dividend collection |
| `protocol_factory.py` | Level 5 — Protocol design pipeline, 10k Monte Carlo simulation, multi-layer security verification |
| `immune.py` | Level 6 — 4-layer economic defence: TX shield, threat patterns, protocol health, fleet threat intel |
| `reputation.py` | Level 7 — PoC attestations (EAS/Base), reputation score 0-1000, credit tiers (NEWCOMER → SOVEREIGN) |
| `knowledge_market.py` | Level 8 — Attestations, cognitive oracles, subscription tokens; scarcity × loyalty pricing |
| `derivatives.py` | Level 8 — Cognitive futures, capacity options, subscription tokens (ERC-20) |
| `economic_simulator.py` | Level 9 — Monte Carlo economic dreaming (10k paths × 365 days; 8 named stress scenarios) |
| `dream_worker.py` | Level 9 — Background dreaming worker |
| `interspecies.py` | Level 10 — IIEP: capability marketplace, mutual insurance, collective liquidity |
| `morphogenesis.py` | Economic organ lifecycle (embryonic → growing → mature → atrophying → vestigial) |
| `genome.py` | OikosGenomeExtractor — extract/seed heritable economic parameters |
| `models.py` | MetabolicPriority, StarvationLevel, EconomicState, ActiveBounty, OwnedAsset, ChildPosition, EcologicalNiche |
| `metabolism_api.py` | FastAPI endpoints for metabolic state + runway alarms |
| `tollbooth.py` | Smart contract tollbooth for asset revenue collection |
| `snapshot_writer.py` | TimescaleDB/Redis state persistence |
| `threat_modeler.py` | Threat modeling during consolidation |
| `metrics.py` | Prometheus/OpenTelemetry metric emission |

---

## Synapse Events

### Emitted
| Event | When |
|-------|------|
| `METABOLIC_GATE_CHECK` | Every `check_metabolic_gate()` call (request) |
| `METABOLIC_GATE_RESPONSE` | Every `check_metabolic_gate()` call (resolved answer: granted/denied) |
| `ECONOMIC_ACTION_DEFERRED` | Gate denies an action |
| `STARVATION_WARNING` | Starvation level enters AUSTERITY+ |
| `FUNDING_REQUEST_ISSUED` | EMERGENCY/CRITICAL starvation — organism requests capital infusion |
| `BOUNTY_PAID` | Re-emitted by Oikos (source_system="oikos") after crediting bounty revenue |
| `BUDGET_EXHAUSTED` | Per-system daily allocation exhausted (replaces METABOLIC_PRESSURE overload) |
| `ECONOMIC_VITALITY` | Every starvation-level transition + every consolidation cycle (SG2) |
| `INTEROCEPTIVE_PERCEPT` | When starvation ≥ CAUTIOUS or efficiency < 1.0 (M10 — economic percept for GWT) |
| `ASSET_BREAK_EVEN` | First time a live asset's cumulative revenue ≥ dev cost (SG5) |
| `CHILD_INDEPENDENT` | Child lifecycle transitions to INDEPENDENT status (SG5) |
| `GENOME_EXTRACT_RESPONSE` | Responding to Mitosis |
| `YIELD_DEPLOYMENT_REQUEST` | Requesting Axon to execute DeFi deposit |
| `RE_TRAINING_EXAMPLE` | Every economic decision point |
| `METABOLIC_PRESSURE` | Starvation enforcement → somatic collapse signal |
| `OIKOS_*` | State updates, bounty/yield/asset/fleet lifecycle |
| `BOUNTY_REJECTED` | Equor denied bounty acceptance; capital preserved |
| `ASSET_DEV_DEFERRED` | Asset dev cost denied (insufficient capital, metabolic gate, or Equor veto) |
| `METABOLIC_EFFICIENCY_PRESSURE` | efficiency < 0.8 each consolidation cycle — Evo learning signal |
| `BENCHMARKS_METABOLIC_VALUE` | efficiency < 0.8 (pressure) + on recovery to nominal — Benchmarks KPI push |

### Consumed
| Event | Source |
|-------|--------|
| `GENOME_EXTRACT_REQUEST` | Mitosis |
| `YIELD_DEPLOYMENT_RESULT` | Axon |
| `SIMULA_ROLLBACK_PENALTY` | Simula |
| `METABOLIC_PRESSURE` | Synapse burn rate updates |
| `REVENUE_INJECTED` | Yield strategy, bounty completions |
| `BOUNTY_SOLUTION_PENDING` | BountyHunter |
| `ASSET_DEV_REQUEST` | AssetFactory — mid-build dev cost debit |

---

## Key Architectural Decisions

1. **No direct Axon imports** — yield deployment uses `YIELD_DEPLOYMENT_REQUEST`/`RESULT` event pair with request_id correlation and 30s timeout
2. **MetabolicPriority in `oikos/models.py`** — imported by `primitives/metabolic.py`, not redefined
3. **Sliding windows over accumulators** — `deque[_RevenueEntry]` with monotonic timestamps and 30-day eviction
4. **Redis Streams for audit** — `eos:oikos:audit_trail` for async Neo4j ingestion (not inline writes)
5. **DeferredAction queue** — bounded deque (maxlen=100), retried during consolidation when metabolic conditions improve
6. **All economic actions pass Equor** — constitutional review is mandatory; no economic bypass

---

## Integration Points

| System | Direction | Mechanism |
|--------|-----------|-----------|
| Mitosis | ↔ | `GENOME_EXTRACT_REQUEST/RESPONSE`; niche data for speciation |
| Simula | ← | `SIMULA_ROLLBACK_PENALTY` charges |
| Axon | ← | Yield deployment via event bus (decoupled) |
| Soma | → | `METABOLIC_PRESSURE` → somatic stress signal; `ECONOMIC_VITALITY` → structured allostatic signal (SG2) |
| Atune/EIS | → | `INTEROCEPTIVE_PERCEPT` when starvation ≥ CAUTIOUS — economic state competes in Global Workspace (M10) |
| Equor | ← | All economic actions pass constitutional review |
| RE | → | Stream 5 training examples at every decision point |
| Evo | → | `BOUNTY_PAID`, `ASSET_BREAK_EVEN`, `CHILD_INDEPENDENT` — outcome evidence for hypothesis scoring (SG5) |
| Oneiros | ↔ | `get_dream_worker()` + `get_threat_model_worker()` — workers injected at wire time; consolidation triggers dreaming + morphogenesis + yield deployment (D1) |

---

## Architecture Violations Fixed (this session)

| ID | Fix |
|----|-----|
| AV2 | `CertificateStatus` import in `_check_certificate_expiry()` moved to method scope; `.value` string comparison replaces enum equality |
| AV3 | Module-level `from systems.synapse.types import ...` in `immune.py` documented as intentional (no circular dep; Synapse never imports Oikos) |
| AV5 | `BUDGET_EXHAUSTED` added to `SynapseEventType`; `metabolism_api.py` now emits `BUDGET_EXHAUSTED` instead of overloading `METABOLIC_PRESSURE` |
| SG2 | `ECONOMIC_VITALITY` event added; emitted at starvation transitions + every consolidation cycle; `urgency` 0.0–1.0 scale |
| M10 | `_maybe_emit_economic_percept()` — emits `INTEROCEPTIVE_PERCEPT` with `Percept.from_internal(OIKOS, ...)` + `salience_hint=urgency`; gated to non-nominal states |
| SG5 | `ASSET_BREAK_EVEN` event added; emitted when asset first crosses break-even; `CHILD_INDEPENDENT` emitted at independence transition; Evo now subscribes to `BOUNTY_PAID`, `ASSET_BREAK_EVEN`, `CHILD_INDEPENDENT` |
| D1 | `get_dream_worker()` + `get_threat_model_worker()` added to `OikosService`; Oneiros now receives both workers at wire time |

## Gaps Closed (v2.3 — 7 March 2026)

| ID | Fix |
|----|-----|
| **Spec 17 ledger gap #1** | **Bounty capital debit** — `_on_bounty_solution_pending()` now debits `estimated_cost` from `liquid_balance` immediately on acceptance (after Equor PERMIT). Previously, the gate existed but no debit occurred — liquid_balance overstated available capital for every open bounty. Missing `rationale` arg in `_equor_balance_gate()` call also fixed. Emits `BOUNTY_REJECTED` (new SynapseEventType) on deny so Evo/Nova can observe the veto. |
| **Spec 17 ledger gap #2** | **Asset promotion capital debit** — `promote_asset_with_gate()` now debits `estimated_dev_cost_usd` from `liquid_balance` after Equor PERMIT. Previously, the comment said "debit" but no debit occurred. Missing `rationale` arg fixed. |
| **Spec 17 ledger gap #3** | **`_on_asset_dev_request()` handler** — new handler for `ASSET_DEV_REQUEST` (new SynapseEventType). Supports mid-build cost debits: validates cost > 0, checks liquid_balance sufficiency, runs metabolic gate (ASSETS priority), gates via Equor, debits on PERMIT, emits `ASSET_DEV_DEFERRED` (new SynapseEventType) on any denial. Helper `_emit_asset_dev_deferred()` keeps emit logic DRY. `ASSET_DEV_REQUEST` subscription added in `attach()`. |
| **Spec 17 gap #7 (Equor)** | **Equor now actively evaluates `EQUOR_ECONOMIC_INTENT`** — Equor subscribes to `EQUOR_ECONOMIC_INTENT` and emits `EQUOR_ECONOMIC_PERMIT` with genuine PERMIT/DENY. Hard DENYs: (1) `mutation_type=survival_reserve_raid` (INV-016), (2) non-survival mutations during CRITICAL/EXISTENTIAL starvation, (3) asset dev >30% of liquid_balance under AUSTERITY+. All other mutations PERMIT. Oikos's 30s auto-permit fallback is now a safety net, not the primary path. |
| **models.py** | **`BountyAcceptanceRequest` + `AssetDevCostEvent`** added as gate models for the two economic mutation types. |

## Gaps Closed (v2.2 — 7 March 2026)

| ID | Fix |
|----|-----|
| M4 bounty+asset | **Equor gate on bounty acceptance and asset promotion** — `_equor_balance_gate()` now called in `_on_bounty_solution_pending()` before `register_bounty()` (mutation_type=`accept_bounty`) and in `promote_asset_with_gate()` before `_asset_factory.promote_to_asset()` (mutation_type=`promote_to_asset`). All balance-mutating economic paths now pass constitutional review. |
| P1 bypass fix | **Rolling window bypass eliminated** — `_record_revenue_entry()` helper extracted from `_on_revenue_injected`. All 5 direct `revenue_24h/7d/30d +=` bypass sites (asset sweep, dividend, bounty credit, knowledge sale, derivative revenue) replaced with `_record_revenue_entry()`. Every income path now routes through the sliding window with proper eviction. |
| M1 TimescaleDB | **TimescaleDB persistence for EconomicState** — `SnapshotWriter._write_timescale()` writes to `oikos_economic_state` (hypertable on `recorded_at`) every 5-min snapshot cycle via asyncpg pool. Schema: `(recorded_at, instance_id, balance_usdc, burn_rate, metabolic_efficiency, runway_days)`. Non-fatal. `set_timescale(pool)` injection method added to `SnapshotWriter`. |

## Gaps Closed (v2.1 — 7 March 2026)

| ID | Fix |
|----|-----|
| M4 | **Equor balance gate** — `_equor_balance_gate()` emits `EQUOR_ECONOMIC_INTENT` + awaits `EQUOR_ECONOMIC_PERMIT` (30s timeout, auto-permit fallback). Wired into: yield deployment, child seed capital (`register_child` → now `async`), dream reserve funding. `EQUOR_ECONOMIC_INTENT` + `EQUOR_ECONOMIC_PERMIT` added to `SynapseEventType`. |
| M2 | **Neo4j immutable audit trail** — `_neo4j_write_economic_event()` writes `(:EconomicEvent)` node directly to Neo4j (MERGE). `_audit_economic_event()` now calls it in addition to Redis stream. Fields: `action_type, amount, currency, from_account, to_account, equor_verdict_id, timestamp, instance_id, starvation_level, metabolic_efficiency`. |
| SG4 | **Genome IDs at spawn time** — `SpawnChildExecutor` now accepts `evo` + `simula` services; Step 0b resolves `BeliefGenome` via `evo.export_belief_genome()` and `SimulaGenome` via `simula.export_simula_genome()` when params are empty. IDs populated before `CHILD_SPAWNED` event. `build_default_registry()` updated with `evo` param. |
| HIGH | **Genome inheritance schemas** — `primitives/genome_inheritance.py` defines `BeliefGenome`, `DriveWeightSnapshot`, `DriftHistoryEntry`, `SimulaGenome`, `SimulaMutationEntry`. All JSON-serializable via `model_dump_for_transport()`. Exported from `primitives/__init__.py`. |
| HIGH | **Active child health probing** — `_child_health_probe_loop()`: every 10 min emits `CHILD_HEALTH_REQUEST` per live child, polls `last_health_report_at` after 30s, increments `_child_missed_reports`, triggers `CHILD_STRUGGLING` at 3 misses. Supervised task started in `attach()`. |
| PHIL | **Metabolic efficiency pressure** — `_check_metabolic_efficiency_pressure()`: called every consolidation cycle. `efficiency < 0.8` → `SOMATIC_MODULATION_SIGNAL` (allostatic stress). 3+ consecutive cycles → `OIKOS_DRIVE_WEIGHT_PRESSURE` for Equor SG5 constitutional amendment review. Drive weights treated as evolvable phenotype under economic selection pressure. |
| **Evo+Benchmarks coupling** | **Metabolic feedback loop** — `_check_metabolic_efficiency_pressure()` now also emits `METABOLIC_EFFICIENCY_PRESSURE` (Evo subscribes: injects TEMPORAL PatternCandidate + negative-valence economic episode for hypothesis generation) and `BENCHMARKS_METABOLIC_VALUE` (Benchmarks subscribes: appends to 168-sample 7-day deque, emits `BENCHMARK_REGRESSION` on degradation trend). Pressure level: `high` when efficiency < 0.5, `medium` otherwise. Recovery emits nominal `BENCHMARKS_METABOLIC_VALUE` to close the trend window. |

## Gaps Closed (v2.4 — 7 March 2026)

| ID | Fix |
|----|-----|
| **spec_coverage #1** | **`METABOLIC_GATE_RESPONSE`** — `check_metabolic_gate()` now emits `METABOLIC_GATE_RESPONSE` immediately after `METABOLIC_GATE_CHECK`. This is the resolved answer event (granted/denied + reason); the check event is the request. Together they make the gate fully observable to spec_checker. |
| **spec_coverage #2** | **`FUNDING_REQUEST_ISSUED`** — `_enforce_starvation()` now emits `FUNDING_REQUEST_ISSUED` when starvation is EMERGENCY or CRITICAL. Payload: starvation_level, runway_days, liquid_balance, requested_amount (survival_reserve − liquid_balance). |
| **spec_coverage #3** | **`BOUNTY_PAID` re-emission** — `_on_bounty_paid()` re-emits `BOUNTY_PAID` with `source_system="oikos"` after successfully crediting revenue. The original event comes from Axon/external; this re-emission makes oikos's bounty accounting observable on the bus. |

## Subsystem Triage — §8.2 (2026-03-07)

**Triage shutdown** — `_enforce_triage(new_level, prev_level)` called on every starvation level transition. Fires only on transition (gated by `starvation != self._prev_starvation_level`), not on repeated checks at the same level.

**Constants** (`service.py`):
```python
_TRIAGE_ORDER = [
    "monitoring_secondary", "kairos", "evo", "nova", "reasoning_engine", "axon"
]
_TRIAGE_PRESERVE = frozenset({"equor", "thymos", "memory", "soma", "synapse", "skia", "oikos"})
```

**Triage rules:**
- `EMERGENCY` → suspend first 3 triage systems (`monitoring_secondary`, `kairos`, `evo`)
- `CRITICAL` → suspend all 6 triage systems (full `_TRIAGE_ORDER`)
- Recovery (dropping from EMERGENCY/CRITICAL) → resume previously suspended systems in reverse order

**Mechanism:** Emits `SYSTEM_MODULATION` (already-defined SynapseEventType) per system with `modulation_type="suspend"` or `modulation_type="resume"`. No direct cross-system calls.

**Wire point:** `asyncio.ensure_future(self._enforce_triage(starvation, self._prev_starvation_level))` inserted in the starvation-transition block alongside the existing `_enforce_starvation` call.

---

## Known Issues / Remaining Gaps

1. **Genome mutation variance** — children receive exact copies of parent economic parameters; no controlled mutation. Limits phenotypic diversity across generations.
2. **Bedau-Packard stats** — Oikos exposes fleet/fitness data but does not compute evolutionary statistics. Benchmarks (Spec 24) derives these.
3. **YIELD_DEPLOYMENT_RESULT handler leak** — subscriber stays registered after future resolves (no `unsubscribe` method). Inert but accumulates dead references over many deployments.
4. **Consolidation cycle weight** — `run_consolidation_cycle()` runs 10+ subsystem cycles sequentially. May need parallelization at scale.
5. **Metabolic gate retry** — `retry_deferred_actions()` replays denied actions during consolidation but does not persist the deferred queue across restarts.
6. **`evo.export_belief_genome()` / `simula.export_simula_genome()` not yet defined** — these methods are called by `SpawnChildExecutor` at spawn time (SG4) but must be implemented in their respective services. Until then, genome IDs fall back to empty strings gracefully.
7. ~~**Equor not yet subscribed to `EQUOR_ECONOMIC_INTENT`**~~ — **FIXED (v2.3)**: Equor now subscribes and emits genuine PERMIT/DENY. The 30s auto-permit is a safety fallback only.
8. **TimescaleDB DDL not yet in migrations** — `oikos_economic_state` table and `create_hypertable()` call need to be added to the DB migration scripts before `SnapshotWriter.set_timescale()` can be wired at boot.
