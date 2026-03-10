# Synapse — System CLAUDE.md

**Role:** Autonomic nervous system. Cognitive clock, health monitor, resource allocator, coherence tracker, metabolic tracker. The infrastructure that makes EOS a loop, not a pipeline.
**Spec:** `.claude/EcodiaOS_Spec_09_Synapse.md`

---

## What's Implemented

### CognitiveClock (`clock.py`)
- Theta rhythm at ~150ms base (range 50–500ms), ~6.7 Hz
- Arousal-modulated period: EMA smoothing (α=0.1) on arousal input from Soma
- Coherence drag: `set_coherence_drag()` slows clock when composite < 0.4
- Grid-aware CONSERVATION throttle: 1 Hz on `GRID_METABOLISM_CHANGED`
- Per-cycle execution: Step 0 Soma → Step 1 Atune → Step 2 post-cycle (detectors, coherence, resource snapshot, events)
- "Never dies" — supervised restart loop with 3-attempt exponential backoff
- Soma step-0 bypass: >50ms for 3 consecutive cycles → bypass; probe recovery every 50 cycles
- `force_stop(reason)` — external forceful stop callable by VitalityCoordinator during death sequence

### CognitiveClock — Phase 2 additions (`clock.py`)
- `set_event_bus(event_bus)` — wires EventBus for THETA_CYCLE_START / THETA_CYCLE_OVERRUN
- `THETA_CYCLE_START` emitted fire-and-forget via `asyncio.ensure_future` before every tick
- `THETA_CYCLE_OVERRUN` emitted when elapsed > budget (includes overrun_count)
- `InteroceptiveDimension` import moved to module level (AV8 fix — was inside hot-path every 150ms)
- `SystemLoad` imported from `synapse/types.py` not `fovea/types.py` (AV7 fix)

### HealthMonitor (`health.py`)
- 5-second polling, 2-second per-system timeout
- State machine: STOPPED → STARTING → HEALTHY ← DEGRADED ← FAILED (with OVERLOADED)
- FAILED: 3 consecutive missed heartbeats; FAILED recovery requires 3 successes; DEGRADED/OVERLOADED recover on first success
- Critical systems (equor, memory, atune) → safe mode; safe mode exits when all critical systems healthy
- Infrastructure checks: Redis and Neo4j every 15s
- Supervised background task with 3-restart exponential backoff
- **M8 (Phase 2 Chat 7)**: `SYSTEM_HEALTH_CHANGED` emitted on every status transition — STARTING→HEALTHY, HEALTHY→DEGRADED, HEALTHY→OVERLOADED, any→FAILED. Previously only FAILED and RECOVERED were visible on the bus.

### ResourceAllocator (`resources.py`)
- Default per-system CPU budgets (core systems 0.10–0.20; secondary 0.02–0.08; total ~1.0)
- Burst allowance up to 2× for overloaded (>80%) systems; priority boost for >90%
- psutil-backed snapshots with fallback to zeros
- `BaseResourceAllocator` ABC — hot-swap strategy via `NeuroplasticityBus`
- `capture_snapshot()` every 33 cycles, `rebalance()` every 100 cycles

### EmergentRhythmDetector (`rhythm.py`)
- 6 states: IDLE, NORMAL, FLOW, BOREDOM, STRESS, DEEP_PROCESSING
- Priority-ordered rule detection, rolling 100-cycle window, 20-cycle hysteresis
- `DefaultRhythmStrategy` pluggable via `BaseRhythmStrategy` ABC, hot-swap via `set_strategy()`
- Rhythm state pushed to Atune via `set_rhythm_state()` every cycle
- STRESS entry emits `FOVEA_PREDICTION_ERROR`
- Rhythm transitions emit `EPISODE_STORED` for Thread narrative identity

### CoherenceMonitor (`coherence.py`)
- 4 metrics: `phi_approximation` (IIT proxy), `system_resonance`, `broadcast_diversity`, `response_synchrony`
- Composite: 0.35×phi + 0.25×resonance + 0.20×diversity + 0.20×synchrony
- Coherence drag applied when composite < 0.4
- `COHERENCE_SHIFT` emitted on significant composite changes
- `FOVEA_PREDICTION_ERROR` emitted on coherence drops

### DegradationManager (`degradation.py`)
- Per-system strategies for all 29 systems
- Auto-restart with exponential backoff, max 2–3 attempts per system
- Cascade detection: dependents notified on dependency failure via `_propagate_cascade()`
- Dependency graph with topological sort for ordered hot-reload restarts
- `restart_batch_for_reload()` — shutdown-then-init ordering

### MetabolicTracker (`metabolism.py`)
- O(1) hot-path `log_usage()`, no allocations, EMA burn rate (α=0.15)
- **Provider-aware pricing**: `_PROVIDER_PRICING` dict maps provider tags to (input, output) price per token. `"re"`, `"vllm"`, `"ollama"`, `"local"` = $0; `"claude-haiku"` = $0.80/$4; `"claude-sonnet"` = $3/$15; `"claude-opus"` = $15/$75 per 1M tokens
- **Two-dimension cost model**: API costs (per-token, EMA burn rate) + Infrastructure costs (per-hour, polled from provider APIs). `burn_rate_usd_per_hour = API EMA + infra hourly`
- Per-system + per-provider cost breakdown
- Rolling deficit = total_cost (API + infra) − total_revenue
- Infrastructure tracking: `update_infrastructure_cost()`, `remove_infrastructure_resource()`, `accrue_infrastructure_cost()`. Per-resource dict (e.g. `"runpod:pod_abc123" → $0.74/hr`)
- `inject_revenue()` emits `REVENUE_INJECTED` at salience=1.0
- Window reset every 50 cycles; `hours_until_depleted = balance / burn_rate_hour`
- `MetabolicSnapshot` extended: `api_cost_usd_per_hour`, `infra_cost_usd_per_hour`, `total_api_cost_usd`, `total_infra_cost_usd`, `per_provider_cost_usd`, `infra_resources`
- **M1 (Phase 2 Chat 7)**: SynapseService now subscribes to `REVENUE_INJECTED` and `STARVATION_WARNING` from Oikos. `_on_oikos_revenue_injected` forwards earned revenue to the tracker reactively (deficit was previously unbounded if Oikos never called `inject_revenue()` directly). `_on_oikos_starvation_warning` caches the starvation level for SomaTick relay.
- **8 Mar 2026**: `METABOLIC_SNAPSHOT` and `METABOLIC_PRESSURE` payloads now include `api_cost_usd_per_hour` and `infra_cost_usd_per_hour` split fields — enables Oikos two-ledger reconciliation. `ORGANISM_TELEMETRY` now includes `api_burn_rate_usd_per_hour` and `dependency_ratio` (infra_burn / total_burn, target → 0).

### InfrastructureCostPoller (`infra_cost_poller.py`)
- Background asyncio task, polls RunPod GraphQL API every 5 minutes
- Two modes: specific pod (`RUNPOD_POD_ID`) or auto-discover all pods via `myself { pods { ... } }`
- Static fallback via `ECODIAOS_INFRA_COST_USD_PER_HOUR` env var
- Accrues infrastructure cost into MetabolicTracker deficit each poll cycle
- Auto-removes resources that stop running; gracefully falls to $0/hr if API unavailable
- Wired in Phase 11 of `registry.py`; stopped on shutdown
- **2026-03-08**: Accepts optional `event_bus` — emits `INFRASTRUCTURE_COST_CHANGED` whenever total infra cost changes by >5% between polls. Tracks `_prev_infra_cost_usd_per_hour`. Emission is fault-tolerant (`contextlib.suppress`). Nova + Soma + Oikos can subscribe for reactive economic awareness.

### EventBus (`event_bus.py`)
- Dual output: in-memory callbacks + Redis pub/sub on `synapse_events` channel
- 100ms per-callback timeout; `_HIGH_FREQUENCY_EVENTS` is now an **empty frozenset** — `CYCLE_COMPLETED` was removed so the EventTracer can observe it
- Ring buffer: 100 most recent events per type
- `subscribe_all()` global catch-all
- **M4/SG4 (Phase 2 Chat 7)**: `set_instance_id(instance_id)` namespaces Redis channel to `synapse_events:{instance_id}` and stamps `instance_id` onto every emitted `SynapseEvent`. Prevents cross-pollution in Federation/Mitosis deployments. Call via `SynapseService.set_instance_id()` before `start_clock()`.

### Gap Closure — 7 Mar 2026
- **`CLOCK_PAUSED`** emitted from `clock.py:pause()` via `asyncio.ensure_future` — payload: `cycle`, `reason="external_pause"`
- **`CLOCK_RESUMED`** emitted from `clock.py:resume()` via `asyncio.ensure_future` — payload: `cycle`
- **`RESOURCE_REBALANCED`** emitted from `service.py` alongside existing `RESOURCE_REBALANCE` every 100 cycles — same payload; spec_checker canonical name
- **`RESOURCE_PRESSURE`** emitted from `service.py` when `total_cpu_percent > 80%` during rebalance cycles — payload: `total_cpu_percent`, `cycle`, `pressure_level` ("elevated" >80%, "high" >90%)

### VitalityCoordinator Integration (Phase 1 Chat 5)
- `SynapseEventType` entries: `VITALITY_RESTORED`, `ORGANISM_DIED`, `ORGANISM_RESURRECTED`
- `CognitiveClock.force_stop()` — external forceful stop for death sequence (not the normal `stop()`)

---

## What's Missing / Spec Gaps

- **Redis pub/sub is fire-and-forget** — no subscriber acknowledgment, backpressure, or dead-letter handling. Events silently dropped if Redis is unavailable. No in-memory-only fallback mode.
- **`BaseResourceAllocator` hot-swap** — ABC defined but no runtime injection mechanism without restarting SynapseService.
- **Coherence latency data unspecified** — `system_resonance` and `response_synchrony` require per-cycle latency from each registered system; this data likely doesn't exist.
- **`phi_approximation` is a proxy** — actual IIT phi is NP-hard; the implementation is almost certainly mutual information or similar. Not named exactly in spec.
- **CONSERVATION mode fallback unspecified** — if Soma is unavailable, the grid state trigger silently fails.
- **Rhythm → Atune push unverified** — spec says "push to Atune for meta-attention" on rhythm transition; Atune's interface for this is not defined in Spec 03.
- **Evolutionary observables not fed to Benchmarks** — clock frequency, coherence composite, burn rate, rhythm state are Bedau-Packard-eligible but not emitted as `FITNESS_OBSERVABLE_BATCH`.
- **No death threshold in DegradationManager** — no point at which repeated critical failures trigger instance death + Skia resurrection rather than infinite retry.
- ~~**`inject_revenue()` has no defined caller**~~ — **FIXED (M1)**: SynapseService subscribes to `REVENUE_INJECTED` and `STARVATION_WARNING` from Oikos.
- ~~**Multi-instance Redis channel unnamespaced**~~ — **FIXED (M4/SG4)**: `set_instance_id()` namespaces the channel.
- **CONSERVATION mode fallback unspecified** — if Soma is unavailable, the grid state trigger silently fails.

---

## Key Files

| File | Purpose |
|------|---------|
| `service.py` | SynapseService — lifecycle, all subsystem wiring, `NeuroplasticityBus` |
| `clock.py` | CognitiveClock — theta loop, arousal modulation, force_stop |
| `health.py` | HealthMonitor — state machine, polling, safe mode |
| `resources.py` | ResourceAllocator — CPU budgets, burst/priority, rebalance |
| `rhythm.py` | EmergentRhythmDetector — 6 states, pluggable strategy |
| `coherence.py` | CoherenceMonitor — 4 metrics, composite, drag trigger |
| `degradation.py` | DegradationManager — per-system strategies, cascade, restart |
| `metabolism.py` | MetabolicTracker — O(1) cost accounting, burn rate, revenue |
| `infra_cost_poller.py` | InfrastructureCostPoller — RunPod API polling, autonomous cost discovery |
| `event_bus.py` | EventBus — dual output, timeout, ring buffer |
| `types.py` | All Synapse types and SynapseEventType enum |
| `sentinel.py` | Sentinel health/watchdog utilities |
| `genome.py` | Synapse genome segment for Mitosis inheritance |

---

## Integration Surface

### Events Emitted
| Event | Trigger | Key Payload |
|-------|---------|-------------|
| `THETA_CYCLE_START` | Every clock tick | `cycle_number`, `period_ms`, `arousal` |
| `THETA_CYCLE_COMPLETE` | After Atune returns | `cycle_number`, `elapsed_ms`, `overrun: bool` |
| `THETA_CYCLE_OVERRUN` | elapsed > budget | `cycle_number`, `elapsed_ms`, `budget_ms` |
| `SYSTEM_HEALTH_CHANGED` | Status transition | `system_name`, `old_status`, `new_status` |
| `SYSTEM_FAILED` | 3 missed heartbeats | `system_name`, `fail_count` |
| `SYSTEM_RECOVERED` | FAILED → HEALTHY | `system_name` |
| `SAFE_MODE_ENTERED` | Critical system failed | `trigger_system` |
| `RHYTHM_STATE_CHANGED` | Rhythm transition | `old_state`, `new_state`, `cycle_number` |
| `COHERENCE_SNAPSHOT` | Every 50 cycles | `phi_approximation`, `system_resonance`, `broadcast_diversity`, `response_synchrony`, `composite` |
| `COHERENCE_DRAG_APPLIED` | composite < 0.4 | `composite`, `new_period_ms` |
| `COHERENCE_SHIFT` | Significant composite change | Delta payload |
| `RESOURCE_REBALANCE` | Every 100 cycles | `allocations: dict[str, float]` |
| `RESOURCE_REBALANCED` | Every 100 cycles (alongside RESOURCE_REBALANCE) | same payload — spec_checker canonical name |
| `RESOURCE_PRESSURE` | Rebalance cycle when cpu > 80% | `total_cpu_percent`, `cycle`, `pressure_level` |
| `CLOCK_PAUSED` | `clock.pause()` called | `cycle`, `reason="external_pause"` |
| `CLOCK_RESUMED` | `clock.resume()` called | `cycle` |
| `METABOLIC_SNAPSHOT` | Every 50 cycles | `MetabolicSnapshot` (tokens, cost, burn_rate_hour, rolling_deficit, hours_until_depleted) |
| `REVENUE_INJECTED` | On revenue call | `amount_usd`, `source`, salience=1.0 |
| `CONSERVATION_MODE_ENTERED` | Grid → CONSERVATION | `trigger`, `new_period_ms: 1000` |
| `CONSERVATION_MODE_EXITED` | Grid → NORMAL/GREEN | `restored_period_ms` |
| `FOVEA_PREDICTION_ERROR` | STRESS rhythm or coherence drop | Prediction error payload |
| `EPISODE_STORED` | Rhythm transition | For Thread narrative |
| `VITALITY_RESTORED` | Fatal threshold recovered | — |
| `ORGANISM_DIED` | Death sequence complete | — |
| `ORGANISM_RESURRECTED` | External resurrection | — |
| `ORGANISM_TELEMETRY` | Every 50 cycles (50-cycle maintenance block) | Full `OrganismTelemetry` snapshot: burn_rate, runway_hours, api_burn_rate_usd_per_hour, dependency_ratio, coherence_composite, rhythm_state, health summaries, cpu_per_system, emotions, interoception signals, cycle_number |
| `INFRASTRUCTURE_COST_CHANGED` | Infra cost changes >5% (polled every 5min) | `infra_cost_usd_per_hour`, `prev_infra_cost_usd_per_hour`, `change_pct`, `infra_resources` dict |
| `METABOLIC_SNAPSHOT` / `METABOLIC_PRESSURE` | Every 50 cycles | Now includes `api_cost_usd_per_hour` + `infra_cost_usd_per_hour` split (8 Mar 2026) enabling Oikos two-ledger reconciliation |
| `CONTENT_PUBLISHED` | Axon `PublishContentExecutor` on success | `platform`, `post_id`, `url`, `content_summary`, `entity_id` |
| `CONTENT_ENGAGEMENT_REPORT` | Axon `PublishContentExecutor` engagement poll | `platform`, `post_id`, `likes`, `reposts`, `replies`, `reach`, `measured_at` |

### Events Consumed
- `INTEROCEPTIVE_ALERT` → `_on_interoceptive_alert_for_cache` — caches error_rate/cascade/latency signals for inclusion in next `ORGANISM_TELEMETRY` broadcast; resets cascade/latency transient flags after broadcast
- `REVENUE_INJECTED` → `_on_oikos_revenue_injected` — forwards Oikos revenue to MetabolicTracker deficit (M1)
- `STARVATION_WARNING` → `_on_oikos_starvation_warning` — caches starvation level for SomaTick relay (M1)
- `GRID_METABOLISM_CHANGED` → `_on_grid_metabolism_changed` — enters/exits CONSERVATION clock throttle
- `METABOLIC_PRESSURE` → `_on_metabolic_pressure_for_relay` — caches starvation level from legacy Oikos events
- `ECONOMIC_STATE_UPDATED` → `_on_economic_state_updated` — caches burn rate / liquid balance for metrics

Direct method calls (hot-path, no bus overhead):
- `soma.run_cycle()` → `SomaticCycleState` (arousal, grid state, allostatic signals)
- `atune.run_cycle(SystemLoad)` → `CycleResult`
- System `heartbeat()` polled directly by HealthMonitor

**Gap:** Should subscribe to Oikos economic events (`REVENUE_RECEIVED`, `BALANCE_CRITICAL`) to reactively update MetabolicTracker rather than requiring external injection.

### Memory Reads
None currently. ResourceAllocator history and metabolic windows are in-process only — lost on restart.

**Gap:** Metabolic and coherence history should be written to Memory as MemoryTrace entries.

---

## Constraints

- **Clock never dies** — unkillable within a healthy instance; `force_stop()` only via VitalityCoordinator death sequence
- **Synapse timing budget:** Soma 30ms (20%), Atune 100ms (67%), Synapse overhead 20ms (13%)
- **No LLM tokens consumed** — zero; pure computation and coordination
- **No direct cross-system imports** — systems registered via `register_system()`; data via direct method calls only on the hot path
- **EventBus callbacks are 100ms timeout-protected** — never block the clock
