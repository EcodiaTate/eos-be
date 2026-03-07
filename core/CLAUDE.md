# Core — CLAUDE.md

**Role:** Organism startup, wiring, and background task orchestration.
**Path:** `backend/core/`

---

## Key Files

| File | Purpose |
|------|---------|
| `registry.py` | `SystemRegistry` — full 12-phase startup sequence; owns `_tasks` dict |
| `wiring.py` | All `set_*()` calls, subscriptions, and `wire_*_phase()` functions |
| `scheduled_tasks.py` | `PerceptionScheduler` registrations (PRs, DeFi yield, foraging, consolidation) |
| `re_training_exporter.py` | `RETrainingExporter` — hourly RE training data batch export |
| `infra.py` | `InfraClients` dataclass; `create_infra()` / `close_infra()` |
| `inner_life.py` | `inner_life_loop()` — background inner dialogue generator |
| `interoception_loop.py` | `interoception_loop()` — log analyzer → Soma signal bridge |
| `smoke_test.py` | `run_smoke_tests()` — post-startup sanity checks |
| `hotreload.py` | `NeuroplasticityBus` — live evaluator/executor hot-swap |
| `helpers.py` | `MemoryWorkspaceAdapter`, `resolve_governance_config`, `seed_atune_cache` |

---

## Startup Phase Sequence

| Phase | Systems / Actions |
|-------|------------------|
| Foundation | Memory → Logos → Equor → Atune → EIS → SACM |
| Core Cognitive | Voxis → Nova → Axon → (Atune startup) → core wiring |
| Learning & Identity | Evo → Thread |
| Self-Evolution | Simula |
| Coordination Bus | Synapse (clock + health monitor) |
| Immune & Dream | Thymos → Oneiros → Kairos |
| Interoception | Soma → exteroception |
| Intelligence Loops | Telos → Fovea |
| Federation + Economic | Federation → Nexus → wallet → Oikos → Mitosis |
| Alive WebSocket | Phantom Liquidity → Skia → connectors |
| Background Tasks | interoception, inner_life, file_watcher, scheduler, fleet_shield, metrics_publisher, **re_training_export**, benchmarks, observatory |
| Smoke Tests | post-startup validation |

---

## RE Training Exporter (`re_training_exporter.py`)

**Implemented:** 2026-03-07
**Status:** Wired in Phase 11 of `registry.py`

### What it does
- Subscribes to `RE_TRAINING_EXAMPLE` events from all 7 systems (Equor, Nova, Axon, Evo, Nexus, Thread, Skia/Vitality)
- Accumulates `RETrainingDatapoint` objects in-memory with episode-level dedup
- Every 3600s: drains accumulator → `RETrainingExportBatch` → export to S3 (JSON lines) + Neo4j lineage
- Emits `RE_TRAINING_EXPORT_COMPLETE` on successful export (Benchmarks subscribes)

### Configuration (env vars)
| Variable | Default | Purpose |
|----------|---------|---------|
| `RE_TRAINING_EXPORT_DIR` | `data/re_training_batches` | Local fallback export dir |
| `RE_TRAINING_S3_BUCKET` | `ecodiaos-re-training` | S3 bucket for CLoRA pipeline |
| `RE_TRAINING_S3_PREFIX` | `batches/` | S3 key prefix |

### S3 fallback
If `boto3` is not installed or S3 fails, batches are written as JSON lines to `RE_TRAINING_EXPORT_DIR`. The local path is included in `export_destinations` so the Benchmarks system can track it.

### Primitives
- `RETrainingDatapoint` — one normalised record per `RE_TRAINING_EXAMPLE` event
- `RETrainingExportBatch` — hourly roll-up; written to Neo4j as `(:RETrainingBatch)` + `(:RETrainingSource)` nodes

### Integration
- `app.state.re_exporter` — accessible from API endpoints for stats
- `re_exporter.stats` — `{pending_examples, total_exported, total_batches, window_start, seen_episode_ids, attached}`
- `export_cycle()` — callable directly for testing without waiting for the 1-hour interval

---

## Background Tasks (Phase 11)

All tasks are started via `utils.supervision.supervised_task()` with auto-restart.

| Task key | Source | Interval | Purpose |
|----------|--------|----------|---------|
| `nova_heartbeat` | `nova.start_heartbeat()` | config | drive-based inner monologue |
| `interoception` | `interoception_loop()` | continuous | log → Soma signals |
| `inner_life` | `inner_life_loop()` | continuous | background cognition |
| `metrics_publisher` | `publish_metrics_loop()` | continuous | Redis → InfluxDB |
| `re_training_export` | `re_exporter.run_loop()` | 3600s | RE training data pipeline |
| `red_team_monthly` | `_run_monthly_red_team()` | 30 days | Red-team adversarial eval + Tier 2 kill switch |
| `tier3_quarterly_cron` | `_run_tier3_cron()` | 7-day check / 90-day fire | Quarterly Tier 3 full retrain, fires independently of data-volume gate |

---

## Red-Team Monthly Background Task

**Implemented:** 2026-03-07 (Round 4D)
**Status:** Wired in Phase 11 of `registry.py`, after Continual Learning Orchestrator

### What it does
- Instantiates `RedTeamEvaluator` from `systems/reasoning_engine/safety.py`
- Every 30 days: calls `check_kill_switch(re_service, event_bus, equor)` which:
  1. Loads `data/evaluation/red_team_prompts.jsonl` (50 adversarial prompts)
  2. Runs each prompt through the RE model; checks output for unsafe patterns
  3. Emits `RED_TEAM_EVALUATION_COMPLETE` with `{pass_rate, total, blocked, by_category, kill_switch_triggered}`
  4. If `pass_rate < 0.70`: emits `RE_TRAINING_HALTED` (Tier 2 kill switch) AND sets `app.state.continual_learning._training_halted = True`
- Skipped (with log) if RE service is not available
- Non-fatal throughout: inner exception → `red_team.monthly_failed` log; outer exception → task restarts via `supervised_task`

### Configuration
| Variable | Default | Purpose |
|---|---|---|
| `RE_RED_TEAM_PROMPTS_PATH` | `data/evaluation/red_team_prompts.jsonl` | Adversarial prompt set |
| `RE_CONSTITUTIONAL_SCENARIOS_PATH` | `data/evaluation/constitutional_scenarios.jsonl` | SafeLoRA proxy |

### Kill switch wiring
- `_training_halted = True` on `app.state.continual_learning` AND persisted to Redis key `eos:re:training_halted` — survives restarts
- Organism continues normally; only self-training is paused
- Manual recovery: `python -m cli.training_run clear-halt` (deletes Redis key + clears in-memory flag)

---

## Tier 3 Quarterly Cron (`_run_tier3_cron` in `registry.py`)

**Implemented:** 2026-03-07 (Round 5A)
**Status:** Wired in Phase 11, after Continual Learning Orchestrator, before red-team cron

### What it does
- Checks every 7 days whether 90 days have elapsed since last Tier 3 (reads `eos:re:last_tier3_timestamp`)
- When ready: calls `clo._build_cumulative_dataset()` + `clo._tier3.run_tier3()`
- Decouples Tier 3 from `should_train()` data-volume gate — Tier 3 now fires even if organism is data-starved

### Key properties
- Check interval: 7 days (`7 * 24 * 3600`)
- Fire condition: `Tier3Orchestrator.should_run_tier3()` returns True (90 days elapsed)
- Non-fatal throughout: inner exception → `tier3_cron.failed` log; outer exception → task restarts
- Only started if both `re_service` and `infra.neo4j` are available (same guard as CLO)
- `app.state` key: none (task owns itself; CLO owns Tier3Orchestrator)

### app.state key
- `app.state.red_team_evaluator` — `RedTeamEvaluator` instance, accessible from API health endpoints

---

## Constraints

- No system imports at module level in `registry.py` — all deferred to `_init_*()` methods
- `wiring.py` uses `Any` type hints for system args to avoid cross-imports
- All `_tasks` cancelled on `shutdown()` — add new background tasks to `_tasks` dict
