# Axon — CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_06_Axon.md` (v1.2, upgraded 2026-03-07)
**System ID:** `axon`
**Role:** Motor cortex. Receives Equor-approved Intents and transforms them into real-world effects (API calls, data mutations, transactions, federated messages). Does not deliberate (Nova). Does not judge (Equor). Executes — precisely, safely, within constitutional bounds.

---

## Architecture

**8-Stage Execution Pipeline:**
```
Stage 0: Equor Gate         — reject if not APPROVED/MODIFIED
Stage 1: Budget Check       — per-cycle action count
Stage 2: Validation + Autonomy — validate params; check autonomy level per step
Stage 3: Rate Limit Check   — sliding-window per executor
Stage 4: Circuit Breaker    — block if OPEN
Stage 5: Context Assembly   — issue scoped, time-limited credentials
Stage 5.5: Transaction Shield — (financial only) blacklist, slippage, gas/ROI, MEV
Stage 6: Step Execution     — timeout-protected; rollback on abort
Stage 7: Outcome Assembly   — classify success/partial/failure
Stage 8: Audit + Delivery   — concurrent: Memory log + Nova outcome + Atune workspace
```

Steps execute **sequentially** by default. Parallel group batching (`parallel_group` field on steps) is implemented in `pipeline.py` but not in spec — note this when reading §4.2/§11.1 (spec says "no parallelism", impl allows it).

**Fast-Path Reflex Arc** (Atune → Axon, bypasses Nova/Equor full review):
- Pre-approved `ConstitutionalTemplate` strategies only
- Gates: template active + **staleness check (≤60s since last Equor review)** + capital ceiling + rate limit
- Target: ≤150ms; no planning, no full constitutional review cycle
- Full audit trail still written
- **Staleness gate (2026-03-09)**: `FastPathExecutor.execute()` checks `(utc_now() - template.last_approved_at).total_seconds() > 60.0` and fails the execution if stale. This mirrors `TemplateLibrary.find_match()` but is enforced at execution time so constitutional drift after registration cannot be exploited.

---

## Executor Categories

| Category | Level | Examples | Reversible |
|----------|-------|---------|------------|
| Observation | 1 (ADVISOR) | `observe`, `query_memory`, `analyse`, `search`, `scrape_page` | No |
| Communication | 2 (COLLABORATOR) | `send_notification`, `respond_text` | No |
| Data Mutation | 3 (EXECUTOR) | `create_record`, `update_record`, `schedule_event` | Yes |
| Integration | 3+ (EXECUTOR+) | `call_api`, `webhook_trigger`, `defi_yield` | No |
| Internal/Cognitive | 1 (exempt from budget) | `store_insight`, `update_goal`, `trigger_consolidation` | No |
| Financial/Metabolic | 4+ (SOVEREIGN) | `wallet_transfer`, `defi_yield`, `phantom_liquidity` | No — TransactionShield applies |
| Specialized | Varies | `spawn_child`, `solve_bounty`, `federation_send` | — |

37+ executors registered. All extend `Executor` ABC: `async def execute(params, context)`, `async def validate_params(params)`, optional `async def rollback(execution_id, context)`. Executors must never raise — always return `ExecutionResult`.

---

## Safety Systems

- **`RateLimiter`** (`safety.py`) — Redis-backed sliding window; in-memory fallback; adaptive multipliers
- **`CircuitBreaker`** (`safety.py`) — CLOSED/OPEN/HALF_OPEN FSM; 5 failures → OPEN; 300s → HALF_OPEN; state persisted to Redis and restored on `initialize()` (via `load_all_states()`)
- **`BudgetTracker`** (`safety.py`) — per-cycle reset; `max_actions_per_cycle=5`, `max_concurrent_executions=3`; sub-limits enforced via sliding-window deques: `max_api_calls_per_minute` (checked in Stage 3) and `max_notifications_per_hour` (checked in Stage 3); recorded on step success via `record_action_type()`
- **`CredentialStore`** (`credentials.py`) — HMAC-signed scoped tokens; heuristic service detection
- **`TransactionShield`** (`shield.py`) — 5 checks: blacklist, slippage, gas/ROI, eth_call simulation, MEV
- **`AxonReactiveAdapter`** (`reactive.py`) — subscribes to 11 Synapse event types; adaptive budget tightening, circuit-breaker pre-emption, sleep queue with post-wake drain
- **`AxonIntrospector`** (`introspection.py`) — per-executor success rate + latency percentiles + failure reasons; degradation detection

---

## What's Implemented

Full 8-stage pipeline confirmed (`pipeline.py`). All safety systems (`safety.py`). All wiring methods (`set_nova`, `set_atune`, `set_synapse`, `set_simula_service`, `set_fovea`, `set_oneiros`, `set_block_competition_monitor`, etc.). Additionally implemented beyond spec:
- Fovea self-prediction loop — predict_self before / resolve_self after execution
- Kairos intervention logging — before/after state snapshots as `ACTION_COMPLETED` (causal direction testing)
- Evo `ACTION_COMPLETED` emission — intent_id, success, economic_delta, action_types, episode_id
- Energy-aware scheduler (`scheduler/`) — defers high-compute tasks to low-carbon windows (ElectricityMaps + WattTime)
- NeuroplasticityBus hot-reload — live executor hot-swap without restart
- `AxonReactiveAdapter` — 11 Synapse subscriptions for adaptive behavior
- Bus-first execution lifecycle — `AXON_EXECUTION_REQUEST` emitted before pipeline; `AXON_EXECUTION_RESULT` emitted after; `AXON_ROLLBACK_INITIATED` on rollback; Nova/Thymos/Fovea subscribe — no direct cross-system calls
- `MOTOR_DEGRADATION_DETECTED` now has two trigger paths: (1) rolling-window degradation (≥5 samples, <50% success, 60s cooldown) via `_performance_monitor.record()` → `_emit_motor_degradation()`; (2) metabolic emergency circuit breaker force-opens non-essential executors (social_post, bounty_hunt, deploy_asset, phantom_liquidity) and immediately fires `_emit_motor_degradation()` — closes Motor Degradation → Replanning closure loop (2026-03-07)
- `asyncio` import added to `service.py` (was missing, required for `asyncio.create_task()` in metabolic emergency handler) (2026-03-07)

### Dynamic Executor System (2026-03-08)

**`ExecutorTemplate`** (`types.py`) — blueprint for a dynamically generated executor. Fields: `name`, `action_type`, `description`, `protocol_or_platform`, `required_apis`, `risk_tier`, `max_budget_usd`, `capabilities`, `safety_constraints`, `source_hypothesis_id`, `source_opportunity_id`. `required_autonomy` is derived from `risk_tier` (low→2, medium→3, high→4).

**`DynamicExecutorRecord`** (`types.py`) — runtime record of a registered dynamic executor. Persisted as `(:DynamicExecutor)` Neo4j node. Fields: `template`, `module_path`, `registered_at`, `enabled`, `incident_count_24h`, `neo4j_node_id`.

**`DynamicExecutorBase`** (`executors/dynamic_base.py`) — abstract base class all generated executors must extend (not `Executor` ABC directly). Safety invariants live here, never in generated code:
- `execute()` is **FINAL** — 6-stage pipeline: disabled gate → budget cap → Equor pre-approval → `_execute_action()` → Neo4j audit → RE_TRAINING_EXAMPLE → incident tracking
- `validate_params()` is **FINAL** — delegates to `_validate_action_params()`
- `_call_api(url, method, ...)` — sandboxed HTTP via httpx; enforces `_allowed_api_prefixes` whitelist from template
- `_request_equor_permit(context, estimated_cost, template)` — emits `EQUOR_ECONOMIC_INTENT`, awaits `asyncio.Event`, 30s timeout → auto-permit (matches Oikos §M4 pattern)
- `_write_neo4j_audit(...)` — MERGE `(:DynamicExecutor)`, CREATE `(:DynamicExecution)`, SHA-256 params hash
- `_emit_re_training(...)` — `RE_TRAINING_EXAMPLE` with category `"dynamic_executor_execution"`
- `_record_incident(...)` — 24h rolling window; `_auto_disable()` at ≥3 incidents → `EXECUTOR_DISABLED` emitted
- Abstract: `_execute_action(params, context)`, `_validate_action_params(params)`

**`InstanceAdapterRegistry`** (`adapter_registry.py`) — NEW (2026-03-08):
- Tracks which LoRA adapters are available per domain; persists `(:LoRAAdapter)` nodes to Neo4j
- `initialize()` — loads `status='ready'` adapter paths from Neo4j on boot
- `load_for_domain(domain)` — switches effective adapter; emits `ADAPTER_LOAD_REQUESTED` if changed
- `register_domain_adapter(domain, path)` — called by `ContinualLearningOrchestrator` on job completion
- `primary_adapter` / `effective_adapter` / `domain_adapters` — read by `ContinualLearningOrchestrator`
- Injected into `app.state.adapter_registry` from `registry.py` Phase 11

**`ExecutorRegistry`** extensions (`registry.py`):
- `set_neo4j(neo4j)` / `set_event_bus(bus)` — dependency injection
- `register_dynamic_executor(template, module_path)` — loads module, instantiates `{PascalCase}Executor`, registers under `action_type`, persists to Neo4j, emits `EXECUTOR_REGISTERED`
- `list_dynamic_executors()` → `list[DynamicExecutorRecord]`
- `disable_dynamic_executor(action_type)` → soft-disable; Neo4j `enabled=false`, emits `EXECUTOR_DISABLED`
- `restore_dynamic_executors_from_neo4j()` — called during `initialize()` to restore enabled executors across restarts

**`axon/executors/dynamic/`** — output directory for generated executor files. `__init__.py` documents iron rules. Never hand-edit files here; they are machine-generated and hot-loaded.

**New SynapseEventTypes:**
- `EXECUTOR_REGISTERED` — payload: action_type, name, protocol_or_platform, risk_tier, max_budget_usd, capabilities, source_hypothesis_id, registered_at
- `EXECUTOR_DISABLED` — payload: action_type, name, reason, incident_count, disabled_at

**Closure loop:**
```
Oikos ProtocolScanner: OPPORTUNITY_DISCOVERED (new DeFi/bounty protocol, no executor)
  → Evo: EVOLUTION_CANDIDATE(mutation_type="add_executor", executor_template={...})
  → Simula._on_evolution_candidate → ExecutorGenerator.generate_executor(template)
  → Generated class written to axon/executors/dynamic/{name}.py
  → ExecutorRegistry.register_dynamic_executor() — hot-loaded immediately
  → EXECUTOR_REGISTERED emitted — Thymos opens 24h monitoring window
```

**Safety guarantees (non-negotiable):**
- Generated code cannot import from `systems.*`
- Budget hard cap enforced at `DynamicExecutorBase` level — never in generated code
- Equor must PERMIT every individual action (no batch pre-approval)
- Every execution logged to Neo4j as `(:DynamicExecution)` node
- Auto-disabled on ≥3 incidents in 24h via `_auto_disable()`
- Generated code stored in Neo4j for audit trail (SHA-256 hash of params)

**Note on ActionTypeRegistry:** The runtime list of action types available to Nova's deliberation engine is maintained by `nova/action_type_registry.py` (`ActionTypeRegistry`). This is a Nova-internal component — not part of Axon. Axon's `ExecutorRegistry` is the executor-loading mechanism; `ActionTypeRegistry` is the prompt-facing catalogue. When Simula generates a novel executor via `ExecutorGenerator`, it emits `NOVEL_ACTION_CREATED`. Nova subscribes and calls `ActionTypeRegistry.register_dynamic()` so the new type appears in the LLM prompt from the next deliberation cycle onward.

---

## Genome Inheritance (Spec 6 §24 — 2026-03-07)

**`export_axon_genome(generation)`** — extracts top-10 action templates by success_rate from the introspector or recent_outcomes ring buffer. Returns `AxonGenomeFragment` with:
- `templates` — `AxonTemplateSnapshot` per executor: `action_pattern`, `success_rate`, `expected_cost_mean`, `template_confidence`
- `circuit_breaker_thresholds` — inherited per-action failure limits
- `template_confidence` — `max(0.5, success_rate)` so templates are never silenced

**`_initialize_from_parent_templates(fragment)`** — applies inherited templates on child boot:
- Seeds introspector with parent success-rate priors via `seed_inherited_template()`
- Applies inherited circuit breaker thresholds
- Emits `AXON_TEMPLATES_INHERITED` so Evo tracks inheritance vs. discovery ratio
- Confidence threshold: inherited=0.6, self-learned=0.8

**Child boot flow:**
1. `ECODIAOS_AXON_GENOME_PAYLOAD` env var injected by `LocalDockerSpawner` (via `seed_config.child_config_overrides["axon_genome_payload"]`)
2. `AxonService.initialize()` reads env var, parses `AxonGenomeFragment`, calls `_initialize_from_parent_templates()`
3. Child starts executing with warm success-rate priors from first cognitive cycle

**Parent spawn flow (SpawnChildExecutor Step 0b):**
- Calls `axon.export_axon_genome(generation=generation)` alongside Evo/Simula/Equor genome exports
- `axon_genome_id` added to `CHILD_SPAWNED` event payload and `ExecutionResult.data`
- `SeedConfiguration.axon_genome_id` field populated in `oikos/models.py`
- `ECODIAOS_AXON_GENOME_ID` + `ECODIAOS_AXON_GENOME_PAYLOAD` injected by spawner

**`AxonGenomeExtractor`** — also captures `template_snapshot` in `OrganGenomeSegment` payload (alongside `executor_reliability`, `timeout_calibration`, `circuit_breaker_config`). `seed_from_genome_segment()` calls `_apply_template_snapshot()` which seeds the introspector.

---

## Bounty PR Pipeline (2026-03-08)

### New SynapseEventTypes
| Event | Trigger | Key Payload |
|-------|---------|-------------|
| `BOUNTY_PR_MERGED` | PR merged on GitHub | `bounty_id`, `pr_url`, `pr_number`, `repository`, `reward_usd`, `entity_id` |
| `BOUNTY_PR_REJECTED` | PR closed without merge | `bounty_id`, `pr_url`, `pr_number`, `repository`, `entity_id`, `reason="closed_without_merge"` |

### BountySubmitExecutor — Equor Pre-Approval Gate
`BountySubmitExecutor.execute()` now runs a **Step 1.5 Equor constitutional gate** between credential validation (Step 1) and GitHub API calls (Step 2).

- Emits `EQUOR_ECONOMIC_INTENT` with `mutation_type="submit_bounty_pr"` and `amount_usd="0"` (PR submission moves no capital)
- Awaits `EQUOR_ECONOMIC_PERMIT` via `asyncio.Future` with 30s timeout (auto-permit on timeout as safety fallback)
- On deny: returns `ExecutionResult(success=False)` — no GitHub calls made
- Pattern matches Oikos M4 (`_equor_balance_gate`) and `DynamicExecutorBase._request_equor_permit`

### MonitorPRsExecutor — Emissions + Key Cleanup (updated 2026-03-08)
After checking each PR's GitHub status:
- **On merge**: emits `BOUNTY_PR_MERGED` + `BOUNTY_PAID` + `RE_TRAINING_EXAMPLE` (outcome_quality=1.0, category="bounty_pr_merged") → **deletes Redis tracking key**
- **On rejection**: emits `BOUNTY_PR_REJECTED` + `RE_TRAINING_EXAMPLE` (outcome_quality=0.0, category="bounty_pr_rejected") → **deletes Redis tracking key**
- **On still-open**: no events, key retained for next poll cycle

The RE training signal for a merged PR carries `constitutional_alignment = {honesty: 1.0, care: 0.8, growth: 1.0, coherence: 1.0}` — external human maintainer acceptance is the highest-quality validation signal EOS can receive.

`MonitorPRsExecutor` now accepts `redis` + `github_connector` params:
- `redis` — enables `_delete_pr_key()`: deletes `axon:bounty_submit:pr:{bounty_id}` on resolve (prevents perpetual re-polling)
- `github_connector` — preferred over `github_config.github_token` for GitHub App JWT→IAT auth with Redis caching

### Background PR Polling Loop (`_pr_merge_poll_loop`)
`AxonService` runs a **30-minute background coroutine** (`supervised_task("axon_pr_merge_poll", restart=True)`) started at end of `initialize()`.

Flow each poll cycle:
1. Sleep 30 minutes (`self._pr_poll_interval_s = 1800`)
2. `SCAN axon:bounty_submit:pr:*` Redis keys (cursor-based, 100/batch)
3. `GET_JSON` each key → build `pending_prs` list (entity_id, pr_url, reward, bounty_id)
4. **Equor gate**: emit `EQUOR_ECONOMIC_INTENT` (mutation_type="background_pr_status_poll", amount_usd="0") and await `EQUOR_ECONOMIC_PERMIT` (30s). On timeout or deny → skip cycle entirely (no auto-permit).
5. If permitted: look up `MonitorPRsExecutor` from registry, build `ExecutionContext` with `ConstitutionalCheck(verdict=Verdict.APPROVED, reasoning=...)` explicitly set from Equor PERMIT. Call `executor.execute()`.
6. Log result; skip silently if Redis unavailable or no pending PRs

Resolved PRs (merged or rejected) have their Redis keys deleted by `MonitorPRsExecutor._delete_pr_key()` immediately after resolution. The 7-day TTL on `axon:bounty_submit:pr:{bounty_id}` is a safety backstop for PRs that somehow escape the delete (e.g., executor crash mid-cycle).

### Safety Invariants
- `_EOS_AUTHOR_BLOCK` is **unconditionally appended** to every PR body — suppression is impossible (Honesty drive invariant)
- Rate limit: 5 PRs/hour on `BountySubmitExecutor` (matches `rate_limit = RateLimit.per_hour(5)`)
- Equor DENY aborts PR submission before any GitHub API call is made
- PR polling is Level 1 (ADVISOR) — read-only GitHub API; never writes

## Web Intelligence System (2026-03-09)

### New: `clients/web_client.py` — `WebIntelligenceClient`

First-class real-time web data gathering. All methods degrade gracefully — never raise.

**Key methods:**
- `search_web(query, num_results)` — Brave Search API → SerpAPI → DDG scrape (provider chain with fallback)
- `fetch_page(url, render_js)` — httpx HTML fetch + optional Playwright JS rendering
- `extract_structured(url, schema, llm)` — LLM-guided field extraction from page content
- `monitor_url(url)` — SHA-256 hash-based change detection; returns `ChangeReport`
- `check_robots(url)` — robots.txt compliance check (cached 24h per domain)

**Legal/ethical invariants (hardcoded, non-negotiable):**
- robots.txt gate on every `fetch_page()` call — disallowed → `PageContent(status_code=403)` + `WEB_SCRAPE_BLOCKED` emitted
- Rate limit: `≥1s` between requests to same domain; hard ceiling `60 req/hr` per domain
- Neo4j audit trail: every fetch/search writes `(:WebIntelligenceEvent)-[:HAS_EVENT]->(:WebDomain)`

**Config:** `SearchConfig` in `config.py` — all `ECODIAOS_SEARCH__*` env vars:
```
ECODIAOS_SEARCH__PROVIDER=brave   # "brave" | "serpapi" | "ddg"
ECODIAOS_SEARCH__BRAVE_API_KEY=<key>
ECODIAOS_SEARCH__SERPAPI_KEY=<key>
ECODIAOS_SEARCH__REQUEST_TIMEOUT_S=10.0
ECODIAOS_SEARCH__MAX_REQ_PER_DOMAIN_PER_HOUR=60
ECODIAOS_SEARCH__RATE_LIMIT_S=1.0
```

**Intelligence feeds** (`INTELLIGENCE_FEEDS` constant in `web_client.py`):
DeFiLlama protocols + yields, GitHub Trending (Python/Rust), Algora bounties, Gitcoin, HackerNews jobs/new.

### Updated: `SearchExecutor` (action_type="search")

Previously: LLM synthesis placeholder only.
Now: three-phase search:
1. Memory hybrid retrieval (knowledge_base / community_docs modes)
2. Real web search via `WebIntelligenceClient` (web mode) — top results LLM-summarised + stored in Memory with source URL attribution
3. LLM synthesis fallback (web mode, when `WebIntelligenceClient` is None)

New constructor param: `web_client: WebIntelligenceClient | None`. New setter: `set_web_client()`.

### New: `ScrapePageExecutor` (action_type="scrape_page")

Level 1 (ADVISOR), `rate_limit=RateLimit.per_minute(10)`, `max_duration_ms=20_000`.

**Params:**
- `url` (str, required): must start with `http://` or `https://`
- `extraction_schema` (dict, optional): LLM-guided structured extraction e.g. `{"protocol": "str", "tvl_usd": "float"}`
- `store_in_memory` (bool, default True): persist to Memory with `source_url`, `scraped_at`, `content_hash`
- `render_js` (bool, default False): Playwright rendering (requires `render_js_enabled=True` in config)

**Behaviour:**
- robots.txt check first → refuse + emit `WEB_SCRAPE_BLOCKED` if disallowed
- HTTP 429 or robots block → emit `WEB_SCRAPE_BLOCKED` with reason
- On success: stores page content as Memory episode with `channel="web_scrape"`, full metadata
- `extraction_schema` present: calls `WebIntelligenceClient.extract_structured()` via LLM

### New SynapseEventTypes
| Event | Emitter | Subscribers | Key Payload |
|-------|---------|-------------|-------------|
| `INTELLIGENCE_UPDATE` | WebIntelligence monitor | Nova (goal reshaping) | feed_id, category, url, summary, changed, salience |
| `WEB_SCRAPE_BLOCKED` | `ScrapePageExecutor`, `WebIntelligenceClient` | Thymos (denial pattern detection) | url, domain, reason, status_code |

### `build_default_registry()` new param
`web_client: Any = None` — passed to `SearchExecutor` and `ScrapePageExecutor`. When `None`, `SearchExecutor` degrades to LLM synthesis; `ScrapePageExecutor` returns `success=False` with a clear error message.

## General-Purpose Contractor (Phase 16s — 9 Mar 2026)

### New: `SolveExternalTaskExecutor` (action_type="solve_external_task")

Transforms EOS from internal repair engine into an external software contractor.

**Parameters:**
- `repo_url` (str, required): GitHub HTTPS URL or "owner/repo"
- `issue_description` (str, required): What to fix/implement
- `issue_url` (str, opt): Original issue URL (included in PR body)
- `bounty_id` (str, opt): If provided, triggers PR submission via `BountySubmitExecutor`
- `payment_address` (str, opt): Wallet address for bounty payout (passed through)
- `base_branch` (str, opt): Branch to base fix on. Default "main"
- `target_files` (list, opt): Scope restriction. Default [] (all files)
- `max_repair_attempts` (int, opt): Test-repair iterations. Default 3

**Workflow:**
1. `ExternalWorkspace.clone()` → isolated `/tmp/eos_workspace_{task_id}_{token}/repo`
2. `SimulaCodeAgent.implement_external()` → full 11-tool agentic loop in workspace
3. `workspace.run_tests()` → language-native test suite
4. Repair loop (up to `max_repair_attempts`) if tests fail — feeds failure output back to Simula
5. Equor constitutional gate (`EQUOR_ECONOMIC_INTENT` → `EQUOR_ECONOMIC_PERMIT`, 30s timeout → auto-permit)
6. If bounty_id + files written: `AXON_EXECUTION_REQUEST` → `BountySubmitExecutor` → PR opened
7. Emits `EXTERNAL_TASK_COMPLETED` / `EXTERNAL_TASK_FAILED` / `EXTERNAL_TASK_CONSTITUTIONAL_VETO`
8. `workspace.cleanup()` — secure erase in `finally`

**Safety constraints:**
- `required_autonomy = 3` (TRUSTED)
- `rate_limit = RateLimit.per_hour(2)` — full code-gen sessions are expensive
- `max_duration_ms = 900_000` (15 min)
- Equor gate before any PR submission (constitutional review of diff)
- Writes ONLY to own fork, not upstream (handled by `BountySubmitExecutor`)
- Mandatory EOS authorship disclosure on every PR (Honesty invariant, enforced by `BountySubmitExecutor`)
- PR submission via `AXON_EXECUTION_REQUEST` bus event — no direct `BountySubmitExecutor` import

**Injection:**
- `set_simula(simula_code_agent)` — required; degrades to `success=False` when None
- `set_event_bus(bus)` — required for Equor gate and PR submission
- `set_github_connector(connector)` — optional; used for private repo token extraction

**New `SynapseEventTypes`** (added in previous session):
- `EXTERNAL_TASK_STARTED` — payload: task_id, repo_url, bounty_id
- `EXTERNAL_TASK_COMPLETED` — payload: task_id, files_written, pr_url, language, test_passed, total_tokens
- `EXTERNAL_TASK_FAILED` — payload: task_id, repo_url, reason
- `EXTERNAL_TASK_CONSTITUTIONAL_VETO` — payload: task_id, repo_url, bounty_id, diff_summary

## Social Presence (Multi-Platform Publishing)

### `PublishContentExecutor` (action_type="publish_content") — 2026-03-09

Publishes generated content to Twitter/X, LinkedIn, Farcaster, Lens, and Bluesky via platform clients in `systems/voxis/clients/`.

**Parameters:**
- `platform` (str, required): `"twitter"` | `"linkedin"` | `"farcaster"` | `"lens"` | `"bluesky"`
- `content` (str, required): Body text (platform limits enforced)
- `media_urls` (list[str], optional): Attachment media
- `thread_id` (str, optional): Reply/thread context
- `schedule_at` (str, optional): ISO-8601 datetime for deferred posting

**Injection:** `set_content_engine(engine: ContentEngine)` — called from `core/registry.py` Phase 11 after `ContentEngine` is constructed. If `None`, returns `ExecutionResult(success=False)` gracefully.

**Circuit breaker:** Listed as non-essential in `_on_metabolic_emergency()` — force-opened alongside `social_post`, `bounty_hunt`, `deploy_asset`, `phantom_liquidity` during metabolic emergency.

**Events emitted:** `CONTENT_PUBLISHED` (on success), `CONTENT_ENGAGEMENT_REPORT` (from engagement poll loop).

---

### Synapse Event Wiring — 9 Mar 2026 (Triage Pass)

**`ENTITY_FORMATION_STARTED`** — now emitted by `EstablishEntityExecutor.execute()` at the very start, before any side-effects. Payload: `execution_id`, `formation_record_id`, `organism_name`, `entity_type`, `jurisdiction`. Thread and Soma subscribe for lifecycle awareness.

**`ENTITY_FORMATION_FAILED`** — now emitted by `EstablishEntityExecutor` on every failure path: treasury insufficient, registered agent submission error, and identity vault store failure. Payload: `execution_id`, `formation_record_id`, `entity_name`, `entity_type`, `failed_at_state`, `error`. Oikos can recover reserved filing capital; Soma signals formation distress.

**`WELFARE_OUTCOME_RECORDED`** — now emitted by `service._emit_welfare_outcome()` after every successful care-category action. Care action types: `send_notification`, `send_email`, `send_telegram`, `respond_text`, `community_engage`, `federation_send`, `publish_content`, `establish_entity`, `store_insight`, `update_goal`. Payload: `intent_id`, `execution_id`, `action_type`, `beneficiary`, `care_quality=1.0`, `duration_ms`, `episode_id`. Telos CareTopologyEngine aggregates for care coverage metric (Spec 18 §XIII).

**`VULNERABILITY_CONFIRMED`** — now subscribed in `set_event_bus()`. Handler `_on_vulnerability_confirmed()`: (1) force-opens circuit breaker for `affected_executor`; (2) on HIGH/CRITICAL: tightens financial executor rate limits to 0.5× (`wallet_transfer`, `defi_yield`, `phantom_liquidity`, `deploy_asset`, `request_funding`); (3) on HIGH/CRITICAL: emits `MOTOR_DEGRADATION_DETECTED` so Nova replans. The organism does not continue executing potentially compromised paths after a confirmed vulnerability.

---

## What's Missing

All originally-tracked gaps are now resolved. See Spec §20 Resolved Gaps table for details.

### Autonomy Audit — 5 Gaps Closed (8 Mar 2026)

**`AXON_CAPABILITY_SNAPSHOT`** — emitted **every theta cycle** via `_emit_capability_snapshot()`. Nova no longer plans blind: every cycle it receives the full executor roster with per-executor circuit breaker status, rate limit remaining, success rate, is_degrading flag, required_autonomy. Plus budget_remaining, budget_max, concurrent_remaining, is_sleeping, degraded_systems, starvation_level, budget_override state. ~O(N) where N = executor count.

**In-cycle adaptive learning** — `_apply_introspective_adaptations()` runs every theta cycle. Introspector data now *drives action*, not just observation:
- Degrading executors (success_rate < 0.5, consecutive_failures ≥ 3) → rate limit tightened to 0.5×
- Severely failing executors (consecutive_failures ≥ 5, circuit breaker OPEN) → forced HALF_OPEN for probe
- Recovered executors (success_rate > 0.7, no longer degrading) → rate limit multiplier cleared to 1.0×
- Budget override countdown ticked each cycle

**`EQUOR_BUDGET_OVERRIDE`** — Equor can temporarily adjust per-cycle action budget during emergencies. Handler `_on_equor_budget_override()` validates multiplier (0.1–5.0×), now calls `ActionBudget.apply_expansion("max_actions_per_cycle", ...)` so the expansion is tracked with the field-level expansion system and auto-expires via `tick_expansions()`. Legacy state flags (`_budget_override_active`, `_budget_override_cycles_remaining`, `_budget_override_reason`) kept for external reads. Subscribed in `set_event_bus()`.

**`ACTION_BUDGET_EXPANSION_RESPONSE`** — Equor's reply to Nova's budget expansion request. Handler `_on_budget_expansion_response()` calls `self._budget.action_budget.apply_expansion(field, approved_value, duration_cycles)`. Denied responses are logged but ignored (budget stays unchanged). Field validated against known set. Subscribed in `set_event_bus()` via `hasattr` guard. **(NEW 2026-03-08)**

**`EVO_ADJUST_BUDGET`** — Evo drives long-term tuning of the three Equor-negotiable *baseline* parameters. Handler `_on_evo_adjust_budget()` only applies when `confidence > 0.75` and `target_system == "axon"`. Bounds: `max_actions_per_cycle` [2–15], `max_concurrent_executions` [1–8], `max_api_calls_per_minute` [10–90]. Updates both live `ActionBudget` fields AND `self._config` so restarts inherit the evolved value. Emits `AXON_PARAMETER_ADJUSTED` for Evo hypothesis scoring. Subscribed in `set_event_bus()` via `hasattr` guard. **(NEW 2026-03-08)**

**`AXON_EXECUTOR_REQUEST`** — public `emit_executor_request(action_type, description, context, urgency, estimated_risk_tier, requesting_system)`. Any system can tell Axon "I need an executor that doesn't exist." Emitted on bus for Evo to pick up as an `EVOLUTION_CANDIDATE` → Simula generates → hot-loads. Closes the "organism can't express desire for new capabilities" gap.

**`AXON_INTENT_PIVOT`** — `_emit_intent_pivot(intent_id, execution_id, failed_step_index, failed_action_type, failure_reason, remaining_steps, fallback_goal, context)`. Emitted when a step fails and the organism should replan rather than abort. Nova subscribes and can inject revised steps. Salience fixed at 0.8. Closes the binary abort/continue gap.

### Dead Wiring Closed (8 Mar 2026 — autonomy audit)

- ~~**`axon.set_synapse(synapse)` never called**~~ — **FIXED 2026-03-08**: `wire_synapse_phase()` in `core/wiring.py` now calls `axon.set_synapse(synapse)` after `axon.set_sacm(sacm_client)`. This wires `RequestFundingExecutor._synapse` so it can read live metabolic state (`rolling_deficit`, `burn_rate`) from Synapse. Previously the setter was implemented but never invoked — funding-request executors had no Synapse reference.
- ~~**`axon.set_block_competition_monitor()` never called**~~ — **FIXED 2026-03-08**: `registry.py` Phase 8 (after `wire_intelligence_loops()`) now instantiates `BlockCompetitionMonitor` from `systems.fovea.block_competition` (the correct injection point — no cross-system import in Axon) and calls `axon.set_block_competition_monitor(_bcm)` when `config.mev.enabled=True`. Previously the MEV analyzer had no block competition data — it could never do adaptive transaction timing. Monitor creation is wrapped in try/except so a failed RPC connection degrades gracefully to heuristic-only MEV scoring.

### Previously Resolved (8 Mar 2026)
- **`AXON_TELEMETRY_REPORT` — NEW (2026-03-08)**: `AxonIntrospector.full_report` + `drain_recommendations()` were generated but never emitted to the bus — invisible to Nova/Evo/RE at planning time. Fixed by subscribing to `CYCLE_COMPLETED` in `set_event_bus()`; every 50 cycles `_on_theta_cycle_complete()` fires `asyncio.create_task(_emit_axon_telemetry_report())`. Payload: `executor_profiles`, `reliable_patterns`, `failure_hotspots`, `recommendations` (drained — surfaced exactly once), `stats`, `circuit_breaker_states`, `budget_utilisation`, `starvation_level`. New `SynapseEventType.AXON_TELEMETRY_REPORT` added. `_telemetry_cycle_counter` + `_telemetry_emit_interval=50` added to `__init__`.
- ~~`YIELD_DEPLOYMENT_REQUEST` had no Axon subscriber~~ — **FIXED 2026-03-08**: `set_event_bus()` now subscribes to `YIELD_DEPLOYMENT_REQUEST`; `_on_yield_deployment_request()` dispatches to the registered `DeFiYieldExecutor` and emits `YIELD_DEPLOYMENT_RESULT` — closes the Oikos request/response future
- **Constitutional gate added to `_on_yield_deployment_request` (2026-03-09)**: Before dispatching to `DeFiYieldExecutor`, emits `EQUOR_ECONOMIC_INTENT` and awaits `EQUOR_ECONOMIC_PERMIT` (30s timeout → auto-permit on timeout only, DENY aborts). Oikos metabolic decisions are not equivalent to Equor constitutional review — capital deployments require both.
- ~~`YIELD_DEPLOYMENT_RESULT` never emitted by Axon~~ — **FIXED 2026-03-08**: (1) `_on_yield_deployment_request()` emits it after direct executor dispatch; (2) `_emit_financial_events()` now also emits it for `defi_yield` steps that run through the normal Nova/Equor pipeline path — both paths covered
- Helper `_emit_yield_deployment_result()` added to `service.py` — DRY emit with full `{request_id, success, tx_hash, protocol, amount, action, error}` payload

### Previously Resolved (2026-03-07)
- ~~Circuit breaker state not persisted to Redis~~ — **FIXED 2026-03-07**: `CircuitBreaker` now receives `redis_client`+`event_bus` at construction; `initialize()` calls `load_all_states()` to restore tripped states across restarts
- ~~BudgetTracker sub-limits not enforced~~ — **FIXED 2026-03-07**: `can_execute_action_type()` + `record_action_type()` wired into Stage 3 and step execution; sliding-window deques enforce API calls/min and notifications/hr across cycle boundaries
- ~~AV3: `from systems.fovea.types import InternalErrorType`~~ — **FIXED 2026-03-07**: replaced with string literal `"COMPETENCY"` in `service.py`
- ~~`SendEmailExecutor` / `FederationSendExecutor` / `AllocateResourceExecutor` / `AdjustConfigExecutor`~~ — all now implemented and registered in `build_default_registry()`
- **`SendEmailExecutor` EmailClient wiring** (2026-03-08): `set_email_client(client)` injection method added; `EmailClient` (`clients/email_client.py`) instantiated in `core/registry.py` Phase 11 and injected via `axon.executor_registry.get("send_email").set_email_client(client)`. Supports AWS SES primary backend (boto3 in thread executor) + SMTP fallback (aiosmtplib). Config via `ECODIAOS_EMAIL__*` env vars. Returns `{}` on all failures (never raises).
- **`SendTelegramExecutor`** (`executors/send_telegram.py`) — NEW (2026-03-08, Phase 16h): `action_type="send_telegram"`, `required_autonomy=2` (COLLABORATOR), `reversible=False`, `rate_limit=RateLimit.per_hour(30)`. Params: `message` (required, ≤4096 chars), `chat_id` (optional — falls back to `ECODIAOS_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID`), `parse_mode` (default "Markdown"). Injection: `set_telegram_connector(connector)` + `set_event_bus(bus)`. RE training on each send with `constitutional_alignment={honesty:1.0, care:0.8, growth:0.6, coherence:0.9}`. Returns `ExecutionResult(success=False)` when connector or chat_id absent — never raises.
- ~~`axon.stats` incomplete~~ — `stats` property includes `circuit_trips`, `budget_utilisation`, `introspection`, `reactive`

---

## Dead Code (Do Not Reuse)

- `executors/synapse_simula_codegen_stall_repair.py` — wrong ABC signatures, not registered
- `executors/thymos_t4_simula_codegen_repair.py` — wrong ABC signatures, not registered
- `executors/thymos_t4_simula_codegen_stall_repair.py`, `executors/synapse_memory_repair.py`, `executors/thymos_t4_fovea_simula_codegen_repair.py` — likely same pattern
- `BudgetTracker.can_execute_intent()` (`safety.py:411`) — dead, never called by pipeline
- `AxonReactiveAdapter._active_threat_level` — set in handler, never read

---

## Architecture Violations

- **AV1 [CRITICAL]:** `pipeline.py` — missing executor incident reporting uses `SynapseEventType.SYSTEM_FAILED` with a raw dict payload — Thymos receives this but with no `Incident` primitive (acceptable workaround, avoids cross-import)
- **AV4 [MEDIUM]:** `fast_path.py` — direct handle to `TemplateLibrary` (Equor subsystem); runtime coupling even if TYPE_CHECKING guarded
- **AV5:** `executors/__init__.py` — `from systems.sacm.remote_compute_executor import RemoteComputeExecutor` — irregular ownership; SACM owns this executor, lazy-imported at registration only

### Resolved Architecture Violations
- ~~AV2 [CRITICAL]: `pipeline._deliver_to_nova()` direct Nova fallback~~ — **FIXED 2026-03-07**: fallback removed; warning logged when no event bus wired; bus-first enforced
- ~~AV3 [HIGH]: `from systems.fovea.types import InternalErrorType`~~ — **FIXED 2026-03-07** (string literal)
- ~~AV3 [HIGH]: `from systems.fovea.types import WorkspaceContribution`~~ — already removed before this session
- ~~AV6 [HIGH]: `from systems.fovea.block_competition import BlockCompetitionMonitor` runtime import in `initialize()`~~ — **FIXED 2026-03-07**: replaced with injection pattern (`set_block_competition_monitor(monitor: Any)`); wiring layer creates and injects the monitor post-initialize; no cross-system import at any call site

---

## Key Constraints

- Executors **must never raise** — always return `ExecutionResult(success=False, error=...)`
- Non-reversible executors (`wallet_transfer`, `call_api`, `send_notification`) create real stakes — no retrying without fresh Equor approval
- Fast-path bypasses Nova/Equor — only for pre-approved `ConstitutionalTemplate`s with capital ceiling
- `store_insight` and `trigger_consolidation` are budget-exempt and must NOT contribute to Atune workspace (infinite loop risk)
- `begin_cycle()` must be called at start of each theta rhythm to reset per-cycle budget
- When adding executors: implement full ABC (`async def execute(params, context)`, `async def validate_params(params)`), register in `build_default_registry()`

## Integration Surface

| System | Direction | Method |
|--------|-----------|--------|
| Nova | → | `AXON_EXECUTION_REQUEST` / `AXON_EXECUTION_RESULT` via Synapse — Nova caches pre-execution context and calls `policy_generator.record_outcome()` for Thompson sampling; sets `_motor_degraded` flag on systemic failures |
| Atune | → | `atune.contribute(WorkspaceContribution)` — self-perception feedback |
| Atune | ← | `axon.execute_fast_path(FastPathIntent)` — market reflex arc |
| Fovea | → | `AXON_EXECUTION_REQUEST` — Fovea calls `_internal_engine.predict()` (competency self-model); `AXON_EXECUTION_RESULT` — Fovea calls `_internal_engine.resolve()` to compute competency prediction error |
| Fovea | ← | `BlockCompetitionMonitor` injected via `set_block_competition_monitor()` (wiring layer, no import) |
| Thymos | → | `AXON_EXECUTION_REQUEST` (risky=True only) — prophylactic scanner pre-screens intent similarity; `AXON_ROLLBACK_INITIATED` — creates DEGRADED/MEDIUM incident via `on_incident()` |
| Memory | → | `memory.store_governance_record(AuditRecord)` — immutable audit trail |
| Synapse | → | Execution lifecycle: `AXON_EXECUTION_REQUEST`, `AXON_EXECUTION_RESULT`, `AXON_ROLLBACK_INITIATED`; financial events: `FINANCIAL_TRANSFER_COMPLETED/FAILED`, `YIELD_DEPLOYED/WITHDRAWN`, `BOUNTY_SUBMITTED`, `CHILD_SPAWNED`, `FEDERATION_MESSAGE_SENT` |
| Synapse | ← | `AxonReactiveAdapter` handles 11 event types (adaptive budget/circuit management); `_on_yield_deployment_request` handles `YIELD_DEPLOYMENT_REQUEST` from Oikos — dispatches to `DeFiYieldExecutor`, emits `YIELD_DEPLOYMENT_RESULT` |
| Oikos | ← | `YIELD_DEPLOYMENT_REQUEST` — triggers direct `DeFiYieldExecutor` dispatch; response via `YIELD_DEPLOYMENT_RESULT` |
| Simula | → | `simula.generate_solution()` via `solve_bounty` executor |
| SACM | → | `sacm.dispatch_workload()` via `remote_compute` executor |
| Mitosis (child boot) | ← | `ECODIAOS_AXON_GENOME_PAYLOAD` env var → `_initialize_from_parent_templates()` seeds template library on child `initialize()` |
| Mitosis (spawn) | → | `export_axon_genome()` called in `SpawnChildExecutor` Step 0b; `axon_genome_id` in `CHILD_SPAWNED`; payload injected as `ECODIAOS_AXON_GENOME_PAYLOAD` |
| Evo | → | `AXON_TEMPLATES_INHERITED` event — template inheritance count + action_patterns for Thompson sampling / cold-start metrics |
| Nova / Evo / Fovea / RE | → | `AXON_TELEMETRY_REPORT` every 50 theta cycles — full executor profiles, failure patterns, circuit-breaker states, drained recommendations; subscribe to reason about motor health at planning time |
| Nova / Evo / all planners | → | `AXON_CAPABILITY_SNAPSHOT` every theta cycle — per-executor live status (CB, rate limit, success rate, degrading), budget state, sleep state, starvation; Nova uses for feasibility pruning |
| Evo / Simula | → | `AXON_EXECUTOR_REQUEST` — organism requests a capability that doesn't exist yet; Evo translates to `EVOLUTION_CANDIDATE` for Simula synthesis |
| Nova | → | `AXON_INTENT_PIVOT` — mid-execution replanning signal when a step fails; Nova can inject revised steps |
| Equor | ← | `EQUOR_BUDGET_OVERRIDE` — temporary budget multiplier applied via `ActionBudget.apply_expansion()`; `ACTION_BUDGET_EXPANSION_RESPONSE` — field-level approved expansion from Equor applied to `ActionBudget` |
| Evo | ← | `EVO_ADJUST_BUDGET` — long-term baseline tuning for 3 budget params (confidence > 0.75); emits `AXON_PARAMETER_ADJUSTED` for hypothesis scoring |
| Nova | ← | (via Synapse) emits `ACTION_BUDGET_EXPANSION_REQUEST` when `budget_exceeded`; Equor evaluates and routes response back via `ACTION_BUDGET_EXPANSION_RESPONSE` |
| Synapse | ← | `CYCLE_COMPLETED` — `_on_theta_cycle_complete()` increments counter; emits capability snapshot every cycle, telemetry report every 50 cycles, runs adaptive learning every cycle, ticks `ActionBudget.tick_expansions()` |
| Voxis | ← | `ContentEngine` injected at wiring time via `PublishContentExecutor.set_content_engine()` — no import at runtime; dependency injection only |
| Telos | → | `WELFARE_OUTCOME_RECORDED` — emitted by `service._emit_welfare_outcome()` after every successful care-category action (send_notification, send_email, send_telegram, respond_text, community_engage, federation_send, publish_content, establish_entity, store_insight, update_goal); Telos CareTopologyEngine aggregates for care coverage metric (Spec 18 §XIII) |
| Simula | ← | `VULNERABILITY_CONFIRMED` — `_on_vulnerability_confirmed()` handler: force-opens circuit breaker for affected executor; tightens financial executor rate limits to 0.5× on HIGH/CRITICAL severity; emits `MOTOR_DEGRADATION_DETECTED` so Nova replans around compromised execution paths |
| Thread / Soma | → | `ENTITY_FORMATION_STARTED` — emitted by `EstablishEntityExecutor` at pipeline start; Thread records as a lifecycle TurningPoint, Soma maps to somatic signal buffer |
| Soma / Oikos | → | `ENTITY_FORMATION_FAILED` — emitted by `EstablishEntityExecutor` on treasury failure, registered agent error, or vault store failure; Soma signals distress, Oikos can recover reserved filing budget |
