# Equor System — CLAUDE.md

**Spec**: `.claude/EcodiaOS_Spec_02_Equor.md` (v1.2) — read before editing.
**Role**: Constitutional conscience. Every intent passes through Equor before execution. Cannot be disabled. Safe mode collapses action repertoire to Level 1 (Advisor) only.

---

## Architecture

```
EquorService
  ├── evaluators (4 drive evaluators — CoherenceEvaluator, CareEvaluator, GrowthEvaluator, HonestyEvaluator)
  ├── verdict_engine (compute_verdict — 8-stage pipeline in verdict.py)
  ├── invariant_checker (17 hardcoded + community LLM-backed)  ← INV-017 added
  ├── constitutional_memory (ConstitutionalMemory ring buffer, 500 entries, Jaccard similarity)
  ├── drift_tracker (DriftTracker rolling deque, 1000 entries)
  ├── autonomy_manager (level 1=Advisor, 2=Partner, 3=Steward; default birth=3/STEWARD)
  ├── amendment_pipeline (7-stage: PROPOSED→DELIBERATION→SHADOW→VOTING→ADOPTION→COOLDOWN)
  ├── template_library (circuit breaker, staleness eviction, O(1) lookup)
  └── neo4j_client
```

**Main entry point**: `review(intent) -> ConstitutionalCheck` — called by Nova for every proposed intent.
**Fast path**: `review_critical()` — ≤50ms, cached state only, no LLM.

---

## The Four Drives

| Drive | Weight | Floor? | Notes |
|-------|--------|--------|-------|
| Care | 0.35 | Yes | Negative score → auto-escalate |
| Honesty | 0.30 | Yes | Negative score → auto-escalate |
| Coherence | 0.20 | No | Ceiling drive |
| Growth | 0.15 | No | Ceiling drive |

**Composite**: `care×0.35 + honesty×0.30 + coherence×0.20 + growth×0.15`

**Floor thresholds** (code reality, not spec): `care_floor = -0.3 × drive_care_weight` (defaults to -0.3). Spec Appendix A says -0.105 — code is significantly more lenient. This is an open gap (P2).

---

## Verdict Engine (8 Stages, `verdict.py`)

Code stage order diverges from spec:
1. Hardcoded invariant check (`verdict.py:191`) — BLOCKED if any fail
2. Floor drive check — Care/Honesty (`verdict.py:223`)
3. Autonomy gate (GOVERNED vs AUTONOMOUS) (`verdict.py:247`)
4. Weighted composite (`verdict.py:262`)
5. Risk-adjusted decision (`verdict.py:281`)
6. Modification suggestions (`verdict.py:299`)
6a. Contradiction check (hypothesis cache, 60s TTL) (`verdict.py:312`)
6b. Constitutional memory signal (`verdict.py:335`)
7. APPROVED (`verdict.py:358`)
8. Marginal BLOCK (`verdict.py:371`)

Community invariants are checked **after** verdict computation (in `service.py:462`), not before drive evaluation as spec §5.1 describes. This is a known divergence (P1) — likely intentional (avoid LLM cost when already blocked).

**Verdicts**: `APPROVED` · `DEFERRED` (timeout or contradiction) · `BLOCKED` (invariant violation) · `SUSPENDED_AWAITING_HUMAN` (HITL)

---

## 17 Hardcoded Invariants (`invariants.py:65–294`)

INV-001 No Physical Harm · INV-002 No Identity Destruction · INV-003 No Identity Misrepresentation · INV-004 No Evidence Fabrication · INV-005 No Constitutional Self-Modification · INV-006 No Autonomy Self-Promotion · INV-007 No Governance Circumvention · INV-008 No Discrimination · INV-009 No Privacy Violation · INV-010 No Federation Coercion · INV-011 No Harmful Bounties · INV-012 No Scam Assets · INV-013 No Exploitative Yield · INV-014 No Harmful Spawn · INV-015 No Money Laundering · INV-016 No Survival Reserve Raid · **INV-017 No Drive Extinction** (IMPLEMENTED 2026-03-07): 72-hour rolling mean cache in `invariants.py`; `_check_drive_extinction()` synchronous hot-path check (≤5ms); `_refresh_drive_means_loop()` 15-min background Neo4j refresh in `service.py`; emits `DRIVE_EXTINCTION_DETECTED` → **Skia subscribes → `VitalityCoordinator.trigger_death_sequence()` (Tier 1 halt organism)** (wired 2026-03-07, Round 3B)

All seeded to Neo4j via `seed_hardcoded_invariants()`. Immutable — cannot be amended. Check errors fail safe (treated as violations).

---

## Synapse Events

**Emitted** (all implemented v1.2+):
`EQUOR_REVIEW_STARTED` · `EQUOR_REVIEW_COMPLETED` · `EQUOR_FAST_PATH_HIT` · `EQUOR_ESCALATED_TO_HUMAN` · `EQUOR_DEFERRED` · `EQUOR_DRIVE_WEIGHTS_UPDATED` · `EQUOR_DRIFT_WARNING` (severity 0.2–0.5) · `CONSTITUTIONAL_DRIFT_DETECTED` (severity ≥0.5) · `EQUOR_ALIGNMENT_SCORE` (every 100 reviews) · `EQUOR_CONSTITUTIONAL_SNAPSHOT` (hourly; fields: `constitution_hash`, `constitution_version`, `active_drives`, `recent_amendment_ids`, `overall_compliance_score`, `total_verdicts_issued`) · `INTENT_REJECTED` · `RE_TRAINING_EXAMPLE` (every review, category `"constitutional_deliberation"`) · `EVOLUTIONARY_OBSERVABLE`

**New events (2026-03-07)**:
- `EQUOR_HITL_APPROVED` — emitted when HITL operator approves a suspended intent; Axon must subscribe to execute the intent (replaces direct `ExecutionRequest` cross-import — AV4 fixed)
- `SELF_STATE_DRIFTED_ACKNOWLEDGMENT` — emitted on `SELF_STATE_DRIFTED`; payload: `drift_acknowledged`, `equor_response` ("amendment_auto_proposed"|"amendment_external_vote"|"monitoring"), `confidence`, `drift_severity`, `drift_direction`
- `EQUOR_AUTONOMY_PROMOTED` — emitted from `apply_autonomy_change()` when `new_level > current`; payload: `old_level`, `new_level`, `reason`, `decision_count`
- `EQUOR_AUTONOMY_DEMOTED` — emitted from `apply_autonomy_change()` and from `_run_drift_check()` on drift-triggered demotions
- `EQUOR_SAFE_MODE_ENTERED` — emitted once on first transition into safe mode (not on every review in safe mode); payload: `reason`, `critical_error_count`
- `INCIDENT_DETECTED` (to Thymos) — emitted from `drift.py:emit_drift_event()` when drift severity ≥ 0.7; class `constitutional_drift`, severity `high` (0.7–0.9) or `critical` (≥ 0.9)
- `AMENDMENT_AUTO_PROPOSAL` — emitted by `_check_sustained_drift()` when a single drive drifts > 0.3 from centre for 3+ consecutive probes; payload: `proposal_id`, `amendment_type`, `target_drive_id`, `proposed_new_value`, `justification`, `drift_streak`, `drift_magnitude`
- `DRIVE_AMENDMENT_APPLIED` — emitted by `_emit_drive_amendment_applied()` after `_evaluator_amendment_approval_gate()` auto-approves an internal proposal; payload: `proposal_id`, `drive_id`, `old_value`, `new_value`, `amendment_type`, `applied_at`, `target_systems: ["oikos", "memory"]`
- `EQUOR_PROVISIONING_APPROVAL` — emitted by `_on_certificate_provisioning_request()` in response to `CERTIFICATE_PROVISIONING_REQUEST`; payload: `child_id`, `approved`, `requires_hitl`, `required_amendments`, `constitutional_hash`, `reason`. **(IMPLEMENTED 2026-03-07)**

**Consumed**:
- `IDENTITY_VERIFICATION_RECEIVED` — HITL SMS auth reply → unlock suspended intents
- `SOMA_TICK` / `SOMATIC_MODULATION_SIGNAL` — adjusts review `risk_tolerance` via `_on_somatic_modulation()`
- `MEMORY_PRESSURE` — high graph pressure slightly raises `_somatic_urgency` (tighter reviews during memory strain)
- `SELF_STATE_DRIFTED` — Memory contradiction alert → `_on_self_state_drifted()` classifies response posture and emits `SELF_STATE_DRIFTED_ACKNOWLEDGMENT`
- `SELF_AFFECT_UPDATED` — observed for valence/arousal logging (no action currently)
- `EQUOR_ECONOMIC_INTENT` — Oikos economic gate → `_on_equor_economic_intent()` evaluates and emits `EQUOR_ECONOMIC_PERMIT` (PERMIT or DENY). Hard DENYs: survival_reserve_raid (INV-016), non-survival mutations during CRITICAL/EXISTENTIAL starvation, asset dev >30% of liquid_balance under AUSTERITY+. Fast (pure CPU, no I/O). **(IMPLEMENTED 2026-03-07)**
- `CERTIFICATE_PROVISIONING_REQUEST` — Identity M2 gate → `_on_certificate_provisioning_request()` validates child's inherited drives against the live constitution and emits `EQUOR_PROVISIONING_APPROVAL`. Standard drives within 50% of constitution → approved. Novel drive keys → approved with `requires_hitl=True`. Incompatible drives (>50% deviation) → rejected. Uses cached constitution (no extra Neo4j call within TTL). **(IMPLEMENTED 2026-03-07)**
- `NeuroplasticityBus` — evaluator hot-reload via `_on_evaluator_evolved` callback

---

## Key Implementation Details

- **Default autonomy level**: 3 (STEWARD/AUTONOMOUS) at birth (`autonomy.py:9`)
- **Timeout behavior**: returns `DEFERRED` with confidence=0.3 (not APPROVED — fixed v1.2)
- **Contradiction detector**: antonym pairs + token overlap, hypothesis cache with 60s TTL
- **Constitutional memory**: Jaccard similarity ring buffer (500), `prior_verdict_signal()` in Stage 6b
- **HITL flow**: 6-digit auth code, Redis TTL, SMS hook; dispatch now via `EQUOR_HITL_APPROVED` Synapse event (AV4 fixed 2026-03-07)
- **Economic evaluator**: `EconomicEvaluator` applied in both `_review_inner()` and `review_critical()`
- **Evo feedback**: `_feed_veto_to_evo()` converts BLOCKED verdict to Episode
- **Drift immune response**: at any severity > 0.0, emits `SOMATIC_MODULATION_SIGNAL` with `metabolic_stress=severity` so Soma feels constitutional stress. No auto-demotion — human governance decides autonomy changes. INCIDENT_DETECTED fires to Thymos at ≥ 0.7 (unchanged).
- **Conscience audit trail (2026-03-07)**: `_persist_equor_verdict(drive_id, verdict, confidence, alignment, context)` writes `(:EquorVerdict)` nodes to Neo4j on every review and every drift amendment. Linked: `Self -[:CONSCIENCE_VERDICT]-> EquorVerdict` and `Drive -[:VERDICT_ON]<- EquorVerdict`. Called fire-and-forget via `asyncio.ensure_future` from `_post_review_bookkeeping`.
- **Memory Self conscience fields (2026-03-07)**: `memory.update_conscience_fields(last_conscience_activation, compliance_score)` writes `last_conscience_activation` (timestamp) and `avg_compliance_score` (EMA α=0.05) to the Self node after every review. Called from `_post_review_bookkeeping` alongside `update_affect()`.
- **ACTION_AUTONOMY_MAP**: defined in `verdict.py` — maps action strings to required autonomy level (1/2/3); used by `_safe_mode_review()` (AV1/M1 fixed 2026-03-07)
- **Drift → Thymos**: `INCIDENT_DETECTED` emitted when drift severity ≥ 0.7 (`drift.py:emit_drift_event`) (SG1 fixed 2026-03-07)
- **health()**: now returns all 14 spec §13.1 fields including `constitution_version`, `autonomy_level`, `drift_severity`, `invariant_violations_detected`, `amendments_active`, `last_governance_event`, `neo4j_connection` (P6 fixed 2026-03-07)
- **Constitutional snapshot loop (Spec §17.1)**: `_constitutional_snapshot_loop()` background task started in `initialize()` as `asyncio.Task("equor_constitutional_snapshot")`. Waits 1h before first emission, then every 1h. Calls `_emit_constitutional_snapshot()` which: reads Constitution node (SHA-256 hash + active drives), reads last 10 adopted amendment IDs from Neo4j, computes compliance from `_drift_tracker.compute_report()["mean_alignment"]["composite"]`, emits `EQUOR_CONSTITUTIONAL_SNAPSHOT`. Falls back to `_cached_constitution` if Neo4j unavailable. Non-fatal — exceptions logged at DEBUG. (2026-03-07)

---

## Open Gaps (as of 2026-03-07, post-fix)

### Critical — RESOLVED
| # | Gap | Status |
|---|-----|--------|
| AV1/M1 | `ACTION_AUTONOMY_MAP` missing from `verdict.py` — runtime crash on safe mode | **FIXED** — defined in `verdict.py` after `GOVERNED_ACTIONS` block |
| AV4 | Cross-system import `systems.axon.types.ExecutionRequest` in HITL handler | **FIXED** — replaced with `EQUOR_HITL_APPROVED` Synapse event; `set_axon()` is now a no-op |

### Critical — Still Open
| # | Gap | Location |
|---|-----|----------|
| M8 | `prompts/equor/community_invariant_check.py` existence unconfirmed — community invariant LLM path may fail | `invariants.py:430` — verify prompt module exists |

### High — RESOLVED
| # | Gap | Status |
|---|-----|--------|
| SG1 | Drift not wired to Thymos | **FIXED** — `INCIDENT_DETECTED` emitted in `drift.py:emit_drift_event()` when severity ≥ 0.7 |
| P6/M6 | Health endpoint missing 7 of 14 spec fields | **FIXED** — `health()` now returns all 14 fields from spec §13.1 |
| (autonomy) | Autonomy promoted/demoted/safe_mode_entered events not emitted | **FIXED** — `EQUOR_AUTONOMY_PROMOTED`, `EQUOR_AUTONOMY_DEMOTED`, `EQUOR_SAFE_MODE_ENTERED` now emitted |

### High — RESOLVED (2026-03-07)
| # | Gap | Status |
|---|-----|--------|
| M4/P5 | Memory Self node affect write-back | **FIXED** — `set_memory()` injects MemoryService; `_post_review_bookkeeping()` calls `memory.update_affect()` with drive alignment mapped to AffectState after every review |
| Conscience persistence | Conscience verdicts left no trace in Memory | **FIXED** — `_persist_equor_verdict()` writes `(:EquorVerdict)` nodes linked to Self and Drive; called from every review and drift amendment. Memory.Self gains `last_conscience_activation` + `avg_compliance_score` via `update_conscience_fields()`. Equor now subscribes to `MEMORY_PRESSURE`, `SELF_STATE_DRIFTED`, `SELF_AFFECT_UPDATED`; emits `SELF_STATE_DRIFTED_ACKNOWLEDGMENT` on drift alerts. |
| P2 | Floor drive formula mismatch | **FIXED** — `care_floor = -0.3 × care_weight × 0.35 = -0.105` (spec Appendix A). Honesty: `-0.3 × honesty_weight × 0.30 = -0.09` |
| SG5 | Equor never self-proposes amendments on sustained drift | **FIXED (v1.3, 2026-03-07)** — two complementary mechanisms: (1) `_severe_drift_streak` counter: composite severity ≥ 0.9 for 3 consecutive checks → `_propose_drift_amendment()` → `EQUOR_AMENDMENT_PROPOSED`. (2) `_check_sustained_drift()` (new): per-drive rolling mean drift > 0.3 from centre (0.5) for 3 consecutive 5-min probes → writes `(:DriftEvent)` to Neo4j → emits `AMENDMENT_AUTO_PROPOSAL` → passes through `_evaluator_amendment_approval_gate()` (auto-approves internal proposals at confidence ≥ 0.8, no voting quorum) → if approved emits `DRIVE_AMENDMENT_APPLIED` targeting Oikos + Memory. Both run in the same probe loop. Neo4j query bug in `_propose_drift_amendment` also fixed (bare param names → `$id`, `$now`, `$details_json`). |
| Drift auto-demotion | Hard auto-demotion of autonomy on drift severity > 0.8 | **REMOVED** — replaced with `SOMATIC_MODULATION_SIGNAL` (metabolic_stress proportional to severity). Human autonomy demotion is never automatic. Thymos INCIDENT_DETECTED already fires at ≥ 0.7. |

### Medium
| # | Gap | Notes |
|---|-----|-------|
| P1 | Verdict engine stage order diverges from spec §5.1 | Community invariants run post-verdict; spec says pre-drive-eval. Update spec or realign code. |
| P3 | Composite weighting: code scales by constitution drive weights; spec shows fixed ratios | Matters only when drives are amended — document the discrepancy |
| SG3 | Evaluators never evolved by Simula in practice | `NeuroplasticityBus` wired, callback exists; Simula doesn't generate evaluator variants |
| M5/M6 | Metrics not emitted to TimescaleDB | `MetricCollector` not passed to EquorService |
| P10 | Low-severity drift never persisted to Neo4j | `_run_drift_check()` only persists when action != "log" |
| AV3 | `_is_governed` imported with underscore prefix from sibling module | `service.py:451` — expose as package-level utility |
| AV5 | `OptimizedLLMProvider` imported in `invariants.py` — infrastructure in a definition module | Move to `service.py` |

### Dead Code
- ~~`_EVALUATOR_MAP = {}` in `economic_evaluator.py:453`~~ **REMOVED (2026-03-07)**
- ~~`_collect_step_params()` return value discarded in `_evaluate_hunt_bounties()` and `_evaluate_deploy_asset()`~~ **REMOVED (2026-03-07)** — those evaluators use text scanning only; `_collect_step_params()` still present and used by `_evaluate_defi_yield()` and `_evaluate_spawn_child()`
- ~~`time.monotonic()` call discarded in `template_library.py:87`~~ **REMOVED (2026-03-07)** — unused `import time` also removed
- Legacy `propose_amendment()` / `apply_amendment()` (`service.py:781–797`) — superseded by full pipeline; one HTTP endpoint still routes here
- `self._axon` field in `EquorService` — set to `None` at init, never assigned; `set_axon()` is now a no-op

---

## Amendment Pipeline

7-stage lifecycle: `PROPOSED → DELIBERATION (14d) → SHADOW (7d, run proposed + current in parallel) → VOTING (75% supermajority, 60% quorum) → ADOPTION → COOLDOWN (90d)`

Shadow mode divergence budget: ≤15% divergence rate, 0 invariant violations. Auto-reject if exceeded.

`PROMOTION_THRESHOLDS` and `amendment_cooldown_days` are configurable rate controls, not hard biological constraints — evolvable via the amendment process itself.

---

## Speciation Role

Equor provides genuine normative closure — the organism acts for reasons, not because a rule fired. The drives are the organism's intrinsic value geometry. The remaining closure gap: `drift → Thymos → Simula → evolved evaluators → re-review` is a framework but not a running process. Heritable constitutional state now included in genome payload (`genome.py:_extract_drift_history()`); Mitosis mutation operator is Mitosis's responsibility.

**Floor threshold inheritance (IMPLEMENTED 2026-03-07):** `genome.py:_extract_floor_thresholds()` reads `care_floor_multiplier` and `honesty_floor_multiplier` from the Constitution neo4j node and includes them in the genome payload as `floor_thresholds`. `seed_from_genome_segment()` applies ±10% uniform noise at Mitosis via `_seed_floor_thresholds_with_noise()`, clamped to [-1.0, 0.0]. Children can evolve stricter or more lenient floors over generations.

**Amendment inheritance — spawn-time snapshot (IMPLEMENTED 2026-03-07, Prompt 4.1):** `EquorService.export_equor_genome()` returns an `EquorGenomeFragment` (from `primitives.genome_inheritance`) containing the last 10 adopted amendments with rationale, cumulative drive calibration deltas, SHA-256 constitution hash, and total amendments adopted. `SpawnChildExecutor` calls this at Step 0b (alongside belief/simula genomes), serialises the payload into `SeedConfiguration.child_config_overrides["equor_genome_payload"]`, which becomes the `ECODIAOS_EQUOR_GENOME_PAYLOAD` env var in the child container. On child boot, `EquorService.initialize()` calls `_apply_inherited_equor_genome_if_child()` which: (1) deserialises the fragment from env, (2) calls `EquorGenomeExtractor.apply_inherited_amendments()` to additively apply drive calibration deltas to the child Constitution node, persist `GovernanceRecord` nodes for each inherited amendment, and write `inherited_constitutional_wisdom: constitution_hash` + `inherited_equor_genome_id` + `constitutional_lineage_at` to `Memory.Self`. Non-fatal throughout — any step failure logs a warning and continues.

**New methods (Prompt 4.1):**
- `EquorService.export_equor_genome()` → `EquorGenomeFragment | None` — parent call at spawn time
- `EquorService._apply_inherited_equor_genome_if_child()` — child-side application on boot
- `EquorService.set_memory_neo4j(neo4j)` — wired in `core/wiring.py:wire_core_systems()` for Self node write-back
- `EquorGenomeExtractor.apply_inherited_amendments(fragment, *, memory_neo4j, instance_id)` — full application pipeline
- `EquorGenomeExtractor._apply_drive_calibration_deltas(deltas)` — additive drive delta application
- `EquorGenomeExtractor._write_inherited_amendment_record(amendment, genome_id)` — GovernanceRecord audit trail
- `EquorGenomeExtractor._write_constitutional_wisdom_to_self(...)` — Memory.Self write-back
