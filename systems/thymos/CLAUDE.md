# Thymos — Immune System (Spec 12)

## What's Implemented

### Core Pipeline
- **6-tier repair**: NOOP → PARAMETER → RESTART → KNOWN_FIX → NOVEL_FIX → ESCALATE
- **5 sentinel types**: Exception (fingerprint = system+exc_type+first-local-frame), Contract (SLA monitoring), FeedbackLoop (15 severed-loop checks), Drift (8 metrics, σ-threshold SPC), CognitiveStall (broadcast ack rate, nova intent rate, etc.)
- **Triage**: composite severity scoring (blast radius 0.25 + recurrence velocity 0.20 + constitutional impact 0.25 + user visibility 0.15 + healing potential 0.15), fingerprint dedup with class-specific windows
- **`DiagnosticEngine`** with LLM-powered hypothesis generation (do NOT modify prompts) — max 3 hypotheses, testable via `DIAGNOSTIC_TEST_REGISTRY`
- **`CausalAnalyzer`**: Neo4j `(System)-[:DEPENDS_ON]->(System)` traversal (5-min cache, hardcoded fallback)
- **`TemporalCorrelator`**: TimescaleDB metric anomaly + system event queries (30s window)
- **`AntibodyLibrary`**: fingerprint lookup, effectiveness tracking (refinement <0.6, retirement <0.3 after 5+ apps), generation lineage in Neo4j
- **Repair validation gates**: Equor review (Tier 3+), blast radius check (>0.5 → escalate), Simula sandbox (Tier 4), rate limits (5/hour, 3 novel/day)
- **`HealingGovernor`**: rate limiting, storm detection, hysteresis-based storm exit
- **`HomeostasisController`**: adaptive baselines (7-day rolling median ± 25%)
- **`DriftSentinel`**: configurable σ-thresholds per metric
- **Embedding-based prophylactic scanner (P2, IMPLEMENTED 2026-03-07)**: 768-dim sentence-transformer cosine similarity >0.85 threshold; antibody fingerprint embeddings cached in `_fingerprint_store`; keyword fallback when embedder unavailable; `check_intent_similarity()` for intent-time gating; `add_fingerprints_from_procedures()` for Oneiros ingestion
- `ThymosService.set_embedding_client()` — hot-swap wiring from main.py; also wires into scanner if already initialized
- **INV-017 Drive Extinction handler (IMPLEMENTED 2026-03-07)**: `_on_drive_extinction()` subscribes to `DRIVE_EXTINCTION_DETECTED` (from Equor). Creates CRITICAL / `IncidentClass.DRIVE_EXTINCTION` incident with `blast_radius=1.0`, `user_visible=True`. No autonomous repair: blast_radius > 0.5 triggers validation gate escalation at Tier 3+ (`prescription.py:467`). Requires human/federation governance review.

### Synapse Integration (AV1 migration)
- All sub-components use `_on_event` callbacks bridged to Synapse by `ThymosService`
- Soma/Oikos health reads migrated from direct calls to cached subscription state
- Sandbox validation via correlation-based SIMULA_SANDBOX_REQUESTED/RESULT (30s timeout, fail-closed)
- 6+ lifecycle events emitted (INCIDENT_CREATED, REPAIR_APPLIED, REPAIR_ROLLED_BACK, etc.)
- Pydantic v2 validation on all 27+ subscribed events (`event_payloads.py`, non-blocking)

### Telemetry Loops
- `_vitality_loop()` — VITALITY_SIGNAL every 60s (antibody count, repair success rate, MTTH, novel ratio, storm status)
- `_drive_pressure_loop()` — THYMOS_DRIVE_PRESSURE every 30s (4 constitutional drives)

### Speciation Features
- Federation antibody sync: `export_for_federation()` / `import_from_federation()` with trust gating (ally/bonded/kin)
- RE training data: full repair episodes emitted as RE_TRAINING_DATA with measured outcome_quality
- Deepened SPECIATION_EVENT handler: antibody cross-ref, DriftSentinel tightening (×0.7), federation quarantine

## What's Missing

| ID | Item | Priority |
|----|------|----------|
| SG1 | Mitosis antibody inheritance (federation sync done) | BLOCKED on Spec 26 |

## Completed (7 Mar 2026)

| ID | Item | Where |
|----|------|-------|
| M7 | Persist `(:Repair)` nodes with `[:REPAIRED_WITH]` edge for all outcomes | `service.py::_persist_repair_node()` |
| M8 | `HomeostasisController.check_drift_warnings()` — broadcasts HOMEOSTASIS_ADJUSTED (warn_only=True) in pre-repair zone | `prophylactic.py` + `service.py::_homeostasis_loop` |
| SG4 | `_try_federation_escalation()` — broadcast INCIDENT_ESCALATED to peers, wait 45s for FEDERATION_ASSISTANCE_ACCEPTED before human escalation | `service.py` |
| SG7 | Subscribe to KAIROS_INVARIANT_DISTILLED → `_on_kairos_invariant()` injects edges into CausalAnalyzer._graph_deps | `service.py` |
| P8 | Per-sentinel try/except → `_raise_sentinel_internal_incident()` creates MEDIUM DEGRADATION incident on sentinel crash | `service.py::_sentinel_scan_loop` |
| P8 | Version-rollback guard → `_request_model_version_rollback()` emits MODEL_ROLLBACK_TRIGGERED on MODEL_HOT_SWAP_FAILED | `service.py` |
| P2 | Upgrade prophylactic scanner to 768-dim embedding cosine similarity (>0.85 threshold); keyword fallback; fingerprint store cache; `check_intent_similarity()`; `set_embedding_client()` | `prophylactic.py` + `service.py` |
| SG8 | Subscribe to ONEIROS_CONSOLIDATION_COMPLETE → query `(:Procedure {thymos_repair: true})` → batch-embed → inject into prophylactic fingerprint store | `service.py::_on_oneiros_consolidation()` |
| Nova feedback | Emit THYMOS_REPAIR_VALIDATED on Tier 3+ verified success so Nova can strengthen recoverability priors | `service.py` + `synapse/types.py` |
| Identity #8 | Subscribe to `VAULT_DECRYPT_FAILED` → MEDIUM `SECURITY` incident; `VAULT_KEY_ROTATION_FAILED` → CRITICAL `SECURITY` incident. `_on_vault_decrypt_failed()` + `_on_vault_key_rotation_failed()` handlers added. New `IncidentClass.SECURITY` added to `primitives/incident.py`. | `service.py` |

## Known Issues
- AV1 migration is incremental — some direct cross-system calls may remain
- AV4: background tasks start before all subscriptions confirmed (race window)
- D1/D3/D4: dead code candidates (`SimulationResult`, `AddressBlacklistEntry`, `record_metric_anomaly`)
- SG4: `_event_bus.subscribe/unsubscribe` signature varies by Synapse implementation; handler may not deregister cleanly on all bus variants

## Key Files
- `service.py` — main ThymosService (~6100+ lines)
- `diagnosis.py` — CausalAnalyzer, DiagnosticEngine
- `antibody.py` — AntibodyLibrary with federation sync
- `governor.py` — HealingGovernor with storm hysteresis
- `prophylactic.py` — HomeostasisController with adaptive baselines + drift warnings
- `event_payloads.py` — 35+ Pydantic v2 payload models

## Integration Surface
- **Emits:** SYSTEM_HEALED, THYMOS_STORM_ENTERED/EXITED, RE_TRAINING_DATA, VITALITY_SIGNAL, THYMOS_DRIVE_PRESSURE, FEDERATION_ANTIBODY_SHARED, FEDERATION_TRUST_UPDATED, SIMULA_SANDBOX_REQUESTED, HOMEOSTASIS_ADJUSTED (warn_only), INCIDENT_ESCALATED (federation_broadcast), MODEL_ROLLBACK_TRIGGERED, **THYMOS_REPAIR_VALIDATED** (Tier 3+ success → Nova recoverability priors), **IMMUNE_CYCLE_COMPLETE** (end of every sentinel scan loop iteration — payload: `cycle_timestamp`, `active_incidents`, `antibody_count`)
- **Consumes:** SYSTEM_FAILED, SYSTEM_DEGRADED, SOMATIC_MODULATION, SPECIATION_EVENT, FEDERATION_KNOWLEDGE_RECEIVED, SIMULA_SANDBOX_RESULT, KAIROS_INVARIANT_DISTILLED, FEDERATION_ASSISTANCE_ACCEPTED, **ONEIROS_CONSOLIDATION_COMPLETE** (→ SG8 repair schema ingestion), **DRIVE_EXTINCTION_DETECTED** (→ INV-017 CRITICAL incident, Tier 5 governance escalation, no autonomous repair), **AXON_EXECUTION_REQUEST** (risky=True only → `_on_axon_execution_request()` prophylactic scanner pre-screens intent similarity), **AXON_ROLLBACK_INITIATED** (→ `_on_axon_rollback_initiated()` creates DEGRADED/MEDIUM incident via `on_incident()`), **VAULT_DECRYPT_FAILED** (→ MEDIUM SECURITY incident), **VAULT_KEY_ROTATION_FAILED** (→ CRITICAL SECURITY incident), + 20 more (see event_payloads.py _PAYLOAD_MODELS)
- **Neo4j:** reads `(System)-[:DEPENDS_ON]->(System)` for causal analysis; writes `(:Repair)-[:REPAIRED_WITH]->(:Incident)`
