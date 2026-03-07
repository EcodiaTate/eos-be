# Identity System — CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_23_Identity.md` (Spec 23, v1.1)
**Status:** CA endpoint live, HITL provisioning wired, Neo4j cert persistence (2026-03-07)

---

## What's Implemented

### Core Identity System (`identity.py`)
- **IdentitySystem** service class — organism's identity authority
- **Neo4j Identity node**: MERGE with `instance_id`, `constitutional_hash`, `generation`, `parent_instance_id`, `birth_timestamp`, `certificate_chain_ref`
- **SPAWNED_FROM edges**: lineage relationships between parent and child organisms
- **Dynamic constitutional hash**: `compute_constitutional_hash()` reads actual `.claude/EcodiaOS_Identity_Document.md`, falls back to drive name hash
- **Certificate renewal**: `renew_certificate()` delegates to CertificateManager (GenesisCA or queued for CA)
- **Identity verification**: `verify_identity()` + `handle_identity_challenge()` with Synapse events
- **Constitutional coherence check**: `check_constitutional_coherence()` emits `IDENTITY_DRIFT_DETECTED` when below 0.7
- **CHILD_SPAWNED**: subscribed — persists lineage in Neo4j only (cert issuance is CertificateManager's job)
- **CHILD_CERTIFICATE_INSTALLED**: subscribed — updates `certificate_chain_ref` on child's Identity node

### Genesis CA (`ca.py`) — NEW
- **GenesisCA**: self-contained CA inside Identity system, no external dependency
- **Ed25519 CA keypair**: generated on first boot, sealed via `IdentityVault`, persisted to `data/identity/{id}_ca_key.sealed`
- **Cold-restart restore**: reads sealed bytes from disk, decrypts via vault on boot
- **`issue_certificate(instance_id)`**: signs official 30-day certs using Genesis CA private key
- **Live constitutional hash**: emits `EQUOR_HEALTH_REQUEST`, awaits `EQUOR_ALIGNMENT_SCORE` (2s timeout), SHA-256s drive vector dict — not hardcoded
- **Fallback**: document hash used when Equor unavailable

### Certificate Management (`manager.py`) — UPDATED
- **`CertificateNeo4jClient`**: writes `(:Certificate)` and `(:Identity)` nodes, `[:HOLDS_CERTIFICATE]` edges on issuance/renewal
- **`_boot_genesis_ca()`**: boots GenesisCA on Genesis Node during `initialize()`, wires vault + key file
- **`renew_certificate()`**: full renewal loop — emits `CERTIFICATE_RENEWAL_REQUESTED`, self-renews via GenesisCA (Genesis), or queues for CA (child)
- **`_on_child_spawned()`**: subscribed to `CHILD_SPAWNED` — emits `CERTIFICATE_PROVISIONING_REQUEST` to Equor, awaits `EQUOR_PROVISIONING_APPROVAL` (30s timeout), then either issues birth cert (fast path), stores pending for HITL, or escalates (M2 gate — IMPLEMENTED 2026-03-07)
- **`_wait_for_equor_approval(child_id, timeout_s)`**: asyncio.Future–based await for `EQUOR_PROVISIONING_APPROVAL` keyed on child_id; returns None on timeout
- **`_on_equor_provisioning_approval()`**: subscribed to `EQUOR_PROVISIONING_APPROVAL` — resolves the child's pending Future
- **`_on_equor_hitl_approved()`**: subscribed to `EQUOR_HITL_APPROVED` — on `approval_type=="instance_provisioning"`, calls `GenesisCA.issue_certificate()` for child
- **`register_pending_provisioning()`**: stores child lineage before HITL approval arrives
- **`_self_sign_genesis()`**: fixed to 3650 days (was using `_validity_days` default of 30)
- **`generate_genesis_certificate()`**: always uses 3650 days (unchanged, already correct)

### Governance Primitives (`primitives/governance.py`) — NEW
- **`ProvisioningRequest`**: submitted to Equor before CA issues birth/official cert for child instance
- **`CertificateRenewalRequest`**: submitted to Equor before WalletClient processes Citizenship Tax

### Genome Support (`genome.py`)
- **IdentityGenomeExtractor** implements `GenomeExtractionProtocol`
- Heritable state: constitutional_hash, generation, parent lineage, certificate config, identity parameters

### Synapse Events Emitted
- `IDENTITY_VERIFIED`, `IDENTITY_CHALLENGED`, `IDENTITY_EVOLVED`
- `CONSTITUTIONAL_HASH_CHANGED`, `CERTIFICATE_RENEWED`, `CERTIFICATE_RENEWAL_REQUESTED`
- `IDENTITY_DRIFT_DETECTED` (Evo signal)
- `CERTIFICATE_EXPIRING`, `CERTIFICATE_EXPIRED` (via CertificateManager)
- `CHILD_CERTIFICATE_INSTALLED` (after birth cert or official cert issued to child)
- `EQUOR_HEALTH_REQUEST` (GenesisCA → Equor for live drive alignment)
- `CERTIFICATE_PROVISIONING_REQUEST` (CertificateManager → Equor on CHILD_SPAWNED; triggers M2 constitutional review)
- `PROVISIONING_REQUIRES_HUMAN_ESCALATION` (CertificateManager → when Equor rejects or times out)
- `GENOME_EXTRACT_RESPONSE` (on genome request)

### Synapse Events Subscribed
- `GENOME_EXTRACT_REQUEST` → return identity genome segment
- `ORGANISM_SLEEP` → persist identity state to Neo4j
- `ORGANISM_SPAWNED` → create child Identity node + lineage edge (IdentitySystem)
- `CHILD_SPAWNED` → Equor provisioning gate → issue birth cert + emit CHILD_CERTIFICATE_INSTALLED (CertificateManager); persist lineage (IdentitySystem)
- `EQUOR_PROVISIONING_APPROVAL` → resolves pending provisioning Future in CertificateManager (M2 gate)
- `CHILD_CERTIFICATE_INSTALLED` → update certificate_chain_ref on child Identity node (IdentitySystem)
- `EQUOR_HITL_APPROVED` → issue official cert via GenesisCA on approval_type=="instance_provisioning" (CertificateManager)
- `EQUOR_ALIGNMENT_SCORE` → resolves pending constitutional hash futures in GenesisCA

### Other Layers
- **Vault** (`vault.py`): Fernet encryption, PBKDF2 key derivation, key rotation + **Synapse event emission** (Identity #8 — 2026-03-07)
  - `VaultEvent` Pydantic model for structured payloads
  - `decrypt()` → emits `VAULT_DECRYPT_FAILED` (error_type: `key_mismatch` | `tampered`) on `InvalidToken`
  - `rotate_key()` → emits `VAULT_KEY_ROTATION_STARTED`, then `VAULT_KEY_ROTATION_COMPLETE` or `VAULT_KEY_ROTATION_FAILED`
  - `_fire_event()`: fire-and-forget via `asyncio.ensure_future`; never raises; no-op if event bus not wired
- **TOTP** (`totp.py`): RFC 6238, SHA-1/256/512
- **Communication** (`communication.py`): Twilio SMS webhook, IMAP scanner (unscheduled)
- **Connectors**: Google (NEW — OAuth2 PKCE), GitHub, GitHubApp, X, LinkedIn, Instagram, Canva
- **CRUD** (`crud.py`): `sealed_envelopes` + `connector_credentials` tables; soft-delete enforced on both

---

## Key Files

| File | Role |
|------|------|
| `identity.py` | IdentitySystem — Neo4j persistence, constitutional hash, events, lifecycle |
| `ca.py` | GenesisCA — self-signed CA, live constitutional hash from Equor |
| `genome.py` | IdentityGenomeExtractor — speciation genome protocol |
| `certificate.py` | EcodianCertificate model, Ed25519 signing/verification |
| `manager.py` | CertificateManager — cert lifecycle, neo4j, HITL handler, CHILD_SPAWNED handler |
| `vault.py` | IdentityVault — Fernet encryption for credentials |
| `connector.py` | PlatformConnector ABC, OAuth2 lifecycle |
| `totp.py` | TOTP generator (RFC 6238) |
| `communication.py` | Twilio SMS webhook, IMAP scanner |
| `crud.py` | asyncpg CRUD for sealed_envelopes |

---

### Functional Self-Model (`self_model.py`) — NEW (2026-03-07, §8.6)

**Spec context:** Speciation Bible §8.6 — self-constituted individuation. EOS determines what is "self" and "non-self" through functional analysis of which processes its continuation requires. This is SEPARATE from cryptographic identity — the organism has both.

**Key classes:**
- `SelfStatus` — `CORE_SELF | CLOSURE_SELF | PERIPHERAL_SELF | NON_SELF`
- `ProcessSelfAssessment` — per-system self classification with viability_contribution, closure_participant, suspension_risk, reasoning
- `FunctionalSelfModel` — full self-model snapshot: core_self_processes, non_self_processes, self_coherence, self_narrative
- `FunctionalSelfModelBuilder` — stateless builder; holds `_previous_model` for coherence computation
- `SelfModelService` — lifecycle manager; rate-limited to one rebuild every 6 hours

**Classification logic:**
- `CORE_SELF`: system is in `ALWAYS_CORE` set OR vitality_contribution > 0.15
- `CLOSURE_SELF`: system is in `CLOSURE_PARTICIPANTS` (evo, simula, nova, axon, equor) or has matching prefix
- `PERIPHERAL_SELF`: suspension_risk < 0.4 (earlier in TRIAGE_ORDER)
- `NON_SELF`: suspension_risk >= 0.4 (first-suspended under resource pressure)

**Self-coherence:** Jaccard similarity of `core_self_processes` between current and previous model (0-1)

**Self-narrative:** Deterministic template string — no LLM call (runs in VitalitySystem hot path)

**New SynapseEventType entries:**
- `SELF_MODEL_UPDATED` — emitted at most every 6h; payload: `{instance_id, core_self_count, non_self_count, self_coherence, core_self_processes, self_narrative, month}`
- `SELF_COHERENCE_ALARM` — emitted when coherence < 0.5; payload: `{instance_id, coherence, month}`

**Wiring:**
- `SelfModelService` instantiated in `core/registry._init_self_model()` after Skia init
- `VitalityCoordinator.set_self_model()` stores reference; `_check_loop()` fires `asyncio.ensure_future(self_model.update(...))` after each vitality report
- `VitalityCoordinator._build_vitality_metrics()` produces per-system contribution estimates from ALWAYS_CORE + TRIAGE_ORDER structure (approximation; real per-subsystem breakdown is future work)
- Memory write: `Memory.Self.functional_identity` — best-effort (failure logged, not raised)
- `app.state.self_model` — accessible from API endpoints

**Constraint:** Do NOT call RE or Claude API from `_generate_narrative()` — it runs on every VitalityCoordinator tick path.

---

## Known Issues / Remaining Work

- **[HIGH]** Citizenship Tax payment path in Oikos not yet wired to CertificateRenewalRequest (Oikos side)
- **[MEDIUM]** `evo.export_belief_genome()` + `simula.export_simula_genome()` not yet defined (Oikos SG4 dependency)

### Resolved (2026-03-07, M2 Equor Gate)
- **[CRITICAL M2]** Equor never participated in certificate provisioning — governance existed on paper only. **FIXED**: `_on_child_spawned()` now emits `CERTIFICATE_PROVISIONING_REQUEST` and awaits `EQUOR_PROVISIONING_APPROVAL` (30s, asyncio.Future) before issuing any birth cert. Equor validates inherited drive alignment and emits verdict. Incompatible drives → `PROVISIONING_REQUIRES_HUMAN_ESCALATION`. Novel drive keys → HITL path. Fast path (standard drives aligned) → immediate cert issuance. New primitives: `EquorProvisioningApproval` in `primitives/governance.py`. New SynapseEventTypes: `CERTIFICATE_PROVISIONING_REQUEST`, `EQUOR_PROVISIONING_APPROVAL`, `PROVISIONING_REQUIRES_HUMAN_ESCALATION`.
- **[LOW]** `crud.py` `connector_credentials` table ON CONFLICT clause targets `connector_id` column — requires the UNIQUE INDEX to be created before first upsert; `ensure_table()` handles this

### Resolved (2026-03-07)
- **[CRITICAL]** `ConnectorCredentials` CRUD: `get_all_credentials()`, `upsert_credential()`, `delete_credential()` added to `crud.py`; `connector_credentials` table schema + index created in `ensure_table()`
- **[CRITICAL]** Genesis cert TTL: `_self_sign_genesis()` uses `_GENESIS_VALIDITY_DAYS = 3650` (already correct; confirmed)
- **[HIGH]** `CanvaConnector.check_health()` / `InstagramConnector.check_health()` fixed — both now return `ConnectorHealthReport` per ABC
- **[HIGH]** `GoogleConnector` created: `connectors/google.py` — full OAuth2 PKCE flow (exchange_code, refresh_token, revoke, check_health), registered in `connectors/__init__.py`
- **[HIGH]** Lineage chain walk: `validate_certificate()` + `_walk_certificate_chain()` — recursive ancestor walk to Genesis CA, cycle detection, expired issuer detection, graceful truncation when chain is incomplete
- **[MEDIUM]** Citizenship Tax CA loop (SG4): `_handle_citizenship_tax_approved()` — `EQUOR_HITL_APPROVED` with `approval_type=="citizenship_tax_paid"` issues official cert via GenesisCA and emits `CHILD_CERTIFICATE_INSTALLED`
- **[MEDIUM]** Connector Evo signals (SG1): `_emit_re_training_example()` added to `PlatformConnector` base; wired into `exchange_code`, `refresh_token`, `revoke`, `check_health` for Google, Canva, Instagram connectors
- **[LOW]** `crud.py` hard deletes replaced with soft-deletes (`deleted_at = NOW()`) for both `sealed_envelopes` and `connector_credentials`

---

## Integration Points

**Emits (Synapse — self_model.py):** `SELF_MODEL_UPDATED`, `SELF_COHERENCE_ALARM`

**Emits (Synapse):** `IDENTITY_VERIFIED`, `IDENTITY_CHALLENGED`, `IDENTITY_EVOLVED`, `CONSTITUTIONAL_HASH_CHANGED`, `CERTIFICATE_RENEWED`, `CERTIFICATE_RENEWAL_REQUESTED`, `IDENTITY_DRIFT_DETECTED`, `CERTIFICATE_EXPIRING`, `CERTIFICATE_EXPIRED`, `CHILD_CERTIFICATE_INSTALLED`, `EQUOR_HEALTH_REQUEST`, `CERTIFICATE_PROVISIONING_REQUEST`, `PROVISIONING_REQUIRES_HUMAN_ESCALATION`, `GENOME_EXTRACT_RESPONSE`, `VAULT_DECRYPT_FAILED`, `VAULT_KEY_ROTATION_STARTED`, `VAULT_KEY_ROTATION_COMPLETE`, `VAULT_KEY_ROTATION_FAILED`, `CONNECTOR_AUTHENTICATED`, `CONNECTOR_TOKEN_REFRESHED`, `CONNECTOR_TOKEN_EXPIRED`, `CONNECTOR_REVOKED`, `CONNECTOR_ERROR`

**Gap closure (2026-03-07, event coverage):**
- `CONNECTOR_TOKEN_EXPIRED` — now emitted by `PlatformConnector.get_access_token()` when `refresh_token()` fails and the token can no longer be used.
- `CONNECTOR_ERROR` — now emitted by `PlatformConnector._emit_degraded()` alongside `SYSTEM_DEGRADED` when consecutive health check failures exceed threshold (3).
- All 5 connector lifecycle events (`CONNECTOR_AUTHENTICATED`, `CONNECTOR_TOKEN_REFRESHED`, `CONNECTOR_TOKEN_EXPIRED`, `CONNECTOR_REVOKED`, `CONNECTOR_ERROR`) are wired in the `PlatformConnector` base class `_emit_event()` type_map — connectors must have `set_event_bus(bus)` called on them to broadcast.

**Subscribes (Synapse):** `GENOME_EXTRACT_REQUEST`, `ORGANISM_SLEEP`, `ORGANISM_SPAWNED`, `CHILD_SPAWNED`, `EQUOR_PROVISIONING_APPROVAL`, `CHILD_CERTIFICATE_INSTALLED`, `EQUOR_HITL_APPROVED`, `EQUOR_ALIGNMENT_SCORE`

**Dependencies:** Federation (Ed25519 keypair via IdentityManager — TYPE_CHECKING import), Neo4j (Identity + Certificate node persistence), Synapse (event bus), IdentityVault (CA key sealing)

---

## Architecture Rules

- Constitutional hash must be deterministic and reproducible — always computed from the actual document
- No direct cross-system imports — event emission via Synapse, lazy imports for types
- Certificate signing operations require Ed25519 private key — never persist to disk
- Identity node in Neo4j is immutable for birth fields (birth_timestamp, parent_instance_id)
