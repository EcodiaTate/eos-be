# Federation System — CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_11_Federation.md` (Spec 11b)
**Status:** Core + population dynamics + peer discovery + RE quality scoring complete (2026-03-07)

---

## What's Implemented

### Core Protocol
- **Identity**: Ed25519 keypair, `InstanceIdentityCard`, dynamic constitutional hash from drive weights
- **Handshake**: 4-phase mTLS protocol (HELLO → CHALLENGE → ACCEPT → CONFIRM), nonce replay prevention
- **Trust**: 5 levels (NONE → ALLY), score thresholds (5/20/50/100), 3x violation penalty, privacy breach instant reset, time-based decay (24h grace)
- **Knowledge Exchange**: `KnowledgeExchangeManager` + IIEP `ExchangeProtocol` (dual paths), trust-gated sharing permissions
- **Privacy Filter**: PII key stripping, email/phone regex, PRIVATE always removed, COMMUNITY_ONLY at COLLEAGUE+
- **Coordination**: Assistance request/response with trust gate (COLLEAGUE+), Equor review
- **Channels**: mTLS via httpx, `ChannelManager` with SSL context, dev mode fallback
- **Link Persistence**: Redis primary + local file fallback
- **IIEP**: Exchange protocol, ingestion pipeline (6-stage + 2 new stages), threat intelligence, reputation staking
- **Certificate Validation**: `_validate_sender_certificate()` on all inbound operations

### Population Dynamics (P2-2 — 2026-03-07)
- **Synapse Events**: 8 outbound events emitted (`_emit_synapse_event` helper), 12 `SynapseEventType` entries (including new `FEDERATION_PRIVACY_VIOLATION`)
- **Synapse Subscriptions**: 5 inbound handlers via `set_event_bus()` — economic state, cert rotation, incidents, sleep consolidation, privacy violations
- **Metabolic Gate**: Trust upgrades blocked when `metabolic_efficiency < 0.1`
- **Sleep Certification**: Knowledge sharing gated on `ONEIROS_CONSOLIDATION_COMPLETE` (bypasses THREAT_ADVISORY)
- **Constitutional Hash**: Dynamic from injectable drive weights via `_compute_constitutional_hash()`, `set_equor()` pulls live weights
- **Novelty Score**: `_compute_novelty_score()` via content hash tracking, included in event payloads
- **Jaccard Fix**: `ReputationBond.claim_content` stores original content for real word-set extraction
- **ExchangeEnvelope Fix**: `send_fragment()` and `request_divergence_profile()` use correct field names
- **RE Training**: `RE_TRAINING_EXAMPLE` events emitted on assistance decisions
- **RE_ADAPTER_DIFF**: `KnowledgeType` added, ALLY trust gate — transport type only, no LoRA logic yet

### Peer Discovery (P2-3 — 2026-03-07)
- **`seed_peers`** field in `FederationConfig` — list of endpoint URLs to auto-connect on startup
- **`seed_retry_interval_seconds`** — configurable retry interval for failed seed connections (default 300s)
- **`_connect_seed_peers()`** — async, fire-and-forget, skips already-linked endpoints, logs success/failure
- **`_retry_seed_peers()`** — single-pass retry loop after interval, converges to empty when all connected
- Resolves Spec §XIII gap #5 (no discovery mechanism). Manual `establish_link(endpoint)` still works.

### Knowledge Stub Consolidation (P2-3 — 2026-03-07)
- **`_retrieve_procedures()`** now delegates to `collect_procedures()` from `exchange.py` (single source of truth)
- **`_retrieve_hypotheses()`** now delegates to `collect_hypotheses()` from `exchange.py`
- `KnowledgeExchangeManager` gains `evo` and `instance_id` constructor params
- `set_evo()` propagates to both `_ingestion._evo` and `_knowledge._evo`
- Resolves Spec §VIII Fix #11 (dead knowledge.py stubs)

### Privacy Violation Detection (P2-3 — 2026-03-07)
- **Ingestion Stage 3.5**: `_detect_privacy_violation()` scans inbound payload content for 24 PII key patterns
- **`_emit_privacy_violation()`**: emits `FEDERATION_PRIVACY_VIOLATION` Synapse event + logs warning
- **`_on_privacy_violation()`** in FederationService: receives the event, records PRIVACY_BREACH interaction, triggers trust zero-reset via `_update_trust_and_emit()`
- `IngestionPipeline` gains `event_bus` constructor param; propagated from `set_event_bus()`
- Resolves Spec §XI gap (`FEDERATION_PRIVACY_VIOLATION` event not yet emitted)

### RE Semantic Quality Scoring (P2-3 — 2026-03-07)
- **Ingestion Stage 4.5**: `_run_re_quality_check()` scores HYPOTHESIS payloads at PARTNER+ trust
- Scores three dimensions via RE/Claude: coherence, novelty, constitutional safety
- Uses harmonic mean — any single weak dimension degrades the composite score
- Below `_RE_QUALITY_THRESHOLD` (0.35): DEFERRED (not rejected — local evidence may validate later)
- `set_re(re)` wirer on `FederationService` propagates to `_ingestion._re`
- Fail-open: no RE wired → all payloads pass Stage 4.5
- Resolves Spec §XII §2 (RE-assisted semantic quality scoring)

---

## Key Files

| File | Role |
|------|------|
| `service.py` | Main orchestrator — wires all subsystems, Synapse integration, link lifecycle, peer discovery |
| `identity.py` | `IdentityManager` — keypair, identity card, constitutional hash |
| `trust.py` | `TrustManager` — scoring, level transitions, decay |
| `knowledge.py` | `KnowledgeExchangeManager` — knowledge exchange, now delegates procedures/hypotheses to IIEP collectors |
| `privacy.py` | `PrivacyFilter` — PII removal, consent enforcement |
| `coordination.py` | `CoordinationManager` — assistance request/response |
| `channel.py` | `ChannelManager` — mTLS channel lifecycle |
| `handshake.py` | 4-phase handshake protocol |
| `exchange.py` | IIEP `ExchangeProtocol` — push/pull with provenance; canonical collectors for hypotheses/procedures |
| `ingestion.py` | IIEP ingestion pipeline — 7 stages: dedup, loop, privacy scan, EIS, RE quality, Equor, routing |
| `iiep.py` | Economic coordination — `CapabilityMarketplace`, `MutualInsurancePool` |
| `reputation_staking.py` | Bond create/forfeit/recover, contradiction detection |
| `threat_intelligence.py` | Signed advisory broadcast, trust-gated receipt |

---

## Known Issues / Remaining Work

- **[MEDIUM]** `systems.nexus.types` still inline lazy imports in `service.py` — should move to shared primitives
- **[MEDIUM]** LoRA diff collection/application for `RE_ADAPTER_DIFF` not implemented (requires RE system)
- **[LOW]** Privacy filter is regex-based, not k-anonymity/differential privacy
- **[LOW]** Nova alignment check is a pass-through flag, not a real Nova call
- **[LOW]** Trust decay contradicts spec philosophy (biological decay vs. digital persistence) — undocumented design choice

---

## Integration Points

**Emits (Synapse):** `FEDERATION_LINK_ESTABLISHED`, `FEDERATION_LINK_DROPPED`, `FEDERATION_TRUST_UPDATED`, `FEDERATION_KNOWLEDGE_SHARED`, `FEDERATION_KNOWLEDGE_RECEIVED`, `FEDERATION_INVARIANT_RECEIVED`, `WORLD_MODEL_FRAGMENT_SHARE`, `FEDERATION_ASSISTANCE_ACCEPTED`, `FEDERATION_ASSISTANCE_DECLINED`, `FEDERATION_PRIVACY_VIOLATION`, `RE_TRAINING_EXAMPLE`

**Gap closure (2026-03-07, event coverage):**
- `WORLD_MODEL_FRAGMENT_SHARE` — now emitted in `FederationService.send_fragment()` when a world model fragment is accepted by the peer (non-REJECTED receipt verdict). Data: `link_id`, `remote_instance_id`, `message_id`, `fragment_type`.
- `FEDERATION_INVARIANT_RECEIVED` — now emitted in `FederationService.handle_exchange_envelope()` when inbound PUSH payloads include at least one accepted hypothesis with confidence ≥ 0.9 (Kairos-distilled causal invariants travel as high-confidence hypotheses). Data: `link_id`, `remote_instance_id`, `invariant_count`, `min_confidence`.

**Subscribes (Synapse):** `ECONOMIC_STATE_UPDATED` (Oikos), `IDENTITY_CERTIFICATE_ROTATED` (Identity), `INCIDENT_DETECTED` (Thymos), `ONEIROS_CONSOLIDATION_COMPLETE` (Oneiros), `FEDERATION_PRIVACY_VIOLATION` (self — triggers trust reset)

**Post-init wirers:** `set_equor()`, `set_evo()`, `set_oikos()`, `set_simula()`, `set_eis()`, `set_re()`, `set_atune()`, `set_certificate_manager()`, `set_event_bus()`

**Dependencies:** Equor (constitutional review), Memory (knowledge retrieval), Redis (link persistence), Identity (certificates — via lazy imports), Nexus (fragment protocol — via lazy imports)

---

## Architecture Rules

- No module-level cross-system imports — use `TYPE_CHECKING` guard + lazy wrappers or inline function-level imports
- All trust updates go through `_update_trust_and_emit()` — never call `self._trust.update_trust()` directly
- All Synapse emissions go through `_emit_synapse_event()` helper (fire-and-forget, logged on error)
- Constitutional hash must stay deterministic (6-decimal precision, sorted drive keys)
- Ingestion stages are ordered: dedup(1) → loop(2) → privacy scan(3.5) → EIS(4) → RE quality(4.5) → Equor(5) → routing(6)
- Peer discovery is fire-and-forget from `initialize()` — never blocks startup
