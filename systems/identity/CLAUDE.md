# Identity System — CLAUDE.md

**Spec:** `.claude/EcodiaOS_Spec_23_Identity.md` (Spec 23, v1.2)
**Status:** CA endpoint live, HITL provisioning wired, Neo4j cert persistence, Telegram channel wired, autonomous account provisioner LIVE (2026-03-08), PersonaEngine LIVE (2026-03-09)

---

## What's Implemented

### PersonaEngine (`persona.py`) — NEW (2026-03-09)

EOS now has a coherent, consistent synthetic AI persona across all external platforms.

**Class:** `PersonaEngine`

**`PersonaProfile`** — canonical persona snapshot:
| Field | Purpose |
|-------|---------|
| `handle` | Synthetic instance-specific name (e.g. `EOS-Convergent-a1b2c3d4`) |
| `display_name` | Human-readable display name (e.g. `Ecodia · Convergent Agent`) |
| `bio_short` | ≤160 chars (X/Twitter bio) — AI disclosure appended unconditionally |
| `bio_long` | Full bio for GitHub/LinkedIn (≤600 chars) |
| `professional_domain` | Primary Telos specialisation |
| `voice_style` | One of: `technical-precise`, `curious-accessible`, `analytical-dry`, `warm-collaborative`, `concise-systematic` |
| `avatar_seed` | Deterministic seed → `https://api.dicebear.com/7.x/bottts/svg?seed={seed}` |
| `website` | Optional public endpoint |
| `ai_disclosure` | **Mandatory** `"Autonomous AI agent (EcodiaOS)"` — cannot be empty |
| `brand_lineage` | Ordered list of ancestor handles (oldest first) |
| `generation` | Lineage depth (1 = genesis) |

**`PersonaEngine` public API:**
- `generate_initial_persona(domain, parent_persona, llm_client)` — LLM generation with deterministic fallback; Equor constitutional gate; seals to vault; emits `PERSONA_CREATED`
- `evolve_persona(event, context, llm_client)` — 24h cooldown enforced; bio refresh; emits `PERSONA_EVOLVED`
- `get_platform_bio(platform)` — Returns bio truncated + disclosure for platform character limit
- `get_platform_handle(platform)` — Returns platform-safe handle variant
- `seal_persona(profile)` / `load_sealed_persona()` — IdentityVault persistence
- `current_handle` — Quick accessor for OrganismTelemetry wiring

**Equor constitutional gate:**
- Emits `CERTIFICATE_PROVISIONING_REQUEST` with `provisioning_type="persona_generation"`
- Awaits `EQUOR_PROVISIONING_APPROVAL` (30s timeout → auto-permit)
- Checks: `ai_disclosure` present, `handle` doesn't appear human
- DENY → falls back to `_generate_safe_default()` (maximally-honest minimal persona)

**Avatar:**
- DiceBear Bottts style: `https://api.dicebear.com/7.x/bottts/svg?seed={avatar_seed}`
- Same seed across all platforms = visually consistent identity
- No auth required
- `avatar_seed` = `{instance_id}-{domain_tag}` for semantic consistency

**Dependency injection:**
- `set_vault(vault)` — IdentityVault for persistence
- `set_event_bus(bus)` — Synapse EventBus for events + Equor subscription

**New SynapseEventTypes:**
- `PERSONA_CREATED` — first persona sealed for this instance
- `PERSONA_EVOLVED` — persona updated due to a life event

**Wiring:** Not yet wired into `registry.py` — call during `IdentitySystem.initialize()` after vault/bus are available. Suggest: `asyncio.ensure_future(persona_engine.generate_initial_persona(...))` after `provision_platform_identities()`.

---

### Autonomous Account Provisioner (`account_provisioner.py`) — NEW (2026-03-08)

EOS can now create its own external platform accounts without human intervention.

**Class:** `AccountProvisioner`

**`provision_platform_identities()`** — Entry point, called fire-and-forget from `IdentitySystem.initialize()` via `asyncio.ensure_future`. Runs in background, never blocks boot. Provisions:
1. Twilio phone number (if not already owned)
2. GitHub account (if not already created)

**`provision_twilio_number(area_code)`** — Fully API-driven (no browser):
- Equor constitutional gate first (economic action, ~$1.15/month)
- POST to `Twilio /IncomingPhoneNumbers` REST API
- Seals `{phone_number, twilio_sid, area_code, webhook_url}` in IdentityVault
- Sets `ECODIAOS_IDENTITY_COMM__TWILIO_FROM_NUMBER` at runtime
- Emits `PHONE_NUMBER_PROVISIONED`
- Emits `DOMAIN_EPISODE_RECORDED` for Oikos cost accounting
- Neo4j audit: `(:AccountProvisioning {platform: "twilio", status: "success"}) -[:PROVISIONED_BY]-> (:Identity)`

**`provision_github_account(username, email)`** — Playwright + 2captcha + OTPCoordinator:
- Equor gate
- GitHub username availability check (REST API `/users/{username}` → 404 = free)
- Playwright browser: github.com/signup → fill email/password/username → CAPTCHA → email OTP
- Generates PAT via Settings → Developer Settings → Personal Access Tokens
- Seals `{username, email, password, pat}` in IdentityVault
- Emits `GITHUB_ACCOUNT_PROVISIONED`
- Neo4j audit node

**`provision_gmail(username)`** — Playwright:
- Equor gate
- accounts.google.com/signup form flow
- Phone verification via provisioned Twilio number + SMS OTP
- Seals `{email, password}` in IdentityVault
- Emits `PLATFORM_ACCOUNT_PROVISIONED`

**`provision_platform_account(platform, config)`** — Generic extensible hook for future platforms.

**Per-instance naming:**
| Platform | Pattern |
|----------|---------|
| GitHub username | `ecodiaos-{instance_id[:8]}` |
| Gmail | `ecodiaos.{instance_id[:8]}@gmail.com` |

**Already-provisioned detection:** Neo4j query on `(:AccountProvisioning {platform, status: "success"})` node. Environment variable `ECODIAOS_IDENTITY_COMM__TWILIO_FROM_NUMBER` also checked for Twilio.

**Equor gate:**
- Emits `CERTIFICATE_PROVISIONING_REQUEST` with `provisioning_type: "platform_account"`
- Waits up to `equor_approval_timeout_s` (default 30s) for `EQUOR_PROVISIONING_APPROVAL`
- Default-allows on timeout (Equor may be initializing)
- Explicit DENY returns `ProvisioningStatus.EQUOR_DENIED`

**Wiring:**
- `IdentitySystem.set_account_provisioner(provisioner)` — called by registry
- `IdentitySystem.set_vault(vault)` — for credential sealing
- `IdentitySystem.set_full_config(config)` — for CaptchaConfig + AccountProvisionerConfig
- `app.state.account_provisioner` — accessible for direct API calls
- On `set_event_bus()`: forwards bus to provisioner (lazy wiring safe)
- On `set_neo4j()`: forwards neo4j to provisioner

**New SynapseEventTypes:**
- `PHONE_NUMBER_PROVISIONED` — Twilio number purchased
- `GITHUB_ACCOUNT_PROVISIONED` — GitHub account created
- `PLATFORM_ACCOUNT_PROVISIONED` — Generic platform account created
- `ACCOUNT_PROVISIONING_FAILED` — Provisioning attempt failed (retryable flag)

**Configuration (all with defaults):**
| Env var | Default | Purpose |
|---------|---------|---------|
| `ECODIAOS_ACCOUNT_PROVISIONER__ENABLED` | `true` | Enable/disable auto-provisioning |
| `ECODIAOS_ACCOUNT_PROVISIONER__GITHUB_USERNAME_PREFIX` | `ecodiaos` | Username prefix |
| `ECODIAOS_ACCOUNT_PROVISIONER__TWILIO_AREA_CODE` | `415` | US area code for number purchase |
| `ECODIAOS_ACCOUNT_PROVISIONER__TWILIO_NUMBER_COST_USD` | `1.15` | Monthly cost for Oikos accounting |
| `ECODIAOS_ACCOUNT_PROVISIONER__BROWSER_HEADLESS` | `true` | Headless Playwright |
| `ECODIAOS_ACCOUNT_PROVISIONER__BROWSER_STEALTH` | `true` | playwright-stealth patches |
| `ECODIAOS_ACCOUNT_PROVISIONER__OTP_WAIT_TIMEOUT_S` | `300` | OTP wait timeout |
| `ECODIAOS_ACCOUNT_PROVISIONER__EQUOR_APPROVAL_TIMEOUT_S` | `30` | Equor gate timeout |

**Cost accounting:** Every provisioning call emits `DOMAIN_EPISODE_RECORDED` with `domain: "account_provisioning"` and `cost_usd` for Oikos to track as infrastructure cost.

---

### CAPTCHA Client (`clients/captcha_client.py`) — NEW (2026-03-08)

**Class:** `CaptchaClient`

Async CAPTCHA solving client for 2captcha and Anti-Captcha providers.

**Methods:**
- `solve_recaptcha_v2(site_key, page_url, invisible)` → token
- `solve_recaptcha_v3(site_key, page_url, action, min_score)` → token
- `solve_hcaptcha(site_key, page_url)` → token
- `solve_image_captcha(image_base64)` → text
- `get_balance()` → float (USD remaining)

**Cost estimates:** reCAPTCHAv2/hCaptcha ~$0.002, v3 ~$0.003, image ~$0.001 per solve.
Each solve emits `DOMAIN_EPISODE_RECORDED` for cost tracking.

**Configuration:**
| Env var | Default | Purpose |
|---------|---------|---------|
| `ECODIAOS_CAPTCHA__TWOCAPTCHA_API_KEY` | `""` | 2captcha API key |
| `ECODIAOS_CAPTCHA__ANTICAPTCHA_API_KEY` | `""` | Anti-Captcha API key |
| `ECODIAOS_CAPTCHA__PROVIDER` | `2captcha` | Active provider |
| `ECODIAOS_CAPTCHA__POLLING_INTERVAL_S` | `5` | Poll interval |
| `ECODIAOS_CAPTCHA__MAX_WAIT_S` | `120` | Max wait before timeout |

---

### Browser Client (`clients/browser_client.py`) — NEW (2026-03-08)

**Class:** `BrowserClient` (async context manager)

Headless Playwright browser for form-based account creation.

**Key methods:**
- `create_page(stealth=True)` → Page (with random viewport, UA, stealth patches)
- `goto(page, url)` → PageLoadResult
- `fill_form(page, fields)` → FormFillResult (human-like typing delays + jitter)
- `click(page, selector)` → bool
- `solve_captcha_on_page(page, captcha_client)` → bool (auto-detects reCAPTCHA/hCaptcha)
- `wait_for_otp(platform, otp_coordinator, source, timeout_s)` → str
- `screenshot(page)` → bytes (PNG, for audit)
- `generate_password(length)` → str (static, cryptographically random)

**Dependencies:**
```bash
pip install playwright playwright-stealth
playwright install chromium
```

**playwright-stealth** is optional — graceful fallback if not installed (warning logged).

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
- `IDENTITY_CERTIFICATE_ROTATED` (CertificateManager → on every `install_certificate()` — Federation caches fingerprint for mTLS)
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

### Token Refresh Scheduler (`connector.py — TokenRefreshScheduler`) — NEW (2026-03-08)

- **`TokenRefreshScheduler`**: background coroutine that iterates over all registered `PlatformConnector` instances every `check_interval_seconds` (default 3600s)
- **Refresh window**: 24h ahead of expiry (`_REFRESH_AHEAD_S = 86_400`) — tokens within 24h of expiry are refreshed proactively
- **Lock safety**: uses same per-platform `asyncio.Lock` as `get_access_token()` — no races with inline refreshes
- **Events emitted**: `CONNECTOR_TOKEN_REFRESHED` on success; `CONNECTOR_TOKEN_EXPIRED` on failure with `source="proactive_scheduler"` field so Thymos can distinguish proactive vs. inline failures
- **Wiring**: `app.state.token_refresh_scheduler`; `supervised_task("token_refresh_scheduler")` in Phase 11 of `registry.py` (before interoception task)
- **`check_and_refresh_all()`**: public — callable directly in tests without waiting for sleep

### GitHub Connector (`connectors/github.py`) — UPDATED (2026-03-08)

- **`create_gist(files, description, public)`**: POST `/gists` — creates secret or public Gist; returns HTML URL
- **`get_active_github_token()`**: returns organism's own IAT (via `app_connector`) if available, else operator PAT from env; used by bounty/PR work to prefer own-account token

### Other Layers
- **Vault** (`vault.py`): Fernet encryption, PBKDF2 key derivation, key rotation + **Synapse event emission** (Identity #8 — 2026-03-07)
  - `VaultEvent` Pydantic model for structured payloads
  - `decrypt()` → emits `VAULT_DECRYPT_FAILED` (error_type: `key_mismatch` | `tampered`) on `InvalidToken`
  - `rotate_key()` → emits `VAULT_KEY_ROTATION_STARTED`, then `VAULT_KEY_ROTATION_COMPLETE` or `VAULT_KEY_ROTATION_FAILED`
  - `_fire_event()`: fire-and-forget via `asyncio.ensure_future`; never raises; no-op if event bus not wired
- **TOTP** (`totp.py`): RFC 6238, SHA-1/256/512
- **Communication** (`communication.py`): Twilio SMS webhook, **IMAPScanner** (scheduled via `supervised_task` in Phase 11), **OTPCoordinator** (2026-03-08)
- **Connectors**: Google (NEW — OAuth2 PKCE), GitHub, GitHubApp, X, LinkedIn, Instagram, Canva
- **CRUD** (`crud.py`): `sealed_envelopes` + `connector_credentials` tables; soft-delete enforced on both

### IMAPScanner (`communication.py`) — IMPLEMENTED (2026-03-08)

Background coroutine that polls an IMAP inbox for inbound OTP / verification codes, then routes them to `OTPCoordinator` and the Synapse bus.

**Key classes / functions:**
- `IMAPScanner` — wraps `_do_imap_scan()` in a supervised loop with configurable interval
  - `__init__(config, event_bus)` — reads `config.identity_comm` for IMAP settings
  - `run()` — exits immediately if `imap_host` is unset; otherwise loops: `sleep(interval_s)` → `_do_imap_scan()`
- `_do_imap_scan(comm, event_bus)` — private async core; runs `imaplib` blocking I/O via `asyncio.to_thread()`
  - Parses `UNSEEN` messages, extracts OTP digits (4–8 digit pattern), marks read
  - Emits `IDENTITY_VERIFICATION_RECEIVED` (for existing 2FA flow handlers)
  - Emits `EMAIL_OTP_RECEIVED` (for channel-specific filtering by OTPCoordinator)
- `scan_imap_inbox(cfg, event_bus)` — backward-compat delegate; calls `_do_imap_scan()`

**Configuration (via `IdentityCommConfig` / env vars):**
| Field | Env var | Default |
|-------|---------|---------|
| `imap_host` | `ECODIAOS_IDENTITY_COMM__IMAP_HOST` | `""` (scanner disabled if empty) |
| `imap_port` | `ECODIAOS_IDENTITY_COMM__IMAP_PORT` | `993` |
| `imap_username` | `ECODIAOS_IDENTITY_COMM__IMAP_USERNAME` | `""` |
| `imap_password` | `ECODIAOS_IDENTITY_COMM__IMAP_PASSWORD` | `""` |
| `imap_mailbox` | `ECODIAOS_IDENTITY_COMM__IMAP_MAILBOX` | `"INBOX"` |
| `imap_scan_interval_s` | `ECODIAOS_IDENTITY_COMM__IMAP_SCAN_INTERVAL_S` | `60.0` |

**Wiring:** `supervised_task("imap_scanner", restart=True, max_restarts=5)` in `core/registry.py` Phase 11 (before interoception loop). `app.state.imap_scanner` holds the instance.

---

### OTPCoordinator (`communication.py`) — NEW (2026-03-08)

Unified OTP coordination layer — receives codes from any channel and resolves pending authentication flows.

**Key classes:**
- `OTPCoordinator` — singleton held on `IdentitySystem._otp_coordinator`; accessed via `identity.otp_coordinator`
- `_PendingOTPFlow` — per-flow state: platform, expected_source, asyncio.Future, created_at, timeout_seconds
- `PLATFORM_HINTS` dict — 15 platforms → message text hints for extraction (case-insensitive)
- `extract_platform_hint(message)` — scans message text for known platform names; returns first match or None

**`wait_for_otp(platform, source="any", timeout=300)`:**
- Registers a pending flow keyed by platform name
- Replaces and cancels stale pending flow for the same platform
- Returns `await asyncio.wait_for(future, timeout)` — raises `asyncio.TimeoutError` on expiry

**`on_otp_received(platform_hint, code, source)`:**
- Tries platform_hint substring match against pending keys first
- Falls back to source-only match when hint is absent and flow accepts `"any"`
- Resolves future, deletes flow entry, emits `OTP_FLOW_RESOLVED` fire-and-forget

**Channel handlers (subscribed via `set_event_bus`):**
- `_on_sms_otp` → `IDENTITY_VERIFICATION_RECEIVED` — extracts hint from `raw_body`
- `_on_telegram_otp` → `TELEGRAM_OTP_RECEIVED` — prefers `platform_hint` field from I-1, falls back to `raw_text` extraction
- `_on_email_otp` → `EMAIL_OTP_RECEIVED` — prefers `platform_hint` field from I-2, falls back to `subject` extraction

**Wiring:**
- `OTPCoordinator` instantiated in `IdentitySystem.__init__()`
- `set_event_bus()` wires all 3 subscriptions — called from both `initialize()` and `set_event_bus()`
- `IdentitySystem.otp_coordinator` property exposes coordinator to callers (e.g. connector OAuth2 flows)

**Usage example (in an OAuth2 connector flow):**
```python
code = await identity.otp_coordinator.wait_for_otp(
    platform="github",
    source="email",   # GitHub sends codes via email
    timeout=300,
)
```

**New SynapseEventTypes:**
- `TELEGRAM_OTP_RECEIVED` — emitted by Telegram bot handler (I-1); payload: `platform_hint`, `code`, `sender_username`, `raw_text`
- `EMAIL_OTP_RECEIVED` — emitted by IMAP scanner (I-2); payload: `platform_hint`, `code`, `sender_address`, `subject`
- `OTP_FLOW_RESOLVED` — emitted by OTPCoordinator on resolution; payload: `platform`, `code`, `source`

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
| `communication.py` | Twilio SMS webhook, IMAPScanner (supervised_task), OTPCoordinator, Telegram webhook handler |
| `telegram_broadcast.py` | `telegram_status_broadcast_loop()` — 6-hour organism status to admin chat |
| `connectors/telegram.py` | `TelegramConnector` — bot token auth, vault storage, send/webhook helpers |
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

- ~~**[HIGH]** `AccountProvisioner` not yet wired in `core/registry.py`~~ — **FIXED (2026-03-09)**: Wired in Phase 11 of `registry.py` immediately after `imap_scanner` block. `set_account_provisioner()`, `set_vault()`, and `set_full_config()` all called. Stored as `app.state.account_provisioner`.
- **[MEDIUM]** `BrowserClient.solve_captcha_on_page()` injects token via JS eval — may break on SPAs that use React state for CAPTCHA responses. May need a page-reload or form re-submit trigger per site.
- **[LOW]** GitHub PAT generation relies on GitHub Settings UI selectors that may change. Monitor for selector breakage and update `_github_generate_pat()` accordingly.
- **[LOW]** Gmail provisioning (`provision_gmail`) may hit Google's phone verification requirement even after reCAPTCHA solve — the Twilio number must be fully provisioned before Gmail provisioning is attempted.
- **[HIGH]** Citizenship Tax: Oikos has no handler for `CERTIFICATE_RENEWAL_REQUESTED` — Identity emits it (`manager.py` line ~1016 `_handle_citizenship_tax_approved()` exists and is ready) but Oikos never processes the citizenship tax debit.
- ~~**[MEDIUM]** `evo.export_belief_genome()` + `simula.export_simula_genome()` not yet defined~~ — **FIXED**: Both methods are implemented (`evo/service.py:4047`, `simula/service.py:4076`).

### Resolved (2026-03-09, Telegram Full Capability)

- **TelegramCommandHandler** (`communication.py`) — subscribes to `TELEGRAM_MESSAGE_RECEIVED`; routes `/ping`, `/status`, `/help`, `/start` commands; replies via `TelegramConnector.send_message()`. `/status` pulls `synapse.metabolic_snapshot()` + `oikos.economic_state()`. Wired in registry Phase 11 as `app.state.telegram_cmd_handler`.
- **TelegramPollingLoop** (`communication.py`) — `getUpdates` long-polling fallback (30s timeout) started when `ECODIAOS_PUBLIC_URL` is not set; emits the same `TELEGRAM_MESSAGE_RECEIVED` + `TELEGRAM_OTP_RECEIVED` events as the webhook. `delete_webhook()` called first to clear any stale webhook. Supervised with `max_restarts=20`.
- **Registry wiring**: polling vs. webhook is automatic — `inbound_mode` log field confirms which path is active.
- **GitHubConnector PAT vault** (`connectors/github.py`) — `vault` param added to `__init__()`; env-var PAT encrypted via `vault.encrypt_token_json()` at construction time (Fernet, CPU-only). `authenticate()` method added — emits `CONNECTOR_AUTHENTICATED` with `auth_mode` (`app_iat` | `pat`) and `pat_sealed` flag. Called fire-and-forget via `asyncio.ensure_future` in `_init_github_connector`.
- **Registry `_init_github_connector`** — resolves `identity.vault` from `app.state.identity` and passes to `GitHubConnector`.

### Resolved (2026-03-08, Telegram Channel — Phase 16h)
- **TelegramConnector** (`connectors/telegram.py`) — bot token auth (not OAuth2); `authenticate()` via `getMe`, `revoke()` via `logOut`; token stored as `OAuthTokenSet(access_token=token, token_type="Bot", expires_in=0)` in vault; registered in `connectors/__init__.py`
- **Telegram webhook handler** (`communication.py`) — `POST /api/v1/identity/comm/telegram/webhook`; validates `X-Telegram-Bot-Api-Secret-Token` via `hmac.compare_digest()`; drops messages not from admin chat ID when set; emits `TELEGRAM_MESSAGE_RECEIVED` (all text) and `TELEGRAM_OTP_RECEIVED` (4-8 digit codes)
- **Telegram status broadcast** (`telegram_broadcast.py`) — `telegram_status_broadcast_loop()` coroutine; 6h default interval (`ECODIAOS_TELEGRAM_STATUS_INTERVAL_S`); reads `synapse.metabolic_snapshot()` + `oikos.economic_state()`; skips silently when `ADMIN_CHAT_ID` not set
- **Registry wiring** (`core/registry.py`) — Phase 11: boot connector, `authenticate()`, wire `SendTelegramExecutor`, `set_webhook(public_url)`, start `supervised_task("telegram_status_broadcast")`
- **New env vars**: `ECODIAOS_CONNECTORS__TELEGRAM__BOT_TOKEN`, `ECODIAOS_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID`, `ECODIAOS_TELEGRAM_WEBHOOK_SECRET`, `ECODIAOS_PUBLIC_URL`, `ECODIAOS_TELEGRAM_STATUS_INTERVAL_S`
- **New SynapseEventType**: `TELEGRAM_MESSAGE_RECEIVED` — inbound non-OTP text from Telegram webhook

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
- **[MEDIUM]** Oikos citizenship tax handler wired (2026-03-08): `OikosService._on_certificate_renewal_requested()` now subscribes to `CERTIFICATE_RENEWAL_REQUESTED`, gates via Equor (`mutation_type="citizenship_tax"`), debits `liquid_balance`, and emits `CERTIFICATE_RENEWAL_FUNDED` (new SynapseEventType). `EQUOR_HEALTH_REQUEST` handler also wired in Equor — GenesisCA now receives a live constitutional hash (drive vector + SHA-256) rather than always falling back to the document hash.
- **[MEDIUM]** Connector Evo signals (SG1): `_emit_re_training_example()` added to `PlatformConnector` base; wired into `exchange_code`, `refresh_token`, `revoke`, `check_health` for Google, Canva, Instagram connectors
- **[LOW]** `crud.py` hard deletes replaced with soft-deletes (`deleted_at = NOW()`) for both `sealed_envelopes` and `connector_credentials`

---

## Integration Points

**Emits (Synapse — self_model.py):** `SELF_MODEL_UPDATED`, `SELF_COHERENCE_ALARM`

**Emits (Synapse):** `IDENTITY_VERIFIED`, `IDENTITY_CHALLENGED`, `IDENTITY_EVOLVED`, `CONSTITUTIONAL_HASH_CHANGED`, `CERTIFICATE_RENEWED`, `CERTIFICATE_RENEWAL_REQUESTED`, `IDENTITY_DRIFT_DETECTED`, `CERTIFICATE_EXPIRING`, `CERTIFICATE_EXPIRED`, `CHILD_CERTIFICATE_INSTALLED`, `EQUOR_HEALTH_REQUEST`, `CERTIFICATE_PROVISIONING_REQUEST`, `PROVISIONING_REQUIRES_HUMAN_ESCALATION`, `GENOME_EXTRACT_RESPONSE`, `VAULT_DECRYPT_FAILED`, `VAULT_KEY_ROTATION_STARTED`, `VAULT_KEY_ROTATION_COMPLETE`, `VAULT_KEY_ROTATION_FAILED`, `CONNECTOR_AUTHENTICATED`, `CONNECTOR_TOKEN_REFRESHED`, `CONNECTOR_TOKEN_EXPIRED`, `CONNECTOR_REVOKED`, `CONNECTOR_ERROR`, `OTP_FLOW_RESOLVED`, `TELEGRAM_MESSAGE_RECEIVED` (inbound webhook — non-OTP text), `TELEGRAM_OTP_RECEIVED` (inbound webhook — OTP extracted)

**Gap closure (2026-03-07, event coverage):**
- `CONNECTOR_TOKEN_EXPIRED` — now emitted by `PlatformConnector.get_access_token()` when `refresh_token()` fails and the token can no longer be used.
- `CONNECTOR_ERROR` — now emitted by `PlatformConnector._emit_degraded()` alongside `SYSTEM_DEGRADED` when consecutive health check failures exceed threshold (3).
- All 5 connector lifecycle events (`CONNECTOR_AUTHENTICATED`, `CONNECTOR_TOKEN_REFRESHED`, `CONNECTOR_TOKEN_EXPIRED`, `CONNECTOR_REVOKED`, `CONNECTOR_ERROR`) are wired in the `PlatformConnector` base class `_emit_event()` type_map — connectors must have `set_event_bus(bus)` called on them to broadcast.

**Subscribes (Synapse):** `GENOME_EXTRACT_REQUEST`, `ORGANISM_SLEEP`, `ORGANISM_SPAWNED`, `CHILD_SPAWNED`, `EQUOR_PROVISIONING_APPROVAL`, `CHILD_CERTIFICATE_INSTALLED`, `EQUOR_HITL_APPROVED`, `EQUOR_ALIGNMENT_SCORE`, `IDENTITY_VERIFICATION_RECEIVED` (OTPCoordinator — SMS), `TELEGRAM_OTP_RECEIVED` (OTPCoordinator — I-1), `EMAIL_OTP_RECEIVED` (OTPCoordinator — I-2)

**Dependencies:** Federation (Ed25519 keypair via IdentityManager — TYPE_CHECKING import), Neo4j (Identity + Certificate node persistence), Synapse (event bus), IdentityVault (CA key sealing)

---

## Architecture Rules

- Constitutional hash must be deterministic and reproducible — always computed from the actual document
- No direct cross-system imports — event emission via Synapse, lazy imports for types
- Certificate signing operations require Ed25519 private key — never persist to disk
- Identity node in Neo4j is immutable for birth fields (birth_timestamp, parent_instance_id)
