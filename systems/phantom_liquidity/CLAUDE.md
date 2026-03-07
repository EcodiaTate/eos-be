# Phantom Liquidity — System CLAUDE.md

**Spec**: `.claude/EcodiaOS_Spec_28_PhantomLiquidity.md`
**System ID**: `phantom_liquidity`
**Last Updated**: 2026-03-07 (v1.2 — all gaps closed)

---

## What's Implemented

### Core Modules

| Module | Status | Notes |
|---|---|---|
| `types.py` | ✅ Complete | `PoolHealth`, `PhantomLiquidityPool`, `PhantomPriceFeed`, `PoolSelectionCandidate` — all use `EOSBaseModel` |
| `pool_selector.py` | ✅ Complete | Static 5-pool curated list; `select_pools()` with TVL filter + budget cap; `compute_tick_range()` with ±80%/±50% spread; `get_static_pools()` public API |
| `price_listener.py` | ✅ Complete | `eth_getLogs` polling; `_decode_swap_data` (all 5 Swap event fields, signed int256/int24); `sqrt_price_x96_to_price` formula; graceful degradation |
| `executor.py` | ✅ Complete | `mint_position()` approve×2 + mint + receipt parse; `burn_position()` decreaseLiquidity → collect → burn; `_parse_mint_receipt` uses **full 32-byte** `IncreaseLiquidity` topic (fixed 2026-03-07) |
| `service.py` | ✅ Complete | Full lifecycle; `get_price()` staleness-aware; `get_price_with_fallback()` CoinGecko; `maintenance_cycle()` staleness + IL; Synapse emission; `get_candidates()` public API |

### Synapse Events

All 7 `PHANTOM_*` events implemented and registered in `synapse/types.py`:
- `PHANTOM_PRICE_UPDATE` — per Swap event
- `PHANTOM_POOL_STALE` — no swaps > `staleness_threshold_s`
- `PHANTOM_POSITION_CRITICAL` — IL > threshold
- `PHANTOM_IL_DETECTED` — IL change during maintenance
- `PHANTOM_FALLBACK_ACTIVATED` — CoinGecko fallback used
- `PHANTOM_RESOURCE_EXHAUSTED` — EMERGENCY/CRITICAL metabolic pressure
- `PHANTOM_METABOLIC_COST` — per maintenance cycle (gas, fees, net P&L)

### Synapse Subscriptions

- `METABOLIC_PRESSURE` → `_on_metabolic_pressure()` — AUSTERITY warns; EMERGENCY/CRITICAL emits `PHANTOM_RESOURCE_EXHAUSTED`
- `GENOME_EXTRACT_REQUEST` → `_on_genome_extract_request()` → `GENOME_EXTRACT_RESPONSE` with pool configs + thresholds

### Genome Extraction Protocol (Mitosis)

`extract_genome_segment()` / `seed_from_genome_segment()` implemented. Exports: pool addresses, pair configs, fee tiers, tick ranges, capital allocation, staleness/IL/poll thresholds + SHA-256 payload hash.

### Oikos Integration

Direct call (acceptable per Spec §9): `register_phantom_position()`, `update_phantom_position()`, `remove_phantom_position()`. Uses `YieldPosition` from `systems.oikos.models`.

### TimescaleDB Persistence

`write_phantom_price` → `phantom_price_history` table. `get_phantom_price_history` for REST API.

### REST API

12 endpoints in `api/routers/phantom_liquidity.py`, wired in `main.py`. Bonus: `/defillama-pools`, `/tick-range`.

---

## What's Missing (Known Gaps)

### MEDIUM

- **Two executors with no clear division**: `systems/phantom_liquidity/executor.py` (used by service/router) vs `axon/executors/phantom_liquidity.py` (Axon Intent-driven, currently zero runtime callers). Both use the correct NPM address. Not blocking but creates maintenance surface.

- **LP key rotation**: `store_lp_key()` seals the key at provisioning time but does not implement periodic rotation or re-sealing under a new vault key version. Key rotation would require calling `vault.rotate_key()` and updating `_lp_key_envelope`.

### LOW

- **RE training annotation** (Spec §23): Price context not attached to Nova/Oikos decision training examples. Price data is emitted via Synapse but not captured as RE episodes.

### Closed Gaps (2026-03-07)

- ~~Identity/wallet key management~~ — `store_lp_key()` / `retrieve_lp_key()` via `IdentityVault`. Never in config/env.
- ~~TimescaleDB → Memory bridge~~ — `_write_price_observation_to_neo4j()` writes `(:PriceObservation)` nodes per swap event.
- ~~Multi-instance price consensus~~ — `PHANTOM_PRICE_OBSERVATION` + `_compute_consensus_price()` (2σ median, 30s window).
- ~~Genome round-trip validation~~ — 4 pytest tests in `tests/unit/systems/phantom_liquidity/test_genome_roundtrip.py`.
- ~~Bedau-Packard contribution~~ — `PHANTOM_SUBSTRATE_OBSERVABLE` emitted per maintenance cycle.

---

## Architecture Notes

- **No cross-system imports at runtime** — all inter-system comms via Synapse events or direct Oikos method call (explicitly permitted in Spec §9)
- `from systems.oikos.models import YieldPosition` in `service.py:register_pool()` creates a cross-system type dependency — acceptable per spec but worth monitoring
- Router no longer imports private `_STATIC_POOLS` — uses `svc.get_candidates()` or `PoolSelector.get_static_pools()` (fixed 2026-03-07)
- `PoolHealth` comparisons in `pool_selector.py` now use enum values, not raw strings (fixed 2026-03-07)

---

## Integration Map

| System | Channel | Direction | What |
|---|---|---|---|
| Synapse | `PHANTOM_PRICE_UPDATE` | → emit | Every swap event; Atune/Nova/Oikos subscribe |
| Synapse | `PHANTOM_POOL_STALE` | → emit | Pool health degraded; Thymos may subscribe |
| Synapse | `PHANTOM_POSITION_CRITICAL` | → emit | IL > 2%; Thymos/Nova may subscribe |
| Synapse | `PHANTOM_RESOURCE_EXHAUSTED` | → emit | EMERGENCY metabolic pressure |
| Synapse | `PHANTOM_METABOLIC_COST` | → emit | Hourly cost/fee report |
| Synapse | `PHANTOM_PRICE_OBSERVATION` | → emit | Raw swap observation for fleet consensus |
| Synapse | `PHANTOM_SUBSTRATE_OBSERVABLE` | → emit | Bedau-Packard evolutionary metrics per maintenance cycle |
| Synapse | `METABOLIC_PRESSURE` | ← consume | Oikos sends; adjusts logging/emits exhaustion |
| Synapse | `GENOME_EXTRACT_REQUEST` | ← consume | Mitosis requests; responds with `GENOME_EXTRACT_RESPONSE` |
| Synapse | `PHANTOM_PRICE_OBSERVATION` | ← consume | Peer observations; aggregated for fleet consensus |
| Oikos | `register_phantom_position()` | → direct call | Position lifecycle tracking |
| TimescaleDB | `phantom_price_history` | → write | Per swap event persistence |
| Neo4j | `(:PriceObservation)` nodes | → write | Per swap event — Memory bridge for Kairos/Memory |
| IdentityVault | `store_lp_key()` / `retrieve_lp_key()` | → call | LP wallet key sealed at rest; decrypted per on-chain op |
