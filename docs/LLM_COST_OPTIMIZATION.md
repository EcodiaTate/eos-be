# EcodiaOS LLM Cost Optimization Framework

## Executive Summary

EcodiaOS makes 31 LLM calls across 16 systems. Without optimization, token costs grow linearly with cycle frequency. This framework implements:

1. **Token budgets** (soft/hard limits per cycle, per hour)
2. **Prompt caching** (Redis-backed, semantic deduplication)
3. **Adaptive model selection** (fast models for quick tasks, precise for high-stakes)
4. **Structured output enforcement** (prevent multi-attempt rewrites)
5. **Heuristic fallbacks** (fast non-LLM paths when safe)
6. **Telemetry & auto-tuning** (track cost/latency tradeoffs, learn weights)

---

## Current State Analysis

### LLM Call Hotspots

| System | Calls | Context | Cost Driver |
|--------|-------|---------|-------------|
| **Nova EFE** | 2 | Pragmatic + epistemic eval | Max policies × 5, every deliberation |
| **Voxis Renderer** | 1 | Expression generation | Every output utterance |
| **Evo Hypothesis** | 1 | Schema induction | Post-outcome learning |
| **Thread** (3×) | 3 | Identity synthesis, coherence | Periodic consolidation |
| **Equor Invariants** | 1 | Constitutional check | Critical intents only |
| **Thymos Diagnosis** | 1 | Affect diagnosis | Trigger-based |
| **Oneiros** (3×) | 3 | Sleep processing, reflection | Off-cycle (consolidated) |
| **Simula** | 1 | Code generation | Episodic |
| **Axon Observation** | 1 | Sensor interpretation | Per percept |

**Key observation:** Nova EFE evaluator dominates cost. Per cycle: 5 policies × 2 LLM calls = 10 calls/cycle at max load.

### Current Budgets (from config.py)

```
max_calls_per_hour: 60
max_tokens_per_hour: 60,000
```

This is **too tight** for active operation. With 150ms cycle period:
- Cycles per hour: 3600s ÷ 0.15s = 24,000
- If Nova uses 2 calls/cycle: 48,000 LLM calls/hour (80× budget)

---

## Optimization Strategies

### 1. Token Budget System

**Goal:** Graceful degradation when approaching limits. Three tiers:

- **Green** (0–70% of limit): All systems active, full LLM
- **Yellow** (70–90%): Disable low-priority LLM calls (Evo, Oneiros); use heuristics
- **Red** (90%–100%): Only critical systems (Equor, emergency); fallback to fast models

**Implementation:**
- `TokenBudget` class tracks cumulative tokens per hour/cycle
- Systems check `budget.can_use_llm(cost_estimate)` before calling
- Auto-tune via `evo/budget_learner.py` → adjust EFE weights away from expensive paths

### 2. Prompt Caching

**Goal:** Avoid re-evaluating identical prompts (semantic cache).

**Use cases:**
- Nova EFE: Same policy evaluated in similar belief states → same score
- Voxis: Repeated expression triggers in same audience context
- Evo/hypothesis: Same observation pattern → same schema induction

**Implementation:**
- `PromptCache` wraps Redis
- Key: `SHA256(system + method + prompt_hash)` (deterministic)
- TTL: configurable per system (Nova 300s, Voxis 60s, Evo 3600s)
- Hit rate target: >30% in active conversations

### 3. Adaptive Model Selection

**Goal:** Use cheaper, faster models for low-risk tasks; reserve precise models for critical decisions.

**Model tiers:**

| Tier | Model | Cost | Latency | Use Case |
|------|-------|------|---------|----------|
| **Fast** | claude-3.5-haiku | 0.8¢/M in | 50ms | EFE quick-path, Voxis outline |
| **Standard** | claude-3.5-sonnet | 3¢/M in | 200ms | Most deliberation, expression gen |
| **Precise** | claude-opus-4 | 15¢/M in | 500ms | Critical review (Equor), identity |

**Rules:**
- Nova (non-critical): Start with Haiku, escalate if EFE too uncertain
- Voxis: Use Haiku for brainstorm, Sonnet for final render
- Equor: Always Opus (constitutional checks are high-stakes)
- Evo: Haiku for hypothesis proposal, Sonnet for induction

### 4. Structured Output Enforcement

**Goal:** Prevent retry loops from malformed JSON/XML.

**Approach:**
- All LLM prompts include format spec (JSON Schema)
- Response validator with auto-correct (e.g., truncate overshooting tokens)
- If parse fails, DON'T retry → fallback to heuristic immediately

### 5. Heuristic Fallbacks

**Goal:** Fast, cheap approximations when LLM unavailable or budget exhausted.

| System | LLM Task | Fallback Heuristic | Cost |
|--------|----------|-------------------|------|
| **Nova EFE** | Pragmatic value | History-weighted probability | ~0 tokens |
| **Nova EFE** | Epistemic value | Belief entropy | ~0 tokens |
| **Voxis Renderer** | Expression | Template + substitution | ~0 tokens |
| **Evo Hypothesis** | Schema induction | Frequency-based pattern | ~0 tokens |
| **Equor Invariants** | Alignment check | Rule-based checklist | ~0 tokens |

---

## Implementation Roadmap

### Phase 1: Token Budget Tracking (Priority 1)

**Files to create/modify:**

1. **`ecodiaos/clients/token_budget.py`** (NEW)
   - `TokenBudget` class with per-cycle and per-hour tracking
   - Methods: `can_use_llm()`, `charge()`, `get_status()`
   - Emit telemetry events on threshold crossing

2. **`ecodiaos/clients/llm.py`** (MODIFY)
   - Wrap `_post_with_retry()` with budget check
   - Return token counts in all response types
   - Log `llm_call_charged` events with cost

3. **`ecodiaos/systems/nova/efe_evaluator.py`** (MODIFY)
   - Check budget before LLM calls
   - Fall back to heuristics in Yellow/Red tiers
   - Add `use_heuristic_fallback` param

4. **`ecodiaos/config.py`** (MODIFY)
   - Add `LLMBudgetTier` enum and tier thresholds
   - Expose `token_budget` on main config

### Phase 2: Prompt Caching (Priority 1)

**Files to create:**

1. **`ecodiaos/clients/prompt_cache.py`** (NEW)
   - `PromptCache` class with Redis backend
   - Key: `hashlib.sha256(system + method + prompt).hexdigest()`
   - Hit/miss telemetry

2. **System integrations** (in each system's cache-worthy method)
   - Nova EFE: cache pragmatic/epistemic evaluation results
   - Voxis: cache expression generation with same trigger + context
   - Evo: cache hypothesis proposals for repeated patterns

### Phase 3: Adaptive Model Selection (Priority 2)

**Files to create/modify:**

1. **`ecodiaos/clients/llm.py`** (MODIFY)
   - Add `model_tier` param to `generate()` / `evaluate()`
   - Route to appropriate provider based on tier
   - Fallback model selection logic

2. **`ecodiaos/systems/nova/efe_evaluator.py`** (MODIFY)
   - Use `model_tier="fast"` for quick-path
   - Use `model_tier="standard"` for slow-path
   - Escalate heuristic uncertainty to Sonnet if needed

3. **System-specific integration** (Voxis, Evo, Equor, etc.)

### Phase 4: Telemetry & Dashboarding (Priority 2)

**Files to create:**

1. **`ecodiaos/systems/telemetry/llm_metrics.py`** (NEW)
   - Aggregate token spend, cost, latency per system
   - Compute cache hit rate, model tier distribution
   - Forecast hourly spend based on current burn rate

2. **`ecodiaos/main.py`** (MODIFY)
   - Add `/metrics/llm` endpoint returning cost dashboard
   - Include tier status, budget utilization, forecast

---

## Tuning Guidelines

### Setting Realistic Budgets

**Measure live for 1 hour, then 10–20× + margin:**

```
Observed tokens/hour: 45,000
Safe budget: 600,000 tokens/hour (13× margin)
Safe calls: 1,000 calls/hour (similar margin)
```

**Why high margin?**
- Spikes from multi-turn conversations
- Evo hypothesis generation during learning
- Oneiros consolidation cycles

### Cache TTL Configuration

| System | TTL | Reasoning |
|--------|-----|-----------|
| Nova EFE | 5min | Beliefs change slower than percepts |
| Voxis | 1min | Personality/affect shift frequently |
| Evo | 1hour | Schema rarely changes mid-session |
| Thread | 6hour | Identity quite stable |

### Model Tier Escalation

If Nova's heuristic epistemic value has `confidence < 0.5`:
→ Re-evaluate with Sonnet (costs ~3¢ extra, but better decision)

If Voxis expression fails parse validation:
→ Use template fallback, don't retry with Opus

---

## Metrics & Monitoring

### Key Metrics

```
llm_tokens_charged (gauge)          — Total tokens used this hour
llm_calls_made (counter)             — Total LLM API calls
llm_cache_hit_rate (gauge)           — % of calls served from cache
llm_budget_tier (enum)               — Current tier (Green/Yellow/Red)
llm_model_tier_distribution (hist)   — Fast/Standard/Precise breakdown
llm_latency_p99 (gauge)              — 99th percentile LLM call latency
llm_cost_per_cycle (gauge)           — Average cost per cognitive cycle
```

### Alerting

- **Alert 1:** Budget → Yellow tier → warn via logs, consider slowing cycle
- **Alert 2:** Latency > 500ms → investigate provider or network
- **Alert 3:** Cache hit rate < 10% → adjust TTL or caching strategy

---

## Rollout Plan

1. **Week 1:** Token budget tracking (Phase 1) → measure baseline
2. **Week 2:** Prompt caching (Phase 2) → expect 25–40% cost reduction
3. **Week 3:** Adaptive model selection (Phase 3) → additional 20–30% reduction
4. **Week 4:** Telemetry dashboard (Phase 4) → visibility & auto-tuning

**Expected total cost reduction:** 60–70% with same capability.

---

## References

- `CLAUDE.md` § "Budgets" — Hard latency limits per system
- `config.py` — LLMConfig, LLMBudget
- `docs/Comprehensive Architecture Audit.txt` — Current LLM usage patterns
- Anthropic Prompt Caching API: https://docs.anthropic.com/en/docs/build-a-system/prompt-caching
