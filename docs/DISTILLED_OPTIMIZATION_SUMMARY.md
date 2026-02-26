# EcodiaOS LLM Cost Optimization — Distilled Summary

## Executive Overview

**Goal:** Reduce LLM token costs by 60–70% without sacrificing capability.

**Current state:** 31 LLM calls across 16 EcodiaOS systems. Budget is too tight (60k tokens/hour → need 600k/hour for steady state).

**Solution:** Five core strategies implemented in Phase 1, ready for per-system integration (Phase 2–5).

---

## Five Core Strategies

### 1. **Token Budget System** (✅ Complete)
- **What:** Tracks cumulative tokens/hour and tokens/cycle
- **How:** Three-tier degradation: Green (normal) → Yellow (careful) → Red (critical)
- **File:** `ecodiaos/clients/token_budget.py`
- **Impact:** Graceful degradation when approaching limits

### 2. **Prompt Caching** (✅ Complete)
- **What:** Redis-backed semantic cache (key = SHA256(system + method + prompt))
- **Expected hit rate:** 30–80% depending on system
- **File:** `ecodiaos/clients/prompt_cache.py`
- **Impact:** 25–40% cost reduction from cache hits alone

### 3. **Output Validation** (✅ Complete)
- **What:** Parse JSON/numbers/enums from LLM responses with auto-correction
- **Key:** No retry loops on parse failure → fall back to heuristic immediately
- **File:** `ecodiaos/clients/output_validator.py`
- **Impact:** Eliminate token waste from malformed responses

### 4. **Heuristic Fallbacks** (✅ Complete)
- **What:** Fast (<10ms) approximations for common LLM tasks
- **Use case:** Yellow/Red budget tiers or validation failures
- **File:** `ecodiaos/systems/nova/efe_heuristics.py`
- **Impact:** 20–30% additional cost reduction

### 5. **Metrics & Telemetry** (✅ Complete)
- **What:** Track tokens, cost (USD), latency, cache hit rate per system
- **File:** `ecodiaos/telemetry/llm_metrics.py`
- **Endpoint:** `GET /metrics/llm` for dashboard data
- **Impact:** Visibility + auto-tuning potential

---

## Implementation Status

| Phase | What | Status | Impact |
|-------|------|--------|--------|
| **Phase 1** | Foundation (all 5 strategies) | ✅ Done | Ready for integration |
| **Phase 2** | Nova EFE integration | 🔄 Pending | 50–70% reduction in Nova |
| **Phase 3** | Voxis renderer integration | 🔄 Pending | 40–50% reduction in Voxis |
| **Phase 4** | System-wide rollout | 🔄 Pending | 60–70% total reduction |
| **Phase 5** | Auto-tuning (Evo weights) | 🔄 Pending | Continuous optimization |

---

## Quick Integration Pattern

All systems follow this 3-step pattern:

```python
# Step 1: Check budget
if not token_budget.can_use_llm(estimated_tokens=500):
    return await fallback_heuristic()

# Step 2: Try cache
cached = await cache.get("system", "method", prompt)
if cached:
    return cached

# Step 3: Call LLM, validate, cache
response = await llm.generate(prompt)
result = OutputValidator.extract_json(response.text)
if result is None:
    return await fallback_heuristic()

await cache.set("system", "method", prompt, result, ttl_seconds=300)
return result
```

---

## Budget Tiers & System Behavior

### Green Tier (0–70% usage)
- All systems use LLM
- Full capability

### Yellow Tier (70–90% usage)
- Low-priority systems degrade
- Nova: Use heuristics
- Voxis: Use templates
- Evo: Skip or use fast path
- Equor: **Always LLM** (critical)

### Red Tier (90–100% usage)
- Only critical systems active
- Everything else uses heuristics
- Equor still has precedence

---

## Cache TTL Guidelines

| System | TTL | Hit Rate Target |
|--------|-----|-----------------|
| Nova EFE | 5 min | 30% |
| Voxis Renderer | 1 min | 40% |
| Evo Hypothesis | 1 hour | 60% |
| Thread Identity | 6 hours | 70% |
| Equor Checks | 30 min | 10% (rare) |

---

## Configuration

### config/default.yaml

```yaml
llm:
  budget:
    max_tokens_per_hour: 600_000    # Was: 60,000 (too tight)
    max_calls_per_hour: 1_000
    hard_limit: false               # Graceful degradation

nova:
  efe_cache_ttl_s: 300

voxis:
  expression_cache_ttl_s: 60

evo:
  hypothesis_cache_ttl_s: 3600
```

---

## Monitoring

### Key Metrics

```
llm_tokens_charged          — Tokens used this hour
llm_cost_estimate           — USD cost estimate
llm_cache_hit_rate          — % calls from cache
llm_budget_tier             — Green/Yellow/Red
llm_latency_p99             — 99th percentile latency
```

### Dashboard Endpoint

```
GET /metrics/llm
→ Returns total cost, per-system breakdown, cache hit rate, budget tier
```

### Alert Rules

| Condition | Action |
|-----------|--------|
| Budget → Yellow | Warn: low-priority systems degrade |
| Budget → Red | Alert: critical-only mode |
| Cache hit < 10% | Adjust TTL or caching strategy |
| Latency p99 > 500ms | Investigate provider/network |

---

## Expected Outcomes

| Strategy | Individual | Combined |
|----------|-----------|----------|
| Caching | 25–40% | 60–70% |
| Heuristics | 20–30% | across all |
| Output validation | 10–15% | systems |

**Total expected reduction: 60–70% cost with same capability.**

---

## Common Integration Questions

### "Will heuristics produce worse decisions?"
No. Used only during:
1. Token budget exhaustion (graceful degradation)
2. Parse validation failures (fallback only)
3. Latency exceeds timeout (safety-critical)

In normal operation (Green tier, cache hits), LLM decisions are used.

### "How do I tune cache TTL?"
1. Monitor hit rate per system
2. If < target: reduce TTL (more misses → more accurate)
3. If >> target: increase TTL (fewer misses → faster)

### "What if I need precise decisions under budget constraints?"
- Configure `hard_limit=false` to allow overage
- Pre-allocate higher budget
- Equor (constitutional checks) always runs regardless of budget tier

---

## Files Reference

```
ecodiaos/
├── clients/
│   ├── token_budget.py          # Budget tracking
│   ├── prompt_cache.py          # Semantic cache
│   ├── output_validator.py      # Response validation
│   └── llm.py                   # Modified: budget integration
├── systems/nova/
│   └── efe_heuristics.py        # Fast fallbacks
├── telemetry/
│   └── llm_metrics.py           # Metrics collection
└── config.py                    # Modified: budget config

docs/
├── LLM_COST_OPTIMIZATION.md     # Strategy & framework
├── LLM_INTEGRATION_GUIDE.md     # Integration details
├── LLM_OPTIMIZATION_README.md   # Implementation summary
├── OPTIMIZATION_FLOW_DIAGRAM.md # Visual flows
└── DISTILLED_OPTIMIZATION_SUMMARY.md  # This file
```

---

## Next Steps

1. **Phase 2:** Integrate Nova EFE (start here)
   - Cache pragmatic/epistemic evaluations
   - Add budget checks before LLM calls
   - Expect 50–70% cost reduction in Nova

2. **Phase 3:** Integrate Voxis renderer
   - Cache expression generation
   - Template fallback in Red tier
   - Expect 40–50% cost reduction in Voxis

3. **Phase 4:** System-wide rollout
   - Integrate Evo, Equor, Thread, Oneiros
   - Tune cache TTLs per system
   - Enable monitoring dashboard

4. **Phase 5:** Auto-tuning
   - Evo learns EFE weights under budget constraints
   - Adaptive model tier selection
   - Cache TTL rebalancing

---

**Status:** Phase 1 foundation complete. Ready for system integration.
