# EcodiaOS LLM Cost Optimization — Integration Guide

This guide shows how to integrate the four optimization tools into existing systems.

---

## Quick Start

### 1. Initialize Budget & Cache (in main.py or system initialization)

```python
from ecodiaos.clients.token_budget import TokenBudget
from ecodiaos.clients.prompt_cache import PromptCache
from ecodiaos.config import EcodiaOSConfig

# Load config
config = EcodiaOSConfig.from_yaml("config/default.yaml")

# Create token budget (from config)
token_budget = TokenBudget(
    max_tokens_per_hour=config.llm.budget.max_tokens_per_hour,
    max_calls_per_hour=config.llm.budget.max_calls_per_hour,
    hard_limit=config.llm.budget.hard_limit,
)

# Create LLM provider with budget
llm_provider = create_llm_provider(config.llm, token_budget=token_budget)

# Create prompt cache (requires redis_client)
prompt_cache = PromptCache(redis_client=redis_conn)
```

### 2. Use Budget Checks Before LLM Calls

```python
# Before calling LLM, check budget
if not token_budget.can_use_llm(estimated_tokens=500):
    # Fall back to heuristic
    logger.warning("LLM budget exhausted, using heuristic fallback")
    result = await my_heuristic_function()
else:
    # Make LLM call (budget is automatically charged)
    result = await llm_provider.generate(...)
```

### 3. Integrate Prompt Caching

```python
# Check cache first
cached_result = await prompt_cache.get(
    system="nova",
    method="pragmatic_value",
    prompt=my_prompt,
)

if cached_result:
    return cached_result

# Cache miss — call LLM
llm_response = await llm_provider.generate(...)
result = parse_response(llm_response)

# Store in cache for next time
await prompt_cache.set(
    system="nova",
    method="pragmatic_value",
    prompt=my_prompt,
    value=result,
    ttl_seconds=300,  # 5 minutes
)

return result
```

### 4. Use Output Validation

```python
from ecodiaos.clients.output_validator import OutputValidator

# Make LLM call
response = await llm_provider.generate(...)

# Extract JSON without retry
data = OutputValidator.extract_json(response.text)

if data is None:
    # Unfixable — use fallback, don't retry
    logger.warning("Output validation failed, using fallback")
    return await my_fallback_function()

# Auto-fix missing keys (no retry)
required = ["score", "reasoning"]
data = OutputValidator.auto_fix_dict(data, required)

return data
```

### 5. Monitor Metrics

```python
from ecodiaos.telemetry.llm_metrics import record_llm_call

# After each LLM call, record metrics
record_llm_call(
    system="nova.efe",
    input_tokens=response.input_tokens,
    output_tokens=response.output_tokens,
    latency_ms=elapsed_ms,
    cache_hit=False,  # or True if from cache
)

# Export metrics (e.g., in /metrics endpoint)
dashboard_data = get_collector().get_dashboard_data()
```

---

## System-Specific Integration

### Nova EFE Evaluator

**Current:** Two LLM calls per policy (pragmatic + epistemic)

**With optimization:**

```python
# nova/efe_evaluator.py

from ecodiaos.systems.nova.efe_heuristics import EFEHeuristics
from ecodiaos.clients.token_budget import BudgetTier

async def evaluate(self, policy, goal, beliefs, affect, drive_weights):
    budget_status = self._budget.get_status()

    # Get pragmatic value
    if budget_status.tier == BudgetTier.RED:
        # Use heuristic in Red tier
        pragmatic = EFEHeuristics.estimate_pragmatic_value_heuristic(
            policy, goal, beliefs
        )
    else:
        # Use cached LLM evaluation
        cached = await self._cache.get(
            system="nova",
            method="pragmatic_value",
            prompt=self._build_pragmatic_prompt(policy, goal, beliefs),
        )

        if cached:
            pragmatic = cached["score"]
        else:
            response = await self._llm.evaluate(
                prompt=self._build_pragmatic_prompt(...)
            )
            pragmatic = OutputValidator.extract_number(response.text)

            await self._cache.set(..., value={"score": pragmatic})

    # Similar for epistemic...

    return EFEScore(pragmatic, epistemic, ...)
```

**Expected savings:** 50–70% fewer LLM calls (cache hits) + 20–30% cost reduction (heuristic fallbacks in Yellow/Red tiers)

---

### Voxis Expression Renderer

**Current:** One LLM call per utterance

**With optimization:**

```python
# voxis/renderer.py

async def render(self, trigger, context):
    # Check cache first (same trigger + audience = same expression often)
    cached = await self._cache.get(
        system="voxis",
        method="render",
        prompt=self._build_expression_prompt(trigger, context),
    )

    if cached:
        return cached

    # Get budget tier
    if budget_status.tier == BudgetTier.RED:
        # Use template fallback
        return self._template_fallback(trigger, context)

    # Call LLM (charges budget)
    response = await self._llm.generate(...)

    # Validate output (no retry)
    expression = OutputValidator.extract_string_list(response.text)
    if not expression:
        return self._template_fallback(trigger, context)

    # Cache successful render
    await self._cache.set(
        system="voxis",
        method="render",
        prompt=...,
        value=expression,
        ttl_seconds=60,  # 1 minute
    )

    return expression
```

**Expected savings:** 40–50% cache hit rate + 10–15% token reduction (structured output validation)

---

### Evo Hypothesis Generation

**Current:** One LLM call per new pattern

**With optimization:**

```python
# evo/hypothesis.py

async def propose_hypothesis(self, observation):
    # Hypotheses are stable — long cache TTL
    cached = await self._cache.get(
        system="evo",
        method="hypothesis",
        prompt=self._build_observation_prompt(observation),
    )

    if cached:
        return cached

    # Skip LLM in Yellow/Red tiers for low-confidence observations
    if budget_status.tier != BudgetTier.GREEN and observation.confidence < 0.7:
        logger.info("Skipping LLM hypothesis in budget constraint")
        return None  # Caller uses heuristic

    response = await self._llm.generate(...)

    # Parse with validation (no retry)
    parsed = OutputValidator.extract_json(response.text)
    if not parsed:
        return None

    # Cache for 1 hour
    await self._cache.set(
        system="evo",
        method="hypothesis",
        prompt=...,
        value=parsed,
        ttl_seconds=3600,
    )

    return parsed
```

**Expected savings:** 70–80% cache hit rate (stable patterns) + cost reduction in Yellow/Red tiers

---

### Equor Constitutional Checks

**Current:** One LLM call per critical intent

**Note:** Equor is ALWAYS LLM (constitutional checks must be precise).
Cache only for identical intents (rare).

```python
# equor/invariants.py

async def check_invariant(self, intent):
    # Constitutional checks: never budget-limit, always precise
    # But cache identical intents for instant response

    cached = await self._cache.get(
        system="equor",
        method="invariant_check",
        prompt=self._build_check_prompt(intent),
    )

    if cached:
        return cached

    # Always call LLM (even in Red tier, Equor is critical)
    response = await self._llm.evaluate(...)

    result = OutputValidator.validate_enum(
        response.text,
        valid_values=["PERMIT", "MODIFY", "ESCALATE", "DENY"],
    )

    await self._cache.set(..., value={"result": result})

    return result
```

**Expected savings:** Minimal (cache misses common) + ensures safety

---

## Configuration

### config/default.yaml

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${ANTHROPIC_API_KEY}

  # Token budget (was too tight, now realistic)
  budget:
    max_tokens_per_hour: 600_000       # Supports ~4,000 cycles
    max_calls_per_hour: 1_000
    hard_limit: false                  # Graceful degradation

synapse:
  cycle_period_ms: 150

# Prompt cache TTLs (from prompt_cache.py)
nova:
  efe_cache_ttl_seconds: 300           # 5 minutes

voxis:
  expression_cache_ttl_seconds: 60     # 1 minute

evo:
  hypothesis_cache_ttl_seconds: 3600   # 1 hour

thread:
  synthesis_cache_ttl_seconds: 21600   # 6 hours
```

---

## Monitoring & Alerts

### Metrics Endpoint

Add to main.py:

```python
@app.get("/metrics/llm")
async def get_llm_metrics():
    """Return LLM cost dashboard."""
    from ecodiaos.telemetry.llm_metrics import get_collector

    collector = get_collector()
    return {
        "status": "ok",
        "dashboard": collector.get_dashboard_data(),
        "summary": collector.summary(),
    }
```

### Log Alerts

Watch for these log events:

- `budget_tier_yellow` — Warn, consider slowing cycle or increasing budget
- `budget_tier_red` — Alert, only critical systems active
- `cache_hit_rate < 0.1` — Warn, adjust cache TTL or caching strategy
- `llm_latency_p99 > 500ms` — Warn, investigate provider or network
- `output_validation_failed` — Warn, caller should use fallback

---

## Tuning Guidelines

### Setting Budget per Deployment

1. **Measure live for 1 hour:**
   - Note tokens/hour (e.g., 45,000)
   - Note calls/hour (e.g., 500)

2. **Set budget with 10–20× margin:**
   ```
   Tokens: 45,000 × 13 = 585,000
   Calls:  500 × 2 = 1,000
   ```

3. **Enable Yellow/Red tier alerts:**
   - Yellow at 70% warns early
   - Red at 90% triggers heuristic fallbacks

### Cache TTL Tuning

| System | Current TTL | Observation |
|--------|-------------|-------------|
| Nova EFE | 5min | If hit rate < 20%, reduce to 2min |
| Voxis | 1min | If < 30%, increase to 2min or check context variation |
| Evo | 1hour | If < 50%, observations too unique; reduce to 30min |
| Thread | 6hour | If < 70%, increase to 12hour (identity stable) |

---

## Common Patterns

### "Graceful Degradation" Pattern

```python
async def complex_decision(self):
    """Make a decision, degrade gracefully if budget constrained."""
    budget = self._budget.get_status()

    if budget.tier == BudgetTier.GREEN:
        # Full precision: try LLM, cache, validate
        return await self._full_llm_decision()

    elif budget.tier == BudgetTier.YELLOW:
        # Conservative: use cached LLM, heuristic fallback
        cached = await self._cache.get(...)
        return cached or await self._heuristic_decision()

    else:  # BudgetTier.RED
        # Crisis: heuristic only
        return await self._heuristic_decision()
```

### "Cache → Validate → Fallback" Pattern

```python
async def get_result(self, prompt):
    """Retrieve or compute result, with fallback."""
    # Step 1: Try cache
    cached = await self._cache.get("system", "method", prompt)
    if cached:
        return cached

    # Step 2: Try LLM (charges budget)
    if not self._budget.can_use_llm(estimate):
        return await self._fallback()

    response = await self._llm.generate(prompt)

    # Step 3: Validate (never retry on parse failure)
    result = OutputValidator.extract_json(response.text)
    if result is None:
        return await self._fallback()

    # Step 4: Cache and return
    await self._cache.set("system", "method", prompt, result)
    return result
```

---

## Testing

### Mock Token Budget

```python
@pytest.fixture
def token_budget():
    budget = TokenBudget(max_tokens_per_hour=10_000)
    yield budget
    budget.reset_window()

def test_graceful_degradation_under_budget(token_budget):
    # Simulate high usage
    token_budget.charge(tokens=9_500, system="test")

    # Next call should fail in hard-limit mode
    budget = TokenBudget(max_tokens_per_hour=10_000, hard_limit=True)
    budget.charge(tokens=9_500, system="test")

    assert not budget.can_use_llm(estimated_tokens=600)
    assert budget.get_status().tier == BudgetTier.RED
```

### Mock Prompt Cache

```python
@pytest.fixture
async def prompt_cache(mocker):
    mock_redis = AsyncMock()
    cache = PromptCache(mock_redis)
    return cache

@pytest.mark.asyncio
async def test_cache_hit(prompt_cache):
    prompt_cache._redis.get.return_value = b'{"value": "cached"}'

    result = await prompt_cache.get("nova", "test", "prompt")
    assert result == "cached"
    assert prompt_cache.get_hit_rate() > 0
```

---

## References

- `docs/LLM_COST_OPTIMIZATION.md` — Framework overview & strategy
- `ecodiaos/clients/token_budget.py` — Budget implementation
- `ecodiaos/clients/prompt_cache.py` — Cache implementation
- `ecodiaos/clients/output_validator.py` — Validation implementation
- `ecodiaos/telemetry/llm_metrics.py` — Metrics collection
- `ecodiaos/systems/nova/efe_heuristics.py` — Heuristic fallbacks
