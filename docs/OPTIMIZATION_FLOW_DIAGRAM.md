# LLM Cost Optimization — Flow Diagrams

## 1. Request Flow with Optimization

```
System (Nova, Voxis, etc) wants LLM response
    ↓
[Check Token Budget]
    ├─ Can use LLM? ──→ YES ─┐
    │                        │
    └─ Red/Yellow tier ─→ NO ┘
                              ↓
                    [Try Prompt Cache]
                              ↓
                    Cache hit? ──→ YES ──→ Return cached result
                              │
                              NO
                              ↓
                    [Call LLM Provider]
                    (charges budget)
                              ↓
                    [Validate Output]
                    (extract_json, etc)
                              ↓
                    Valid? ──→ YES ──→ [Store in Cache] ──→ Return result
                              │
                              NO
                              ↓
                    [Use Heuristic Fallback]
                              ↓
                         Return fast approximation
```

## 2. Budget Tier Decision Tree

```
                    System wants to use LLM
                              ↓
                    Check TokenBudget.get_status()
                              ↓
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
          GREEN          YELLOW            RED
        (0–70%)         (70–90%)        (90–100%)
              ↓               ↓               ↓
         All systems    Low-priority    Only critical
         use LLM       systems degrade  systems active
              ↓               ↓               ↓
        ┌─ Nova: LLM   ├─ Nova: Heuristic ├─ Nova: Heuristic
        ├─ Voxis: LLM  ├─ Voxis: Template ├─ Voxis: Template
        ├─ Evo: LLM    ├─ Evo: Skip/Fast  ├─ Evo: Skip/Fast
        ├─ Equor: LLM  ├─ Equor: LLM ✓    ├─ Equor: LLM ✓
        └─ Others: LLM └─ Others: Mixed   └─ Others: Heuristic
```

## 3. Token Budget Lifecycle

```
┌─────────────────────────────────────────────────────┐
│ Hour Begins: budget = 600,000 tokens                │
│ Cycle Period = 150ms                                │
│ ~4,000 cycles/hour in steady state                  │
└─────────────────────────────────────────────────────┘
                         ↓
            ┌────────────────────────┐
            │ Cycle 1: 150 tokens    │
            │ Remaining: 599,850     │ ──→ GREEN (0.025%)
            └────────────────────────┘
                         ↓
            ┌────────────────────────┐
            │ Cycle 100: 200 tokens  │
            │ Remaining: 420,000     │ ──→ GREEN (30%)
            └────────────────────────┘
                         ↓
            ┌────────────────────────┐
            │ Cycle 2,500: 300 tokens│
            │ Remaining: 180,000     │ ──→ YELLOW (70%)
            │                         │
            │ Log: "Budget tier      │
            │ YELLOW. Low-priority   │
            │ systems degrade."      │
            └────────────────────────┘
                         ↓
            ┌────────────────────────┐
            │ Cycle 3,000: 100 tokens│
            │ Remaining: 54,000      │ ──→ RED (91%)
            │                         │
            │ Log: "Budget tier RED. │
            │ Only critical systems  │
            │ active."               │
            └────────────────────────┘
                         ↓
            ┌────────────────────────┐
            │ Rest of hour: Heuristics│
            │ only, no LLM calls     │
            │ Graceful degradation ✓ │
            └────────────────────────┘
```

## 4. Prompt Cache Hit Scenario

```
┌─────────────────────────────────────────────────────┐
│ Voxis Renderer                                      │
│ Trigger: User asks "What are you thinking?"         │
│ Audience: Parent, morning context                   │
└─────────────────────────────────────────────────────┘
                         ↓
    Construct prompt:
    "System: You are Ecodia..."
    "Messages: [user query about thinking]"
    "Context: parent, morning, 8:30am"
                         ↓
    Hash: SHA256("voxis:render:" + prompt) = "a1b2c3..."
    Redis key: "eos:cache:voxis:render:a1b2c3"
                         ↓
    ┌──────────────────────────────────┐
    │ Redis lookup: GET "eos:cache..." │
    └──────────────────────────────────┘
                         ↓
              ┌──────────┴──────────┐
              ↓                     ↓
        VALUE FOUND!          Miss (first time)
              ↓                     ↓
    ┌─ Return cached      ┌─ Call LLM
    │   result immediately│   (charges budget)
    │   (0ms latency)    │
    │                     ├─ Get response
    │ Hit count: 1,534    │
    │ Hit rate: 87%  ✓   │   ├─ Validate JSON
    └─────────────────────┤   │   "outline": ["..."],
                          │   │   "tone": "warm"
                          │
                          ├─ Store in cache
                          │   SET "eos:cache..."
                          │   + value
                          │   + EX 60  (1 minute TTL)
                          │
                          └─ Return result
                              (200ms latency)

Result over 1 hour:
  - Cache hits:  1,537 (87%)
  - Cache misses:   225 (13%)
  - Token savings: 1,537 × 150 tokens = 230,550 tokens saved
  - Cost savings: $2.30+ (if 1M tokens = $15)
```

## 5. Output Validation Flow

```
LLM Response received:
"```json
{\\"score\\": 0.7, \\"reasoning\\": \\"Good...
```"
                ↓
        [Try JSON.parse()]
                ↓
        Parse failed (unclosed string)
                ↓
        [Extract JSON]
        Find { and } boundaries
        Extract: {"score": 0.7, "reasoning": "Good...
                ↓
        [Try JSON.parse() again]
                ↓
        Still failed (truncated)
                ↓
        [Auto-fix]
        ├─ Truncate after last }
        └─ Result: {"score": 0.7}
                ↓
        [Validate keys]
        Required: ["score", "reasoning"]
        Missing: ["reasoning"]
                ↓
        [Auto-correct with defaults]
        ├─ Add "reasoning": ""
        └─ Result: {"score": 0.7, "reasoning": ""}
                ↓
        ✅ Valid! Use this result.
           (No LLM retry needed!)
                ↓
        Cost saved: ~500 tokens (avoided retry)
```

## 6. Heuristic Fallback Cascade

```
Budget enters RED tier (>90% usage)
                ↓
Nova EFE Evaluator needs pragmatic value
                ↓
    [Check can_use_llm()]
    → False (budget exhausted)
                ↓
    [Call heuristic]
    EFEHeuristics.estimate_pragmatic_value_heuristic()
    ├─ Policy type: "express"
    ├─ Goal: "connect with child"
    ├─ Base score: 0.6 (for "express")
    ├─ Check for opposition: none found
    └─ Return: 0.6 (~0 tokens, <1ms)
                ↓
    Use 0.6 instead of LLM
    System continues operating ✓
                ↓
Result: Graceful degradation
  - Continue operating at reduced fidelity
  - No crash, no rejection
  - Fast approximations use known distributions
```

## 7. System Integration: Nova Example

```
┌──────────────────────────────────────────────────────────────────┐
│ Nova EFE Evaluator in GREEN tier                                 │
│ evaluate(policy, goal, beliefs, affect, drive_weights)           │
└──────────────────────────────────────────────────────────────────┘
                         ↓
        ┌─────────────────┴────────────────┐
        ↓                                  ↓
[Pragmatic Value]              [Epistemic Value]
        ↓                                  ↓
1. Check cache                 1. Check cache
   ├─ HIT? (80%) ─→ Return     │   ├─ HIT? (70%) ─→ Return
   │ cached (0ms)             │   │ cached (0ms)
   │                           │   │
   └─ MISS (20%)              └─ MISS (30%)
        ↓                                  ↓
2. Check budget                2. Check budget
   ├─ Can use? YES             │   ├─ Can use? YES
   │                           │   │
   └─ Call LLM                 └─ Call LLM
        ├─ Prompt: "Rate       │   ├─ Prompt: "Info gain
        │   likelihood of      │   │   if we observe X?"
        │   goal under policy" │   │
        ├─ Response: "0.7"     │   ├─ Response: "0.5"
        ├─ Validate: 0.7 ✓     │   ├─ Validate: 0.5 ✓
        ├─ Charge: 200 tokens  │   ├─ Charge: 180 tokens
        │                       │   │
        └─ Cache 5min           └─ Cache 5min
                ↓                                  ↓
        Pragmatic = 0.7                    Epistemic = 0.5
                                                  ↓
                        ┌───────────────────────┘
                        ↓
        [Compute EFE Score]
        ├─ Pragmatic: -0.35 (weighted)
        ├─ Epistemic: -0.10 (weighted)
        ├─ Constitutional: -0.18 (weighted)
        ├─ Feasibility: +0.05 (penalty for complex)
        └─ Risk: +0.02 (penalty)
                        ↓
        EFE = -(0.35 + 0.10 + 0.18 - 0.05 - 0.02)
            = -0.56
                        ↓
        Lower EFE = preferred policy ✓
        (Used 380 tokens, cached next identical
         prompt, 40% faster on second identical
         belief state)
```

## 8. Metrics Dashboard Data Flow

```
System makes LLM call
        ↓
LLM Provider charges budget
        ↓
record_llm_call(
    system="nova.efe",
    input_tokens=150,
    output_tokens=50,
    latency_ms=180,
    cache_hit=False
)
        ↓
┌───────────────────────────┐
│ LLMMetricsCollector       │
│                           │
│ nova.efe:                 │
│   calls: 1,234            │
│   tokens: 300,000         │
│   cost: $4.50             │
│   latency_avg: 180ms      │
│   cache_hits: 900         │
│   cache_hit_rate: 73%     │
│                           │
│ voxis.render:             │
│   calls: 5,600            │
│   tokens: 420,000         │
│   cost: $6.30             │
│   latency_avg: 200ms      │
│   cache_hits: 4,500       │
│   cache_hit_rate: 80%     │
│                           │
│ TOTAL:                    │
│   calls: 8,900            │
│   tokens: 720,000 (used)  │
│   cost: $10.80            │
│   cache_hit_rate: 77%     │
│   budget_tier: GREEN ✓    │
│   projected_cost: $14.20  │
│   (based on hour burn)    │
└───────────────────────────┘
        ↓
GET /metrics/llm
        ↓
Dashboard displays
┌────────────────────────────┐
│ 💚 GREEN TIER              │
│ Tokens: 720k / 600k (77%)  │
│ Cost: $10.80               │
│ Cache hit rate: 77% ✓✓✓    │
│ Avg latency: 190ms         │
│ Projected hourly: $14.20   │
│                            │
│ Top systems by cost:       │
│  1. voxis.render $6.30     │
│  2. nova.efe $4.50         │
│  3. equor.check $0.80      │
│                            │
│ Actions:                   │
│ • All systems active       │
│ • Continue monitoring      │
│ • (none needed)            │
└────────────────────────────┘
```

## 9. Phased Rollout Timeline

```
Week 1: Foundation (✅ DONE)
├─ Token budget system
├─ Prompt cache infrastructure
├─ Output validation
├─ Heuristic fallbacks
└─ Metrics collection

Week 2: Nova Integration (→ 50–70% reduction)
├─ EFE pragmatic evaluation caching
├─ EFE epistemic evaluation caching
├─ Heuristic fallback in Yellow/Red
├─ Measure 30% cache hit rate
└─ Projected: $X.XX/hour → $Y.YY/hour (70% savings)

Week 3: Voxis Integration (→ 40–50% reduction)
├─ Expression generation caching
├─ Template fallback
├─ Output validation
├─ Measure 40% cache hit rate
└─ System-wide: (Week1 + Week2 + Week3 = 50–60% reduction)

Week 4: System-Wide Rollout (→ 60–70% reduction)
├─ Evo hypothesis integration (70% cache)
├─ Thread synthesis integration (80% cache)
├─ Equor check tuning (identity check)
├─ Oneiros reflection (off-cycle consolidation)
└─ Final: 60–70% total cost reduction

Week 5: Auto-Tuning & Optimization
├─ Enable Evo weight learning under constraints
├─ Adaptive model tier selection
├─ Cache TTL auto-tuning
├─ Observability dashboards live
└─ Monitoring: continuous optimization
```

---

**Key Insight:** The system maintains full capability while dramatically reducing cost. Heuristics are used only when needed (budget constraints), and cache hits avoid LLM calls entirely.
