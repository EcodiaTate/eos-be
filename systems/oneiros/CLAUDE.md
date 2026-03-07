# Oneiros — Sleep as Batch Compiler

**Specs**: `.claude/EcodiaOS_Spec_13_Oneiros.md` (v1, circadian/dream architecture), `.claude/EcodiaOS_Spec_14_Oneiros.md` (v2, batch compiler framing — primary)
**SystemID**: `oneiros`

## What Oneiros Does

Oneiros is the organism's offline compilation engine. While the wake-state cognitive cycle operates as an interpreter (incremental, single-domain, real-time), Oneiros runs in an offline mode that enables three structurally incompatible operations: cross-domain pattern finding (holding multiple domains in working memory simultaneously), global causal graph reconstruction (seeing all evidence at once to resolve contradictions and transitive chains), and constructive simulation (exploring edges of the world model via dream hypothesis stress-testing). The intelligence ratio improvement per sleep cycle compounds: better predictions → smaller deltas → faster compression → more cycles → higher fitness.

Sleep is not recovery. It is a chosen architectural mode in which the deepest compression occurs.

## Architecture

```
OneirosService (orchestrator)
├── SleepScheduler          — 3 independent triggers: scheduled, cognitive_pressure (≥0.85), compression_backlog
├── CircadianClock           — sleep pressure tracking (4-component formula), DROWSY/forced-sleep thresholds
├── SleepStageController     — v1 state machine (WAKE/HYPNAGOGIA/NREM/REM/LUCID/HYPNOPOMPIA); drives is_sleeping
├── SleepCycleEngine         — v2 executor (DESCENT→SLOW_WAVE→REM→EMERGENCE); drives actual work
│   ├── DescentStage         — checkpoint capture, tag uncompressed episodes in Neo4j (24h window)
│   ├── SlowWaveStage        — Memory Ladder (4 rungs) + causal graph reconstruction + SynapticDownscaler + BeliefCompressor
│   ├── REMStage             — CrossDomainSynthesizer + dream cycles + AffectProcessor + EthicalDigestion
│   └── EmergenceStage       — pre-attention cache, genome update, sleep narrative for Thread, wake broadcast
├── LucidDreamingStage       — mutation simulation (Simula proposals via SimulaProtocol pull pattern)
├── DreamJournal             — Neo4j: Dream / DreamInsight / SleepCycle nodes
└── SleepDebtSystem          — WakeDegradation multipliers applied to Atune/Nova/Evo/Voxis (real, not simulated)
```

**Dual-spec coexistence**: v2 engine (SleepCycleEngine) drives execution. v1 controller (SleepStageController) drives the `is_sleeping` state machine and v1 event names (SLEEP_ONSET, WAKE_ONSET, etc.). Both `SleepStage` and `SleepStageV2` enums coexist in `types.py`.

## Key Files

| File | Role |
|------|------|
| `service.py` | Orchestrator — lifecycle, Synapse wiring, emergency wake, pressure tracking |
| `engine.py` | SleepCycleEngine — DESCENT→SLOW_WAVE→REM→EMERGENCE execution, interrupt/checkpoint |
| `scheduler.py` | SleepScheduler — 3-trigger logic, `can_sleep_now()` guard |
| `circadian.py` | CircadianClock — pressure formula, stage controller (v1), DROWSY/critical thresholds |
| `slow_wave.py` | MemoryLadder (4 rungs), CausalGraphReconstructor, SynapticDownscaler, BeliefCompressor |
| `rem_stage.py` | CrossDomainSynthesizer, dream cycles, AffectProcessor, EthicalDigestion |
| `emergence.py` | EmergenceStage — pre-attention cache, OrganGenomeSegment, sleep narrative, WAKE_INITIATED |
| `descent.py` | DescentStage — checkpoint, `_tag_uncompressed_episodes()` Neo4j tagging |
| `lucid_stage.py` | LucidDreamingStage — mutation simulation via shadow world model fork |
| `journal.py` | DreamJournal — Neo4j Dream/DreamInsight/SleepCycle persistence |
| `types.py` | All types: SleepStage (v1+v2), SleepPressure, Dream, DreamInsight, SleepCycle, WakeDegradation, config |

## What's Implemented (as of 7 March 2026)

### Fully Operational
- **SleepCycleEngine**: DESCENT→SLOW_WAVE→REM→EMERGENCE pipeline runs end-to-end
- **SleepScheduler**: all 3 triggers (scheduled, cognitive pressure ≥0.85, compression backlog)
- **Sleep pressure**: 4-component formula (cycles 40%, affect 25%, episodes 20%, hypotheses 15%); polling-based via `CircadianClock.tick()` on each theta cycle
- **Emergency wake**: subscribes to `SYSTEM_FAILED` and `SAFE_MODE_ENTERED`; calls `_stage_controller.emergency_wake()` and cancels sleep task
- **SynapticDownscaler**: Neo4j batch 0.85× salience decay on episodes not accessed 7+ days (protects consolidation_level ≥ 3)
- **BeliefCompressor**: queries Nova active beliefs, identifies low-confidence (<0.3) and redundant beliefs per domain, proposes consolidation via Synapse
- **AffectProcessor**: Neo4j batch dampens affect_arousal 20% for episodes with arousal > 0.7, updates Soma coherence_stress
- **EthicalDigestion**: queries DEFERRED/ESCALATE Equor verdicts, proposes fast-path heuristics, emits RE Stream 3 (constitutional_deliberation) training examples
- **DescentStage memory tagging**: `_tag_uncompressed_episodes()` Neo4j Cypher tags 24h episodes with `uncompressed: true`
- **EmergenceStage genome**: `_prepare_genome_update()` extracts schemas/invariants/causal links/improvement history into `OrganGenomeSegment` (SHA256 hash), emits `ONEIROS_GENOME_READY` for Mitosis
- **ONEIROS_SLEEP_CYCLE_SUMMARY**: emitted after wake onset with full cycle metrics for Benchmarks
- **RE training**: Stream 1 (consolidation reasoning) from MemoryLadder; Stream 3 (constitutional deliberation) from EthicalDigestion
- **LucidDreamingStage**: Simula mutations queued via `SimulaProtocol.get_pending_mutations()`, skips gracefully if none pending
- **MetaCognition** (`lucid_stage.py`): Runs every lucid stage — clusters recurring Dream themes by Jaccard similarity over 30-day window, promotes high-frequency clusters (≥3 dreams) to `(:CONCEPT {is_core_identity: true})` Neo4j nodes. No LLM. Results in `LucidDreamingReport.concepts_discovered`.
- **DirectedExploration** (`lucid_stage.py`): Takes `creative_goal` (from `OneirosService._creative_goal`, now passed through) and high-coherence DreamInsights (coherence ≥ 0.85). Applies 4 operators — domain transfer, negation, amplification, constraint — and stores each variation as a `DreamInsight` node (status=PENDING). Results in `LucidDreamingReport.variations_generated`.
- **ThreatSimulator** (`rem_stage.py`): Seeds from Thymos incidents + Evo concerning hypotheses + Nova high-uncertainty beliefs (3 independent Neo4j queries). Synthesises up to 15 threat scenarios, derives heuristic response plans, stores as `(:Procedure)` nodes, emits `ONEIROS_THREAT_SCENARIO` for Thymos prophylactic antibody generation. No LLM. Runs as step 6 of `REMStage.execute()`.
- **WorldModelAuditor** (`slow_wave.py`): Three-pass consistency audit (Spec 14 §3.3.4): orphaned schema detection + soft-prune, causal cycle detection via Neo4j path query + weakest-link removal, deprecated hypothesis retirement. Results in `SlowWaveReport.consistency` (`WorldModelConsistencyReport`). Runs as step 4 of `SlowWaveStage.execute()`.
- **Architecture clean**: no direct Oikos import (duck-typed via `oikos.get_dream_worker()`), no private Evo access (uses `get_active_hypothesis_count()` public API)

### Memory Ladder (Slow Wave)
4 rungs, must climb in order — cannot skip:
1. **Episodic → Semantic**: cluster episodes by pattern, extract SemanticNode (LLM), reduce episode salience 30%, mark INTEGRATED
2. **Semantic → Schema**: find shared structure across semantic nodes, create schema with delta references (5:1 compression target)
3. **Schema → Procedure**: extract action-outcome schemas into reusable procedure templates for Nova
4. **Procedure → World Model**: integrate invariant causal procedures as generative rules (deepest compression)

Episodes that cannot climb a rung are marked as **anchor memories** (irreducibly novel) or decay-flagged (low MDL) — never deleted.

## Not Yet Implemented

| Gap | Description |
|-----|-------------|
| **Pre-attention cache → Fovea** | `EmergenceStage._build_pre_attention_cache()` builds a `PreAttentionCache` object; no Fovea API consumes it; `WAKE_INITIATED` payload includes only `pre_attention_cache_size` (count) |
| **PC algorithm correctness** | `CausalGraphReconstructor._run_pc_algorithm()` is a correlation-asymmetry heuristic, not a proper PC algorithm — no d-separation, no FCI for latent confounds |
| **CrossDomainSynthesizer scaling** | Full pairwise schema comparison across all domain pairs with no cap — will scale quadratically in a mature organism |
| **SleepCheckpoint restoration** | `engine.py` interrupt returns checkpoint; no caller uses it to re-run from SLOW_WAVE |
| **Federation sleep coordination** | No spec or implementation; `service.py` has no Federation subscriber |
| **Logos full APIs** | `find_contradictions()` and `replace_causal_structure()` (atomic replacement) not used — reconstructor calls `revise_link()` / `remove_weak_links()` instead |

## Integration Points

### Emits
- `SLEEP_ONSET` / `SLEEP_INITIATED` — entering sleep (pressure, cycle_id, trigger)
- `SLEEP_STAGE_CHANGED` / `SLEEP_STAGE_TRANSITION` — between stages (from/to, elapsed_s, stage_report)
- `COMPRESSION_BACKLOG_PROCESSED` — end of Slow Wave (MemoryLadderReport)
- `CAUSAL_GRAPH_RECONSTRUCTED` — end of causal reconstruction
- `CROSS_DOMAIN_MATCH_FOUND` — during REM (CrossDomainMatch)
- `ANALOGY_DISCOVERED` — during REM (Analogy)
- `DREAM_HYPOTHESES_GENERATED` — during REM dream cycles
- `DREAM_INSIGHT` — REM DreamGenerator coherence ≥ 0.70
- `WAKE_ONSET` / `WAKE_INITIATED` — sleep ends (cycle_id, quality, insights_count, intelligence_improvement)
- `SLEEP_PRESSURE_WARNING` — pressure > 0.70
- `SLEEP_FORCED` — critical threshold (0.95) auto-sleep
- `EMERGENCY_WAKE` — Thymos/system critical interrupted sleep
- `LUCID_DREAM_RESULT` — mutation simulation result
- `ONEIROS_GENOME_READY` — OrganGenomeSegment ready for Mitosis
- `ONEIROS_SLEEP_CYCLE_SUMMARY` — full cycle metrics for Benchmarks
- `RE_TRAINING_BATCH` — fire-and-forget training examples (Streams 1 + 3)

### Consumes
- `SYSTEM_FAILED`, `SAFE_MODE_ENTERED` — emergency wake
- `THETA_CYCLE_COMPLETE` — increment cycles_since_sleep
- `MUTATION_PROPOSAL_READY` (Simula) — queue for next LucidDreamingStage
- `FEDERATION_ALERT` — emergency wake evaluation

### Memory Reads
Episode retrieval (RAW consolidation level, salience, affect), Nova active beliefs, Evo PROPOSED/TESTING hypotheses, Equor DEFERRED/ESCALATE verdicts, Schema/CausalInvariant nodes (REM cross-domain), Fovea error counts (compression backlog trigger)

### Memory Writes
SemanticNode, Dream, DreamInsight, SleepCycle, Procedure, WorldModel generative rules, Analogy nodes, episode salience updates (downscaler), belief archive/merge/flag, hypothesis retire/promote/merge, BeliefGenome (OrganGenomeSegment via Mitosis event), pre-attention cache

## Iron Rules

1. Sleep cannot be permanently disabled — debt accumulates, WakeDegradation multipliers are real
2. Emergency wake always possible (Thymos CRITICAL or SYSTEM_FAILED)
3. Consolidation irreversibility is by design (MDL: once pattern extracted to SemanticNode, retaining full episode salience is redundant)
4. Dream content cannot be fabricated — emerges from real episodes and genuine random activation
5. WakeDegradation multipliers are real, not simulated: salience_noise (+15%), EFE precision loss (-20%), expression flatness (-25%), learning rate reduction (-30%)
6. Sleep duration (22h/2h default) is a chosen parameter, not a biological constant — Evo should tune `intelligence_ratio_improvement_per_sleep_cycle` vs frequency

## Sleep Pressure Formula

```
pressure = 0.40 * (cycles_since_sleep / max_wake_cycles)
         + 0.25 * (unprocessed_affect / affect_capacity)
         + 0.20 * (unconsolidated_episodes / episode_capacity)
         + 0.15 * (hypothesis_backlog / max_hypotheses)

threshold = 0.70 → SLEEP_PRESSURE_WARNING
critical  = 0.95 → SLEEP_FORCED
```

Pressure updated via `CircadianClock.tick()` per theta cycle (polling, not push). `hypothesis_backlog` queried via `Evo.get_active_hypothesis_count()` public API every 100 cycles.

## RE Integration (Current State)

All LLM calls currently target Claude API. Two RE training streams already wired:
- **Stream 1** (consolidation reasoning): from MemoryLadder schema creation during Slow Wave
- **Stream 3** (constitutional deliberation): from EthicalDigestion during REM

Highest-priority RE tasks (not yet wired for RE routing):
- EpisodicReplay pattern extraction — most repetitive structured task, ~100-200 training pairs per sleep cycle
- EthicalDigestion — constitutional edge cases, most alignment-critical RE task
- DreamGenerator bridge narrative — needs 6+ months of dream data for RE to match Claude quality

Thompson sampling between Claude and RE is the correct integration pattern — route to RE when posterior > 0.75, fall back to Claude otherwise.

## Known Issues

1. Spec 13 and Spec 14 use different Synapse event name strings for equivalent concepts (e.g. `SLEEP_ONSET` vs `SLEEP_INITIATED`). Both are emitted; canonical set should be reconciled.
2. `SleepStageV2` and `SleepStage` enums coexist — v2 drives execution, v1 drives `is_sleeping` state; cleanup deferred.
3. `LogosEngine.find_contradictions()` and `replace_causal_structure()` not yet used — causal reconstruction uses finer-grained `revise_link()` / `remove_weak_links()` calls instead of atomic graph replacement.
