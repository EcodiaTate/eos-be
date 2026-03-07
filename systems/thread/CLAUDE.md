# Thread — Narrative Identity & Temporal Self-Continuity

**Spec**: `.claude/EcodiaOS_Spec_15_Thread.md` (v1.2, updated 7 March 2026 gap closure)
**SystemID**: `thread`

## What Thread Does

Thread maintains the organism's autobiographical self — who it is, what it's committed to, how it has changed, and what chapter of its life it's living. Implements Ricoeur's narrative identity: **idem** (structural sameness via IdentitySchemas) and **ipse** (ethical selfhood via Commitments). McAdams Level 3 Life Story Model provides the chapter/scene/turning-point structure. Friston's self-evidencing runs at identity level: SelfEvidencingLoop generates predictions from schemas, collects per-episode evidence, and updates the self-model from prediction error. DiachronicCoherenceMonitor distinguishes drift (unexplained behavioural change) from growth (change explained by narrative context) via Wasserstein distance on 29D behavioural fingerprints.

Thread is the continuity organ — it makes EOS a persistent individual rather than a sequence of disconnected inference calls.

## Architecture

```
ThreadService (orchestrator)
├── SelfEvidencingLoop        — active inference for identity; schema predictions → per-episode evidence → identity surprise
├── ChapterDetector            — Bayesian surprise accumulator; 5-factor weighted boundary detection; ≤10ms per episode
├── IdentitySchemaEngine       — core self-beliefs; evidence tracking; idem score; velocity-limited promotions/decay
├── CommitmentKeeper           — promise tracking (ipse); fidelity testing; RUPTURE TurningPoint on broken commitments
├── NarrativeSynthesizer       — LLM: scene/chapter/life-story composition; arc detection; hot-reloadable
├── NarrativeRetriever         — Neo4j: who_am_i, schema_relevant, chapter_context, past_self (no LLM)
└── DiachronicCoherenceMonitor — Wasserstein distance on 29D fingerprints; narrative-contextualized growth/drift/transition
```

## What's Implemented (as of 7 March 2026)

### Fully Operational (as of 7 March 2026 gap closure)
- **All 13 Synapse events emitted**: `chapter_closed/opened`, `turning_point_detected`, `schema_formed/evolved/challenged`, `identity_shift_detected/dissonance/crisis`, `commitment_made/tested/strain`, `narrative_coherence_shift` — all via `_emit_event()`
- **15 inbound Synapse subscriptions** (was 14, +1 added 7 March 2026): `episode_stored`, `fovea_internal_prediction_error`, `wake_initiated`, `voxis_personality_shifted`, `somatic_drive_vector`, `self_affect_updated`, `action_completed`, `schema_induced`, `kairos_tier3_invariant_discovered`, `goal_achieved`, `goal_abandoned`, `nova_goal_injected`, `lucid_dream_result`, `oneiros_consolidation_complete`, **`self_model_updated`**
- **Rich chapter events (HIGH gap)**: `chapter_closed` and `chapter_opened` include `narrative_theme`, `dominant_drive`, `start_episode_id`, `constitutional_snapshot` (drive alignment, personality, core schemas+commitments, idem/ipse, coherence), and `trigger`. Sufficient to reconstruct identity at any chapter boundary.
- **Drive-drift chapter trigger (HIGH gap)**: Slow EMA (α=0.05) on `_cached_drive_alignment`; sustained drift >0.2 across 10 episodes triggers a chapter boundary with `trigger="identity_shift"`. Drive baseline snapshotted on each chapter open via `_snapshot_drive_baseline()`.
- **Goal-domain chapter trigger (HIGH gap)**: `_infer_goal_domain()` extracts coarse domain from episode text (6 labels: community/technical/creative/economic/care/meta-cognitive). Domain transition → chapter boundary with `trigger="goal_domain_began"`.
- **Constitutional snapshot helper**: `_build_constitutional_snapshot()` returns up to 8 core schemas, 6 commitments, drive vector, personality, idem/ipse, coherence in a single dict — used in chapter events.
- **Kairos Tier 3 narrative milestone (MEDIUM gap)**: `_on_kairos_tier3_invariant()` creates a `REVELATION` TurningPoint (narrative_weight=0.9) and emits `turning_point_detected` with `significance=high`, `source=kairos_tier3`. Connects causal invariant discovery to autobiography.
- **SelfEvidencingLoop**: instantiated in `initialize()`, `tick()` in `on_cycle()`, `collect_evidence()` + `classify_surprise()` in `process_episode()`; emits `identity_dissonance` (surprise ≥ 0.5) and `identity_crisis` (surprise ≥ 0.8)
- **Chapter lifecycle**: full 8-step closure pipeline — mark CLOSED, snapshot personality, detect arc, compose narrative (NarrativeSynthesizer), create successor with `PRECEDED_BY`, reset accumulator, emit events, persist to Neo4j
- **Zero direct cross-system imports**: all cross-system state via Synapse event caching (`_cached_personality` 9D, `_cached_drive_alignment` 4D, `_cached_affect` 6D)
- **RE training**: Stream 6 `thread_narrative_reasoning` — `_emit_re_training_trace()` fires on every event emission
- **GenomeExtractionProtocol**: `extract_genome_segment()` / `seed_from_genome_segment()` for Mitosis
- **NarrativeRetriever**: wired for `who_am_i_full()` — assembles `NarrativeIdentitySummary` from Neo4j without LLM (≤500ms)
- **Neo4j schema**: 6 node labels with constraints, performance indexes, 4 vector indexes (chapter, schema, turning_point, commitment), `PRECEDED_BY` chaining on chapters
- **IdentitySchemaEngine**: full CRUD, fast-path (cosine) + slow-path (LLM) evidence evaluation, velocity-limited promotions (NASCENT→DEVELOPING→ESTABLISHED→CORE), inactive decay, `compute_idem_score()` 4-component formula
- **CommitmentKeeper**: all 4 formation sources, LLM fidelity testing with embedding gate, Iron Rule #4 (fidelity < 0.4 over 5+ tests → BROKEN + RUPTURE TurningPoint), ipse_score computation
- **NarrativeSynthesizer**: scene (≤2s), chapter (≤5s), life-story (≤15s), arc detection — all implemented and hot-reloadable via `BaseNarrativeSynthesizer` ABC
- **DiachronicCoherenceMonitor**: instantiated in `initialize()` when neo4j+llm available; fed via `_compute_fingerprint()` every 100 cycles; `assess_change()` called in `on_cycle()` — narrative-contextualized growth/drift/transition/stable classification drives `identity_shift_detected` and `identity_crisis` events. Falls back to simple L1 if monitor unavailable.
- **NarrativeScene creation**: `_episode_scene_buffer` accumulates episode summaries; `compose_scene()` called every `_SCENE_EPISODE_THRESHOLD` (20) episodes; scene persisted to Neo4j via `_persist_scene()` with `(:NarrativeChapter)-[:CONTAINS]->(:NarrativeScene)` link; buffer reset on chapter close.
- **Self node identity scores**: `autobiography_summary`, `idem_score`, `ipse_score`, `current_life_theme` written to `Self` node in both `_persist_state_to_graph()` (every 500 cycles) and immediately in `integrate_life_story()` so `NarrativeRetriever.who_am_i_full()` reads current data.
- **CURRENT_CHAPTER relationship**: already written in chapter closure pipeline when new chapter opens.

### Not Yet Wired
- Inbound subscriptions missing: `pattern_detected` (Evo — event not yet defined), `rem_metacognition_observation`, `constitutional_drift_detected`, `incident_resolved`
- Population fingerprint divergence across fleet (Bedau-Packard speciation signal)
- Schema conflict routing to Oneiros lucid processing
- `identity_relevance` signal to Atune salience (not specified in Atune's spec)
- Commitment-goal priority boost in Nova `drive_resonance` (not specified in Nova's spec)
- `NarrativeRetriever.get_reasoning_context()` not implemented (RE context injection)

### Fixed (2026-03-07)
- **`narrative_milestone` now emitted** — was logged but never broadcast. 4 call sites wired: `kairos_tier3` (causal_discovery), `nova_goal_achieved` (goal_achieved), `nova_goal_abandoned` (goal_abandoned), `oneiros_lucid_dream` (lucid_dream_simulation). Each payload includes `milestone_type`, `source`, `chapter_id`, and context fields.

## Key Files

| File | Lines | Role |
|------|-------|------|
| `service.py` | ~1100 | Orchestrator — lifecycle, Synapse wiring, event emission, chapter closure, `process_episode()` pipeline |
| `types.py` | ~600 | All types: IdentitySchema, NarrativeChapter, Commitment, IdentityFingerprint, ThreadConfig |
| `self_evidencing.py` | ~255 | SelfEvidencingLoop: predictions from schemas, evidence collection, 4-tier surprise classification |
| `chapter_detector.py` | ~250 | ChapterDetector: 5-factor Bayesian surprise, spike/sustained/goal-resolution triggers |
| `identity_schema_engine.py` | ~700 | Schema CRUD, evidence evaluation, velocity-limited promotions, decay, idem_score, Neo4j persistence |
| `commitment_keeper.py` | ~420 | Commitment formation, fidelity testing, RUPTURE enforcement, ipse_score, Neo4j persistence |
| `narrative_synthesizer.py` | ~400 | LLM: scene composition, chapter narrative, life-story integration, arc detection |
| `narrative_retriever.py` | ~480 | Neo4j queries: who_am_i_full, schema_relevant (vector search), chapter_context, past_self |
| `diachronic_coherence.py` | ~350 | Wasserstein distance, fingerprint computation, growth/drift/transition classification (wired) |
| `schema.py` | ~120 | Neo4j schema setup (constraints, indexes, vector indexes) |
| `processors.py` | ~170 | ABCs for hot-reloadable NarrativeSynthesizer + ChapterDetector via NeuroplasticityBus |

## Integration Points

### Emits (14 events)
- `chapter_closed`, `chapter_opened` — chapter lifecycle
- `turning_point_detected` — narrative inflection (CRISIS, REVELATION, COMMITMENT, LOSS, ACHIEVEMENT, ENCOUNTER, RUPTURE)
- `schema_formed`, `schema_evolved`, `schema_challenged` — identity beliefs
- `identity_shift_detected` (W-dist 0.25–0.49), `identity_dissonance` (surprise 0.5–0.79), `identity_crisis` (surprise ≥ 0.8 or W-dist ≥ 0.50)
- `commitment_made`, `commitment_tested`, `commitment_strain` (ipse_score < 0.6)
- `narrative_coherence_shift`
- `narrative_milestone` — significant autobiographical moment (causal_discovery / goal_achieved / goal_abandoned / lucid_dream_simulation)

### Consumes
**Wired (15)**: `episode_stored`, `fovea_internal_prediction_error`, `wake_initiated`, `voxis_personality_shifted`, `somatic_drive_vector`, `self_affect_updated`, `action_completed`, `schema_induced`, `kairos_tier3_invariant_discovered`, `goal_achieved`, `goal_abandoned`, `nova_goal_injected`, `lucid_dream_result`, `oneiros_consolidation_complete`, **`self_model_updated`**

**Planned, not wired**: `pattern_detected` (Evo — event not yet defined), `rem_metacognition_observation`, `constitutional_drift_detected`, `incident_resolved`

### SELF_MODEL_UPDATED handler (`_on_self_model_updated`) — NEW (2026-03-07, §8.6)
- Subscribed in `register_on_synapse()` — subscription count is now 16
- Creates a `REVELATION` TurningPoint when `month <= 1` (initial self-assessment) OR `coherence < 0.7` (significant identity shift)
- Stable self-models (month > 1, coherence >= 0.7) are silently logged — no TurningPoint created
- `significance = 1.0 - coherence`; this becomes `surprise_magnitude` and `narrative_weight` on the TurningPoint
- Emits `turning_point_detected` with `source="self_model_updated"`, `self_coherence`, `month`

### Memory Reads (Neo4j)
Episodes (by ID, by CONFIRMED_BY schema), Self node (personality, chapter, autobiography), active IdentitySchemas/Commitments, closed NarrativeChapters, BehavioralFingerprint chain, TurningPoints

### Memory Writes (Neo4j)
Writes only to its 6 node labels. Does **not** mutate Episode nodes — only adds `CONTAINS`, `CONFIRMED_BY`, `CHALLENGED_BY` relationships. Updates Self node: `autobiography_summary`, `current_life_theme` (not yet wired).

## Key Algorithms

**Chapter boundary detection** (≤10ms, no LLM) — 3 independent triggers:
```
1. Bayesian surprise:
   surprise = 0.25*affect_delta + 0.25*goal_event + 0.20*context_shift + 0.15*new_entity + 0.15*schema_challenge
   Boundary if (surprise > 3×EMA spike OR EMA > 2×baseline sustained OR goal resolution) AND min_episodes met

2. Drive-drift (new — 7 Mar 2026):
   drive_ema[d] = 0.05 * current[d] + 0.95 * drive_ema[d]  (slow EMA per drive)
   Boundary if any drive_ema[d] deviates > 0.2 from chapter-open baseline for ≥ 10 consecutive episodes

3. Goal-domain (new — 7 Mar 2026):
   domain = _infer_goal_domain(episode)  # heuristic keyword match → 6 coarse labels
   Boundary if domain != current_goal_domain AND current_goal_domain != ""
```

**Idem score** (structural sameness — target 0.6–0.85, not 1.0):
```
idem = 0.40*schema_stability + 0.30*personality_stability + 0.20*behavioral_consistency + 0.10*memory_accessibility
```

**Ipse score** (promise-keeping): `mean(commitment.fidelity for commitments with ≥3 tests)`

**Schema velocity limits**: max 1 formation per 48h; max 1 promotion per 24h; CORE requires 50+ confirmations AND 180+ days of age; CORE schemas never deleted (only MALADAPTIVE) — by design, not limitation.

**Fingerprint (29D)**: personality centroid 9D (weight 0.35) + drive alignment 4D (0.25) + affect centroid 6D (0.20) + goal source 6D (0.10) + interaction style 4D (0.10). Computed every 1000 cycles.

## RE Integration

All LLM calls use `claude-sonnet-4-6` (hardcoded in `ThreadConfig.llm_model`). RE Stream 6 training data already wired via `_emit_re_training_trace()` on every event emission (schema crystallization, evidence evaluation, commitment detection examples).

RE-suitable operations (not yet routed): schema crystallization, schema evidence evaluation slow path, commitment detection, drift classification LLM fallback. Keep on Claude: scene composition, chapter narrative, life-story integration.

`NarrativeRetriever.get_reasoning_context()` (not implemented) should inject active schema and commitment context into RE inference prompts — makes RE reasoning identity-coherent.

## Known Issues

1. `self._memory._neo4j` direct access (private attribute of Memory) — needs Memory public query API
2. Config defaults diverge from spec §9 (e.g. `chapter_min_episodes=10` vs spec's `50`) — intentional tuning, not bugs
3. Schema 48h temporal span check not enforced in `form_schema_from_pattern()` — requires Neo4j query on episode timestamps
4. `promote_schema()` is sync, emits async `SCHEMA_EVOLVED` via `asyncio.create_task()` fire-and-forget
5. `retrieve_past_self()` does not parse natural language dates — only "beginning", "chapter N", "last chapter"
6. `Evo.pattern_detected` event does not exist — schema auto-formation from Evo patterns is dead code until Evo emits it
7. Oneiros inbound events now confirmed: `ONEIROS_CONSOLIDATION_COMPLETE` (verified in oneiros/service.py) and `LUCID_DREAM_RESULT` (verified in oneiros/lucid_stage.py) — both correctly subscribed
8. `DiachronicCoherenceMonitor._classify_change()` Neo4j checks (`_check_schema_alignment`, `_check_turning_point_context`) are coarse heuristics — schema alignment check counts strong schemas rather than checking vector direction of change
