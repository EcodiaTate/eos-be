# Next Steps: Soma Follow-Up Work Phases 4-9

This document provides detailed next steps for completing the remaining consumer-side modifications.

## Quick Reference

| Phase | System | Status | Owner Task |
|-------|--------|--------|-----------|
| 4 | Nova | ⏳ TODO | Allostatic deliberation trigger on urgency > 0.7 |
| 5 | Memory | ⏳ TODO | Somatic markers on write, reranking on read |
| 6 | Evo | ⏳ TODO | Curiosity modulation, dynamics matrix update |
| 7 | Oneiros | ⏳ TODO | Sleep pressure from energy errors, REM counterfactuals |
| 8 | Thymos | ⏳ TODO | Constitutional gating based on integrity precision |
| 9 | Voxis | ⏳ TODO | Expression style modulation (arousal/valence) |

---

## Phase 4: Nova — Allostatic Deliberation Trigger

### File: `ecodiaos/systems/nova/service.py`

**Step 1**: Add Soma reference
```python
def __init__(self, ...):
    # ... existing code ...
    self._soma = None  # Soma for allostatic signal reading
```

**Step 2**: Add setter
```python
def set_soma(self, soma: Any) -> None:
    """Wire Soma service for allostatic urgency-based deliberation."""
    self._soma = soma
```

**Step 3**: In `handle_broadcast()` method (around line 258), before calling deliberation_engine:
```python
# Check for allostatic urgency
allostatic_mode = False
dominant_error_dim = None
if self._soma is not None:
    try:
        signal = self._soma.get_current_signal()
        if signal.urgency > self._soma.urgency_threshold:
            allostatic_mode = True
            dominant_error_dim = signal.dominant_error
    except Exception as exc:
        self._logger.debug("soma_urgency_check_error", error=str(exc))

intent, record = await self._deliberation_engine.deliberate(
    broadcast=broadcast,
    belief_state=self._belief_updater.beliefs,
    affect=broadcast.affect,
    belief_delta_is_conflicting=delta.involves_belief_conflict(),
    memory_traces=memory_traces,
    allostatic_mode=allostatic_mode,  # NEW
    allostatic_error_dim=dominant_error_dim,  # NEW
)
```

**Step 4**: Update DeliberationEngine.deliberate() signature
```python
async def deliberate(
    self,
    broadcast: WorkspaceBroadcast,
    belief_state: BeliefState,
    affect: AffectState,
    belief_delta_is_conflicting: bool = False,
    memory_traces: list[dict] | None = None,
    allostatic_mode: bool = False,  # NEW
    allostatic_error_dim: Any = None,  # NEW
) -> tuple[Intent | None, DecisionRecord]:
```

**Step 5**: In `_deliberate_inner()`, when allostatic_mode is True:
```python
if allostatic_mode and allostatic_error_dim is not None:
    # Reorder active goals to prioritize those addressing the dominant error
    # E.g., if dominant_error is ENERGY, prioritize recovery/rest goals
    # If dominant_error is SOCIAL_CHARGE, prioritize engagement goals
    self._reorder_goals_for_allostatic_mode(
        active_goals=priority_ctx.ordered_goals,
        error_dimension=allostatic_error_dim,
    )
```

**Step 6**: Wire in main.py (after `await nova.initialize()`)
```python
nova.set_soma(soma)
```

---

## Phase 5: Memory — Somatic Markers & Reranking

### Files: `ecodiaos/systems/memory/service.py`, `trace_writer.py`, `retrieval.py`

**Step 1**: Add Soma reference to MemoryService.__init__
```python
self._soma = None  # Soma for somatic markers
```

**Step 2**: Add setter
```python
def set_soma(self, soma: Any) -> None:
    """Wire Soma for somatic marker stamping and reranking."""
    self._soma = soma
```

**Step 3**: Pass Soma reference to trace_writer
```python
self._trace_writer = TraceWriter(..., soma_service=self._soma)
```

**Step 4**: In `trace_writer.py`, when writing traces:
```python
somatic_marker = None
if self._soma_service is not None:
    try:
        somatic_marker = self._soma_service.get_somatic_marker()
    except Exception as exc:
        logger.debug("soma_marker_error", error=str(exc))

trace = MemoryTrace(
    ...existing fields...,
    somatic_marker=somatic_marker,  # NEW
)
```

**Step 5**: In `retrieval.py`, after search returns candidates:
```python
async def search_relevant(self, query_vector, top_k=10):
    # Vector search
    candidates = await self._vector_search(query_vector, top_k)

    # Somatic reranking (if Soma available)
    if self._soma_service is not None:
        try:
            candidates = self._soma_service.somatic_rerank(candidates)
        except Exception as exc:
            logger.debug("somatic_rerank_error", error=str(exc))

    return candidates
```

**Step 6**: Wire in main.py
```python
memory.set_soma(soma)
```

---

## Phase 6: Evo — Curiosity Modulation & Dynamics Update

### File: `ecodiaos/systems/evo/service.py`

**Step 1**: Add Soma reference
```python
def __init__(self, ...):
    self._soma = None
```

**Step 2**: Add setter
```python
def set_soma(self, soma: Any) -> None:
    """Wire Soma for curiosity modulation and dynamics learning."""
    self._soma = soma
```

**Step 3**: In the main cycle loop, read curiosity_drive:
```python
curiosity_multiplier = 1.0  # Default
if self._soma is not None:
    try:
        signal = self._soma.get_current_signal()
        curiosity_drive = signal.state.sensed.get("curiosity_drive", 0.5)
        # Scale hypothesis generation by curiosity
        # 0.7 = normal, 1.0 = maximum exploration
        curiosity_multiplier = 0.5 + curiosity_drive * 1.0
    except Exception:
        pass

hypothesis_budget = int(self._base_hypothesis_count * curiosity_multiplier)
```

**Step 4**: When discovering systematic mis-predictions, update dynamics:
```python
# Evo detects that certain transitions are poorly predicted
if self._discovered_mis_prediction_pattern():
    new_dynamics = self._compute_updated_dynamics()

    if self._soma is not None:
        try:
            self._soma.update_dynamics_matrix(new_dynamics)
            logger.info("evo_dynamics_update", new_dynamics_dim=len(new_dynamics))
        except Exception as exc:
            logger.error("soma_dynamics_update_error", error=str(exc))
```

**Step 5**: Wire in main.py
```python
evo.set_soma(soma)
```

---

## Phase 7: Oneiros — Sleep Pressure & REM Counterfactuals

### File: `ecodiaos/systems/oneiros/service.py`

**Step 1**: Add Soma reference
```python
def __init__(self, ...):
    self._soma = None
```

**Step 2**: Add setter
```python
def set_soma(self, soma: Any) -> None:
    """Wire Soma for sleep pressure and counterfactual REM replay."""
    self._soma = soma
```

**Step 3**: Monitor energy errors for sleep pressure:
```python
async def run_cycle(self):
    sleep_pressure = self._compute_base_sleep_pressure()

    if self._soma is not None:
        try:
            signal = self._soma.get_current_signal()
            errors = signal.state.errors.get("immediate", {})
            energy_error = errors.get("energy", 0.0)

            # Large negative energy error = depleted → high sleep pressure
            if energy_error < -0.3:
                sleep_pressure += (abs(energy_error) * 0.5)
        except Exception:
            pass

    # Trigger sleep if sleep_pressure > threshold
    if sleep_pressure > 0.8:
        await self._initiate_sleep()
```

**Step 4**: During REM, call counterfactual engine:
```python
async def _run_rem_sleep(self):
    # ... existing REM setup ...

    for decision_record in self._near_miss_episodes:
        if self._soma is not None:
            try:
                counterfactual = await self._soma.generate_counterfactual(
                    decision_id=decision_record.id,
                    actual_trajectory=decision_record.trajectory,
                    alternative_description=f"Alternative to {decision_record.context}",
                    alternative_initial_impact={
                        "energy": decision_record.energy_impact * 1.2,
                        "valence": decision_record.valence_impact * 1.1,
                        # ... other dimensions ...
                    },
                    num_steps=10,
                )
                # Store counterfactual for learning
                await self._store_counterfactual_lesson(counterfactual)
            except Exception as exc:
                logger.debug("counterfactual_error", error=str(exc))
```

**Step 5**: Wire in main.py
```python
oneiros.set_soma(soma)
```

---

## Phase 8: Thymos — Constitutional Health Gating

### File: `ecodiaos/systems/thymos/service.py`

**Step 1**: Add Soma reference
```python
def __init__(self, ...):
    self._soma = None
```

**Step 2**: Add setter
```python
def set_soma(self, soma: Any) -> None:
    """Wire Soma for integrity-based constitutional health gating."""
    self._soma = soma
```

**Step 3**: When scoring risks, apply integrity gating:
```python
async def evaluate_risk(self, context):
    # Base risk score
    risk_score = self._compute_base_risk(context)

    # Apply integrity precision gating
    if self._soma is not None:
        try:
            signal = self._soma.get_current_signal()
            precision_weights = signal.precision_weights
            integrity_precision = precision_weights.get("integrity", 1.0)

            # If integrity is well-predicted (high precision), weight health signals more
            if integrity_precision > 0.7:
                constitutional_health_weight = 1.5  # Amplify constitutional concerns
            else:
                constitutional_health_weight = 1.0  # Normal weight

            # Re-score with health weighting
            risk_score = self._apply_constitutional_weighting(
                risk_score,
                weight=constitutional_health_weight
            )
        except Exception:
            pass

    return risk_score
```

**Step 4**: Wire in main.py
```python
thymos.set_soma(soma)
```

---

## Phase 9: Voxis — Somatic Expression Modulation

### File: `ecodiaos/systems/voxis/service.py`

**Step 1**: Add Soma reference
```python
def __init__(self, ...):
    self._soma = None
```

**Step 2**: Add setter
```python
def set_soma(self, soma: Any) -> None:
    """Wire Soma for somatic expression modulation."""
    self._soma = soma
```

**Step 3**: Modulate expression based on arousal/valence:
```python
async def express(self, intent):
    # Extract expression parameters
    arousal = 0.5  # Default
    valence = 0.0  # Default neutral

    if self._soma is not None:
        try:
            signal = self._soma.get_current_signal()
            state = signal.state
            arousal = state.sensed.get("arousal", 0.5)
            valence = state.sensed.get("valence", 0.0)
        except Exception:
            pass

    # Adjust tone and style based on arousal/valence
    expression_params = {
        "urgency_level": arousal,  # High arousal → urgent tone
        "hedging_level": 1.0 - max(0, valence),  # Low valence → more hedging
        "sentence_length": 10 if arousal > 0.7 else 20,  # High arousal = shorter
        "collaborativeness": 0.5 + valence * 0.5,  # High valence = collaborative
    }

    # Generate expression with adjusted parameters
    expression = await self._generate_expression(
        intent=intent,
        **expression_params
    )

    return expression
```

**Step 4**: Wire in main.py
```python
voxis.set_soma(soma)
```

---

## Integration Testing Strategy

### Unit Test Template (per system)

```python
@pytest.mark.asyncio
async def test_system_reads_soma_signal():
    """Verify system reads Soma signal safely."""
    system = SystemService()
    mock_soma = MagicMock()
    mock_signal = MagicMock()
    mock_signal.urgency = 0.8
    mock_soma.get_current_signal.return_value = mock_signal

    system.set_soma(mock_soma)

    # Run operation, verify signal was read
    await system.handle_cycle(test_input)
    mock_soma.get_current_signal.assert_called()

@pytest.mark.asyncio
async def test_system_graceful_fallback_without_soma():
    """Verify system works without Soma."""
    system = SystemService()
    # Don't wire Soma (set to None)

    # Should not raise
    await system.handle_cycle(test_input)
```

### Integration Test

```python
@pytest.mark.asyncio
async def test_full_theta_cycle_with_soma():
    """E2E: Verify Soma runs first, all consumers read signal."""
    # Create all systems with real dependencies
    soma = SomaService(config)
    await soma.initialize()

    synapse = SynapseService(atune, config)
    synapse.set_soma(soma)

    atune = AtuneService(embed_fn)
    atune.set_soma(soma)

    # ... wire other systems ...

    # Run cycle
    signal = await soma.run_cycle()
    broadcast = await atune.run_cycle(SystemLoad())

    # Verify signal was consumed
    assert signal.urgency >= 0.0
    assert broadcast is not None
```

---

## Verification Checklist

- [ ] All 6 systems have `set_soma()` method
- [ ] All systems pass Soma reference through __init__ or setter
- [ ] All systems wrap Soma reads in try/except + safe defaults
- [ ] All systems wire Soma in main.py after initialization
- [ ] DB migrations run successfully
- [ ] 1000+ cycle load test passes
- [ ] Telemetry: `interoceptive_state` table has > 0 rows
- [ ] Neo4j: Vector index query runs and completes < 100ms
- [ ] Logs: No Soma-related errors during normal operation
- [ ] Graceful fallback: Disable Soma, verify all systems still work

---

## Estimated Effort

- Phase 4 (Nova): ~2-3 hours (goal reordering logic)
- Phase 5 (Memory): ~2-3 hours (marker stamping + reranking)
- Phase 6 (Evo): ~1-2 hours (curiosity scaling)
- Phase 7 (Oneiros): ~2-3 hours (sleep pressure + counterfactuals)
- Phase 8 (Thymos): ~1 hour (gating logic)
- Phase 9 (Voxis): ~1-2 hours (tone modulation)

**Total**: ~9-14 hours of implementation + testing

---

## Support References

- **Soma Spec**: `.claude/15_soma_specification.md`
- **Integration Guide**: `SOMA_INTEGRATION_GUIDE.md`
- **Status Doc**: `SOMA_FOLLOW_UP_STATUS.md`
- **Project CLAUDE.md**: Root project instructions
