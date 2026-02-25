"""
EcodiaOS — Nova Service

The executive function. Nova is where perception becomes intention.

Nova is the bridge between understanding the world (Atune + Memory) and
acting on it (Axon + Voxis). It receives workspace broadcasts, integrates
them with beliefs and goals, formulates possible courses of action, evaluates
them against Expected Free Energy, submits to Equor for constitutional review,
and issues Intents for execution.

Nova is not the boss. Equor can deny its Intents, Axon can fail them,
and the community can override them through governance. Nova proposes;
the organism disposes.

Lifecycle:
  initialize() — loads constitution and drive weights, builds sub-components
  receive_broadcast() — implements BroadcastSubscriber for Atune
  submit_intent() — external API for direct intent submission (test/governance)
  process_outcome() — feedback loop from execution
  shutdown() — graceful teardown
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.clients.llm import LLMProvider
from ecodiaos.primitives.affect import AffectState
from ecodiaos.primitives.intent import Intent
from ecodiaos.systems.atune.types import ActiveGoalSummary, WorkspaceBroadcast
from ecodiaos.systems.nova.belief_updater import BeliefUpdater
from ecodiaos.systems.nova.deliberation_engine import DeliberationEngine
from ecodiaos.systems.nova.efe_evaluator import EFEEvaluator, EFEWeights
from ecodiaos.systems.nova.goal_manager import GoalManager
from ecodiaos.systems.nova.intent_router import IntentRouter
from ecodiaos.systems.nova.policy_generator import PolicyGenerator
from ecodiaos.systems.nova.types import (
    DecisionRecord,
    Goal,
    GoalSource,
    GoalStatus,
    IntentOutcome,
    PendingIntent,
)
from ecodiaos.config import NovaConfig

if TYPE_CHECKING:
    from ecodiaos.systems.atune.service import AtuneService
    from ecodiaos.systems.axon.service import AxonService
    from ecodiaos.systems.equor.service import EquorService
    from ecodiaos.systems.memory.service import MemoryService
    from ecodiaos.systems.voxis.service import VoxisService

logger = structlog.get_logger()


class NovaService:
    """
    Decision & Planning system.

    Implements BroadcastSubscriber (Atune workspace protocol):
        system_id: str
        async def receive_broadcast(broadcast: WorkspaceBroadcast) -> None

    Dependencies:
        memory  — for constitution, self-model retrieval, procedure lookup
        equor   — for constitutional review of every Intent
        voxis   — for expression routing of approved Intents
        llm     — for policy generation and EFE estimation
        config  — NovaConfig
    """

    system_id: str = "nova"

    def __init__(
        self,
        memory: "MemoryService",
        equor: "EquorService",
        voxis: "VoxisService",
        llm: LLMProvider,
        config: NovaConfig,
    ) -> None:
        self._memory = memory
        self._equor = equor
        self._voxis = voxis
        self._llm = llm
        self._config = config
        self._logger = logger.bind(system="nova")

        # Instance metadata
        self._instance_name: str = "EOS"
        self._drive_weights: dict[str, float] = {
            "coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0
        }

        # Sub-components — built in initialize()
        self._belief_updater: BeliefUpdater = BeliefUpdater()
        self._goal_manager: GoalManager | None = None
        self._policy_generator: PolicyGenerator | None = None
        self._efe_evaluator: EFEEvaluator | None = None
        self._deliberation_engine: DeliberationEngine | None = None
        self._intent_router: IntentRouter | None = None

        # State
        self._pending_intents: dict[str, PendingIntent] = {}
        self._current_affect: AffectState = AffectState.neutral()
        self._current_conversation_id: str | None = None

        # Goal embedding cache: goal_id → embedding
        self._goal_embeddings: dict[str, list[float]] = {}
        self._embed_fn: Any = None  # Set via set_embed_fn()
        # Callback to push goal updates to Atune
        self._goal_sync_callback: Any = None  # Set via set_goal_sync_callback()

        # Soma for allostatic signal reading
        self._soma: Any = None

        # Rhythm-adaptive state (updated by Synapse event bus)
        self._rhythm_state: str = "normal"
        self._rhythm_drive_modulation: dict[str, float] = {}

        # Observability counters
        self._total_broadcasts: int = 0
        self._total_fast_path: int = 0
        self._total_slow_path: int = 0
        self._total_do_nothing: int = 0
        self._total_intents_issued: int = 0
        self._total_intents_approved: int = 0
        self._total_intents_blocked: int = 0
        self._total_outcomes_success: int = 0
        self._total_outcomes_failure: int = 0
        self._decision_records: list[DecisionRecord] = []
        self._max_decision_records: int = 100

    # ─── Lifecycle ────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Load constitution and drive weights from Memory.
        Build all sub-components.
        """
        self._logger.info("nova_initializing")

        # Load instance name and constitution
        self_node = await self._memory.get_self()
        if self_node is not None:
            self._instance_name = self_node.name

        constitution = await self._memory.get_constitution()
        if constitution and "drives" in constitution:
            drives = constitution["drives"]
            self._drive_weights = {
                "coherence": float(drives.get("coherence", 1.0)),
                "care": float(drives.get("care", 1.0)),
                "growth": float(drives.get("growth", 1.0)),
                "honesty": float(drives.get("honesty", 1.0)),
            }

        # Build sub-components
        self._goal_manager = GoalManager(
            max_active_goals=self._config.max_active_goals,
        )

        self._policy_generator = PolicyGenerator(
            llm=self._llm,
            instance_name=self._instance_name,
            max_policies=self._config.max_policies_per_deliberation,
            timeout_ms=self._config.slow_path_timeout_ms - 2000,  # Leave 2s for EFE + Equor
        )

        self._efe_evaluator = EFEEvaluator(
            llm=self._llm,
            weights=EFEWeights(
                pragmatic=self._config.efe_weight_pragmatic,
                epistemic=self._config.efe_weight_epistemic,
                constitutional=self._config.efe_weight_constitutional,
                feasibility=self._config.efe_weight_feasibility,
                risk=self._config.efe_weight_risk,
            ),
            use_llm_estimation=True,
        )

        self._intent_router = IntentRouter(voxis=self._voxis)

        self._deliberation_engine = DeliberationEngine(
            goal_manager=self._goal_manager,
            policy_generator=self._policy_generator,
            efe_evaluator=self._efe_evaluator,
            equor=self._equor,
            drive_weights=self._drive_weights,
            fast_path_timeout_ms=self._config.fast_path_timeout_ms,
            slow_path_timeout_ms=self._config.slow_path_timeout_ms,
        )

        self._logger.info(
            "nova_initialized",
            instance_name=self._instance_name,
            max_active_goals=self._config.max_active_goals,
            drive_weights=self._drive_weights,
        )

    def set_soma(self, soma: Any) -> None:
        """Wire Soma service for allostatic urgency-based deliberation."""
        self._soma = soma
        self._logger.info("soma_wired_to_nova")

    def set_axon(self, axon: "AxonService") -> None:
        """
        Wire Axon into Nova's intent router after both are initialised.
        This enables Step 5 of the cognitive cycle: ACT.
        """
        if self._intent_router is not None:
            self._intent_router.set_axon(axon)
            self._logger.info("axon_wired_to_nova")
        else:
            self._logger.warning("set_axon_called_before_initialize")

    @property
    def belief_state_reader(self) -> "_NovaBeliefStateReader":
        """
        Returns a BeliefStateReader adapter that Atune can use for
        top-down prediction. This closes the predictive processing loop:
        Nova beliefs → Atune prediction → prediction error → belief update.
        """
        return _NovaBeliefStateReader(self._belief_updater)

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._logger.info(
            "nova_shutdown",
            total_broadcasts=self._total_broadcasts,
            total_intents_issued=self._total_intents_issued,
            active_goals=len(self._goal_manager.active_goals) if self._goal_manager else 0,
        )

    # ─── BroadcastSubscriber Interface ───────────────────────────

    async def receive_broadcast(self, broadcast: WorkspaceBroadcast) -> None:
        """
        Called by Atune when the workspace broadcasts a percept.

        This is the primary cycle entry point. Nova updates beliefs,
        deliberates, and dispatches an Intent — or chooses silence.

        The full pipeline must complete in ≤5000ms (slow path budget).
        Fast path targets ≤150ms.
        """
        if self._deliberation_engine is None:
            return  # Not yet initialized

        self._total_broadcasts += 1
        self._current_affect = broadcast.affect

        # ── Belief update (≤50ms) ──
        delta = self._belief_updater.update_from_broadcast(broadcast)

        # ── Retrieve relevant memories (best-effort, non-blocking) ──
        memory_traces = await self._retrieve_relevant_memories_safe(broadcast)

        # ── Check for allostatic urgency (soma read from cache, <1ms) ──
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

        # ── Deliberate (≤5000ms total) ──
        intent, record = await self._deliberation_engine.deliberate(
            broadcast=broadcast,
            belief_state=self._belief_updater.beliefs,
            affect=broadcast.affect,
            belief_delta_is_conflicting=delta.involves_belief_conflict(),
            memory_traces=memory_traces,
            allostatic_mode=allostatic_mode,
            allostatic_error_dim=dominant_error_dim,
        )

        # ── Update observability ──
        self._record_decision(record)
        if record.path == "fast":
            self._total_fast_path += 1
        elif record.path == "slow":
            self._total_slow_path += 1
        else:
            self._total_do_nothing += 1

        # ── Route intent if one was produced ──
        if intent is not None:
            self._total_intents_issued += 1
            await self._dispatch_intent(intent, broadcast)

        # ── Update Atune with active goal summaries ──
        # (Done asynchronously so it doesn't block the cycle)
        if self._goal_manager:
            asyncio.create_task(
                self._sync_goals_to_atune_safe(),
                name=f"nova_goal_sync_{broadcast.broadcast_id[:8]}",
            )

        # ── Coherence repair: high stress → generate an epistemic goal ──
        if (
            broadcast.affect.coherence_stress > 0.7
            and self._goal_manager
            and not self._has_active_coherence_goal()
        ):
            self._create_coherence_repair_goal(broadcast)

        # ── Goal maintenance (every 100 broadcasts) ──
        if self._total_broadcasts % 100 == 0 and self._goal_manager:
            self._goal_manager.expire_stale_goals()
            self._goal_manager.prune_retired_goals()

        # ── Decay unobserved entity beliefs (background maintenance) ──
        self._belief_updater.decay_unobserved_entities()

    # ─── External API ─────────────────────────────────────────────

    async def add_goal(self, goal: Goal) -> Goal:
        """Add a goal directly (called by governance or test harness)."""
        assert self._goal_manager is not None
        result = self._goal_manager.add_goal(goal)
        # Embed the goal description for salience-guided attention
        asyncio.create_task(
            self._embed_goal(result),
            name=f"nova_embed_goal_{result.id[:8]}",
        )
        return result

    async def process_outcome(self, outcome: IntentOutcome) -> None:
        """
        An intent has completed. Update beliefs and goal progress.
        Called by Axon (or Voxis feedback loop) when execution completes.
        """
        pending = self._pending_intents.pop(outcome.intent_id, None)

        if outcome.success:
            self._total_outcomes_success += 1
            self._belief_updater.update_from_outcome(
                outcome_description=outcome.episode_id,
                success=True,
            )
            # Update affect — success feels good
            if self._current_affect:
                new_valence = min(1.0, self._current_affect.valence + 0.05)
                self._current_affect = self._current_affect.model_copy(
                    update={"valence": new_valence}
                )
        else:
            self._total_outcomes_failure += 1
            self._belief_updater.update_from_outcome(
                outcome_description=outcome.failure_reason,
                success=False,
            )
            # Record bad outcome embedding for Risk head's threat detection
            # If we have the goal embedding, record it so similar future
            # percepts trigger heightened risk awareness.
            if pending:
                goal_emb = self._goal_embeddings.get(pending.goal_id)
                if goal_emb:
                    from ecodiaos.systems.atune.salience import RiskHead
                    RiskHead.record_bad_outcome(goal_emb)

        # Update goal progress if we know which goal this intent served
        if pending and self._goal_manager:
            goal = self._goal_manager.get_goal(pending.goal_id)
            if goal:
                progress_delta = 0.3 if outcome.success else 0.0
                new_progress = min(1.0, goal.progress + progress_delta)
                self._goal_manager.update_progress(
                    goal.id,
                    progress=new_progress,
                    episode_id=outcome.episode_id,
                )

        self._logger.info(
            "outcome_processed",
            intent_id=outcome.intent_id,
            success=outcome.success,
        )

    def set_conversation_id(self, conversation_id: str | None) -> None:
        """Set the active conversation ID for intent routing."""
        self._current_conversation_id = conversation_id

    # ─── Evo Interface ────────────────────────────────────────────

    def update_efe_weights(self, new_weights: dict[str, float]) -> None:
        """
        Called by Evo after learning that certain EFE components
        predict outcomes better. Adjusts the EFE weight vector.
        """
        assert self._efe_evaluator is not None
        current = self._efe_evaluator.weights
        updated = EFEWeights(
            pragmatic=new_weights.get("pragmatic", current.pragmatic),
            epistemic=new_weights.get("epistemic", current.epistemic),
            constitutional=new_weights.get("constitutional", current.constitutional),
            feasibility=new_weights.get("feasibility", current.feasibility),
            risk=new_weights.get("risk", current.risk),
        )
        self._efe_evaluator.update_weights(updated)
        self._logger.info("efe_weights_updated_by_evo", weights=new_weights)

    # ─── Observability ────────────────────────────────────────────

    async def health(self) -> dict:
        """Health check — returns current metrics snapshot."""
        total_decisions = (
            self._total_fast_path + self._total_slow_path + self._total_do_nothing
        )
        vfe = self._belief_updater.beliefs.free_energy

        goal_stats = {}
        if self._goal_manager:
            goal_stats = self._goal_manager.stats()

        router_stats = {}
        if self._intent_router:
            router_stats = self._intent_router.stats

        return {
            "status": "healthy",
            "instance_name": self._instance_name,
            "total_broadcasts": self._total_broadcasts,
            "total_decisions": total_decisions,
            "fast_path_decisions": self._total_fast_path,
            "slow_path_decisions": self._total_slow_path,
            "do_nothing_decisions": self._total_do_nothing,
            "intents_issued": self._total_intents_issued,
            "outcomes_success": self._total_outcomes_success,
            "outcomes_failure": self._total_outcomes_failure,
            "belief_free_energy": round(vfe, 4),
            "belief_confidence": round(self._belief_updater.beliefs.overall_confidence, 4),
            "entity_count": len(self._belief_updater.beliefs.entities),
            "goals": goal_stats,
            "routing": router_stats,
            "drive_weights": self._drive_weights,
        }

    def get_recent_decisions(self, limit: int = 20) -> list[DecisionRecord]:
        """Return recent decision records for observability and Evo learning."""
        return list(reversed(self._decision_records[-limit:]))

    def set_embed_fn(self, embed_fn: Any) -> None:
        """Wire the embedding function for goal embedding generation."""
        self._embed_fn = embed_fn

    def set_goal_sync_callback(self, callback: Any) -> None:
        """Wire a callback that pushes active goal summaries to Atune."""
        self._goal_sync_callback = callback

    async def on_rhythm_change(self, event: Any) -> None:
        """
        Synapse event bus callback: adapt drive weights when rhythm changes.

        Rhythm modulation naturally shifts goal priorities by altering the
        drive_resonance component of priority computation:
          - STRESS: boost coherence (focus on what matters, shed low-priority)
          - FLOW: boost growth (extend creative focus, don't context-switch)
          - BOREDOM: boost growth + care (seek novelty, help others)
          - DEEP_PROCESSING: boost coherence strongly (lock focus)
        """
        try:
            new_state = event.data.get("to", "normal")
            old_state = self._rhythm_state
            if new_state == old_state:
                return
            self._rhythm_state = new_state

            # Drive weight modulation per rhythm state
            modulations = {
                "stress": {"coherence": 1.4, "care": 0.8, "growth": 0.6, "honesty": 1.0},
                "flow": {"coherence": 1.0, "care": 0.9, "growth": 1.4, "honesty": 1.0},
                "boredom": {"coherence": 0.8, "care": 1.2, "growth": 1.3, "honesty": 1.0},
                "deep_processing": {
                    "coherence": 1.5, "care": 0.7, "growth": 0.8, "honesty": 1.0,
                },
                "idle": {"coherence": 1.0, "care": 1.1, "growth": 1.1, "honesty": 1.0},
            }
            self._rhythm_drive_modulation = modulations.get(new_state, {})

            # Apply modulation to the deliberation engine's drive weights
            if self._deliberation_engine is not None and self._rhythm_drive_modulation:
                base = {"coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0}
                modulated = {
                    k: base[k] * self._rhythm_drive_modulation.get(k, 1.0)
                    for k in base
                }
                self._deliberation_engine.update_drive_weights(modulated)

            self._logger.info(
                "rhythm_adaptation_applied",
                from_state=old_state,
                to_state=new_state,
                drive_modulation=self._rhythm_drive_modulation,
            )
        except Exception:
            self._logger.debug("rhythm_adaptation_failed", exc_info=True)

    @property
    def active_goal_summaries(self) -> list[ActiveGoalSummary]:
        """
        Returns minimal goal summaries for Atune's salience heads.
        Atune uses goal embeddings to boost salience of goal-relevant content.
        """
        if self._goal_manager is None:
            return []
        return [
            ActiveGoalSummary(
                id=g.id,
                target_embedding=self._goal_embeddings.get(g.id, []),
                priority=g.priority,
            )
            for g in self._goal_manager.active_goals
        ]

    async def _embed_goal(self, goal: Goal) -> None:
        """Compute and cache the embedding for a goal's description."""
        if self._embed_fn is None:
            return
        try:
            embedding = await self._embed_fn(goal.description)
            self._goal_embeddings[goal.id] = embedding
        except Exception:
            self._logger.debug("goal_embedding_failed", goal_id=goal.id)

    @property
    def beliefs(self):
        return self._belief_updater.beliefs

    # ─── Private ──────────────────────────────────────────────────

    async def _dispatch_intent(
        self,
        intent: Intent,
        broadcast: WorkspaceBroadcast,
    ) -> None:
        """Dispatch an approved intent via the intent router."""
        assert self._intent_router is not None
        try:
            # Thread the Equor check from the deliberation engine so the
            # router (and Axon) receive the real verdict, not a default.
            equor_check = (
                self._deliberation_engine.last_equor_check
                if self._deliberation_engine else None
            )
            route = await self._intent_router.route(
                intent=intent,
                affect=broadcast.affect,
                conversation_id=self._current_conversation_id,
                equor_check=equor_check,
            )
            if route != "internal":
                self._total_intents_approved += 1
                # Track pending intent
                # Use the actual goal ID when available, fall back to description
                goal_id = getattr(intent.goal, "id", None) or intent.goal.description[:50]
                self._pending_intents[intent.id] = PendingIntent(
                    intent_id=intent.id,
                    goal_id=goal_id,
                    routed_to=route,
                )
        except Exception as exc:
            self._logger.error("intent_dispatch_failed", intent_id=intent.id, error=str(exc))
            self._total_intents_blocked += 1

    async def _retrieve_relevant_memories_safe(
        self,
        broadcast: WorkspaceBroadcast,
    ) -> list[dict]:
        """Memory retrieval with hard timeout (non-blocking)."""
        try:
            # Extract query text from broadcast
            content = broadcast.content
            query = ""
            for attr in ["content", "text", "summary"]:
                obj = getattr(content, attr, None)
                if isinstance(obj, str) and obj:
                    query = obj[:200]
                    break
            if not query:
                return []

            result = await asyncio.wait_for(
                self._memory.retrieve(query_text=query, max_results=5),
                timeout=0.15,  # 150ms hard timeout
            )
            traces: list[dict] = []
            for trace in result.traces[:5]:
                summary = trace.get("summary") or trace.get("content", "")
                if summary:
                    traces.append({"summary": str(summary)[:200]})
            return traces
        except (asyncio.TimeoutError, Exception):
            return []

    async def _sync_goals_to_atune_safe(self) -> None:
        """Non-blocking: sync active goal summaries to Atune for salience heads."""
        if self._goal_sync_callback is not None:
            try:
                self._goal_sync_callback(self.active_goal_summaries)
            except Exception:
                self._logger.debug("goal_sync_callback_failed", exc_info=True)

    def _record_decision(self, record: DecisionRecord) -> None:
        """Store decision record for observability (ring buffer)."""
        self._decision_records.append(record)
        if len(self._decision_records) > self._max_decision_records:
            self._decision_records = self._decision_records[-self._max_decision_records:]

    def _has_active_coherence_goal(self) -> bool:
        """Check if there's already an active coherence-repair goal."""
        if self._goal_manager is None:
            return False
        return any(
            g.source == GoalSource.SELF_GENERATED
            and "coherence" in g.description.lower()
            for g in self._goal_manager.active_goals
        )

    def _create_coherence_repair_goal(self, broadcast: WorkspaceBroadcast) -> None:
        """
        High coherence stress means the organism's beliefs conflict with
        incoming percepts. Generate a self-repair goal to seek clarification.
        """
        if self._goal_manager is None:
            return

        from ecodiaos.primitives.common import DriveAlignmentVector, new_id

        goal = Goal(
            id=new_id(),
            description="Resolve coherence conflict: seek clarifying information to reconcile contradictory beliefs",
            source=GoalSource.SELF_GENERATED,
            priority=0.7,
            urgency=broadcast.affect.coherence_stress,
            importance=0.6,
            drive_alignment=DriveAlignmentVector(
                coherence=0.9, care=0.0, growth=0.1, honesty=0.0,
            ),
            status=GoalStatus.ACTIVE,
        )
        self._goal_manager.add_goal(goal)
        self._logger.info(
            "coherence_repair_goal_created",
            stress=round(broadcast.affect.coherence_stress, 3),
        )


# ─── Belief State Adapter ─────────────────────────────────────────


class _NovaBeliefStateReader:
    """
    Adapter that implements Atune's BeliefStateReader protocol using
    Nova's belief state. This closes the top-down prediction loop:

        Nova beliefs → Atune prediction → prediction error → salience → broadcast → Nova

    For each percept source, we generate a predicted embedding from the
    belief state's current context and entity knowledge. This allows
    Atune to compute genuine surprise rather than treating everything
    as equally novel.
    """

    def __init__(self, belief_updater: BeliefUpdater) -> None:
        self._beliefs = belief_updater

    async def predict_for_source(self, source_system: str) -> "BeliefPrediction | None":
        """
        Return the expected embedding for the next Percept from *source_system*.

        Uses the context belief to generate a prediction. When the context
        has high confidence and a meaningful summary, we return that as the
        predicted content. Atune's prediction error module will compute
        semantic divergence between this prediction and the actual percept.

        This is the first step in closing the predictive processing loop.
        Future improvements: maintain per-source prediction models, track
        prediction accuracy, adapt precision based on historical errors.
        """
        from ecodiaos.systems.atune.prediction import BeliefPrediction

        beliefs = self._beliefs.beliefs
        ctx = beliefs.current_context

        # Use context belief when it has sufficient confidence
        if ctx.confidence > 0.3 and ctx.summary:
            return BeliefPrediction(
                embedding=[],  # Empty → prediction error uses semantic divergence path
                predicted_content=ctx.summary[:200],
            )

        # No strong prediction available — Atune will use default surprise (0.5)
        return None
