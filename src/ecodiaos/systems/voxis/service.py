"""
EcodiaOS — Voxis Service

The expression and voice system — the organism's primary communicative interface.

VoxisService is the public API for all expression. It:
- Implements BroadcastSubscriber to receive workspace broadcasts from Atune
- Orchestrates the full 9-step expression pipeline via ContentRenderer
- Manages conversation state via ConversationManager
- Makes silence decisions via SilenceEngine
- Reports expression feedback to Atune (closing the perception-action loop)
- Maintains the live personality vector (updated by Evo over time)

Architecture note on async/sync:
  on_broadcast() is called by Atune's workspace synchronously during the
  theta cycle. The silence decision is made synchronously (≤10ms). If speaking,
  the full expression pipeline is spawned as an asyncio task so it never
  blocks the workspace cycle. Expressions are delivered asynchronously.
"""

from __future__ import annotations

import asyncio
from typing import Callable

import structlog

from ecodiaos.clients.llm import LLMProvider
from ecodiaos.clients.redis import RedisClient
from ecodiaos.config import VoxisConfig
from ecodiaos.primitives.affect import AffectState
from ecodiaos.primitives.common import new_id, utc_now
from ecodiaos.primitives.expression import Expression, PersonalityVector
from ecodiaos.systems.memory.service import MemoryService
from ecodiaos.systems.voxis.affect_colouring import AffectColouringEngine
from ecodiaos.systems.voxis.audience import AudienceProfiler
from ecodiaos.systems.voxis.conversation import ConversationManager
from ecodiaos.systems.voxis.personality import PersonalityEngine
from ecodiaos.systems.voxis.renderer import ContentRenderer
from ecodiaos.systems.voxis.silence import SilenceEngine
from ecodiaos.systems.voxis.types import (
    AudienceProfile,
    ExpressionContext,
    ExpressionFeedback,
    ExpressionIntent,
    ExpressionTrigger,
    ReceptionEstimate,
    SilenceContext,
)

logger = structlog.get_logger()

# Type alias for expression delivery callback
ExpressionCallback = Callable[[Expression], None]


class VoxisService:
    """
    Expression and voice system.

    Dependencies:
        memory  — for personality loading, instance name, memory retrieval
        redis   — for conversation state persistence
        llm     — for expression generation and conversation summarisation
        config  — VoxisConfig

    Lifecycle:
        initialize()  — load personality from Memory, set up sub-components
        shutdown()    — flush any queued state
    """

    system_id: str = "voxis"

    def __init__(
        self,
        memory: MemoryService,
        redis: RedisClient,
        llm: LLMProvider,
        config: VoxisConfig,
    ) -> None:
        self._memory = memory
        self._redis = redis
        self._llm = llm
        self._config = config
        self._logger = logger.bind(system="voxis")

        # Sub-components — initialised in initialize()
        self._personality_engine: PersonalityEngine | None = None
        self._affect_engine = AffectColouringEngine()
        self._audience_profiler = AudienceProfiler()
        self._silence_engine = SilenceEngine(
            min_expression_interval_minutes=config.min_expression_interval_minutes,
        )
        self._conversation_manager: ConversationManager | None = None
        self._renderer: ContentRenderer | None = None

        # Instance metadata — loaded in initialize()
        self._instance_name: str = "EOS"
        self._drive_weights: dict[str, float] = {
            "coherence": 1.0,
            "care": 1.0,
            "growth": 1.0,
            "honesty": 1.0,
        }

        # Expression delivery callbacks (registered by WebSocket handlers, etc.)
        self._expression_callbacks: list[ExpressionCallback] = []

        # Expression feedback callbacks (for Evo personality learning, Nova outcome tracking)
        self._feedback_callbacks: list[Callable[[ExpressionFeedback], None]] = []

        # Affect state before the last expression (for affect delta tracking)
        self._affect_before_expression: AffectState | None = None

        # Background task tracking — prevents fire-and-forget error loss
        self._background_tasks: set[asyncio.Task] = set()
        self._background_task_failures: int = 0

        # Observability counters
        self._total_expressions: int = 0
        self._total_silence: int = 0
        self._total_speak: int = 0
        self._honesty_rejections: int = 0
        self._expressions_by_trigger: dict[str, int] = {}
        self._expressions_by_channel: dict[str, int] = {}

    # ─── Lifecycle ────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Load personality vector from Memory, build sub-components.
        Called during application startup after Memory is ready.
        """
        self._logger.info("voxis_initializing")

        # Load instance name and personality from Self node
        personality_vector = PersonalityVector()  # Default: neutral seed
        instance = await self._memory.get_self()
        if instance is not None:
            self._instance_name = instance.name
            # Load personality from Self node — stored as personality_json (dict)
            # or personality_vector (ordered list of 9 floats from birth)
            raw_json = getattr(instance, "personality_json", None)
            raw_vector = getattr(instance, "personality_vector", None)

            if raw_json and isinstance(raw_json, dict):
                try:
                    personality_vector = PersonalityVector(**raw_json)
                    self._logger.info(
                        "personality_loaded_from_json",
                        instance_name=self._instance_name,
                    )
                except Exception:
                    self._logger.warning("personality_json_load_failed", exc_info=True)
            elif raw_vector and isinstance(raw_vector, list) and len(raw_vector) >= 9:
                # Birth stores personality as an ordered list of 9 floats:
                # [warmth, directness, verbosity, formality, curiosity_expression,
                #  humour, empathy_expression, confidence_display, metaphor_use]
                _KEYS = [
                    "warmth", "directness", "verbosity", "formality",
                    "curiosity_expression", "humour", "empathy_expression",
                    "confidence_display", "metaphor_use",
                ]
                personality_dict = dict(zip(_KEYS, raw_vector[:9]))
                try:
                    personality_vector = PersonalityVector(**personality_dict)
                    self._logger.info(
                        "personality_loaded_from_vector",
                        instance_name=self._instance_name,
                        warmth=personality_vector.warmth,
                        empathy=personality_vector.empathy_expression,
                    )
                except Exception:
                    self._logger.warning("personality_vector_load_failed", exc_info=True)
            else:
                self._logger.warning(
                    "personality_not_found_using_defaults",
                    has_json=raw_json is not None,
                    has_vector=raw_vector is not None,
                    vector_len=len(raw_vector) if raw_vector else 0,
                )

            # Load drive weights from constitution
            constitution = await self._memory.get_constitution()
            if constitution and "drives" in constitution:
                drives = constitution["drives"]
                self._drive_weights = {
                    "coherence": float(drives.get("coherence", 1.0)),
                    "care": float(drives.get("care", 1.0)),
                    "growth": float(drives.get("growth", 1.0)),
                    "honesty": float(drives.get("honesty", 1.0)),
                }

        self._personality_engine = PersonalityEngine(personality_vector)

        self._conversation_manager = ConversationManager(
            redis=self._redis,
            llm=self._llm,
            history_window=self._config.conversation_history_window,
            context_window_max_tokens=self._config.context_window_max_tokens,
            summary_threshold=self._config.conversation_summary_threshold,
            max_active_conversations=self._config.max_active_conversations,
        )

        self._renderer = ContentRenderer(
            llm=self._llm,
            personality_engine=self._personality_engine,
            affect_engine=self._affect_engine,
            audience_profiler=self._audience_profiler,
            base_temperature=self._config.temperature_base,
            honesty_check_enabled=self._config.honesty_check_enabled,
            max_expression_length=self._config.max_expression_length,
        )

        self._logger.info(
            "voxis_initialized",
            instance_name=self._instance_name,
            drive_weights=self._drive_weights,
        )

    async def shutdown(self) -> None:
        """Graceful shutdown — nothing stateful to flush beyond what Redis persists."""
        self._logger.info(
            "voxis_shutdown",
            total_expressions=self._total_expressions,
            total_silence=self._total_silence,
        )

    # ─── BroadcastSubscriber Interface ───────────────────────────

    async def on_broadcast(self, broadcast: object) -> None:
        """
        Called by Atune when the workspace broadcasts a percept.

        The silence decision is made synchronously (≤10ms).
        If speaking, the expression pipeline is spawned as a background task
        so this method returns quickly and never blocks the theta cycle.
        """
        if self._renderer is None:
            return

        # Extract affect from broadcast if available, otherwise use neutral
        affect = getattr(broadcast, "affect", None) or AffectState.neutral()
        content = getattr(broadcast, "content", None)
        if content is None:
            return

        # Build a minimal intent from the broadcast
        content_text = getattr(content, "content", None)
        if content_text is None:
            raw = getattr(content, "raw", None)
            content_text = str(raw) if raw else ""

        if not content_text:
            return

        intent = ExpressionIntent(
            trigger=ExpressionTrigger.ATUNE_DIRECT_ADDRESS,
            content_to_express=content_text,
            urgency=float(getattr(broadcast, "salience", type("", (), {"composite": 0.5})()).composite),
        )

        # Silence decision — synchronous, fast
        silence_ctx = SilenceContext(
            trigger=intent.trigger,
            minutes_since_last_expression=self._silence_engine.minutes_since_last_expression,
            min_expression_interval=self._config.min_expression_interval_minutes,
            insight_value=intent.insight_value,
            urgency=intent.urgency,
        )
        decision = self._silence_engine.evaluate(silence_ctx)

        if not decision.speak:
            self._total_silence += 1
            return

        # Spawn expression pipeline as background task
        asyncio.create_task(
            self._express_background(intent, affect),
            name=f"voxis_express_{intent.id}",
        )

    # ─── Primary Expression API ───────────────────────────────────

    async def express(
        self,
        content: str,
        trigger: ExpressionTrigger = ExpressionTrigger.NOVA_RESPOND,
        conversation_id: str | None = None,
        addressee_id: str | None = None,
        addressee_name: str | None = None,
        affect: AffectState | None = None,
        intent_id: str | None = None,
        urgency: float = 0.5,
        insight_value: float = 0.5,
    ) -> Expression:
        """
        Generate and deliver an expression. The primary external API.

        Called by:
        - Nova (deliberate communicative intents)
        - API endpoints (chat/message)
        - Test harness

        Returns the completed Expression (also delivers via registered callbacks).
        """
        assert self._renderer is not None, "VoxisService not initialized"
        assert self._conversation_manager is not None

        current_affect = affect or AffectState.neutral()

        # Capture affect before expression for delta tracking
        self._affect_before_expression = current_affect

        # Silence check
        silence_ctx = SilenceContext(
            trigger=trigger,
            minutes_since_last_expression=self._silence_engine.minutes_since_last_expression,
            min_expression_interval=self._config.min_expression_interval_minutes,
            insight_value=insight_value,
            urgency=urgency,
        )
        decision = self._silence_engine.evaluate(silence_ctx)

        if not decision.speak:
            self._total_silence += 1
            self._logger.debug(
                "expression_suppressed",
                trigger=trigger.value,
                reason=decision.reason,
            )
            return Expression(
                is_silence=True,
                silence_reason=decision.reason,
                conversation_id=conversation_id,
                affect_valence=current_affect.valence,
                affect_arousal=current_affect.arousal,
                affect_dominance=current_affect.dominance,
                affect_curiosity=current_affect.curiosity,
                affect_care_activation=current_affect.care_activation,
                affect_coherence_stress=current_affect.coherence_stress,
            )

        # Fetch/create conversation state
        conv_state = await self._conversation_manager.get_or_create(conversation_id)
        conversation_history = await self._conversation_manager.prepare_context(conv_state)

        # Build audience profile
        audience = await self._build_audience_profile(
            addressee_id=addressee_id,
            addressee_name=addressee_name,
            conversation_id=conv_state.conversation_id,
            interaction_count=len(conv_state.messages),
        )

        # Build intent
        intent = ExpressionIntent(
            trigger=trigger,
            content_to_express=content,
            conversation_id=conv_state.conversation_id,
            addressee_id=addressee_id,
            intent_id=intent_id,
            insight_value=insight_value,
            urgency=urgency,
        )

        # Retrieve relevant memories (best-effort, non-blocking with timeout)
        relevant_memories = await self._retrieve_relevant_memories(content, current_affect)

        # Build full context
        context = ExpressionContext(
            instance_name=self._instance_name,
            personality=self._personality_engine.current,  # type: ignore[union-attr]
            affect=current_affect,
            audience=audience,
            conversation_history=conversation_history,
            relevant_memories=relevant_memories,
            intent=intent,
        )

        # Render
        expression = await self._renderer.render(intent, context, self._drive_weights)

        # Post-render: update state
        self._silence_engine.record_expression()
        self._total_expressions += 1
        self._total_speak += 1
        self._expressions_by_trigger[trigger.value] = (
            self._expressions_by_trigger.get(trigger.value, 0) + 1
        )
        self._expressions_by_channel[expression.channel] = (
            self._expressions_by_channel.get(expression.channel, 0) + 1
        )
        if expression.generation_trace and not expression.generation_trace.honesty_check_passed:
            self._honesty_rejections += 1

        # Append EOS side of exchange to conversation
        await self._conversation_manager.append_message(
            state=conv_state,
            role="assistant",
            content=expression.content,
            affect_valence=current_affect.valence,
        )

        # Async: update topics (tracked, not fire-and-forget)
        self._spawn_tracked_task(
            self._update_topics_async(conv_state),
            name=f"voxis_topics_{conv_state.conversation_id}",
        )

        # Deliver via callbacks (WebSocket handlers etc.)
        for cb in self._expression_callbacks:
            try:
                cb(expression)
            except Exception:
                self._logger.warning("expression_callback_failed", exc_info=True)

        # Generate ExpressionFeedback for Evo personality learning loop
        feedback = ExpressionFeedback(
            expression_id=expression.id,
            trigger=trigger.value,
            conversation_id=conv_state.conversation_id,
            content_summary=expression.content[:200] if expression.content else "",
            strategy_register=expression.strategy.register if expression.strategy else "neutral",
            personality_warmth=self._personality_engine.current.warmth if self._personality_engine else 0.0,
            affect_before_valence=self._affect_before_expression.valence if self._affect_before_expression else 0.0,
            affect_after_valence=current_affect.valence,
            affect_delta=current_affect.valence - (self._affect_before_expression.valence if self._affect_before_expression else 0.0),
        )

        # Dispatch feedback to all registered listeners (Evo, Nova)
        for fb_cb in self._feedback_callbacks:
            try:
                fb_cb(feedback)
            except Exception:
                self._logger.debug("feedback_callback_failed", exc_info=True)

        # Track affect state for next delta computation
        self._affect_before_expression = current_affect

        return expression

    async def ingest_user_message(
        self,
        message: str,
        conversation_id: str | None = None,
        speaker_id: str | None = None,
        affect_valence: float | None = None,
    ) -> str:
        """
        Record a user message into the conversation state.
        Returns the conversation_id (for use in the response call).
        """
        assert self._conversation_manager is not None
        conv_state = await self._conversation_manager.get_or_create(conversation_id)
        updated = await self._conversation_manager.append_message(
            state=conv_state,
            role="user",
            content=message,
            speaker_id=speaker_id,
            affect_valence=affect_valence,
        )
        return updated.conversation_id

    # ─── Personality Update (called by Evo) ──────────────────────

    def update_personality(self, delta: dict[str, float]) -> PersonalityVector:
        """
        Apply an incremental personality adjustment.
        Called by Evo after accumulating sufficient evidence.
        Returns the new PersonalityVector.
        """
        assert self._personality_engine is not None
        new_vector = self._personality_engine.apply_delta(delta)
        self._personality_engine = PersonalityEngine(new_vector)
        self._logger.info(
            "personality_updated_by_evo",
            dimensions=list(delta.keys()),
        )
        return new_vector

    # ─── Observability ────────────────────────────────────────────

    @property
    def current_personality(self) -> PersonalityVector:
        assert self._personality_engine is not None
        return self._personality_engine.current

    def register_expression_callback(self, callback: ExpressionCallback) -> None:
        """Register a callback to be called with every delivered expression."""
        self._expression_callbacks.append(callback)

    def register_feedback_callback(self, callback: Callable[[ExpressionFeedback], None]) -> None:
        """
        Register a callback for ExpressionFeedback.

        Used by:
        - Evo: observes expression reception to evolve personality over time
        - Nova: tracks expression outcomes for goal progress
        """
        self._feedback_callbacks.append(callback)

    async def health(self) -> dict:
        """Health check — returns current metrics snapshot."""
        total_decisions = self._total_speak + self._total_silence
        silence_rate = self._total_silence / max(1, total_decisions)

        return {
            "status": "healthy",
            "instance_name": self._instance_name,
            "total_expressions": self._total_expressions,
            "silence_rate": round(silence_rate, 4),
            "honesty_rejections": self._honesty_rejections,
            "expressions_by_trigger": dict(self._expressions_by_trigger),
            "expressions_by_channel": dict(self._expressions_by_channel),
            "personality": {
                "warmth": round(self.current_personality.warmth, 3),
                "directness": round(self.current_personality.directness, 3),
                "verbosity": round(self.current_personality.verbosity, 3),
                "empathy_expression": round(self.current_personality.empathy_expression, 3),
                "curiosity_expression": round(self.current_personality.curiosity_expression, 3),
            },
        }

    # ─── Private Helpers ──────────────────────────────────────────

    async def _express_background(
        self,
        intent: ExpressionIntent,
        affect: AffectState,
    ) -> None:
        """Background task wrapper for broadcast-triggered expressions."""
        try:
            await self.express(
                content=intent.content_to_express,
                trigger=intent.trigger,
                conversation_id=intent.conversation_id,
                affect=affect,
                urgency=intent.urgency,
            )
        except Exception:
            self._logger.error("background_expression_failed", exc_info=True)

    async def _build_audience_profile(
        self,
        addressee_id: str | None,
        addressee_name: str | None,
        conversation_id: str,
        interaction_count: int,
    ) -> AudienceProfile:
        """Build an AudienceProfile, pulling facts from Memory where available."""
        memory_facts: list[dict] = []

        if addressee_id:
            try:
                # Retrieve entity-level facts about this individual from Memory
                # The retrieve call is best-effort; we fall back to defaults on failure
                result = await asyncio.wait_for(
                    self._memory.retrieve(
                        query_text=f"individual person entity {addressee_id}",
                        max_results=5,
                    ),
                    timeout=0.1,  # Hard 100ms timeout — don't block expression for this
                )
                for trace in result.traces:
                    for entity in result.entities:
                        if entity.get("name") == addressee_id or entity.get("id") == addressee_id:
                            # Extract known facts
                            props = entity.get("properties", {})
                            for k, v in props.items():
                                memory_facts.append({"type": k, "value": v})
            except (asyncio.TimeoutError, Exception):
                pass  # Fall back to profile built from conversation context alone

        return self._audience_profiler.build_profile(
            addressee_id=addressee_id,
            addressee_name=addressee_name,
            interaction_count=interaction_count,
            memory_facts=memory_facts,
        )

    async def _retrieve_relevant_memories(
        self,
        query: str,
        affect: AffectState,
    ) -> list[str]:
        """
        Retrieve relevant memory traces as plain text summaries.
        Best-effort with hard 150ms timeout to stay within the cycle budget.
        """
        try:
            result = await asyncio.wait_for(
                self._memory.retrieve(query_text=query, max_results=5),
                timeout=0.15,
            )
            summaries: list[str] = []
            for trace in result.traces[:5]:
                summary = trace.get("summary") or trace.get("content", "")
                if summary:
                    summaries.append(str(summary)[:300])
            return summaries
        except (asyncio.TimeoutError, Exception):
            return []

    def _spawn_tracked_task(self, coro, name: str = "") -> asyncio.Task:
        """
        Spawn a background task with lifecycle tracking.

        Unlike bare ``asyncio.create_task``, this:
        * Keeps a strong reference so the task isn't garbage-collected.
        * Logs and counts failures instead of silently dropping them.
        * Automatically removes completed tasks from the tracking set.
        """
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)
        return task

    def _on_background_task_done(self, task: asyncio.Task) -> None:
        """Callback when a background task completes."""
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._background_task_failures += 1
            self._logger.warning(
                "background_task_failed",
                task_name=task.get_name(),
                error=str(exc),
            )

    async def _update_topics_async(self, conv_state: object) -> None:
        """Background: extract active topics and update conversation state."""
        assert self._conversation_manager is not None
        try:
            topics = await self._conversation_manager.extract_topics_async(conv_state)  # type: ignore[arg-type]
            if topics:
                await self._conversation_manager.update_topics(conv_state, topics)  # type: ignore[arg-type]
        except Exception:
            self._logger.debug("topic_update_failed", exc_info=True)
