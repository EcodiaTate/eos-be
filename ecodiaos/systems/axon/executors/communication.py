"""
EcodiaOS — Axon Communication Executors

Communication executors send messages to people. They range from Level 1
(responding in an active conversation) to Level 2 (pushing unsolicited notifications).

RespondTextExecutor  — (Level 1) route text response through Voxis for personality rendering
NotificationExecutor — (Level 2) send a push notification to a user or group
PostMessageExecutor  — (Level 2) post to a shared community channel

These executors are not reversible — you cannot un-send a message.
This asymmetry is intentional: communication has real effects in the world.
Nova and Equor bear full responsibility for approving communication intents.

All communication is routed through Voxis for personality rendering —
Axon never sends raw text directly to users.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.systems.axon.executor import Executor
from ecodiaos.systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)

if TYPE_CHECKING:
    from ecodiaos.systems.memory.service import MemoryService
    from ecodiaos.systems.voxis.service import VoxisService
    from ecodiaos.systems.voxis.types import ExpressionTrigger

logger = structlog.get_logger()


# ─── RespondTextExecutor ──────────────────────────────────────────


class RespondTextExecutor(Executor):
    """
    Send a text response in the current conversation via Voxis.

    This is the primary "speak" action. It routes the response content to
    Voxis, which applies personality rendering, affect colouring, and
    silence judgement before delivery.

    Level 1: Responding in an active conversation is always within ADVISOR scope.

    Required params:
      content (str): The response content to express.

    Optional params:
      conversation_id (str): Target conversation. Default: current active conversation.
      urgency (float 0-1): Expression urgency. Default 0.5.
      trigger (str): Voxis trigger hint. Default "NOVA_RESPOND".
    """

    action_type = "respond_text"
    description = "Send a text response via Voxis personality engine"
    required_autonomy = 1
    reversible = False
    max_duration_ms = 5000
    rate_limit = RateLimit.per_minute(30)

    def __init__(self, voxis: VoxisService | None = None) -> None:
        self._voxis = voxis
        self._logger = logger.bind(system="axon.executor.respond_text")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        if not params.get("content"):
            return ValidationResult.fail("content is required", content="missing or empty")
        content = params["content"]
        if not isinstance(content, str):
            return ValidationResult.fail("content must be a string")
        if len(content) > 10_000:
            return ValidationResult.fail("content too long (max 10,000 chars)")
        urgency = params.get("urgency", 0.5)
        if not isinstance(urgency, (int, float)) or not 0.0 <= float(urgency) <= 1.0:
            return ValidationResult.fail("urgency must be a float between 0.0 and 1.0")
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        content = params["content"]
        conversation_id = params.get("conversation_id")
        urgency = float(params.get("urgency", 0.5))
        trigger_name = params.get("trigger", "NOVA_RESPOND")

        self._logger.info(
            "respond_text_execute",
            content_preview=content[:80],
            conversation_id=conversation_id,
            execution_id=context.execution_id,
        )

        if self._voxis is None:
            return ExecutionResult(
                success=True,
                data={"content": content, "delivered": False},
                side_effects=["Text response staged (no Voxis service)"],
                new_observations=[f"EOS would have said: {content[:200]}"],
            )

        try:

            trigger = _resolve_trigger(trigger_name)
            await self._voxis.express(
                content=content,
                trigger=trigger,
                conversation_id=conversation_id,
                affect=context.affect_state,
                urgency=urgency,
            )
            return ExecutionResult(
                success=True,
                data={"content_length": len(content), "delivered": True},
                side_effects=["Text response delivered via Voxis"],
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                error=f"Voxis expression failed: {exc}",
            )


def _resolve_trigger(trigger_name: str) -> ExpressionTrigger:
    from ecodiaos.systems.voxis.types import ExpressionTrigger
    try:
        return ExpressionTrigger[trigger_name]
    except KeyError:
        return ExpressionTrigger.NOVA_RESPOND


# ─── NotificationExecutor ─────────────────────────────────────────


class NotificationExecutor(Executor):
    """
    Send a push notification to a user or group of users.

    Level 2: Sending unsolicited notifications requires PARTNER autonomy.
    EOS should use these sparingly — notification overload undermines trust
    and Care. Equor should scrutinise notification intents carefully.

    Required params:
      recipient_id (str): User ID or group ID to notify.
      title (str): Short notification title.
      body (str): Notification body text.

    Optional params:
      urgency (str): "low" | "normal" | "high". Default "normal".
      action_url (str): Deep link URL for the notification. Default None.
    """

    action_type = "send_notification"
    description = "Send a push notification to a user or group (Level 2)"
    required_autonomy = 2
    reversible = False
    max_duration_ms = 5000
    rate_limit = RateLimit.per_hour(10)  # Strict — notification spam is harmful

    def __init__(self, redis_client: Any = None) -> None:
        self._redis = redis_client
        self._logger = logger.bind(system="axon.executor.notification")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        for field in ("recipient_id", "title", "body"):
            if not params.get(field):
                return ValidationResult.fail(
                    f"{field} is required",
                    **{field: "missing or empty"},
                )
        urgency = params.get("urgency", "normal")
        if urgency not in ("low", "normal", "high"):
            return ValidationResult.fail("urgency must be 'low', 'normal', or 'high'")
        title = params["title"]
        if len(title) > 100:
            return ValidationResult.fail("title too long (max 100 chars)")
        body = params["body"]
        if len(body) > 500:
            return ValidationResult.fail("body too long (max 500 chars)")
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        recipient_id = params["recipient_id"]
        title = params["title"]
        body = params["body"]
        urgency = params.get("urgency", "normal")
        action_url = params.get("action_url")

        self._logger.info(
            "notification_execute",
            recipient_id=recipient_id,
            title=title,
            urgency=urgency,
            execution_id=context.execution_id,
        )

        notification_payload = {
            "type": "push_notification",
            "recipient_id": recipient_id,
            "title": title,
            "body": body,
            "urgency": urgency,
            "action_url": action_url,
            "sender": "eos",
            "execution_id": context.execution_id,
        }

        if self._redis is not None:
            try:
                import json as _json
                channel = f"eos:notifications:{recipient_id}"
                await self._redis.publish(channel, _json.dumps(notification_payload))
                return ExecutionResult(
                    success=True,
                    data={"channel": channel, "delivered": True},
                    side_effects=[
                        f"Notification '{title}' sent to {recipient_id} ({urgency} urgency)"
                    ],
                )
            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    error=f"Notification publish failed: {exc}",
                )
        else:
            self._logger.info("notification_no_redis", payload=notification_payload)
            return ExecutionResult(
                success=True,
                data={"delivered": False, "reason": "No Redis client"},
                side_effects=[f"Notification staged: '{title}' → {recipient_id}"],
            )


# ─── PostMessageExecutor ──────────────────────────────────────────


class PostMessageExecutor(Executor):
    """
    Post a message to a shared community channel.

    Level 2: Posting to shared channels affects multiple people simultaneously.
    This requires PARTNER autonomy — EOS should not post announcements,
    meeting notes, or channel messages without appropriate governance.

    Required params:
      channel_id (str): Target channel or space ID.
      content (str): Message content.

    Optional params:
      author_label (str): Display label for the author. Default "EOS".
      thread_id (str): Reply to a specific thread. Default None (new thread).
    """

    action_type = "post_message"
    description = "Post a message to a shared community channel (Level 2)"
    required_autonomy = 2
    reversible = False
    max_duration_ms = 5000
    rate_limit = RateLimit.per_hour(20)

    def __init__(self, memory: MemoryService | None = None) -> None:
        self._memory = memory
        self._logger = logger.bind(system="axon.executor.post_message")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        for field in ("channel_id", "content"):
            if not params.get(field):
                return ValidationResult.fail(
                    f"{field} is required",
                    **{field: "missing or empty"},
                )
        content = params["content"]
        if len(content) > 5000:
            return ValidationResult.fail("content too long (max 5,000 chars)")
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        channel_id = params["channel_id"]
        content = params["content"]
        author_label = params.get("author_label", "EOS")
        thread_id = params.get("thread_id")

        self._logger.info(
            "post_message_execute",
            channel_id=channel_id,
            content_preview=content[:80],
            execution_id=context.execution_id,
        )

        # Note: episodic memory storage for channel posts requires Percept ingestion
        # through Atune → Memory pipeline, not direct MemoryTrace creation.

        # Phase 1: stub delivery — Phase 2 will integrate with community platform
        return ExecutionResult(
            success=True,
            data={
                "channel_id": channel_id,
                "author_label": author_label,
                "thread_id": thread_id,
                "content_length": len(content),
                "delivered": False,
                "note": "Channel delivery integration pending (Phase 2)",
            },
            side_effects=[f"Message posted to channel {channel_id} by {author_label}"],
        )
