"""
EcodiaOS — Federation Send Executor

Sends data to a federated EOS instance via Synapse event bus.
Used for cross-instance knowledge sharing, antibody sync, and coordination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import ExecutionContext, ExecutionResult, RateLimit, ValidationResult
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


class FederationSendExecutor(Executor):
    action_type = "federation_send"
    description = "Send data to a federated EOS instance"
    required_autonomy = 2
    reversible = False
    max_duration_ms = 30_000
    rate_limit = RateLimit.per_minute(10)

    def __init__(self, event_bus: Any = None) -> None:
        self._event_bus = event_bus

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        target = params.get("target_instance_id")
        if not target or not isinstance(target, str):
            return ValidationResult.fail("'target_instance_id' is required")
        payload = params.get("payload")
        if payload is None:
            return ValidationResult.fail("'payload' is required")
        message_type = params.get("message_type")
        if not message_type or not isinstance(message_type, str):
            return ValidationResult.fail("'message_type' is required and must be a string")
        return ValidationResult.ok()

    async def execute(self, params: dict[str, Any], context: ExecutionContext) -> ExecutionResult:
        target = params["target_instance_id"]
        payload = params["payload"]
        message_type = params["message_type"]

        try:
            if self._event_bus is None:
                return ExecutionResult(
                    success=False,
                    error="No event bus configured for federation send",
                )

            # Emit federation message via Synapse for the Federation system to pick up
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.ACTION_EXECUTED,
                source_system="axon",
                data={
                    "action_type": self.action_type,
                    "target_instance_id": target,
                    "message_type": message_type,
                    "payload": payload,
                    "execution_id": context.execution_id,
                },
            ))

            await self._emit_re_trace(context, params, success=True)

            return ExecutionResult(
                success=True,
                data={
                    "target_instance_id": target,
                    "message_type": message_type,
                },
                side_effects=[f"Federation message sent to {target}: {message_type}"],
            )
        except Exception as exc:
            if self._event_bus is not None:
                try:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.ACTION_FAILED,
                        source_system="axon",
                        data={
                            "action_type": self.action_type,
                            "error": str(exc),
                            "execution_id": context.execution_id,
                        },
                    ))
                except Exception:
                    pass
            return ExecutionResult(success=False, error=str(exc))

    async def _emit_re_trace(
        self, context: ExecutionContext, params: dict[str, Any], success: bool
    ) -> None:
        if self._event_bus is None:
            return
        try:
            from primitives.common import DriveAlignmentVector, SystemID, utc_now
            from primitives.re_training import RETrainingExample

            trace = RETrainingExample(
                source_system=SystemID.AXON,
                instruction=f"Send federation message to {params.get('target_instance_id', 'unknown')}",
                input_context=f"message_type={params.get('message_type', '')}",
                output=f"success={success}",
                outcome_quality=1.0 if success else 0.0,
                category="federation_communication",
                constitutional_alignment=DriveAlignmentVector(),
                timestamp=utc_now(),
            )
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="axon",
                data=trace.model_dump(mode="json"),
            ))
        except Exception:
            pass
