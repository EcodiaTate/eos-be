"""
EcodiaOS — Allocate Resource Executor

Allocates compute or memory resources for the organism's subsystems.
Emits ACTION_EXECUTED / ACTION_FAILED events and RE training traces.
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

_VALID_RESOURCE_TYPES = frozenset({"compute", "memory", "storage", "network", "gpu"})


class AllocateResourceExecutor(Executor):
    action_type = "allocate_resource"
    description = "Allocate compute, memory, or storage resources for subsystems"
    required_autonomy = 2
    reversible = True
    max_duration_ms = 30_000
    rate_limit = RateLimit.per_minute(10)

    def __init__(self, event_bus: Any = None, resource_manager: Any = None) -> None:
        self._event_bus = event_bus
        self._resource_manager = resource_manager

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        resource_type = params.get("resource_type")
        if not resource_type or resource_type not in _VALID_RESOURCE_TYPES:
            return ValidationResult.fail(
                f"'resource_type' must be one of {sorted(_VALID_RESOURCE_TYPES)}"
            )
        amount = params.get("amount")
        if amount is None or not isinstance(amount, (int, float)) or amount <= 0:
            return ValidationResult.fail("'amount' must be a positive number")
        target_system = params.get("target_system")
        if not target_system or not isinstance(target_system, str):
            return ValidationResult.fail("'target_system' is required")
        return ValidationResult.ok()

    async def execute(self, params: dict[str, Any], context: ExecutionContext) -> ExecutionResult:
        resource_type = params["resource_type"]
        amount = params["amount"]
        target_system = params["target_system"]
        duration_s = params.get("duration_s", 3600)

        try:
            allocation_id = ""
            if self._resource_manager is not None:
                result = await self._resource_manager.allocate(
                    resource_type=resource_type,
                    amount=amount,
                    target_system=target_system,
                    duration_s=duration_s,
                )
                allocation_id = result.get("allocation_id", "")
            else:
                logger.warning(
                    "allocate_resource_no_manager",
                    resource_type=resource_type,
                    target_system=target_system,
                )
                # Proceed as a no-op allocation (informational)

            await self._emit_event(
                SynapseEventType.ACTION_EXECUTED,
                {
                    "action_type": self.action_type,
                    "resource_type": resource_type,
                    "amount": amount,
                    "target_system": target_system,
                    "allocation_id": allocation_id,
                    "execution_id": context.execution_id,
                },
            )

            await self._emit_re_trace(context, params, success=True)

            return ExecutionResult(
                success=True,
                data={
                    "allocation_id": allocation_id,
                    "resource_type": resource_type,
                    "amount": amount,
                    "target_system": target_system,
                    "duration_s": duration_s,
                },
                side_effects=[
                    f"Allocated {amount} {resource_type} for {target_system} ({duration_s}s)"
                ],
            )
        except Exception as exc:
            await self._emit_event(
                SynapseEventType.ACTION_FAILED,
                {
                    "action_type": self.action_type,
                    "error": str(exc),
                    "execution_id": context.execution_id,
                },
            )
            return ExecutionResult(success=False, error=str(exc))

    async def _emit_event(self, event_type: SynapseEventType, data: dict[str, Any]) -> None:
        if self._event_bus is None:
            return
        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=event_type,
                source_system="axon",
                data=data,
            ))
        except Exception:
            pass

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
                instruction=f"Allocate {params.get('resource_type', '')} for {params.get('target_system', '')}",
                input_context=f"amount={params.get('amount', 0)}, duration_s={params.get('duration_s', 3600)}",
                output=f"success={success}",
                outcome_quality=1.0 if success else 0.0,
                category="resource_allocation",
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
