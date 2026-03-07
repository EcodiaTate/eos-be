"""
EcodiaOS — Query Memory Executor

Queries the organism's Neo4j memory graph for knowledge retrieval.
This is a read-only executor that retrieves structured knowledge
without modifying the graph.
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


class QueryMemoryGraphExecutor(Executor):
    action_type = "query_memory"
    description = "Query the organism's Neo4j memory graph for knowledge retrieval"
    required_autonomy = 1
    reversible = False
    max_duration_ms = 15_000
    rate_limit = RateLimit.per_minute(30)
    counts_toward_budget = False  # Read-only, should not consume budget
    emits_to_atune = False  # Avoid feedback loops from memory reads

    def __init__(self, memory: Any = None, event_bus: Any = None) -> None:
        self._memory = memory
        self._event_bus = event_bus

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        query_type = params.get("query_type")
        valid_types = {"cypher", "semantic", "entity_lookup", "recent_episodes"}
        if not query_type or query_type not in valid_types:
            return ValidationResult.fail(
                f"'query_type' must be one of {sorted(valid_types)}"
            )
        if query_type == "cypher":
            cypher = params.get("cypher")
            if not cypher or not isinstance(cypher, str):
                return ValidationResult.fail("'cypher' is required for cypher query_type")
            # Safety: reject mutations
            upper = cypher.upper()
            if any(kw in upper for kw in ("CREATE", "MERGE", "DELETE", "SET ", "REMOVE ")):
                return ValidationResult.fail("Mutation queries are not allowed via query_memory")
        elif query_type == "semantic":
            if not params.get("query_text"):
                return ValidationResult.fail("'query_text' is required for semantic query_type")
        elif query_type == "entity_lookup":
            if not params.get("entity_id") and not params.get("entity_name"):
                return ValidationResult.fail(
                    "'entity_id' or 'entity_name' required for entity_lookup"
                )
        return ValidationResult.ok()

    async def execute(self, params: dict[str, Any], context: ExecutionContext) -> ExecutionResult:
        query_type = params["query_type"]
        limit = params.get("limit", 20)

        try:
            if self._memory is None:
                return ExecutionResult(
                    success=False,
                    error="No memory service configured",
                )

            results: list[dict[str, Any]] = []

            if query_type == "cypher":
                raw = await self._memory.query(params["cypher"], limit=limit)
                results = raw if isinstance(raw, list) else [raw] if raw else []

            elif query_type == "semantic":
                raw = await self._memory.semantic_search(
                    query=params["query_text"],
                    limit=limit,
                )
                results = raw if isinstance(raw, list) else []

            elif query_type == "entity_lookup":
                entity_id = params.get("entity_id", "")
                entity_name = params.get("entity_name", "")
                if entity_id:
                    raw = await self._memory.get_entity(entity_id)
                else:
                    raw = await self._memory.find_entity(name=entity_name)
                results = [raw] if raw else []

            elif query_type == "recent_episodes":
                raw = await self._memory.recent_episodes(limit=limit)
                results = raw if isinstance(raw, list) else []

            await self._emit_event(
                SynapseEventType.ACTION_EXECUTED,
                {
                    "action_type": self.action_type,
                    "query_type": query_type,
                    "result_count": len(results),
                    "execution_id": context.execution_id,
                },
            )

            await self._emit_re_trace(context, params, success=True, result_count=len(results))

            return ExecutionResult(
                success=True,
                data={
                    "query_type": query_type,
                    "result_count": len(results),
                    "results": results[:limit],
                },
                new_observations=[
                    f"Memory query ({query_type}) returned {len(results)} results"
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
        self,
        context: ExecutionContext,
        params: dict[str, Any],
        success: bool,
        result_count: int = 0,
    ) -> None:
        if self._event_bus is None:
            return
        try:
            from primitives.common import DriveAlignmentVector, SystemID, utc_now
            from primitives.re_training import RETrainingExample

            trace = RETrainingExample(
                source_system=SystemID.AXON,
                instruction=f"Query memory graph ({params.get('query_type', '')})",
                input_context=str(params)[:500],
                output=f"success={success}, results={result_count}",
                outcome_quality=1.0 if success and result_count > 0 else 0.5,
                category="memory_query",
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
