"""
EcodiaOS — Axon Service

The motor cortex. Axon receives approved Intents from Nova and turns them into
real-world effects — memory writes, API calls, scheduled tasks, notifications,
and federated messages.

Axon does not decide. It does not judge. It executes.
Decision authority lives in Nova. Ethical authority lives in Equor.
Axon is the disciplined hand that carries out the will.

Lifecycle:
  initialize() — builds the executor registry, wires safety systems
  execute()    — main entry point: accepts ExecutionRequest, returns AxonOutcome
  set_nova()   — wires the Nova feedback loop for outcome delivery
  shutdown()   — graceful teardown

Interface contracts (from spec):
  Validation (all steps):       ≤50ms
  Rate limit check:             ≤5ms
  Context assembly:             ≤30ms
  Simple intent (1-2 internal): ≤300ms end-to-end
  Complex intent (external):    ≤15,000ms end-to-end
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.core.hotreload import NeuroplasticityBus
from ecodiaos.systems.axon.audit import AuditLogger
from ecodiaos.systems.axon.credentials import CredentialStore
from ecodiaos.systems.axon.executor import Executor
from ecodiaos.systems.axon.executors import build_default_registry
from ecodiaos.systems.axon.pipeline import ExecutionPipeline
from ecodiaos.systems.axon.safety import BudgetTracker, CircuitBreaker, RateLimiter
from ecodiaos.systems.axon.shield import TransactionShield
from ecodiaos.systems.axon.types import AxonOutcome, ExecutionRequest

if TYPE_CHECKING:
    from ecodiaos.config import AxonConfig
    from ecodiaos.systems.axon.registry import ExecutorRegistry
    from ecodiaos.systems.memory.service import MemoryService
    from ecodiaos.systems.nova.service import NovaService
    from ecodiaos.systems.synapse.event_bus import EventBus
    from ecodiaos.systems.voxis.service import VoxisService

logger = structlog.get_logger()


class AxonService:
    """
    Axon — the EOS action execution system.

    AxonService is the single entry point for action execution.
    It owns and coordinates all sub-systems:
      - ExecutorRegistry: maps action types to handler implementations
      - ExecutionPipeline: runs the 7-stage execution pipeline
      - BudgetTracker: enforces per-cycle action limits
      - RateLimiter: sliding-window per-executor rate limits
      - CircuitBreaker: per-executor open/closed/half-open state
      - CredentialStore: issues scoped, time-limited credentials
      - AuditLogger: records every execution permanently

    All sub-systems are constructed in initialize() and are immutable at runtime.
    """

    system_id: str = "axon"

    def __init__(
        self,
        config: AxonConfig,
        memory: MemoryService | None = None,
        voxis: VoxisService | None = None,
        neuroplasticity_bus: NeuroplasticityBus | None = None,
        redis_client: Any = None,
        wallet: Any = None,
        synapse: Any = None,
        instance_id: str = "eos-default",
    ) -> None:
        self._config = config
        self._memory = memory
        self._voxis = voxis
        self._bus = neuroplasticity_bus
        self._redis = redis_client
        self._wallet = wallet
        self._synapse = synapse
        self._instance_id = instance_id
        self._logger = logger.bind(system="axon")
        self._initialized = False

        # Sub-systems -- built in initialize()
        self._registry: ExecutorRegistry | None = None
        self._pipeline: ExecutionPipeline | None = None
        self._budget: BudgetTracker | None = None
        self._rate_limiter: RateLimiter | None = None
        self._circuit_breaker: CircuitBreaker | None = None
        self._credential_store: CredentialStore | None = None
        self._audit: AuditLogger | None = None
        self._shield: TransactionShield | None = None

        # Metrics
        self._total_executions: int = 0
        self._successful_executions: int = 0
        self._failed_executions: int = 0

        # Recent outcomes ring buffer — last 50 executions for /api/v1/axon/outcomes
        self._recent_outcomes: deque[AxonOutcome] = deque(maxlen=50)

        # NeuroplasticityBus registration — done in initialize() when bus is available

    async def initialize(self) -> None:
        """
        Initialise all Axon sub-systems and build the executor registry.

        Must be called before any execute() calls.
        Idempotent — safe to call multiple times.
        """
        if self._initialized:
            return

        self._logger.info("axon_initializing", instance_id=self._instance_id)

        # Safety systems
        self._budget = BudgetTracker(self._config)
        self._rate_limiter = RateLimiter()
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout_s=300,
            half_open_max_calls=1,
        )

        # Credential store
        self._credential_store = CredentialStore()

        # Audit logger
        self._audit = AuditLogger(memory=self._memory)

        # Transaction shield (Layer 1: Economic Immune System)
        self._shield = TransactionShield(
            wallet=self._wallet,
            max_slippage_bps=50,
        )

        # Build executor registry with all built-in executors
        self._registry = build_default_registry(
            memory=self._memory,
            voxis=self._voxis,
            redis_client=self._redis,
            wallet=self._wallet,
            synapse=self._synapse,
        )

        # Execution pipeline
        self._pipeline = ExecutionPipeline(
            registry=self._registry,
            budget=self._budget,
            rate_limiter=self._rate_limiter,
            circuit_breaker=self._circuit_breaker,
            credential_store=self._credential_store,
            audit_logger=self._audit,
            instance_id=self._instance_id,
            shield=self._shield,
        )

        self._initialized = True
        self._logger.info(
            "axon_initialized",
            executors=len(self._registry),
            executor_types=self._registry.list_types(),
        )

        # Register with the NeuroplasticityBus for hot-reload of Executor subclasses.
        if self._bus is not None:
            self._bus.register(
                base_class=Executor,
                registration_callback=self._on_executor_evolved,
                system_id="axon",
                # Skip abstract stubs that have no action_type set
                instance_qualifier=lambda cls: bool(cls.action_type),
            )

    def set_nova(self, nova: NovaService) -> None:
        """
        Wire the Nova feedback loop.

        Call this after both Nova and Axon are initialised.
        Must be called before execute() for outcome delivery to work.
        """
        if self._pipeline is None:
            raise RuntimeError("AxonService.initialize() must be called before set_nova()")
        self._pipeline.set_nova(nova)
        self._logger.info("nova_wired", system="axon")

    def set_atune(self, atune: Any) -> None:
        """
        Wire Atune so execution outcomes become workspace percepts.

        The organism should perceive its own actions — closing the
        intention→execution→perception loop.
        """
        if self._pipeline is None:
            raise RuntimeError("AxonService.initialize() must be called before set_atune()")
        self._pipeline.set_atune(atune)
        self._logger.info("atune_wired", system="axon")

    def set_wallet(self, wallet: Any) -> None:
        """
        Wire the Wallet client for metabolic actions.

        Enables Axon to execute financial actions: spending, transfers,
        on-chain transactions, and cost tracking against the energy budget.
        """
        self._wallet = wallet
        self._logger.info("wallet_wired", system="axon")

    def set_synapse(self, synapse: Any) -> None:
        """
        Wire the SynapseService so funding-request executors can read live
        metabolic state (rolling_deficit, burn_rate) and emit events.

        Call this after both AxonService and SynapseService are initialised,
        before the first cognitive cycle begins.
        """
        self._synapse = synapse
        # Propagate into the already-registered RequestFundingExecutor if the
        # registry has been built (i.e. initialize() was called before this).
        if self._registry is not None:
            executor = self._registry.get("request_funding")
            if executor is not None:
                # duck-type update — avoids importing RequestFundingExecutor here
                executor._synapse = synapse  # type: ignore[attr-defined]
        self._logger.info("synapse_wired", system="axon")

    def set_event_bus(self, event_bus: "EventBus") -> None:
        """
        Wire the Synapse event bus so Axon can emit financial events.

        Call this after both AxonService and the event bus are initialised.
        When a wallet_transfer succeeds, Axon emits WALLET_TRANSFER_CONFIRMED
        which the Memory system picks up and encodes as a salience=1.0 episode.
        """
        self._event_bus = event_bus
        self._logger.info("event_bus_wired", system="axon")

    def configure_credentials(self, credentials: dict[str, str]) -> None:
        """
        Load credentials into the CredentialStore.

        Called at startup to configure external service secrets.
        Format: {"service_name": "raw_secret_or_api_key"}
        """
        if self._credential_store is None:
            raise RuntimeError("AxonService.initialize() must be called first")
        self._credential_store.configure(credentials)

    def begin_cycle(self) -> None:
        """
        Notify Axon that a new cognitive cycle is starting.

        Resets the per-cycle execution budget. Call from Synapse at the
        start of each theta rhythm cycle.
        """
        if self._budget is not None:
            self._budget.begin_cycle()

    async def execute(self, request: ExecutionRequest) -> AxonOutcome:
        """
        Execute an approved Intent.

        This is the main external interface — Nova calls this via IntentRouter
        after Equor has approved the Intent.

        Args:
            request: ExecutionRequest containing the approved Intent and Equor verdict.

        Returns:
            AxonOutcome with full step-level detail and outcome summary.
            Never raises — failures are captured in the outcome.
        """
        if not self._initialized or self._pipeline is None:
            raise RuntimeError(
                "AxonService.initialize() must be called before execute()"
            )

        self._total_executions += 1

        self._logger.info(
            "execute_start",
            intent_id=request.intent.id,
            goal=request.intent.goal.description[:60],
            steps=len(request.intent.plan.steps),
        )

        try:
            outcome = await self._pipeline.execute(request)
        except Exception as exc:
            # Pipeline should never raise, but guard anyway
            self._logger.error(
                "pipeline_raised_unexpectedly",
                intent_id=request.intent.id,
                error=str(exc),
            )
            from ecodiaos.primitives.common import new_id
            from ecodiaos.systems.axon.types import ExecutionStatus, FailureReason
            outcome = AxonOutcome(
                intent_id=request.intent.id,
                execution_id=new_id(),
                success=False,
                status=ExecutionStatus.FAILURE,
                failure_reason=FailureReason.EXECUTION_EXCEPTION.value,
                error=str(exc),
            )

        if outcome.success:
            self._successful_executions += 1
            # Emit WALLET_TRANSFER_CONFIRMED so Memory encodes it at salience=1.0
            if self._event_bus is not None:
                await self._emit_financial_events(outcome)
        else:
            self._failed_executions += 1

        self._recent_outcomes.append(outcome)
        return outcome

    async def _emit_financial_events(self, outcome: AxonOutcome) -> None:
        """Emit WALLET_TRANSFER_CONFIRMED for any successful wallet_transfer steps."""
        from ecodiaos.systems.synapse.types import SynapseEvent, SynapseEventType

        for step in outcome.step_outcomes:
            if step.action_type == "wallet_transfer" and step.result.success:
                data = dict(step.result.data)
                data["execution_id"] = outcome.execution_id
                event = SynapseEvent(
                    event_type=SynapseEventType.WALLET_TRANSFER_CONFIRMED,
                    data=data,
                    source_system="axon",
                )
                try:
                    await self._event_bus.emit(event)
                    self._logger.info(
                        "wallet_transfer_event_emitted",
                        tx_hash=data.get("tx_hash", ""),
                        token=data.get("token", ""),
                        amount=data.get("amount", ""),
                    )
                except Exception as exc:
                    self._logger.error(
                        "wallet_transfer_event_emit_failed", error=str(exc)
                    )

    def register_executor(self, executor: Any) -> None:
        """
        Register a custom executor at runtime.

        For use by integrations, plugins, and governance-approved extensions.
        The executor must be an instance of Executor ABC.
        """
        if self._registry is None:
            raise RuntimeError("AxonService.initialize() must be called first")
        self._registry.register(executor)
        self._logger.info(
            "executor_registered_runtime",
            action_type=executor.action_type,
        )

    async def health(self) -> dict[str, Any]:
        """Self-health report (implements ManagedSystem protocol)."""
        return {
            "status": "healthy" if self._initialized else "starting",
            "total_executions": self._total_executions,
            "successful": self._successful_executions,
            "failed": self._failed_executions,
            "executor_count": len(self._registry) if self._registry else 0,
        }

    async def shutdown(self) -> None:
        """Graceful shutdown — log final stats."""
        if self._bus is not None:
            self._bus.deregister(Executor)

        self._logger.info(
            "axon_shutdown",
            total_executions=self._total_executions,
            successful=self._successful_executions,
            failed=self._failed_executions,
            circuit_trips=self._circuit_breaker.trip_count()
            if self._circuit_breaker else 0,
            audit_stats=self._audit.stats if self._audit else {},
        )

    def _on_executor_evolved(self, executor: Executor) -> None:
        """
        Registration callback for NeuroplasticityBus.

        Called once per Executor subclass found in a hot-reloaded file.
        Registers the new instance into the live ExecutorRegistry with
        replace=True so the existing entry (if any) is atomically replaced.
        """
        if self._registry is None:
            return
        self._registry.register(executor, replace=True)
        self._logger.info(
            "axon_executor_hot_reloaded",
            action_type=executor.action_type,
            executor=type(executor).__name__,
        )

    @property
    def recent_outcomes(self) -> list[AxonOutcome]:
        """Return recent execution outcomes (newest first), up to 50."""
        return list(reversed(self._recent_outcomes))

    @property
    def stats(self) -> dict[str, Any]:
        """Return current operational statistics."""
        return {
            "initialized": self._initialized,
            "total_executions": self._total_executions,
            "successful_executions": self._successful_executions,
            "failed_executions": self._failed_executions,
            "executor_count": len(self._registry) if self._registry else 0,
            "executor_types": self._registry.list_types() if self._registry else [],
            "circuit_trips": self._circuit_breaker.trip_count()
            if self._circuit_breaker
            else 0,
            "budget_utilisation": self._budget.utilisation if self._budget else 0.0,
            "audit": self._audit.stats if self._audit else {},
        }
