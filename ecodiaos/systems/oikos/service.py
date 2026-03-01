"""
EcodiaOS — Oikos Service (Phases 16a + 16d + 16e + 16h + 16k + 16l)

The organism's economic engine. Oikos maintains a real-time EconomicState
by subscribing to Synapse metabolic events and WalletClient balance updates.
It is the single source of truth for: "Can we afford to keep running?"

Phase 16a (Metabolism):
  - Subscribes to Synapse EventBus for METABOLIC_PRESSURE, REVENUE_INJECTED,
    and WALLET_TRANSFER_CONFIRMED events.
  - Periodically polls WalletClient for on-chain balance (liquid_balance).
  - Computes BMR via a hot-swappable BaseCostModel (neuroplastic).
  - Derives runway_hours, runway_days, starvation_level.

Phase 16d (Entrepreneurship):
  - Owns the AssetFactory for ideating, evaluating, and deploying assets.
  - Owns the TollboothManager for smart contract lifecycle.
  - Tracks owned_assets in EconomicState with break-even monitoring.
  - Periodic asset_maintenance_cycle() checks for terminations.

Phase 16h (Knowledge Markets):
  - Owns the CognitivePricingEngine for dynamic knowledge pricing.
  - Owns the SubscriptionManager for client tracking and tier management.
  - Exposes quote_price() for Nova / external API router.
  - Records knowledge sales as revenue in the income statement.

Phase 16k (Cognitive Derivatives):
  - Owns the DerivativesManager for futures contracts and subscription tokens.
  - Tracks derivative liabilities on the balance sheet.
  - Enforces combined 80% capacity ceiling across subscriptions + derivatives.
  - Periodic derivatives_maintenance_cycle() expires/settles contracts.

Performance contract:
  - snapshot() is a cheap read of pre-computed state (~0us).
  - Event handlers are async and must complete within Synapse's 100ms
    callback timeout (they do pure math, no I/O).
  - Balance polling is periodic (configurable interval), not per-cycle.

Lifecycle:
  initialize()                      — wire refs, register neuroplasticity handler
  attach(event_bus)                  — subscribe to Synapse events
  poll_balance()                     — fetch on-chain balance (call periodically)
  snapshot()                         — return current EconomicState
  asset_maintenance_cycle()          — check terminations, sweep revenue (periodic)
  derivatives_maintenance_cycle()    — expire/settle derivative contracts (periodic)
  quote_price()                      — generate a knowledge market price quote
  shutdown()                         — deregister from neuroplasticity bus
  health()                           — self-health report for Synapse
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.systems.oikos.asset_factory import AssetFactory
from ecodiaos.systems.oikos.base import BaseCostModel, BaseMitosisStrategy
from ecodiaos.systems.oikos.derivatives import DerivativesManager
from ecodiaos.systems.oikos.knowledge_market import (
    CognitivePricingEngine,
    KnowledgeProductType,
    KnowledgeSale,
    PriceQuote,
    SubscriptionManager,
)
from ecodiaos.systems.oikos.mitosis import MitosisEngine
from ecodiaos.systems.oikos.morphogenesis import MorphogenesisResult, OrganLifecycleManager
from ecodiaos.systems.oikos.models import (
    AssetStatus,
    ChildPosition,
    ChildStatus,
    DividendRecord,
    EcologicalNiche,
    EconomicState,
    MetabolicPriority,
    MetabolicRate,
    SeedConfiguration,
    StarvationLevel,
)
from ecodiaos.systems.oikos.tollbooth import TollboothManager

if TYPE_CHECKING:
    from ecodiaos.clients.redis import RedisClient
    from ecodiaos.clients.wallet import WalletClient
    from ecodiaos.config import OikosConfig
    from ecodiaos.core.hotreload import NeuroplasticityBus
    from ecodiaos.systems.identity.manager import CertificateManager
    from ecodiaos.systems.synapse.event_bus import EventBus
    from ecodiaos.systems.synapse.metabolism import MetabolicTracker
    from ecodiaos.systems.synapse.types import SynapseEvent


logger = structlog.get_logger("ecodiaos.oikos")


# ─── Default Cost Model ─────────────────────────────────────────


class DefaultCostModel(BaseCostModel):
    """
    Default BMR strategy: treats the EMA burn rate from MetabolicTracker
    as the organism's basal metabolic rate directly.

    This is the simplest correct model — actual observed spend IS the
    minimum cost, since the organism is already running at minimum
    viable capacity. Evolved strategies can add cloud cost projections,
    time-of-day weighting, or multi-model pricing.
    """

    @property
    def model_name(self) -> str:
        return "default_ema"

    def compute_bmr(
        self,
        burn_rate_usd_per_hour: Decimal,
        per_system_cost_usd: dict[str, Decimal],
        measurement_window_hours: int,
    ) -> MetabolicRate:
        return MetabolicRate.from_hourly(
            usd_per_hour=burn_rate_usd_per_hour,
            breakdown=per_system_cost_usd,
        )

    def compute_runway(
        self,
        liquid_balance: Decimal,
        survival_reserve: Decimal,
        bmr: MetabolicRate,
    ) -> tuple[Decimal, Decimal]:
        if bmr.usd_per_hour <= Decimal("0"):
            return Decimal("Infinity"), Decimal("Infinity")

        # Runway = liquid capital / hourly burn. Survival reserve is NOT
        # counted — it is only touched during metabolic emergency.
        total_available = liquid_balance
        hours = total_available / bmr.usd_per_hour
        days = hours / Decimal("24")
        return hours.quantize(Decimal("0.01")), days.quantize(Decimal("0.01"))


# ─── Oikos Service ───────────────────────────────────────────────


class OikosService:
    """
    The organism's economic engine.

    Maintains a live EconomicState by reacting to Synapse metabolic events
    and on-chain wallet balance updates. The BMR calculation strategy is
    neuroplastic — Evo can evolve it at runtime via the NeuroplasticityBus.

    Thread-safety: NOT thread-safe. Designed for the single-threaded
    asyncio event loop, same as all EOS services.
    """

    system_id: str = "oikos"

    # ── Redis key for durable state ──────────────────────────────
    _STATE_KEY = "oikos:state"

    def __init__(
        self,
        config: OikosConfig,
        wallet: WalletClient | None = None,
        metabolism: MetabolicTracker | None = None,
        instance_id: str = "eos-default",
        redis: RedisClient | None = None,
    ) -> None:
        self._config = config
        self._wallet = wallet
        self._metabolism = metabolism
        self._instance_id = instance_id
        self._redis = redis
        self._logger = logger.bind(component="oikos")

        # ── Hot-swappable cost model (neuroplastic) ──
        self._cost_model: BaseCostModel = DefaultCostModel()

        # ── Phase 16e: Mitosis Engine (neuroplastic strategy) ──
        self._mitosis = MitosisEngine(config=config)

        # ── Pre-computed state ──
        self._state = EconomicState(instance_id=instance_id)
        self._event_bus: EventBus | None = None
        self._bus: NeuroplasticityBus | None = None

        # ── Accumulators for rolling income statement ──
        self._total_revenue_usd: Decimal = Decimal("0")
        self._total_costs_usd: Decimal = Decimal("0")

        # ── Phase 16d: Entrepreneurship (Asset Creation) ──
        self._asset_factory: AssetFactory = AssetFactory(oikos=self)
        self._tollbooth_manager: TollboothManager = TollboothManager(wallet=wallet)

        # ── Phase 16h: Knowledge Markets (Cognition as Commodity) ──
        self._pricing_engine = CognitivePricingEngine()
        self._subscription_manager = SubscriptionManager()

        # ── Phase 16k: Cognitive Derivatives ──
        self._derivatives_manager = DerivativesManager(
            total_monthly_capacity=self._subscription_manager._total_monthly_capacity,
            max_capacity_pct=Decimal(str(config.derivatives_max_capacity_commitment)),
            futures_base_discount=Decimal(str(config.derivatives_futures_base_discount)),
            futures_collateral_rate=Decimal(str(config.derivatives_futures_collateral_rate)),
        )

        # ── Phase 16l: Economic Morphogenesis ──
        self._morphogenesis = OrganLifecycleManager(config=config)

        # ── Phase 16g: Certificate of Alignment tracking ──
        self._certificate_manager: CertificateManager | None = None

        # ── Phase 16i: Economic Dreaming result cache ──
        self._last_dream_result: Any = None

        self._logger.info(
            "oikos_constructed",
            instance_id=instance_id,
            cost_model=self._cost_model.model_name,
            survival_reserve_days=config.survival_reserve_days,
        )

    # ─── Lifecycle ────────────────────────────────────────────────

    def initialize(self, bus: NeuroplasticityBus | None = None) -> None:
        """
        Wire neuroplasticity handler for hot-swappable cost models.

        Call once after construction. Safe to call with bus=None if
        neuroplasticity is not available (the default cost model will be used).

        After calling this, call ``await load_state()`` to restore durable
        state from Redis before attaching to the event bus.
        """
        if bus is not None:
            self._bus = bus
            bus.register(
                base_class=BaseCostModel,
                registration_callback=self._on_cost_model_evolved,
                system_id=self.system_id,
            )
            bus.register(
                base_class=BaseMitosisStrategy,
                registration_callback=self._on_mitosis_strategy_evolved,
                system_id=self.system_id,
            )
            self._logger.info(
                "neuroplasticity_registered",
                base_classes=["BaseCostModel", "BaseMitosisStrategy"],
            )

    async def load_state(self) -> None:
        """
        Restore durable EconomicState and organ dictionary from Redis.

        Call after ``initialize()`` and before ``attach()``.  If no saved
        state exists the service starts with a fresh $0 ledger.  On any
        deserialization error the saved blob is discarded and the service
        starts fresh — a conservative fallback that prevents corrupted data
        from keeping the organism permanently broken.
        """
        if self._redis is None:
            self._logger.info("oikos_state_load_skipped", reason="no redis client")
            return

        try:
            blob = await self._redis.get_json(self._STATE_KEY)
        except Exception as exc:
            self._logger.warning("oikos_state_load_failed", error=str(exc))
            return

        if blob is None:
            self._logger.info("oikos_state_fresh_start", reason="no saved state in redis")
            return

        try:
            # ── Restore EconomicState ──────────────────────────────────────
            self._state = EconomicState.model_validate(blob["state"])

            # ── Restore rolling accumulators ──────────────────────────────
            self._total_revenue_usd = Decimal(blob.get("total_revenue_usd", "0"))
            self._total_costs_usd = Decimal(blob.get("total_costs_usd", "0"))

            # ── Restore morphogenesis organs ───────────────────────────────
            from ecodiaos.systems.oikos.morphogenesis import EconomicOrgan

            raw_organs: dict[str, Any] = blob.get("organs", {})
            restored_organs: dict[str, EconomicOrgan] = {
                organ_id: EconomicOrgan.model_validate(organ_data)
                for organ_id, organ_data in raw_organs.items()
            }
            self._morphogenesis._organs = restored_organs

            self._logger.info(
                "oikos_state_restored",
                liquid_balance=str(self._state.liquid_balance),
                net_worth=str(self._state.total_net_worth),
                starvation=self._state.starvation_level.value,
                organs=len(restored_organs),
                assets=len(self._state.owned_assets),
                children=len(self._state.child_instances),
            )
        except Exception as exc:
            self._logger.error(
                "oikos_state_deserialize_failed",
                error=str(exc),
                action="starting_fresh",
            )
            self._state = EconomicState(instance_id=self._instance_id)
            self._total_revenue_usd = Decimal("0")
            self._total_costs_usd = Decimal("0")
            self._morphogenesis._organs = {}

    async def _persist_state(self) -> None:
        """
        Serialize the current EconomicState and organ dictionary to Redis.

        Uses Pydantic's model_dump(mode="json") for safe Decimal serialization.
        Called automatically from ``_recalculate_derived_metrics()`` via a
        fire-and-forget asyncio task so the synchronous call path is never
        blocked by I/O.
        """
        if self._redis is None:
            return

        try:
            blob = {
                "state": self._state.model_dump(mode="json"),
                "total_revenue_usd": str(self._total_revenue_usd),
                "total_costs_usd": str(self._total_costs_usd),
                "organs": {
                    organ_id: organ.model_dump(mode="json")
                    for organ_id, organ in self._morphogenesis._organs.items()
                },
            }
            await self._redis.set_json(self._STATE_KEY, blob)
        except Exception as exc:
            # Persist failures are non-fatal — the in-memory state remains
            # authoritative; we just lose durability for this cycle.
            self._logger.warning("oikos_state_persist_failed", error=str(exc))

    def attach(self, event_bus: EventBus) -> None:
        """
        Subscribe to Synapse events for metabolic and financial data.

        Call after both OikosService and SynapseService are initialised.
        """
        from ecodiaos.systems.synapse.types import SynapseEventType

        self._event_bus = event_bus

        event_bus.subscribe(
            SynapseEventType.METABOLIC_PRESSURE,
            self._on_metabolic_pressure,
        )
        event_bus.subscribe(
            SynapseEventType.REVENUE_INJECTED,
            self._on_revenue_injected,
        )
        event_bus.subscribe(
            SynapseEventType.WALLET_TRANSFER_CONFIRMED,
            self._on_wallet_transfer,
        )

        # Phase 16e: Mitosis lifecycle events
        event_bus.subscribe(
            SynapseEventType.CHILD_HEALTH_REPORT,
            self._on_child_health_report,
        )
        event_bus.subscribe(
            SynapseEventType.DIVIDEND_RECEIVED,
            self._on_dividend_received,
        )

        # Phase 16l: Wire morphogenesis to the event bus
        self._morphogenesis.attach(event_bus)

        self._logger.info(
            "oikos_attached",
            subscriptions=[
                SynapseEventType.METABOLIC_PRESSURE.value,
                SynapseEventType.REVENUE_INJECTED.value,
                SynapseEventType.WALLET_TRANSFER_CONFIRMED.value,
                SynapseEventType.CHILD_HEALTH_REPORT.value,
                SynapseEventType.DIVIDEND_RECEIVED.value,
            ],
        )

    async def shutdown(self) -> None:
        """Deregister from neuroplasticity bus."""
        if self._bus is not None:
            self._bus.deregister(BaseCostModel)
            self._bus.deregister(BaseMitosisStrategy)
            self._logger.info("neuroplasticity_deregistered")

    # ─── Event Handlers ──────────────────────────────────────────

    async def _on_metabolic_pressure(self, event: SynapseEvent) -> None:
        """
        Handle METABOLIC_PRESSURE from Synapse.

        Fires every ~50 cycles when burn rate exceeds the pressure threshold.
        We use this to update BMR, current burn rate, runway, and starvation level.
        """
        data = event.data
        burn_rate_usd_per_hour = Decimal(str(data.get("burn_rate_usd_per_hour", 0)))
        rolling_deficit_usd = Decimal(str(data.get("rolling_deficit_usd", 0)))

        # Build per-caller cost breakdown (convert to Decimal)
        raw_costs: dict[str, Any] = data.get("per_system_cost_usd", {})
        per_caller_costs = {
            sid: Decimal(str(cost)) for sid, cost in raw_costs.items()
        }

        # Track cumulative costs
        self._total_costs_usd = abs(rolling_deficit_usd) + self._total_revenue_usd

        # Compute BMR via the active (possibly evolved) cost model
        bmr = self._cost_model.compute_bmr(
            burn_rate_usd_per_hour=burn_rate_usd_per_hour,
            per_system_cost_usd=per_caller_costs,
            measurement_window_hours=self._config.bmr_measurement_window_hours,
        )

        # Current burn rate = actual observed spend rate
        current_burn = MetabolicRate.from_hourly(
            usd_per_hour=burn_rate_usd_per_hour,
            breakdown=per_caller_costs,
        )

        # Compute runway
        runway_hours, runway_days = self._cost_model.compute_runway(
            liquid_balance=self._state.liquid_balance,
            survival_reserve=self._state.survival_reserve,
            bmr=bmr,
        )

        # Derive starvation level from config thresholds
        starvation = self._classify_starvation(runway_days)

        # Compute survival reserve target
        reserve_target = bmr.usd_per_day * Decimal(str(self._config.survival_reserve_days))

        # Update state atomically (single-threaded, no races)
        self._state.basal_metabolic_rate = bmr
        self._state.current_burn_rate = current_burn
        self._state.runway_hours = runway_hours
        self._state.runway_days = runway_days
        self._state.starvation_level = starvation
        self._state.survival_reserve_target = reserve_target
        self._state.costs_24h = current_burn.usd_per_day
        self._state.costs_7d = current_burn.usd_per_day * Decimal("7")
        self._state.costs_30d = current_burn.usd_per_day * Decimal("30")

        # Net income = revenue - costs (rolling)
        self._state.net_income_24h = self._state.revenue_24h - self._state.costs_24h
        self._state.net_income_7d = self._state.revenue_7d - self._state.costs_7d
        self._state.net_income_30d = self._state.revenue_30d - self._state.costs_30d

        # Metabolic efficiency
        if self._state.costs_7d > Decimal("0"):
            self._state.metabolic_efficiency = (
                self._state.revenue_7d / self._state.costs_7d
            ).quantize(Decimal("0.001"))
        else:
            self._state.metabolic_efficiency = Decimal("0")

        self._logger.debug(
            "oikos_metabolic_update",
            bmr_usd_hr=str(bmr.usd_per_hour),
            burn_usd_hr=str(burn_rate_usd_per_hour),
            runway_days=str(runway_days),
            starvation=starvation.value,
        )

    async def _on_revenue_injected(self, event: SynapseEvent) -> None:
        """
        Handle REVENUE_INJECTED from Synapse.

        Fires when external revenue arrives (wallet top-up, bounty payment, etc.).
        Updates the revenue side of the income statement and credits liquid_balance.
        """
        data = event.data
        amount = Decimal(str(data.get("amount_usd", 0)))

        self._total_revenue_usd += amount

        # Credit liquid_balance — revenue arrives in the hot wallet
        self._state.liquid_balance += amount

        # Distribute across rolling windows (simplified — Phase 16i will
        # implement proper windowed accounting with TimescaleDB)
        self._state.revenue_24h += amount
        self._state.revenue_7d += amount
        self._state.revenue_30d += amount

        # Recalculate derived metrics (net income, efficiency, runway)
        self._recalculate_derived_metrics()

        self._logger.info(
            "oikos_revenue_recorded",
            amount_usd=str(amount),
            total_revenue_usd=str(self._total_revenue_usd),
            is_positive=self._state.is_metabolically_positive,
        )

    async def _on_wallet_transfer(self, event: SynapseEvent) -> None:
        """
        Handle WALLET_TRANSFER_CONFIRMED from Axon.

        Fires when an on-chain transfer succeeds. We refresh the wallet
        balance to keep liquid_balance current.
        """
        self._logger.info(
            "oikos_wallet_transfer_noted",
            tx_hash=event.data.get("tx_hash", ""),
        )
        await self.poll_balance()

    # ─── Balance Polling ─────────────────────────────────────────

    async def poll_balance(self) -> None:
        """
        Fetch on-chain balance from WalletClient and update liquid_balance.

        Call this periodically (e.g. every 60s from main loop or a background
        task). Not called on every cycle — on-chain reads are too slow.
        """
        if self._wallet is None:
            return

        try:
            usdc_balance = await self._wallet.get_usdc_balance()
            self._state.liquid_balance = usdc_balance

            # Also inform MetabolicTracker so its hours_until_depleted is accurate
            if self._metabolism is not None:
                self._metabolism.snapshot(available_balance_usd=float(usdc_balance))

            # Recompute all derived metrics (runway, efficiency, liabilities)
            self._recalculate_derived_metrics()

            self._logger.debug(
                "oikos_balance_polled",
                usdc=str(usdc_balance),
                runway_days=str(self._state.runway_days),
            )
        except Exception as exc:
            self._logger.warning(
                "oikos_balance_poll_failed",
                error=str(exc),
            )

    # ─── Snapshot ────────────────────────────────────────────────

    def snapshot(self) -> EconomicState:
        """
        Return the current economic state.

        This is a cheap read of pre-computed values — no I/O, no computation.
        Safe to call from hot paths (Soma, Nova).
        """
        return self._state

    @property
    def is_metabolically_positive(self) -> bool:
        """Convenience property — True when 7d net income > 0."""
        return self._state.is_metabolically_positive

    @property
    def starvation_level(self) -> StarvationLevel:
        """Current starvation classification."""
        return self._state.starvation_level

    @property
    def runway_days_value(self) -> Decimal:
        """Days of operation remaining at current burn rate."""
        return self._state.runway_days

    # ─── Phase 16d: Asset Factory & Tollbooth ───────────────────

    @property
    def asset_factory(self) -> AssetFactory:
        """The organism's entrepreneurship engine for asset lifecycle."""
        return self._asset_factory

    @property
    def tollbooth_manager(self) -> TollboothManager:
        """Manager for tollbooth smart contract lifecycle."""
        return self._tollbooth_manager

    async def asset_maintenance_cycle(self) -> dict[str, Any]:
        """
        Periodic maintenance for all owned assets.

        Should be called during consolidation cycles or on a timer.
        Performs:
          1. Sweep revenue from all live tollbooths
          2. Record swept revenue against each asset
          3. Check for assets that should be terminated
          4. Recompute total_asset_value

        Returns a summary of actions taken.
        """
        swept_total = Decimal("0")
        terminated_ids: list[str] = []

        # Sweep revenue from tollbooths
        for asset in self._asset_factory.get_live_assets():
            try:
                swept = await self._tollbooth_manager.sweep_revenue(asset.asset_id)
                if swept > Decimal("0"):
                    self._asset_factory.record_revenue(asset.asset_id, swept)
                    swept_total += swept
            except Exception as exc:
                self._logger.warning(
                    "asset_sweep_failed",
                    asset_id=asset.asset_id,
                    error=str(exc),
                )

        # Inject swept revenue into the income statement and credit liquid_balance
        if swept_total > Decimal("0"):
            self._state.liquid_balance += swept_total
            self._state.revenue_24h += swept_total
            self._state.revenue_7d += swept_total
            self._state.revenue_30d += swept_total
            self._total_revenue_usd += swept_total
            self._recalculate_derived_metrics()
            self._logger.info(
                "asset_revenue_swept",
                total_usd=str(swept_total),
                liquid_balance=str(self._state.liquid_balance),
            )

        # Check terminations (90-day break-even + 30-day decline)
        terminated = self._asset_factory.check_terminations()
        terminated_ids = [a.asset_id for a in terminated]

        result = {
            "revenue_swept_usd": str(swept_total),
            "assets_terminated": len(terminated),
            "terminated_ids": terminated_ids,
            "live_assets": len(self._asset_factory.get_live_assets()),
            "building_assets": len(self._asset_factory.get_building_assets()),
        }

        if terminated:
            self._logger.info("asset_maintenance_terminations", **result)

        return result

    # ─── Phase 16e: Mitosis (Child Fleet Management) ────────────

    @property
    def mitosis(self) -> MitosisEngine:
        """Access the MitosisEngine for reproductive evaluation."""
        return self._mitosis

    def register_child(self, child: ChildPosition) -> None:
        """
        Register a newly spawned child in the economic state.

        Called by SpawnChildExecutor after successful seed transfer.
        Debits liquid_balance for the seed capital (funds leave the hot wallet).
        """
        self._state.child_instances.append(child)

        # Seed capital leaves the parent's liquid balance
        self._state.liquid_balance -= child.seed_capital_usd
        self._recompute_fleet_equity()
        self._recalculate_derived_metrics()

        self._logger.info(
            "child_registered",
            child_id=child.instance_id,
            niche=child.niche,
            seed=str(child.seed_capital_usd),
            liquid_balance=str(self._state.liquid_balance),
        )

    def record_dividend(self, record: DividendRecord) -> None:
        """
        Record a dividend payment from a child and update fleet accounting.

        Credits liquid_balance (dividend arrives in hot wallet) and updates
        the income statement.
        """
        self._mitosis.record_dividend(record)

        # Update the child's cumulative dividend total
        for child in self._state.child_instances:
            if child.instance_id == record.child_instance_id:
                child.total_dividends_paid_usd += record.amount_usd
                break

        # Dividend counts as revenue for the parent — credit liquid_balance
        self._state.liquid_balance += record.amount_usd
        self._total_revenue_usd += record.amount_usd
        self._state.revenue_24h += record.amount_usd
        self._state.revenue_7d += record.amount_usd
        self._state.revenue_30d += record.amount_usd
        self._recalculate_derived_metrics()

        self._logger.info(
            "dividend_recorded_in_oikos",
            child=record.child_instance_id,
            amount=str(record.amount_usd),
        )

    async def _on_child_health_report(self, event: SynapseEvent) -> None:
        """
        Handle CHILD_HEALTH_REPORT from Synapse.

        Fires when a child instance reports its current metrics. We update
        the child's position and evaluate whether its status should change.
        """
        from ecodiaos.primitives.common import utc_now

        data = event.data
        child_id = str(data.get("child_instance_id", ""))

        for child in self._state.child_instances:
            if child.instance_id == child_id:
                # Update child metrics from report
                child.current_net_worth_usd = Decimal(str(data.get("net_worth_usd", child.current_net_worth_usd)))
                child.current_runway_days = Decimal(str(data.get("runway_days", child.current_runway_days)))
                child.current_efficiency = Decimal(str(data.get("efficiency", child.current_efficiency)))
                child.consecutive_positive_days = int(data.get("consecutive_positive_days", child.consecutive_positive_days))
                child.last_health_report_at = utc_now()

                # Re-evaluate status
                new_status = self._mitosis.evaluate_child_health(child)
                if new_status != child.status:
                    old_status = child.status
                    child.status = new_status
                    self._logger.info(
                        "child_status_changed",
                        child_id=child_id,
                        old=old_status.value,
                        new=new_status.value,
                    )
                break

        self._recompute_fleet_equity()

    async def _on_dividend_received(self, event: SynapseEvent) -> None:
        """
        Handle DIVIDEND_RECEIVED from Synapse.

        This handler exists for dividends arriving via federation or other
        paths. The DividendCollectorExecutor calls record_dividend() directly
        for the primary path.
        """
        data = event.data
        child_id = str(data.get("child_instance_id", ""))
        self._logger.debug(
            "dividend_event_received",
            child_id=child_id,
            amount=data.get("amount_usd"),
        )

    def _recompute_fleet_equity(self) -> None:
        """Recompute total_fleet_equity from active child positions."""
        self._state.total_fleet_equity = sum(
            (c.current_net_worth_usd for c in self._state.child_instances
             if c.status != ChildStatus.DEAD),
            Decimal("0"),
        )

    # ─── Phase 16h: Knowledge Markets ───────────────────────────

    @property
    def pricing_engine(self) -> CognitivePricingEngine:
        """The organism's cognitive pricing engine."""
        return self._pricing_engine

    @property
    def subscription_manager(self) -> SubscriptionManager:
        """Manager for external client subscriptions and purchase history."""
        return self._subscription_manager

    def quote_price(
        self,
        product_type: KnowledgeProductType,
        estimated_tokens: int,
        client_id: str,
    ) -> PriceQuote:
        """
        Generate an instant price quote for a cognitive task.

        This is the primary entry point for Nova or the external API router.
        If the client is unknown, they are auto-registered.

        Args:
            product_type: Which knowledge product is being requested.
            estimated_tokens: Estimated token consumption for the task.
            client_id: External identifier of the buyer (human or agent).

        Returns:
            A PriceQuote with full price breakdown and 5-minute validity.
        """
        from ecodiaos.systems.oikos.knowledge_market import quote_price as _quote

        return _quote(
            product_type=product_type,
            estimated_tokens=estimated_tokens,
            client_id=client_id,
            pricing_engine=self._pricing_engine,
            subscription_manager=self._subscription_manager,
        )

    def record_knowledge_sale(self, sale: KnowledgeSale) -> None:
        """
        Record a completed knowledge sale.

        Updates the client's purchase history (driving loyalty discount),
        credits liquid_balance, and injects the revenue into the income statement.
        """
        self._subscription_manager.record_purchase(sale)

        # Knowledge sale revenue flows into the income statement and hot wallet
        self._state.liquid_balance += sale.price_usd
        self._total_revenue_usd += sale.price_usd
        self._state.revenue_24h += sale.price_usd
        self._state.revenue_7d += sale.price_usd
        self._state.revenue_30d += sale.price_usd
        self._recalculate_derived_metrics()

        self._logger.info(
            "knowledge_sale_recorded",
            sale_id=sale.sale_id,
            client_id=sale.client_id,
            product=sale.product_type.value,
            price_usd=str(sale.price_usd),
        )

    # ─── Phase 16k: Cognitive Derivatives ───────────────────────

    @property
    def derivatives_manager(self) -> DerivativesManager:
        """Manager for cognitive futures and subscription tokens."""
        return self._derivatives_manager

    def record_derivative_revenue(self, amount_usd: Decimal, source: str = "derivative") -> None:
        """
        Record revenue from a derivative sale (future or token mint).

        Credits liquid_balance (payment arrives in hot wallet) and injects
        into the income statement the same way as knowledge sales.
        """
        self._state.liquid_balance += amount_usd
        self._total_revenue_usd += amount_usd
        self._state.revenue_24h += amount_usd
        self._state.revenue_7d += amount_usd
        self._state.revenue_30d += amount_usd
        self._recalculate_derived_metrics()

        self._logger.info(
            "derivative_revenue_recorded",
            amount_usd=str(amount_usd),
            source=source,
            liquid_balance=str(self._state.liquid_balance),
        )

    async def derivatives_maintenance_cycle(self) -> dict[str, int | str]:
        """
        Periodic maintenance for derivative instruments.

        Should be called during consolidation cycles. Performs:
          1. Expire futures past their delivery window
          2. Expire tokens past their validity date
          3. Credit liquid_balance for released collateral
          4. Recalculate derivative liabilities on the balance sheet

        Returns a summary of actions taken.
        """
        # Snapshot collateral BEFORE maintenance to detect releases
        collateral_before = self._derivatives_manager.locked_collateral_usd

        result = self._derivatives_manager.maintenance_cycle()

        # Detect collateral released by settled/expired futures
        collateral_after = self._derivatives_manager.locked_collateral_usd
        collateral_released = collateral_before - collateral_after

        # Released collateral returns to liquid_balance — it was previously
        # locked as a performance guarantee and is now freed
        if collateral_released > Decimal("0"):
            self._state.liquid_balance += collateral_released
            self._recalculate_derived_metrics()
            result["collateral_released_usd"] = str(collateral_released)

        # Update balance sheet: derivative liabilities reduce available capital
        liabilities = self._derivatives_manager.total_liabilities_usd
        result["total_liabilities_usd"] = str(liabilities)
        result["locked_collateral_usd"] = str(collateral_after)

        self._logger.info(
            "derivatives_maintenance_complete",
            **result,
        )
        return result

    @property
    def derivative_liabilities_usd(self) -> Decimal:
        """Total outstanding liabilities from derivative commitments."""
        return self._derivatives_manager.total_liabilities_usd

    @property
    def combined_capacity_committed_pct(self) -> Decimal:
        """
        Fraction of total capacity committed across subscriptions AND derivatives.

        This is the number the 80% ceiling is enforced against.
        """
        sub_committed = self._subscription_manager._committed_monthly_requests()
        deriv_committed = self._derivatives_manager.derivatives_committed_requests()
        total_capacity = self._subscription_manager._total_monthly_capacity
        if total_capacity <= 0:
            return Decimal("0")
        return (
            Decimal(str(sub_committed + deriv_committed))
            / Decimal(str(total_capacity))
        ).quantize(Decimal("0.001"))

    # ── Phase 16l: Economic Morphogenesis ────────────────────────

    @property
    def morphogenesis(self) -> OrganLifecycleManager:
        """The organism's organ lifecycle manager."""
        return self._morphogenesis

    async def morphogenesis_cycle(self) -> MorphogenesisResult:
        """
        Run the morphogenesis consolidation cycle.

        Should be called during the organism's consolidation/sleep phase.
        Evaluates all organ lifecycles, applies transitions, normalises
        resource allocations, and emits events for Synapse.

        Returns a summary of transitions and new allocations.
        """
        result = await self._morphogenesis.run_consolidation_cycle()

        self._logger.info(
            "morphogenesis_cycle_integrated",
            active_organs=result.total_active_organs,
            transitions=len(result.transitions),
        )

        return result

    # ── Phase 16g: Certificate of Alignment Tracking ────────────

    def set_certificate_manager(self, cert_mgr: CertificateManager) -> None:
        """Wire CertificateManager for certificate expiry monitoring."""
        self._certificate_manager = cert_mgr
        self._logger.info("certificate_manager_wired")

    async def check_certificate_expiry(self) -> None:
        """
        Check certificate validity and trigger renewal if expiring.

        Called periodically (default every hour via config.certificate_check_interval_s).
        When the certificate has < 7 days remaining, emits an OBLIGATIONS-priority
        intent to renew it by paying the Citizenship Tax.
        When the certificate expires, the event is emitted for Thymos to raise
        a CRITICAL survival incident.
        """
        if self._certificate_manager is None:
            return

        from ecodiaos.systems.identity.certificate import CertificateStatus

        status = await self._certificate_manager.check_expiry()
        if status is None:
            return

        remaining = self._certificate_manager.certificate_remaining_days

        if status == CertificateStatus.EXPIRING_SOON and self._event_bus is not None:
            # Trigger OBLIGATIONS-priority renewal intent via Synapse
            from ecodiaos.systems.synapse.types import SynapseEvent, SynapseEventType

            renewal_cost = Decimal(str(self._config.certificate_renewal_cost_usd))
            ca_address = self._config.certificate_ca_address

            if ca_address and renewal_cost > Decimal("0"):
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.CERTIFICATE_EXPIRING,
                    source_system="oikos",
                    data={
                        "action": "renew_certificate",
                        "priority": MetabolicPriority.OBLIGATIONS.value,
                        "remaining_days": remaining,
                        "renewal_cost_usd": str(renewal_cost),
                        "ca_address": ca_address,
                        "instance_id": self._instance_id,
                    },
                ))
                self._logger.warning(
                    "certificate_renewal_triggered",
                    remaining_days=f"{remaining:.1f}",
                    cost_usd=str(renewal_cost),
                )

        elif status == CertificateStatus.EXPIRED:
            self._logger.error(
                "certificate_expired_critical",
                instance_id=self._instance_id,
            )

    @property
    def certificate_validity_days(self) -> float:
        """Days remaining on the current certificate. -1 if none."""
        if self._certificate_manager is None:
            return -1.0
        return self._certificate_manager.certificate_remaining_days

    # ── Phase 16i: Economic Dreaming Integration ────────────────

    def integrate_dream_result(self, result: Any) -> None:
        """
        Integrate results from economic dreaming into the live EconomicState.

        Called by OneirosService after the Monte Carlo simulation completes
        during a consolidation cycle. Updates survival_probability_30d and
        logs recommendations for downstream consumption (Nova, Evo).
        """
        from ecodiaos.systems.oikos.dreaming_types import EconomicDreamResult

        if not isinstance(result, EconomicDreamResult):
            self._logger.warning(
                "invalid_dream_result_type",
                type_name=type(result).__name__,
            )
            return

        # Update the survival probability estimate on live state
        self._state.survival_probability_30d = result.survival_probability_30d

        # Store latest dream result for observability
        self._last_dream_result = result

        self._logger.info(
            "economic_dream_integrated",
            ruin_probability=str(result.ruin_probability),
            survival_30d=str(result.survival_probability_30d),
            resilience_score=str(result.resilience_score),
            recommendations=len(result.recommendations),
            total_paths=result.total_paths_simulated,
            duration_ms=result.duration_ms,
        )

        for rec in result.recommendations:
            self._logger.warning(
                "economic_dream_recommendation",
                action=rec.action,
                description=rec.description[:200],
                priority=rec.priority,
                parameter_path=rec.parameter_path,
                confidence=str(rec.confidence),
            )

    @property
    def last_dream_result(self) -> Any:
        """Most recent economic dream result, for observability."""
        return getattr(self, "_last_dream_result", None)

    # ─── Neuroplasticity ─────────────────────────────────────────

    def _on_cost_model_evolved(self, new_model: BaseCostModel) -> None:
        """
        Callback from NeuroplasticityBus when a new BaseCostModel subclass
        is discovered. Hot-swaps the cost model without restart.
        """
        old_name = self._cost_model.model_name
        self._cost_model = new_model
        self._logger.info(
            "cost_model_evolved",
            old=old_name,
            new=new_model.model_name,
        )

    def _on_mitosis_strategy_evolved(self, new_strategy: BaseMitosisStrategy) -> None:
        """
        Callback from NeuroplasticityBus when a new BaseMitosisStrategy subclass
        is discovered. Hot-swaps the niche detection strategy without restart.
        """
        self._mitosis.set_strategy(new_strategy)

    # ─── Derived Metrics Recalculation ──────────────────────────

    def _recalculate_derived_metrics(self) -> None:
        """
        Recalculate net income, metabolic efficiency, and runway after
        any revenue or cost change.

        Called from every code path that mutates the income statement
        or liquid_balance to keep derived metrics consistent.
        """
        s = self._state

        # Net income = revenue - costs (rolling windows)
        s.net_income_24h = s.revenue_24h - s.costs_24h
        s.net_income_7d = s.revenue_7d - s.costs_7d
        s.net_income_30d = s.revenue_30d - s.costs_30d

        # Metabolic efficiency = revenue / costs over 7d window
        if s.costs_7d > Decimal("0"):
            s.metabolic_efficiency = (
                s.revenue_7d / s.costs_7d
            ).quantize(Decimal("0.001"))
        else:
            s.metabolic_efficiency = Decimal("0")

        # Sync derivative liabilities onto the balance sheet
        s.derivative_liabilities = self._derivatives_manager.total_liabilities_usd

        # Recompute runway with current liquid balance
        if s.basal_metabolic_rate.usd_per_hour > Decimal("0"):
            hours, days = self._cost_model.compute_runway(
                liquid_balance=s.liquid_balance,
                survival_reserve=s.survival_reserve,
                bmr=s.basal_metabolic_rate,
            )
            s.runway_hours = hours
            s.runway_days = days
            s.starvation_level = self._classify_starvation(days)

        # Persist durably to Redis after every state mutation.  Fire-and-forget
        # so the synchronous call path is never blocked by I/O.
        if self._redis is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._persist_state())
            except RuntimeError:
                pass  # No running event loop (unit tests, startup before attach)

    # ─── Starvation Classification ───────────────────────────────

    def _classify_starvation(self, runway_days: Decimal) -> StarvationLevel:
        """
        Map runway_days to a StarvationLevel using config thresholds.

        Thresholds (from spec Section XVII):
          critical  <= 1 day
          emergency <= 3 days
          austerity <= 7 days
          cautious  <= 14 days
          nominal   > 14 days
        """
        try:
            days_float = float(runway_days)
        except Exception:
            return StarvationLevel.NOMINAL

        if days_float <= self._config.critical_threshold_days:
            return StarvationLevel.CRITICAL
        if days_float <= self._config.emergency_threshold_days:
            return StarvationLevel.EMERGENCY
        if days_float <= self._config.austerity_threshold_days:
            return StarvationLevel.AUSTERITY
        if days_float <= self._config.cautious_threshold_days:
            return StarvationLevel.CAUTIOUS
        return StarvationLevel.NOMINAL

    # ─── Health ──────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Self-health report for Synapse health monitoring."""
        live_count = len(self._asset_factory.get_live_assets())
        building_count = len(self._asset_factory.get_building_assets())
        return {
            "status": "healthy",
            "cost_model": self._cost_model.model_name,
            "runway_days": str(self._state.runway_days),
            "starvation_level": self._state.starvation_level.value,
            "liquid_balance": str(self._state.liquid_balance),
            "is_metabolically_positive": self._state.is_metabolically_positive,
            "bmr_usd_per_hour": str(self._state.basal_metabolic_rate.usd_per_hour),
            "assets_live": live_count,
            "assets_building": building_count,
            "total_asset_value": str(self._state.total_asset_value),
            "fleet_children": len(self._state.child_instances),
            "fleet_equity": str(self._state.total_fleet_equity),
            "mitosis_strategy": self._mitosis.strategy.strategy_name,
            "knowledge_market_clients": self._subscription_manager.stats["total_clients"],
            "knowledge_market_subscribers": self._subscription_manager.active_subscribers,
            # Phase 16k: Cognitive Derivatives
            "derivatives_active_futures": len(self._derivatives_manager.get_active_futures()),
            "derivatives_active_tokens": len(self._derivatives_manager.get_active_tokens()),
            "derivatives_liabilities_usd": str(self._derivatives_manager.total_liabilities_usd),
            "derivatives_locked_collateral_usd": str(self._derivatives_manager.locked_collateral_usd),
            "combined_capacity_committed_pct": str(self.combined_capacity_committed_pct),
            # Phase 16l: Economic Morphogenesis
            "morpho_active_organs": len(self._morphogenesis.active_organs),
            "morpho_total_organs": len(self._morphogenesis.all_organs),
            "certificate_validity_days": f"{self.certificate_validity_days:.1f}",
        }

    # ─── Stats ───────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        """Expose economic state for aggregation / observability."""
        s = self._state
        return {
            "liquid_balance": str(s.liquid_balance),
            "survival_reserve": str(s.survival_reserve),
            "total_net_worth": str(s.total_net_worth),
            "bmr_usd_per_hour": str(s.basal_metabolic_rate.usd_per_hour),
            "bmr_usd_per_day": str(s.basal_metabolic_rate.usd_per_day),
            "burn_rate_usd_per_hour": str(s.current_burn_rate.usd_per_hour),
            "runway_hours": str(s.runway_hours),
            "runway_days": str(s.runway_days),
            "starvation_level": s.starvation_level.value,
            "is_metabolically_positive": s.is_metabolically_positive,
            "metabolic_efficiency": str(s.metabolic_efficiency),
            "net_income_7d": str(s.net_income_7d),
            "revenue_7d": str(s.revenue_7d),
            "costs_7d": str(s.costs_7d),
            "cost_model": self._cost_model.model_name,
            "total_asset_value": str(s.total_asset_value),
            "owned_assets_count": len(s.owned_assets),
            "asset_factory": self._asset_factory.stats,
            "fleet_children": len(s.child_instances),
            "fleet_equity": str(s.total_fleet_equity),
            "mitosis_strategy": self._mitosis.strategy.strategy_name,
            "total_dividends_received": str(sum(
                (r.amount_usd for r in self._mitosis.dividend_history),
                Decimal("0"),
            )),
            # Phase 16h: Knowledge Markets
            "knowledge_market": self._subscription_manager.stats,
            # Phase 16k: Cognitive Derivatives
            "derivatives": self._derivatives_manager.stats,
            "derivative_liabilities_usd": str(self._derivatives_manager.total_liabilities_usd),
            "combined_capacity_committed_pct": str(self.combined_capacity_committed_pct),
            # Phase 16l: Economic Morphogenesis
            "morphogenesis": self._morphogenesis.stats,
            # Phase 16g: Certificate of Alignment
            "certificate_validity_days": f"{self.certificate_validity_days:.1f}",
            "certificate_status": (
                self._certificate_manager.certificate.status.value
                if self._certificate_manager and self._certificate_manager.certificate
                else "none"
            ),
            # Phase 16i: Economic Dreaming
            "survival_probability_30d": str(s.survival_probability_30d),
            "dreaming_ruin_probability": str(
                self._last_dream_result.ruin_probability
                if getattr(self, "_last_dream_result", None) is not None
                else "n/a"
            ),
            "dreaming_resilience_score": str(
                self._last_dream_result.resilience_score
                if getattr(self, "_last_dream_result", None) is not None
                else "n/a"
            ),
        }
