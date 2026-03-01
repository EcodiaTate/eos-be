"""
EcodiaOS -- Oikos (Economic Engine)

The organism's metabolic layer -- the capacity to acquire, allocate,
conserve, and generate resources autonomously.

Phase 16a: The Ledger -- economic state tracking, BMR, runway, starvation.
Phase 16d: Entrepreneurship -- asset factory, tollbooth, revenue tracking.
Phase 16e: Speciation -- mitosis engine, child fleet, dividend architecture.
Phase 16h: Knowledge Markets -- cognitive pricing, subscriptions, client tracking.
Phase 16i: Economic Dreaming -- Monte Carlo simulation during consolidation.
Phase 16k: Cognitive Derivatives -- futures contracts, subscription tokens, capacity ceiling.
Phase 16l: Economic Morphogenesis -- organ lifecycle, resource rebalancing.
"""

from ecodiaos.systems.oikos.asset_factory import AssetFactory, AssetPolicy
from ecodiaos.systems.oikos.base import BaseCostModel, BaseMitosisStrategy
from ecodiaos.systems.oikos.derivatives import (
    CognitiveFuture,
    DerivativesManager,
    FutureStatus,
    SubscriptionToken,
    TokenStatus,
)
from ecodiaos.systems.oikos.dreaming_types import (
    EconomicDreamResult,
    EconomicRecommendation,
    PathStatistics,
    StressScenario,
    StressScenarioConfig,
    StressTestResult,
)
from ecodiaos.systems.oikos.knowledge_market import (
    ClientRecord,
    CognitivePricingEngine,
    KnowledgeCategory,
    KnowledgeProduct,
    KnowledgeProductType,
    KnowledgeSale,
    PriceQuote,
    SubscriptionManager,
    SubscriptionTier,
    SubscriptionTierName,
    quote_price,
)
from ecodiaos.systems.oikos.morphogenesis import (
    EconomicOrgan,
    MorphogenesisResult,
    OrganCategory,
    OrganLifecycleManager,
    OrganMaturity,
    OrganTransition,
)
from ecodiaos.systems.oikos.mitosis import (
    DefaultMitosisStrategy,
    MitosisEngine,
    ReproductiveFitness,
)
from ecodiaos.systems.oikos.models import (
    ActiveBounty,
    AssetCandidate,
    AssetStatus,
    ChildPosition,
    ChildStatus,
    DividendRecord,
    EcologicalNiche,
    EconomicState,
    MetabolicPriority,
    MetabolicRate,
    OwnedAsset,
    SeedConfiguration,
    Settlement,
    StarvationLevel,
    TollboothConfig,
    YieldPosition,
)
from ecodiaos.systems.oikos.service import OikosService
from ecodiaos.systems.oikos.tollbooth import (
    TollboothDeployment,
    TollboothManager,
    TollboothReceipt,
)

# Economic Dreaming worker and simulator are imported lazily to avoid
# circular dependency with oneiros (dream_worker imports BaseOneirosWorker
# which triggers oneiros/__init__.py). Use:
#   from ecodiaos.systems.oikos.dream_worker import EconomicDreamWorker
#   from ecodiaos.systems.oikos.economic_simulator import EconomicSimulator

__all__ = [
    # Strategy ABCs
    "BaseCostModel",
    "BaseMitosisStrategy",
    # Mitosis Engine (Phase 16e)
    "MitosisEngine",
    "DefaultMitosisStrategy",
    "ReproductiveFitness",
    # Asset Factory (Phase 16d)
    "AssetFactory",
    "AssetPolicy",
    "AssetCandidate",
    "AssetStatus",
    "TollboothConfig",
    # Tollbooth (Phase 16d)
    "TollboothManager",
    "TollboothDeployment",
    "TollboothReceipt",
    # Knowledge Market (Phase 16h)
    "CognitivePricingEngine",
    "SubscriptionManager",
    "ClientRecord",
    "KnowledgeCategory",
    "KnowledgeProduct",
    "KnowledgeProductType",
    "KnowledgeSale",
    "PriceQuote",
    "SubscriptionTier",
    "SubscriptionTierName",
    "quote_price",
    # Cognitive Derivatives (Phase 16k)
    "DerivativesManager",
    "CognitiveFuture",
    "FutureStatus",
    "SubscriptionToken",
    "TokenStatus",
    # Economic Morphogenesis (Phase 16l)
    "OrganLifecycleManager",
    "EconomicOrgan",
    "OrganCategory",
    "OrganMaturity",
    "OrganTransition",
    "MorphogenesisResult",
    # Economic Dreaming Types (Phase 16i)
    "EconomicDreamResult",
    "EconomicRecommendation",
    "PathStatistics",
    "StressScenario",
    "StressScenarioConfig",
    "StressTestResult",
    # Models
    "ActiveBounty",
    "ChildPosition",
    "ChildStatus",
    "DividendRecord",
    "EcologicalNiche",
    "EconomicState",
    "MetabolicPriority",
    "MetabolicRate",
    "OikosService",
    "OwnedAsset",
    "SeedConfiguration",
    "Settlement",
    "StarvationLevel",
    "YieldPosition",
]
