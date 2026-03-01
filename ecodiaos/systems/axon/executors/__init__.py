"""
EcodiaOS — Axon Built-in Executors

All built-in executors for the EOS action system.

Executors are organised by capability category:
  observation    — ObserveExecutor, QueryMemoryExecutor, AnalyseExecutor, SearchExecutor
  communication  — RespondTextExecutor, NotificationExecutor, PostMessageExecutor
  data           — CreateRecordExecutor, UpdateRecordExecutor, ScheduleExecutor, ReminderExecutor
  integration    — APICallExecutor, WebhookExecutor
  internal       — StoreInsightExecutor, UpdateGoalExecutor, ConsolidationExecutor
  financial      — WalletTransferExecutor, RequestFundingExecutor, DeFiYieldExecutor
  foraging       — BountyHunterExecutor
  entrepreneurship — DeployAssetExecutor (Phase 16d)
  mitosis        — SpawnChildExecutor, DividendCollectorExecutor (Phase 16e)

Import build_default_registry() to get a fully-populated ExecutorRegistry.
"""

from __future__ import annotations

from typing import Any

from ecodiaos.systems.axon.executors.bounty_hunter import BountyHunterExecutor
from ecodiaos.systems.axon.executors.defi_yield import DeFiYieldExecutor
from ecodiaos.systems.axon.executors.deploy_asset import DeployAssetExecutor
from ecodiaos.systems.axon.executors.communication import (
    NotificationExecutor,
    PostMessageExecutor,
    RespondTextExecutor,
)
from ecodiaos.systems.axon.executors.data import (
    CreateRecordExecutor,
    ReminderExecutor,
    ScheduleExecutor,
    UpdateRecordExecutor,
)
from ecodiaos.systems.axon.executors.financial import (
    RequestFundingExecutor,
    WalletTransferExecutor,
)
from ecodiaos.systems.axon.executors.integration import (
    APICallExecutor,
    WebhookExecutor,
)
from ecodiaos.systems.axon.executors.mitosis import (
    DividendCollectorExecutor,
    SpawnChildExecutor,
)
from ecodiaos.systems.axon.executors.internal import (
    ConsolidationExecutor,
    StoreInsightExecutor,
    UpdateGoalExecutor,
)
from ecodiaos.systems.axon.executors.observation import (
    AnalyseExecutor,
    ObserveExecutor,
    QueryMemoryExecutor,
    SearchExecutor,
)
from ecodiaos.systems.axon.registry import ExecutorRegistry

__all__ = [
    "build_default_registry",
    # Observation
    "ObserveExecutor",
    "QueryMemoryExecutor",
    "AnalyseExecutor",
    "SearchExecutor",
    # Communication
    "RespondTextExecutor",
    "NotificationExecutor",
    "PostMessageExecutor",
    # Data
    "CreateRecordExecutor",
    "UpdateRecordExecutor",
    "ScheduleExecutor",
    "ReminderExecutor",
    # Integration
    "APICallExecutor",
    "WebhookExecutor",
    # Financial
    "WalletTransferExecutor",
    "RequestFundingExecutor",
    "DeFiYieldExecutor",
    # Foraging
    "BountyHunterExecutor",
    # Entrepreneurship (Phase 16d)
    "DeployAssetExecutor",
    # Internal
    "StoreInsightExecutor",
    "UpdateGoalExecutor",
    "ConsolidationExecutor",
    # Mitosis (Phase 16e)
    "SpawnChildExecutor",
    "DividendCollectorExecutor",
]


def build_default_registry(
    memory: Any = None,
    voxis: Any = None,
    redis_client: Any = None,
    wallet: Any = None,
    synapse: Any = None,
    oikos: Any = None,
    asset_factory: Any = None,
    simula: Any = None,
) -> ExecutorRegistry:
    """
    Build and return a fully-populated ExecutorRegistry with all built-in executors.

    Args:
        memory: MemoryService instance (for memory-backed executors)
        voxis: VoxisService instance (for RespondTextExecutor)
        redis_client: Redis client (for scheduled tasks and reminders)
        wallet: WalletClient instance (for WalletTransferExecutor; omit to skip registration)
        synapse: SynapseService instance (for RequestFundingExecutor metabolic reads)
        oikos: OikosService instance (for SpawnChildExecutor fleet registration)
        asset_factory: AssetFactory instance (for DeployAssetExecutor candidate evaluation)
        simula: SimulaService instance (for DeployAssetExecutor code generation)
    """
    registry = ExecutorRegistry()

    # ── Observation (Level 1) ──────────────────────────────────────
    registry.register(ObserveExecutor(memory=memory))
    registry.register(QueryMemoryExecutor(memory=memory))
    registry.register(AnalyseExecutor(memory=memory))
    registry.register(SearchExecutor())

    # ── Communication (Level 1-2) ─────────────────────────────────
    registry.register(RespondTextExecutor(voxis=voxis))
    registry.register(NotificationExecutor(redis_client=redis_client))
    registry.register(PostMessageExecutor(memory=memory))

    # ── Data Operations (Level 2) ─────────────────────────────────
    registry.register(CreateRecordExecutor(memory=memory))
    registry.register(UpdateRecordExecutor(memory=memory))
    registry.register(ScheduleExecutor(redis_client=redis_client))
    registry.register(ReminderExecutor(redis_client=redis_client))

    # ── Integration (Level 2-3) ───────────────────────────────────
    registry.register(APICallExecutor())
    registry.register(WebhookExecutor())

    # ── Financial (Level 1–3) ─────────────────────────────────────
    # RequestFundingExecutor: always registered — it only emits an event,
    # moves no funds, and requires only AWARE autonomy (level 1).
    # WalletTransferExecutor + DeFiYieldExecutor: only registered when
    # a WalletClient is provided (on-chain operations require a wallet).
    registry.register(RequestFundingExecutor(wallet=wallet, synapse=synapse))
    if wallet is not None:
        registry.register(WalletTransferExecutor(wallet=wallet))
        registry.register(DeFiYieldExecutor(wallet=wallet))

    # ── Foraging (Level 2) ─────────────────────────────────────────
    registry.register(BountyHunterExecutor(synapse=synapse))

    # ── Entrepreneurship / Phase 16d (Level 3) ──────────────────
    # DeployAssetExecutor requires STEWARD autonomy (level 3) and
    # is only registered when an AssetFactory is provided.
    if asset_factory is not None:
        registry.register(
            DeployAssetExecutor(
                asset_factory=asset_factory,
                oikos=oikos,
                simula=simula,
            )
        )

    # ── Internal (Level 1) ───────────────────────────────────────
    registry.register(StoreInsightExecutor(memory=memory))
    registry.register(UpdateGoalExecutor())
    registry.register(ConsolidationExecutor(memory=memory))

    # ── Mitosis / Phase 16e (Level 1–3) ──────────────────────────
    # SpawnChildExecutor: only registered when WalletClient is available
    # (seed capital transfer requires on-chain access).
    # DividendCollectorExecutor: always registered (recording only, no funds moved).
    registry.register(DividendCollectorExecutor(oikos=oikos, synapse=synapse))
    if wallet is not None:
        registry.register(SpawnChildExecutor(wallet=wallet, oikos=oikos, synapse=synapse))

    return registry
