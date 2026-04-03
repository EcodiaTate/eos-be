"""
EcodiaOS - Infrastructure Client Initialization

Creates and connects all shared infrastructure clients (data stores,
LLM, embedding, caches) that cognitive systems depend on.

Extracted from main.py to keep the entry point thin and enable
per-system hot-reload without touching the startup module.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from clients.wallet import WalletClient

from clients.embedding import create_embedding_client
from clients.llm import create_llm_provider
from clients.neo4j import Neo4jClient
from clients.optimized_llm import OptimizedLLMProvider
from clients.prompt_cache import PromptCache
from clients.redis import RedisClient
from clients.timescaledb import TimescaleDBClient
from clients.token_budget import TokenBudget
from core.hotreload import NeuroplasticityBus
from telemetry.llm_metrics import LLMMetricsCollector
from telemetry.logging import setup_logging
from telemetry.metrics import MetricCollector

logger = structlog.get_logger()

# Timeout for infrastructure connect() calls (seconds).
# Prevents startup from hanging indefinitely when a data store is unreachable.
INFRA_CONNECT_TIMEOUT_S: float = 10.0


@dataclass
class InfraClients:
    """Container for all shared infrastructure clients."""

    config: Any
    neo4j: Neo4jClient | None = field(init=False, default=None)
    tsdb: TimescaleDBClient | None = field(init=False, default=None)
    redis: RedisClient | None = field(init=False, default=None)
    llm: OptimizedLLMProvider = field(init=False)
    raw_llm: Any = field(init=False)
    embedding: Any = field(init=False)
    token_budget: TokenBudget = field(init=False)
    llm_metrics: LLMMetricsCollector = field(init=False)
    prompt_cache: PromptCache | None = field(init=False, default=None)
    neuroplasticity_bus: NeuroplasticityBus = field(init=False)
    metrics: MetricCollector = field(init=False)
    wallet: WalletClient | None = field(init=False, default=None)
    tollbooth_ledger: Any = field(init=False, default=None)


async def create_infra(config: Any) -> InfraClients:
    """
    Instantiate and connect all infrastructure clients.

    Follows the Infrastructure Architecture spec section 3.2:
    1. Logging
    2. Data stores (Neo4j, TimescaleDB, Redis)
    3. LLM + embedding
    4. Neuroplasticity bus
    5. Telemetry

    Raises RuntimeError on critical failures (data store connections).
    """
    infra = InfraClients(config=config)

    # ── 1. Logging ────────────────────────────────────────────
    setup_logging(config.logging, instance_id=config.instance_id)
    logger.info(
        "ecodiaos_starting",
        instance_id=config.instance_id,
        config_path=os.environ.get("ORGANISM_CONFIG_PATH", "config/default.yaml"),
    )

    # ── 2. Data stores ────────────────────────────────────────
    # All data store failures are non-fatal.  The organism starts in
    # degraded / safe-mode rather than crashing outright, so the /health
    # endpoint remains reachable and external monitors see a degraded
    # organism instead of "unresponsive after N health-check failures".
    #
    # Each connect() is wrapped in asyncio.wait_for() with a 10s timeout
    # to prevent hanging on unreachable hosts (half-open TCP, DNS stall).
    # Without this, a hung connect() blocks lifespan() indefinitely and
    # the HTTP server never starts — causing "organism unresponsive".
    try:
        infra.neo4j = Neo4jClient(config.neo4j)
        await asyncio.wait_for(infra.neo4j.connect(), timeout=INFRA_CONNECT_TIMEOUT_S)
    except (TimeoutError, asyncio.TimeoutError):
        logger.error(
            "neo4j_connect_timeout",
            timeout_s=INFRA_CONNECT_TIMEOUT_S,
            dependents="Memory, Nova, Evo, Thread",
            note="Organism will start in degraded mode; Neo4j connect timed out",
        )
        infra.neo4j = None
    except Exception as exc:
        logger.error(
            "neo4j_init_failed_non_fatal",
            error=str(exc),
            dependents="Memory, Nova, Evo, Thread",
            note="Organism will start in degraded mode; Memory/critical systems will report unhealthy",
            exc_info=True,
        )
        infra.neo4j = None  # Downstream systems guard on None and degrade gracefully

    try:
        infra.tsdb = TimescaleDBClient(config.timescaledb)
        await asyncio.wait_for(infra.tsdb.connect(), timeout=INFRA_CONNECT_TIMEOUT_S)
    except (TimeoutError, asyncio.TimeoutError):
        logger.warning(
            "timescaledb_connect_timeout",
            timeout_s=INFRA_CONNECT_TIMEOUT_S,
            dependents="Metrics, Skia",
        )
        infra.tsdb = None
    except Exception as exc:
        logger.warning(
            "timescaledb_init_failed_non_fatal",
            error=str(exc),
            dependents="Metrics, Skia",
        )
        infra.tsdb = None  # Metrics and Skia will degrade gracefully

    try:
        infra.redis = RedisClient(config.redis)
        await asyncio.wait_for(infra.redis.connect(), timeout=INFRA_CONNECT_TIMEOUT_S)
    except (TimeoutError, asyncio.TimeoutError):
        logger.error(
            "redis_connect_timeout",
            timeout_s=INFRA_CONNECT_TIMEOUT_S,
            dependents="NeuroplasticityBus, Alive, Voxis, Axon, PromptCache",
            note="Organism will start in degraded mode; Redis connect timed out",
        )
        infra.redis = None
    except Exception as exc:
        logger.error(
            "redis_init_failed_non_fatal",
            error=str(exc),
            dependents="NeuroplasticityBus, Alive, Voxis, Axon, PromptCache",
            note="Organism will start in degraded mode; EventBus falls back to in-memory only",
            exc_info=True,
        )
        infra.redis = None  # EventBus and downstream systems degrade to in-memory

    # ── 2a. Tollbooth credit ledger ───────────────────────────
    if infra.redis is not None:
        from api.monetization.ledger import CreditLedger

        infra.tollbooth_ledger = CreditLedger(
            redis=infra.redis.client,
            prefix=config.redis.prefix,
        )

    # ── 2b. Neuroplasticity bus ───────────────────────────────
    infra.neuroplasticity_bus = NeuroplasticityBus(redis_client=infra.redis)
    infra.neuroplasticity_bus.start()

    # ── 3. LLM + embedding ────────────────────────────────────
    infra.raw_llm = create_llm_provider(config.llm)

    infra.token_budget = TokenBudget(
        max_tokens_per_hour=config.llm.budget.max_tokens_per_hour,
        max_calls_per_hour=config.llm.budget.max_calls_per_hour,
        hard_limit=config.llm.budget.hard_limit,
    )

    infra.llm_metrics = LLMMetricsCollector()

    if infra.redis is not None:
        try:
            infra.prompt_cache = PromptCache(
                redis_client=infra.redis.client,
                prefix="eos:llmcache",
            )
            logger.info("prompt_cache_initialized")
        except Exception as exc:
            logger.warning("prompt_cache_init_failed", error=str(exc))

    infra.llm = OptimizedLLMProvider(
        inner=infra.raw_llm,
        cache=infra.prompt_cache,
        budget=infra.token_budget,
        metrics=infra.llm_metrics,
    )

    logger.info(
        "llm_optimization_active",
        budget_tokens_per_hour=config.llm.budget.max_tokens_per_hour,
        budget_calls_per_hour=config.llm.budget.max_calls_per_hour,
        cache_enabled=infra.prompt_cache is not None,
    )

    infra.embedding = create_embedding_client(config.embedding)

    # ── 4. Telemetry ──────────────────────────────────────────
    infra.metrics = MetricCollector(infra.tsdb)
    await infra.metrics.start_writer()

    return infra


async def close_infra(infra: InfraClients) -> None:
    """Close all infrastructure connections (Phase 2 of shutdown)."""

    async def _safe(name: str, coro: Any) -> None:
        try:
            await coro
        except Exception as e:
            logger.warning(f"{name}_close_failed", error=str(e))

    closers = [
        ("embedding", infra.embedding.close()),
        ("llm", infra.llm.close()),
    ]
    if infra.redis is not None:
        closers.append(("redis", infra.redis.close()))
    if infra.tsdb is not None:
        closers.append(("tsdb", infra.tsdb.close()))
    if infra.neo4j is not None:
        closers.append(("neo4j", infra.neo4j.close()))
    if infra.wallet is not None:
        closers.insert(0, ("wallet", infra.wallet.close()))

    import asyncio
    async with asyncio.timeout(1.0):
        await asyncio.gather(
            *[_safe(name, coro) for name, coro in closers],
            return_exceptions=True,
        )
