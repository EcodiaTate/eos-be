"""
EcodiaOS — TimescaleDB Client

Async connection management for telemetry, metrics, and audit logs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import asyncpg
import structlog

if TYPE_CHECKING:
    from ecodiaos.config import TimescaleDBConfig

logger = structlog.get_logger()

# SQL for initialising the TimescaleDB schema on first boot.
INIT_SQL = """
-- Metrics hypertable
CREATE TABLE IF NOT EXISTS metrics (
    time        TIMESTAMPTZ NOT NULL,
    system      TEXT NOT NULL,
    metric      TEXT NOT NULL,
    value       DOUBLE PRECISION NOT NULL,
    labels      JSONB DEFAULT '{}'
);

SELECT create_hypertable('metrics', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_metrics_system_metric ON metrics (system, metric, time DESC);

-- Audit log (immutable)
CREATE TABLE IF NOT EXISTS audit_log (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    time        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    system      TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    intent_id   UUID,
    details     JSONB NOT NULL,
    affect      JSONB,
    checksum    TEXT NOT NULL
);

SELECT create_hypertable('audit_log', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_audit_system ON audit_log (system, time DESC);
CREATE INDEX IF NOT EXISTS idx_audit_type ON audit_log (event_type, time DESC);

-- Affect state history
CREATE TABLE IF NOT EXISTS affect_history (
    time              TIMESTAMPTZ NOT NULL,
    valence           DOUBLE PRECISION,
    arousal           DOUBLE PRECISION,
    dominance         DOUBLE PRECISION,
    curiosity         DOUBLE PRECISION,
    care_activation   DOUBLE PRECISION,
    coherence_stress  DOUBLE PRECISION,
    source_event      TEXT
);

SELECT create_hypertable('affect_history', 'time', if_not_exists => TRUE);

-- Cycle performance log
CREATE TABLE IF NOT EXISTS cycle_log (
    time            TIMESTAMPTZ NOT NULL,
    cycle_number    BIGINT NOT NULL,
    period_ms       INTEGER,
    actual_ms       INTEGER,
    broadcast       BOOLEAN,
    salience_max    DOUBLE PRECISION,
    systems_acked   INTEGER
);

SELECT create_hypertable('cycle_log', 'time', if_not_exists => TRUE);
"""


class TimescaleDBClient:
    """
    Async TimescaleDB client with connection pooling.
    Handles metrics, audit logs, and affect history.
    """

    def __init__(self, config: TimescaleDBConfig) -> None:
        self._config = config
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create connection pool and initialise schema."""
        self._pool = await asyncpg.create_pool(
            dsn=self._config.dsn,
            min_size=2,
            max_size=self._config.pool_size,
            ssl="require" if self._config.ssl else None,
        )
        logger.info(
            "timescaledb_connected",
            host=self._config.host,
            database=self._config.database,
        )
        # Initialise schema
        await self._init_schema()

    async def _init_schema(self) -> None:
        """Create tables and hypertables if they don't exist."""
        async with self.pool.acquire() as conn:
            # Execute each statement individually (hypertable creation can't be in a
            # multi-statement transaction easily, so we split them)
            for statement in INIT_SQL.split(";"):
                statement = statement.strip()
                if statement:
                    try:
                        await conn.execute(statement)
                    except Exception as e:
                        # Ignore "already exists" type errors
                        if "already exists" not in str(e).lower():
                            logger.warning("tsdb_init_statement_warning", error=str(e))
        logger.info("timescaledb_schema_initialised")

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("timescaledb_disconnected")

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("TimescaleDB client not connected. Call connect() first.")
        return self._pool

    async def health_check(self) -> dict[str, Any]:
        """Check connectivity."""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return {"status": "connected", "latency_ms": 0}
        except Exception as e:
            logger.error("tsdb_health_check_failed", error=str(e))
            return {"status": "disconnected", "error": str(e)}

    async def write_metrics(self, points: list[dict[str, Any]]) -> None:
        """Batch write metric points."""
        if not points:
            return
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO metrics (time, system, metric, value, labels)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                """,
                [
                    (p["time"], p["system"], p["metric"], p["value"],
                     str(p.get("labels", {})).replace("'", '"'))
                    for p in points
                ],
            )

    async def write_affect(self, state: dict[str, Any]) -> None:
        """Write a single affect state snapshot."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO affect_history
                    (time, valence, arousal, dominance, curiosity,
                     care_activation, coherence_stress, source_event)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                state.get("time"),
                state.get("valence", 0.0),
                state.get("arousal", 0.0),
                state.get("dominance", 0.0),
                state.get("curiosity", 0.0),
                state.get("care_activation", 0.0),
                state.get("coherence_stress", 0.0),
                state.get("source_event", ""),
            )
