"""
EcodiaOS — Structured Logging

All logging via structlog. Every log entry includes system context.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from ecodiaos.config import LoggingConfig


# ─── SSE Log Broadcast ───────────────────────────────────────────────────────
# A lightweight in-process fanout: the SSELogHandler enqueues every log record
# into all active subscriber queues. The /api/v1/admin/logs/stream endpoint
# creates a queue per connection and removes it on disconnect.

_log_subscribers: list[asyncio.Queue[dict[str, Any]]] = []
_MAX_QUEUE = 500  # cap per-subscriber to avoid unbounded memory


def subscribe_logs() -> asyncio.Queue[dict[str, Any]]:
    """Register a new SSE subscriber and return its queue."""
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_MAX_QUEUE)
    _log_subscribers.append(q)
    return q


def unsubscribe_logs(q: asyncio.Queue[dict[str, Any]]) -> None:
    """Remove a subscriber queue (called when the SSE connection closes)."""
    try:
        _log_subscribers.remove(q)
    except ValueError:
        pass


def _format_record_time(record: logging.LogRecord) -> str:
    """ISO-8601 timestamp from the record's created float (no formatter needed)."""
    import datetime
    return (
        datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    )


class SSELogHandler(logging.Handler):
    """
    Standard-library logging handler that fans out each record to all
    active SSE subscriber queues as a structured dict.

    We deliberately use put_nowait + discard on full so that a slow browser
    never blocks the logging path.
    """

    def emit(self, record: logging.LogRecord) -> None:
        if not _log_subscribers:
            return

        # Build a lean dict — only the fields we want to stream
        entry: dict[str, Any] = {
            "ts": _format_record_time(record),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }

        # Structlog attaches extra fields as record.__dict__ keys
        skip = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for k, v in record.__dict__.items():
            if k not in skip and not k.startswith("_"):
                try:
                    # Only include JSON-serialisable primitives
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        entry[k] = v
                except Exception:
                    pass

        for q in list(_log_subscribers):
            try:
                q.put_nowait(entry)
            except asyncio.QueueFull:
                # Subscriber is too slow — drop oldest entry and retry
                try:
                    q.get_nowait()
                    q.put_nowait(entry)
                except Exception:
                    pass


def setup_logging(config: LoggingConfig, instance_id: str = "") -> None:
    """
    Configure structured logging for the entire application.
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if instance_id:
        shared_processors.insert(
            0,
            structlog.processors.CallsiteParameterAdder(
                parameters=[structlog.processors.CallsiteParameter.FUNC_NAME]
            ),
        )

    renderer: Any
    if config.format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    sse_handler = SSELogHandler()

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.addHandler(sse_handler)
    root_logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    # Quiet noisy libraries
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
