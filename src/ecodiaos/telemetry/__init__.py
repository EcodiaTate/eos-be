"""
EcodiaOS — Observability Infrastructure

Structured logging, metrics collection, and tracing.
"""

from ecodiaos.telemetry.logging import setup_logging
from ecodiaos.telemetry.metrics import MetricCollector

__all__ = ["setup_logging", "MetricCollector"]
