"""
EcodiaOS — Oneiros: Base Worker ABC

Formal strategy pattern root for all Oneiros sleep workers (NREM, REM, Lucid).
Enables the NeuroplasticityBus to discover and hot-reload evolved worker
implementations without interrupting an active sleep cycle.

The per-category ABCs (BaseEpisodicReplay, BaseDreamGenerator, etc.) live
in their respective modules (nrem.py, rem.py, lucid.py) to avoid circular
imports with the result types defined there.  This module defines only the
root ABC and re-exports the category ABCs for convenience.
"""

from __future__ import annotations

from abc import ABC


class BaseOneirosWorker(ABC):
    """
    Root abstract base for every Oneiros sleep-cycle worker.

    Subclasses MUST set ``worker_type`` to a non-empty string so the
    NeuroplasticityBus qualifier can filter out the ABCs themselves.

    The ``run()`` contract is intentionally *not* defined here because
    NREM, REM, and Lucid workers have different signatures.  Instead,
    each category has its own ABC in its respective module.
    """

    worker_type: str = ""  # e.g. "nrem.episodic_replay", "rem.dream_generator"
