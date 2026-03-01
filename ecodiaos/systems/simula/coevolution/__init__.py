"""
EcodiaOS -- Simula Co-Evolving Agents (Stage 6B)

Autonomous failure case extraction from failure history and
continuous self-improvement via robustness test generation.
Feeds into GRPO training loop (Stage 4B).
"""

from ecodiaos.systems.simula.coevolution.robustness_tester import (
    RobustnessTestGenerator,
)
from ecodiaos.systems.simula.coevolution.failure_analyzer import FailureAnalyzer

__all__ = [
    "FailureAnalyzer",
    "RobustnessTestGenerator",
]
