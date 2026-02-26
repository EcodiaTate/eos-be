"""
EcodiaOS -- Simula Co-Evolving Agents (Stage 6B)

Autonomous hard negative generation from failure history and
continuous self-improvement via adversarial test generation.
Feeds into GRPO training loop (Stage 4B).
"""

from ecodiaos.systems.simula.coevolution.adversarial_tester import (
    AdversarialTestGenerator,
)
from ecodiaos.systems.simula.coevolution.hard_negative_miner import HardNegativeMiner

__all__ = [
    "HardNegativeMiner",
    "AdversarialTestGenerator",
]
