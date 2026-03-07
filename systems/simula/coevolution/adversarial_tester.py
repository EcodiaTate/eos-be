"""
EcodiaOS -- Adversarial Tester (Stage 6B.2 re-export)

Re-exports RobustnessTestGenerator under the canonical module path used by
tests and external consumers. The implementation lives in robustness_tester.py.

The adversarial tester generates perturbed variants of candidate mutations
to stress-test them: edge cases, boundary conditions, malformed inputs, and
exploits of known failure patterns. Tests that find bugs feed back into GRPO
training as hard negatives, closing the self-improvement loop.
"""

from systems.simula.coevolution.robustness_tester import RobustnessTestGenerator

__all__ = ["RobustnessTestGenerator"]
