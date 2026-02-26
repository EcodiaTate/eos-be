"""
EcodiaOS — Simula Hunter: Zero-Day Discovery Engine

Hunter inverts Simula's internal verification logic — instead of proving
code is *correct*, it proves code is *exploitable* by translating Z3 SAT
counterexamples into weaponized exploit proofs-of-concept.

Public API:
  TargetWorkspace      — abstraction for internal or external codebases
  TargetType           — INTERNAL_EOS | EXTERNAL_REPO
  AttackSurface        — discovered exploitable entry point
  VulnerabilityReport  — proven vulnerability with Z3 counterexample + PoC
  HuntResult           — aggregated results from a full hunt
  HunterConfig         — authorization and resource limits
"""

from ecodiaos.systems.simula.hunter.workspace import TargetWorkspace
from ecodiaos.systems.simula.hunter.types import (
    AttackSurface,
    AttackSurfaceType,
    HunterConfig,
    HuntResult,
    TargetType,
    VulnerabilityReport,
    VulnerabilitySeverity,
)

__all__ = [
    "TargetWorkspace",
    "TargetType",
    "AttackSurface",
    "AttackSurfaceType",
    "VulnerabilityReport",
    "VulnerabilitySeverity",
    "HuntResult",
    "HunterConfig",
]
