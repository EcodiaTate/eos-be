"""
EcodiaOS — Evolution Primitives

Shared self-evolution types used across systems (Thymos, Simula, Evo).
Pure enum definitions with no cross-system dependencies — safe to import
from any system without violating the Synapse-only communication rule.

Richer types (ChangeSpec, EvolutionProposal, SimulationResult, ProposalResult)
remain in systems/simula/evolution_types.py because they depend on
Simula-internal models.
"""

from __future__ import annotations

import enum


class ChangeCategory(enum.StrEnum):
    ADD_EXECUTOR = "add_executor"
    ADD_INPUT_CHANNEL = "add_input_channel"
    ADD_PATTERN_DETECTOR = "add_pattern_detector"
    ADJUST_BUDGET = "adjust_budget"
    MODIFY_CONTRACT = "modify_contract"
    ADD_SYSTEM_CAPABILITY = "add_system_capability"
    MODIFY_CYCLE_TIMING = "modify_cycle_timing"
    CHANGE_CONSOLIDATION = "change_consolidation"
    # BUG_FIX: runtime errors that Simula can autonomously fix.
    BUG_FIX = "bug_fix"
    # CODE: generic code-level change (bounty solutions, external issue fixes).
    CODE = "code"
    # Constitutional amendment via formal amendment pipeline.
    CONSTITUTIONAL_AMENDMENT = "constitutional_amendment"
    MODIFY_EQUOR = "modify_equor"
    MODIFY_CONSTITUTION = "modify_constitution"
    MODIFY_INVARIANTS = "modify_invariants"
    MODIFY_SELF_EVOLUTION = "modify_self_evolution"


class ProposalStatus(enum.StrEnum):
    PROPOSED = "proposed"
    SIMULATING = "simulating"
    AWAITING_GOVERNANCE = "awaiting_governance"
    APPROVED = "approved"
    APPLYING = "applying"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"
