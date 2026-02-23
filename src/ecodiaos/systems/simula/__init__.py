"""
EcodiaOS — Simula: Self-Evolution System

The organism's capacity for metamorphosis. Where Evo adjusts the knobs,
Simula redesigns the dashboard.

Public API:
  SimulaService              — main service, wired in main.py
  EvoSimulaBridge            — translates Evo proposals to Simula format
  EvolutionAnalyticsEngine   — evolution quality tracking
  ProposalIntelligence       — dedup, prioritize, dependency analysis
  EvolutionProposal          — submitted by Evo when a hypothesis reaches SUPPORTED
  ProposalResult             — outcome of process_proposal()
  ChangeCategory             — taxonomy of allowed (and forbidden) change types
  ChangeSpec                 — formal specification of what to change
  EnrichedSimulationResult   — deep multi-strategy simulation output
"""

from ecodiaos.systems.simula.service import SimulaService
from ecodiaos.systems.simula.analytics import EvolutionAnalyticsEngine
from ecodiaos.systems.simula.bridge import EvoSimulaBridge
from ecodiaos.systems.simula.proposal_intelligence import ProposalIntelligence
from ecodiaos.systems.simula.types import (
    CategorySuccessRate,
    ChangeCategory,
    ChangeSpec,
    ConfigVersion,
    CounterfactualResult,
    DependencyImpact,
    EnrichedSimulationResult,
    EvolutionAnalytics,
    EvolutionProposal,
    EvolutionRecord,
    EvoProposalEnriched,
    ProposalCluster,
    ProposalPriority,
    ProposalResult,
    ProposalStatus,
    ResourceCostEstimate,
    RiskLevel,
    SimulationResult,
    FORBIDDEN,
    GOVERNANCE_REQUIRED,
    SELF_APPLICABLE,
    SIMULA_IRON_RULES,
)

__all__ = [
    # Services
    "SimulaService",
    "EvoSimulaBridge",
    "EvolutionAnalyticsEngine",
    "ProposalIntelligence",
    # Core types
    "ChangeCategory",
    "ChangeSpec",
    "ConfigVersion",
    "EvolutionProposal",
    "EvolutionRecord",
    "ProposalResult",
    "ProposalStatus",
    "RiskLevel",
    "SimulationResult",
    # Enriched types
    "EnrichedSimulationResult",
    "CounterfactualResult",
    "DependencyImpact",
    "ResourceCostEstimate",
    "EvoProposalEnriched",
    "ProposalPriority",
    "ProposalCluster",
    "CategorySuccessRate",
    "EvolutionAnalytics",
    # Constants
    "FORBIDDEN",
    "GOVERNANCE_REQUIRED",
    "SELF_APPLICABLE",
    "SIMULA_IRON_RULES",
]
