"""
EcodiaOS — Thymos (System #12)

The immune system. Detects failures, diagnoses root causes, prescribes
repairs, maintains an antibody library of learned fixes, and prevents
future errors through prophylactic scanning and homeostatic regulation.

Every error becomes an Incident — the organism feels it break and heals itself.
"""

from ecodiaos.systems.thymos.antibody import AntibodyLibrary
from ecodiaos.systems.thymos.diagnosis import (
    CausalAnalyzer,
    DiagnosticEngine,
    TemporalCorrelator,
)
from ecodiaos.systems.thymos.governor import HealingGovernor
from ecodiaos.systems.thymos.prescription import RepairPrescriber, RepairValidator
from ecodiaos.systems.thymos.prophylactic import HomeostasisController, ProphylacticScanner
from ecodiaos.systems.thymos.sentinels import (
    BankruptcySentinel,
    BaseThymosSentinel,
    CognitiveStallSentinel,
    ContractSentinel,
    DriftSentinel,
    ExceptionSentinel,
    FeedbackLoopSentinel,
)
from ecodiaos.systems.thymos.service import ThymosService
from ecodiaos.systems.thymos.triage import (
    IncidentDeduplicator,
    ResponseRouter,
    SeverityScorer,
)
from ecodiaos.systems.thymos.types import (
    Antibody,
    CausalChain,
    ContractSLA,
    Diagnosis,
    DiagnosticEvidence,
    DiagnosticHypothesis,
    DiagnosticTestResult,
    DriftConfig,
    FeedbackLoop,
    HealingBudgetState,
    HealingMode,
    Incident,
    IncidentClass,
    IncidentSeverity,
    ParameterAdjustment,
    ParameterFix,
    ProphylacticWarning,
    RepairSpec,
    RepairStatus,
    RepairTier,
    StallConfig,
    TemporalCorrelation,
    ThymosHealthSnapshot,
    ValidationResult,
)

__all__ = [
    # Service
    "ThymosService",
    # Sub-systems
    "AntibodyLibrary",
    "BankruptcySentinel",
    "BaseThymosSentinel",
    "CausalAnalyzer",
    "CognitiveStallSentinel",
    "ContractSentinel",
    "DiagnosticEngine",
    "DriftSentinel",
    "ExceptionSentinel",
    "FeedbackLoopSentinel",
    "HealingGovernor",
    "HomeostasisController",
    "IncidentDeduplicator",
    "ProphylacticScanner",
    "RepairPrescriber",
    "RepairValidator",
    "ResponseRouter",
    "SeverityScorer",
    "TemporalCorrelator",
    # Types — Enums
    "HealingMode",
    "IncidentClass",
    "IncidentSeverity",
    "RepairStatus",
    "RepairTier",
    # Types — Models
    "Antibody",
    "CausalChain",
    "ContractSLA",
    "Diagnosis",
    "DiagnosticEvidence",
    "DiagnosticHypothesis",
    "DiagnosticTestResult",
    "DriftConfig",
    "FeedbackLoop",
    "HealingBudgetState",
    "Incident",
    "ParameterAdjustment",
    "ParameterFix",
    "ProphylacticWarning",
    "RepairSpec",
    "StallConfig",
    "TemporalCorrelation",
    "ThymosHealthSnapshot",
    "ValidationResult",
]
