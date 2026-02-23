"""
Atune — Perception, Attention & Global Workspace.

EOS's sensory cortex and consciousness.  Receives all input, determines
what matters through seven-head salience scoring, and broadcasts the
winning content to all cognitive systems via the Global Workspace.
"""

from .service import AtuneConfig, AtuneService
from .types import (
    ActiveGoalSummary,
    Alert,
    AttentionContext,
    AtuneCache,
    EntityCandidate,
    ExtractionResult,
    InputChannel,
    LearnedPattern,
    MemoryContext,
    MetaContext,
    PendingDecision,
    PredictionError,
    PredictionErrorDirection,
    RawInput,
    RelationCandidate,
    RiskCategory,
    SalienceVector,
    SystemLoad,
    WorkspaceBroadcast,
    WorkspaceCandidate,
    WorkspaceContext,
    WorkspaceContribution,
)
from .workspace import BroadcastSubscriber, GlobalWorkspace

__all__ = [
    # Service
    "AtuneService",
    "AtuneConfig",
    # Workspace
    "GlobalWorkspace",
    "BroadcastSubscriber",
    # Types
    "ActiveGoalSummary",
    "Alert",
    "AttentionContext",
    "AtuneCache",
    "EntityCandidate",
    "ExtractionResult",
    "InputChannel",
    "LearnedPattern",
    "MemoryContext",
    "MetaContext",
    "PendingDecision",
    "PredictionError",
    "PredictionErrorDirection",
    "RawInput",
    "RelationCandidate",
    "RiskCategory",
    "SalienceVector",
    "SystemLoad",
    "WorkspaceBroadcast",
    "WorkspaceCandidate",
    "WorkspaceContext",
    "WorkspaceContribution",
]
