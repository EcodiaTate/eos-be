"""
EcodiaOS -- Simula Neurosymbolic Synthesis Subsystem (Stage 5A)

Fast-path alternatives to the expensive CEGIS agentic loop:
  - HySynth:      Probabilistic CFG bottom-up beam search (4x speedup target)
  - Sketch+Solve: LLM template + symbolic hole-filling via Z3
  - ChopChop:     Type-directed constrained generation (generate-then-verify)
  - Strategy Selector: Routes proposals to best-fit strategy
"""

from ecodiaos.systems.simula.synthesis.chopchop import ChopChopEngine
from ecodiaos.systems.simula.synthesis.hysynth import HySynthEngine
from ecodiaos.systems.simula.synthesis.sketch_solver import SketchSolver
from ecodiaos.systems.simula.synthesis.strategy_selector import (
    SynthesisStrategySelector,
)
from ecodiaos.systems.simula.synthesis.types import (
    CFGRule,
    ChopChopResult,
    GrammarConstraint,
    HoleKind,
    HySynthResult,
    SketchHole,
    SketchSolveResult,
    SketchTemplate,
    SynthesisResult,
    SynthesisSelectionReason,
    SynthesisStatus,
    SynthesisStrategy,
)

__all__ = [
    # Engines
    "HySynthEngine",
    "SketchSolver",
    "ChopChopEngine",
    "SynthesisStrategySelector",
    # Types
    "SynthesisStrategy",
    "SynthesisStatus",
    "HoleKind",
    "CFGRule",
    "HySynthResult",
    "SketchHole",
    "SketchTemplate",
    "SketchSolveResult",
    "GrammarConstraint",
    "ChopChopResult",
    "SynthesisSelectionReason",
    "SynthesisResult",
]
