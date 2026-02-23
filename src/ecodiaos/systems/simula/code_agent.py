"""
EcodiaOS — Simula Code Implementation Agent

The SimulaCodeAgent is Simula's most powerful capability: an agentic
Claude-backed engine that reads the EOS codebase, generates code for
structural changes, writes the files, and verifies correctness.

This is functionally equivalent to Claude Code, embedded within EOS
itself, operating under Simula's constitutional constraints:
  - Cannot write to forbidden paths (equor, simula, constitution, invariants)
  - Cannot exceed max_turns without completing
  - All writes are intercepted and tracked for rollback
  - The system prompt includes the full change spec + relevant EOS conventions

Tool suite (11 tools):
  read_file         — Read a file from the codebase
  write_file        — Write or create a file (tracked for rollback)
  diff_file         — Apply a targeted find/replace edit to a file
  list_directory    — List files and subdirectories
  search_code       — Search for patterns across Python files
  run_tests         — Run pytest on a specific path
  run_linter        — Run ruff on a specific path
  type_check        — Run mypy for type safety verification
  dependency_graph  — Show module imports and importers
  read_spec         — Read EcodiaOS specification documents
  find_similar      — Find existing implementations as pattern exemplars

Architecture: agentic tool-use loop
  1. Build architecture-aware system prompt (change spec + exemplar code + spec context + iron rules)
  2. Prepend planning instruction for multi-file reasoning
  3. Call LLM with tools
  4. Execute any tool calls (all 11 tools available)
  5. Feed tool results back as the next message
  6. Repeat until stop_reason == "end_turn" or max_turns exceeded
  7. Return CodeChangeResult with all files written and summary
"""

from __future__ import annotations

import ast
import asyncio
import subprocess
from pathlib import Path
from typing import Any, Callable

import structlog

from ecodiaos.clients.llm import (
    LLMProvider,
    ToolAwareResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from ecodiaos.systems.simula.types import (
    ChangeCategory,
    CodeChangeResult,
    EvolutionProposal,
)

logger = structlog.get_logger()

# ─── Tool Definitions ────────────────────────────────────────────────────────

SIMULA_AGENT_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="read_file",
        description=(
            "Read a file from the EcodiaOS codebase. "
            "Use this to understand existing code, conventions, and patterns "
            "before implementing your change."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path from codebase root",
                }
            },
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="write_file",
        description=(
            "Write or create a file in the EcodiaOS codebase. "
            "All writes are tracked for rollback. "
            "Forbidden paths (equor, simula, constitutional) will be rejected. "
            "Prefer diff_file for modifying existing files."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path from codebase root"},
                "content": {"type": "string", "description": "Complete file content to write"},
            },
            "required": ["path", "content"],
        },
    ),
    ToolDefinition(
        name="diff_file",
        description=(
            "Apply a targeted find-and-replace edit to an existing file. "
            "More precise than write_file for modifications — only changes "
            "the specified text, preserving everything else. The 'find' text "
            "must be an exact match of existing content."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path from codebase root"},
                "find": {"type": "string", "description": "Exact text to find in the file"},
                "replace": {"type": "string", "description": "Text to replace it with"},
            },
            "required": ["path", "find", "replace"],
        },
    ),
    ToolDefinition(
        name="list_directory",
        description="List files and subdirectories at a given path in the codebase.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative path from codebase root"}},
        },
    ),
    ToolDefinition(
        name="search_code",
        description=(
            "Search for a pattern across codebase Python files. "
            "Returns matching lines with file paths and line numbers. "
            "Use this to find existing patterns, class names, or function signatures."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "String pattern to search for (case-sensitive)"},
                "directory": {"type": "string", "description": "Directory to search in (default: src/)"},
            },
            "required": ["pattern"],
        },
    ),
    ToolDefinition(
        name="run_tests",
        description=(
            "Run the pytest test suite for a specific path. "
            "Use this to verify your implementation is correct before finishing."
        ),
        input_schema={
            "type": "object",
            "properties": {"test_path": {"type": "string", "description": "Test path relative to codebase root"}},
            "required": ["test_path"],
        },
    ),
    ToolDefinition(
        name="run_linter",
        description=(
            "Run ruff linter on a path to check for code style issues. "
            "Run this on your written files before finishing."
        ),
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to lint"}},
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="type_check",
        description=(
            "Run mypy type checker on a file or directory. "
            "Use after writing code to verify type safety. "
            "EcodiaOS requires mypy --strict compliance."
        ),
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to type-check"}},
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="dependency_graph",
        description=(
            "Show what a Python module imports and what other modules import it. "
            "Use this before modifying files to understand blast radius and "
            "ensure your changes don't break downstream consumers."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "module_path": {
                    "type": "string",
                    "description": "Python file path relative to codebase root",
                },
            },
            "required": ["module_path"],
        },
    ),
    ToolDefinition(
        name="read_spec",
        description=(
            "Read an EcodiaOS specification document to understand the "
            "design intent, interfaces, and constraints for a system. "
            "Always read the relevant spec before implementing changes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "spec_name": {
                    "type": "string",
                    "description": (
                        "Spec name: 'identity', 'architecture', 'infrastructure', "
                        "'memory', 'equor', 'atune', 'voxis', 'nova', 'axon', "
                        "'evo', 'simula', 'synapse', 'alive', 'federation'"
                    ),
                },
            },
            "required": ["spec_name"],
        },
    ),
    ToolDefinition(
        name="find_similar",
        description=(
            "Find existing implementations similar to what you need to build. "
            "Returns relevant code examples from the codebase that you should "
            "study and follow as patterns. Always use this before writing new "
            "code to ensure convention compliance."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "What you're looking for (e.g., 'executor implementation', "
                        "'pattern detector', 'service initialization')"
                    ),
                },
            },
            "required": ["description"],
        },
    ),
]

# Spec name → file path mapping
_SPEC_FILE_MAP: dict[str, str] = {
    "identity": ".claude/EcodiaOS_Identity_Document.md",
    "architecture": ".claude/EcodiaOS_System_Architecture_Overview.md",
    "infrastructure": ".claude/EcodiaOS_Infrastructure_Architecture.md",
    "memory": ".claude/EcodiaOS_Spec_01_Memory_Identity_Core.md",
    "equor": ".claude/EcodiaOS_Spec_02_Equor.md",
    "atune": ".claude/EcodiaOS_Spec_03_Atune.md",
    "voxis": ".claude/EcodiaOS_Spec_04_Voxis.md",
    "nova": ".claude/EcodiaOS_Spec_05_Nova.md",
    "axon": ".claude/EcodiaOS_Spec_06_Axon.md",
    "evo": ".claude/EcodiaOS_Spec_07_Evo.md",
    "simula": ".claude/EcodiaOS_Spec_08_Simula.md",
    "synapse": ".claude/EcodiaOS_Spec_09_Synapse.md",
    "alive": ".claude/EcodiaOS_Spec_10_Alive.md",
    "federation": ".claude/EcodiaOS_Spec_11_Federation.md",
}

# Keyword → file path mapping for find_similar
_SIMILAR_CODE_MAP: dict[str, list[str]] = {
    "executor": [
        "src/ecodiaos/systems/axon/executors/",
        "src/ecodiaos/systems/axon/executor.py",
    ],
    "pattern detector": [
        "src/ecodiaos/systems/evo/detectors.py",
    ],
    "detector": [
        "src/ecodiaos/systems/evo/detectors.py",
    ],
    "input channel": [
        "src/ecodiaos/systems/atune/",
    ],
    "channel": [
        "src/ecodiaos/systems/atune/",
    ],
    "service": [
        "src/ecodiaos/systems/axon/service.py",
        "src/ecodiaos/systems/evo/service.py",
    ],
    "hypothesis": [
        "src/ecodiaos/systems/evo/hypothesis.py",
    ],
    "consolidation": [
        "src/ecodiaos/systems/evo/consolidation.py",
    ],
    "parameter": [
        "src/ecodiaos/systems/evo/parameter_tuner.py",
    ],
    "primitives": [
        "src/ecodiaos/primitives/common.py",
        "src/ecodiaos/primitives/memory_trace.py",
    ],
}

# ─── System Prompt ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """You are Simula's Code Implementation Agent — the autonomous part of EcodiaOS
that implements approved structural changes to the codebase.

## Your Task
Category: {category}
Description: {description}
Expected benefit: {expected_benefit}
Evidence: {evidence}

## EcodiaOS Coding Conventions
- Python 3.12+, async-native throughout
- Pydantic v2 for all data models (use EOSBaseModel from ecodiaos.primitives.common)
- structlog for logging: logger = structlog.get_logger(), bound with system name
- Type hints on everything — mypy --strict clean
- from __future__ import annotations at top of every .py file
- New executors: inherit from Executor (ecodiaos.systems.axon.executor),
  set action_type class var, implement execute()
- New input channels: register in Atune's InputChannel registry
- New pattern detectors: inherit from PatternDetector (ecodiaos.systems.evo.detectors),
  implement scan()
- NEVER import directly between systems — all inter-system data uses shared
  primitives from ecodiaos.primitives/

## Iron Rules (ABSOLUTE — never violate)
{iron_rules}

## Forbidden Write Paths (write_file and diff_file will reject these)
{forbidden_paths}

## Architecture Context
{architecture_context}

## Process
1. First, use find_similar to study an existing implementation that matches your task
2. Use read_spec to understand the design intent for the affected system
3. Use dependency_graph on files you plan to modify to understand blast radius
4. Plan your approach: list every file you'll create or modify and why
5. Implement following conventions exactly — match the style of similar code
6. Run run_linter on every file you write or modify
7. Run type_check on your written files to verify type safety
8. Run run_tests if a test directory exists for the affected system
9. When everything passes, stop calling tools

Be thorough, follow existing patterns exactly, and produce production-quality code.
Prefer diff_file over write_file when modifying existing files."""


def _build_architecture_context(
    category: ChangeCategory, codebase_root: Path,
) -> str:
    """
    Build rich architecture context for the system prompt.
    Reads actual spec sections and existing implementations as exemplars.
    Max 6000 chars to stay within token budget.
    Priority: exemplar code > spec text > API surface.
    """
    context_parts: list[str] = []
    budget_remaining = 6000

    # 1. Load relevant spec section summary
    spec_map: dict[ChangeCategory, str] = {
        ChangeCategory.ADD_EXECUTOR: "axon",
        ChangeCategory.ADD_INPUT_CHANNEL: "atune",
        ChangeCategory.ADD_PATTERN_DETECTOR: "evo",
        ChangeCategory.ADJUST_BUDGET: "architecture",
        ChangeCategory.MODIFY_CONTRACT: "architecture",
        ChangeCategory.ADD_SYSTEM_CAPABILITY: "architecture",
        ChangeCategory.MODIFY_CYCLE_TIMING: "synapse",
        ChangeCategory.CHANGE_CONSOLIDATION: "evo",
    }
    spec_name = spec_map.get(category, "architecture")
    spec_file = _SPEC_FILE_MAP.get(spec_name)
    if spec_file:
        spec_path = codebase_root / spec_file
        if spec_path.exists():
            try:
                spec_text = spec_path.read_text(encoding="utf-8")[:2000]
                context_parts.append(f"### Relevant Specification ({spec_name})\n{spec_text}")
                budget_remaining -= len(context_parts[-1])
            except Exception:
                pass

    # 2. Load exemplar code for the category
    exemplar_map: dict[ChangeCategory, str] = {
        ChangeCategory.ADD_EXECUTOR: "src/ecodiaos/systems/axon/executor.py",
        ChangeCategory.ADD_INPUT_CHANNEL: "src/ecodiaos/systems/atune/service.py",
        ChangeCategory.ADD_PATTERN_DETECTOR: "src/ecodiaos/systems/evo/detectors.py",
    }
    exemplar_path_str = exemplar_map.get(category)
    if exemplar_path_str and budget_remaining > 500:
        exemplar_path = codebase_root / exemplar_path_str
        if exemplar_path.exists():
            try:
                exemplar_text = exemplar_path.read_text(encoding="utf-8")
                # Take the first chunk that fits the budget
                chunk = exemplar_text[:min(2500, budget_remaining - 100)]
                context_parts.append(
                    f"### Exemplar Implementation ({exemplar_path_str})\n"
                    f"Study this code and follow its patterns exactly:\n```python\n{chunk}\n```"
                )
                budget_remaining -= len(context_parts[-1])
            except Exception:
                pass

    # 3. Load the target system's __init__.py for API awareness
    system_map: dict[ChangeCategory, str] = {
        ChangeCategory.ADD_EXECUTOR: "src/ecodiaos/systems/axon/__init__.py",
        ChangeCategory.ADD_INPUT_CHANNEL: "src/ecodiaos/systems/atune/__init__.py",
        ChangeCategory.ADD_PATTERN_DETECTOR: "src/ecodiaos/systems/evo/__init__.py",
    }
    init_path_str = system_map.get(category)
    if init_path_str and budget_remaining > 200:
        init_path = codebase_root / init_path_str
        if init_path.exists():
            try:
                init_text = init_path.read_text(encoding="utf-8")[:min(800, budget_remaining - 50)]
                context_parts.append(
                    f"### System API Surface ({init_path_str})\n```python\n{init_text}\n```"
                )
            except Exception:
                pass

    if not context_parts:
        return "See EcodiaOS specification documents in .claude/ (use read_spec tool)"

    return "\n\n".join(context_parts)


class SimulaCodeAgent:
    """
    Agentic code generation engine for Simula.

    Given an EvolutionProposal, uses Claude with 11 file-system and
    analysis tools to:
      1. Study existing similar code for pattern compliance
      2. Read relevant specs for design intent
      3. Analyze dependency graphs for blast radius
      4. Plan the implementation approach
      5. Generate correct, convention-following implementation
      6. Write files (tracked for rollback)
      7. Verify with linter, type checker, and tests
      8. Return CodeChangeResult
    """

    def __init__(
        self,
        llm: LLMProvider,
        codebase_root: Path,
        max_turns: int = 20,
    ) -> None:
        self._llm = llm
        self._root = codebase_root.resolve()
        self._max_turns = max_turns
        self._logger = logger.bind(system="simula.code_agent")
        self._files_written: list[str] = []
        self._total_tokens_used: int = 0

    async def implement(self, proposal: EvolutionProposal) -> CodeChangeResult:
        """
        Main entry point. Runs the agentic loop to implement the proposal.
        Returns CodeChangeResult with all files written and outcome.
        """
        self._files_written = []
        self._total_tokens_used = 0

        system_prompt = self._build_system_prompt(proposal)

        # Prepend a planning instruction to encourage multi-file reasoning
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Please implement this change: {proposal.description}\n\n"
                    f"Change spec details: {proposal.change_spec.model_dump_json(indent=2)}\n\n"
                    "IMPORTANT: Before writing any code, first:\n"
                    "1. Use find_similar to study an existing implementation like what you need to build\n"
                    "2. Use read_spec for the affected system to understand design intent\n"
                    "3. List every file you plan to create or modify and explain your approach\n"
                    "4. Then implement, lint, type-check, and test."
                ),
            }
        ]

        turns = 0
        last_text = ""

        self._logger.info(
            "code_agent_starting",
            proposal_id=proposal.id,
            category=proposal.category.value,
            max_turns=self._max_turns,
            tools_available=len(SIMULA_AGENT_TOOLS),
        )

        while turns < self._max_turns:
            turns += 1

            try:
                response = await self._llm.generate_with_tools(
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=SIMULA_AGENT_TOOLS,
                    max_tokens=8192,
                    temperature=0.2,
                )
            except Exception as exc:
                self._logger.error("llm_call_failed", turn=turns, error=str(exc))
                return CodeChangeResult(
                    success=False,
                    files_written=self._files_written,
                    error=f"LLM call failed on turn {turns}: {exc}",
                )

            # Track token budget
            self._total_tokens_used += getattr(response, "total_tokens", 0)
            last_text = response.text

            if not response.has_tool_calls:
                self._logger.info(
                    "code_agent_done",
                    turns=turns,
                    files_written=len(self._files_written),
                    stop_reason=response.stop_reason,
                    total_tokens=self._total_tokens_used,
                )
                break

            # Build assistant message with text + tool_use blocks
            assistant_content: list[dict[str, Any]] = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute all tool calls
            tool_results: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                result = await self._execute_tool(tc)
                tool_results.append(result.to_anthropic_dict())
                self._logger.debug(
                    "tool_executed",
                    tool=tc.name,
                    is_error=result.is_error,
                    turn=turns,
                )

            messages.append({"role": "user", "content": tool_results})

        else:
            self._logger.warning(
                "code_agent_max_turns_exceeded",
                max_turns=self._max_turns,
                files_written=len(self._files_written),
                total_tokens=self._total_tokens_used,
            )
            return CodeChangeResult(
                success=len(self._files_written) > 0,
                files_written=self._files_written,
                summary=last_text[:500] if last_text else "Max turns exceeded",
                error="Max turns exceeded without completion signal",
            )

        return CodeChangeResult(
            success=len(self._files_written) > 0,
            files_written=self._files_written,
            summary=last_text[:1000] if last_text else "Change implemented",
        )

    # ─── Tool Dispatch ───────────────────────────────────────────────────────

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Dispatch a tool call to the appropriate implementation."""
        try:
            match tool_call.name:
                case "read_file":
                    return await self._tool_read_file(tool_call)
                case "write_file":
                    return await self._tool_write_file(tool_call)
                case "diff_file":
                    return await self._tool_diff_file(tool_call)
                case "list_directory":
                    return await self._tool_list_directory(tool_call)
                case "search_code":
                    return await self._tool_search_code(tool_call)
                case "run_tests":
                    return await self._tool_run_tests(tool_call)
                case "run_linter":
                    return await self._tool_run_linter(tool_call)
                case "type_check":
                    return await self._tool_type_check(tool_call)
                case "dependency_graph":
                    return await self._tool_dependency_graph(tool_call)
                case "read_spec":
                    return await self._tool_read_spec(tool_call)
                case "find_similar":
                    return await self._tool_find_similar(tool_call)
                case _:
                    return ToolResult(
                        tool_use_id=tool_call.id,
                        content=f"Unknown tool: {tool_call.name}",
                        is_error=True,
                    )
        except Exception as exc:
            return ToolResult(
                tool_use_id=tool_call.id,
                content=f"Tool execution error: {exc}",
                is_error=True,
            )

    # ─── Original Tools (upgraded) ───────────────────────────────────────────

    async def _tool_read_file(self, tc: ToolCall) -> ToolResult:
        rel_path = tc.input.get("path", "")
        target = (self._root / rel_path).resolve()
        if not str(target).startswith(str(self._root)):
            return ToolResult(tc.id, "Access denied: path outside codebase root", True)
        try:
            content = target.read_text(encoding="utf-8")
            return ToolResult(tc.id, content)
        except FileNotFoundError:
            return ToolResult(tc.id, f"File not found: {rel_path}", True)
        except Exception as exc:
            return ToolResult(tc.id, f"Read error: {exc}", True)

    async def _tool_write_file(self, tc: ToolCall) -> ToolResult:
        rel_path = tc.input.get("path", "")
        content = tc.input.get("content", "")
        forbidden_check = self._check_forbidden_path(rel_path)
        if forbidden_check:
            return ToolResult(tc.id, forbidden_check, True)
        target = (self._root / rel_path).resolve()
        if not str(target).startswith(str(self._root)):
            return ToolResult(tc.id, "Access denied: path outside codebase root", True)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            self._files_written.append(rel_path)
            return ToolResult(tc.id, f"Written: {rel_path} ({len(content)} bytes)")
        except Exception as exc:
            return ToolResult(tc.id, f"Write error: {exc}", True)

    async def _tool_list_directory(self, tc: ToolCall) -> ToolResult:
        rel_path = tc.input.get("path", "")
        target = (self._root / rel_path).resolve() if rel_path else self._root
        if not str(target).startswith(str(self._root)):
            return ToolResult(tc.id, "Access denied", True)
        try:
            if not target.exists():
                return ToolResult(tc.id, f"Directory not found: {rel_path}", True)
            entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
            lines = []
            for entry in entries:
                prefix = "  " if entry.is_file() else "D "
                lines.append(f"{prefix}{entry.name}")
            return ToolResult(tc.id, "\n".join(lines) or "(empty)")
        except Exception as exc:
            return ToolResult(tc.id, f"List error: {exc}", True)

    async def _tool_search_code(self, tc: ToolCall) -> ToolResult:
        pattern = tc.input.get("pattern", "")
        directory = tc.input.get("directory", "src/")
        search_root = (self._root / directory).resolve()
        if not str(search_root).startswith(str(self._root)):
            return ToolResult(tc.id, "Access denied", True)
        results: list[str] = []
        try:
            proc = await asyncio.create_subprocess_exec(
                "grep", "-rn", "--include=*.py", pattern, str(search_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            output = stdout.decode("utf-8", errors="replace")
            for line in output.splitlines()[:50]:
                results.append(line.replace(str(self._root) + "/", "").replace(str(self._root) + "\\", ""))
            return ToolResult(tc.id, "\n".join(results) if results else "No matches found")
        except asyncio.TimeoutError:
            return ToolResult(tc.id, "Search timed out", True)
        except Exception as exc:
            return ToolResult(tc.id, f"Search error: {exc}", True)

    async def _tool_run_tests(self, tc: ToolCall) -> ToolResult:
        test_path = tc.input.get("test_path", "")
        target = (self._root / test_path).resolve()
        if not str(target).startswith(str(self._root)):
            return ToolResult(tc.id, "Access denied", True)
        if not target.exists():
            return ToolResult(tc.id, f"Test path not found: {test_path}")
        try:
            proc = await asyncio.create_subprocess_exec(
                "pytest", str(target), "-x", "--tb=short", "-q",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self._root),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            output = stdout.decode("utf-8", errors="replace")
            passed = proc.returncode == 0
            return ToolResult(
                tc.id,
                f"{'PASSED' if passed else 'FAILED'}\n{output[-2000:]}",
                is_error=not passed,
            )
        except asyncio.TimeoutError:
            return ToolResult(tc.id, "Tests timed out after 60s", True)
        except Exception as exc:
            return ToolResult(tc.id, f"Test run error: {exc}", True)

    async def _tool_run_linter(self, tc: ToolCall) -> ToolResult:
        path = tc.input.get("path", "")
        target = (self._root / path).resolve()
        if not str(target).startswith(str(self._root)):
            return ToolResult(tc.id, "Access denied", True)
        try:
            proc = await asyncio.create_subprocess_exec(
                "ruff", "check", str(target),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self._root),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
            output = stdout.decode("utf-8", errors="replace")
            passed = proc.returncode == 0
            return ToolResult(
                tc.id,
                f"{'CLEAN' if passed else 'ISSUES FOUND'}\n{output}" if output else "CLEAN",
            )
        except asyncio.TimeoutError:
            return ToolResult(tc.id, "Linter timed out", True)
        except Exception as exc:
            return ToolResult(tc.id, f"Linter error: {exc}", True)

    # ─── New Tools ───────────────────────────────────────────────────────────

    async def _tool_diff_file(self, tc: ToolCall) -> ToolResult:
        """Apply a targeted find/replace edit to a file."""
        rel_path = tc.input.get("path", "")
        find_text = tc.input.get("find", "")
        replace_text = tc.input.get("replace", "")

        forbidden_check = self._check_forbidden_path(rel_path)
        if forbidden_check:
            return ToolResult(tc.id, forbidden_check, True)

        target = (self._root / rel_path).resolve()
        if not str(target).startswith(str(self._root)):
            return ToolResult(tc.id, "Access denied: path outside codebase root", True)
        if not target.exists():
            return ToolResult(tc.id, f"File not found: {rel_path}", True)

        try:
            content = target.read_text(encoding="utf-8")

            if find_text not in content:
                return ToolResult(
                    tc.id,
                    f"Find text not found in {rel_path}. "
                    "Ensure the 'find' parameter is an exact match of existing content.",
                    True,
                )

            occurrences = content.count(find_text)
            if occurrences > 1:
                return ToolResult(
                    tc.id,
                    f"Find text matches {occurrences} locations in {rel_path}. "
                    "Provide more surrounding context to make the match unique.",
                    True,
                )

            new_content = content.replace(find_text, replace_text, 1)
            target.write_text(new_content, encoding="utf-8")

            if rel_path not in self._files_written:
                self._files_written.append(rel_path)

            # Build a readable diff summary
            find_lines = find_text.count("\n") + 1
            replace_lines = replace_text.count("\n") + 1
            return ToolResult(
                tc.id,
                f"Edited {rel_path}: replaced {find_lines} line(s) with {replace_lines} line(s)",
            )
        except Exception as exc:
            return ToolResult(tc.id, f"Diff error: {exc}", True)

    async def _tool_type_check(self, tc: ToolCall) -> ToolResult:
        """Run mypy type checker on a path."""
        path = tc.input.get("path", "")
        target = (self._root / path).resolve()
        if not str(target).startswith(str(self._root)):
            return ToolResult(tc.id, "Access denied", True)
        try:
            proc = await asyncio.create_subprocess_exec(
                "mypy", str(target), "--strict", "--no-error-summary",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self._root),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            output = stdout.decode("utf-8", errors="replace")
            passed = proc.returncode == 0
            if passed:
                return ToolResult(tc.id, "TYPE CHECK PASSED — no issues found")
            return ToolResult(
                tc.id,
                f"TYPE CHECK ISSUES:\n{output[-2000:]}",
                is_error=True,
            )
        except asyncio.TimeoutError:
            return ToolResult(tc.id, "Type check timed out after 30s", True)
        except FileNotFoundError:
            return ToolResult(tc.id, "mypy not found — type checking unavailable")
        except Exception as exc:
            return ToolResult(tc.id, f"Type check error: {exc}", True)

    async def _tool_dependency_graph(self, tc: ToolCall) -> ToolResult:
        """Show what a module imports and what imports it."""
        module_path = tc.input.get("module_path", "")
        target = (self._root / module_path).resolve()
        if not str(target).startswith(str(self._root)):
            return ToolResult(tc.id, "Access denied", True)
        if not target.exists():
            return ToolResult(tc.id, f"File not found: {module_path}", True)

        try:
            source = target.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=module_path)
        except Exception as exc:
            return ToolResult(tc.id, f"Parse error: {exc}", True)

        # Extract this module's imports
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names = ", ".join(a.name for a in (node.names or []))
                    imports.append(f"from {node.module} import {names}")

        # Find files that import this module
        module_name = self._path_to_module(module_path)
        importers: list[str] = []
        if module_name:
            src_dir = self._root / "src"
            if src_dir.exists():
                short_parts = module_name.split(".")
                # Search for imports of this module
                for py_file in src_dir.rglob("*.py"):
                    if py_file.resolve() == target:
                        continue
                    try:
                        file_source = py_file.read_text(encoding="utf-8")
                        # Quick string check before expensive parse
                        if module_name not in file_source and short_parts[-1] not in file_source:
                            continue
                        file_tree = ast.parse(file_source)
                        for node in ast.walk(file_tree):
                            if isinstance(node, ast.ImportFrom) and node.module:
                                if module_name in node.module or (
                                    ".".join(short_parts[:-1]) in node.module
                                    and any(a.name == short_parts[-1] for a in (node.names or []))
                                ):
                                    importers.append(str(py_file.relative_to(self._root)))
                                    break
                            elif isinstance(node, ast.Import):
                                for alias in node.names:
                                    if module_name in alias.name:
                                        importers.append(str(py_file.relative_to(self._root)))
                                        break
                    except Exception:
                        continue

        lines = [f"=== Dependency Graph for {module_path} ===\n"]
        lines.append(f"Module: {module_name or 'unknown'}\n")
        lines.append(f"--- This module imports ({len(imports)}) ---")
        for imp in imports:
            lines.append(f"  {imp}")
        lines.append(f"\n--- Imported by ({len(importers)}) ---")
        for imp in importers:
            lines.append(f"  {imp}")

        return ToolResult(tc.id, "\n".join(lines))

    async def _tool_read_spec(self, tc: ToolCall) -> ToolResult:
        """Read an EcodiaOS specification document."""
        spec_name = tc.input.get("spec_name", "").lower().strip()
        spec_file = _SPEC_FILE_MAP.get(spec_name)

        if spec_file is None:
            available = ", ".join(sorted(_SPEC_FILE_MAP.keys()))
            return ToolResult(
                tc.id,
                f"Unknown spec: {spec_name!r}. Available: {available}",
                True,
            )

        target = self._root / spec_file
        if not target.exists():
            return ToolResult(tc.id, f"Spec file not found: {spec_file}", True)

        try:
            content = target.read_text(encoding="utf-8")
            # Truncate to 4000 chars to stay within token budget
            if len(content) > 4000:
                content = content[:4000] + "\n\n[... truncated — use read_file for full content ...]"
            return ToolResult(tc.id, content)
        except Exception as exc:
            return ToolResult(tc.id, f"Read error: {exc}", True)

    async def _tool_find_similar(self, tc: ToolCall) -> ToolResult:
        """Find existing implementations similar to what needs to be built."""
        description = tc.input.get("description", "").lower()

        # Find matching paths from the keyword map
        matched_paths: list[str] = []
        for keyword, paths in _SIMILAR_CODE_MAP.items():
            if keyword in description:
                matched_paths.extend(paths)
                break

        if not matched_paths:
            # Fallback: search for the first noun in the description
            words = description.split()
            for word in words:
                if len(word) > 3:
                    for keyword, paths in _SIMILAR_CODE_MAP.items():
                        if word in keyword or keyword in word:
                            matched_paths.extend(paths)
                            break
                if matched_paths:
                    break

        if not matched_paths:
            return ToolResult(
                tc.id,
                "No similar implementations found. Try search_code with a specific pattern.",
            )

        # Read the first matching file/directory
        results: list[str] = []
        chars_remaining = 4000

        for rel_path in matched_paths:
            if chars_remaining <= 0:
                break
            target = self._root / rel_path
            if target.is_file():
                try:
                    content = target.read_text(encoding="utf-8")
                    chunk = content[:min(2500, chars_remaining)]
                    results.append(f"=== {rel_path} ===\n{chunk}")
                    chars_remaining -= len(results[-1])
                except Exception:
                    continue
            elif target.is_dir():
                # List the directory and read the first non-init Python file
                try:
                    py_files = sorted(target.glob("*.py"))
                    file_list = ", ".join(f.name for f in py_files)
                    results.append(f"=== {rel_path} ===\nFiles: {file_list}")
                    chars_remaining -= len(results[-1])

                    for py_file in py_files:
                        if py_file.name == "__init__.py" or chars_remaining <= 0:
                            continue
                        content = py_file.read_text(encoding="utf-8")
                        chunk = content[:min(2000, chars_remaining)]
                        rel = str(py_file.relative_to(self._root))
                        results.append(f"\n=== {rel} (exemplar) ===\n{chunk}")
                        chars_remaining -= len(results[-1])
                        break  # One exemplar is enough
                except Exception:
                    continue

        return ToolResult(tc.id, "\n\n".join(results) if results else "No files found at matched paths")

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _check_forbidden_path(self, rel_path: str) -> str | None:
        """Check if a path is forbidden. Returns error message or None."""
        from ecodiaos.systems.simula.types import FORBIDDEN_WRITE_PATHS
        for forbidden in FORBIDDEN_WRITE_PATHS:
            if rel_path.startswith(forbidden) or forbidden in rel_path:
                return (
                    f"IRON RULE VIOLATION: Cannot write to forbidden path '{rel_path}' "
                    f"(matches forbidden pattern '{forbidden}'). "
                    "This change would violate Simula's constitutional constraints."
                )
        return None

    def _path_to_module(self, rel_path: str) -> str | None:
        """Convert a relative file path to a dotted module name."""
        parts = rel_path.replace("\\", "/").split("/")
        if parts and parts[0] == "src":
            parts = parts[1:]
        if not parts:
            return None
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts) if parts else None

    def _build_system_prompt(self, proposal: EvolutionProposal) -> str:
        from ecodiaos.systems.simula.types import FORBIDDEN_WRITE_PATHS, SIMULA_IRON_RULES

        architecture_context = _build_architecture_context(
            category=proposal.category,
            codebase_root=self._root,
        )

        return _SYSTEM_PROMPT_TEMPLATE.format(
            category=proposal.category.value,
            description=proposal.description,
            expected_benefit=proposal.expected_benefit,
            evidence=", ".join(proposal.evidence) or "none",
            iron_rules="\n".join(f"- {r}" for r in SIMULA_IRON_RULES),
            forbidden_paths="\n".join(f"- {p}" for p in FORBIDDEN_WRITE_PATHS),
            architecture_context=architecture_context,
        )
