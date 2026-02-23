"""
EcodiaOS -- Simula Health Checker

After a change is applied, the health checker verifies the codebase
is still functional. Three checks run in sequence:
  1. Syntax check -- ast.parse() on all written Python files
  2. Import check -- attempt to import the affected module
  3. Unit tests -- run pytest on the affected system's test directory

If any check fails, Simula rolls back the change. The goal is to never
leave EOS in a broken state.

Target: health check completes <=5s (as part of the <=5s change application budget).
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
from pathlib import Path

import structlog

from ecodiaos.systems.simula.types import HealthCheckResult

logger = structlog.get_logger().bind(system="simula.health")


class HealthChecker:
    """
    Verifies post-apply codebase health via syntax, import, and test checks.
    Any failure triggers rollback.
    """

    def __init__(self, codebase_root: Path, test_command: str = "pytest") -> None:
        self._root = codebase_root
        self._test_command = test_command
        self._log = logger

    async def check(self, files_written: list[str]) -> HealthCheckResult:
        """""""""
        Run all health checks in sequence.  Returns on first failure.
        """""""""
        # 1. Syntax check
        syntax_errors = await self._check_syntax(files_written)
        if syntax_errors:
            self._log.warning("health_syntax_failed", errors=syntax_errors)
            return HealthCheckResult(healthy=False, issues=syntax_errors)
        self._log.info("health_syntax_passed", files=len(files_written))

        # 2. Import check
        import_errors = await self._check_imports(files_written)
        if import_errors:
            self._log.warning("health_import_failed", errors=import_errors)
            return HealthCheckResult(healthy=False, issues=import_errors)
        self._log.info("health_import_passed", files=len(files_written))

        # 3. Tests
        tests_passed, test_output = await self._run_tests(files_written)
        if not tests_passed:
            self._log.warning("health_tests_failed", output=test_output[:500])
            return HealthCheckResult(
                healthy=False,
                issues=[f"Test suite failed:\n{test_output[:1000]}"],
            )
        self._log.info("health_tests_passed")

        return HealthCheckResult(healthy=True)

    async def _check_syntax(self, files: list[str]) -> list[str]:
        """""""""
        Parse each .py file with ast.parse().  Collect syntax errors.
        Returns a list of error strings (empty list = all pass).
        """""""""
        errors: list[str] = []
        for filepath in files:
            if not filepath.endswith(".py"):
                continue
            path = Path(filepath)
            if not path.exists():
                errors.append(f"Syntax check: file not found: {filepath}")
                continue
            try:
                source = path.read_text(encoding="utf-8")
                ast.parse(source, filename=filepath)
            except SyntaxError as exc:
                errors.append(f"Syntax error in {filepath}:{exc.lineno}: {exc.msg}")
            except Exception as exc:
                errors.append(f"Failed to read {filepath}: {exc}")
        return errors

    async def _check_imports(self, files: list[str]) -> list[str]:
        """""""""
        Derive dotted module paths from written file paths and check
        whether importlib can locate them.  Returns error strings.
        """""""""
        errors: list[str] = []
        for filepath in files:
            if not filepath.endswith(".py"):
                continue
            module_path = self._derive_module_path(filepath)
            if module_path is None:
                continue
            try:
                spec = importlib.util.find_spec(module_path)
                if spec is None:
                    errors.append(f"Import check: module not found: {module_path}")
            except ModuleNotFoundError as exc:
                errors.append(f"Import check: {module_path}: {exc}")
            except Exception as exc:
                errors.append(f"Import check failed for {module_path}: {exc}")
        return errors

    def _derive_module_path(self, src_file: str) -> str | None:
        """""""""
        Convert a source file path to a dotted module path.
        Example: src/ecodiaos/systems/axon/executors/my.py
                 -> ecodiaos.systems.axon.executors.my
        """""""""
        try:
            path = Path(src_file)
            # Make relative to codebase root if possible
            try:
                rel = path.relative_to(self._root)
            except ValueError:
                rel = path
            parts = list(rel.parts)
            if parts and parts[0] == "src":
                parts = parts[1:]
            # Strip .py extension from last part
            if parts:
                parts[-1] = parts[-1].removesuffix(".py")
            return ".".join(parts) if parts else None
        except Exception:
            return None

    async def _run_tests(self, files: list[str]) -> tuple[bool, str]:
        """""""""
        Derive the test directory from the written files and run pytest.
        Returns (passed, output).  If no test directory found, returns (True, ...).
        30-second subprocess timeout.
        """""""""
        test_path = None
        for filepath in files:
            candidate = self._derive_test_path(filepath)
            if candidate:
                test_dir = Path(candidate)
                if test_dir.is_dir():
                    test_path = candidate
                    break

        if test_path is None:
            self._log.info("health_no_tests", files=files)
            return True, "no tests found"

        try:
            proc = await asyncio.create_subprocess_exec(
                self._test_command,
                test_path,
                "-x",
                "--tb=short",
                "-q",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self._root),
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return False, "Test run timed out after 30s"
            output = stdout.decode("utf-8", errors="replace")
            passed = proc.returncode == 0
            self._log.info(
                "health_test_run",
                test_path=test_path,
                passed=passed,
                returncode=proc.returncode,
            )
            return passed, output
        except FileNotFoundError:
            msg = f"Test command {self._test_command!r} not found"
            self._log.warning("health_test_command_missing", command=self._test_command)
            return False, msg
        except Exception as exc:
            return False, f"Test run error: {exc}"

    def _derive_test_path(self, src_file: str) -> str | None:
        """""""""
        Map a source file path to a test directory path.
        Example: src/ecodiaos/systems/axon/executor.py
                 -> tests/unit/systems/axon/
        """""""""
        try:
            path = Path(src_file)
            try:
                rel = path.relative_to(self._root)
            except ValueError:
                rel = path
            parts = list(rel.parts)
            # Expect: src / ecodiaos / systems / <system_name> / ...
            if len(parts) >= 4 and parts[0] == "src" and parts[2] == "systems":
                system_name = parts[3]
                test_path = self._root / "tests" / "unit" / "systems" / system_name
                return str(test_path)
            return None
        except Exception:
            return None
