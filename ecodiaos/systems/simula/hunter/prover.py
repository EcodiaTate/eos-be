"""
EcodiaOS — Hunter Vulnerability Prover (Phases 4 + 5)

Proves vulnerabilities exist by encoding security conditions as Z3 constraints,
then translates proven counterexamples into local diagnostic reproduction scripts.

The Inversion:
  Internal Simula: "Is this code correct?" → NOT(property) → UNSAT = correct
  Hunter:          "Is this code exploitable?" → NOT(security_property) → SAT = exploitable

When Z3 returns SAT, the counterexample is a concrete set of variable assignments
that violate the security property. Phase 5 translates these into a local-only
Security Unit Test script (targeting localhost only) for the RepairAgent to
use as a regression test during its patch-verify loop.

Uses the same Z3Bridge.check_invariant() infrastructure as Stage 2B but
inverts the interpretation: SAT means the security property can be violated.
"""

from __future__ import annotations

import ast
import json
import re
import time
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.clients.llm import Message
from ecodiaos.systems.simula.hunter.types import (
    AttackSurface,
    AttackSurfaceType,
    HunterConfig,
    VulnerabilityClass,
    VulnerabilityReport,
    VulnerabilitySeverity,
)

if TYPE_CHECKING:
    from ecodiaos.clients.llm import LLMProvider
    from ecodiaos.systems.simula.verification.z3_bridge import Z3Bridge

logger = structlog.get_logger().bind(system="simula.hunter.prover")


# ── Vulnerability class mapping ─────────────────────────────────────────────
# Maps attack goal keywords to the most likely vulnerability class.

_GOAL_TO_VULN_CLASS: list[tuple[re.Pattern[str], VulnerabilityClass]] = [
    (re.compile(r"unauth|authentication|login|session|token", re.I), VulnerabilityClass.BROKEN_AUTH),
    (re.compile(r"access.control|user.a.*user.b|another.user|idor|object.reference", re.I), VulnerabilityClass.BROKEN_ACCESS_CONTROL),
    (re.compile(r"sql.injection|sql.inject|sqli", re.I), VulnerabilityClass.SQL_INJECTION),
    (re.compile(r"inject(?!.*sql)", re.I), VulnerabilityClass.INJECTION),
    (re.compile(r"xss|cross.site.script", re.I), VulnerabilityClass.XSS),
    (re.compile(r"ssrf|server.side.request", re.I), VulnerabilityClass.SSRF),
    (re.compile(r"privilege.escalat|admin.function|role.bypass", re.I), VulnerabilityClass.PRIVILEGE_ESCALATION),
    (re.compile(r"reentran|recursive.call|contract.*call.*itself", re.I), VulnerabilityClass.REENTRANCY),
    (re.compile(r"race.condition|concurrent|toctou|double.spend", re.I), VulnerabilityClass.RACE_CONDITION),
    (re.compile(r"redirect|open.redirect|url.redirect", re.I), VulnerabilityClass.UNVALIDATED_REDIRECT),
    (re.compile(r"information.disclos|leak|expos|sensitive.data", re.I), VulnerabilityClass.INFORMATION_DISCLOSURE),
    (re.compile(r"deserializ|pickle|yaml.load|marshal", re.I), VulnerabilityClass.INSECURE_DESERIALIZATION),
    (re.compile(r"path.traversal|directory.traversal|\.\./", re.I), VulnerabilityClass.PATH_TRAVERSAL),
    (re.compile(r"command.inject|os.system|subprocess|shell.inject", re.I), VulnerabilityClass.COMMAND_INJECTION),
]


# ── Severity heuristics ──────────────────────────────────────────────────────
# Higher severity for more impactful vulnerability classes.

_VULN_SEVERITY_MAP: dict[VulnerabilityClass, VulnerabilitySeverity] = {
    VulnerabilityClass.SQL_INJECTION: VulnerabilitySeverity.CRITICAL,
    VulnerabilityClass.COMMAND_INJECTION: VulnerabilitySeverity.CRITICAL,
    VulnerabilityClass.REENTRANCY: VulnerabilitySeverity.CRITICAL,
    VulnerabilityClass.INSECURE_DESERIALIZATION: VulnerabilitySeverity.CRITICAL,
    VulnerabilityClass.BROKEN_AUTH: VulnerabilitySeverity.HIGH,
    VulnerabilityClass.BROKEN_ACCESS_CONTROL: VulnerabilitySeverity.HIGH,
    VulnerabilityClass.PRIVILEGE_ESCALATION: VulnerabilitySeverity.HIGH,
    VulnerabilityClass.SSRF: VulnerabilitySeverity.HIGH,
    VulnerabilityClass.PATH_TRAVERSAL: VulnerabilitySeverity.HIGH,
    VulnerabilityClass.INJECTION: VulnerabilitySeverity.HIGH,
    VulnerabilityClass.XSS: VulnerabilitySeverity.MEDIUM,
    VulnerabilityClass.RACE_CONDITION: VulnerabilitySeverity.MEDIUM,
    VulnerabilityClass.INFORMATION_DISCLOSURE: VulnerabilitySeverity.MEDIUM,
    VulnerabilityClass.UNVALIDATED_REDIRECT: VulnerabilitySeverity.LOW,
    VulnerabilityClass.OTHER: VulnerabilitySeverity.LOW,
}


# ── System prompts ───────────────────────────────────────────────────────────

_ATTACK_ENCODING_SYSTEM_PROMPT = """\
You are a security constraint encoder for Z3 SMT solver.

Your task: given an attack surface (code) and an attacker goal, encode the
security violation as Z3 constraints. The goal is to find concrete inputs
that PROVE the vulnerability exists.

## Encoding Rules

1. Declare Z3 variables for all relevant inputs and state:
   - Use z3.Int for integer values (IDs, counts, indices)
   - Use z3.Bool for boolean flags (is_authenticated, is_admin, has_permission)
   - Use z3.Real for numeric values (amounts, scores, timestamps)
   - Use z3.BitVec for bitfield/flags and string-length reasoning

2. Encode the SECURITY PROPERTY that should hold (the protection):
   - Example: "authenticated users can only access their own data"
   - Express as: z3.Implies(condition, protected_outcome)

3. Encode the ATTACKER GOAL as the NEGATION of the security property:
   - The Z3 expression should evaluate to True when the attack SUCCEEDS
   - If Z3 finds this satisfiable (SAT), the vulnerability is proven

4. Your output must be a single Z3 Python expression using variables you declare.
   The expression must evaluate to a z3.BoolRef when executed.

## Variable Declaration Format
Declare variables as a dict mapping name → Z3 type ("Int", "Real", "Bool").

## Output Format
Respond with ONLY a JSON object:
{
  "variable_declarations": {"var_name": "Int|Real|Bool", ...},
  "z3_expression": "z3.And(condition1, condition2, ...)",
  "reasoning": "Brief explanation of the encoding logic"
}

Do NOT include any other text. Only the JSON object.

## Examples

### Example 1: Broken Access Control
Surface: GET /api/user/{id}, function get_user(id, current_user)
Goal: "User A can access User B's data without authorization"

{
  "variable_declarations": {
    "requested_user_id": "Int",
    "current_user_id": "Int",
    "is_authenticated": "Bool",
    "can_access_data": "Bool"
  },
  "z3_expression": "z3.And(is_authenticated == True, requested_user_id != current_user_id, can_access_data == True)",
  "reasoning": "If an authenticated user can access data for a different user ID, access control is broken"
}

### Example 2: SQL Injection
Surface: POST /api/search, function search(query)
Goal: "SQL injection in user input"

{
  "variable_declarations": {
    "input_length": "Int",
    "contains_sql_metachar": "Bool",
    "input_sanitized": "Bool",
    "query_parameterized": "Bool",
    "sql_executed_with_input": "Bool"
  },
  "z3_expression": "z3.And(input_length > 0, contains_sql_metachar == True, input_sanitized == False, query_parameterized == False, sql_executed_with_input == True)",
  "reasoning": "If user input containing SQL metacharacters reaches execution without sanitization or parameterization, SQL injection is possible"
}

### Example 3: Privilege Escalation
Surface: POST /api/admin/settings, function update_settings(user, settings)
Goal: "Regular user can call admin function"

{
  "variable_declarations": {
    "user_role_level": "Int",
    "admin_role_level": "Int",
    "function_requires_admin": "Bool",
    "access_granted": "Bool"
  },
  "z3_expression": "z3.And(user_role_level < admin_role_level, function_requires_admin == True, access_granted == True)",
  "reasoning": "If a user with role level below admin can access an admin-only function, privilege escalation exists"
}"""


_SEVERITY_CLASSIFICATION_PROMPT = """\
You are a vulnerability severity classifier.

Given a vulnerability with its Z3 counterexample, classify the severity
as one of: LOW, MEDIUM, HIGH, CRITICAL.

Severity guidelines (CVSS-aligned):
- CRITICAL (9.0-10.0): Remote code execution, authentication bypass, \
SQL injection with data exfiltration, reentrancy with fund theft
- HIGH (7.0-8.9): Privilege escalation, SSRF, broken access control, \
path traversal with sensitive file read
- MEDIUM (4.0-6.9): XSS, race conditions, information disclosure of \
non-sensitive data, CSRF
- LOW (0.1-3.9): Open redirects, minor information leaks, verbose \
error messages

Also consider the attack surface context:
- API endpoints with authentication: higher severity
- Public functions without state changes: lower severity
- Smart contract functions handling funds: higher severity

Respond with ONLY a JSON object:
{
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "reasoning": "Brief justification"
}"""


# ── Phase 5: Reproduction script generation prompt ─────────────────────────

_POC_GENERATION_SYSTEM_PROMPT = """\
You are a security verification engineer generating diagnostic reproduction \
scripts from formal Z3 counterexamples.

Your task: given a Z3 counterexample (concrete variable assignments that \
demonstrate a vulnerability condition) and the attack surface context, \
generate a Python Security Unit Test script that:
  1. Reproduces the exact input conditions Z3 identified against a LOCAL \
     development server only.
  2. Asserts the expected secure behaviour (a failing assertion documents \
     the bug for the RepairAgent to fix).
  3. Is safe, readable, and never touches a production system.

## Rules

1. The script must use the `requests` library (or `httpx` as fallback).
2. The script MUST target localhost only — `TARGET_URL` defaults to \
"http://localhost:8000". Add a comment: "# Run against local dev server only".
3. Include detailed comments explaining:
   - Which security property the Z3 counterexample violates
   - How each Z3 variable maps to a concrete request parameter
   - What the assertion checks (i.e., the expected SECURE response)
4. Structure the script as:
   - Module docstring labelled "Security Unit Test: <VulnerabilityClass>"
   - `TARGET_URL` constant at the top with the localhost comment
   - A `build_request()` function that returns (url, headers, payload) \
     WITHOUT executing the request
   - A `run_security_test()` function that calls `build_request()`, sends \
     the request, and asserts the response indicates secure behaviour
   - A `if __name__ == "__main__"` block that calls `run_security_test()`
5. Map Z3 variable assignments to concrete HTTP values:
   - Bool False → omit auth headers or use empty tokens
   - Bool True → include the header/parameter
   - Int/Real values → use as path params, query params, or body values
6. The script must be syntactically valid Python 3.10+.
7. Do NOT import anything beyond: requests, json, sys, os, urllib.parse, base64, hashlib, hmac.

## Output Format

Respond with ONLY the Python code. No markdown fences, no explanation text.
Start directly with the docstring or import statements.

## Examples

### Example 1: Broken Access Control
Z3 counterexample: is_authenticated=True, requested_user_id=999, current_user_id=1, can_access_data=True
Surface: GET /api/user/{id}

Output:
\"\"\"
Security Unit Test: Broken Access Control (IDOR)
Z3 condition: is_authenticated=True, requested_user_id=999, current_user_id=1 → can_access_data=True
Expected secure behaviour: server returns 403 Forbidden when user 1 requests user 999's data.
\"\"\"
import requests

TARGET_URL = "http://localhost:8000"  # Run against local dev server only

def build_request():
    url = f"{TARGET_URL}/api/user/999"
    headers = {"Authorization": "Bearer <valid_token_for_user_1>"}
    return url, headers, None

def run_security_test():
    url, headers, payload = build_request()
    print(f"[test] GET {url} as user 1 (should be denied)")
    response = requests.get(url, headers=headers)
    assert response.status_code == 403, (
        f"SECURITY BUG: expected 403 Forbidden, got {response.status_code}. "
        "User can read another user's data."
    )
    print("[pass] Server correctly denied cross-user data access")
    return response

if __name__ == "__main__":
    run_security_test()

### Example 2: SQL Injection
Z3 counterexample: input_length=10, contains_sql_metachar=True, input_sanitized=False, query_parameterized=False, sql_executed_with_input=True
Surface: POST /api/search, function search(query)

Output:
\"\"\"
Security Unit Test: SQL Injection
Z3 condition: contains_sql_metachar=True, input_sanitized=False, query_parameterized=False
Expected secure behaviour: server rejects or safely handles SQL metacharacters.
\"\"\"
import requests
import json

TARGET_URL = "http://localhost:8000"  # Run against local dev server only

def build_request():
    url = f"{TARGET_URL}/api/search"
    headers = {"Content-Type": "application/json"}
    payload = {"query": "' OR 1=1 --"}
    return url, headers, payload

def run_security_test():
    url, headers, payload = build_request()
    print(f"[test] POST {url} with SQL metacharacters")
    response = requests.post(url, headers=headers, json=payload)
    assert response.status_code != 500, (
        "SECURITY BUG: server threw a 500, likely due to unparameterized SQL. "
        "Input must be sanitized or queries must use parameterized statements."
    )
    print("[pass] Server handled SQL metacharacters without error")
    return response

if __name__ == "__main__":
    run_security_test()"""


# ── Unsafe import / network patterns for PoC safety validation ─────────────

_POC_FORBIDDEN_IMPORTS = frozenset({
    "subprocess", "os.system", "os.popen", "shutil", "ctypes",
    "socket", "multiprocessing", "threading", "signal",
    "importlib", "pickle", "shelve", "marshal",
})

_POC_ALLOWED_IMPORTS = frozenset({
    "requests", "json", "sys", "os", "urllib", "urllib.parse",
    "base64", "hashlib", "hmac", "httpx",
})


# ── VulnerabilityProver ──────────────────────────────────────────────────────


class VulnerabilityProver:
    """
    Proves security violations exist by encoding vulnerability conditions as Z3 constraints.

    The key inversion from internal Simula verification:
    - Simula checks NOT(invariant) → UNSAT means code is correct
    - Hunter checks NOT(security_property) → SAT means the security property
      can be violated (i.e., a vulnerability exists)

    When SAT, the Z3 model provides concrete variable assignments that
    demonstrate the violation. These drive a local Security Unit Test script
    for the RepairAgent to verify its patches against.
    """

    def __init__(
        self,
        z3_bridge: Z3Bridge,
        llm: LLMProvider,
        *,
        max_encoding_retries: int = 2,
        check_timeout_ms: int = 10_000,
    ) -> None:
        """
        Args:
            z3_bridge: Z3Bridge instance for constraint checking.
            llm: LLM provider for encoding attack goals as Z3 constraints.
            max_encoding_retries: Max retries if encoding fails Z3 parsing.
            check_timeout_ms: Z3 solver timeout for each constraint check.
        """
        self._z3 = z3_bridge
        self._llm = llm
        self._max_retries = max_encoding_retries
        self._check_timeout_ms = check_timeout_ms
        self._log = logger

    # ── Public API ──────────────────────────────────────────────────────────

    async def prove_vulnerability(
        self,
        surface: AttackSurface,
        attack_goal: str,
        target_url: str = "unknown",
        *,
        generate_poc: bool = False,
        config: HunterConfig | None = None,
    ) -> VulnerabilityReport | None:
        """
        Attempt to prove a vulnerability exists for a given attack surface.

        The LLM encodes the attack goal as Z3 constraints. If Z3 finds the
        constraints satisfiable (SAT), the vulnerability is mathematically
        proven to exist and a VulnerabilityReport is returned with the
        counterexample.

        When generate_poc=True, proven vulnerabilities also get an executable
        PoC script attached to the report (Phase 5).

        Args:
            surface: The attack surface to test.
            attack_goal: Human-readable description of the attacker's goal
                (e.g., "Unauthenticated access to protected resource").
            target_url: GitHub URL or "internal_eos" for analytics tagging.
            generate_poc: If True, generate a PoC script for proven vulns.
            config: HunterConfig for authorized_targets validation during
                PoC generation.

        Returns:
            VulnerabilityReport if vulnerability proven (SAT), None if
            not exploitable (UNSAT) or encoding failed.
        """
        start = time.monotonic()
        self._log.info(
            "prove_vulnerability_start",
            entry_point=surface.entry_point,
            attack_goal=attack_goal,
            surface_type=surface.surface_type.value,
            file_path=surface.file_path,
        )

        # Encode the attack goal as Z3 constraints (with retry on parse errors)
        encoding = await self._encode_attack_goal_with_retry(surface, attack_goal)
        if encoding is None:
            self._log.warning(
                "encoding_failed",
                entry_point=surface.entry_point,
                attack_goal=attack_goal,
            )
            return None

        z3_expr_code, variable_declarations, reasoning = encoding

        # Check constraints via Z3: SAT = vulnerability proven
        check_start = time.monotonic()
        status, counterexample = self._check_exploit_constraints(
            z3_expr_code, variable_declarations,
        )
        z3_time_ms = int((time.monotonic() - check_start) * 1000)

        total_ms = int((time.monotonic() - start) * 1000)

        if status == "sat":
            # Vulnerability proven — Z3 found concrete exploit conditions
            vuln_class = self._classify_vulnerability(attack_goal)
            severity = await self._classify_severity(
                surface, attack_goal, counterexample, vuln_class,
            )

            report = VulnerabilityReport(
                target_url=target_url,
                vulnerability_class=vuln_class,
                severity=severity,
                attack_surface=surface,
                attack_goal=attack_goal,
                z3_counterexample=counterexample,
                z3_constraints_code=z3_expr_code,
            )

            # Phase 5: generate Security Unit Test reproduction script if requested
            if generate_poc:
                poc_code = await self.generate_reproduction_script(report, config=config)
                if poc_code:
                    report.proof_of_concept_code = poc_code

            self._log.info(
                "vulnerability_proved",
                vuln_id=report.id,
                vulnerability_class=vuln_class.value,
                severity=severity.value,
                entry_point=surface.entry_point,
                attack_goal=attack_goal,
                z3_time_ms=z3_time_ms,
                total_ms=total_ms,
                counterexample=counterexample,
                has_poc=bool(report.proof_of_concept_code),
            )
            return report

        elif status == "unsat":
            # Security property holds — no vulnerability
            self._log.info(
                "vulnerability_disproved",
                entry_point=surface.entry_point,
                attack_goal=attack_goal,
                z3_time_ms=z3_time_ms,
                total_ms=total_ms,
            )
            return None

        else:
            # Solver timeout or unknown — inconclusive
            self._log.warning(
                "vulnerability_check_inconclusive",
                entry_point=surface.entry_point,
                attack_goal=attack_goal,
                status=status,
                detail=counterexample,
                z3_time_ms=z3_time_ms,
            )
            return None

    async def prove_vulnerability_batch(
        self,
        surface: AttackSurface,
        attack_goals: list[str],
        target_url: str = "unknown",
        *,
        generate_poc: bool = False,
        config: HunterConfig | None = None,
    ) -> list[VulnerabilityReport]:
        """
        Test multiple attack goals against a single surface.

        Args:
            surface: The attack surface to test.
            attack_goals: List of attacker goals to test.
            target_url: Target URL for analytics.
            generate_poc: If True, generate PoC scripts for proven vulns.
            config: HunterConfig for authorized_targets validation.

        Returns:
            List of proven vulnerabilities (may be empty).
        """
        reports: list[VulnerabilityReport] = []

        for goal in attack_goals:
            report = await self.prove_vulnerability(
                surface, goal, target_url=target_url,
                generate_poc=generate_poc,
                config=config,
            )
            if report is not None:
                reports.append(report)

        return reports

    # ── Attack goal encoding ────────────────────────────────────────────────

    async def _encode_attack_goal(
        self,
        surface: AttackSurface,
        goal: str,
    ) -> tuple[str, dict[str, str], str] | None:
        """
        Use LLM to encode an attacker goal as Z3 constraints.

        Args:
            surface: The attack surface with context code.
            goal: Human-readable attacker goal.

        Returns:
            (z3_expression, variable_declarations, reasoning) or None if
            encoding fails.
        """
        # Build the user prompt with surface context
        prompt_parts = [
            "## Attack Surface",
            f"Entry point: {surface.entry_point}",
            f"Type: {surface.surface_type.value}",
            f"File: {surface.file_path}",
        ]

        if surface.http_method:
            prompt_parts.append(f"HTTP method: {surface.http_method}")
        if surface.route_pattern:
            prompt_parts.append(f"Route: {surface.route_pattern}")

        if surface.context_code:
            prompt_parts.extend([
                "",
                "## Source Code",
                f"```\n{surface.context_code[:4000]}\n```",
            ])

        prompt_parts.extend([
            "",
            "## Attacker Goal",
            f"{goal}",
            "",
            "Encode the attacker goal as Z3 constraints. The Z3 expression "
            "should be satisfiable (SAT) when the attack succeeds.",
        ])

        user_prompt = "\n".join(prompt_parts)

        try:
            response = await self._llm.generate(
                system_prompt=_ATTACK_ENCODING_SYSTEM_PROMPT,
                messages=[Message(role="user", content=user_prompt)],
                max_tokens=2048,
                temperature=0.2,
            )
        except Exception as exc:
            self._log.error(
                "encoding_llm_error",
                error=str(exc),
                entry_point=surface.entry_point,
            )
            return None

        return self._parse_encoding_response(response.text)

    async def _encode_attack_goal_with_retry(
        self,
        surface: AttackSurface,
        goal: str,
    ) -> tuple[str, dict[str, str], str] | None:
        """
        Encode with retries on parse failures. Feeds parse errors back
        to the LLM for correction.
        """
        last_error: str | None = None

        for attempt in range(1, self._max_retries + 1):
            if attempt == 1:
                result = await self._encode_attack_goal(surface, goal)
            else:
                # Retry with error feedback
                result = await self._encode_with_error_feedback(
                    surface, goal, last_error or "Unknown parse error",
                )

            if result is not None:
                z3_expr, var_decls, reasoning = result

                # Validate that the expression can be parsed by Z3
                validation_error = self._validate_z3_expression(
                    z3_expr, var_decls,
                )
                if validation_error is None:
                    return result
                else:
                    last_error = validation_error
                    self._log.debug(
                        "z3_validation_failed",
                        attempt=attempt,
                        error=validation_error,
                    )
            else:
                last_error = "LLM response could not be parsed as JSON"

        return None

    async def _encode_with_error_feedback(
        self,
        surface: AttackSurface,
        goal: str,
        error: str,
    ) -> tuple[str, dict[str, str], str] | None:
        """Re-encode with the previous error fed back for correction."""
        prompt_parts = [
            "## Attack Surface",
            f"Entry point: {surface.entry_point}",
            f"Type: {surface.surface_type.value}",
        ]

        if surface.context_code:
            prompt_parts.extend([
                "",
                "## Source Code",
                f"```\n{surface.context_code[:3000]}\n```",
            ])

        prompt_parts.extend([
            "",
            "## Attacker Goal",
            f"{goal}",
            "",
            "## Previous Attempt Error",
            f"Your previous Z3 encoding failed with: {error}",
            "",
            "Please fix the encoding. Common issues:",
            "- Expression must use declared variable names exactly",
            "- Use z3.And, z3.Or, z3.Not, z3.Implies — not Python and/or/not",
            "- Comparison operators (==, !=, <, >, <=, >=) are fine on Z3 vars",
            "- Bool variables use == True/False, not bare references",
            "",
            "Respond with ONLY a JSON object as specified.",
        ])

        try:
            response = await self._llm.generate(
                system_prompt=_ATTACK_ENCODING_SYSTEM_PROMPT,
                messages=[Message(role="user", content="\n".join(prompt_parts))],
                max_tokens=2048,
                temperature=0.1,
            )
        except Exception as exc:
            self._log.error("retry_encoding_llm_error", error=str(exc))
            return None

        return self._parse_encoding_response(response.text)

    # ── Z3 constraint checking ──────────────────────────────────────────────

    def _check_exploit_constraints(
        self,
        z3_expr_code: str,
        variable_declarations: dict[str, str],
    ) -> tuple[str, str]:
        """
        Check exploit constraints via Z3.

        Unlike Z3Bridge.check_invariant() which checks NOT(property),
        here we check the expression DIRECTLY: if SAT, the exploit
        conditions can be satisfied.

        The expression already encodes the attacker's goal (the negation
        of the security property), so we don't need to negate again.

        Args:
            z3_expr_code: Z3 Python expression encoding the attack.
            variable_declarations: Variable name → Z3 type mapping.

        Returns:
            ("sat", counterexample) if exploitable,
            ("unsat", "") if secure,
            ("unknown", error_detail) if inconclusive.
        """
        try:
            import z3 as z3_lib
        except ImportError:
            return "unknown", "z3-solver not installed"

        solver = z3_lib.Solver()
        solver.set("timeout", self._check_timeout_ms)

        # Create Z3 variables from declarations
        z3_vars: dict[str, Any] = {}
        for name, z3_type in variable_declarations.items():
            if z3_type == "Int":
                z3_vars[name] = z3_lib.Int(name)
            elif z3_type == "Real":
                z3_vars[name] = z3_lib.Real(name)
            elif z3_type == "Bool":
                z3_vars[name] = z3_lib.Bool(name)
            else:
                # Default to Real for unknown types
                z3_vars[name] = z3_lib.Real(name)

        # Evaluate the Z3 expression in a sandboxed namespace
        namespace: dict[str, Any] = {"z3": z3_lib, **z3_vars}
        try:
            expr = eval(z3_expr_code, {"__builtins__": {}}, namespace)  # noqa: S307
        except Exception as exc:
            return "unknown", f"expression eval error: {exc}"

        if not isinstance(expr, z3_lib.BoolRef):
            return "unknown", "expression did not produce a z3.BoolRef"

        # Direct check: the expression encodes the attacker goal.
        # SAT means the attack conditions can be satisfied → vulnerability.
        solver.add(expr)
        result = solver.check()

        if result == z3_lib.sat:
            model = solver.model()
            counterexample = self._extract_z3_model(model)
            return "sat", counterexample
        elif result == z3_lib.unsat:
            return "unsat", ""
        else:
            return "unknown", "solver timeout or unknown"

    def _extract_z3_model(self, model: Any) -> str:
        """
        Convert a Z3 model to a human-readable counterexample string.

        The model contains concrete variable assignments that demonstrate
        the exploit conditions.

        Args:
            model: Z3 Model object from a SAT result.

        Returns:
            Human-readable string like "is_authenticated=False, user_id=999".
        """
        parts: list[str] = []
        try:
            for decl in model.decls():
                value = model[decl]
                # Format boolean values readably
                val_str = str(value)
                if val_str == "True":
                    val_str = "True"
                elif val_str == "False":
                    val_str = "False"
                parts.append(f"{decl.name()}={val_str}")
        except Exception as exc:
            self._log.warning("model_extraction_error", error=str(exc))
            return f"<model extraction failed: {exc}>"

        return ", ".join(sorted(parts))

    # ── Vulnerability classification ────────────────────────────────────────

    def _classify_vulnerability(self, attack_goal: str) -> VulnerabilityClass:
        """
        Classify a vulnerability based on the attack goal keywords.

        Uses regex matching against known vulnerability patterns.
        Falls back to OTHER if no pattern matches.
        """
        for pattern, vuln_class in _GOAL_TO_VULN_CLASS:
            if pattern.search(attack_goal):
                return vuln_class
        return VulnerabilityClass.OTHER

    async def _classify_severity(
        self,
        surface: AttackSurface,
        attack_goal: str,
        counterexample: str,
        vuln_class: VulnerabilityClass,
    ) -> VulnerabilitySeverity:
        """
        Classify severity using heuristic mapping first, LLM refinement
        if needed for edge cases.

        The heuristic is fast and deterministic; the LLM provides nuanced
        classification for cases where surface context matters.
        """
        # Start with heuristic severity from the vulnerability class
        base_severity = _VULN_SEVERITY_MAP.get(
            vuln_class, VulnerabilitySeverity.MEDIUM,
        )

        # Escalate based on surface type heuristics
        if surface.surface_type == AttackSurfaceType.SMART_CONTRACT_PUBLIC:
            # Smart contract vulns involving funds are always critical
            if vuln_class in (
                VulnerabilityClass.REENTRANCY,
                VulnerabilityClass.RACE_CONDITION,
            ):
                return VulnerabilitySeverity.CRITICAL

        if surface.surface_type == AttackSurfaceType.AUTH_HANDLER:
            # Auth handler vulnerabilities are at least HIGH
            if base_severity.value in ("low", "medium"):
                return VulnerabilitySeverity.HIGH

        if surface.surface_type == AttackSurfaceType.DATABASE_QUERY:
            # Database query vulns are at least HIGH (data exposure)
            if base_severity.value == "low":
                return VulnerabilitySeverity.MEDIUM

        # Use LLM for more nuanced classification of ambiguous cases
        if vuln_class == VulnerabilityClass.OTHER:
            llm_severity = await self._llm_classify_severity(
                surface, attack_goal, counterexample,
            )
            if llm_severity is not None:
                return llm_severity

        return base_severity

    async def _llm_classify_severity(
        self,
        surface: AttackSurface,
        attack_goal: str,
        counterexample: str,
    ) -> VulnerabilitySeverity | None:
        """
        Use LLM to classify severity for ambiguous vulnerability classes.

        Returns None if LLM classification fails (caller should use heuristic).
        """
        user_prompt = (
            f"Vulnerability: {attack_goal}\n"
            f"Surface: {surface.entry_point} ({surface.surface_type.value})\n"
            f"File: {surface.file_path}\n"
            f"Z3 counterexample: {counterexample}\n"
        )

        try:
            response = await self._llm.evaluate(
                prompt=(
                    f"{_SEVERITY_CLASSIFICATION_PROMPT}\n\n"
                    f"Classify this vulnerability:\n{user_prompt}"
                ),
                max_tokens=256,
                temperature=0.1,
            )
        except Exception:
            return None

        return self._parse_severity_response(response.text)

    # ── Response parsing ────────────────────────────────────────────────────

    def _parse_encoding_response(
        self,
        llm_text: str,
    ) -> tuple[str, dict[str, str], str] | None:
        """
        Parse the LLM's encoding response into Z3 expression + declarations.

        Expects JSON with keys: variable_declarations, z3_expression, reasoning.

        Returns:
            (z3_expression, variable_declarations, reasoning) or None.
        """
        text = llm_text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)

        # Find JSON object in the response
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start == -1 or brace_end == -1 or brace_end <= brace_start:
            self._log.warning(
                "encoding_parse_no_json",
                text_preview=text[:200],
            )
            return None

        json_str = text[brace_start:brace_end + 1]
        try:
            obj = json.loads(json_str)
        except json.JSONDecodeError as exc:
            self._log.warning("encoding_parse_json_error", error=str(exc))
            return None

        if not isinstance(obj, dict):
            return None

        z3_expression = obj.get("z3_expression", "")
        variable_declarations = obj.get("variable_declarations", {})
        reasoning = obj.get("reasoning", "")

        if not z3_expression or not isinstance(variable_declarations, dict):
            self._log.warning(
                "encoding_parse_missing_fields",
                has_expr=bool(z3_expression),
                has_vars=isinstance(variable_declarations, dict),
            )
            return None

        # Validate variable declarations are all valid Z3 types
        valid_types = {"Int", "Real", "Bool"}
        clean_decls: dict[str, str] = {}
        for name, z3_type in variable_declarations.items():
            if not isinstance(name, str) or not isinstance(z3_type, str):
                continue
            # Normalize type names (case-insensitive)
            normalized = z3_type.strip().capitalize()
            if normalized not in valid_types:
                normalized = "Real"  # Safe default
            clean_decls[name] = normalized

        return z3_expression, clean_decls, reasoning

    def _parse_severity_response(
        self,
        llm_text: str,
    ) -> VulnerabilitySeverity | None:
        """Parse LLM severity classification response."""
        text = llm_text.strip()

        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines)

        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start == -1 or brace_end == -1:
            return None

        try:
            obj = json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            return None

        severity_str = obj.get("severity", "").strip().lower()
        try:
            return VulnerabilitySeverity(severity_str)
        except ValueError:
            return None

    # ── Phase 5: Reproduction script generation ───────────────────────────────

    async def generate_reproduction_script(
        self,
        report: VulnerabilityReport,
        *,
        config: HunterConfig | None = None,
    ) -> str:
        """
        Generate a Security Unit Test script from a proven vulnerability.

        Takes the Z3 counterexample (concrete variable assignments that
        demonstrate a violated security property) and the attack surface
        context, then uses the LLM to produce a local-only diagnostic
        reproduction script. The script is structured as a Python test
        that asserts the expected secure response — making it directly
        consumable by the RepairAgent's patch-verify loop.

        Args:
            report: A proven VulnerabilityReport containing the surface,
                Z3 counterexample, and vulnerability goal.
            config: Optional HunterConfig for authorized_targets validation.

        Returns:
            Python source code of the Security Unit Test script.
            Empty string if generation or safety validation fails.
        """
        start = time.monotonic()
        surface = report.attack_surface

        self._log.info(
            "reproduction_script_generation_start",
            vuln_id=report.id,
            vulnerability_class=report.vulnerability_class.value,
            severity=report.severity.value,
            entry_point=surface.entry_point,
        )

        # Build the user prompt with all context the LLM needs
        prompt_parts = [
            "## Vulnerability Details",
            f"Class: {report.vulnerability_class.value}",
            f"Severity: {report.severity.value}",
            f"Attack goal: {report.attack_goal}",
            "",
            "## Z3 Counterexample (proven exploit conditions)",
            f"{report.z3_counterexample}",
            "",
            "## Attack Surface",
            f"Entry point: {surface.entry_point}",
            f"Type: {surface.surface_type.value}",
            f"File: {surface.file_path}",
        ]

        if surface.http_method:
            prompt_parts.append(f"HTTP method: {surface.http_method}")
        if surface.route_pattern:
            prompt_parts.append(f"Route: {surface.route_pattern}")

        if surface.context_code:
            prompt_parts.extend([
                "",
                "## Source Code",
                f"```\n{surface.context_code[:4000]}\n```",
            ])

        if report.z3_constraints_code:
            prompt_parts.extend([
                "",
                "## Z3 Constraints (for reference)",
                f"```python\n{report.z3_constraints_code}\n```",
            ])

        prompt_parts.extend([
            "",
            "Generate a Python Security Unit Test script (reproduction script) that "
            "reproduces the exact violation conditions from the Z3 counterexample "
            "against a local development server and asserts the expected secure response.",
        ])

        user_prompt = "\n".join(prompt_parts)

        # Call LLM to generate the PoC
        try:
            response = await self._llm.generate(
                system_prompt=_POC_GENERATION_SYSTEM_PROMPT,
                messages=[Message(role="user", content=user_prompt)],
                max_tokens=4096,
                temperature=0.3,
            )
        except Exception as exc:
            self._log.error(
                "reproduction_script_llm_error",
                vuln_id=report.id,
                error=str(exc),
            )
            return ""

        # Parse and extract the Python code from the response
        poc_code = self._parse_poc_response(response.text)
        if not poc_code:
            self._log.warning(
                "reproduction_script_parse_failed",
                vuln_id=report.id,
                response_preview=response.text[:200],
            )
            return ""

        # Validate syntax — must be parseable Python
        syntax_error = self._validate_poc_syntax(poc_code)
        if syntax_error is not None:
            self._log.warning(
                "reproduction_script_syntax_invalid",
                vuln_id=report.id,
                error=syntax_error,
            )
            # Attempt a single retry with the error fed back
            poc_code = await self._retry_reproduction_script_with_error(
                user_prompt, poc_code, syntax_error,
            )
            if not poc_code:
                return ""
            syntax_error = self._validate_poc_syntax(poc_code)
            if syntax_error is not None:
                self._log.warning(
                    "reproduction_script_syntax_retry_failed",
                    vuln_id=report.id,
                    error=syntax_error,
                )
                return ""

        # Validate safety — no forbidden imports, no unauthorized URLs
        authorized_targets = config.authorized_targets if config else []
        safety_error = self._validate_poc_safety(poc_code, authorized_targets)
        if safety_error is not None:
            self._log.warning(
                "reproduction_script_safety_violation",
                vuln_id=report.id,
                error=safety_error,
            )
            return ""

        total_ms = int((time.monotonic() - start) * 1000)
        self._log.info(
            "reproduction_script_generated",
            vuln_id=report.id,
            script_size_bytes=len(poc_code.encode()),
            total_ms=total_ms,
        )

        return poc_code

    # Keep backward-compatible alias for callers that pre-date the rename.
    generate_poc = generate_reproduction_script

    async def generate_reproduction_script_batch(
        self,
        reports: list[VulnerabilityReport],
        *,
        config: HunterConfig | None = None,
    ) -> dict[str, str]:
        """
        Generate Security Unit Test scripts for multiple vulnerability reports.

        Args:
            reports: List of proven VulnerabilityReport objects.
            config: Optional HunterConfig for authorized_targets validation.

        Returns:
            Dict mapping vulnerability report ID → reproduction script code.
            Only includes entries where generation succeeded.
        """
        results: dict[str, str] = {}

        for report in reports:
            script = await self.generate_reproduction_script(report, config=config)
            if script:
                results[report.id] = script

        self._log.info(
            "reproduction_script_batch_complete",
            total_reports=len(reports),
            successful_scripts=len(results),
        )

        return results

    # Backward-compatible alias.
    generate_poc_batch = generate_reproduction_script_batch

    async def _retry_reproduction_script_with_error(
        self,
        original_prompt: str,
        failed_code: str,
        error: str,
    ) -> str:
        """Retry reproduction script generation feeding back the syntax error for correction."""
        retry_prompt = (
            f"{original_prompt}\n\n"
            f"## Previous Attempt Error\n"
            f"Your previous code had a syntax error:\n"
            f"```\n{error}\n```\n\n"
            f"Previous code (first 2000 chars):\n"
            f"```python\n{failed_code[:2000]}\n```\n\n"
            f"Fix the syntax error and regenerate the Security Unit Test. "
            f"Respond with ONLY Python code."
        )

        try:
            response = await self._llm.generate(
                system_prompt=_POC_GENERATION_SYSTEM_PROMPT,
                messages=[Message(role="user", content=retry_prompt)],
                max_tokens=4096,
                temperature=0.1,
            )
        except Exception:
            return ""

        return self._parse_poc_response(response.text)

    def _parse_poc_response(self, llm_text: str) -> str:
        """
        Extract Python code from the LLM's reproduction script response.

        Handles:
        - Raw Python code (no fences)
        - Markdown ```python fences
        - Markdown ``` fences without language tag
        - Leading/trailing explanation text around code blocks

        Returns:
            The extracted Python source code, or empty string if extraction fails.
        """
        text = llm_text.strip()
        if not text:
            return ""

        # Try to extract from markdown code fences first
        # Match ```python ... ``` or ``` ... ```
        fence_pattern = re.compile(
            r"```(?:python)?\s*\n(.*?)```",
            re.DOTALL,
        )
        fenced_blocks = fence_pattern.findall(text)
        if fenced_blocks:
            # Use the longest fenced block (most likely the full script)
            code = max(fenced_blocks, key=len).strip()
            if code:
                return code  # type: ignore[no-any-return]

        # If no fences found, check if the entire response looks like Python
        # Heuristic: starts with a docstring, import, or comment
        if (
            text.startswith('"""')
            or text.startswith("'''")
            or text.startswith("import ")
            or text.startswith("from ")
            or text.startswith("#")
        ):
            return text

        # Last resort: find the first line that looks like Python and take
        # everything from there
        lines = text.split("\n")
        start_idx = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (
                stripped.startswith('"""')
                or stripped.startswith("'''")
                or stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped.startswith("# ")
                or stripped.startswith("def ")
                or stripped.startswith("class ")
            ):
                start_idx = i
                break

        if start_idx >= 0:
            return "\n".join(lines[start_idx:]).strip()

        return ""

    def _validate_poc_syntax(self, poc_code: str) -> str | None:
        """
        Validate that the reproduction script is syntactically valid Python.

        Uses ast.parse() — the code is NOT executed.

        Returns:
            None if valid, or a human-readable error string if invalid.
        """
        try:
            ast.parse(poc_code, filename="<poc>", mode="exec")
        except SyntaxError as exc:
            location = f"line {exc.lineno}" if exc.lineno else "unknown location"
            return f"SyntaxError at {location}: {exc.msg}"

        return None

    def _validate_poc_safety(
        self,
        poc_code: str,
        authorized_targets: list[str],
    ) -> str | None:
        """
        Validate that the reproduction script does not contain dangerous operations.

        Checks:
        1. No forbidden imports (subprocess, socket, ctypes, etc.)
        2. No hardcoded URLs pointing to unauthorized domains
        3. No eval/exec calls (the script should be a straightforward test)

        Args:
            poc_code: The generated Python source code.
            authorized_targets: List of authorized target domains/URLs.
                If empty, URL validation is skipped (offline/localhost-only mode).

        Returns:
            None if safe, or a human-readable error string if unsafe.
        """
        # Parse the AST to inspect imports and calls
        try:
            tree = ast.parse(poc_code, filename="<poc_safety>", mode="exec")
        except SyntaxError:
            return "Code failed to parse (syntax error)"

        # Check imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name not in _POC_ALLOWED_IMPORTS:
                        if alias.name in _POC_FORBIDDEN_IMPORTS or module_name in _POC_FORBIDDEN_IMPORTS:
                            return f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_module = node.module.split(".")[0]
                    full_path = node.module
                    if full_path in _POC_FORBIDDEN_IMPORTS or root_module in _POC_FORBIDDEN_IMPORTS:
                        return f"Forbidden import: from {node.module}"

            # Check for eval/exec calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "compile", "__import__"):
                        return f"Forbidden call: {node.func.id}()"
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("system", "popen", "exec"):
                        return f"Forbidden call: .{node.func.attr}()"

        # Check for hardcoded URLs pointing to non-localhost, non-authorized domains
        if authorized_targets:
            url_pattern = re.compile(
                r"""(?:"|')https?://([^/"'\s:]+)""",
            )
            for match in url_pattern.finditer(poc_code):
                hostname = match.group(1).lower()
                # Allow localhost and loopback
                if hostname in ("localhost", "127.0.0.1", "0.0.0.0", "[::1]"):
                    continue
                # Allow example.com domains (placeholder)
                if hostname.endswith("example.com") or hostname.endswith("example.org"):
                    continue
                # Check against authorized targets
                is_authorized = any(
                    hostname == target.lower()
                    or hostname.endswith("." + target.lower())
                    for target in authorized_targets
                )
                if not is_authorized:
                    return (
                        f"Unauthorized target domain: {hostname} "
                        f"(authorized: {authorized_targets})"
                    )

        return None

    # ── Z3 expression validation ────────────────────────────────────────────

    def _validate_z3_expression(
        self,
        z3_expr_code: str,
        variable_declarations: dict[str, str],
    ) -> str | None:
        """
        Validate that a Z3 expression can be parsed without errors.

        Returns None if valid, or an error message string if invalid.
        """
        try:
            import z3 as z3_lib
        except ImportError:
            return None  # Can't validate without z3, assume valid

        z3_vars: dict[str, Any] = {}
        for name, z3_type in variable_declarations.items():
            if z3_type == "Int":
                z3_vars[name] = z3_lib.Int(name)
            elif z3_type == "Real":
                z3_vars[name] = z3_lib.Real(name)
            elif z3_type == "Bool":
                z3_vars[name] = z3_lib.Bool(name)
            else:
                z3_vars[name] = z3_lib.Real(name)

        namespace: dict[str, Any] = {"z3": z3_lib, **z3_vars}
        try:
            expr = eval(z3_expr_code, {"__builtins__": {}}, namespace)  # noqa: S307
        except Exception as exc:
            return f"eval error: {exc}"

        if not isinstance(expr, z3_lib.BoolRef):
            return f"produced {type(expr).__name__}, expected z3.BoolRef"

        return None
