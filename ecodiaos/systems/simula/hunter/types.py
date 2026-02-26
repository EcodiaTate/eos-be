"""
EcodiaOS — Hunter Domain Types

All data models for the vulnerability discovery pipeline.
Uses EOSBaseModel for consistency with the rest of EOS.
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import Field, field_validator

from ecodiaos.primitives.common import EOSBaseModel, new_id, utc_now


# ── Enums ────────────────────────────────────────────────────────────────────


class TargetType(str, enum.Enum):
    """Whether the hunt target is internal EOS or an external repository."""

    INTERNAL_EOS = "internal_eos"
    EXTERNAL_REPO = "external_repo"


class AttackSurfaceType(str, enum.Enum):
    """Classification of an exploitable entry point."""

    API_ENDPOINT = "api_endpoint"
    MIDDLEWARE = "middleware"
    SMART_CONTRACT_PUBLIC = "smart_contract_public"
    FUNCTION_EXPORT = "function_export"
    CLI_COMMAND = "cli_command"
    WEBSOCKET_HANDLER = "websocket_handler"
    GRAPHQL_RESOLVER = "graphql_resolver"
    EVENT_HANDLER = "event_handler"
    DATABASE_QUERY = "database_query"
    FILE_UPLOAD = "file_upload"
    AUTH_HANDLER = "auth_handler"
    DESERIALIZATION = "deserialization"


class VulnerabilitySeverity(str, enum.Enum):
    """CVSS-aligned severity classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VulnerabilityClass(str, enum.Enum):
    """Common vulnerability taxonomy (OWASP-aligned)."""

    BROKEN_AUTH = "broken_authentication"
    BROKEN_ACCESS_CONTROL = "broken_access_control"
    INJECTION = "injection"
    SQL_INJECTION = "sql_injection"
    XSS = "cross_site_scripting"
    SSRF = "server_side_request_forgery"
    IDOR = "insecure_direct_object_reference"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    REENTRANCY = "reentrancy"
    RACE_CONDITION = "race_condition"
    UNVALIDATED_REDIRECT = "unvalidated_redirect"
    INFORMATION_DISCLOSURE = "information_disclosure"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    OTHER = "other"


# ── Data Models ──────────────────────────────────────────────────────────────


class AttackSurface(EOSBaseModel):
    """A discovered exploitable entry point in the target codebase."""

    id: str = Field(default_factory=new_id)
    entry_point: str = Field(
        ...,
        description="Qualified name of the entry point (e.g., 'app.routes.get_user')",
    )
    surface_type: AttackSurfaceType
    file_path: str = Field(
        ...,
        description="Relative path within the workspace to the source file",
    )
    line_number: int | None = Field(
        default=None,
        description="Starting line number of the entry point in the file",
    )
    context_code: str = Field(
        default="",
        description="Surrounding function/class source code for Z3 encoding",
    )
    http_method: str | None = Field(
        default=None,
        description="HTTP method if this is an API endpoint (GET, POST, etc.)",
    )
    route_pattern: str | None = Field(
        default=None,
        description="URL route pattern (e.g., '/api/user/{id}')",
    )
    discovered_at: datetime = Field(default_factory=utc_now)


class VulnerabilityReport(EOSBaseModel):
    """A proven vulnerability with Z3 counterexample and optional PoC."""

    id: str = Field(default_factory=new_id)
    target_url: str = Field(
        ...,
        description="GitHub URL or 'internal_eos'",
    )
    vulnerability_class: VulnerabilityClass
    severity: VulnerabilitySeverity
    attack_surface: AttackSurface
    attack_goal: str = Field(
        ...,
        description="The attacker goal that was proven satisfiable",
    )
    z3_counterexample: str = Field(
        ...,
        description="Human-readable Z3 model showing the exploit conditions",
    )
    z3_constraints_code: str = Field(
        default="",
        description="The Z3 Python code that was checked",
    )
    proof_of_concept_code: str = Field(
        default="",
        description="Generated exploit script (Python)",
    )
    verified: bool = Field(
        default=False,
        description="Whether the PoC was sandbox-verified",
    )
    discovered_at: datetime = Field(default_factory=utc_now)

    @field_validator("severity")
    @classmethod
    def _validate_severity(cls, v: VulnerabilitySeverity) -> VulnerabilitySeverity:
        if v not in VulnerabilitySeverity:
            raise ValueError(f"Invalid severity: {v}")
        return v


class HuntResult(EOSBaseModel):
    """Aggregated results from a full hunt against a target."""

    id: str = Field(default_factory=new_id)
    target_url: str
    target_type: TargetType
    surfaces_mapped: int = 0
    attack_surfaces: list[AttackSurface] = Field(default_factory=list)
    vulnerabilities_found: list[VulnerabilityReport] = Field(default_factory=list)
    generated_patches: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of vulnerability ID → patch diff",
    )
    total_duration_ms: int = 0
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None

    @property
    def vulnerability_count(self) -> int:
        return len(self.vulnerabilities_found)

    @property
    def critical_count(self) -> int:
        return sum(
            1 for v in self.vulnerabilities_found
            if v.severity == VulnerabilitySeverity.CRITICAL
        )

    @property
    def high_count(self) -> int:
        return sum(
            1 for v in self.vulnerabilities_found
            if v.severity == VulnerabilitySeverity.HIGH
        )


class HunterConfig(EOSBaseModel):
    """Configuration for a Hunter instance. Enforces safety constraints."""

    authorized_targets: list[str] = Field(
        default_factory=list,
        description="List of authorized target domains/URLs for PoC execution",
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Max concurrent attack surface analysis workers",
    )
    sandbox_timeout_seconds: int = Field(
        default=30,
        gt=0,
        description="Timeout for sandboxed PoC execution",
    )
    log_vulnerability_analytics: bool = Field(
        default=True,
        description="Whether to emit structlog analytics events for discoveries",
    )
    clone_depth: int = Field(
        default=1,
        ge=1,
        description="Git clone depth (1 = shallow clone for speed)",
    )

    @field_validator("authorized_targets")
    @classmethod
    def _validate_targets(cls, v: list[str]) -> list[str]:
        """Ensure authorized targets are non-empty strings."""
        for target in v:
            if not target.strip():
                raise ValueError("Authorized target cannot be an empty string")
        return v
