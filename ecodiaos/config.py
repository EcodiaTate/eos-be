"""
EcodiaOS — Configuration System

All configuration is Pydantic-validated and loaded from:
1. default.yaml (defaults)
2. Environment variables (overrides)
3. Seed config (instance birth parameters)

Every tunable parameter in the system lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── Sub-configs ──────────────────────────────────────────────────


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    ws_port: int = 8001
    federation_port: int = 8002
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    api_key_header: str = "X-EOS-API-Key"
    # API keys for authentication. When empty, auth is disabled (dev mode).
    # Set via ECODIAOS_SERVER__API_KEYS or config YAML.
    api_keys: list[str] = Field(default_factory=list)


class Neo4jConfig(BaseModel):
    uri: str = "neo4j+s://localhost:7687"
    username: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    max_connection_pool_size: int = 20


class TimescaleDBConfig(BaseModel):
    host: str = "timescaledb"
    port: int = 5432
    database: str = "ecodiaos"
    schema_name: str = Field(default="public", alias="schema")
    username: str = "ecodiaos"
    password: str = "ecodiaos_dev"
    pool_size: int = 10
    ssl: bool = False

    model_config = {"populate_by_name": True}

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class RedisConfig(BaseModel):
    url: str = "redis://redis:6379/0"
    prefix: str = "eos"
    password: str = "ecodiaos_dev"

    @property
    def full_url(self) -> str:
        """Build URL with password injected."""
        clean_pw = self.password.strip() if self.password else ""
        if clean_pw and "://" in self.url:
            scheme, rest = self.url.split("://", 1)
            return f"{scheme}://:{clean_pw}@{rest}"
        return self.url


class LLMBudget(BaseModel):
    max_calls_per_hour: int = 60
    max_tokens_per_hour: int = 60000


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    fallback_provider: str | None = None
    fallback_model: str | None = None
    budgets: dict[str, LLMBudget] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _strip_api_key(self) -> "LLMConfig":
        # GCP Secret Manager can inject trailing \r\n into env vars
        if self.api_key:
            object.__setattr__(self, "api_key", self.api_key.strip())
        return self


class EmbeddingConfig(BaseModel):
    strategy: str = "local"  # "local" | "api" | "sidecar"
    local_model: str = "sentence-transformers/all-mpnet-base-v2"
    local_device: str = "cpu"
    dimension: int = 768
    max_batch_size: int = 32
    cache_embeddings: bool = True
    cache_ttl_seconds: int = 3600


class SynapseConfig(BaseModel):
    cycle_period_ms: int = 150
    min_cycle_period_ms: int = 80
    max_cycle_period_ms: int = 500
    health_check_interval_ms: int = 5000
    health_failure_threshold: int = 3
    coherence_update_interval: int = 50
    rhythm_enabled: bool = True


class AtuneConfig(BaseModel):
    ignition_threshold: float = 0.3
    workspace_buffer_size: int = 32
    spontaneous_recall_base_probability: float = 0.02
    max_percept_queue_size: int = 100


class NovaConfig(BaseModel):
    max_active_goals: int = 20
    fast_path_timeout_ms: int = 100
    slow_path_timeout_ms: int = 5000
    max_policies_per_deliberation: int = 5
    # EFE component weights (Evo adjusts these over time)
    efe_weight_pragmatic: float = 0.35
    efe_weight_epistemic: float = 0.20
    efe_weight_constitutional: float = 0.20
    efe_weight_feasibility: float = 0.15
    efe_weight_risk: float = 0.10
    # Memory retrieval timeout for belief enrichment
    memory_retrieval_timeout_ms: int = 150
    # Whether to use LLM for pragmatic/epistemic EFE estimation (vs. heuristics)
    use_llm_efe_estimation: bool = True


class EquorConfig(BaseModel):
    standard_review_timeout_ms: int = 500
    critical_review_timeout_ms: int = 50
    care_floor_multiplier: float = -0.3
    honesty_floor_multiplier: float = -0.3
    drift_window_size: int = 1000
    drift_report_interval: int = 1000  # every N reviews


class AxonConfig(BaseModel):
    max_actions_per_cycle: int = 5
    max_api_calls_per_minute: int = 30
    max_notifications_per_hour: int = 10
    max_concurrent_executions: int = 3
    total_timeout_per_cycle_ms: int = 30000


class VoxisConfig(BaseModel):
    max_expression_length: int = 2000
    min_expression_interval_minutes: int = 1
    voice_synthesis_enabled: bool = False
    # Proactive expression threshold — ambient insights below this are suppressed
    insight_expression_threshold: float = 0.6
    # Rolling message window kept verbatim in conversation context
    conversation_history_window: int = 50
    # Max tokens to pass to LLM as conversation history (older messages summarised)
    context_window_max_tokens: int = 4000
    # Summarise messages older than this count (keep last N verbatim)
    conversation_summary_threshold: int = 10
    # Whether to report expression feedback to Atune
    feedback_enabled: bool = True
    # Whether to run post-generation honesty check
    honesty_check_enabled: bool = True
    # Base LLM temperature before strategy modulation
    temperature_base: float = 0.7
    # Maximum concurrent tracked conversations in Redis
    max_active_conversations: int = 50


class EvoConfig(BaseModel):
    consolidation_interval_hours: int = 6
    consolidation_cycle_threshold: int = 10000
    max_active_hypotheses: int = 50
    max_parameter_delta_per_cycle: float = 0.03
    min_evidence_for_integration: int = 10


class SimulaConfig(BaseModel):
    max_simulation_episodes: int = 200
    regression_threshold_unacceptable: float = 0.10
    regression_threshold_high: float = 0.05
    # Code agent settings
    codebase_root: str = "."
    code_agent_model: str = "claude-opus-4-6"
    max_code_agent_turns: int = 20
    test_command: str = "pytest"
    auto_apply_self_applicable: bool = True


class ThymosConfig(BaseModel):
    # Sentinel scan interval (seconds)
    sentinel_scan_interval_s: float = 30.0
    # Homeostasis check interval (seconds)
    homeostasis_interval_s: float = 30.0
    # Post-repair verification timeout (seconds)
    post_repair_verify_timeout_s: float = 10.0
    # Healing governor limits
    max_concurrent_diagnoses: int = 3
    max_concurrent_codegen: int = 1
    storm_threshold: int = 10  # incidents per 60 seconds
    max_repairs_per_hour: int = 5
    max_novel_repairs_per_day: int = 3
    # Antibody library
    antibody_refinement_threshold: float = 0.6
    antibody_retirement_threshold: float = 0.3
    # Resource budget
    cpu_budget_fraction: float = 0.05
    burst_cpu_fraction: float = 0.15
    memory_budget_mb: int = 256


class OneirosConfig(BaseModel):
    # Circadian rhythm
    wake_duration_target_s: float = 79200.0     # 22 hours
    sleep_duration_target_s: float = 7200.0     # 2 hours

    # Sleep pressure
    pressure_threshold: float = 0.70
    pressure_critical: float = 0.95
    max_wake_cycles: int = 528000

    # Pressure weights
    pressure_weight_cycles: float = 0.40
    pressure_weight_affect: float = 0.25
    pressure_weight_episodes: float = 0.20
    pressure_weight_hypotheses: float = 0.15

    # NREM
    nrem_fraction: float = 0.40
    max_episodes_per_nrem: int = 200
    replay_batch_size: int = 10
    salience_decay_factor: float = 0.85
    salience_pruning_threshold: float = 0.05

    # REM
    rem_fraction: float = 0.40
    max_dreams_per_rem: int = 50
    dream_coherence_insight_threshold: float = 0.70
    dream_coherence_fragment_threshold: float = 0.40
    max_affect_traces_per_rem: int = 100
    affect_dampening_factor: float = 0.50
    max_threats_per_rem: int = 15
    max_ethical_cases_per_rem: int = 10

    # Lucid
    lucid_fraction: float = 0.10
    lucid_insight_threshold: float = 0.85
    max_explorations_per_lucid: int = 10

    # Sleep debt
    debt_salience_noise_max: float = 0.15
    debt_efe_precision_loss_max: float = 0.20
    debt_expression_flatness_max: float = 0.25
    debt_learning_rate_reduction_max: float = 0.30

    # Transitions
    hypnagogia_duration_s: float = 30.0
    hypnopompia_duration_s: float = 30.0


class FederationConfig(BaseModel):
    enabled: bool = False
    endpoint: str | None = None
    tls_cert_path: str | None = None
    tls_key_path: str | None = None
    ca_cert_path: str | None = None
    private_key_path: str | None = None  # Ed25519 signing key

    # Trust model
    auto_accept_links: bool = False
    trust_decay_enabled: bool = True
    trust_decay_rate_per_day: float = 0.1
    max_trust_level: int = 4  # TrustLevel.ALLY

    # Connection management
    link_timeout_ms: int = 3000
    knowledge_request_timeout_ms: int = 2000
    identity_verification_timeout_ms: int = 500
    heartbeat_interval_seconds: int = 30
    max_concurrent_links: int = 50

    # Privacy
    privacy_filter_enabled: bool = True
    allow_individual_data_sharing: bool = False  # NEVER true without individual consent

    # Knowledge exchange
    max_knowledge_items_per_request: int = 100
    knowledge_cache_ttl_seconds: int = 300

    # Local data directory for file-based fallback persistence
    data_dir: str = "data/federation"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "console"  # "console" | "json"


# ─── Seed Config (Birth Parameters) ──────────────────────────────


class PersonalityConfig(BaseModel):
    warmth: float = 0.0
    directness: float = 0.0
    verbosity: float = 0.0
    formality: float = 0.0
    curiosity_expression: float = 0.0
    humour: float = 0.0
    empathy_expression: float = 0.0
    confidence_display: float = 0.0
    metaphor_use: float = 0.0


class IdentityConfig(BaseModel):
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    traits: list[str] = Field(default_factory=list)
    voice_id: str | None = None


class ConstitutionalDrives(BaseModel):
    coherence: float = 1.0
    care: float = 1.0
    growth: float = 1.0
    honesty: float = 1.0


class GovernanceConfig(BaseModel):
    amendment_supermajority: float = 0.75
    amendment_quorum: float = 0.60
    amendment_deliberation_days: int = 14
    amendment_cooldown_days: int = 90


class ConstitutionConfig(BaseModel):
    drives: ConstitutionalDrives = Field(default_factory=ConstitutionalDrives)
    autonomy_level: int = 1
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)


class InitialEntity(BaseModel):
    name: str
    type: str
    description: str
    is_core_identity: bool = False


class InitialGoal(BaseModel):
    """An initial goal to seed at birth, giving Nova something to work toward."""

    description: str
    source: str = "self_generated"  # GoalSource value
    priority: float = 0.5
    importance: float = 0.5
    drive_alignment: dict[str, float] = Field(
        default_factory=lambda: {"coherence": 0.0, "care": 0.0, "growth": 0.0, "honesty": 0.0}
    )


class CommunityConfig(BaseModel):
    context: str = ""
    initial_entities: list[InitialEntity] = Field(default_factory=list)
    initial_goals: list[InitialGoal] = Field(default_factory=list)


class InstanceConfig(BaseModel):
    name: str = "EOS"
    description: str = ""


class SeedConfig(BaseModel):
    """The birth configuration for a new EOS instance."""

    instance: InstanceConfig = Field(default_factory=InstanceConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    constitution: ConstitutionConfig = Field(default_factory=ConstitutionConfig)
    community: CommunityConfig = Field(default_factory=CommunityConfig)


# ─── Root Configuration ──────────────────────────────────────────


class EcodiaOSConfig(BaseSettings):
    """
    Root configuration. Loads from YAML, overridable by env vars.
    """

    model_config = SettingsConfigDict(
        env_prefix="ECODIAOS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Instance identity
    instance_id: str = "eos-default"

    # Sub-configurations
    server: ServerConfig = Field(default_factory=ServerConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    timescaledb: TimescaleDBConfig = Field(default_factory=TimescaleDBConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    synapse: SynapseConfig = Field(default_factory=SynapseConfig)
    atune: AtuneConfig = Field(default_factory=AtuneConfig)
    nova: NovaConfig = Field(default_factory=NovaConfig)
    equor: EquorConfig = Field(default_factory=EquorConfig)
    axon: AxonConfig = Field(default_factory=AxonConfig)
    voxis: VoxisConfig = Field(default_factory=VoxisConfig)
    evo: EvoConfig = Field(default_factory=EvoConfig)
    simula: SimulaConfig = Field(default_factory=SimulaConfig)
    thymos: ThymosConfig = Field(default_factory=ThymosConfig)
    oneiros: OneirosConfig = Field(default_factory=OneirosConfig)
    federation: FederationConfig = Field(default_factory=FederationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path | None = None) -> EcodiaOSConfig:
    """
    Load configuration from YAML file, then apply environment variable overrides.
    """
    raw: dict[str, Any] = {}

    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}

    # Inject secrets from environment
    import os

    if neo4j_uri := os.environ.get("ECODIAOS_NEO4J_URI"):
        raw.setdefault("neo4j", {})["uri"] = neo4j_uri
    if neo4j_pw := os.environ.get("ECODIAOS_NEO4J_PASSWORD"):
        raw.setdefault("neo4j", {})["password"] = neo4j_pw
    if neo4j_db := os.environ.get("ECODIAOS_NEO4J_DATABASE"):
        raw.setdefault("neo4j", {})["database"] = neo4j_db
    if neo4j_user := os.environ.get("ECODIAOS_NEO4J_USERNAME"):
        raw.setdefault("neo4j", {})["username"] = neo4j_user
    if tsdb_host := os.environ.get("ECODIAOS_TIMESCALEDB__HOST"):
        raw.setdefault("timescaledb", {})["host"] = tsdb_host
    if tsdb_port := os.environ.get("ECODIAOS_TIMESCALEDB__PORT"):
        raw.setdefault("timescaledb", {})["port"] = int(tsdb_port)
    if tsdb_db := os.environ.get("ECODIAOS_TIMESCALEDB__DATABASE"):
        raw.setdefault("timescaledb", {})["database"] = tsdb_db
    if tsdb_user := os.environ.get("ECODIAOS_TIMESCALEDB__USERNAME"):
        raw.setdefault("timescaledb", {})["username"] = tsdb_user
    if tsdb_pw := os.environ.get("ECODIAOS_TSDB_PASSWORD"):
        raw.setdefault("timescaledb", {})["password"] = tsdb_pw
    if tsdb_ssl := os.environ.get("ECODIAOS_TIMESCALEDB__SSL"):
        raw.setdefault("timescaledb", {})["ssl"] = tsdb_ssl.lower() in ("true", "1", "yes")
    if redis_url := os.environ.get("ECODIAOS_REDIS__URL"):
        raw.setdefault("redis", {})["url"] = redis_url
    if redis_pw := os.environ.get("ECODIAOS_REDIS_PASSWORD"):
        raw.setdefault("redis", {})["password"] = redis_pw
    if llm_key := os.environ.get("ECODIAOS_LLM_API_KEY"):
        raw.setdefault("llm", {})["api_key"] = llm_key
    if llm_provider := os.environ.get("ECODIAOS_LLM__PROVIDER"):
        raw.setdefault("llm", {})["provider"] = llm_provider
    if llm_model := os.environ.get("ECODIAOS_LLM__MODEL"):
        raw.setdefault("llm", {})["model"] = llm_model
    if instance_id := os.environ.get("ECODIAOS_INSTANCE_ID"):
        raw["instance_id"] = instance_id

    return EcodiaOSConfig(**raw)


def load_seed(seed_path: str | Path) -> SeedConfig:
    """Load a seed configuration for birthing a new instance."""
    path = Path(seed_path)
    if not path.exists():
        raise FileNotFoundError(f"Seed config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return SeedConfig(**raw)