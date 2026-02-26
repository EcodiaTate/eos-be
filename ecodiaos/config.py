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
    uri: str = ""  # Required: set via ECODIAOS_NEO4J_URI (e.g., neo4j+s://xxx.databases.neo4j.io)
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
    max_calls_per_hour: int = 1_000
    max_tokens_per_hour: int = 600_000
    hard_limit: bool = False  # If True, reject requests when budget exhausted


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    fallback_provider: str | None = None
    fallback_model: str | None = None
    budget: LLMBudget = Field(default_factory=LLMBudget)
    budgets: dict[str, LLMBudget] = Field(default_factory=dict)  # Deprecated, use budget

    @model_validator(mode="after")
    def _strip_api_key(self) -> LLMConfig:
        # GCP Secret Manager can inject trailing \r\n into env vars
        if self.api_key:
            object.__setattr__(self, "api_key", self.api_key.strip())
        return self


class EmbeddingConfig(BaseModel):
    strategy: str = "local"  # "local" | "api" | "sidecar" | "mock"
    local_model: str = "sentence-transformers/all-mpnet-base-v2"
    local_device: str = "cpu"
    sidecar_url: str | None = None
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
    # Stage 1A: Extended-thinking model for governance-required and high-risk proposals
    thinking_model: str = "o3"
    thinking_model_provider: str = "openai"
    thinking_model_api_key: str = ""
    thinking_budget_tokens: int = 16384  # max reasoning tokens for extended thinking
    # Stage 1B: Code embeddings for semantic similarity
    embedding_model: str = "voyage-code-3"
    embedding_api_key: str = ""
    # Stage 1C: KV cache compression
    kv_compression_ratio: float = 0.3  # prune ratio for KVzip (0.0 = no pruning, 1.0 = max)
    kv_compression_enabled: bool = True
    # Stage 2A: Dafny formal verification (Clover pattern)
    dafny_enabled: bool = False  # requires Dafny binary on PATH
    dafny_binary_path: str = "dafny"
    dafny_verify_timeout_s: float = 30.0
    dafny_max_clover_rounds: int = 8
    dafny_blocking: bool = True  # Dafny failure blocks triggerable categories
    # Stage 2B: Z3 invariant discovery
    z3_enabled: bool = False  # requires z3-solver pip package
    z3_check_timeout_ms: int = 5000
    z3_max_discovery_rounds: int = 6
    z3_blocking: bool = False  # advisory by default, graduates to blocking
    # Stage 2C: Static analysis gates
    static_analysis_enabled: bool = True  # on by default (pip packages)
    static_analysis_max_fix_iterations: int = 3
    static_analysis_blocking: bool = True  # ERROR findings block proposals
    # Stage 2D: AgentCoder (separated test/code/execute pipeline)
    agent_coder_enabled: bool = False  # opt-in 3-agent pipeline
    agent_coder_max_iterations: int = 3
    agent_coder_test_timeout_s: float = 60.0
    # Stage 3A: Salsa Incremental Verification
    incremental_verification_enabled: bool = True  # dependency-aware memoization
    incremental_hot_ttl_seconds: int = 3600  # Redis hot cache TTL (1 hour)
    # Stage 3B: SWE-grep Agentic Retrieval
    swe_grep_enabled: bool = True  # multi-hop code search for bridge + code agent
    swe_grep_max_hops: int = 4  # serial retrieval turns (4 × 8 parallel tools)
    # Stage 3C: LILO Library Learning
    lilo_enabled: bool = True  # abstraction extraction from successful proposals
    lilo_max_library_size: int = 200  # cap on :LibraryAbstraction nodes
    lilo_consolidation_interval_proposals: int = 10  # consolidate every N applied proposals
    # Stage 4A: Lean 4 Proof Generation (DeepSeek-Prover-V2 pattern)
    lean_enabled: bool = False  # requires Lean 4 + Mathlib on PATH
    lean_binary_path: str = "lean"  # path to Lean 4 binary
    lean_project_path: str = ""  # path to lakefile.lean project (for Mathlib deps)
    lean_verify_timeout_s: float = 60.0  # per-proof verification timeout
    lean_max_attempts: int = 5  # max proof generation attempts
    lean_blocking: bool = True  # Lean failure blocks for proof-requiring categories
    lean_copilot_enabled: bool = True  # use Lean Copilot for tactic automation
    lean_dojo_enabled: bool = True  # use LeanDojo for proof search and retrieval
    lean_proof_library_max_size: int = 500  # cap on :ProvenLemma nodes in Neo4j
    # Stage 4B: GRPO Domain Fine-Tuning
    grpo_enabled: bool = False  # opt-in self-improvement training loop
    grpo_base_model: str = "deepseek-coder-7b"  # 7B base model for fine-tuning
    grpo_min_training_examples: int = 100  # minimum examples before first SFT
    grpo_sft_epochs: int = 3  # supervised fine-tuning epochs
    grpo_rollouts_per_example: int = 2  # 2-rollout contrastive (matches 16-rollout)
    grpo_batch_size: int = 8  # training batch size
    grpo_learning_rate: float = 2e-5  # fine-tuning learning rate
    grpo_retrain_interval_proposals: int = 50  # retrain every N applied proposals
    grpo_gpu_ids: list[int] = Field(default_factory=lambda: [0])  # GPU allocation
    grpo_use_finetuned: bool = False  # route code agent to fine-tuned model
    grpo_ab_test_fraction: float = 0.2  # fraction of proposals routed to fine-tuned
    # Stage 4C: Diffusion-Based Code Repair
    diffusion_repair_enabled: bool = False  # opt-in last-mile repair agent
    diffusion_model: str = "diffucoder-7b"  # diffusion model ID
    diffusion_max_denoise_steps: int = 10  # maximum denoising iterations
    diffusion_timeout_s: float = 120.0  # total repair timeout
    diffusion_sketch_first: bool = False  # True = skeleton mode, False = iterative denoise
    diffusion_handoff_after_failures: int = 2  # hand off to diffusion after N code agent failures
    # Stage 5A: Neurosymbolic Synthesis (beyond CEGIS)
    synthesis_enabled: bool = False  # opt-in neurosymbolic synthesis routing
    hysynth_enabled: bool = True  # probabilistic CFG-guided search (within synthesis)
    hysynth_max_candidates: int = 200  # max candidate programs per synthesis
    hysynth_beam_width: int = 10  # beam search width for bottom-up enumeration
    hysynth_timeout_s: float = 60.0  # per-synthesis timeout
    sketch_synthesis_enabled: bool = True  # LLM sketch + symbolic hole-filling
    sketch_max_holes: int = 20  # max holes per template
    sketch_solver_timeout_ms: int = 5000  # Z3/constraint solver timeout per hole
    chopchop_enabled: bool = True  # type-directed constrained generation
    chopchop_chunk_size_lines: int = 10  # lines per constrained generation chunk
    chopchop_max_retries: int = 3  # retries per chunk on constraint violation
    chopchop_timeout_s: float = 90.0  # total timeout
    # Stage 5B: Neural Program Repair (SRepair pattern)
    repair_agent_enabled: bool = False  # opt-in FSM-guided repair
    repair_diagnosis_model: str = "claude-opus-4-6"  # reasoning model for root cause
    repair_generation_model: str = "claude-sonnet-4-20250514"  # code model for fix gen
    repair_max_retries: int = 3  # max repair attempts per failure
    repair_cost_budget_usd: float = 0.10  # hard cap per repair attempt
    repair_timeout_s: float = 180.0  # total repair timeout
    repair_use_similar_fixes: bool = True  # query Neo4j for similar past repairs
    # Stage 5C: Multi-Agent Orchestration
    orchestration_enabled: bool = False  # opt-in multi-agent pipeline
    orchestration_max_agents_per_stage: int = 2  # per "overcrowding" finding
    orchestration_multi_file_threshold: int = 3  # files >= this triggers orchestrator
    orchestration_max_dag_nodes: int = 50  # cap on task decomposition DAG size
    orchestration_timeout_s: float = 300.0  # total orchestration timeout
    # Stage 5D: Causal Debugging
    causal_debugging_enabled: bool = False  # opt-in causal analysis on failure
    causal_max_interventions: int = 5  # max interventional queries per diagnosis
    causal_fault_injection_enabled: bool = False  # active causal learning (staging only)
    causal_timeout_s: float = 60.0  # per-diagnosis timeout
    # Stage 5E: Autonomous Issue Resolution
    issue_resolution_enabled: bool = False  # opt-in autonomous resolution
    issue_max_autonomy_level: str = "test_fix"  # lint|dependency|test_fix|logic_bug
    issue_abstention_confidence_threshold: float = 0.8  # below this, abstain
    issue_perf_regression_enabled: bool = True  # detect perf regressions post-apply
    issue_security_scan_enabled: bool = True  # enhanced security scanning
    issue_degradation_window_hours: int = 24  # monitor window for subtle degradation
    # Stage 6A: Cryptographic Auditability
    hash_chain_enabled: bool = False  # SHA-256 hash chains on EvolutionRecord nodes
    c2pa_enabled: bool = False  # C2PA content credentials for code provenance
    c2pa_signing_key_path: str = ""  # path to Ed25519 private key for signing
    c2pa_issuer_name: str = "EcodiaOS Simula"  # issuer name in C2PA manifests
    verifiable_credentials_enabled: bool = False  # tamper-evident governance approval chain
    credential_verification_timeout_s: float = 10.0  # timeout for credential verification
    regulatory_framework: str = ""  # ""|"finance_sox"|"healthcare_hipaa"|"defense_cmmc"|"general_audit"
    # Stage 6B: Co-Evolving Agents
    coevolution_enabled: bool = False  # autonomous hard negative mining + adversarial testing
    hard_negative_mining_interval_proposals: int = 10  # mine every N applied proposals
    adversarial_test_generation_enabled: bool = False  # LLM generates edge-case tests
    adversarial_max_tests_per_cycle: int = 20  # cap on adversarial tests per cycle
    coevolution_idle_compute_enabled: bool = False  # run adversarial generation on idle cycles
    # Stage 6C: Formal Spec Generation
    formal_spec_generation_enabled: bool = False  # auto-generate Dafny/TLA+/Alloy specs
    dafny_spec_generation_enabled: bool = True  # Dafny spec gen (within formal_spec_generation)
    dafny_bench_coverage_target: float = 0.96  # DafnyBench 96% coverage target
    tla_plus_enabled: bool = False  # TLA+ specs for distributed interactions
    tla_plus_binary_path: str = "tlc"  # path to TLC model checker binary
    tla_plus_model_check_timeout_s: float = 120.0  # per-model-check timeout
    alloy_enabled: bool = False  # Alloy for property checking on system invariants
    alloy_binary_path: str = "alloy"  # path to Alloy analyzer binary
    alloy_scope: int = 10  # Alloy scope (bound on universe size)
    self_spec_dsl_enabled: bool = False  # LLMs invent task-specific DSLs
    # Stage 6D: Equality Saturation (E-graphs)
    egraph_enabled: bool = False  # e-graph refactoring with semantic equivalence
    egraph_max_iterations: int = 1000  # max saturation iterations
    egraph_timeout_s: float = 30.0  # per-equivalence-check timeout
    egraph_blocking: bool = False  # advisory by default, equivalence failures don't block
    # Stage 6E: Hybrid Symbolic Execution
    symbolic_execution_enabled: bool = False  # Z3 SMT for mission-critical logic proofs
    symbolic_execution_timeout_ms: int = 10000  # Z3 per-property timeout
    symbolic_execution_blocking: bool = True  # proved properties are hard guarantees
    symbolic_execution_domains: list[str] = Field(
        default_factory=lambda: ["budget_calculation", "risk_scoring"],
    )  # domains to target for symbolic execution
    # Stage 7: Hunter — Zero-Day Discovery Engine
    hunter_enabled: bool = False  # opt-in vulnerability hunting
    hunter_max_workers: int = 4  # concurrent surface × goal analysis workers (1-16)
    hunter_sandbox_timeout_s: int = 30  # PoC sandbox execution timeout
    hunter_clone_depth: int = 1  # git clone depth (1 = shallow)
    hunter_log_analytics: bool = True  # emit structlog analytics events
    hunter_authorized_targets: list[str] = Field(
        default_factory=list,
    )  # domains allowed for PoC execution
    hunter_generate_pocs: bool = False  # auto-generate exploit PoC scripts
    hunter_generate_patches: bool = False  # auto-generate + verify patches
    hunter_remediation_enabled: bool = False  # enable HunterRepairOrchestrator


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


class SomaConfig(BaseModel):
    """Configuration for Soma — the interoceptive predictive substrate."""

    # Master on/off
    cycle_enabled: bool = True
    # Phase-space update frequency (every N theta cycles)
    phase_space_update_interval: int = 100
    # Trajectory ring buffer size (~150s at 150ms/tick)
    trajectory_buffer_size: int = 1000
    # EWM smoothing span for velocity estimation
    prediction_ewm_span: int = 20
    # EMA smoothing for setpoint context transitions
    setpoint_adaptation_alpha: float = 0.05
    # Urgency threshold for Nova allostatic deliberation
    urgency_threshold: float = 0.3
    # Minimum dwell cycles to declare a new attractor
    attractor_min_dwell_cycles: int = 50
    # Enable bifurcation boundary detection
    bifurcation_detection_enabled: bool = True
    # Maximum discoverable attractors
    max_attractors: int = 20
    # Enable somatic marker stamping on memory traces
    somatic_marker_enabled: bool = True
    # Maximum salience boost from somatic similarity
    somatic_rerank_boost: float = 0.3
    # Enable developmental stage gating
    developmental_gating_enabled: bool = True
    # Boot developmental stage
    initial_stage: str = "reflexive"


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
    soma: SomaConfig = Field(default_factory=SomaConfig)
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
    # Simula Stage 1 config
    if thinking_key := os.environ.get("ECODIAOS_SIMULA__THINKING_MODEL_API_KEY"):
        raw.setdefault("simula", {})["thinking_model_api_key"] = thinking_key
    if embedding_key := os.environ.get("ECODIAOS_SIMULA__EMBEDDING_API_KEY"):
        raw.setdefault("simula", {})["embedding_api_key"] = embedding_key
    # Simula Stage 4 config
    if lean_path := os.environ.get("ECODIAOS_SIMULA__LEAN_PROJECT_PATH"):
        raw.setdefault("simula", {})["lean_project_path"] = lean_path
    if grpo_gpus := os.environ.get("ECODIAOS_SIMULA__GRPO_GPU_IDS"):
        raw.setdefault("simula", {})["grpo_gpu_ids"] = [
            int(g.strip()) for g in grpo_gpus.split(",") if g.strip()
        ]

    return EcodiaOSConfig(**raw)


def load_seed(seed_path: str | Path) -> SeedConfig:
    """Load a seed configuration for birthing a new instance."""
    path = Path(seed_path)
    if not path.exists():
        raise FileNotFoundError(f"Seed config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return SeedConfig(**raw)
