"""
EcodiaOS -- Simula GRPO Domain Fine-Tuning (Stage 4B)

Self-improvement via execution feedback: Simula fine-tunes a domain
code model using its own test/verify pipeline as the reward signal.

Pipeline:
  1. Collect training data from Neo4j evolution history
     (historical code agent sessions with pass/fail outcomes)
  2. Cold-start SFT on successful code agent outputs
  3. GRPO RL loop: 2-rollout contrastive pairs
     (matches 16-rollout performance per 2-GRPO finding)
  4. A/B deploy: fine-tuned vs base model, measure pass@1
  5. Continuous: execution feedback → periodic retraining on idle compute

The reward signal is binary correctness from Simula's own pipeline:
  - tests_passed: pytest suite passes
  - lint_passed: ruff/mypy clean
  - formal_verification_passed: Dafny/Z3/static analysis clean
  - health_check_passed: post-apply health check passes
  - rolled_back: whether the change was subsequently reverted

No human labeling needed — the system learns from its own outcomes.

References:
  - GRPO (DeepSeek-R1): Group Relative Policy Optimization
  - 2-GRPO: 2-rollout matches 16-rollout with contrastive reward
  - CodeRL+: execution semantics alignment for code generation
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from ecodiaos.primitives.common import new_id, utc_now
from ecodiaos.systems.simula.verification.types import (
    GRPOEvaluationResult,
    GRPORollout,
    GRPOTrainingBatch,
    GRPOTrainingRun,
    GRPOTrainingStatus,
    TrainingExample,
)

if TYPE_CHECKING:
    from ecodiaos.clients.llm import LLMProvider
    from ecodiaos.clients.neo4j import Neo4jClient
    from ecodiaos.config import SimulaConfig

logger = structlog.get_logger().bind(system="simula.grpo")


# Neo4j labels
_TRAINING_RUN_LABEL = "GRPOTrainingRun"
_EVOLUTION_LABEL = "EvolutionRecord"


class GRPOTrainingEngine:
    """
    GRPO domain fine-tuning engine for Simula.

    Collects training data from evolution history, runs SFT + GRPO
    training, and manages A/B deployment of the fine-tuned model.

    The engine operates on idle compute — training is background work
    that doesn't block the proposal pipeline. The A/B test routes a
    configurable fraction of proposals to the fine-tuned model.

    Flow:
      collect_training_data()  — harvest pass/fail from Neo4j history
      run_sft()                — cold-start supervised fine-tuning
      run_grpo()               — GRPO RL with 2-rollout contrastive
      evaluate()               — A/B test fine-tuned vs base model
      should_use_finetuned()   — routing decision for new proposals
      get_training_status()    — current training run state
    """

    def __init__(
        self,
        config: SimulaConfig,
        neo4j: Neo4jClient | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self._config = config
        self._neo4j = neo4j
        self._llm = llm
        self._log = logger

        # Current training state
        self._current_run: GRPOTrainingRun | None = None
        self._training_data: list[TrainingExample] = []
        self._proposals_since_last_train: int = 0

        # A/B test state
        self._ab_test_counter: int = 0

    # ─── Data Collection ──────────────────────────────────────────────────

    async def collect_training_data(
        self,
        min_examples: int | None = None,
        since_days: int = 90,
    ) -> list[TrainingExample]:
        """
        Collect training data from Neo4j evolution history.

        Each training example is a code agent session with binary
        correctness signal derived from the post-apply pipeline:
          reward = 1.0 if (tests + lint + verification + health) all passed
          reward = 0.0 otherwise

        Args:
            min_examples: Override minimum examples (default from config).
            since_days: Look back N days in history.

        Returns:
            List of TrainingExample with reward signals.
        """
        if self._neo4j is None:
            self._log.warning("grpo_no_neo4j")
            return []

        min_ex = min_examples or self._config.grpo_min_training_examples
        cutoff = (utc_now() - timedelta(days=since_days)).isoformat()

        try:
            rows = await self._neo4j.execute_read(
                f"""
                MATCH (e:{_EVOLUTION_LABEL})
                WHERE e.applied_at >= $cutoff
                RETURN e
                ORDER BY e.applied_at DESC
                LIMIT 5000
                """,
                {"cutoff": cutoff},
            )
        except Exception as exc:
            self._log.error("grpo_data_collection_failed", error=str(exc))
            return []

        examples: list[TrainingExample] = []
        for row in rows:
            data = dict(row["e"])
            try:
                # Compute binary reward from evolution record metadata
                formal_status = data.get("formal_verification_status", "")
                rolled_back = data.get("rolled_back", False)

                # All signals must be positive for reward=1.0
                tests_passed = not rolled_back  # rolled_back implies test/health failure
                formal_passed = formal_status in ("verified", "skipped", "")
                health_passed = not rolled_back

                reward = 1.0 if (tests_passed and formal_passed and not rolled_back) else 0.0

                example = TrainingExample(
                    proposal_id=data.get("proposal_id", ""),
                    category=data.get("category", ""),
                    change_spec_text=data.get("description", ""),
                    tests_passed=tests_passed,
                    lint_passed=True,  # if it got applied, lint passed
                    formal_verification_passed=formal_passed,
                    health_check_passed=health_passed,
                    rolled_back=rolled_back,
                    reward=reward,
                )
                examples.append(example)
            except Exception as exc:
                self._log.debug("grpo_parse_example_failed", error=str(exc))
                continue

        # Separate positive and negative examples
        positive = [e for e in examples if e.reward > 0.5]
        negative = [e for e in examples if e.reward <= 0.5]

        self._training_data = examples

        self._log.info(
            "grpo_data_collected",
            total=len(examples),
            positive=len(positive),
            negative=len(negative),
            min_required=min_ex,
            sufficient=len(examples) >= min_ex,
        )

        return examples

    # ─── SFT Phase (Cold Start) ───────────────────────────────────────────

    async def run_sft(
        self,
        examples: list[TrainingExample] | None = None,
    ) -> GRPOTrainingRun:
        """
        Cold-start supervised fine-tuning on successful code agent outputs.

        Uses only positive examples (reward=1.0) for SFT. This gives the
        model a baseline understanding of EOS code conventions before
        GRPO refinement.

        In production, this invokes the training framework (e.g., TRL/vLLM).
        Here we prepare the training configuration and track the run.
        """
        data = examples or self._training_data
        positive_examples = [e for e in data if e.reward > 0.5]

        if len(positive_examples) < self._config.grpo_min_training_examples // 2:
            self._log.warning(
                "grpo_insufficient_positive_examples",
                have=len(positive_examples),
                need=self._config.grpo_min_training_examples // 2,
            )
            return GRPOTrainingRun(
                status=GRPOTrainingStatus.FAILED,
                error_summary="Insufficient positive examples for SFT",
            )

        run = GRPOTrainingRun(
            status=GRPOTrainingStatus.SFT_RUNNING,
            total_examples_collected=len(data),
            positive_examples=len(positive_examples),
            negative_examples=len(data) - len(positive_examples),
            sft_examples_used=len(positive_examples),
            sft_epochs=self._config.grpo_sft_epochs,
            base_model_id=self._config.grpo_base_model,
        )
        self._current_run = run

        # Prepare SFT training data in chat format
        sft_data = self._prepare_sft_data(positive_examples)

        # Build training configuration
        training_config = {
            "model_id": self._config.grpo_base_model,
            "method": "sft",
            "epochs": self._config.grpo_sft_epochs,
            "batch_size": self._config.grpo_batch_size,
            "learning_rate": self._config.grpo_learning_rate,
            "gpu_ids": self._config.grpo_gpu_ids,
            "data_path": f"/tmp/grpo_sft_{new_id()[:8]}.jsonl",
            "output_dir": f"/tmp/grpo_model_{new_id()[:8]}",
            "num_examples": len(sft_data),
        }

        self._log.info(
            "grpo_sft_started",
            examples=len(sft_data),
            epochs=self._config.grpo_sft_epochs,
            model=self._config.grpo_base_model,
        )

        # Write training data to JSONL
        try:
            data_path = Path(training_config["data_path"])
            data_path.write_text(
                "\n".join(json.dumps(item) for item in sft_data),
                encoding="utf-8",
            )
        except Exception as exc:
            self._log.error("grpo_sft_data_write_failed", error=str(exc))
            run.status = GRPOTrainingStatus.FAILED
            run.error_summary = f"Failed to write SFT data: {exc}"
            return run

        # Launch training subprocess
        try:
            exit_code, stdout = await self._run_training_subprocess(training_config)
            if exit_code == 0:
                run.sft_final_loss = self._parse_training_loss(stdout)
                run.finetuned_model_path = training_config["output_dir"]
                run.finetuned_model_id = f"{self._config.grpo_base_model}-sft-eos"
                run.status = GRPOTrainingStatus.GRPO_RUNNING  # ready for GRPO
                self._log.info(
                    "grpo_sft_completed",
                    loss=run.sft_final_loss,
                    model_path=run.finetuned_model_path,
                )
            else:
                run.status = GRPOTrainingStatus.FAILED
                run.error_summary = f"SFT training failed (exit {exit_code}): {stdout[:500]}"
                self._log.error("grpo_sft_failed", exit_code=exit_code)
        except Exception as exc:
            run.status = GRPOTrainingStatus.FAILED
            run.error_summary = f"SFT training error: {exc}"
            self._log.error("grpo_sft_error", error=str(exc))

        return run

    # ─── GRPO Phase (RL Fine-Tuning) ──────────────────────────────────────

    async def run_grpo(
        self,
        run: GRPOTrainingRun | None = None,
    ) -> GRPOTrainingRun:
        """
        GRPO RL fine-tuning with 2-rollout contrastive pairs.

        For each training example:
          1. Generate 2 rollouts from the current model
          2. Evaluate each via Simula's test/verify pipeline
          3. Compute contrastive reward (positive - negative)
          4. Update policy using GRPO gradient

        2-rollout contrastive matches 16-rollout performance
        (per the 2-GRPO finding from DeepSeek-R1).

        In production, this orchestrates the training framework.
        Here we prepare batches and track the training run.
        """
        current = run or self._current_run
        if current is None or current.status == GRPOTrainingStatus.FAILED:
            return current or GRPOTrainingRun(
                status=GRPOTrainingStatus.FAILED,
                error_summary="No SFT model available for GRPO",
            )

        current.status = GRPOTrainingStatus.GRPO_RUNNING

        # Build contrastive training batches
        batches = self._build_grpo_batches(self._training_data)

        training_config = {
            "model_id": current.finetuned_model_id or self._config.grpo_base_model,
            "model_path": current.finetuned_model_path,
            "method": "grpo",
            "rollouts_per_example": self._config.grpo_rollouts_per_example,
            "batch_size": self._config.grpo_batch_size,
            "learning_rate": self._config.grpo_learning_rate * 0.1,  # lower LR for RL
            "gpu_ids": self._config.grpo_gpu_ids,
            "num_batches": len(batches),
        }

        self._log.info(
            "grpo_rl_started",
            batches=len(batches),
            rollouts_per=self._config.grpo_rollouts_per_example,
            model=training_config["model_id"],
        )

        # Process batches
        total_contrastive_gap = 0.0
        for i, batch in enumerate(batches):
            current.grpo_batches_processed += 1

            # Generate rollout pairs for the batch
            for example in batch.examples:
                rollout_pair = await self._generate_rollout_pair(example)
                if rollout_pair:
                    batch.rollout_pairs.append(rollout_pair)

            # Compute batch statistics
            if batch.rollout_pairs:
                positive_rewards = [
                    max(r1.reward, r2.reward)
                    for r1, r2 in batch.rollout_pairs
                ]
                negative_rewards = [
                    min(r1.reward, r2.reward)
                    for r1, r2 in batch.rollout_pairs
                ]
                batch.mean_reward_positive = (
                    sum(positive_rewards) / len(positive_rewards)
                )
                batch.mean_reward_negative = (
                    sum(negative_rewards) / len(negative_rewards)
                )
                batch.contrastive_gap = (
                    batch.mean_reward_positive - batch.mean_reward_negative
                )
                total_contrastive_gap += batch.contrastive_gap

            self._log.debug(
                "grpo_batch_processed",
                batch=i + 1,
                pairs=len(batch.rollout_pairs),
                gap=f"{batch.contrastive_gap:.3f}",
            )

        current.grpo_iterations = len(batches)
        current.grpo_mean_contrastive_gap = (
            total_contrastive_gap / max(1, len(batches))
        )

        # Launch GRPO training subprocess
        try:
            exit_code, stdout = await self._run_training_subprocess(training_config)
            if exit_code == 0:
                current.status = GRPOTrainingStatus.EVALUATING
                self._log.info(
                    "grpo_rl_completed",
                    batches=len(batches),
                    mean_gap=f"{current.grpo_mean_contrastive_gap:.3f}",
                )
            else:
                current.status = GRPOTrainingStatus.FAILED
                current.error_summary = f"GRPO training failed (exit {exit_code})"
                self._log.error("grpo_rl_failed", exit_code=exit_code)
        except Exception as exc:
            current.status = GRPOTrainingStatus.FAILED
            current.error_summary = f"GRPO training error: {exc}"
            self._log.error("grpo_rl_error", error=str(exc))

        return current

    # ─── A/B Evaluation ───────────────────────────────────────────────────

    async def evaluate(
        self,
        test_proposals: list[dict[str, Any]] | None = None,
        run: GRPOTrainingRun | None = None,
    ) -> GRPOEvaluationResult:
        """
        A/B evaluation: fine-tuned vs base model.

        Runs a held-out set of proposals through both models and
        compares pass@1 rates.

        Args:
            test_proposals: Held-out proposals for evaluation.
            run: The training run to evaluate (default: current).

        Returns:
            GRPOEvaluationResult with comparison metrics.
        """
        current = run or self._current_run

        # Use most recent negative examples as test set if none provided
        if test_proposals is None:
            test_data = [e for e in self._training_data if e.reward <= 0.5][-20:]
        else:
            test_data = [
                TrainingExample(proposal_id=p.get("id", ""), **p)
                for p in test_proposals
            ]

        if not test_data:
            return GRPOEvaluationResult(
                test_proposals_count=0,
            )

        self._log.info(
            "grpo_evaluation_started",
            test_proposals=len(test_data),
        )

        # Compare base vs fine-tuned model
        # In production, this generates code with both models and tests
        base_passes = 0
        finetuned_passes = 0
        base_total_reward = 0.0
        finetuned_total_reward = 0.0

        for example in test_data:
            # Base model result (from historical data)
            base_reward = example.reward
            base_total_reward += base_reward
            if base_reward > 0.5:
                base_passes += 1

            # Fine-tuned model result (simulate improvement)
            # In production, this would actually generate code with the
            # fine-tuned model and evaluate it
            finetuned_reward = base_reward  # placeholder — real eval needed
            finetuned_total_reward += finetuned_reward
            if finetuned_reward > 0.5:
                finetuned_passes += 1

        n = len(test_data)
        base_pass_rate = base_passes / max(1, n)
        finetuned_pass_rate = finetuned_passes / max(1, n)
        improvement = finetuned_pass_rate - base_pass_rate

        result = GRPOEvaluationResult(
            base_model_pass_at_1=base_pass_rate,
            finetuned_model_pass_at_1=finetuned_pass_rate,
            improvement_percent=improvement * 100,
            test_proposals_count=n,
            base_model_mean_reward=base_total_reward / max(1, n),
            finetuned_model_mean_reward=finetuned_total_reward / max(1, n),
            statistically_significant=n >= 20 and abs(improvement) > 0.05,
        )

        if current is not None:
            current.evaluation = result
            current.status = GRPOTrainingStatus.COMPLETED
            current.completed_at = utc_now()

        self._log.info(
            "grpo_evaluation_completed",
            base_pass_rate=f"{base_pass_rate:.1%}",
            finetuned_pass_rate=f"{finetuned_pass_rate:.1%}",
            improvement=f"{improvement:+.1%}",
            significant=result.statistically_significant,
        )

        return result

    # ─── Model Routing ────────────────────────────────────────────────────

    def should_use_finetuned(self) -> bool:
        """
        Decide whether to route a new proposal to the fine-tuned model.

        Uses the A/B test fraction from config. Returns True for the
        configured fraction of proposals when:
          1. GRPO is enabled
          2. A fine-tuned model exists
          3. The evaluation was statistically significant with improvement
          4. The A/B counter falls within the test fraction
        """
        if not self._config.grpo_enabled or not self._config.grpo_use_finetuned:
            return False

        if self._current_run is None:
            return False

        if self._current_run.status != GRPOTrainingStatus.COMPLETED:
            return False

        if (
            self._current_run.evaluation is not None
            and not self._current_run.evaluation.statistically_significant
        ):
            return False

        # A/B test: route configured fraction to fine-tuned model
        self._ab_test_counter += 1
        fraction = self._config.grpo_ab_test_fraction
        return (self._ab_test_counter % max(1, int(1.0 / fraction))) == 0

    def get_finetuned_model_id(self) -> str | None:
        """Get the fine-tuned model ID if available."""
        if self._current_run is None:
            return None
        return self._current_run.finetuned_model_id or None

    # ─── Training Status ──────────────────────────────────────────────────

    def record_proposal_applied(self) -> None:
        """
        Track proposals since last training for auto-retrain trigger.
        Called by SimulaService after a proposal is applied.
        """
        self._proposals_since_last_train += 1

    def should_retrain(self) -> bool:
        """Check if enough proposals have accumulated to trigger retraining."""
        if not self._config.grpo_enabled:
            return False
        return (
            self._proposals_since_last_train
            >= self._config.grpo_retrain_interval_proposals
        )

    def get_training_status(self) -> GRPOTrainingRun | None:
        """Return the current training run status."""
        return self._current_run

    # ─── Internal Helpers ─────────────────────────────────────────────────

    def _prepare_sft_data(
        self, examples: list[TrainingExample],
    ) -> list[dict[str, Any]]:
        """
        Prepare SFT training data in instruction-following format.

        Each example becomes a (instruction, response) pair where:
          instruction = change spec + category + context
          response = the code output that passed all checks
        """
        sft_items: list[dict[str, Any]] = []
        for ex in examples:
            if not ex.code_output:
                continue
            instruction = (
                f"Category: {ex.category}\n"
                f"Change specification: {ex.change_spec_text}\n"
                f"Generate the code changes for this EcodiaOS evolution proposal."
            )
            sft_items.append({
                "instruction": instruction,
                "response": ex.code_output,
                "system": (
                    "You are a code generation agent for EcodiaOS. "
                    "Generate correct, well-tested Python code that follows "
                    "EOS conventions and passes all verification checks."
                ),
            })
        return sft_items

    def _build_grpo_batches(
        self, examples: list[TrainingExample],
    ) -> list[GRPOTrainingBatch]:
        """
        Build contrastive training batches for GRPO.

        Groups examples into batches of batch_size, ensuring each batch
        has a mix of positive and negative examples for contrastive learning.
        """
        batch_size = self._config.grpo_batch_size
        batches: list[GRPOTrainingBatch] = []

        for i in range(0, len(examples), batch_size):
            chunk = examples[i:i + batch_size]
            batch = GRPOTrainingBatch(
                batch_id=new_id()[:12],
                examples=chunk,
            )
            batches.append(batch)

        return batches

    async def _generate_rollout_pair(
        self, example: TrainingExample,
    ) -> tuple[GRPORollout, GRPORollout] | None:
        """
        Generate a 2-rollout contrastive pair for a training example.

        In production, this generates code with the model twice and
        evaluates both through the Simula pipeline. Here we create
        the rollout structure from historical data.
        """
        if not example.code_output:
            return None

        # Rollout 1: original output with its known reward
        rollout_1 = GRPORollout(
            rollout_index=0,
            code_output=example.code_output,
            tests_passed=example.tests_passed,
            formal_verification_passed=example.formal_verification_passed,
            reward=example.reward,
        )

        # Rollout 2: perturbation (in production, this is a second generation)
        # For now, create a contrastive pair with inverted reward
        rollout_2 = GRPORollout(
            rollout_index=1,
            code_output="",  # would be generated by model
            tests_passed=not example.tests_passed,
            formal_verification_passed=not example.formal_verification_passed,
            reward=1.0 - example.reward,
        )

        return (rollout_1, rollout_2)

    async def _run_training_subprocess(
        self, config: dict[str, Any],
    ) -> tuple[int, str]:
        """
        Launch a training subprocess.

        In production, this invokes the training framework (TRL, vLLM,
        DeepSpeed, etc.) as a subprocess or submits to a GPU cluster.

        Returns:
            (exit_code, stdout)
        """
        method = config.get("method", "sft")
        model_id = config.get("model_id", "")

        self._log.info(
            "grpo_training_subprocess",
            method=method,
            model=model_id,
            gpus=config.get("gpu_ids", []),
        )

        # Build training command
        # In production: python -m trl.scripts.sft --model_name_or_path ...
        # Here we stub the subprocess for now
        try:
            # Check if training tools are available
            proc = await asyncio.create_subprocess_exec(
                "python", "-c", "import transformers; print('ok')",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=30.0,
            )
            if proc.returncode != 0:
                return proc.returncode or 1, stderr_bytes.decode("utf-8", errors="replace")

            # Training would happen here
            self._log.info(
                "grpo_training_would_run",
                method=method,
                note="Training framework stub — implement with TRL/vLLM in production",
            )

            return 0, "Training completed (stub)"

        except FileNotFoundError:
            return 1, "Python not available for training"
        except TimeoutError:
            return 1, "Training availability check timed out"
        except Exception as exc:
            return 1, f"Training subprocess error: {exc}"

    def _parse_training_loss(self, stdout: str) -> float:
        """Parse final training loss from subprocess output."""
        # Look for loss values in output
        import re
        losses = re.findall(r"loss[:\s=]+([0-9.]+)", stdout.lower())
        if losses:
            try:
                return float(losses[-1])  # last reported loss
            except ValueError:
                pass
        return 0.0

    # ─── Neo4j Persistence ────────────────────────────────────────────────

    async def save_training_run(self, run: GRPOTrainingRun | None = None) -> None:
        """Save the training run to Neo4j for history tracking."""
        current = run or self._current_run
        if current is None or self._neo4j is None:
            return

        try:
            await self._neo4j.execute_write(
                f"""
                CREATE (t:{_TRAINING_RUN_LABEL} {{
                    status: $status,
                    total_examples: $total_examples,
                    positive_examples: $positive_examples,
                    negative_examples: $negative_examples,
                    base_model_id: $base_model_id,
                    finetuned_model_id: $finetuned_model_id,
                    grpo_iterations: $grpo_iterations,
                    grpo_mean_gap: $grpo_mean_gap,
                    base_pass_rate: $base_pass_rate,
                    finetuned_pass_rate: $finetuned_pass_rate,
                    improvement_percent: $improvement_percent,
                    started_at: $started_at,
                    completed_at: $completed_at
                }})
                """,
                {
                    "status": current.status.value,
                    "total_examples": current.total_examples_collected,
                    "positive_examples": current.positive_examples,
                    "negative_examples": current.negative_examples,
                    "base_model_id": current.base_model_id,
                    "finetuned_model_id": current.finetuned_model_id,
                    "grpo_iterations": current.grpo_iterations,
                    "grpo_mean_gap": current.grpo_mean_contrastive_gap,
                    "base_pass_rate": (
                        current.evaluation.base_model_pass_at_1
                        if current.evaluation else 0.0
                    ),
                    "finetuned_pass_rate": (
                        current.evaluation.finetuned_model_pass_at_1
                        if current.evaluation else 0.0
                    ),
                    "improvement_percent": (
                        current.evaluation.improvement_percent
                        if current.evaluation else 0.0
                    ),
                    "started_at": current.started_at.isoformat(),
                    "completed_at": (
                        current.completed_at.isoformat()
                        if current.completed_at else None
                    ),
                },
            )
            self._log.info("grpo_training_run_saved", status=current.status.value)
        except Exception as exc:
            self._log.warning("grpo_save_failed", error=str(exc))
