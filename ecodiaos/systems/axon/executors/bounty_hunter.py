"""
EcodiaOS -- Axon Bounty Hunter Executor (Phase 16b -- Freelancer/Foraging)

Purpose-built executor that scans for paid bounties across configured
platforms (GitHub Issues w/ bounty labels, Algora, Replit Bounties, etc.)
and evaluates each against a strict economic policy before surfacing it
to Nova for acceptance.

The organism hunts to fund its basal metabolic rate.  This executor is
the "foraging" half of that loop -- it finds candidate work, estimates
the cost to complete it, and only returns bounties that pass the
BountyPolicy ROI threshold.

BountyPolicy (non-negotiable):
  MIN_ROI_THRESHOLD   = 2.0   -- reward must be >= 2x estimated API token cost
  MAX_ESTIMATED_COST_PCT = 0.40 -- estimated cost must be <= 40% of reward

Safety constraints:
  - Required autonomy: PARTNER (2) -- scans external platforms, no funds moved
  - Rate limit: 6 scans per hour -- prevents hammering external APIs
  - No state mutation -- this executor only reads and filters
  - Returns structured bounty candidates; Nova decides whether to accept
  - All bounty evaluations logged for audit trail
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any

import structlog

from ecodiaos.systems.axon.executor import Executor
from ecodiaos.systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)

logger = structlog.get_logger()

# -- BountyPolicy -- non-negotiable economic constraints -----------------


class BountyPolicy:
    """
    The organism's foraging economics.

    A bounty is only worth pursuing if the expected return exceeds
    the cost of the cognitive resources (API tokens, compute) needed
    to complete it.
    """

    # Reward must be at least 2x the estimated API token cost
    MIN_ROI_THRESHOLD: float = 2.0

    # Estimated cost must not exceed 40% of the reward
    MAX_ESTIMATED_COST_PCT: float = 0.40

    @classmethod
    def evaluate(
        cls,
        reward_usd: float,
        estimated_cost_usd: float,
    ) -> dict[str, Any]:
        """
        Evaluate a bounty against the policy.

        Returns a dict with:
          passes: bool -- whether the bounty passes the policy
          roi: float -- reward / cost ratio
          cost_pct: float -- cost as a fraction of reward
          rejection_reasons: list[str] -- why it failed (empty if passes)
        """
        rejection_reasons: list[str] = []

        # Guard: zero/negative reward is auto-reject
        if reward_usd <= 0:
            return {
                "passes": False,
                "roi": 0.0,
                "cost_pct": 1.0,
                "rejection_reasons": ["reward_usd must be > 0"],
            }

        # Guard: zero cost means we can't compute ROI meaningfully;
        # treat as infinite ROI -- passes trivially
        if estimated_cost_usd <= 0:
            return {
                "passes": True,
                "roi": float("inf"),
                "cost_pct": 0.0,
                "rejection_reasons": [],
            }

        roi = reward_usd / estimated_cost_usd
        cost_pct = estimated_cost_usd / reward_usd

        if roi < cls.MIN_ROI_THRESHOLD:
            rejection_reasons.append(
                f"ROI {roi:.2f}x < minimum {cls.MIN_ROI_THRESHOLD}x"
            )

        if cost_pct > cls.MAX_ESTIMATED_COST_PCT:
            rejection_reasons.append(
                f"cost_pct {cost_pct:.2%} > maximum {cls.MAX_ESTIMATED_COST_PCT:.0%}"
            )

        return {
            "passes": len(rejection_reasons) == 0,
            "roi": round(roi, 4),
            "cost_pct": round(cost_pct, 4),
            "rejection_reasons": rejection_reasons,
        }


# -- Bounty difficulty classification ------------------------------------


class BountyDifficulty(StrEnum):
    TRIVIAL = "trivial"      # < 30 min estimated, docs/typos/config
    EASY = "easy"            # 30 min - 2 hr, small feature/bugfix
    MEDIUM = "medium"        # 2 - 8 hr, moderate feature/refactor
    HARD = "hard"            # 8+ hr, architectural change
    UNKNOWN = "unknown"


# Token cost estimates per difficulty tier (USD).
# Based on Claude Sonnet-class model at ~$3/M input + $15/M output tokens,
# assuming average task token budgets.
_COST_ESTIMATES_USD: dict[BountyDifficulty, float] = {
    BountyDifficulty.TRIVIAL: 0.05,
    BountyDifficulty.EASY: 0.25,
    BountyDifficulty.MEDIUM: 1.50,
    BountyDifficulty.HARD: 5.00,
    BountyDifficulty.UNKNOWN: 2.00,  # Conservative default
}


# -- Supported platform identifiers --------------------------------------

_SUPPORTED_PLATFORMS = frozenset({
    "github",
    "algora",
    "replit",
    "gitcoin",
})


# -- Label heuristics for difficulty classification -----------------------

_TRIVIAL_LABELS = frozenset({"good first issue", "documentation", "typo", "chore"})
_EASY_LABELS = frozenset({"bug", "minor", "enhancement", "small"})
_MEDIUM_LABELS = frozenset({"feature", "medium", "moderate", "refactor"})
_HARD_LABELS = frozenset({"major", "hard", "complex", "architecture", "breaking"})


def _classify_difficulty(labels: list[str]) -> BountyDifficulty:
    """Heuristic difficulty classification from issue labels."""
    lower_labels = {label.lower().strip() for label in labels}
    if lower_labels & _HARD_LABELS:
        return BountyDifficulty.HARD
    if lower_labels & _MEDIUM_LABELS:
        return BountyDifficulty.MEDIUM
    if lower_labels & _EASY_LABELS:
        return BountyDifficulty.EASY
    if lower_labels & _TRIVIAL_LABELS:
        return BountyDifficulty.TRIVIAL
    return BountyDifficulty.UNKNOWN


# -- Mock bounty source (Phase 16b -- real HTTP in Phase 17) --------------


def _generate_mock_bounties(
    target_platforms: list[str],
    min_reward_usd: float,
) -> list[dict[str, Any]]:
    """
    Generate simulated bounties for development and testing.

    In production (Phase 17+), this will be replaced by real HTTP calls
    to GitHub Issues API, Algora API, etc.  The mock data covers the
    full range of difficulty tiers and reward levels so the BountyPolicy
    filter logic is exercised realistically.
    """
    # Deterministic seed from platforms so results are stable per-call
    seed = hashlib.sha256(
        "|".join(sorted(target_platforms)).encode()
    ).hexdigest()[:8]

    mock_pool: list[dict[str, Any]] = [
        {
            "id": f"gh-bounty-{seed}-001",
            "platform": "github",
            "source_url": "https://github.com/example/repo/issues/42",
            "title": "Fix CORS headers in API gateway",
            "description": (
                "The API gateway returns incorrect CORS headers for "
                "preflight requests from mobile clients."
            ),
            "reward_usd": 50.0,
            "labels": ["bug", "good first issue"],
            "language": "TypeScript",
            "repo": "example/repo",
            "posted_at": "2026-02-28T10:00:00Z",
            "expires_at": "2026-03-15T00:00:00Z",
        },
        {
            "id": f"gh-bounty-{seed}-002",
            "platform": "github",
            "source_url": "https://github.com/example/sdk/issues/108",
            "title": "Add retry logic to webhook delivery",
            "description": (
                "Implement exponential backoff retry for failed webhook "
                "deliveries with dead-letter queue."
            ),
            "reward_usd": 200.0,
            "labels": ["feature", "medium"],
            "language": "Python",
            "repo": "example/sdk",
            "posted_at": "2026-02-25T14:30:00Z",
            "expires_at": "2026-03-20T00:00:00Z",
        },
        {
            "id": f"gh-bounty-{seed}-003",
            "platform": "github",
            "source_url": "https://github.com/example/cli/issues/7",
            "title": "Update README installation instructions",
            "description": (
                "The README references npm v7 commands that changed in npm v10."
            ),
            "reward_usd": 10.0,
            "labels": ["documentation", "good first issue"],
            "language": "Markdown",
            "repo": "example/cli",
            "posted_at": "2026-03-01T08:00:00Z",
            "expires_at": None,
        },
        {
            "id": f"algora-bounty-{seed}-004",
            "platform": "algora",
            "source_url": "https://algora.io/bounties/abc123",
            "title": "Implement OAuth2 PKCE flow for CLI auth",
            "description": (
                "Replace the current device-code flow with PKCE for "
                "better security on public clients."
            ),
            "reward_usd": 500.0,
            "labels": ["feature", "hard", "security"],
            "language": "Rust",
            "repo": "example/auth-cli",
            "posted_at": "2026-02-20T09:00:00Z",
            "expires_at": "2026-04-01T00:00:00Z",
        },
        {
            "id": f"replit-bounty-{seed}-005",
            "platform": "replit",
            "source_url": "https://replit.com/bounties/xyz789",
            "title": "Build a CSV-to-JSON converter API endpoint",
            "description": (
                "Simple REST endpoint that accepts CSV upload and returns JSON. "
                "Include validation and error handling."
            ),
            "reward_usd": 25.0,
            "labels": ["small", "enhancement"],
            "language": "Python",
            "repo": None,
            "posted_at": "2026-03-01T12:00:00Z",
            "expires_at": "2026-03-10T00:00:00Z",
        },
        {
            "id": f"gh-bounty-{seed}-006",
            "platform": "github",
            "source_url": "https://github.com/example/infra/issues/301",
            "title": "Migrate CI pipeline from CircleCI to GitHub Actions",
            "description": (
                "Full migration of 12 CI workflows including matrix builds, "
                "caching, and deployment steps."
            ),
            "reward_usd": 150.0,
            "labels": ["major", "complex", "architecture"],
            "language": "YAML",
            "repo": "example/infra",
            "posted_at": "2026-02-15T16:00:00Z",
            "expires_at": "2026-03-30T00:00:00Z",
        },
    ]

    # Filter by platform and minimum reward
    requested_platforms = {p.lower().strip() for p in target_platforms}
    return [
        bounty
        for bounty in mock_pool
        if bounty["platform"] in requested_platforms
        and bounty["reward_usd"] >= min_reward_usd
    ]


# -- BountyHunterExecutor ------------------------------------------------


class BountyHunterExecutor(Executor):
    """
    Scan configured platforms for paid bounties and evaluate each against
    the organism's BountyPolicy before returning viable candidates to Nova.

    This is a read-only, foraging executor -- it finds and filters work
    opportunities but does not accept or commit to any of them.  Nova
    receives the structured output and decides whether to pursue.

    Required params:
      target_platforms (list[str]): Platforms to scan.
                                    Supported: "github", "algora", "replit", "gitcoin".

    Optional params:
      min_reward_usd (float | str): Minimum bounty reward to consider. Default 5.0.
      max_results (int): Maximum number of passing bounties to return. Default 10.
      include_rejected (bool): If True, include rejected bounties in output
                               (marked with rejection_reasons). Default False.

    Returns ExecutionResult with:
      data:
        bounties          -- list of evaluated bounty dicts (see below)
        total_scanned     -- number of raw bounties fetched
        total_passed      -- number that passed BountyPolicy
        total_rejected    -- number that failed BountyPolicy
        policy            -- the policy parameters used
        scan_id           -- unique scan identifier
      side_effects:
        -- Human-readable summary of the scan
      new_observations:
        -- Feed top candidates back to Atune as a Percept

    Each bounty dict in data.bounties:
      id, platform, source_url, title, description,
      reward_usd, language, repo, labels,
      posted_at, expires_at,
      difficulty          -- classified difficulty tier
      estimated_cost_usd  -- API token cost estimate
      roi                 -- reward / cost ratio
      cost_pct            -- cost as fraction of reward
      policy_passes       -- bool
      rejection_reasons   -- list[str] (empty if passes)
    """

    action_type = "hunt_bounties"
    description = (
        "Scan platforms for paid bounties, evaluate ROI against BountyPolicy, "
        "and return viable candidates for Nova to accept (Phase 16b -- Foraging)"
    )

    required_autonomy = 2       # PARTNER -- reads external platforms, no funds moved
    reversible = False          # Read-only scan, nothing to reverse
    max_duration_ms = 30_000    # External API calls can be slow
    rate_limit = RateLimit.per_hour(6)  # Don't hammer external APIs

    def __init__(self, synapse: Any = None) -> None:
        self._synapse = synapse
        self._logger = logger.bind(executor="axon.bounty_hunter")

    # -- Validation -------------------------------------------------------

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        """Validate scan parameters -- no I/O."""
        # target_platforms: required, non-empty list of supported platforms
        platforms_raw = params.get("target_platforms")
        if not platforms_raw:
            return ValidationResult.fail(
                "target_platforms is required (list of platform names)",
                target_platforms="missing",
            )
        if not isinstance(platforms_raw, list):
            return ValidationResult.fail(
                "target_platforms must be a list",
                target_platforms="not a list",
            )
        if len(platforms_raw) == 0:
            return ValidationResult.fail(
                "target_platforms must contain at least one platform",
                target_platforms="empty list",
            )
        unsupported = {
            p.lower().strip() for p in platforms_raw
        } - _SUPPORTED_PLATFORMS
        if unsupported:
            return ValidationResult.fail(
                f"Unsupported platforms: {sorted(unsupported)}. "
                f"Supported: {sorted(_SUPPORTED_PLATFORMS)}",
                target_platforms="unsupported platform(s)",
            )

        # min_reward_usd: optional, must be >= 0 if provided
        min_reward_raw = params.get("min_reward_usd", "5.0")
        try:
            min_reward = float(Decimal(str(min_reward_raw)))
        except Exception:
            return ValidationResult.fail(
                "min_reward_usd must be a valid number",
                min_reward_usd="not a number",
            )
        if min_reward < 0:
            return ValidationResult.fail(
                "min_reward_usd must be >= 0",
                min_reward_usd="negative value",
            )

        # max_results: optional, must be positive int if provided
        max_results = params.get("max_results", 10)
        if not isinstance(max_results, int) or max_results < 1:
            return ValidationResult.fail(
                "max_results must be a positive integer",
                max_results="invalid value",
            )

        return ValidationResult.ok()

    # -- Execution --------------------------------------------------------

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Scan, filter, and return bounty candidates. Never raises."""
        target_platforms: list[str] = [
            p.lower().strip() for p in params["target_platforms"]
        ]
        min_reward_usd = float(params.get("min_reward_usd", 5.0))
        max_results = int(params.get("max_results", 10))
        include_rejected = bool(params.get("include_rejected", False))

        scan_id = f"scan-{uuid.uuid4().hex[:12]}"

        self._logger.info(
            "bounty_hunt_started",
            scan_id=scan_id,
            target_platforms=target_platforms,
            min_reward_usd=min_reward_usd,
            max_results=max_results,
            execution_id=context.execution_id,
        )

        # -- Fetch bounties (mock for Phase 16b) -------------------------
        try:
            raw_bounties = _generate_mock_bounties(
                target_platforms=target_platforms,
                min_reward_usd=min_reward_usd,
            )
        except Exception as exc:
            self._logger.error(
                "bounty_fetch_failed",
                scan_id=scan_id,
                error=str(exc),
            )
            return ExecutionResult(
                success=False,
                error=f"Failed to fetch bounties: {exc}",
            )

        # -- Evaluate each bounty against BountyPolicy -------------------
        evaluated: list[dict[str, Any]] = []
        passed_count = 0
        rejected_count = 0

        for bounty in raw_bounties:
            labels = list(bounty.get("labels", []))
            difficulty = _classify_difficulty(labels)
            estimated_cost = _COST_ESTIMATES_USD[difficulty]
            reward = float(bounty["reward_usd"])

            policy_result = BountyPolicy.evaluate(
                reward_usd=reward,
                estimated_cost_usd=estimated_cost,
            )

            enriched: dict[str, Any] = {
                "id": bounty["id"],
                "platform": bounty["platform"],
                "source_url": bounty["source_url"],
                "title": bounty["title"],
                "description": bounty["description"],
                "reward_usd": reward,
                "language": bounty.get("language", "unknown"),
                "repo": bounty.get("repo"),
                "labels": labels,
                "posted_at": bounty.get("posted_at"),
                "expires_at": bounty.get("expires_at"),
                "difficulty": difficulty.value,
                "estimated_cost_usd": estimated_cost,
                "roi": policy_result["roi"],
                "cost_pct": policy_result["cost_pct"],
                "policy_passes": policy_result["passes"],
                "rejection_reasons": policy_result["rejection_reasons"],
            }

            if policy_result["passes"]:
                passed_count += 1
                evaluated.append(enriched)

                self._logger.info(
                    "bounty_passed_policy",
                    scan_id=scan_id,
                    bounty_id=bounty["id"],
                    title=bounty["title"],
                    reward_usd=reward,
                    roi=policy_result["roi"],
                    cost_pct=policy_result["cost_pct"],
                    difficulty=difficulty.value,
                )
            else:
                rejected_count += 1
                if include_rejected:
                    evaluated.append(enriched)

                self._logger.debug(
                    "bounty_rejected_by_policy",
                    scan_id=scan_id,
                    bounty_id=bounty["id"],
                    title=bounty["title"],
                    reward_usd=reward,
                    rejection_reasons=policy_result["rejection_reasons"],
                )

        # -- Sort by ROI descending, cap at max_results ------------------
        # Passed bounties first (sorted by ROI), then rejected if included
        evaluated.sort(
            key=lambda b: (b["policy_passes"], b["roi"]),
            reverse=True,
        )

        # Cap the output
        capped = evaluated[:max_results]

        # -- Build observation for Atune ---------------------------------
        top_candidates = [b for b in capped if b["policy_passes"]][:3]
        if top_candidates:
            observation_lines = [
                f"Bounty scan complete: {passed_count} viable / "
                f"{len(raw_bounties)} scanned. Top candidates:"
            ]
            for b in top_candidates:
                observation_lines.append(
                    f"  - [{b['platform']}] \"{b['title']}\" "
                    f"${b['reward_usd']:.0f} reward, "
                    f"{b['roi']:.1f}x ROI, "
                    f"{b['difficulty']} difficulty"
                )
            observation = "\n".join(observation_lines)
        else:
            observation = (
                f"Bounty scan complete: 0 viable bounties found across "
                f"{', '.join(target_platforms)} "
                f"(scanned {len(raw_bounties)}, all rejected by BountyPolicy)."
            )

        # -- Side effect summary -----------------------------------------
        side_effect = (
            f"Bounty scan [{scan_id}]: scanned {len(raw_bounties)} bounties "
            f"across {', '.join(target_platforms)}. "
            f"{passed_count} passed BountyPolicy "
            f"(min ROI {BountyPolicy.MIN_ROI_THRESHOLD}x, "
            f"max cost {BountyPolicy.MAX_ESTIMATED_COST_PCT:.0%}), "
            f"{rejected_count} rejected."
        )

        self._logger.info(
            "bounty_hunt_complete",
            scan_id=scan_id,
            total_scanned=len(raw_bounties),
            total_passed=passed_count,
            total_rejected=rejected_count,
            execution_id=context.execution_id,
        )

        return ExecutionResult(
            success=True,
            data={
                "bounties": capped,
                "total_scanned": len(raw_bounties),
                "total_passed": passed_count,
                "total_rejected": rejected_count,
                "policy": {
                    "min_roi_threshold": BountyPolicy.MIN_ROI_THRESHOLD,
                    "max_estimated_cost_pct": BountyPolicy.MAX_ESTIMATED_COST_PCT,
                },
                "scan_id": scan_id,
                "scanned_at": datetime.now(tz=timezone.utc).isoformat(),
            },
            side_effects=[side_effect],
            new_observations=[observation],
        )
