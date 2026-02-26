"""
EcodiaOS — Hunter Engine smoke-test runner.

Runs the full Clone → Map → Prove pipeline against a single authorized
target repository and prints a structured summary of what was found.

Usage:
    ANTHROPIC_API_KEY=sk-... python run_hunter.py [--target <github_url>]

The target must be listed in AUTHORIZED_TARGETS below (or passed via --target).
The script only reads from the repository — no writes, no network calls to the
target, no side effects beyond structured console output.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# ── Authorized honeypot targets (edit to add your own) ────────────────────────

AUTHORIZED_TARGETS: list[str] = [
    # Point this at any public repo you own or have written authorization to test.
    # Example: "https://github.com/yourusername/nextjs-honeypot"
]


async def main(target_url: str) -> int:
    # Deferred imports — avoids loading the full EOS stack until we're sure
    # the venv is activated and the target is authorized.
    from ecodiaos.clients.llm import create_llm_provider
    from ecodiaos.config import LLMConfig
    from ecodiaos.systems.simula.hunter.prover import VulnerabilityProver
    from ecodiaos.systems.simula.hunter.service import HunterService
    from ecodiaos.systems.simula.hunter.types import HunterConfig
    from ecodiaos.systems.simula.verification.z3_bridge import Z3Bridge

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[ERROR] ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 1

    print(f"[*] Booting EcodiaOS Simula Hunter Engine...")
    print(f"[*] Target: {target_url}")

    # ── Core components ────────────────────────────────────────────────────────

    llm_config = LLMConfig(
        provider="anthropic",
        api_key=api_key,
        model="claude-sonnet-4-20250514",
    )
    llm = create_llm_provider(llm_config)

    z3_bridge = Z3Bridge(check_timeout_ms=10_000)

    prover = VulnerabilityProver(
        z3_bridge=z3_bridge,
        llm=llm,
    )

    config = HunterConfig(
        authorized_targets=[target_url],
        max_workers=2,
        sandbox_timeout_seconds=60,
        log_vulnerability_analytics=False,
        clone_depth=1,
    )

    hunter = HunterService(
        prover=prover,
        config=config,
        eos_root=Path(__file__).parent,
    )

    # ── Pull the trigger ───────────────────────────────────────────────────────

    print("[*] Commencing hunt...")
    result = await hunter.hunt_external_repo(
        github_url=target_url,
        generate_pocs=True,
        generate_patches=False,
    )

    # ── Output ─────────────────────────────────────────────────────────────────

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Hunt complete in {result.total_duration_ms}ms")
    print(f"Surfaces mapped : {result.surfaces_mapped}")
    print(f"Vulnerabilities : {len(result.vulnerabilities_found)}")

    if not result.vulnerabilities_found:
        print("\n[OK] No vulnerabilities proved — target appears secure.")
        await llm.close()
        return 0

    for vuln in result.vulnerabilities_found:
        print(f"\n{'─' * 60}")
        print(f"[FINDING] {vuln.vulnerability_class.value.upper()}")
        print(f"Severity  : {vuln.severity.value}")
        print(f"Surface   : {vuln.attack_surface.file_path}:{vuln.attack_surface.line_number}")
        print(f"Entry     : {vuln.attack_surface.entry_point}")
        print(f"Goal      : {vuln.attack_goal}")
        print(f"\nZ3 counterexample:\n{vuln.z3_counterexample}")
        if vuln.proof_of_concept_code:
            print(f"\nReproduction script (Security Unit Test):\n{vuln.proof_of_concept_code}")

    print(f"\n{sep}")
    print(
        f"[SUMMARY] {len(result.vulnerabilities_found)} finding(s)  "
        f"critical={result.critical_count}  high={result.high_count}"
    )

    await llm.close()
    return 0 if not result.vulnerabilities_found else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hunter engine smoke-test runner")
    parser.add_argument(
        "--target",
        default=AUTHORIZED_TARGETS[0] if AUTHORIZED_TARGETS else "",
        help="GitHub HTTPS URL of the authorized target repository",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.target:
        print(
            "[ERROR] No target specified.\n"
            "  Either add a URL to AUTHORIZED_TARGETS in this script\n"
            "  or pass --target https://github.com/yourusername/nextjs-honeypot",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(asyncio.run(main(args.target)))
