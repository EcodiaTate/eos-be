"""
EcodiaOS — Governance Approval Script

Approves a Simula proposal that is paused at AWAITING_GOVERNANCE.

Governance is enforced for GOVERNANCE_REQUIRED categories (ADD_SYSTEM_CAPABILITY,
MODIFY_CONTRACT, MODIFY_CYCLE_TIMING, CHANGE_CONSOLIDATION). When a proposal in
one of these categories completes simulation, the SimulaService pauses the
pipeline, writes a GovernanceProposal node to Neo4j, and returns a
governance_record_id. This script sends the approval signal so the service
resumes from Step 4 (Apply) and writes the evolved code to disk.

The approval HTTP endpoint lives in the running FastAPI process (not the worker),
because the proposal object is held in SimulaService._active_proposals memory.

Usage — interactive (paste IDs from the AWAITING_GOVERNANCE result):
    cd backend
    python approve_mutation.py --proposal-id <ULID> --governance-id <gov_id>

Usage — non-interactive (hardcode for a single test run):
    Edit PROPOSAL_ID and GOVERNANCE_ID below, then:
    cd backend
    python approve_mutation.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import httpx
from dotenv import load_dotenv

# load_dotenv MUST run before any os.getenv() call
load_dotenv()

# ── Hardcoded fallbacks (edit for a one-shot test run) ────────────────────────
# These are used only when --proposal-id / --governance-id are not supplied.
PROPOSAL_ID: str = ""   # e.g. "01JPPQR2S3T4U5V6W7X8Y9Z0AB"
GOVERNANCE_ID: str = "" # e.g. "gov_01jppqr2s3t4u5v6w7x8y9z0"

# ── API base URL ───────────────────────────────────────────────────────────────
# Override via ECODIAOS_API_URL env var if the service runs on a different host/port.
API_BASE = os.getenv("ECODIAOS_API_URL", "http://localhost:8000")

# ── Timeout for the approve HTTP call ─────────────────────────────────────────
# _apply_change() runs the code agent, lints, health-checks — allow up to 10 min.
APPLY_TIMEOUT_S = 600


# ── Argument parsing ──────────────────────────────────────────────────────────


def _parse_args() -> tuple[str, str]:
    parser = argparse.ArgumentParser(
        description="Approve a Simula proposal that is paused at AWAITING_GOVERNANCE."
    )
    parser.add_argument(
        "--proposal-id",
        metavar="ULID",
        default=PROPOSAL_ID,
        help="The proposal ULID from the AWAITING_GOVERNANCE result.",
    )
    parser.add_argument(
        "--governance-id",
        metavar="GOV_ID",
        default=GOVERNANCE_ID,
        help='The governance_record_id from the result (e.g. "gov_...").',
    )
    args = parser.parse_args()

    proposal_id = args.proposal_id.strip()
    governance_id = args.governance_id.strip()

    if not proposal_id:
        print(
            "[!] --proposal-id is required.\n"
            "    Run with --help for usage, or set PROPOSAL_ID in the script."
        )
        sys.exit(1)
    if not governance_id:
        print(
            "[!] --governance-id is required.\n"
            "    Run with --help for usage, or set GOVERNANCE_ID in the script."
        )
        sys.exit(1)

    return proposal_id, governance_id


# ── Main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    proposal_id, governance_id = _parse_args()

    url = f"{API_BASE}/api/v1/simula/proposals/{proposal_id}/approve"
    payload = {"governance_record_id": governance_id}

    print()
    print(f"[*] Proposal ID   : {proposal_id}")
    print(f"[*] Governance ID : {governance_id}")
    print(f"[*] Endpoint      : POST {url}")
    print(f"[*] Timeout       : {APPLY_TIMEOUT_S}s (code agent may take a while)")
    print()

    async with httpx.AsyncClient(timeout=APPLY_TIMEOUT_S) as client:
        try:
            response = await client.post(url, json=payload)
        except httpx.ConnectError:
            print(
                f"[!] Could not connect to {API_BASE}\n"
                "    Is the EcodiaOS FastAPI service running?\n"
                "    Override with: ECODIAOS_API_URL=http://host:port python approve_mutation.py ..."
            )
            sys.exit(1)
        except httpx.TimeoutException:
            print(
                f"[!] Request timed out after {APPLY_TIMEOUT_S}s.\n"
                "    The code agent may still be working — check the service logs."
            )
            sys.exit(1)

    # ── Parse response ────────────────────────────────────────────────────
    try:
        body = response.json()
    except Exception:
        print(f"[!] Non-JSON response (HTTP {response.status_code}):\n{response.text}")
        sys.exit(1)

    if response.status_code != 200:
        print(f"[!] HTTP {response.status_code}: {json.dumps(body, indent=2)}")
        sys.exit(1)

    if "error" in body:
        print(f"[!] API error: {body['error']}")
        sys.exit(1)

    status = body.get("status", "unknown")
    reason = body.get("reason", "")
    files_changed = body.get("files_changed", [])
    version = body.get("version")

    print("=" * 60)
    print(f"  STATUS        : {status.upper()}")
    print(f"  VERSION       : {version}")
    print(f"  REASON        : {reason or 'n/a'}")
    print(f"  FILES CHANGED : {files_changed or 'n/a'}")
    print("=" * 60)

    if status == "applied":
        print(
            "\n[OK] Proposal APPLIED — EcodiaOS evolved.\n"
            "     NeuroplasticityBus will hot-reload the changed files on the\n"
            "     next eos:events:code_evolved publication."
        )
    elif status == "rejected":
        print(f"\n[FAIL] Proposal REJECTED: {reason}")
        sys.exit(1)
    elif status == "rolled_back":
        print(f"\n[WARN] Proposal ROLLED BACK: {reason}")
        sys.exit(1)
    else:
        print(f"\n[?] Unexpected status: {status!r}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
