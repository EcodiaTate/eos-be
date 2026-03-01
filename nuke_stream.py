"""
EcodiaOS — Nuclear Stream Reset

Connects to Upstash (or local) Redis, destroys ALL Simula and Inspector
streams, consumer groups, and pending entries.  Use this to get a clean
slate after a migration or when ghost messages haunt the PEL.

Usage:
    cd backend
    python nuke_stream.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

# Every stream/group pair the system uses.
STREAMS = [
    "eos:simula:tasks",
    "eos:simula:results",
    "eos:inspector:tasks",
    "eos:inspector:results",
]

CONSUMER_GROUPS = {
    "eos:simula:tasks": ["simula-workers"],
    "eos:simula:results": ["simula-proxy"],
    "eos:inspector:tasks": ["inspector-workers"],
    "eos:inspector:results": ["inspector-proxy"],
}


def _build_url() -> str:
    """Build authenticated Redis URL from env vars."""
    url = os.getenv("ECODIAOS_REDIS__URL", "").strip()
    if not url:
        url = "redis://localhost:6379/0"
        print("[!] ECODIAOS_REDIS__URL not set — falling back to localhost")

    pw = os.getenv("ECODIAOS_REDIS_PASSWORD", "").strip()
    if pw and "://" in url and "@" not in url:
        scheme, rest = url.split("://", 1)
        url = f"{scheme}://:{pw}@{rest}"

    return url


async def nuke() -> None:
    url = _build_url()
    masked = url.split("@")[-1] if "@" in url else url
    print(f"[*] Connecting to Redis: {masked}")

    r = await aioredis.from_url(url, decode_responses=True)

    try:
        pong = await r.ping()
        if not pong:
            print("[!] Redis PING failed — check credentials")
            sys.exit(1)
        print("[+] AUTH OK")
    except Exception as exc:
        print(f"[!] Connection failed: {exc}")
        sys.exit(1)

    # ── 1. Destroy consumer groups (must happen before key deletion) ──
    for stream, groups in CONSUMER_GROUPS.items():
        for group in groups:
            try:
                await r.xgroup_destroy(stream, group)
                print(f"[+] Destroyed group {group!r} on {stream!r}")
            except Exception as exc:
                if "NOGROUP" in str(exc) or "no such key" in str(exc).lower():
                    print(f"[~] Group {group!r} on {stream!r} — already gone")
                else:
                    print(f"[!] Error destroying {group!r} on {stream!r}: {exc}")

    # ── 2. Delete all stream keys ────────────────────────────────────
    deleted = await r.delete(*STREAMS)
    print(f"[+] Deleted {deleted}/{len(STREAMS)} stream keys")

    # ── 3. Verify nothing is left ────────────────────────────────────
    for stream in STREAMS:
        exists = await r.exists(stream)
        if exists:
            print(f"[!] WARNING: {stream!r} still exists after delete!")
        else:
            print(f"[OK] {stream!r} confirmed gone")

    await r.aclose()
    print()
    print("[DONE] All Simula & Inspector streams nuked. Clean slate.")


if __name__ == "__main__":
    asyncio.run(nuke())
