"""Stream diagnostic — run once to inspect Redis state and optionally reset."""
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

import redis.asyncio as aioredis


def _redis_url() -> str:
    url = os.getenv("ECODIAOS_REDIS__URL", "").strip()
    pw = os.getenv("ECODIAOS_REDIS_PASSWORD", "").strip()
    if pw and "://" in url and "@" not in url:
        scheme, rest = url.split("://", 1)
        url = f"{scheme}://:{pw}@{rest}"
    return url


async def main(reset: bool = False, trim: bool = False) -> None:
    client = await aioredis.from_url(_redis_url(), decode_responses=True)
    await client.ping()
    print("Connected OK\n")

    for stream in ("eos:simula:tasks", "eos:simula:results"):
        n = await client.xlen(stream)
        print(f"{stream}: {n} messages")
        msgs = await client.xrevrange(stream, count=5)
        for mid, fields in msgs:
            key = "proposal_id" if "proposal_id" in fields else list(fields.keys())[0]
            print(f"  {mid}  {key}={fields.get(key, '?')[:60]}")

    print("\nConsumer groups on eos:simula:tasks:")
    try:
        for g in await client.xinfo_groups("eos:simula:tasks"):
            print(
                f"  name={g['name']}"
                f"  pending={g['pending']}"
                f"  consumers={g['consumers']}"
                f"  last-delivered={g['last-delivered-id']}"
            )
    except Exception as e:
        print(f"  (no groups or error: {e})")

    if reset or trim:
        print("\nResetting consumer group...")
        try:
            await client.xgroup_destroy("eos:simula:tasks", "simula-workers")
            print("  Deleted simula-workers — worker will recreate it on next start.")
        except Exception as e:
            print(f"  xgroup_destroy: {e}")

    if trim:
        print("\nTrimming streams to 0 messages (clean slate)...")
        for stream in ("eos:simula:tasks", "eos:simula:results"):
            try:
                await client.xtrim(stream, maxlen=0, approximate=False)
                print(f"  Trimmed {stream}")
            except Exception as e:
                print(f"  xtrim {stream}: {e}")
        print("  Done — stream is empty. Start the worker, then run test_hello_world.py.")

    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main(
        reset="--reset" in sys.argv,
        trim="--trim" in sys.argv,
    ))
