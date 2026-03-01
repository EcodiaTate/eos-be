"""
EcodiaOS — Simula Worker

Standalone process that consumes both evolution proposals and inspector
hunt requests from Redis Streams, runs them through the real pipelines,
and publishes results back.

Streams consumed:
  ``eos:simula:tasks``    → EvolutionProposal  → SimulaService.process_proposal()
  ``eos:inspector:tasks`` → Inspector hunt      → InspectorService.hunt_external_repo()

Designed to run on a dedicated GCE/GKE node with its own connection
pools so the main Cloud Run monolith's 150ms Synapse clock is never
blocked by heavy simulation or Z3 work.

Usage:
    python -m ecodiaos.simula_worker
    python -m ecodiaos.simula_worker --config /etc/ecodiaos/config.yaml

Graceful shutdown:
    Handles SIGINT and SIGTERM. In-flight tasks complete before exit;
    no orphaned tasks are left in the stream.
"""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Any

import orjson
import structlog
from dotenv import load_dotenv
from redis.asyncio import Redis

# Explicitly resolve .env relative to the backend/ directory (parent of this
# package) so it works regardless of cwd — critical on WSL where case-
# insensitive path resolution can cause bare load_dotenv() to miss the file.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_BACKEND_DIR / ".env", override=True)

from ecodiaos.clients.llm import create_llm_provider
from ecodiaos.clients.neo4j import Neo4jClient
from ecodiaos.clients.redis import RedisClient
from ecodiaos.clients.timescaledb import TimescaleDBClient
from ecodiaos.config import load_config
from ecodiaos.systems.simula.evolution_types import (
    EvolutionProposal,
    ProposalResult,
    ProposalStatus,
)
from ecodiaos.systems.simula.service import SimulaService

logger = structlog.get_logger()

# Must match the keys in proxy.py exactly.
TASKS_STREAM = "eos:simula:tasks"
RESULTS_STREAM = "eos:simula:results"
GOVERNANCE_STREAM = "eos:simula:governance"  # Proxy → Worker approval signals

INSPECTOR_TASKS_STREAM = "eos:inspector:tasks"
INSPECTOR_RESULTS_STREAM = "eos:inspector:results"

# Consumer group names for worker replicas.
TASKS_GROUP = "simula-workers"
GOVERNANCE_GROUP = "simula-governance"
INSPECTOR_TASKS_GROUP = "inspector-workers"


async def run_worker(config_path: str | None = None) -> None:
    """
    Main worker loop.

    1. Load config and build its own connection pools.
    2. Instantiate and initialise the real SimulaService.
    3. Consume from eos:simula:tasks via XREADGROUP.
    4. Run process_proposal and XADD the result to eos:simula:results.
    5. XACK the task message.
    6. On shutdown signal, finish the current proposal then exit.
    """
    config = load_config(config_path)
    log = logger.bind(worker="simula")

    # ── Build connection pools (independent of the monolith) ──────────
    neo4j_client = Neo4jClient(config.neo4j)
    await neo4j_client.connect()
    log.info("neo4j_connected")

    tsdb_client = TimescaleDBClient(config.timescaledb)
    await tsdb_client.connect()
    log.info("tsdb_connected")

    redis_client = RedisClient(config.redis)
    await redis_client.connect()
    log.info("redis_connected")

    llm_client = create_llm_provider(config.llm)
    log.info("llm_provider_created")

    # ── Build SimulaService ───────────────────────────────────────────
    simula = SimulaService(
        config=config.simula,
        llm=llm_client,
        neo4j=neo4j_client,
        memory=None,  # Worker doesn't need MemoryService for the pipeline
        codebase_root=Path(config.simula.codebase_root).resolve(),
        instance_name=config.instance_id,
        tsdb=tsdb_client,
        redis=redis_client,
    )
    await simula.initialize()
    log.info("simula_initialized")

    # ── Build InspectorService for hunt offloading ────────────────────
    inspector = _build_inspector(config, llm_client)
    log.info("inspector_initialized")

    # ── Ensure consumer groups exist ──────────────────────────────────
    raw: Redis = redis_client.client
    for stream, group in [
        (TASKS_STREAM, TASKS_GROUP),
        (GOVERNANCE_STREAM, GOVERNANCE_GROUP),
        (INSPECTOR_TASKS_STREAM, INSPECTOR_TASKS_GROUP),
    ]:
        try:
            await raw.xgroup_create(stream, group, id="0", mkstream=True)
            log.info("xgroup_created", stream=stream, group=group)
        except Exception as exc:
            err_msg = str(exc)
            if "BUSYGROUP" in err_msg:
                log.info("xgroup_already_exists", stream=stream, group=group)
                # Do NOT reset the cursor — resetting to "0" causes every
                # previously-ACKed message to be re-delivered as new,
                # which is the "ghost memories" bug.  The group's cursor
                # is already at the right position from the last run.
            else:
                log.error("xgroup_create_failed", stream=stream, group=group, error=err_msg)

    # ── Evict ghost consumers and drain stale PEL ──────────────────
    # Previous worker runs leave behind dead consumers (e.g. worker-12345)
    # that still compete for new messages via the consumer group.  With 15
    # ghost consumers, new messages are round-robined to dead endpoints
    # and never processed.  Delete every consumer except the one we're
    # about to register.
    shutdown_event = asyncio.Event()
    pid = os.getpid()
    consumer_name = f"worker-{pid}"

    for stream, group in [
        (TASKS_STREAM, TASKS_GROUP),
        (GOVERNANCE_STREAM, GOVERNANCE_GROUP),
        (INSPECTOR_TASKS_STREAM, INSPECTOR_TASKS_GROUP),
    ]:
        try:
            consumers = await raw.xinfo_consumers(stream, group)
            for c in consumers:
                cname = c.get("name", "")
                if cname and cname != consumer_name:
                    # ACK any pending messages owned by this ghost consumer
                    # before deleting it, so they re-enter the undelivered pool.
                    pending_count = c.get("pending", 0)
                    if pending_count > 0:
                        stale = await raw.xpending_range(
                            stream, group, min="-", max="+", count=pending_count,
                            consumername=cname,
                        )
                        for entry in stale:
                            msg_id = entry.get("message_id", "")
                            if msg_id:
                                await raw.xack(stream, group, msg_id)
                        log.info(
                            "ghost_consumer_pending_acked",
                            stream=stream,
                            consumer=cname,
                            acked=len(stale),
                        )
                    await raw.xgroup_delconsumer(stream, group, cname)
                    log.info("ghost_consumer_evicted", stream=stream, consumer=cname)
            remaining = await raw.xinfo_consumers(stream, group)
            log.info(
                "consumer_cleanup_done",
                stream=stream,
                remaining_consumers=len(remaining),
            )
        except Exception as exc:
            log.warning("consumer_cleanup_error", stream=stream, error=str(exc))

    def _signal_handler() -> None:
        log.info("shutdown_signal_received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler; fall back to
            # signal.signal which is less clean but functional.
            signal.signal(sig, lambda s, f: _signal_handler())

    # ── Main consume loop ─────────────────────────────────────────────
    # Read from both streams in a single XREADGROUP call so the worker
    # handles evolution proposals and inspector hunts interleaved.
    log.info("worker_started", consumer=consumer_name)

    _diag_counter = 0
    while not shutdown_event.is_set():
        # ── Simula proposals ──────────────────────────────────────
        try:
            entries = await raw.xreadgroup(
                groupname=TASKS_GROUP,
                consumername=consumer_name,
                streams={TASKS_STREAM: ">"},
                count=1,
                block=500,
            )
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log.error("xreadgroup_error", error=str(exc), error_type=type(exc).__name__)
            await asyncio.sleep(2.0)
            continue

        # Periodic diagnostics — log stream state every ~5s (10 iterations × 500ms block)
        _diag_counter += 1
        if _diag_counter % 10 == 1:
            try:
                slen = await raw.xlen(TASKS_STREAM)
                groups = await raw.xinfo_groups(TASKS_STREAM)
                group_info = [
                    {
                        "name": g.get("name"),
                        "pending": g.get("pending"),
                        "consumers": g.get("consumers"),
                        "last_delivered": g.get("last-delivered-id"),
                    }
                    for g in groups
                ]
                log.info(
                    "stream_diag",
                    stream=TASKS_STREAM,
                    length=slen,
                    groups=group_info,
                    xreadgroup_returned=repr(entries),
                )
            except Exception as diag_exc:
                log.warning("stream_diag_error", error=str(diag_exc))

        if entries:
            for _stream_name, messages in entries:
                for msg_id, fields in messages:
                    log.info(
                        "xreadgroup_received",
                        msg_id=msg_id,
                        field_keys=list(fields.keys()),
                    )
                    await _process_one(
                        fields=fields,
                        msg_id=msg_id,
                        raw=raw,
                        simula=simula,
                        log=log,
                    )

        # ── Governance approvals ──────────────────────────────────
        try:
            g_entries = await raw.xreadgroup(
                groupname=GOVERNANCE_GROUP,
                consumername=consumer_name,
                streams={GOVERNANCE_STREAM: ">"},
                count=10,
                block=0,  # non-blocking — don't stall the proposal loop
            )
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log.error("governance_xreadgroup_error", error=str(exc))
            g_entries = []

        if g_entries:
            for _stream_name, messages in g_entries:
                for msg_id, fields in messages:
                    await _process_governance_approval(
                        fields=fields,
                        msg_id=msg_id,
                        raw=raw,
                        simula=simula,
                        log=log,
                    )

        # ── Inspector hunts ───────────────────────────────────────
        try:
            i_entries = await raw.xreadgroup(
                groupname=INSPECTOR_TASKS_GROUP,
                consumername=consumer_name,
                streams={INSPECTOR_TASKS_STREAM: ">"},
                count=1,
                block=500,
            )
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log.error("inspector_xreadgroup_error", error=str(exc))
            await asyncio.sleep(2.0)
            continue

        if i_entries:
            for _stream_name, messages in i_entries:
                for msg_id, fields in messages:
                    await _process_inspector_hunt(
                        fields=fields,
                        msg_id=msg_id,
                        raw=raw,
                        inspector=inspector,
                        log=log,
                    )

    # ── Graceful teardown ─────────────────────────────────────────────
    log.info("worker_draining")
    await simula.shutdown()
    await llm_client.close()
    await redis_client.close()
    await tsdb_client.close()
    await neo4j_client.close()
    log.info("worker_shutdown_complete")


async def _process_one(
    *,
    fields: dict[str, str],
    msg_id: str,
    raw: Redis,
    simula: SimulaService,
    log: Any,
) -> None:
    """Process a single task message end-to-end."""
    proposal_id = fields.get("proposal_id", "")
    payload_json = fields.get("payload", "")
    plog = log.bind(proposal_id=proposal_id, msg_id=msg_id)

    try:
        data = orjson.loads(payload_json)
        proposal = EvolutionProposal.model_validate(data)
    except Exception as exc:
        plog.error("payload_parse_error", error=str(exc))
        # ACK the bad message so it doesn't block the stream, and push
        # a rejection result so the proxy doesn't hang.
        result = ProposalResult(
            status=ProposalStatus.REJECTED,
            reason=f"Worker failed to parse proposal: {exc}",
        )
        await _publish_result(raw, proposal_id, result, plog)
        await raw.xack(TASKS_STREAM, TASKS_GROUP, msg_id)
        return

    plog.info("processing_proposal", category=proposal.category.value)

    try:
        result = await simula.process_proposal(proposal)
    except Exception as exc:
        plog.error("pipeline_error", error=str(exc))
        result = ProposalResult(
            status=ProposalStatus.REJECTED,
            reason=f"Worker pipeline crashed: {exc}",
        )

    await _publish_result(raw, proposal_id, result, plog)
    await raw.xack(TASKS_STREAM, TASKS_GROUP, msg_id)
    plog.info("proposal_completed", status=result.status.value)

    # Publish neuroplasticity event so Axon can hot-reload evolved executors.
    if result.status == ProposalStatus.APPLIED and result.files_changed:
        await _publish_code_evolved(raw, result.files_changed, plog)


async def _recover_proposal_from_stream(
    proposal_id: str,
    governance_record_id: str,
    raw: Redis,
    simula: SimulaService,
    log: Any,
) -> bool:
    """
    Re-hydrate a proposal into SimulaService._active_proposals after a worker restart.

    Scans ``eos:simula:tasks`` for the original payload, reconstructs the
    ``EvolutionProposal``, and injects it at ``AWAITING_GOVERNANCE`` status so
    ``approve_governed_proposal()`` can resume normally.

    Returns True if recovery succeeded, False otherwise.
    """
    try:
        # XRANGE reads all messages; the stream is small enough in practice.
        all_msgs = await raw.xrange(TASKS_STREAM, "-", "+")
    except Exception as exc:
        log.error("governance_recovery_xrange_failed", error=str(exc))
        return False

    payload_json: str | None = None
    for _entry_id, entry_fields in all_msgs:
        if entry_fields.get("proposal_id") == proposal_id:
            payload_json = entry_fields.get("payload")
            break

    if not payload_json:
        log.error("governance_recovery_payload_not_found", proposal_id=proposal_id)
        return False

    try:
        data = orjson.loads(payload_json)
        proposal = EvolutionProposal.model_validate(data)
    except Exception as exc:
        log.error("governance_recovery_parse_error", error=str(exc))
        return False

    # Inject at AWAITING_GOVERNANCE so the service's guard passes.
    proposal.status = ProposalStatus.AWAITING_GOVERNANCE
    proposal.governance_record_id = governance_record_id
    simula._active_proposals[proposal_id] = proposal
    log.info("governance_recovery_success", proposal_id=proposal_id)
    return True


async def _process_governance_approval(
    *,
    fields: dict[str, str],
    msg_id: str,
    raw: Redis,
    simula: SimulaService,
    log: Any,
) -> None:
    """
    Resume a proposal that is paused at AWAITING_GOVERNANCE.

    The proxy pushes {proposal_id, governance_record_id} onto
    ``eos:simula:governance``.  We call
    ``SimulaService.approve_governed_proposal()`` (which runs _apply_change)
    and publish the result back to ``eos:simula:results`` so the proxy's
    existing listener resolves the parked future.

    If the proposal is not in memory (worker restarted), we recover it from
    the tasks stream before proceeding.
    """
    proposal_id = fields.get("proposal_id", "")
    governance_record_id = fields.get("governance_record_id", "")
    glog = log.bind(proposal_id=proposal_id, governance_record_id=governance_record_id)

    if not proposal_id or not governance_record_id:
        glog.error("governance_approval_missing_fields")
        await raw.xack(GOVERNANCE_STREAM, GOVERNANCE_GROUP, msg_id)
        return

    glog.info("governance_approval_received")

    # If the worker restarted, _active_proposals will be empty.  Recover the
    # proposal from the tasks stream before calling approve_governed_proposal.
    if proposal_id not in simula._active_proposals:
        glog.warning("governance_proposal_not_in_memory_recovering")
        recovered = await _recover_proposal_from_stream(
            proposal_id, governance_record_id, raw, simula, glog
        )
        if not recovered:
            result = ProposalResult(
                status=ProposalStatus.REJECTED,
                reason=(
                    f"Proposal {proposal_id} not found in memory or tasks stream. "
                    "It may have been submitted before this worker started and the "
                    "stream was trimmed."
                ),
            )
            await _publish_result(raw, proposal_id, result, glog)
            await raw.xack(GOVERNANCE_STREAM, GOVERNANCE_GROUP, msg_id)
            return

    try:
        result = await simula.approve_governed_proposal(proposal_id, governance_record_id)
    except Exception as exc:
        glog.error("governance_apply_error", error=str(exc))
        result = ProposalResult(
            status=ProposalStatus.REJECTED,
            reason=f"Worker failed to apply governed proposal: {exc}",
        )

    await _publish_result(raw, proposal_id, result, glog)
    await raw.xack(GOVERNANCE_STREAM, GOVERNANCE_GROUP, msg_id)
    glog.info("governance_approval_completed", status=result.status.value)

    if result.status == ProposalStatus.APPLIED and result.files_changed:
        await _publish_code_evolved(raw, result.files_changed, glog)


_CODE_EVOLVED_CHANNEL = "eos:events:code_evolved"


async def _publish_code_evolved(
    raw: Redis,
    files_changed: list[str],
    log: Any,
) -> None:
    """Publish a Pub/Sub event so Axon (and any other subscriber) can
    hot-reload executors that were written by this evolution cycle."""
    try:
        payload = orjson.dumps({"files_changed": files_changed}).decode()
        await raw.publish(_CODE_EVOLVED_CHANNEL, payload)
        log.info("code_evolved_published", files_changed=files_changed)
    except Exception as exc:
        log.error("code_evolved_publish_error", error=str(exc))


async def _publish_result(
    raw: Redis,
    proposal_id: str,
    result: ProposalResult,
    log: Any,
) -> None:
    """XADD the result to the results stream."""
    try:
        await raw.xadd(
            RESULTS_STREAM,
            {
                "proposal_id": proposal_id,
                "result": result.model_dump_json(),
            },
        )
    except Exception as exc:
        log.error("result_publish_error", proposal_id=proposal_id, error=str(exc))


# ═══════════════════════════════════════════════════════════════════════
# Inspector hunt processing
# ═══════════════════════════════════════════════════════════════════════


def _build_inspector(config: Any, llm_client: Any) -> Any:
    """Construct an InspectorService with the full agent swarm."""
    from ecodiaos.systems.simula.inspector.prover import VulnerabilityProver
    from ecodiaos.systems.simula.inspector.remediation import RepairAgent
    from ecodiaos.systems.simula.inspector.service import InspectorService
    from ecodiaos.systems.simula.inspector.slicer import SemanticSlicer
    from ecodiaos.systems.simula.inspector.temporal import ConcurrencyProver
    from ecodiaos.systems.simula.inspector.types import InspectorConfig
    from ecodiaos.systems.simula.inspector.verifier import AdversarialVerifier
    from ecodiaos.systems.simula.inspector.shield import AutonomousShield
    from ecodiaos.systems.simula.verification.z3_bridge import Z3Bridge

    z3_bridge = Z3Bridge(check_timeout_ms=10_000)
    prover = VulnerabilityProver(z3_bridge=z3_bridge, llm=llm_client)
    temporal_prover = ConcurrencyProver(llm=llm_client, z3_bridge=z3_bridge)
    slicer = SemanticSlicer(llm=llm_client)
    verifier = AdversarialVerifier(llm=llm_client)
    repair_agent = RepairAgent(llm=llm_client, prover=prover, max_retries=3)
    shield = AutonomousShield(llm=llm_client)

    inspector_config = InspectorConfig(
        authorized_targets=getattr(config.simula, "authorized_targets", []),
        max_workers=2,
        sandbox_timeout_seconds=60,
        log_vulnerability_analytics=False,
        clone_depth=1,
    )
    return InspectorService(
        prover=prover,
        config=inspector_config,
        eos_root=Path(config.simula.codebase_root).resolve(),
        temporal_prover=temporal_prover,
        slicer=slicer,
        verifier=verifier,
        repair_agent=repair_agent,
        shield=shield,
    )


async def _process_inspector_hunt(
    *,
    fields: dict[str, str],
    msg_id: str,
    raw: Redis,
    inspector: Any,
    log: Any,
) -> None:
    """Process a single inspector hunt task end-to-end."""
    hunt_id = fields.get("hunt_id", "")
    payload_json = fields.get("payload", "")
    hlog = log.bind(hunt_id=hunt_id, msg_id=msg_id)

    try:
        data = orjson.loads(payload_json)
    except Exception as exc:
        hlog.error("hunt_payload_parse_error", error=str(exc))
        await _publish_inspector_error(raw, hunt_id, str(exc), hlog)
        await raw.xack(INSPECTOR_TASKS_STREAM, INSPECTOR_TASKS_GROUP, msg_id)
        return

    github_url: str = data.get("github_url", "")
    attack_goals: list[str] | None = data.get("attack_goals")
    generate_pocs: bool = data.get("generate_pocs", True)
    generate_patches: bool = data.get("generate_patches", False)

    hlog.info("processing_hunt", target=github_url)

    # Dynamically authorize the target URL for this hunt — the proxy sends
    # ephemeral file:///tmp/phantom_workspace_* paths that can't be in the
    # static config.  This mirrors what run_inspector.py does: it constructs
    # a fresh InspectorConfig(authorized_targets=[target_url]).
    if github_url and github_url not in inspector._config.authorized_targets:
        inspector._config.authorized_targets.append(github_url)

    try:
        result = await inspector.hunt_external_repo(
            github_url,
            attack_goals=attack_goals,
            generate_pocs=generate_pocs,
            generate_patches=generate_patches,
        )
    except Exception as exc:
        hlog.error("hunt_pipeline_error", error=str(exc))
        await _publish_inspector_error(raw, hunt_id, str(exc), hlog)
        await raw.xack(INSPECTOR_TASKS_STREAM, INSPECTOR_TASKS_GROUP, msg_id)
        return

    # Publish the full InspectionResult back to the proxy.
    try:
        await raw.xadd(
            INSPECTOR_RESULTS_STREAM,
            {
                "hunt_id": hunt_id,
                "result": result.model_dump_json(),
            },
        )
    except Exception as exc:
        hlog.error("hunt_result_publish_error", error=str(exc))

    await raw.xack(INSPECTOR_TASKS_STREAM, INSPECTOR_TASKS_GROUP, msg_id)
    hlog.info(
        "hunt_completed",
        vulns=len(result.vulnerabilities_found),
        duration_ms=result.total_duration_ms,
    )


async def _publish_inspector_error(
    raw: Redis,
    hunt_id: str,
    error_msg: str,
    log: Any,
) -> None:
    """Publish an empty InspectionResult so the proxy doesn't hang."""
    from ecodiaos.systems.simula.inspector.types import (
        InspectionResult,
        TargetType,
    )

    empty = InspectionResult(
        target_url="error",
        target_type=TargetType.EXTERNAL_REPO,
    )
    try:
        await raw.xadd(
            INSPECTOR_RESULTS_STREAM,
            {
                "hunt_id": hunt_id,
                "result": empty.model_dump_json(),
            },
        )
    except Exception as exc:
        log.error("inspector_error_publish_failed", error=str(exc))


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="EcodiaOS Simula Worker")
    parser.add_argument(
        "--config",
        default=os.getenv("ECODIAOS_CONFIG_PATH"),
        help="Path to YAML config file (default: ECODIAOS_CONFIG_PATH env var)",
    )
    args = parser.parse_args()
    asyncio.run(run_worker(args.config))


if __name__ == "__main__":
    main()
