from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from tests.core.test_abandoned_session_resume import _child, _Proposal
from tests.core.test_runtime import VersionedFakeProvider
from tests.core.test_session_execution_presence import _app, _consume, _stores

from cayu import (
    Message,
    ModelStreamEvent,
    ResumeRequest,
    RunRequest,
    SessionExecutionInProgress,
    SessionStatus,
    ToolEffect,
)
from cayu.sessions import _process_liveness as processes
from cayu.sessions.execution import _ExecutionOwner
from cayu.sessions.records import SessionIdentity


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_resume_does_not_take_successor_before_its_presence_is_published(
    backend, request, sqlite_resources, tmp_path, monkeypatch
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, reopen):
            sid = "successor-" + uuid4().hex
            receipt = tmp_path / "proposal-receipt"
            child = _child(store, backend, request, receipt, sid, ToolEffect.IDEMPOTENT, "crash")
            try:
                stdout, stderr = await asyncio.wait_for(asyncio.to_thread(child.communicate), 30)
                assert child.returncode == 137, (stdout, stderr)
            finally:
                if child.poll() is None:
                    child.kill()
                    await asyncio.to_thread(child.wait)
            loser_store, winner_store = reopen(), reopen()
            reserve_entered, release_reserve = asyncio.Event(), asyncio.Event()
            successor_entered, release_successor = asyncio.Event(), asyncio.Event()
            original_reserve = loser_store.reserve_stalled_run_recovery
            original_claim = winner_store._claim_session_execution

            async def reserve(*args, **kwargs):
                if not reserve_entered.is_set():
                    reserve_entered.set()
                    await release_reserve.wait()
                return await original_reserve(*args, **kwargs)

            async def claim(*args, **kwargs):
                checkpoint = await winner_store.load_checkpoint(sid) or {}
                if checkpoint.get("incomplete_session_recovery_claim") is None:
                    successor_entered.set()
                    await release_successor.wait()
                return await original_claim(*args, **kwargs)

            monkeypatch.setattr(loser_store, "reserve_stalled_run_recovery", reserve)
            monkeypatch.setattr(winner_store, "_claim_session_execution", claim)

            def app(target):
                return _app(
                    target,
                    VersionedFakeProvider([ModelStreamEvent.completed({"finish_reason": "stop"})]),
                    tools=[_Proposal(ToolEffect.IDEMPOTENT, receipt)],
                )

            resume = ResumeRequest(session_id=sid, messages=[Message.text("user", "Any update?")])
            loser = asyncio.create_task(_consume(app(loser_store).resume(resume)))
            winner = None
            try:
                await asyncio.wait_for(reserve_entered.wait(), 30)
                winner = asyncio.create_task(_consume(app(winner_store).resume(resume)))
                await asyncio.wait_for(successor_entered.wait(), 30)
                successor = await store.load(sid)
                assert successor.status is SessionStatus.RUNNING
                release_reserve.set()
                with pytest.raises(SessionExecutionInProgress):
                    await asyncio.wait_for(loser, 30)
                assert (await store.load(sid)).run_epoch == successor.run_epoch
                release_successor.set()
                await asyncio.wait_for(winner, 30)
                assert (await store.load(sid)).status is SessionStatus.COMPLETED
                # Only the winning continuation replays the interrupted call, once.
                assert receipt.read_text() == "committed\ncommitted\n"
            finally:
                release_reserve.set()
                release_successor.set()
                await asyncio.gather(
                    loser, *([] if winner is None else [winner]), return_exceptions=True
                )

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_remote_owner_requires_store_lease_expiry(backend, request, sqlite_resources):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, _):
            sid = "remote-" + uuid4().hex
            await store.create(
                RunRequest(
                    agent_name="assistant", session_id=sid, messages=[Message.text("user", "go")]
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            session = await store.transition_status(
                sid, from_statuses={SessionStatus.PENDING}, to_status=SessionStatus.RUNNING
            )
            await store._claim_session_execution(
                sid,
                token=uuid4().hex,
                owner_id=uuid4().hex,
                owner_kind="in_process_runner",
                owner_label=None,
                process_identity=processes.ProcessIdentity(
                    host_boot_id="another-host", pid=os.getpid()
                ),
                lease_seconds=0.3,
            )
            assert (await store.inspect_session_execution(sid)).state == "executing"
            assert (
                await store.fence_stalled_run(
                    sid, statuses={SessionStatus.RUNNING}, inactive_for_seconds=0
                )
                is None
            )
            await asyncio.sleep(0.35)
            assert (await store.inspect_session_execution(sid)).state == "owner_lost"
            fenced = await store.fence_stalled_run(
                sid, statuses={SessionStatus.RUNNING}, inactive_for_seconds=0
            )
            assert fenced.run_epoch == session.run_epoch + 1

    asyncio.run(scenario())


def test_process_identity_never_crosses_host_or_namespace(monkeypatch):
    monkeypatch.setattr(processes, "_host_boot_id", lambda pid: "current-host-and-namespace")
    identity = processes.ProcessIdentity(host_boot_id="other-host-or-namespace", pid=os.getpid())
    assert processes.process_liveness(identity) == "unknown"


def test_missing_procfs_without_process_death_is_not_abandonment(monkeypatch):
    monkeypatch.setattr(processes.sys, "platform", "linux")
    monkeypatch.setattr(processes, "_host_boot_id", lambda pid: "same-host")

    def hidden(pid):
        raise FileNotFoundError

    monkeypatch.setattr(processes, "_linux_process", hidden)
    monkeypatch.setattr(processes.os, "kill", lambda pid, signal: None)
    identity = processes.ProcessIdentity(host_boot_id="same-host", pid=42, start_id="original")
    assert processes.process_liveness(identity) == "unknown"


def test_pid_reuse_cannot_resurrect_the_original_owner(monkeypatch):
    monkeypatch.setattr(processes.sys, "platform", "linux")
    monkeypatch.setattr(processes, "_host_boot_id", lambda pid: "same-host")
    monkeypatch.setattr(processes, "_linux_process", lambda pid: ("S", "successor"))
    identity = processes.ProcessIdentity(host_boot_id="same-host", pid=42, start_id="original")
    assert processes.process_liveness(identity) == "dead"


def test_local_process_evidence_only_ends_ownership_early(monkeypatch):
    monkeypatch.setattr(processes.sys, "platform", "linux")
    monkeypatch.setattr(processes, "_host_boot_id", lambda pid: "same-host")
    process = {"start": "original"}
    monkeypatch.setattr(processes, "_linux_process", lambda pid: ("S", process["start"]))
    now = datetime.now(UTC)
    owner = _ExecutionOwner(
        session_id="session",
        session_instance_id="incarnation",
        run_epoch=1,
        token="token",
        owner_kind="in_process_runner",
        owner_id="opaque",
        owner_label=None,
        operation_id=None,
        claimed_at=now - timedelta(seconds=10),
        heartbeat_at=now - timedelta(seconds=10),
        lease_expires_at=now - timedelta(seconds=1),
        last_progress_at=now - timedelta(seconds=10),
        process_identity=processes.ProcessIdentity(
            host_boot_id="same-host", pid=42, start_id="original"
        ),
    )
    # A live process cannot keep an expired lease: a successor may fence it.
    assert not processes.execution_owner_is_live(owner, now)
    renewed = owner.model_copy(update={"lease_expires_at": now + timedelta(seconds=30)})
    assert processes.execution_owner_is_live(renewed, now)
    # Same-host death ends ownership before the lease expires.
    process["start"] = "reused-pid"
    assert not processes.execution_owner_is_live(renewed, now)
