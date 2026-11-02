"""A hand release owns its allocation exclusively, and its own writes don't look like run activity."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from tests.environments.test_durable_shared_allocations import (
    _crash_after_plan,
    _PlanThenSolve,
    _ReconnectingFactory,
    _records,
    _SandboxBinding,
    _World,
)
from tests.environments.test_shared_allocation_release import _run
from tests.sandbox_broker.test_brokered_shared_allocation import _call, _done

from cayu import EventType
from cayu.environments import SharedAllocationReleaseRefused
from cayu.environments._finalization_disposal import checkpoint_finalization_disposal
from cayu.environments.bindings import SyncBindingSourceConflictError
from cayu.runtime._shared_allocation_records import SharedAllocationRecords


def test_a_second_release_is_refused_while_a_copy_back_runs(tmp_path: Path, monkeypatch):
    world = _World(tmp_path)
    finalizing = asyncio.Event()
    go_on = asyncio.Event()
    original = _SandboxBinding.finalize

    async def slow_finalize(self, bound, *, outcome=None, metadata=None):
        finalizing.set()
        await go_on.wait()
        return await original(self, bound, outcome=outcome, metadata=metadata)

    async def scenario():
        await _crash_after_plan(world, "run-exclusive")
        monkeypatch.setattr(_SandboxBinding, "finalize", slow_finalize)
        operator = world.app()
        (allocation,) = (await operator.list_shared_allocations()).allocations
        copy_back = asyncio.ensure_future(
            operator.release_shared_allocation(
                allocation.allocation_id, mode="copy_back", inactive_for_seconds=0
            )
        )
        await asyncio.wait_for(finalizing.wait(), 30)
        # Another operator (here, the same app) discards it mid-copy-back.
        with pytest.raises(SharedAllocationReleaseRefused) as refused:
            await operator.release_shared_allocation(
                allocation.allocation_id, mode="discard", inactive_for_seconds=0, force=True
            )
        # Recovery's reaping can't fence it either while the release holds it.
        records = SharedAllocationRecords(operator._runtime_session_store)
        snapshot = await records.load("run-exclusive")
        assert snapshot is not None
        (held,) = snapshot.all()
        reaped = await records.mark_reaping(held)
        disposed_mid_copy_back = list(world.sandboxes.disposed)
        go_on.set()
        release = await copy_back
        return refused.value, reaped, disposed_mid_copy_back, release

    refused, reaped, disposed_mid_copy_back, release = _run(world, scenario)
    assert refused.reason == "changed"
    assert reaped is None
    assert disposed_mid_copy_back == []
    assert "back to the source workspace" in release.message
    assert (world.source / "plan.md").read_bytes() == b"replace a - b with a + b\n"
    assert world.sandboxes.finalizations == ["completed"]
    assert len(world.sandboxes.disposed) == 1


def test_an_expired_release_lease_no_longer_blocks_reaping(tmp_path: Path):
    world = _World(tmp_path)

    async def scenario():
        await _crash_after_plan(world, "run-lapsed")
        operator = world.app()
        records = SharedAllocationRecords(operator._runtime_session_store)
        snapshot = await records.load("run-lapsed")
        assert snapshot is not None
        (record,) = snapshot.all()
        # A release whose process died: its lease lapses.
        held = await records.claim_release(record, token="release-dead", lease_seconds=0)
        assert held is not None and held.state == "reaping"
        assert await records.claim_release(record, token="release-stale", lease_seconds=60) is None
        return (
            held,
            await records.mark_reaping(held),
            await records.claim_release(held, token="release-next", lease_seconds=60),
        )

    held, reaped, next_release = _run(world, scenario)
    assert reaped is not None and reaped.allocation_session_id == held.allocation_session_id
    assert next_release is not None and next_release.claim_generation == held.claim_generation + 1


def test_a_retry_after_a_failed_copy_back_is_not_refused_as_activity(tmp_path: Path):
    world = _World(tmp_path)

    async def scenario():
        await _crash_after_plan(world, "run-retry")
        operator = world.app()
        (allocation,) = (await operator.list_shared_allocations()).allocations
        # The run's journal has been quiet for longer than the window.
        time.sleep(1.2)
        (world.source / "plan.md").write_bytes(b"someone else's plan\n")
        with pytest.raises(SyncBindingSourceConflictError):
            await operator.release_shared_allocation(
                allocation.allocation_id, mode="copy_back", inactive_for_seconds=1
            )
        # The failed release wrote the run's records, but the run did nothing.
        (listed,) = (await operator.list_shared_allocations(inactive_for_seconds=1)).allocations
        retry = await operator.release_shared_allocation(
            allocation.allocation_id, mode="discard", inactive_for_seconds=1
        )
        return listed, retry, await _records(operator, "run-retry")

    listed, retry, records = _run(world, scenario)
    assert (listed.state, listed.run_state) == ("reaping", "stopped")
    assert retry.forced is False
    assert records == {}
    assert len(world.sandboxes.disposed) == 1


def test_a_binding_left_mid_finalization_is_settled_by_the_next_release(
    tmp_path: Path, monkeypatch
):
    world = _World(tmp_path)
    original_finalize = _SandboxBinding.finalize
    failures = {"left": 1}

    async def finalize_once_failing(self, bound, *, outcome=None, metadata=None):
        if failures["left"]:
            failures["left"] -= 1
            # Like Docker coding after a copy-back whose container removal failed:
            # it checkpointed its disposal handoff, and the binding keeps its
            # finalization state and its workspace reservations.
            await checkpoint_finalization_disposal({"sandbox": self._name})
            raise RuntimeError("removing the sandbox failed")
        return await original_finalize(self, bound, outcome=outcome, metadata=metadata)

    async def recover_disposal(self, request, state):
        world.sandboxes.dispose(state["sandbox"])

    async def scenario():
        await _crash_after_plan(world, "run-retained")
        monkeypatch.setattr(_SandboxBinding, "finalize", finalize_once_failing)
        monkeypatch.setattr(_SandboxBinding, "abandon", lambda self, bound: False)
        monkeypatch.setattr(_ReconnectingFactory, "recover_finalization_disposal", recover_disposal)
        operator = world.app()
        (allocation,) = (await operator.list_shared_allocations()).allocations
        with pytest.raises(RuntimeError, match="removing the sandbox failed"):
            await operator.release_shared_allocation(
                allocation.allocation_id, mode="copy_back", inactive_for_seconds=0
            )
        retry = await operator.release_shared_allocation(
            allocation.allocation_id, mode="copy_back", inactive_for_seconds=0
        )
        # The source workspace is free again: a resumed run binds it.
        resumed = world.app(_call("solve"), _done())
        events = [event async for event in _PlanThenSolve(resumed).run("run-retained")]
        return retry, events

    retry, events = _run(world, scenario)
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    assert (world.source / "plan.md").read_bytes() == b"replace a - b with a + b\n"
    # The retry finished the retained finalization instead of starting another.
    assert "earlier release" in retry.message


def test_a_live_step_past_the_first_page_of_running_steps_counts():
    from types import SimpleNamespace

    from cayu.runtime._shared_allocation_operations import _running_step_reason
    from cayu.sessions.execution import SessionExecutionState
    from cayu.sessions.records import SessionStatus

    def page(ids, next_cursor=None):
        return SimpleNamespace(
            sessions=[SimpleNamespace(id=session_id) for session_id in ids],
            next_cursor=next_cursor,
        )

    class Store:
        async def list_sessions(self, query):
            if query.status is not SessionStatus.RUNNING:
                return page([])
            if query.cursor is None:
                return page([f"lost-{index}" for index in range(query.limit)], "page-2")
            return page(["live"])

        async def inspect_session_execution(self, session_id):
            state = "executing" if session_id == "live" else "owner_lost"
            return SessionExecutionState(session_id=session_id, state=state, run_epoch=0)

    reason = asyncio.run(_running_step_reason(Store(), "run"))  # type: ignore[arg-type]
    assert reason == "Its step session 'live' is running."
