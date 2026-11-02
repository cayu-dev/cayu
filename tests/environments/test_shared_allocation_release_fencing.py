"""A hand release owns its allocation exclusively, and its own writes don't look like run activity."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from tests.environments.test_durable_shared_allocations import (
    _crash,
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
from cayu.environments import EnvironmentFactoryOperation, SharedAllocationReleaseRefused
from cayu.environments._finalization_disposal import checkpoint_finalization_disposal
from cayu.environments.bindings import SyncBindingSourceConflictError
from cayu.runtime._shared_allocation_records import SharedAllocationRecords
from cayu.sessions.records import SessionStatus
from cayu.sessions.recovery import IncompleteSessionsRecoveryRequest


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
        # Recovery takes the lapsed lease over. Recovery then holds it, so a
        # release, whether it read the record before or after, is refused.
        reaped = await records.mark_reaping(held)
        assert reaped is not None
        refused = (
            await records.claim_release(held, token="release-early", lease_seconds=60),
            await records.claim_release(reaped, token="release-late", lease_seconds=60),
        )
        # Once recovery gives its lease up (it couldn't dispose), a release can take it.
        await records.drop_release(reaped)
        snapshot = await records.load("run-lapsed")
        assert snapshot is not None
        (dropped,) = snapshot.all()
        taken = await records.claim_release(dropped, token="release-next", lease_seconds=60)
        return held, reaped, refused, taken

    held, reaped, refused, taken = _run(world, scenario)
    assert reaped is not None and reaped.allocation_session_id == held.allocation_session_id
    assert reaped.claim_generation == held.claim_generation + 1
    assert refused == (None, None)
    assert taken is not None and taken.claim_generation == reaped.claim_generation + 1


def test_a_release_is_refused_while_recovery_holds_the_reaping_fence(tmp_path: Path):
    world = _World(tmp_path)

    async def scenario():
        await _crash_after_plan(world, "run-fenced-first")
        operator = world.app()
        records = SharedAllocationRecords(operator._runtime_session_store)
        snapshot = await records.load("run-fenced-first")
        assert snapshot is not None
        (read_before,) = snapshot.all()
        reaped = await records.mark_reaping(read_before)
        assert reaped is not None
        return (
            await records.claim_release(read_before, token="release-a", lease_seconds=60),
            await records.claim_release(reaped, token="release-b", lease_seconds=60),
            await records.mark_reaping(read_before),
            await records.mark_reaping(reaped),
        )

    # Neither a release nor a second reaper gets an allocation recovery is reaping.
    assert _run(world, scenario) == (None, None, None, None)


def test_a_copy_back_is_refused_while_recovery_disposes_of_the_allocation(
    tmp_path: Path, monkeypatch
):
    world = _World(tmp_path)
    reconnecting = asyncio.Event()
    go_on = asyncio.Event()
    original_create = _ReconnectingFactory.create
    blocked = {"first": True}

    async def slow_reconnect(self, request):
        if request.operation is EnvironmentFactoryOperation.RECONNECT and blocked["first"]:
            blocked["first"] = False
            reconnecting.set()
            await go_on.wait()
        return await original_create(self, request)

    async def scenario():
        await _crash_after_plan(world, "run-reaped")
        operator = world.app()
        registry = operator._environment_lifecycle.shared_allocations

        async def ended(owner):
            return True

        # The run has ended, so recovery reaps its record.
        monkeypatch.setattr(registry, "_owner_ended", ended)
        monkeypatch.setattr(_ReconnectingFactory, "create", slow_reconnect)
        (allocation,) = (await operator.list_shared_allocations()).allocations
        reaping = asyncio.ensure_future(registry.reap_recorded("run-reaped"))
        await asyncio.wait_for(reconnecting.wait(), 30)
        # The operator read the allocation before recovery fenced it.
        with pytest.raises(SharedAllocationReleaseRefused) as refused:
            await operator.release_shared_allocation(
                allocation.allocation_id, mode="copy_back", inactive_for_seconds=0
            )
        go_on.set()
        return refused.value, await reaping, await _records(operator, "run-reaped")

    refused, reaped, records = _run(world, scenario)
    assert refused.reason == "changed"
    assert reaped == 1 and records == {}
    # Recovery discarded it; nothing was copied back alongside.
    assert world.sandboxes.finalizations == []
    assert not (world.source / "plan.md").exists()
    assert len(world.sandboxes.disposed) == 1


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


def test_a_hand_release_is_refused_while_recovery_disposes_of_the_allocation(
    tmp_path: Path, monkeypatch
):
    world = _World(tmp_path)
    disposing = asyncio.Event()
    go_on = asyncio.Event()
    original_create = _ReconnectingFactory.create

    async def journal_completion_then_crash(ctx):
        await ctx._append_current_attempt_event(
            ctx.event(EventType.WORKFLOW_COMPLETED, payload={"workflow": ctx.workflow_name})
        )
        raise RuntimeError("the process died before releasing")

    async def slow_reconnect(self, request):
        # Only recovery's reconnect waits, so a copy-back that wrongly got
        # through fails the test instead of waiting with it.
        if request.operation is EnvironmentFactoryOperation.RECONNECT and not disposing.is_set():
            disposing.set()
            await go_on.wait()
        return await original_create(self, request)

    async def scenario():
        ended = world.app(_call("write_plan"), _done(), _call("solve"), _done())
        with pytest.raises(RuntimeError, match="died before releasing"):
            workflow = _PlanThenSolve(ended, before_completion=journal_completion_then_crash)
            async for _ in workflow.run("run-reaping"):
                pass
        _crash(ended)
        operator = world.app()
        (allocation,) = (await operator.list_shared_allocations()).allocations
        monkeypatch.setattr(_ReconnectingFactory, "create", slow_reconnect)
        recovering = world.app()
        recovery = asyncio.ensure_future(
            recovering.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(statuses={SessionStatus.COMPLETED})
            )
        )
        await asyncio.wait_for(disposing.wait(), 30)
        # Recovery is disposing of the ended run's allocation. A copy-back now
        # would read the sandbox while recovery removes it, so it is refused,
        # whether the operator listed the record before recovery fenced it or
        # after.
        with pytest.raises(SharedAllocationReleaseRefused) as stale:
            await operator.release_shared_allocation(
                allocation.allocation_id, mode="copy_back", inactive_for_seconds=0, force=True
            )
        (fenced,) = (await operator.list_shared_allocations()).allocations
        with pytest.raises(SharedAllocationReleaseRefused) as fresh:
            await operator.release_shared_allocation(
                fenced.allocation_id, mode="copy_back", inactive_for_seconds=0, force=True
            )
        finalizations_mid_disposal = list(world.sandboxes.finalizations)
        go_on.set()
        await recovery
        return (
            stale.value,
            fresh.value,
            finalizations_mid_disposal,
            await _records(operator, "run-reaping"),
        )

    stale, fresh, finalizations_mid_disposal, records = _run(world, scenario)
    assert stale.reason == "changed"
    assert fresh.reason == "changed"
    assert finalizations_mid_disposal == []
    # Recovery finished its disposal: nothing copied back, the record is gone.
    assert world.sandboxes.finalizations == []
    assert len(world.sandboxes.disposed) == 1
    assert records == {}


def test_a_recovery_cancelled_mid_disposal_gives_its_lease_up(tmp_path: Path, monkeypatch):
    world = _World(tmp_path)
    reconnecting = asyncio.Event()
    original_create = _ReconnectingFactory.create

    async def hanging_reconnect(self, request):
        if request.operation is EnvironmentFactoryOperation.RECONNECT:
            reconnecting.set()
            await asyncio.Event().wait()
        return await original_create(self, request)

    async def scenario():
        await _crash_after_plan(world, "run-cancelled")
        operator = world.app()
        registry = operator._environment_lifecycle.shared_allocations

        async def ended(owner):
            return True

        monkeypatch.setattr(registry, "_owner_ended", ended)
        monkeypatch.setattr(_ReconnectingFactory, "create", hanging_reconnect)
        reaping = asyncio.ensure_future(registry.reap_recorded("run-cancelled"))
        await asyncio.wait_for(reconnecting.wait(), 30)
        reaping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reaping
        monkeypatch.setattr(_ReconnectingFactory, "create", original_create)
        # The next recovery takes the allocation at once and disposes of it.
        return await registry.reap_recorded("run-cancelled"), await _records(
            operator, "run-cancelled"
        )

    reaped, records = _run(world, scenario)
    assert reaped == 1 and records == {}
    assert len(world.sandboxes.disposed) == 1
