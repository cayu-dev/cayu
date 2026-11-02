"""An operator lists a crashed workflow run's shared allocation and releases it by hand.

Each "process" is its own CayuApp on one SQLite file, as in
``test_durable_shared_allocations``; the first one crashes after the plan step
and is never resumed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.core.test_human_review import identity
from tests.environments.test_durable_shared_allocations import (
    _crash_after_plan,
    _keyring,
    _PlanThenSolve,
    _records,
    _World,
)
from tests.sandbox_broker.fake_broker import FakeBroker
from tests.sandbox_broker.test_brokered_sandbox import _factory, _project
from tests.sandbox_broker.test_brokered_shared_allocation import (
    _call,
    _done,
    _ResumablePlanThenSolve,
    _WritePlan,
)
from tests.sandbox_broker.test_brokered_shared_allocation import _Crash as _BrokeredCrash

from cayu import AgentSpec, CayuApp, EventType
from cayu.environments import (
    EnvironmentFactoryOperation,
    EnvironmentSpec,
    SharedAllocationReleaseMode,
    SharedAllocationReleaseRefused,
)
from cayu.environments.bindings import SyncBindingSourceConflictError
from cayu.evals import ScriptedModelProvider
from cayu.runtime.public_authority import PublicAuthorityAliasCodec
from cayu.storage import SQLiteSessionStore


def _run(world: _World, scenario):
    async def wrapped():
        try:
            return await scenario()
        finally:
            await world.close()

    return asyncio.run(wrapped())


def test_listing_shows_a_crashed_runs_allocation_without_its_private_state(tmp_path: Path):
    world = _World(tmp_path)

    async def scenario():
        await _crash_after_plan(world, "run-list")
        operator = world.app()
        recent = await operator.list_shared_allocations()
        idle = await operator.list_shared_allocations(
            workflow_run_id="run-list", inactive_for_seconds=0
        )
        other = await operator.list_shared_allocations(environment_name="elsewhere")
        missing = await operator.list_shared_allocations(workflow_run_id="no-such-run")
        return recent, idle, other, missing

    recent, idle, other, missing = _run(world, scenario)
    (allocation,) = recent.allocations
    created = world.sandboxes.requests[0]
    assert allocation.allocation_id == created.session_id
    assert allocation.workflow_run_id == "run-list"
    assert allocation.workflow_name == "plan-then-solve"
    assert (allocation.environment_name, allocation.agent_name) == ("sandbox", "planner")
    assert (allocation.scope, allocation.scope_id) == ("parent", "run-list")
    assert (allocation.state, allocation.reconnectable) == ("live", True)
    assert allocation.claim_generation == 1 and allocation.attempt_id
    assert allocation.created_at is not None
    # It journaled the plan step a moment ago, so it still looks like it's running.
    assert allocation.run_state == "running"
    assert "inactivity window" in (allocation.run_state_reason or "")
    assert recent.next_cursor is None
    # Reconnect metadata and the binding's recovery state never leave the record.
    assert "sandbox" not in str(allocation.metadata)
    assert not hasattr(allocation, "reconnect_metadata")
    (stopped,) = idle.allocations
    assert (stopped.run_state, stopped.run_state_reason) == ("stopped", None)
    assert other.allocations == () and missing.allocations == ()


def test_listing_pages_through_workflow_runs_newest_first(tmp_path: Path):
    world = _World(tmp_path)

    async def scenario():
        await _crash_after_plan(world, "run-older")
        await _crash_after_plan(world, "run-newer")
        operator = world.app()
        first = await operator.list_shared_allocations(limit=1)
        assert first.next_cursor is not None
        second = await operator.list_shared_allocations(limit=1, cursor=first.next_cursor)
        return first, second

    first, second = _run(world, scenario)
    assert [row.workflow_run_id for row in first.allocations] == ["run-newer"]
    assert [row.workflow_run_id for row in second.allocations] == ["run-older"]


def test_discarding_a_crashed_runs_allocation_disposes_of_it_without_copying_back(
    tmp_path: Path,
):
    world = _World(tmp_path)

    async def scenario():
        await _crash_after_plan(world, "run-discard")
        operator = world.app()
        (allocation,) = (await operator.list_shared_allocations()).allocations
        release = await operator.release_shared_allocation(
            allocation.allocation_id, mode="discard", inactive_for_seconds=0
        )
        with pytest.raises(KeyError):
            await operator.release_shared_allocation(
                allocation.allocation_id, mode="discard", inactive_for_seconds=0
            )
        return release, await _records(operator, "run-discard")

    release, records = _run(world, scenario)
    assert release.mode is SharedAllocationReleaseMode.DISCARD and release.forced is False
    assert "nothing was copied back" in release.message
    # Reconnected (to reach it) and discarded: no finalization, no copy-back.
    assert [request.operation for request in world.sandboxes.requests] == [
        EnvironmentFactoryOperation.CREATE,
        EnvironmentFactoryOperation.RECONNECT,
    ]
    assert world.sandboxes.finalizations == []
    assert len(world.sandboxes.disposed) == 1
    assert not (world.source / "plan.md").exists()
    assert records == {}


def test_copying_back_publishes_the_work_and_a_resumed_run_builds_on_it(tmp_path: Path):
    world = _World(tmp_path)

    async def scenario():
        await _crash_after_plan(world, "run-copy")
        operator = world.app()
        (allocation,) = (await operator.list_shared_allocations()).allocations
        release = await operator.release_shared_allocation(
            allocation.allocation_id, mode="copy_back", inactive_for_seconds=0
        )
        records_after_release = await _records(operator, "run-copy")
        # The run is resumed after all: its next step starts a fresh sandbox from
        # the source, which now has the plan.
        resumed = world.app(_call("solve"), _done())
        events = [event async for event in _PlanThenSolve(resumed).run("run-copy")]
        return release, records_after_release, events

    release, records_after_release, events = _run(world, scenario)
    assert release.mode is SharedAllocationReleaseMode.COPY_BACK
    assert "back to the source workspace" in release.message
    assert records_after_release == {}
    assert (world.source / "plan.md").read_bytes() == b"replace a - b with a + b\n"
    # Copied in once, rebuilt from its recorded state, finalized as completed.
    first, reconnect, fresh = world.sandboxes.requests[:3]
    assert reconnect.operation is EnvironmentFactoryOperation.RECONNECT
    assert reconnect.session_id == first.session_id
    assert fresh.operation is EnvironmentFactoryOperation.CREATE
    assert fresh.session_id != first.session_id
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    assert (world.source / "solution.py").read_bytes() == b"def add(a, b):\n    return a + b\n"
    assert world.sandboxes.finalizations == ["completed", "completed"]


def test_release_is_refused_while_the_run_may_be_running_unless_forced(tmp_path: Path):
    world = _World(tmp_path)

    async def scenario():
        crashed = await _crash_after_plan(world, "run-busy")
        operator = world.app()
        (allocation,) = (await operator.list_shared_allocations()).allocations
        with pytest.raises(SharedAllocationReleaseRefused) as recent:
            await operator.release_shared_allocation(allocation.allocation_id, mode="discard")
        # The crashed process still has the run open: it releases its own, and
        # force doesn't change that.
        with pytest.raises(SharedAllocationReleaseRefused) as here:
            await crashed.release_shared_allocation(
                allocation.allocation_id, mode="discard", force=True, inactive_for_seconds=0
            )
        intact = await _records(operator, "run-busy")
        forced = await operator.release_shared_allocation(
            allocation.allocation_id, mode="discard", force=True
        )
        return recent.value, here.value, intact, forced

    recent, here, intact, forced = _run(world, scenario)
    assert recent.reason == "owner_running"
    assert "force" in str(recent)
    assert here.reason == "held_here"
    (record,) = intact["records"].values()
    assert record["state"] == "live"
    assert forced.forced is True and forced.allocation.run_state == "running"
    assert len(world.sandboxes.disposed) == 1


def test_a_failed_copy_back_stays_fenced_and_is_never_reattached(tmp_path: Path):
    world = _World(tmp_path)

    async def scenario():
        await _crash_after_plan(world, "run-fenced")
        operator = world.app()
        (allocation,) = (await operator.list_shared_allocations()).allocations
        # A copy-back that conflicts with the source leaves the allocation and its
        # fenced record for another try.
        (world.source / "plan.md").write_bytes(b"someone else's plan\n")
        with pytest.raises(SyncBindingSourceConflictError):
            await operator.release_shared_allocation(
                allocation.allocation_id, mode="copy_back", inactive_for_seconds=0
            )
        (listed,) = (await operator.list_shared_allocations()).allocations
        fenced = await _records(operator, "run-fenced")
        # The run resumes anyway: it starts a fresh sandbox instead of reattaching
        # the fenced one, which becomes an orphan an operator can discard.
        (world.source / "plan.md").write_bytes(b"replace a - b with a + b\n")
        resumed = world.app(_call("solve"), _done())
        events = [event async for event in _PlanThenSolve(resumed).run("run-fenced")]
        (orphan,) = (await operator.list_shared_allocations()).allocations
        discarded = await operator.release_shared_allocation(
            orphan.allocation_id, mode="discard", inactive_for_seconds=0
        )
        left = await _records(operator, "run-fenced")
        return allocation, listed, fenced, events, orphan, discarded, left

    allocation, listed, fenced, events, orphan, discarded, left = _run(world, scenario)
    assert listed.allocation_id == allocation.allocation_id
    assert (listed.state, listed.reconnectable) == ("reaping", False)
    (record,) = fenced["records"].values()
    assert record["state"] == "reaping"
    assert events[-1].type == EventType.WORKFLOW_COMPLETED, events[-1].payload
    first, copy_back, resumed_create, discard = world.sandboxes.requests
    assert copy_back.session_id == discard.session_id == first.session_id
    assert resumed_create.operation is EnvironmentFactoryOperation.CREATE
    assert resumed_create.session_id != first.session_id
    assert orphan.allocation_id == allocation.allocation_id
    assert (orphan.state, orphan.run_state) == ("reaping", "ended")
    assert discarded.mode is SharedAllocationReleaseMode.DISCARD
    assert left == {}
    # The resumed run's sandbox and the discarded one are both gone.
    assert len(world.sandboxes.disposed) == 2


def test_unknown_or_malformed_allocations_are_not_found(tmp_path: Path):
    world = _World(tmp_path)

    async def scenario():
        await _crash_after_plan(world, "run-missing")
        operator = world.app()
        for allocation_id in (
            "run-missing.shared-0000000000000000",
            "no-such-run.shared-0123456789abcdef",
            "not-an-allocation",
        ):
            with pytest.raises(KeyError):
                await operator.release_shared_allocation(allocation_id, mode="discard")
        with pytest.raises(ValueError):
            await operator.release_shared_allocation("x.shared-0123456789abcdef", mode="keep")
        with pytest.raises(ValueError):
            await operator.list_shared_allocations(limit=0)

    _run(world, scenario)


def test_a_brokered_sandbox_can_only_be_discarded(tmp_path: Path):
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    keyring = _keyring()
    stores: list[SQLiteSessionStore] = []

    def app(*turns) -> CayuApp:
        store = SQLiteSessionStore(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=PublicAuthorityAliasCodec(keyring),
        )
        stores.append(store)
        built = CayuApp(
            session_store=store, enable_logging=False, public_authority_alias_keyring=keyring
        )
        built.register_provider(ScriptedModelProvider(list(turns)), default=True)
        built.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=identity("brokered")),
            _factory(broker, source),
            default=True,
        )
        built.register_agent(AgentSpec(name="planner", model="scripted"), tools=[_WritePlan()])
        return built

    async def scenario():
        try:
            first = app(_call("write_plan"), _done())
            with pytest.raises(_BrokeredCrash):
                async for _ in _ResumablePlanThenSolve(first, crash_after_plan=True).run("run-b"):
                    pass
            # The process dies: its lease keeper stops, and it lets go of what it held.
            registry = first._environment_lifecycle.shared_allocations
            for entry in tuple(registry._live.values()):
                await entry.runner.lease_keeper.stop()
                registry._forget(entry)
            registry._live.clear()
            operator = app()
            page = await operator.list_shared_allocations(inactive_for_seconds=0)
            (allocation,) = page.allocations
            with pytest.raises(SharedAllocationReleaseRefused) as refused:
                await operator.release_shared_allocation(
                    allocation.allocation_id, mode="copy_back", inactive_for_seconds=0
                )
            live_before = [sandbox.id for sandbox in broker.live()]
            release = await operator.release_shared_allocation(
                allocation.allocation_id, mode="discard", inactive_for_seconds=0
            )
            return allocation, refused.value, live_before, release
        finally:
            for store in stores:
                await store.close()

    allocation, refused, live_before, release = asyncio.run(scenario())
    assert (allocation.state, allocation.reconnectable) == ("live", False)
    assert refused.reason == "not_copyable" and "reconnect" in str(refused)
    (sandbox,) = broker.sandboxes.values()
    assert live_before == [sandbox.id]
    assert release.mode is SharedAllocationReleaseMode.DISCARD
    assert sandbox.status == "destroyed" and sandbox.destroy_reason == "requested"
    assert broker.live() == []
