"""Disposal stops when its authority is lost, even while renewal is blocked."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from tests.environments.test_durable_shared_allocations import (
    _crash_after_plan,
    _ReconnectingFactory,
    _SandboxBinding,
    _World,
)
from tests.environments.test_shared_allocation_release import _run

from cayu.environments import EnvironmentFactoryOperation
from cayu.runtime import _shared_allocation_records as leases
from cayu.runtime._shared_allocation_records import (
    SharedAllocationClaimLost,
    SharedAllocationRecord,
    SharedAllocationRecords,
    keeping_disposal_lease,
)


def _record(seconds=60):
    return SharedAllocationRecord(
        allocation_session_id="run.shared-test",
        scope="parent",
        scope_id="run",
        owner_session_id="run",
        environment_name="sandbox",
        agent_name="agent",
        attempt_id="attempt",
        claim_generation=1,
        state="reaping",
        release_lease={
            "token": "holder",
            "expires_at": (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat(),
        },
    )


@pytest.mark.parametrize("renewal", ["lost", "error", "hang"])
def test_lease_loss_or_expiry_cancels_and_drains_the_body(monkeypatch, renewal):
    monkeypatch.setattr(leases, "DISPOSAL_RENEW_SECONDS", 0.01)

    async def scenario():
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        renewal_stopped = asyncio.Event()

        class Records:
            async def renew_release(self, record, *, lease_seconds):
                await entered.wait()
                if renewal == "lost":
                    return None
                if renewal == "error":
                    raise OSError("store unavailable")
                try:
                    await asyncio.Event().wait()
                finally:
                    renewal_stopped.set()

        with pytest.raises(SharedAllocationClaimLost, match="lost or expired"):
            async with asyncio.timeout(5):
                async with keeping_disposal_lease(Records(), _record(0.2)):  # type: ignore[arg-type]
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cancelled.set()
        assert cancelled.is_set()
        if renewal == "hang":
            assert renewal_stopped.is_set()

    asyncio.run(scenario())


def test_successful_renewal_extends_the_guard(monkeypatch):
    monkeypatch.setattr(leases, "DISPOSAL_RENEW_SECONDS", 0)

    async def scenario():
        renewed = asyncio.Event()
        finish_renewal = asyncio.Event()

        class Records:
            async def renew_release(self, record, *, lease_seconds):
                if not renewed.is_set():
                    renewed.set()
                    return replace(record, release_lease=_record(60).release_lease)
                await finish_renewal.wait()

        async with asyncio.timeout(5):
            async with keeping_disposal_lease(Records(), _record(0.1)) as hold:  # type: ignore[arg-type]
                await renewed.wait()
                # Wait beyond the original deadline; a successful renewal moved it.
                await asyncio.sleep(0.2)
                assert hold.record.release_leased(datetime.now(UTC))

    asyncio.run(scenario())


def test_expired_lease_never_enters_cleanup():
    async def scenario():
        with pytest.raises(SharedAllocationClaimLost, match="expired"):
            async with keeping_disposal_lease(None, _record(-1)):  # type: ignore[arg-type]
                pytest.fail("cleanup ran after its lease expired")

    asyncio.run(scenario())


def test_external_cancellation_is_not_reported_as_lease_loss():
    async def scenario():
        entered = asyncio.Event()

        async def work():
            async with keeping_disposal_lease(None, _record()):  # type: ignore[arg-type]
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(work())
        await entered.wait()
        task.cancel("caller stopped")
        with pytest.raises(asyncio.CancelledError, match="caller stopped"):
            await task
        assert task.cancelled()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["copy_back", "discard", "recovery"])
def test_lost_lease_stops_public_cleanup_and_preserves_new_owner(tmp_path, monkeypatch, mode):
    world = _World(tmp_path)
    monkeypatch.setattr(leases, "DISPOSAL_RENEW_SECONDS", 0)
    original_create = _ReconnectingFactory.create

    async def scenario():
        await _crash_after_plan(world, "run-lost-lease")
        operator = world.app()
        registry = operator._environment_lifecycle.shared_allocations
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def block():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        async def finalize(self, bound, *, outcome=None, metadata=None):
            await block()

        async def create(self, request):
            if request.operation is EnvironmentFactoryOperation.RECONNECT:
                await block()
            return await original_create(self, request)

        async def lose_lease(self, record, *, lease_seconds):
            await started.wait()
            await self.drop_release(record)
            replacement = await self.claim_release(record, token="replacement", lease_seconds=60)
            assert replacement is not None
            return None

        async def ended(owner):
            return True

        monkeypatch.setattr(SharedAllocationRecords, "renew_release", lose_lease)
        if mode == "copy_back":
            monkeypatch.setattr(_SandboxBinding, "finalize", finalize)
        else:
            monkeypatch.setattr(_ReconnectingFactory, "create", create)
        (allocation,) = (await operator.list_shared_allocations()).allocations
        expectation = (
            contextlib.nullcontext()
            if mode == "recovery"
            else pytest.raises(SharedAllocationClaimLost)
        )
        with expectation:
            async with asyncio.timeout(30):
                if mode == "recovery":
                    monkeypatch.setattr(registry, "_owner_ended", ended)
                    assert await registry.reap_recorded("run-lost-lease") == 0
                else:
                    await operator.release_shared_allocation(
                        allocation.allocation_id, mode=mode, inactive_for_seconds=0
                    )
        assert stopped.is_set()
        snapshot = await SharedAllocationRecords(operator._runtime_session_store).load(
            "run-lost-lease"
        )
        assert snapshot is not None
        (retained,) = snapshot.all()
        assert retained.release_lease is not None
        assert retained.release_lease["token"] == "replacement"
        assert world.sandboxes.disposed == []
        assert not (world.source / "plan.md").exists()

    _run(world, scenario)


def test_expired_holder_cannot_renew_its_record(tmp_path):
    world = _World(tmp_path)

    async def scenario():
        await _crash_after_plan(world, "run-expired")
        records = SharedAllocationRecords(world.app()._runtime_session_store)
        snapshot = await records.load("run-expired")
        assert snapshot is not None
        (record,) = snapshot.all()
        held = await records.claim_release(record, token="expired", lease_seconds=0)
        assert held is not None
        assert await records.renew_release(held, lease_seconds=60) is None
        replacement = await records.claim_release(held, token="next", lease_seconds=60)
        assert replacement is not None
        assert await records.renew_release(held, lease_seconds=60) is None

    _run(world, scenario)


def test_body_timeout_is_not_reported_as_lease_loss():
    async def scenario():
        with pytest.raises(TimeoutError, match="provider timed out"):
            async with keeping_disposal_lease(None, _record()):  # type: ignore[arg-type]
                raise TimeoutError("provider timed out")

    asyncio.run(scenario())


def test_swallowed_cancellation_is_detected_before_removing_record(monkeypatch):
    monkeypatch.setattr(leases, "DISPOSAL_RENEW_SECONDS", 0)

    class Records:
        async def renew_release(self, record, *, lease_seconds):
            return None

    async def scenario():
        checked = False
        with pytest.raises(SharedAllocationClaimLost):
            async with asyncio.timeout(5):
                async with keeping_disposal_lease(Records(), _record()) as hold:  # type: ignore[arg-type]
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.Event().wait()
                    with pytest.raises(SharedAllocationClaimLost):
                        hold.check_active()
                    checked = True
        assert checked

    asyncio.run(scenario())


@pytest.mark.parametrize("policy", ["sandbox", "command_or_sandbox"])
@pytest.mark.parametrize("kill_succeeds", [True, False])
def test_lease_expiry_during_docker_workspace_read_never_removes_sandbox(
    monkeypatch, policy, kill_succeeds
):
    from cayu.runners._cleanup import preserve_sandbox_on_command_cleanup
    from cayu.runners.base import ExecResult
    from cayu.runners.docker import DockerRunner
    from cayu.workspaces import RunnerWorkspace

    issued = []
    verified_stops = []

    async def stop_verified_tree(self):
        verified_stops.append(self)
        return kill_succeeds

    monkeypatch.setattr(
        "cayu.runners.docker._DockerCommandHandle._kill_verified_tree", stop_verified_tree
    )

    async def run_subprocess(command, **kwargs):
        issued.append(command.argv)
        if len(issued) == 1:
            await asyncio.Event().wait()
        return ExecResult(exit_code=0 if kill_succeeds else 1)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", run_subprocess)

    async def scenario():
        runner = DockerRunner(
            "retained", docker_path="/usr/bin/docker", cancellation_cleanup=policy
        )
        with pytest.raises(SharedAllocationClaimLost):
            async with asyncio.timeout(5):
                async with keeping_disposal_lease(None, _record(0.1)):  # type: ignore[arg-type]
                    await RunnerWorkspace(runner).list()
        assert not runner.container_removed
        assert runner.lifecycle_state == ("reusable" if kill_succeeds else "poisoned")
        assert not preserve_sandbox_on_command_cleanup.get()

    asyncio.run(scenario())
    assert len(verified_stops) == 1
    assert len(issued) == 1  # No unverified kill script or sandbox deletion.
    assert all(argv[1] == "exec" for argv in issued)


def test_terminal_removal_does_not_race_renewal(monkeypatch):
    monkeypatch.setattr(leases, "DISPOSAL_RENEW_SECONDS", 0)

    async def scenario():
        removed = asyncio.Event()
        return_response = asyncio.Event()
        renewals = 0

        class Records:
            async def renew_release(self, record, *, lease_seconds):
                nonlocal renewals
                renewals += 1
                assert not removed.is_set(), "renewal observed our own deletion"
                return record

            async def remove_reaped(self, record):
                removed.set()
                await return_response.wait()

        async def release():
            async with keeping_disposal_lease(Records(), _record()) as hold:  # type: ignore[arg-type]
                await hold.remove_record(Records())  # type: ignore[arg-type]
                # Keep the scope alive so an uncoordinated keeper would see deletion.
                await asyncio.sleep(0.01)

        async with asyncio.timeout(5):
            task = asyncio.create_task(release())
            await removed.wait()
            before = renewals
            await asyncio.sleep(0.01)
            assert renewals == before
            return_response.set()
            await task

    asyncio.run(scenario())


def test_recovery_continues_after_one_candidate_loses_its_lease(tmp_path, monkeypatch):
    from types import SimpleNamespace

    world = _World(tmp_path)
    monkeypatch.setattr(leases, "DISPOSAL_RENEW_SECONDS", 0)

    async def scenario():
        await _crash_after_plan(world, "run-two-orphans")
        registry = world.app()._environment_lifecycle.shared_allocations
        first = replace(_record(), owner_session_id="run-two-orphans")
        second = replace(first, allocation_session_id="run.shared-second")
        started = asyncio.Event()
        disposed = []
        removed = []

        class Records:
            async def load(self, owner):
                return SimpleNamespace(orphans={"a": first, "b": second}, records={})

            async def mark_reaping(self, record):
                return record

            async def renew_release(self, record, *, lease_seconds):
                if record is first:
                    await started.wait()
                    return None
                return record

            async def drop_release(self, record):
                pass

            async def remove_reaped(self, record):
                removed.append(record.allocation_session_id)

        async def dispose(factory, record, records):
            if record is first:
                started.set()
                await asyncio.Event().wait()
            disposed.append(record.allocation_session_id)
            return True

        async def ended(owner):
            return False

        monkeypatch.setattr(registry, "_records", Records())
        monkeypatch.setattr(registry, "_owner_ended", ended)
        monkeypatch.setattr(
            "cayu.runtime._shared_environment_allocations._dispose_recorded", dispose
        )
        async with asyncio.timeout(5):
            assert await registry.reap_recorded("run-two-orphans") == 1
        assert disposed == removed == [second.allocation_session_id]

    _run(world, scenario)


def test_guarded_docker_cleanup_rejects_unverified_stop_acknowledgement(monkeypatch):
    from cayu.runners.base import ExecResult
    from cayu.runners.docker import DockerRunner
    from cayu.workspaces import RunnerWorkspace

    issued = []

    async def run_subprocess(command, **kwargs):
        issued.append(command.argv)
        if len(issued) == 1:
            await asyncio.Event().wait()
        # A guest could forge this success if the unverified kill script ran.
        return ExecResult(exit_code=0)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", run_subprocess)

    async def scenario():
        runner = DockerRunner(
            "retained", docker_path="/usr/bin/docker", cancellation_cleanup="sandbox"
        )
        with pytest.raises(SharedAllocationClaimLost):
            async with asyncio.timeout(5):
                async with keeping_disposal_lease(None, _record(0.1)):  # type: ignore[arg-type]
                    await RunnerWorkspace(runner).list()
        assert not runner.container_removed
        assert runner.lifecycle_state == "poisoned"
        with pytest.raises(RuntimeError):
            await RunnerWorkspace(runner).list()

    asyncio.run(scenario())
    assert len(issued) == 1
