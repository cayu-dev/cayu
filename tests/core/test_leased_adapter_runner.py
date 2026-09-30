from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest

from cayu.runtime._leased_adapter_runner import LeasedAdapterLease, LeasedAdapterRunner


def test_cancelled_single_flight_waiter_cannot_split_an_active_queue() -> None:
    async def scenario() -> None:
        runner = LeasedAdapterRunner[object]()
        entered = asyncio.Event()
        release = asyncio.Event()
        successor_entered = asyncio.Event()

        async def first() -> None:
            with runner.single_flight("same-proposal") as lock:
                async with lock:
                    entered.set()
                    await release.wait()

        async def successor() -> None:
            with runner.single_flight("same-proposal") as lock:
                async with lock:
                    successor_entered.set()

        holder = asyncio.create_task(first())
        await entered.wait()
        waiter = asyncio.create_task(successor())
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        successor_task = asyncio.create_task(successor())
        await asyncio.sleep(0)
        assert not successor_entered.is_set()
        release.set()
        await asyncio.gather(holder, successor_task)
        assert successor_entered.is_set()
        assert not runner._locks

    asyncio.run(scenario())


def test_lease_loss_and_heartbeat_failure_cancel_the_same_drain_only_once() -> None:
    async def scenario() -> None:
        runner = LeasedAdapterRunner[object]()
        lost = asyncio.get_running_loop().create_future()
        renewal_failed = asyncio.Event()
        cancellation_seen = asyncio.Event()
        cleanup_allowed = asyncio.Event()
        marker = object()
        messages = []

        async def heartbeat() -> None:
            await renewal_failed.wait()
            raise RuntimeError("late renewal failure")

        async def adapter() -> str:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                messages.append(cancellation.args)
                cancellation_seen.set()
                await cleanup_allowed.wait()
            return "late result"

        heartbeat_task = runner.start_heartbeat(heartbeat(), name="test-lease-renewal")
        invocation = await runner.run(
            "same-proposal",
            adapter,
            name="test-adapter",
            timeout_seconds=0.01,
            lease=LeasedAdapterLease(lost, heartbeat_task, marker),
        )
        assert invocation.outcome.timed_out
        assert runner.active_adapter_count == 1
        runner.retain_drain("same-proposal", invocation.task)
        lost.set_result(RuntimeError("lease expired"))
        await cancellation_seen.wait()
        renewal_failed.set()
        with pytest.raises(RuntimeError, match="late renewal failure"):
            await heartbeat_task
        await asyncio.sleep(0)
        assert messages == [(marker,)]
        assert not invocation.task.done()
        cleanup_allowed.set()
        assert (await invocation.task).result == "late result"
        await asyncio.sleep(0)
        assert runner.active_adapter_count == 0
        assert runner.active_heartbeat_count == 0
        assert runner.draining("same-proposal") is invocation.task

    asyncio.run(scenario())


def test_stale_drain_acknowledgement_does_not_release_a_successor() -> None:
    runner = LeasedAdapterRunner[object]()
    original, successor = object(), object()
    runner.retain_drain("decision", original)
    runner.retain_drain("decision", successor)
    runner.acknowledge_drain("decision", original)
    assert runner.draining("decision") is successor
    runner.acknowledge_drain("decision", successor)
    assert runner.draining("decision") is None


def test_tracking_a_heartbeat_preserves_its_cancellation_cleanup_cause() -> None:
    async def scenario() -> None:
        runner = LeasedAdapterRunner[object]()
        entered = asyncio.Event()

        async def heartbeat() -> None:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                raise cancellation from ValueError("late publication renewal cleanup")

        task = runner.start_heartbeat(heartbeat(), name="test-heartbeat")
        await entered.wait()
        task.cancel("owned shutdown")
        while not task.done():
            await asyncio.sleep(0)
        await asyncio.sleep(0)  # Run bookkeeping before the actual settlement owner.
        with pytest.raises(asyncio.CancelledError) as captured:
            await task
        assert isinstance(captured.value.__cause__, ValueError)
        assert str(captured.value.__cause__) == "late publication renewal cleanup"

    asyncio.run(scenario())


@pytest.mark.parametrize("active", ["lock", "capacity", "adapter", "heartbeat", "drain", "domain"])
def test_inherited_execution_requires_settlement_before_generation_refresh(active: str) -> None:
    async def scenario() -> None:
        runner = LeasedAdapterRunner[object]()
        runner._process_id = -1
        release = asyncio.Event()

        def inherited_failure() -> RuntimeError:
            return RuntimeError("inherited active ownership")

        def require_rebuild() -> None:
            with pytest.raises(RuntimeError, match="inherited active ownership"):
                runner.ensure_process_local(inherited_failure, additional_active=active == "domain")
            assert runner._process_id == -1

        if active == "lock":
            with runner.single_flight("proposal"):
                require_rebuild()
        elif active == "capacity":
            reservation = runner.reserve_capacity(1, inherited_failure)
            require_rebuild()
            runner.release_capacity(reservation)
        elif active == "drain":
            drain = object()
            runner.retain_drain("proposal", drain)
            require_rebuild()
            runner.acknowledge_drain("proposal", drain)
        elif active == "domain":
            require_rebuild()
        else:
            if active == "adapter":
                invocation = await runner.run(
                    "proposal", release.wait, name="test-adapter", timeout_seconds=0
                )
                task = invocation.task
            else:
                task = runner.start_heartbeat(release.wait(), name="test-heartbeat")
            require_rebuild()
            release.set()
            await task
            await asyncio.sleep(0)
        assert runner.ensure_process_local(inherited_failure)
        assert not runner.ensure_process_local(inherited_failure)

    asyncio.run(scenario())


def test_worker_settlement_retains_cleanup_failure_and_repeated_caller_cancellation() -> None:
    async def scenario() -> None:
        runner = LeasedAdapterRunner[object]()
        started = asyncio.Event()
        settling = asyncio.Event()
        release = asyncio.Event()

        async def adapter() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                settling.set()
                await release.wait()
                raise cancellation from ValueError("owned cleanup failed")

        async def heartbeat(stop: asyncio.Event) -> bool:
            await stop.wait()
            return True

        operation = asyncio.create_task(runner.run_to_settlement(adapter, heartbeat))
        await started.wait()
        operation.cancel("first request")
        await settling.wait()
        operation.cancel("second request")
        await asyncio.sleep(0)
        assert not operation.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await operation
        assert captured.value.args == ("first request",)
        assert operation.cancelling() == 2
        assert "owned cleanup failed" in str(captured.value.__cause__)
        await asyncio.sleep(0)
        assert runner.active_adapter_count == runner.active_heartbeat_count == 0

    asyncio.run(scenario())


def test_successful_dispatch_detaches_its_lease_watch() -> None:
    async def scenario() -> None:
        runner = LeasedAdapterRunner[object]()
        lost = asyncio.get_running_loop().create_future()
        stop = asyncio.Event()

        async def heartbeat() -> None:
            await stop.wait()

        async def adapter() -> str:
            return "accepted"

        heartbeat_task = runner.start_heartbeat(heartbeat(), name="test-heartbeat")
        invocation = await runner.run(
            "proposal",
            adapter,
            name="test-adapter",
            timeout_seconds=1,
            lease=LeasedAdapterLease(lost, heartbeat_task, object()),
        )
        assert invocation.outcome.result is not None
        assert invocation.outcome.result.result == "accepted"
        assert not lost._callbacks
        stop.set()
        with suppress(asyncio.CancelledError):
            await heartbeat_task

    asyncio.run(scenario())
