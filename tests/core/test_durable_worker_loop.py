from __future__ import annotations

import asyncio

import pytest

from cayu.runtime._durable_worker_loop import (
    DurableWorkerCadence,
    DurableWorkerStep,
    lease_heartbeat_interval,
    run_durable_lease_heartbeat,
    run_durable_worker_loop,
)


def test_shared_worker_loop_runs_claim_to_terminal_cycle() -> None:
    pending = ["task-a", "task-b"]
    terminalized: list[str] = []

    async def scenario() -> int:
        async def claim_handle_terminalize(
            _now: float,
            _handled: int,
        ) -> DurableWorkerStep:
            task_id = pending.pop(0)
            terminalized.append(task_id)
            return DurableWorkerStep(handled=1, continue_immediately=True)

        return await run_durable_worker_loop(
            claim_handle_terminalize,
            poll_interval_s=1.0,
            stop=None,
            max_handled=2,
        )

    assert asyncio.run(scenario()) == 2
    assert terminalized == ["task-a", "task-b"]


def test_shared_worker_loop_propagates_handler_failure() -> None:
    handler_failure = RuntimeError("handler failed")

    async def scenario() -> None:
        async def claim_and_handle(_now: float, _handled: int) -> DurableWorkerStep:
            raise handler_failure

        await run_durable_worker_loop(
            claim_and_handle,
            poll_interval_s=1.0,
            stop=None,
        )

    with pytest.raises(RuntimeError, match="handler failed") as raised:
        asyncio.run(scenario())
    assert raised.value is handler_failure


def test_shared_worker_loop_honors_adapter_deadline_and_stop() -> None:
    observed_waits: list[float] = []

    async def scenario() -> int:
        stop = asyncio.Event()

        async def step(now: float, handled: int) -> DurableWorkerStep:
            assert handled == 0
            return DurableWorkerStep(idle=True, next_wake_at=now + 0.25)

        async def wait(seconds: float, wait_stop: asyncio.Event | None) -> bool:
            assert wait_stop is stop
            observed_waits.append(seconds)
            stop.set()
            return True

        return await run_durable_worker_loop(
            step,
            poll_interval_s=10.0,
            stop=stop,
            wait=wait,
        )

    assert asyncio.run(scenario()) == 0
    assert observed_waits == [pytest.approx(0.25, abs=0.01)]


def test_shared_worker_loop_propagates_cancellation_while_idle() -> None:
    wait_started = asyncio.Event()

    async def scenario() -> None:
        async def step(_now: float, _handled: int) -> DurableWorkerStep:
            return DurableWorkerStep(idle=True)

        async def wait(_seconds: float, _stop: asyncio.Event | None) -> bool:
            wait_started.set()
            await asyncio.Event().wait()
            return False

        worker = asyncio.create_task(
            run_durable_worker_loop(
                step,
                poll_interval_s=1.0,
                stop=None,
                wait=wait,
            )
        )
        await wait_started.wait()
        worker.cancel()
        await worker

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())


def test_shared_worker_loop_owns_max_handled_boundary() -> None:
    observed_counts: list[int] = []

    async def scenario() -> int:
        async def step(_now: float, handled: int) -> DurableWorkerStep:
            observed_counts.append(handled)
            return DurableWorkerStep(handled=1, continue_immediately=True)

        return await run_durable_worker_loop(
            step,
            poll_interval_s=1.0,
            stop=None,
            max_handled=2,
        )

    assert asyncio.run(scenario()) == 2
    assert observed_counts == [0, 1]


def test_shared_worker_step_rejects_conflicting_wait_disposition() -> None:
    with pytest.raises(ValueError, match="idle worker step"):
        DurableWorkerStep(idle=True, continue_immediately=True)


def test_shared_worker_cadence_owns_timed_and_every_cycle_reclaim_policy() -> None:
    timed_runs: list[float] = []
    every_cycle_runs: list[float] = []
    current_time = 0.0

    async def scenario() -> None:
        nonlocal current_time
        timed = DurableWorkerCadence(every_s=5.0)
        every_cycle = DurableWorkerCadence(every_s=None)

        async def timed_reclaim() -> None:
            timed_runs.append(current_time)

        async def every_cycle_reclaim() -> None:
            every_cycle_runs.append(current_time)

        ran, _ = await timed.run_if_due(
            timed_reclaim,
            now=current_time,
            clock=lambda: current_time,
        )
        assert ran is True
        assert timed.next_run_at == 5.0

        current_time = 4.0
        ran, _ = await timed.run_if_due(
            timed_reclaim,
            now=current_time,
            clock=lambda: current_time,
        )
        assert ran is False

        current_time = 5.0
        ran, _ = await timed.run_if_due(
            timed_reclaim,
            now=current_time,
            clock=lambda: current_time,
        )
        assert ran is True

        await every_cycle.run_if_due(
            every_cycle_reclaim,
            now=current_time,
            clock=lambda: current_time,
        )
        await every_cycle.run_if_due(
            every_cycle_reclaim,
            now=current_time,
            clock=lambda: current_time,
        )

    asyncio.run(scenario())
    assert timed_runs == [0.0, 5.0]
    assert every_cycle_runs == [5.0, 5.0]


def test_shared_heartbeat_uses_one_third_lease_with_optional_ceiling() -> None:
    assert lease_heartbeat_interval(9) == 3
    assert lease_heartbeat_interval(9, maximum_s=1) == 1


def test_shared_heartbeat_returns_adapter_outcome() -> None:
    observed_waits: list[float] = []
    observed_updates: list[str] = []

    async def scenario() -> str:
        stop = asyncio.Event()

        async def wait(seconds: float, wait_stop: asyncio.Event | None) -> bool:
            assert wait_stop is stop
            observed_waits.append(seconds)
            return False

        async def heartbeat() -> str:
            return "renewed"

        async def inspect(update: str) -> str | None:
            observed_updates.append(update)
            return "terminal"

        return await run_durable_lease_heartbeat(
            heartbeat,
            lease_seconds=9,
            stop=stop,
            stopped_outcome="stopped",
            maximum_interval_s=1,
            after_heartbeat=inspect,
            wait=wait,
        )

    assert asyncio.run(scenario()) == "terminal"
    assert observed_waits == [1]
    assert observed_updates == ["renewed"]


def test_shared_heartbeat_reconciles_post_heartbeat_inspection_failure() -> None:
    inspection_failure = RuntimeError("inspection failed")
    reconciled: list[Exception] = []

    async def scenario() -> str:
        stop = asyncio.Event()

        async def wait(_seconds: float, _stop: asyncio.Event | None) -> bool:
            return False

        async def heartbeat() -> str:
            return "renewed"

        async def inspect(update: str) -> str | None:
            assert update == "renewed"
            raise inspection_failure

        async def reconcile(error: Exception) -> str | None:
            reconciled.append(error)
            return "terminal"

        return await run_durable_lease_heartbeat(
            heartbeat,
            lease_seconds=3,
            stop=stop,
            stopped_outcome="stopped",
            after_heartbeat=inspect,
            on_failure=reconcile,
            wait=wait,
        )

    assert asyncio.run(scenario()) == "terminal"
    assert reconciled == [inspection_failure]


def test_shared_heartbeat_returns_adapter_lost_lease_outcome() -> None:
    heartbeat_failure = KeyError("lease lost")

    async def scenario() -> str:
        stop = asyncio.Event()

        async def wait(_seconds: float, _stop: asyncio.Event | None) -> bool:
            return False

        async def heartbeat() -> object:
            raise heartbeat_failure

        async def reconcile(error: Exception) -> str | None:
            assert error is heartbeat_failure
            return "lost"

        return await run_durable_lease_heartbeat(
            heartbeat,
            lease_seconds=3,
            stop=stop,
            stopped_outcome="stopped",
            on_failure=reconcile,
            wait=wait,
        )

    assert asyncio.run(scenario()) == "lost"


def test_shared_heartbeat_propagates_unreconciled_failure() -> None:
    heartbeat_failure = KeyError("heartbeat unavailable")

    async def scenario() -> None:
        stop = asyncio.Event()

        async def wait(_seconds: float, _stop: asyncio.Event | None) -> bool:
            return False

        async def heartbeat() -> object:
            raise heartbeat_failure

        await run_durable_lease_heartbeat(
            heartbeat,
            lease_seconds=3,
            stop=stop,
            stopped_outcome=None,
            wait=wait,
        )

    with pytest.raises(KeyError, match="heartbeat unavailable") as raised:
        asyncio.run(scenario())
    assert raised.value is heartbeat_failure
