from __future__ import annotations

import asyncio

import pytest

from cayu.runtime._durable_worker_loop import (
    DurableWorkerCadence,
    DurableWorkerDemandPolicy,
    DurableWorkerMetrics,
    DurableWorkerPollerGroup,
    DurableWorkerStep,
    DurableWorkerWaitResult,
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
            demand_policy=DurableWorkerDemandPolicy(
                dispatch_latency_s=10.0,
                minimum_idle_delay_s=1.0,
                jitter_ratio=0.0,
            ),
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


def test_worker_polling_backoff_jitters_within_bounds_and_resets_on_claim() -> None:
    now = 0.0
    samples = iter((1.0, 0.0, 0.5))
    policy = DurableWorkerDemandPolicy(
        dispatch_latency_s=4.0,
        minimum_idle_delay_s=1.0,
        maximum_idle_delay_s=4.0,
        backoff_multiplier=2.0,
        jitter_ratio=0.25,
    )
    poller = DurableWorkerPollerGroup().subscribe(
        policy,
        clock=lambda: now,
        random_source=lambda: next(samples),
    )

    async def scenario() -> None:
        nonlocal now

        async def empty() -> None:
            return None

        async def task() -> str:
            return "task-a"

        poller.begin_step()
        assert (await poller.claim(empty)).attempted is True
        assert poller.next_wake_at() == pytest.approx(1.25)

        now = 1.25
        poller.begin_step()
        assert (await poller.claim(empty)).attempted is True
        assert poller.next_wake_at() == pytest.approx(2.75)

        now = 2.75
        poller.begin_step()
        assert (await poller.claim(task)).value == "task-a"
        assert poller.next_wake_at() == pytest.approx(2.75)

        poller.begin_step()
        assert (await poller.claim(empty)).attempted is True
        assert poller.next_wake_at() == pytest.approx(3.75)

    asyncio.run(scenario())


def test_worker_metrics_snapshot_is_content_free_and_bounded() -> None:
    now = 0.0
    group = DurableWorkerPollerGroup()
    policy = DurableWorkerDemandPolicy(
        dispatch_latency_s=1.0,
        minimum_idle_delay_s=0.1,
        maximum_idle_delay_s=1.0,
        jitter_ratio=0.0,
    )
    poller = group.subscribe(policy, clock=lambda: now)

    async def scenario() -> None:
        nonlocal now

        async def empty() -> None:
            return None

        async def task() -> str:
            return "opaque-task-id-is-not-recorded"

        poller.begin_step()
        await poller.claim(empty)
        now = 0.1
        poller.note_hint()
        now = 0.35
        poller.begin_step()
        await poller.claim(task)

    try:
        asyncio.run(scenario())
        snapshot = poller.metrics_snapshot()
    finally:
        poller.close()
    assert snapshot.configured_handler_capacity == 1
    assert snapshot.active_handlers == 0
    assert snapshot.active_pollers == 0
    assert snapshot.claim_attempts == 2
    assert snapshot.empty_claims == 1
    assert snapshot.successful_claims == 1
    assert snapshot.failed_claims == 0
    assert snapshot.cancelled_claims == 0
    assert snapshot.wake_hints_received == 1
    assert snapshot.wake_hints_accepted == 1
    assert snapshot.hint_to_claim_latency_samples == 1
    assert snapshot.hint_to_claim_latency_total_s == pytest.approx(0.25)
    assert snapshot.hint_to_claim_latency_max_s == pytest.approx(0.25)


def test_worker_metrics_keep_failed_and_cancelled_claims_out_of_empty_claims() -> None:
    now = 0.0
    metrics = DurableWorkerMetrics()
    poller = DurableWorkerPollerGroup().subscribe(
        DurableWorkerDemandPolicy(
            dispatch_latency_s=1.0,
            minimum_idle_delay_s=0.1,
            jitter_ratio=0.0,
        ),
        clock=lambda: now,
    )
    poller.set_metrics(metrics)

    async def scenario() -> None:
        nonlocal now

        async def failed_claim() -> None:
            raise RuntimeError("store unavailable")

        async def cancelled_claim() -> None:
            raise asyncio.CancelledError

        poller.begin_step()
        with pytest.raises(RuntimeError, match="store unavailable"):
            await poller.claim(failed_claim)
        now = 0.1
        poller.begin_step()
        with pytest.raises(asyncio.CancelledError):
            await poller.claim(cancelled_claim)

    try:
        asyncio.run(scenario())
        snapshot = metrics.snapshot()
    finally:
        poller.close()
    assert snapshot.claim_attempts == 2
    assert snapshot.empty_claims == 0
    assert snapshot.successful_claims == 0
    assert snapshot.failed_claims == 1
    assert snapshot.cancelled_claims == 1
    assert snapshot.store_failures == 1


@pytest.mark.parametrize(
    ("with_maintenance_deadline", "expected_fallback_polls"),
    ((False, 1), (True, 0)),
)
def test_worker_metrics_count_only_claim_audit_timeouts_as_fallback_polls(
    *,
    with_maintenance_deadline: bool,
    expected_fallback_polls: int,
) -> None:
    now = 0.0
    metrics = DurableWorkerMetrics()
    stop = asyncio.Event()

    async def scenario() -> None:
        nonlocal now

        async def step(step_now: float, _handled: int) -> DurableWorkerStep:
            return DurableWorkerStep(
                idle=True,
                next_wake_at=step_now + 0.25 if with_maintenance_deadline else None,
            )

        async def wait(
            seconds: float,
            _wait_stop: asyncio.Event | None,
        ) -> DurableWorkerWaitResult:
            nonlocal now
            now += seconds
            stop.set()
            return DurableWorkerWaitResult.TIMEOUT

        await run_durable_worker_loop(
            step,
            poll_interval_s=10.0,
            stop=stop,
            wait=wait,
            demand_policy=DurableWorkerDemandPolicy(
                dispatch_latency_s=10.0,
                minimum_idle_delay_s=1.0,
                jitter_ratio=0.0,
            ),
            clock=lambda: now,
            metrics=metrics,
        )

    asyncio.run(scenario())
    assert metrics.snapshot().fallback_poll_activations == expected_fallback_polls


def test_hundred_worker_empty_cohort_has_one_fair_authoritative_poller() -> None:
    now = 0.0
    attempts: list[int] = []
    group = DurableWorkerPollerGroup()
    policy = DurableWorkerDemandPolicy(
        dispatch_latency_s=8.0,
        minimum_idle_delay_s=1.0,
        maximum_idle_delay_s=8.0,
        jitter_ratio=0.0,
    )
    pollers = [group.subscribe(policy, clock=lambda: now) for _ in range(100)]

    async def scenario() -> None:
        nonlocal now

        async def empty(index: int) -> None:
            attempts.append(index)
            assert group.active_poller_count == 1
            await asyncio.sleep(0)
            return None

        for _ in range(5):
            for poller in pollers:
                poller.begin_step()
            outcomes = await asyncio.gather(
                *(
                    poller.claim(lambda index=index: empty(index))
                    for index, poller in enumerate(pollers)
                )
            )
            assert sum(outcome.attempted for outcome in outcomes) == 1
            now = min(poller.next_wake_at() for poller in pollers)

    try:
        asyncio.run(scenario())
    finally:
        for poller in pollers:
            poller.close()
    assert attempts == [0, 1, 2, 3, 4]
    assert group.subscriber_count == 0


def test_stalled_claim_gate_expires_for_a_successor_audit() -> None:
    now = 0.0
    group = DurableWorkerPollerGroup()
    policy = DurableWorkerDemandPolicy(
        dispatch_latency_s=1.0,
        minimum_idle_delay_s=0.1,
        maximum_idle_delay_s=1.0,
        jitter_ratio=0.0,
    )
    first = group.subscribe(policy, clock=lambda: now)
    second = group.subscribe(policy, clock=lambda: now)

    async def scenario() -> None:
        nonlocal now
        claim_started = asyncio.Event()
        release_acknowledgement = asyncio.Event()

        async def stalled_claim() -> str:
            claim_started.set()
            await release_acknowledgement.wait()
            return "stale-task"

        async def successor_claim() -> str:
            return "replacement-task"

        stale = asyncio.create_task(first.claim(stalled_claim, maximum_active_s=1.0))
        await claim_started.wait()
        now = 2.0
        replacement = await second.claim(successor_claim, maximum_active_s=1.0)
        assert replacement.value == "replacement-task"
        release_acknowledgement.set()
        assert (await stale).value == "stale-task"

    try:
        asyncio.run(scenario())
    finally:
        first.close()
        second.close()
    assert group.active_poller_count == 0


def test_matching_hint_interrupts_max_backoff_and_forces_a_prompt_audit() -> None:
    now = 0.0
    attempt_times: list[float] = []
    observed_waits: list[float] = []
    policy = DurableWorkerDemandPolicy(
        dispatch_latency_s=4.0,
        minimum_idle_delay_s=1.0,
        maximum_idle_delay_s=4.0,
        jitter_ratio=0.0,
    )
    poller = DurableWorkerPollerGroup().subscribe(policy, clock=lambda: now)

    async def scenario() -> int:
        nonlocal now

        async def empty() -> None:
            attempt_times.append(now)
            return None

        async def step(_step_now: float, _handled: int) -> DurableWorkerStep:
            await poller.claim(empty)
            if len(attempt_times) == 4:
                return DurableWorkerStep(stop=True)
            return DurableWorkerStep(idle=True)

        async def wait(
            seconds: float,
            _stop: asyncio.Event | None,
        ) -> DurableWorkerWaitResult:
            nonlocal now
            observed_waits.append(seconds)
            if len(observed_waits) == 3:
                return DurableWorkerWaitResult.HINT
            now += seconds
            return DurableWorkerWaitResult.TIMEOUT

        return await run_durable_worker_loop(
            step,
            poll_interval_s=4.0,
            stop=None,
            wait=wait,
            demand_policy=policy,
            poller=poller,
            clock=lambda: now,
        )

    assert asyncio.run(scenario()) == 0
    assert observed_waits == [1.0, 2.0, 4.0]
    assert attempt_times == [0.0, 1.0, 3.0, 3.0]
    poller.close()


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
