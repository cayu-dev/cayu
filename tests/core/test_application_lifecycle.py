from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging

import pytest

from cayu.runtime.application_lifecycle import (
    ApplicationAdmission,
    ApplicationAdmissionsSealed,
    ApplicationShutdown,
    ShutdownBudget,
    ShutdownStepSpec,
    _AdmissionLease,
    _admitted,
    _admitted_entrance,
    _tracked_entrance,
)


class _Owner:
    def __init__(self) -> None:
        self._admission = ApplicationAdmission()
        self.calls: list[str] = []

    @_admitted_entrance
    async def work(self, release: asyncio.Event | None = None) -> str:
        self.calls.append("work")
        if release is not None:
            await release.wait()
        return "done"

    @_admitted_entrance
    async def outer(self) -> str:
        self._admission.seal()
        # Already admitted: a nested public call is not refused mid-flight.
        return await self.work()

    @_admitted_entrance
    async def stream(self, release: asyncio.Event | None = None):
        yield 1
        if release is not None:
            await release.wait()
        yield 2


def _shutdown(stages, **kwargs) -> tuple[ApplicationShutdown, ApplicationAdmission]:
    admission = ApplicationAdmission()
    return ApplicationShutdown(admission=admission, stages=lambda: stages, **kwargs), admission


def _settled(name: str, calls: list[str] | None = None) -> ShutdownStepSpec:
    async def run(budget_s: float) -> bool:
        if calls is not None:
            calls.append(name)
        return True

    return ShutdownStepSpec(subsystem=name, run=run)


@pytest.mark.parametrize("timeout_s", [0, -1, float("inf"), float("nan"), True, "1"])
def test_budget_rejects_invalid_timeouts(timeout_s) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        ShutdownBudget(timeout_s)


def test_budget_counts_down_and_floors() -> None:
    budget = ShutdownBudget(0.5)
    assert 0 < budget.remaining() <= 0.5
    assert budget.step_budget(floor_s=10.0) == 10.0
    assert not budget.expired()


def test_admission_refuses_after_seal_and_allows_nested_calls() -> None:
    async def scenario() -> None:
        owner = _Owner()
        assert await owner.work() == "done"
        assert await owner.outer() == "done"
        with pytest.raises(ApplicationAdmissionsSealed):
            await owner.work()
        assert owner._admission.in_flight == 0

    asyncio.run(scenario())


def test_stream_admission_is_checked_on_first_advance_and_released_on_close() -> None:
    async def scenario() -> None:
        owner = _Owner()
        stream = owner.stream(asyncio.Event())
        assert owner._admission.in_flight == 0
        assert await anext(stream) == 1
        assert owner._admission.in_flight == 1
        await stream.aclose()
        assert owner._admission.in_flight == 0
        owner._admission.seal()
        sealed = owner.stream()
        with pytest.raises(ApplicationAdmissionsSealed):
            await anext(sealed)
        assert owner._admission.in_flight == 0

    asyncio.run(scenario())


def test_idle_shutdown_settles_and_is_final() -> None:
    async def scenario() -> None:
        calls: list[str] = []
        shutdown, admission = _shutdown([[_settled("a", calls)], [_settled("b", calls)]])
        assert shutdown.state == "open"
        outcome = await shutdown.aclose(timeout_s=1.0)
        assert outcome.settled and outcome.attempt == 1
        assert [step.subsystem for step in outcome.steps] == ["open_operations", "a", "b"]
        assert calls == ["a", "b"]
        assert shutdown.state == "closed" and admission.sealed
        assert await shutdown.aclose(timeout_s=1.0) is outcome
        assert calls == ["a", "b"]

    asyncio.run(scenario())


def test_a_failing_step_does_not_skip_later_steps() -> None:
    async def scenario() -> None:
        calls: list[str] = []

        async def broken(budget_s: float) -> bool:
            raise RuntimeError("drain broke")

        shutdown, _ = _shutdown(
            [[ShutdownStepSpec(subsystem="broken", run=broken)], [_settled("after", calls)]]
        )
        outcome = await shutdown.aclose(timeout_s=1.0)
        assert outcome.status == "failed"
        broken_step = outcome.step("broken")
        assert broken_step is not None
        assert broken_step.failure_type == "RuntimeError"
        assert calls == ["after"]

    asyncio.run(scenario())


def test_incomplete_work_is_reported_and_a_retry_can_settle() -> None:
    async def scenario() -> None:
        results = [False, True]

        async def draining(budget_s: float) -> bool:
            return results.pop(0)

        shutdown, _ = _shutdown([[ShutdownStepSpec(subsystem="drain", run=draining)]])
        first = await shutdown.aclose(timeout_s=1.0)
        assert first.status == "incomplete"
        step = first.step("drain")
        assert step is not None and step.reason == "still_draining"
        assert shutdown.state == "closing"
        second = await shutdown.aclose(timeout_s=1.0)
        assert second.settled and second.attempt == 2

    asyncio.run(scenario())


def test_exhausted_deadline_skips_unprotected_steps_but_runs_floor_protected_ones() -> None:
    async def scenario() -> None:
        calls: list[str] = []

        async def slow(budget_s: float) -> bool:
            await asyncio.sleep(budget_s)
            return False

        async def protected(budget_s: float) -> bool:
            calls.append("protected")
            return True

        shutdown, _ = _shutdown(
            [
                [ShutdownStepSpec(subsystem="slow", run=slow)],
                [_settled("skipped", calls)],
                [ShutdownStepSpec(subsystem="protected", run=protected, floor_protected=True)],
            ]
        )
        outcome = await shutdown.aclose(timeout_s=0.1)
        assert outcome.status == "incomplete"
        skipped = outcome.step("skipped")
        assert skipped is not None and skipped.reason == "deadline_exhausted"
        assert calls == ["protected"]
        assert outcome.elapsed_seconds < 1.0

    asyncio.run(scenario())


def test_an_overrunning_step_is_retained_and_rejoined_by_the_next_attempt() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        starts = 0

        async def stuck(budget_s: float) -> bool:
            nonlocal starts
            starts += 1
            await release.wait()
            return True

        shutdown, _ = _shutdown([[ShutdownStepSpec(subsystem="stuck", run=stuck)]])
        first = await shutdown.aclose(timeout_s=0.05)
        step = first.step("stuck")
        assert step is not None and step.reason == "overran_budget"
        release.set()
        second = await shutdown.aclose(timeout_s=1.0)
        assert second.settled
        # The retained drain was joined, then the subsystem drained again.
        assert starts == 2

    asyncio.run(scenario())


def test_concurrent_callers_share_one_attempt() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        runs = 0

        async def gated(budget_s: float) -> bool:
            nonlocal runs
            runs += 1
            await release.wait()
            return True

        shutdown, _ = _shutdown([[ShutdownStepSpec(subsystem="gated", run=gated)]])
        first = asyncio.create_task(shutdown.aclose(timeout_s=5.0))
        second = asyncio.create_task(shutdown.aclose(timeout_s=5.0))
        await asyncio.sleep(0.01)
        release.set()
        assert (await first) is (await second)
        assert runs == 1

    asyncio.run(scenario())


def test_caller_cancellation_propagates_while_the_attempt_finishes() -> None:
    async def scenario() -> None:
        release = asyncio.Event()

        async def gated(budget_s: float) -> bool:
            await release.wait()
            return True

        shutdown, _ = _shutdown([[ShutdownStepSpec(subsystem="gated", run=gated)]])
        caller = asyncio.create_task(shutdown.aclose(timeout_s=5.0))
        await asyncio.sleep(0.01)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert caller.cancelled()
        release.set()
        outcome = await shutdown.aclose(timeout_s=5.0)
        assert outcome.settled and outcome.attempt == 1

    asyncio.run(scenario())


def test_process_control_signal_is_reraised_after_later_steps() -> None:
    async def scenario() -> None:
        calls: list[str] = []

        async def interrupted(budget_s: float) -> bool:
            raise KeyboardInterrupt

        shutdown, _ = _shutdown(
            [
                [ShutdownStepSpec(subsystem="interrupted", run=interrupted)],
                [_settled("after", calls)],
            ]
        )
        with pytest.raises(KeyboardInterrupt):
            await shutdown.aclose(timeout_s=1.0)
        assert calls == ["after"]
        assert shutdown.outcome is not None and shutdown.outcome.status == "failed"

    asyncio.run(scenario())


class _Resource:
    def __init__(self, log: list[str], name: str) -> None:
        self.log = log
        self.name = name

    async def close(self) -> None:
        self.log.append(self.name)


def test_owned_resources_close_once_in_reverse_order_only_when_settled() -> None:
    async def scenario() -> None:
        log: list[str] = []
        results = [False, True]

        async def draining(budget_s: float) -> bool:
            return results.pop(0)

        shutdown, _ = _shutdown(
            [[ShutdownStepSpec(subsystem="drain", run=draining)]],
            owned_resources=(_Resource(log, "first"), _Resource(log, "second")),
        )
        incomplete = await shutdown.aclose(timeout_s=1.0)
        assert incomplete.owned_resources == "retained" and log == []
        settled = await shutdown.aclose(timeout_s=1.0)
        assert settled.owned_resources == "released"
        assert log == ["second", "first"]

    asyncio.run(scenario())


def test_owned_resources_require_async_close() -> None:
    class SyncClose:
        def close(self) -> None:
            pass

    with pytest.raises(TypeError, match="async close"):
        ApplicationShutdown(
            admission=ApplicationAdmission(),
            stages=lambda: (),
            owned_resources=(SyncClose(),),  # ty: ignore[invalid-argument-type]
        )


def test_open_operations_are_waited_for_then_reported() -> None:
    async def scenario() -> None:
        owner = _Owner()
        shutdown = ApplicationShutdown(admission=owner._admission, stages=lambda: ())
        release = asyncio.Event()
        running = asyncio.create_task(owner.work(release))
        await asyncio.sleep(0)
        outcome = await shutdown.aclose(timeout_s=0.05)
        assert outcome.status == "incomplete" and outcome.open_operations == 1
        step = outcome.step("open_operations")
        assert step is not None and step.reason == "open_operations"
        release.set()
        assert await running == "done"
        assert (await shutdown.aclose(timeout_s=1.0)).settled

    asyncio.run(scenario())


def test_late_work_downgrades_only_settled_steps_with_its_reason() -> None:
    async def scenario() -> None:
        async def unfinished(budget_s: float) -> bool:
            return False

        shutdown, _ = _shutdown(
            [
                [_settled("interruptions")],
                [_settled("providers")],
                [ShutdownStepSpec(subsystem="unfinished", run=unfinished)],
            ],
            late_work=lambda: {
                "interruptions": "late_work",
                "providers": "unowned_cancellations",
                "unfinished": "late_work",
            },
        )
        outcome = await shutdown.aclose(timeout_s=1.0)
        assert outcome.status == "incomplete"
        reasons = {step.subsystem: step.reason for step in outcome.steps}
        assert reasons["interruptions"] == "late_work"
        assert reasons["providers"] == "unowned_cancellations"
        # A step already incomplete keeps its own reason.
        assert reasons["unfinished"] == "still_draining"

    asyncio.run(scenario())


def test_a_failed_close_is_not_released_and_a_retry_closes_it_again() -> None:
    async def scenario() -> None:
        calls: list[int] = []

        class Flaky:
            async def close(self) -> None:
                calls.append(1)
                if len(calls) == 1:
                    raise RuntimeError("close failed")

        shutdown, _ = _shutdown([[_settled("drain")]], owned_resources=(Flaky(),))
        first = await shutdown.aclose(timeout_s=1.0)
        assert first.status == "failed" and first.owned_resources == "failed"
        second = await shutdown.aclose(timeout_s=1.0)
        assert second.settled and second.owned_resources == "released"
        assert len(calls) == 2

    asyncio.run(scenario())


def test_a_hanging_close_is_bounded_and_rejoined() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        calls: list[int] = []

        class Slow:
            async def close(self) -> None:
                calls.append(1)
                await release.wait()

        shutdown, _ = _shutdown([[_settled("drain")]], owned_resources=(Slow(),))
        first = await shutdown.aclose(timeout_s=0.2)
        assert first.status == "incomplete" and first.owned_resources == "retained"
        assert first.elapsed_seconds < 1.0
        step = first.step("owned_resources")
        assert step is not None and step.reason == "overran_budget"
        release.set()
        second = await shutdown.aclose(timeout_s=1.0)
        assert second.settled and second.owned_resources == "released"
        assert len(calls) == 1

    asyncio.run(scenario())


def test_a_step_cancelled_on_its_own_fails_without_cancelling_the_caller() -> None:
    async def scenario() -> None:
        async def cancelled(budget_s: float) -> bool:
            raise asyncio.CancelledError("step cancelled itself")

        shutdown, _ = _shutdown(
            [[ShutdownStepSpec(subsystem="cancelled", run=cancelled)], [_settled("after")]]
        )
        outcome = await shutdown.aclose(timeout_s=1.0)
        current = asyncio.current_task()
        assert current is not None and current.cancelling() == 0
        assert outcome.status == "failed"
        step = outcome.step("cancelled")
        assert step is not None and step.failure_type == "CancelledError"
        after = outcome.step("after")
        assert after is not None and after.status == "settled"

    asyncio.run(scenario())


def test_a_step_skipped_for_an_exhausted_deadline_still_seals() -> None:
    async def scenario() -> None:
        sealed: list[str] = []

        async def slow(budget_s: float) -> bool:
            await asyncio.sleep(budget_s)
            return False

        async def never_runs(budget_s: float) -> bool:
            raise AssertionError("skipped")

        shutdown, _ = _shutdown(
            [
                [ShutdownStepSpec(subsystem="slow", run=slow)],
                [
                    ShutdownStepSpec(
                        subsystem="sealed", run=never_runs, seal=lambda: sealed.append("sealed")
                    )
                ],
            ]
        )
        outcome = await shutdown.aclose(timeout_s=0.1)
        step = outcome.step("sealed")
        assert step is not None and step.reason == "deadline_exhausted"
        assert sealed == ["sealed"]

    asyncio.run(scenario())


class _Spawner:
    def __init__(self) -> None:
        self._admission = ApplicationAdmission()
        self.spawned: list[asyncio.Task] = []

    @_admitted_entrance
    async def work(self, release: asyncio.Event | None = None) -> str:
        if release is not None:
            await release.wait()
        return "done"

    @_admitted_entrance
    async def spawn(self, start: asyncio.Event, release: asyncio.Event) -> None:
        async def later() -> str:
            await start.wait()
            return await self.work(release)

        # The background task outlives this admitted call.
        self.spawned.append(asyncio.create_task(later()))


def test_work_from_a_finished_operation_is_refused_after_sealing() -> None:
    async def scenario() -> None:
        owner = _Spawner()
        start, release = asyncio.Event(), asyncio.Event()
        await owner.spawn(start, release)
        owner._admission.seal()
        start.set()
        with pytest.raises(ApplicationAdmissionsSealed):
            await asyncio.wait_for(owner.spawned[0], 5)
        assert owner._admission.in_flight == 0

    asyncio.run(scenario())


def test_admitted_work_still_running_after_the_stages_is_reported() -> None:
    async def scenario() -> None:
        owner = _Spawner()
        release = asyncio.Event()
        started = asyncio.Event()

        async def starts_work(budget_s: float) -> bool:
            # A drain whose cleanup admits work that is still running afterwards.
            async def descendant() -> None:
                await owner.work(release)

            token = _admitted.set((_AdmissionLease(owner._admission, counted=False),))
            try:
                owner.spawned.append(asyncio.create_task(descendant()))
            finally:
                _admitted.reset(token)
            await asyncio.sleep(0)
            started.set()
            return True

        shutdown = ApplicationShutdown(
            admission=owner._admission,
            stages=lambda: [[ShutdownStepSpec(subsystem="drain", run=starts_work)]],
        )
        outcome = await shutdown.aclose(timeout_s=1.0)
        assert started.is_set()
        assert outcome.status == "incomplete" and outcome.open_operations == 1
        step = outcome.step("open_operations")
        assert step is not None and step.reason == "late_work"
        release.set()
        await asyncio.wait_for(asyncio.gather(*owner.spawned), 5)
        assert (await shutdown.aclose(timeout_s=1.0)).settled

    asyncio.run(scenario())


class _Tracked:
    def __init__(self) -> None:
        self._admission = ApplicationAdmission()

    @_admitted_entrance
    async def gated(self) -> str:
        return "gated"

    @_tracked_entrance
    async def allowed(self, release: asyncio.Event | None = None) -> str:
        if release is not None:
            await release.wait()
        return "allowed"

    @_tracked_entrance
    async def allowed_then_gated(self) -> str:
        return await self.gated()

    @_admitted_entrance
    async def gated_then_nested(self, release: asyncio.Event) -> str:
        await self.allowed()
        await release.wait()
        return await self.gated()


def test_tracked_entrances_are_counted_but_never_refused() -> None:
    async def scenario() -> None:
        owner = _Tracked()
        release = asyncio.Event()
        running = asyncio.create_task(owner.allowed(release))
        await asyncio.sleep(0)
        assert owner._admission.in_flight == 1
        owner._admission.seal()
        assert await owner.allowed() == "allowed"
        release.set()
        assert await running == "allowed"
        assert owner._admission.in_flight == 0

    asyncio.run(scenario())


def test_a_tracked_entrance_does_not_let_gated_work_past_the_seal() -> None:
    async def scenario() -> None:
        owner = _Tracked()
        owner._admission.seal()
        with pytest.raises(ApplicationAdmissionsSealed):
            await owner.allowed_then_gated()
        assert owner._admission.in_flight == 0

    asyncio.run(scenario())


def test_nested_calls_in_the_same_task_count_once() -> None:
    async def scenario() -> None:
        owner = _Tracked()
        release = asyncio.Event()
        running = asyncio.create_task(owner.gated_then_nested(release))
        await asyncio.sleep(0)
        assert owner._admission.in_flight == 1
        release.set()
        assert await running == "gated"
        assert owner._admission.in_flight == 0

    asyncio.run(scenario())


def test_closing_a_counted_call_from_another_context_still_releases_it() -> None:
    async def scenario() -> None:
        owner = _Tracked()
        call = owner.allowed(asyncio.Event())
        # Start the call so its context variables are set, then close it from a
        # fresh context, as garbage collection of a pending task would.
        call.send(None)
        assert owner._admission.in_flight == 1
        with contextlib.suppress(ValueError):
            contextvars.Context().run(call.close)
        assert owner._admission.in_flight == 0

    asyncio.run(scenario())


class _Caller:
    def __init__(self) -> None:
        self._admission = ApplicationAdmission()

    @_admitted_entrance
    async def gated(self) -> str:
        return "gated"

    @_admitted_entrance
    async def seal_then_call(self, other: _Caller) -> str:
        self._admission.seal()
        return await other.call_back(self)

    @_admitted_entrance
    async def call_back(self, other: _Caller) -> str:
        return await other.gated()


def test_work_nested_across_two_applications_keeps_each_admission() -> None:
    async def scenario() -> None:
        first, second = _Caller(), _Caller()
        # first -> second -> first: first's own operation is still running, so
        # its nested call is admitted though another app's call sits between.
        assert await first.seal_then_call(second) == "gated"
        with pytest.raises(ApplicationAdmissionsSealed):
            await first.gated()

    asyncio.run(scenario())


@pytest.mark.parametrize("exhausted", [False, True])
def test_a_failing_seal_is_a_failed_step_and_later_steps_still_run(exhausted: bool) -> None:
    async def scenario() -> None:
        calls: list[str] = []

        async def slow(budget_s: float) -> bool:
            await asyncio.sleep(budget_s)
            return False

        def broken_seal() -> None:
            raise RuntimeError("seal broke")

        async def drained(budget_s: float) -> bool:
            calls.append("drained")
            return True

        stages: list[list[ShutdownStepSpec]] = [
            [ShutdownStepSpec(subsystem="sealed", run=drained, seal=broken_seal)],
            [_settled("after", calls)],
        ]
        if exhausted:
            stages.insert(0, [ShutdownStepSpec(subsystem="slow", run=slow)])
        shutdown, _ = _shutdown(stages, owned_resources=(_Resource([], "store"),))
        outcome = await shutdown.aclose(timeout_s=0.1 if exhausted else 1.0)
        step = outcome.step("sealed")
        assert step is not None and step.status == "failed"
        assert step.failure_type == "RuntimeError"
        assert outcome.status == "failed" and outcome.owned_resources == "retained"
        after = outcome.step("after")
        assert after is not None
        if not exhausted:
            assert calls == ["after"]

    asyncio.run(scenario())


def test_a_seal_raising_a_process_control_signal_is_reraised_after_cleanup() -> None:
    async def scenario() -> None:
        calls: list[str] = []

        def interrupted_seal() -> None:
            raise KeyboardInterrupt

        async def unused(budget_s: float) -> bool:
            raise AssertionError("the drain does not run after its seal failed")

        shutdown, _ = _shutdown(
            [
                [ShutdownStepSpec(subsystem="sealed", run=unused, seal=interrupted_seal)],
                [_settled("after", calls)],
            ]
        )
        with pytest.raises(KeyboardInterrupt):
            await shutdown.aclose(timeout_s=1.0)
        assert calls == ["after"]

    asyncio.run(scenario())


@pytest.mark.parametrize("late_error", [RuntimeError("late"), KeyboardInterrupt()])
def test_a_rejoined_drain_reports_how_it_ended(late_error: BaseException) -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        runs = 0

        async def overrunning(budget_s: float) -> bool:
            nonlocal runs
            runs += 1
            if runs == 1:
                await release.wait()
                raise late_error
            return True

        shutdown, _ = _shutdown([[ShutdownStepSpec(subsystem="drain", run=overrunning)]])
        first = await shutdown.aclose(timeout_s=0.05)
        step = first.step("drain")
        assert step is not None and step.reason == "overran_budget"
        release.set()
        await asyncio.sleep(0)
        if isinstance(late_error, KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                await shutdown.aclose(timeout_s=1.0)
            second = shutdown.outcome
        else:
            second = await shutdown.aclose(timeout_s=1.0)
        assert second is not None and second.status == "failed"
        failed = second.step("drain")
        assert failed is not None and failed.failure_type == type(late_error).__qualname__
        # The failure was reported once; the next attempt drains afresh.
        third = await shutdown.aclose(timeout_s=1.0)
        assert third.settled and runs == 2

    asyncio.run(scenario())


_CANARY = "sk-live-shutdown-canary-7f3a"


def test_shutdown_failure_logs_never_contain_exception_content(caplog, capsys) -> None:
    async def scenario() -> None:
        async def leaking(budget_s: float) -> bool:
            try:
                raise ValueError(f"cause {_CANARY}")
            except ValueError as cause:
                raise ExceptionGroup(
                    f"group {_CANARY}", [RuntimeError(f"member {_CANARY}")]
                ) from cause

        class LeakingResource:
            async def close(self) -> None:
                raise ConnectionError(f"postgres://user:{_CANARY}@db/app")

        shutdown, _ = _shutdown(
            [[ShutdownStepSpec(subsystem="leaking", run=leaking)]],
            owned_resources=(LeakingResource(),),
        )
        with caplog.at_level(logging.DEBUG):
            first = await shutdown.aclose(timeout_s=1.0)
        assert first.status == "failed"
        step = first.step("leaking")
        assert step is not None and step.failure_type == "ExceptionGroup"

    asyncio.run(scenario())
    logged = "\n".join(
        record.getMessage()
        + (logging.Formatter().formatException(record.exc_info) if record.exc_info else "")
        for record in caplog.records
    )
    captured = capsys.readouterr()
    assert "leaking" in logged
    assert _CANARY not in logged
    assert _CANARY not in captured.out + captured.err


@pytest.mark.parametrize(
    ("members", "process_control"),
    [
        ([asyncio.CancelledError()], False),
        ([asyncio.CancelledError(), RuntimeError("cleanup")], False),
        ([ExceptionGroup("inner", [ValueError("x")]), asyncio.CancelledError()], False),
        ([asyncio.CancelledError(), KeyboardInterrupt()], True),
    ],
)
def test_exception_groups_are_classified_by_their_members(
    members: list[BaseException], process_control: bool
) -> None:
    async def scenario() -> None:
        calls: list[str] = []

        async def grouped(budget_s: float) -> bool:
            raise BaseExceptionGroup("cleanup failed", members)

        shutdown, _ = _shutdown(
            [[ShutdownStepSpec(subsystem="grouped", run=grouped)], [_settled("after", calls)]]
        )
        if process_control:
            with pytest.raises(BaseExceptionGroup) as raised:
                await shutdown.aclose(timeout_s=1.0)
            # The original group is kept, with its genuine signal inside.
            assert raised.value.subgroup(KeyboardInterrupt) is not None
            outcome = shutdown.outcome
        else:
            outcome = await shutdown.aclose(timeout_s=1.0)
        assert outcome is not None and outcome.status == "failed"
        assert calls == ["after"]

    asyncio.run(scenario())


def test_a_resource_handed_over_twice_closes_once() -> None:
    async def scenario() -> None:
        log: list[str] = []
        resource = _Resource(log, "store")
        shutdown, _ = _shutdown([[_settled("drain")]], owned_resources=(resource, resource))
        outcome = await shutdown.aclose(timeout_s=1.0)
        assert outcome.settled and log == ["store"]

    asyncio.run(scenario())
