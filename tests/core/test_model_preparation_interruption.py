from __future__ import annotations

import asyncio
from contextlib import suppress
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from tests.core._execution_profile_fixtures import profiled_session_identity

from cayu.runtime._model_step_executor import (
    _classify_pre_dispatch_failure,
    _model_preparation_lost_to_interrupt,
    _raise_after_model_pre_dispatch_cleanup,
    _terminate_pre_dispatch_context_exposure,
)
from cayu.runtime._run_limits import (
    BudgetReservationLeaseLost,
    BudgetReservationLeaseLostBeforeModelDispatch,
)
from cayu.runtime._session_control import SessionInterruptedByRequest
from cayu.sessions._execution_profile_checkpoint import ActiveInvocationExecutionProfile
from cayu.sessions.cleanup import (
    RecoveryCleanupCapacityExceeded,
    RecoveryCleanupDeadlineExceeded,
    RecoveryCleanupPolicy,
    RecoveryCleanupSupervisor,
)
from cayu.sessions.records import SessionStatus


@pytest.mark.parametrize("control", [asyncio.CancelledError, GeneratorExit, SystemExit])
def test_context_exposure_cleanup_preserves_process_control(control):
    async def run():
        signal = control("stop cleanup")
        store = SimpleNamespace(load_context_exposure=AsyncMock(side_effect=signal))
        exposure = SimpleNamespace(session_id="session", exposure_id="exposure")
        original = RuntimeError("preparation refused")
        with pytest.raises(control) as raised:
            await _terminate_pre_dispatch_context_exposure(
                store=store,
                exposure=exposure,
                failure=original,
                evidence_ref="test",
            )
        assert raised.value is signal
        assert any(control.__name__ in note for note in original.__notes__)

    asyncio.run(run())


@pytest.mark.parametrize(
    "difference",
    [
        "none",
        "instance",
        "epoch",
        "status",
        "marker",
        "marker-id",
        "profile",
        "profile-epoch",
        "profile-session",
    ],
)
def test_preparation_refusal_requires_same_interrupting_invocation(difference):
    async def run():
        original = SimpleNamespace(id="session", instance_id="instance", run_epoch=1)
        current = SimpleNamespace(
            instance_id="replacement" if difference == "instance" else "instance",
            run_epoch=2 if difference == "epoch" else 1,
            status=SessionStatus.RUNNING if difference == "status" else SessionStatus.INTERRUPTING,
        )
        profile = profiled_session_identity(
            provider_name="scripted", model="test"
        ).execution_profile
        active_profile = ActiveInvocationExecutionProfile(
            session_id="other" if difference == "profile-session" else "session",
            interaction_id="interaction",
            run_epoch=2 if difference == "profile-epoch" else 1,
            profile=profile,
        )
        marker = {"reason": "pause", "interruption_request_id": "request"}
        if difference == "marker-id":
            marker.pop("interruption_request_id")
        checkpoint = {
            "pending_session_interrupt": marker,
            "active_invocation_execution_profile": active_profile.model_dump(mode="json"),
        }
        if difference == "marker":
            checkpoint.pop("pending_session_interrupt")
        if difference == "profile":
            checkpoint.pop("active_invocation_execution_profile")
        store = SimpleNamespace(
            load=AsyncMock(return_value=current),
            load_checkpoint=AsyncMock(return_value=checkpoint),
        )
        assert await _model_preparation_lost_to_interrupt(store, original) is (difference == "none")

    asyncio.run(run())


@pytest.mark.parametrize("requests", [1, 2])
def test_pre_dispatch_settlement_retains_cancellation_count_and_typed_failure(requests):
    async def run():
        started = asyncio.Event()
        release = asyncio.Event()
        settled = asyncio.Event()
        original = BudgetReservationLeaseLost("lease expired")

        async def cleanup():
            started.set()
            await release.wait()
            original.add_note("stage retained because exposure termination is not durable")
            settled.set()
            raise BudgetReservationLeaseLostBeforeModelDispatch("before dispatch") from original

        async def caller():
            # A handled historical request must not be consumed by later cleanup.
            asyncio.current_task().cancel("historical")
            with suppress(asyncio.CancelledError):
                await asyncio.sleep(0)
            await _raise_after_model_pre_dispatch_cleanup(
                cleanup,
                unsettled_failure=lambda: original,
                supervisor=RecoveryCleanupSupervisor(),
            )

        task = asyncio.create_task(caller())
        try:
            await asyncio.wait_for(started.wait(), 5)
            for index in range(requests):
                task.cancel("caller stop" if index == 0 else "repeated stop")
                await asyncio.sleep(0)
            assert not settled.is_set()
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError, match="caller stop") as raised:
                await task
            assert settled.is_set()
            assert task.cancelling() == 1 + requests
            cause = raised.value.__cause__
            assert isinstance(cause, BudgetReservationLeaseLostBeforeModelDispatch)
            assert cause.__cause__ is original
            assert "stage retained" in original.__notes__[0]
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("deadline_scope", ["step", "overall"])
@pytest.mark.parametrize("cancel_caller", [False, True])
def test_stuck_pre_dispatch_store_is_bounded_and_remains_supervised(deadline_scope, cancel_caller):
    from cayu import CayuApp, CayuConfig, OperationsConfig

    async def run():
        policy = RecoveryCleanupPolicy(
            step_timeout_seconds=0.03 if deadline_scope == "step" else 1,
            overall_timeout_seconds=0.03 if deadline_scope == "overall" else 1,
        )
        app = CayuApp(
            config=CayuConfig(operations=OperationsConfig(recovery_cleanup_policy=policy))
        )
        supervisor = app._model_step_executor._recovery_cleanup_supervisor
        assert supervisor is app._recovery_cleanup_supervisor
        started = asyncio.Event()
        release = asyncio.Event()
        stopped = asyncio.Event()

        async def stuck_read():
            started.set()
            # Model a driver that does not settle when its await is cancelled.
            while not release.is_set():
                with suppress(asyncio.CancelledError):
                    await release.wait()
            stopped.set()

        store = SimpleNamespace(load_checkpoint=stuck_read)
        refusal = RuntimeError("original model preparation failure")
        interrupt_winner = SessionInterruptedByRequest("session")
        classified = refusal

        async def cleanup():
            nonlocal classified
            # Classification finished before the store stopped responding.
            classified = interrupt_winner
            await store.load_checkpoint()
            raise classified

        caller = asyncio.create_task(
            _raise_after_model_pre_dispatch_cleanup(
                cleanup,
                unsettled_failure=lambda: classified,
                supervisor=supervisor,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 1)
            if cancel_caller:
                caller.cancel("stop stuck cleanup")
                await asyncio.sleep(0)
                caller.cancel("stop again")
                done, _ = await asyncio.wait({caller}, timeout=1)
                assert caller in done
                with pytest.raises(asyncio.CancelledError, match="stop stuck cleanup") as raised:
                    await caller
                failure = raised.value.__cause__
                assert caller.cancelling() == 2
            else:
                done, _ = await asyncio.wait({caller}, timeout=1)
                assert caller in done
                with pytest.raises(SessionInterruptedByRequest) as raised:
                    await caller
                failure = raised.value
            # The deadline explains the unknown cleanup; it does not replace the
            # interrupt classification the runtime must finalize.
            assert failure is interrupt_winner
            deadline = failure.__cause__
            assert isinstance(deadline, RecoveryCleanupDeadlineExceeded)
            assert any("did not settle" in note for note in failure.__notes__)
            assert deadline.scope.value == deadline_scope
            assert not stopped.is_set()
            assert app.recovery_cleanup_status().retained_tasks == 1
            shutdown = await asyncio.wait_for(app.aclose(timeout_s=0.03), 1)
            assert not shutdown.settled
            assert not stopped.is_set()
        finally:
            release.set()
            await asyncio.gather(caller, return_exceptions=True)
            assert await supervisor.drain(timeout_s=1)
            assert app.recovery_cleanup_status().retained_tasks == 0
            await app.aclose(timeout_s=1)

    asyncio.run(run())


@pytest.mark.parametrize("cancel_caller", [False, True])
def test_pre_dispatch_capacity_refusal_keeps_classified_failure(cancel_caller):
    async def run():
        supervisor = RecoveryCleanupSupervisor(
            RecoveryCleanupPolicy(
                step_timeout_seconds=0.03,
                overall_timeout_seconds=0.03,
                max_supervised_tasks=1,
            )
        )
        release = asyncio.Event()

        async def stuck_cleanup():
            while not release.is_set():
                with suppress(asyncio.CancelledError):
                    await release.wait()
            raise RuntimeError("late")

        occupant = RuntimeError("occupant")
        with pytest.raises(RuntimeError, match="occupant"):
            await _raise_after_model_pre_dispatch_cleanup(
                stuck_cleanup, unsettled_failure=lambda: occupant, supervisor=supervisor
            )
        assert supervisor.snapshot().retained_tasks == 1

        cleanup_ran = False

        async def refused_cleanup():
            nonlocal cleanup_ran
            cleanup_ran = True
            raise AssertionError("cleanup must not start without capacity")

        lease_lost = BudgetReservationLeaseLost("lease expired")

        async def caller():
            if cancel_caller:
                asyncio.current_task().cancel("caller stop")
            await _raise_after_model_pre_dispatch_cleanup(
                refused_cleanup,
                unsettled_failure=partial(_classify_pre_dispatch_failure, lease_lost),
                supervisor=supervisor,
            )

        try:
            if cancel_caller:
                with pytest.raises(asyncio.CancelledError, match="caller stop") as raised:
                    await asyncio.create_task(caller())
                failure = raised.value.__cause__
            else:
                with pytest.raises(BudgetReservationLeaseLostBeforeModelDispatch) as raised:
                    await caller()
                failure = raised.value
            assert isinstance(failure, BudgetReservationLeaseLostBeforeModelDispatch)
            assert not cleanup_ran
            # Capacity evidence joins, rather than replaces, the lease cause.
            cause = failure.__cause__
            assert isinstance(cause, BaseExceptionGroup)
            assert isinstance(cause.exceptions[0], RecoveryCleanupCapacityExceeded)
            assert cause.exceptions[1] is lease_lost
        finally:
            release.set()
            assert await supervisor.drain(timeout_s=1)

    asyncio.run(run())
