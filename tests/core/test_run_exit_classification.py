"""One decision table classifies every run, resume and task exit."""

from __future__ import annotations

import asyncio

import pytest

from cayu.runtime._run_exit import RunExitKind, classify_run_exit, delivered_cancellation
from cayu.runtime._session_finalization import _mark_interaction_transition_run_fence
from cayu.sessions.authority import SessionRunFenced


def _classify_in_task(error: BaseException | None, *, cancel_requests: int, baseline: int = 0):
    async def scenario():
        task = asyncio.current_task()
        assert task is not None
        for _ in range(cancel_requests):
            task.cancel()
        try:
            return classify_run_exit(error, cancellation_baseline=baseline)
        finally:
            while task.cancelling():
                task.uncancel()

    async def outer():
        return await asyncio.create_task(scenario())

    return asyncio.run(outer())


def test_return_is_returned():
    exit = classify_run_exit(None)
    assert exit.kind is RunExitKind.RETURNED
    assert exit.error is None


@pytest.mark.parametrize(
    "error",
    [ValueError("boom"), ExceptionGroup("grouped", [ValueError("a"), OSError("b")])],
    ids=["exception", "exception-group"],
)
def test_ordinary_failures_are_failed(error):
    exit = _classify_in_task(error, cancel_requests=1)
    assert exit.kind is RunExitKind.FAILED
    assert exit.error is error
    assert exit.cancellation is None


def test_plain_cancellation_is_cancelled_without_a_delivered_request():
    cancellation = asyncio.CancelledError("caller left")
    exit = _classify_in_task(cancellation, cancel_requests=0)
    assert exit.kind is RunExitKind.CANCELLED
    assert exit.cancellation is cancellation
    assert exit.error is cancellation


def test_grouped_delivered_cancellation_is_cancelled_and_untouched():
    cancellation = asyncio.CancelledError("caller left")
    failure = OSError("sink close failed")
    group = BaseExceptionGroup("delivery", [cancellation, failure])
    exit = _classify_in_task(group, cancel_requests=1)
    assert exit.kind is RunExitKind.CANCELLED
    assert exit.cancellation is cancellation
    assert exit.error is group
    assert list(group.exceptions) == [cancellation, failure]


def test_grouped_undelivered_cancellation_is_a_failure():
    group = BaseExceptionGroup("child cleanup", [asyncio.CancelledError(), OSError("x")])
    exit = _classify_in_task(group, cancel_requests=1, baseline=1)
    assert exit.kind is RunExitKind.FAILED
    assert exit.cancellation is None


def test_delivered_cancellation_respects_the_baseline():
    async def scenario():
        task = asyncio.current_task()
        assert task is not None
        group = BaseExceptionGroup("g", [asyncio.CancelledError()])
        assert delivered_cancellation(group, cancellation_baseline=0) is None
        task.cancel()
        try:
            assert delivered_cancellation(group, cancellation_baseline=0) is group.exceptions[0]
            assert delivered_cancellation(group, cancellation_baseline=1) is None
        finally:
            task.uncancel()

    async def outer():
        await asyncio.create_task(scenario())

    asyncio.run(outer())


@pytest.mark.parametrize("signal_type", [SystemExit, KeyboardInterrupt])
def test_process_control_leaf_wins_over_delivered_cancellation(signal_type):
    signal = signal_type(7)
    cancellation = asyncio.CancelledError()
    group = BaseExceptionGroup(
        "parallel",
        [cancellation, BaseExceptionGroup("process stopping", [signal])],
    )
    exit = _classify_in_task(group, cancel_requests=1)
    assert exit.kind is RunExitKind.PROCESS_CONTROL
    assert exit.supervisory == (signal,)
    assert exit.cancellation is cancellation
    assert exit.error is group


def test_bare_process_control_is_process_control():
    signal = SystemExit(3)
    exit = classify_run_exit(signal)
    assert exit.kind is RunExitKind.PROCESS_CONTROL
    assert exit.supervisory == (signal,)


def test_process_control_in_a_cause_is_causal_history_only():
    cancellation = asyncio.CancelledError()
    cancellation.__cause__ = SystemExit(7)
    exit = classify_run_exit(cancellation)
    assert exit.kind is RunExitKind.CANCELLED
    assert exit.supervisory == ()


def test_generator_exit_is_abandoned_bare_and_grouped():
    assert classify_run_exit(GeneratorExit()).kind is RunExitKind.ABANDONED
    group = BaseExceptionGroup("closing", [GeneratorExit(), OSError("x")])
    exit = _classify_in_task(group, cancel_requests=0)
    assert exit.kind is RunExitKind.ABANDONED
    assert exit.error is group


def test_delivered_cancellation_wins_over_generator_exit():
    cancellation = asyncio.CancelledError()
    group = BaseExceptionGroup("closing", [GeneratorExit(), cancellation])
    exit = _classify_in_task(group, cancel_requests=1)
    assert exit.kind is RunExitKind.CANCELLED
    assert exit.cancellation is cancellation


def test_authenticated_fence_loss_wins_over_everything():
    fenced = SessionRunFenced("stale epoch")
    _mark_interaction_transition_run_fence(fenced)
    cancellation = asyncio.CancelledError()
    cancellation.__context__ = None
    group = BaseExceptionGroup("lost", [cancellation, fenced, SystemExit(1)])
    exit = _classify_in_task(group, cancel_requests=1)
    assert exit.kind is RunExitKind.FENCED
    assert exit.error is group


def test_unauthenticated_fence_error_is_an_ordinary_failure():
    exit = classify_run_exit(SessionRunFenced("not marked by this owner"))
    assert exit.kind is RunExitKind.FAILED


def test_other_base_exception_leaves_are_process_control():
    class WorkerLost(BaseException):
        pass

    lost = WorkerLost()
    exit = classify_run_exit(BaseExceptionGroup("stopping", [lost, ValueError("x")]))
    assert exit.kind is RunExitKind.PROCESS_CONTROL
    assert exit.supervisory == (lost,)
