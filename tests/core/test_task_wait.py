from __future__ import annotations

import asyncio
import contextlib

import pytest

import cayu._task_wait as task_wait
from cayu._task_wait import await_shielded_task_outcome


@pytest.mark.parametrize("child_fails", [False, True])
def test_shielded_task_outcome_ignores_handled_historical_cancellation(
    child_fails: bool,
) -> None:
    child_error = RuntimeError("child failed")

    async def child() -> str:
        if child_fails:
            raise child_error
        return "done"

    async def run() -> task_wait.ShieldedTaskOutcome[str]:
        current = asyncio.current_task()
        assert current is not None
        current.cancel("preexisting cancellation")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.sleep(0)
        assert exc_info.value.args == ("preexisting cancellation",)
        try:
            outcome = await await_shielded_task_outcome(asyncio.create_task(child()))
            assert current.cancelling() == 1
            return outcome
        finally:
            current.uncancel()

    outcome = asyncio.run(run())

    assert outcome.cancellation is None
    if child_fails:
        assert outcome.result is None
        assert outcome.error is child_error
    else:
        assert outcome.result == "done"
        assert outcome.error is None


def test_shielded_task_outcome_retains_explicitly_owned_cancellation() -> None:
    async def run() -> task_wait.ShieldedTaskOutcome[str]:
        current = asyncio.current_task()
        assert current is not None
        current.cancel("owned cancellation")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.sleep(0)
        outcome = await await_shielded_task_outcome(
            asyncio.create_task(asyncio.sleep(0, result="done")),
            cancellation=exc_info.value,
        )
        assert current.cancelling() == 0
        return outcome

    outcome = asyncio.run(run())

    assert outcome.result == "done"
    assert outcome.error is None
    assert isinstance(outcome.cancellation, asyncio.CancelledError)
    assert outcome.cancellation.args == ("owned cancellation",)


def test_shielded_task_outcome_preserves_history_before_explicit_cancellation() -> None:
    async def run() -> task_wait.ShieldedTaskOutcome[str]:
        current = asyncio.current_task()
        assert current is not None
        current.cancel("historical cancellation")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        current.cancel("owned cancellation")
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.sleep(0)
        try:
            outcome = await await_shielded_task_outcome(
                asyncio.create_task(asyncio.sleep(0, result="done")),
                cancellation=exc_info.value,
            )
            assert current.cancelling() == 1
            return outcome
        finally:
            current.uncancel()

    outcome = asyncio.run(run())

    assert outcome.result == "done"
    assert outcome.error is None
    assert isinstance(outcome.cancellation, asyncio.CancelledError)
    assert outcome.cancellation.args == ("owned cancellation",)


def test_shielded_task_outcome_retains_explicit_unwinding_cancellation() -> None:
    child_error = RuntimeError("cleanup failed")
    outcomes: list[task_wait.ShieldedTaskOutcome[None]] = []

    async def child() -> None:
        raise child_error

    async def run() -> None:
        current = asyncio.current_task()
        assert current is not None
        current.cancel("unwinding cancellation")
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as cancellation:
            outcome = await await_shielded_task_outcome(
                asyncio.create_task(child()),
                cancellation=cancellation,
            )
            assert current.cancelling() == 0
            outcomes.append(outcome)
            raise

    with pytest.raises(asyncio.CancelledError, match="unwinding cancellation"):
        asyncio.run(run())

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.result is None
    assert outcome.error is child_error
    assert isinstance(outcome.cancellation, asyncio.CancelledError)
    assert outcome.cancellation.args == ("unwinding cancellation",)


def test_shielded_task_outcome_does_not_claim_child_cancellation_from_historical_count() -> None:
    child_cancellation = asyncio.CancelledError("child cancellation")

    async def child() -> None:
        raise child_cancellation

    async def run() -> task_wait.ShieldedTaskOutcome[None]:
        current = asyncio.current_task()
        assert current is not None
        current.cancel("historical cancellation")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        try:
            outcome = await await_shielded_task_outcome(asyncio.create_task(child()))
            assert current.cancelling() == 1
            return outcome
        finally:
            current.uncancel()

    outcome = asyncio.run(run())

    assert outcome.cancellation is None
    assert outcome.error is child_cancellation


def test_shielded_task_outcome_preserves_historical_count_during_new_cancellation() -> None:
    async def run() -> task_wait.ShieldedTaskOutcome[str]:
        current = asyncio.current_task()
        assert current is not None
        current.cancel("historical cancellation")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)

        child_started = asyncio.Event()
        allow_child = asyncio.Event()

        async def child() -> str:
            child_started.set()
            await allow_child.wait()
            return "done"

        async def cancel_during_wait() -> None:
            await child_started.wait()
            current.cancel("new cancellation")
            allow_child.set()

        canceller = asyncio.create_task(cancel_during_wait())
        try:
            outcome = await await_shielded_task_outcome(asyncio.create_task(child()))
            await canceller
            assert current.cancelling() == 1
            return outcome
        finally:
            current.uncancel()

    outcome = asyncio.run(run())

    assert outcome.result == "done"
    assert outcome.error is None
    assert isinstance(outcome.cancellation, asyncio.CancelledError)
    assert outcome.cancellation.args == ("new cancellation",)


def test_shielded_task_outcome_observes_real_cancellation_during_wait() -> None:
    async def run() -> task_wait.ShieldedTaskOutcome[str]:
        child_started = asyncio.Event()
        allow_child = asyncio.Event()

        async def child() -> str:
            child_started.set()
            await allow_child.wait()
            return "done"

        waiter = asyncio.create_task(await_shielded_task_outcome(asyncio.create_task(child())))
        await child_started.wait()
        waiter.cancel("cancel during shielded wait")
        await asyncio.sleep(0)
        assert not waiter.done()
        allow_child.set()
        outcome = await waiter
        assert not waiter.cancelled()
        assert waiter.cancelling() == 0
        return outcome

    outcome = asyncio.run(run())

    assert outcome.result == "done"
    assert outcome.error is None
    assert isinstance(outcome.cancellation, asyncio.CancelledError)
    assert outcome.cancellation.args == ("cancel during shielded wait",)


def test_shielded_task_outcome_preserves_not_yet_delivered_cancellation_message() -> None:
    async def child() -> str:
        await asyncio.sleep(0)
        return "done"

    async def run() -> task_wait.ShieldedTaskOutcome[str]:
        current = asyncio.current_task()
        assert current is not None
        current.cancel("pending cancellation")
        return await await_shielded_task_outcome(asyncio.create_task(child()))

    outcome = asyncio.run(run())

    assert outcome.result == "done"
    assert isinstance(outcome.cancellation, asyncio.CancelledError)
    assert outcome.cancellation.args == ("pending cancellation",)


def test_shielded_task_outcome_preserves_history_before_pending_cancellation() -> None:
    async def child() -> str:
        await asyncio.sleep(0)
        return "done"

    async def run() -> task_wait.ShieldedTaskOutcome[str]:
        current = asyncio.current_task()
        assert current is not None
        current.cancel("historical cancellation")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        current.cancel("pending cancellation")
        try:
            outcome = await await_shielded_task_outcome(asyncio.create_task(child()))
            assert current.cancelling() == 1
            return outcome
        finally:
            current.uncancel()

    outcome = asyncio.run(run())

    assert outcome.result == "done"
    assert outcome.error is None
    assert isinstance(outcome.cancellation, asyncio.CancelledError)
    assert outcome.cancellation.args == ("pending cancellation",)


def test_shielded_task_outcome_observes_success_completed_during_timeout_checkpoint() -> None:
    async def child() -> str:
        return "done"

    async def run() -> task_wait.ShieldedTaskOutcome[str]:
        return await await_shielded_task_outcome(asyncio.create_task(child()), timeout_s=0)

    outcome = asyncio.run(run())

    assert outcome.timed_out is False
    assert outcome.result == "done"


def test_shielded_task_outcome_observes_failure_completed_during_timeout_checkpoint() -> None:
    child_error = RuntimeError("cleanup failed")

    async def child() -> None:
        raise child_error

    async def run() -> task_wait.ShieldedTaskOutcome[None]:
        return await await_shielded_task_outcome(asyncio.create_task(child()), timeout_s=0)

    outcome = asyncio.run(run())

    assert outcome.timed_out is False
    assert outcome.error is child_error


def test_shielded_task_outcome_zero_timeout_includes_initial_checkpoint() -> None:
    async def run() -> tuple[task_wait.ShieldedTaskOutcome[str], bool]:
        child = asyncio.create_task(asyncio.sleep(0, result="done"))
        outcome = await await_shielded_task_outcome(child, timeout_s=0)
        await child
        return outcome, child.done()

    outcome, child_done = asyncio.run(run())

    assert child_done is True
    assert outcome.timed_out is True
    assert outcome.result is None


def test_shielded_task_outcome_allows_already_completed_task_at_zero_timeout() -> None:
    async def run() -> task_wait.ShieldedTaskOutcome[str]:
        child = asyncio.create_task(asyncio.sleep(0, result="done"))
        await child
        return await await_shielded_task_outcome(child, timeout_s=0)

    outcome = asyncio.run(run())

    assert outcome.timed_out is False
    assert outcome.result == "done"


def test_shielded_task_outcome_counts_checkpoint_time_toward_positive_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> task_wait.ShieldedTaskOutcome[None]:
        loop = asyncio.get_running_loop()
        now = 10.0
        original_sleep = task_wait.asyncio.sleep

        async def advance_clock(delay: float) -> None:
            nonlocal now
            assert delay == 0
            now += 2
            await original_sleep(0)

        monkeypatch.setattr(loop, "time", lambda: now)
        monkeypatch.setattr(task_wait.asyncio, "sleep", advance_clock)
        child = asyncio.create_task(asyncio.Event().wait())
        try:
            return await await_shielded_task_outcome(child, timeout_s=1)
        finally:
            child.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await child

    outcome = asyncio.run(run())

    assert outcome.timed_out is True
    assert outcome.result is None


@pytest.mark.parametrize("timeout_s", [None, 1.0])
def test_shielded_task_outcome_does_not_mask_fatal_signal_with_pending_child(
    monkeypatch: pytest.MonkeyPatch,
    timeout_s: float | None,
) -> None:
    fatal_signal = GeneratorExit("consumer abandoned stream")

    async def raise_fatal(_task: asyncio.Task[object]) -> None:
        raise fatal_signal

    async def run() -> None:
        child = asyncio.create_task(asyncio.Event().wait())
        monkeypatch.setattr(task_wait.asyncio, "shield", raise_fatal)
        try:
            with pytest.raises(GeneratorExit) as exc_info:
                await await_shielded_task_outcome(child, timeout_s=timeout_s)
            assert exc_info.value is fatal_signal
        finally:
            child.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await child

    asyncio.run(run())
