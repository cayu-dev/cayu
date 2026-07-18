from __future__ import annotations

import asyncio
import contextlib

import pytest

import cayu._task_wait as task_wait
from cayu._task_wait import await_shielded_task_outcome


@pytest.mark.parametrize("child_fails", [False, True])
def test_shielded_task_outcome_retains_preexisting_cancellation(
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
        current.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        try:
            return await await_shielded_task_outcome(asyncio.create_task(child()))
        finally:
            current.uncancel()

    outcome = asyncio.run(run())

    assert isinstance(outcome.cancellation, asyncio.CancelledError)
    if child_fails:
        assert outcome.result is None
        assert outcome.error is child_error
    else:
        assert outcome.result == "done"
        assert outcome.error is None


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
