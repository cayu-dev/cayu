"""Timer/cleanup characterization; full runtime coverage lives in the runtime witness."""

import asyncio

import pytest

from cayu.deadlines import ExecutionDeadline, ExecutionDeadlineExceeded
from cayu.runtime._temporary_service_execution import drive_temporary_service

pytestmark = pytest.mark.anyio


def deadline(timeout_ms):
    return ExecutionDeadline.after(
        timeout_ms / 1000, source="clarification_service", scope="temporary_service"
    )


@pytest.mark.parametrize("suppress", (False, True))
async def test_service_expiry_cancels_real_stream_and_closes_it(suppress):
    cancelled = []
    closed = []

    async def stream():
        try:
            try:
                await asyncio.Future()
            except asyncio.CancelledError as error:
                cancelled.append(error)
                if not suppress:
                    raise
        finally:
            closed.append(True)
        if False:
            yield

    with pytest.raises(ExecutionDeadlineExceeded if suppress else TimeoutError) as caught:
        await drive_temporary_service(stream(), deadline(20))
    assert len(cancelled) == 1 and closed == [True]
    assert caught.value.__dict__["temporary_service_deadline"]["scope"] == "temporary_service"


async def test_external_cancellation_remains_plain_cancellation():
    entered = asyncio.Event()
    closed = []

    async def stream():
        try:
            entered.set()
            await asyncio.Future()
        finally:
            closed.append(True)
        if False:
            yield

    task = asyncio.create_task(drive_temporary_service(stream(), deadline(60_000)))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert task.cancelled() and task.cancelling() == 1
    assert not hasattr(caught.value, "temporary_service_deadline")
    assert closed == [True]


async def test_ordinary_failure_preserves_original_exception():
    failure = RuntimeError("provider failure")

    async def stream():
        raise failure
        yield

    with pytest.raises(RuntimeError) as caught:
        await drive_temporary_service(stream(), deadline(60_000))
    assert caught.value is failure
    assert not hasattr(caught.value, "temporary_service_deadline")


async def test_success_does_not_leave_a_cancellation_request():
    task = asyncio.current_task()
    assert task is not None
    baseline = task.cancelling()
    completed = []

    async def stream():
        await asyncio.sleep(0)
        completed.append(True)
        if False:
            yield

    await drive_temporary_service(stream(), deadline(60_000))
    assert completed == [True] and task.cancelling() == baseline
