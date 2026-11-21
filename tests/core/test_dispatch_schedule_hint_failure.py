"""An optional schedule hint must not terminate an otherwise healthy dispatch worker."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest
from tests.core.test_dispatch import _batch, _build, _dispatch_request

from cayu.messages import Message
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("fail_observation", [False, True])
async def test_dispatch_worker_survives_one_schedule_hint_read_failure(
    tmp_path, monkeypatch, backend, fail_observation
):
    tasks = (
        InMemoryTaskStore()
        if backend == "memory"
        else SQLiteTaskStore(tmp_path / "schedule-hint.sqlite")
    )
    h = _build([_batch("initial"), _batch("queued result")], task_store=tasks)
    session_id = "schedule-hint-recovery"
    async for _ in h.app.run(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "initial")],
        )
    ):
        pass

    stop = asyncio.Event()
    original_observe = tasks.next_task_schedule_wakeup
    original_acknowledge = h.app._acknowledge_queued_dispatch
    observations = 0
    submitted = None

    async def observe(query=None):
        nonlocal observations, submitted
        observations += 1
        if observations == 1:
            # The worker has just made an empty claim. Publish ordinary,
            # unscheduled work while its optional scheduling hint is read.
            submitted = await h.app.dispatch(_dispatch_request(session_id, "schedule-hint"))
            if fail_observation:
                error_type = ConnectionError if backend == "memory" else sqlite3.OperationalError
                raise error_type("one transient schedule hint read failure")
        return await original_observe(query)

    async def acknowledge(envelope, *, dispatch_status, receipt=None):
        await original_acknowledge(envelope, dispatch_status=dispatch_status, receipt=receipt)
        stop.set()

    monkeypatch.setattr(tasks, "next_task_schedule_wakeup", observe)
    monkeypatch.setattr(h.app, "_acknowledge_queued_dispatch", acknowledge)
    processing = asyncio.create_task(
        h.dispatcher.run_worker(
            h.app,
            worker_id="schedule-worker",
            stop=stop,
            poll_interval_s=0.02,
        )
    )
    try:
        # Ordinary polling must remain available when this optimization fails.
        # A single failed read must not escape run_worker and abandon the queue.
        await asyncio.wait_for(processing, 10)
        assert submitted is not None
        current = await tasks.load_task(submitted.metadata["queue_task_id"])
        assert current is not None and current.status is TaskStatus.COMPLETED
        assert len(h.provider.requests) == 2
    finally:
        stop.set()
        if not processing.done():
            processing.cancel()
        await asyncio.gather(processing, return_exceptions=True)
        await h.app.aclose()
        if backend == "sqlite":
            await tasks.close()
