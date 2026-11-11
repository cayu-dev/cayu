"""Cancellation must retain task disposition across a transient fence lookup failure."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_dispatch import FakeProvider, _batch, _configured_app, _dispatch_request

from cayu.messages import Message
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.dispatch import TaskStoreDispatcher
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("fail_fence_read", [False, True])
async def test_cancelled_dispatch_settles_after_one_fence_read_failure(
    monkeypatch, tmp_path, backend, fail_fence_read
):
    now = [datetime.now(UTC)]
    tasks = (
        InMemoryTaskStore(ownership_clock=lambda: now[0])
        if backend == "memory"
        else SQLiteTaskStore(tmp_path / "tasks.sqlite", ownership_clock=lambda: now[0])
    )
    sessions = InMemorySessionStore()
    dispatcher = TaskStoreDispatcher(tasks)
    producer = _configured_app(
        session_store=sessions,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=FakeProvider([_batch("initial")]),
    )
    session_id = "fence-read-failure"
    async for _ in producer.run(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "initial")],
        )
    ):
        pass
    submitted = await producer.dispatch(_dispatch_request(session_id, "fence-read-failure"))
    task_id = submitted.metadata["queue_task_id"]

    class BlockingProvider(FakeProvider):
        def __init__(self):
            super().__init__([_batch("queued result")])
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request):
            self.started.set()
            await self.release.wait()
            async for event in super().stream(request):
                yield event

    provider = BlockingProvider()
    worker = _configured_app(
        session_store=sessions,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=provider,
    )
    processing = asyncio.create_task(dispatcher.process_next(worker, worker_id="owner"))
    read_attempted = asyncio.Event()
    original_load = tasks.load_task
    failures = 0

    async def load_with_one_failed_fence_read(identity):
        nonlocal failures
        if identity == task_id and not read_attempted.is_set():
            read_attempted.set()
            if fail_fence_read:
                failures += 1
                raise ConnectionError("temporary cancellation-fence lookup failure")
        return await original_load(identity)

    try:
        await asyncio.wait_for(provider.started.wait(), 10)
        monkeypatch.setattr(tasks, "load_task", load_with_one_failed_fence_read)
        processing.cancel("worker shutdown")
        await asyncio.wait_for(read_attempted.wait(), 10)
        # Storage is healthy for the rest of the operation, including its natural return.
        provider.release.set()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await asyncio.wait_for(processing, 10)
        if fail_fence_read:
            assert isinstance(cancelled.value.__cause__, ConnectionError)
        assert len(provider.requests) == 1
        assert failures == int(fail_fence_read)
        assert (await sessions.load(session_id)).status.value == "completed"

        current = await tasks.load_task(task_id)
        assert current is not None
        if current.lease_expires_at is not None:
            now[0] = current.lease_expires_at + timedelta(seconds=1)
        await tasks.reclaim_expired()
        replacement = TaskStoreDispatcher(tasks)
        for _ in range(3):
            assert await replacement.process_next(worker, worker_id="replacement") is None
        current = await tasks.load_task(task_id)
        assert current is not None
        assert current.status is TaskStatus.CANCELLED, (current.status, current.status_reason)
    finally:
        provider.release.set()
        if not processing.done():
            processing.cancel()
        await asyncio.gather(processing, return_exceptions=True)
        if backend == "sqlite":
            await tasks.close()
