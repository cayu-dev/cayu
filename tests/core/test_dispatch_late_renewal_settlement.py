"""Late renewal acknowledgements must not orphan naturally drained dispatches."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from tests.core.test_dispatch import _dispatch_request, _SecretFreeDispatchRuntime

from cayu.events import Event, EventType
from cayu.tasks.dispatch import DispatchStatus, TaskStoreDispatcher
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus


async def _scenario() -> None:
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    renewal_committed = asyncio.Event()
    release_acknowledgement = asyncio.Event()
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()

    class LateAcknowledgementStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def heartbeat(self, *args, **kwargs):
            now[0] += timedelta(milliseconds=300)
            renewed = await super().heartbeat(*args, **kwargs)
            renewal_committed.set()
            await release_acknowledgement.wait()
            return renewed

    class BlockingRuntime(_SecretFreeDispatchRuntime):
        async def dispatch_inline(self, request):
            dispatch_started.set()
            await release_dispatch.wait()
            yield Event(type=EventType.SESSION_COMPLETED, session_id=request.session_id)

    tasks = LateAcknowledgementStore(ownership_clock=lambda: now[0])
    dispatcher = TaskStoreDispatcher(tasks, lease_seconds=1)
    runtime = BlockingRuntime()
    submitted = await dispatcher.submit(runtime, _dispatch_request("late-session", "late-ack"))
    task_id = submitted.metadata["queue_task_id"]
    processing = asyncio.create_task(dispatcher.process_next(runtime, worker_id="owner"))
    try:
        await asyncio.wait_for(dispatch_started.wait(), 5)
        await asyncio.wait_for(renewal_committed.wait(), 5)
        # The store has renewed, but its acknowledgement arrives after the
        # dispatcher's last positively known local deadline (2/3 second).
        await asyncio.sleep(0.8)
        release_acknowledgement.set()
        current = await tasks.load_task(task_id)
        assert current is not None and current.lease_expires_at is not None
        now[0] = current.lease_expires_at + timedelta(seconds=1)
        await tasks.reclaim_expired()
        release_dispatch.set()
        outcome = (await asyncio.wait_for(asyncio.gather(processing, return_exceptions=True), 5))[0]
        assert runtime._test_terminal_status is DispatchStatus.COMPLETED
        terminal = await tasks.load_task(task_id)
        assert terminal is not None
        assert terminal.status is TaskStatus.CANCELLED, (terminal.status_reason, repr(outcome))
    finally:
        release_acknowledgement.set()
        release_dispatch.set()
        if not processing.done():
            processing.cancel()
        await asyncio.gather(processing, return_exceptions=True)


def test_committed_late_renewal_still_settles_drained_dispatch() -> None:
    asyncio.run(_scenario())
