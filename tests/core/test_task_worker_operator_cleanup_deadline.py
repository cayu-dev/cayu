"""An operator's first interrupt must not be followed by a deadline interrupt in cleanup."""

from __future__ import annotations

import asyncio

import pytest

from cayu import CayuApp
from cayu.deadlines import current_handler_time_limit
from cayu.storage import SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskStatus,
    current_task_cancellation,
    run_task_worker,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("handler_timeout", [None, 1.7])
def test_handler_deadline_does_not_cancel_operator_cleanup_again(
    backend, handler_timeout, tmp_path
):
    async def scenario():
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        created = await store.create_task(TaskCreate(type="job"))
        entered = asyncio.Event()
        cleaning = asyncio.Event()
        release_cleanup = asyncio.Event()
        cleaned = asyncio.Event()
        cleanup_interrupted = asyncio.Event()
        first_interrupt = []

        async def handler(_app, _task, _worker_id):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                limit = current_handler_time_limit()
                first_interrupt.append(
                    (current_task_cancellation().requested, limit is not None and limit.cancelled)
                )
                cleaning.set()
                try:
                    await release_cleanup.wait()
                    cleaned.set()
                except asyncio.CancelledError:
                    cleanup_interrupted.set()
                    raise

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="owner",
                max_tasks=1,
                execution_timeout_s=handler_timeout,
                execution_grace_s=0.0,
                reclaim=False,
                recover_interrupted_handoffs=False,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await store.cancel_task(created.id, {"code": "operator_stop"})
            # The next heartbeat (~1s) delivers the first interruption before
            # the handler's 1.7s deadline. No second cancel is sent by this test.
            await asyncio.wait_for(cleaning.wait(), 5)
            assert first_interrupt == [(True, False)], "The operator must interrupt first"
            done, _ = await asyncio.wait({worker}, timeout=1.1)
            final = await store.load_task(created.id)
            assert not cleanup_interrupted.is_set(), (
                "The handler time limit interrupted cleanup already entered by "
                "the worker's operator-cancellation interrupt",
                cleaned.is_set(),
                worker in done,
                final.status,
            )
            assert not done, "The worker abandoned the handler's unfinished cleanup"
            release_cleanup.set()
            assert await asyncio.wait_for(worker, 5) == 1
            assert cleaned.is_set()
            final = await store.load_task(created.id)
            assert final.status is TaskStatus.CANCELLED
            assert final.error == {"code": "operator_stop"}
        finally:
            release_cleanup.set()
            await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 10)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
