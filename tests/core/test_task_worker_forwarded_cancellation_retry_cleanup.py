"""A forwarded operator interrupt is the one interrupt a retry attempt's handler gets.

An outer handler runs a nested worker whose task has a retry policy. When the
outer task is cancelled, the nested supervisor forwards the interrupt to its
handler, which starts its cleanup. A cancellation of the nested task itself that
arrives during that cleanup must wait for it, not cancel the handler again.
"""

from __future__ import annotations

import asyncio

import pytest

from cayu import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    complete_managed_task,
    run_task_worker,
)
from cayu.tasks.records import TaskRetryPolicy
from cayu.tasks.retry import TaskRetryAttemptDisposition, TaskRetryAttemptReport

# Long enough for the nested attempt's heartbeat to see its own cancellation
# request and for a second interrupt, if one were sent, to land.
_SECOND_INTERRUPT_WINDOW_SECONDS = 3.0


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_nested_retry_cancellation_does_not_interrupt_forwarded_cleanup(backend, tmp_path):
    store = (
        InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.sqlite")
    )
    app = CayuApp(task_store=store, enable_logging=False)
    root = await store.create_task(TaskCreate(type="root"))
    nested = await store.create_task(
        TaskCreate(type="leaf", retry_policy=TaskRetryPolicy(max_attempts=2))
    )
    entered, release = asyncio.Event(), asyncio.Event()
    cleanup_started, release_cleanup = asyncio.Event(), asyncio.Event()
    cleanup_completed, cleanup_interrupted = asyncio.Event(), asyncio.Event()

    async def work(kind, handler):
        return await run_task_worker(
            app,
            store,
            handler,
            worker_id=kind,
            query=TaskQuery(type=kind),
            max_tasks=1,
            reclaim=False,
            recover_interrupted_handoffs=False,
            execution_grace_s=0,
        )

    async def leaf(_app, _task, _worker_id):
        entered.set()
        try:
            await release.wait()
            return TaskRetryAttemptReport(
                idempotency_key="leaf-report",
                disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                result={"done": True},
            )
        finally:
            cleanup_started.set()
            try:
                await release_cleanup.wait()
                cleanup_completed.set()
            except asyncio.CancelledError:
                cleanup_interrupted.set()
                raise

    async def outer(_app, task, worker_id):
        await work("leaf", leaf)
        await complete_managed_task(store, task, worker_id, {"done": True})

    worker = asyncio.create_task(work("root", outer))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await store.cancel_task(root.id, {"code": "operator_cancelled"})
        await asyncio.wait_for(cleanup_started.wait(), 5)

        await store.cancel_task(nested.id, {"code": "operator_cancelled"})
        done, _ = await asyncio.wait({worker}, timeout=_SECOND_INTERRUPT_WINDOW_SECONDS)
        assert not cleanup_interrupted.is_set()
        assert not done

        release_cleanup.set()
        await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 10)
        assert cleanup_completed.is_set() and not cleanup_interrupted.is_set()
        assert (await store.load_task(root.id)).status is TaskStatus.CANCELLED
        assert (await store.load_task(nested.id)).status is TaskStatus.CANCELLED
    finally:
        release.set()
        release_cleanup.set()
        await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 10)
        await app.aclose()
        if backend == "sqlite":
            await store.close()
