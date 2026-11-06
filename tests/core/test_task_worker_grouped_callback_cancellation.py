"""Grouped callback cancellation must settle the claim just like bare cancellation."""

import asyncio

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import TaskCreate, TaskStatus
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.worker import run_task_worker


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("grouped", [False, True])
def test_callback_cancellation_with_cleanup_failure_does_not_strand_claim(
    backend, grouped, tmp_path
):
    async def scenario():
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        created = await store.create_task(TaskCreate(type="job"))
        stopped = asyncio.Event()

        async def handler(_app, _task, _worker_id):
            try:
                cancellation = asyncio.CancelledError("provider operation cancelled")
                if grouped:
                    raise BaseExceptionGroup(
                        "operation and cleanup failed",
                        [cancellation, RuntimeError("provider cleanup failed")],
                    )
                raise cancellation
            finally:
                stopped.set()

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="owner",
                max_tasks=1,
                reclaim=False,
                recover_interrupted_handoffs=False,
            )
        )
        try:
            await asyncio.wait_for(asyncio.wait({worker}), 10)
            # Observe either propagated cancellation shape without imposing a
            # public exception-type contract. The callback is positively stopped.
            await asyncio.gather(worker, return_exceptions=True)
            assert stopped.is_set()
            current = await store.load_task(created.id)
            assert current.status in {TaskStatus.CANCELLED, TaskStatus.FAILED}, (
                current.status,
                current.worker_id,
                current.status_reason,
            )
            assert current.worker_id is None
            if grouped:
                # Raised by the handler beside a cleanup failure, not delivered to
                # the worker: not a worker-owner cancellation (which permits restart).
                evidence = repr((current.error, current.status_payload))
                assert "task_handler_cancelled" in evidence, evidence
        finally:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
