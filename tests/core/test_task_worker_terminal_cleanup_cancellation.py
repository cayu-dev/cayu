"""Terminal task state must not let shutdown abandon a live opaque callback."""

import asyncio
import threading

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import TaskCreate, TaskStatus, complete_managed_task, run_task_worker
from cayu.tasks.memory import InMemoryTaskStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("terminal_before_cleanup", [False, True])
def test_worker_cancellation_waits_for_post_terminal_cleanup(
    backend, terminal_before_cleanup, tmp_path
):
    async def scenario():
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        task = await store.create_task(TaskCreate(type="job"))
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        callback_cancelled = []

        def external_cleanup():
            started.set()
            release.wait()
            finished.set()

        async def handler(_app, claimed, worker_id):
            if terminal_before_cleanup:
                await complete_managed_task(store, claimed, worker_id, {"ok": True})
            try:
                await asyncio.to_thread(external_cleanup)
            except asyncio.CancelledError:
                callback_cancelled.append(True)
                raise

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
            assert await asyncio.to_thread(started.wait, 5)
            # With the default 300-second lease, native heartbeats still run
            # once per second. Let one observe the completed task first.
            if terminal_before_cleanup:
                await asyncio.sleep(1.2)
            worker.cancel("shutdown")
            done, _ = await asyncio.wait({worker}, timeout=0.2)
            assert not finished.is_set(), "The external operation is deliberately still live"
            assert (bool(done), callback_cancelled) == (False, []), (
                "Worker returned and cancelled its wrapper while external cleanup still runs",
                bool(done),
                callback_cancelled,
            )
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(worker, 5)
            assert finished.is_set()
            current = await store.load_task(task.id)
            assert current.status is (
                TaskStatus.COMPLETED if terminal_before_cleanup else TaskStatus.CANCELLED
            )
        finally:
            release.set()
            await asyncio.gather(worker, return_exceptions=True)
            await asyncio.to_thread(finished.wait, 5)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
