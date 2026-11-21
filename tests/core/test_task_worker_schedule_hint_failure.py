"""Optional schedule hints must not stop task workers or their concurrency slots."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from cayu.applications import CayuApp
from cayu.runtime._durable_worker_loop import DurableWorkerMetrics
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskStatus,
    complete_managed_task,
    run_task_worker,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("concurrency", [1, 3])
@pytest.mark.parametrize("fail_observation", [False, True])
def test_task_worker_survives_schedule_hint_read_failure(
    backend, concurrency, fail_observation, monkeypatch, sqlite_resources, caplog
):
    async def scenario():
        async with sqlite_resources as resources:
            store = (
                resources.own(SQLiteTaskStore(resources.path()))
                if backend == "sqlite"
                else InMemoryTaskStore()
            )
            app = CayuApp(task_store=store, enable_logging=False)
            metrics = DurableWorkerMetrics()
            original_observe = store.next_task_schedule_wakeup
            observations = 0
            submitted = None

            async def observe(query=None):
                nonlocal observations, submitted
                observations += 1
                if observations == 1:
                    # Publish work after an empty claim, just as the hint fails.
                    submitted = await store.create_task(TaskCreate(type="job"))
                    if fail_observation:
                        error_type = (
                            ConnectionError if backend == "memory" else sqlite3.OperationalError
                        )
                        raise error_type("private store diagnostic")
                return await original_observe(query)

            async def unavailable_subscription(queries):
                return None

            async def handler(_app, task, worker_id):
                await complete_managed_task(store, task, worker_id, {"done": True})

            monkeypatch.setattr(store, "next_task_schedule_wakeup", observe)
            # Ensure recovery uses ordinary polling, not an admission notification.
            monkeypatch.setattr(store, "_task_admission_wakeup", unavailable_subscription)
            worker = asyncio.create_task(
                run_task_worker(
                    app,
                    store,
                    handler,
                    worker_id="schedule-worker",
                    concurrency=concurrency,
                    max_tasks=1,
                    poll_interval_s=0.02,
                    metrics=metrics,
                    reclaim=False,
                    recover_interrupted_handoffs=False,
                )
            )
            try:
                assert await asyncio.wait_for(worker, 10) == 1
                assert submitted is not None
                current = await store.load_task(submitted.id)
                assert current is not None and current.status is TaskStatus.COMPLETED
                assert metrics.snapshot().store_failures == int(fail_observation)
                assert "private store diagnostic" not in caplog.text
                if fail_observation:
                    assert "schedule wakeup discovery failed" in caplog.text
            finally:
                if not worker.done():
                    worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                await app.aclose()

    asyncio.run(scenario())
