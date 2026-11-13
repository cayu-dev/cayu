"""One worker loop claims tasks of several types, in the store's claim order."""

import asyncio

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import TaskCreate, TaskQuery, TaskStatus, complete_managed_task, run_task_worker
from cayu.tasks.memory import InMemoryTaskStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_one_worker_claims_every_listed_type_and_leaves_the_rest(backend, tmp_path):
    async def scenario():
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        for task_id, task_type in (("pass-1", "pass"), ("other-1", "other"), ("check-1", "check")):
            await store.create_task(TaskCreate(task_id=task_id, type=task_type))
        handled: list[str] = []

        async def handler(_app, task, worker_id):
            handled.append(task.id)
            await complete_managed_task(store, task, worker_id, {})

        try:
            count = await run_task_worker(
                app,
                store,
                handler,
                worker_id="one-loop",
                query=TaskQuery(types=("check", "pass")),
                max_tasks=2,
                poll_interval_s=0.01,
                recover_interrupted_handoffs=False,
            )
            assert count == 2
            assert handled == ["pass-1", "check-1"]
            other = await store.load_task("other-1")
            assert other is not None and other.status is TaskStatus.PENDING
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(asyncio.wait_for(scenario(), 30))
