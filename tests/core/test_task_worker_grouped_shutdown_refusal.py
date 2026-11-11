"""Shutdown admission refusal must not destroy a task inside a TaskGroup."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_task_worker_shutdown import _app, _run_task, _task_store

from cayu.tasks import TaskCreate, TaskStatus, run_task_worker


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("grouped", [False, True])
def test_shutdown_refused_task_group_returns_work_to_the_queue(backend, grouped, tmp_path):
    async def scenario():
        store = _task_store(backend, tmp_path)
        app = _app(store)
        reopened = None
        created = await app.create_task(TaskCreate(type="demo", title="still needs processing"))

        async def handler(owned_app, task, worker_id):
            # As in the existing shutdown tests, the app seals after the task
            # was claimed but before its first run can be admitted.
            owned_app.seal_admissions()
            if grouped:
                async with asyncio.TaskGroup() as group:
                    group.create_task(_run_task(owned_app, task, worker_id))
            else:
                await _run_task(owned_app, task, worker_id)

        try:
            assert (
                await asyncio.wait_for(
                    run_task_worker(app, store, handler, worker_id="closing-worker", max_tasks=1),
                    10,
                )
                == 1
            )
            current = await store.load_task(created.id)
            assert current is not None
            assert (current.status, current.worker_id, current.error) == (
                TaskStatus.PENDING,
                None,
                None,
            ), "A shutdown refusal wrapped by TaskGroup permanently failed undispatched work"

            # It must be processable by a new application, without an operator
            # having to repair or recreate a falsely failed task.
            reopened = _app(store)
            assert (
                await asyncio.wait_for(
                    run_task_worker(
                        reopened, store, _run_task, worker_id="new-worker", max_tasks=1
                    ),
                    10,
                )
                == 1
            )
            assert (await store.load_task(created.id)).status is TaskStatus.COMPLETED
        finally:
            await app.aclose()
            if reopened is not None:
                await reopened.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
