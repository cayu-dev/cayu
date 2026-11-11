"""A shutdown-refused task still settles an operator cancellation that wins release."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_task_worker_shutdown import _app, _run_task, _task_store

from cayu.tasks import TaskCreate, TaskStatus, run_task_worker


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("cancel_during_release", [False, True])
def test_shutdown_refusal_settles_cancellation_before_returning(
    backend, cancel_during_release, tmp_path, monkeypatch
):
    async def scenario():
        store = _task_store(backend, tmp_path)
        app = _app(store)
        created = await app.create_task(TaskCreate(type="demo", title="not yet processed"))
        release_entered = asyncio.Event()
        release_allowed = asyncio.Event()
        original_release = store.release_task
        operator_error = {"code": "operator_cancelled"}

        async def release_task(task_id, *args, **kwargs):
            # Pause between the worker's read of its unattached claim and the
            # real store transaction, without changing either store's logic.
            if cancel_during_release and task_id == created.id:
                release_entered.set()
                await release_allowed.wait()
            return await original_release(task_id, *args, **kwargs)

        monkeypatch.setattr(store, "release_task", release_task)

        async def handler(owned_app, task, worker_id):
            if not cancel_during_release:
                # Control: cancellation already visible at refusal readback.
                await store.cancel_task(task.id, operator_error)
            owned_app.seal_admissions()
            await _run_task(owned_app, task, worker_id)

        worker = asyncio.create_task(
            run_task_worker(app, store, handler, worker_id="closing-owner", max_tasks=1)
        )
        try:
            if cancel_during_release:
                await asyncio.wait_for(release_entered.wait(), 5)
                current = await store.load_task(created.id)
                assert current.status is TaskStatus.CLAIMED
                assert current.started_at is not None
                assert current.session_id is None
                # This is an accepted cancellation of a still-owned claim.
                requested = await store.cancel_task(created.id, operator_error)
                assert requested.status_reason == "cancellation_requested"
                release_allowed.set()

            assert await asyncio.wait_for(worker, 5) == 1
            current = await store.load_task(created.id)
            assert current.status is TaskStatus.CANCELLED, (
                "The closing worker returned without settling the cancellation "
                "that defeated its shutdown-refusal release",
                current.status,
                current.status_reason,
                current.worker_id,
            )
            assert current.worker_id is None
            assert current.error == operator_error
        finally:
            release_allowed.set()
            await asyncio.gather(worker, return_exceptions=True)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
