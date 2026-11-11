"""A cancellation that wins the retry shutdown release must still be settled."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskRetryPolicy, TaskStatus, run_task_worker


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("operator_cancels", [False, True])
def test_retry_shutdown_settles_cancellation_winning_unreported_release(
    backend, operator_cancels, tmp_path, monkeypatch
):
    async def scenario():
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        created = await store.create_task(
            TaskCreate(type="job", retry_policy=TaskRetryPolicy(max_attempts=2))
        )
        started = asyncio.Event()
        finish_handler = asyncio.Event()
        handler_stopped = asyncio.Event()
        release_started = asyncio.Event()
        allow_release = asyncio.Event()
        original_release = store.release_task

        async def delayed_release(*args, **kwargs):
            # The shutdown owner has joined the callback and stopped its
            # heartbeat, but the release transaction has not acquired storage.
            release_started.set()
            await allow_release.wait()
            return await original_release(*args, **kwargs)

        monkeypatch.setattr(store, "release_task", delayed_release)

        async def handler(_app, _task, _worker_id):
            started.set()
            await finish_handler.wait()
            handler_stopped.set()
            # A clean unreported attempt is normally released on worker shutdown.
            return None

        worker = asyncio.create_task(
            run_task_worker(app, store, handler, worker_id="owner", max_tasks=1)
        )
        try:
            await asyncio.wait_for(started.wait(), 10)
            worker.cancel("worker shutdown")
            await asyncio.sleep(0)
            finish_handler.set()
            await asyncio.wait_for(release_started.wait(), 10)
            assert handler_stopped.is_set()
            if operator_cancels:
                requested = await store.cancel_task(created.id, {"code": "operator_stop"})
                assert requested.status_reason == "retry_cancellation_requested"
                assert requested.lease_expires_at > datetime.now(UTC)
            allow_release.set()
            with pytest.raises(asyncio.CancelledError, match="worker shutdown"):
                await asyncio.wait_for(worker, 10)

            current = await store.load_task(created.id)
            expected = TaskStatus.CANCELLED if operator_cancels else TaskStatus.PENDING
            assert current.status is expected, (
                "Shutdown abandoned the cancellation that defeated its clean release",
                current.status,
                current.status_reason,
                current.worker_id,
            )
            assert current.worker_id is None
            if operator_cancels:
                assert current.error == {"code": "operator_stop"}
                assert current.retry_series.disposition.value == "cancelled"
        finally:
            finish_handler.set()
            allow_release.set()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
