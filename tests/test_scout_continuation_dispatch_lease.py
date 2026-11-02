"""Continuations must still own a live lease after pre-dispatch event reads."""

import asyncio
import contextlib
from datetime import UTC, datetime

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskClaimLost
from cayu.tasks.worker import TaskHandlerOutcome, run_task_worker
from tests.core.test_task_worker import _seed_receipt_backed_continuation


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_continuation_never_dispatches_with_expired_lease(backend, tmp_path, monkeypatch):
    async def scenario():
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        await _seed_receipt_backed_continuation(
            app, store, task_id="continued", session_id="paused"
        )
        refresh = store.refresh_task_retry_key
        delayed = False
        dispatched_expired = []

        async def slow_refresh(key):
            nonlocal delayed
            if not delayed:
                delayed = True
                # Reads can queue behind a database lock while no heartbeat exists yet.
                await asyncio.sleep(1.1)
            return await refresh(key)

        monkeypatch.setattr(store, "refresh_task_retry_key", slow_refresh)

        async def handler(app, task, worker_id):
            current = await store.load_task(task.id)
            dispatched_expired.append(
                current.worker_id != worker_id
                or current.lease_expires_at is None
                or current.lease_expires_at <= datetime.now(UTC)
            )
            return TaskHandlerOutcome.SESSION_INTERRUPTED

        try:
            # Failing closed before dispatch is also safe; running work first is not.
            with contextlib.suppress(TaskClaimLost):
                await run_task_worker(
                    app,
                    store,
                    handler,
                    worker_id="worker",
                    lease_seconds=1,
                    max_tasks=1,
                    reclaim=False,
                    recover_interrupted_handoffs=False,
                    recovered_interrupted_task_handler=handler,
                )
            assert delayed
            assert not any(dispatched_expired), dispatched_expired
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
