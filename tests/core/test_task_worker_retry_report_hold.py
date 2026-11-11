"""A retry report returning after an operator hold must not stop the worker."""

from __future__ import annotations

import asyncio

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskClaimLost,
    TaskCreate,
    TaskRetryAttemptDisposition,
    TaskRetryAttemptReport,
    TaskRetryPolicy,
    TaskStatus,
    complete_managed_task,
    run_task_worker,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("pause", [False, True])
def test_operator_hold_of_returning_retry_does_not_abort_worker(
    backend, pause, tmp_path, monkeypatch
):
    async def scenario():
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        started, finish = asyncio.Event(), asyncio.Event()
        heartbeat_saw_hold = asyncio.Event()
        heartbeat = store.heartbeat

        async def observe_heartbeat(*args, **kwargs):
            try:
                return await heartbeat(*args, **kwargs)
            except TaskClaimLost:
                heartbeat_saw_hold.set()
                raise

        monkeypatch.setattr(store, "heartbeat", observe_heartbeat)
        seen = []
        first = await store.create_task(
            TaskCreate(
                task_id="retry-job",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2),
            )
        )
        second = await store.create_task(TaskCreate(task_id="next-job", type="job"))

        async def handler(_app, task, worker_id):
            seen.append(task.id)
            if task.id == first.id:
                started.set()
                await finish.wait()
                return TaskRetryAttemptReport(
                    idempotency_key="finished-attempt",
                    disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                    result={"ok": True},
                )
            await complete_managed_task(store, task, worker_id, {"ok": True})

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="owner",
                max_tasks=2,
                lease_seconds=3,
                reclaim=False,
                recover_interrupted_handoffs=False,
                poll_interval_s=0.01,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 5)
            if pause:
                await store.pause_task(first.id, reason="operator paused this job")
                # The heartbeat already recognizes this intentional ownership
                # release. The failure is in later report disposition, not the
                # heartbeat's handling of a held claim.
                await asyncio.wait_for(heartbeat_saw_hold.wait(), 5)
            finish.set()
            done, _ = await asyncio.wait({worker}, timeout=5)
            assert done, "The worker failed to drain"
            result = (await asyncio.gather(worker, return_exceptions=True))[0]
            first_state = await store.load_task(first.id)
            second_state = await store.load_task(second.id)
            assert result == 2, (
                "Retry-report disposition aborted after a valid operator hold",
                repr(result),
                first_state.status,
                second_state.status,
            )
            assert seen == [first.id, second.id]
            assert (await store.load_task(first.id)).status is (
                TaskStatus.PAUSED if pause else TaskStatus.COMPLETED
            )
            assert (await store.load_task(second.id)).status is TaskStatus.COMPLETED
        finally:
            finish.set()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
