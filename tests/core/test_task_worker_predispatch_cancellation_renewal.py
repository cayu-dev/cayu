"""A background renewal must not invalidate pre-dispatch cancellation settlement."""

import asyncio

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskClaimLost, TaskRetryPolicy, TaskStatus
from cayu.tasks.worker import run_task_worker


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("retry_series", [False, True])
def test_predispatch_cancellation_survives_renewal(backend, retry_series, tmp_path):
    async def scenario():
        base = InMemoryTaskStore if backend == "memory" else SQLiteTaskStore

        class SlowCancellationStore(base):
            verified_work_mutations_are_cancellation_quiescent = True
            heartbeat_calls = 0
            delayed = False

            async def heartbeat(self, *args, **kwargs):
                self.heartbeat_calls += 1
                renewed = await super().heartbeat(*args, **kwargs)
                if self.heartbeat_calls == 1:
                    return await self.cancel_task(renewed.id, {"code": "operator_cancel"})
                return renewed

            async def delay_once(self):
                if not self.delayed:
                    self.delayed = True
                    # Far shorter than the live lease, but longer than the
                    # worker's one-second maximum heartbeat interval.
                    await asyncio.sleep(1.1)

            async def terminalize_task(self, request):
                await self.delay_once()
                return await super().terminalize_task(request)

            async def settle_task_retry_attempt(self, request):
                await self.delay_once()
                return await super().settle_task_retry_attempt(request)

        store = (
            SlowCancellationStore()
            if backend == "memory"
            else SlowCancellationStore(tmp_path / "tasks.sqlite")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        created = await store.create_task(
            TaskCreate(
                task_id="cancel-before-dispatch",
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2) if retry_series else None,
            )
        )

        async def handler(*args):
            pytest.fail("Cancellation won before callback dispatch")

        try:
            failure = None
            handled = None
            try:
                handled = await run_task_worker(
                    app,
                    store,
                    handler,
                    worker_id="cancel-owner",
                    lease_seconds=30,
                    max_tasks=1,
                    reclaim=False,
                    recover_interrupted_handoffs=False,
                )
            except TaskClaimLost as exc:
                failure = exc
            assert store.delayed
            current = await store.load_task(created.id)
            assert current.status is TaskStatus.CANCELLED, (
                current.status,
                current.status_reason,
                str(failure),
            )
            assert failure is None
            assert handled == 1
            assert current.worker_id is None
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
