"""Timeout fence writes must remain owned through application shutdown."""

import asyncio

import pytest

import cayu.tasks.worker as worker_module
from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskRetryPolicy,
    run_task_worker,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("retry_series", [False, True])
def test_shutdown_retains_store_until_timeout_fence_write_finishes(backend, retry_series, tmp_path):
    async def scenario():
        fencing, release_write = asyncio.Event(), asyncio.Event()
        base = InMemoryTaskStore if backend == "memory" else SQLiteTaskStore

        class SlowTimeoutFenceStore(base):
            verified_work_mutations_are_cancellation_quiescent = True

            def __init__(self):
                super().__init__(*((tmp_path / "tasks.sqlite",) if backend == "sqlite" else ()))
                self.closed = False
                self.writes_after_close = 0

            async def request_claimed_task_cancellation(self, *args, **kwargs):
                fencing.set()
                await release_write.wait()
                if self.closed:
                    self.writes_after_close += 1
                return await super().request_claimed_task_cancellation(*args, **kwargs)

            async def close(self):
                self.closed = True
                if backend == "sqlite":
                    await super().close()

        store = SlowTimeoutFenceStore()
        app = CayuApp(task_store=store, owned_resources=(store,), enable_logging=False)
        await store.create_task(
            TaskCreate(
                type="job",
                retry_policy=TaskRetryPolicy(max_attempts=2) if retry_series else None,
            )
        )

        async def handler(*args):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Cleanup outlasts the timeout, so the native watcher starts its fence.
                await fencing.wait()
            return None

        before = set(worker_module._HANDLER_TIME_LIMIT_WATCHERS)
        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="timeout-owner",
                lease_seconds=300,
                max_tasks=1,
                reclaim=False,
                recover_interrupted_handoffs=False,
                execution_timeout_s=0.05,
                execution_grace_s=0,
            )
        )
        try:
            await asyncio.wait_for(fencing.wait(), 5)
            # An ordinary task's disposition holds the lease lock and waits for
            # the fence. Retry timeout dispositions settle outside that lock.
            await asyncio.wait({worker}, timeout=0.2)
            watchers = set(worker_module._HANDLER_TIME_LIMIT_WATCHERS) - before
            assert any(not watcher.done() for watcher in watchers)
            outcome = await app.aclose(timeout_s=0.1)
            assert (outcome.status, store.closed) == ("incomplete", False), (
                "Shutdown released a store with a timeout-fence write still running",
                outcome.status,
                store.closed,
            )
        finally:
            release_write.set()
            await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 5)
            watchers = set(worker_module._HANDLER_TIME_LIMIT_WATCHERS) - before
            await asyncio.wait_for(asyncio.gather(*watchers, return_exceptions=True), 5)
            await app.aclose(timeout_s=2)
        assert store.writes_after_close == 0

    asyncio.run(scenario())
