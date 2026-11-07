"""An old handoff must not release a replacement using the same worker name."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.task_invocation_fixtures import stored_session_invocation
from tests.core.test_task_worker import _seed_interrupted_worker_handoff

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.worker import (
    TaskHandlerOutcome,
    _recover_expired_interrupted_task_handoffs,
    run_task_worker,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("reuse_worker_id", [False, True])
def test_delayed_handoff_preserves_replacement_claim(backend, reuse_worker_id, tmp_path):
    async def scenario():
        now = [datetime.now(UTC)]
        base = InMemoryTaskStore if backend == "memory" else SQLiteTaskStore

        class PreflightBarrierStore(base):
            supports_interrupted_task_handoffs = True
            verified_work_mutations_are_cancellation_quiescent = True

            def __init__(self, *args):
                super().__init__(*args, ownership_clock=lambda: now[0])
                self.block_next_read = False
                self.preflight_started = asyncio.Event()
                self.release_preflight = asyncio.Event()

            async def load_task(self, task_id):
                if self.block_next_read:
                    self.block_next_read = False
                    self.preflight_started.set()
                    await self.release_preflight.wait()
                return await super().load_task(task_id)

        store = PreflightBarrierStore(
            *(() if backend == "memory" else (tmp_path / "tasks.sqlite",))
        )
        app = CayuApp(task_store=store, enable_logging=False)
        await _seed_interrupted_worker_handoff(app, store, task_id="job", session_id="session")
        replacement_started, replacement_return = asyncio.Event(), asyncio.Event()
        replacement_stopped = asyncio.Event()

        async def original_handler(_app, task, worker_id):
            await store.attach_task(
                task.id,
                session_id="session",
                session_invocation=await stored_session_invocation(app.session_store, "session"),
                worker_id=worker_id,
                lease_expires_at=task.lease_expires_at,
            )
            store.block_next_read = True
            return TaskHandlerOutcome.SESSION_INTERRUPTED

        async def continuation(_app, _task, _worker_id):
            replacement_started.set()
            try:
                await replacement_return.wait()
            finally:
                replacement_stopped.set()
            return TaskHandlerOutcome.SESSION_INTERRUPTED

        options = dict(max_tasks=1, reclaim=False, recover_interrupted_handoffs=False)
        original = asyncio.create_task(
            run_task_worker(app, store, original_handler, worker_id="shared", **options)
        )
        replacement = None
        try:
            await asyncio.wait_for(store.preflight_started.wait(), 5)
            old_claim = await store.load_task("job")
            assert old_claim is not None and old_claim.lease_expires_at is not None
            now[0] = old_claim.lease_expires_at + timedelta(seconds=1)
            recovered = await _recover_expired_interrupted_task_handoffs(
                app, store, worker_id="scanner", after=None, limit=10, stop=None
            )
            assert recovered.recovered == 1
            replacement = asyncio.create_task(
                run_task_worker(
                    app,
                    store,
                    original_handler,
                    worker_id="shared" if reuse_worker_id else "replacement",
                    recovered_interrupted_task_handler=continuation,
                    **options,
                )
            )
            await asyncio.wait_for(replacement_started.wait(), 5)
            new_claim = await store.load_task("job")
            assert new_claim is not None and new_claim.interrupted_handoff_id is not None
            assert new_claim.lease_expires_at != old_claim.lease_expires_at
            store.release_preflight.set()
            await asyncio.wait_for(asyncio.gather(original, return_exceptions=True), 5)
            assert not replacement_stopped.is_set()
            current = await store.load_task("job")
            assert current is not None
            assert (current.worker_id, current.interrupted_handoff_id) == (
                new_claim.worker_id,
                new_claim.interrupted_handoff_id,
            ), "The old handoff released a replacement whose callback is still running"
            assert current.lease_expires_at is not None
        finally:
            store.release_preflight.set()
            replacement_return.set()
            await asyncio.gather(
                original,
                *((replacement,) if replacement is not None else ()),
                return_exceptions=True,
            )
            await app.aclose()

    asyncio.run(scenario())
