"""A stalled handoff must respect the ordinary worker's disposition deadline."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from tests.core.task_invocation_fixtures import stored_session_invocation
from tests.core.test_task_worker import _seed_interrupted_worker_handoff

import cayu.tasks.worker as worker_module
from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import TaskHandlerOutcome, run_task_worker
from cayu.tasks.memory import InMemoryTaskStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("handoff", [False, True])
def test_ordinary_disposition_deadline_stops_renewing_a_stalled_handoff(
    backend, handoff, tmp_path, monkeypatch
):
    async def scenario():
        base = InMemoryTaskStore if backend == "memory" else SQLiteTaskStore

        class DelayedDispositionStore(base):
            supports_interrupted_task_handoffs = True
            verified_work_mutations_are_cancellation_quiescent = True

            def __init__(self, *args):
                super().__init__(*args)
                self.disposing = asyncio.Event()
                self.release_disposition = asyncio.Event()
                self.delay_failure_read = False

            async def release_interrupted_task_worker(self, request):
                self.disposing.set()
                # This is cancellation-cooperative, like a pending storage call.
                await self.release_disposition.wait()
                return await super().release_interrupted_task_worker(request)

            async def load_task(self, task_id):
                if self.delay_failure_read:
                    self.disposing.set()
                    await self.release_disposition.wait()
                return await super().load_task(task_id)

        store = DelayedDispositionStore(
            *(() if backend == "memory" else (tmp_path / "tasks.sqlite",))
        )
        app = CayuApp(task_store=store, enable_logging=False)
        # Keep the production max(lease_seconds, floor) rule, shrinking only the
        # five-minute floor. A one-second lease then gives a one-second deadline.
        monkeypatch.setattr(worker_module, "_MIN_TASK_DISPOSITION_DEADLINE_S", 0.05)
        await _seed_interrupted_worker_handoff(
            app, store, task_id="job", session_id="interrupted-session"
        )

        async def handler(_app, task, worker_id):
            if not handoff:
                store.delay_failure_read = True
                return None
            await store.attach_task(
                task.id,
                session_id="interrupted-session",
                session_invocation=await stored_session_invocation(
                    app.session_store, "interrupted-session"
                ),
                worker_id=worker_id,
                claim_generation=task.claim_generation,
            )
            return TaskHandlerOutcome.SESSION_INTERRUPTED

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="owner",
                lease_seconds=1,
                max_tasks=1,
                reclaim=False,
                recover_interrupted_handoffs=False,
            )
        )
        try:
            await asyncio.wait_for(store.disposing.wait(), 5)
            # Do not deliver test cancellation to the worker. After the one-
            # second disposition deadline and another full lease, its last
            # acknowledged claim must have expired, even if shutdown still
            # needs to join the outstanding store operation.
            await asyncio.sleep(2.4)
            current = await base.load_task(store, "job")
            assert current.lease_expires_at is None or current.lease_expires_at <= datetime.now(
                UTC
            ), (
                "The handoff swallowed the disposition timeout and kept its "
                "claim renewing after the handler had already stopped",
                current.lease_expires_at,
                datetime.now(UTC),
            )
            if worker.done():
                assert await worker == 1
        finally:
            store.delay_failure_read = False
            store.release_disposition.set()
            await asyncio.wait_for(asyncio.gather(worker, return_exceptions=True), 5)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
