"""A handler's deliberate task hold must not look like lease loss to its worker."""

import asyncio

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import TaskClaimLost, TaskCreate, TaskStatus, complete_managed_task, run_task_worker
from cayu.tasks.memory import InMemoryTaskStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    ("hold", "held_status"),
    [
        ("pause_task", TaskStatus.PAUSED),
        ("block_task", TaskStatus.BLOCKED),
        ("mark_task_needs_attention", TaskStatus.NEEDS_ATTENTION),
    ],
)
def test_worker_continues_when_heartbeat_observes_handler_hold(
    backend, hold, held_status, tmp_path
):
    async def scenario():
        heartbeat_saw_hold = asyncio.Event()
        base = InMemoryTaskStore if backend == "memory" else SQLiteTaskStore

        class ObservedStore(base):
            verified_work_mutations_are_cancellation_quiescent = True

            async def heartbeat(self, *args, **kwargs):
                try:
                    return await super().heartbeat(*args, **kwargs)
                except TaskClaimLost:
                    heartbeat_saw_hold.set()
                    raise

        store = ObservedStore(*(() if backend == "memory" else (tmp_path / "tasks.db",)))
        app = CayuApp(task_store=store, enable_logging=False)
        first = await store.create_task(TaskCreate(type="job"))
        second = await store.create_task(TaskCreate(type="job"))
        handled = []

        async def handler(_app, task, worker_id):
            handled.append(task.id)
            if task.id == first.id:
                await getattr(store, hold)(task.id, reason="awaiting an external decision")
                # Cleanup after the hold outlasts a heartbeat interval.
                await asyncio.wait_for(heartbeat_saw_hold.wait(), 5)
            else:
                await complete_managed_task(store, task, worker_id, {"ok": True})

        try:
            count = await asyncio.wait_for(
                run_task_worker(
                    app,
                    store,
                    handler,
                    worker_id="owner",
                    lease_seconds=1,
                    max_tasks=2,
                    poll_interval_s=0.01,
                    reclaim=False,
                    recover_interrupted_handoffs=False,
                ),
                10,
            )
            assert count == 2
            assert handled == [first.id, second.id]
            assert (await store.load_task(first.id)).status is held_status
            assert (await store.load_task(second.id)).status is TaskStatus.COMPLETED
        finally:
            await app.aclose()

    asyncio.run(scenario())
