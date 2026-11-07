"""A failure disposition must not abandon its task while the store is merely slow."""

import asyncio

from cayu.applications import CayuApp
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus
from cayu.tasks.worker import run_task_worker


class SlowFailureReadStore(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self):
        super().__init__()
        self.delay_failure_read = False
        self.delayed = False

    async def load_task(self, task_id):
        if self.delay_failure_read and not self.delayed:
            self.delayed = True
            # A slow read, not an unavailable store: heartbeats can still run.
            await asyncio.sleep(1.2)
        return await super().load_task(task_id)


def test_failure_read_retains_lease_until_disposition():
    async def scenario():
        store = SlowFailureReadStore()
        app = CayuApp(task_store=store, enable_logging=False)
        await store.create_task(TaskCreate(task_id="failed-handler", type="job"))

        async def handler(app, task, worker_id):
            store.delay_failure_read = True
            raise RuntimeError("handler failed")

        try:
            handled = await run_task_worker(
                app,
                store,
                handler,
                worker_id="failure-owner",
                lease_seconds=1,
                max_tasks=1,
                reclaim=False,
                recover_interrupted_handoffs=False,
            )
            assert handled == 1
            assert store.delayed
            task = await store.load_task("failed-handler")
            assert task is not None
            assert task.status is TaskStatus.FAILED, (task.status, task.worker_id)
            assert task.error["message"] == "handler failed"
        finally:
            await app.aclose()

    asyncio.run(scenario())
