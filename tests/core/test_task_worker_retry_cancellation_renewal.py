"""A retry attempt's cancellation fallback must not lose to its own lease renewal."""

import asyncio

from cayu.applications import CayuApp
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskRetryPolicy, TaskStatus
from cayu.tasks.retry import TaskRetryAttemptDisposition, TaskRetryAttemptReport
from cayu.tasks.worker import run_task_worker


class RenewingAfterReadStore(InMemoryTaskStore):
    """Renews the claim right after the worker reads a cancellation-requested attempt."""

    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self) -> None:
        super().__init__()
        self.settlement_refused = False
        self.renewed_after_read = False

    async def settle_task_retry_attempt(self, request):
        try:
            return await super().settle_task_retry_attempt(request)
        except Exception:
            self.settlement_refused = True
            raise

    async def load_task(self, task_id):
        task = await super().load_task(task_id)
        if (
            self.settlement_refused
            and not self.renewed_after_read
            and task is not None
            and task.status is TaskStatus.CLAIMED
        ):
            # A heartbeat commits between the fallback's read and its settlement.
            self.renewed_after_read = True
            await super().heartbeat(
                task.id,
                task.worker_id,
                claim_generation=task.claim_generation,
                extend_seconds=600,
            )
        return task


def test_cancellation_fallback_settles_after_a_renewal_moves_the_lease():
    async def scenario():
        store = RenewingAfterReadStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await store.create_task(
            TaskCreate(type="job", retry_policy=TaskRetryPolicy(max_attempts=2))
        )

        async def handler(_app, claimed, _worker_id):
            # An operator cancellation wins before the attempt's report settles.
            await store.cancel_task(claimed.id, {"code": "operator"})
            return TaskRetryAttemptReport(
                idempotency_key="completed-report",
                disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                result={"ok": True},
            )

        try:
            await run_task_worker(
                app,
                store,
                handler,
                worker_id="report-owner",
                max_tasks=1,
                reclaim=False,
                recover_interrupted_handoffs=False,
            )
            current = await store.load_task(task.id)
            assert store.renewed_after_read
            assert current.status is TaskStatus.CANCELLED, (current.status, current.status_reason)
        finally:
            await app.aclose()

    asyncio.run(scenario())
