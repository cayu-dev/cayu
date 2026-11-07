"""Reliable retry settlement must not lose to its own lease renewals."""

import asyncio

from cayu.applications import CayuApp
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskClaimLost, TaskRetryPolicy, TaskStatus
from cayu.tasks.retry import TaskRetryAttemptDisposition, TaskRetryAttemptReport
from cayu.tasks.worker import run_task_worker


class SlowRetrySettlementStore(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    async def settle_task_retry_attempt(self, request):
        # The store is healthy, but each request takes longer to reach its
        # transaction than the worker's one-second heartbeat interval.
        await asyncio.sleep(1.1)
        return await super().settle_task_retry_attempt(request)


def test_retry_report_settles_despite_one_second_store_latency():
    async def scenario():
        store = SlowRetrySettlementStore()
        app = CayuApp(task_store=store, enable_logging=False)
        task = await store.create_task(
            TaskCreate(type="job", retry_policy=TaskRetryPolicy(max_attempts=2))
        )

        async def handler(_app, _task, _worker_id):
            return TaskRetryAttemptReport(
                idempotency_key="completed-report",
                disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                result={"ok": True},
                token_count=17,
            )

        failure = None
        try:
            try:
                # Exercise the default 300-second lease, not a short-lease setup.
                await run_task_worker(
                    app,
                    store,
                    handler,
                    worker_id="report-owner",
                    max_tasks=1,
                    reclaim=False,
                    recover_interrupted_handoffs=False,
                )
            except TaskClaimLost as exc:
                failure = exc
            current = await store.load_task(task.id)
            assert current.status is TaskStatus.COMPLETED, (current.status, str(failure))
            assert current.result == {"ok": True}
            assert current.retry_series.cumulative_tokens == 17
        finally:
            await app.aclose()

    asyncio.run(scenario())
