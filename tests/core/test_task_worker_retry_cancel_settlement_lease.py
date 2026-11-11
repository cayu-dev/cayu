"""A cancelled retry attempt keeps its lease through slow final settlement."""

import asyncio
from decimal import Decimal

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskRetryAttemptDisposition,
    TaskRetryAttemptReport,
    TaskRetryPolicy,
    TaskStatus,
    current_task_cancellation,
    run_task_worker,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("settlement_delay", [0, 1.25])
def test_retry_operator_cancellation_renews_until_settlement(backend, settlement_delay, tmp_path):
    async def scenario():
        base = InMemoryTaskStore if backend == "memory" else SQLiteTaskStore

        class SlowSettlementStore(base):
            verified_work_mutations_are_cancellation_quiescent = True

            async def settle_task_retry_attempt(self, request):
                # Storage admission is slow, not unavailable. Other operations,
                # including lease renewals, are still able to commit.
                await asyncio.sleep(settlement_delay)
                return await super().settle_task_retry_attempt(request)

        store = (
            SlowSettlementStore()
            if backend == "memory"
            else SlowSettlementStore(tmp_path / "tasks.db")
        )
        app = CayuApp(task_store=store, enable_logging=False)
        created = await store.create_task(
            TaskCreate(type="job", retry_policy=TaskRetryPolicy(max_attempts=2))
        )
        ready = asyncio.Event()

        async def handler(_app, _task, _worker_id):
            signal = current_task_cancellation()
            ready.set()
            await signal.wait()
            return TaskRetryAttemptReport(
                idempotency_key="cancelled-report",
                disposition=TaskRetryAttemptDisposition.CANCELLED,
                error={"code": "handler_stopped"},
                token_count=17,
                estimated_cost=Decimal("0.25"),
            )

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="owner",
                lease_seconds=1,
                max_tasks=1,
                poll_interval_s=0.01,
                execution_grace_s=5,
                reclaim=False,
                recover_interrupted_handoffs=False,
            )
        )
        try:
            await asyncio.wait_for(ready.wait(), 5)
            await store.cancel_task(created.id, {"code": "operator"})
            outcome = (await asyncio.gather(worker, return_exceptions=True))[0]
            current = await store.load_task(created.id)
            assert current.status is TaskStatus.CANCELLED, (
                "Quiescent retry cancellation lost its lease during final settlement",
                current.status,
                current.status_reason,
                repr(outcome),
            )
            assert current.error == {"code": "operator"}
            assert current.retry_series.cumulative_tokens == 17
            assert current.retry_series.cumulative_estimated_cost == Decimal("0.25")
            assert outcome == 1
        finally:
            if not worker.done():
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
