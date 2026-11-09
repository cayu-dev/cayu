"""A transient drain renewal failure must not erase a stopped retry handler's usage."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from cayu.applications import CayuApp
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import (
    TaskCreate,
    TaskRetryAttemptDisposition,
    TaskRetryAttemptReport,
    TaskRetryPolicy,
    TaskStatus,
    current_task_cancellation,
    run_task_worker,
)
from cayu.tasks.cancellation import _task_retry_cancellation_requested
from cayu.tasks.memory import InMemoryTaskStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("transient_failure", [False, True])
def test_retry_cancellation_keeps_report_after_transient_drain_heartbeat_error(
    backend, transient_failure, tmp_path
):
    async def scenario():
        base = InMemoryTaskStore if backend == "memory" else SQLiteTaskStore

        class TransientDrainHeartbeatStore(base):
            verified_work_mutations_are_cancellation_quiescent = True

            def __init__(self, *args):
                super().__init__(*args)
                self.cancellation_renewals = 0
                self.failed = False
                self.recovered = asyncio.Event()

            async def heartbeat(self, task_id, worker_id, **kwargs):
                current = await super().load_task(task_id)
                if current is not None and _task_retry_cancellation_requested(current):
                    self.cancellation_renewals += 1
                    # The first renewal tells the main heartbeat to start the
                    # cancellation drain. Fail only its first replacement renewal.
                    if transient_failure and self.cancellation_renewals == 2:
                        self.failed = True
                        raise ConnectionError("one transient drain renewal failure")
                renewed = await super().heartbeat(task_id, worker_id, **kwargs)
                if self.cancellation_renewals >= 3:
                    self.recovered.set()
                return renewed

        store = TransientDrainHeartbeatStore(
            *(() if backend == "memory" else (tmp_path / "tasks.sqlite",))
        )
        app = CayuApp(task_store=store, enable_logging=False)
        created = await store.create_task(
            TaskCreate(type="job", retry_policy=TaskRetryPolicy(max_attempts=2))
        )
        started = asyncio.Event()
        operator_error = {"code": "operator_cancelled"}

        async def handler(_app, _task, _worker_id):
            started.set()
            signal = current_task_cancellation()
            assert signal is not None
            await signal.wait()
            # Finish cooperatively after a successful renewal following the
            # transient error, well inside the still-live 30-second claim.
            await store.recovered.wait()
            return TaskRetryAttemptReport(
                idempotency_key="stopped-with-accounting",
                disposition=TaskRetryAttemptDisposition.CANCELLED,
                error={"code": "cooperatively_stopped"},
                token_count=1234,
                estimated_cost=Decimal("0.25"),
            )

        worker = asyncio.create_task(
            run_task_worker(
                app,
                store,
                handler,
                worker_id="owner",
                max_tasks=1,
                lease_seconds=30,
                reclaim=False,
                recover_interrupted_handoffs=False,
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 10)
            await store.cancel_task(created.id, operator_error)
            assert await asyncio.wait_for(worker, 10) == 1
            assert store.failed is transient_failure
            assert store.recovered.is_set()
            current = await store.load_task(created.id)
            assert current is not None and current.status is TaskStatus.CANCELLED
            assert current.error == operator_error
            assert current.retry_series is not None
            assert (
                current.retry_series.cumulative_tokens,
                current.retry_series.cumulative_estimated_cost,
            ) == (1234, Decimal("0.25")), "The drain discarded the quiescent handler's usage report"
        finally:
            store.recovered.set()
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
