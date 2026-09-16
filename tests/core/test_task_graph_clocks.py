"""Graph retry admission preserves the stores' separate clock domains."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from cayu import CayuApp
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.base import (
    InMemoryTaskStore,
    TaskCreate,
    TaskQuery,
    TaskRetryAttemptDisposition,
    TaskRetryPolicy,
    TaskRetrySettlementRequest,
    TaskStatus,
)
from cayu.tasks.graphs import TaskGraphCreate, TaskGraphEventType, TaskGraphNode
from cayu.tasks.scheduling import TaskSchedulePolicy


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("year", [2000, 2100], ids=["past", "future"])
@pytest.mark.parametrize("scheduled", [False, True], ids=["ordinary", "scheduled"])
def test_graph_retry_clock_is_independent_of_lease_clock(
    backend, year, scheduled, request, tmp_path
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None
    evidence_now = datetime(year, 1, 1, tzinfo=UTC)
    admitted_at = evidence_now
    ownership_now = datetime.now(UTC)
    identity = uuid4().hex

    def open_store():
        if backend == "memory":
            return InMemoryTaskStore(
                clock=lambda: evidence_now, ownership_clock=lambda: ownership_now
            )
        if backend == "sqlite":
            return SQLiteTaskStore(
                tmp_path / "graph-clocks.sqlite",
                clock=lambda: evidence_now,
                ownership_clock=lambda: ownership_now,
            )
        return PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE, clock=lambda: evidence_now)

    def root(task_id):
        return TaskCreate(
            task_id=task_id,
            type=task_id,
            retry_policy=TaskRetryPolicy(max_attempts=2, max_elapsed_seconds=60),
            available_at=admitted_at if scheduled else None,
            schedule_policy=TaskSchedulePolicy() if scheduled else None,
        )

    def graph(suffix):
        graph_id = f"{identity}-{suffix}"
        return TaskGraphCreate(
            graph_id=graph_id,
            nodes=(
                TaskGraphNode(task=root(graph_id)),
                TaskGraphNode(
                    task=TaskCreate(task_id=f"{graph_id}-child", type=f"{graph_id}-child"),
                    prerequisite_task_ids=(graph_id,),
                ),
            ),
        )

    async def run():
        nonlocal evidence_now
        store = open_store()
        try:
            app = CayuApp(task_store=store, enable_logging=False)
            successful, expiring = graph("success"), graph("expire")
            receipts = [await app.create_task_graph(value) for value in (successful, expiring)]
            control = await store.create_task(root(f"{identity}-control"))
            assert control.retry_series is not None
            for value, receipt in zip((successful, expiring), receipts, strict=True):
                task = await store.load_task(value.graph_id)
                assert task is not None and task.retry_series is not None
                assert (
                    task.retry_series.started_at == control.retry_series.started_at == admitted_at
                )
                assert task.retry_series.elapsed_deadline == admitted_at + timedelta(seconds=60)
                assert receipt.accepted_at == admitted_at
                if scheduled:
                    assert task.created_at == control.created_at == admitted_at
                events = await app.list_task_graph_events(value.graph_id)
                assert all(event.occurred_at == admitted_at for event in events)

            # Fresh persistent instances must reconstruct the same retry authority.
            if backend != "memory":
                await store.close()
                store = open_store()
                app = CayuApp(task_store=store, enable_logging=False)
            before_claim = datetime.now(UTC)
            claimed = await store.claim_task("worker", TaskQuery(type=successful.graph_id))
            assert claimed is not None and claimed.retry_series is not None
            assert claimed.lease_expires_at is not None
            if backend == "postgres":
                assert before_claim + timedelta(seconds=300) <= claimed.lease_expires_at
                assert claimed.lease_expires_at <= datetime.now(UTC) + timedelta(seconds=300)
            else:
                assert claimed.lease_expires_at == ownership_now + timedelta(seconds=300)
            await store.settle_task_retry_attempt(
                TaskRetrySettlementRequest(
                    task_id=claimed.id,
                    worker_id="worker",
                    lease_expires_at=claimed.lease_expires_at,
                    causal_budget_id=claimed.retry_series.causal_budget_id,
                    idempotency_key="success",
                    disposition=TaskRetryAttemptDisposition.SUCCEEDED,
                    result={"ok": True},
                )
            )
            child = await store.claim_task("child-worker", TaskQuery(type=f"{claimed.id}-child"))
            assert child is not None
            await store.complete_task(
                child.id,
                {},
                worker_id="child-worker",
                lease_expires_at=child.lease_expires_at,
            )

            # The retry boundary is inclusive and independent of the still-valid
            # physical lease clock. Expiry must atomically skip the dependent.
            evidence_now += timedelta(seconds=60)
            assert await store.claim_task("late-worker", TaskQuery(type=expiring.graph_id)) is None
            expired = await store.load_task(expiring.graph_id)
            skipped = await store.load_task(f"{expiring.graph_id}-child")
            assert expired is not None and expired.status is TaskStatus.FAILED
            assert skipped is not None and skipped.status is TaskStatus.DEPENDENCY_SKIPPED
            events = await app.list_task_graph_events(expiring.graph_id)
            assert sum(event.type is TaskGraphEventType.TERMINAL for event in events) == 1
            assert sum(event.type is TaskGraphEventType.SKIPPED for event in events) == 1
            snapshot = await app.load_task_graph(expiring.graph_id)
            if backend != "memory":
                await store.close()
                store = open_store()
                app = CayuApp(task_store=store, enable_logging=False)
            for value, receipt in zip((successful, expiring), receipts, strict=True):
                assert await app.create_task_graph(value) == receipt
            assert await app.load_task_graph(expiring.graph_id) == snapshot
            assert await app.list_task_graph_events(expiring.graph_id) == events
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(run())
