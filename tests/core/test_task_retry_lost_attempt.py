"""A retry attempt whose worker was lost can be settled as a retryable failure."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks import TaskCreate, retry_lost_task_attempt
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import Task, TaskRetryPolicy, TaskRetrySeriesDisposition, TaskStatus
from cayu.tasks.store import TaskStore
from cayu.tasks.terminalization import TaskTerminalizationConflict

LEASE_SECONDS = 30
EVIDENCE = {
    "validator_id": "service.single-worker",
    "validator_version": "1",
    "evidence_id": "worker-a-gone",
}
BACKENDS = ["memory", "sqlite", "postgres"]


class _Clock:
    def __init__(self) -> None:
        self.value = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=1)

    def __call__(self) -> datetime:
        return self.value


def _store(backend: str, tmp_path: Path, clock: _Clock, request: pytest.FixtureRequest):
    if backend == "memory":
        return InMemoryTaskStore(clock=clock, ownership_clock=clock)
    if backend == "sqlite":
        return SQLiteTaskStore(tmp_path / "tasks.db", clock=clock, ownership_clock=clock)
    from cayu.storage import PostgresTaskStore

    return PostgresTaskStore(
        request.getfixturevalue("postgres_dsn"), clock=clock, schema_mode=SchemaMode.MIGRATE
    )


async def _close(store: TaskStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


async def _lose_worker(store: TaskStore, clock: _Clock, task_type: str) -> Task:
    # PostgreSQL leases run on the database clock, so let a short lease really expire.
    real_time = not isinstance(store, InMemoryTaskStore | SQLiteTaskStore)
    claimed = await store.claim_task(
        "worker-a", TaskQuery(type=task_type), lease_seconds=1 if real_time else LEASE_SECONDS
    )
    assert claimed is not None and claimed.lease_expires_at is not None
    await store.mark_claimed_task_execution_started(
        claimed.id, "worker-a", claimed.lease_expires_at
    )
    if real_time:
        await asyncio.sleep(1.5)
    clock.value = claimed.lease_expires_at + timedelta(seconds=1)
    await store.reclaim_expired(query=TaskQuery(type=task_type))
    fenced = await store.load_task(claimed.id)
    assert fenced is not None and fenced.status_reason == "retry_cancellation_requested"
    return fenced


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_lost_attempt_queues_the_next_and_unknown_usage_ends_a_capped_series(
    backend: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(backend, tmp_path, clock, request)
        await store.create_task(
            TaskCreate(
                task_id=f"scan-{backend}",
                type=f"scan-{backend}",
                retry_policy=TaskRetryPolicy(max_attempts=3, initial_backoff_seconds=0),
            )
        )
        fenced = await _lose_worker(store, clock, f"scan-{backend}")
        settled = await retry_lost_task_attempt(store, fenced, **EVIDENCE)
        assert settled.task.status is TaskStatus.FAILED
        assert settled.task.retry_series is not None
        assert settled.task.retry_series.disposition is TaskRetrySeriesDisposition.RETRY_SCHEDULED
        assert settled.task.error is not None and settled.task.error["usage_known"] is False
        successor = settled.successor
        assert successor is not None and successor.retry_series is not None
        assert successor.retry_series.attempt == 2
        stored = await store.load_task(successor.id)
        assert stored is not None and stored.status is TaskStatus.PENDING
        # Repeating the call returns the same settlement.
        again = await retry_lost_task_attempt(store, fenced, **EVIDENCE)
        assert again.successor is not None and again.successor.id == successor.id

        # A token cap: unknown usage counts as the rest of the budget, so it ends.
        await store.create_task(
            TaskCreate(
                task_id=f"capped-{backend}",
                type=f"capped-{backend}",
                retry_policy=TaskRetryPolicy(max_attempts=3, max_total_tokens=1000),
            )
        )
        capped = await _lose_worker(store, clock, f"capped-{backend}")
        ended = await retry_lost_task_attempt(store, capped, **EVIDENCE)
        assert ended.successor is None
        assert ended.task.retry_series is not None
        assert ended.task.retry_series.disposition is TaskRetrySeriesDisposition.TOKENS_EXHAUSTED
        assert ended.task.retry_series.cumulative_tokens == 1000
        await _close(store)

    asyncio.run(scenario())


def test_known_usage_is_charged_and_the_series_goes_on() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = InMemoryTaskStore(clock=clock, ownership_clock=clock)
        await store.create_task(
            TaskCreate(
                task_id="known",
                type="known",
                retry_policy=TaskRetryPolicy(
                    max_attempts=3,
                    max_total_tokens=1000,
                    max_estimated_cost=Decimal("5"),
                    initial_backoff_seconds=0,
                ),
            )
        )
        fenced = await _lose_worker(store, clock, "known")
        settled = await retry_lost_task_attempt(
            store, fenced, **EVIDENCE, token_count=200, estimated_cost=Decimal("1.5")
        )
        assert settled.successor is not None
        series = settled.successor.retry_series
        assert series is not None
        assert (series.cumulative_tokens, series.cumulative_estimated_cost) == (
            200,
            Decimal("1.5"),
        )
        assert settled.task.error is not None and settled.task.error["usage_known"] is True

    asyncio.run(scenario())


def test_an_attempt_someone_asked_to_cancel_is_not_retried() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = InMemoryTaskStore(clock=clock, ownership_clock=clock)
        await store.create_task(
            TaskCreate(task_id="asked", type="asked", retry_policy=TaskRetryPolicy(max_attempts=3))
        )
        claimed = await store.claim_task("worker-a", TaskQuery(type="asked"), lease_seconds=30)
        assert claimed is not None and claimed.lease_expires_at is not None
        await store.mark_claimed_task_execution_started(
            claimed.id, "worker-a", claimed.lease_expires_at
        )
        await store.cancel_task(claimed.id)
        clock.value = claimed.lease_expires_at + timedelta(seconds=1)
        fenced = await store.load_task(claimed.id)
        assert fenced is not None
        with pytest.raises(TaskTerminalizationConflict):
            await retry_lost_task_attempt(store, fenced, **EVIDENCE)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_a_cancellation_asked_for_after_the_worker_was_lost_is_kept(
    backend: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = _store(backend, tmp_path, clock, request)
        await store.create_task(
            TaskCreate(task_id="late", type="late", retry_policy=TaskRetryPolicy(max_attempts=3))
        )
        await _lose_worker(store, clock, "late")
        await store.cancel_task("late")
        fenced = await store.load_task("late")
        assert fenced is not None and fenced.status_reason == "retry_cancellation_requested"
        with pytest.raises(TaskTerminalizationConflict):
            await retry_lost_task_attempt(store, fenced, **EVIDENCE)
        await _close(store)

    asyncio.run(scenario())


def test_usage_is_known_only_when_both_figures_are_given() -> None:
    async def scenario() -> None:
        clock = _Clock()
        store = InMemoryTaskStore(clock=clock, ownership_clock=clock)
        await store.create_task(
            TaskCreate(
                task_id="half",
                type="half",
                retry_policy=TaskRetryPolicy(max_attempts=3, initial_backoff_seconds=0),
            )
        )
        fenced = await _lose_worker(store, clock, "half")
        settled = await retry_lost_task_attempt(store, fenced, **EVIDENCE, token_count=10)
        assert settled.task.error is not None and settled.task.error["usage_known"] is False
        assert settled.successor is not None

    asyncio.run(scenario())


def test_a_lost_attempt_of_an_event_that_already_succeeded_is_cancelled() -> None:
    from cayu.tasks._cancellation import _reconciled_task_retry_cancellation
    from cayu.tasks.cancellation import prepare_task_retry_cancellation_reconciliation

    class _Capture:
        async def reconcile_task_retry_cancellation(self, request):
            return request

    async def scenario() -> None:
        clock = _Clock()
        store = InMemoryTaskStore(clock=clock, ownership_clock=clock)
        await store.create_task(
            TaskCreate(task_id="done", type="done", retry_policy=TaskRetryPolicy(max_attempts=3))
        )
        fenced = await _lose_worker(store, clock, "done")
        request = await retry_lost_task_attempt(_Capture(), fenced, **EVIDENCE)  # type: ignore[arg-type]
        request, request_sha256 = prepare_task_retry_cancellation_reconciliation(request)
        # Another task with the same event key succeeded first.
        receipt = _reconciled_task_retry_cancellation(
            fenced,
            request,
            request_sha256=request_sha256,
            committed_at=clock.value,
            event_refusal=TaskRetrySeriesDisposition.CANCELLED,
        )
        assert receipt.task.status is TaskStatus.CANCELLED and receipt.successor is None
        assert receipt.reconciliation is not None and receipt.reconciliation.retried is False

    asyncio.run(scenario())
