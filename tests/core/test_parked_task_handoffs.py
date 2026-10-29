"""Parked interrupted-task handoffs: only their bound trigger continues them."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from tests.core.task_invocation_fixtures import task_backed_session_invocation

from cayu.storage import SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore
from cayu.tasks.continuation_claims import TaskContinuationExpectation
from cayu.tasks.creation import TaskCreate
from cayu.tasks.handoff import TaskParkedWait, interrupted_task_handoff_request
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import Task, TaskClaimLost, TaskStatus

_WAKE = TaskParkedWait(kind="session_wake", wait_id="wake-1")
_OPERATION = "0" * 64


def _store(backend: str, tmp_path, request):
    if backend == "memory":
        return InMemoryTaskStore()
    if backend == "sqlite":
        return SQLiteTaskStore(tmp_path / "tasks.db")
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    return PostgresTaskStore(request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE)


async def _close(store) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


async def _released(store, task_type: str, *, parked: TaskParkedWait | None) -> Task:
    task_id = f"{task_type}-{uuid4().hex}"
    session_id = f"session-{task_id}"
    await store.create_task(TaskCreate(task_id=task_id, type=task_type))
    claimed = await store.claim_task("first-worker", TaskQuery(type=task_type))
    assert claimed is not None and claimed.id == task_id
    attached = await store.attach_task(
        task_id,
        session_id=session_id,
        session_invocation=await task_backed_session_invocation(store, task_id, session_id),
        worker_id="first-worker",
        lease_expires_at=claimed.lease_expires_at,
    )
    receipt = await store.release_interrupted_task_worker(
        interrupted_task_handoff_request(attached, session_run_epoch=1, parked=parked)
    )
    assert receipt.request.parked == parked
    loaded = await store.load_interrupted_task_handoff_receipt(task_id, receipt.request.handoff_id)
    assert loaded == receipt
    return receipt.task


def _expected(task: Task, parked_wait: TaskParkedWait | None) -> TaskContinuationExpectation:
    assert task.session_id is not None and task.session_instance_id is not None
    assert task.interrupted_handoff_id is not None
    return TaskContinuationExpectation(
        task_id=task.id,
        session_id=task.session_id,
        session_instance_id=task.session_instance_id,
        prior_handoff_id=task.interrupted_handoff_id,
        operation_sha256=_OPERATION,
        parked_wait=parked_wait,
    )


async def _claim(store, task: Task, *, expected: TaskContinuationExpectation | None = None, **kw):
    bound = {} if expected is None else {"task_id": task.id, "expected": expected}
    return await store.claim_interrupted_task_continuation(
        "next-worker", TaskQuery(type=task.type), handoff_id=str(uuid4()), **bound, **kw
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_a_parked_handoff_is_never_claimed_by_the_generic_scan(backend, tmp_path, request):
    async def scenario() -> None:
        store = _store(backend, tmp_path, request)
        try:
            parked = await _released(store, "parked-scan", parked=_WAKE)
            page = await _claim(store, parked)
            assert page.task is None and page.exhausted
            # An exact selector without the bound wait is not the trigger either.
            exact = await store.claim_interrupted_task_continuation(
                "next-worker", None, handoff_id=str(uuid4()), task_id=parked.id
            )
            assert exact.task is None
            current = await store.load_task(parked.id)
            assert current.status is TaskStatus.RUNNING and current.worker_id is None

            ordinary = await _released(store, "parked-scan", parked=None)
            page = await _claim(store, ordinary)
            assert page.task is not None and page.task.id == ordinary.id
        finally:
            await _close(store)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_only_the_bound_trigger_continues_a_parked_handoff(backend, tmp_path, request):
    async def scenario() -> None:
        store = _store(backend, tmp_path, request)
        try:
            parked = await _released(store, "parked-bound", parked=_WAKE)
            other = TaskParkedWait(kind="session_wake", wait_id="wake-2")
            for wrong in (None, other):
                with pytest.raises(TaskClaimLost):
                    await _claim(store, parked, expected=_expected(parked, wrong))
            page = await _claim(store, parked, expected=_expected(parked, _WAKE))
            assert page.task is not None and page.task.worker_id == "next-worker"

            # A bound claim that presents a wait cannot take an ordinary handoff.
            ordinary = await _released(store, "parked-bound-ordinary", parked=None)
            with pytest.raises(TaskClaimLost):
                await _claim(store, ordinary, expected=_expected(ordinary, _WAKE))
            page = await _claim(store, ordinary, expected=_expected(ordinary, None))
            assert page.task is not None
        finally:
            await _close(store)

    asyncio.run(scenario())


def test_the_receipt_digest_of_an_unparked_handoff_is_unchanged() -> None:
    from cayu.tasks.handoff import prepare_interrupted_task_handoff

    async def scenario() -> None:
        store = InMemoryTaskStore()
        task = await _released(store, "digest", parked=None)
        receipt = await store.load_interrupted_task_handoff_receipt(
            task.id, task.interrupted_handoff_id
        )
        assert "parked" not in receipt.request.model_dump(mode="json")
        _, digest = prepare_interrupted_task_handoff(receipt.request)
        assert digest == receipt.request_sha256

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_revision_137_indexes_parked_handoff_receipts(backend, tmp_path, request):
    async def scenario() -> str:
        store = _store(backend, tmp_path, request)
        try:
            await _released(store, "parked-index", parked=_WAKE)
        finally:
            await _close(store)
        if backend == "sqlite":
            import sqlite3

            with sqlite3.connect(tmp_path / "tasks.db") as connection:
                (sql,) = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'index' "
                    "AND name = 'idx_cayu_task_handoff_receipts_parked'"
                ).fetchone()
            return sql
        import psycopg

        with psycopg.connect(request.getfixturevalue("postgres_dsn")) as connection:
            (sql,) = connection.execute(
                "SELECT indexdef FROM pg_indexes "
                "WHERE indexname = 'idx_cayu_task_handoff_receipts_parked'"
            ).fetchone()
        return sql

    definition = asyncio.run(scenario())
    assert "cayu_task_interrupted_handoff_receipts" in definition
    assert "parked" in definition.split("WHERE", 1)[1]


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_a_store_lists_only_its_current_parked_handoffs_in_pages(backend, tmp_path, request):
    async def scenario() -> None:
        store = _store(backend, tmp_path, request)
        try:
            before = {r.request.task_id for r in await _all_parked(store)}
            parked = [await _released(store, "listed", parked=_WAKE) for _ in range(3)]
            await _released(store, "listed", parked=None)
            claimed = await _released(store, "listed", parked=_WAKE)
            await _claim(store, claimed, expected=_expected(claimed, _WAKE))

            listed = await _all_parked(store, page=2)
            ids = [r.request.task_id for r in listed if r.request.task_id not in before]
            assert ids == sorted(task.id for task in parked)
            assert all(r.request.parked == _WAKE for r in listed)
            assert {r.task.interrupted_handoff_id for r in listed if r.request.task_id in ids} == {
                task.interrupted_handoff_id for task in parked
            }
        finally:
            await _close(store)

    asyncio.run(scenario())


async def _all_parked(store, page: int = 100):
    listed, after = [], None
    while True:
        batch = await store.list_parked_interrupted_task_handoffs(after=after, limit=page)
        listed.extend(batch)
        if len(batch) < page:
            return listed
        after = batch[-1].request.task_id
