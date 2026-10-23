"""Exact Inbox continuation selection through native task-store ownership."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from tests.core.task_invocation_fixtures import task_backed_session_invocation
from tests.core.test_task_store import _close_store, _make_store

from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.continuation_claims import TaskContinuationExpectation
from cayu.tasks.creation import TaskCreate
from cayu.tasks.handoff import interrupted_task_handoff_request
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskClaimLost


@pytest.mark.parametrize("store_factory", [InMemoryTaskStore, SQLiteTaskStore])
def test_exact_continuation_binds_session_incarnation_before_claim(store_factory, tmp_path):
    store = _make_store(store_factory, tmp_path)

    async def exercise():
        nonlocal store
        try:
            await store.create_task(TaskCreate(task_id="selected", type="job"))
            owned = await store.claim_task("original")
            assert owned is not None
            attached = await store.attach_task(
                owned.id,
                session_id="selected-session",
                session_invocation=await task_backed_session_invocation(
                    store, owned.id, "selected-session"
                ),
                worker_id=owned.worker_id,
                lease_expires_at=owned.lease_expires_at,
            )
            await store.release_interrupted_task_worker(
                interrupted_task_handoff_request(attached, session_run_epoch=1)
            )
            before = await store.load_task(owned.id)
            assert before is not None
            generation = str(uuid4())
            expectation = TaskContinuationExpectation(
                task_id=before.id,
                session_id=before.session_id,
                session_instance_id=before.session_instance_id,
                prior_handoff_id=before.interrupted_handoff_id,
                operation_sha256="a" * 64,
            )
            for session_id, instance in [
                ("other-session", before.session_instance_id),
                (before.session_id, str(uuid4())),
            ]:
                with pytest.raises(TaskClaimLost):
                    await store.claim_interrupted_task_continuation(
                        "dashboard",
                        handoff_id=generation,
                        task_id=before.id,
                        expected=expectation.model_copy(
                            update={
                                "session_id": session_id,
                                "session_instance_id": instance,
                            }
                        ),
                    )
                assert await store.load_task(owned.id) == before
            selected = await store.claim_interrupted_task_continuation(
                "dashboard",
                handoff_id=generation,
                task_id=before.id,
                expected=expectation,
            )
            assert selected.task is not None
            assert selected.task.worker_id == "dashboard"
            if store_factory is SQLiteTaskStore:
                await _close_store(store)
                store = _make_store(store_factory, tmp_path)
            replay = await store.claim_interrupted_task_continuation(
                "dashboard",
                handoff_id=generation,
                task_id=before.id,
                expected=expectation,
            )
            assert replay.replayed
            with pytest.raises(TaskClaimLost):
                await store.claim_interrupted_task_continuation(
                    "dashboard",
                    handoff_id=generation,
                    task_id=before.id,
                    expected=expectation.model_copy(update={"operation_sha256": "b" * 64}),
                )
            with pytest.raises(TaskClaimLost):
                await store.claim_interrupted_task_continuation(
                    "dashboard", handoff_id=generation, task_id=before.id
                )
            assert await store.load_task(owned.id) == selected.task
        finally:
            await _close_store(store)

    asyncio.run(exercise())


def test_postgres_bound_continuation_replays_exact_persisted_expectation(postgres_dsn):
    from cayu.storage import PostgresTaskStore
    from cayu.storage.migrations import SchemaMode

    async def exercise():
        store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await store.create_task(TaskCreate(task_id="selected", type="job"))
            owned = await store.claim_task("original")
            attached = await store.attach_task(
                owned.id,
                session_id="selected-session",
                session_invocation=await task_backed_session_invocation(
                    store, owned.id, "selected-session"
                ),
                worker_id=owned.worker_id,
                lease_expires_at=owned.lease_expires_at,
            )
            released = await store.release_interrupted_task_worker(
                interrupted_task_handoff_request(attached, session_run_epoch=1)
            )
            before = released.task
            expected = TaskContinuationExpectation(
                task_id=before.id,
                session_id=before.session_id,
                session_instance_id=before.session_instance_id,
                prior_handoff_id=before.interrupted_handoff_id,
                operation_sha256="a" * 64,
            )
            generation = str(uuid4())
            with pytest.raises(TaskClaimLost):
                await store.claim_interrupted_task_continuation(
                    "dashboard",
                    task_id=before.id,
                    handoff_id=generation,
                    expected=expected.model_copy(update={"prior_handoff_id": "changed"}),
                )
            assert await store.load_task(before.id) == before
            claimed = await store.claim_interrupted_task_continuation(
                "dashboard",
                task_id=before.id,
                handoff_id=generation,
                expected=expected,
            )
            await store.close()
            store = PostgresTaskStore(postgres_dsn)
            replay = await store.claim_interrupted_task_continuation(
                "dashboard",
                task_id=before.id,
                handoff_id=generation,
                expected=expected,
            )
            assert replay.replayed and replay.task == claimed.task
            with pytest.raises(TaskClaimLost):
                await store.claim_interrupted_task_continuation(
                    "dashboard",
                    task_id=before.id,
                    handoff_id=generation,
                    expected=expected.model_copy(update={"operation_sha256": "b" * 64}),
                )
        finally:
            await store.close()

    asyncio.run(exercise())
