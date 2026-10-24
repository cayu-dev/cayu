"""Exact declared task creation through the admitted application entrance."""

from __future__ import annotations

import asyncio

import pytest

from cayu import CayuApp
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskQuery


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_task_creation_operation_retries_once_and_conflicts_on_changed_input(
    backend, tmp_path, request
) -> None:
    from cayu.tasks.operations import TaskCreationConflict, TaskCreationOperation
    from cayu.tasks.types import TaskTypeSpec

    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario() -> None:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.sqlite import SQLiteTaskStore
        from cayu.storage.tasks_postgres import PostgresTaskStore

        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "tasks.sqlite")
            if backend == "sqlite"
            else PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
        )
        app = CayuApp(task_store=store, enable_logging=False)
        declaration = TaskTypeSpec(name="report", input_schema={"type": "object"})
        app.register_task_type(declaration)
        operation = TaskCreationOperation(operation_id="op-1", declaration=declaration)
        request = TaskCreate(task_id="new-task", type="report", input={"answer": True})
        first = await app.create_task_operation(request, operation)
        repeated = await app.create_task_operation(request, operation)
        assert first.id == repeated.id == "new-task"
        assert len(await store.list_tasks(TaskQuery())) == 1
        with pytest.raises(TaskCreationConflict):
            await app.create_task_operation(
                TaskCreate(task_id="new-task", type="report", input={"answer": 1}), operation
            )
        assert (await store.load_task("new-task")).input == {"answer": True}
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_replay_checks_attachment_then_survives_source_and_result_deletion(
    backend, tmp_path, request
) -> None:
    from tests.core.task_invocation_fixtures import task_backed_session_invocation

    from cayu.storage.migrations import SchemaMode
    from cayu.storage.sqlite import SQLiteTaskStore
    from cayu.storage.tasks_postgres import PostgresTaskStore
    from cayu.tasks.operations import (
        TaskCreationConflict,
        TaskCreationOperation,
        TaskCreationResultDeleted,
        task_replay_source,
    )
    from cayu.tasks.types import TaskTypeSpec

    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario() -> None:
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "replay.sqlite")
            if backend == "sqlite"
            else PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
        )
        app = CayuApp(task_store=store, enable_logging=False)
        spec = TaskTypeSpec(name="report", input_schema={"type": "object"})
        app.register_task_type(spec)
        source = await app.create_task(
            TaskCreate(task_id="source", type="report", input={"secret": "private-value"})
        )
        stale = TaskCreationOperation(
            operation_id="replay", declaration=spec, replay_source=task_replay_source(source)
        )
        binding = await task_backed_session_invocation(store, source.id, "source-session")
        source = await store.start_task(
            source.id, session_id=binding.id, session_invocation=binding
        )
        create = TaskCreate(task_id="replayed", type="report")
        with pytest.raises(TaskCreationConflict):
            await app.create_task_operation(create, stale)
        assert await store.load_task(create.task_id) is None
        operation = TaskCreationOperation(
            operation_id="replay", declaration=spec, replay_source=task_replay_source(source)
        )
        first, duplicate = await asyncio.gather(
            app.create_task_operation(create, operation),
            app.create_task_operation(create, operation),
        )
        assert first.id == duplicate.id == "replayed"
        assert first.input == {"secret": "private-value"}
        assert first.session_id is None
        assert first.invocation.root_invocation_id != source.invocation.root_invocation_id
        await store.complete_task(source.id, {})
        await store.delete_session_tasks("source-session", task_ids=(source.id,), policy=None)
        assert await store.load_task(source.id) is None
        assert (await app.create_task_operation(create, operation)).id == first.id
        binding = await task_backed_session_invocation(store, first.id, "result-session")
        await store.start_task(first.id, session_id=binding.id, session_invocation=binding)
        await store.complete_task(first.id, {})
        await store.delete_session_tasks("result-session", task_ids=(first.id,), policy=None)
        with pytest.raises(TaskCreationResultDeleted):
            await app.create_task_operation(create, operation)
        with pytest.raises((TaskCreationConflict, ValueError)):
            await app.create_task(TaskCreate(task_id=first.id, type="report"))
        assert await store.load_task(first.id) is None
        await app.aclose()

    asyncio.run(scenario())


def test_task_creation_post_commit_cancellation_keeps_exact_retry_receipt() -> None:
    from cayu.tasks.operations import TaskCreationOperation
    from cayu.tasks.types import TaskTypeSpec

    async def scenario() -> None:
        committed = asyncio.Event()
        release = asyncio.Event()

        class DelayedAcknowledgement(InMemoryTaskStore):
            hold = True

            async def create_task_operation(self, request, operation):
                task = await super().create_task_operation(request, operation)
                if self.hold:
                    self.hold = False
                    committed.set()
                    await release.wait()
                return task

        store = DelayedAcknowledgement()
        app = CayuApp(task_store=store, enable_logging=False)
        spec = TaskTypeSpec(name="report", input_schema={"type": "object"})
        app.register_task_type(spec)
        operation = TaskCreationOperation(operation_id="cancelled", declaration=spec)
        request = TaskCreate(task_id="cancelled-task", type="report")
        owner = asyncio.create_task(app.create_task_operation(request, operation))
        await committed.wait()
        assert (await store.load_task("cancelled-task")).status.value == "pending"
        owner.cancel()
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert owner.cancelling() == 2
        assert owner.cancelled()
        assert (await app.create_task_operation(request, operation)).id == "cancelled-task"
        release.set()
        assert len(await store.list_tasks(TaskQuery())) == 1
        await app.aclose()

    asyncio.run(scenario())


def test_sqlite_creation_identity_fence_is_required_on_reopen(tmp_path) -> None:
    import sqlite3

    from cayu.storage.sqlite import SQLiteTaskStore

    async def scenario() -> None:
        path = tmp_path / "fence.sqlite"
        store = SQLiteTaskStore(path)
        await store.list_tasks(TaskQuery())
        await store.close()
        with sqlite3.connect(path) as connection:
            connection.execute("DROP TRIGGER cayu_task_creation_identity_fence")
        with pytest.raises(RuntimeError, match="creation.*fence"):
            reopened = SQLiteTaskStore(path)
            try:
                await reopened.list_tasks(TaskQuery())
            finally:
                await reopened.close()

    asyncio.run(scenario())


def test_sqlite_creation_receipt_survives_restart_and_catalog_change(tmp_path) -> None:
    from cayu.storage.sqlite import SQLiteTaskStore
    from cayu.tasks import TaskCreationConflict, TaskCreationOperation, TaskTypeSpec

    async def scenario() -> None:
        path = tmp_path / "restart.sqlite"
        declaration = TaskTypeSpec(name="report", input_schema={"type": "object"})
        operation = TaskCreationOperation(operation_id="durable", declaration=declaration)
        request = TaskCreate(task_id="durable-result", type="report", input={"topic": "same"})
        first = CayuApp(task_store=SQLiteTaskStore(path), enable_logging=False)
        first.register_task_type(declaration)
        created = await first.create_task_operation(request, operation)
        await first.aclose()
        second = CayuApp(task_store=SQLiteTaskStore(path), enable_logging=False)
        second.register_task_type(declaration)
        assert (await second.create_task_operation(request, operation)).id == created.id
        assert len(await second.task_store.list_tasks(TaskQuery())) == 1
        await second.aclose()
        third = CayuApp(task_store=SQLiteTaskStore(path), enable_logging=False)
        changed = TaskTypeSpec(
            name="report", description="changed", input_schema={"type": "object"}
        )
        third.register_task_type(changed)
        try:
            with pytest.raises(TaskCreationConflict):
                await third.create_task_operation(
                    request, TaskCreationOperation(operation_id="durable", declaration=changed)
                )
            assert len(await third.task_store.list_tasks(TaskQuery())) == 1
        finally:
            await third.aclose()

    asyncio.run(scenario())


def test_sqlite_receipt_write_failure_rolls_back_task_before_retry(tmp_path) -> None:
    import sqlite3

    from cayu.storage.sqlite import SQLiteTaskStore
    from cayu.tasks import TaskCreationOperation, TaskTypeSpec

    async def scenario() -> None:
        path = tmp_path / "receipt-failure.sqlite"
        store = SQLiteTaskStore(path)
        app = CayuApp(task_store=store, enable_logging=False)
        declaration = TaskTypeSpec(name="report", input_schema={"type": "object"})
        app.register_task_type(declaration)
        operation = TaskCreationOperation(operation_id="atomic", declaration=declaration)
        request = TaskCreate(task_id="atomic-result", type="report", input={"topic": "stable"})
        with sqlite3.connect(path) as connection:
            connection.executescript("""
                CREATE TRIGGER test_fail_creation_receipt BEFORE INSERT ON cayu_task_creation_operations
                BEGIN SELECT RAISE(ABORT, 'synthetic receipt failure'); END;
            """)
        with pytest.raises(sqlite3.IntegrityError, match="synthetic receipt failure"):
            await app.create_task_operation(request, operation)
        assert await store.load_task("atomic-result") is None
        with sqlite3.connect(path) as connection:
            assert connection.execute(
                "SELECT count(*) FROM cayu_task_creation_operations"
            ).fetchone() == (0,)
            connection.execute("DROP TRIGGER test_fail_creation_receipt")
        assert (await app.create_task_operation(request, operation)).id == "atomic-result"
        assert len(await store.list_tasks(TaskQuery())) == 1
        await app.aclose()

    asyncio.run(scenario())
