"""The task outbox: a task queued in the application's own transaction, created by a drain."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.outbox import (
    SQLiteTaskOutbox,
    drain_task_outbox,
    run_task_outbox,
    task_outbox_ddl,
    task_outbox_row,
)
from cayu.tasks.queries import TaskQuery


def _app_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS invoices (id TEXT PRIMARY KEY)")
        connection.execute(task_outbox_ddl("task_outbox", dialect="sqlite"))


def _write(path: Path, invoice: str, request: TaskCreate, *, commit: bool) -> str:
    """Write an application row and queue its task in one transaction."""

    row = task_outbox_row(request)
    connection = sqlite3.connect(path)
    try:
        connection.execute("BEGIN")
        connection.execute("INSERT INTO invoices (id) VALUES (?)", (invoice,))
        connection.execute(
            "INSERT INTO task_outbox (id, request_json) VALUES (:id, :request_json)", row
        )
        if commit:
            connection.commit()
        else:
            connection.rollback()
    finally:
        connection.close()
    return row["id"]


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_a_committed_row_becomes_one_task_and_a_rolled_back_one_none(
    backend: str, tmp_path: Path
) -> None:
    path = tmp_path / "app.db"
    _app_database(path)
    kept = _write(path, "inv-1", TaskCreate(type="review", input={"invoice": "inv-1"}), commit=True)
    _write(path, "inv-2", TaskCreate(type="review", input={"invoice": "inv-2"}), commit=False)

    async def scenario() -> None:
        store = InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(path)
        outbox = SQLiteTaskOutbox(path)
        first = await drain_task_outbox(store, outbox)
        assert first.created == (kept,) and first.existing == ()
        assert await outbox.read(10) == []
        [task] = await store.list_tasks(TaskQuery())
        assert task.id == kept and task.input == {"invoice": "inv-1"}
        again = await drain_task_outbox(store, outbox)
        assert again.created == () and again.existing == ()

    asyncio.run(scenario())


def test_a_row_whose_task_exists_is_removed_without_a_second_task(tmp_path: Path) -> None:
    path = tmp_path / "app.db"
    _app_database(path)
    request = TaskCreate(task_id="review-inv-1", type="review")
    _write(path, "inv-1", request, commit=True)

    async def scenario() -> None:
        store = InMemoryTaskStore()
        # A drain that created the task and stopped before removing the row.
        await store.create_task(request)
        drained = await drain_task_outbox(store, SQLiteTaskOutbox(path))
        assert drained.existing == ("review-inv-1",) and drained.created == ()
        assert len(await store.list_tasks(TaskQuery())) == 1

    asyncio.run(scenario())


def test_the_loop_drains_until_stopped(tmp_path: Path) -> None:
    path = tmp_path / "app.db"
    _app_database(path)

    async def scenario() -> None:
        store = InMemoryTaskStore()
        stop = asyncio.Event()
        loop = asyncio.create_task(
            run_task_outbox(store, SQLiteTaskOutbox(path), interval_s=0.05, stop=stop)
        )
        task_id = _write(path, "inv-1", TaskCreate(type="review"), commit=True)
        for _ in range(200):
            if await store.load_task(task_id) is not None:
                break
            await asyncio.sleep(0.02)
        stop.set()
        await asyncio.wait_for(loop, 5)
        assert await store.load_task(task_id) is not None

    asyncio.run(scenario())


def test_postgres_outbox_drains_into_the_task_store(postgres_dsn: str) -> None:
    import psycopg

    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore
    from cayu.tasks.outbox import PostgresTaskOutbox

    with psycopg.connect(postgres_dsn) as connection:
        connection.execute("CREATE SCHEMA IF NOT EXISTS app")
        connection.execute(task_outbox_ddl("app.task_outbox", dialect="postgresql"))
        connection.execute("CREATE TABLE IF NOT EXISTS app.invoices (id TEXT PRIMARY KEY)")
    with psycopg.connect(postgres_dsn) as connection:
        row = task_outbox_row(TaskCreate(type="review"))
        connection.execute("INSERT INTO app.invoices (id) VALUES ('inv-1')")
        connection.execute(
            "INSERT INTO app.task_outbox (id, request_json) VALUES (%(id)s, %(request_json)s)",
            row,
        )

    async def scenario() -> None:
        store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            outbox = PostgresTaskOutbox(postgres_dsn, table="app.task_outbox")
            drained = await drain_task_outbox(store, outbox)
            assert drained.created == (row["id"],)
            assert await outbox.read(10) == []
            assert await store.load_task(row["id"]) is not None
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("table", ["", "a.b.c", "x; DROP TABLE y", '"quoted"'])
def test_table_names_are_plain_identifiers(table: str) -> None:
    with pytest.raises(ValueError):
        task_outbox_ddl(table, dialect="sqlite")


def test_a_parent_and_child_queued_together_drain_in_order_and_a_refusal_blocks_nothing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "app.db"
    _app_database(path)
    with sqlite3.connect(path) as connection:
        for request in (
            TaskCreate(task_id="zroot", type="review"),
            TaskCreate(task_id="achild", type="review", parent_task_id="zroot"),
            TaskCreate(task_id="orphan", type="review", parent_task_id="missing"),
            TaskCreate(task_id="later", type="review"),
        ):
            connection.execute(
                "INSERT INTO task_outbox (id, request_json, created_at) "
                "VALUES (:id, :request_json, '2026-01-01T00:00:00.000')",
                task_outbox_row(request),
            )
        connection.execute(
            "INSERT INTO task_outbox (id, request_json) VALUES ('broken', 'not json')"
        )

    async def scenario() -> None:
        store = InMemoryTaskStore()
        outbox = SQLiteTaskOutbox(path)
        drained = await drain_task_outbox(store, outbox)
        assert drained.created == ("zroot", "achild", "later")
        assert [item.id for item in drained.refused] == ["orphan", "broken"]
        assert await outbox.read(10) == []

    asyncio.run(scenario())


def test_a_request_with_runtime_provenance_is_refused_before_the_commit() -> None:
    from cayu.sessions.invocation import TaskExecutionSource
    from cayu.tasks.creation import task_create_with_execution_source

    request = task_create_with_execution_source(
        TaskCreate(type="review"), source=TaskExecutionSource.WEBHOOK
    )
    with pytest.raises(ValueError, match="provenance"):
        task_outbox_row(request)
