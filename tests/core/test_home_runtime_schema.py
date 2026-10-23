"""Current bound claim storage composes with idempotent migration and reset owners."""

from __future__ import annotations

import asyncio
import importlib
import sqlite3

import pytest


def test_bound_registry_survives_sqlite_idempotent_migration(tmp_path):
    from cayu.storage import SQLiteTaskStore
    from cayu.storage.migrations import LATEST_REVISION, SchemaMode
    from cayu.tasks.creation import TaskCreate

    database = tmp_path / "current.db"

    async def seed():
        store = SQLiteTaskStore(database)
        await store.create_task(TaskCreate(task_id="preserved", type="work"))
        await store.close()

    asyncio.run(seed())
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE cayu_local_execution_attempts")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 66")
        connection.execute("PRAGMA user_version = 65")

    async def migrate():
        store = SQLiteTaskStore(database, schema_mode=SchemaMode.MIGRATE)
        assert (await store.load_task("preserved")).id == "preserved"
        await store.close()

    asyncio.run(migrate())
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (LATEST_REVISION,)
        assert "expected_sha256" in {
            r[1]
            for r in connection.execute(
                "PRAGMA table_info(cayu_task_interrupted_continuation_claims)"
            )
        }


@pytest.mark.parametrize(
    "module,tables_name",
    [
        ("session_store_conformance.support", "_POSTGRES_TABLES"),
        ("test_event_watchers", "_POSTGRES_TABLES"),
        ("test_postgres_budget_ledger", "_TABLES"),
        ("postgres_contention_support", "POSTGRES_CONTENTION_TABLES"),
        ("test_postgres_session_store", "_TABLES"),
        ("test_postgres_task_store", "_TABLES"),
        ("test_postgres_migrations", "_TABLES"),
        ("test_postgres_knowledge_store", "_TABLES"),
    ],
)
def test_bound_registry_survives_postgres_reset(postgres_dsn, module, tables_name):
    import psycopg
    from psycopg import sql

    from cayu.storage import PostgresTaskStore
    from cayu.storage.migrations import SchemaMode

    tables = getattr(importlib.import_module(f"tests.core.{module}"), tables_name)

    async def run():
        store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        await store.list_tasks()
        await store.close()
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            for table in tables:
                await connection.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table))
                )
        for mode in (SchemaMode.CREATE, SchemaMode.VALIDATE):
            store = PostgresTaskStore(postgres_dsn, schema_mode=mode)
            assert await store.list_tasks() == []
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("invalid", ["missing", "wrong_type"])
def test_current_bound_registry_still_rejects_invalid_shape(backend, invalid, tmp_path, request):
    from cayu.storage import PostgresTaskStore, SQLiteTaskStore
    from cayu.storage.migrations import SchemaMode

    database = tmp_path / "invalid.db"

    async def run():
        if backend == "sqlite":
            store = SQLiteTaskStore(database)
        else:
            import psycopg

            dsn = request.getfixturevalue("postgres_dsn")
            store = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
        await store.list_tasks()
        await store.close()
        table = "cayu_task_interrupted_continuation_claims"
        statements = [f"ALTER TABLE {table} DROP COLUMN expected_sha256"]
        if invalid == "wrong_type":
            statements.append(f"ALTER TABLE {table} ADD COLUMN expected_sha256 INTEGER")
        if backend == "sqlite":
            with sqlite3.connect(database) as connection:
                for statement in statements:
                    connection.execute(statement)

        else:
            async with await psycopg.AsyncConnection.connect(dsn) as connection:
                for statement in statements:
                    await connection.execute(statement)

        validator = None
        try:
            with pytest.raises(RuntimeError, match="revision-76"):
                validator = (
                    SQLiteTaskStore(database, schema_mode=SchemaMode.VALIDATE)
                    if backend == "sqlite"
                    else PostgresTaskStore(dsn, schema_mode=SchemaMode.VALIDATE)
                )
                await validator.list_tasks()
        finally:
            if validator is not None:
                await validator.close()
            if backend == "postgres":
                async with await psycopg.AsyncConnection.connect(dsn) as connection:
                    await connection.execute(
                        f"ALTER TABLE {table} DROP COLUMN IF EXISTS expected_sha256"
                    )
                    await connection.execute(
                        f'ALTER TABLE {table} ADD COLUMN expected_sha256 TEXT COLLATE "C"'
                    )

    asyncio.run(run())
