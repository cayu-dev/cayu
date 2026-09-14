from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from cayu import TaskCreate, TaskQuery, TaskSchedulePolicy
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_schedule_migration_preserves_unmanaged_tasks_and_accounting(
    tmp_path, postgres_dsn, backend
):
    async def run():
        path = tmp_path / "schedule-migration.sqlite"
        if backend == "sqlite":
            creator = SQLiteTaskStore(path)
        else:
            creator = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            original = await creator.create_task(TaskCreate(task_id="ordinary", type="ordinary"))
        finally:
            await creator.close()

        # Reconstruct revision 89's task schema without changing its accounting
        # definitions. The public migrator must add scheduling, not reinterpret
        # previously accepted task content or rebuild auxiliary accounting.
        if backend == "sqlite":
            with sqlite3.connect(path) as connection:
                connection.execute("DROP TABLE cayu_task_schedule_receipts")
                connection.execute("DROP TABLE cayu_task_schedule_events")
                connection.execute("DROP INDEX idx_cayu_tasks_next_schedule")
                connection.execute("ALTER TABLE cayu_tasks DROP COLUMN schedule_json")
                connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 90")
                connection.execute("PRAGMA user_version = 89")
                generation = connection.execute(
                    "SELECT generation FROM cayu_accounting_state"
                ).fetchone()[0]
            migrated = SQLiteTaskStore(path, schema_mode=SchemaMode.MIGRATE)
        else:
            import psycopg

            async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                await connection.execute("DROP TABLE cayu_task_schedule_receipts")
                await connection.execute("DROP TABLE cayu_task_schedule_events")
                await connection.execute("DROP INDEX idx_cayu_tasks_next_schedule")
                await connection.execute("ALTER TABLE cayu_tasks DROP COLUMN schedule")
                await connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 90")
                cursor = await connection.execute("SELECT generation FROM cayu_accounting_state")
                generation = (await cursor.fetchone())[0]
            migrated = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            assert await migrated.load_task("ordinary") == original
            assert await migrated.list_task_schedule_events("ordinary") == []
            future = await migrated.create_task(
                TaskCreate(
                    task_id="future",
                    type="future",
                    available_at=datetime.now(UTC) + timedelta(hours=1),
                    schedule_policy=TaskSchedulePolicy(),
                )
            )
            assert future.schedule is not None and future.schedule.revision == 1
            assert await migrated.claim_task("early", TaskQuery(type="future")) is None
            claimed = await migrated.claim_task("ordinary-worker", TaskQuery(type="ordinary"))
            assert claimed is not None and claimed.id == original.id and claimed.schedule is None
        finally:
            await migrated.close()

        if backend == "sqlite":
            with sqlite3.connect(path) as connection:
                assert connection.execute(
                    "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 90"
                ).fetchone() == ("breaking", 90)
                assert connection.execute(
                    "SELECT generation FROM cayu_accounting_state"
                ).fetchone() == (generation,)
            reopened = SQLiteTaskStore(path, schema_mode=SchemaMode.VALIDATE)
        else:
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                cursor = await connection.execute(
                    "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 90"
                )
                assert await cursor.fetchone() == ("breaking", 90)
                cursor = await connection.execute("SELECT generation FROM cayu_accounting_state")
                assert await cursor.fetchone() == (generation,)
            reopened = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            assert await reopened.load_task("future") == future
            assert len(await reopened.list_task_schedule_events("future")) == 1
        finally:
            await reopened.close()

    asyncio.run(run())
