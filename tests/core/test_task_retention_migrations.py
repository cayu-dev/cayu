"""Task stores require the retention epoch floor before accepting work."""

import asyncio
import sqlite3
from uuid import uuid4

import pytest

from cayu.storage import PostgresTaskStore, SQLiteTaskStore
from cayu.storage.migrations import SchemaMode, SchemaTooOld
from cayu.tasks import TaskCreate, TaskSessionClosureClaim


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_task_retention_epoch_requires_migration(backend, tmp_path, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        path = tmp_path / "retention.db"
        creator = (
            SQLiteTaskStore(path)
            if backend == "sqlite"
            else PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
        )
        original = await creator.create_task(TaskCreate(type="original"))
        await creator.close()
        if backend == "sqlite":
            with sqlite3.connect(path) as connection:
                connection.execute("DROP TABLE cayu_task_session_retention_epochs")
                connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 123")
                connection.execute("PRAGMA user_version = 122")
            with pytest.raises(SchemaTooOld):
                SQLiteTaskStore(path, schema_mode=SchemaMode.VALIDATE)
            migrated = SQLiteTaskStore(path, schema_mode=SchemaMode.MIGRATE)
        else:
            import psycopg

            async with await psycopg.AsyncConnection.connect(dsn) as connection:
                await connection.execute("DROP TABLE cayu_task_session_retention_epochs")
                await connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 123")
            rejected = PostgresTaskStore(dsn, schema_mode=SchemaMode.VALIDATE)
            try:
                with pytest.raises(SchemaTooOld):
                    await rejected.load_task(original.id)
            finally:
                await rejected.close()
            migrated = PostgresTaskStore(dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            assert await migrated.load_task(original.id) == original
            instance_id = str(uuid4())
            await migrated.advance_session_retention_epoch(
                "session", session_instance_id=instance_id, run_epoch=1
            )
            with pytest.raises(ValueError, match="superseded invocation"):
                await migrated.claim_session_closure(
                    TaskSessionClosureClaim(
                        session_id="session",
                        plan_id="a" * 64,
                        task_ids=(),
                        purpose="retention",
                        retention_instance_id=instance_id,
                        retention_run_epoch=0,
                    )
                )
        finally:
            await migrated.close()

    asyncio.run(scenario())
