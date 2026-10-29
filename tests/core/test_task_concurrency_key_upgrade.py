"""Forward migrations from fresh disposable revision-122 databases."""

from __future__ import annotations

import asyncio

import pytest

from cayu.storage import PostgresSessionStore, PostgresTaskStore, SQLiteTaskStore, migrations
from cayu.storage import _postgres_support as pg
from cayu.storage import _sqlite_connection as sqlite_connection
from cayu.storage import _sqlite_records as sq
from cayu.storage import _sqlite_support as sqlite_support
from cayu.tasks import InMemoryTaskStore, TaskCreate


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_task_key_forward_upgrade(backend, tmp_path, request, monkeypatch):
    # postgres_dsn is a new module-scoped database created by the shared test
    # fixture on its disposable local PostgreSQL server; it has no existing data.
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None
    path = tmp_path / "revision-122.sqlite"

    async def scenario():
        original = await InMemoryTaskStore().create_task(TaskCreate(type="legacy-job"))
        with monkeypatch.context() as legacy:
            legacy.setattr(
                migrations,
                "REVISIONS",
                tuple(item for item in migrations.REVISIONS if item.revision <= 122),
            )
            if backend == "sqlite":
                with sqlite_connection.connect(path) as connection:
                    sqlite_support.reconcile_schema(
                        connection, migrations.SchemaMode.MIGRATE, app_min_supported=122
                    )
                    columns = (
                        "id,type,title,description,status,session_id,session_instance_id,"
                        "parent_task_id,assigned_agent_name,available_at,worker_id,"
                        "lease_expires_at,interrupted_handoff_id,status_reason,status_payload_json,"
                        "input_json,result_json,error_json,metadata_json,created_at,updated_at,"
                        "started_at,completed_at,invocation_json,retry_series_json,work_contract_json,"
                        "schedule_json,restartable,restart_count"
                    )
                    # Revision 122 predates the concurrency key and the revision-135 columns.
                    values = sq.task_to_row_values(original)[:-3]
                    connection.execute(
                        f"INSERT INTO cayu_tasks ({columns}) VALUES ({','.join('?' for _ in values)})",
                        values,
                    )
            else:
                import psycopg

                historical = PostgresSessionStore(dsn, schema_mode=migrations.SchemaMode.MIGRATE)
                historical._min_required_revision = 122
                try:
                    await historical.ensure_schema()
                finally:
                    await historical.close()
                async with await psycopg.AsyncConnection.connect(dsn) as connection:
                    columns = pg.TASK_COLUMNS.rsplit(", ", 3)[0]
                    values = pg.task_insert_values(original)[:-3]
                    await connection.execute(
                        f"INSERT INTO cayu_tasks ({columns}) VALUES ({','.join('%s' for _ in values)})",
                        values,
                    )
        if backend == "sqlite":
            with pytest.raises(migrations.SchemaTooOld):
                SQLiteTaskStore(path, schema_mode=migrations.SchemaMode.VALIDATE)
            store = SQLiteTaskStore(path, schema_mode=migrations.SchemaMode.MIGRATE)
        else:
            invalid = PostgresTaskStore(dsn, schema_mode=migrations.SchemaMode.VALIDATE)
            try:
                with pytest.raises(migrations.SchemaTooOld):
                    await invalid.load_task(original.id)
            finally:
                await invalid.close()
            store = PostgresTaskStore(dsn, schema_mode=migrations.SchemaMode.MIGRATE)
        try:
            assert await store.load_task(original.id) == original
            keyed = await store.create_task(TaskCreate(type="new-job", concurrency_key="repo#1"))
            assert (await store.load_task(keyed.id)).concurrency_key == "repo#1"
        finally:
            await store.close()

    asyncio.run(scenario())
