"""Application foreign keys do not grant Cayu ownership of referencing rows."""

import asyncio
from uuid import uuid4

import pytest
from tests.core.session_closure_conformance import create_closure_session

from cayu import CayuApp
from cayu.storage._session_closure_sql import TASK_CLOSURE_DEPENDENCIES
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks.base import TaskCreate


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("table_prefix", ["app_audit", "cayu_app_audit"])
def test_public_closure_preserves_application_references(backend, table_prefix, tmp_path, request):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        if backend == "sqlite":
            sessions = SQLiteSessionStore(tmp_path / "ownership.db")
            tasks = SQLiteTaskStore(tmp_path / "ownership.db")
        else:
            from cayu.storage.postgres import PostgresSessionStore, PostgresTaskStore

            sessions = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
            tasks = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)

        async def execute(statement, parameters=()):
            if backend == "sqlite":
                cursor = tasks._connection.execute(statement, parameters)
                try:
                    rows = [tuple(row) for row in cursor.fetchall()]
                    tasks._connection.commit()
                    return rows
                finally:
                    cursor.close()
            async with tasks._connection() as connection, connection.cursor() as cursor:
                await cursor.execute(statement.replace("?", "%s"), parameters)
                return [] if cursor.description is None else await cursor.fetchall()

        root, selected, other, attempt = (uuid4().hex for _ in range(4))
        table = f"{table_prefix}_{uuid4().hex}"
        created = False
        try:
            await create_closure_session(sessions, root)
            for task_id, session_id in ((selected, root), (other, None)):
                await tasks.create_task(
                    TaskCreate(type="test", task_id=task_id, session_id=session_id)
                )
                await tasks.complete_task(task_id, {})
            # Pin the ownership mapping to the installed Cayu schema. New native
            # dependencies require an explicit decision, not automatic authority.
            if backend == "sqlite":
                native_dependencies = set()
                for (name,) in await execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name GLOB 'cayu_*'"
                ):
                    for foreign_key in await execute(f"PRAGMA foreign_key_list('{name}')"):
                        if foreign_key[2] == "cayu_tasks":
                            native_dependencies.add((name, foreign_key[3]))
            else:
                native_dependencies = set(
                    await execute(
                        "SELECT DISTINCT kcu.table_name, kcu.column_name "
                        "FROM information_schema.table_constraints AS tc "
                        "JOIN information_schema.key_column_usage AS kcu "
                        "ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema "
                        "JOIN information_schema.constraint_column_usage AS ccu "
                        "ON tc.constraint_name = ccu.constraint_name AND tc.constraint_schema = ccu.constraint_schema "
                        "WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = current_schema() "
                        "AND ccu.table_name = 'cayu_tasks' AND ccu.column_name = 'id'"
                    )
                )
            assert native_dependencies == TASK_CLOSURE_DEPENDENCIES
            # A real Cayu-owned dependent is deleted first, then must roll back
            # when the application's restrictive reference prevents task deletion.
            await execute(
                "INSERT INTO cayu_work_attempts "
                "(attempt_id, task_id, ordinal, request_sha256, started_at, attempt_json) "
                "VALUES (?, ?, 1, ?, ?, ?)",
                (attempt, selected, "a" * 64, "2026-09-14T00:00:00+00:00", "{}"),
            )
            await execute(
                f"CREATE TABLE {table} (task_id TEXT PRIMARY KEY "
                "REFERENCES cayu_tasks(id) ON DELETE RESTRICT, audit TEXT NOT NULL)"
            )
            created = True
            for task_id in (selected, other):
                await execute(f"INSERT INTO {table} VALUES (?, ?)", (task_id, "application-owned"))
            app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
            first = await app.erase_session_closure(root)
            assert not first.complete
            assert await sessions.load(root) is not None
            assert await tasks.load_task(selected) is not None
            assert set(await execute(f"SELECT task_id, audit FROM {table}")) == {
                (selected, "application-owned"),
                (other, "application-owned"),
            }
            assert await execute(
                "SELECT attempt_id FROM cayu_work_attempts WHERE task_id = ?", (selected,)
            ) == [(attempt,)]

            # Only the application removes its blocking reference. The same
            # retained closure plan can then finish normal Cayu-owned cleanup.
            await execute(f"DELETE FROM {table} WHERE task_id = ?", (selected,))
            second = await CayuApp(
                session_store=sessions, task_store=tasks, enable_logging=False
            ).erase_session_closure(root)
            assert second.complete and second.plan_id == first.plan_id
            assert await tasks.load_task(selected) is None
            assert await sessions.load(root) is None
            assert (
                await execute(
                    "SELECT attempt_id FROM cayu_work_attempts WHERE task_id = ?", (selected,)
                )
                == []
            )
            assert await tasks.load_task(other) is not None
            assert await execute(f"SELECT task_id, audit FROM {table}") == [
                (other, "application-owned")
            ]
        finally:
            if created:
                await execute(f"DROP TABLE {table}")
            await tasks.close()
            await sessions.close()

    asyncio.run(scenario())
