"""Session-store retention: one shared conformance suite for SQLite and Postgres."""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from collections.abc import Iterator
from datetime import timedelta

import pytest
from tests.core.session_retention_conformance import (
    SCENARIOS,
    RetentionClock,
    RetentionHarness,
    assert_policy_contract,
    create_retention_session,
    policy,
)

from cayu.sessions.base import SessionStore
from cayu.snapshots.base import SQLiteAgentSnapshotStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.retention import (
    RetentionAuditUnavailable,
    RetentionMode,
    RetentionProtection,
    SessionRetentionPolicy,
)
from cayu.storage.sqlite import SQLiteSessionStore, SQLiteTaskStore

# Tables a retention scenario may populate. Schema, migration receipts and the
# public-authority alias keyring stay; everything else is cleared per test.
_KEPT_POSTGRES_TABLES = (
    "cayu_schema_migrations",
    "cayu_schema_migration_receipts",
    "cayu_public_authority_alias_config",
    "cayu_public_authority_alias_keys",
    "cayu_transcript_search_configuration",
    "cayu_accounting_state",
)


def _sqlite_scenario(sqlite_resources, scenario) -> None:
    async def run() -> None:
        async with sqlite_resources as resources:
            path = resources.path()
            clock = RetentionClock()
            store = resources.own(SQLiteSessionStore(path, ownership_clock=clock))
            tasks = resources.own(SQLiteTaskStore(path))

            async def execute(sql, parameters=(), *, foreign_keys=True):
                with contextlib.closing(sqlite3.connect(path, timeout=5)) as connection:
                    connection.execute(f"PRAGMA foreign_keys = {'ON' if foreign_keys else 'OFF'}")
                    connection.execute(sql, parameters)
                    connection.commit()

            async def snapshot_tables():
                SQLiteAgentSnapshotStore(path)

            await scenario(
                RetentionHarness(
                    store=store,
                    tasks=tasks,
                    clock=clock,
                    execute=execute,
                    snapshot_tables=snapshot_tables,
                )
            )

    asyncio.run(run())


@pytest.fixture(scope="module")
def retention_postgres_dsn(postgres_dsn) -> Iterator[str]:
    async def initialize() -> None:
        from cayu.storage.postgres import PostgresSessionStore, PostgresTaskStore

        for store_type in (PostgresSessionStore, PostgresTaskStore):
            store = store_type(postgres_dsn, schema_mode=SchemaMode.CREATE)
            try:
                await store._ensure_ready()
            finally:
                await store.close()

    asyncio.run(initialize())
    yield postgres_dsn


async def _clear_postgres(dsn: str) -> None:
    import psycopg
    from psycopg import sql

    async with await psycopg.AsyncConnection.connect(dsn) as connection:
        rows = await (
            await connection.execute(
                "SELECT tablename FROM pg_catalog.pg_tables "
                "WHERE schemaname = current_schema() AND tablename LIKE 'cayu%%'"
            )
        ).fetchall()
        tables = [row[0] for row in rows if row[0] not in _KEPT_POSTGRES_TABLES]
        if tables:
            await connection.execute(
                sql.SQL("TRUNCATE {} CASCADE").format(
                    sql.SQL(", ").join(sql.Identifier(table) for table in tables)
                )
            )
        await connection.commit()


def _postgres_scenario(dsn: str, scenario) -> None:
    async def run() -> None:
        import psycopg

        from cayu.storage.postgres import PostgresSessionStore, PostgresTaskStore

        await _clear_postgres(dsn)
        store = PostgresSessionStore(dsn, min_size=1, max_size=4)
        tasks = PostgresTaskStore(dsn, min_size=1, max_size=2)

        async def execute(statement, parameters=(), *, foreign_keys=True):
            async with await psycopg.AsyncConnection.connect(dsn) as connection:
                if not foreign_keys:
                    await connection.execute("SET session_replication_role = replica")
                await connection.execute(statement.replace("?", "%s"), parameters)
                await connection.commit()

        try:
            await scenario(
                RetentionHarness(store=store, tasks=tasks, execute=execute, postgres=True)
            )
        finally:
            await store.close()
            await tasks.close()

    asyncio.run(run())


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.__name__)
def test_sqlite_session_retention_conformance(sqlite_resources, scenario) -> None:
    _sqlite_scenario(sqlite_resources, scenario)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.__name__)
def test_postgres_session_retention_conformance(retention_postgres_dsn, scenario) -> None:
    _postgres_scenario(retention_postgres_dsn, scenario)


def test_session_retention_policy_contract() -> None:
    asyncio.run(assert_policy_contract())


def test_retention_is_an_explicit_store_capability() -> None:
    from cayu.sessions.base import InMemorySessionStore
    from cayu.storage.postgres import PostgresSessionStore

    assert SessionStore.supports_storage_retention is False
    assert InMemorySessionStore.supports_storage_retention is False
    assert SQLiteSessionStore.supports_storage_retention is True
    assert PostgresSessionStore.supports_storage_retention is True


def test_inspect_session_retention_reports_direct_protections(sqlite_resources) -> None:
    async def run() -> None:
        async with sqlite_resources as resources:
            path = resources.path()
            store = resources.own(SQLiteSessionStore(path))
            tasks = resources.own(SQLiteTaskStore(path))
            harness = RetentionHarness(store=store, tasks=tasks)
            await create_retention_session(harness, "kept")
            await create_retention_session(harness, "free")
            from cayu.tasks.creation import TaskCreate

            await tasks.create_task(TaskCreate(type="test", task_id="t", session_id="kept"))
            result = await store.inspect_session_retention(["kept", "free", "missing"])
            assert result == {"kept": (RetentionProtection.LIVE_TASK,), "free": ()}

    asyncio.run(run())


def test_apply_requires_audit_tables_but_dry_run_does_not(sqlite_resources) -> None:
    async def run() -> None:
        async with sqlite_resources as resources:
            path = resources.path()
            clock = RetentionClock()
            store = resources.own(SQLiteSessionStore(path, ownership_clock=clock))
            harness = RetentionHarness(store=store, clock=clock)
            await create_retention_session(harness, "old")
            await harness.age()
            with contextlib.closing(sqlite3.connect(path, timeout=5)) as connection:
                connection.execute("DROP TABLE cayu_storage_retention_entries")
                connection.execute("DROP TABLE cayu_storage_retention_runs")
                connection.commit()
            report = await store.apply_retention_policy(policy())
            assert [item.item_id for item in report.items] == ["old"]
            with pytest.raises(RetentionAuditUnavailable, match="cayu storage migrate"):
                await store.apply_retention_policy(policy(dry_run=False))
            assert len(await store.load_events("old")) == 6

    asyncio.run(run())


def test_read_only_store_can_dry_run_but_not_apply(sqlite_resources) -> None:
    async def run() -> None:
        async with sqlite_resources as resources:
            path = resources.path()
            writer = resources.own(SQLiteSessionStore(path))
            await create_retention_session(RetentionHarness(store=writer), "kept")
            reader = resources.own(
                SQLiteSessionStore(path, schema_mode=SchemaMode.VALIDATE, read_only=True)
            )
            report = await reader.apply_retention_policy(
                SessionRetentionPolicy(older_than=timedelta(seconds=1), mode=RetentionMode.DELETE)
            )
            assert report.dry_run is True
            with pytest.raises(PermissionError):
                await reader.apply_retention_policy(policy(dry_run=False))

    asyncio.run(run())
