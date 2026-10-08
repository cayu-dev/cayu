"""Cleanup acknowledgements require exact, session-independent native storage."""

import sqlite3

import pytest

from cayu.sessions.base import InMemorySessionStore
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize(
    "method",
    [
        "_read_native_producer_attachment",
        "_complete_native_producer_cleanup",
        "_read_completed_native_producer_cleanup",
        "_retire_native_producer_cleanup",
        "prepare_model_completion_stage",
        "_prepare_model_completion_stage_atomic",
        "deliver_queued_session_messages",
    ],
)
def test_opaque_cleanup_override_does_not_inherit_native_qualification(method):
    async def opaque(*args, **kwargs):
        raise AssertionError("Opaque cleanup must not run")

    store = type("OpaqueCleanupStore", (InMemorySessionStore,), {method: opaque})()
    assert not store._supports_producer_attachment_protocol()


@pytest.mark.anyio
@pytest.mark.parametrize("corruption", ["missing", "foreign_key", "wrong_key"])
async def test_sqlite_cleanup_schema_refuses_missing_or_session_owned_receipts(
    tmp_path, corruption
):
    path = tmp_path / "cleanup.sqlite"
    store = SQLiteSessionStore(path)
    await store.close()
    connection = sqlite3.connect(path)
    try:
        assert (
            connection.execute("PRAGMA foreign_key_list(cayu_producer_cleanup_receipts)").fetchall()
            == []
        )
        connection.execute("DROP TABLE cayu_producer_cleanup_receipts")
        if corruption != "missing":
            connection.execute(
                "CREATE TABLE cayu_producer_cleanup_receipts (operation_key TEXT NOT NULL "
                + (
                    "PRIMARY KEY REFERENCES cayu_sessions(id) ON DELETE CASCADE"
                    if corruption == "foreign_key"
                    else ""
                )
                + ", receipt_json TEXT NOT NULL)"
            )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(RuntimeError, match="producer cleanup receipt table"):
        SQLiteSessionStore(path, schema_mode=SchemaMode.VALIDATE)


@pytest.mark.anyio
@pytest.mark.parametrize("corruption", ["missing", "foreign_key", "wrong_key"])
async def test_postgres_cleanup_schema_refuses_missing_or_session_owned_receipts(
    postgres_dsn, corruption
):
    import psycopg

    from cayu.storage.postgres import PostgresSessionStore

    store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
    try:
        await store._ensure_ready()
    finally:
        await store.close()
    async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
        if corruption == "missing":
            await connection.execute("DROP TABLE cayu_producer_cleanup_receipts")
        elif corruption == "foreign_key":
            await connection.execute(
                "ALTER TABLE cayu_producer_cleanup_receipts ADD FOREIGN KEY (operation_key) "
                "REFERENCES cayu_sessions(id) ON DELETE CASCADE"
            )
        else:
            await connection.execute(
                "ALTER TABLE cayu_producer_cleanup_receipts DROP CONSTRAINT cayu_producer_cleanup_receipts_pkey"
            )
    reopened = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
    try:
        with pytest.raises(RuntimeError, match="producer cleanup|Producer cleanup"):
            await reopened._ensure_ready()
    finally:
        await reopened.close()
        # postgres_dsn is module-scoped. Restore only this test-created empty
        # table so the next corruption case starts from the qualified schema.
        from cayu.storage._postgres_schema_history import _MIGRATION_STEPS

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            await connection.execute("DROP TABLE IF EXISTS cayu_producer_cleanup_receipts")
            for statement in _MIGRATION_STEPS[110]:
                await connection.execute(statement)


@pytest.mark.anyio
@pytest.mark.parametrize("corruption", ["missing_fence", "missing_index", "wrong_index"])
async def test_sqlite_retirement_requires_durable_fence_and_ordered_index(tmp_path, corruption):
    path = tmp_path / "retirement.sqlite"
    store = SQLiteSessionStore(path)
    await store.close()
    with sqlite3.connect(path) as connection:
        if corruption == "missing_fence":
            connection.execute("DROP TABLE cayu_producer_cleanup_retirements")
        else:
            connection.execute("DROP INDEX idx_cayu_producer_cleanup_namespace")
            if corruption == "wrong_index":
                connection.execute(
                    "CREATE INDEX idx_cayu_producer_cleanup_namespace ON cayu_producer_cleanup_receipts(generation, namespace_key, operation_key)"
                )
    with pytest.raises(RuntimeError, match="producer cleanup"):
        SQLiteSessionStore(path, schema_mode=SchemaMode.VALIDATE)


@pytest.mark.anyio
@pytest.mark.parametrize("corruption", ["missing_fence", "missing_index", "wrong_index"])
async def test_postgres_retirement_requires_durable_fence_and_ordered_index(
    postgres_dsn, corruption
):
    import psycopg

    from cayu.storage._postgres_schema_history import _MIGRATION_STEPS
    from cayu.storage.postgres import PostgresSessionStore

    store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
    try:
        await store._ensure_ready()
    finally:
        await store.close()
    async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
        if corruption == "missing_fence":
            await connection.execute("DROP TABLE cayu_producer_cleanup_retirements")
        else:
            await connection.execute("DROP INDEX idx_cayu_producer_cleanup_namespace")
            if corruption == "wrong_index":
                await connection.execute(
                    "CREATE INDEX idx_cayu_producer_cleanup_namespace ON cayu_producer_cleanup_receipts(generation, namespace_key, operation_key)"
                )
    reopened = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
    try:
        with pytest.raises(RuntimeError, match="Producer cleanup"):
            await reopened._ensure_ready()
    finally:
        await reopened.close()
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            await connection.execute("DROP INDEX IF EXISTS idx_cayu_producer_cleanup_namespace")
            await connection.execute("DROP TABLE IF EXISTS cayu_producer_cleanup_retirements")
            for statement in _MIGRATION_STEPS[110]:
                await connection.execute(statement)
