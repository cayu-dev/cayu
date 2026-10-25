"""Revision 129: stored transcript message sizes and the root-session-tree index."""

from __future__ import annotations

import asyncio
import random
import sqlite3
import string

import pytest

from cayu.messages import Message
from cayu.sessions import RunRequest, SessionIdentity, SessionQuery
from cayu.storage import SQLiteSessionStore
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage import migrations as schema
from cayu.storage.migrations import SchemaMode

_IDENTITY = SessionIdentity(provider_name="fake", model="fake")
_INDEX = "idx_cayu_sessions_root_session_created"


async def _seed(store) -> None:
    await store.create(
        RunRequest(agent_name="scout", session_id="root", messages=[]), identity=_IDENTITY
    )
    for index in range(3):
        await store.create(
            RunRequest(
                agent_name="hunter",
                session_id=f"child-{index}",
                parent_session_id="root",
                messages=[],
            ),
            identity=_IDENTITY,
        )
    await store.append_transcript_messages(
        "root", [Message.text("user", "héllo"), Message.text("assistant", "x" * 5000)]
    )


def test_revision_129_is_additive_and_required_by_session_stores() -> None:
    from cayu.storage import postgres, sqlite

    revision = schema.revision(129)
    assert revision.kind is schema.RevisionKind.ADDITIVE
    assert revision.compatible_from == schema.revision(128).compatible_from
    assert schema.LATEST_REVISION == 129
    assert sqlite._SQLITE_SESSION_MIN_REQUIRED_REVISION == 129
    assert postgres._POSTGRES_SESSION_MIN_REQUIRED_REVISION == 129


def test_sqlite_revision_129_backfills_sizes_fills_old_writers_and_indexes_trees(
    tmp_path,
) -> None:
    path = tmp_path / "sessions.sqlite"

    async def seed() -> None:
        store = SQLiteSessionStore(path)
        try:
            await _seed(store)
        finally:
            await store.close()

    asyncio.run(seed())
    connection = sqlite3.connect(path)
    try:
        sizes = connection.execute(
            "SELECT message_bytes, length(CAST(message_json AS BLOB)) "
            "FROM cayu_transcript_messages ORDER BY sequence"
        ).fetchall()
        assert len(sizes) == 2 and all(stored == measured for stored, measured in sizes)
        # Rewind to revision 128.
        connection.execute("DROP TRIGGER cayu_transcript_messages_bytes_insert")
        connection.execute(f"DROP INDEX {_INDEX}")
        connection.execute("ALTER TABLE cayu_transcript_messages DROP COLUMN message_bytes")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 129")
        connection.execute("PRAGMA user_version = 128")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(schema.SchemaTooOld, match="cayu storage migrate"):
        SQLiteSessionStore(path, schema_mode=SchemaMode.VALIDATE)

    async def migrate() -> None:
        store = SQLiteSessionStore(path, schema_mode=SchemaMode.MIGRATE)
        try:
            window = await store.load_bounded_transcript_window(
                "root", start_index=0, limit=5, max_record_bytes=100, max_total_bytes=100
            )
            assert window.records[0].message is not None
            assert window.records[1].message is None and window.records[1].stored_bytes > 5000

            def old_writer(connection: sqlite3.Connection) -> None:
                # A writer that predates the column leaves it out; the trigger fills it.
                connection.execute(
                    "INSERT INTO cayu_transcript_messages "
                    "(session_id, role, interaction_id, message_json, "
                    "transcript_search_document) "
                    "SELECT session_id, role, interaction_id, message_json, "
                    "transcript_search_document FROM cayu_transcript_messages "
                    "WHERE sequence = 1"
                )
                connection.commit()

            await store._run_write(old_writer)
        finally:
            await store.close()

    asyncio.run(migrate())
    connection = sqlite3.connect(path)
    try:
        backfilled = connection.execute(
            "SELECT message_bytes, length(CAST(message_json AS BLOB)) "
            "FROM cayu_transcript_messages ORDER BY sequence"
        ).fetchall()
        assert backfilled == [*sizes, sizes[0]]
        plan = session_store_sql.build_session_query_sql(
            SessionQuery(root_session_id="root", order_by="created_at_desc", limit=2),
            dialect=_sqlite_dialect(),
        )
        explained = " ".join(
            str(row[3])
            for row in connection.execute(
                f"EXPLAIN QUERY PLAN SELECT id FROM cayu_sessions {plan.page_where_sql} "
                f"ORDER BY {plan.order_sql} {plan.pagination_sql}",
                plan.page_params,
            )
        )
        assert _INDEX in explained and "TEMP B-TREE" not in explained
        connection.execute("DROP TRIGGER cayu_transcript_messages_bytes_insert")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises((schema.SchemaError, RuntimeError), match="bytes_insert"):
        SQLiteSessionStore(path, schema_mode=SchemaMode.VALIDATE)


def _sqlite_dialect():
    from cayu.storage import _sqlite_session_queries

    return _sqlite_session_queries.SQL_DIALECT


def test_postgres_revision_129_backfills_sizes_fills_old_writers_and_indexes_trees(
    postgres_dsn,
) -> None:
    import psycopg

    from cayu.storage import PostgresSessionStore, _postgres_session_queries

    async def run() -> None:
        # Incompressible maximum-length IDs used to overflow the root index:
        # it stored the root ID twice for a root session (over 4 KiB total).
        long_root = "".join(
            random.Random(129).choices(string.ascii_letters + string.digits, k=2048)
        )
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await _seed(store)
            await store.create(
                RunRequest(agent_name="scout", session_id=long_root, messages=[]),
                identity=_IDENTITY,
            )
            await store.create(
                RunRequest(
                    agent_name="hunter",
                    session_id="long-root-child",
                    parent_session_id=long_root,
                    messages=[],
                ),
                identity=_IDENTITY,
            )
        finally:
            await store.close()
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            await conn.execute(
                "DROP TRIGGER cayu_record_transcript_message_bytes ON cayu_transcript_messages"
            )
            await conn.execute("DROP FUNCTION cayu_record_transcript_message_bytes()")
            await conn.execute(f"DROP INDEX {_INDEX}")
            await conn.execute("ALTER TABLE cayu_transcript_messages DROP COLUMN message_bytes")
            await conn.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 129")
            await conn.commit()

        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises(schema.SchemaTooOld, match="cayu storage migrate"):
                await validator.ensure_schema()
        finally:
            await validator.close()
        migrator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.MIGRATE)
        try:
            await migrator.ensure_schema()
            tree = await migrator.list_sessions(SessionQuery(root_session_id=long_root))
            assert {session.id for session in tree.sessions} == {long_root, "long-root-child"}
            window = await migrator.load_bounded_transcript_window(
                "root", start_index=0, limit=5, max_record_bytes=100, max_total_bytes=100
            )
            assert window.records[0].message is not None
            assert window.records[1].message is None and window.records[1].stored_bytes > 5000
        finally:
            await migrator.close()

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            cur = await conn.execute(
                "SELECT message_bytes = octet_length(message::text) FROM cayu_transcript_messages"
            )
            assert [row[0] for row in await cur.fetchall()] == [True, True]
            await conn.execute(
                "INSERT INTO cayu_transcript_messages "
                "(session_id, interaction_id, message, transcript_search_document) "
                "SELECT session_id, interaction_id, message, transcript_search_document "
                "FROM cayu_transcript_messages ORDER BY sequence LIMIT 1"
            )
            cur = await conn.execute(
                "SELECT message_bytes = octet_length(message::text) "
                "FROM cayu_transcript_messages ORDER BY sequence DESC LIMIT 1"
            )
            assert await cur.fetchone() == (True,)
            plan = session_store_sql.build_session_query_sql(
                SessionQuery(root_session_id="root", order_by="created_at_desc", limit=2),
                dialect=_postgres_session_queries.SQL_DIALECT,
            )
            await conn.execute("SET LOCAL enable_seqscan = off")
            cur = await conn.execute(
                f"EXPLAIN SELECT id FROM cayu_sessions {plan.page_where_sql} "  # ty: ignore[invalid-argument-type]
                f"ORDER BY {plan.order_sql} {plan.pagination_sql}",
                plan.page_params,
            )
            explained = " ".join(str(row[0]) for row in await cur.fetchall())
            assert _INDEX in explained and "Sort" not in explained
            await conn.rollback()
            await conn.execute(
                "DROP TRIGGER cayu_record_transcript_message_bytes ON cayu_transcript_messages"
            )
            await conn.commit()
        validator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        try:
            with pytest.raises((schema.SchemaError, RuntimeError), match="message size trigger"):
                await validator.ensure_schema()
        finally:
            await validator.close()

    asyncio.run(run())
