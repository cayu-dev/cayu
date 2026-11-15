"""Environment counts read a covering index, not every session row."""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3

import pytest

from cayu.sessions.queries import SessionQuery
from cayu.storage.sqlite import SQLiteSessionStore

_ACTIVITY = (
    "SELECT environment_name, COUNT(*), MAX(last_activity_at) FROM cayu_sessions "
    "WHERE environment_name IN ({marks}) GROUP BY environment_name"
)


def test_sqlite_environment_activity_reads_only_the_covering_index(tmp_path) -> None:
    path = tmp_path / "sessions.sqlite"

    async def create() -> None:
        store = SQLiteSessionStore(path)
        try:
            # Any read opens the database and creates its schema.
            await store.list_sessions(SessionQuery(limit=1))
        finally:
            await store.close()

    asyncio.run(create())
    with contextlib.closing(sqlite3.connect(path)) as connection:
        plan = "\n".join(
            str(row[-1])
            for row in connection.execute(
                "EXPLAIN QUERY PLAN " + _ACTIVITY.format(marks="?, ?"), ("a", "b")
            )
        )
    assert "COVERING INDEX idx_cayu_sessions_environment_activity" in plan, plan


def test_postgres_environment_activity_can_read_only_the_index(request) -> None:
    import psycopg

    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    dsn = request.getfixturevalue("postgres_dsn")

    async def scenario() -> str:
        store = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
        try:
            await store.ensure_schema()
        finally:
            await store.close()
        async with await psycopg.AsyncConnection.connect(dsn) as conn, conn.cursor() as cur:
            await cur.execute("SET enable_seqscan = off")
            await cur.execute("SET enable_bitmapscan = off")
            await cur.execute(
                "EXPLAIN " + _ACTIVITY.format(marks="%s, %s"),  # type: ignore[arg-type]
                ("a", "b"),
            )
            return "\n".join(row[0] for row in await cur.fetchall())

    plan = asyncio.run(scenario())
    assert "Index Only Scan using idx_cayu_sessions_environment_activity" in plan, plan


pytestmark = pytest.mark.store_semantics
