"""Artifact index operations for the SQLite and Postgres session stores."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Any, cast

from cayu._validation import require_clean_nonblank
from cayu.artifacts._index import (
    ARTIFACT_INDEX_BATCH_MAX,
    ARTIFACT_INDEX_PAGE_MAX,
    ArtifactIndexCursor,
    ArtifactIndexPage,
    ArtifactIndexQuery,
    ArtifactIndexRow,
    ArtifactIndexValues,
    decode_metadata,
    parse_index_time,
)
from cayu.storage import _artifact_index_sql as sql
from cayu.storage._artifact_index_sql import Dialect


def _ids(values: Sequence[str], name: str, *, at_most: int) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise TypeError(f"{name} must be a sequence.")
    page = tuple(dict.fromkeys(require_clean_nonblank(value, name) for value in values))
    if len(page) > at_most:
        raise ValueError(f"An artifact index operation takes at most {at_most} {name} values.")
    return page


def _live_rows(store_id: str, artifacts: Sequence[Any]) -> tuple[ArtifactIndexValues, ...]:
    store_id = require_clean_nonblank(store_id, "store_id")
    rows = tuple(ArtifactIndexValues.of(store_id, artifact) for artifact in artifacts)
    if len(rows) > ARTIFACT_INDEX_BATCH_MAX:
        raise ValueError("An artifact index write takes at most 500 artifacts.")
    return rows


def _page(query: ArtifactIndexQuery, rows: Sequence[Any], total: int | None) -> ArtifactIndexPage:
    decoded = tuple(
        ArtifactIndexRow(store_id=row[0], artifact=decode_metadata(row[1]), title=row[2])
        for row in rows[: query.limit]
    )
    next_cursor = (
        ArtifactIndexCursor.after(query.order_by, decoded[-1])
        if len(rows) > query.limit and decoded
        else None
    )
    return ArtifactIndexPage(rows=decoded, total_count=total, next_cursor=next_cursor)


def _stores(rows: Sequence[Any]) -> dict[str, datetime | None]:
    return {row[0]: None if row[1] is None else parse_index_time(row[1]) for row in rows}


# ---------------------------------------------------------------------------
# SQLite: ``run`` is the store's ``_run_read`` or ``_run_write``.

SQLiteRunner = Callable[[Callable[[sqlite3.Connection], Any]], Any]
_SQLITE: Dialect = "sqlite"


async def sqlite_record_live(
    run_write: SQLiteRunner, store_id: str, artifacts: Sequence[Any], observed_at: datetime
) -> None:
    rows = _live_rows(store_id, artifacts)
    if not rows:
        return
    statement = sql.upsert_live_sql(_SQLITE)

    def write(connection: sqlite3.Connection) -> None:
        with connection:
            connection.executemany(
                statement, [sql.upsert_live_params(_SQLITE, row, observed_at) for row in rows]
            )

    await run_write(write)


async def sqlite_record_deleted(
    run_write: SQLiteRunner, store_id: str, artifact_ids: Sequence[str], deleted_at: datetime
) -> None:
    store_id = require_clean_nonblank(store_id, "store_id")
    ids = _ids(artifact_ids, "artifact_id", at_most=ARTIFACT_INDEX_BATCH_MAX)
    if not ids:
        return
    statement = sql.upsert_tombstone_sql(_SQLITE)

    def write(connection: sqlite3.Connection) -> None:
        with connection:
            connection.executemany(
                statement,
                [sql.upsert_tombstone_params(_SQLITE, store_id, item, deleted_at) for item in ids],
            )

    await run_write(write)


async def sqlite_mark_store(
    run_write: SQLiteRunner, store_id: str, backfilled_at: datetime | None
) -> None:
    store_id = require_clean_nonblank(store_id, "store_id")

    def write(connection: sqlite3.Connection) -> None:
        with connection:
            connection.execute(
                sql.mark_store_sql(_SQLITE), sql.mark_store_params(_SQLITE, store_id, backfilled_at)
            )

    await run_write(write)


async def sqlite_read_stores(
    run_read: SQLiteRunner, store_ids: Sequence[str]
) -> dict[str, datetime | None]:
    ids = _ids(store_ids, "store_id", at_most=ARTIFACT_INDEX_PAGE_MAX)
    if not ids:
        return {}
    statement, params = sql.read_stores_sql(_SQLITE, ids)
    return await run_read(
        lambda connection: _stores(connection.execute(statement, params).fetchall())
    )


async def sqlite_scan(
    run_read: SQLiteRunner, store_id: str, after: str | None, limit: int
) -> tuple[tuple[str, str], ...]:
    store_id = require_clean_nonblank(store_id, "store_id")
    if type(limit) is not int or not 1 <= limit <= ARTIFACT_INDEX_BATCH_MAX:
        raise ValueError("An artifact index scan reads 1..500 rows at a time.")
    statement, params = sql.scan_sql(_SQLITE, store_id, after, limit)
    return await run_read(
        lambda connection: tuple(
            (row[0], row[1]) for row in connection.execute(statement, params).fetchall()
        )
    )


async def sqlite_count_sessions(
    run_read: SQLiteRunner, store_ids: Sequence[str], session_ids: Sequence[str]
) -> dict[str, int]:
    stores = _ids(store_ids, "store_id", at_most=ARTIFACT_INDEX_PAGE_MAX)
    sessions = _ids(session_ids, "session_id", at_most=ARTIFACT_INDEX_PAGE_MAX)
    counts = dict.fromkeys(sessions, 0)
    if not stores or not sessions:
        return counts
    statement, params = sql.count_sessions_sql(_SQLITE, stores, sessions)

    def read(connection: sqlite3.Connection) -> dict[str, int]:
        for session_id, count in connection.execute(statement, params):
            counts[session_id] = int(count)
        return counts

    return await run_read(read)


async def sqlite_query(run_read: SQLiteRunner, query: ArtifactIndexQuery) -> ArtifactIndexPage:
    if type(query) is not ArtifactIndexQuery:
        raise TypeError("query must be an ArtifactIndexQuery.")
    page_statement, page_params = sql.page_sql(_SQLITE, query)
    total_statement, total_params = sql.total_sql(_SQLITE, query)

    def read(connection: sqlite3.Connection) -> ArtifactIndexPage:
        rows = connection.execute(page_statement, page_params).fetchall()
        total = None
        if query.include_total:
            total = int(connection.execute(total_statement, total_params).fetchone()[0])
        return _page(query, rows, total)

    return await run_read(read)


# ---------------------------------------------------------------------------
# Postgres: ``connect`` is the store's ``_connection``.

PostgresConnect = Callable[[], AbstractAsyncContextManager[Any]]
_POSTGRES: Dialect = "postgres"


async def _postgres(connect: PostgresConnect, run: Callable[[Any], Any]) -> Any:
    async with connect() as conn:
        try:
            async with conn.cursor() as cursor:
                result = await run(cursor)
            await conn.commit()
        except BaseException:
            await conn.rollback()
            raise
    return result


async def postgres_record_live(
    connect: PostgresConnect, store_id: str, artifacts: Sequence[Any], observed_at: datetime
) -> None:
    rows = _live_rows(store_id, artifacts)
    if not rows:
        return
    statement = cast("Any", sql.upsert_live_sql(_POSTGRES))

    async def run(cursor: Any) -> None:
        await cursor.executemany(
            statement, [sql.upsert_live_params(_POSTGRES, row, observed_at) for row in rows]
        )

    await _postgres(connect, run)


async def postgres_record_deleted(
    connect: PostgresConnect, store_id: str, artifact_ids: Sequence[str], deleted_at: datetime
) -> None:
    store_id = require_clean_nonblank(store_id, "store_id")
    ids = _ids(artifact_ids, "artifact_id", at_most=ARTIFACT_INDEX_BATCH_MAX)
    if not ids:
        return
    statement = cast("Any", sql.upsert_tombstone_sql(_POSTGRES))

    async def run(cursor: Any) -> None:
        await cursor.executemany(
            statement,
            [sql.upsert_tombstone_params(_POSTGRES, store_id, item, deleted_at) for item in ids],
        )

    await _postgres(connect, run)


async def postgres_mark_store(
    connect: PostgresConnect, store_id: str, backfilled_at: datetime | None
) -> None:
    store_id = require_clean_nonblank(store_id, "store_id")

    async def run(cursor: Any) -> None:
        await cursor.execute(
            cast("Any", sql.mark_store_sql(_POSTGRES)),
            sql.mark_store_params(_POSTGRES, store_id, backfilled_at),
        )

    await _postgres(connect, run)


async def postgres_read_stores(
    connect: PostgresConnect, store_ids: Sequence[str]
) -> dict[str, datetime | None]:
    ids = _ids(store_ids, "store_id", at_most=ARTIFACT_INDEX_PAGE_MAX)
    if not ids:
        return {}
    statement, params = sql.read_stores_sql(_POSTGRES, ids)

    async def run(cursor: Any) -> dict[str, datetime | None]:
        await cursor.execute(cast("Any", statement), params)
        return _stores(await cursor.fetchall())

    return await _postgres(connect, run)


async def postgres_scan(
    connect: PostgresConnect, store_id: str, after: str | None, limit: int
) -> tuple[tuple[str, str], ...]:
    store_id = require_clean_nonblank(store_id, "store_id")
    if type(limit) is not int or not 1 <= limit <= ARTIFACT_INDEX_BATCH_MAX:
        raise ValueError("An artifact index scan reads 1..500 rows at a time.")
    statement, params = sql.scan_sql(_POSTGRES, store_id, after, limit)

    async def run(cursor: Any) -> tuple[tuple[str, str], ...]:
        await cursor.execute(cast("Any", statement), params)
        return tuple((row[0], row[1]) for row in await cursor.fetchall())

    return await _postgres(connect, run)


async def postgres_count_sessions(
    connect: PostgresConnect, store_ids: Sequence[str], session_ids: Sequence[str]
) -> dict[str, int]:
    stores = _ids(store_ids, "store_id", at_most=ARTIFACT_INDEX_PAGE_MAX)
    sessions = _ids(session_ids, "session_id", at_most=ARTIFACT_INDEX_PAGE_MAX)
    counts = dict.fromkeys(sessions, 0)
    if not stores or not sessions:
        return counts
    statement, params = sql.count_sessions_sql(_POSTGRES, stores, sessions)

    async def run(cursor: Any) -> dict[str, int]:
        await cursor.execute(cast("Any", statement), params)
        for session_id, count in await cursor.fetchall():
            counts[session_id] = int(count)
        return counts

    return await _postgres(connect, run)


async def postgres_query(connect: PostgresConnect, query: ArtifactIndexQuery) -> ArtifactIndexPage:
    if type(query) is not ArtifactIndexQuery:
        raise TypeError("query must be an ArtifactIndexQuery.")
    page_statement, page_params = sql.page_sql(_POSTGRES, query)
    total_statement, total_params = sql.total_sql(_POSTGRES, query)

    async def run(cursor: Any) -> ArtifactIndexPage:
        await cursor.execute(cast("Any", page_statement), page_params)
        rows = await cursor.fetchall()
        total = None
        if query.include_total:
            await cursor.execute(cast("Any", total_statement), total_params)
            total = int((await cursor.fetchone())[0])
        return _page(query, rows, total)

    return await _postgres(connect, run)
