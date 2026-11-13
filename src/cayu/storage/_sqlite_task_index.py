"""SQLite task index reads on dedicated read connections.

Task reads run on read-only WAL connections, so a list or task page never waits
behind the writer's lock. A native read filters, pages and counts in SQL from one
read snapshot (see ``_task_index_sql``); a read with a caller predicate still
visits every eligible task.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from cayu.resource_access import current_data_bounds
from cayu.storage import _sqlite_connection as sqlite_connection
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage._phase_timing import TimedStoreLock
from cayu.storage._sqlite_connection import _run_off_thread_with_connection_ownership
from cayu.storage._task_index_sql import (
    TaskIndexDialect,
    count_page,
    eligibility_filters,
    plan_native_read,
    search_page,
)
from cayu.tasks.access import sql_predicate
from cayu.tasks.index import (
    TaskIndexAccumulator,
    TaskIndexPredicate,
    TaskIndexQuery,
    TaskIndexResult,
    TaskSearchMatch,
)

if TYPE_CHECKING:
    from cayu.storage.tasks_sqlite import SQLiteTaskStore
    from cayu.tasks.records import Task

_T = TypeVar("_T")
_DIALECT = TaskIndexDialect(postgres=False)

#: Read connections per task store. Each runs one read at a time.
TASK_READERS = 2


class TaskReadConnections:
    """A small pool of read-only connections, or the writer for in-memory stores."""

    def __init__(
        self,
        path: Path | None,
        connect: Callable[[Path], sqlite3.Connection],
        *,
        writer_lock: asyncio.Lock,
        writer: sqlite3.Connection,
    ) -> None:
        self._readers: list[tuple[asyncio.Lock, sqlite3.Connection]] = []
        if path is None:
            self._readers.append((writer_lock, writer))
            self._shared = True
        else:
            self._shared = False
            try:
                for _ in range(TASK_READERS):
                    self._readers.append((TimedStoreLock(), connect(path)))
            except BaseException:
                self._close_connections()
                raise
        self._available: asyncio.LifoQueue[tuple[asyncio.Lock, sqlite3.Connection]] = (
            asyncio.LifoQueue()
        )
        for reader in reversed(self._readers):
            self._available.put_nowait(reader)
        self._closed = False

    async def run(self, read: Callable[[sqlite3.Connection], _T]) -> _T:
        reader = await self._available.get()
        try:
            if self._closed:
                raise RuntimeError("SQLite task store is closed.")
            lock, connection = reader
            return await _run_off_thread_with_connection_ownership(
                lock, connection, read, interrupt_on_cancellation=True
            )
        finally:
            self._available.put_nowait(reader)

    async def close(self) -> None:
        self._closed = True
        if self._shared:
            return
        for lock, connection in self._readers:
            async with lock:
                connection.close()

    def _close_connections(self) -> None:
        for _, connection in self._readers:
            connection.close()


def _bounds_clauses(bounds) -> tuple[list[str], list[object]]:
    if bounds is None:
        return [], []
    clause, values = sql_predicate(bounds, postgres=False)
    return [clause], list(values)


async def query_index(
    store: SQLiteTaskStore,
    query: TaskIndexQuery,
    *,
    matches: TaskIndexPredicate | None = None,
    search: TaskSearchMatch | None = None,
) -> TaskIndexResult:
    query = TaskIndexQuery.model_validate(query.model_dump(mode="python", warnings=False))
    bounds = await current_data_bounds("tasks")
    if matches is not None:
        return await _visit_index(store, query, matches=matches, bounds=bounds)
    plan = plan_native_read(
        query,
        store._task_list_clauses(eligibility_filters(query), bounds),
        dialect=_DIALECT,
        bounded=bounds is not None,
        search=search is not None,
    )

    def read(connection: sqlite3.Connection) -> TaskIndexResult:
        with sqlite_connection._transaction(connection, begin_immediate=False):
            as_of = store._clock()
            if query.filters.q is not None:
                sql, params = plan.search_scan()
                page = search_page(
                    query,
                    connection.execute(sql, params),
                    search,
                    parse_timestamp=sqlite_records.parse_datetime,
                    format_timestamp=sqlite_records.format_datetime,
                )
                identities = [key[1] for key in page.keys]
                rows = (
                    connection.execute(
                        plan.by_identity("*", len(identities)), identities
                    ).fetchall()
                    if identities
                    else []
                )
                by_id = {row["id"]: row for row in rows}
                items = [sqlite_records.task_from_row(by_id[identity]) for identity in identities]
                return page.result(items, as_of)
            sql, params = plan.counts()
            page = count_page(query, connection.execute(sql, params).fetchall())
            sql, params = plan.page("*", sqlite_records.format_datetime)
            rows = connection.execute(sql, params).fetchall()
            items = [sqlite_records.task_from_row(row) for row in rows[: query.filters.limit]]
            page.has_more = len(rows) > query.filters.limit
            page.keys = [(getattr(task, plan.order_column), task.id) for task in items]
            return page.result(items, as_of)

    return await store._task_readers.run(read)


async def _visit_index(
    store: SQLiteTaskStore, query: TaskIndexQuery, *, matches: TaskIndexPredicate, bounds
) -> TaskIndexResult:
    clauses, params = store._task_filter_clauses(query.filters)
    bound_clauses, bound_params = _bounds_clauses(bounds)
    clauses.extend(bound_clauses)
    params.extend(bound_params)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    order = sqlite_records.task_order_sql(query.filters.order_by)

    def read(connection: sqlite3.Connection) -> TaskIndexResult:
        result = TaskIndexAccumulator(query, matches)
        with sqlite_connection._transaction(connection, begin_immediate=False):
            as_of = store._clock()
            cursor = connection.execute(
                f"SELECT * FROM cayu_tasks {where} ORDER BY {order}, id ASC", params
            )
            try:
                while rows := cursor.fetchmany(256):
                    for row in rows:
                        result.visit(sqlite_records.task_from_row(row))
            finally:
                cursor.close()
        return result.result(as_of)

    return await store._task_readers.run(read)


async def load_tasks(store: SQLiteTaskStore, identities: Sequence[str], bounds) -> list[Task]:
    """The visible tasks among ``identities``, in that order, each once."""

    unique = list(dict.fromkeys(identities))
    if not unique:
        return []
    clauses, params = _bounds_clauses(bounds)
    clauses.insert(0, f"id IN ({_DIALECT.marks(len(unique))})")
    sql = f"SELECT * FROM cayu_tasks WHERE {' AND '.join(clauses)}"

    def read(connection: sqlite3.Connection) -> list[Task]:
        rows = connection.execute(sql, [*unique, *params]).fetchall()
        by_id = {row["id"]: sqlite_records.task_from_row(row) for row in rows}
        return [by_id[identity] for identity in unique if identity in by_id]

    return await store._task_readers.run(read)


async def task_counts(store: SQLiteTaskStore) -> tuple[list[tuple], list[tuple]]:
    """The stored counts and a recount of ``cayu_tasks``, from one snapshot."""

    def read(connection: sqlite3.Connection) -> tuple[list[tuple], list[tuple]]:
        with sqlite_connection._transaction(connection, begin_immediate=False):
            stored = connection.execute(
                "SELECT is_internal, status, type, assigned_agent_name, task_count "
                "FROM cayu_task_counts WHERE task_count != 0"
            ).fetchall()
            recount = connection.execute(
                "SELECT substr(type, 1, 5) = 'cayu.', status, type, "
                "coalesce(assigned_agent_name, ''), COUNT(*) FROM cayu_tasks GROUP BY 1, 2, 3, 4"
            ).fetchall()
        return [tuple(row) for row in stored], [tuple(row) for row in recount]

    return await store._task_readers.run(read)
