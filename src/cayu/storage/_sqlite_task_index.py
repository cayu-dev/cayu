"""SQLite task index scan keeps connection ownership through cancellation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu.resource_access import current_data_bounds
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage._sqlite_connection import _run_off_thread_with_connection_ownership
from cayu.tasks.access import sql_predicate
from cayu.tasks.index import (
    TaskIndexAccumulator,
    TaskIndexPredicate,
    TaskIndexQuery,
    TaskIndexResult,
)

if TYPE_CHECKING:
    from cayu.storage.tasks_sqlite import SQLiteTaskStore


async def query_index(
    store: SQLiteTaskStore, query: TaskIndexQuery, *, matches: TaskIndexPredicate
) -> TaskIndexResult:
    query = TaskIndexQuery.model_validate(query.model_dump(mode="python", warnings=False))
    bounds = await current_data_bounds("tasks")
    clauses, params = store._task_filter_clauses(query.filters)
    if bounds is not None:
        clause, values = sql_predicate(bounds, postgres=False)
        clauses.append(clause)
        params.extend(values)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    order = sqlite_records.task_order_sql(query.filters.order_by)

    def read(connection):
        result = TaskIndexAccumulator(query, matches)
        as_of = store._clock()
        cursor = connection.execute(
            f"SELECT * FROM cayu_tasks {where} ORDER BY {order}, id ASC", params
        )
        try:
            while rows := cursor.fetchmany(256):
                for row in rows:
                    result.visit(sqlite_records.task_from_row(row))
            return result.result(as_of)
        finally:
            cursor.close()

    return await _run_off_thread_with_connection_ownership(
        store._lock, store._connection, read, interrupt_on_cancellation=True
    )
