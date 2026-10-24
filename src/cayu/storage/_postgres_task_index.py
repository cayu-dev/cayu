"""PostgreSQL task index streams a single repeatable-read snapshot."""

from __future__ import annotations

from typing import TYPE_CHECKING, LiteralString, cast
from uuid import uuid4

from cayu.resource_access import current_data_bounds
from cayu.storage import _postgres_support as pg_support
from cayu.tasks.access import sql_predicate
from cayu.tasks.index import (
    TaskIndexAccumulator,
    TaskIndexPredicate,
    TaskIndexQuery,
    TaskIndexResult,
)

if TYPE_CHECKING:
    from cayu.storage.tasks_postgres import PostgresTaskStore


async def query_index(
    store: PostgresTaskStore, query: TaskIndexQuery, *, matches: TaskIndexPredicate
) -> TaskIndexResult:
    query = TaskIndexQuery.model_validate(query.model_dump(mode="python", warnings=False))
    bounds = await current_data_bounds("tasks")
    clauses, params = store._task_filter_clauses(query.filters)
    if bounds is not None:
        clause, values = sql_predicate(bounds, postgres=True)
        clauses.append(clause)
        params.extend(values)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    order = pg_support.task_order_sql(query.filters.order_by)
    result = TaskIndexAccumulator(query, matches)
    async with store._ready_connection() as connection:
        async with connection.cursor() as control:
            await control.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            await control.execute("SELECT transaction_timestamp()")
            row = await control.fetchone()
            if row is None:
                raise RuntimeError("Task index snapshot timestamp is unavailable.")
            as_of = row[0]
        async with connection.cursor(name=f"task_index_{uuid4().hex}") as cursor:
            await cursor.execute(
                cast(
                    "LiteralString",
                    f'SELECT {pg_support.TASK_COLUMNS} FROM cayu_tasks {where} ORDER BY {order}, id COLLATE "C" ASC',
                ),
                params,
            )
            while rows := await cursor.fetchmany(256):
                for row in rows:
                    result.visit(pg_support.task_from_row(row))
    return result.result(as_of)
