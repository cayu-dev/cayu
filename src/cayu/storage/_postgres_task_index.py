"""PostgreSQL task index reads from a single repeatable-read snapshot.

A native read filters, pages and counts in SQL (see ``_task_index_sql``). Before
reading counts it folds pending ``cayu_task_count_changes`` into
``cayu_task_counts`` in a short transaction of its own; the read is exact either
way, since it sums both.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, LiteralString, cast
from uuid import uuid4

from cayu.resource_access import current_data_bounds
from cayu.storage import _postgres_support as pg_support
from cayu.storage._task_index_schema import POSTGRES_FOLD_TASK_COUNT_CHANGES
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
    from cayu.storage.tasks_postgres import PostgresTaskStore
    from cayu.tasks.records import Task

_DIALECT = TaskIndexDialect(postgres=True)


def _sql(statement: str) -> LiteralString:
    # Statements are built from constant fragments; values bind through %s.
    return cast("LiteralString", statement)


async def _begin_snapshot(cursor) -> Any:
    await cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
    await cursor.execute("SELECT transaction_timestamp()")
    row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("Task index snapshot timestamp is unavailable.")
    return row[0]


async def fold_task_count_changes(store: PostgresTaskStore) -> None:
    """Fold pending count changes, unless another reader is already folding.

    Folding only keeps the changes table short: every count read adds the pending
    changes to the stored counts. A read-only store, which can't write, skips it.
    """

    if store._read_only:
        return
    lock, *fold = POSTGRES_FOLD_TASK_COUNT_CHANGES
    async with store._ready_connection() as connection, connection.cursor() as cursor:
        await cursor.execute(_sql(lock))
        row = await cursor.fetchone()
        if row is None or not row[0]:
            return
        for statement in fold:
            await cursor.execute(_sql(statement))


async def query_index(
    store: PostgresTaskStore,
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
    if plan.uses_counts_table:
        await fold_task_count_changes(store)
    async with store._ready_connection() as connection, connection.cursor() as cursor:
        as_of = await _begin_snapshot(cursor)
        if query.filters.q is not None:
            sql, params = plan.search_scan()
            await cursor.execute(_sql(sql), params)
            page = search_page(
                query,
                await cursor.fetchall(),
                search,
                parse_timestamp=lambda value: value,
                format_timestamp=lambda value: value,
            )
            identities = [key[1] for key in page.keys]
            by_id: dict[str, Task] = {}
            if identities:
                await cursor.execute(
                    _sql(plan.by_identity(pg_support.TASK_COLUMNS, len(identities))),
                    (identities,),
                )
                by_id = {
                    task.id: task
                    for task in (pg_support.task_from_row(row) for row in await cursor.fetchall())
                }
            return page.result([by_id[identity] for identity in identities], as_of)
        sql, params = plan.counts()
        await cursor.execute(_sql(sql), params)
        page = count_page(query, await cursor.fetchall())
        sql, params = plan.page(pg_support.TASK_COLUMNS, lambda value: value)
        await cursor.execute(_sql(sql), params)
        rows = await cursor.fetchall()
    items = [pg_support.task_from_row(row) for row in rows[: query.filters.limit]]
    page.has_more = len(rows) > query.filters.limit
    page.keys = [(getattr(task, plan.order_column), task.id) for task in items]
    return page.result(items, as_of)


async def _visit_index(
    store: PostgresTaskStore, query: TaskIndexQuery, *, matches: TaskIndexPredicate, bounds
) -> TaskIndexResult:
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
            as_of = await _begin_snapshot(control)
        async with connection.cursor(name=f"task_index_{uuid4().hex}") as cursor:
            await cursor.execute(
                _sql(
                    f"SELECT {pg_support.TASK_COLUMNS} FROM cayu_tasks {where} "
                    f'ORDER BY {order}, id COLLATE "C" ASC'
                ),
                params,
            )
            while rows := await cursor.fetchmany(256):
                for row in rows:
                    result.visit(pg_support.task_from_row(row))
    return result.result(as_of)


async def load_tasks(store: PostgresTaskStore, identities: Sequence[str], bounds) -> list[Task]:
    """The visible tasks among ``identities``, in that order, each once."""

    unique = list(dict.fromkeys(identities))
    if not unique:
        return []
    clauses: list[str] = ["id = ANY(%s)"]
    params: list[Any] = [unique]
    if bounds is not None:
        clause, values = sql_predicate(bounds, postgres=True)
        clauses.append(clause)
        params.extend(values)
    async with store._ready_connection() as connection, connection.cursor() as cursor:
        await cursor.execute(
            _sql(f"SELECT {pg_support.TASK_COLUMNS} FROM cayu_tasks WHERE {' AND '.join(clauses)}"),
            params,
        )
        by_id = {
            task.id: task
            for task in (pg_support.task_from_row(row) for row in await cursor.fetchall())
        }
    return [by_id[identity] for identity in unique if identity in by_id]


async def task_counts(store: PostgresTaskStore) -> tuple[list[tuple], list[tuple]]:
    """The stored counts and a recount of ``cayu_tasks``, from one snapshot."""

    await fold_task_count_changes(store)
    async with store._ready_connection() as connection, connection.cursor() as cursor:
        await _begin_snapshot(cursor)
        await cursor.execute(
            "SELECT is_internal, status, type, assigned_agent_name, SUM(n) FROM ("
            "SELECT is_internal, status, type, assigned_agent_name, task_count AS n "
            "FROM cayu_task_counts UNION ALL "
            "SELECT is_internal, status, type, assigned_agent_name, delta AS n "
            "FROM cayu_task_count_changes) AS counted "
            "GROUP BY 1, 2, 3, 4 HAVING SUM(n) <> 0"
        )
        stored = await cursor.fetchall()
        await cursor.execute(
            "SELECT left(type, 5) = 'cayu.', status, type, coalesce(assigned_agent_name, ''), "
            "COUNT(*) FROM cayu_tasks GROUP BY 1, 2, 3, 4"
        )
        recount = await cursor.fetchall()
    return stored, recount
