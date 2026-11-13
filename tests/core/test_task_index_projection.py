"""Trigger-maintained task counts and edges agree with a recount for random histories.

Each history creates tasks through the store, then changes their status, assigned
agent and prerequisites and deletes some of them with plain SQL, as an older
binary or an operator would. After it, the stored counts must equal a recount, and
every native index read (SQL filters, keyset pages, counts) must equal the
reference read that visits every task in Python.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
from typing import LiteralString, cast

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cayu.storage._task_index_schema import CASEFOLD_TO_ASCII
from cayu.tasks import TaskCreate
from cayu.tasks.index import (
    TASK_SEARCH_FIELDS,
    TaskIndexQuery,
    compare_task_counts,
    decode_task_index_cursor,
)
from cayu.tasks.queries import TaskOrder, TaskQuery
from cayu.tasks.records import TaskStatus

_TYPES = ("demo.review", "demo.build", "cayu.internal")
_AGENTS = (None, "reviewer", "builder")
# Statuses a task outside a graph may hold without other fields.
_PLAIN_STATUSES = (
    TaskStatus.PENDING,
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
)
_GRAPH = "graph-index-test"

_operation = st.one_of(
    st.tuples(st.just("create"), st.integers(0, len(_TYPES) - 1), st.integers(0, 2)),
    st.tuples(st.just("status"), st.integers(0, 40), st.sampled_from(_PLAIN_STATUSES)),
    st.tuples(st.just("agent"), st.integers(0, 40), st.integers(0, 2)),
    st.tuples(st.just("prerequisites"), st.integers(0, 40), st.sets(st.integers(0, 40))),
    st.tuples(st.just("delete"), st.integers(0, 40), st.just(None)),
)
_history = st.lists(_operation, min_size=1, max_size=30)


def test_casefold_to_ascii_lists_every_such_character() -> None:
    expected = "".join(
        chr(code)
        for code in range(0x80, 0x110000)
        if unicodedata.category(chr(code)) != "Cs"
        and any(ord(folded) < 128 for folded in chr(code).casefold())
    )
    assert expected == CASEFOLD_TO_ASCII


def test_compare_task_counts_reports_each_difference() -> None:
    check = compare_task_counts(
        [(0, "pending", "a", "", 2), (1, "failed", "cayu.x", "agent", 1)],
        [(0, "pending", "a", "", 3), (1, "failed", "cayu.x", "agent", 1)],
    )
    assert not check.matches
    assert check.by_status["pending"] == 2
    assert check.recounted_by_status["pending"] == 3
    [difference] = check.differences
    assert (difference.type, difference.stored, difference.recounted) == ("a", 2, 3)
    assert difference.assigned_agent_name is None


class _SQLite:
    def __init__(self) -> None:
        self.path = Path(tempfile.mkdtemp()) / "tasks.db"

    @asynccontextmanager
    async def store(self):
        from cayu.storage.tasks_sqlite import SQLiteTaskStore

        store = SQLiteTaskStore(self.path)
        try:
            yield store
        finally:
            await store.close()

    async def execute(self, statement: str, params: tuple) -> None:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute(statement.replace("%s", "?"), params)
            connection.commit()
        finally:
            connection.close()


class _Postgres:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    @asynccontextmanager
    async def store(self):
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.tasks_postgres import PostgresTaskStore

        store = PostgresTaskStore(self.dsn, schema_mode=SchemaMode.CREATE)
        try:
            yield store
        finally:
            await store.close()

    async def execute(self, statement: str, params: tuple) -> None:
        import psycopg

        async with await psycopg.AsyncConnection.connect(self.dsn) as connection:
            await connection.execute(cast("LiteralString", statement), params)
            await connection.commit()


async def _apply(backend, store, history) -> None:
    created: list[str] = []
    prerequisites_column = (
        "prerequisite_task_ids_json" if isinstance(backend, _SQLite) else "prerequisite_task_ids"
    )
    for kind, target, value in history:
        if kind == "create":
            task = await store.create_task(
                TaskCreate(
                    type=_TYPES[target],
                    title=f"task {len(created)} for {_TYPES[target]}",
                    assigned_agent_name=_AGENTS[value],
                    input={},
                )
            )
            created.append(task.id)
            continue
        if not created:
            continue
        task_id = created[target % len(created)]
        if kind == "status":
            await backend.execute(
                "UPDATE cayu_tasks SET status = %s WHERE id = %s AND graph_id IS NULL",
                (str(value), task_id),
            )
        elif kind == "agent":
            await backend.execute(
                "UPDATE cayu_tasks SET assigned_agent_name = %s WHERE id = %s",
                (_AGENTS[value], task_id),
            )
        elif kind == "prerequisites":
            others = sorted({created[index % len(created)] for index in value} - {task_id})
            await backend.execute(
                f"UPDATE cayu_tasks SET graph_id = %s, {prerequisites_column} = %s, "
                "status = 'pending' WHERE id = %s",
                (_GRAPH if others else None, json.dumps(others), task_id),
            )
        else:
            await backend.execute("DELETE FROM cayu_tasks WHERE id = %s", (task_id,))
            created.remove(task_id)


def _reference_match(query: TaskQuery):
    needle = None if query.q is None else query.q.casefold()

    def matches(task) -> bool:
        if query.type is not None and task.type != query.type:
            return False
        if query.assigned_agent_name is not None and (
            task.assigned_agent_name != query.assigned_agent_name
        ):
            return False
        if needle is None:
            return True
        return any(
            isinstance(value := getattr(task, name), str) and needle in str(value).casefold()
            for name in TASK_SEARCH_FIELDS
        )

    return matches


def _search(needle: str):
    def matches(fields) -> bool:
        return any(value is not None and needle in value.casefold() for value in fields)

    return matches


async def _read_all(store, query: TaskIndexQuery, *, native: bool, search: bool):
    pages = []
    after = None
    while True:
        page_query = query.model_copy(update={"after": after})
        if native:
            result = await store.query_task_index(
                page_query,
                search=None
                if not search or query.filters.q is None
                else _search(query.filters.q.casefold()),
            )
        else:
            reference = page_query.model_copy(
                update={
                    "filters": query.filters.model_copy(
                        update={"type": None, "assigned_agent_name": None}
                    )
                }
            )
            result = await store.query_task_index(
                reference, matches=_reference_match(query.filters)
            )
        pages.append(
            ([task.id for task in result.items], result.total_count, result.counts_by_status)
        )
        if result.next_position is None:
            return pages
        after = decode_task_index_cursor(result.next_position)
        assert len(pages) < 200


_QUERIES = (
    TaskIndexQuery(filters=TaskQuery(limit=3)),
    TaskIndexQuery(filters=TaskQuery(limit=2), include_internal=False),
    TaskIndexQuery(
        filters=TaskQuery(limit=4, order_by=TaskOrder.CREATED_AT_ASC),
        statuses=(TaskStatus.PENDING, TaskStatus.FAILED),
    ),
    TaskIndexQuery(filters=TaskQuery(limit=3, status=TaskStatus.COMPLETED, type="demo.review")),
    TaskIndexQuery(
        filters=TaskQuery(limit=2, assigned_agent_name="reviewer"), include_internal=False
    ),
    TaskIndexQuery(filters=TaskQuery(limit=3, q="build")),
    TaskIndexQuery(filters=TaskQuery(limit=3, q="TASK 1", order_by=TaskOrder.UPDATED_AT_ASC)),
)


async def _check(backend, history) -> None:
    async with backend.store() as store:
        await _apply(backend, store, history)
        check = await store.check_task_counts()
        assert check.matches, check.differences
        for query in _QUERIES:
            reference = await _read_all(store, query, native=False, search=False)
            assert await _read_all(store, query, native=True, search=False) == reference
            assert await _read_all(store, query, native=True, search=True) == reference
        tasks = await store.list_tasks(TaskQuery(limit=1000))
        for task in tasks[:5]:
            dependents = await _read_all(
                store,
                TaskIndexQuery(filters=TaskQuery(limit=2), prerequisite_task_id=task.id),
                native=True,
                search=False,
            )
            expected = sorted(
                (other for other in tasks if task.id in other.prerequisite_task_ids),
                key=lambda other: (other.updated_at, other.id),
                reverse=False,
            )
            listed = [identity for page in dependents for identity in page[0]]
            assert sorted(listed) == sorted(other.id for other in expected)
        identities = [task.id for task in reversed(tasks)][:7]
        loaded = await store.load_tasks([*identities, "missing-task", *identities[:2]])
        assert [task.id for task in loaded] == identities


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(history=_history)
def test_sqlite_task_counts_and_index_match_a_recount(history) -> None:
    asyncio.run(_check(_SQLite(), history))


@pytest.mark.usefixtures("postgres_dsn")
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(history=_history)
def test_postgres_task_counts_and_index_match_a_recount(postgres_dsn: str, history) -> None:
    asyncio.run(_check(_Postgres(postgres_dsn), history))


def test_sqlite_migration_backfills_counts_and_edges(tmp_path: Path) -> None:
    from cayu.storage._sqlite_catalog import _iter_statements
    from cayu.storage._task_index_schema import SQLITE_TASK_INDEX_DDL

    async def run() -> None:
        backend = _SQLite()
        backend.path = tmp_path / "tasks.db"
        async with backend.store() as store:
            await _apply(
                backend,
                store,
                [
                    ("create", 0, 1),
                    ("create", 2, 0),
                    ("create", 1, 2),
                    ("prerequisites", 0, {1, 2}),
                ],
            )
        connection = sqlite3.connect(backend.path)
        try:
            connection.execute("DELETE FROM cayu_task_counts")
            connection.execute("DELETE FROM cayu_task_prerequisites")
            connection.execute("INSERT INTO cayu_task_counts VALUES (0, 'pending', 'x', '', 9)")
            connection.commit()
            for statement in _iter_statements(SQLITE_TASK_INDEX_DDL):
                connection.execute(statement)
            connection.commit()
            assert (
                connection.execute("SELECT COUNT(*) FROM cayu_task_prerequisites").fetchone()[0]
                == 2
            )
        finally:
            connection.close()
        async with backend.store() as store:
            assert (await store.check_task_counts()).matches

    asyncio.run(run())


@pytest.mark.usefixtures("postgres_dsn")
def test_postgres_read_only_store_counts_pending_changes_without_folding(
    postgres_dsn: str,
) -> None:
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    async def run() -> None:
        writer = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            for index in range(3):
                await writer.create_task(TaskCreate(type="demo.review", title=f"t{index}"))
        finally:
            await writer.close()
        reader = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE, read_only=True)
        try:
            check = await reader.check_task_counts()
            result = await reader.query_task_index(TaskIndexQuery(filters=TaskQuery(limit=2)))
        finally:
            await reader.close()
        assert check.matches, check.differences
        assert check.by_status["pending"] >= 3
        # Earlier tests in the module share the database, so compare with the check.
        assert int(result.total_count) == sum(check.by_status.values())
        assert {
            status: int(count) for status, count in result.counts_by_status.model_dump().items()
        } == check.by_status

    asyncio.run(run())
