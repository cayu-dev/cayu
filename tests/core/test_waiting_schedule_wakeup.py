"""Graph-gated schedules still need expiry wakeups before graph admission.

A scheduled task waiting on prerequisites or on its group can still expire, and
the claim sweep settles it without waiting for the graph. Every store's wakeup
observation must therefore report its expiry, as a paused task's is reported.

The memory and SQLite stores observe their own state. For PostgreSQL, the
production SELECTs run on SQLite with only timestamp casts, interval arithmetic
and BOOL_OR adapted; the status predicates and parameters are kept intact.
"""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest

from cayu.storage.sqlite import SQLiteTaskStore
from cayu.storage.tasks_postgres import PostgresTaskStore
from cayu.tasks import (
    InMemoryTaskStore,
    TaskCreate,
    TaskGraphCreate,
    TaskGraphNode,
    TaskGroupCreate,
    TaskGroupPolicy,
    TaskGroupQuiescencePolicy,
    TaskQuery,
    TaskSchedulePolicy,
    TaskStatus,
)


class _BoolOr:
    def __init__(self):
        self.value = False

    def step(self, value):
        self.value |= bool(value)

    def finalize(self):
        return self.value


class _QueryCursor:
    def __init__(self, database):
        self.database = database
        self.result = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, statement, parameters):
        statement = statement.replace(
            "EXTRACT(EPOCH FROM (%s::timestamptz - available_at))",
            "((julianday(%s) - julianday(available_at)) * 86400)",
        )
        statement = statement.replace("::timestamptz", "").replace("::bigint", "")
        statement = statement.replace("%s", "?")
        parameters = [
            value.isoformat().replace("+00:00", "Z") if isinstance(value, datetime) else value
            for value in parameters
        ]
        self.result = self.database.execute(statement, parameters)

    async def fetchone(self):
        row = self.result.fetchone()
        # psycopg returns a Python bool for BOOL_OR, not SQLite's integer.
        return (row[0], bool(row[1])) if len(row) == 2 else row


class _Connection:
    def __init__(self, database):
        self.database = database

    def cursor(self):
        return _QueryCursor(self.database)


class _Pool:
    def __init__(self, database):
        self.database = database

    @asynccontextmanager
    async def connection(self):
        yield _Connection(self.database)


async def _postgres_observation(memory, clock, query):
    """Observe through PostgresTaskStore's SQL, with rows copied from ``memory``."""

    database = sqlite3.connect(":memory:")
    database.create_aggregate("BOOL_OR", 1, _BoolOr)
    database.executescript(
        "CREATE TABLE cayu_tasks (id TEXT, type TEXT, status TEXT, session_id TEXT, "
        "available_at TEXT, retry_series TEXT, schedule TEXT);"
        "CREATE TABLE cayu_local_execution_attempts "
        "(task_id TEXT, retry_series_id TEXT, retry_admissible BOOLEAN);"
    )
    for task in await memory.list_tasks():
        database.execute(
            "INSERT INTO cayu_tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                task.id,
                task.type,
                task.status.value,
                task.session_id,
                None
                if task.available_at is None
                else task.available_at.isoformat().replace("+00:00", "Z"),
                None if task.retry_series is None else task.retry_series.model_dump_json(),
                None if task.schedule is None else task.schedule.model_dump_json(),
            ),
        )
    postgres = object.__new__(PostgresTaskStore)
    postgres._pool = _Pool(database)
    postgres._clock_is_injected = True
    postgres._clock = lambda: clock[0]

    async def ready():
        pass

    postgres._ensure_ready = ready
    try:
        return await postgres.next_task_schedule_wakeup(query)
    finally:
        database.close()


async def _scenario(store_kind, gate, elapsed, tmp_path):
    now = datetime(2026, 10, 1, tzinfo=UTC)
    expiry = now + timedelta(seconds=20)
    clock = [now]
    if store_kind == "sqlite":
        store = SQLiteTaskStore(tmp_path / "tasks.sqlite", clock=lambda: clock[0])
    else:
        # PostgreSQL's rows are planned by the memory store, then observed
        # through the production PostgreSQL SELECTs.
        store = InMemoryTaskStore(clock=lambda: clock[0])
    try:
        root = TaskGraphNode(task=TaskCreate(task_id="root", type="root"))
        scheduled = TaskGraphNode(
            task=TaskCreate(
                task_id="scheduled",
                type="scheduled-work",
                available_at=now + timedelta(seconds=10),
                schedule_policy=TaskSchedulePolicy(expires_at=expiry),
            ),
            prerequisite_task_ids=("root",) if gate == "dependencies" else (),
        )
        graph = TaskGraphCreate(graph_id="graph", nodes=(root, scheduled))
        if gate in {"dependencies", "held"}:
            await store.create_task_graph(graph)
            if gate == "held":
                await store.pause_task("scheduled")
                expected_status = TaskStatus.PAUSED
            else:
                expected_status = TaskStatus.WAITING_DEPENDENCIES
        else:
            await store.create_task_group(
                TaskGroupCreate(
                    group_id="group",
                    graph=graph,
                    member_task_ids=("root",),
                    finalizer_task_id="scheduled",
                    policy=TaskGroupPolicy(kind="all"),
                    quiescence=TaskGroupQuiescencePolicy(timeout_seconds=60),
                )
            )
            expected_status = TaskStatus.WAITING_GROUP
        await store.pause_task("root")
        task = await store.load_task("scheduled")
        assert task.status is expected_status and task.schedule.admitted_at is None
        if elapsed:
            clock[0] = expiry + timedelta(seconds=1)

        query = TaskQuery(type="scheduled-work")
        if store_kind == "postgres":
            observation = await _postgres_observation(store, clock, query)
        else:
            observation = await store.next_task_schedule_wakeup(query)
        assert observation.next_available_at is None  # The graph still gates dispatch.
        if elapsed:
            # Expiry is not gated on prerequisite/group success: the real sweep
            # must, and does, settle this task while the root remains unfinished.
            assert await store.claim_task("sweeper", query) is None
            assert (await store.load_task("scheduled")).status is TaskStatus.CANCELLED
            assert observation.maintenance_required, (
                f"The expiry sweep can settle {expected_status}, but the {store_kind} "
                "wakeup observation says no maintenance is due."
            )
            assert observation.next_expiry_at is None
        else:
            assert observation.next_expiry_at == expiry, (
                f"The unadmitted {expected_status} schedule lost its {store_kind} expiry wakeup."
            )
            assert not observation.maintenance_required
    finally:
        if store_kind == "sqlite":
            await store.close()


@pytest.mark.store_semantics
@pytest.mark.parametrize("store_kind", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("gate", ["dependencies", "group", "held"])
@pytest.mark.parametrize("elapsed", [False, True])
def test_graph_gated_schedule_expiry_is_visible_to_the_wakeup_query(
    store_kind, gate, elapsed, tmp_path
):
    asyncio.run(_scenario(store_kind, gate, elapsed, tmp_path))
