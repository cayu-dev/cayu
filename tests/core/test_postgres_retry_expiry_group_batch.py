"""A group transition may terminalize later rows in the retry-expiry batch."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from cayu import CayuApp
from cayu.storage import _postgres_support as pg
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
    TaskRetryPolicy,
    TaskStatus,
)
from cayu.tasks.retry import _task_retry_attempt_elapsed


async def _expired_group():
    clock = [datetime(2026, 9, 1, tzinfo=UTC)]
    memory = InMemoryTaskStore(clock=lambda: clock[0])
    app = CayuApp(task_store=memory, enable_logging=False)
    await app.create_task_group(
        TaskGroupCreate(
            group_id="expiry-group",
            graph=TaskGraphCreate(
                graph_id="expiry-graph",
                nodes=tuple(
                    TaskGraphNode(
                        task=TaskCreate(
                            task_id=identity,
                            type="expiry-job",
                            retry_policy=TaskRetryPolicy(max_attempts=2, max_elapsed_seconds=10),
                            metadata={
                                "execution_profile_fingerprint": "b" * 64,
                                "effect_fingerprint": "c" * 64,
                            },
                        )
                    )
                    for identity in ("a", "b")
                ),
            ),
            member_task_ids=("a", "b"),
            policy=TaskGroupPolicy(kind="all"),
            quiescence=TaskGroupQuiescencePolicy(timeout_seconds=60),
        )
    )
    clock[0] += timedelta(seconds=11)
    assert all(_task_retry_attempt_elapsed(t, series_now=clock[0]) for t in memory._tasks.values())
    return memory, clock[0]


class _ExpiryCursor:
    """Driver double for the claim admission's SQL, not for group policy.

    Rows and update rowcounts follow the current task snapshots. Group writes
    are produced by the real native in-memory group transition owner, which
    invokes the same graph/group planners as the PostgreSQL implementation.
    """

    def __init__(self, memory, now):
        self.memory = memory
        self.now = now
        self.rows = []
        self.rowcount = 0
        self.updated_ids = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def _expired(self):
        return [
            t
            for t in sorted(self.memory._tasks.values(), key=lambda t: (t.created_at, t.id))
            if _task_retry_attempt_elapsed(t, series_now=self.now)
        ]

    async def execute(self, statement, parameters=()):
        statement = " ".join(str(statement).split())
        self.rows = []
        self.rowcount = 0
        if statement in {"SELECT transaction_timestamp()", "SELECT clock_timestamp()"}:
            self.rows = [(self.now,)]
        elif statement.startswith("SELECT pg_advisory_xact_lock"):
            pass
        elif statement.startswith("SELECT graph_id FROM cayu_task_graph_members"):
            self.rows = [("expiry-graph",)]
        elif statement.startswith("SELECT id FROM cayu_tasks WHERE"):
            if "retry_series->>'disposition' = %s" in statement:
                self.rows = [(t.id,) for t in self._expired()]
            # No schedules, and both pending members have elapsed deadlines.
        elif statement.startswith("SELECT ") and "FROM cayu_tasks WHERE" in statement:
            if "WHERE id = %s" in statement:
                task = self.memory._tasks.get(parameters[0])
                self.rows = [] if task is None else [pg.task_insert_values(task)]
            elif "retry_series->>'disposition' = %s" in statement:
                # fetchall captures both rows before the first transition.
                self.rows = [pg.task_insert_values(t) for t in self._expired()]
        elif statement.startswith("UPDATE cayu_tasks SET status = %s"):
            task_id = parameters[-2]
            task = self.memory._tasks[task_id]
            self.rowcount = int(task.status.value == parameters[-1] and task.session_id is None)
            if self.rowcount:
                self.updated_ids.append(task_id)
            # _record_task_transition below publishes this write with its real
            # planner side effects, preserving the original pre-update snapshot.
        elif statement.startswith("INSERT INTO cayu_task_retry_settlements"):
            pass
        elif statement.startswith("WITH candidate AS"):
            pass  # No nonexpired tasks remain to dispatch.
        else:
            raise AssertionError(f"Unexpected SQL: {statement}")

    async def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    async def fetchall(self):
        rows, self.rows = self.rows, []
        return rows


class _Connection:
    def __init__(self, cursor):
        self.value = cursor
        self.committed = False

    def cursor(self):
        return self.value

    async def commit(self):
        self.committed = True


async def _retry_expiry_scenario():
    memory, now = await _expired_group()
    cursor = _ExpiryCursor(memory, now)
    connection = _Connection(cursor)
    store = object.__new__(PostgresTaskStore)
    store._clock_is_injected = True
    store._clock = lambda: now

    @asynccontextmanager
    async def owned_connection(_owner):
        yield connection

    async def record_transition(cur, prior, current, **kwargs):
        assert cur is cursor
        assert memory._tasks[prior.id] == prior
        memory._store_task(current)
        # The first expiry makes the all-members policy impossible and the
        # production group planner cancels the still-pending second member.
        assert memory._tasks["b"].status is TaskStatus.CANCELLED

    store._owned_store_connection = owned_connection
    store._record_task_transition = record_transition
    result = await store._claim_task_unowned(
        "maintenance-worker",
        TaskQuery(),
        connection_owner=None,
        excluded_keys=set(),
    )
    assert result == (None, None)
    assert connection.committed
    assert cursor.updated_ids == ["a"]
    assert memory._tasks["a"].status is TaskStatus.FAILED
    assert memory._tasks["b"].status is TaskStatus.CANCELLED


def test_retry_expiry_rechecks_rows_cancelled_by_an_earlier_group_member():
    asyncio.run(_retry_expiry_scenario())
