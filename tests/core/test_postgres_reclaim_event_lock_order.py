"""Disjoint reclaim batches must not acquire shared event locks in opposite orders.

No PostgreSQL service is needed: the driver double models transaction-scoped
exclusive advisory locks and detects the same wait-for cycle PostgreSQL aborts.
Task construction, row codecs, reclaim decisions and event transitions are real.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from psycopg.errors import DeadlockDetected

from cayu.storage import _postgres_support as pg
from cayu.storage.tasks_postgres import PostgresTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskQuery, TaskStatus
from cayu.tasks.retry_budgets import TaskRetryBudget


class _AdvisoryLocks:
    def __init__(self):
        self.holders = {}
        self.waiting_for = {}
        self.changed = asyncio.Event()

    async def acquire(self, owner, identity):
        while (holder := self.holders.get(identity)) is not None and holder != owner:
            self.waiting_for[owner] = holder
            cursor = holder
            while cursor in self.waiting_for:
                cursor = self.waiting_for[cursor]
                if cursor == owner:
                    assert set(self.waiting_for.values()) == {"left", "right"}
                    assert {key for key in self.holders if ":retry-key:" in key} == {
                        "cayu:verified-work:retry-key:event-a",
                        "cayu:verified-work:retry-key:event-b",
                    }
                    raise DeadlockDetected(f"advisory-lock cycle while acquiring {identity}")
            event = self.changed
            await event.wait()
        self.waiting_for.pop(owner, None)
        self.holders[identity] = owner
        # Give the other independent connection a turn. This models a possible
        # network/driver scheduling point, without barriers on particular keys.
        await asyncio.sleep(0)

    def release(self, owner):
        self.holders = {key: holder for key, holder in self.holders.items() if holder != owner}
        self.waiting_for.pop(owner, None)
        changed, self.changed = self.changed, asyncio.Event()
        changed.set()


class _ReclaimCursor:
    def __init__(self, owner, locks, tasks, keys, now):
        self.owner = owner
        self.locks = locks
        self.tasks = tasks
        self.keys = keys
        self.now = now
        self.rows = []
        self.rowcount = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def _candidates(self, task_type):
        return sorted(
            (
                t
                for t in self.tasks.values()
                if t.type == task_type
                and t.status is TaskStatus.CLAIMED
                and t.lease_expires_at <= self.now
            ),
            key=lambda t: (t.lease_expires_at, t.id),
        )

    async def execute(self, statement, parameters=()):
        statement = " ".join(str(statement).split())
        self.rows = []
        self.rowcount = 0
        if statement.startswith("SELECT pg_advisory_xact_lock"):
            await self.locks.acquire(self.owner, parameters[0])
        elif statement.startswith("SELECT graph_id FROM cayu_task_graph_members"):
            pass  # These ordinary queue tasks are not graph/group members.
        elif statement.startswith("WITH timing AS MATERIALIZED"):
            # Both the unlocked scope selection and the locked selection use
            # the caller's actual type filter. The two batches have no common
            # task rows, so neither has a row-lock wait or a SKIP LOCKED loss.
            tasks = self._candidates(parameters[3])
            if "SELECT task.id FROM" in statement:
                self.rows = [(t.id,) for t in tasks]
            else:
                assert "FOR UPDATE OF task SKIP LOCKED" in statement
                selected_ids = parameters[-2]
                self.rows = [
                    (*pg.task_insert_values(t), self.now) for t in tasks if t.id in selected_ids
                ]
        elif statement.startswith("SELECT 1 FROM cayu_local_execution_attempts"):
            pass
        elif statement.startswith("UPDATE cayu_tasks SET status = %s, session_id = %s"):
            task = self.tasks[parameters[-1]]
            self.tasks[task.id] = task.model_copy(
                update={
                    "status": TaskStatus(parameters[0]),
                    "worker_id": parameters[3],
                    "lease_expires_at": parameters[4],
                    "updated_at": parameters[9],
                    "claim_generation": parameters[-2],
                }
            )
            self.rowcount = 1
        elif statement.startswith("SELECT state_json FROM cayu_task_retry_keys"):
            self.rows = [(self.keys[parameters[0]].model_dump_json(),)]
        elif statement == "SELECT clock_timestamp()":
            self.rows = [(self.now,)]
        elif statement.startswith("UPDATE cayu_task_retry_keys SET state_json"):
            self.keys[parameters[1]] = type(self.keys[parameters[1]]).model_validate_json(
                parameters[0]
            )
        elif statement.startswith("SELECT graph_id, task_id FROM cayu_task_graph_members"):
            pass  # Real _record_task_transition needs no graph publication.
        else:
            raise AssertionError(f"Unexpected SQL: {statement}; parameters={parameters}")

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

    async def rollback(self):
        pass


async def _reclaim_batches():
    now = datetime(2026, 10, 1, tzinfo=UTC)
    memory = InMemoryTaskStore(clock=lambda: now, ownership_clock=lambda: now)
    # Each event admits two tasks, one of each type. Opposite expiration
    # orders are normal: the claims can have been taken by different workers.
    for task_id, task_type, key in (
        ("left-1", "left", "event-a"),
        ("left-2", "left", "event-b"),
        ("right-1", "right", "event-b"),
        ("right-2", "right", "event-a"),
    ):
        await memory.create_task(
            TaskCreate(
                task_id=task_id,
                type=task_type,
                retry_key=key,
                retry_budget=TaskRetryBudget(),
            )
        )
    for task_type in ("left", "right"):
        for _ in range(2):
            assert await memory.claim_task(task_type, TaskQuery(type=task_type), lease_seconds=1)
    tasks = {t.id: t.model_copy(deep=True) for t in await memory.list_tasks()}
    keys = {key: await memory.load_task_retry_key(key) for key in ("event-a", "event-b")}
    now += timedelta(seconds=2)
    locks = _AdvisoryLocks()
    connections = []
    stores = []
    for owner in ("left", "right"):
        store = object.__new__(PostgresTaskStore)
        store._clock_is_injected = False
        connection = _Connection(_ReclaimCursor(owner, locks, tasks, keys, now))
        connections.append(connection)

        def make_connection(connection=connection, owner=owner):
            @asynccontextmanager
            async def owned_connection(_owner):
                try:
                    yield connection
                finally:
                    locks.release(owner)

            return owned_connection

        store._owned_store_connection = make_connection()
        stores.append(store)
    results = await asyncio.wait_for(
        asyncio.gather(
            *(
                store._reclaim_expired_unowned(
                    query=TaskQuery(type=owner),
                    max_reclaims=100,
                    connection_owner=None,
                )
                for store, owner in zip(stores, ("left", "right"), strict=True)
            )
        ),
        timeout=5,
    )
    assert all(connection.committed for connection in connections)
    assert [{task.id for task in result} for result in results] == [
        {"left-1", "left-2"},
        {"right-1", "right-2"},
    ]
    assert all(task.status is TaskStatus.PENDING for task in tasks.values())


def test_disjoint_reclaimers_share_a_canonical_event_key_lock_order():
    asyncio.run(_reclaim_batches())
