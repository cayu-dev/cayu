"""Duplicate admission must not borrow a second connection during rollback."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from psycopg.errors import UniqueViolation
from psycopg_pool import PoolTimeout

from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.targets import DEFAULT_DATABASE_POOL_MAX


class _Cursor:
    """Only the SQL results needed for a duplicate interaction-start admission."""

    rowcount = 1

    def __init__(self, insert_barrier):
        self.insert_barrier = insert_barrier
        self.row = None

    async def execute(self, statement, params=None):
        sql = " ".join(statement.split())
        self.row = None
        if sql.startswith("SELECT clock_timestamp()"):
            self.row = (datetime.now(UTC),)
        elif sql.startswith("UPDATE cayu_sessions"):
            self.row = (1,)
        elif sql.startswith("INSERT INTO cayu_events"):
            # Independent sessions can all reach their duplicate insert together.
            await self.insert_barrier.wait()
            raise UniqueViolation("duplicate interaction-start event")
        elif sql.startswith("SELECT 1 FROM cayu_events") or sql.startswith(
            "SELECT 1 FROM cayu_sessions"
        ):
            self.row = (1,)

    async def executemany(self, statement, rows):
        for params in rows:
            await self.execute(statement, params)

    async def fetchone(self):
        return self.row

    async def fetchall(self):
        return []


class _Connection:
    def __init__(self, insert_barrier):
        self.insert_barrier = insert_barrier
        self.rollbacks = 0

    @asynccontextmanager
    async def cursor(self):
        yield _Cursor(self.insert_barrier)

    async def rollback(self):
        self.rollbacks += 1

    async def commit(self):
        pass


class _BoundedPool:
    """Rollback ends a transaction, not the pool checkout."""

    def __init__(self, capacity):
        self.slot = asyncio.Semaphore(capacity)
        self.insert_barrier = asyncio.Barrier(capacity)
        self.connections = []

    @asynccontextmanager
    async def connection(self):
        try:
            await asyncio.wait_for(self.slot.acquire(), timeout=0.1)
        except TimeoutError:
            raise PoolTimeout("all connections are still checked out") from None
        conn = _Connection(self.insert_barrier)
        self.connections.append(conn)
        try:
            yield conn
        finally:
            self.slot.release()


@pytest.mark.parametrize("capacity", [1, DEFAULT_DATABASE_POOL_MAX])
@pytest.mark.parametrize("publication", ["admission", "append"])
def test_duplicate_event_reuses_rolled_back_connection(monkeypatch, capacity, publication):
    async def run():
        memory = InMemorySessionStore()
        sessions = {}
        for index in range(capacity):
            session = await memory.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"sess_duplicate_admission_{index}",
                    messages=[Message.text("user", "hi")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            sessions[session.id] = session
        pool = _BoundedPool(capacity)
        store = object.__new__(PostgresSessionStore)
        store._pool = pool
        store._read_only = False
        store._public_authority_alias_codec = None
        store._alias_keyring_optional = True
        monkeypatch.setattr(store, "_ensure_ready", AsyncMock())
        monkeypatch.setattr(store, "_ensure_aliases_reconciled", AsyncMock())
        monkeypatch.setattr(
            store,
            "_load_for_update",
            AsyncMock(side_effect=lambda _cur, session_id: sessions[session_id]),
        )
        monkeypatch.setattr(store, "_require_external_wait_admission", AsyncMock())

        async def admit(session):
            event = Event(
                id=f"already_retained_start_{session.id}",
                type=EventType.INTERACTION_STARTED,
                session_id=session.id,
                interaction_id=f"interaction_{session.id}",
            )
            if publication == "append":
                await store.append_event(session.id, event)
                return
            await store.transition_status_and_checkpoint(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
                checkpoint_transform=lambda _session, checkpoint: checkpoint,
                interaction_started_event=event,
                interaction_source_messages=[Message.text("user", "hi")],
            )

        results = await asyncio.gather(
            *(admit(session) for session in sessions.values()), return_exceptions=True
        )
        assert sum(conn.rollbacks for conn in pool.connections) == capacity
        assert all(
            isinstance(result, ValueError) and "Event already exists" in str(result)
            for result in results
        ), results

    asyncio.run(run())
