"""Native query operations compose with execution capabilities independently of adapters."""

from __future__ import annotations

import asyncio
import importlib
import os
import sqlite3
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

import cayu
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runtime._usage_accounting import SessionUsageCache
from cayu.sessions.base import RunRequest, SessionIdentity
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.queries import SessionQuery


@pytest.mark.parametrize("backend", ["sqlite"])
def test_native_session_queries_import_without_store_adapters(backend):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import sys

class RejectAdapters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"cayu.storage.sqlite", "cayu.storage.postgres"}:
            raise AssertionError(f"Query owner imported its adapter: {fullname}")

sys.meta_path.insert(0, RejectAdapters())
owner = importlib.import_module(f"cayu.storage._{sys.argv[1]}_session_queries")
assert callable(owner.query_events)
assert callable(owner.list_sessions)
""",
            backend,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["sqlite"])
def test_native_session_queries_compose_with_direct_connections(backend, tmp_path, request):
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore
    from cayu.storage.sqlite import SQLiteSessionStore

    async def exercise():
        queries = importlib.import_module(f"cayu.storage._{backend}_session_queries")
        path = tmp_path / "sessions.sqlite"
        dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None
        store = (
            SQLiteSessionStore(path)
            if backend == "sqlite"
            else PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
        )
        session_id = f"query-owner-{uuid4()}"
        connection = None
        try:
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                    labels={"query-owner": session_id},
                ),
                identity=SessionIdentity(provider_name="provider", model="model"),
            )
            events = [
                Event(
                    id=f"event-{index}",
                    session_id=session_id,
                    type=EventType.SESSION_STARTED,
                    timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                )
                for index in range(3)
            ]
            await store.append_events(session_id, events)
            event_query = EventQuery(session_id=session_id, limit=2)
            session_query = SessionQuery(labels={"query-owner": session_id})
            expected_events = await store.query_events(event_query)
            expected_sessions = await store.list_sessions(session_query)
            expected_usage = await store.read_usage_accounting(event_query)

            # The receiving functions get a direct connection capability, never
            # a store instance. The adapter is only used to prepare native data.
            if backend == "sqlite":
                connection = sqlite3.connect(path)
                connection.row_factory = sqlite3.Row

                async def run_read(operation):
                    return operation(connection)

                actual_events = await queries.query_events(run_read, event_query)
                actual_sessions = await queries.list_sessions(
                    run_read,
                    session_query,
                    pending_interruption_cascade_only=False,
                    ownership_clock=lambda: datetime.now(UTC),
                )
                actual_usage = await queries.read_usage_accounting(
                    run_read, event_query, usage_cache=SessionUsageCache()
                )
                assert not connection.in_transaction
            else:
                import psycopg

                @asynccontextmanager
                async def connect():
                    async with await psycopg.AsyncConnection.connect(dsn) as native:
                        yield native

                actual_events = await queries.query_events(connect, event_query)
                actual_sessions = await queries.list_sessions(
                    connect, session_query, pending_interruption_cascade_only=False
                )
                actual_usage = await queries.read_usage_accounting(
                    connect, event_query, usage_cache=SessionUsageCache()
                )

            assert actual_events == expected_events
            assert [record.event.id for record in actual_events] == ["event-0", "event-1"]
            assert actual_sessions == expected_sessions
            assert [session.id for session in actual_sessions.sessions] == [session_id]
            assert actual_usage == expected_usage
        finally:
            if connection is not None:
                connection.close()
            await store.close()

    asyncio.run(exercise())
