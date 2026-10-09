"""Transcript operations compose independently of the native store adapters."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import cayu
from cayu.messages import Message, MessageRole
from cayu.sessions.base import RunRequest
from cayu.sessions.records import SessionIdentity
from cayu.sessions.transcript_queries import TranscriptQuery, TranscriptSearchQuery


def _assert_import_without_adapters(backend):
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
            raise AssertionError(f"Transcript owner imported an adapter: {fullname}")

sys.meta_path.insert(0, RejectAdapters())
owner = importlib.import_module(f"cayu.storage._{sys.argv[1]}_transcript")
assert callable(owner.append_transcript_messages)
assert callable(owner.query_transcript)
""",
            backend,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sqlite_transcript_imports_without_store_adapters():
    _assert_import_without_adapters("sqlite")


def test_postgres_transcript_imports_without_store_adapters():
    _assert_import_without_adapters("postgres")


def test_sqlite_transcript_composes_with_direct_connection(tmp_path):
    from cayu.storage import _sqlite_connection as sqlite_connection
    from cayu.storage import _sqlite_transcript as transcript
    from cayu.storage.sqlite import SQLiteSessionStore

    async def exercise():
        path = tmp_path / "transcript.sqlite"
        store = SQLiteSessionStore(path)
        initial = Message.text("user", "find this durable phrase")
        connection = None
        try:
            await store.create(
                RunRequest(session_id="transcript", agent_name="assistant", messages=[]),
                identity=SessionIdentity(provider_name="provider", model="model"),
            )
            await store.append_transcript_messages("transcript", [initial], interaction_id=None)
            connection = sqlite_connection.connect(path)
            store._register_public_authority_alias_sql_function(connection)
        finally:
            await store.close()

        # The adapter prepares schema/data and the connection's authority UDFs.
        # Transcript execution below uses that independent connection after
        # the adapter has been closed.
        assert connection is not None
        callbacks = []
        now = datetime(2026, 1, 1, tzinfo=UTC)

        async def execute(operation):
            return operation(connection)

        def closure_owners(targets, *, connection=None):
            callbacks.append(("closure", tuple(targets)))
            return ()

        def touch_activity(native, session_id, activity_at):
            assert native is connection and native.in_transaction
            callbacks.append(("activity", session_id, activity_at))

        try:
            appended = Message.text("assistant", "another durable phrase")
            await transcript.append_transcript_messages(
                execute,
                "transcript",
                [appended],
                interaction_id=None,
                ownership_clock=lambda: now,
                closure_owners=closure_owners,
                touch_activity=touch_activity,
            )
            assert callbacks == [("closure", ("transcript",)), ("activity", "transcript", now)]
            assert not connection.in_transaction
            assert await transcript.load_transcript(execute, "transcript") == [initial, appended]
            snapshot = await transcript.load_transcript_snapshot(execute, "transcript")
            assert snapshot.cursor == 2
            assert [record.index for record in snapshot.records] == [0, 1]
            page = await transcript.query_transcript(
                execute, TranscriptQuery(session_id="transcript", role=MessageRole.ASSISTANT)
            )
            assert [record.message for record in page.records] == [appended]
            results = await transcript.search_transcript(
                execute, TranscriptSearchQuery(session_ids=("transcript",), text="durable")
            )
            assert {hit.transcript_index for hit in results.hits} == {0, 1}
            assert await transcript.compact_transcript(execute, "transcript", keep_last=1) == 1
            assert await transcript.load_transcript_cursor(execute, "transcript") == 2
            assert await transcript.load_transcript(execute, "transcript") == [appended]
            assert not connection.in_transaction
        finally:
            connection.close()

    asyncio.run(exercise())


def test_postgres_transcript_composes_with_direct_connection(postgres_dsn):
    import psycopg

    from cayu.storage import _postgres_transcript as transcript
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def exercise():
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        session_id = f"transcript-{uuid4()}"
        initial = Message.text("user", "find this durable phrase")
        try:
            await store.create(
                RunRequest(session_id=session_id, agent_name="assistant", messages=[]),
                identity=SessionIdentity(provider_name="provider", model="model"),
            )
            await store.append_transcript_messages(session_id, [initial], interaction_id=None)
        finally:
            await store.close()

        callbacks = []
        now = datetime(2026, 1, 1, tzinfo=UTC)

        @asynccontextmanager
        async def connect():
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                yield connection

        async def closure_owners(cur, targets):
            callbacks.append(("closure", tuple(targets)))
            return ()

        async def register_authorities(cur, sid, *, interaction_ids=()):
            callbacks.append(("authority", sid, interaction_ids))

        async def store_now(cur):
            return now

        async def touch_activity(cur, sid, activity_at):
            assert cur.connection.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
            callbacks.append(("activity", sid, activity_at))

        async def load_session(cur, sid):
            # These operator reads do not need an authorization row. A scoped
            # read must receive a real loader; native access conformance covers it.
            raise AssertionError("Unscoped transcript read loaded authorization state")

        appended = Message.text("assistant", "another durable phrase")
        await transcript.append_transcript_messages(
            connect,
            session_id,
            [appended],
            interaction_id=None,
            closure_owners=closure_owners,
            register_authorities=register_authorities,
            store_now=store_now,
            touch_activity=touch_activity,
        )
        assert callbacks == [
            ("closure", (session_id,)),
            ("authority", session_id, ()),
            ("activity", session_id, now),
        ]
        assert await transcript.load_transcript(connect, session_id, load_session=load_session) == [
            initial,
            appended,
        ]
        snapshot = await transcript.load_transcript_snapshot(
            connect, session_id, load_session=load_session
        )
        assert snapshot.cursor == 2
        assert [record.index for record in snapshot.records] == [0, 1]
        page = await transcript.query_transcript(
            connect,
            TranscriptQuery(session_id=session_id, role=MessageRole.ASSISTANT),
            load_session=load_session,
        )
        assert [record.message for record in page.records] == [appended]
        results = await transcript.search_transcript(
            connect, TranscriptSearchQuery(session_ids=(session_id,), text="durable")
        )
        assert {hit.transcript_index for hit in results.hits} == {0, 1}
        assert await transcript.load_transcript_cursor(connect, session_id) == 2

    asyncio.run(exercise())
