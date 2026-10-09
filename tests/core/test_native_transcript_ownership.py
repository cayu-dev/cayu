"""Transcript operations compose independently of the native store adapters."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import cayu
from cayu.messages import Message, MessageRole
from cayu.sessions.base import RunRequest, SessionIdentity
from cayu.sessions.transcript_queries import TranscriptQuery, TranscriptSearchQuery


def test_sqlite_transcript_imports_without_store_adapters():
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
owner = importlib.import_module("cayu.storage._sqlite_transcript")
assert callable(owner.append_transcript_messages)
assert callable(owner.query_transcript)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
