"""Memory transcript operations compose without a session-store implementation."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests._session_provenance import fixture_session_invocation

import cayu
from cayu.messages import Message, MessageRole, ToolCallPart
from cayu.sessions import _memory_transcripts as transcripts
from cayu.sessions.records import Session
from cayu.sessions.transcript_queries import TranscriptQuery, TranscriptSearchQuery


def _state():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    session = Session(
        id="session",
        agent_name="assistant",
        provider_name="fake",
        model="fake",
        causal_budget_id="session",
        invocation=fixture_session_invocation("session"),
        created_at=now,
        updated_at=now,
        last_activity_at=now,
    )
    indexes = transcripts.MemoryTranscriptIndexes(
        messages={session.id: []},
        interaction_ids={session.id: []},
        indices_by_interaction={},
        latest_indexes_by_role={},
        search_documents={},
        search_postings={},
    )
    return transcripts.MemoryTranscriptReadState(indexes, {session.id: session}, {})


def _append(indexes, messages, interaction_id):
    indexes.messages["session"].extend(messages)
    transcripts.extend_search(indexes, "session", messages)
    transcripts.extend_attribution(indexes, "session", [interaction_id] * len(messages))


def test_memory_transcripts_import_without_store_implementations():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import sys
class RejectStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"cayu.sessions.base", "cayu.sessions.store", "cayu.sessions.memory",
                        "cayu.storage.sqlite", "cayu.storage.postgres"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectStores())
owner = importlib.import_module("cayu.sessions._memory_transcripts")
assert callable(owner.query_transcript)
assert callable(owner.search_transcript)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_transcript_reads_compose_with_independent_indexes():
    async def scenario():
        state = _state()
        lock = asyncio.Lock()

        @asynccontextmanager
        async def read():
            async with lock:
                yield state

        _append(state.indexes, [Message.text("user", "Atlas launch")], "first")
        _append(
            state.indexes,
            [
                Message(
                    role="assistant",
                    content=[
                        ToolCallPart(
                            tool_call_id="call", tool_name="tool", arguments={"x": {"v": 1}}
                        )
                    ],
                ),
                Message.text("assistant", "Atlas launch tomorrow"),
            ],
            "second",
        )
        page = await transcripts.query_transcript(
            read, TranscriptQuery(session_id="session", interaction_id="second", limit=1)
        )
        assert page.total_records == 2
        assert [record.index for record in page.records] == [1]
        page.records[0].message.content[0].arguments["x"]["v"] = 9
        again = await transcripts.query_transcript(
            read, TranscriptQuery(session_id="session", offset=1, limit=1)
        )
        assert again.records[0].message.content[0].arguments["x"]["v"] == 1

        # Append after building the lazy interaction projection; it must stay current.
        _append(state.indexes, [Message.text("assistant", "Atlas launch complete")], "second")
        page = await transcripts.query_transcript(
            read, TranscriptQuery(session_id="session", interaction_id="second")
        )
        assert [record.index for record in page.records] == [1, 2, 3]
        snapshot = await transcripts.load_transcript_snapshot(read, "session")
        assert snapshot.cursor == await transcripts.load_transcript_cursor(read, "session") == 4
        assert [r.interaction_id for r in snapshot.records] == [
            "first",
            "second",
            "second",
            "second",
        ]
        latest = await transcripts.load_latest_transcript_message(
            read, "session", role=MessageRole.ASSISTANT
        )
        assert latest is not None and latest.index == 3
        assert await transcripts.load_latest_transcript_text(
            read, "session", role=MessageRole.ASSISTANT, max_chars=5
        ) == ("Atlas", True)
        query = TranscriptSearchQuery(text="Atlas", session_ids=("session",), limit=1)
        first = await transcripts.search_transcript(read, query)
        assert [hit.transcript_index for hit in first.hits] == [3]
        assert first.next_cursor is not None
        second = await transcripts.search_transcript(
            read, query.model_copy(update={"cursor": first.next_cursor})
        )
        assert [hit.transcript_index for hit in second.hits] == [2]
        assert not lock.locked()

    asyncio.run(scenario())


def test_transcript_read_view_observes_indexes_replaced_while_waiting():
    from cayu.sessions.memory import InMemorySessionStore

    async def scenario():
        store = InMemorySessionStore()
        state = _state()
        _append(state.indexes, [Message.text("user", "replacement")], "interaction")
        await store._lock.acquire()
        pending = asyncio.create_task(store.load_transcript_snapshot("session"))
        try:
            await asyncio.sleep(0)
            assert not pending.done()
            store._sessions = state.sessions
            store._transcripts = state.indexes.messages
            store._transcript_interaction_ids = state.indexes.interaction_ids
        finally:
            store._lock.release()
        result = await pending
        assert result.cursor == 1
        assert result.records[0].interaction_id == "interaction"
        store._transcripts = {"session": []}
        store._transcript_interaction_ids = {"session": []}
        assert (await store.load_transcript_snapshot("session")).cursor == 0

    asyncio.run(scenario())


def test_transcript_search_cancellation_releases_independent_context():
    async def scenario():
        state = _state()
        _append(
            state.indexes, [Message.text("user", "searchable") for _ in range(300)], "interaction"
        )
        lock = asyncio.Lock()
        started = asyncio.Event()

        @asynccontextmanager
        async def read():
            async with lock:
                started.set()
                yield state

        pending = asyncio.create_task(
            transcripts.search_transcript(
                read, TranscriptSearchQuery(text="searchable", session_ids=("session",))
            )
        )
        await started.wait()
        assert not pending.done() and lock.locked()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not lock.locked()
        assert (await transcripts.load_transcript_snapshot(read, "session")).cursor == 300

    asyncio.run(scenario())
