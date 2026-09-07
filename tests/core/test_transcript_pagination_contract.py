"""Retained-row pagination is distinct from absolute transcript identity."""

from __future__ import annotations

import asyncio
import json
from io import StringIO
from uuid import uuid4

import pytest

from cayu import SQLiteSessionStore, TranscriptQuery
from cayu.core import Message, ThinkingPart
from cayu.runtime import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.runtime.sessions import TranscriptSnapshot


async def _create(store):
    return await store.create(
        RunRequest(agent_name="assistant", session_id=f"pagination-{uuid4().hex}", messages=[]),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )


async def _assert_queries(store):
    session = await _create(store)
    empty = await store.load_transcript_snapshot(session.id)
    assert empty.cursor == empty.retained_position(0) == 0
    assert empty.records == []
    await store.append_transcript_messages(
        session.id,
        [
            Message.text("user", "first"),
            Message(role="assistant", content=[ThinkingPart(text="private")]),
        ],
        interaction_id="one",
    )
    await store.append_transcript_messages(
        session.id,
        [Message.text("assistant", "answer"), Message.text("user", "next")],
        interaction_id="two",
    )
    page = await store.query_transcript(
        TranscriptQuery(
            session_id=session.id,
            role="assistant",
            offset=0,
            limit=1,
            include_thinking=False,
        )
    )
    assert page.total_records == 2
    assert page.records == []  # Projection is after retained-row pagination.
    page = await store.query_transcript(
        TranscriptQuery(
            session_id=session.id,
            role="assistant",
            offset=1,
            limit=1,
            include_thinking=False,
        )
    )
    assert page.total_records == 2
    assert [record.index for record in page.records] == [2]
    filtered = await store.query_transcript(
        TranscriptQuery(
            session_id=session.id,
            interaction_id="two",
            role="user",
            offset=0,
            limit=1,
        )
    )
    assert filtered.total_records == 1
    assert [record.index for record in filtered.records] == [3]
    assert (
        await store.query_transcript(
            TranscriptQuery(
                session_id=session.id,
                interaction_id="missing",
            )
        )
    ).total_records == 0
    snapshot = await store.load_transcript_snapshot(session.id)
    assert snapshot.cursor == await store.load_transcript_cursor(session.id) == 4
    assert len(snapshot.records) == 4
    for start, expected in [(0, [0, 1]), (2, [2, 3]), (4, []), (5, [])]:
        window = await store.load_transcript_window(session.id, start_index=start, limit=2)
        assert window.cursor == 4
        assert [record.index for record in window.records] == expected
    # Serialization does not change exact-cursor semantics.
    restored = TranscriptSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored.retained_position(4) == 4
    with pytest.raises(ValueError):
        restored.retained_position(5)
    for start, limit, error in [
        (True, 1, TypeError),
        (-1, 1, ValueError),
        (0, True, TypeError),
        (0, 0, ValueError),
        (0, 5001, ValueError),
    ]:
        with pytest.raises(error):
            await store.load_transcript_window(session.id, start_index=start, limit=limit)
    with pytest.raises(KeyError):
        await store.load_transcript_window("missing-session", start_index=0, limit=1)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_transcript_pagination_contract(tmp_path, backend):
    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        try:
            await _assert_queries(store)
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(run())


def test_sqlite_concurrent_append_and_retention_keep_absolute_authority(tmp_path):
    async def run():
        path = tmp_path / "concurrent.sqlite"
        writer, retainer = SQLiteSessionStore(path), SQLiteSessionStore(path)
        try:
            for _ in range(8):
                session = await _create(writer)
                await writer.append_transcript_messages(
                    session.id, [Message.text("user", str(i)) for i in range(5)]
                )
                # Separate connections contend at SQLite's actual write boundary.
                _, removed = await asyncio.gather(
                    writer.append_transcript_messages(
                        session.id, [Message.text("assistant", "five")]
                    ),
                    retainer.compact_transcript(session.id, keep_last=2),
                )
                snapshot = await writer.load_transcript_snapshot(session.id)
                assert snapshot.cursor == 6
                assert removed in (3, 4)
                assert [record.index for record in snapshot.records] == list(range(removed, 6))
                page = await retainer.query_transcript(TranscriptQuery(session_id=session.id))
                assert page.total_records == 6 - removed
                assert page.records == snapshot.records
        finally:
            await writer.close()
            await retainer.close()

    asyncio.run(run())


def test_postgres_transcript_pagination_contract(postgres_dsn):
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    async def run():
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await _assert_queries(store)
        finally:
            await store.close()

    asyncio.run(run())


def test_sqlite_retention_cycles_preserve_extent_and_pagination(tmp_path):
    path = tmp_path / "sessions.sqlite"

    async def run():
        store = SQLiteSessionStore(path)
        try:
            session = await _create(store)
            await store.append_transcript_messages(
                session.id, [Message.text("user", str(i)) for i in range(5)]
            )
            assert await store.compact_transcript(session.id, keep_last=2) == 3
            snapshot = await store.load_transcript_snapshot(session.id)
            assert snapshot.cursor == 5
            assert [record.index for record in snapshot.records] == [3, 4]
            with pytest.raises(ValueError, match="not available"):
                snapshot.retained_position(1)
            assert snapshot.retained_position(3) == 0
            assert snapshot.retained_position(5) == 2
            from cayu.storage.jsonl_export import export_sessions

            exported = StringIO()
            assert await export_sessions(store, stream=exported) == 1
            document = json.loads(exported.getvalue())
            assert document["snapshot"]["transcript_cursor"] == 5
            assert [record["index"] for record in document["transcript_records"]] == [3, 4]
            window = await store.load_transcript_window(session.id, start_index=1, limit=1)
            assert window.cursor == 5
            assert [record.index for record in window.records] == [3]
            page = await store.query_transcript(
                TranscriptQuery(session_id=session.id, offset=1, limit=1)
            )
            assert page.total_records == 2
            assert [record.index for record in page.records] == [4]
            await store.append_transcript_messages(session.id, [Message.text("assistant", "five")])
            assert await store.compact_transcript(session.id, keep_last=1) == 2
            filtered = await store.query_transcript(
                TranscriptQuery(session_id=session.id, role="user")
            )
            assert filtered.total_records == 0
            assert filtered.records == []
            assert await store.compact_transcript(session.id, keep_last=0) == 1
        finally:
            await store.close()
        reopened = SQLiteSessionStore(path)
        try:
            empty = await reopened.load_transcript_snapshot(session.id)
            assert empty.cursor == 6
            assert empty.records == []
            assert empty.retained_position(6) == 0
            with pytest.raises(ValueError, match="not available"):
                empty.retained_position(0)
            assert (
                await reopened.query_transcript(TranscriptQuery(session_id=session.id))
            ).total_records == 0
            assert (
                await reopened.load_transcript_window(session.id, start_index=0, limit=1)
            ).cursor == 6
            await reopened.append_transcript_messages(session.id, [Message.text("user", "six")])
            final = await reopened.load_transcript_snapshot(session.id)
            assert final.cursor == 7
            assert [record.index for record in final.records] == [6]
        finally:
            await reopened.close()

    asyncio.run(run())


def test_cli_pages_retained_rows_without_treating_indexes_as_offsets(tmp_path, capsys):
    from cayu.cli import main

    path = tmp_path / "cli.sqlite"

    async def seed():
        store = SQLiteSessionStore(path)
        try:
            session = await _create(store)
            await store.append_transcript_messages(
                session.id, [Message.text("user", str(i)) for i in range(5)]
            )
            await store.compact_transcript(session.id, keep_last=2)
            return session.id
        finally:
            await store.close()

    session_id = asyncio.run(seed())
    offset = 0
    indexes = []
    for _ in range(3):
        assert (
            main(
                [
                    "session",
                    "transcript",
                    session_id,
                    "--sqlite",
                    str(path),
                    "--offset",
                    str(offset),
                    "--limit",
                    "1",
                    "--json",
                ]
            )
            == 0
        )
        page = json.loads(capsys.readouterr().out)
        assert page["total_messages"] == 2
        indexes.extend(record["index"] for record in page["messages"])
        if not page["has_more"]:
            assert page["next_offset"] is None
            break
        assert page["next_offset"] == offset + 1
        offset = page["next_offset"]
    else:
        pytest.fail("Retained-row pagination did not terminate")
    assert indexes == [3, 4]
