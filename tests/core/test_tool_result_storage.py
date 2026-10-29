"""Tool results are stored once, in their tool event, and read back unchanged."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import Message, MessageRole, ToolResultPart
from cayu.providers.base import ModelStreamEvent
from cayu.sessions._access_rules import SessionAccessRule, SessionAccessScope, SessionAccessSelector
from cayu.sessions.access import ScopedSessionAccess
from cayu.sessions.requests import ForkSessionRequest, ResumeRequest, RunRequest
from cayu.sessions.transcript_queries import TranscriptQuery
from cayu.storage import _transcript_tool_results as tool_results
from cayu.storage.retention import (
    RetentionMode,
    SessionCompactionLevel,
    SessionRetentionPolicy,
)
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec

# Both stores own this format; keep the Postgres variants in PR CI.
pytestmark = pytest.mark.store_semantics

REFERENCE = tool_results.CONTENT_REFERENCE_KEY
#: About what one tool-heavy review-service session produces: seven tool calls
#: of roughly 16 KiB each, plus one result too small to reference.
LARGE_RESULTS = 7
SMALL_RESULT = "ok"


def _large_result(index: int) -> str:
    return f"result {index} é\n" + "".join(
        f"line {index}.{line}: some file content\n" for line in range(600)
    )


class _Read(Tool):
    spec = ToolSpec(
        name="read",
        description="Read one numbered file.",
        input_schema={"type": "object", "properties": {"n": {"type": "integer"}}},
        max_terminal_payload_bytes=64 * 1024,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        index = args["n"]
        return ToolResult(content=SMALL_RESULT if index < 0 else _large_result(index))


def _expected_results() -> dict[str, str]:
    results = {f"call-{index}": _large_result(index) for index in range(LARGE_RESULTS)}
    results["call-small"] = SMALL_RESULT
    return results


def _script(final: str) -> list[list[ModelStreamEvent]]:
    steps = [
        [
            ModelStreamEvent.tool_call(name="read", arguments={"n": index}, id=f"call-{index}"),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
        for index in range(LARGE_RESULTS)
    ]
    steps.append(
        [
            ModelStreamEvent.tool_call(name="read", arguments={"n": -1}, id="call-small"),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    steps.append(
        [
            ModelStreamEvent.text_delta(final),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    return steps


class _Backend:
    """A session store plus direct access to its rows."""

    def __init__(self, name: str, tmp_path, request) -> None:
        self.name = name
        if name == "sqlite":
            from cayu.storage.sqlite import SQLiteSessionStore

            self.path = tmp_path / "sessions.sqlite"
            self.store = SQLiteSessionStore(self.path)
        else:
            from cayu.storage.migrations import SchemaMode
            from cayu.storage.postgres import PostgresSessionStore

            self.dsn = request.getfixturevalue("postgres_dsn")
            self.store = PostgresSessionStore(self.dsn, schema_mode=SchemaMode.CREATE)

    async def rows(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        if self.name == "sqlite":
            connection = sqlite3.connect(self.path)
            try:
                return connection.execute(sql, parameters).fetchall()
            finally:
                connection.close()
        import psycopg

        async with await psycopg.AsyncConnection.connect(self.dsn) as conn:
            cursor = await conn.execute(sql.replace("?", "%s"), parameters)
            return await cursor.fetchall()

    async def run(self, sql: str, parameters: tuple[Any, ...] = ()) -> None:
        if self.name == "sqlite":
            # The store's connection has the functions its triggers call.
            def write(connection: sqlite3.Connection) -> None:
                connection.execute(sql, parameters)
                connection.commit()

            await self.store._run_write(write)
            return
        import psycopg

        async with await psycopg.AsyncConnection.connect(self.dsn) as conn:
            await conn.execute(sql.replace("?", "%s"), parameters)
            await conn.commit()

    @property
    def message(self) -> str:
        return "message_json" if self.name == "sqlite" else "message"

    async def transcript_rows(self, session_id: str) -> list[tuple[int, Any, int]]:
        rows = await self.rows(
            f"SELECT sequence, {self.message}, message_bytes FROM cayu_transcript_messages "
            "WHERE session_id = ? ORDER BY session_order",
            (session_id,),
        )
        return [
            (row[0], json.loads(row[1]) if isinstance(row[1], str) else row[1], row[2])
            for row in rows
        ]

    async def row_bytes(self, session_id: str) -> dict[str, int]:
        """Bytes of the stored JSON documents that carry tool results."""

        if self.name == "sqlite":
            event = "COALESCE(SUM(length(CAST(payload_json AS BLOB))), 0)"
            transcript = "COALESCE(SUM(length(CAST(message_json AS BLOB))), 0)"
        else:
            event = "COALESCE(SUM(octet_length(payload::text) + octet_length(event::text)), 0)"
            transcript = "COALESCE(SUM(octet_length(message::text)), 0)"
        [(events,)] = await self.rows(
            f"SELECT {event} FROM cayu_events WHERE session_id = ?", (session_id,)
        )
        [(messages,)] = await self.rows(
            f"SELECT {transcript} FROM cayu_transcript_messages WHERE session_id = ?",
            (session_id,),
        )
        return {"events": int(events), "transcript": int(messages)}

    async def inline_message_bytes(self, document: dict[str, Any]) -> int:
        if self.name == "sqlite":
            from cayu.storage._sqlite_records import json_dumps

            return len(json_dumps(document).encode("utf-8"))
        [(size,)] = await self.rows(
            "SELECT octet_length((?::jsonb)::text)", (json.dumps(document),)
        )
        return int(size)

    async def rewrite_inline(self, session_id: str) -> None:
        """Store every tool result inline, the form stores used before revision 134."""

        contents = {
            event_id: content
            for event_id, content in await self.rows(
                (
                    "SELECT event_id, json_extract(payload_json, '$.result.content') "
                    if self.name == "sqlite"
                    else "SELECT event_id, payload -> 'result' ->> 'content' "
                )
                + "FROM cayu_events WHERE session_id = ? AND event_type IN (?, ?)",
                (session_id, *tool_results.TOOL_RESULT_EVENT_TYPES),
            )
        }
        for sequence, document, _ in await self.transcript_rows(session_id):
            if not tool_results.referenced_event_ids(document):
                continue
            inline = tool_results.resolve_message_document(document, contents)
            if self.name == "sqlite":
                encoded = json.dumps(inline, ensure_ascii=False, separators=(",", ":"))
                await self.run(
                    "UPDATE cayu_transcript_messages SET message_json = ?, message_bytes = ? "
                    "WHERE sequence = ?",
                    (encoded, len(encoded.encode("utf-8")), sequence),
                )
            else:
                await self.run(
                    "UPDATE cayu_transcript_messages SET message = ?::jsonb, "
                    "message_bytes = NULL WHERE sequence = ?",
                    (json.dumps(inline), sequence),
                )


@pytest.fixture(params=["sqlite", "postgres"])
async def backend(request, tmp_path):
    selected = _Backend(request.param, tmp_path, request)
    try:
        yield selected
    finally:
        await selected.store.close()


def _app(backend: _Backend, provider: ScriptedModelProvider) -> CayuApp:
    app = CayuApp(session_store=backend.store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="reader", model="scripted"), tools=[_Read()])
    return app


async def _run_session(backend: _Backend) -> tuple[CayuApp, ScriptedModelProvider, str]:
    session_id = "tool-results-" + uuid4().hex
    provider = ScriptedModelProvider(
        [
            *_script("done"),
            [
                ModelStreamEvent.text_delta("resumed"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("forked"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = _app(backend, provider)
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="reader",
                session_id=session_id,
                labels={"suite": "tool-results"},
                max_steps=LARGE_RESULTS + 3,
                messages=[Message.text("user", "Read the files.")],
            )
        )
    ]
    assert events[-1].type == EventType.SESSION_COMPLETED, events[-1]
    return app, provider, session_id


def _tool_contents(messages: list[Message]) -> dict[str, str]:
    return {
        part.tool_call_id: part.content
        for message in messages
        if message.role is MessageRole.TOOL
        for part in message.content
        if type(part) is ToolResultPart
    }


async def _every_transcript_read(backend: _Backend, session_id: str) -> dict[str, list[Message]]:
    store = backend.store
    reads: dict[str, list[Message]] = {"load_transcript": await store.load_transcript(session_id)}
    snapshot = await store.load_transcript_snapshot(session_id)
    reads["snapshot"] = [record.message for record in snapshot.records]
    window = await store.load_transcript_window(session_id, start_index=0, limit=100)
    reads["window"] = [record.message for record in window.records]
    bounded = await store.load_bounded_transcript_window(
        session_id,
        start_index=0,
        limit=100,
        max_record_bytes=1_000_000,
        max_total_bytes=10_000_000,
    )
    reads["bounded_window"] = [record.message for record in bounded.records]
    page = await store.query_transcript(TranscriptQuery(session_id=session_id, limit=100))
    reads["query"] = [record.message for record in page.records]
    latest = await store.load_latest_transcript_message(session_id, role=MessageRole.TOOL)
    reads["latest_tool_message"] = [latest.message]
    export = await store.load_session_export_snapshot(session_id)
    reads["export"] = [record.message for record in export.transcript_records]
    evidence = await store.load_terminal_session_evidence(session_id)
    reads["terminal_evidence"] = [record.message for record in evidence.transcript]
    closure = await store.load_session_closure_records(
        session_id, max_records=10_000, max_bytes=50_000_000
    )
    reads["closure_records"] = [
        Message.model_validate(record["message"]) for record in closure["records"]["transcript"]
    ]
    scope = SessionAccessScope(
        read=(
            SessionAccessRule(
                selectors=(SessionAccessSelector(key="suite", values=("tool-results",)),)
            ),
        )
    )

    async def resolve() -> SessionAccessScope:
        return scope

    access = ScopedSessionAccess(store, admitted=scope, resolve=resolve)
    records = await access.read_records(session_id, kind="transcript", max_bytes=4_000_000)
    reads["access_records"] = [Message.model_validate(record) for record in records.records]
    return reads


@pytest.mark.anyio
async def test_tool_results_are_stored_once_and_every_reader_sees_them_unchanged(
    backend: _Backend,
):
    app, provider, session_id = await _run_session(backend)
    expected = _expected_results()
    # The model saw each full result on the following step.
    assert _tool_contents(provider.requests[LARGE_RESULTS + 1].messages) == expected

    stored = await backend.transcript_rows(session_id)
    tool_rows = [row for row in stored if row[1]["role"] == "tool"]
    assert len(tool_rows) == LARGE_RESULTS + 1
    referenced = [row for row in tool_rows if tool_results.referenced_event_ids(row[1])]
    assert len(referenced) == LARGE_RESULTS
    for _, document, message_bytes in referenced:
        [part] = document["content"]
        assert part["content"] == ""
        assert part[REFERENCE]["bytes"] == len(expected[part["tool_call_id"]].encode("utf-8"))
        # The recorded size is the full message's, so bounded reads budget for it.
        inline = {
            **document,
            "content": [
                {
                    **{key: value for key, value in part.items() if key != REFERENCE},
                    "content": expected[part["tool_call_id"]],
                }
            ],
        }
        assert message_bytes == await backend.inline_message_bytes(inline)
    [small] = [row for row in tool_rows if row not in referenced]
    assert small[1]["content"][0]["content"] == SMALL_RESULT
    assert REFERENCE not in small[1]["content"][0]

    events = await backend.store.load_events(session_id)
    completed = {
        event.payload["tool_call_id"]: event.payload["result"]["content"]
        for event in events
        if event.type == EventType.TOOL_CALL_COMPLETED
    }
    assert completed == expected

    reads = await _every_transcript_read(backend, session_id)
    baseline = reads["load_transcript"]
    assert _tool_contents(baseline) == expected
    for name, messages in reads.items():
        if name == "latest_tool_message":
            assert messages == [baseline[-2]], name
        else:
            assert messages == baseline, name

    deduplicated = await backend.row_bytes(session_id)

    # A store written before this change kept each result inline in both
    # places. Its rows read back as the same messages.
    await backend.rewrite_inline(session_id)
    assert all(
        not tool_results.referenced_event_ids(document)
        for _, document, _ in await backend.transcript_rows(session_id)
    )
    inline = await backend.row_bytes(session_id)
    assert await _every_transcript_read(backend, session_id) == reads
    result_bytes = sum(len(content.encode("utf-8")) for content in expected.values())
    saved = inline["transcript"] - deduplicated["transcript"]
    print(
        f"\n{backend.name}: tool results {result_bytes} B; "
        f"inline events {inline['events']} B + transcript {inline['transcript']} B; "
        f"stored once: events {deduplicated['events']} B + transcript "
        f"{deduplicated['transcript']} B; saved {saved} B"
    )
    assert deduplicated["events"] == inline["events"]
    assert saved > 0.95 * (result_bytes - len(SMALL_RESULT))
    assert deduplicated["transcript"] < 0.1 * inline["transcript"]

    # Resume and fork read the stored transcript; the model sees every result.
    await _restore_references(backend, session_id, stored)
    resumed = [
        event
        async for event in app.resume(
            ResumeRequest(session_id=session_id, messages=[Message.text("user", "And now?")])
        )
    ]
    assert resumed[-1].type == EventType.SESSION_COMPLETED, resumed[-1]
    assert _tool_contents(provider.requests[-1].messages) == expected
    fork_id = session_id + "-fork"
    forked = [
        event
        async for event in app.fork_session(
            ForkSessionRequest(source_session_id=session_id, session_id=fork_id)
        )
    ]
    assert forked
    assert _tool_contents(await backend.store.load_transcript(fork_id)) == expected


async def _restore_references(
    backend: _Backend, session_id: str, stored: list[tuple[int, Any, int]]
) -> None:
    for sequence, document, message_bytes in stored:
        if not tool_results.referenced_event_ids(document):
            continue
        if backend.name == "sqlite":
            await backend.run(
                "UPDATE cayu_transcript_messages SET message_json = ?, message_bytes = ? "
                "WHERE sequence = ?",
                (
                    json.dumps(document, ensure_ascii=False, separators=(",", ":")),
                    message_bytes,
                    sequence,
                ),
            )
        else:
            await backend.run(
                "UPDATE cayu_transcript_messages SET message = ?::jsonb, message_bytes = ? "
                "WHERE sequence = ?",
                (json.dumps(document), message_bytes, sequence),
            )
    assert await backend.transcript_rows(session_id) == stored


@pytest.mark.anyio
async def test_a_changed_tool_event_fails_the_transcript_read(backend: _Backend):
    _, _, session_id = await _run_session(backend)
    if backend.name == "sqlite":
        statement = (
            "UPDATE cayu_events SET payload_json = "
            "json_set(payload_json, '$.result.content', 'forged') "
            "WHERE session_id = ? AND json_extract(payload_json, '$.tool_call_id') = 'call-0'"
        )
    else:
        statement = (
            "UPDATE cayu_events SET payload = "
            "jsonb_set(payload, '{result,content}', to_jsonb('forged'::text)) "
            "WHERE session_id = ? AND payload ->> 'tool_call_id' = 'call-0'"
        )
    await backend.run(statement, (session_id,))
    with pytest.raises(ValueError, match="no longer matches its tool event"):
        await backend.store.load_transcript(session_id)


async def _compact(backend: _Backend, session_id: str, level: SessionCompactionLevel):
    store = backend.store
    while (claim := await store.claim_persisted_event_side_effect()) is not None:
        await store.mark_persisted_event_side_effect_delivered(claim)
    report = await store.apply_retention_policy(
        SessionRetentionPolicy(
            mode=RetentionMode.COMPACT,
            older_than=timedelta(microseconds=1),
            compaction_level=level,
            compact_tool_output_min_bytes=4096,
            dry_run=False,
        )
    )
    return [item for item in report.items if item.item_id == session_id]


@pytest.mark.anyio
async def test_event_compaction_leaves_results_the_transcript_reads_from_events(
    backend: _Backend,
):
    _, _, session_id = await _run_session(backend)
    before = await backend.store.load_transcript(session_id)
    for item in await _compact(backend, session_id, SessionCompactionLevel.EVENTS):
        assert item.counts["tool_outputs_compacted"] == 0
    assert await backend.store.load_transcript(session_id) == before
    events = await backend.store.load_events(session_id)
    assert {
        event.payload["tool_call_id"]: event.payload["result"]["content"]
        for event in events
        if event.type == EventType.TOOL_CALL_COMPLETED
    } == _expected_results()


@pytest.mark.anyio
async def test_transcript_compaction_marks_both_copies_and_receipts_still_verify(
    backend: _Backend,
):
    app, provider, session_id = await _run_session(backend)
    [item] = await _compact(backend, session_id, SessionCompactionLevel.TRANSCRIPT)
    assert item.counts["tool_outputs_compacted"] == LARGE_RESULTS
    assert item.counts["transcript_tool_results_compacted"] == LARGE_RESULTS
    assert item.bytes > 0
    contents = _tool_contents(await backend.store.load_transcript(session_id))
    for call_id, original in _expected_results().items():
        if original == SMALL_RESULT:
            assert contents[call_id] == SMALL_RESULT
            continue
        encoded = original.encode("utf-8")
        assert contents[call_id] == (
            f"[cayu storage retention removed {len(encoded)} bytes of tool output; "
            f"sha256 {hashlib.sha256(encoded).hexdigest()}]"
        )
    assert all(
        not tool_results.referenced_event_ids(document)
        for _, document, _ in await backend.transcript_rows(session_id)
    )
    resumed = [
        event
        async for event in app.resume(
            ResumeRequest(session_id=session_id, messages=[Message.text("user", "And now?")])
        )
    ]
    assert resumed[-1].type == EventType.SESSION_COMPLETED, resumed[-1]
    assert _tool_contents(provider.requests[-1].messages) == contents


def _tool_message(call_id: str, content: str) -> dict[str, Any]:
    return Message.tool_result(tool_call_id=call_id, tool_name="read", content=content).model_dump(
        mode="json"
    )


def _completed(event_id: str, call_id: str, content: str) -> dict[str, Any]:
    return {
        "type": "tool.call.completed",
        "id": event_id,
        "payload": {"tool_call_id": call_id, "result": {"content": content}},
    }


def test_only_one_identical_large_terminal_result_is_referenced():
    large = "x" * tool_results.MIN_REFERENCED_CONTENT_BYTES
    small = "x" * (tool_results.MIN_REFERENCED_CONTENT_BYTES - 1)
    messages = [
        _tool_message("same", large),
        _tool_message("different", large),
        _tool_message("small", small),
        _tool_message("twice", large),
    ]
    events = [
        _completed("e-same", "same", large),
        _completed("e-different", "different", large + "!"),
        _completed("e-small", "small", small),
        _completed("e-twice-1", "twice", large),
        {**_completed("e-twice-2", "twice", large), "type": "tool.call.failed"},
    ]
    stored = tool_results.reference_event_contents(messages, events)
    assert [tool_results.referenced_event_ids(document) for document in stored] == [
        ["e-same"],
        [],
        [],
        [],
    ]
    assert stored[1:] == messages[1:]
    assert tool_results.resolve_message_document(stored[0], {"e-same": large}) == messages[0]
    with pytest.raises(ValueError, match="missing tool event"):
        tool_results.resolve_message_document(stored[0], {})


def test_unknown_reference_formats_are_rejected():
    stored = tool_results.reference_event_contents(
        [_tool_message("call", "y" * 2048)], [_completed("event", "call", "y" * 2048)]
    )[0]
    part = stored["content"][0]
    future = {**stored, "content": [{**part, REFERENCE: {**part[REFERENCE], "format": 2}}]}
    with pytest.raises(ValueError, match="malformed or unsupported"):
        tool_results.referenced_event_ids(future)


def test_revision_134_fences_binaries_that_cannot_read_references():
    from cayu.storage import migrations, postgres, sqlite

    revision = migrations.revision(134)
    assert revision.kind is migrations.RevisionKind.BREAKING
    assert revision.compatible_from == 134
    assert sqlite._SQLITE_SESSION_MIN_REQUIRED_REVISION == 134
    assert postgres._POSTGRES_SESSION_MIN_REQUIRED_REVISION == 134


@pytest.mark.anyio
async def test_sqlite_reads_restore_results_from_their_own_snapshot(tmp_path, monkeypatch):
    """A compaction committed between the row read and the event read is not seen."""

    from cayu.storage import _sqlite_transcript_content as transcript_content

    backend = _Backend("sqlite", tmp_path, None)
    try:
        _, _, session_id = await _run_session(backend)
        before = await backend.store.load_transcript(session_id)
        original = transcript_content.event_contents

        def compact_between_reads(connection, session, event_ids):
            writer = sqlite3.connect(backend.path)
            try:
                writer.execute(
                    "UPDATE cayu_events SET payload_json = "
                    "json_set(payload_json, '$.result.content', 'compacted') "
                    "WHERE session_id = ? AND event_type = 'tool.call.completed'",
                    (session,),
                )
                writer.commit()
            finally:
                writer.close()
            return original(connection, session, event_ids)

        monkeypatch.setattr(transcript_content, "event_contents", compact_between_reads)
        # The reader keeps the snapshot its rows came from. The event was
        # changed only by this test, so a later read fails its digest check.
        assert await backend.store.load_transcript(session_id) == before
    finally:
        await backend.store.close()
