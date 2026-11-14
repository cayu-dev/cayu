"""The narrow ``cayu_trace_events`` copy that the session trace and event heads read.

Durable stores keep the copy with triggers on ``cayu_events``. These tests hold
its contract on SQLite and Postgres: every event is copied with its payload's
size and only the payload members the trace reads, payload updates and
deletions follow, the migration copies events retained before it, and native
reads equal the reference that reduces whole events.
"""

from __future__ import annotations

import json
import random
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import LiteralString, cast
from uuid import uuid4

import pytest

from cayu import Event, EventType, Message, RunRequest
from cayu import _event_schema as event_schema
from cayu._trace_payload import (
    TRACE_PAYLOAD_KEYS,
    TRACE_RESULT_STRUCTURED_EVENT_TYPES,
    TRACE_RESULT_STRUCTURED_KEYS,
    TRACE_WHOLE_MEMBERS,
    reduce_trace_payload,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import SessionIdentity
from cayu.sessions.store import SessionStore
from cayu.sessions.trace_events import check_trace_events
from cayu.storage._trace_event_schema import TRACE_EVENTS_REVISION
from cayu.storage.migrations import SchemaMode, SchemaTooOld
from cayu.storage.sqlite import SQLiteSessionStore

pytestmark = pytest.mark.anyio
NOW = datetime(2026, 3, 1, 12, tzinfo=UTC)
_TRACE_FIELDS = {"model_step_id", "tool_round_id", "tool_call_id", "execution_id"}


def _events(session_id: str) -> list[Event]:
    def event(index: int, event_type: EventType, payload: dict, tool_name: str | None = None):
        return Event(
            id=f"{session_id}-{index}",
            session_id=session_id,
            interaction_id="turn-1" if index else None,
            type=event_type,
            timestamp=NOW + timedelta(seconds=index),
            tool_name=tool_name,
            payload=payload,
        )

    return [
        event(0, EventType.MODEL_STARTED, {"model_step_id": "step-1", "messages": ["x" * 5000]}),
        event(
            1,
            EventType.MODEL_COMPLETED,
            {"model_step_id": "step-1", "tool_round_id": "round-1", "response": "y" * 3000},
        ),
        event(
            2,
            EventType.TOOL_CALL_STARTED,
            {
                "model_step_id": "step-1",
                "tool_round_id": "round-1",
                "tool_call_id": "call-1",
                "arguments": {"cmd": "ls"},
            },
            "exec_command",
        ),
        event(
            3,
            EventType.TOOL_CALL_COMPLETED,
            {
                "model_step_id": "step-1",
                "tool_round_id": "round-1",
                "tool_call_id": "call-1",
                "interrupted": False,
                "result": {
                    "content": "z" * 20000,
                    "structured": {"tool_call_id": "call-1", "rows": [1, 2, 3]},
                },
            },
            "exec_command",
        ),
        event(
            4,
            EventType.RUNNER_EXEC_COMPLETED,
            {"execution_id": "exec-1", "exit_code": 0, "timed_out": None, "error_type": "ü"},
        ),
        event(
            5,
            EventType.SESSION_AWAITING_USER_INPUT,
            {
                "input_id": "input-1",
                "model_step_id": "step-1",
                "tool_calls": [{"tool_call_id": "call-2", "arguments": {}}],
                "approval": {"ignored": True},
            },
        ),
        event(
            6,
            EventType.SESSION_INTERRUPTED,
            {"approval": {"tool_call_id": "call-3"}, "user_input": None, "reason": "stop"},
        ),
        event(7, EventType.SESSION_COMPLETED, {"summary": "done", "cancelled": True}),
    ]


async def _populate(store, session_id: str) -> None:
    await store.create(
        RunRequest(
            session_id=session_id,
            agent_name="assistant",
            messages=[Message.text("user", "hello")],
        ),
        identity=SessionIdentity(provider_name="anthropic", model="claude-x"),
    )
    for event in _events(session_id):
        await store.append_event(session_id, event)


class _Backend:
    """A durable store plus raw access to its event tables."""

    def __init__(self, name: str, tmp_path, request) -> None:
        self.name = name
        if name == "sqlite":
            self.path = tmp_path / "trace-copy.sqlite"
        else:
            self.dsn = request.getfixturevalue("postgres_dsn")

    async def open(self, schema_mode: SchemaMode = SchemaMode.CREATE):
        if self.name == "sqlite":
            return SQLiteSessionStore(self.path, schema_mode=schema_mode)
        from cayu.storage.postgres import PostgresSessionStore

        store = PostgresSessionStore(self.dsn, schema_mode=schema_mode)
        try:
            await store.ensure_schema()
        except BaseException:
            await store.close()
            raise
        return store

    async def execute(self, sql: str, params: tuple = ()) -> list[tuple]:
        if self.name == "sqlite":
            connection = sqlite3.connect(self.path)
            try:
                rows = [tuple(row) for row in connection.execute(sql, params).fetchall()]
                connection.commit()
                return rows
            finally:
                connection.close()
        import psycopg

        async with await psycopg.AsyncConnection.connect(self.dsn, autocommit=True) as conn:
            cursor = await conn.execute(cast("LiteralString", sql.replace("?", "%s")), params)
            return [tuple(row) for row in await cursor.fetchall()] if cursor.description else []

    async def copies(self, session_id: str) -> list[dict]:
        payload = "payload_json" if self.name == "sqlite" else "payload"
        rows = await self.execute(
            f"SELECT {payload} FROM cayu_trace_events WHERE session_id = ? ORDER BY sequence",
            (session_id,),
        )
        return [json.loads(value) if isinstance(value, str) else value for (value,) in rows]

    async def set_payload(self, session_id: str, event_id: str, payload: dict) -> None:
        if self.name == "sqlite":
            sql = "UPDATE cayu_events SET payload_json = ? WHERE session_id = ? AND event_id = ?"
            params: tuple = (json.dumps(payload), session_id, event_id)
        else:
            # Postgres keeps the payload twice; writers rewrite both.
            sql = (
                "UPDATE cayu_events SET payload = ?::jsonb, "
                "event = jsonb_set(event, '{payload}', ?::jsonb) "
                "WHERE session_id = ? AND event_id = ?"
            )
            params = (json.dumps(payload), json.dumps(payload), session_id, event_id)
        await self.execute(sql, params)


@pytest.fixture(params=["sqlite", "postgres"])
def backend(request, tmp_path) -> _Backend:
    return _Backend(request.param, tmp_path, request)


def _comparable(rows) -> list[tuple]:
    return [
        (
            row.sequence,
            row.event.id,
            str(row.event.type),
            row.event.session_id,
            row.event.interaction_id,
            row.event.timestamp,
            row.event.tool_name,
            row.event.payload,
        )
        for row in rows
    ]


async def _assert_matches_reference(store, session_id: str) -> None:
    query = EventQuery(session_id=session_id, limit=5000)
    native = await store.query_trace_events(query)
    reference = await SessionStore.query_trace_events(store, query)
    assert _comparable(native) == _comparable(reference)
    assert all(row.payload_bytes > 0 for row in native)
    assert (await check_trace_events(store, session_id)).mismatched_sequences == ()


def test_trace_members_cover_every_identity_the_event_schema_owns() -> None:
    # The DDL is fixed per revision. If the schema gains a nested identity path,
    # the copy must grow in a new revision before the trace can read it.
    for event_type in EventType:
        policy = event_schema.event_payload_policy(event_type)
        for path in policy.aliased_nested_authority_paths:
            if path[-1] not in _TRACE_FIELDS:
                continue
            if "*" in path:
                assert path[0] in TRACE_WHOLE_MEMBERS.get(str(event_type), ()), (event_type, path)
            elif path[0] == "result":
                assert path[1:-1] == ("structured",)
                assert path[-1] in TRACE_RESULT_STRUCTURED_KEYS
                assert str(event_type) in TRACE_RESULT_STRUCTURED_EVENT_TYPES
            else:
                assert path[0] in TRACE_WHOLE_MEMBERS.get(str(event_type), ()), (event_type, path)
        for key in _TRACE_FIELDS & (
            set(policy.authority_keys) | set(policy.aliased_authority_keys)
        ):
            assert key in TRACE_PAYLOAD_KEYS


async def test_copy_keeps_only_trace_members(backend) -> None:
    session_id = "trace-" + uuid4().hex
    store = await backend.open()
    try:
        await _populate(store, session_id)
        copies = await backend.copies(session_id)
        assert copies == [
            reduce_trace_payload(str(event.type), event.payload) for event in _events(session_id)
        ]
        assert copies[0] == {"model_step_id": "step-1"}
        assert copies[3] == {
            "model_step_id": "step-1",
            "tool_round_id": "round-1",
            "tool_call_id": "call-1",
            "interrupted": False,
            "result": {"structured": {"tool_call_id": "call-1"}},
        }
        assert copies[4] == {
            "execution_id": "exec-1",
            "exit_code": 0,
            "timed_out": None,
            "error_type": "ü",
        }
        assert set(copies[5]) == {"input_id", "model_step_id", "tool_calls"}
        assert copies[6] == {"approval": {"tool_call_id": "call-3"}, "user_input": None}
        assert copies[7] == {"cancelled": True}
        await _assert_matches_reference(store, session_id)
        heads = await store.query_trace_events(EventQuery(session_id=session_id, limit=10))
        assert heads[3].payload_bytes > 20000
        assert heads[0].payload_bytes < heads[3].payload_bytes
        await store.delete_session(session_id)
    finally:
        await store.close()


async def test_copy_follows_payload_updates_and_deletions(backend) -> None:
    session_id = "trace-" + uuid4().hex
    store = await backend.open()
    try:
        await _populate(store, session_id)
        await backend.set_payload(
            session_id, f"{session_id}-3", {"tool_call_id": "call-9", "result": {"content": "x"}}
        )
        assert (await backend.copies(session_id))[3] == {"tool_call_id": "call-9"}
        await _assert_matches_reference(store, session_id)
        await store.delete_session(session_id)
        assert await backend.copies(session_id) == []
    finally:
        await store.close()


def _random_value(rng: random.Random, depth: int = 0):
    choices = ["str", "int", "float", "bool", "none", "blank"]
    if depth < 2:
        choices += ["dict", "list"]
    kind = rng.choice(choices)
    if kind == "str":
        return rng.choice(["a", "step-1", "ü'\"\\", "x" * rng.randint(0, 50)])
    if kind == "int":
        return rng.randint(-5, 2**40)
    if kind == "float":
        return rng.choice([0.5, -1.25, 3.0])
    if kind == "bool":
        return rng.choice([True, False])
    if kind == "none":
        return None
    if kind == "blank":
        return "  "
    if kind == "list":
        return [_random_value(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    keys = ["tool_call_id", "tool_round_id", "structured", "content", "other"]
    return {rng.choice(keys): _random_value(rng, depth + 1) for _ in range(rng.randint(0, 4))}


def _random_payload(rng: random.Random) -> dict:
    keys = [*TRACE_PAYLOAD_KEYS, "approval", "user_input", "tool_calls", "result", "arguments"]
    payload = {key: _random_value(rng) for key in rng.sample(keys, rng.randint(0, 6))}
    if rng.random() < 0.4:
        payload["result"] = {
            "content": "c" * rng.randint(0, 2000),
            "structured": _random_value(rng)
            if rng.random() < 0.3
            else {
                key: _random_value(rng, 2)
                for key in rng.sample(["tool_call_id", "tool_round_id", "rows"], rng.randint(0, 3))
            },
        }
    return payload


async def test_random_histories_match_the_reference(backend) -> None:
    rng = random.Random(3945)
    types = [
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.SESSION_INTERRUPTED,
        EventType.SESSION_AWAITING_USER_INPUT,
        EventType.MODEL_COMPLETED,
        EventType.RUNNER_EXEC_COMPLETED,
        EventType.KNOWLEDGE_LOOKUP,
    ]
    store = await backend.open()
    try:
        for history in range(4):
            session_id = f"trace-random-{history}-" + uuid4().hex
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="anthropic", model="claude-x"),
            )
            events = [
                Event(
                    id=f"{session_id}-{index}",
                    session_id=session_id,
                    type=rng.choice(types),
                    timestamp=NOW + timedelta(seconds=index),
                    payload=_random_payload(rng),
                )
                for index in range(30)
            ]
            for event in events:
                await store.append_event(session_id, event)
            # Retention-like rewrites and deletions of single events.
            for index in rng.sample(range(30), 5):
                await backend.set_payload(session_id, f"{session_id}-{index}", _random_payload(rng))
            for index in rng.sample(range(30), 5):
                await backend.execute(
                    "DELETE FROM cayu_events WHERE session_id = ? AND event_id = ?",
                    (session_id, f"{session_id}-{index}"),
                )
            await _assert_matches_reference(store, session_id)
            await store.delete_session(session_id)
    finally:
        await store.close()


async def test_migration_copies_events_retained_before_it(backend) -> None:
    session_id = "trace-" + uuid4().hex
    store = await backend.open()
    try:
        await _populate(store, session_id)
    finally:
        await store.close()
    on_table = "" if backend.name == "sqlite" else " ON cayu_events"
    for trigger in ("insert", "delete", "update"):
        await backend.execute(f"DROP TRIGGER cayu_trace_events_{trigger}{on_table}")
    await backend.execute("DROP TABLE cayu_trace_events")
    # Back to the revision before the trace copy, and before every later revision.
    await backend.execute(
        "DELETE FROM cayu_schema_migrations WHERE revision >= ?", (TRACE_EVENTS_REVISION,)
    )
    with pytest.raises(SchemaTooOld, match="requires >= "):
        await (await backend.open()).close()
    store = await backend.open(SchemaMode.MIGRATE)
    try:
        assert len(await backend.copies(session_id)) == len(_events(session_id))
        await _assert_matches_reference(store, session_id)
        await store.delete_session(session_id)
    finally:
        await store.close()


async def test_copy_keeps_an_empty_payload_for_a_payload_that_is_not_an_object(backend) -> None:
    # The copy must never refuse a write the event table accepts.
    session_id = "trace-" + uuid4().hex
    store = await backend.open()
    try:
        await _populate(store, session_id)
        event_id = f"{session_id}-3"
        if backend.name == "sqlite":
            await backend.execute(
                "UPDATE cayu_events SET payload_json = ? WHERE session_id = ? AND event_id = ?",
                ("not-json", session_id, event_id),
            )
        else:
            await backend.execute(
                "UPDATE cayu_events SET event = jsonb_set(event, '{payload}', ?::jsonb) "
                "WHERE session_id = ? AND event_id = ?",
                (json.dumps("not an object"), session_id, event_id),
            )
        assert (await backend.copies(session_id))[3] == {}
        await store.delete_session(session_id)
        assert await backend.copies(session_id) == []
    finally:
        await store.close()
