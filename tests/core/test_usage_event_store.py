"""The narrow ``cayu_usage_events`` copy that usage rollups and tool totals read.

Durable stores keep the copy with triggers on ``cayu_events``. These tests hold
its contract on SQLite and Postgres: every aggregated event is copied with only
the payload members aggregates read, payload updates and deletions follow, the
migration copies events retained before it, and rollups and tool totals over the
copy equal the in-memory store's, which reads the events themselves.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import LiteralString, cast
from uuid import uuid4

import pytest

from cayu import Event, EventType, Message, RunRequest
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.queries import SessionAggregateFilter
from cayu.sessions.records import SessionIdentity
from cayu.sessions.usage import UsageRollupQuery
from cayu.storage._usage_projection_schema import POSTGRES_PENDING_TABLES, PROJECTION_TABLES
from cayu.storage.migrations import SchemaMode, SchemaTooOld
from cayu.storage.sqlite import SQLiteSessionStore

pytestmark = pytest.mark.anyio
NOW = datetime(2026, 3, 1, 12, tzinfo=UTC)


def _events(session_id: str) -> list[Event]:
    def event(index: int, event_type: EventType, payload: dict, tool_name: str | None = None):
        return Event(
            id=f"{session_id}-{index}",
            session_id=session_id,
            type=event_type,
            timestamp=NOW + timedelta(minutes=index),
            tool_name=tool_name,
            payload=payload,
        )

    metrics = {
        "provider_name": "anthropic",
        "model": "claude-x",
        "input_tokens": 1200,
        "output_tokens": 300,
        "total_tokens": 1500,
        "cache": {"read_tokens": 100, "write_tokens": 20},
        "billing_identity": {"provider_name": "anthropic", "model": "claude-x"},
    }
    return [
        event(
            0,
            EventType.MODEL_COMPLETED,
            {
                "usage_metrics": metrics,
                "usage": {"cost": "0.0123", "cost_currency": "USD", "cost_status": "reported"},
                "id": "request-1",
                "response": {"content": "x" * 4000},
            },
        ),
        event(
            1,
            EventType.MODEL_COMPLETED,
            {"usage_metrics": {**metrics, "model": "claude-y"}, "usage_normalization_failed": True},
        ),
        event(
            2,
            EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
            {"usage_metrics": {**metrics, "input_tokens": 7, "total_tokens": 7}},
        ),
        event(3, EventType.TOOL_CALL_STARTED, {"arguments": {"cmd": "ls"}}, "exec_command"),
        event(4, EventType.TOOL_CALL_COMPLETED, {"output": "y" * 4000}, "exec_command"),
        event(5, EventType.TOOL_CALL_STARTED, {"arguments": {}}, "read_file"),
        event(6, EventType.TOOL_CALL_FAILED, {"error": "no"}, "read_file"),
        event(7, EventType.TOOL_CALL_APPROVED, {}, "exec_command"),
        event(8, EventType.TOOL_CALL_APPROVAL_DENIED, {}, "write_file"),
        event(9, EventType.TOOL_CALL_COMPLETED, {}, None),
        event(
            10,
            EventType.MODEL_HOSTED_TOOL_CALL,
            {
                "tool_type": "web_search",
                "status": "failed",
                "provider_name": "openai",
                "model": "gpt-x",
            },
        ),
        event(
            11,
            EventType.MODEL_HOSTED_TOOL_CALL,
            {"tool_type": "web_search", "status": "in_progress", "provider_name": "openai"},
        ),
        event(12, EventType.SESSION_COMPLETED, {"summary": "done"}),
    ]


def _query(session_id: str, **update) -> UsageRollupQuery:
    return UsageRollupQuery(
        start_at=NOW - timedelta(hours=1),
        end_at=NOW + timedelta(hours=1),
        sessions=SessionAggregateFilter(labels={"copy": session_id}),
        session_group_limit=10,
        include_pricing_inputs=True,
        **update,
    )


async def _populate(store, session_id: str) -> None:
    await store.create(
        RunRequest(
            session_id=session_id,
            agent_name="assistant",
            labels={"copy": session_id},
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
            self.path = tmp_path / "usage-copy.sqlite"
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

    async def copies(self, session_id: str) -> list[tuple[str, str | None, dict]]:
        payload = "payload_json" if self.name == "sqlite" else "payload"
        rows = await self.execute(
            f"SELECT event_type, tool_name, {payload} FROM cayu_usage_events "
            "WHERE session_id = ? ORDER BY sequence",
            (session_id,),
        )
        return [
            (event_type, tool_name, json.loads(value) if isinstance(value, str) else value)
            for event_type, tool_name, value in rows
        ]


@pytest.fixture(params=["sqlite", "postgres"])
def backend(request, tmp_path) -> _Backend:
    return _Backend(request.param, tmp_path, request)


async def _assert_matches_memory(store, session_id: str) -> None:
    memory = InMemorySessionStore()
    await _populate(memory, session_id)
    for update in ({}, {"series": {"unit": "hour", "timezone": "UTC"}}):
        durable = await store.aggregate_usage(_query(session_id, **update))
        expected = await memory.aggregate_usage(_query(session_id, **update))
        assert durable.model_dump(exclude={"as_of"}) == expected.model_dump(exclude={"as_of"})
    durable_tools, _ = await store.tool_usage_totals()
    expected_tools, _ = await memory.tool_usage_totals()
    assert durable_tools == expected_tools


async def test_copy_keeps_only_aggregated_events_and_payload_members(backend) -> None:
    session_id = "copy-" + uuid4().hex
    store = await backend.open()
    try:
        await _populate(store, session_id)
        copies = await backend.copies(session_id)
        assert [event_type for event_type, _, _ in copies] == [
            str(event.type) for event in _events(session_id)[:-1]
        ]
        first = copies[0][2]
        assert set(first) == {"usage_metrics", "usage", "id"}
        assert first["usage_metrics"]["cache"] == {"read_tokens": 100, "write_tokens": 20}
        assert copies[1][2]["usage_normalization_failed"] is True
        assert copies[4] == ("tool.call.completed", "exec_command", {})
        assert copies[10][2] == {
            "tool_type": "web_search",
            "status": "failed",
            "provider_name": "openai",
            "model": "gpt-x",
        }
        await _assert_matches_memory(store, session_id)
        await store.delete_session(session_id)
    finally:
        await store.close()


async def test_copy_follows_payload_updates_and_deletions(backend) -> None:
    session_id = "copy-" + uuid4().hex
    store = await backend.open()
    try:
        await _populate(store, session_id)
        payload = "payload_json" if backend.name == "sqlite" else "payload"
        cast = "" if backend.name == "sqlite" else "::jsonb"
        await backend.execute(
            f"UPDATE cayu_events SET {payload} = ?{cast} WHERE session_id = ? AND event_id = ?",
            (json.dumps({"usage_metrics": {"total_tokens": 9}}), session_id, f"{session_id}-0"),
        )
        copies = await backend.copies(session_id)
        assert copies[0][2] == {"usage_metrics": {"total_tokens": 9}}
        await store.delete_session(session_id)
        assert await backend.copies(session_id) == []
    finally:
        await store.close()


async def test_migration_copies_events_retained_before_it(backend) -> None:
    session_id = "copy-" + uuid4().hex
    store = await backend.open()
    try:
        await _populate(store, session_id)
    finally:
        await store.close()
    on_table = "" if backend.name == "sqlite" else " ON cayu_events"
    for trigger in ("insert", "delete", "update"):
        await backend.execute(f"DROP TRIGGER cayu_usage_events_{trigger}{on_table}")
    # The stored totals of the usage-projection revision go too; on Postgres its
    # functions take a cayu_usage_events row, so they drop with the table.
    cascade = "" if backend.name == "sqlite" else " CASCADE"
    await backend.execute(f"DROP TABLE cayu_usage_events{cascade}")
    stored = PROJECTION_TABLES + (() if backend.name == "sqlite" else POSTGRES_PENDING_TABLES)
    for table in stored:
        await backend.execute(f"DROP TABLE {table}")
    # Back to revision 148, as before the usage copy and every later revision.
    await backend.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 149")
    with pytest.raises(SchemaTooOld, match="requires >= "):
        await (await backend.open()).close()
    store = await backend.open(SchemaMode.MIGRATE)
    try:
        assert len(await backend.copies(session_id)) == len(_events(session_id)) - 1
        await _assert_matches_memory(store, session_id)
        await store.delete_session(session_id)
    finally:
        await store.close()
