"""Stored events decode without re-walking their payloads, and decode the same.

``event_from_row`` (SQLite) and ``event_from_document`` (PostgreSQL) rebuild
rows the store's own ingestion validated. These tests pin that they equal the
validating decode, private authority included, that a tampered row is still
rejected, and that no payload is walked or stamped on the way.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import cayu._validation as validation
import cayu.events as events_module
from cayu import AgentSpec, CayuApp, Message, RunRequest, Tool, ToolResult, ToolSpec
from cayu.evals import ScriptedModelProvider
from cayu.events import Event, EventType, copy_event
from cayu.messages import MessageRole
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.sessions.event_delivery import (
    MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY,
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    restore_persisted_event_authority,
)
from cayu.sessions.event_queries import EventQuery
from cayu.storage import SQLiteSessionStore
from cayu.storage import _postgres_stored_events as pg_stored_events
from cayu.storage import _sqlite_records as sqlite_records
from cayu.tools.base import ToolContext

_PROOF_ATTRIBUTES = ("_payload_validation_stamp", "_stored_payload_json")


def _private_state(event: Event) -> dict[str, Any]:
    """Private attributes that carry meaning; validation proofs differ by design."""

    return {
        name: value
        for name, value in (event.__pydantic_private__ or {}).items()
        if name not in _PROOF_ATTRIBUTES
    }


_ROW_COLUMNS = ("sequence", *sqlite_records.EVENT_COLUMN_NAMES)


def _as_row(values: dict[str, Any]) -> sqlite3.Row:
    """A real ``sqlite3.Row`` holding exactly these column values."""

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        columns = ", ".join(f"? AS {name}" for name in _ROW_COLUMNS)
        return connection.execute(
            f"SELECT {columns}", [values[name] for name in _ROW_COLUMNS]
        ).fetchone()
    finally:
        connection.close()


def _assert_same_decode(trusted: Event, validated: Event) -> None:
    assert trusted == validated
    assert type(trusted.type) is type(validated.type)
    assert trusted.timestamp.tzinfo == validated.timestamp.tzinfo
    assert trusted.model_fields_set == validated.model_fields_set
    assert _private_state(trusted) == _private_state(validated)
    assert json.dumps(trusted.payload) == json.dumps(validated.payload)
    # A copy of the trusted event is as cheap and as equal as a copy of the other.
    copied = copy_event(trusted)
    assert copied == validated
    assert _private_state(copied) == _private_state(validated)


# --- Random rows -------------------------------------------------------------

_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    max_size=12,
)
_SCALARS = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**53), max_value=2**53)
    | st.floats(allow_nan=False, allow_infinity=False, width=64).filter(
        lambda value: not value.is_integer()
    )
    | _TEXT
)
_JSON = st.recursive(
    _SCALARS,
    lambda children: st.lists(children, max_size=4) | st.dictionaries(_TEXT, children, max_size=4),
    max_leaves=20,
)
_AUTHORITY_VALUE = st.text(alphabet="abcdef0123456789-", min_size=1, max_size=16)
_EVENT_TYPES = st.sampled_from(
    [
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.SESSION_STARTED,
        EventType.SESSION_RESUMED,
        EventType.SESSION_FORKED,
        EventType.MODEL_STARTED,
        EventType.SESSION_EXPORT_PUBLISHED,
        "custom.example.thing",
    ]
)
_NAMES = st.none() | st.text(alphabet="abcdefghij-_", min_size=1, max_size=10)


@st.composite
def _rows(draw: Any) -> dict[str, Any]:
    event_type = draw(_EVENT_TYPES)
    payload = draw(st.dictionaries(_TEXT, _JSON, max_size=6))
    # Fields the store attests for each type, as strings, sometimes absent.
    for field_name in (
        "parent_session_id",
        "source_session_id",
        "export_commitment",
        SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
        MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY,
    ):
        if draw(st.booleans()):
            payload[field_name] = draw(_AUTHORITY_VALUE)
    timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(
        microseconds=draw(st.integers(min_value=0, max_value=10**13))
    )
    return {
        "sequence": draw(st.integers(min_value=1, max_value=2**40)),
        "session_id": draw(st.text(alphabet="abcdef0123456789", min_size=1, max_size=20)),
        "event_id": draw(st.text(alphabet="abcdef0123456789-", min_size=1, max_size=36)),
        "interaction_id": draw(_NAMES),
        "event_type": str(event_type),
        "timestamp": sqlite_records.format_datetime(timestamp),
        "agent_name": draw(_NAMES),
        "environment_name": draw(_NAMES),
        "workflow_name": draw(_NAMES),
        "tool_name": draw(_NAMES),
        "payload_json": sqlite_records.json_dumps(payload),
        "input_contract_runtime_owned": draw(st.sampled_from([0, 1])),
        "file_attachment_attestations_runtime_owned": draw(st.sampled_from([0, 1])),
    }


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(row=_rows())
def test_trusted_sqlite_decode_equals_validating_decode(row: dict[str, Any]) -> None:
    stored = _as_row(row)
    _assert_same_decode(
        sqlite_records.event_from_row(stored), sqlite_records.validating_event_from_row(stored)
    )


def _document(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["event_id"],
        "type": row["event_type"],
        "session_id": row["session_id"],
        "interaction_id": row["interaction_id"],
        "timestamp": row["timestamp"],
        "agent_name": row["agent_name"],
        "environment_name": row["environment_name"],
        "workflow_name": row["workflow_name"],
        "tool_name": row["tool_name"],
        "payload": json.loads(row["payload_json"]),
    }


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(row=_rows(), restore=st.booleans())
def test_trusted_postgres_decode_equals_validating_decode(
    row: dict[str, Any], restore: bool
) -> None:
    flags = {
        "input_contract_runtime_owned": row["input_contract_runtime_owned"] == 1,
        "file_attachment_attestations_runtime_owned": (
            row["file_attachment_attestations_runtime_owned"] == 1
        ),
    }
    document = _document(row)
    validated = Event(**json.loads(json.dumps(document)))
    if restore:
        validated = restore_persisted_event_authority(validated, **flags)
        trusted = pg_stored_events.event_from_document(document, restore_authority=True, **flags)
    else:
        trusted = pg_stored_events.event_from_document(document)
    _assert_same_decode(trusted, validated)


# --- Tampered rows -----------------------------------------------------------


def _honest_row() -> dict[str, Any]:
    return {
        "sequence": 7,
        "session_id": "session-1",
        "event_id": "event-1",
        "interaction_id": None,
        "event_type": str(EventType.SESSION_STARTED),
        "timestamp": "2026-01-01T00:00:00.000000+00:00",
        "agent_name": "writer",
        "environment_name": None,
        "workflow_name": None,
        "tool_name": None,
        "payload_json": json.dumps({"parent_session_id": "parent-1", "note": "ok"}),
        "input_contract_runtime_owned": 0,
        "file_attachment_attestations_runtime_owned": 0,
    }


_TAMPERED = {
    "unparseable payload": {"payload_json": '{"note": '},
    "payload not an object": {"payload_json": "[1, 2]"},
    "NaN in payload": {"payload_json": '{"note": NaN}'},
    "infinite float": {"payload_json": '{"note": 1e400}'},
    "integral float out of range": {"payload_json": '{"note": 1e300}'},
    "integer out of range": {"payload_json": '{"note": 9223372036854775808}'},
    "negative integer out of range": {"payload_json": '{"note": -9223372036854775809}'},
    "raw NUL": {"payload_json": '{"note": "a\x00b"}'},
    "escaped NUL": {"payload_json": '{"note": "a\\u0000b"}'},
    "escaped lone surrogate": {"payload_json": '{"note": "\\ud800"}'},
    "blank attested field": {"payload_json": '{"parent_session_id": "  "}'},
    "unknown event type": {"event_type": "not.a.type"},
    "blank session id": {"session_id": "  "},
    "id too long": {"event_id": "x" * 513},
    "id with NUL": {"event_id": "a\x00b"},
    "bad timestamp": {"timestamp": "yesterday"},
    "integer session id": {"session_id": 12},
    "authority flag out of range": {"input_contract_runtime_owned": 2},
    "authority flag as text": {"file_attachment_attestations_runtime_owned": "1"},
}


@pytest.mark.parametrize("change", _TAMPERED.values(), ids=_TAMPERED.keys())
def test_a_tampered_sqlite_row_is_rejected_as_before(change: dict[str, Any]) -> None:
    row = _as_row({**_honest_row(), **change})
    with pytest.raises((ValueError, TypeError)) as validating:
        sqlite_records.validating_event_from_row(row)
    with pytest.raises(type(validating.value)) as trusted:
        sqlite_records.event_from_row(row)
    assert str(trusted.value) == str(validating.value)


_TAMPERED_DOCUMENTS = {
    "payload not an object": {"payload": [1, 2]},
    "integral float out of range": {"payload": {"note": 1e300}},
    "integer out of range": {"payload": {"note": 2**63}},
    "blank attested field": {"payload": {"parent_session_id": "  "}},
    "unknown event type": {"type": "not.a.type"},
    "missing id": {"id": None},
    "extra key": {"surprise": True},
}


@pytest.mark.parametrize("change", _TAMPERED_DOCUMENTS.values(), ids=_TAMPERED_DOCUMENTS.keys())
def test_a_tampered_postgres_document_is_rejected_as_before(change: dict[str, Any]) -> None:
    document = {**_document(_honest_row()), **change}
    with pytest.raises((ValueError, TypeError)) as validating:
        restore_persisted_event_authority(Event(**document))
    with pytest.raises(type(validating.value)) as trusted:
        pg_stored_events.event_from_document(document, restore_authority=True)
    assert str(trusted.value) == str(validating.value)


def test_an_edited_payload_is_validated_again_when_copied() -> None:
    event = sqlite_records.event_from_row(_as_row(_honest_row()))
    event.payload["note"] = math.nan
    with pytest.raises(ValueError):
        copy_event(event)
    event.payload["note"] = "changed"
    copied = copy_event(event)
    assert copied.payload["note"] == "changed"
    assert copied._stored_payload_json is None
    assert copied._payload_validation_stamp is not None


# --- Real events, through the stores -----------------------------------------


class _Inspect(Tool):
    spec = ToolSpec(
        name="inspect",
        description="Read part of the change.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        return ToolResult(content="diff --git a/x b/x\n+changed line\n" * 20)


def _respond(request: ModelRequest) -> list[ModelStreamEvent]:
    usage = {"input_tokens": 1200, "output_tokens": 80}
    if any(message.role is MessageRole.TOOL for message in request.messages):
        return [
            ModelStreamEvent.text_delta("No issues found."),
            ModelStreamEvent.completed({"finish_reason": "stop", "usage": usage}),
        ]
    return [
        ModelStreamEvent.text_delta("Looking at the change."),
        ModelStreamEvent.tool_call(id="call-1", name="inspect", arguments={"path": "x"}),
        ModelStreamEvent.completed({"finish_reason": "tool_calls", "usage": usage}),
    ]


def _app(store: Any) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(ScriptedModelProvider(name="scripted", response_factory=_respond))
    app.register_agent(AgentSpec(name="reviewer", model="scripted-model"), tools=[_Inspect()])
    return app


async def _run(store: Any) -> str:
    """Run one session to completion and return its id."""

    session_id = f"decode-{uuid4().hex}"
    app = _app(store)
    async for _event in app.run(
        RunRequest(
            agent_name="reviewer",
            session_id=session_id,
            messages=[Message.text("user", "Review this change.")],
        )
    ):
        pass
    # Settle background side-effect delivery before anything is counted.
    await app.aclose()
    return session_id


def _store(backend: str, request: pytest.FixtureRequest, tmp_path: Any) -> Any:
    if backend == "sqlite":
        return SQLiteSessionStore(tmp_path / "sessions.db")
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    return PostgresSessionStore(
        request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
    )


async def _validated_records(
    backend: str, store: Any, tmp_path: Any, session_id: str
) -> list[tuple[int, Event]]:
    """Every stored event decoded through full validation, straight from the rows."""

    if backend == "sqlite":
        with sqlite3.connect(tmp_path / "sessions.db") as connection:
            connection.row_factory = sqlite3.Row
            columns = ", ".join(sqlite_records.EVENT_COLUMN_NAMES)
            rows = connection.execute(
                f"SELECT sequence, {columns} FROM cayu_events WHERE session_id = ? "
                "ORDER BY sequence",
                (session_id,),
            ).fetchall()
        return [(row["sequence"], sqlite_records.validating_event_from_row(row)) for row in rows]
    async with store._connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT sequence, event, input_contract_runtime_owned, "
            "file_attachment_attestations_runtime_owned FROM cayu_events "
            "WHERE session_id = %s ORDER BY sequence",
            (session_id,),
        )
        rows = await cur.fetchall()
    return [
        (
            row[0],
            restore_persisted_event_authority(
                Event(**row[1]),
                input_contract_runtime_owned=row[2],
                file_attachment_attestations_runtime_owned=row[3],
            ),
        )
        for row in rows
    ]


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_stored_events_read_back_as_validated(
    backend: str, request: pytest.FixtureRequest, tmp_path: Any
) -> None:
    asyncio.run(_read_back_as_validated(_store(backend, request, tmp_path), backend, tmp_path))


async def _read_back_as_validated(store: Any, backend: str, tmp_path: Any) -> None:
    try:
        session_id = await _run(store)
        expected = await _validated_records(backend, store, tmp_path, session_id)
        records = await store.query_events(EventQuery(session_id=session_id, limit=1000))
        assert [record.sequence for record in records] == [sequence for sequence, _ in expected]
        assert {str(event.type) for _, event in expected} >= {
            "session.started",
            "model.completed",
            "tool.call.started",
            "tool.call.completed",
        }
        for record, (_, validated) in zip(records, expected, strict=True):
            _assert_same_decode(record.event, validated)
    finally:
        await store.close()


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_reading_stored_events_walks_no_payload(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asyncio.run(_read_without_walks(_store(backend, request, tmp_path), monkeypatch))


async def _read_without_walks(store: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        session_id = await _run(store)
        walks: list[str] = []
        stamps: list[str] = []
        walk = validation._walk_bounded_durable_json
        stamp = events_module._durable_payload_stamp

        def counting_walk(value: Any, field_name: str, **kwargs: Any) -> Any:
            # A copy validates its small envelope around an empty payload; only
            # a walk that reaches payload content counts.
            payload = value.get("payload") if field_name == "event" else value
            if field_name in {"payload", "event"} and payload:
                walks.append(field_name)
            return walk(value, field_name, **kwargs)

        def counting_stamp(value: Any) -> object:
            stamps.append(type(value).__name__)
            return stamp(value)

        monkeypatch.setattr(validation, "_walk_bounded_durable_json", counting_walk)
        monkeypatch.setattr(events_module, "_durable_payload_stamp", counting_stamp)
        records = await store.query_events(EventQuery(session_id=session_id, limit=1000))
        assert len(records) > 10
        assert walks == []
        assert stamps == []
        # A later copy still reuses the stored text instead of walking the payload.
        copied = [copy_event(record.event) for record in records]
        assert walks == []
        assert copied == [record.event for record in records]
    finally:
        await store.close()
