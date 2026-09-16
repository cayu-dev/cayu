from __future__ import annotations

import asyncio
import io
import json

import pytest

from cayu._validation import DurableValueError
from cayu.events import Event, EventType
from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.sessions.exports import SessionExportLimits, SessionExportTooLarge
from cayu.storage import SQLiteSessionStore
from cayu.storage.jsonl_export import export_sessions, import_sessions


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("shape", ["bytes", "nodes"])
def test_admitted_history_exceeds_single_value_limits_and_roundtrips(tmp_path, backend, shape):
    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "history.sqlite")
        )
        try:
            await store.create(
                RunRequest(session_id="history", agent_name="test", messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            payload = {"text": "x" * (1024 * 1024)} if shape == "bytes" else {"items": [0] * 30000}
            events = []
            for index in range(18 if shape == "bytes" else 9):
                event = Event(
                    session_id="history",
                    type=EventType.TOOL_CALL_STARTED,
                    payload={**payload, "index": index},
                )
                await store.append_event("history", event)
                events.append(event)
            stream = io.StringIO()
            assert await export_sessions(store, stream=stream) == 1
            line = stream.getvalue()
            if shape == "bytes":
                assert len(line.encode()) > 16 * 1024 * 1024
            [restored] = import_sessions([line])
            assert [e.model_dump(mode="json") for e in restored.events] == [
                e.model_dump(mode="json") for e in events
            ]
            assert restored.boundary is not None
            assert len(restored.boundary.event_sequences) == len(events)
            second = InMemorySessionStore()
            await second.create(
                RunRequest(session_id="history", agent_name="test", messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            for event in restored.events:
                await second.append_event("history", event)
            output = io.StringIO()
            await export_sessions(second, stream=output)
            [again] = import_sessions([output.getvalue()])
            assert [e.model_dump(mode="json") for e in again.events] == [
                e.model_dump(mode="json") for e in restored.events
            ]
        finally:
            if hasattr(store, "close"):
                await store.close()

    asyncio.run(run())


def test_explicit_envelope_boundary_is_inclusive_and_never_truncates():
    async def run():
        store = InMemorySessionStore()
        await store.create(
            RunRequest(session_id="small", agent_name="test", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        stream = io.StringIO()
        await export_sessions(store, stream=stream)
        line = stream.getvalue()
        size = len(line.encode())
        exact = io.StringIO()
        await export_sessions(store, stream=exact, limits=SessionExportLimits(max_bytes=size))
        assert exact.getvalue() == line
        assert len(list(import_sessions([line], max_bytes=size))) == 1
        with pytest.raises(DurableValueError, match="byte limit"):
            list(import_sessions([line], max_bytes=size - 1))
        rejected = io.StringIO()
        with pytest.raises(SessionExportTooLarge):
            await export_sessions(
                store, stream=rejected, limits=SessionExportLimits(max_bytes=size - 1)
            )
        assert rejected.getvalue() == ""

    asyncio.run(run())


@pytest.mark.parametrize("literal", ["NaN", "9223372036854775808"])
def test_larger_session_envelope_keeps_portable_number_validation(literal):
    with pytest.raises(DurableValueError):
        list(
            import_sessions(
                ['{"type":"session","value":' + literal + "}"], max_bytes=128 * 1024 * 1024
            )
        )


def test_larger_session_envelope_keeps_individual_event_limits():
    async def run():
        store = InMemorySessionStore()
        await store.create(
            RunRequest(session_id="small", agent_name="test", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        stream = io.StringIO()
        await export_sessions(store, stream=stream)
        document = json.loads(stream.getvalue())
        event = Event(session_id="small", type=EventType.TOOL_CALL_STARTED).model_dump(mode="json")
        event["payload"] = {"text": "x" * (17 * 1024 * 1024)}
        document["events"] = [event]
        document["snapshot"]["event_sequences"] = [1]
        document["snapshot"]["through_sequence"] = 1
        with pytest.raises(ValueError):
            list(import_sessions([json.dumps(document)], max_bytes=128 * 1024 * 1024))

    asyncio.run(run())


def test_native_cli_respects_explicit_session_size(tmp_path, capsys):
    from cayu.cli import main

    database = tmp_path / "cli.sqlite"

    async def seed():
        store = SQLiteSessionStore(database)
        try:
            await store.create(
                RunRequest(session_id="cli-history", agent_name="test", messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            for index in range(2):
                await store.append_event(
                    "cli-history",
                    Event(
                        session_id="cli-history",
                        type=EventType.TOOL_CALL_STARTED,
                        payload={"index": index, "text": "x" * 700000},
                    ),
                )
        finally:
            await store.close()

    asyncio.run(seed())
    output = tmp_path / "export.jsonl"
    common = ["storage", "export", "--sqlite", str(database), "--jsonl", "--output", str(output)]
    assert main([*common, "--max-session-bytes", "1048576"]) == 1
    assert not output.exists() or output.stat().st_size == 0
    capsys.readouterr()
    assert main([*common, "--max-session-bytes", "4194304", "--max-record-bytes", "1048576"]) == 0
    [restored] = import_sessions(output.read_text().splitlines(), max_bytes=4194304)
    assert [e.payload["index"] for e in restored.events] == [0, 1]
