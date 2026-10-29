"""Keyed argument identity on tool starts, so readers can spot exact repeats."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

from pydantic import SecretStr
from tests.core.session_store_conformance.support import _public_authority_alias_codec
from tests.core.test_fork_controller_reads import _controller_server
from tests.core.test_session_fork_overlays import _http
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events
from tests.runtime.runtime_tool_support import EchoTool

from cayu import AgentSpec, CayuApp, Event, EventType, Message, RunRequest
from cayu.fork_controllers import ForkController
from cayu.providers.base import ModelStreamEvent
from cayu.server.fork_controller import _controller_event
from cayu.sessions import _correlation as correlation
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import EventRecord
from cayu.storage import SQLiteSessionStore
from cayu.tools._argument_publication import ARGUMENTS_IDENTITY_FIELD, arguments_identity_fields
from cayu.vaults import SecretRedactor

AGENT = "assistant"
SUBJECT = "svc:compass"
FIELD = "arguments_correlation_id"
FIRST = "sess_first"
SECOND = "sess_second"
CALLS = (("call_a1", "a"), ("call_a2", "a"), ("call_b", "b"))


def _turns() -> list[list[ModelStreamEvent]]:
    # Each session proposes the same three calls in one round, then stops.
    round_ = [
        *(
            ModelStreamEvent.tool_call(id=call_id, name="echo", arguments={"text": text})
            for call_id, text in CALLS
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]
    done = [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]
    return [round_, done, round_, done]


def _app(store) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(FakeProvider(_turns()), default=True)
    app.register_agent(AgentSpec(name=AGENT, model="fake-model"), tools=[EchoTool()])
    app.enroll_fork_controller(ForkController(name="compass", subjects=(SUBJECT,), agents=(AGENT,)))
    return app


def _digest(text: str, session_id: str = FIRST) -> str:
    """The private digest the Runtime records for an echo call in a session."""

    return arguments_identity_fields({"text": text}, session_id=session_id, publish_arguments=True)[
        ARGUMENTS_IDENTITY_FIELD
    ]


async def _reads(store) -> dict[str, Any]:
    """Run both sessions and read their tool starts through every read path."""

    app = _app(store)
    streamed = {}
    for session_id in (FIRST, SECOND):
        streamed[session_id] = await collect_events(
            app,
            RunRequest(
                agent_name=AGENT,
                session_id=session_id,
                messages=[Message.text("user", "go")],
            ),
        )
    records = await store.query_events(EventQuery(session_ids=(FIRST, SECOND), limit=1000))
    compass = app.fork_controller_access(SUBJECT)
    server = _controller_server(app, SUBJECT)
    controller = {}
    for session_id in (FIRST, SECOND):
        response = await _http(
            server, "GET", f"/fork-controller/sessions/{session_id}/events", params={"limit": 1000}
        )
        assert response.status_code == 200, response.text
        controller[session_id] = response.json()["events"]
    return {
        "records": records,
        "streamed": streamed,
        "control_plane": [
            app._project_persisted_event_record_for_exposure(record).event for record in records
        ],
        "controller": controller,
        "sdk": [compass.event_correlation_ids(record) for record in records],
    }


def _starts(events, session_id: str | None = None) -> list[dict[str, Any]]:
    return [
        event if type(event) is dict else event.model_dump(mode="json")
        for event in events
        if (event["type"] if type(event) is dict else event.type) == "tool.call.started"
        and (
            session_id is None
            or (event["session_id"] if type(event) is dict else event.session_id) == session_id
        )
    ]


def test_same_arguments_in_one_session_share_an_identity_that_reveals_nothing() -> None:
    store = InMemorySessionStore(public_authority_alias_codec=_public_authority_alias_codec())
    reads = asyncio.run(_reads(store))

    # The durable start keeps only the private digest of the proposed arguments.
    private_starts = [
        r.event for r in reads["records"] if r.event.type is EventType.TOOL_CALL_STARTED
    ]
    assert [start.payload[ARGUMENTS_IDENTITY_FIELD] for start in private_starts] == [
        _digest(text, session_id) for session_id in (FIRST, SECOND) for _call_id, text in CALLS
    ]
    assert all("arguments" not in start.payload for start in private_starts)

    for session_id in (FIRST, SECOND):
        controller = _starts(reads["controller"][session_id])
        streamed = _starts(reads["streamed"][session_id])
        published = _starts(reads["control_plane"], session_id)
        assert len(controller) == len(streamed) == len(published) == 3
        values = [event["payload"][FIELD] for event in controller]
        # Every read path publishes the same value.
        assert [event["payload"][FIELD] for event in streamed] == values
        assert [event["payload"][FIELD] for event in published] == values
        a1, a2, b = values
        assert a1 == a2 and a1 != b
        assert all(".arguments_correlation." in value for value in values)
    # The same arguments in another session have an unrelated value.
    first = {e["payload"][FIELD] for e in _starts(reads["controller"][FIRST])}
    second = {e["payload"][FIELD] for e in _starts(reads["controller"][SECOND])}
    assert len(first) == len(second) == 2 and not first & second

    # The SDK returns the values the HTTP routes add.
    by_sequence = dict(zip((r.sequence for r in reads["records"]), reads["sdk"], strict=True))
    for session_id in (FIRST, SECOND):
        for event in _starts(reads["controller"][session_id]):
            assert by_sequence[event["sequence"]][FIELD] == event["payload"][FIELD]
    # Terminals and model events carry none.
    for session_id in (FIRST, SECOND):
        for event in reads["controller"][session_id]:
            if event["type"] != "tool.call.started":
                assert FIELD not in event["payload"], event["type"]

    # Neither the digest nor the raw arguments reach a public or controller read.
    text = json.dumps(
        [
            reads["controller"],
            [e.model_dump(mode="json") for events in reads["streamed"].values() for e in events],
            [e.model_dump(mode="json") for e in reads["control_plane"]],
        ]
    )
    for text_value in ("a", "b"):
        for session_id in (FIRST, SECOND):
            digest = _digest(text_value, session_id)
            assert digest not in text and digest.removeprefix("sha256:") not in text
    assert ARGUMENTS_IDENTITY_FIELD not in text


def test_keyless_stores_withhold_the_arguments_identity(tmp_path) -> None:
    store = SQLiteSessionStore(tmp_path / "keyless.sqlite")

    async def scenario():
        try:
            return await _reads(store)
        finally:
            await store.close()

    reads = asyncio.run(scenario())
    private_starts = [
        r.event for r in reads["records"] if r.event.type is EventType.TOOL_CALL_STARTED
    ]
    # The private digest needs no key, so adding a keyring later works.
    assert private_starts
    assert all(ARGUMENTS_IDENTITY_FIELD in start.payload for start in private_starts)
    for session_id in (FIRST, SECOND):
        for events in (
            reads["controller"][session_id],
            _starts(reads["streamed"][session_id]),
            _starts(reads["control_plane"], session_id),
        ):
            assert events
            for event in events:
                payload = event["payload"]
                assert FIELD not in payload and ARGUMENTS_IDENTITY_FIELD not in payload
    assert all(FIELD not in ids for ids in reads["sdk"])


def test_producers_cannot_supply_or_forge_the_arguments_identity() -> None:
    from cayu.runtime._event_projection import (
        prepare_new_runtime_event,
        project_persisted_runtime_event,
        project_runtime_event,
    )

    codec = _public_authority_alias_codec()
    store = InMemorySessionStore(public_authority_alias_codec=codec)

    async def scenario():
        app = _app(store)
        await collect_events(
            app,
            RunRequest(agent_name=AGENT, session_id=FIRST, messages=[Message.text("user", "go")]),
        )
        return app, await store.query_events(EventQuery(session_id=FIRST, limit=1000))

    app, records = asyncio.run(scenario())
    compass = app.fork_controller_access(SUBJECT)
    [start, *_] = [r for r in records if r.event.type is EventType.TOOL_CALL_STARTED]
    genuine = correlation.arguments_correlation_value(start.event, codec)
    assert genuine is not None

    # A producer-written public value is dropped from every read; the derived one wins.
    forged = EventRecord(
        sequence=start.sequence,
        event=start.event.model_copy(update={"payload": {**start.event.payload, FIELD: "forged"}}),
    )
    controller = _controller_event(app, compass, forged)
    assert controller["payload"][FIELD] == genuine
    redactor = SecretRedactor()
    public = project_persisted_runtime_event(
        forged.event,
        sequence=forged.sequence,
        redactor=redactor,
        public_authority_alias_codec=codec,
    )
    assert public.payload[FIELD] == genuine
    # A caller-supplied record is not durable evidence: no value, forged or derived.
    untrusted = project_runtime_event(
        forged.event,
        sequence=forged.sequence,
        redactor=redactor,
        public_authority_alias_codec=codec,
    )
    assert FIELD not in untrusted.payload
    assert ARGUMENTS_IDENTITY_FIELD not in untrusted.payload

    # A start whose digest the Runtime did not attest keeps no digest durably.
    unattested = Event(
        type=EventType.TOOL_CALL_STARTED,
        session_id=FIRST,
        tool_name="echo",
        payload={
            "tool_call_id": "call_forged",
            "arguments_state": "quarantined",
            ARGUMENTS_IDENTITY_FIELD: _digest("a"),
        },
    )
    prepared = prepare_new_runtime_event(unattested, redactor=redactor)
    assert ARGUMENTS_IDENTITY_FIELD not in prepared.payload
    # A malformed stored digest yields no value rather than a misleading one.
    malformed = start.event.model_copy(
        update={"payload": {**start.event.payload, ARGUMENTS_IDENTITY_FIELD: "not-a-digest"}}
    )
    assert correlation.arguments_correlation_value(malformed, codec) is None
    assert FIELD not in correlation.event_correlation_ids(malformed, codec)


def test_starts_recorded_before_the_digest_still_read_without_an_identity() -> None:
    from cayu.runtime._event_projection import project_persisted_runtime_event

    codec = _public_authority_alias_codec()
    store = InMemorySessionStore(public_authority_alias_codec=codec)

    async def scenario():
        app = _app(store)
        await collect_events(
            app,
            RunRequest(agent_name=AGENT, session_id=FIRST, messages=[Message.text("user", "go")]),
        )
        return app, await store.query_events(EventQuery(session_id=FIRST, limit=1000))

    app, records = asyncio.run(scenario())
    compass = app.fork_controller_access(SUBJECT)
    [start, *_] = [r for r in records if r.event.type is EventType.TOOL_CALL_STARTED]
    payload = {k: v for k, v in start.event.payload.items() if k != ARGUMENTS_IDENTITY_FIELD}
    legacy = EventRecord(
        sequence=start.sequence, event=start.event.model_copy(update={"payload": payload})
    )
    controller = _controller_event(app, compass, legacy)
    assert FIELD not in controller["payload"]
    assert controller["payload"]["tool_call_correlation_id"]
    assert FIELD not in compass.event_correlation_ids(legacy)
    public = project_persisted_runtime_event(
        legacy.event,
        sequence=legacy.sequence,
        redactor=SecretRedactor(),
        public_authority_alias_codec=codec,
    )
    assert FIELD not in public.payload and public.payload["arguments_state"] == "quarantined"


def test_no_event_but_a_derived_tool_start_carries_the_arguments_identity() -> None:
    from cayu.runtime._event_projection import project_persisted_runtime_event

    # A provider's completion metadata could hold the name; public reads still drop it.
    for kind in (EventType.MODEL_STARTED, EventType.MODEL_COMPLETED, EventType.TOOL_CALL_STARTED):
        event = Event(type=kind, session_id=FIRST, payload={FIELD: "forged"})
        public = project_persisted_runtime_event(event, sequence=1, redactor=SecretRedactor())
        assert FIELD not in public.payload


def _rotated(codec):
    encoded = base64.urlsafe_b64encode(bytes([29]) * 32).decode("ascii").rstrip("=")
    return codec.rotated(active_key_id="rotated", key=SecretStr(encoded))


def test_identical_arguments_match_across_a_key_rotation_and_a_move_to_another_store() -> None:
    old = _public_authority_alias_codec()
    new = _rotated(old)

    async def starts(codec):
        store = InMemorySessionStore(public_authority_alias_codec=codec)
        await _reads(store)
        records = await store.query_events(EventQuery(session_id=FIRST, limit=1000))
        return [r.event for r in records if r.event.type is EventType.TOOL_CALL_STARTED]

    # The same session's calls recorded under the old key and under the rotated one (or in
    # another store with another keyring) store the same private digests.
    before = asyncio.run(starts(old))
    after = asyncio.run(starts(new))
    digests = [start.payload[ARGUMENTS_IDENTITY_FIELD] for start in before]
    assert digests == [start.payload[ARGUMENTS_IDENTITY_FIELD] for start in after]
    # So a call recorded before the rotation and an identical one after it still match when
    # read with the rotated keyring, and the rotation moves every value of the session at once.
    a1_before, a2_before, _ = before
    _, a2_after, b_after = after
    value = correlation.arguments_correlation_value
    assert value(a1_before, new) == value(a2_after, new) == value(a2_before, new)
    assert value(a1_before, new) != value(b_after, new)
    assert value(a1_before, new) != value(a1_before, old)


def test_tools_that_dont_publish_arguments_get_no_digest() -> None:
    assert (
        arguments_identity_fields({"pin": "4821"}, session_id=FIRST, publish_arguments=False) == {}
    )
    published = _digest("a")
    assert published.startswith("sha256:") and published != _digest("a", SECOND)
