"""Native JSON depth survives the extra transcript representation envelope."""

import json
from uuid import uuid4

import pytest
from tests.runtime.runtime_provider_support import NativeStructuredOutputFakeProvider

from cayu import AgentSpec, CayuApp
from cayu.messages import Message
from cayu.providers import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import ResumeRequest, RunRequest
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("lost_ack", [False, True])
async def test_native_structured_depth_survives_publication(
    tmp_path, backend, lost_ack, request, monkeypatch
):
    session_id = "deep-" + uuid4().hex
    output = 1
    for _ in range(62):
        output = [output]
    if backend == "memory":
        store = InMemorySessionStore()
    elif backend == "sqlite":
        store = SQLiteSessionStore(tmp_path / "deep.db")
    else:
        from cayu.storage import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        store = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    original = store.complete_model_completion_stage
    injected = False

    async def complete(*args, **kwargs):
        nonlocal injected
        result = await original(*args, **kwargs)
        if lost_ack and not injected:
            injected = True
            raise OSError("publication acknowledgement lost")
        return result

    monkeypatch.setattr(store, "complete_model_completion_stage", complete)
    provider = NativeStructuredOutputFakeProvider(
        [
            [ModelStreamEvent.text_delta(json.dumps(output)), ModelStreamEvent.completed()],
            [ModelStreamEvent.text_delta("[]"), ModelStreamEvent.completed()],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    try:
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "Answer")],
                    structured_output={"strategy": "native", "json_schema": {"type": "array"}},
                )
            )
        ]
        assert any(event.type.value == "session.completed" for event in events), [
            (
                event.type.value,
                {key: value for key, value in event.payload.items() if "path" not in key},
            )
            for event in events
            if event.type.value == "session.failed"
        ]
        assert injected is lost_ack
        assert len(provider.requests) == 1
        transcript = await store.load_transcript(session_id)
        assert [
            part.output
            for message in transcript
            for part in message.content
            if part.type == "structured_output"
        ] == [output]
        continued = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "Again")],
                    structured_output={"strategy": "native", "json_schema": {"type": "array"}},
                )
            )
        ]
        assert any(event.type.value == "session.completed" for event in continued)
        assert len(provider.requests) == 2
        assert all(
            part.type != "structured_output"
            for message in provider.requests[-1].messages
            for part in message.content
        )
        transcript = await store.load_transcript(session_id)
        assert [
            part.output
            for message in transcript
            for part in message.content
            if part.type == "structured_output"
        ] == [output, []]
        if backend == "sqlite":
            await store.close()
            store = SQLiteSessionStore(tmp_path / "deep.db")
            assert transcript == await store.load_transcript(session_id)
    finally:
        if backend != "memory":
            await store.close()


def test_transcript_envelope_keeps_payload_and_other_part_depth_limits():
    from cayu._validation import DurableValueError, copy_durable_json_value
    from cayu.messages import StructuredOutputPart, ToolCallPart

    output = 1
    for _ in range(64):
        output = [output]
    assert copy_durable_json_value(output, "output") == output
    with pytest.raises(DurableValueError):
        copy_durable_json_value([output], "output")
    with pytest.raises(ValueError):
        StructuredOutputPart(name=None, output=[output])
    arguments = 1
    for _ in range(61):
        arguments = [arguments]
    call = ToolCallPart(tool_call_id="call", tool_name="tool", arguments={"value": arguments})
    for evidence in ((), (StructuredOutputPart(name=None, output={"ok": True}),)):
        with pytest.raises(ValueError):
            Message(role="assistant", content=(call, *evidence))
