"""Completed search calls retain identity without inventing unavailable metadata."""

from copy import deepcopy

import pytest
from tests.core.test_openai_optional_search_evidence import response_fixture, stream_fixture
from tests.core.test_openai_provider import RecordingTransport

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    HostedToolCallPart,
    Message,
    ModelRequest,
    OpenAIProvider,
    OpenAIWebSearch,
    ProviderStatePart,
    RunRequest,
    SQLiteSessionStore,
)
from cayu.providers import OpenAIProtocolError, build_openai_payload, openai_response_events
from cayu.providers.openai import _normalized_web_search_call


@pytest.mark.anyio
@pytest.mark.parametrize("terminal_only", [False, True])
@pytest.mark.parametrize("explicit_null", [False, True])
async def test_completed_missing_action_persists_and_replays(
    tmp_path, terminal_only, explicit_null
):
    response = response_fixture(None)
    if not explicit_null:
        response["output"][0].pop("action")
    parsed = openai_response_events(response)
    assert parsed[-1].payload["hosted_tool_usage"] == {
        "web_search_calls": 1,
        "web_search_outcome_unknown": 0,
    }
    database = tmp_path / "search.sqlite"
    store = SQLiteSessionStore(database)
    transport = RecordingTransport(stream_events=[stream_fixture(response, terminal_only)])
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(OpenAIProvider(api_key="offline", transport=transport), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-5.6"), hosted_tools=[OpenAIWebSearch()]
    )
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant", session_id="missing", messages=[Message.text("user", "go")]
            )
        )
    ]
    assert not [e for e in events if e.type in {EventType.MODEL_ERROR, EventType.MODEL_RETRY}]
    assert len(transport.calls) == 1
    await store.close()
    reopened = SQLiteSessionStore(database)
    transcript = await reopened.load_transcript("missing")
    parts = [p for m in transcript for p in m.content if isinstance(p, HostedToolCallPart)]
    assert len(parts) == 1
    assert parts[0].status == "completed"
    assert parts[0].call_id == "ws_optional"
    assert parts[0].action is None
    assert HostedToolCallPart.model_validate_json(parts[0].model_dump_json()) == parts[0]
    durable = await reopened.load_events("missing")
    completed = [
        e
        for e in durable
        if e.type == EventType.MODEL_HOSTED_TOOL_CALL and e.payload["status"] == "completed"
    ]
    assert len(completed) == 1
    assert completed[0].payload.get("action") is None
    assert "source_count" not in completed[0].payload
    # Exercise both opaque provider state and neutral transcript reconstruction.
    for neutral in [False, True]:
        messages = [
            m.model_copy(
                update={
                    "content": [
                        p for p in m.content if not (neutral and isinstance(p, ProviderStatePart))
                    ]
                }
            )
            for m in transcript
        ]
        payload = build_openai_payload(
            ModelRequest(model="gpt-5.6", messages=messages),
            reasoning_state="server" if neutral else "inline",
            chain=not neutral,
        )
        calls = [i for i in payload["input"] if i.get("type") == "web_search_call"]
        assert calls == [{"type": "web_search_call", "id": "ws_optional", "status": "completed"}]
    await reopened.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", ""),
        ("id", " "),
        ("id", 4),
        ("status", "unknown"),
        ("action", {}),
        ("action", []),
        ("action", {"type": "screenshot"}),
    ],
)
def test_missing_action_does_not_relax_invalid_fields(field, value):
    item = deepcopy(response_fixture(None)["output"][0])
    item[field] = value
    with pytest.raises(OpenAIProtocolError):
        _normalized_web_search_call(item, item_index=0)
