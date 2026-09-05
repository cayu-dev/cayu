"""Hermetic regressions for optional Responses hosted-search evidence (#1393)."""

from copy import deepcopy

import pytest
from pydantic import ValidationError
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
    WebSearchAction,
)
from cayu.providers import (
    ModelStreamEventType,
    OpenAIProtocolError,
    build_openai_payload,
    openai_response_events,
)

ACTIONS = [
    {"type": "open_page"},
    {"type": "open_page", "url": None},
    {"type": "open_page", "url": "https://example.com/"},
    {"type": "search"},
    {"type": "search", "queries": []},
    {"type": "search", "sources": []},
]


def response_fixture(action):
    return {
        "id": "resp_optional",
        "model": "gpt-5.6",
        "status": "completed",
        "output": [
            {
                "type": "web_search_call",
                "id": "ws_optional",
                "status": "completed",
                "action": action,
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done."}],
            },
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def stream_fixture(response, terminal_only=False):
    events = []
    if not terminal_only:
        events = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "web_search_call", "id": "ws_optional", "status": "in_progress"},
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": deepcopy(response["output"][0]),
            },
        ]
    return [*events, {"type": "response.completed", "response": response}]


@pytest.mark.parametrize("action", ACTIONS)
def test_optional_action_model_and_nonstream_replay(action):
    evidence = WebSearchAction.model_validate(action)
    assert evidence.url == action.get("url")
    assert WebSearchAction.model_validate_json(evidence.model_dump_json()) == evidence
    events = openai_response_events(response_fixture(action))
    hosted = [event for event in events if event.type == ModelStreamEventType.HOSTED_TOOL_CALL]
    expected = {key: value for key, value in action.items() if value is not None}
    assert len(hosted) == 1
    assert hosted[0].payload["status"] == "completed"
    assert hosted[0].payload["action"] == expected
    assert events[-1].payload["hosted_tool_usage"] == {
        "web_search_calls": 1,
        "web_search_outcome_unknown": 0,
    }
    state = events[-1].payload["provider_state"]
    payload = build_openai_payload(
        ModelRequest(
            model="gpt-5.6",
            messages=[
                Message(
                    role="assistant",
                    content=[ProviderStatePart.model_validate(part) for part in state],
                )
            ],
        )
    )
    assert payload["input"][0]["action"] == expected


@pytest.mark.anyio
@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("terminal_only", [False, True])
async def test_optional_action_public_stream(action, terminal_only):
    response = response_fixture(action)
    transport = RecordingTransport(stream_events=[stream_fixture(response, terminal_only)])
    provider = OpenAIProvider(api_key="offline", transport=transport)
    try:
        events = [
            event
            async for event in provider.stream(
                ModelRequest(model="gpt-5.6", messages=[Message.text("user", "go")])
            )
        ]
    finally:
        await provider.aclose()
    assert events[-1].type == ModelStreamEventType.COMPLETED
    hosted = [event for event in events if event.type == ModelStreamEventType.HOSTED_TOOL_CALL]
    assert [event.payload["status"] for event in hosted] == (
        ["completed"] if terminal_only else ["in_progress", "completed"]
    )
    assert hosted[-1].payload["action"] == {
        key: value for key, value in action.items() if value is not None
    }
    assert events[-1].payload["hosted_tool_usage"] == {
        "web_search_calls": 1,
        "web_search_outcome_unknown": 0,
    }


@pytest.mark.anyio
@pytest.mark.parametrize("action", ACTIONS)
async def test_optional_action_durable_runtime_readback(tmp_path, action):
    database = tmp_path / "optional.sqlite"
    store = SQLiteSessionStore(database)
    transport = RecordingTransport(stream_events=[stream_fixture(response_fixture(action))])
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(OpenAIProvider(api_key="offline", transport=transport), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-5.6"), hosted_tools=[OpenAIWebSearch()]
    )
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant", session_id="optional", messages=[Message.text("user", "go")]
            )
        )
    ]
    assert not [
        event for event in events if event.type in {EventType.MODEL_ERROR, EventType.MODEL_RETRY}
    ]
    assert len(transport.calls) == 1
    await store.close()
    reopened = SQLiteSessionStore(database)
    transcript = await reopened.load_transcript("optional")
    parts = [
        part
        for message in transcript
        for part in message.content
        if isinstance(part, HostedToolCallPart)
    ]
    assert len(parts) == 1
    assert parts[0].status == "completed"
    assert parts[0].action == WebSearchAction.model_validate(action)
    persisted = await reopened.load_events("optional")
    hosted = [event for event in persisted if event.type == EventType.MODEL_HOSTED_TOOL_CALL]
    assert [event.payload["status"] for event in hosted] == ["in_progress", "completed"]
    # Replay the actual persisted assistant message, including provider state.
    payload = build_openai_payload(ModelRequest(model="gpt-5.6", messages=transcript))
    calls = [item for item in payload["input"] if item.get("type") == "web_search_call"]
    assert len(calls) == 1
    assert calls[0]["action"] == {key: value for key, value in action.items() if value is not None}
    await reopened.close()


@pytest.mark.parametrize(
    "action",
    [
        *[
            {"type": "open_page", "url": value}
            for value in [
                "",
                "javascript:alert(1)",
                "https://user@example.com/",
                "https://bad host/",
                "https://example.com/" + "x" * 4096,
                42,
            ]
        ],
        {"type": "unsupported"},
        {"type": "find_in_page", "pattern": "text"},
        {"type": "find_in_page", "url": "https://example.com/"},
        {"type": "search", "query": " "},
        {"type": "search", "queries": [""]},
    ],
)
def test_optional_action_preserves_invalid_evidence_rejection(action):
    with pytest.raises(ValidationError):
        WebSearchAction.model_validate(action)
    with pytest.raises(OpenAIProtocolError):
        openai_response_events(response_fixture(action))


@pytest.mark.anyio
@pytest.mark.parametrize("conflict", ["identity", "lifecycle", "terminal_action"])
async def test_missing_url_does_not_weaken_stream_consistency(conflict):
    raw = stream_fixture(response_fixture({"type": "open_page"}))
    if conflict == "identity":
        raw[1]["item"]["id"] = "ws_other"
    elif conflict == "lifecycle":
        raw.insert(
            1,
            {
                "type": "response.web_search_call.searching",
                "output_index": 0,
                "item_id": "ws_other",
            },
        )
    else:
        raw[-1]["response"]["output"][0]["action"]["url"] = "https://example.com/"
    provider = OpenAIProvider(api_key="offline", transport=RecordingTransport(stream_events=[raw]))
    try:
        events = [
            event
            async for event in provider.stream(
                ModelRequest(model="gpt-5.6", messages=[Message.text("user", "go")])
            )
        ]
    finally:
        await provider.aclose()
    assert events[-1].type == ModelStreamEventType.ERROR
    assert not any(event.type == ModelStreamEventType.COMPLETED for event in events)
