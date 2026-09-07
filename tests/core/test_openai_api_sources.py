"""Observed API source shape, with an explicitly redacted replacement name."""

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError
from tests.core.test_openai_optional_search_evidence import (
    API_SOURCE,
    response_fixture,
    stream_fixture,
)
from tests.core.test_openai_provider import RecordingTransport

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ModelRequest,
    OpenAIProvider,
    OpenAIWebSearch,
    RetryPolicy,
    RunRequest,
    SQLiteSessionStore,
    WebSearchAction,
    WebSearchAPISource,
    WebSearchSource,
)
from cayu.providers import ModelStreamEventType, OpenAIProtocolError, openai_response_events

REASON = "web_search_action_sources_name_is_invalid"
INVALID_NAMES = [
    None,
    True,
    42,
    [],
    {},
    "",
    " ",
    " redacted",
    "redacted ",
    "x" * 1025,
    "secret\x00name",
    "secret\ud800name",
]


@pytest.mark.parametrize("name", INVALID_NAMES)
def test_api_source_invalid_name_is_safe(name):
    with pytest.raises(ValidationError):
        WebSearchAPISource(name=name)
    with pytest.raises(OpenAIProtocolError) as caught:
        openai_response_events(
            response_fixture(
                {
                    "type": "search",
                    "sources": [
                        {"type": "api", "name": name},
                    ],
                }
            )
        )
    assert caught.value.reason_code == REASON
    assert "secret" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_api_source_exact_name_and_url_compatibility():
    name = "redacted / API 名称"  # Synthetic non-enum name, retained exactly.
    action = WebSearchAction(
        type="search",
        sources=[
            WebSearchSource(url="https://example.com"),
            {"type": "api", "name": name},
            {"url": "https://example.org"},
        ],
    )
    assert action.sources[1].model_dump() == {"type": "api", "name": name}
    assert WebSearchAction.model_validate_json(action.model_dump_json()) == action
    for invalid in (
        {"name": "redacted"},
        {"type": "api", "url": "https://example.com"},
        {"type": "unknown", "name": "redacted"},
    ):
        with pytest.raises(ValidationError):
            WebSearchAction(type="search", sources=[invalid])
    with pytest.raises(ValidationError):
        WebSearchAPISource(name="redacted", url="https://example.com")


@pytest.mark.anyio
@pytest.mark.parametrize("terminal_only", [False, True])
async def test_malformed_api_source_bounded_durable_failure(tmp_path, terminal_only):
    response = response_fixture({"type": "search", "sources": [{"type": "api"}]})
    raw = stream_fixture(response, terminal_only)
    transport = RecordingTransport(stream_events=[deepcopy(raw) for _ in range(5)])
    provider = OpenAIProvider(api_key="offline", transport=transport)
    database = tmp_path / "malformed.sqlite"
    store = SQLiteSessionStore(database)
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-5.6"), hosted_tools=[OpenAIWebSearch()]
    )
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="malformed",
                messages=[Message.text("user", "go")],
                retry_policy=RetryPolicy(max_attempts=5, max_unknown_attempts=2, initial_delay_s=0),
            )
        )
    ]
    assert 1 <= len(transport.calls) <= 2
    assert events[-1].type == EventType.SESSION_FAILED
    await store.close()
    reopened = SQLiteSessionStore(database)
    persisted = await reopened.load_events("malformed")
    errors = [e for e in persisted if e.type == EventType.MODEL_ERROR]
    assert errors
    for error in errors:
        assert error.payload["provider_protocol_reason"] == REASON
        assert error.payload["provider_protocol_field"] == "output[].action.sources[].name"
        assert error.payload["effective_max_attempts"] <= 2
        assert "redacted" not in json.dumps(error.payload)
    await reopened.close()
    await provider.aclose()


@pytest.mark.anyio
async def test_api_name_conflict_between_item_and_terminal_is_rejected():
    raw = stream_fixture(response_fixture({"type": "search", "sources": [dict(API_SOURCE)]}))
    raw[-1]["response"]["output"][0]["action"]["sources"][0]["name"] = "redacted-other"
    provider = OpenAIProvider(api_key="offline", transport=RecordingTransport(stream_events=[raw]))
    try:
        events = [
            e
            async for e in provider.stream(
                ModelRequest(
                    model="gpt-5.6",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
    finally:
        await provider.aclose()
    assert events[-1].type == ModelStreamEventType.ERROR
    assert not any(e.type == ModelStreamEventType.COMPLETED for e in events)
