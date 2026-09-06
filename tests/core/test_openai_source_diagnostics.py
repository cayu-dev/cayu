"""Synthetic discriminator boundary fixtures, not reproductions of GAIA #114."""

import json

import pytest
from tests.core.test_openai_optional_search_evidence import response_fixture, stream_fixture
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
)
from cayu.providers import ModelStreamEventType, OpenAIProtocolError, openai_response_events
from cayu.providers._openai_protocol import SearchSourceDiagnostic, source_diagnostic_fields

REASON = "web_search_action_sources_type_is_unsupported"
VALUES = [
    ("api", "string", "api"),
    ("future_extension", "string", None),
    (None, "null", None),
    (True, "boolean", None),
    (42, "number", None),
    (1.5, "number", None),
    ({"secret": "source-body"}, "object", None),
    (["source-body"], "array", None),
    ("x" * 10000, "string", None),
    ("api\nAuthorization: secret", "string", None),
    ("https://user:secret@example.com", "string", None),
    ("sk-secret", "string", None),
    ("privatecredential", "string", None),
]


def bad_response(value):
    return response_fixture(
        {
            "type": "search",
            "sources": [
                {"type": "url", "url": "https://example.com"},
                {"type": value, "url": "https://private.example.com", "title": "source-body"},
            ],
        }
    )


def expected_fields(kind, label):
    return {
        "provider_protocol_source_index": 1,
        "provider_protocol_source_type_kind": kind,
        "provider_protocol_source_supported_types": "url",
        "provider_protocol_source_type_value_status": "retained" if label else "omitted",
        **({"provider_protocol_source_type_value": label} if label else {}),
    }


@pytest.mark.parametrize(("value", "kind", "label"), VALUES)
def test_completed_response_source_diagnostics(value, kind, label):
    with pytest.raises(OpenAIProtocolError) as caught:
        openai_response_events(bad_response(value))
    error = caught.value
    assert error.reason_code == REASON
    assert source_diagnostic_fields(error.source_diagnostic) == expected_fields(kind, label)
    assert "source-body" not in repr(vars(error)) + str(error)
    assert "private.example" not in repr(vars(error)) + str(error)
    if label is None and isinstance(value, str):
        assert value not in repr(vars(error)) + str(error)


@pytest.mark.anyio
@pytest.mark.parametrize("terminal_only", [False, True])
@pytest.mark.parametrize(("value", "kind", "label"), VALUES)
async def test_stream_source_diagnostics(value, kind, label, terminal_only, caplog):
    transport = RecordingTransport(
        stream_events=[stream_fixture(bad_response(value), terminal_only)]
    )
    provider = OpenAIProvider(
        api_key="privatecredential", transport=transport, base_url="https://codex-lb.cayu.ai"
    )
    events = [
        event
        async for event in provider.stream(
            ModelRequest(model="gpt-5.6", messages=[Message.text("user", "go")])
        )
    ]
    error = next(event for event in events if event.type == ModelStreamEventType.ERROR)
    assert error.payload["provider_protocol_reason"] == REASON
    assert {
        k: v for k, v in error.payload.items() if k.startswith("provider_protocol_source_")
    } == expected_fields(kind, label)
    rendered = json.dumps(error.payload) + caplog.text
    for secret in ("source-body", "private.example", "privatecredential", "sk-secret"):
        assert secret not in rendered
    assert not any(event.type == ModelStreamEventType.COMPLETED for event in events)
    assert transport.calls[0]["url"] == "https://codex-lb.cayu.ai/v1/responses"
    await provider.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("succeed", [False, True])
async def test_durable_source_diagnostics_survive_bounded_retry(tmp_path, succeed):
    raw = stream_fixture(bad_response("api"))
    success = {
        "id": "resp_ok",
        "model": "gpt-5.6",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_ok",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    transport = RecordingTransport(
        stream_events=[
            raw,
            [{"type": "response.completed", "response": success}] if succeed else raw,
        ]
    )
    database = tmp_path / "sources.sqlite3"
    store = SQLiteSessionStore(database)
    app = CayuApp(session_store=store, enable_logging=False)
    provider = OpenAIProvider(api_key="offline", transport=transport)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-5.6"), hosted_tools=[OpenAIWebSearch()]
    )
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="source",
                messages=[Message.text("user", "go")],
                retry_policy=RetryPolicy(max_attempts=5, max_unknown_attempts=2, initial_delay_s=0),
            )
        )
    ]
    assert len(transport.calls) == 2
    assert len([e for e in events if e.type == EventType.MODEL_RETRY]) == 1
    assert events[-1].type == (EventType.SESSION_COMPLETED if succeed else EventType.SESSION_FAILED)
    await store.close()
    reopened = SQLiteSessionStore(database)
    persisted = await reopened.load_events("source")
    errors = [e for e in persisted if e.type == EventType.MODEL_ERROR]
    assert len(errors) == (1 if succeed else 2)
    for error in errors:
        assert error.payload["provider_protocol_reason"] == REASON
        assert error.payload["effective_max_attempts"] == 2
        assert {
            k: v for k, v in error.payload.items() if k.startswith("provider_protocol_source_")
        } == expected_fields("string", "api")
    hosted = [e for e in persisted if e.type == EventType.MODEL_HOSTED_TOOL_CALL]
    assert any(e.payload["status"] == "outcome_unknown" for e in hosted)
    await reopened.close()
    await provider.aclose()


@pytest.mark.parametrize(
    "source", [{"type": "url", "url": "https://example.com"}, {"url": "https://example.com"}]
)
def test_url_and_omitted_discriminator_unchanged(source):
    events = openai_response_events(response_fixture({"type": "search", "sources": [source]}))
    hosted = next(e for e in events if e.type == ModelStreamEventType.HOSTED_TOOL_CALL)
    assert hosted.payload["action"]["sources"] == [{"type": "url", "url": "https://example.com"}]


@pytest.mark.parametrize(
    ("sources", "reason"),
    [
        ({"secret": "payload"}, "web_search_action_sources_must_be_a_bounded_list"),
        ([{}] * 101, "web_search_action_sources_must_be_a_bounded_list"),
        ([None], "web_search_action_sources_must_be_an_object"),
    ],
)
def test_malformed_source_lists(sources, reason):
    with pytest.raises(OpenAIProtocolError) as caught:
        openai_response_events(response_fixture({"type": "search", "sources": sources}))
    assert caught.value.reason_code == reason
    assert caught.value.source_diagnostic is None


@pytest.mark.anyio
async def test_allowlisted_label_matching_header_credential_is_omitted():
    provider = OpenAIProvider(
        api_key="offline",
        extra_headers={"X-Private": "api"},
        transport=RecordingTransport(stream_events=[stream_fixture(bad_response("api"))]),
    )
    events = [
        e
        async for e in provider.stream(
            ModelRequest(model="gpt-5.6", messages=[Message.text("user", "go")])
        )
    ]
    error = next(e for e in events if e.type == ModelStreamEventType.ERROR)
    assert error.payload["provider_protocol_source_type_value_status"] == "omitted"
    assert "provider_protocol_source_type_value" not in error.payload
    await provider.aclose()


@pytest.mark.parametrize(
    "diagnostic",
    [
        None,
        {"index": 0},
        SearchSourceDiagnostic(True, "string", "api"),
        SearchSourceDiagnostic(100, "string", "api"),
        SearchSourceDiagnostic(0, "secret", "api"),
    ],
)
def test_reject_untrusted_diagnostic_attributes(diagnostic):
    assert source_diagnostic_fields(diagnostic) == {}


@pytest.mark.anyio
@pytest.mark.parametrize("reconnect", [False, True])
@pytest.mark.parametrize("credential", ["offline", "api"])
async def test_background_recovery_retains_diagnostics_without_redispatch(
    tmp_path, reconnect, credential
):
    from tests.core.test_openai_background_operations import (
        BackgroundTransport,
        SimulatedWorkerLoss,
        _created,
    )

    from cayu.runtime import IncompleteSessionRecoveryRequest
    from cayu.runtime.provider_operations import (
        ProviderOperationUnavailableReason,
        inspect_provider_operation,
    )

    transport = BackgroundTransport()
    start = [_created()]
    if reconnect:
        start.append(
            {"type": "response.output_text.delta", "sequence_number": 1, "delta": "accepted"}
        )
    transport.start_batches.append([*start, SimulatedWorkerLoss("worker lost")])
    response = bad_response("api")
    response["id"] = "resp_background_123"
    if reconnect:
        transport.reconnect_batches.append(
            [{"type": "response.completed", "sequence_number": 2, "response": response}]
        )
    else:
        transport.retrieve_responses.append(response)
    database = tmp_path / "recovery.sqlite3"
    store = SQLiteSessionStore(database)
    app = CayuApp(session_store=store, enable_logging=False)
    provider = OpenAIProvider(api_key=credential, background=True, transport=transport)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="gpt-5.6"), hosted_tools=[OpenAIWebSearch()]
    )
    with pytest.raises(SimulatedWorkerLoss):
        _ = [
            e
            async for e in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="recovery",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(session_id="recovery", inactive_for_seconds=0)
    )
    inspection = await inspect_provider_operation(store, "recovery")
    assert inspection.recovery_reason is ProviderOperationUnavailableReason.MALFORMED
    assert len(transport.start_calls) == 1
    assert len(transport.retrieve_calls) == int(not reconnect)
    assert len(transport.reconnect_calls) == int(reconnect)
    await store.close()
    reopened = SQLiteSessionStore(database)
    events = await reopened.load_events("recovery")
    required = next(
        e for e in reversed(events) if e.type == EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED
    )
    assert required.payload["provider_protocol_reason"] == REASON
    assert {
        k: v for k, v in required.payload.items() if k.startswith("provider_protocol_source_")
    } == expected_fields("string", None if credential == "api" else "api")
    assert "source-body" not in json.dumps(required.payload)
    assert "private.example" not in json.dumps(required.payload)
    assert not any(e.type == EventType.MODEL_RETRY for e in events)
    await reopened.close()
    await provider.aclose()
