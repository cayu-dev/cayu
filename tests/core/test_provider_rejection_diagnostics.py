from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.cli import main
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers._http import credential_safe_error_event, credential_safe_provider_exception
from cayu.providers._rejection_diagnostics import project_rejection_response
from cayu.providers.anthropic import _anthropic_api_error_from_response
from cayu.providers.chat_completions import _chat_api_error_from_response
from cayu.providers.openai import (
    HttpxOpenAITransport,
    OpenAIProvider,
    _openai_api_error_from_response,
)
from cayu.providers.vertex import _vertex_api_error_from_response
from cayu.runtime._model_errors import (
    copy_model_provider_error_control,
    model_provider_error_from_payload,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.sessions.base import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.workflows.base import WorkflowSpec
from cayu.workflows.models import StepError
from cayu.workflows.workflow import WorkflowBase, step

FACTORIES = [
    (_openai_api_error_from_response, "openai"),
    (_anthropic_api_error_from_response, "anthropic"),
    (_chat_api_error_from_response, "chat_completions"),
    (_vertex_api_error_from_response, "vertex"),
]


@pytest.mark.parametrize("factory,provider", FACTORIES)
@pytest.mark.parametrize("credentials", [(), ("private-credential",)])
@pytest.mark.parametrize(
    "error",
    [
        {"message": "Unsupported parameter: 'temperature'."},
        {
            "code": "unsupported_parameter",
            "param": "temperature",
            "message": "private-credential customer@example.com",
        },
    ],
)
def test_safe_explanation_across_adapters_and_detachment(factory, provider, credentials, error):
    exc = factory(
        httpx.Response(400, json={"error": error, "request_id": "private-credential"}),
        "private-credential",
        None,
    )
    event = credential_safe_error_event(
        exc, provider_label=provider, provider_name=provider, credential_values=credentials
    )
    assert event.payload["status_code"] == 400
    assert event.payload.get("retryable") is not True
    assert event.payload["provider_rejection_parameter"] == "temperature"
    assert "Remove this parameter" in event.payload["provider_rejection_explanation"]
    assert "private-credential" not in json.dumps(event.payload)
    safe = credential_safe_provider_exception(
        exc, provider_label=provider, provider_name=provider, credential_values=credentials
    )
    copied = copy_model_provider_error_control(safe)
    rebuilt = model_provider_error_from_payload(event.payload, fallback_provider=provider)
    for failure in (safe, copied, rebuilt):
        assert failure.status_code == 400
        assert failure.error_payload_fields()["provider_rejection_parameter"] == "temperature"


@pytest.mark.parametrize(
    "body,reason",
    [
        (
            {"error": {"message": "secret", "param": "secret", "code": "secret"}},
            "unrecognized_details",
        ),
        (
            {"error": {"code": "unsupported_parameter", "param": "customer@example.com"}},
            "unsafe_parameter",
        ),
        (
            {"error": {"message": "Unsupported parameter: 'temperature'. secret"}},
            "unrecognized_details",
        ),
        ({"error": {"details": {"message": "secret"}}}, "absent_details"),
        ({"error": {}}, "absent_details"),
        ({"error": []}, "malformed_body"),
        ({"error": {"message": "secret" * 20000}}, "body_too_large"),
    ],
)
def test_unrecognized_and_sensitive_fields(body, reason):
    result = project_rejection_response(
        httpx.Response(400, json=body, headers={"x-request-id": "secret"})
    )
    assert result["provider_rejection_unavailable_reason"] == reason
    assert "secret" not in json.dumps(result)
    assert "customer@" not in json.dumps(result)


@pytest.mark.parametrize(
    "content,content_type,reason",
    [
        (b"", "application/json", "absent_body"),
        (b"secret", "text/html", "non_json_body"),
        (b'{"message":"secret', "application/json", "malformed_body"),
        (b"[" * 2000, "application/json", "malformed_body"),
        (b"\xff", "application/json", "malformed_body"),
    ],
)
def test_malformed_response(content, content_type, reason):
    result = project_rejection_response(
        httpx.Response(400, content=content, headers={"content-type": content_type})
    )
    assert result["provider_rejection_unavailable_reason"] == reason


def test_known_credential_overlaps_parameter():
    exc = _openai_api_error_from_response(
        httpx.Response(
            400, json={"error": {"code": "unsupported_parameter", "param": "temperature"}}
        ),
        "safe",
        None,
    )
    event = credential_safe_error_event(
        exc, provider_label="OpenAI", provider_name="openai", credential_values=("temperature",)
    )
    assert "temperature" not in json.dumps(event.payload)
    assert event.payload["provider_rejection_unavailable_reason"] == "credential_overlap"


@pytest.mark.parametrize("workflow", [False, True])
def test_http_400_is_durable_and_never_redispatched(tmp_path, capsys, workflow):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            400,
            json={
                "error": {
                    "type": "invalid_request_error",
                    "message": "Unsupported parameter: 'temperature'.",
                }
            },
            headers={"x-request-id": "secret-request"},
        )

    async def run():
        store = SQLiteSessionStore(tmp_path / "session.sqlite")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            transport = HttpxOpenAITransport()
            transport._client._client = client
            provider = OpenAIProvider(api_key="synthetic", transport=transport)
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="probe", model="synthetic"))
            session_id = "rejection"
            if workflow:

                class Probe(WorkflowBase):
                    spec = WorkflowSpec(name="probe")

                    async def run(self, session_id):
                        yield await self.context(session_id).start()

                ctx = Probe(app).context("parent")
                await ctx.start()
                with pytest.raises(StepError) as raised:
                    await step(ctx, agent="probe", step_id="reject", prompt="hello")
                failure = raised.value
                session_id = failure.session_id
                assert failure.evidence.session_id == session_id
                assert failure.evidence.terminal_event_id
                events = await store.load_events(session_id)
            else:
                events = [
                    event
                    async for event in app.run(
                        RunRequest(
                            agent_name="probe",
                            session_id="rejection",
                            messages=[Message.text("user", "hello")],
                            retry_policy=RetryPolicy(max_attempts=3),
                        )
                    )
                ]
            stored = await store.load_events(session_id)
            for collection in (events, stored):
                errors = [event for event in collection if event.type is EventType.MODEL_ERROR]
                assert len(errors) == 1
                payload = errors[0].payload
                assert payload["provider_rejection_parameter"] == "temperature"
                assert payload["status_code"] == 400
                assert payload["retry_disposition"] == "explicit_nonretryable"
                assert payload["retryable"] is False
                assert payload["model_attempt_id"]
                assert collection[-1].type is EventType.SESSION_FAILED
                assert not any(event.type is EventType.MODEL_RETRY for event in collection)
                assert "secret-request" not in json.dumps([event.payload for event in collection])
        assert len(calls) == 1
        return session_id

    session_id = asyncio.run(run())
    assert (
        main(
            [
                "session",
                "events",
                session_id,
                "--sqlite",
                str(tmp_path / "session.sqlite"),
                "--include-payload",
                "10000",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "temperature" in output
    assert "Remove this parameter" in output
    assert "secret-request" not in output


@pytest.mark.parametrize("provider", ["openai", "chat_completions", "anthropic", "vertex"])
def test_streaming_rejection_uses_same_safe_projection(provider):
    from cayu.providers.anthropic import anthropic_stream_events
    from cayu.providers.chat_completions import _stream_error_chunk_exception
    from cayu.providers.openai import _openai_stream_error_exception
    from cayu.providers.vertex import VertexAPIError, VertexContextOverflowError

    error = {"type": "invalid_request_error", "message": "max_tokens: must be a positive integer"}

    async def run():
        if provider == "openai":
            return _openai_stream_error_exception({"type": "error", "error": error})
        if provider == "chat_completions":
            return _stream_error_chunk_exception(error)

        async def source():
            yield {"type": "error", "error": error}

        kwargs = (
            {}
            if provider == "anthropic"
            else {
                "provider_label": "Vertex",
                "api_error": VertexAPIError,
                "context_overflow_error": VertexContextOverflowError,
            }
        )
        with pytest.raises(Exception) as raised:
            _ = [event async for event in anthropic_stream_events(source(), **kwargs)]
        return raised.value

    exc = asyncio.run(run())
    event = credential_safe_error_event(
        exc, provider_label=provider, provider_name=provider, credential_values=()
    )
    assert event.payload["provider_rejection_parameter"] == "max_tokens"
    assert event.payload["provider_rejection_reason"] == "positive_integer_required"


@pytest.mark.parametrize("factory,provider", FACTORIES)
@pytest.mark.parametrize("content", [b"[" * 2000, b"x" * 70000, b'{"error":'])
def test_bad_body_does_not_replace_http_400(factory, provider, content):
    failure = factory(
        httpx.Response(400, content=content, headers={"content-type": "application/json"}),
        "safe",
        None,
    )
    event = credential_safe_error_event(
        failure, provider_label=provider, provider_name=provider, credential_values=()
    )
    assert event.payload["status_code"] == 400
    assert event.payload.get("retryable") is not True
    assert "provider_rejection_unavailable_reason" in event.payload


@pytest.mark.parametrize("failure_kind", ["flat_sse", "oversized_quota"])
def test_bundled_transport_preserves_rejection_and_retry_decisions(tmp_path, failure_kind):
    calls = []

    class ResponseStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "error",
                        "status_code": 400,
                        "code": "unsupported_parameter",
                        "param": "temperature",
                        "message": "Unsupported parameter: 'temperature'.",
                    }
                )
                + "\n\n"
            ).encode()

    def handle(request):
        calls.append(request)
        if failure_kind == "flat_sse":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=ResponseStream()
            )
        return httpx.Response(
            429,
            json={
                "error": {
                    "type": "insufficient_quota",
                    "code": "insufficient_quota",
                    "message": "private detail " * 6000,
                }
            },
        )

    async def run():
        store = SQLiteSessionStore(tmp_path / "transport.sqlite")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            transport = HttpxOpenAITransport()
            transport._client._client = client
            provider = OpenAIProvider(
                api_key="synthetic-key",
                transport=transport,
                streaming=failure_kind == "flat_sse",
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="probe", model="gpt-test"))
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="probe",
                        session_id="transport-rejection",
                        messages=[Message.text("user", "hello")],
                        retry_policy=RetryPolicy(
                            max_attempts=3, initial_delay_s=0.0, max_delay_s=0.0, jitter_s=0.0
                        ),
                    )
                )
            ]
            stored = await store.load_events("transport-rejection")
        for collection in (events, stored):
            errors = [event for event in collection if event.type is EventType.MODEL_ERROR]
            assert len(errors) == 1
            payload = errors[0].payload
            if failure_kind == "flat_sse":
                assert payload["provider_rejection_parameter"] == "temperature"
                assert payload["provider_rejection_reason"] == "unsupported_parameter"
                assert "Remove this parameter" in payload["provider_rejection_explanation"]
            else:
                assert payload["status_code"] == 429
                assert payload["retryable"] is False
                assert payload["retry_disposition"] == "explicit_nonretryable"
                assert payload["provider_rejection_unavailable_reason"] == "body_too_large"
                assert "private detail" not in json.dumps(payload)
            assert not any(event.type is EventType.MODEL_RETRY for event in collection)
            assert collection[-1].type is EventType.SESSION_FAILED
        assert len(calls) == 1

    asyncio.run(run())


def test_oversized_vertex_error_preserves_context_recovery_classification():
    from cayu.providers.vertex import _raise_vertex_context_overflow_if_applicable

    response = httpx.Response(
        413,
        json={"error": {"status": "PERMISSION_DENIED", "message": "private detail " * 6000}},
    )
    # An explicit non-overflow identity must prevent status-only context recovery.
    _raise_vertex_context_overflow_if_applicable(response)
    failure = _vertex_api_error_from_response(response, "safe", None)
    assert failure.error_type == "PERMISSION_DENIED"
    assert failure.rejection_diagnostic["provider_rejection_unavailable_reason"] == "body_too_large"
