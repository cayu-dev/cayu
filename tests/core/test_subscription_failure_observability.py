"""Safe transport identities must survive provider projection and durable events."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RetryPolicy, RunRequest, SQLiteSessionStore
from cayu.providers import HttpxOpenAITransport, OpenAIAPIError, OpenAISubscriptionProvider
from cayu.providers._http import _SAFE_INTERNAL_PROVIDER_ERROR_TYPES
from cayu.providers.openai_subscription import (
    OpenAISubscriptionCredentials,
    _safe_subscription_error_event,
)

CANARY = "secret-credential-response-url-canary"


class StaticAuth:
    async def credentials(self):
        return OpenAISubscriptionCredentials(
            access_token=CANARY,
            refresh_token="refresh-secret-canary",
            expires_at=2_000_000_000,
        )


@pytest.mark.parametrize("error_type", sorted(_SAFE_INTERNAL_PROVIDER_ERROR_TYPES))
def test_subscription_preserves_safe_internal_failure_identity(error_type):
    error = OpenAIAPIError(
        CANARY,
        error_type=error_type,
        error_code=CANARY,
        request_id=CANARY,
        response_body=CANARY,
        retryable=False,
    )
    event = _safe_subscription_error_event(error, None, provider_name="openai_subscription")
    assert event.payload.get("provider_error_type") == error_type
    assert event.payload["retryable"] is False
    assert CANARY not in json.dumps(event.payload)
    assert "provider_error_code" not in event.payload
    assert "request_id" not in event.payload


@pytest.mark.parametrize("error_type", [CANARY, "ReadError:" + CANARY])
def test_subscription_drops_unrecognized_failure_identity(error_type):
    event = _safe_subscription_error_event(
        OpenAIAPIError(CANARY, error_type=error_type),
        None,
        provider_name="openai_subscription",
    )
    assert "provider_error_type" not in event.payload
    assert CANARY not in json.dumps(event.payload)


def test_internal_type_does_not_become_an_allowed_api_error_code():
    event = _safe_subscription_error_event(
        OpenAIAPIError(CANARY, error_type="ReadError", error_code="ReadError"),
        None,
        provider_name="openai_subscription",
    )
    assert event.payload["provider_error_type"] == "ReadError"
    assert "provider_error_code" not in event.payload


@pytest.mark.parametrize(
    "error_type,retryable",
    [("ConnectTimeout", True), ("ReadError", False), ("SseEventLimitError", False)],
)
def test_subscription_failure_identity_survives_sqlite_reopen(tmp_path, error_type, retryable):
    class FailingTransport:
        async def stream_response_events(self, **kwargs):
            raise OpenAIAPIError(
                CANARY,
                error_type=error_type,
                retryable=retryable,
                response_body=CANARY,
            )
            yield  # Make this a streaming transport, without contacting a provider.

    _assert_durable_failure(tmp_path, FailingTransport(), error_type, retryable)


@pytest.mark.parametrize(
    "exception_type,after_headers,retryable",
    [
        (httpx.ConnectError, False, True),
        (httpx.ConnectTimeout, False, True),
        (httpx.ReadError, True, False),
        (httpx.ReadTimeout, True, False),
        (httpx.RemoteProtocolError, True, False),
    ],
)
def test_http_failure_survives_subscription_and_storage(
    tmp_path, monkeypatch, exception_type, after_headers, retryable
):
    class InterruptedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"type":"response.created","response":{"id":"test-response"}}\n\n'
            raise exception_type(CANARY)

    def handler(request):
        if not after_headers:
            raise exception_type(CANARY, request=request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=InterruptedStream()
        )

    monkeypatch.setattr(
        "cayu.providers._http.new_async_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    _assert_durable_failure(tmp_path, HttpxOpenAITransport(), exception_type.__name__, retryable)


def _assert_durable_failure(tmp_path, transport, error_type, retryable):
    database = tmp_path / "sessions.sqlite"
    app = CayuApp(session_store=SQLiteSessionStore(database), enable_logging=False)
    app.register_provider(
        OpenAISubscriptionProvider(auth=StaticAuth(), transport=transport),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="test-model"))

    async def run():
        try:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="failure-audit",
                        messages=[Message.text("user", "hello")],
                        retry_policy=RetryPolicy(max_attempts=1),
                    )
                )
            ]
        finally:
            if isinstance(transport, HttpxOpenAITransport):
                await transport.aclose()

    events = asyncio.run(run())
    persisted = asyncio.run(SQLiteSessionStore(database).load_events("failure-audit"))
    for source in (events, persisted):
        errors = [event.payload for event in source if event.type is EventType.MODEL_ERROR]
        assert len(errors) == 1
        payload = errors[0]
        assert payload.get("provider_error_type") == error_type
        assert payload["provider_retryable"] is retryable
        assert payload["model_step_id"]
        assert payload["model_attempt_id"]
        assert payload["retry"] is False
        assert payload["retry_disposition"] == (
            "configured_attempt_exhaustion" if retryable else "explicit_nonretryable"
        )
        assert CANARY not in json.dumps([event.payload for event in source])
    live_error = next(event.payload for event in events if event.type is EventType.MODEL_ERROR)
    stored_error = next(event.payload for event in persisted if event.type is EventType.MODEL_ERROR)
    # Public events intentionally redact private execution authority. The
    # authorized store must retain the actual IDs for attempt correlation.
    assert stored_error["model_step_id"].startswith("mstep_")
    assert stored_error["model_attempt_id"].startswith("matt_")
    for key in ("provider_error_type", "retry_disposition"):
        assert live_error[key] == stored_error[key]
