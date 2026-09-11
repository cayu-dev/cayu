from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from tests.core.test_openai_subscription_provider import StaticSubscriptionAuth

from cayu import AgentSpec, CayuApp, EventType, Message, RetryPolicy, RunRequest, SQLiteSessionStore
from cayu.providers import OpenAIProvider
from cayu.providers._http import credential_safe_error_event
from cayu.providers.openai import _openai_api_error_from_response, _openai_stream_error_exception
from cayu.providers.openai_subscription import (
    OpenAISubscriptionProvider,
    _safe_subscription_error_event,
)


@pytest.mark.parametrize("adapter", ["api", "subscription"])
@pytest.mark.parametrize(
    "identity,status,outer,reason,retryable",
    [
        ({}, None, None, "absent_identity", None),
        ({"code": "secret-unknown"}, None, None, "unsupported_identity", None),
        ({"code": "secret-unknown"}, 500, 400, "explicit_status_conflict", False),
        ({"code": "rate_limit_exceeded"}, None, None, "recognized_identity", True),
        ({"code": "insufficient_quota"}, None, None, "recognized_identity", False),
        (
            {"code": "server_error", "type": "authentication_error"},
            None,
            None,
            "identity_conflict",
            False,
        ),
        ({"code": "rate_limit_exceeded"}, 401, None, "status_identity_conflict", False),
        ({}, 503, None, "explicit_status", None),
    ],
)
def test_classification_survives_projection_and_sqlite(
    tmp_path, adapter, identity, status, outer, reason, retryable
):
    raw = {"type": "response.failed", "response": {"error": identity}}
    if status is not None:
        raw["response"]["status_code"] = status
    if outer is not None:
        raw["status_code"] = outer
    exc = _openai_stream_error_exception(raw)
    assert exc.retryable is retryable
    api = credential_safe_error_event(
        exc, provider_label="OpenAI", provider_name="openai", credential_values=["secret-unknown"]
    )
    subscription = _safe_subscription_error_event(exc, None, provider_name="openai_subscription")
    for event in (api, subscription):
        assert event.payload["provider_api_classification_reason"] == reason
        assert event.payload["provider_api_classification_origin"] == "stream"
        assert "secret-unknown" not in json.dumps(event.payload)

    class Transport:
        async def stream_response_events(self, **kwargs):
            yield raw

    database = tmp_path / "sessions.sqlite"
    app = CayuApp(session_store=SQLiteSessionStore(database), enable_logging=False)
    provider = (
        OpenAIProvider(api_key="secret-unknown", transport=Transport())
        if adapter == "api"
        else OpenAISubscriptionProvider(auth=StaticSubscriptionAuth(), transport=Transport())
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="diagnostic",
                    messages=[Message.text("user", "hello")],
                    retry_policy=RetryPolicy(max_attempts=1),
                )
            )
        ]

    events = asyncio.run(run())
    stored = asyncio.run(SQLiteSessionStore(database).load_events("diagnostic"))
    for collection in (events, stored):
        errors = [event.payload for event in collection if event.type is EventType.MODEL_ERROR]
        assert errors
        assert errors[-1]["provider_api_classification_reason"] == reason
        assert errors[-1]["provider_api_classification_origin"] == "stream"


@pytest.mark.parametrize(
    "status,identity,reason",
    [
        (503, {}, "explicit_status"),
        (503, {"code": "server_is_overloaded"}, "recognized_identity"),
        (401, {"code": "server_is_overloaded"}, "status_identity_conflict"),
        (
            404,
            {"type": "invalid_request_error", "code": "previous_response_not_found"},
            "recognized_identity",
        ),
    ],
)
def test_http_classification(status, identity, reason):
    exc = _openai_api_error_from_response(
        httpx.Response(status, json={"error": identity}), "safe", None
    )
    for event in (
        credential_safe_error_event(
            exc, provider_label="OpenAI", provider_name="openai", credential_values=["synthetic"]
        ),
        _safe_subscription_error_event(exc, None, provider_name="openai_subscription"),
    ):
        assert event.payload["provider_api_classification_origin"] == "http"
        assert event.payload["provider_api_classification_reason"] == reason


@pytest.mark.parametrize("field,secret", [("reason", "recognized_identity"), ("origin", "stream")])
def test_classification_credential_overlap_is_omitted(field, secret):
    exc = _openai_stream_error_exception(
        {"type": "response.failed", "response": {"error": {"code": "server_error"}}}
    )
    events = [
        credential_safe_error_event(
            exc, provider_label="OpenAI", provider_name="openai", credential_values=[secret]
        ),
        _safe_subscription_error_event(
            exc, None, provider_name="openai_subscription", extra_header_values=(secret,)
        ),
    ]
    for event in events:
        assert "provider_api_classification_" + field not in event.payload
        assert secret not in json.dumps(event.payload)
