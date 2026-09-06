from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    RetryPolicy,
    RunRequest,
    SQLiteSessionStore,
)
from cayu.providers import OpenAIProvider
from cayu.providers._http import credential_safe_error_event
from cayu.providers.base import ModelStreamEvent, ModelStreamEventType
from cayu.providers.openai import (
    _openai_api_error_from_response,
    _openai_stream_error_exception,
)
from cayu.runtime._model_errors import runtime_owned_model_stream_error_event

SECRET = "synthetic-credential-secret"
OVERLOAD = "server_is_overloaded"


def failure(code=OVERLOAD, error_type=None, status=None):
    error = {"code": code, "message": SECRET * 1000, "request_id": SECRET}
    if error_type is not None:
        error["type"] = error_type
    response = {"id": "fixture", "error": error}
    if status is not None:
        response["status_code"] = status
    return {"type": "response.failed", "response": response}


@pytest.mark.parametrize(
    "code,error_type,status,expected_status,retryable,safe_code",
    [
        (OVERLOAD, None, None, None, True, OVERLOAD),
        (OVERLOAD, "server_error", None, None, True, OVERLOAD),
        (OVERLOAD, None, 503, 503, True, OVERLOAD),
        (OVERLOAD, None, 429, 429, False, OVERLOAD),
        (OVERLOAD, "authentication_error", None, None, False, OVERLOAD),
        (OVERLOAD, "insufficient_quota", None, None, False, OVERLOAD),
        (OVERLOAD, "invalid_request_error", None, None, False, OVERLOAD),
        pytest.param(SECRET * 1000, SECRET, None, None, None, None, id="hostile-unknown"),
    ],
)
def test_safe_stream_identity(code, error_type, status, expected_status, retryable, safe_code):
    exc = _openai_stream_error_exception(failure(code, error_type, status))
    event = credential_safe_error_event(
        exc, provider_label="OpenAI", provider_name="openai", credential_values=[SECRET]
    )
    serialized = json.dumps(event.payload)
    assert SECRET not in serialized
    assert len(serialized) < 600
    # Reconstruct only serialized data, with no retained exception object.
    _, restored = runtime_owned_model_stream_error_event(
        ModelStreamEvent(type=ModelStreamEventType.ERROR, payload=json.loads(serialized)),
        fallback_provider="openai",
    )
    assert restored is not None
    assert restored.provider == "openai"
    assert restored.status_code == expected_status
    assert restored.retryable is retryable
    assert restored.error_code == safe_code
    assert restored.response_body is None


@pytest.mark.parametrize(
    "code,error_type,recover,limit,calls,disposition",
    [
        (OVERLOAD, None, True, 3, 2, "retry_scheduled"),
        (OVERLOAD, None, False, 3, 3, "configured_attempt_exhaustion"),
        (OVERLOAD, None, False, 1, 1, "configured_attempt_exhaustion"),
        pytest.param(
            SECRET * 1000, None, False, 5, 2, "unknown_provider_attempt_cap", id="unknown-cap"
        ),
        (OVERLOAD, "authentication_error", False, 3, 1, "explicit_nonretryable"),
        (OVERLOAD, "insufficient_quota", False, 3, 1, "explicit_nonretryable"),
        (OVERLOAD, "invalid_request_error", False, 3, 1, "explicit_nonretryable"),
    ],
)
def test_public_stream_runtime_retry(
    tmp_path, code, error_type, recover, limit, calls, disposition
):
    class Transport:
        calls = 0

        async def stream_response_events(self, **kwargs):
            self.calls += 1
            if recover and self.calls > 1:
                yield {"type": "response.output_text.delta", "delta": "ok"}
                yield {
                    "type": "response.completed",
                    "response": {
                        "id": "response-success",
                        "model": "fake-model",
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 2, "output_tokens": 1},
                    },
                }
            else:
                yield failure(code, error_type)

    transport = Transport()
    database = tmp_path / "sessions.sqlite"
    app = CayuApp(session_store=SQLiteSessionStore(database), enable_logging=False)
    app.register_provider(OpenAIProvider(api_key=SECRET, transport=transport), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run():
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="overload",
                    messages=[Message.text("user", "hello")],
                    retry_policy=RetryPolicy(max_attempts=limit, initial_delay_s=0.0),
                )
            )
        ]

    events = asyncio.run(run())
    assert transport.calls == calls
    assert events[-1].type is (EventType.SESSION_COMPLETED if recover else EventType.SESSION_FAILED)
    errors = [e for e in events if e.type is EventType.MODEL_ERROR]
    assert errors[-1].payload["retry_disposition"] == disposition
    assert len([e for e in events if e.type is EventType.MODEL_RETRY]) == calls - 1
    stored = asyncio.run(SQLiteSessionStore(database).load_events("overload"))
    retained = [e.payload for e in stored if e.type is EventType.MODEL_ERROR]
    for stored_error, public in zip(retained, errors, strict=True):
        assert {
            k: v for k, v in stored_error.items() if k not in {"model_step_id", "model_attempt_id"}
        } == {
            k: v
            for k, v in public.payload.items()
            if k not in {"model_step_id", "model_attempt_id"}
        }
    assert SECRET not in json.dumps(retained)
    if code == OVERLOAD:
        assert all(e["provider_error_code"] == OVERLOAD for e in retained)
        assert all("status_code" not in e for e in retained)
    else:
        assert all("provider_error_code" not in e for e in retained)


@pytest.mark.parametrize("credentials", [[], [SECRET]])
def test_unknown_origin_survives_missing_credentials(credentials):
    exc = _openai_stream_error_exception(failure(SECRET, SECRET))
    event = credential_safe_error_event(
        exc, provider_label="OpenAI", provider_name="openai", credential_values=credentials
    )
    _, restored = runtime_owned_model_stream_error_event(event, fallback_provider="openai")
    assert restored is not None
    assert restored.retryable is None
    assert restored.error_code is None
    assert restored.error_type is None
    assert SECRET not in json.dumps(event.payload)


@pytest.mark.parametrize("marker", [None, False, "true", 1])
def test_untyped_error_does_not_gain_provider_retry_authority(marker):
    event = ModelStreamEvent(
        type=ModelStreamEventType.ERROR,
        payload={"error": "failure", "provider": "openai", "model_provider_error": marker},
    )
    _, restored = runtime_owned_model_stream_error_event(event, fallback_provider="openai")
    assert restored is None


def test_overload_conflicting_status_fields_fail_closed():
    raw = failure(status=503)
    raw["status_code"] = 401
    exc = _openai_stream_error_exception(raw)
    assert exc.status_code is None
    assert exc.retryable is False


def test_overload_preserves_only_trusted_retry_after():
    from cayu.providers._http import _TrustedSseJsonEvent

    raw = failure()
    raw["retry_after_s"] = 900
    assert _openai_stream_error_exception(raw).retry_after_s is None
    exc = _openai_stream_error_exception(_TrustedSseJsonEvent(raw, retry_after_s=4.0))
    event = credential_safe_error_event(
        exc, provider_label="OpenAI", provider_name="openai", credential_values=[SECRET]
    )
    _, restored = runtime_owned_model_stream_error_event(event, fallback_provider="openai")
    assert restored is not None
    assert restored.retry_after_s == 4.0
    assert restored.retryable is True
    assert restored.status_code is None


@pytest.mark.parametrize("status,retryable", [(500, True), (503, True), (401, False), (429, False)])
def test_overload_http_error_retains_observed_status(status, retryable):
    response = httpx.Response(status, json={"error": {"code": OVERLOAD}})
    exc = _openai_api_error_from_response(response, "OpenAI failed", 4.0)
    assert exc.status_code == status
    assert exc.retryable is retryable
    assert exc.retry_after_s == 4.0
    assert exc.error_code == OVERLOAD


def test_top_level_stream_overload_has_no_http_status():
    exc = _openai_stream_error_exception({"type": "error", "code": OVERLOAD})
    assert exc.retryable is True
    assert exc.status_code is None
