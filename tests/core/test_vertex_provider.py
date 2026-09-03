from __future__ import annotations

import gzip
from collections.abc import Mapping
from typing import Any

import httpx
import pytest
from tests.provider_traceback_assertions import assert_cayu_traceback_does_not_retain

from cayu import (
    AgentSpec,
    AnthropicProvider,
    CayuApp,
    EventType,
    Message,
    RecentTurnsContextPolicy,
    RetryPolicy,
    RunRequest,
    StructuredOutputSpec,
)
from cayu.core.messages import TextPart, ThinkingPart
from cayu.providers import (
    HttpxVertexTransport,
    InputTokenCountConfidence,
    InputTokenCountMethod,
    ModelContextOverflowError,
    ModelRequest,
    ModelStreamEventType,
    VertexAPIError,
    VertexContextOverflowError,
    VertexProtocolError,
    VertexProvider,
)
from cayu.providers._http import MAX_PROVIDER_ERROR_BODY_CHARS, _TrustedSseJsonEvent
from cayu.providers.vertex import (
    VERTEX_OAUTH_SCOPE,
    _import_google,
    _resolve_credentials,
    _safe_gcp_error,
)
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME


class RecordingTransport:
    def __init__(self, responses: list[Mapping[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create_message(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {"url": url, "headers": dict(headers), "payload": dict(payload), "timeout_s": timeout_s}
        )
        if not self.responses:
            raise AssertionError("No fake Vertex response queued.")
        return self.responses.pop(0)


class FailingTransport:
    async def create_message(self, *, url, headers, payload, timeout_s) -> Mapping[str, Any]:
        raise VertexAPIError("Vertex AI request failed with HTTP 429: quota exhausted")


class FakeCredentials:
    def __init__(self, *, token: str = "fake-token", valid: bool = True) -> None:
        self.token = token
        self.valid = valid
        self.refresh_calls: list[Any] = []

    def refresh(self, request: Any) -> None:
        self.refresh_calls.append(request)
        self.valid = True
        self.token = "refreshed-token"


def _provider(transport: Any, **kwargs: Any) -> VertexProvider:
    options: dict[str, Any] = {
        "project_id": "demo-project",
        "region": "us-east5",
        "credentials": FakeCredentials(),
        "transport": transport,
    }
    options.update(kwargs)
    return VertexProvider(**options)


def _request() -> ModelRequest:
    return ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "Say hello.")],
    )


_OK_RESPONSE: dict[str, Any] = {
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {},
}


def test_vertex_provider_projects_privacy_safe_anthropic_options() -> None:
    provider = _provider(RecordingTransport([]))
    request = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[Message.text("user", "Say hello.")],
        options={
            "anthropic": {
                "temperature": 0.5,
                "metadata": {"private": "provider-option-secret"},
            }
        },
    )

    assert provider.request_footprint_options(request) == {
        "anthropic": {
            "max_tokens": provider.max_tokens,
            "temperature": 0.5,
        }
    }
    assert provider.request_fingerprint_options(request) == {
        "anthropic": {
            "max_tokens": provider.max_tokens,
            "temperature": 0.5,
            "metadata": {"private": "provider-option-secret"},
        }
    }

    different_default = _provider(RecordingTransport([]), max_tokens=1234)
    assert different_default.request_footprint_options(_request()) == {
        "anthropic": {"max_tokens": 1234}
    }


@pytest.mark.anyio
async def test_vertex_provider_emits_text_and_completed_events() -> None:
    transport = RecordingTransport(
        [
            {
                "id": "msg_1",
                "model": "claude-sonnet-4-6",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        ]
    )
    provider = _provider(transport)

    events = [event async for event in provider.stream(_request())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].delta == "hello"
    assert events[1].payload["stop_reason"] == "end_turn"
    call = transport.calls[0]
    assert call["url"] == (
        "https://us-east5-aiplatform.googleapis.com/v1/projects/demo-project"
        "/locations/us-east5/publishers/anthropic/models/"
        "claude-sonnet-4-6:rawPredict"
    )
    assert call["headers"]["Authorization"] == "Bearer fake-token"
    assert call["headers"]["content-type"] == "application/json"
    # The model lives in the URL, not the body; the version moves into the body.
    assert "model" not in call["payload"]
    assert call["payload"]["anthropic_version"] == "vertex-2023-10-16"


@pytest.mark.anyio
async def test_vertex_replays_only_its_matching_anthropic_protocol_state() -> None:
    transport = RecordingTransport(
        [
            {
                "id": "msg_reasoning",
                "model": "claude-sonnet-4-6",
                "stop_reason": "end_turn",
                "content": [
                    {"type": "thinking", "thinking": "vertex thought", "signature": "vertex-sig"},
                    {"type": "text", "text": "first answer"},
                ],
                "usage": {},
            },
            _OK_RESPONSE,
        ]
    )
    provider = _provider(transport)
    version_attribute = "anthropic_version"
    with pytest.raises(AttributeError):
        setattr(provider, version_attribute, "mutated-version")

    first_events = [event async for event in provider.stream(_request())]
    state = first_events[0].payload["provider_state"]
    assert state == {
        "provider": "vertex",
        "protocol": "messages",
        "protocol_version": "vertex-2023-10-16",
        "type": "thinking",
        "signature": "vertex-sig",
    }
    assert transport.calls[0]["payload"]["anthropic_version"] == "vertex-2023-10-16"

    continuation = ModelRequest(
        model="claude-sonnet-4-6",
        messages=[
            Message.text("user", "first"),
            Message(
                role="assistant",
                content=[
                    ThinkingPart(text="vertex thought", provider_state=state),
                    ThinkingPart(
                        text="direct Anthropic thought",
                        provider_state={
                            "provider": "anthropic",
                            "protocol": "messages",
                            "protocol_version": "2023-06-01",
                            "type": "thinking",
                            "signature": "anthropic-sig",
                        },
                    ),
                    TextPart(text="first answer"),
                ],
            ),
            Message.text("user", "continue"),
        ],
    )

    _ = [event async for event in provider.stream(continuation)]

    assert transport.calls[1]["payload"]["messages"][1]["content"] == [
        {"type": "thinking", "thinking": "vertex thought", "signature": "vertex-sig"},
        {"type": "text", "text": "first answer"},
    ]


@pytest.mark.anyio
async def test_vertex_provider_emits_tool_call_events() -> None:
    transport = RecordingTransport(
        [
            {
                "id": "msg_2",
                "model": "claude-sonnet-4-6",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "tool_use", "id": "tool_1", "name": "lookup", "input": {"q": "x"}},
                ],
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }
        ]
    )
    provider = _provider(transport)

    events = [event async for event in provider.stream(_request())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TOOL_CALL,
        ModelStreamEventType.COMPLETED,
    ]
    assert events[0].payload["name"] == "lookup"
    assert events[0].payload["arguments"] == {"q": "x"}


@pytest.mark.anyio
async def test_vertex_provider_honors_base_url_override() -> None:
    transport = RecordingTransport([_OK_RESPONSE])
    # Multi-region host pairs with region="us" (the GCP location), so locations/us is
    # what the URL must carry.
    provider = _provider(
        transport, region="us", base_url="https://aiplatform.us.rep.googleapis.com"
    )

    _ = [event async for event in provider.stream(_request())]

    assert transport.calls[0]["url"] == (
        "https://aiplatform.us.rep.googleapis.com/v1/projects/demo-project"
        "/locations/us/publishers/anthropic/models/claude-sonnet-4-6:rawPredict"
    )


@pytest.mark.anyio
async def test_vertex_provider_builds_non_default_region_host() -> None:
    transport = RecordingTransport([_OK_RESPONSE])
    provider = _provider(transport, region="europe-west1")

    _ = [event async for event in provider.stream(_request())]

    assert transport.calls[0]["url"] == (
        "https://europe-west1-aiplatform.googleapis.com/v1/projects/demo-project"
        "/locations/europe-west1/publishers/anthropic/models/"
        "claude-sonnet-4-6:rawPredict"
    )


@pytest.mark.anyio
async def test_vertex_provider_global_region_host() -> None:
    # `global` (GCP's recommended region) uses the bare aiplatform.googleapis.com host,
    # NOT the {region}-aiplatform template — matching the official AnthropicVertex SDK.
    transport = RecordingTransport([_OK_RESPONSE])
    provider = _provider(transport, region="global")

    _ = [event async for event in provider.stream(_request())]

    assert transport.calls[0]["url"] == (
        "https://aiplatform.googleapis.com/v1/projects/demo-project"
        "/locations/global/publishers/anthropic/models/"
        "claude-sonnet-4-6:rawPredict"
    )


@pytest.mark.anyio
async def test_vertex_provider_defaults_to_global_region() -> None:
    transport = RecordingTransport([_OK_RESPONSE])
    provider = VertexProvider(
        project_id="demo-project",
        credentials=FakeCredentials(),
        transport=transport,
    )

    _ = [event async for event in provider.stream(_request())]

    assert transport.calls[0]["url"] == (
        "https://aiplatform.googleapis.com/v1/projects/demo-project"
        "/locations/global/publishers/anthropic/models/"
        "claude-sonnet-4-6:rawPredict"
    )


@pytest.mark.anyio
@pytest.mark.parametrize("region", ["us", "eu"])
async def test_vertex_provider_multi_region_host(region: str) -> None:
    transport = RecordingTransport([_OK_RESPONSE])
    provider = _provider(transport, region=region)

    _ = [event async for event in provider.stream(_request())]

    assert transport.calls[0]["url"] == (
        f"https://aiplatform.{region}.rep.googleapis.com/v1/projects/demo-project"
        f"/locations/{region}/publishers/anthropic/models/"
        "claude-sonnet-4-6:rawPredict"
    )


def test_vertex_provider_rejects_non_https_base_url() -> None:
    with pytest.raises(ValueError, match="https"):
        VertexProvider(
            project_id="p",
            credentials=FakeCredentials(),
            base_url="http://insecure.example.com",
        )


@pytest.mark.anyio
async def test_vertex_provider_errors_on_empty_access_token() -> None:
    transport = RecordingTransport([_OK_RESPONSE])
    provider = _provider(transport, credentials=FakeCredentials(token="", valid=True))

    events = [event async for event in provider.stream(_request())]

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload["error"] == "VertexError: Vertex provider failed"
    assert transport.calls == []  # never reached the POST


@pytest.mark.anyio
async def test_vertex_provider_refreshes_expired_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setattr("cayu.providers.vertex._auth_request", lambda: sentinel)
    credentials = FakeCredentials(valid=False)
    transport = RecordingTransport([_OK_RESPONSE])
    provider = _provider(transport, credentials=credentials)

    _ = [event async for event in provider.stream(_request())]

    assert credentials.refresh_calls == [sentinel]
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer refreshed-token"


@pytest.mark.anyio
async def test_vertex_provider_wraps_api_error_as_single_event() -> None:
    provider = _provider(FailingTransport())

    events = [event async for event in provider.stream(_request())]

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload["error"] == "VertexAPIError: Vertex provider failed"
    # Typed classification fields survive into the error event payload.
    assert events[0].payload["error_type"] == "VertexAPIError"
    assert events[0].payload["provider"] == "vertex"


@pytest.mark.anyio
async def test_vertex_provider_stream_propagates_context_overflow() -> None:
    overflow = ModelContextOverflowError(
        "Vertex model context overflow",
        provider="vertex",
        status_code=400,
    )

    class OverflowTransport:
        async def create_message(self, *, url, headers, payload, timeout_s) -> Mapping[str, Any]:
            raise overflow

    provider = _provider(OverflowTransport())

    with pytest.raises(ModelContextOverflowError) as exc_info:
        [event async for event in provider.stream(_request())]

    # Overflow escapes as a fresh detached typed exception so runtime recovery
    # can retry without retaining transport traceback locals containing headers.
    assert exc_info.value is not overflow
    assert isinstance(exc_info.value, VertexContextOverflowError)
    assert exc_info.value.status_code == 400
    assert exc_info.value.retryable is False


def test_vertex_provider_rejects_multiple_credential_sources() -> None:
    with pytest.raises(ValueError, match="at most one"):
        VertexProvider(
            project_id="p",
            credentials=FakeCredentials(),
            service_account_file="/tmp/sa.json",
        )


def test_resolve_credentials_uses_adc_without_google_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    # The lazy import seam lets ADC/service-account resolution be exercised with a
    # fake google module, so the test runs without google-auth installed.
    resolved = FakeCredentials(token="adc-token")

    class FakeAuth:
        @staticmethod
        def default(*, scopes: list[str]) -> tuple[Any, str]:
            assert scopes == ["https://www.googleapis.com/auth/cloud-platform"]
            return resolved, "demo-project"

    monkeypatch.setattr(
        "cayu.providers.vertex._import_google",
        lambda name: (
            FakeAuth if name == "google.auth" else pytest.fail(f"unexpected import {name}")
        ),
    )
    credentials = _resolve_credentials(
        credentials=None, service_account_info=None, service_account_file=None
    )
    assert credentials is resolved


def test_safe_gcp_error_extracts_object_and_array_envelopes() -> None:
    obj = _safe_gcp_error({"error": {"code": 403, "status": "PERMISSION_DENIED", "message": "no"}})
    assert '"code":403' in obj and "PERMISSION_DENIED" in obj
    arr = _safe_gcp_error(
        [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "x"}}]
    )
    assert "RESOURCE_EXHAUSTED" in arr


def _mock_client_factory(handler: Any):
    # Capture the real client before monkeypatch replaces httpx.AsyncClient, so the
    # factory does not recurse into itself.
    real_client = httpx.AsyncClient

    def make(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("verify", None)
        kwargs.pop("timeout", None)
        return real_client(transport=httpx.MockTransport(handler))

    return make


_VERTEX_URL = (
    "https://us-east5-aiplatform.googleapis.com/v1/projects/p/locations/us-east5"
    "/publishers/anthropic/models/claude:rawPredict"
)


@pytest.mark.anyio
async def test_httpx_vertex_transport_wraps_gcp_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"error": {"code": 403, "status": "PERMISSION_DENIED", "message": "denied"}},
        )

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _mock_client_factory(handler))
    with pytest.raises(VertexAPIError) as exc_info:
        await HttpxVertexTransport().create_message(
            url=_VERTEX_URL, headers={}, payload={"a": 1}, timeout_s=10.0
        )
    assert str(exc_info.value) == (
        "Vertex AI request failed with HTTP 403: [provider response body omitted]"
    )
    assert exc_info.value.status_code == 403
    assert exc_info.value.error_type == "PERMISSION_DENIED"
    assert exc_info.value.retryable is False


@pytest.mark.anyio
async def test_httpx_vertex_transport_rejects_non_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json", headers={"content-type": "text/plain"})

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _mock_client_factory(handler))
    with pytest.raises(VertexProtocolError, match="valid JSON"):
        await HttpxVertexTransport().create_message(
            url=_VERTEX_URL, headers={}, payload={}, timeout_s=10.0
        )


@pytest.mark.anyio
async def test_httpx_vertex_transport_rejects_non_object(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2])

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _mock_client_factory(handler))
    with pytest.raises(VertexProtocolError, match="must be a JSON object"):
        await HttpxVertexTransport().create_message(
            url=_VERTEX_URL, headers={}, payload={}, timeout_s=10.0
        )


@pytest.mark.parametrize(
    ("kwarg", "value"),
    [
        ("service_account_info", {"type": "service_account"}),
        ("service_account_file", "/secrets/sa.json"),
    ],
)
def test_resolve_credentials_from_service_account(
    monkeypatch: pytest.MonkeyPatch, kwarg: str, value: Any
) -> None:
    resolved = FakeCredentials(token="sa-token")
    seen: dict[str, Any] = {}

    class FakeServiceAccount:
        class Credentials:
            @staticmethod
            def from_service_account_info(info: Any, *, scopes: list[str]) -> Any:
                seen.update(arg=info, scopes=scopes)
                return resolved

            @staticmethod
            def from_service_account_file(path: str, *, scopes: list[str]) -> Any:
                seen.update(arg=path, scopes=scopes)
                return resolved

    monkeypatch.setattr(
        "cayu.providers.vertex._import_google",
        lambda name: (
            FakeServiceAccount
            if name == "google.oauth2.service_account"
            else pytest.fail(f"unexpected import {name}")
        ),
    )
    kwargs: dict[str, Any] = {
        "credentials": None,
        "service_account_info": None,
        "service_account_file": None,
        kwarg: value,
    }
    credentials = _resolve_credentials(**kwargs)
    assert credentials is resolved
    assert seen == {"arg": value, "scopes": [VERTEX_OAUTH_SCOPE]}


def _raise_missing(name: str):
    def _raise(_module_name: str):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    return _raise


def test_vertex_provider_requires_google_auth_when_uninstalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No credentials passed -> ADC path -> lazy import of google.auth. Simulate it
    # being absent so the test is deterministic regardless of the env.
    monkeypatch.setattr(
        "cayu.providers.vertex.importlib.import_module", _raise_missing("google.auth")
    )
    with pytest.raises(RuntimeError, match=r"pip install cayu\[vertex\]"):
        VertexProvider(project_id="p")


@pytest.mark.parametrize(
    ("missing_name", "expected"),
    [
        ("google", RuntimeError),
        ("google.auth", RuntimeError),
        ("cachetools", ModuleNotFoundError),
    ],
)
def test_import_google_remaps_google_but_reraises_others(
    monkeypatch: pytest.MonkeyPatch, missing_name: str, expected: type[Exception]
) -> None:
    monkeypatch.setattr(
        "cayu.providers.vertex.importlib.import_module", _raise_missing(missing_name)
    )
    with pytest.raises(expected):
        _import_google("google.auth")


@pytest.mark.anyio
async def test_vertex_provider_supports_structured_output_via_tools() -> None:
    # Structured output is runtime-level (TOOL strategy): the runtime injects the
    # submit tool, and Vertex returns it as a tool_use that the runtime validates.
    transport = RecordingTransport(
        [
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_final",
                        "name": STRUCTURED_OUTPUT_TOOL_NAME,
                        "input": {"output": {"answer": "ok"}},
                    }
                ],
                "usage": {},
            }
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(_provider(transport), default=True)
    app.register_agent(AgentSpec(name="assistant", model="claude-sonnet-4-6"))

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "answer with structured output")],
                structured_output=StructuredOutputSpec(
                    name="answer",
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                ),
            )
        )
    ]

    validated = next(e for e in events if e.type == EventType.STRUCTURED_OUTPUT_VALIDATED)
    assert validated.payload["output"] == {"answer": "ok"}
    # The runtime's structured-output tool was sent to Vertex in the request body.
    sent_tools = transport.calls[0]["payload"]["tools"]
    assert any(tool["name"] == STRUCTURED_OUTPUT_TOOL_NAME for tool in sent_tools)


@pytest.mark.anyio
async def test_vertex_coexists_with_anthropic_in_one_app() -> None:
    # Vertex registers alongside AnthropicProvider (distinct names) and, as the
    # default, serves the run while Anthropic remains registered.
    transport = RecordingTransport([_OK_RESPONSE])
    app = CayuApp(enable_logging=False)
    app.register_provider(AnthropicProvider(api_key="test-key"))
    app.register_provider(_provider(transport), default=True)
    app.register_agent(AgentSpec(name="assistant", model="claude-sonnet-4-6"))

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hi")],
                max_steps=1,
            )
        )
    ]

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert transport.calls  # the Vertex (default) provider served the run


class StreamingRecordingTransport:
    """Fake transport exposing the SSE streaming and token-count seams."""

    def __init__(
        self,
        event_batches: list[list[Mapping[str, Any]]] | None = None,
        count_responses: list[Mapping[str, Any]] | None = None,
    ) -> None:
        self.event_batches = list(event_batches or [])
        self.count_responses = list(count_responses or [])
        self.calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []

    async def count_message_tokens(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        self.count_calls.append(
            {"url": url, "headers": dict(headers), "payload": dict(payload), "timeout_s": timeout_s}
        )
        if not self.count_responses:
            raise AssertionError("No fake Vertex count response queued.")
        return self.count_responses.pop(0)

    async def create_message(self, *, url, headers, payload, timeout_s) -> Mapping[str, Any]:
        raise AssertionError("create_message must not be used when streaming is available.")

    async def stream_message_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        transport_idle_timeout_s: float,
        protocol_idle_timeout_s: float,
        semantic_progress_timeout_s: float,
        absolute_stream_timeout_s: float,
    ):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_s": timeout_s,
                "transport_idle_timeout_s": transport_idle_timeout_s,
                "protocol_idle_timeout_s": protocol_idle_timeout_s,
                "semantic_progress_timeout_s": semantic_progress_timeout_s,
                "absolute_stream_timeout_s": absolute_stream_timeout_s,
            }
        )
        if not self.event_batches:
            raise AssertionError("No fake Vertex stream queued.")
        for event in self.event_batches.pop(0):
            yield event


@pytest.mark.anyio
async def test_vertex_provider_streams_sse_events_incrementally() -> None:
    transport = StreamingRecordingTransport(
        event_batches=[
            [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_v1",
                        "model": "claude-sonnet-4-6",
                        "usage": {"input_tokens": 9},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hel"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "lo"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 4},
                },
                {"type": "message_stop"},
            ]
        ]
    )
    provider = _provider(transport)

    events = [event async for event in provider.stream(_request())]

    assert [event.type for event in events] == [
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.TEXT_DELTA,
        ModelStreamEventType.COMPLETED,
    ]
    assert [events[0].delta, events[1].delta] == ["hel", "lo"]
    assert events[2].payload["stop_reason"] == "end_turn"
    assert events[2].payload["usage"] == {"input_tokens": 9, "output_tokens": 4}

    call = transport.calls[0]
    assert call["url"] == (
        "https://us-east5-aiplatform.googleapis.com/v1/projects/demo-project"
        "/locations/us-east5/publishers/anthropic/models/"
        "claude-sonnet-4-6:streamRawPredict"
    )
    assert call["payload"]["stream"] is True
    assert "model" not in call["payload"]
    assert call["payload"]["anthropic_version"] == "vertex-2023-10-16"
    assert call["headers"]["Authorization"] == "Bearer fake-token"
    assert call["transport_idle_timeout_s"] == 120.0
    assert call["protocol_idle_timeout_s"] == 120.0
    assert call["semantic_progress_timeout_s"] == 120.0
    assert call["absolute_stream_timeout_s"] == 600.0


@pytest.mark.anyio
async def test_vertex_provider_rejects_duplicate_content_block_start() -> None:
    start = {
        "type": "message_start",
        "message": {"id": "msg_v1", "model": "claude-sonnet-4-6"},
    }
    block = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "tool-1", "name": "read_file"},
    }
    transport = StreamingRecordingTransport(event_batches=[[start, block, block]])
    provider = _provider(transport)

    events = [event async for event in provider.runtime_stream(_request())]

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload["error_type"] == "VertexProtocolError"
    assert "provider_deadline_kind" not in events[0].payload


@pytest.mark.anyio
async def test_vertex_provider_rejects_post_stop_output_before_completion() -> None:
    start = {
        "type": "message_start",
        "message": {"id": "msg_v1", "model": "claude-sonnet-4-6"},
    }
    transport = StreamingRecordingTransport(
        event_batches=[[start, {"type": "message_stop"}, {"type": "message_delta", "delta": {}}]]
    )
    provider = _provider(transport)

    events = [event async for event in provider.stream(_request())]

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload["error_type"] == "VertexProtocolError"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error_type", "expected_status", "expected_retryable"),
    [
        pytest.param("api_error", 500, True, id="api"),
        pytest.param("authentication_error", 401, False, id="authentication"),
        pytest.param("billing_error", 402, False, id="billing"),
        pytest.param("conflict_error", 409, False, id="conflict"),
        pytest.param("invalid_request_error", 400, False, id="invalid-request"),
        pytest.param("not_found_error", 404, False, id="not-found"),
        pytest.param("overloaded_error", 529, True, id="overloaded"),
        pytest.param("permission_error", 403, False, id="permission"),
        pytest.param("rate_limit_error", 429, True, id="rate-limit"),
        pytest.param("timeout_error", 504, True, id="timeout"),
    ],
)
async def test_vertex_provider_stream_error_event_uses_vertex_typed_errors(
    error_type: str,
    expected_status: int,
    expected_retryable: bool,
) -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    transport = StreamingRecordingTransport(
        event_batches=[
            [
                {
                    "type": "error",
                    "request_id": "req_vertex_cutoff",
                    "error": {
                        "type": error_type,
                        "code": "provider_code",
                        "message": "x" * (MAX_PROVIDER_ERROR_BODY_CHARS - 10) + secret,
                    },
                }
            ]
        ]
    )
    provider = _provider(transport)

    events = [event async for event in provider.stream(_request())]

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload["error"] == "VertexAPIError: Vertex provider failed"
    assert events[0].payload["error_type"] == "VertexAPIError"
    assert events[0].payload["provider"] == "vertex"
    assert events[0].payload["status_code"] == expected_status
    assert events[0].payload["provider_error_type"] == error_type
    assert events[0].payload["retryable"] is expected_retryable
    rendered = repr([event.model_dump(mode="json") for event in events])
    assert not any(secret[:size] in rendered for size in range(8, len(secret) + 1))


@pytest.mark.anyio
async def test_vertex_provider_stream_error_preserves_trusted_retry_after() -> None:
    raw_event = _TrustedSseJsonEvent(
        {"type": "error", "error": {"type": "api_error"}},
        retry_after_s=4.5,
    )
    transport = StreamingRecordingTransport(event_batches=[[raw_event]])
    provider = _provider(transport)

    events = [event async for event in provider.stream(_request())]

    assert [event.type for event in events] == [ModelStreamEventType.ERROR]
    assert events[0].payload["status_code"] == 500
    assert events[0].payload["retryable"] is True
    assert events[0].payload["retry_after_s"] == 4.5


@pytest.mark.anyio
async def test_vertex_provider_stream_error_event_propagates_context_overflow() -> None:
    secret = "workload-secret-canary-ABCDEFGHIJKLMNOP"
    transport = StreamingRecordingTransport(
        event_batches=[
            [
                {
                    "type": "error",
                    "request_id": "req_vertex_overflow",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "context_length_exceeded",
                        "message": (
                            "x" * (MAX_PROVIDER_ERROR_BODY_CHARS + 1)
                            + " prompt is too long "
                            + secret
                        ),
                    },
                }
            ]
        ]
    )
    provider = _provider(transport)

    with pytest.raises(VertexContextOverflowError) as exc_info:
        [event async for event in provider.stream(_request())]

    assert isinstance(exc_info.value, ModelContextOverflowError)
    assert exc_info.value.error_type == "invalid_request_error"
    assert secret not in repr((str(exc_info.value), vars(exc_info.value)))
    assert exc_info.value.response_body is None
    assert exc_info.value.provider == "vertex"
    assert exc_info.value.retryable is False


@pytest.mark.anyio
async def test_vertex_provider_counts_input_tokens_with_official_endpoint() -> None:
    transport = StreamingRecordingTransport(count_responses=[{"input_tokens": 42}])
    provider = _provider(transport)

    result = await provider.count_input_tokens(_request())

    assert result is not None
    assert result.input_tokens == 42
    assert result.method == InputTokenCountMethod.OFFICIAL
    assert result.confidence == InputTokenCountConfidence.HIGH
    assert result.metadata == {
        "endpoint": "count-tokens:rawPredict",
        "provider_billing_status": "not_documented",
    }
    call = transport.count_calls[0]
    assert call["url"] == (
        "https://us-east5-aiplatform.googleapis.com/v1/projects/demo-project"
        "/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict"
    )
    # Unlike message creation, the real model stays in the count-tokens body
    # (the URL model segment is the literal "count-tokens").
    assert call["payload"]["model"] == "claude-sonnet-4-6"
    assert call["payload"]["anthropic_version"] == "vertex-2023-10-16"
    assert "max_tokens" not in call["payload"]
    assert call["headers"]["Authorization"] == "Bearer fake-token"


@pytest.mark.anyio
async def test_vertex_token_count_failure_drops_access_token_and_exception_graph() -> None:
    canary = "provider-vertex-token-canary-0123456789"

    class CredentialFailingTransport(StreamingRecordingTransport):
        async def count_message_tokens(self, **_kwargs: Any) -> Mapping[str, Any]:
            error = RuntimeError(f"Authorization: Bearer {canary}")
            error.headers = {"Authorization": f"Bearer {canary}"}
            error.add_note(f"transport retained {canary}")
            raise error

    provider = _provider(
        CredentialFailingTransport(),
        credentials=FakeCredentials(token=canary),
    )

    with pytest.raises(VertexAPIError) as exc_info:
        await provider.count_input_tokens(_request())

    assert_cayu_traceback_does_not_retain(exc_info.value, provider)
    retained = str(exc_info.value) + repr(exc_info.value) + repr(vars(exc_info.value))
    assert canary not in retained
    assert str(exc_info.value) == "RuntimeError: Vertex provider failed"
    assert exc_info.value.response_body is None
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.anyio
async def test_vertex_provider_count_input_tokens_unavailable_without_transport_support() -> None:
    # Transports predating the count seam stay source-compatible: counting is
    # reported unavailable instead of raising AttributeError.
    provider = _provider(FailingTransport())

    assert await provider.count_input_tokens(_request()) is None


@pytest.mark.anyio
async def test_vertex_provider_rejects_invalid_token_count_response() -> None:
    transport = StreamingRecordingTransport(count_responses=[{"input_tokens": "42"}])
    provider = _provider(transport)

    with pytest.raises(VertexProtocolError, match="input_tokens"):
        await provider.count_input_tokens(_request())


@pytest.mark.anyio
@pytest.mark.parametrize("status_code", [400, 413])
async def test_httpx_vertex_transport_classifies_prompt_too_long(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={
                "error": {
                    "code": status_code,
                    "status": "INVALID_ARGUMENT",
                    "message": "prompt is too long: 250000 tokens > 200000 maximum",
                }
            },
        )

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _mock_client_factory(handler))
    with pytest.raises(VertexContextOverflowError) as exc_info:
        await HttpxVertexTransport().create_message(
            url=_VERTEX_URL, headers={}, payload={"a": 1}, timeout_s=10.0
        )

    assert isinstance(exc_info.value, ModelContextOverflowError)
    assert exc_info.value.provider == "vertex"
    assert exc_info.value.status_code == status_code
    assert exc_info.value.error_type == "INVALID_ARGUMENT"
    assert exc_info.value.retryable is False


@pytest.mark.anyio
async def test_runtime_recovers_from_compressed_vertex_413_without_reading_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ResponseBody(httpx.AsyncByteStream):
        def __init__(self, content: bytes) -> None:
            self.content = content
            self.yielded = False
            self.closed = False

        async def __aiter__(self):
            self.yielded = True
            yield self.content

        async def aclose(self) -> None:
            self.closed = True

    compressed_body = ResponseBody(
        gzip.compress(b'{"error":{"status":"INVALID_ARGUMENT","message":"request too large"}}')
    )
    success_body = ResponseBody(
        b'data: {"type":"message_start","message":{"id":"msg_v1",'
        b'"model":"claude-sonnet-4-6","usage":{"input_tokens":1}}}\n\n'
        b'data: {"type":"content_block_start","index":0,'
        b'"content_block":{"type":"text","text":""}}\n\n'
        b'data: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"text_delta","text":"ok"}}\n\n'
        b'data: {"type":"content_block_stop","index":0}\n\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        b'"usage":{"output_tokens":1}}\n\n'
        b'data: {"type":"message_stop"}\n\n'
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                413,
                headers={
                    "content-encoding": "gzip",
                    "content-type": "application/json",
                },
                stream=compressed_body,
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=success_body,
            request=request,
        )

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _mock_client_factory(handler))
    app = CayuApp(enable_logging=False)
    app.register_provider(_provider(HttpxVertexTransport()), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="claude-sonnet-4-6"),
        context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
    )

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            )
        )
    ]

    assert calls == 2
    assert not compressed_body.yielded
    assert compressed_body.closed
    assert success_body.yielded
    assert success_body.closed
    assert [
        event.type
        for event in events
        if event.type
        in {
            EventType.CONTEXT_OVERFLOW_DETECTED,
            EventType.CONTEXT_OVERFLOW_RECOVERING,
            EventType.CONTEXT_OVERFLOW_FAILED,
            EventType.SESSION_COMPLETED,
        }
    ] == [
        EventType.CONTEXT_OVERFLOW_DETECTED,
        EventType.CONTEXT_OVERFLOW_RECOVERING,
        EventType.SESSION_COMPLETED,
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error_type", "expected_public_type", "expected_retryable"),
    [
        pytest.param("UNAUTHENTICATED", "UNAUTHENTICATED", False, id="authentication"),
        pytest.param("FUTURE_STATUS", None, None, id="unknown"),
        pytest.param(True, None, None, id="boolean"),
        pytest.param("", None, None, id="blank"),
    ],
)
async def test_runtime_does_not_recover_from_conflicting_vertex_413(
    monkeypatch: pytest.MonkeyPatch,
    error_type: Any,
    expected_public_type: str | None,
    expected_retryable: bool | None,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            413,
            headers={"content-type": "application/json"},
            json={
                "error": {
                    "code": 413,
                    "status": error_type,
                    "message": "denied",
                }
            },
            request=request,
        )

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _mock_client_factory(handler))
    app = CayuApp(
        enable_logging=False,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )
    app.register_provider(_provider(HttpxVertexTransport()), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="claude-sonnet-4-6"),
        context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
    )

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_conflicting_vertex_413_{error_type}",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            )
        )
    ]

    assert calls == 1
    assert not {
        EventType.CONTEXT_OVERFLOW_DETECTED,
        EventType.CONTEXT_OVERFLOW_RECOVERING,
        EventType.MODEL_RETRY,
    }.intersection(event.type for event in events)
    model_errors = [event for event in events if event.type == EventType.MODEL_ERROR]
    assert len(model_errors) == 1
    assert model_errors[0].payload["status_code"] == 413
    assert model_errors[0].payload.get("provider_error_type") == expected_public_type
    assert model_errors[0].payload.get("retryable") is expected_retryable
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.anyio
async def test_httpx_vertex_transport_populates_typed_error_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "4"},
            json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota"}},
        )

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _mock_client_factory(handler))
    with pytest.raises(VertexAPIError) as exc_info:
        await HttpxVertexTransport().create_message(
            url=_VERTEX_URL, headers={}, payload={"a": 1}, timeout_s=10.0
        )

    assert exc_info.value.status_code == 429
    assert exc_info.value.error_type == "RESOURCE_EXHAUSTED"
    assert exc_info.value.retryable is True
    assert exc_info.value.retry_after_s == 4.0


@pytest.mark.anyio
async def test_httpx_vertex_transport_defers_unknown_error_identity_to_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "error": {
                    "code": 500,
                    "status": "FUTURE_TRANSIENT_STATUS",
                    "message": "future failure",
                }
            },
        )

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _mock_client_factory(handler))
    with pytest.raises(VertexAPIError) as exc_info:
        await HttpxVertexTransport().create_message(
            url=_VERTEX_URL, headers={}, payload={"a": 1}, timeout_s=10.0
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.error_type == "FUTURE_TRANSIENT_STATUS"
    assert exc_info.value.retryable is None


@pytest.mark.anyio
async def test_runtime_does_not_retry_conflicting_buffered_vertex_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            500,
            headers={"retry-after": "4"},
            json={
                "error": {
                    "code": 500,
                    "status": "UNAUTHENTICATED",
                    "message": "denied",
                }
            },
        )

    monkeypatch.setattr("cayu.providers._http.httpx.AsyncClient", _mock_client_factory(handler))
    app = CayuApp(
        enable_logging=False,
        retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
    )
    app.register_provider(_provider(HttpxVertexTransport()), default=True)
    app.register_agent(AgentSpec(name="assistant", model="claude-sonnet-4-6"))

    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_conflicting_buffered_vertex_identity",
                messages=[Message.text("user", "hello")],
            )
        )
    ]

    assert calls == 1
    assert EventType.MODEL_RETRY not in {event.type for event in events}
    model_errors = [event for event in events if event.type == EventType.MODEL_ERROR]
    assert len(model_errors) == 1
    assert model_errors[0].payload["status_code"] == 500
    assert model_errors[0].payload["provider_error_type"] == "UNAUTHENTICATED"
    assert model_errors[0].payload["retryable"] is False
    assert model_errors[0].payload["retry_after_s"] == 4.0
    assert events[-1].type == EventType.SESSION_FAILED


def test_vertex_provider_rejects_invalid_transport_idle_timeout() -> None:
    with pytest.raises(ValueError, match="transport_idle_timeout_s"):
        _provider(FailingTransport(), transport_idle_timeout_s=0)
    with pytest.raises(TypeError, match="transport_idle_timeout_s"):
        _provider(FailingTransport(), transport_idle_timeout_s="60")


@pytest.mark.parametrize("timeout_s", [float("nan"), float("inf"), float("-inf"), 10**1000])
def test_vertex_provider_rejects_nonfinite_timeout(timeout_s: int | float) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        _provider(FailingTransport(), timeout_s=timeout_s)
