from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from cayu import (
    AgentSpec,
    AnthropicProvider,
    CayuApp,
    EventType,
    Message,
    RunRequest,
    StructuredOutputSpec,
)
from cayu.providers import (
    HttpxVertexTransport,
    ModelContextOverflowError,
    ModelRequest,
    ModelStreamEventType,
    VertexAPIError,
    VertexProtocolError,
    VertexProvider,
)
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
    assert "did not produce an access token" in events[0].payload["error"]
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
    assert "429" in events[0].payload["error"]
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

    # Overflow must escape as a typed exception (not an ERROR event) so
    # runtime context-overflow recovery can shrink context and retry.
    assert exc_info.value is overflow
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
    with pytest.raises(VertexAPIError, match="PERMISSION_DENIED"):
        await HttpxVertexTransport().create_message(
            url=_VERTEX_URL, headers={}, payload={"a": 1}, timeout_s=10.0
        )


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
