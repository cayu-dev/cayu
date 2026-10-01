from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from tests.provider_traceback_assertions import assert_cayu_traceback_does_not_retain

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, run_to_completion
from cayu.context.base import ContextPolicy, ContextRequest
from cayu.context.counting import ContextCountingConfig, ContextCountingMode
from cayu.providers import OpenAIProvider
from cayu.providers._http import (
    OMITTED_PROVIDER_ERROR_BODY,
    attach_provider_error_text,
    bind_provider_error_workload_redactor,
    credential_safe_provider_exception,
    post_json,
    provider_error_body_text,
    reset_provider_error_workload_redactor,
)
from cayu.providers.anthropic import (
    AnthropicProvider,
    HttpxAnthropicTransport,
    anthropic_stream_events,
)
from cayu.providers.anthropic import _safe_error_response_text as anthropic_error_text
from cayu.providers.base import ModelProviderError
from cayu.providers.chat_completions import ChatCompletionsProvider, HttpxChatCompletionsTransport
from cayu.providers.chat_completions import (
    _safe_error_response_text as chat_completions_error_text,
)
from cayu.providers.openai import HttpxOpenAITransport
from cayu.providers.openai import _safe_error_response_text as openai_error_text
from cayu.providers.vertex import HttpxVertexTransport, VertexProvider
from cayu.providers.vertex import _safe_error_response_text as vertex_error_text
from cayu.sessions.outcomes import RunOutcome
from cayu.vaults import SecretRedactor
from cayu.vaults.redaction import REDACTED_SECRET

_BUNDLED_HTTP_ERROR_FORMATTERS: tuple[Callable[[httpx.Response], str], ...] = (
    openai_error_text,
    anthropic_error_text,
    chat_completions_error_text,
    vertex_error_text,
)


@pytest.mark.parametrize("format_error", _BUNDLED_HTTP_ERROR_FORMATTERS)
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(
            500,
            request=httpx.Request("POST", "https://provider.invalid"),
            headers={"content-type": "application/json"},
            content=(
                b'{"error":{"type":"server_error","message":'
                b'"prefix-provider\\u002derror\\u002dsecret-suffix"}}'
            ),
        ),
        httpx.Response(
            500,
            request=httpx.Request("POST", "https://provider.invalid"),
            headers={"content-type": "application/json"},
            text='{"error":{"message":"x' + "provider-error-secret"[:10],
        ),
        httpx.Response(
            500,
            request=httpx.Request("POST", "https://provider.invalid"),
            headers={"content-type": "text/plain"},
            text="x" * 1_995 + "provider-error-secret",
        ),
    ],
    ids=("json-escaped", "malformed-json", "unstructured"),
)
def test_bundled_provider_http_errors_omit_untrusted_bodies(
    format_error: Callable[[httpx.Response], str],
    response: httpx.Response,
) -> None:
    rendered = format_error(response)

    assert rendered == "[provider response body omitted]"
    assert "provider-error-secret" not in rendered
    assert "provider-error" not in rendered


def _error_response(body: bytes, *, content_type: str = "application/json") -> httpx.Response:
    return httpx.Response(
        400,
        request=httpx.Request(
            "POST",
            "https://provider.invalid/v1/responses?key=query-credential-123&api-version=1",
            headers={"authorization": "Bearer header-credential-123", "accept": "identity"},
        ),
        headers={"content-type": content_type},
        content=body,
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            b'{"error":{"type":"invalid_request_error","message":"The model does not exist"}}',
            "The model does not exist",
        ),
        (b'[{"error":{"code":403,"message":"Permission denied"}}]', "Permission denied"),
        (
            b'{"detail":"not supported with a ChatGPT account"}',
            "not supported with a ChatGPT account",
        ),
        (b"plain gateway failure", "plain gateway failure"),
    ],
    ids=("openai", "gcp-array", "fastapi-detail", "plain-text"),
)
def test_provider_error_body_text_extracts_the_message(body: bytes, expected: str) -> None:
    assert provider_error_body_text(_error_response(body)) == expected


def test_provider_error_body_text_removes_the_requests_credentials() -> None:
    body = b'{"error":{"message":"bad key header-credential-123 query-credential-123 identity"}}'

    text = provider_error_body_text(_error_response(body))

    assert text == f"bad key {REDACTED_SECRET} {REDACTED_SECRET} identity"


def test_provider_error_body_text_omits_oversized_bodies() -> None:
    assert provider_error_body_text(_error_response(b"x" * (64 * 1024 + 1))) is None


def _attached_failure(text: str) -> ModelProviderError:
    failure = ModelProviderError(
        f"OpenAI API request failed with HTTP 400: {OMITTED_PROVIDER_ERROR_BODY}",
        provider="openai",
        status_code=400,
        error_code="model_not_found",
        request_id="req_123",
    )
    attach_provider_error_text(failure, text)
    return failure


def test_public_boundary_shows_the_message_with_known_secrets_removed() -> None:
    failure = _attached_failure("bad request for api-key-123 and workload-canary-123")
    token = bind_provider_error_workload_redactor(SecretRedactor(["workload-canary-123"]))
    try:
        public = credential_safe_provider_exception(
            failure,
            provider_label="OpenAI",
            provider_name="openai",
            credential_values=("api-key-123",),
        )
    finally:
        reset_provider_error_workload_redactor(token)

    assert str(public) == (
        f"OpenAI API request failed with HTTP 400: bad request for {REDACTED_SECRET} "
        f"and {REDACTED_SECRET}"
    )
    assert public.error_code == "model_not_found"
    assert public.request_id == "req_123"
    # The raw failure itself never carries the provider text.
    assert "bad request" not in str(failure)
    assert "bad request" not in repr(vars(failure))


@pytest.mark.parametrize("bound", [True, False])
def test_public_boundary_truncates_only_with_the_complete_registry(bound: bool) -> None:
    secret = "workload-canary-ABCDEFGHIJKLMNOP"
    failure = _attached_failure("x" * 1_990 + secret + " tail")
    token = bind_provider_error_workload_redactor(SecretRedactor([secret]) if bound else None)
    try:
        public = credential_safe_provider_exception(
            failure,
            provider_label="OpenAI",
            provider_name="openai",
            credential_values=("api-key-123",),
        )
    finally:
        reset_provider_error_workload_redactor(token)

    message = str(public)
    assert not any(secret[:size] in message for size in range(8, len(secret) + 1))
    if bound:
        assert message.endswith("...[truncated]")
        assert len(message.encode()) <= 2_048
    else:
        # Without the app's registry a cut secret could escape a later pass.
        assert message == "ModelProviderError: OpenAI provider failed"


def test_public_boundary_keeps_the_fixed_message_without_credentials() -> None:
    public = credential_safe_provider_exception(
        _attached_failure("unchecked provider text"),
        provider_label="OpenAI",
        provider_name="openai",
        credential_values=(),
    )

    assert str(public) == "ModelProviderError: OpenAI provider failed"
    assert public.request_id is None


def test_runtime_events_show_provider_errors_without_known_secrets() -> None:
    workload_secret = "workload-canary-0123456789"
    body = {
        "error": {
            "type": "invalid_request_error",
            "code": "model_not_found",
            "message": f"The model does not exist. key sk-test-key {workload_secret}",
        }
    }

    async def run() -> RunOutcome:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404, json=body, headers={"x-request-id": "req_abc123"}, request=request
            )

        transport = HttpxOpenAITransport()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport._client._client = client
            app = CayuApp(enable_logging=False, secret_redactor=SecretRedactor([workload_secret]))
            app.register_provider(
                OpenAIProvider(api_key="sk-test-key", transport=transport), default=True
            )
            app.register_agent(AgentSpec(name="assistant", model="gpt-missing"))
            return await run_to_completion(
                app,
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "hello")],
                    max_steps=1,
                ),
            )

    outcome = asyncio.run(run())

    expected = (
        "OpenAI API request failed with HTTP 404: The model does not exist. "
        f"key {REDACTED_SECRET} {REDACTED_SECRET}"
    )
    model_error = next(event for event in outcome.events if event.type == EventType.MODEL_ERROR)
    assert model_error.payload["error"] == expected
    assert model_error.payload["provider_error_code"] == "model_not_found"
    assert model_error.payload["request_id"] == "req_abc123"
    assert outcome.error == expected
    rendered = json.dumps([event.payload for event in outcome.events], default=str)
    assert "sk-test-key" not in rendered
    assert workload_secret not in rendered


class _ErrorBodyStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.content


_QUERY_SECRET = "gateway-query-canary-0123456789"
_PROXY_SECRET = "proxy-header-canary-0123456789"
_API_SECRET = "provider-api-canary-0123456789"
_COOKIE_SECRET = "session-cookie-canary-0123456789"
_QUOTED_COOKIE_SECRET = "quoted-cookie-canary-0123456789"
_ERROR_TRANSPORT_CASES = [
    pytest.param(protocol, mode, id=f"{mode}-{protocol}")
    for protocol in ("chat", "openai", "anthropic")
    for mode in ("buffered-http", "streamed-http", "sse")
] + [
    pytest.param("openai", mode, id=f"{mode}-openai")
    for mode in ("final-json", "background-sse", "background-error")
]


async def _run_provider_error(
    protocol: str,
    response_mode: str,
    *,
    message: str,
    workload_secret: str | None = None,
    identity_secret: str | None = None,
    request_id_location: str | None = None,
    header_request_id: str = "req_correlation_123",
    context_counting: bool = False,
    context_policy: ContextPolicy | None = None,
) -> RunOutcome:
    def handler(request: httpx.Request) -> httpx.Response:
        # Both values are known to HTTPX, but are not the provider's API key
        # or extra_headers. Exercise the real transport, including its copy
        # of bounded HTTP error responses, rather than a parser-only fixture.
        assert request.url.params["key"] == _QUERY_SECRET
        assert request.headers["x-proxy-token"] == _PROXY_SECRET
        body: dict[str, Any] = {
            "type": "error",
            "request_id": identity_secret or _PROXY_SECRET,
            "error": {
                "type": "invalid_request_error",
                "code": identity_secret or _QUERY_SECRET,
                "message": message,
                "request_id": identity_secret or _PROXY_SECRET,
            },
        }
        headers = {
            "x-request-id": identity_secret or _QUERY_SECRET,
            "request-id": identity_secret or _QUERY_SECRET,
            "content-type": "application/json",
        }
        if request_id_location is not None:
            body.pop("request_id")
            body["error"].pop("request_id")
            headers.pop("request-id")
            headers.pop("x-request-id")
            if request_id_location == "top-level":
                body["request_id"] = "req_correlation_123"
            elif request_id_location == "nested":
                body["error"]["request_id"] = "req_correlation_123"
            else:
                headers[request_id_location] = header_request_id
        if response_mode == "final-json":
            body.update({"id": "resp_failed", "status": "failed"})
            return httpx.Response(200, json=body, headers=headers, request=request)
        if response_mode == "buffered-http":
            return httpx.Response(400, json=body, headers=headers, request=request)
        content = json.dumps(body).encode()
        if response_mode in {"background-sse", "background-error"}:
            created = {
                "type": "response.created",
                "sequence_number": 0,
                "response": {"id": "resp_background", "status": "in_progress"},
            }
            failed = (
                {
                    "type": "response.failed",
                    "sequence_number": 1,
                    "response": {**body, "id": "resp_background", "status": "failed"},
                }
                if response_mode == "background-sse"
                else {**body, "sequence_number": 1}
            )
            content = b"".join(
                b"data: " + json.dumps(event).encode() + b"\n\n" for event in (created, failed)
            )
            headers["content-type"] = "text/event-stream"
        elif response_mode == "sse":
            content = b"data: " + content + b"\n\n"
            headers["content-type"] = "text/event-stream"
        return httpx.Response(
            200 if response_mode in {"sse", "background-sse", "background-error"} else 400,
            stream=_ErrorBodyStream(content),
            headers=headers,
            request=request,
        )

    if protocol == "openai":
        transport = HttpxOpenAITransport()
        provider = OpenAIProvider(
            api_key=_API_SECRET,
            transport=transport,
            streaming=response_mode != "final-json",
            background=response_mode in {"background-sse", "background-error"},
        )
    elif protocol == "anthropic":
        transport = HttpxAnthropicTransport()
        provider = AnthropicProvider(api_key=_API_SECRET, transport=transport)
    elif protocol == "vertex":
        transport = HttpxVertexTransport()
        provider = VertexProvider(
            project_id="test-project",
            region="us-east5",
            credentials=SimpleNamespace(valid=True, token=_API_SECRET),
            transport=transport,
        )
    else:
        transport = HttpxChatCompletionsTransport()
        provider = ChatCompletionsProvider(
            api_key=_API_SECRET,
            endpoint_url=f"https://provider.invalid/chat/completions?key={_QUERY_SECRET}",
            transport=transport,
        )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={
            "x-proxy-token": _PROXY_SECRET,
            "cookie": f'session={_COOKIE_SECRET}; other="{_QUOTED_COOKIE_SECRET}"',
        },
        params={"key": _QUERY_SECRET},
    ) as client:
        transport._client._client = client
        app = CayuApp(
            enable_logging=False,
            secret_redactor=SecretRedactor(workload_secret),
            context_counting=ContextCountingConfig(
                mode=ContextCountingMode.OBSERVE if context_counting else ContextCountingMode.OFF
            ),
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="test-model"), context_policy=context_policy
        )
        return await run_to_completion(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "hello")],
                max_steps=1,
            ),
        )


@pytest.mark.parametrize(("protocol", "response_mode"), _ERROR_TRANSPORT_CASES)
def test_runtime_redacts_request_credentials_in_messages_and_identifiers(
    protocol: str,
    response_mode: str,
) -> None:
    outcome = asyncio.run(
        _run_provider_error(
            protocol,
            response_mode,
            message=(
                f"Request rejected: {_QUERY_SECRET} {_PROXY_SECRET} {_API_SECRET} "
                f"{_COOKIE_SECRET} {_QUOTED_COOKIE_SECRET}"
            ),
        )
    )

    model_error = next(event for event in outcome.events if event.type == EventType.MODEL_ERROR)
    assert (
        f"Request rejected: {REDACTED_SECRET} {REDACTED_SECRET} {REDACTED_SECRET}"
        in (model_error.payload["error"])
    )
    assert model_error.payload["provider_error_type"] == "invalid_request_error"
    assert "provider_error_code" not in model_error.payload
    assert "request_id" not in model_error.payload
    assert outcome.error == model_error.payload["error"]
    rendered = json.dumps([event.payload for event in outcome.events], default=str)
    for secret in (
        _QUERY_SECRET,
        _PROXY_SECRET,
        _API_SECRET,
        _COOKIE_SECRET,
        _QUOTED_COOKIE_SECRET,
    ):
        assert secret not in rendered


@pytest.mark.parametrize(("protocol", "response_mode"), _ERROR_TRANSPORT_CASES)
@pytest.mark.parametrize(
    ("secret", "echo", "sensitive_fragment"),
    [
        (
            "-----BEGIN PRIVATE KEY-----\r\nworkload-canary-ABCDEFGHIJKLMN\r\n-----END PRIVATE KEY-----",
            "-----BEGIN PRIVATE KEY-----\r\nworkload-canary-ABCDEFGHIJKLMN\r\n-----END PRIVATE KEY-----",
            "workload-canary-ABCDEFGHIJKLMN",
        ),
        ("  padded-workload-canary  ", "  padded-workload-canary  ", "padded-workload-canary"),
        (
            f"workload-prefix-{_API_SECRET}-suffix",
            f"workload-prefix-{_API_SECRET}-suffix",
            "workload-prefix",
        ),
        ("normalized workload-canary", "normalized\x00workload-canary", "workload-canary"),
    ],
    ids=["crlf", "edge-whitespace", "overlapping-credentials", "normalization-join"],
)
def test_runtime_redacts_workload_secrets_before_and_after_formatting(
    protocol: str, response_mode: str, secret: str, echo: str, sensitive_fragment: str
) -> None:
    outcome = asyncio.run(
        _run_provider_error(protocol, response_mode, message=echo, workload_secret=secret)
    )

    model_error = next(event for event in outcome.events if event.type == EventType.MODEL_ERROR)
    assert model_error.payload["error"].endswith(REDACTED_SECRET)
    assert outcome.error == model_error.payload["error"]
    rendered = json.dumps([event.payload for event in outcome.events], default=str)
    assert sensitive_fragment not in rendered


@pytest.mark.parametrize(("protocol", "response_mode"), _ERROR_TRANSPORT_CASES)
def test_runtime_drops_secret_identifiers_after_whitespace_normalization(
    protocol: str, response_mode: str
) -> None:
    secret = "  padded-identity-canary-0123456789  "
    outcome = asyncio.run(
        _run_provider_error(
            protocol,
            response_mode,
            message="Request rejected",
            workload_secret=secret,
            identity_secret=secret,
        )
    )

    model_error = next(event for event in outcome.events if event.type == EventType.MODEL_ERROR)
    assert model_error.payload["error"].endswith("Request rejected")
    assert model_error.payload["provider_error_type"] == "invalid_request_error"
    assert "provider_error_code" not in model_error.payload
    assert "request_id" not in model_error.payload
    rendered = json.dumps([event.payload for event in outcome.events], default=str)
    assert secret.strip() not in rendered


@pytest.mark.parametrize("protocol", ["chat", "openai", "anthropic", "vertex"])
@pytest.mark.parametrize("response_mode", ["buffered-http", "streamed-http"])
@pytest.mark.parametrize("location", ["x-request-id", "request-id", "top-level", "nested"])
def test_runtime_preserves_http_correlation_ids(
    protocol: str, response_mode: str, location: str
) -> None:
    outcome = asyncio.run(
        _run_provider_error(
            protocol, response_mode, message="Request rejected", request_id_location=location
        )
    )
    error = next(event for event in outcome.events if event.type == EventType.MODEL_ERROR)
    assert error.payload["request_id"] == "req_correlation_123"
    assert error.payload["error"].endswith("Request rejected")


@pytest.mark.parametrize("protocol", ["chat", "openai", "anthropic", "vertex"])
@pytest.mark.parametrize("header", ["x-request-id", "request-id"])
def test_runtime_preserves_sse_header_correlation_ids(protocol: str, header: str) -> None:
    outcome = asyncio.run(
        _run_provider_error(protocol, "sse", message="Request rejected", request_id_location=header)
    )
    error = next(event for event in outcome.events if event.type == EventType.MODEL_ERROR)
    assert error.payload["request_id"] == "req_correlation_123"
    assert error.payload["error"].endswith("Request rejected")


def test_anthropic_error_parser_accepts_builtin_exception_factory() -> None:
    raw_event = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "private-provider-body-canary"},
    }

    async def events() -> AsyncIterator[dict[str, Any]]:
        yield raw_event

    async def run() -> None:
        with pytest.raises(RuntimeError, match="provider response body omitted") as raised:
            async for _ in anthropic_stream_events(
                events(), api_error=lambda message, **kwargs: RuntimeError(message)
            ):
                pass
        assert type(raised.value) is RuntimeError
        assert_cayu_traceback_does_not_retain(raised.value, raw_event)

    asyncio.run(run())


def test_http_errors_accept_builtin_exception_factory() -> None:
    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(400, json={"error": {"message": "private"}})
            )
        ) as client:
            with pytest.raises(RuntimeError, match="provider response body omitted") as raised:
                await post_json(
                    client=client,
                    url="https://provider.invalid",
                    headers={},
                    payload={},
                    timeout_s=1,
                    request_label="Custom API",
                    response_label="Custom",
                    api_error=lambda message, **kwargs: RuntimeError(message),
                    protocol_error=RuntimeError,
                    error_response_text=lambda response: OMITTED_PROVIDER_ERROR_BODY,
                )
            assert type(raised.value) is RuntimeError
            assert "private" not in str(raised.value)

    asyncio.run(run())


@pytest.mark.parametrize("seam", ["observe", "context-policy"])
def test_runtime_count_failures_use_workload_redaction_before_formatting(seam: str) -> None:
    secret = "-----BEGIN PRIVATE KEY-----\r\ncount-workload-canary-123\r\n-----END PRIVATE KEY-----"
    observed: list[str] = []

    class CountingContextPolicy(ContextPolicy):
        async def build(self, request: ContextRequest) -> list[Message]:
            assert request.count_input_tokens is not None
            with pytest.raises(ModelProviderError) as raised:
                await request.count_input_tokens(request.messages)
            observed.append(str(raised.value))
            return request.messages

    outcome = asyncio.run(
        _run_provider_error(
            "openai",
            "buffered-http",
            message=f"Invalid request: {secret}",
            workload_secret=secret,
            context_counting=seam == "observe",
            context_policy=CountingContextPolicy() if seam == "context-policy" else None,
        )
    )
    if seam == "observe":
        count_error = next(
            event for event in outcome.events if event.type == EventType.CONTEXT_COUNT_FAILED
        )
        observed.append(count_error.payload["error"])
    assert observed
    assert all(message.endswith(f"Invalid request: {REDACTED_SECRET}") for message in observed)
    rendered = json.dumps([event.payload for event in outcome.events], default=str)
    assert "count-workload-canary" not in rendered


@pytest.mark.parametrize("text", ["secret\r\ncanary", "  padded-canary  "])
def test_unbound_provider_errors_remain_safe_for_later_workload_redaction(text: str) -> None:
    public = credential_safe_provider_exception(
        _attached_failure(text),
        provider_label="OpenAI",
        provider_name="openai",
        credential_values=("test-key",),
    )
    final = SecretRedactor([text]).redact_text(str(public))
    assert "canary" not in final
    if "\r" in text:
        assert final == "ModelProviderError: OpenAI provider failed"
    else:
        assert final.endswith(REDACTED_SECRET)


@pytest.mark.parametrize("header", ["x-request-id", "request-id"])
@pytest.mark.parametrize("mode", ["final-json", "background-sse", "background-error"])
def test_runtime_preserves_header_correlation_ids_in_additional_transports(
    header: str, mode: str
) -> None:
    outcome = asyncio.run(
        _run_provider_error("openai", mode, message="Request rejected", request_id_location=header)
    )
    error = next(event for event in outcome.events if event.type == EventType.MODEL_ERROR)
    assert error.payload["request_id"] == "req_correlation_123"
    assert error.payload["error"].endswith("Request rejected")


@pytest.mark.parametrize("mode", ["background-sse", "background-error"])
def test_runtime_redacts_background_header_credentials(mode: str) -> None:
    outcome = asyncio.run(
        _run_provider_error(
            "openai",
            mode,
            message="Request rejected",
            request_id_location="x-request-id",
            header_request_id=_API_SECRET,
        )
    )
    error = next(event for event in outcome.events if event.type == EventType.MODEL_ERROR)
    assert "request_id" not in error.payload
    assert error.payload["error"].endswith("Request rejected")
    assert _API_SECRET not in json.dumps([event.payload for event in outcome.events], default=str)
