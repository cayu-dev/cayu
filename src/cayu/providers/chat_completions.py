from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import unquote_plus, urlencode, urlsplit, urlunsplit

from cayu._validation import (
    copy_durable_json_value,
    copy_json_value,
    require_clean_nonblank,
    require_finite,
)
from cayu.artifacts import (
    FileAttachmentKind,
    file_attachment_from_payload,
    resolved_file_attachments_from_options,
)
from cayu.core.messages import (
    CitationPart,
    FilePart,
    HostedToolCallPart,
    Message,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
)
from cayu.providers._api_keys import resolve_api_key
from cayu.providers._credential_boundary import (
    aclosing_provider_stream,
    detach_provider_stream_traceback,
)
from cayu.providers._http import (
    OMITTED_PROVIDER_ERROR_BODY,
    SharedAsyncClient,
    _trusted_sse_retry_after_s,
    aclose_transport,
    copy_headers,
    credential_safe_error_event,
    credential_safe_post_completion_failure,
    credential_safe_provider_exception,
    credential_sanitization_values,
    optional_error_string,
    response_json_object,
    safe_error_json,
    safe_error_response_text,
    sanitize_provider_cancellation,
    stream_sse_json_events,
    validate_url,
)
from cayu.providers.base import (
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    UsageDialect,
    _preflight_provider_portable_messages,
    copy_usage_dialect,
    privacy_safe_provider_option_projection,
)

if TYPE_CHECKING:
    import httpx

# base_url follows the OpenAI-SDK convention: it includes the version path, and
# the endpoint appends only "/chat/completions". So OpenAI is ".../v1", Gemini is
# ".../v1beta/openai", Together is ".../v1", Azure is ".../deployments/<dep>".
DEFAULT_CHAT_COMPLETIONS_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CHAT_COMPLETIONS_TIMEOUT_SECONDS = 60.0
DEFAULT_CHAT_COMPLETIONS_STREAM_IDLE_TIMEOUT_SECONDS = 120.0
DEFAULT_CHAT_COMPLETIONS_API_KEY_ENV = "OPENAI_API_KEY"
# OpenAI/Together use `Authorization: Bearer <key>`; Azure uses `api-key: <key>`.
DEFAULT_CHAT_COMPLETIONS_AUTH_HEADER = "Authorization"
DEFAULT_CHAT_COMPLETIONS_AUTH_VALUE_PREFIX = "Bearer "
_USAGE_DIALECT_UNDECLARED = object()

_RESERVED_CHAT_COMPLETIONS_OPTIONS = {
    "model",
    "messages",
    "tools",
    "stream",
    "stream_options",
}
_CHAT_COMPLETIONS_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# JSON Schema keys rejected broadly enough by OpenAI-compatible vendors to strip
# whenever schema cleaning is enabled. Gemini additionally rejects
# ``additionalProperties``; endpoint detection or an explicit compatibility
# option selects that projection so other providers retain closed schemas.
_UNSUPPORTED_SCHEMA_KEYS = {"$schema"}
# JSON Schema keys whose values are name->subschema maps (arbitrary property
# names, not schema keywords), so their keys must be preserved when cleaning.
_SUBSCHEMA_MAP_KEYS = {"properties", "patternProperties", "$defs", "definitions"}
# How PDF/document attachments are encoded as a content part. OpenAI/Azure expect
# the `file` part; Google Gemini's compatible endpoint rejects `file` and instead
# accepts a PDF data URL through the `image_url` part. There is no single portable
# shape, so this is selectable per provider instance.
DEFAULT_DOCUMENT_ENCODING = "file"
_VALID_DOCUMENT_ENCODINGS = {"file", "image_url"}
_MAX_ROUTER_ATTEMPT = 1_000_000
_MAX_ROUTER_EVIDENCE_BYTES = 256
_CHAT_COMPLETIONS_STATE_PROTOCOL = "openai-chat-completions"
_CHAT_COMPLETIONS_STATE_PROTOCOL_VERSION = 1
_CHAT_COMPLETIONS_STATE_TARGET_VERSION = 1
_OPENROUTER_ROUTER_STRATEGIES = frozenset(
    {
        "alias",
        "auto",
        "bodybuilder",
        "direct",
        "fallback",
        "free",
        "fusion",
        "latest",
        "pareto",
    }
)
# Snapshot of public OpenRouter provider identities on 2026-08-22, plus
# documented display aliases. Future identities are retained only as a
# domain-separated digest until this allowlist is updated.
_OPENROUTER_UPSTREAM_PROVIDER_IDENTITIES: dict[str, str] = {
    value.casefold(): value
    for value in (
        "AI21",
        "AionLabs",
        "AkashML",
        "Alibaba",
        "Ambient",
        "Amazon Bedrock",
        "Amazon Nova",
        "Anthropic",
        "Arcee AI",
        "AtlasCloud",
        "AWS Bedrock",
        "Azure",
        "Baidu",
        "BaseTen",
        "Black Forest Labs",
        "Cerebras",
        "Chutes",
        "Cirrascale",
        "Clarifai",
        "Cloudflare",
        "Cohere",
        "CoreWeave",
        "Crucible",
        "Crusoe",
        "Darkbloom",
        "Databricks",
        "Decart",
        "Deepgram",
        "DeepInfra",
        "DeepSeek",
        "DekaLLM",
        "DigitalOcean",
        "Featherless",
        "Fireworks",
        "Fish Audio",
        "Friendli",
        "GMICloud",
        "Google",
        "Google AI Studio",
        "Google Vertex",
        "Groq",
        "HeyGen",
        "Hyperbolic",
        "Inception",
        "Inceptron",
        "InferenceNet",
        "Inferact vLLM",
        "Infermatic",
        "Inflection",
        "Ionstream",
        "Io Net",
        "Krea",
        "Liquid",
        "Makora",
        "Mancer 2",
        "Mara",
        "Meta",
        "Minimax",
        "Mistral",
        "Modal",
        "ModelRun",
        "Modular",
        "Moonshot AI",
        "Morph",
        "NCompass",
        "Nebius",
        "NextBit",
        "Nex AGI",
        "Nvidia",
        "Novita",
        "NovitaAI",
        "NVIDIA",
        "OpenAI",
        "OpenInference",
        "Parasail",
        "Perceptron",
        "Perplexity",
        "Phala",
        "Poolside",
        "Quiver",
        "Recraft",
        "Reka",
        "Relace",
        "Runway",
        "Sail Research",
        "Sakana AI",
        "SambaNova",
        "Seed",
        "SiliconFlow",
        "Sourceful",
        "Stealth",
        "StepFun",
        "StreamLake",
        "Switchpoint",
        "Tencent",
        "Tencent Cloud",
        "Tenstorrent",
        "Thinking Machines",
        "Together",
        "Upstage",
        "Venice",
        "Vertex AI",
        "VoyageAI by MongoDB",
        "Wafer",
        "Xiaomi",
        "xAI",
        "Z.AI",
    )
}

_TOOL_RESULT_ATTACHMENT_LEAD_IN = "The previous tool result returned file content for inspection."


class ChatCompletionsError(RuntimeError):
    """Base error for Chat Completions provider failures."""


class ChatCompletionsAPIError(ChatCompletionsError, ModelProviderError):
    """Raised when the Chat Completions HTTP API returns an error response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        retryable: bool | None = None,
        retry_after_s: float | None = None,
        response_body: str | None = None,
    ) -> None:
        ModelProviderError.__init__(
            self,
            message,
            provider="chat_completions",
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
            retryable=retryable,
            retry_after_s=retry_after_s,
            response_body=response_body,
        )


class ChatCompletionsContextOverflowError(
    ChatCompletionsAPIError,
    ModelContextOverflowError,
):
    """Raised when a Chat Completions provider reports context overflow."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        response_body: str | None = None,
    ) -> None:
        ModelContextOverflowError.__init__(
            self,
            message,
            provider="chat_completions",
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
            response_body=response_body,
        )


class ChatCompletionsProtocolError(ChatCompletionsError):
    """Raised when data does not match the expected Chat Completions shape."""


class ChatCompletionsTransport(Protocol):
    def stream_chat_completions(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """POST a streaming Chat Completions payload and yield decoded SSE data objects."""


class HttpxChatCompletionsTransport:
    """HTTP transport with explicit certifi-backed TLS verification.

    Owns one shared httpx.AsyncClient (created lazily) that is reused across
    requests so each model call does not pay for a fresh TLS handshake. Close it
    with :meth:`aclose` when the transport is no longer needed.
    """

    def __init__(self, *, allow_http: bool = False) -> None:
        if type(allow_http) is not bool:
            raise TypeError("allow_http must be a bool.")
        self.allow_http = allow_http
        self._client = SharedAsyncClient()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream_chat_completions(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        timeout_s = _validate_timeout_s(timeout_s)
        url = _validate_url(url, "url", allow_http=self.allow_http)
        events = stream_sse_json_events(
            client=self._client.get(),
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
            stream_idle_timeout_s=stream_idle_timeout_s,
            request_label="Chat Completions API",
            response_label="Chat Completions",
            api_error=ChatCompletionsAPIError,
            protocol_error=ChatCompletionsProtocolError,
            error_response_text=_safe_error_response_text,
            raise_context_overflow=_raise_chat_context_overflow_if_applicable,
            api_error_from_response=_chat_api_error_from_response,
        )
        async with aclosing_provider_stream(events):
            async for event in events:
                yield event


class ChatCompletionsProvider(ModelProvider):
    """Adapter for OpenAI-compatible ``/v1/chat/completions`` services.

    Many providers expose the OpenAI Chat Completions wire format without the
    newer Responses API: Google Gemini (AI Studio), Azure OpenAI, Together,
    Fireworks, Mistral, Ollama, vLLM, and others. This single adapter targets
    that shared format so those providers work through Cayu's provider-neutral
    runtime. ``OpenAIProvider`` remains the adapter for OpenAI's Responses API.

    The model is resolved from the agent's ``AgentSpec`` (and ``ModelRequest``),
    not from this provider, matching ``OpenAIProvider``/``AnthropicProvider``.
    """

    name = "openai_chat"
    usage_dialect = UsageDialect.OPENAI

    def preflight_portable_messages(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> None:
        _preflight_provider_portable_messages(
            model=model,
            messages=messages,
            tools=tools,
            supports_system_messages=True,
            supports_tool_history=True,
            supports_tool_definitions=True,
            supports_file_attachments=True,
            tool_name_validator=_validate_chat_completions_tool_name,
            tool_definition_validator=lambda tool: _chat_completions_tool(
                tool,
                clean_schemas=self.clean_schemas,
                strip_additional_properties=self.strip_additional_properties,
            ),
        )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        name: str = "openai_chat",
        base_url: str = DEFAULT_CHAT_COMPLETIONS_BASE_URL,
        endpoint_url: str | None = None,
        api_key_env: str = DEFAULT_CHAT_COMPLETIONS_API_KEY_ENV,
        auth_header: str = DEFAULT_CHAT_COMPLETIONS_AUTH_HEADER,
        auth_value_prefix: str = DEFAULT_CHAT_COMPLETIONS_AUTH_VALUE_PREFIX,
        allow_http: bool = False,
        stream_include_usage: bool = True,
        timeout_s: float = DEFAULT_CHAT_COMPLETIONS_TIMEOUT_SECONDS,
        stream_idle_timeout_s: float = DEFAULT_CHAT_COMPLETIONS_STREAM_IDLE_TIMEOUT_SECONDS,
        transport: ChatCompletionsTransport | None = None,
        extra_headers: Mapping[str, str] | None = None,
        openrouter_http_referer: str | None = None,
        openrouter_app_title: str | None = None,
        openrouter_router_metadata: bool = False,
        api_version: str | None = None,
        clean_schemas: bool = True,
        strip_additional_properties: bool | None = None,
        document_encoding: str = DEFAULT_DOCUMENT_ENCODING,
        usage_dialect: UsageDialect | str | None = None,
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.api_key_env = require_clean_nonblank(api_key_env, "api_key_env")
        self.api_key = resolve_api_key(
            api_key=api_key,
            env_var=self.api_key_env,
            provider_name="ChatCompletionsProvider",
            missing_hint=(
                f"set the {self.api_key_env} environment variable or pass api_key=... "
                "to ChatCompletionsProvider(...)."
            ),
        )
        # Auth header is configurable: OpenAI/Together use Authorization: Bearer,
        # Azure uses an `api-key` header (empty prefix).
        self.auth_header = require_clean_nonblank(auth_header, "auth_header")
        if type(auth_value_prefix) is not str:
            raise TypeError("auth_value_prefix must be a string.")
        self.auth_value_prefix = auth_value_prefix
        if type(allow_http) is not bool:
            raise TypeError("allow_http must be a bool.")
        self.allow_http = allow_http
        self.base_url = _validate_base_url(base_url, allow_http=allow_http)
        self.endpoint_url = (
            _validate_url(endpoint_url, "endpoint_url", allow_http=allow_http)
            if endpoint_url is not None
            else None
        )
        effective_url = self.endpoint_url or self.base_url
        if usage_dialect is not None:
            self.usage_dialect = copy_usage_dialect(usage_dialect)
        else:
            declared_usage_dialect = _declared_subclass_usage_dialect(type(self))
            if declared_usage_dialect is _USAGE_DIALECT_UNDECLARED:
                self.usage_dialect = (
                    UsageDialect.GEMINI
                    if urlsplit(effective_url).hostname == "generativelanguage.googleapis.com"
                    else UsageDialect.OPENAI
                )
            else:
                self.usage_dialect = copy_usage_dialect(
                    declared_usage_dialect,
                    f"{type(self).__name__}.usage_dialect",
                )
        self.timeout_s = _validate_timeout_s(timeout_s)
        if type(stream_idle_timeout_s) not in {int, float}:
            raise TypeError("stream_idle_timeout_s must be a number.")
        stream_idle_timeout_s = require_finite(
            float(stream_idle_timeout_s), "stream_idle_timeout_s"
        )
        if stream_idle_timeout_s <= 0:
            raise ValueError("stream_idle_timeout_s must be greater than zero.")
        self.stream_idle_timeout_s = stream_idle_timeout_s
        # A caller-supplied transport manages its own scheme policy; the default
        # transport inherits allow_http so a local http endpoint actually connects.
        self.transport = (
            transport
            if transport is not None
            else HttpxChatCompletionsTransport(allow_http=allow_http)
        )
        if type(openrouter_router_metadata) is not bool:
            raise TypeError("openrouter_router_metadata must be a bool.")
        self.openrouter_http_referer = (
            None
            if openrouter_http_referer is None
            else require_clean_nonblank(openrouter_http_referer, "openrouter_http_referer")
        )
        self.openrouter_app_title = (
            None
            if openrouter_app_title is None
            else require_clean_nonblank(openrouter_app_title, "openrouter_app_title")
        )
        self.openrouter_router_metadata = openrouter_router_metadata
        # Protect the headers we set (content-type, auth, and explicit OpenRouter
        # controls) from being clobbered by arbitrary extra_headers.
        protected_headers = {"content-type", self.auth_header.lower()}
        if self.openrouter_http_referer is not None:
            protected_headers.add("http-referer")
        if self.openrouter_app_title is not None:
            protected_headers.add("x-openrouter-title")
        if self.openrouter_router_metadata:
            protected_headers.add("x-openrouter-metadata")
        self.extra_headers = copy_headers(extra_headers, protected=protected_headers)
        if api_version is not None and not require_clean_nonblank(api_version, "api_version"):
            raise ValueError("api_version must be a nonblank string.")
        self.api_version = api_version
        if type(stream_include_usage) is not bool:
            raise TypeError("stream_include_usage must be a bool.")
        self.stream_include_usage = stream_include_usage
        if type(clean_schemas) is not bool:
            raise TypeError("clean_schemas must be a bool.")
        self.clean_schemas = clean_schemas
        if (
            strip_additional_properties is not None
            and type(strip_additional_properties) is not bool
        ):
            raise TypeError("strip_additional_properties must be a bool or None.")
        self.strip_additional_properties = (
            urlsplit(effective_url).hostname == "generativelanguage.googleapis.com"
            if strip_additional_properties is None
            else strip_additional_properties
        )
        self.document_encoding = _validate_document_encoding(document_encoding)

    def request_footprint_options(self, request: ModelRequest) -> dict[str, Any]:
        effective_options = _effective_chat_completions_request_options(
            request.options,
            options_key=self.name,
        )
        projected = privacy_safe_provider_option_projection(effective_options)
        return {self.name: projected} if projected else {}

    def request_fingerprint_options(self, request: ModelRequest) -> dict[str, Any]:
        effective = _effective_chat_completions_request_options(
            request.options,
            options_key=self.name,
        )
        return {self.name: effective} if effective else {}

    async def aclose(self) -> None:
        """Close the transport's shared HTTP client, if it owns one."""
        await aclose_transport(self.transport)

    @detach_provider_stream_traceback
    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        cancellation: asyncio.CancelledError | None = None
        overflow_failure: ChatCompletionsContextOverflowError | None = None
        post_completion_failure: ModelProviderError | None = None
        error_event: ModelStreamEvent | None = None
        completion_emitted = False
        try:
            endpoint = self._endpoint()
            provider_state_target_sha256 = _chat_completions_state_target_sha256(
                provider_name=self.name,
                endpoint_url=endpoint,
                model=request.model,
            )
            payload = build_chat_completions_payload(
                request,
                stream=True,
                clean_schemas=self.clean_schemas,
                strip_additional_properties=self.strip_additional_properties,
                options_key=self.name,
                document_encoding=self.document_encoding,
                include_usage=self.stream_include_usage,
                provider_state_target_sha256=provider_state_target_sha256,
            )
            raw_events = self.transport.stream_chat_completions(
                url=endpoint,
                headers=self._headers(),
                payload=payload,
                timeout_s=self.timeout_s,
                stream_idle_timeout_s=self.stream_idle_timeout_s,
            )
            events = chat_completions_stream_events(
                raw_events,
                requested_model=request.model,
                provider_state_target_sha256=provider_state_target_sha256,
            )
            async with aclosing_provider_stream(raw_events), aclosing_provider_stream(events):
                # Chat completion is synthesized only after the raw stream
                # terminates. Exhaust the translator so a deferred transport-
                # cleanup failure reaches runtime after the completion event.
                async for event in events:
                    if event.type == ModelStreamEventType.COMPLETED:
                        completion_emitted = True
                    yield event
        except asyncio.CancelledError as exc:
            cancellation = sanitize_provider_cancellation(
                exc,
                provider_label="Chat Completions",
                credential_values=credential_sanitization_values(
                    self.api_key,
                    extra_headers=self.extra_headers,
                ),
            )
        except ModelContextOverflowError as exc:
            credential_values = credential_sanitization_values(
                self.api_key,
                extra_headers=self.extra_headers,
            )
            if completion_emitted:
                post_completion_failure = credential_safe_post_completion_failure(
                    exc,
                    provider_label="Chat Completions",
                    provider_name="chat_completions",
                    credential_values=credential_values,
                )
            else:
                # Overflow must reach runtime recovery as a typed exception; an
                # error event would flatten it into unrecoverable message text.
                safe = credential_safe_provider_exception(
                    exc,
                    provider_label="Chat Completions",
                    provider_name="chat_completions",
                    credential_values=credential_values,
                )
                overflow_failure = ChatCompletionsContextOverflowError(
                    str(safe),
                    status_code=safe.status_code,
                    error_type=safe.error_type,
                    error_code=safe.error_code,
                    request_id=safe.request_id,
                    response_body=None,
                )
        except Exception as exc:
            credential_values = credential_sanitization_values(
                self.api_key,
                extra_headers=self.extra_headers,
            )
            if completion_emitted:
                post_completion_failure = credential_safe_post_completion_failure(
                    exc,
                    provider_label="Chat Completions",
                    provider_name="chat_completions",
                    credential_values=credential_values,
                )
            else:
                error_event = credential_safe_error_event(
                    exc,
                    provider_label="Chat Completions",
                    provider_name="chat_completions",
                    credential_values=credential_values,
                )
        if cancellation is not None:
            raise cancellation from None
        if overflow_failure is not None:
            raise overflow_failure from None
        if post_completion_failure is not None:
            raise post_completion_failure from None
        if error_event is not None:
            yield error_event

    def _endpoint(self) -> str:
        # OpenAI-SDK convention: base_url already carries the version path, so
        # append only "/chat/completions". `endpoint_url` is a full override.
        if self.endpoint_url is not None and self.api_version is None:
            return self.endpoint_url
        endpoint_override = self.endpoint_url is not None
        url = self.endpoint_url if self.endpoint_url is not None else self.base_url
        parts = urlsplit(url)
        path = parts.path
        if not endpoint_override:
            path = f"{path.rstrip('/')}/chat/completions"
        query = parts.query
        if self.api_version is not None:
            query_parts = (
                []
                if not query
                else [
                    part
                    for part in query.split("&")
                    if unquote_plus(part.partition("=")[0]) != "api-version"
                ]
            )
            query_parts.append(urlencode({"api-version": self.api_version}))
            query = "&".join(query_parts)
        return urlunsplit(parts._replace(path=path, query=query))

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            self.auth_header: f"{self.auth_value_prefix}{self.api_key}",
        }
        if self.openrouter_http_referer is not None:
            headers["HTTP-Referer"] = self.openrouter_http_referer
        if self.openrouter_app_title is not None:
            headers["X-OpenRouter-Title"] = self.openrouter_app_title
        if self.openrouter_router_metadata:
            headers["X-OpenRouter-Metadata"] = "enabled"
        headers.update(self.extra_headers)
        return headers


def build_chat_completions_payload(
    request: ModelRequest,
    *,
    stream: bool = False,
    clean_schemas: bool = True,
    strip_additional_properties: bool = False,
    options_key: str = "openai",
    document_encoding: str = DEFAULT_DOCUMENT_ENCODING,
    include_usage: bool = True,
    provider_state_target_sha256: str | None = None,
) -> dict[str, Any]:
    if type(request) is not ModelRequest:
        raise TypeError("request must be a ModelRequest.")
    if type(clean_schemas) is not bool:
        raise TypeError("clean_schemas must be a bool.")
    if type(strip_additional_properties) is not bool:
        raise TypeError("strip_additional_properties must be a bool.")
    if type(include_usage) is not bool:
        raise TypeError("include_usage must be a bool.")
    provider_state_target_sha256 = _validate_provider_state_target_sha256(
        provider_state_target_sha256
    )
    document_encoding = _validate_document_encoding(document_encoding)

    options = _effective_chat_completions_request_options(
        request.options,
        options_key=options_key,
    )
    resolved_attachments = resolved_file_attachments_from_options(request.options)

    messages: list[dict[str, Any]] = []
    system_text = _system_text(request.messages)
    if system_text:
        messages.append({"role": "system", "content": system_text})
    for message in request.messages:
        messages.extend(
            _chat_completions_messages(
                message,
                resolved_attachments=resolved_attachments,
                document_encoding=document_encoding,
                provider_state_target_sha256=provider_state_target_sha256,
            )
        )
    if not messages:
        raise ValueError("Chat Completions requests require at least one message.")

    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
    }
    tools = [
        _chat_completions_tool(
            tool,
            clean_schemas=clean_schemas,
            strip_additional_properties=strip_additional_properties,
        )
        for tool in request.tools
    ]
    if tools:
        payload["tools"] = tools
    if stream:
        payload["stream"] = True
        # Some OpenAI-compatible servers reject stream_options; make it opt-out.
        if include_usage:
            payload["stream_options"] = {"include_usage": True}
    payload.update(options)
    return copy_json_value(payload, "chat_completions_payload")


def _chat_completions_reasoning_options(neutral: Mapping[str, Any]) -> dict[str, Any]:
    """Map the neutral ``options["thinking"]`` payload to Chat Completions request keys.

    The portable knob is ``reasoning_effort`` (low/medium/high), which OpenAI-compatible
    reasoning providers accept. There is no portable way to *disable* reasoning here (the
    ``reasoning_effort="none"`` value is backend-specific — Gemini/DeepSeek accept it,
    OpenAI/Azure reject it), and this generic adapter can't know the backend, so
    ``enabled=False`` is a no-op; pass a raw ``reasoning_effort`` via provider_options to
    target a backend that supports it. There is no portable token budget, so ``max_tokens``
    is not mapped.
    """
    if not neutral.get("enabled", True):
        return {}
    effort = neutral.get("effort")
    if effort is not None:
        return {"reasoning_effort": effort}
    return {}


def _apply_thinking_options(payload: dict[str, Any], neutral: Any) -> None:
    """Merge the mapped reasoning config into the payload (typed config wins)."""
    if not isinstance(neutral, Mapping):
        return
    payload.update(_chat_completions_reasoning_options(neutral))


def _chat_completions_state_target_sha256(
    *,
    provider_name: str,
    endpoint_url: str,
    model: str,
) -> str:
    """Bind opaque state to one exact Chat Completions protocol target."""

    material = {
        "endpoint_url": require_clean_nonblank(endpoint_url, "endpoint_url"),
        "model": require_clean_nonblank(model, "model"),
        "protocol": _CHAT_COMPLETIONS_STATE_PROTOCOL,
        "protocol_version": _CHAT_COMPLETIONS_STATE_PROTOCOL_VERSION,
        "provider_name": require_clean_nonblank(provider_name, "provider_name"),
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _provider_state_target(target_sha256: str) -> dict[str, Any]:
    return {
        "protocol": _CHAT_COMPLETIONS_STATE_PROTOCOL,
        "protocol_version": _CHAT_COMPLETIONS_STATE_PROTOCOL_VERSION,
        "version": _CHAT_COMPLETIONS_STATE_TARGET_VERSION,
        "sha256": target_sha256,
    }


def _validate_provider_state_target_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("provider_state_target_sha256 must be a lowercase SHA-256 value.")
    return value


def _provider_state_target_matches(state: Mapping[str, Any], target_sha256: str | None) -> bool:
    if target_sha256 is None:
        return False
    target = state.get("target")
    return isinstance(target, Mapping) and target == _provider_state_target(target_sha256)


def _router_evidence_string(value: Any) -> str | None:
    if type(value) is not str:
        return None
    normalized = value.strip()
    if not normalized or len(normalized.encode("utf-8")) > _MAX_ROUTER_EVIDENCE_BYTES:
        return None
    return normalized


def _router_evidence_sha256(value: Any, *, domain: str) -> str | None:
    normalized = _router_evidence_string(value)
    if normalized is None:
        return None
    encoded = f"cayu:openrouter:{domain}:v1\0{normalized}".encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_openrouter_provider_identity(value: Any) -> str | None:
    normalized = _router_evidence_string(value)
    if normalized is None:
        return None
    return _OPENROUTER_UPSTREAM_PROVIDER_IDENTITIES.get(normalized.casefold())


def _openrouter_provider_evidence(value: Any) -> tuple[str, str] | None:
    provider = _safe_openrouter_provider_identity(value)
    if provider is not None:
        return "provider", provider
    digest = _router_evidence_sha256(value, domain="provider")
    return None if digest is None else ("provider_sha256", digest)


def _matching_router_model_identity(value: Any, *, expected: str | None) -> str | None:
    normalized = _router_evidence_string(value)
    expected_normalized = _router_evidence_string(expected)
    if normalized is None or expected_normalized is None or normalized != expected_normalized:
        return None
    return normalized


def _safe_openrouter_metadata(
    value: Any,
    *,
    requested_model: str | None,
    effective_model: str | None,
) -> dict[str, Any] | None:
    """Project additive OpenRouter metadata onto a fixed, bounded allowlist.

    Router ``pipeline``, ``attempts``, ``params``, human summaries, and unknown
    future fields are deliberately omitted. They are free-form and can contain
    provider- or plugin-specific data that does not belong in public runtime
    evidence.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ChatCompletionsProtocolError("OpenRouter metadata must be an object.")
    projected: dict[str, Any] = {}
    requested = _matching_router_model_identity(
        value.get("requested"),
        expected=requested_model,
    )
    if requested is not None:
        projected["requested"] = requested
    strategy = _router_evidence_string(value.get("strategy"))
    if strategy in _OPENROUTER_ROUTER_STRATEGIES:
        projected["strategy"] = strategy
    elif strategy is not None:
        projected["strategy_sha256"] = _router_evidence_sha256(
            strategy,
            domain="strategy",
        )
    attempt = value.get("attempt")
    if type(attempt) is int and 0 <= attempt <= _MAX_ROUTER_ATTEMPT:
        projected["attempt"] = attempt
    is_byok = value.get("is_byok")
    if type(is_byok) is bool:
        projected["is_byok"] = is_byok

    endpoints = value.get("endpoints")
    if isinstance(endpoints, Mapping):
        total = endpoints.get("total")
        if type(total) is int and 0 <= total <= _MAX_ROUTER_ATTEMPT:
            projected["endpoint_total"] = total
        available = endpoints.get("available")
        if type(available) is list and len(available) <= 256:
            selected: dict[str, str] | None = None
            for endpoint in available:
                if not isinstance(endpoint, Mapping) or endpoint.get("selected") is not True:
                    continue
                provider_evidence = _openrouter_provider_evidence(endpoint.get("provider"))
                model = _matching_router_model_identity(
                    endpoint.get("model"),
                    expected=effective_model,
                )
                candidate: dict[str, str] = {}
                if provider_evidence is not None:
                    candidate[provider_evidence[0]] = provider_evidence[1]
                if model is not None:
                    candidate["model"] = model
                if not candidate or selected is not None:
                    selected = None
                    break
                selected = candidate
            if selected is not None:
                projected["selected_endpoint"] = selected
    return projected


async def chat_completions_stream_events(
    events: AsyncIterator[Mapping[str, Any]],
    *,
    requested_model: str | None = None,
    provider_state_target_sha256: str | None = None,
) -> AsyncIterator[ModelStreamEvent]:
    provider_state_target_sha256 = _validate_provider_state_target_sha256(
        provider_state_target_sha256
    )
    tool_calls = _ToolCallAccumulator()
    reasoning_details = _ReasoningDetailsAccumulator()
    response_id: str | None = None
    model: str | None = None
    upstream_provider_evidence: tuple[str, str] | None = None
    openrouter_metadata: dict[str, Any] | None = None
    choice_index: int | None = None
    finish_reason: str | None = None
    usage: Any = None
    post_terminal_failure: BaseException | None = None

    iterator = events.__aiter__()
    while True:
        try:
            event = await iterator.__anext__()
        except StopAsyncIteration:
            break
        except asyncio.CancelledError as exc:
            # Preserve real task cancellation without losing authoritative
            # completion evidence already carried by a finish reason. Runtime
            # publishes the completion before restoring the same cancellation.
            if finish_reason is None:
                raise
            post_terminal_failure = exc
            break
        except Exception as exc:
            # A finish reason is authoritative completion evidence. Preserve
            # the accumulated response and usage when the real HTTP iterator
            # fails while closing after that terminal chunk; the provider will
            # surface this failure only after runtime has observed COMPLETED.
            if finish_reason is None:
                raise
            post_terminal_failure = exc
            break
        if not isinstance(event, Mapping):
            raise ChatCompletionsProtocolError(
                "Chat Completions stream event must be a JSON object."
            )
        response_id = response_id or _optional_string(event, "id")
        model = model or _optional_string(event, "model")
        chunk_upstream_provider_evidence = _openrouter_provider_evidence(event.get("provider"))
        if chunk_upstream_provider_evidence is not None:
            if (
                upstream_provider_evidence is not None
                and chunk_upstream_provider_evidence != upstream_provider_evidence
            ):
                raise ChatCompletionsProtocolError(
                    "Chat Completions stream emitted conflicting upstream providers."
                )
            upstream_provider_evidence = chunk_upstream_provider_evidence
        chunk_router_metadata = _safe_openrouter_metadata(
            event.get("openrouter_metadata"),
            requested_model=requested_model,
            effective_model=model,
        )
        if chunk_router_metadata is not None:
            if openrouter_metadata is not None and chunk_router_metadata != openrouter_metadata:
                raise ChatCompletionsProtocolError(
                    "Chat Completions stream emitted conflicting OpenRouter metadata."
                )
            openrouter_metadata = chunk_router_metadata
        chunk_usage = event.get("usage")
        if chunk_usage is not None:
            if not isinstance(chunk_usage, Mapping):
                raise ChatCompletionsProtocolError("Chat Completions usage must be an object.")
            usage = chunk_usage

        # Some OpenAI-compatible servers report a fault after the stream opens by
        # emitting a data chunk carrying an ``error`` object instead of an HTTP
        # error. Such a chunk has no ``choices``; surfacing it here avoids the
        # misleading "ended before a finish_reason" protocol error it would
        # otherwise trigger downstream.
        error = event.get("error")
        if error is not None:
            failure = _stream_error_chunk_exception(
                error,
                request_id=_optional_string(event, "request_id"),
                retry_after_s=_trusted_sse_retry_after_s(event),
            )
            # The exported parser is itself a public exception boundary. Do
            # not retain the raw provider envelope in traceback frame locals.
            error = None
            event = {}
            del events
            raise failure from None

        choices = event.get("choices")
        if choices is None:
            continue
        if not isinstance(choices, list):
            raise ChatCompletionsProtocolError("Chat Completions choices must be a list.")
        if len(choices) > 1:
            raise ChatCompletionsProtocolError(
                "Chat Completions stream emitted multiple choices in one chunk."
            )
        if not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ChatCompletionsProtocolError("Chat Completions choice must be an object.")
        chunk_choice_index = choice.get("index")
        if chunk_choice_index is not None:
            if type(chunk_choice_index) is not int or chunk_choice_index < 0:
                raise ChatCompletionsProtocolError(
                    "Chat Completions choice index must be a non-negative integer."
                )
            if choice_index is not None and chunk_choice_index != choice_index:
                raise ChatCompletionsProtocolError(
                    "Chat Completions stream emitted conflicting choice indexes."
                )
            choice_index = chunk_choice_index
        delta = choice.get("delta")
        if delta is not None:
            if not isinstance(delta, Mapping):
                raise ChatCompletionsProtocolError("Chat Completions delta must be an object.")
            reasoning_details.record(delta.get("reasoning_details"))
            reasoning = delta.get("reasoning_content")
            if not (isinstance(reasoning, str) and reasoning):
                # Fall back to `reasoning` unless reasoning_content is a non-empty
                # string, so an empty/absent reasoning_content can't shadow it.
                reasoning = delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                # Display-only reasoning surfaced by OpenAI-compatible reasoning
                # providers (DeepSeek/OpenRouter); no round-trip state.
                yield ModelStreamEvent.thinking(reasoning)
            content = delta.get("content")
            if content is not None:
                if not isinstance(content, str):
                    raise ChatCompletionsProtocolError(
                        "Chat Completions delta content must be a string."
                    )
                if content:
                    yield ModelStreamEvent.text_delta(content)
            tool_calls.record(delta.get("tool_calls"))
        choice_finish = choice.get("finish_reason")
        if choice_finish is not None:
            if not isinstance(choice_finish, str):
                raise ChatCompletionsProtocolError(
                    "Chat Completions finish_reason must be a string."
                )
            if finish_reason is not None and _canonical_chat_finish_reason(
                choice_finish
            ) != _canonical_chat_finish_reason(finish_reason):
                raise ChatCompletionsProtocolError(
                    "Chat Completions stream emitted conflicting finish_reason values."
                )
            if finish_reason is None:
                finish_reason = choice_finish

    # Tool calls are emitted once, after the upstream stream, before Cayu's terminal
    # completed event. Deferring normalization lets trailing usage or repeated
    # identical finish metadata arrive without producing multiple terminal events.
    provider_state = [
        *reasoning_details.provider_state_items(
            target_sha256=provider_state_target_sha256,
        ),
        *tool_calls.provider_state_items(
            target_sha256=provider_state_target_sha256,
        ),
    ]
    if tool_calls.has_pending():
        for tool_call_event in tool_calls.events():
            yield tool_call_event
        # Gemini's OpenAI-compatible endpoint can return finish_reason="stop"
        # on a chunk that also carries tool_calls. Cayu's runtime needs the
        # provider-neutral completion reason to reflect the actual next step.
        if finish_reason in {"stop", "end_turn"}:
            finish_reason = "tool_calls"

    if finish_reason is None:
        raise ChatCompletionsProtocolError(
            "Chat Completions streaming response ended before a finish_reason."
        )

    completed_payload = {
        "id": response_id,
        "model": model,
        "finish_reason": finish_reason,
        "usage": copy_json_value(usage, "usage"),
    }
    if provider_state:
        completed_payload["provider_state"] = provider_state
    if upstream_provider_evidence is not None:
        upstream_key, upstream_value = upstream_provider_evidence
        completed_payload[f"upstream_{upstream_key}"] = upstream_value
    if openrouter_metadata is not None:
        completed_payload["openrouter_metadata"] = openrouter_metadata
    yield ModelStreamEvent.completed(completed_payload)
    if post_terminal_failure is not None:
        failure = post_terminal_failure
        post_terminal_failure = None
        event = {}
        del iterator, events
        raise failure from None


def _stream_error_chunk_exception(
    error: Any,
    *,
    request_id: str | None = None,
    retry_after_s: float | None = None,
) -> ChatCompletionsError:
    """Build the typed exception for a mid-stream ``{"error": ...}`` chunk.

    The error is surfaced as a context-overflow error when its code/type/message
    indicate one (so runtime recovery can see it), else as a plain API error.
    """
    error_mapping = error if isinstance(error, Mapping) else {}
    error_type, code, reported_status, identity_conflict = _chat_error_identity(error_mapping)
    message = optional_error_string(error_mapping.get("message"))
    request_id = optional_error_string(error_mapping.get("request_id")) or request_id
    safe_message = f"Chat Completions stream reported an error: {OMITTED_PROVIDER_ERROR_BODY}"
    if not identity_conflict and _is_chat_context_overflow(
        status_code=reported_status,
        error_type=error_type,
        code=code,
        message=message,
    ):
        return ChatCompletionsContextOverflowError(
            safe_message,
            status_code=reported_status,
            error_type=error_type,
            error_code=code,
            request_id=request_id,
            response_body=None,
        )
    status_code, retryable = _chat_retry_metadata(
        transport_status_code=reported_status,
        error_type=error_type,
        error_code=code,
    )
    if identity_conflict:
        retryable = False
    return ChatCompletionsAPIError(
        safe_message,
        status_code=status_code,
        error_type=error_type,
        error_code=code,
        request_id=request_id,
        retryable=retryable,
        retry_after_s=retry_after_s,
        response_body=None,
    )


_CHAT_ERROR_TYPE_CLASSIFICATION = {
    "authentication": (401, False),
    "authentication_error": (401, False),
    "content_policy_violation": (403, False),
    "context_length_exceeded": (400, False),
    "invalid_prompt": (400, False),
    "invalid_request": (400, False),
    "invalid_request_error": (400, False),
    "not_found": (404, False),
    "not_found_error": (404, False),
    "payment_required": (402, False),
    "permission_denied": (403, False),
    "permission_error": (403, False),
    "provider_overloaded": (503, True),
    "provider_unavailable": (502, True),
    "rate_limit_exceeded": (429, True),
    "rate_limit_error": (429, True),
    "server": (500, True),
    "server_error": (500, True),
    "timeout": (408, True),
}
_CHAT_ERROR_CODE_CLASSIFICATION = {
    "context_length_exceeded": (400, False),
    "internal_error": (500, True),
    "rate_limit_exceeded": (429, True),
    "server_error": (500, True),
}


def _chat_error_identity(
    error: Mapping[str, Any],
) -> tuple[str | None, str | None, int | None, bool]:
    """Extract OpenAI-compatible and OpenRouter typed error evidence.

    OpenRouter's Chat Completions skin carries its stable type in
    ``error.metadata.error_type``, the upstream code in ``provider_code``, and
    the HTTP identity as a numeric ``error.code`` even after streaming begins.
    Preserve those fields without letting contradictory aliases authorize a
    retry.
    """

    metadata_value = error.get("metadata")
    metadata = metadata_value if isinstance(metadata_value, Mapping) else {}
    canonical_types = [
        value
        for value in (
            optional_error_string(metadata.get("error_type")),
            optional_error_string(error.get("error_type")),
            optional_error_string(error.get("type")),
        )
        if value is not None
    ]
    identity_conflict = bool(
        canonical_types and any(value != canonical_types[0] for value in canonical_types[1:])
    )
    error_type = None if identity_conflict else (canonical_types[0] if canonical_types else None)
    provider_codes = [
        value
        for value in (
            optional_error_string(metadata.get("provider_code")),
            optional_error_string(error.get("provider_code")),
        )
        if value is not None
    ]
    if provider_codes and any(value != provider_codes[0] for value in provider_codes[1:]):
        identity_conflict = True
        provider_code = None
    else:
        provider_code = provider_codes[0] if provider_codes else None
    wire_code = error.get("code")
    string_code = optional_error_string(wire_code)
    error_code = provider_code or string_code

    reported_statuses = [
        value
        for value in (
            _optional_http_status(wire_code),
            _optional_http_status(error.get("http_status")),
        )
        if value is not None
    ]
    if reported_statuses and any(value != reported_statuses[0] for value in reported_statuses[1:]):
        identity_conflict = True
        reported_status = None
    else:
        reported_status = reported_statuses[0] if reported_statuses else None
    return error_type, error_code, reported_status, identity_conflict


def _optional_http_status(value: Any) -> int | None:
    if type(value) is int and 100 <= value <= 599:
        return value
    return None


def _chat_retry_metadata(
    *,
    transport_status_code: int | None,
    error_type: str | None,
    error_code: str | None,
) -> tuple[int | None, bool | None]:
    """Classify recognized HTTP/stream identities; conflicts fail closed."""
    classifications = {
        classification
        for classification in (
            _CHAT_ERROR_TYPE_CLASSIFICATION.get(error_type or ""),
            _CHAT_ERROR_CODE_CLASSIFICATION.get(error_code or ""),
        )
        if classification is not None
    }
    if not classifications:
        return transport_status_code, None
    if len(classifications) != 1:
        return transport_status_code, False
    canonical_status, retryable = next(iter(classifications))
    if transport_status_code is not None and transport_status_code != canonical_status:
        return transport_status_code, False
    return canonical_status, retryable


def _tool_call_names_function(tool_call: Mapping[str, Any]) -> bool:
    """Whether a streamed tool-call fragment carries a ``function.name``."""
    function = tool_call.get("function")
    if not isinstance(function, Mapping):
        return False
    return _optional_string(function, "name") is not None


def _canonical_chat_finish_reason(value: str) -> str:
    return "tool_calls" if value == "function_call" else value


class _ReasoningDetailsAccumulator:
    """Collect opaque streamed reasoning blocks in provider order for exact replay."""

    def __init__(self) -> None:
        self._seen = False
        self._details: list[dict[str, Any]] = []

    def record(self, value: Any) -> None:
        if value is None:
            return
        self._seen = True
        if type(value) is not list:
            raise ChatCompletionsProtocolError(
                "Chat Completions delta reasoning_details must be a list."
            )
        for detail in value:
            if type(detail) is not dict:
                raise ChatCompletionsProtocolError(
                    "Chat Completions reasoning_detail must be an object."
                )
            try:
                copied = copy_durable_json_value(detail, "reasoning_detail")
            except (TypeError, ValueError):
                raise ChatCompletionsProtocolError(
                    "Chat Completions reasoning_detail is not durable JSON."
                ) from None
            if type(copied) is not dict:
                raise ChatCompletionsProtocolError(
                    "Chat Completions reasoning_detail must be an object."
                )
            self._details.append(copied)

    def provider_state_items(self, *, target_sha256: str | None) -> list[dict[str, Any]]:
        if not self._seen:
            return []
        if target_sha256 is None:
            raise ChatCompletionsProtocolError(
                "Chat Completions reasoning_details have no exact replay target."
            )
        return [
            {
                "provider": "chat_completions",
                "state": {
                    "type": "reasoning_details",
                    "version": 1,
                    "target": _provider_state_target(target_sha256),
                    "details": copy_durable_json_value(self._details, "reasoning_details"),
                },
            }
        ]


class _PendingToolCall:
    def __init__(self) -> None:
        self.index: int | None = None
        self.call_id: str | None = None
        self.name: str | None = None
        self.arguments_parts: list[str] = []
        self.extra_content: dict[str, Any] | None = None

    @property
    def arguments(self) -> str:
        return "".join(self.arguments_parts)


class _ToolCallAccumulator:
    """Accumulates streamed tool-call fragments into ordered tool-call events.

    Providers correlate fragments differently. OpenAI puts an ``index`` on each
    ``tool_calls[]`` entry; Gemini's OpenAI-compatible endpoint omits it and
    sends the complete call (with an ``id``) in a single delta. Index and ID are
    aliases only after one fragment proves that they identify the same call.
    Keyless argument-only fragments retain the compatible most-recent fallback.
    """

    def __init__(self) -> None:
        self._pending: list[_PendingToolCall] = []
        self._by_index: dict[int, _PendingToolCall] = {}
        self._by_id: dict[str, _PendingToolCall] = {}
        self._last_pending: _PendingToolCall | None = None

    def record(self, tool_calls: Any) -> None:
        if tool_calls is None:
            return
        if not isinstance(tool_calls, list):
            raise ChatCompletionsProtocolError("Chat Completions delta tool_calls must be a list.")
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                raise ChatCompletionsProtocolError("Chat Completions tool_call must be an object.")
            call_id = _optional_string(tool_call, "id")
            pending = self._pending_for(tool_call, call_id)
            extra_content = tool_call.get("extra_content")
            if extra_content is not None:
                if not isinstance(extra_content, Mapping):
                    raise ChatCompletionsProtocolError(
                        "Chat Completions tool_call extra_content must be an object."
                    )
                pending.extra_content = copy_json_value(extra_content, "tool_call.extra_content")
            function = tool_call.get("function")
            if function is None:
                continue
            if not isinstance(function, Mapping):
                raise ChatCompletionsProtocolError(
                    "Chat Completions tool_call function must be an object."
                )
            name = _optional_string(function, "name")
            if name is not None:
                if pending.name is not None:
                    raise ChatCompletionsProtocolError(
                        "Chat Completions tool_call emitted more than one function name."
                    )
                pending.name = name
            arguments = function.get("arguments")
            if arguments is not None:
                if not isinstance(arguments, str):
                    raise ChatCompletionsProtocolError(
                        "Chat Completions tool_call arguments must be a string."
                    )
                pending.arguments_parts.append(arguments)

    def _pending_for(
        self,
        tool_call: Mapping[str, Any],
        call_id: str | None,
    ) -> _PendingToolCall:
        index = tool_call.get("index")
        if index is not None and (type(index) is not int or index < 0):
            raise ChatCompletionsProtocolError(
                "Chat Completions tool_call index must be a non-negative integer."
            )
        indexed = self._by_index.get(index) if index is not None else None
        identified = self._by_id.get(call_id) if call_id is not None else None
        if indexed is not None and identified is not None and indexed is not identified:
            raise ChatCompletionsProtocolError(
                "Chat Completions tool_call index and id identify different calls."
            )
        pending = indexed or identified
        if pending is None and (
            index is None
            and call_id is None
            and self._last_pending is not None
            and not _tool_call_names_function(tool_call)
        ):
            # A keyless fragment that names no function continues the most recent
            # call (providers stream arguments across chunks that carry only the
            # index-less function.arguments). One that *does* name a function is a
            # distinct call, so fall through to a fresh slot instead of merging it
            # into the previous call's arguments.
            pending = self._last_pending
        if pending is None:
            pending = _PendingToolCall()
            self._pending.append(pending)
        if index is not None:
            if pending.index is not None and pending.index != index:
                raise ChatCompletionsProtocolError("Chat Completions tool_call id changed indexes.")
            existing = self._by_index.get(index)
            if existing is not None and existing is not pending:
                raise ChatCompletionsProtocolError(
                    "Chat Completions tool_call index identifies multiple calls."
                )
            pending.index = index
            self._by_index[index] = pending
        if call_id is not None:
            if pending.call_id is not None and pending.call_id != call_id:
                raise ChatCompletionsProtocolError("Chat Completions tool_call index changed ids.")
            existing = self._by_id.get(call_id)
            if existing is not None and existing is not pending:
                raise ChatCompletionsProtocolError(
                    "Chat Completions tool_call id identifies multiple calls."
                )
            pending.call_id = call_id
            self._by_id[call_id] = pending
        self._last_pending = pending
        return pending

    def has_pending(self) -> bool:
        return bool(self._pending)

    def events(self) -> list[ModelStreamEvent]:
        tool_call_events: list[ModelStreamEvent] = []
        for position, pending in enumerate(self._pending):
            if pending.call_id is None or not pending.call_id.strip():
                raise ChatCompletionsProtocolError(
                    f"Chat Completions tool_call {position} is missing an id."
                )
            if pending.name is None or not pending.name.strip():
                raise ChatCompletionsProtocolError(
                    f"Chat Completions tool_call {position} is missing a name."
                )
            raw_arguments = pending.arguments or "{}"
            try:
                decoded_arguments = json.loads(raw_arguments)
            except ValueError as exc:
                raise ChatCompletionsProtocolError(
                    f"Chat Completions tool_call {position} arguments were not valid JSON."
                ) from exc
            if type(decoded_arguments) is not dict:
                raise ChatCompletionsProtocolError(
                    f"Chat Completions tool_call {position} arguments must decode to an object."
                )
            tool_call_events.append(
                ModelStreamEvent.tool_call(
                    id=pending.call_id,
                    name=pending.name,
                    arguments=copy_json_value(decoded_arguments, "arguments"),
                )
            )
        return tool_call_events

    def provider_state_items(self, *, target_sha256: str | None) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for pending in self._pending:
            if pending.extra_content is None:
                continue
            if pending.call_id is None or not pending.call_id.strip():
                raise ChatCompletionsProtocolError(
                    "Chat Completions tool_call with extra_content is missing an id."
                )
            if target_sha256 is None:
                raise ChatCompletionsProtocolError(
                    "Chat Completions tool_call extra_content has no exact replay target."
                )
            items.append(
                {
                    "provider": "chat_completions",
                    "state": {
                        "type": "tool_call_extra_content",
                        "version": 1,
                        "target": _provider_state_target(target_sha256),
                        "tool_call_id": pending.call_id,
                        "extra_content": copy_json_value(
                            pending.extra_content, "tool_call.extra_content"
                        ),
                    },
                }
            )
        return items


def _system_text(messages: list[Message]) -> str:
    system_parts: list[str] = []
    for message in messages:
        if message.role != MessageRole.SYSTEM:
            continue
        for part in message.content:
            if type(part) is TextPart:
                system_parts.append(part.text)
    return "\n\n".join(system_parts)


def _chat_completions_messages(
    message: Message,
    *,
    resolved_attachments: dict[str, dict[str, Any]],
    document_encoding: str,
    provider_state_target_sha256: str | None,
) -> list[dict[str, Any]]:
    if message.role == MessageRole.SYSTEM:
        return []
    if message.role == MessageRole.USER:
        return [_user_message(message.content, resolved_attachments, document_encoding)]
    if message.role == MessageRole.ASSISTANT:
        return [
            _assistant_message(
                message,
                provider_state_target_sha256=provider_state_target_sha256,
            )
        ]
    if message.role == MessageRole.TOOL:
        messages: list[dict[str, Any]] = []
        attachment_parts: list[dict[str, Any]] = []
        for part in message.content:
            if type(part) is not ToolResultPart:
                raise ChatCompletionsProtocolError(
                    "Tool messages can only contain tool_result parts."
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": part.tool_call_id,
                    "content": part.content,
                }
            )
            attachment_parts.extend(
                _file_attachment_parts(part, resolved_attachments, document_encoding)
            )
        if attachment_parts:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _TOOL_RESULT_ATTACHMENT_LEAD_IN},
                        *attachment_parts,
                    ],
                }
            )
        return messages
    raise ChatCompletionsProtocolError(f"Unsupported Cayu message role: {message.role!r}.")


def _assistant_message(
    message: Message,
    *,
    provider_state_target_sha256: str | None = None,
) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning_details = _reasoning_details(
        message,
        provider_state_target_sha256=provider_state_target_sha256,
    )
    tool_call_extra_content = _tool_call_extra_content_by_id(
        message,
        provider_state_target_sha256=provider_state_target_sha256,
    )
    for part in message.content:
        if type(part) is TextPart:
            text_parts.append(part.text)
        elif type(part) is ToolCallPart:
            tool_call = {
                "id": part.tool_call_id,
                "type": "function",
                "function": {
                    "name": part.tool_name,
                    "arguments": _json_arguments(part.arguments),
                },
            }
            extra_content = tool_call_extra_content.get(part.tool_call_id)
            if extra_content is not None:
                tool_call["extra_content"] = extra_content
            tool_calls.append(tool_call)
        elif type(part) not in {
            ProviderStatePart,
            ThinkingPart,
            HostedToolCallPart,
            CitationPart,
        }:
            raise ChatCompletionsProtocolError(
                "Assistant messages can only contain portable assistant parts."
            )
    assistant: dict[str, Any] = {"role": "assistant"}
    # Chat Completions requires a content key; tool-call-only turns use null.
    assistant["content"] = "\n".join(text_parts) or None
    if reasoning_details is not None:
        assistant["reasoning_details"] = reasoning_details
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    return assistant


def _reasoning_details(
    message: Message,
    *,
    provider_state_target_sha256: str | None,
) -> list[dict[str, Any]] | None:
    details: list[dict[str, Any]] | None = None
    for part in message.content:
        if type(part) is not ProviderStatePart or part.provider != "chat_completions":
            continue
        state = part.state
        if state.get("type") != "reasoning_details":
            continue
        if not _provider_state_target_matches(state, provider_state_target_sha256):
            continue
        if details is not None:
            raise ChatCompletionsProtocolError(
                "Chat Completions assistant message has duplicate reasoning_details state."
            )
        if state.get("version") != 1:
            raise ChatCompletionsProtocolError(
                "Chat Completions reasoning_details state has an unsupported version."
            )
        raw_details = state.get("details")
        if type(raw_details) is not list:
            raise ChatCompletionsProtocolError(
                "Chat Completions provider_state reasoning_details must be a list."
            )
        try:
            copied = copy_durable_json_value(
                raw_details,
                "provider_state.reasoning_details",
            )
        except (TypeError, ValueError):
            raise ChatCompletionsProtocolError(
                "Chat Completions provider_state reasoning_details is not durable JSON."
            ) from None
        if type(copied) is not list or any(type(item) is not dict for item in copied):
            raise ChatCompletionsProtocolError(
                "Chat Completions provider_state reasoning_details must contain objects."
            )
        details = copied
    return details


def _tool_call_extra_content_by_id(
    message: Message,
    *,
    provider_state_target_sha256: str | None,
) -> dict[str, dict[str, Any]]:
    extra_content_by_id: dict[str, dict[str, Any]] = {}
    for part in message.content:
        if type(part) is not ProviderStatePart or part.provider != "chat_completions":
            continue
        state = part.state
        if state.get("type") != "tool_call_extra_content":
            continue
        if not _provider_state_target_matches(state, provider_state_target_sha256):
            continue
        if state.get("version") != 1:
            raise ChatCompletionsProtocolError(
                "Chat Completions tool_call extra_content state has an unsupported version."
            )
        tool_call_id = state.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise ChatCompletionsProtocolError(
                "Chat Completions provider_state tool_call_id must be a non-empty string."
            )
        extra_content = state.get("extra_content")
        if not isinstance(extra_content, Mapping):
            raise ChatCompletionsProtocolError(
                "Chat Completions provider_state extra_content must be an object."
            )
        extra_content_by_id[tool_call_id] = copy_json_value(
            extra_content, "provider_state.extra_content"
        )
    return extra_content_by_id


def _user_message(
    content: tuple[
        TextPart
        | ToolCallPart
        | ToolResultPart
        | ProviderStatePart
        | ThinkingPart
        | FilePart
        | HostedToolCallPart
        | CitationPart,
        ...,
    ],
    resolved_attachments: dict[str, dict[str, Any]],
    document_encoding: str,
) -> dict[str, Any]:
    # Text-only turns keep the plain-string content shape for maximum vendor
    # compatibility; file parts require the content-part list form.
    if all(type(part) is TextPart for part in content):
        return {
            "role": "user",
            "content": "\n".join(part.text for part in content if type(part) is TextPart),
        }
    parts: list[dict[str, Any]] = []
    for part in content:
        if type(part) is TextPart:
            parts.append({"type": "text", "text": part.text})
            continue
        if type(part) is FilePart:
            parts.append(
                _file_attachment_part(
                    _resolved_user_attachment(part, resolved_attachments),
                    document_encoding,
                )
            )
            continue
        raise ChatCompletionsProtocolError("User messages can only contain text and file parts.")
    return {"role": "user", "content": parts}


def _resolved_user_attachment(
    part: FilePart,
    resolved_attachments: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    attachment = file_attachment_from_payload(part.attachment)
    if attachment is None:
        raise ChatCompletionsProtocolError("User file parts require a file attachment payload.")
    resolved = resolved_attachments.get(attachment.artifact_id)
    if resolved is None:
        raise ChatCompletionsProtocolError(
            f"Missing resolved file attachment: {attachment.artifact_id}"
        )
    return resolved


def _file_attachment_parts(
    part: ToolResultPart,
    resolved_attachments: dict[str, dict[str, Any]],
    document_encoding: str,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for payload in part.artifacts:
        attachment = file_attachment_from_payload(payload)
        if attachment is None:
            continue
        resolved = resolved_attachments.get(attachment.artifact_id)
        if resolved is None:
            raise ChatCompletionsProtocolError(
                f"Missing resolved file attachment: {attachment.artifact_id}"
            )
        parts.append(_file_attachment_part(resolved, document_encoding))
    return parts


def _file_attachment_part(resolved: dict[str, Any], document_encoding: str) -> dict[str, Any]:
    kind = FileAttachmentKind(resolved["kind"])
    data_url = f"data:{resolved['content_type']};base64,{resolved['data_base64']}"
    if kind == FileAttachmentKind.IMAGE:
        return {"type": "image_url", "image_url": {"url": data_url}}
    if kind == FileAttachmentKind.DOCUMENT:
        if document_encoding == "image_url":
            # Google Gemini's compatible endpoint carries PDFs through image_url.
            return {"type": "image_url", "image_url": {"url": data_url}}
        # OpenAI/Azure Chat Completions file-input content part. Vendors that do
        # not implement it reject it with a normal API error, like any other
        # unsupported feature.
        return {
            "type": "file",
            "file": {"filename": resolved["filename"], "file_data": data_url},
        }
    raise ChatCompletionsProtocolError(f"Unsupported file attachment kind: {kind!r}")


def _json_arguments(arguments: Mapping[str, Any]) -> str:
    copied = copy_json_value(arguments, "arguments")
    if type(copied) is not dict:
        raise ChatCompletionsProtocolError("Tool call arguments must be an object.")
    return json.dumps(copied, sort_keys=True, separators=(",", ":"))


def _chat_completions_tool(
    tool: Mapping[str, Any],
    *,
    clean_schemas: bool,
    strip_additional_properties: bool,
) -> dict[str, Any]:
    if not isinstance(tool, Mapping):
        raise ValueError("Tool definitions must be objects.")
    name = _require_mapping_string(tool, "name")
    _validate_chat_completions_tool_name(name)
    description = tool.get("description", "")
    if not isinstance(description, str):
        raise ValueError("Tool description must be a string.")
    input_schema = tool.get("input_schema", {})
    if type(input_schema) is not dict:
        raise ValueError("Tool input_schema must be an object.")
    # Both paths produce a fresh structure; the final whole-payload copy_json_value
    # re-validates JSON-safety, so a separate per-schema copy here would be redundant.
    parameters = (
        _clean_schema(
            input_schema,
            strip_additional_properties=strip_additional_properties,
        )
        if clean_schemas
        else copy_json_value(input_schema, "input_schema")
    )
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _validate_chat_completions_tool_name(name: str) -> None:
    if not _CHAT_COMPLETIONS_TOOL_NAME_RE.fullmatch(name):
        raise ValueError(
            "Chat Completions tool names must contain 1-64 letters, numbers, "
            "underscores, or hyphens."
        )


def _clean_schema(
    schema: Any,
    *,
    strip_additional_properties: bool,
    in_property_map: bool = False,
) -> Any:
    """Recursively strip JSON Schema keywords some vendors reject (e.g. Gemini).

    Keys in ``_UNSUPPORTED_SCHEMA_KEYS`` are dropped only where they are schema
    *keywords* (direct keys of a schema object). Inside name->subschema maps
    (``properties``, ``$defs``, ...) the keys are arbitrary names, so a property
    literally named ``additionalProperties`` is preserved and only its subschema
    value is cleaned. When ``strip_additional_properties`` is true, occurrences
    of that schema keyword are removed as a vendor-specific compatibility step.
    """
    if isinstance(schema, dict):
        if in_property_map:
            return {
                name: _clean_schema(
                    value,
                    strip_additional_properties=strip_additional_properties,
                )
                for name, value in schema.items()
            }
        return {
            key: _clean_schema(
                value,
                strip_additional_properties=strip_additional_properties,
                in_property_map=key in _SUBSCHEMA_MAP_KEYS,
            )
            for key, value in schema.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYS
            and not (strip_additional_properties and key == "additionalProperties")
        }
    if isinstance(schema, list):
        return [
            _clean_schema(
                item,
                strip_additional_properties=strip_additional_properties,
            )
            for item in schema
        ]
    return schema


def _chat_completions_options(options: Mapping[str, Any], options_key: str) -> dict[str, Any]:
    raw_options = options.get(options_key, {})
    if raw_options is None:
        return {}
    if type(raw_options) is not dict:
        raise ValueError(f"ModelRequest options.{options_key} must be an object.")
    copied = copy_json_value(raw_options, f"options.{options_key}")
    for key in copied:
        if key in _RESERVED_CHAT_COMPLETIONS_OPTIONS:
            raise ValueError(f"Chat Completions option is reserved: {key}")
    return copied


def _effective_chat_completions_request_options(
    options: Mapping[str, Any],
    *,
    options_key: str,
) -> dict[str, Any]:
    effective = _chat_completions_options(options, options_key)
    # Cayu models one provider response as one assistant step; n>1 would return
    # multiple `choices` that the stream loop cannot represent. Reject it.
    if "n" in effective and (type(effective["n"]) is not int or effective["n"] != 1):
        raise ValueError(
            "Chat Completions n must be 1 (multi-candidate responses are unsupported)."
        )
    _apply_thinking_options(effective, options.get("thinking"))
    return effective


def _require_mapping_string(value: Mapping[str, Any], key: str) -> str:
    raw_value = value.get(key)
    if not isinstance(raw_value, str):
        raise ValueError(f"Tool {key} must be a string.")
    return require_clean_nonblank(raw_value, f"tool.{key}")


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    raw_value = value.get(key)
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ChatCompletionsProtocolError(f"Chat Completions field {key} must be a string.")
    return raw_value


def _validate_document_encoding(value: object) -> str:
    if type(value) is not str or value not in _VALID_DOCUMENT_ENCODINGS:
        raise ValueError(f"document_encoding must be one of {sorted(_VALID_DOCUMENT_ENCODINGS)}.")
    return value


def _validate_timeout_s(value: float) -> float:
    if type(value) not in {int, float}:
        raise TypeError("timeout_s must be a number.")
    try:
        normalized = float(value)
    except OverflowError:
        raise ValueError("timeout_s must be finite and greater than zero.") from None
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("timeout_s must be finite and greater than zero.")
    return normalized


def _declared_subclass_usage_dialect(
    provider_type: type[ChatCompletionsProvider],
) -> object:
    for candidate in provider_type.__mro__:
        if candidate is ChatCompletionsProvider:
            break
        if "usage_dialect" in candidate.__dict__:
            return candidate.__dict__["usage_dialect"]
    return _USAGE_DIALECT_UNDECLARED


def _validate_base_url(base_url: str, *, allow_http: bool = False) -> str:
    validated = validate_url(
        base_url,
        "base_url",
        provider_label="Chat Completions",
        allow_http=allow_http,
        allow_http_hint=True,
    )
    parts = urlsplit(validated)
    return urlunsplit(parts._replace(path=parts.path.rstrip("/")))


def _validate_url(url: str, field_name: str, *, allow_http: bool = False) -> str:
    return validate_url(
        url,
        field_name,
        provider_label="Chat Completions",
        allow_http=allow_http,
        allow_http_hint=True,
    )


def _safe_error_response_text(response: httpx.Response) -> str:
    return safe_error_response_text(response, format_error_json=_format_error_json)


def _format_error_json(decoded: Any) -> str | None:
    if not isinstance(decoded, Mapping):
        return None
    return safe_error_json(decoded)


def _chat_api_error_from_response(
    response: httpx.Response,
    message: str,
    retry_after_s: float | None,
) -> ChatCompletionsAPIError:
    """Build a buffered error with the same typed identity rules as streaming."""

    decoded = response_json_object(response)
    error: Mapping[str, Any] = {}
    if decoded is not None:
        raw_error = decoded.get("error")
        error = raw_error if isinstance(raw_error, Mapping) else decoded
    error_type, error_code, reported_status, identity_conflict = _chat_error_identity(error)
    if reported_status is not None and reported_status != response.status_code:
        identity_conflict = True
    status_code, retryable = _chat_retry_metadata(
        transport_status_code=response.status_code,
        error_type=error_type,
        error_code=error_code,
    )
    if identity_conflict:
        retryable = False
    request_id = optional_error_string(error.get("request_id"))
    if request_id is None and decoded is not None:
        request_id = optional_error_string(decoded.get("request_id"))
    return ChatCompletionsAPIError(
        message,
        status_code=status_code,
        error_type=error_type,
        error_code=error_code,
        request_id=request_id,
        retryable=retryable,
        retry_after_s=retry_after_s,
        response_body=_safe_error_response_text(response),
    )


def _raise_chat_context_overflow_if_applicable(response: httpx.Response) -> None:
    decoded = response_json_object(response)
    if decoded is None:
        return
    error = decoded.get("error")
    if not isinstance(error, Mapping):
        error = decoded
    error_type, code, reported_status, identity_conflict = _chat_error_identity(error)
    if error_type is None and not identity_conflict:
        error_type = optional_error_string(error.get("status"))
    message = optional_error_string(error.get("message"))
    status_code = response.status_code
    if reported_status is not None and reported_status != status_code:
        identity_conflict = True
    if (
        not _is_chat_context_overflow(
            status_code=status_code,
            error_type=error_type,
            code=code,
            message=message,
        )
        or identity_conflict
    ):
        return
    raise ChatCompletionsContextOverflowError(
        "Chat Completions model context overflow",
        status_code=status_code,
        error_type=error_type,
        error_code=code,
        response_body=_safe_error_response_text(response),
    )


def _is_chat_context_overflow(
    *,
    status_code: int | None,
    error_type: str | None,
    code: str | None,
    message: str | None,
) -> bool:
    structured_statuses = {
        classification[0]
        for classification in (
            _CHAT_ERROR_TYPE_CLASSIFICATION.get(error_type or ""),
            _CHAT_ERROR_CODE_CLASSIFICATION.get(code or ""),
        )
        if classification is not None
    }
    # Structured type/code identities are authoritative. Only a coherent 400
    # identity can support context recovery; message text must not override a
    # recognized transient, authentication, permission, or rate-limit failure.
    if len(structured_statuses) > 1 or (structured_statuses and structured_statuses != {400}):
        return False
    if status_code is not None:
        # Some compatible providers (notably Gemini) report context overflow as
        # an otherwise-unclassified HTTP 500/504. Preserve that compatibility,
        # but reject a transport status that conflicts with a structured 400.
        if status_code not in {400, 500, 504}:
            return False
        if structured_statuses and status_code not in structured_statuses:
            return False
    if code == "context_length_exceeded":
        return True
    if error_type == "context_length_exceeded":
        return True
    if message is None:
        return False
    normalized = message.lower()
    if any(
        phrase in normalized
        for phrase in (
            "context_length_exceeded",
            "context length exceeded",
            "maximum context length",
            "input context is too long",
            "context is too long",
            "context too large",
            "prompt too large",
            "exceeds the context window",
        )
    ):
        return True
    return status_code in {400, 500, 504} and "context" in normalized and "too large" in normalized
