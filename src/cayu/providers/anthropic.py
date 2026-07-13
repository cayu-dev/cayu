from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.artifacts import (
    FileAttachmentKind,
    file_attachment_from_payload,
    resolved_file_attachments_from_options,
)
from cayu.core.messages import (
    FilePart,
    Message,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
)
from cayu.providers._api_keys import resolve_api_key
from cayu.providers._http import (
    SharedAsyncClient,
    aclose_transport,
    copy_headers,
    exception_message,
    json_error_text,
    optional_error_string,
    post_json,
    response_json_object,
    safe_error_response_text,
    stream_sse_json_events,
    truncate_error_text,
    validate_base_url,
    validate_url,
)
from cayu.providers.base import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelContextOverflowError,
    ModelContextPressureProfile,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    UsageDialect,
)
from cayu.providers.cache import (
    CacheBreakpoint,
    CachePolicy,
    resolve_cache_policy,
)
from cayu.proxies import CredentialProxy
from cayu.vaults import SecretRef

if TYPE_CHECKING:
    import httpx

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MAX_TOKENS = 4096
DEFAULT_ANTHROPIC_TIMEOUT_SECONDS = 60.0
DEFAULT_ANTHROPIC_STREAM_IDLE_TIMEOUT_SECONDS = 120.0
ANTHROPIC_CONTEXT_PRESSURE_IMAGE_MIN_TOKENS = 100
ANTHROPIC_CONTEXT_PRESSURE_DOCUMENT_MIN_TOKENS = 1800

_RESERVED_ANTHROPIC_OPTIONS = {
    "model",
    "messages",
    "system",
    "tools",
    "stream",
}
_PROTECTED_HEADER_NAMES = {
    "anthropic-version",
    "content-type",
    "x-api-key",
}
_ANTHROPIC_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class AnthropicError(RuntimeError):
    """Base error for Anthropic provider failures."""


class AnthropicAPIError(AnthropicError, ModelProviderError):
    """Raised when the Anthropic HTTP API returns an error response."""

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
            provider="anthropic",
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
            retryable=retryable,
            retry_after_s=retry_after_s,
            response_body=response_body,
        )


class AnthropicContextOverflowError(AnthropicAPIError, ModelContextOverflowError):
    """Raised when Anthropic reports that the request exceeds context limits."""

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
            provider="anthropic",
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
            response_body=response_body,
        )


class AnthropicProtocolError(AnthropicError):
    """Raised when Anthropic data does not match the expected Messages shape."""


class AnthropicTransport(Protocol):
    async def count_message_tokens(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        """POST a Messages token-count payload and return decoded JSON."""

    async def create_message(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        """POST a Messages API payload and return decoded JSON."""

    def stream_message_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        """POST a streaming Messages API payload and yield decoded SSE data objects."""


class HttpxAnthropicTransport:
    """HTTP transport with explicit certifi-backed TLS verification.

    Owns one shared httpx.AsyncClient (created lazily) that is reused across
    requests so each model call does not pay for a fresh TLS handshake. Close it
    with :meth:`aclose` when the transport is no longer needed.
    """

    def __init__(self) -> None:
        self._client = SharedAsyncClient()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def count_message_tokens(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._post_json(
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
        )

    async def create_message(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        return await self._post_json(
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
        )

    async def stream_message_events(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
        stream_idle_timeout_s: float,
    ) -> AsyncIterator[Mapping[str, Any]]:
        url = _validate_url(url, "url")
        events = stream_sse_json_events(
            client=self._client.get(),
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
            stream_idle_timeout_s=stream_idle_timeout_s,
            request_label="Anthropic API",
            response_label="Anthropic",
            api_error=AnthropicAPIError,
            protocol_error=AnthropicProtocolError,
            error_response_text=_safe_error_response_text,
            raise_context_overflow=_raise_anthropic_context_overflow_if_applicable,
            api_error_from_response=_anthropic_api_error_from_response,
        )
        async for event in events:
            yield event

    async def _post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        url = _validate_url(url, "url")
        return await post_json(
            client=self._client.get(),
            url=url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
            request_label="Anthropic API",
            response_label="Anthropic",
            api_error=AnthropicAPIError,
            protocol_error=AnthropicProtocolError,
            error_response_text=_safe_error_response_text,
            raise_context_overflow=_raise_anthropic_context_overflow_if_applicable,
            api_error_from_response=_anthropic_api_error_from_response,
        )


class AnthropicProvider(ModelProvider):
    """Anthropic Messages API adapter for Cayu's provider-neutral runtime.

    The API key comes from one of two credential sources: a plain ``api_key``
    string (or ``ANTHROPIC_API_KEY``), or an async vault-backed path — pass
    ``api_key_ref`` (a ``SecretRef``) plus ``credential_proxy`` (e.g.
    ``AllowlistProxy``). With a ref, every request first authorizes the
    provider destination against the proxy and then resolves the key at the
    last moment, so raw credentials never live in provider config.
    """

    name = "anthropic"
    usage_dialect = UsageDialect.ANTHROPIC

    @property
    def context_pressure_profile(self) -> ModelContextPressureProfile:
        return ModelContextPressureProfile(
            image_min_tokens=ANTHROPIC_CONTEXT_PRESSURE_IMAGE_MIN_TOKENS,
            document_min_tokens=ANTHROPIC_CONTEXT_PRESSURE_DOCUMENT_MIN_TOKENS,
        )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_key_ref: SecretRef | None = None,
        credential_proxy: CredentialProxy | None = None,
        name: str = "anthropic",
        base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
        max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
        timeout_s: float = DEFAULT_ANTHROPIC_TIMEOUT_SECONDS,
        stream_idle_timeout_s: float = DEFAULT_ANTHROPIC_STREAM_IDLE_TIMEOUT_SECONDS,
        transport: AnthropicTransport | None = None,
        extra_headers: Mapping[str, str] | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        if api_key_ref is not None:
            if type(api_key_ref) is not SecretRef:
                raise TypeError("api_key_ref must be a SecretRef.")
            if api_key is not None:
                raise ValueError("Pass either api_key or api_key_ref, not both.")
            if not isinstance(credential_proxy, CredentialProxy):
                raise TypeError("api_key_ref requires a credential_proxy (CredentialProxy).")
            self.api_key = None
        else:
            if credential_proxy is not None:
                raise ValueError("credential_proxy requires api_key_ref.")
            self.api_key = resolve_api_key(
                api_key=api_key,
                env_var="ANTHROPIC_API_KEY",
                provider_name="AnthropicProvider",
                missing_hint=(
                    "set the ANTHROPIC_API_KEY environment variable, pass api_key=..., "
                    "or pass api_key_ref=SecretRef(...) with a credential_proxy for "
                    "deferred resolution."
                ),
            )
        self.api_key_ref = api_key_ref
        self.credential_proxy = credential_proxy
        self.base_url = _validate_base_url(base_url)
        self.anthropic_version = require_clean_nonblank(
            anthropic_version,
            "anthropic_version",
        )
        if type(max_tokens) is not int:
            raise TypeError("max_tokens must be an integer.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")
        if type(timeout_s) not in {int, float}:
            raise TypeError("timeout_s must be a number.")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero.")
        self.max_tokens = max_tokens
        self.timeout_s = float(timeout_s)
        if type(stream_idle_timeout_s) not in {int, float}:
            raise TypeError("stream_idle_timeout_s must be a number.")
        if stream_idle_timeout_s <= 0:
            raise ValueError("stream_idle_timeout_s must be greater than zero.")
        self.stream_idle_timeout_s = float(stream_idle_timeout_s)
        self.transport = transport if transport is not None else HttpxAnthropicTransport()
        self.extra_headers = _copy_headers(extra_headers)
        if cache_policy is not None and type(cache_policy) is not CachePolicy:
            raise TypeError("cache_policy must be a CachePolicy.")
        self.cache_policy = cache_policy

    async def aclose(self) -> None:
        """Close the transport's shared HTTP client, if it owns one."""
        await aclose_transport(self.transport)

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        try:
            headers = self._headers(await self._resolve_api_key())
            policy = resolve_cache_policy(self.cache_policy, request.options)
            payload = build_anthropic_payload(
                request,
                default_max_tokens=self.max_tokens,
                cache_policy=policy,
            )
            stream_transport = getattr(self.transport, "stream_message_events", None)
            if stream_transport is None:
                # Back-compat: transports predating SSE support fall back to one
                # buffered POST and a synthetic event replay.
                response = await self.transport.create_message(
                    url=f"{self.base_url}/v1/messages",
                    headers=headers,
                    payload=payload,
                    timeout_s=self.timeout_s,
                )
                for event in anthropic_response_events(response):
                    yield event
            else:
                payload["stream"] = True
                raw_events = stream_transport(
                    url=f"{self.base_url}/v1/messages",
                    headers=headers,
                    payload=payload,
                    timeout_s=self.timeout_s,
                    stream_idle_timeout_s=self.stream_idle_timeout_s,
                )
                async for event in anthropic_stream_events(raw_events):
                    yield event
        except ModelContextOverflowError:
            # Overflow must reach runtime recovery as a typed exception; an
            # error event would flatten it into unrecoverable message text.
            raise
        except Exception as exc:
            yield ModelStreamEvent.error(
                exception_message(exc, provider_label="Anthropic"),
                cause=exc,
            )

    async def count_input_tokens(
        self,
        request: ModelRequest,
    ) -> InputTokenCountResult | None:
        policy = resolve_cache_policy(self.cache_policy, request.options)
        payload = build_anthropic_token_count_payload(
            request,
            default_max_tokens=self.max_tokens,
            cache_policy=policy,
        )
        response = await self.transport.count_message_tokens(
            url=f"{self.base_url}/v1/messages/count_tokens",
            headers=self._headers(await self._resolve_api_key()),
            payload=payload,
            timeout_s=self.timeout_s,
        )
        return InputTokenCountResult(
            input_tokens=_anthropic_input_tokens_from_count_response(response),
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
            metadata={
                "endpoint": "messages/count_tokens",
                "provider_billing_status": "documented_free",
                "provider_rate_limit": "separate_rpm_limit",
            },
        )

    async def _resolve_api_key(self) -> str:
        """Return the API key from the configured credential source.

        With ``api_key_ref``, the destination is authorized against the
        credential proxy (egress allowlist) and the key resolved per request,
        so denials fail closed and rotated secrets are picked up live.
        """

        if self.api_key_ref is None or self.credential_proxy is None:
            if self.api_key is None:
                raise AnthropicError("Anthropic API key is not configured.")
            return self.api_key
        authorization = await self.credential_proxy.authorize_request(
            destination=self.base_url,
            credential=self.api_key_ref,
            action="anthropic.messages",
        )
        if not authorization.allowed:
            raise AnthropicAPIError(
                "Anthropic credential use denied by credential proxy for "
                f"{self.base_url}: {authorization.reason}",
                retryable=False,
            )
        resolved = await self.credential_proxy.resolve(
            self.api_key_ref,
            scope={"destination": self.base_url, "provider": self.name},
        )
        return resolved.value.get_secret_value()

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": self.anthropic_version,
        }
        headers.update(self.extra_headers)
        return headers


def build_anthropic_payload(
    request: ModelRequest,
    *,
    default_max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
    cache_policy: CachePolicy | None = None,
) -> dict[str, Any]:
    if type(request) is not ModelRequest:
        raise TypeError("request must be a ModelRequest.")
    if type(default_max_tokens) is not int:
        raise TypeError("default_max_tokens must be an integer.")
    if default_max_tokens <= 0:
        raise ValueError("default_max_tokens must be greater than zero.")

    options = _anthropic_options(request.options)
    payload: dict[str, Any] = {
        "model": request.model,
        "max_tokens": _max_tokens(options.pop("max_tokens", default_max_tokens)),
        "messages": [],
    }
    system = _system_text(request.messages)
    if system:
        payload["system"] = system

    resolved_attachments = resolved_file_attachments_from_options(request.options)
    messages = [
        _anthropic_message(message, resolved_attachments=resolved_attachments)
        for message in request.messages
    ]
    payload["messages"] = [message for message in messages if message is not None]
    if not payload["messages"]:
        raise ValueError("Anthropic requests require at least one non-system message.")
    tools = [_anthropic_tool(tool) for tool in request.tools]
    if tools:
        payload["tools"] = tools
    payload.update(options)
    _apply_thinking_options(payload, request.options.get("thinking"))
    _reconcile_thinking_budget(payload, default_max_tokens=default_max_tokens)
    if cache_policy is not None:
        _apply_cache_breakpoints(payload, cache_policy)
    return copy_json_value(payload, "anthropic_payload")


def build_anthropic_token_count_payload(
    request: ModelRequest,
    *,
    default_max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
    cache_policy: CachePolicy | None = None,
) -> dict[str, Any]:
    payload = build_anthropic_payload(
        request,
        default_max_tokens=default_max_tokens,
        cache_policy=cache_policy,
    )
    payload.pop("max_tokens", None)
    return payload


def _anthropic_thinking_options(neutral: Mapping[str, Any]) -> dict[str, Any]:
    """Map the neutral ``options["thinking"]`` payload to Anthropic request keys.

    Field-driven so no model lookup is needed (the request shape differs by model
    generation): ``effort`` -> adaptive thinking + ``output_config.effort``;
    ``max_tokens`` -> legacy ``thinking.budget_tokens``; otherwise adaptive.
    ``display="summarized"`` is requested whenever thinking is enabled so the newest
    models (where the default is ``omitted``) still return readable reasoning text.
    """
    if not neutral.get("enabled", True):
        return {"thinking": {"type": "disabled"}}
    adaptive = {"thinking": {"type": "adaptive", "display": "summarized"}}
    effort = neutral.get("effort")
    if effort is not None:
        return {**adaptive, "output_config": {"effort": effort}}
    max_tokens = neutral.get("max_tokens")
    if max_tokens is not None:
        return {
            "thinking": {"type": "enabled", "budget_tokens": max_tokens, "display": "summarized"}
        }
    return adaptive


def _apply_thinking_options(payload: dict[str, Any], neutral: Any) -> None:
    """Merge the mapped thinking config into the payload (typed config wins).

    ``thinking`` is a mode-exclusive unit, so it replaces any raw value; ``output_config``
    is merged so a caller's unrelated sibling keys (e.g. a raw ``output_config.format``)
    survive.
    """
    if not isinstance(neutral, Mapping):
        return
    mapped = _anthropic_thinking_options(neutral)
    payload["thinking"] = mapped["thinking"]
    output_config = mapped.get("output_config")
    if output_config:
        existing = payload.get("output_config")
        payload["output_config"] = (
            {**existing, **output_config} if isinstance(existing, dict) else output_config
        )


def _reconcile_thinking_budget(payload: dict[str, Any], *, default_max_tokens: int) -> None:
    """Ensure `max_tokens` exceeds a legacy thinking `budget_tokens`.

    Anthropic requires `max_tokens > thinking.budget_tokens` (the budget is part of the
    output allowance). A caller who only set a thinking budget would otherwise leave
    `max_tokens` at the default and get a 400, so raise it to leave response headroom
    above the budget.
    """
    thinking = payload.get("thinking")
    if not isinstance(thinking, dict):
        return
    budget = thinking.get("budget_tokens")
    max_tokens = payload.get("max_tokens")
    if isinstance(budget, int) and isinstance(max_tokens, int) and budget >= max_tokens:
        payload["max_tokens"] = budget + default_max_tokens


def _apply_cache_breakpoints(payload: dict[str, Any], policy: CachePolicy) -> None:
    if CacheBreakpoint.SYSTEM_PROMPT in policy.breakpoints:
        # `_system_text` emits a flat string today; if a future structured-system-prompt
        # feature makes it a block list, this guard skips it (no marker) rather than crash.
        system = payload.get("system")
        if isinstance(system, str) and system:
            payload["system"] = [{"type": "text", "text": system, "cache_control": policy.marker()}]
    if CacheBreakpoint.TOOL_DEFINITIONS in policy.breakpoints:
        tools = payload.get("tools")
        if tools:
            tools[-1]["cache_control"] = policy.marker()
    if CacheBreakpoint.CONVERSATION_PREFIX in policy.breakpoints:
        _mark_conversation_prefix(payload, policy)


def _mark_conversation_prefix(payload: dict[str, Any], policy: CachePolicy) -> None:
    if policy.conversation_prefix_strategy == "none":
        return
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return
    skip = (
        policy.conversation_prefix_n
        if policy.conversation_prefix_strategy == "all_but_last_n"
        else 1
    )
    boundary = len(messages) - 1 - skip
    if boundary < 0:
        return  # too few messages for a stable cacheable prefix
    content = messages[boundary].get("content")
    if isinstance(content, list) and content and isinstance(content[-1], dict):
        # Anthropic rejects cache_control on thinking/redacted_thinking blocks, so a
        # boundary turn ending in reasoning (e.g. a think-only turn) cannot be marked.
        if content[-1].get("type") in {"thinking", "redacted_thinking"}:
            return
        content[-1]["cache_control"] = policy.marker()


def anthropic_response_events(
    response: Mapping[str, Any],
) -> list[ModelStreamEvent]:
    if not isinstance(response, Mapping):
        raise AnthropicProtocolError("Anthropic response must be a JSON object.")
    content = response.get("content")
    if not isinstance(content, list):
        raise AnthropicProtocolError("Anthropic response content must be a list.")
    usage = response.get("usage")
    if usage is not None and not isinstance(usage, Mapping):
        raise AnthropicProtocolError("Anthropic response usage must be an object.")

    events: list[ModelStreamEvent] = []
    for index, block in enumerate(content):
        if not isinstance(block, Mapping):
            raise AnthropicProtocolError(f"Anthropic content block {index} must be an object.")
        block = cast("Mapping[str, Any]", block)
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise AnthropicProtocolError(f"Anthropic text block {index} requires string text.")
            if text:
                events.append(ModelStreamEvent.text_delta(text))
        elif block_type == "thinking":
            thinking_text = block.get("thinking", "")
            if not isinstance(thinking_text, str):
                raise AnthropicProtocolError(
                    f"Anthropic thinking block {index} requires string thinking."
                )
            provider_state: dict[str, Any] = {"type": "thinking"}
            signature = block.get("signature")
            if isinstance(signature, str) and signature:
                provider_state["signature"] = signature
            events.append(ModelStreamEvent.thinking(thinking_text, provider_state=provider_state))
        elif block_type == "redacted_thinking":
            data = block.get("data")
            if not isinstance(data, str) or not data:
                raise AnthropicProtocolError(
                    f"Anthropic redacted_thinking block {index} requires string data."
                )
            events.append(
                ModelStreamEvent.thinking(
                    provider_state={"type": "redacted_thinking", "data": data}
                )
            )
        elif block_type == "tool_use":
            tool_id = block.get("id")
            name = block.get("name")
            tool_input = block.get("input", {})
            if not isinstance(tool_id, str) or not tool_id.strip():
                raise AnthropicProtocolError(
                    f"Anthropic tool_use block {index} requires nonblank id."
                )
            if not isinstance(name, str) or not name.strip():
                raise AnthropicProtocolError(
                    f"Anthropic tool_use block {index} requires nonblank name."
                )
            if type(tool_input) is not dict:
                raise AnthropicProtocolError(
                    f"Anthropic tool_use block {index} input must be an object."
                )
            events.append(
                ModelStreamEvent.tool_call(
                    id=tool_id,
                    name=name,
                    arguments=copy_json_value(tool_input, "tool_input"),
                )
            )
        else:
            raise AnthropicProtocolError(
                f"Unsupported Anthropic content block type: {block_type!r}."
            )

    events.append(
        ModelStreamEvent.completed(
            {
                "id": _optional_string(response, "id"),
                "model": _optional_string(response, "model"),
                "stop_reason": _optional_string(response, "stop_reason"),
                "stop_sequence": _optional_string(response, "stop_sequence"),
                "usage": copy_json_value(None if usage is None else dict(usage), "usage"),
            }
        )
    )
    return events


class _PendingContentBlock:
    """Accumulates one streamed Anthropic content block until content_block_stop."""

    def __init__(self, block_type: str) -> None:
        self.type = block_type
        self.thinking_parts: list[str] = []
        self.signature_parts: list[str] = []
        self.json_parts: list[str] = []
        self.data: str | None = None
        self.tool_id: str | None = None
        self.tool_name: str | None = None


async def anthropic_stream_events(
    events: AsyncIterator[Mapping[str, Any]],
    *,
    provider_label: str = "Anthropic",
    api_error: Callable[..., Exception] = AnthropicAPIError,
    protocol_error: type[Exception] = AnthropicProtocolError,
    context_overflow_error: Callable[..., Exception] = AnthropicContextOverflowError,
) -> AsyncIterator[ModelStreamEvent]:
    """Translate Anthropic Messages SSE events into Cayu model stream events.

    Text streams incrementally as it arrives. Thinking and tool_use blocks are
    buffered per content-block index and emitted whole at ``content_block_stop``:
    a thinking block's signature is computed over its complete text (a partial
    block cannot round-trip), and tool_use arguments arrive as partial JSON that
    only parses once complete. Usage merges ``message_start`` input counts with
    the ``message_delta`` output counts. The error classes are parameterized so
    Anthropic-compatible hosts (Vertex) surface their own typed errors.
    """
    blocks: dict[int, _PendingContentBlock] = {}
    message_id: str | None = None
    model: str | None = None
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: dict[str, Any] | None = None
    completed = False

    def optional_string(mapping: Mapping[str, Any], key: str) -> str | None:
        value = mapping.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise protocol_error(f"{provider_label} stream {key} must be a string.")
        return value

    def block_index(event: Mapping[str, Any]) -> int:
        index = event.get("index")
        if type(index) is not int or index < 0:
            raise protocol_error(f"{provider_label} stream event requires non-negative index.")
        return index

    def required_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
        value = mapping.get(key)
        if not isinstance(value, str) or not value.strip():
            raise protocol_error(f"{provider_label} {label} requires nonblank {key}.")
        return value

    async for event in events:
        if not isinstance(event, Mapping):
            raise protocol_error(f"{provider_label} stream event must be a JSON object.")
        event_type = event.get("type")
        if event_type == "ping":
            continue
        if event_type == "message_start":
            message = event.get("message")
            if not isinstance(message, Mapping):
                raise protocol_error(f"{provider_label} message_start requires message object.")
            message_id = optional_string(message, "id")
            model = optional_string(message, "model")
            start_usage = message.get("usage")
            if start_usage is not None and not isinstance(start_usage, Mapping):
                raise protocol_error(f"{provider_label} message_start usage must be an object.")
            if isinstance(start_usage, Mapping):
                usage = {**(usage or {}), **start_usage}
            continue
        if event_type == "content_block_start":
            index = block_index(event)
            content_block = event.get("content_block")
            if not isinstance(content_block, Mapping):
                raise protocol_error(
                    f"{provider_label} content_block_start requires content_block object."
                )
            block_type = content_block.get("type")
            if block_type == "text":
                blocks[index] = _PendingContentBlock("text")
                text = content_block.get("text")
                if isinstance(text, str) and text:
                    yield ModelStreamEvent.text_delta(text)
            elif block_type == "thinking":
                pending = _PendingContentBlock("thinking")
                initial = content_block.get("thinking")
                if isinstance(initial, str) and initial:
                    pending.thinking_parts.append(initial)
                signature = content_block.get("signature")
                if isinstance(signature, str) and signature:
                    pending.signature_parts.append(signature)
                blocks[index] = pending
            elif block_type == "redacted_thinking":
                pending = _PendingContentBlock("redacted_thinking")
                pending.data = required_string(content_block, "data", "redacted_thinking block")
                blocks[index] = pending
            elif block_type == "tool_use":
                pending = _PendingContentBlock("tool_use")
                pending.tool_id = required_string(content_block, "id", "tool_use block")
                pending.tool_name = required_string(content_block, "name", "tool_use block")
                blocks[index] = pending
            else:
                raise protocol_error(
                    f"Unsupported {provider_label} content block type: {block_type!r}."
                )
            continue
        if event_type == "content_block_delta":
            index = block_index(event)
            pending = blocks.get(index)
            if pending is None:
                raise protocol_error(
                    f"{provider_label} content_block_delta arrived before content_block_start."
                )
            delta = event.get("delta")
            if not isinstance(delta, Mapping):
                raise protocol_error(f"{provider_label} content_block_delta requires delta object.")
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = delta.get("text")
                if not isinstance(text, str):
                    raise protocol_error(f"{provider_label} text_delta requires string text.")
                if text:
                    yield ModelStreamEvent.text_delta(text)
            elif delta_type == "thinking_delta":
                thinking = delta.get("thinking")
                if not isinstance(thinking, str):
                    raise protocol_error(
                        f"{provider_label} thinking_delta requires string thinking."
                    )
                if thinking:
                    pending.thinking_parts.append(thinking)
            elif delta_type == "signature_delta":
                signature = delta.get("signature")
                if not isinstance(signature, str):
                    raise protocol_error(
                        f"{provider_label} signature_delta requires string signature."
                    )
                if signature:
                    pending.signature_parts.append(signature)
            elif delta_type == "input_json_delta":
                partial_json = delta.get("partial_json")
                if not isinstance(partial_json, str):
                    raise protocol_error(
                        f"{provider_label} input_json_delta requires string partial_json."
                    )
                if partial_json:
                    pending.json_parts.append(partial_json)
            else:
                raise protocol_error(
                    f"Unsupported {provider_label} stream delta type: {delta_type!r}."
                )
            continue
        if event_type == "content_block_stop":
            index = block_index(event)
            pending = blocks.pop(index, None)
            if pending is None:
                raise protocol_error(
                    f"{provider_label} content_block_stop arrived before content_block_start."
                )
            if pending.type == "thinking":
                provider_state: dict[str, Any] = {"type": "thinking"}
                signature = "".join(pending.signature_parts)
                if signature:
                    provider_state["signature"] = signature
                yield ModelStreamEvent.thinking(
                    "".join(pending.thinking_parts),
                    provider_state=provider_state,
                )
            elif pending.type == "redacted_thinking":
                yield ModelStreamEvent.thinking(
                    provider_state={"type": "redacted_thinking", "data": pending.data}
                )
            elif pending.type == "tool_use":
                joined = "".join(pending.json_parts)
                if joined:
                    try:
                        arguments = json.loads(joined)
                    except ValueError as exc:
                        raise protocol_error(
                            f"{provider_label} tool_use input was not valid JSON."
                        ) from exc
                else:
                    arguments = {}
                if type(arguments) is not dict:
                    raise protocol_error(
                        f"{provider_label} tool_use input must decode to an object."
                    )
                if pending.tool_name is None:
                    raise protocol_error(f"{provider_label} tool_use block is missing a name.")
                yield ModelStreamEvent.tool_call(
                    id=pending.tool_id,
                    name=pending.tool_name,
                    arguments=copy_json_value(arguments, "tool_input"),
                )
            continue
        if event_type == "message_delta":
            delta = event.get("delta")
            if isinstance(delta, Mapping):
                stop_reason = optional_string(delta, "stop_reason") or stop_reason
                stop_sequence = optional_string(delta, "stop_sequence") or stop_sequence
            delta_usage = event.get("usage")
            if delta_usage is not None and not isinstance(delta_usage, Mapping):
                raise protocol_error(f"{provider_label} message_delta usage must be an object.")
            if isinstance(delta_usage, Mapping):
                usage = {**(usage or {}), **delta_usage}
            continue
        if event_type == "message_stop":
            if blocks:
                raise protocol_error(
                    f"{provider_label} message_stop arrived with unfinished content blocks."
                )
            completed = True
            yield ModelStreamEvent.completed(
                {
                    "id": message_id,
                    "model": model,
                    "stop_reason": stop_reason,
                    "stop_sequence": stop_sequence,
                    "usage": copy_json_value(usage, "usage"),
                }
            )
            continue
        if event_type == "error":
            raw_error = event.get("error")
            error = raw_error if isinstance(raw_error, Mapping) else {}
            error_type = optional_error_string(error.get("type"))
            error_message = optional_error_string(error.get("message"))
            if _is_anthropic_stream_error_context_overflow(
                error_type=error_type,
                message=error_message,
            ):
                raise context_overflow_error(
                    f"{provider_label} model context overflow",
                    error_type=error_type,
                )
            detail = error_message or error_type or "unknown error"
            raise api_error(
                f"{provider_label} streaming error: {detail}",
                error_type=error_type,
            )
        # Unknown event types are ignored for forward compatibility, as the
        # Anthropic streaming docs require.

    if not completed:
        raise protocol_error(f"{provider_label} streaming response ended before message_stop.")


def _is_anthropic_stream_error_context_overflow(
    *,
    error_type: str | None,
    message: str | None,
) -> bool:
    if error_type == "request_too_large":
        return True
    if error_type != "invalid_request_error" or message is None:
        return False
    return _anthropic_overflow_message(message)


def _anthropic_input_tokens_from_count_response(response: Mapping[str, Any]) -> int:
    if not isinstance(response, Mapping):
        raise AnthropicProtocolError("Anthropic token count response must be a JSON object.")
    input_tokens = response.get("input_tokens")
    if type(input_tokens) is not int or input_tokens < 0:
        raise AnthropicProtocolError("Anthropic token count response requires input_tokens.")
    return input_tokens


def _anthropic_options(options: Mapping[str, Any]) -> dict[str, Any]:
    raw_options = options.get("anthropic", {})
    if raw_options is None:
        return {}
    if type(raw_options) is not dict:
        raise ValueError("ModelRequest options.anthropic must be an object.")
    copied = copy_json_value(raw_options, "options.anthropic")
    for key in copied:
        if key in _RESERVED_ANTHROPIC_OPTIONS:
            raise ValueError(f"Anthropic option is reserved: {key}")
    return copied


def _system_text(messages: list[Message]) -> str:
    system_parts: list[str] = []
    for message in messages:
        if message.role != MessageRole.SYSTEM:
            continue
        for part in message.content:
            if type(part) is TextPart:
                system_parts.append(part.text)
    return "\n\n".join(system_parts)


def _anthropic_message(
    message: Message,
    *,
    resolved_attachments: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if message.role == MessageRole.SYSTEM:
        return None
    if message.role == MessageRole.USER:
        return {
            "role": "user",
            "content": [
                _user_block(part, resolved_attachments=resolved_attachments)
                for part in message.content
            ],
        }
    if message.role == MessageRole.ASSISTANT:
        # Thinking blocks must lead the assistant content and round-trip verbatim with
        # their signature/data (Anthropic rejects modified blocks during tool use); the
        # runtime assembles them first, so preserving content order keeps that ordering.
        content: list[dict[str, Any]] = []
        for part in message.content:
            if type(part) is ProviderStatePart:
                continue
            if type(part) is ThinkingPart:
                thinking_block = _assistant_thinking_block(part)
                if thinking_block is not None:
                    content.append(thinking_block)
                continue
            content.append(_assistant_block(part))
        if not content:
            return None
        return {
            "role": "assistant",
            "content": content,
        }
    if message.role == MessageRole.TOOL:
        return {
            "role": "user",
            "content": [
                _tool_result_block(part, resolved_attachments=resolved_attachments)
                for part in message.content
            ],
        }
    raise AnthropicProtocolError(f"Unsupported Cayu message role: {message.role!r}.")


def _user_block(
    part: TextPart | ToolCallPart | ToolResultPart | ProviderStatePart | ThinkingPart | FilePart,
    *,
    resolved_attachments: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if type(part) is TextPart:
        return {"type": "text", "text": part.text}
    if type(part) is FilePart:
        return _anthropic_file_attachment_block(
            _resolved_user_attachment(part, resolved_attachments)
        )
    raise AnthropicProtocolError("User messages can only contain text and file blocks.")


def _resolved_user_attachment(
    part: FilePart,
    resolved_attachments: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    attachment = file_attachment_from_payload(part.attachment)
    if attachment is None:
        raise AnthropicProtocolError("User file parts require a file attachment payload.")
    resolved = resolved_attachments.get(attachment.artifact_id)
    if resolved is None:
        raise AnthropicProtocolError(f"Missing resolved file attachment: {attachment.artifact_id}")
    return resolved


def _assistant_block(
    part: TextPart | ToolCallPart | ToolResultPart | ProviderStatePart | ThinkingPart | FilePart,
) -> dict[str, Any]:
    if type(part) is TextPart:
        return {"type": "text", "text": part.text}
    if type(part) is ToolCallPart:
        return {
            "type": "tool_use",
            "id": part.tool_call_id,
            "name": part.tool_name,
            "input": copy_json_value(part.arguments, "arguments"),
        }
    raise AnthropicProtocolError("Assistant messages can only contain text and tool_call blocks.")


def _assistant_thinking_block(part: ThinkingPart) -> dict[str, Any] | None:
    """Round-trip a thinking block back to Anthropic, or drop it when not echoable.

    Anthropic requires a tool-use loop's prior thinking/redacted_thinking blocks to be
    echoed verbatim with their ``signature``/``data``. A `ThinkingPart` lacking that
    opaque state (another provider's reasoning, or a cross-model switch) cannot be
    echoed, so it is dropped rather than triggering a 400.
    """
    state = part.provider_state or {}
    if state.get("type") == "redacted_thinking":
        data = state.get("data")
        if isinstance(data, str) and data:
            return {"type": "redacted_thinking", "data": data}
        return None
    signature = state.get("signature")
    if isinstance(signature, str) and signature:
        return {"type": "thinking", "thinking": part.text, "signature": signature}
    return None


def _tool_result_block(
    part: TextPart | ToolCallPart | ToolResultPart | ProviderStatePart | ThinkingPart | FilePart,
    *,
    resolved_attachments: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if type(part) is not ToolResultPart:
        raise AnthropicProtocolError("Tool messages can only contain tool_result blocks.")
    content_blocks = _tool_result_content_blocks(part, resolved_attachments)
    has_file_attachments = any(
        file_attachment_from_payload(payload) is not None for payload in part.artifacts
    )
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": part.tool_call_id,
        "content": content_blocks if has_file_attachments else part.content,
    }
    if part.is_error:
        block["is_error"] = True
    return block


def _tool_result_content_blocks(
    part: ToolResultPart,
    resolved_attachments: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    content_blocks: list[dict[str, Any]] = []
    if part.content:
        content_blocks.append({"type": "text", "text": part.content})
    for payload in part.artifacts:
        attachment = file_attachment_from_payload(payload)
        if attachment is None:
            continue
        resolved = resolved_attachments.get(attachment.artifact_id)
        if resolved is None:
            raise AnthropicProtocolError(
                f"Missing resolved file attachment: {attachment.artifact_id}"
            )
        content_blocks.append(_anthropic_file_attachment_block(resolved))
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})
    return content_blocks


def _anthropic_file_attachment_block(resolved: dict[str, Any]) -> dict[str, Any]:
    kind = FileAttachmentKind(resolved["kind"])
    if kind == FileAttachmentKind.IMAGE:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": resolved["content_type"],
                "data": resolved["data_base64"],
            },
        }
    if kind == FileAttachmentKind.DOCUMENT:
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": resolved["content_type"],
                "data": resolved["data_base64"],
            },
        }
    raise AnthropicProtocolError(f"Unsupported file attachment kind: {kind!r}")


def _anthropic_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(tool, Mapping):
        raise ValueError("Tool definitions must be objects.")
    name = _require_mapping_string(tool, "name")
    if not _ANTHROPIC_TOOL_NAME_RE.fullmatch(name):
        raise ValueError(
            "Anthropic tool names must contain 1-64 letters, numbers, underscores, or hyphens."
        )
    description = tool.get("description", "")
    if not isinstance(description, str):
        raise ValueError("Tool description must be a string.")
    input_schema = tool.get("input_schema", {})
    if type(input_schema) is not dict:
        raise ValueError("Tool input_schema must be an object.")
    return {
        "name": name,
        "description": description,
        "input_schema": copy_json_value(input_schema, "input_schema"),
    }


def _require_mapping_string(value: Mapping[str, Any], key: str) -> str:
    raw_value = value.get(key)
    if not isinstance(raw_value, str):
        raise ValueError(f"Tool {key} must be a string.")
    return require_clean_nonblank(raw_value, f"tool.{key}")


def _optional_string(response: Mapping[str, Any], key: str) -> str | None:
    value = response.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise AnthropicProtocolError(f"Anthropic response {key} must be a string.")
    return value


def _copy_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    return copy_headers(headers, protected=_PROTECTED_HEADER_NAMES)


def _validate_base_url(base_url: str) -> str:
    return validate_base_url(base_url, provider_label="Anthropic")


def _validate_url(url: str, field_name: str) -> str:
    return validate_url(url, field_name, provider_label="Anthropic")


def _max_tokens(value: Any) -> int:
    if type(value) is not int:
        raise ValueError("Anthropic max_tokens must be an integer.")
    if value <= 0:
        raise ValueError("Anthropic max_tokens must be greater than zero.")
    return value


def _safe_error_response_text(response: httpx.Response) -> str:
    return safe_error_response_text(response, format_error_json=_format_error_json)


def _format_error_json(decoded: Any) -> str | None:
    if not isinstance(decoded, Mapping):
        return None
    return _safe_error_json(decoded)


def _raise_anthropic_context_overflow_if_applicable(response: httpx.Response) -> None:
    decoded = response_json_object(response)
    if decoded is None:
        return
    error = decoded.get("error")
    request_id = decoded.get("request_id")
    if not isinstance(error, Mapping):
        return
    error_type = optional_error_string(error.get("type"))
    message = optional_error_string(error.get("message"))
    if not _is_anthropic_context_overflow(
        status_code=response.status_code,
        error_type=error_type,
        message=message,
    ):
        return
    raise AnthropicContextOverflowError(
        "Anthropic model context overflow",
        status_code=response.status_code,
        error_type=error_type,
        request_id=request_id if isinstance(request_id, str) else None,
        response_body=_safe_error_response_text(response),
    )


def _anthropic_api_error_from_response(
    response: httpx.Response,
    message: str,
) -> AnthropicAPIError:
    """Build a structured `AnthropicAPIError` from an HTTP error response.

    Preserves the response status code, the Anthropic error body's typed
    identity (`type`/`code`), the `request-id` header, and any `Retry-After`
    directive so runtime retry classification keys off typed fields instead of
    reparsing the flattened message text.
    """
    decoded = response_json_object(response)
    error: Mapping[str, Any] = {}
    if decoded is not None:
        raw_error = decoded.get("error")
        error = raw_error if isinstance(raw_error, Mapping) else decoded
    return AnthropicAPIError(
        message,
        status_code=response.status_code,
        error_type=optional_error_string(error.get("type")),
        error_code=optional_error_string(error.get("code")),
        request_id=optional_error_string(response.headers.get("request-id")),
        retry_after_s=_anthropic_retry_after_seconds(response),
        response_body=_safe_error_response_text(response),
    )


def _anthropic_retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse a `Retry-After` header (delta-seconds or HTTP-date) into seconds."""
    raw = response.headers.get("retry-after")
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return _anthropic_retry_after_from_http_date(raw)
    return seconds if seconds >= 0 else None


def _anthropic_retry_after_from_http_date(raw: str) -> float | None:
    try:
        target = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if target is None:
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    delta = (target - datetime.now(UTC)).total_seconds()
    return max(0.0, delta)


def _is_anthropic_context_overflow(
    *,
    status_code: int,
    error_type: str | None,
    message: str | None,
) -> bool:
    if status_code == 413 and error_type == "request_too_large":
        return True
    if status_code != 400 or error_type != "invalid_request_error" or message is None:
        return False
    return _anthropic_overflow_message(message)


def _anthropic_overflow_message(message: str) -> bool:
    """Whether an Anthropic-shaped error message describes a context overflow."""
    normalized = message.lower()
    return (
        "prompt is too long" in normalized
        or "context window" in normalized
        or "maximum context" in normalized
        or ("token" in normalized and ("too long" in normalized or "exceed" in normalized))
    )


def _safe_error_json(decoded: Mapping[str, Any]) -> str:
    error = decoded.get("error")
    request_id = decoded.get("request_id")
    if isinstance(error, Mapping):
        error_type = error.get("type")
        message = error.get("message")
        safe_error: dict[str, str] = {}
        if isinstance(error_type, str):
            safe_error["type"] = error_type
        if isinstance(message, str):
            safe_error["message"] = truncate_error_text(message)
        if isinstance(request_id, str):
            safe_error["request_id"] = request_id
        if safe_error:
            return json_error_text(safe_error)
    return truncate_error_text(json_error_text(dict(decoded)))
