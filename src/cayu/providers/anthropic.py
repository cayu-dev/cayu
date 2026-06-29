from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol, cast
from urllib.parse import urlparse

import certifi
import httpx

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.artifacts import (
    FileAttachmentKind,
    file_attachment_from_payload,
    resolved_file_attachments_from_options,
)
from cayu.core.messages import (
    Message,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from cayu.providers.base import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelContextOverflowError,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.providers.cache import (
    CacheBreakpoint,
    CachePolicy,
    resolve_cache_policy,
)

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MAX_TOKENS = 4096
DEFAULT_ANTHROPIC_TIMEOUT_SECONDS = 60.0
MAX_PROVIDER_ERROR_BODY_CHARS = 2_000

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


class AnthropicAPIError(AnthropicError):
    """Raised when the Anthropic HTTP API returns an error response."""


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


class HttpxAnthropicTransport:
    """HTTP transport with explicit certifi-backed TLS verification."""

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

    async def _post_json(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        url = _validate_url(url, "url")
        try:
            async with httpx.AsyncClient(
                timeout=timeout_s,
                verify=certifi.where(),
            ) as client:
                response = await client.post(
                    url,
                    headers=dict(headers),
                    json=dict(payload),
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            try:
                _raise_anthropic_context_overflow_if_applicable(exc.response)
            except ModelContextOverflowError as overflow:
                raise overflow from exc
            raise AnthropicAPIError(
                "Anthropic API request failed with HTTP "
                f"{exc.response.status_code}: "
                f"{_safe_error_response_text(exc.response)}"
            ) from exc
        except httpx.RequestError as exc:
            raise AnthropicAPIError(f"Anthropic API request failed for {url}: {exc}") from exc

        try:
            decoded = response.json()
        except ValueError as exc:
            raise AnthropicProtocolError("Anthropic response was not valid JSON.") from exc
        if not isinstance(decoded, Mapping):
            raise AnthropicProtocolError("Anthropic response must be a JSON object.")
        return decoded


class AnthropicProvider(ModelProvider):
    """Anthropic Messages API adapter for Cayu's provider-neutral runtime."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        name: str = "anthropic",
        base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
        max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
        timeout_s: float = DEFAULT_ANTHROPIC_TIMEOUT_SECONDS,
        transport: AnthropicTransport | None = None,
        extra_headers: Mapping[str, str] | None = None,
        cache_policy: CachePolicy | None = None,
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.api_key = require_nonblank(
            api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", ""),
            "api_key",
        )
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
        self.transport = transport if transport is not None else HttpxAnthropicTransport()
        self.extra_headers = _copy_headers(extra_headers)
        if cache_policy is not None and type(cache_policy) is not CachePolicy:
            raise TypeError("cache_policy must be a CachePolicy.")
        self.cache_policy = cache_policy

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        try:
            policy = resolve_cache_policy(self.cache_policy, request.options)
            payload = build_anthropic_payload(
                request,
                default_max_tokens=self.max_tokens,
                cache_policy=policy,
            )
            response = await self.transport.create_message(
                url=f"{self.base_url}/v1/messages",
                headers=self._headers(),
                payload=payload,
                timeout_s=self.timeout_s,
            )
            for event in anthropic_response_events(response):
                yield event
        except Exception as exc:
            yield ModelStreamEvent.error(_exception_message(exc))

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
            headers=self._headers(),
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

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key,
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
        content[-1]["cache_control"] = policy.marker()


def anthropic_response_events(
    response: Mapping[str, Any],
) -> list[ModelStreamEvent]:
    if not isinstance(response, Mapping):
        raise AnthropicProtocolError("Anthropic response must be a JSON object.")
    content = response.get("content")
    if not isinstance(content, list):
        raise AnthropicProtocolError("Anthropic response content must be a list.")

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
                "usage": copy_json_value(response.get("usage"), "usage"),
            }
        )
    )
    return events


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
            "content": [_text_block(part) for part in message.content],
        }
    if message.role == MessageRole.ASSISTANT:
        content = [
            _assistant_block(part)
            for part in message.content
            if type(part) is not ProviderStatePart
        ]
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


def _text_block(
    part: TextPart | ToolCallPart | ToolResultPart | ProviderStatePart,
) -> dict[str, str]:
    if type(part) is not TextPart:
        raise AnthropicProtocolError("User messages can only contain text blocks.")
    return {"type": "text", "text": part.text}


def _assistant_block(
    part: TextPart | ToolCallPart | ToolResultPart | ProviderStatePart,
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


def _tool_result_block(
    part: TextPart | ToolCallPart | ToolResultPart | ProviderStatePart,
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
    if headers is None:
        return {}
    copied: dict[str, str] = {}
    for key, value in headers.items():
        header_name = require_clean_nonblank(key, "header name")
        if header_name.lower() in _PROTECTED_HEADER_NAMES:
            raise ValueError(f"extra_headers cannot override {header_name}.")
        copied[header_name] = require_nonblank(
            value,
            f"header {key}",
        )
    return copied


def _validate_base_url(base_url: str) -> str:
    return _validate_url(base_url, "base_url").rstrip("/")


def _validate_url(url: str, field_name: str) -> str:
    value = require_clean_nonblank(url, field_name)
    parsed = urlparse(value)
    if parsed.scheme != "https":
        raise ValueError(f"Anthropic {field_name} must use https.")
    if not parsed.netloc:
        raise ValueError(f"Anthropic {field_name} must include a host.")
    return value


def _max_tokens(value: Any) -> int:
    if type(value) is not int:
        raise ValueError("Anthropic max_tokens must be an integer.")
    if value <= 0:
        raise ValueError("Anthropic max_tokens must be greater than zero.")
    return value


def _safe_error_response_text(response: httpx.Response) -> str:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            decoded = response.json()
        except ValueError:
            return _truncate_error_text(response.text)
        if isinstance(decoded, Mapping):
            return _safe_error_json(decoded)
    return _truncate_error_text(response.text)


def _raise_anthropic_context_overflow_if_applicable(response: httpx.Response) -> None:
    decoded = _response_json_object(response)
    if decoded is None:
        return
    error = decoded.get("error")
    request_id = decoded.get("request_id")
    if not isinstance(error, Mapping):
        return
    error_type = _optional_error_string(error.get("type"))
    message = _optional_error_string(error.get("message"))
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
    normalized = message.lower()
    return (
        "prompt is too long" in normalized
        or "context window" in normalized
        or "maximum context" in normalized
        or ("token" in normalized and ("too long" in normalized or "exceed" in normalized))
    )


def _response_json_object(response: httpx.Response) -> Mapping[str, Any] | None:
    content_type = response.headers.get("content-type", "")
    if "application/json" not in content_type:
        return None
    try:
        decoded = response.json()
    except ValueError:
        return None
    if not isinstance(decoded, Mapping):
        return None
    return decoded


def _optional_error_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


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
            safe_error["message"] = _truncate_error_text(message)
        if isinstance(request_id, str):
            safe_error["request_id"] = request_id
        if safe_error:
            return _json_error_text(safe_error)
    return _truncate_error_text(_json_error_text(dict(decoded)))


def _json_error_text(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(value)


def _truncate_error_text(value: str) -> str:
    if len(value) <= MAX_PROVIDER_ERROR_BODY_CHARS:
        return value
    return value[:MAX_PROVIDER_ERROR_BODY_CHARS] + "... [truncated]"


def _exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return f"{type(exc).__name__}: Anthropic provider failed"
