from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol
from urllib.parse import urlparse

import certifi
import httpx

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.core.messages import (
    Message,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from cayu.providers.base import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
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


class AnthropicProtocolError(AnthropicError):
    """Raised when Anthropic data does not match the expected Messages shape."""


class AnthropicTransport(Protocol):
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

    async def create_message(
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

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        try:
            payload = build_anthropic_payload(
                request,
                default_max_tokens=self.max_tokens,
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

    messages = [_anthropic_message(message) for message in request.messages]
    payload["messages"] = [message for message in messages if message is not None]
    if not payload["messages"]:
        raise ValueError("Anthropic requests require at least one non-system message.")
    tools = [_anthropic_tool(tool) for tool in request.tools]
    if tools:
        payload["tools"] = tools
    payload.update(options)
    return copy_json_value(payload, "anthropic_payload")


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


def _anthropic_message(message: Message) -> dict[str, Any] | None:
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
            "content": [_tool_result_block(part) for part in message.content],
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
) -> dict[str, Any]:
    if type(part) is not ToolResultPart:
        raise AnthropicProtocolError("Tool messages can only contain tool_result blocks.")
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": part.tool_call_id,
        "content": part.content,
    }
    if part.is_error:
        block["is_error"] = True
    return block


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
