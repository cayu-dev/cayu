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
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 60.0
MAX_PROVIDER_ERROR_BODY_CHARS = 2_000

_RESERVED_OPENAI_OPTIONS = {
    "model",
    "input",
    "instructions",
    "previous_response_id",
    "store",
    "tools",
    "stream",
}
_PROTECTED_HEADER_NAMES = {
    "authorization",
    "content-type",
}
_OPENAI_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class OpenAIError(RuntimeError):
    """Base error for OpenAI provider failures."""


class OpenAIAPIError(OpenAIError):
    """Raised when the OpenAI HTTP API returns an error response."""


class OpenAIProtocolError(OpenAIError):
    """Raised when OpenAI data does not match the expected Responses shape."""


class OpenAITransport(Protocol):
    async def create_response(
        self,
        *,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Mapping[str, Any]:
        """POST a Responses API payload and return decoded JSON."""


class HttpxOpenAITransport:
    """HTTP transport with explicit certifi-backed TLS verification."""

    async def create_response(
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
            raise OpenAIAPIError(
                "OpenAI API request failed with HTTP "
                f"{exc.response.status_code}: "
                f"{_safe_error_response_text(exc.response)}"
            ) from exc
        except httpx.RequestError as exc:
            raise OpenAIAPIError(f"OpenAI API request failed for {url}: {exc}") from exc

        try:
            decoded = response.json()
        except ValueError as exc:
            raise OpenAIProtocolError("OpenAI response was not valid JSON.") from exc
        if not isinstance(decoded, Mapping):
            raise OpenAIProtocolError("OpenAI response must be a JSON object.")
        return decoded


class OpenAIProvider(ModelProvider):
    """OpenAI Responses API adapter for Cayu's provider-neutral runtime."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        name: str = "openai",
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        timeout_s: float = DEFAULT_OPENAI_TIMEOUT_SECONDS,
        transport: OpenAITransport | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.api_key = require_nonblank(
            api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", ""),
            "api_key",
        )
        self.base_url = _validate_base_url(base_url)
        if type(timeout_s) not in {int, float}:
            raise TypeError("timeout_s must be a number.")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero.")
        self.timeout_s = float(timeout_s)
        self.transport = transport if transport is not None else HttpxOpenAITransport()
        self.extra_headers = _copy_headers(extra_headers)

    async def stream(
        self,
        request: ModelRequest,
    ) -> AsyncIterator[ModelStreamEvent]:
        try:
            payload = build_openai_payload(request)
            response = await self.transport.create_response(
                url=f"{self.base_url}/v1/responses",
                headers=self._headers(),
                payload=payload,
                timeout_s=self.timeout_s,
            )
            for event in openai_response_events(response):
                yield event
        except Exception as exc:
            yield ModelStreamEvent.error(_exception_message(exc))

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }
        headers.update(self.extra_headers)
        return headers


def build_openai_payload(request: ModelRequest) -> dict[str, Any]:
    if type(request) is not ModelRequest:
        raise TypeError("request must be a ModelRequest.")

    options = _openai_options(request.options)
    payload: dict[str, Any] = {
        "model": request.model,
        "input": [],
        "store": False,
    }
    instructions = _system_text(request.messages)
    if instructions:
        payload["instructions"] = instructions

    resolved_attachments = resolved_file_attachments_from_options(request.options)
    input_items: list[dict[str, Any]] = []
    for message in request.messages:
        input_items.extend(_openai_input_items(message, resolved_attachments=resolved_attachments))
    if not input_items:
        raise ValueError("OpenAI requests require at least one non-system input item.")
    payload["input"] = input_items

    tools = [_openai_tool(tool) for tool in request.tools]
    if tools:
        payload["tools"] = tools
    payload.update(options)
    return copy_json_value(payload, "openai_payload")


def openai_response_events(response: Mapping[str, Any]) -> list[ModelStreamEvent]:
    if not isinstance(response, Mapping):
        raise OpenAIProtocolError("OpenAI response must be a JSON object.")

    error = response.get("error")
    if error is not None:
        raise OpenAIProtocolError(f"OpenAI response error: {_safe_error_value(error)}")

    output = response.get("output")
    if not isinstance(output, list):
        raise OpenAIProtocolError("OpenAI response output must be a list.")

    events: list[ModelStreamEvent] = []
    provider_state_items: list[dict[str, Any]] = []
    for index, item in enumerate(output):
        if not isinstance(item, Mapping):
            raise OpenAIProtocolError(f"OpenAI output item {index} must be an object.")
        item = cast("Mapping[str, Any]", item)
        item_type = item.get("type")
        if item_type == "message":
            events.extend(_message_output_events(item, index))
            provider_state_items.append(
                {"provider": "openai", "state": copy_json_value(item, "output_item")}
            )
        elif item_type == "function_call":
            events.append(_function_call_event(item, index))
            provider_state_items.append(
                {"provider": "openai", "state": copy_json_value(item, "output_item")}
            )
        elif item_type == "reasoning":
            provider_state_items.append(
                {"provider": "openai", "state": copy_json_value(item, "output_item")}
            )
            continue
        else:
            raise OpenAIProtocolError(f"Unsupported OpenAI output item type: {item_type!r}.")

    events.append(
        ModelStreamEvent.completed(
            {
                "id": _optional_string(response, "id"),
                "model": _optional_string(response, "model"),
                "status": _optional_string(response, "status"),
                "provider_state": provider_state_items,
                "usage": copy_json_value(response.get("usage"), "usage"),
                "incomplete_details": copy_json_value(
                    response.get("incomplete_details"),
                    "incomplete_details",
                ),
            }
        )
    )
    return events


def _message_output_events(
    item: Mapping[str, Any],
    item_index: int,
) -> list[ModelStreamEvent]:
    role = item.get("role")
    if role != "assistant":
        raise OpenAIProtocolError(
            f"OpenAI message output item {item_index} must have assistant role."
        )
    content = item.get("content")
    if not isinstance(content, list):
        raise OpenAIProtocolError(
            f"OpenAI message output item {item_index} content must be a list."
        )
    events: list[ModelStreamEvent] = []
    for content_index, part in enumerate(content):
        if not isinstance(part, Mapping):
            raise OpenAIProtocolError(
                f"OpenAI message output content {item_index}.{content_index} must be an object."
            )
        part = cast("Mapping[str, Any]", part)
        part_type = part.get("type")
        if part_type != "output_text":
            raise OpenAIProtocolError(
                f"Unsupported OpenAI message output content type: {part_type!r}."
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise OpenAIProtocolError("OpenAI output_text content requires string text.")
        if text:
            events.append(ModelStreamEvent.text_delta(text))
    return events


def _function_call_event(
    item: Mapping[str, Any],
    item_index: int,
) -> ModelStreamEvent:
    call_id = item.get("call_id")
    name = item.get("name")
    arguments = item.get("arguments")
    if not isinstance(call_id, str) or not call_id.strip():
        raise OpenAIProtocolError(
            f"OpenAI function_call item {item_index} requires nonblank call_id."
        )
    if not isinstance(name, str) or not name.strip():
        raise OpenAIProtocolError(f"OpenAI function_call item {item_index} requires nonblank name.")
    if not isinstance(arguments, str):
        raise OpenAIProtocolError(
            f"OpenAI function_call item {item_index} requires string arguments."
        )
    try:
        decoded_arguments = json.loads(arguments)
    except ValueError as exc:
        raise OpenAIProtocolError(
            f"OpenAI function_call item {item_index} arguments were not valid JSON."
        ) from exc
    if type(decoded_arguments) is not dict:
        raise OpenAIProtocolError(
            f"OpenAI function_call item {item_index} arguments must decode to an object."
        )
    return ModelStreamEvent.tool_call(
        id=call_id,
        name=name,
        arguments=copy_json_value(decoded_arguments, "arguments"),
    )


def _openai_options(options: Mapping[str, Any]) -> dict[str, Any]:
    raw_options = options.get("openai", {})
    if raw_options is None:
        return {}
    if type(raw_options) is not dict:
        raise ValueError("ModelRequest options.openai must be an object.")
    copied = copy_json_value(raw_options, "options.openai")
    for key in copied:
        if key in _RESERVED_OPENAI_OPTIONS:
            raise ValueError(f"OpenAI option is reserved: {key}")
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


def _openai_input_items(
    message: Message,
    *,
    resolved_attachments: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if message.role == MessageRole.SYSTEM:
        return []
    if message.role == MessageRole.USER:
        return [
            {
                "role": "user",
                "content": [_input_text_part(part) for part in message.content],
            }
        ]
    if message.role == MessageRole.ASSISTANT:
        provider_state_items = _openai_provider_state_items(message)
        if provider_state_items:
            return provider_state_items

        items: list[dict[str, Any]] = []
        text_parts = [_output_text_part(part) for part in message.content if type(part) is TextPart]
        if text_parts:
            items.append(
                {
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": text_parts,
                }
            )
        for part in message.content:
            if type(part) is ToolCallPart:
                items.append(_function_call_input_item(part))
            elif type(part) not in {TextPart, ProviderStatePart}:
                raise OpenAIProtocolError(
                    "Assistant messages can only contain text, tool_call, and provider_state parts."
                )
        return items
    if message.role == MessageRole.TOOL:
        items: list[dict[str, Any]] = []
        attachment_parts: list[dict[str, Any]] = []
        for part in message.content:
            items.append(_function_call_output_item(part))
            if type(part) is ToolResultPart:
                attachment_parts.extend(_openai_file_attachment_parts(part, resolved_attachments))
        if attachment_parts:
            items.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "The previous tool result returned file content for inspection.",
                        },
                        *attachment_parts,
                    ],
                }
            )
        return items
    raise OpenAIProtocolError(f"Unsupported Cayu message role: {message.role!r}.")


def _openai_provider_state_items(message: Message) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for part in message.content:
        if type(part) is not ProviderStatePart:
            continue
        if part.provider != "openai":
            continue
        state = copy_json_value(part.state, "provider_state")
        if type(state) is not dict:
            raise OpenAIProtocolError("OpenAI provider state must be an object.")
        item_type = state.get("type")
        if item_type not in {"reasoning", "message", "function_call"}:
            raise OpenAIProtocolError(
                f"Unsupported OpenAI provider state item type: {item_type!r}."
            )
        items.append(state)
    return items


def _input_text_part(
    part: TextPart | ToolCallPart | ToolResultPart | ProviderStatePart,
) -> dict[str, str]:
    if type(part) is not TextPart:
        raise OpenAIProtocolError("User messages can only contain text parts.")
    return {"type": "input_text", "text": part.text}


def _output_text_part(
    part: TextPart | ToolCallPart | ToolResultPart | ProviderStatePart,
) -> dict[str, str]:
    if type(part) is not TextPart:
        raise OpenAIProtocolError("Assistant text output requires a text part.")
    return {"type": "output_text", "text": part.text}


def _function_call_input_item(part: ToolCallPart) -> dict[str, Any]:
    return {
        "type": "function_call",
        "call_id": part.tool_call_id,
        "name": part.tool_name,
        "arguments": _json_arguments(part.arguments),
        "status": "completed",
    }


def _function_call_output_item(
    part: TextPart | ToolCallPart | ToolResultPart | ProviderStatePart,
) -> dict[str, Any]:
    if type(part) is not ToolResultPart:
        raise OpenAIProtocolError("Tool messages can only contain tool_result parts.")
    return {
        "type": "function_call_output",
        "call_id": part.tool_call_id,
        "output": part.content,
    }


def _openai_file_attachment_parts(
    part: ToolResultPart,
    resolved_attachments: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for payload in part.artifacts:
        attachment = file_attachment_from_payload(payload)
        if attachment is None:
            continue
        resolved = resolved_attachments.get(attachment.artifact_id)
        if resolved is None:
            raise OpenAIProtocolError(f"Missing resolved file attachment: {attachment.artifact_id}")
        parts.append(_openai_file_attachment_part(resolved))
    return parts


def _openai_file_attachment_part(resolved: dict[str, Any]) -> dict[str, Any]:
    kind = FileAttachmentKind(resolved["kind"])
    data_url = f"data:{resolved['content_type']};base64,{resolved['data_base64']}"
    if kind == FileAttachmentKind.IMAGE:
        return {
            "type": "input_image",
            "image_url": data_url,
        }
    if kind == FileAttachmentKind.DOCUMENT:
        return {
            "type": "input_file",
            "filename": resolved["filename"],
            "file_data": data_url,
        }
    raise OpenAIProtocolError(f"Unsupported file attachment kind: {kind!r}")


def _json_arguments(arguments: Mapping[str, Any]) -> str:
    copied = copy_json_value(arguments, "arguments")
    if type(copied) is not dict:
        raise OpenAIProtocolError("Tool call arguments must be an object.")
    return json.dumps(copied, sort_keys=True, separators=(",", ":"))


def _openai_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(tool, Mapping):
        raise ValueError("Tool definitions must be objects.")
    name = _require_mapping_string(tool, "name")
    if not _OPENAI_TOOL_NAME_RE.fullmatch(name):
        raise ValueError(
            "OpenAI tool names must contain 1-64 letters, numbers, underscores, or hyphens."
        )
    description = tool.get("description", "")
    if not isinstance(description, str):
        raise ValueError("Tool description must be a string.")
    input_schema = tool.get("input_schema", {})
    if type(input_schema) is not dict:
        raise ValueError("Tool input_schema must be an object.")
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": copy_json_value(input_schema, "input_schema"),
        "strict": False,
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
        raise OpenAIProtocolError(f"OpenAI response {key} must be a string.")
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
        raise ValueError(f"OpenAI {field_name} must use https.")
    if not parsed.netloc:
        raise ValueError(f"OpenAI {field_name} must include a host.")
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
        safe_error = _safe_flat_error_json(error)
        if isinstance(request_id, str):
            safe_error["request_id"] = request_id
        if safe_error:
            return _json_error_text(safe_error)
    safe_error = _safe_flat_error_json(decoded)
    if safe_error:
        return _json_error_text(safe_error)
    return _truncate_error_text(_json_error_text(dict(decoded)))


def _safe_error_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return _safe_error_json(value)
    if isinstance(value, str):
        return _truncate_error_text(value)
    return _truncate_error_text(str(value))


def _safe_flat_error_json(error: Mapping[str, Any]) -> dict[str, str]:
    error_type = error.get("type")
    message = error.get("message")
    code = error.get("code")
    safe_error: dict[str, str] = {}
    if isinstance(error_type, str):
        safe_error["type"] = error_type
    if isinstance(code, str):
        safe_error["code"] = code
    if isinstance(message, str):
        safe_error["message"] = _truncate_error_text(message)
    return safe_error


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
    return f"{type(exc).__name__}: OpenAI provider failed"
