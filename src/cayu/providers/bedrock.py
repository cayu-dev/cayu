from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib
import json
import threading
from collections.abc import AsyncIterator, Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.artifacts import (
    FileAttachmentKind,
    file_attachment_from_payload,
    resolved_file_attachments_from_options,
)
from cayu.core.messages import (
    FilePart,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
)
from cayu.providers._http import exception_message
from cayu.providers.base import (
    InputTokenCountConfidence,
    InputTokenCountMethod,
    InputTokenCountResult,
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    UsageDialect,
)

DEFAULT_BEDROCK_MAX_TOKENS = 4096
DEFAULT_BEDROCK_STREAM_IDLE_TIMEOUT_SECONDS = 120.0
DEFAULT_BEDROCK_STREAM_CLOSE_TIMEOUT_SECONDS = 5.0
BEDROCK_STREAM_QUEUE_SIZE = 32

_RESERVED_BEDROCK_OPTIONS = frozenset({"modelId", "messages", "system", "toolConfig"})
_STREAM_ERROR_STATUS = {
    "internalServerException": 500,
    "modelStreamErrorException": 424,
    "serviceUnavailableException": 503,
    "throttlingException": 429,
    "validationException": 400,
}


class BedrockError(RuntimeError):
    """Base error for Amazon Bedrock provider failures."""


class BedrockAPIError(BedrockError, ModelProviderError):
    """Amazon Bedrock request or stream failure with provider-neutral fields."""

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
            provider="bedrock",
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
            retryable=retryable,
            retry_after_s=retry_after_s,
            response_body=response_body,
        )


class BedrockContextOverflowError(BedrockAPIError, ModelContextOverflowError):
    """Bedrock rejected a request because it exceeded the model context."""

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
            provider="bedrock",
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
            response_body=response_body,
        )


class BedrockProtocolError(BedrockError):
    """Bedrock returned a stream that violates the Converse contract."""


@dataclass
class _PendingToolUse:
    id: str
    name: str
    input_json: str = ""


@dataclass
class _PendingReasoning:
    text_parts: list[str] = field(default_factory=list)
    signature_parts: list[str] = field(default_factory=list)
    redacted_parts: list[bytes] = field(default_factory=list)


class BedrockProvider(ModelProvider):
    """Amazon Bedrock Converse adapter for Cayu's provider-neutral runtime."""

    name = "bedrock"
    usage_dialect = UsageDialect.ANTHROPIC

    def __init__(
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        endpoint_url: str | None = None,
        client: Any | None = None,
        name: str = "bedrock",
        max_tokens: int = DEFAULT_BEDROCK_MAX_TOKENS,
        stream_idle_timeout_s: float = DEFAULT_BEDROCK_STREAM_IDLE_TIMEOUT_SECONDS,
        stream_close_timeout_s: float = DEFAULT_BEDROCK_STREAM_CLOSE_TIMEOUT_SECONDS,
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.region_name = _optional_clean_string(region_name, "region_name")
        self.profile_name = _optional_clean_string(profile_name, "profile_name")
        self.endpoint_url = _optional_clean_string(endpoint_url, "endpoint_url")
        if client is not None and (self.profile_name is not None or self.endpoint_url is not None):
            raise ValueError(
                "An injected client cannot be combined with profile_name or endpoint_url."
            )
        if type(max_tokens) is not int:
            raise TypeError("max_tokens must be an integer.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")
        self.max_tokens = max_tokens
        self.stream_idle_timeout_s = _positive_float(stream_idle_timeout_s, "stream_idle_timeout_s")
        self.stream_close_timeout_s = _positive_float(
            stream_close_timeout_s, "stream_close_timeout_s"
        )
        self._owns_client = client is None
        self._client = client
        self._client_lock = threading.Lock()

    async def aclose(self) -> None:
        if not self._owns_client:
            return
        await asyncio.to_thread(self._close_owned_client)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        try:
            payload = build_bedrock_converse_payload(request, default_max_tokens=self.max_tokens)
            client = await self._get_client()
            raw_events = _boto3_stream_events(
                client,
                payload,
                idle_timeout_s=self.stream_idle_timeout_s,
                close_timeout_s=self.stream_close_timeout_s,
            )
            async for event in bedrock_converse_stream_events(raw_events):
                yield event
        except ModelContextOverflowError:
            raise
        except Exception as exc:
            wrapped = _bedrock_error_from_exception(exc)
            if isinstance(wrapped, ModelContextOverflowError):
                raise wrapped from exc
            yield ModelStreamEvent.error(
                exception_message(wrapped, provider_label="Bedrock"),
                cause=wrapped,
            )

    async def count_input_tokens(self, request: ModelRequest) -> InputTokenCountResult | None:
        payload = build_bedrock_converse_payload(request, default_max_tokens=self.max_tokens)
        model_id = payload.pop("modelId")
        count_input = {
            key: payload[key]
            for key in ("messages", "system", "toolConfig", "additionalModelRequestFields")
            if key in payload
        }
        try:
            client = await self._get_client()
            response = await asyncio.to_thread(
                client.count_tokens,
                modelId=model_id,
                input={"converse": count_input},
            )
        except Exception as exc:
            if _is_count_tokens_unsupported(exc):
                return None
            raise _bedrock_error_from_exception(exc) from exc
        input_tokens = response.get("inputTokens") if isinstance(response, Mapping) else None
        if type(input_tokens) is not int or input_tokens < 0:
            raise BedrockProtocolError("Bedrock CountTokens returned invalid inputTokens.")
        return InputTokenCountResult(
            input_tokens=input_tokens,
            method=InputTokenCountMethod.OFFICIAL,
            confidence=InputTokenCountConfidence.HIGH,
            metadata={
                "endpoint": "CountTokens",
                "provider_billing_status": "not_documented",
            },
        )

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        return await asyncio.to_thread(self._get_or_create_client)

    def _get_or_create_client(self) -> Any:
        with self._client_lock:
            if self._client is not None:
                return self._client
            self._client = self._create_client()
            return self._client

    def _create_client(self) -> Any:
        boto3 = _boto3_module()
        session_options: dict[str, Any] = {}
        if self.profile_name is not None:
            session_options["profile_name"] = self.profile_name
        session = boto3.Session(**session_options)
        client_options: dict[str, Any] = {}
        if self.region_name is not None:
            client_options["region_name"] = self.region_name
        if self.endpoint_url is not None:
            client_options["endpoint_url"] = self.endpoint_url
        return session.client("bedrock-runtime", **client_options)

    def _close_owned_client(self) -> None:
        with self._client_lock:
            client = self._client
            if client is None:
                return
            close = getattr(client, "close", None)
            if callable(close):
                close()
            self._client = None


def build_bedrock_converse_payload(
    request: ModelRequest,
    *,
    default_max_tokens: int = DEFAULT_BEDROCK_MAX_TOKENS,
) -> dict[str, Any]:
    if type(request) is not ModelRequest:
        raise TypeError("request must be a ModelRequest.")
    if type(default_max_tokens) is not int or default_max_tokens <= 0:
        raise ValueError("default_max_tokens must be a positive integer.")
    options = request.options.get("bedrock", {})
    if options is None:
        options = {}
    if type(options) is not dict:
        raise ValueError("ModelRequest.options['bedrock'] must be an object.")
    copied_options = copy_json_value(options, "bedrock options")
    for key in _RESERVED_BEDROCK_OPTIONS:
        if key in copied_options:
            raise ValueError(f"Bedrock option {key!r} is owned by BedrockProvider.")

    inference_config = copied_options.pop("inferenceConfig", {})
    if type(inference_config) is not dict:
        raise ValueError("Bedrock inferenceConfig must be an object.")
    inference_config.setdefault("maxTokens", default_max_tokens)

    resolved_attachments = resolved_file_attachments_from_options(request.options)
    system: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role == MessageRole.SYSTEM:
            system.extend(_bedrock_system_content(message.content))
            continue
        content = _bedrock_message_content(
            message.content,
            resolved_attachments,
            role=message.role,
        )
        if content:
            role = "user" if message.role == MessageRole.TOOL else str(message.role)
            messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("Bedrock requests require at least one non-system message.")

    payload: dict[str, Any] = {
        "modelId": request.model,
        "messages": messages,
        "inferenceConfig": inference_config,
    }
    if system:
        payload["system"] = system
    if request.tools:
        payload["toolConfig"] = {"tools": [_bedrock_tool(tool) for tool in request.tools]}
    payload.update(copied_options)
    # Boto3 accepts attachment bodies as bytes. Every non-byte value above was
    # already copied through Cayu's JSON-safe request/options contracts, and
    # decoded attachment bytes are newly allocated for this payload.
    return payload


async def bedrock_converse_stream_events(
    raw_events: AsyncIterator[Mapping[str, Any]],
) -> AsyncIterator[ModelStreamEvent]:
    tool_blocks: dict[int, _PendingToolUse] = {}
    reasoning_blocks: dict[int, _PendingReasoning] = {}
    stop_reason: str | None = None
    metadata: dict[str, Any] = {}
    saw_message_stop = False

    async for raw in raw_events:
        if not isinstance(raw, Mapping):
            raise BedrockProtocolError("Bedrock stream events must be objects.")
        _raise_stream_error(raw)
        if "contentBlockStart" in raw:
            start_event = _mapping(raw["contentBlockStart"], "contentBlockStart")
            index = _nonnegative_int(start_event.get("contentBlockIndex"), "contentBlockIndex")
            start = _mapping(start_event.get("start"), "contentBlockStart.start")
            tool_use = start.get("toolUse")
            if tool_use is not None:
                tool = _mapping(tool_use, "toolUse")
                tool_blocks[index] = _PendingToolUse(
                    id=_required_string(tool, "toolUseId"),
                    name=_required_string(tool, "name"),
                )
            continue
        if "contentBlockDelta" in raw:
            delta_event = _mapping(raw["contentBlockDelta"], "contentBlockDelta")
            index = _nonnegative_int(delta_event.get("contentBlockIndex"), "contentBlockIndex")
            delta = _mapping(delta_event.get("delta"), "contentBlockDelta.delta")
            text = delta.get("text")
            if text is not None:
                if type(text) is not str:
                    raise BedrockProtocolError("Bedrock text delta must be a string.")
                if text:
                    yield ModelStreamEvent.text_delta(text)
            tool_delta = delta.get("toolUse")
            if tool_delta is not None:
                tool = tool_blocks.get(index)
                if tool is None:
                    raise BedrockProtocolError("Bedrock tool delta arrived before tool start.")
                tool_input = _mapping(tool_delta, "toolUse delta").get("input", "")
                if type(tool_input) is not str:
                    raise BedrockProtocolError("Bedrock tool input delta must be a string.")
                tool.input_json += tool_input
            reasoning_delta = delta.get("reasoningContent")
            if reasoning_delta is not None:
                reasoning = _mapping(reasoning_delta, "reasoningContent delta")
                pending = reasoning_blocks.setdefault(index, _PendingReasoning())
                members = 0
                if "text" in reasoning:
                    members += 1
                    reasoning_text = reasoning["text"]
                    if type(reasoning_text) is not str:
                        raise BedrockProtocolError("Bedrock reasoning text delta must be a string.")
                    pending.text_parts.append(reasoning_text)
                if "signature" in reasoning:
                    members += 1
                    signature = reasoning["signature"]
                    if type(signature) is not str:
                        raise BedrockProtocolError(
                            "Bedrock reasoning signature delta must be a string."
                        )
                    pending.signature_parts.append(signature)
                if "redactedContent" in reasoning:
                    members += 1
                    redacted = reasoning["redactedContent"]
                    if not isinstance(redacted, (bytes, bytearray, memoryview)):
                        raise BedrockProtocolError(
                            "Bedrock redacted reasoning delta must contain bytes."
                        )
                    pending.redacted_parts.append(bytes(redacted))
                if members != 1:
                    raise BedrockProtocolError(
                        "Bedrock reasoning delta must contain exactly one union member."
                    )
            continue
        if "contentBlockStop" in raw:
            stop_event = _mapping(raw["contentBlockStop"], "contentBlockStop")
            index = _nonnegative_int(stop_event.get("contentBlockIndex"), "contentBlockIndex")
            tool = tool_blocks.pop(index, None)
            reasoning = reasoning_blocks.pop(index, None)
            if tool is not None and reasoning is not None:
                raise BedrockProtocolError(
                    "Bedrock content block mixed tool use and reasoning content."
                )
            if tool is not None:
                raw_input = tool.input_json
                try:
                    arguments = json.loads(raw_input or "{}")
                except json.JSONDecodeError as exc:
                    raise BedrockProtocolError("Bedrock tool input was not valid JSON.") from exc
                if type(arguments) is not dict:
                    raise BedrockProtocolError("Bedrock tool input must decode to an object.")
                yield ModelStreamEvent.tool_call(
                    id=tool.id,
                    name=tool.name,
                    arguments=arguments,
                )
            elif reasoning is not None:
                if reasoning.redacted_parts:
                    if reasoning.text_parts or reasoning.signature_parts:
                        raise BedrockProtocolError(
                            "Bedrock reasoning block mixed redacted and plaintext content."
                        )
                    yield ModelStreamEvent.thinking(
                        provider_state={
                            "type": "redacted_content",
                            "data_base64": base64.b64encode(
                                b"".join(reasoning.redacted_parts)
                            ).decode("ascii"),
                        }
                    )
                else:
                    provider_state: dict[str, Any] = {"type": "reasoning_text"}
                    signature = "".join(reasoning.signature_parts)
                    if signature:
                        provider_state["signature"] = signature
                    yield ModelStreamEvent.thinking(
                        "".join(reasoning.text_parts),
                        provider_state=provider_state,
                    )
            continue
        if "messageStop" in raw:
            message_stop = _mapping(raw["messageStop"], "messageStop")
            stop_reason = _required_string(message_stop, "stopReason")
            saw_message_stop = True
            continue
        if "metadata" in raw:
            metadata = dict(_mapping(raw["metadata"], "metadata"))

    if tool_blocks:
        raise BedrockProtocolError("Bedrock stream ended with unfinished tool blocks.")
    if reasoning_blocks:
        raise BedrockProtocolError("Bedrock stream ended with unfinished reasoning blocks.")
    if not saw_message_stop or stop_reason is None:
        raise BedrockProtocolError("Bedrock stream ended before messageStop.")
    payload: dict[str, Any] = {"stop_reason": stop_reason}
    raw_usage = metadata.get("usage")
    if raw_usage is not None:
        usage = _mapping(raw_usage, "metadata.usage")
        payload["usage"] = _canonical_bedrock_usage(usage)
        payload["bedrock_usage"] = copy_json_value(dict(usage), "bedrock_usage")
    metrics = metadata.get("metrics")
    if metrics is not None:
        payload["metrics"] = copy_json_value(metrics, "bedrock metrics")
    yield ModelStreamEvent.completed(payload)


async def _boto3_stream_events(
    client: Any,
    payload: Mapping[str, Any],
    *,
    idle_timeout_s: float,
    close_timeout_s: float,
) -> AsyncIterator[Mapping[str, Any]]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=BEDROCK_STREAM_QUEUE_SIZE)
    stop = threading.Event()
    stream_holder: list[Any] = []

    def put(kind: str, value: Any) -> bool:
        if stop.is_set():
            return False
        future = asyncio.run_coroutine_threadsafe(queue.put((kind, value)), loop)
        while not stop.is_set():
            try:
                future.result(timeout=0.25)
                return True
            except FutureTimeoutError:
                continue
        future.cancel()
        return False

    def consume() -> None:
        try:
            response = client.converse_stream(**dict(payload))
            if not isinstance(response, Mapping):
                raise BedrockProtocolError("Bedrock converse_stream returned a non-object.")
            stream = response.get("stream")
            if stream is None:
                raise BedrockProtocolError("Bedrock converse_stream response omitted stream.")
            stream_holder.append(stream)
            for raw in stream:
                if stop.is_set() or not put("event", raw):
                    break
        except BaseException as exc:
            put("error", exc)
        finally:
            put("done", None)

    worker = asyncio.create_task(asyncio.to_thread(consume))
    try:
        while True:
            try:
                kind, value = await asyncio.wait_for(queue.get(), timeout=idle_timeout_s)
            except TimeoutError as exc:
                raise BedrockProtocolError(
                    f"Bedrock stream produced no event for {idle_timeout_s:g} seconds."
                ) from exc
            if kind == "event":
                yield value
            elif kind == "error":
                raise value
            elif kind == "done":
                break
    finally:
        stop.set()
        close_deadline = loop.time() + close_timeout_s
        if stream_holder:
            close = getattr(stream_holder[0], "close", None)
            if callable(close):
                remaining = max(0.0, close_deadline - loop.time())
                if remaining > 0:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(asyncio.to_thread(close), timeout=remaining)
        if not worker.done():
            remaining = max(0.0, close_deadline - loop.time())
            if remaining > 0:
                with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(worker), timeout=remaining)


def _bedrock_system_content(parts: Sequence[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart) and part.text:
            result.append({"text": part.text})
    return result


def _bedrock_message_content(
    parts: Sequence[Any],
    resolved_attachments: Mapping[str, dict[str, Any]],
    *,
    role: MessageRole,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            result.append({"text": part.text})
        elif isinstance(part, ProviderStatePart):
            continue
        elif isinstance(part, ThinkingPart):
            if role != MessageRole.ASSISTANT:
                continue
            reasoning_block = _bedrock_reasoning_block(part)
            if reasoning_block is not None:
                result.append(reasoning_block)
        elif isinstance(part, ToolCallPart):
            result.append(
                {
                    "toolUse": {
                        "toolUseId": part.tool_call_id,
                        "name": part.tool_name,
                        "input": copy_json_value(part.arguments, "tool arguments"),
                    }
                }
            )
        elif isinstance(part, ToolResultPart):
            tool_content: list[dict[str, Any]] = []
            if part.content:
                tool_content.append({"text": part.content})
            for payload in part.artifacts:
                attachment = file_attachment_from_payload(payload)
                if attachment is None:
                    continue
                tool_content.append(
                    _bedrock_attachment_block(
                        _resolved_attachment(attachment.artifact_id, resolved_attachments)
                    )
                )
            if not tool_content:
                tool_content.append({"text": ""})
            result.append(
                {
                    "toolResult": {
                        "toolUseId": part.tool_call_id,
                        "content": tool_content,
                        "status": "error" if part.is_error else "success",
                    }
                }
            )
        elif isinstance(part, FilePart):
            attachment = file_attachment_from_payload(part.attachment)
            if attachment is None:
                raise BedrockProtocolError("FilePart did not contain a valid file attachment.")
            result.append(
                _bedrock_attachment_block(
                    _resolved_attachment(attachment.artifact_id, resolved_attachments)
                )
            )
    return result


def _bedrock_reasoning_block(part: ThinkingPart) -> dict[str, Any] | None:
    state = part.provider_state or {}
    if state.get("type") == "reasoning_text":
        signature = state.get("signature")
        if isinstance(signature, str) and signature:
            return {
                "reasoningContent": {"reasoningText": {"text": part.text, "signature": signature}}
            }
        return None
    if state.get("type") == "redacted_content":
        data_base64 = state.get("data_base64")
        if not isinstance(data_base64, str) or not data_base64:
            return None
        try:
            data = base64.b64decode(data_base64, validate=True)
        except ValueError:
            return None
        return {"reasoningContent": {"redactedContent": data}}
    return None


def _resolved_attachment(
    artifact_id: str,
    resolved_attachments: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    resolved = resolved_attachments.get(artifact_id)
    if resolved is None:
        raise BedrockProtocolError(f"Missing resolved file attachment: {artifact_id}")
    return resolved


def _bedrock_attachment_block(resolved: Mapping[str, Any]) -> dict[str, Any]:
    try:
        content = base64.b64decode(resolved["data_base64"], validate=True)
    except Exception as exc:
        raise BedrockProtocolError("Resolved file attachment contained invalid base64.") from exc
    kind = FileAttachmentKind(resolved["kind"])
    content_type = resolved["content_type"]
    if kind == FileAttachmentKind.IMAGE:
        image_formats = {
            "image/jpeg": "jpeg",
            "image/png": "png",
            "image/gif": "gif",
            "image/webp": "webp",
        }
        image_format = image_formats.get(content_type)
        if image_format is None:
            raise BedrockProtocolError(f"Unsupported Bedrock image content type: {content_type}")
        return {"image": {"format": image_format, "source": {"bytes": content}}}
    if kind == FileAttachmentKind.DOCUMENT:
        if content_type != "application/pdf":
            raise BedrockProtocolError(f"Unsupported Bedrock document content type: {content_type}")
        return {
            "document": {
                "format": "pdf",
                "name": "document.pdf",
                "source": {"bytes": content},
            }
        }
    raise BedrockProtocolError(f"Unsupported Bedrock file attachment kind: {kind}")


def _bedrock_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(tool, Mapping):
        raise ValueError("Tool definitions must be objects.")
    name = _required_string(tool, "name")
    description = tool.get("description", "")
    if type(description) is not str:
        raise ValueError("Tool descriptions must be strings.")
    schema = tool.get("input_schema")
    if type(schema) is not dict:
        raise ValueError("Tool input_schema must be an object.")
    return {
        "toolSpec": {
            "name": name,
            "description": description,
            "inputSchema": {"json": copy_json_value(schema, "tool input schema")},
        }
    }


def _canonical_bedrock_usage(usage: Mapping[str, Any]) -> dict[str, int]:
    keys = {
        "inputTokens": "input_tokens",
        "outputTokens": "output_tokens",
        "totalTokens": "total_tokens",
        "cacheReadInputTokens": "cache_read_input_tokens",
        "cacheWriteInputTokens": "cache_creation_input_tokens",
    }
    result: dict[str, int] = {}
    for source, target in keys.items():
        if source not in usage:
            continue
        value = usage[source]
        if type(value) is not int or value < 0:
            raise BedrockProtocolError(f"Bedrock usage {source} must be a non-negative integer.")
        result[target] = value
    return result


def _raise_stream_error(raw: Mapping[str, Any]) -> None:
    for key, status in _STREAM_ERROR_STATUS.items():
        if key not in raw:
            continue
        details = raw[key]
        message = f"Bedrock stream failed with {key}."
        if isinstance(details, Mapping):
            if key == "modelStreamErrorException":
                original_message = details.get("originalMessage")
                if isinstance(original_message, str) and original_message.strip():
                    message = original_message
                original_status = details.get("originalStatusCode")
                if type(original_status) is int:
                    status = original_status
            if message.startswith("Bedrock stream failed") and isinstance(
                details.get("message"), str
            ):
                message = details["message"]
        overflow = key == "validationException" and _is_context_overflow_message(message)
        error_options: dict[str, Any] = {
            "status_code": status,
            "error_type": key,
            "error_code": key,
            "response_body": json.dumps(details, default=str),
        }
        if overflow:
            raise BedrockContextOverflowError(message, **error_options)
        raise BedrockAPIError(
            message,
            retryable=key == "modelStreamErrorException" or status in {429, 500, 503},
            **error_options,
        )


def _bedrock_error_from_exception(exc: Exception) -> Exception:
    if isinstance(exc, (BedrockError, ModelProviderError)):
        return exc
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return BedrockError(str(exc) or type(exc).__name__)
    error = response.get("Error", {})
    metadata = response.get("ResponseMetadata", {})
    code = error.get("Code") if isinstance(error, Mapping) else None
    message = error.get("Message") if isinstance(error, Mapping) else None
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
    request_id = metadata.get("RequestId") if isinstance(metadata, Mapping) else None
    headers = metadata.get("HTTPHeaders", {}) if isinstance(metadata, Mapping) else {}
    retry_after_s: float | None = None
    if isinstance(headers, Mapping):
        raw_retry_after = headers.get("retry-after")
        if isinstance(raw_retry_after, str):
            with contextlib.suppress(ValueError):
                parsed_retry_after = float(raw_retry_after)
                if parsed_retry_after >= 0:
                    retry_after_s = parsed_retry_after
    if type(status) is not int:
        status = None
    if type(code) is not str:
        code = type(exc).__name__
    if type(message) is not str or not message.strip():
        message = str(exc) or code
    overflow = code == "ValidationException" and _is_context_overflow_message(message)
    error_class = BedrockContextOverflowError if overflow else BedrockAPIError
    kwargs: dict[str, Any] = {
        "status_code": status,
        "error_type": type(exc).__name__,
        "error_code": code,
        "request_id": request_id if type(request_id) is str else None,
        "response_body": json.dumps(response, default=str),
    }
    if error_class is BedrockAPIError:
        kwargs["retry_after_s"] = retry_after_s
        kwargs["retryable"] = code in {
            "ThrottlingException",
            "ModelNotReadyException",
            "ModelTimeoutException",
            "InternalServerException",
            "ServiceUnavailableException",
            "ModelErrorException",
        }
    return error_class(message, **kwargs)


def _is_context_overflow_message(message: str) -> bool:
    lowered = message.lower()
    return any(
        phrase in lowered
        for phrase in (
            "context window",
            "context length",
            "too many tokens",
            "input is too long",
            "maximum context",
        )
    )


def _is_count_tokens_unsupported(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return False
    error = response.get("Error")
    if not isinstance(error, Mapping) or error.get("Code") != "ValidationException":
        return False
    message = error.get("Message")
    if type(message) is not str:
        return False
    lowered = message.lower()
    return "support counting tokens" in lowered or "support counttokens" in lowered


def _boto3_module() -> Any:
    try:
        return importlib.import_module("boto3")
    except ModuleNotFoundError as exc:
        if exc.name != "boto3":
            raise
        raise RuntimeError(
            "BedrockProvider requires the optional AWS dependencies; install cayu[aws]."
        ) from exc


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BedrockProtocolError(f"Bedrock {field_name} must be an object.")
    return value


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if type(value) is not str or not value.strip():
        raise BedrockProtocolError(f"Bedrock {key} must be a non-empty string.")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise BedrockProtocolError(f"Bedrock {field_name} must be a non-negative integer.")
    return value


def _optional_clean_string(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return require_clean_nonblank(value, field_name)


def _positive_float(value: float, field_name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{field_name} must be a number.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return float(value)
