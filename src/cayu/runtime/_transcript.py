from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from cayu._validation import copy_json_value
from cayu.core.messages import (
    Message,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    copy_message_part,
)
from cayu.runtime._runtime_records import ToolCallOutcome, ToolCallRequest


@dataclass
class AssistantTextPart:
    text: str


def initial_messages(
    *,
    system_prompt: str | None,
    request_messages: list[Message],
) -> list[Message]:
    messages: list[Message] = []
    if system_prompt and system_prompt.strip():
        messages.append(Message.text("system", system_prompt))
    messages.extend(message.model_copy(deep=True) for message in request_messages)
    return messages


def assistant_message(
    *,
    content_parts: list[AssistantTextPart | ToolCallPart],
    provider_state_parts: list[ProviderStatePart],
) -> Message | None:
    content: list[TextPart | ToolCallPart | ToolResultPart | ProviderStatePart] = []
    for part in content_parts:
        if type(part) is AssistantTextPart:
            if part.text.strip():
                content.append(TextPart(text=part.text))
            continue
        if type(part) is ToolCallPart:
            content.append(copy_message_part(part))
            continue
        raise TypeError("Assistant content must contain text buffers or tool calls.")
    content.extend(provider_state_parts)
    if not content:
        return None
    return Message(role=MessageRole.ASSISTANT, content=content)


def append_assistant_text_delta(
    content_parts: list[AssistantTextPart | ToolCallPart],
    delta: str,
) -> None:
    if not delta:
        return
    if content_parts and type(content_parts[-1]) is AssistantTextPart:
        previous = content_parts[-1]
        previous.text = f"{previous.text}{delta}"
        return
    content_parts.append(AssistantTextPart(text=delta))


def tool_call_part(tool_call: ToolCallRequest) -> ToolCallPart:
    return ToolCallPart(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        arguments=deepcopy(tool_call.arguments),
    )


def provider_state_parts(payload: dict[str, Any]) -> list[ProviderStatePart]:
    raw_parts = payload.get("provider_state", [])
    if raw_parts is None:
        return []
    if type(raw_parts) is not list:
        raise ValueError("Model completed payload provider_state must be a list.")
    parts: list[ProviderStatePart] = []
    for index, raw_part in enumerate(raw_parts):
        if type(raw_part) is not dict:
            raise ValueError(f"Model completed payload provider_state[{index}] must be an object.")
        raw_part = cast("dict[str, Any]", raw_part)
        provider = raw_part.get("provider")
        state = raw_part.get("state")
        if type(provider) is not str:
            raise ValueError(
                f"Model completed payload provider_state[{index}].provider must be a string."
            )
        if type(state) is not dict:
            raise ValueError(
                f"Model completed payload provider_state[{index}].state must be an object."
            )
        parts.append(ProviderStatePart(provider=provider, state=state))
    return parts


def model_completed_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    copied = copy_json_value(payload, "payload")
    if type(copied) is not dict:
        raise ValueError("Model completed payload must be an object.")
    copied.pop("provider_state", None)
    return copied


def parse_tool_call(payload: dict[str, Any]) -> ToolCallRequest:
    return ToolCallRequest(
        id=_optional_payload_string(payload, "id") or str(uuid4()),
        name=_require_payload_string(payload, "name"),
        arguments=copy_json_value(_require_payload_dict(payload, "arguments"), "arguments"),
    )


def tool_result_messages(outcomes: list[ToolCallOutcome]) -> list[Message]:
    return [
        Message.tool_result(
            results=[
                ToolResultPart(
                    tool_call_id=outcome.call.id,
                    tool_name=outcome.call.name,
                    content=outcome.result.content,
                    structured=deepcopy(outcome.result.structured),
                    artifacts=deepcopy(outcome.result.artifacts),
                    is_error=outcome.result.is_error,
                )
                for outcome in outcomes
            ],
        )
    ]


def _require_payload_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Model tool call payload requires non-empty string `{key}`.")
    return value


def _require_payload_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if type(value) is not dict:
        raise ValueError(f"Model tool call payload requires object `{key}`.")
    return value


def _optional_payload_string(payload: dict[str, Any], key: str) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    return _require_payload_string(payload, key)
