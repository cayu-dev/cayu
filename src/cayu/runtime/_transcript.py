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
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
    copy_message_part,
)
from cayu.runtime._runtime_records import ToolCallOutcome, ToolCallRequest


@dataclass
class AssistantTextPart:
    text: str


@dataclass
class AssistantThinkingPart:
    text: str
    provider_state: dict[str, Any] | None = None
    include: bool = True


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
    content_parts: list[AssistantTextPart | AssistantThinkingPart | ToolCallPart],
    provider_state_parts: list[ProviderStatePart],
) -> Message | None:
    content: list[TextPart | ToolCallPart | ToolResultPart | ProviderStatePart | ThinkingPart] = []
    for part in content_parts:
        if type(part) is AssistantTextPart:
            if part.text.strip():
                content.append(TextPart(text=part.text))
            continue
        if type(part) is AssistantThinkingPart:
            thinking = _materialize_thinking(part)
            if thinking is not None:
                content.append(thinking)
            continue
        if type(part) is ToolCallPart:
            content.append(copy_message_part(part))
            continue
        raise TypeError("Assistant content must contain text buffers, thinking, or tool calls.")
    content.extend(provider_state_parts)
    if not content:
        return None
    return Message(role=MessageRole.ASSISTANT, content=content)


def _materialize_thinking(part: AssistantThinkingPart) -> ThinkingPart | None:
    # include_in_transcript=False drops reasoning that round-trips out of band (no
    # provider_state — e.g. an OpenAI display-only summary). A signed/redacted Anthropic
    # block is kept INTACT: its signature is computed over the original text, so blanking
    # the text would make it a modified block and break tool-use continuation. So opting
    # out only redacts providers whose reasoning round-trips separately, not Anthropic.
    if part.provider_state is None and (not part.include or not part.text):
        return None
    return ThinkingPart(text=part.text, provider_state=part.provider_state)


def append_assistant_text_delta(
    content_parts: list[AssistantTextPart | AssistantThinkingPart | ToolCallPart],
    delta: str,
) -> None:
    if not delta:
        return
    if content_parts and type(content_parts[-1]) is AssistantTextPart:
        previous = content_parts[-1]
        previous.text = f"{previous.text}{delta}"
        return
    content_parts.append(AssistantTextPart(text=delta))


def append_assistant_thinking_delta(
    content_parts: list[AssistantTextPart | AssistantThinkingPart | ToolCallPart],
    delta: str,
    *,
    provider_state: dict[str, Any] | None = None,
    include: bool = True,
) -> None:
    # A delta carrying provider_state is a complete, self-contained block (Anthropic
    # emits one event per thinking/redacted block) -> always its own part so each
    # signature round-trips independently. Stateless deltas (OpenAI streamed summary
    # text) accumulate onto a trailing stateless thinking buffer.
    if provider_state is not None:
        content_parts.append(
            AssistantThinkingPart(text=delta, provider_state=provider_state, include=include)
        )
        return
    if not delta:
        return
    if (
        content_parts
        and type(content_parts[-1]) is AssistantThinkingPart
        and content_parts[-1].provider_state is None
    ):
        content_parts[-1].text = f"{content_parts[-1].text}{delta}"
        return
    content_parts.append(AssistantThinkingPart(text=delta, include=include))


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
