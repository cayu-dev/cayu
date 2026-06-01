from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import copy_json_value, require_nonblank
from cayu.core.agents import AgentSpec
from cayu.core.messages import Message, MessageRole, ToolCallPart, copy_message
from cayu.runtime.sessions import Session


class ContextRequest(BaseModel):
    """Input passed to an agent context policy before each model request."""

    model_config = ConfigDict(extra="forbid")

    session: Session
    agent: AgentSpec
    messages: list[Message]
    step: StrictInt = Field(ge=1)
    environment_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        return [copy_message(message) for message in value]

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("environment_name")
    @classmethod
    def validate_optional_environment_name(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, "environment_name")


class ContextPolicy(ABC):
    """Builds the model-facing context for a runtime step.

    Policies may trim, summarize, replace tool results, or inject retrieved
    context. They must not be used as durable transcript storage.
    """

    @abstractmethod
    async def build(self, request: ContextRequest) -> list[Message]:
        """Return provider-neutral messages for one model request."""


class DefaultContextPolicy(ContextPolicy):
    """Default policy that sends the current runtime transcript unchanged."""

    async def build(self, request: ContextRequest) -> list[Message]:
        return [copy_message(message) for message in request.messages]


def copy_context_messages(messages: list[Message]) -> list[Message]:
    if type(messages) is not list:
        raise TypeError("ContextPolicy.build() must return a list of Message instances.")
    if not messages:
        raise ValueError("ContextPolicy.build() must return at least one message.")
    copied_messages = [copy_message(message) for message in messages]
    validate_context_messages(copied_messages)
    return copied_messages


def trim_context_messages(
    messages: list[Message],
    *,
    max_messages: int,
    preserve_system: bool = True,
) -> list[Message]:
    """Return a recent valid suffix without cutting through a tool round."""

    if type(max_messages) is not int:
        raise TypeError("max_messages must be an integer.")
    if type(preserve_system) is not bool:
        raise TypeError("preserve_system must be a bool.")
    if max_messages < 1:
        raise ValueError("max_messages must be greater than zero.")
    copied_messages = [copy_message(message) for message in messages]
    system_prefix, body = _split_system_prefix(copied_messages, preserve_system)
    candidate = system_prefix + body
    if len(candidate) <= max_messages:
        validate_context_messages(candidate)
        return [copy_message(message) for message in candidate]

    body_limit = max(1, max_messages - len(system_prefix))
    start = max(0, len(body) - body_limit)
    for index in range(start, len(body)):
        candidate = system_prefix + body[index:]
        try:
            validate_context_messages(candidate)
        except ValueError:
            continue
        return [copy_message(message) for message in candidate]
    raise ValueError("Cannot trim context without cutting through a tool round.")


def trim_context_turns(
    messages: list[Message],
    *,
    max_user_turns: int,
    preserve_system: bool = True,
) -> list[Message]:
    """Return the latest user turns with complete assistant/tool follow-up."""

    if type(max_user_turns) is not int:
        raise TypeError("max_user_turns must be an integer.")
    if type(preserve_system) is not bool:
        raise TypeError("preserve_system must be a bool.")
    if max_user_turns < 1:
        raise ValueError("max_user_turns must be greater than zero.")

    copied_messages = [copy_message(message) for message in messages]
    validate_context_messages(copied_messages)

    system_prefix, body = _split_system_prefix(copied_messages, preserve_system)
    turn_starts = [
        index
        for index, message in enumerate(body)
        if message.role == MessageRole.USER
    ]
    if not turn_starts:
        candidate = system_prefix + body
        validate_context_messages(candidate)
        return [copy_message(message) for message in candidate]
    if len(turn_starts) <= max_user_turns:
        candidate = system_prefix + body
        validate_context_messages(candidate)
        return [copy_message(message) for message in candidate]

    start = turn_starts[-max_user_turns]
    candidate = system_prefix + body[start:]
    validate_context_messages(candidate)
    return [copy_message(message) for message in candidate]


def validate_context_messages(messages: list[Message]) -> None:
    if type(messages) is not list:
        raise TypeError("Context messages must be a list of Message instances.")
    if not messages:
        raise ValueError("Context messages cannot be empty.")

    pending_tool_call_ids: set[str] | None = None
    for index, message in enumerate(messages):
        if type(message) is not Message:
            raise TypeError("Context messages must be Message instances.")

        if pending_tool_call_ids is not None:
            if message.role != MessageRole.TOOL:
                raise ValueError(
                    "Context messages contain assistant tool calls that are not "
                    "followed by matching tool results."
                )
            result_ids = [part.tool_call_id for part in message.content]
            if len(result_ids) != len(set(result_ids)):
                raise ValueError("Context messages contain duplicate tool result ids.")
            if set(result_ids) != pending_tool_call_ids:
                raise ValueError(
                    "Context messages contain tool results that do not match the "
                    "preceding assistant tool calls."
                )
            pending_tool_call_ids = None
            continue

        if message.role == MessageRole.TOOL:
            raise ValueError(
                "Context messages contain tool results without preceding assistant "
                "tool calls."
            )

        if message.role == MessageRole.ASSISTANT:
            tool_call_ids = [
                part.tool_call_id
                for part in message.content
                if type(part) is ToolCallPart
            ]
            if len(tool_call_ids) != len(set(tool_call_ids)):
                raise ValueError("Context messages contain duplicate tool call ids.")
            if tool_call_ids:
                pending_tool_call_ids = set(tool_call_ids)

    if pending_tool_call_ids is not None:
        raise ValueError(
            "Context messages end with assistant tool calls that have no matching "
            "tool results."
        )


def _split_system_prefix(
    messages: list[Message],
    preserve_system: bool,
) -> tuple[list[Message], list[Message]]:
    system_prefix: list[Message] = []
    for message in messages:
        if message.role != MessageRole.SYSTEM:
            break
        system_prefix.append(message)
    body = messages[len(system_prefix) :]
    if not preserve_system:
        return [], body
    return system_prefix, body
