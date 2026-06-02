from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import copy_json_value, require_nonblank
from cayu.core.agents import AgentSpec
from cayu.core.messages import Message, MessageRole, TextPart, ToolCallPart, ToolResultPart, copy_message
from cayu.runtime.sessions import Session


_COMPACTION_CHECKPOINT_KEY = "context_compaction"
_COMPACTION_CHECKPOINT_VERSION = 1


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


class ContextBuildResult(BaseModel):
    """Runtime-managed context result that may include checkpoint updates."""

    model_config = ConfigDict(extra="forbid")

    messages: list[Message]
    checkpoint: dict[str, Any] | None = None
    checkpoint_event_payload: dict[str, Any] | None = None

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        return [copy_message(message) for message in value]

    @field_validator("checkpoint", "checkpoint_event_payload", mode="before")
    @classmethod
    def copy_optional_json_data(cls, value, info):
        if value is None:
            return None
        return copy_json_value(value, info.field_name)


class RuntimeManagedContextPolicy(ContextPolicy):
    """Context policy whose checkpoint writes are owned by the runtime."""

    @abstractmethod
    async def build_with_checkpoint(
        self,
        request: ContextRequest,
        *,
        checkpoint: dict[str, Any] | None,
    ) -> ContextBuildResult:
        """Return model-facing context and optional checkpoint updates."""

    async def build(self, request: ContextRequest) -> list[Message]:
        result = await self.build_with_checkpoint(request, checkpoint=None)
        return result.messages


class DefaultContextPolicy(ContextPolicy):
    """Default policy that sends the current runtime transcript unchanged."""

    async def build(self, request: ContextRequest) -> list[Message]:
        return [copy_message(message) for message in request.messages]


class MessageWindowContextPolicy(ContextPolicy):
    """Built-in policy that keeps a valid recent message window."""

    def __init__(
        self,
        *,
        max_messages: int,
        preserve_system: bool = True,
    ) -> None:
        if type(max_messages) is not int:
            raise TypeError("max_messages must be an integer.")
        if type(preserve_system) is not bool:
            raise TypeError("preserve_system must be a bool.")
        if max_messages < 1:
            raise ValueError("max_messages must be greater than zero.")
        self.max_messages = max_messages
        self.preserve_system = preserve_system

    async def build(self, request: ContextRequest) -> list[Message]:
        return trim_context_messages(
            request.messages,
            max_messages=self.max_messages,
            preserve_system=self.preserve_system,
        )


class RecentTurnsContextPolicy(ContextPolicy):
    """Built-in policy that keeps recent user turns and complete tool rounds."""

    def __init__(
        self,
        *,
        max_user_turns: int,
        preserve_system: bool = True,
    ) -> None:
        if type(max_user_turns) is not int:
            raise TypeError("max_user_turns must be an integer.")
        if type(preserve_system) is not bool:
            raise TypeError("preserve_system must be a bool.")
        if max_user_turns < 1:
            raise ValueError("max_user_turns must be greater than zero.")
        self.max_user_turns = max_user_turns
        self.preserve_system = preserve_system

    async def build(self, request: ContextRequest) -> list[Message]:
        return trim_context_turns(
            request.messages,
            max_user_turns=self.max_user_turns,
            preserve_system=self.preserve_system,
        )


class CompactionRequest(BaseModel):
    """Input passed to a compactor when older context needs summarizing."""

    model_config = ConfigDict(extra="forbid")

    session: Session
    agent: AgentSpec
    messages: list[Message]
    existing_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        return [copy_message(message) for message in value]

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("existing_summary")
    @classmethod
    def validate_optional_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, "existing_summary")


class CompactionResult(BaseModel):
    """Compacted representation of older model-facing context."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return require_nonblank(value, "summary")

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")


class ContextCompactor(ABC):
    """Summarizes older context into durable checkpoint data."""

    @abstractmethod
    async def compact(self, request: CompactionRequest) -> CompactionResult:
        """Return a compact summary for older transcript messages."""


class TranscriptDigestCompactor(ContextCompactor):
    """Deterministic fallback compactor that stores a clipped text digest."""

    def __init__(self, *, max_summary_chars: int = 8000) -> None:
        if type(max_summary_chars) is not int:
            raise TypeError("max_summary_chars must be an integer.")
        if max_summary_chars < 200:
            raise ValueError("max_summary_chars must be at least 200.")
        self.max_summary_chars = max_summary_chars

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        sections: list[str] = []
        if request.existing_summary is not None:
            sections.append("Previous summary:\n" + request.existing_summary)
        if request.messages:
            sections.append("Newly compacted transcript:\n" + _messages_digest(request.messages))
        summary = "\n\n".join(sections)
        if len(summary) > self.max_summary_chars:
            summary = summary[-self.max_summary_chars :]
            summary = "[summary clipped to latest content]\n" + summary
        return CompactionResult(
            summary=summary,
            metadata={
                "compactor": type(self).__name__,
                "max_summary_chars": self.max_summary_chars,
            },
        )


class CheckpointCompactionContextPolicy(RuntimeManagedContextPolicy):
    """Checkpoint-backed context policy for long-running sessions.

    It keeps the durable transcript intact, stores a compact summary in the
    session checkpoint, and sends system messages + summary + recent turns to
    the model.
    """

    def __init__(
        self,
        *,
        compactor: ContextCompactor | None = None,
        max_user_turns: int = 10,
        compact_after_messages: int = 40,
        summary_prefix: str = "Previous session context summary:",
    ) -> None:
        if compactor is None:
            self.compactor = TranscriptDigestCompactor()
        elif isinstance(compactor, ContextCompactor):
            self.compactor = compactor
        else:
            raise TypeError("compactor must be a ContextCompactor.")
        if type(max_user_turns) is not int:
            raise TypeError("max_user_turns must be an integer.")
        if type(compact_after_messages) is not int:
            raise TypeError("compact_after_messages must be an integer.")
        if max_user_turns < 1:
            raise ValueError("max_user_turns must be greater than zero.")
        if compact_after_messages < 1:
            raise ValueError("compact_after_messages must be greater than zero.")
        self.max_user_turns = max_user_turns
        self.compact_after_messages = compact_after_messages
        self.summary_prefix = require_nonblank(summary_prefix, "summary_prefix")

    async def build_with_checkpoint(
        self,
        request: ContextRequest,
        *,
        checkpoint: dict[str, Any] | None,
    ) -> ContextBuildResult:
        checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        previous = _compaction_checkpoint(checkpoint)
        previous_summary = previous.get("summary") if previous is not None else None
        previous_count = (
            previous.get("compacted_message_count") if previous is not None else 0
        )
        if type(previous_summary) is not str:
            previous_summary = None
        if type(previous_count) is not int or previous_count < 0:
            previous_count = 0

        system_prefix, compactable_messages, recent_messages = _split_recent_turns(
            request.messages,
            max_user_turns=self.max_user_turns,
        )
        newly_compactable = compactable_messages[previous_count:]
        should_compact = (
            len(compactable_messages) >= self.compact_after_messages
            and bool(newly_compactable)
        )

        checkpoint_update = None
        checkpoint_event_payload = None
        summary = previous_summary
        compacted_count = min(previous_count, len(compactable_messages))
        if should_compact:
            result = await self.compactor.compact(
                CompactionRequest(
                    session=request.session,
                    agent=request.agent,
                    messages=newly_compactable,
                    existing_summary=previous_summary,
                    metadata=request.metadata,
                )
            )
            summary = result.summary
            compacted_count = len(compactable_messages)
            checkpoint_update = copy_json_value(checkpoint, "checkpoint")
            checkpoint_update[_COMPACTION_CHECKPOINT_KEY] = {
                "version": _COMPACTION_CHECKPOINT_VERSION,
                "summary": summary,
                "compacted_message_count": compacted_count,
                "metadata": result.metadata,
            }
            checkpoint_event_payload = {
                "checkpoint": _COMPACTION_CHECKPOINT_KEY,
                "compacted_message_count": compacted_count,
                "newly_compacted_message_count": len(newly_compactable),
                "recent_message_count": len(recent_messages),
            }

        messages = [copy_message(message) for message in system_prefix]
        if summary is not None:
            messages.append(Message.text(MessageRole.SYSTEM, f"{self.summary_prefix}\n{summary}"))
        messages.extend(copy_message(message) for message in recent_messages)
        validate_context_messages(messages)
        return ContextBuildResult(
            messages=messages,
            checkpoint=checkpoint_update,
            checkpoint_event_payload=checkpoint_event_payload,
        )


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


def _split_recent_turns(
    messages: list[Message],
    *,
    max_user_turns: int,
) -> tuple[list[Message], list[Message], list[Message]]:
    copied_messages = [copy_message(message) for message in messages]
    validate_context_messages(copied_messages)
    system_prefix, body = _split_system_prefix(copied_messages, True)
    turn_starts = [
        index
        for index, message in enumerate(body)
        if message.role == MessageRole.USER
    ]
    if not turn_starts or len(turn_starts) <= max_user_turns:
        return system_prefix, [], body
    recent_start = turn_starts[-max_user_turns]
    return system_prefix, body[:recent_start], body[recent_start:]


def _compaction_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    value = checkpoint.get(_COMPACTION_CHECKPOINT_KEY)
    if type(value) is not dict:
        return None
    if value.get("version") != _COMPACTION_CHECKPOINT_VERSION:
        return None
    return copy_json_value(value, _COMPACTION_CHECKPOINT_KEY)


def _messages_digest(messages: list[Message]) -> str:
    lines: list[str] = []
    for message in messages:
        parts = [_message_part_digest(part) for part in message.content]
        lines.append(f"{message.role}: " + " ".join(parts))
    return "\n".join(lines)


def _message_part_digest(part: TextPart | ToolCallPart | ToolResultPart) -> str:
    if type(part) is TextPart:
        return part.text
    if type(part) is ToolCallPart:
        return (
            f"[tool_call id={part.tool_call_id} name={part.tool_name} "
            f"arguments={copy_json_value(part.arguments, 'arguments')}]"
        )
    if type(part) is ToolResultPart:
        return (
            f"[tool_result id={part.tool_call_id} name={part.tool_name} "
            f"error={part.is_error} content={part.content}]"
        )
    raise TypeError("Unsupported message part.")
