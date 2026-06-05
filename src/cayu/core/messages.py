from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _require_nonblank("text", value)


class ToolCallPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    @classmethod
    def copy_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "arguments")

    @field_validator("tool_call_id", "tool_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return _require_clean_nonblank(info.field_name, value)


class ToolResultPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: str = ""
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    is_error: StrictBool = False

    @field_validator("structured", "artifacts", mode="before")
    @classmethod
    def copy_result_data(cls, value, info):
        return copy_json_value(value, info.field_name)

    @field_validator("tool_call_id", "tool_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return _require_clean_nonblank(info.field_name, value)


class ProviderStatePart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["provider_state"] = "provider_state"
    provider: str
    state: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        return _require_clean_nonblank("provider", value)

    @field_validator("state", mode="before")
    @classmethod
    def copy_state(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "state")


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: list[TextPart | ToolCallPart | ToolResultPart | ProviderStatePart] = Field(
        default_factory=list
    )

    @field_validator("content")
    @classmethod
    def copy_content(cls, value):
        return [copy_message_part(part) for part in value]

    @model_validator(mode="after")
    def validate_role_content(self) -> Message:
        if not self.content:
            raise ValueError("Message content cannot be empty.")
        if self.role in {MessageRole.USER, MessageRole.SYSTEM}:
            _require_parts(self.role, self.content, TextPart)
        elif self.role == MessageRole.ASSISTANT:
            _require_parts(
                self.role,
                self.content,
                TextPart,
                ToolCallPart,
                ProviderStatePart,
            )
        elif self.role == MessageRole.TOOL:
            _require_parts(self.role, self.content, ToolResultPart)
        return self

    @classmethod
    def text(cls, role: MessageRole | str, text: str) -> Message:
        return cls(role=MessageRole(role), content=[TextPart(text=text)])

    @classmethod
    def tool_call(
        cls,
        *,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        calls: list[ToolCallPart] | None = None,
    ) -> Message:
        content: list[TextPart | ToolCallPart | ToolResultPart | ProviderStatePart]
        if calls is not None:
            if tool_call_id is not None or tool_name is not None or arguments is not None:
                raise ValueError(
                    "`calls` cannot be combined with `tool_call_id`, `tool_name`, or `arguments`."
                )
            if not calls:
                raise ValueError("`calls` cannot be empty.")
            content = list(calls)
        else:
            content = [
                ToolCallPart(
                    tool_call_id=_require_value("tool_call_id", tool_call_id),
                    tool_name=_require_value("tool_name", tool_name),
                    arguments={} if arguments is None else arguments,
                )
            ]
        return cls(
            role=MessageRole.ASSISTANT,
            content=content,
        )

    @classmethod
    def tool_result(
        cls,
        *,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        content: str = "",
        structured: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        is_error: bool = False,
        results: list[ToolResultPart] | None = None,
    ) -> Message:
        result_parts: list[TextPart | ToolCallPart | ToolResultPart | ProviderStatePart]
        if not isinstance(content, str):
            raise ValueError("`content` must be a string.")
        if not isinstance(is_error, bool):
            raise ValueError("`is_error` must be a bool.")
        if results is not None:
            if (
                tool_call_id is not None
                or tool_name is not None
                or content != ""
                or structured is not None
                or artifacts is not None
                or is_error is not False
            ):
                raise ValueError("`results` cannot be combined with scalar result fields.")
            if not results:
                raise ValueError("`results` cannot be empty.")
            result_parts = list(results)
        else:
            result_parts = [
                ToolResultPart(
                    tool_call_id=_require_value("tool_call_id", tool_call_id),
                    tool_name=_require_value("tool_name", tool_name),
                    content=content,
                    structured=structured,
                    artifacts=[] if artifacts is None else artifacts,
                    is_error=is_error,
                )
            ]
        return cls(
            role=MessageRole.TOOL,
            content=result_parts,
        )


def _require_parts(
    role: MessageRole,
    content: list[TextPart | ToolCallPart | ToolResultPart | ProviderStatePart],
    *allowed_types: (
        type[TextPart] | type[ToolCallPart] | type[ToolResultPart] | type[ProviderStatePart]
    ),
) -> None:
    invalid_parts = [part.type for part in content if not isinstance(part, allowed_types)]
    if invalid_parts:
        allowed = ", ".join(part_type.__name__ for part_type in allowed_types)
        invalid = ", ".join(invalid_parts)
        raise ValueError(f"{role.value} messages only support {allowed}; got {invalid}.")


def _require_value(name: str, value: str | None) -> str:
    if value is None:
        raise ValueError(f"`{name}` is required.")
    return _require_clean_nonblank(name, value)


def copy_message(message: Message) -> Message:
    if type(message) is not Message:
        raise TypeError("Messages must be Message instances.")
    content = getattr(message, "content", None)
    if type(content) is not list:
        raise ValueError("Message content must be a list.")
    return Message(
        role=message.role,
        content=[copy_message_part(part) for part in content],
    )


def copy_message_part(
    part: TextPart | ToolCallPart | ToolResultPart | ProviderStatePart,
) -> TextPart | ToolCallPart | ToolResultPart | ProviderStatePart:
    if type(part) is TextPart:
        return TextPart(text=part.text)
    if type(part) is ToolCallPart:
        return ToolCallPart(
            tool_call_id=part.tool_call_id,
            tool_name=part.tool_name,
            arguments=copy_json_value(part.arguments, "arguments"),
        )
    if type(part) is ToolResultPart:
        return ToolResultPart(
            tool_call_id=part.tool_call_id,
            tool_name=part.tool_name,
            content=part.content,
            structured=copy_json_value(part.structured, "structured"),
            artifacts=copy_json_value(part.artifacts, "artifacts"),
            is_error=part.is_error,
        )
    if type(part) is ProviderStatePart:
        return ProviderStatePart(
            provider=part.provider,
            state=copy_json_value(part.state, "state"),
        )
    raise TypeError("Message content must contain supported message parts.")


def _require_nonblank(name: str, value: str) -> str:
    return require_nonblank(value, name)


def _require_clean_nonblank(name: str, value: str) -> str:
    return require_clean_nonblank(value, name)
