from __future__ import annotations

from collections.abc import Sequence
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
from cayu.artifacts.attachments import FileAttachment


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["text"] = "text"
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _require_nonblank("text", value)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> TextPart:
        return self


class ToolCallPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> ToolCallPart:
        return self


class ToolResultPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> ToolResultPart:
        return self


class ProviderStatePart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> ProviderStatePart:
        return self


class ThinkingPart(BaseModel):
    """Model reasoning/thinking content from a single reasoning block.

    `text` may be empty (the provider returned the reasoning in an omitted/redacted
    form). `provider_state` carries the opaque round-trip payload — the Anthropic
    `signature` or `redacted_thinking` data, or an OpenAI encrypted reasoning blob —
    needed to send the block back to the provider on a later turn.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["thinking"] = "thinking"
    text: str = ""
    provider_state: dict[str, Any] | None = None

    @field_validator("provider_state", mode="before")
    @classmethod
    def copy_provider_state(cls, value):
        return copy_json_value(value, "provider_state")

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> ThinkingPart:
        return self


class FilePart(BaseModel):
    """User-supplied file input (image or document) for a multimodal request.

    `attachment` carries a JSON-safe `cayu.file_attachment.v1` payload
    referencing a stored artifact — never file bytes. The runtime resolves the
    artifact from the active ArtifactStore immediately before each provider
    request, exactly like tool-result attachments.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["file"] = "file"
    attachment: dict[str, Any]

    @field_validator("attachment", mode="before")
    @classmethod
    def validate_attachment(cls, value):
        copied = copy_json_value(value, "attachment")
        if type(copied) is not dict:
            raise ValueError("`attachment` must be a file attachment object.")
        return FileAttachment.model_validate(copied).model_dump(mode="json")

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> FilePart:
        return self


class _ValidatedContent(
    tuple[
        TextPart | ToolCallPart | ToolResultPart | ProviderStatePart | ThinkingPart | FilePart,
        ...,
    ]
):
    """Marker type for content produced by full `Message` validation.

    Pydantic runs after-model-validators even when an existing instance passes
    through a model-typed field unrevalidated, so an instance flag set in a
    validator cannot distinguish a validated Message from a `model_construct`
    bypass. The content tuple type can: only the `copy_content` field
    validator — which runs exclusively during full validation — produces it.
    """

    __slots__ = ()


class Message(BaseModel):
    """Frozen transcript message.

    Messages and their parts are immutable once constructed: attribute
    assignment is rejected, `content` is a tuple, and every part entering the
    message is copied so the message exclusively owns its JSON payloads. This
    makes construction the single copy-at-trust-boundary; sharing a stored
    `Message` afterwards is safe, and hot-path "copies" are no-ops.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: MessageRole
    content: tuple[
        TextPart | ToolCallPart | ToolResultPart | ProviderStatePart | ThinkingPart | FilePart,
        ...,
    ] = ()

    @field_validator("content")
    @classmethod
    def copy_content(cls, value):
        return _ValidatedContent(copy_message_part(part) for part in value)

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Message:
        return self

    @model_validator(mode="after")
    def validate_role_content(self) -> Message:
        if not self.content:
            raise ValueError("Message content cannot be empty.")
        if self.role == MessageRole.USER:
            _require_parts(self.role, self.content, TextPart, FilePart)
        elif self.role == MessageRole.SYSTEM:
            _require_parts(self.role, self.content, TextPart)
        elif self.role == MessageRole.ASSISTANT:
            _require_parts(
                self.role,
                self.content,
                TextPart,
                ToolCallPart,
                ProviderStatePart,
                ThinkingPart,
            )
        elif self.role == MessageRole.TOOL:
            _require_parts(self.role, self.content, ToolResultPart)
        return self

    @classmethod
    def text(cls, role: MessageRole | str, text: str) -> Message:
        return cls(role=MessageRole(role), content=(TextPart(text=text),))

    @classmethod
    def tool_call(
        cls,
        *,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        calls: list[ToolCallPart] | None = None,
    ) -> Message:
        content: list[
            TextPart | ToolCallPart | ToolResultPart | ProviderStatePart | ThinkingPart | FilePart
        ]
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
            content=tuple(content),
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
        result_parts: list[
            TextPart | ToolCallPart | ToolResultPart | ProviderStatePart | ThinkingPart | FilePart
        ]
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
            content=tuple(result_parts),
        )


def _require_parts(
    role: MessageRole,
    content: Sequence[
        TextPart | ToolCallPart | ToolResultPart | ProviderStatePart | ThinkingPart | FilePart
    ],
    *allowed_types: (
        type[TextPart]
        | type[ToolCallPart]
        | type[ToolResultPart]
        | type[ProviderStatePart]
        | type[ThinkingPart]
        | type[FilePart]
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


_MESSAGE_PART_TYPES = (
    TextPart,
    ToolCallPart,
    ToolResultPart,
    ProviderStatePart,
    ThinkingPart,
    FilePart,
)


def copy_message(message: Message) -> Message:
    """Validate `message` and return it unchanged.

    `Message` is frozen and copied every part (with its JSON payloads) at
    construction, so a validated instance can be shared safely: this "copy" is
    a no-op. Copying happens once, at the construction trust boundary —
    callers must treat nested payload dicts on returned messages as read-only.
    Instances that bypassed validation (`model_construct`) are rebuilt through
    full validation instead.
    """
    if type(message) is not Message:
        raise TypeError("Messages must be Message instances.")
    if type(message.content) is _ValidatedContent:
        return message
    return Message(role=message.role, content=message.content)


def copy_message_part(
    part: TextPart | ToolCallPart | ToolResultPart | ProviderStatePart | ThinkingPart | FilePart,
) -> TextPart | ToolCallPart | ToolResultPart | ProviderStatePart | ThinkingPart | FilePart:
    """Return an owned copy of `part`.

    Parts are frozen, but the caller that constructed a part may still hold
    references to its mutable JSON payloads (`arguments`, `structured`,
    `artifacts`, `state`, `provider_state`, `attachment`). Copying a part
    generically — a
    dump/validate round-trip through the part's own validators — detaches
    those payloads and revalidates `model_construct`-bypassed parts without a
    per-field copier that can drift as fields are added.
    """
    part_type = type(part)
    if part_type not in _MESSAGE_PART_TYPES:
        raise TypeError("Message content must contain supported message parts.")
    # warnings=False: dumps of `model_construct`-bypassed parts may hold
    # ill-typed values; validation below reports them properly.
    return part_type.model_validate(part.model_dump(warnings=False))


def _require_nonblank(name: str, value: str) -> str:
    return require_nonblank(value, name)


def _require_clean_nonblank(name: str, value: str) -> str:
    return require_clean_nonblank(value, name)
