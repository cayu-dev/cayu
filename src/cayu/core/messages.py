from __future__ import annotations

import ipaddress
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    copy_durable_json_value,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    require_durable_text,
    require_execution_unit_id,
)
from cayu.artifacts.attachments import FileAttachment


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    type: Literal["text"] = "text"
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _require_nonblank("text", value)


class ToolCallPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_round_id: str | None = None
    model_step_id: str | None = None
    model_attempt_id: str | None = None

    @field_validator("arguments", mode="before")
    @classmethod
    def copy_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_value(value, "arguments")

    @field_validator("tool_call_id", "tool_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return _require_clean_nonblank(info.field_name, value)

    @field_validator("tool_round_id", "model_step_id", "model_attempt_id")
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_execution_unit_id(value, info.field_name)

    @model_validator(mode="after")
    def validate_complete_identity(self) -> ToolCallPart:
        _require_complete_tool_round_identity(
            self.tool_round_id,
            self.model_step_id,
            self.model_attempt_id,
        )
        return self


class ToolResultPart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: str = ""
    structured: dict[str, Any] | None = None
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    is_error: StrictBool = False
    tool_round_id: str | None = None
    model_step_id: str | None = None
    model_attempt_id: str | None = None

    @field_validator("structured", "artifacts", mode="before")
    @classmethod
    def copy_result_data(cls, value, info):
        return copy_durable_json_value(value, info.field_name)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        return require_durable_text(value, "content")

    @field_validator("tool_call_id", "tool_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return _require_clean_nonblank(info.field_name, value)

    @field_validator("tool_round_id", "model_step_id", "model_attempt_id")
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_execution_unit_id(value, info.field_name)

    @model_validator(mode="after")
    def validate_complete_identity(self) -> ToolResultPart:
        _require_complete_tool_round_identity(
            self.tool_round_id,
            self.model_step_id,
            self.model_attempt_id,
        )
        return self


class ProviderStatePart(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

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
        return copy_durable_json_value(value, "state")


class ThinkingPart(BaseModel):
    """Model reasoning/thinking content from a single reasoning block.

    `text` may be empty (the provider returned the reasoning in an omitted/redacted
    form). `provider_state` carries the opaque round-trip payload — the Anthropic
    `signature` or `redacted_thinking` data, or an OpenAI encrypted reasoning blob —
    needed to send the block back to the provider on a later turn. Built-in providers
    tag opaque reasoning state with its provider family, protocol, and protocol version;
    request adapters replay it only when all three match. Legacy untagged state remains
    readable in the durable transcript but is not authoritative for opaque replay.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    type: Literal["thinking"] = "thinking"
    text: str = ""
    provider_state: dict[str, Any] | None = None

    @field_validator("provider_state", mode="before")
    @classmethod
    def copy_provider_state(cls, value):
        return copy_durable_json_value(value, "provider_state")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return require_durable_text(value, "text")


class FilePart(BaseModel):
    """User-supplied file input (image or document) for a multimodal request.

    `attachment` carries a JSON-safe `cayu.file_attachment.v1` payload
    referencing a stored artifact — never file bytes. The runtime resolves the
    artifact from the active ArtifactStore immediately before each provider
    request, exactly like tool-result attachments.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    type: Literal["file"] = "file"
    attachment: dict[str, Any]

    @field_validator("attachment", mode="before")
    @classmethod
    def validate_attachment(cls, value):
        copied = copy_durable_json_value(value, "attachment")
        if type(copied) is not dict:
            raise ValueError("`attachment` must be a file attachment object.")
        return FileAttachment.model_validate(copied).model_dump(mode="json")


def _is_valid_url_hostname(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        try:
            ascii_hostname = hostname.removesuffix(".").encode("idna").decode("ascii")
        except UnicodeError:
            return False
        labels = ascii_hostname.split(".")
        return (
            0 < len(ascii_hostname) <= 253
            and all(0 < len(label) <= 63 for label in labels)
            and all(label[0].isalnum() and label[-1].isalnum() for label in labels)
            and all(
                character.isalnum() or character == "-" for label in labels for character in label
            )
        )
    return True


class WebSearchSource(BaseModel):
    """Bounded external source returned by a provider-hosted web search."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    type: Literal["url"] = "url"
    url: str = Field(max_length=4096)
    title: str | None = Field(default=None, max_length=1024)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = _require_clean_nonblank("url", value)
        parsed = urlsplit(value)
        try:
            hostname = parsed.hostname
            _port = parsed.port
        except ValueError as exc:
            raise ValueError("Web search source URLs must use a valid http or https host.") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or hostname is None
            or not hostname
            or not _is_valid_url_hostname(hostname)
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("Web search source URLs must use http or https.")
        return value

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_nonblank("title", value)


class WebSearchAction(BaseModel):
    """Bounded provider-neutral terminal web-search action evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    type: Literal["search", "open_page", "find_in_page"]
    query: str | None = Field(default=None, max_length=4096)
    queries: tuple[str, ...] = Field(default=(), max_length=100)
    url: str | None = Field(default=None, max_length=4096)
    pattern: str | None = Field(default=None, max_length=4096)
    sources: tuple[WebSearchSource, ...] = Field(default=(), max_length=100)

    @field_validator("query", "pattern")
    @classmethod
    def validate_optional_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _require_nonblank(info.field_name, value)

    @field_validator("url")
    @classmethod
    def validate_optional_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return WebSearchSource.validate_url(value)

    @field_validator("queries", mode="before")
    @classmethod
    def copy_queries(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("queries must be a list or tuple.")
        queries: list[str] = []
        for item in value:
            if type(item) is not str:
                raise ValueError("queries must contain strings.")
            queries.append(_require_nonblank("query", item))
        if any(len(query) > 4096 for query in queries):
            raise ValueError("queries must contain bounded strings.")
        return tuple(queries)

    @field_validator("sources", mode="before")
    @classmethod
    def copy_sources(cls, value: object) -> tuple[WebSearchSource, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("sources must be a list or tuple.")
        return tuple(WebSearchSource.model_validate(source) for source in value)

    @model_validator(mode="after")
    def validate_action_fields(self) -> WebSearchAction:
        if self.type == "search" and self.query is None and not self.queries:
            raise ValueError("Search actions require query or queries.")
        if self.type == "open_page" and self.url is None:
            raise ValueError("Open-page actions require url.")
        if self.type == "find_in_page" and (self.url is None or self.pattern is None):
            raise ValueError("Find-in-page actions require url and pattern.")
        return self


class HostedToolCallPart(BaseModel):
    """Terminal provider-hosted execution evidence in an assistant transcript."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    type: Literal["hosted_tool_call"] = "hosted_tool_call"
    hosted_tool: Literal["web_search"] = "web_search"
    call_id: str = Field(max_length=512)
    status: Literal["completed", "incomplete", "failed", "outcome_unknown"]
    action: WebSearchAction | None = None
    provider_name: str = Field(max_length=128)
    model: str = Field(max_length=512)
    model_step_id: str
    model_attempt_id: str

    @field_validator("call_id", "provider_name", "model")
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return _require_clean_nonblank(info.field_name, value)

    @field_validator("model_step_id", "model_attempt_id")
    @classmethod
    def validate_execution_identity(cls, value: str, info) -> str:
        return require_execution_unit_id(value, info.field_name)

    @field_validator("action", mode="before")
    @classmethod
    def copy_action(cls, value: object) -> object:
        if value is None:
            return None
        return WebSearchAction.model_validate(value)

    @model_validator(mode="after")
    def validate_terminal_action(self) -> HostedToolCallPart:
        if self.status == "completed" and self.action is None:
            raise ValueError("Completed hosted tool calls require terminal action evidence.")
        return self


class CitationProvenance(BaseModel):
    """Trust label for provider-hosted citation evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    provider_name: str = Field(max_length=128)
    hosted_tool: Literal["web_search"] = "web_search"
    untrusted_external_evidence: Literal[True] = True

    @field_validator("provider_name")
    @classmethod
    def validate_provider_name(cls, value: str) -> str:
        return _require_clean_nonblank("provider_name", value)


class CitationPart(BaseModel):
    """Provider-neutral citation annotation over assistant text."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    type: Literal["citation"] = "citation"
    citation_type: Literal["url_citation"] = "url_citation"
    url: str = Field(max_length=4096)
    title: str | None = Field(default=None, max_length=1024)
    start_index: StrictInt | None = Field(default=None, ge=0)
    end_index: StrictInt | None = Field(default=None, gt=0)
    provenance: CitationProvenance
    model_step_id: str
    model_attempt_id: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return WebSearchSource.validate_url(value)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_nonblank("title", value)

    @field_validator("model_step_id", "model_attempt_id")
    @classmethod
    def validate_execution_identity(cls, value: str, info) -> str:
        return require_execution_unit_id(value, info.field_name)

    @field_validator("provenance", mode="before")
    @classmethod
    def copy_provenance(cls, value: object) -> CitationProvenance:
        return CitationProvenance.model_validate(value)

    @model_validator(mode="after")
    def validate_offsets(self) -> CitationPart:
        if (self.start_index is None) != (self.end_index is None):
            raise ValueError("Citation offsets must be supplied together.")
        if (
            self.start_index is not None
            and self.end_index is not None
            and self.end_index <= self.start_index
        ):
            raise ValueError("Citation end_index must be greater than start_index.")
        return self


class _ValidatedContent(
    tuple[
        TextPart
        | ToolCallPart
        | ToolResultPart
        | ProviderStatePart
        | ThinkingPart
        | FilePart
        | HostedToolCallPart
        | CitationPart,
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
    message is copied so the message exclusively owns its JSON payloads. The
    freeze is shallow, though — nested payload dicts stay mutable — so sharing
    a validated instance is safe only while every holder treats payloads as
    read-only. Hot-path `copy_message` "copies" are no-ops under that
    contract; `detach_message` produces a genuinely isolated copy for
    storage/trust boundaries, and `copy.deepcopy` isolates as well.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    role: MessageRole
    content: tuple[
        TextPart
        | ToolCallPart
        | ToolResultPart
        | ProviderStatePart
        | ThinkingPart
        | FilePart
        | HostedToolCallPart
        | CitationPart,
        ...,
    ] = ()

    @field_validator("content")
    @classmethod
    def copy_content(cls, value):
        return _ValidatedContent(copy_message_part(part) for part in value)

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
                HostedToolCallPart,
                CitationPart,
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
        tool_round_id: str | None = None,
        model_step_id: str | None = None,
        model_attempt_id: str | None = None,
        calls: list[ToolCallPart] | None = None,
    ) -> Message:
        content: list[
            TextPart
            | ToolCallPart
            | ToolResultPart
            | ProviderStatePart
            | ThinkingPart
            | FilePart
            | HostedToolCallPart
            | CitationPart
        ]
        if calls is not None:
            if any(
                value is not None
                for value in (
                    tool_call_id,
                    tool_name,
                    arguments,
                    tool_round_id,
                    model_step_id,
                    model_attempt_id,
                )
            ):
                raise ValueError("`calls` cannot be combined with individual tool-call fields.")
            if not calls:
                raise ValueError("`calls` cannot be empty.")
            content = list(calls)
        else:
            content = [
                ToolCallPart(
                    tool_call_id=_require_value("tool_call_id", tool_call_id),
                    tool_name=_require_value("tool_name", tool_name),
                    arguments={} if arguments is None else arguments,
                    tool_round_id=tool_round_id,
                    model_step_id=model_step_id,
                    model_attempt_id=model_attempt_id,
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
        tool_round_id: str | None = None,
        model_step_id: str | None = None,
        model_attempt_id: str | None = None,
        results: list[ToolResultPart] | None = None,
    ) -> Message:
        result_parts: list[
            TextPart
            | ToolCallPart
            | ToolResultPart
            | ProviderStatePart
            | ThinkingPart
            | FilePart
            | HostedToolCallPart
            | CitationPart
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
                or tool_round_id is not None
                or model_step_id is not None
                or model_attempt_id is not None
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
                    tool_round_id=tool_round_id,
                    model_step_id=model_step_id,
                    model_attempt_id=model_attempt_id,
                )
            ]
        return cls(
            role=MessageRole.TOOL,
            content=tuple(result_parts),
        )


def _require_parts(
    role: MessageRole,
    content: Sequence[
        TextPart
        | ToolCallPart
        | ToolResultPart
        | ProviderStatePart
        | ThinkingPart
        | FilePart
        | HostedToolCallPart
        | CitationPart
    ],
    *allowed_types: (
        type[TextPart]
        | type[ToolCallPart]
        | type[ToolResultPart]
        | type[ProviderStatePart]
        | type[ThinkingPart]
        | type[FilePart]
        | type[HostedToolCallPart]
        | type[CitationPart]
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
    HostedToolCallPart,
    CitationPart,
)


def detach_message(message: Message) -> Message:
    """Return an isolated copy of `message` with detached JSON payloads.

    Unlike `copy_message`, this always rebuilds through full validation, so
    the result shares no mutable state (`arguments`, `structured`,
    `artifacts`, `state`, `provider_state`, `attachment`) with the input.
    Boundary primitive for code that hands messages across an ownership
    boundary — e.g. an in-memory store returning or ingesting transcripts.
    """
    if type(message) is not Message:
        raise TypeError("Messages must be Message instances.")
    return Message(role=message.role, content=message.content)


def copy_message(message: Message) -> Message:
    """Validate `message` and return it unchanged.

    `Message` is frozen and copied every part (with its JSON payloads) at
    construction, so a validated instance can be shared on hot paths: this
    "copy" is a no-op. It does NOT isolate nested payload dicts — every holder
    must treat them as read-only. Use `detach_message` where isolation is
    required (storage and other trust boundaries). Instances that bypassed
    validation (`model_construct`) are rebuilt through full validation
    instead.
    """
    if type(message) is not Message:
        raise TypeError("Messages must be Message instances.")
    if type(message.content) is _ValidatedContent:
        return message
    return detach_message(message)


def copy_message_part(
    part: TextPart
    | ToolCallPart
    | ToolResultPart
    | ProviderStatePart
    | ThinkingPart
    | FilePart
    | HostedToolCallPart
    | CitationPart,
) -> (
    TextPart
    | ToolCallPart
    | ToolResultPart
    | ProviderStatePart
    | ThinkingPart
    | FilePart
    | HostedToolCallPart
    | CitationPart
):
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
    return require_durable_nonblank(value, name)


def _require_clean_nonblank(name: str, value: str) -> str:
    return require_durable_clean_nonblank(value, name)


def _require_complete_tool_round_identity(
    tool_round_id: str | None,
    model_step_id: str | None,
    model_attempt_id: str | None,
) -> None:
    present = sum(value is not None for value in (tool_round_id, model_step_id, model_attempt_id))
    if present not in {0, 3}:
        raise ValueError(
            "Tool round identity requires tool_round_id, model_step_id, "
            "and model_attempt_id together."
        )
