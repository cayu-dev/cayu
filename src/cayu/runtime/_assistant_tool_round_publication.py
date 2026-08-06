"""Durable publication-safe assistant state for quarantined tool rounds."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import require_clean_nonblank, require_durable_text
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, detach_message

_TOOL_ROUND_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
)


class StagedToolCallTerminal(BaseModel):
    """Private progressively sanitized terminal evidence awaiting publication."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    tool_call_id: str
    event: Event
    hooks_state: Literal["pending", "finalized", "observational", "completed"] = "pending"

    @field_validator("tool_call_id")
    @classmethod
    def validate_tool_call_id(cls, value: str) -> str:
        return require_clean_nonblank(
            require_durable_text(value, "tool_call_id"),
            "tool_call_id",
        )

    @field_validator("event")
    @classmethod
    def copy_event(cls, value: Event) -> Event:
        if type(value) is not Event:
            raise TypeError("Staged terminal event must be an Event.")
        return value.model_copy(deep=True)

    @model_validator(mode="after")
    def validate_terminal_identity(self) -> StagedToolCallTerminal:
        if self.event.type not in _TOOL_ROUND_TERMINAL_EVENT_TYPES:
            raise ValueError("Staged terminal evidence requires a terminal tool event.")
        if self.event.payload.get("tool_call_id") != self.tool_call_id:
            raise ValueError("Staged terminal event conflicts with its tool-call identity.")
        return self


class AssistantToolRoundPublication(BaseModel):
    """Progressively sanitized assistant turn and its invocation coverage."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    state: Literal["pending", "ready", "blocked"]
    message: Message | None = None
    covered_tool_call_ids: list[str] = Field(default_factory=list)
    secret_resolution_scope: Literal["static", "dynamic", "unknown"] = "unknown"
    reason: (
        Literal[
            "opaque_provider_state_secret",
            "incomplete_invocation_secret_scope",
            "projection_evidence_unavailable",
        ]
        | None
    ) = None

    @field_validator("message")
    @classmethod
    def copy_message(cls, value: Message | None) -> Message | None:
        return None if value is None else detach_message(value)

    @field_validator("covered_tool_call_ids")
    @classmethod
    def validate_covered_ids(cls, value: list[str]) -> list[str]:
        copied = [require_clean_nonblank(item, "covered_tool_call_id") for item in value]
        if len(set(copied)) != len(copied):
            raise ValueError("Assistant publication cannot repeat covered tool-call IDs.")
        return copied

    @model_validator(mode="after")
    def validate_state(self) -> AssistantToolRoundPublication:
        if self.state == "blocked":
            if self.message is not None or self.reason is None:
                raise ValueError("Blocked assistant publication requires only a fixed reason.")
        elif self.message is None or self.reason is not None:
            raise ValueError(
                "Publishable assistant state requires a message and no blocked reason."
            )
        return self


def copy_assistant_tool_round_publication(
    publication: AssistantToolRoundPublication | None,
) -> AssistantToolRoundPublication | None:
    if publication is None:
        return None
    if type(publication) is not AssistantToolRoundPublication:
        raise TypeError("assistant publication must be an AssistantToolRoundPublication.")
    return AssistantToolRoundPublication.model_validate(publication.model_dump(mode="json"))
