"""Durable publication-safe assistant state for quarantined tool rounds."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import require_clean_nonblank, require_durable_text
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, detach_message
from cayu.core.tools import ToolResult
from cayu.runtime._policy_evidence import ToolPolicyEvidence
from cayu.runtime.tool_exposure import (
    NOT_EXPOSED_IN_REQUEST_REASON,
    ResolvedToolExposureAuthority,
    unexposed_tool_result,
)
from cayu.vaults.redaction import REDACTED_SECRET

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


def validate_tool_exposure_terminal_event(
    event: Event,
    *,
    tool_exposure: ResolvedToolExposureAuthority,
) -> None:
    """Bind one exposure block to its exact durable request authority."""

    if type(event) is not Event:
        raise TypeError("Tool-exposure terminal must be an Event.")
    if type(tool_exposure) is not ResolvedToolExposureAuthority:
        raise TypeError("tool_exposure must be a ResolvedToolExposureAuthority.")
    exposure = tool_exposure
    payload = event.payload
    try:
        result = ToolResult.model_validate(payload.get("result"))
    except Exception as exc:
        raise ValueError("Tool-exposure terminal lost its fixed error result.") from exc
    fixed_result = unexposed_tool_result()
    if (
        event.type is not EventType.TOOL_CALL_BLOCKED
        or payload.get("blocked_by") != "tool_exposure"
        or payload.get("reason") != NOT_EXPOSED_IN_REQUEST_REASON
        or payload.get("profile_id") != exposure.profile_id
        or payload.get("exposure_fingerprint") != exposure.fingerprint
        or payload.get("arguments_state") != "unavailable"
        or "arguments" in payload
        or "effective_arguments" in payload
        or result.is_error is not True
        or result.structured is not None
        or bool(result.artifacts)
        or not _is_fixed_or_redacted_text(result.content, fixed_result.content)
    ):
        raise ValueError("Tool-exposure terminal conflicts with its frozen exposure authority.")


def _is_fixed_or_redacted_text(value: str, fixed: str) -> bool:
    """Accept only the fixed text or a marker projection derived from it."""

    if value == fixed:
        return True
    if REDACTED_SECRET not in value:
        return False
    pieces = value.split(REDACTED_SECRET)
    first = pieces[0]
    if first and not fixed.startswith(first):
        return False
    cursor = len(first)
    for index, piece in enumerate(pieces[1:], start=1):
        if index == len(pieces) - 1:
            if not piece:
                return cursor < len(fixed)
            position = len(fixed) - len(piece)
            return position >= cursor + 1 and fixed.endswith(piece)
        if not piece:
            cursor += 1
            if cursor > len(fixed):
                return False
            continue
        position = fixed.find(piece, cursor + 1)
        if position < 0:
            return False
        cursor = position + len(piece)
    return False


def validate_staged_tool_exposure_terminal(
    staged: StagedToolCallTerminal,
    *,
    policy_evidence: ToolPolicyEvidence | None,
    tool_exposure: ResolvedToolExposureAuthority | None,
) -> None:
    """Require staged exposure evidence to match its call classification and owner."""

    if type(staged) is not StagedToolCallTerminal:
        raise TypeError("Staged terminal must be a StagedToolCallTerminal.")
    payload = staged.event.payload
    claims_exposure_block = (
        staged.event.type is EventType.TOOL_CALL_BLOCKED
        and payload.get("blocked_by") == "tool_exposure"
    )
    carries_exposure_authority = any(
        field_name in payload for field_name in ("profile_id", "exposure_fingerprint")
    )
    if policy_evidence is ToolPolicyEvidence.UNEXPOSED:
        if tool_exposure is None:
            raise ValueError("Unexposed staged terminal lost its frozen exposure authority.")
        validate_tool_exposure_terminal_event(
            staged.event,
            tool_exposure=tool_exposure,
        )
        if staged.hooks_state != "completed":
            raise ValueError("Unexposed staged terminal cannot retain executable hook work.")
        return
    if claims_exposure_block or carries_exposure_authority:
        raise ValueError("Staged tool-exposure terminal requires unexposed tool-call evidence.")


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
