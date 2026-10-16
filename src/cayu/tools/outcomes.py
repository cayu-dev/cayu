"""Detached, observational summaries of public tool-call events."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from cayu._validation import copy_durable_json_object
from cayu.events import EventType
from cayu.tools.base import ToolResult

_TOOL_CALL_TERMINAL_DISPOSITIONS: dict[EventType, Literal["completed", "failed", "blocked"]] = {
    EventType.TOOL_CALL_COMPLETED: "completed",
    EventType.TOOL_CALL_FAILED: "failed",
    EventType.TOOL_CALL_BLOCKED: "blocked",
    EventType.TOOL_CALL_APPROVAL_DENIED: "blocked",
    EventType.TOOL_CALL_APPROVAL_EXPIRED: "blocked",
}


class ToolCallSummary(BaseModel):
    """One public call, correlated by session, round and call identity.

    Arguments are the redacted public projection, never private execution
    authority. ``None`` means unavailable; an empty object is a real argument
    value. Event indices refer to the invocation's consumed stream. A terminal
    event can appear without a start when resuming or recovering a prior call.
    A restarted call reports its latest attempt. Malformed optional arguments
    or results are unavailable; the original events remain in RunOutcome.events.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: str
    interaction_id: str | None = None
    tool_call_id: str
    correlation_id: str | None = None
    tool_round_id: str | None = None
    tool_name: str | None = None
    idempotency_key: str | None = None
    disposition: Literal["completed", "failed", "blocked", "incomplete"]
    requested_arguments: dict[str, Any] | None = None
    effective_arguments: dict[str, Any] | None = None
    arguments_exact: bool | None = None
    result: ToolResult | None = None
    reason: str | None = None
    blocked_by: str | None = None
    start_event_id: str | None = None
    terminal_event_id: str | None = None
    start_index: int | None = None
    terminal_index: int | None = None

    @field_validator("requested_arguments", "effective_arguments", mode="before")
    @classmethod
    def detach_arguments(cls, value: object) -> dict[str, Any] | None:
        return None if value is None else copy_durable_json_object(value, "arguments")

    @field_validator("result", mode="before")
    @classmethod
    def detach_result(cls, value: object) -> ToolResult | None:
        if value is None:
            return None
        if isinstance(value, ToolResult):
            value = value.model_dump(mode="python")
        return ToolResult.model_validate(value)
