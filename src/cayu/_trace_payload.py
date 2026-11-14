"""The payload members the session trace reads, and the reduction that keeps them.

Shared by :mod:`cayu.sessions.trace_events` and the storage DDL that keeps the
reduced copy, so neither imports the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cayu.events import EventType

#: Top-level members copied whole from every event: span identity, wait identity
#: and the outcome members that decide a tool call's or command's status.
TRACE_PAYLOAD_KEYS: tuple[str, ...] = (
    "model_step_id",
    "tool_round_id",
    "tool_call_id",
    "execution_id",
    "input_id",
    "approval_id",
    "interruption_type",
    "interrupted",
    "cancelled",
    "timed_out",
    "error_type",
    "exit_code",
)
#: Members copied whole only for the event type that carries nested identity in them.
TRACE_WHOLE_MEMBERS: dict[str, tuple[str, ...]] = {
    str(EventType.SESSION_INTERRUPTED): ("approval", "user_input"),
    str(EventType.SESSION_AWAITING_USER_INPUT): ("tool_calls",),
    str(EventType.TOOL_CALL_APPROVAL_REQUESTED): ("approval",),
}
#: Tool events carry their call and round again under ``result.structured``.
TRACE_RESULT_STRUCTURED_EVENT_TYPES: tuple[str, ...] = (
    str(EventType.TOOL_CALL_STARTED),
    str(EventType.TOOL_CALL_COMPLETED),
    str(EventType.TOOL_CALL_FAILED),
    str(EventType.TOOL_CALL_BLOCKED),
    str(EventType.TOOL_CALL_APPROVAL_REQUESTED),
    str(EventType.TOOL_CALL_APPROVAL_DENIED),
)
TRACE_RESULT_STRUCTURED_KEYS: tuple[str, ...] = ("tool_call_id", "tool_round_id")


def reduce_trace_payload(event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the members of ``payload`` the trace reads, as the stores' triggers do."""

    reduced = {key: payload[key] for key in TRACE_PAYLOAD_KEYS if key in payload}
    for key in TRACE_WHOLE_MEMBERS.get(str(event_type), ()):
        if key in payload:
            reduced[key] = payload[key]
    if str(event_type) in TRACE_RESULT_STRUCTURED_EVENT_TYPES:
        result = payload.get("result")
        structured = result.get("structured") if type(result) is dict else None
        if type(structured) is dict:
            reduced["result"] = {
                "structured": {
                    key: structured[key]
                    for key in TRACE_RESULT_STRUCTURED_KEYS
                    if key in structured
                }
            }
    return reduced
