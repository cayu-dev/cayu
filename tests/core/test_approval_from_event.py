"""Tests for ``PendingToolApproval.from_event`` (the nested-payload accessor)."""

from __future__ import annotations

import pytest

from cayu import Event, EventType, PendingToolApproval, PendingToolCallApproval


def _pending() -> PendingToolApproval:
    return PendingToolApproval(
        approval_id="ap_1",
        tool_call_id="call_1",
        tool_name="send_email",
        agent_name="assistant",
        tool_calls=[PendingToolCallApproval(tool_call_id="call_1", tool_name="send_email")],
    )


def _approval_event() -> Event:
    return Event(
        type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
        payload={"approval": _pending().model_dump(mode="json")},
        session_id="sess_1",
    )


def test_from_event_reads_the_nested_approval() -> None:
    event = _approval_event()
    got = PendingToolApproval.from_event(event)
    assert got.approval_id == "ap_1"
    assert got.tool_call_id == "call_1"
    # The mistake from_event exists to prevent: the id is NOT at the top level.
    assert event.payload.get("approval_id") is None


def test_from_event_rejects_wrong_event_type() -> None:
    event = Event(type=EventType.SESSION_STARTED, payload={}, session_id="s")
    with pytest.raises(ValueError, match="approval_requested"):
        PendingToolApproval.from_event(event)


def test_from_event_rejects_missing_approval_payload() -> None:
    event = Event(type=EventType.TOOL_CALL_APPROVAL_REQUESTED, payload={}, session_id="s")
    with pytest.raises(ValueError, match="approval"):
        PendingToolApproval.from_event(event)
