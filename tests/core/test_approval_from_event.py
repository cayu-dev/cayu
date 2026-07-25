"""Tests for ``PendingToolApproval.from_event`` (the nested-payload accessor)."""

from __future__ import annotations

import pytest

from cayu import Event, EventType, PendingToolApproval, PendingToolCallApproval


def _pending() -> PendingToolApproval:
    return PendingToolApproval(
        approval_id="ap_1",
        model_step_id=f"mstep_{'1' * 32}",
        model_attempt_id=f"matt_{'2' * 32}",
        tool_round_id=f"tround_{'3' * 32}",
        tool_call_id="call_1",
        tool_name="send_email",
        agent_name="assistant",
        tool_calls=[PendingToolCallApproval(tool_call_id="call_1", tool_name="send_email")],
    )


def _approval_event() -> Event:
    pending = _pending()
    return Event(
        type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
        payload={
            "approval_id": pending.approval_id,
            "tool_call_id": pending.tool_call_id,
            "model_step_id": pending.model_step_id,
            "model_attempt_id": pending.model_attempt_id,
            "tool_round_id": pending.tool_round_id,
            "approval": pending.model_dump(mode="json"),
        },
        session_id="sess_1",
    )


def test_from_event_reads_the_nested_approval() -> None:
    event = _approval_event()
    got = PendingToolApproval.from_event(event)
    assert got.approval_id == "ap_1"
    assert got.tool_call_id == "call_1"
    assert event.payload["approval_id"] == got.approval_id
    assert event.payload["tool_call_id"] == got.tool_call_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("approval_id", "other"),
        ("tool_call_id", "other"),
        ("model_step_id", f"mstep_{'4' * 32}"),
        ("model_attempt_id", f"matt_{'5' * 32}"),
        ("tool_round_id", f"tround_{'6' * 32}"),
    ],
)
def test_from_event_rejects_conflicting_direct_identity(field: str, value: str) -> None:
    event = _approval_event()
    event.payload[field] = value

    with pytest.raises(ValueError, match="identity"):
        PendingToolApproval.from_event(event)


def test_from_event_rejects_wrong_event_type() -> None:
    event = Event(type=EventType.SESSION_STARTED, payload={}, session_id="s")
    with pytest.raises(ValueError, match="approval_requested"):
        PendingToolApproval.from_event(event)


def test_from_event_rejects_missing_approval_payload() -> None:
    event = Event(type=EventType.TOOL_CALL_APPROVAL_REQUESTED, payload={}, session_id="s")
    with pytest.raises(ValueError, match="approval"):
        PendingToolApproval.from_event(event)
