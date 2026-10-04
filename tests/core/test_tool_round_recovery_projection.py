"""Secret quarantine preserves terminal evidence of whether a tool ran."""

import pytest

from cayu import Event, EventType
from cayu.approvals.tools import PendingToolCallApproval
from cayu.runtime import _tool_round_recovery as recovery
from cayu.runtime.execution_units import new_model_step_identity
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions._assistant_tool_round_publication import StagedToolCallTerminal
from cayu.tools.base import ToolResult


@pytest.mark.parametrize("projection", ["hook", "round", "round-then-hook"])
@pytest.mark.parametrize(
    "event_type",
    [
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
    ],
)
def test_quarantine_preserves_nonexecution_evidence(projection, event_type):
    identity = new_model_step_identity().new_attempt().new_tool_round()
    event = Event(
        type=event_type,
        session_id="recovery-projection",
        tool_name="tool",
        payload={
            **identity.payload(),
            "tool_call_id": "call",
            "idempotency_key": "original-key",
            "result": ToolResult(
                content="private-secret-canary",
                # Caller-controlled result fields cannot reclassify an executed event.
                structured={"executed": False, "outcome_unknown": False},
            ).model_dump(mode="json"),
        },
    )
    original = event.model_dump(mode="json")
    projected = event
    if projection != "hook":
        pending = pending_rounds.PendingToolRound(
            **identity.payload(),
            agent_name="assistant",
            tool_calls=[
                PendingToolCallApproval(tool_call_id="call", tool_name="tool", arguments={})
            ],
            staged_terminals=[StagedToolCallTerminal(tool_call_id="call", event=event)],
        )
        pending_before = pending.model_dump(mode="json")
        [stage] = recovery.staged_terminal_records(pending)
        projected = stage.event
        assert stage.hooks_state == "finalized"
        assert pending.model_dump(mode="json") == pending_before
    if projection != "round":
        projected = recovery.hook_scope_unavailable_recovery_event(projected)

    never_executed = event_type in {
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
    assert projected.type is (event_type if never_executed else EventType.TOOL_CALL_FAILED)
    assert projected.id == event.id
    assert projected.payload["tool_call_id"] == "call"
    assert projected.payload["idempotency_key"] == "original-key"
    assert identity.matches_payload(projected.payload)
    result = projected.payload["result"]
    assert result["is_error"] is True
    assert result["structured"]["outcome_unknown"] is not never_executed
    if never_executed:
        assert result["structured"]["executed"] is False
        # A generic quarantine error cannot acquire command-policy denial authority.
        assert "error" not in result["structured"]
    assert "private-secret-canary" not in projected.model_dump_json()
    assert event.model_dump(mode="json") == original
