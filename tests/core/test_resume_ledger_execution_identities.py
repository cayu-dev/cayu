from __future__ import annotations

import pytest

from cayu.core import Event, EventType, ToolResult
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime.approvals import (
    PendingToolApproval,
    PendingToolCallApproval,
    ToolApprovalDecision,
)
from cayu.runtime.execution_units import ToolRoundIdentity


def _identity() -> ToolRoundIdentity:
    return ToolRoundIdentity(
        model_step_id=f"mstep_{'1' * 32}",
        model_attempt_id=f"matt_{'2' * 32}",
        tool_round_id=f"tround_{'3' * 32}",
    )


def _pending_call() -> PendingToolCallApproval:
    return PendingToolCallApproval(
        tool_call_id="call-1",
        tool_name="side_effect",
    )


def _pending_approval() -> PendingToolApproval:
    identity = _identity()
    return PendingToolApproval(
        approval_id="approval-1",
        **identity.payload(),
        tool_call_id="call-1",
        tool_name="side_effect",
        agent_name="assistant",
        publish_arguments=True,
        tool_calls=[_pending_call()],
    )


def _tool_event(
    event_type: EventType,
    *,
    identity: ToolRoundIdentity | None = None,
    approval_id: str | None = None,
    input_id: str | None = None,
    manual_recovery: bool = False,
    tool_name: str = "side_effect",
) -> Event:
    effective_identity = _identity() if identity is None else identity
    payload: dict[str, object] = {
        **effective_identity.payload(),
        "tool_call_id": "call-1",
    }
    if approval_id is not None:
        payload["approval_id"] = approval_id
    if input_id is not None:
        payload["input_id"] = input_id
    if event_type == EventType.TOOL_CALL_STARTED:
        payload["arguments"] = {}
    else:
        payload["result"] = ToolResult(
            content=str(event_type),
            is_error=event_type != EventType.TOOL_CALL_COMPLETED,
        ).model_dump(mode="json")
    if manual_recovery:
        payload["manual_recovery"] = True
    return Event(
        type=event_type,
        session_id="session-1",
        tool_name=tool_name,
        payload=payload,
    )


def _conflicting_parent_identity() -> ToolRoundIdentity:
    identity = _identity()
    return identity.model_copy(update={"model_step_id": f"mstep_{'9' * 32}"})


def _pending_round() -> tool_round_recovery.PendingToolRound:
    identity = _identity()
    _, pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="assistant",
        environment_name=None,
        task_id=None,
        tool_calls=[
            runtime_records.ToolCallRequest(
                id="call-1",
                name="side_effect",
                arguments={},
            )
        ],
        policy_outcomes=None,
        structured_output=None,
        tool_round_identity=identity,
    )
    return pending_round


def test_approval_recovery_fails_closed_on_conflicting_parent_identity() -> None:
    approval = _pending_approval()
    started = _tool_event(
        EventType.TOOL_CALL_STARTED,
        identity=_conflicting_parent_identity(),
        approval_id=approval.approval_id,
    )

    with pytest.raises(
        approval_support.ToolApprovalManualRecoveryRequired,
        match="started without a terminal result",
    ):
        approval_support.recorded_tool_outcomes(
            events=[started],
            approval=approval,
        )

    approval_support.validate_recovery_target(
        events=[started],
        approval=approval,
        tool_call_id="call-1",
    )


def test_approval_recovery_rejects_unknown_call_in_exact_round() -> None:
    approval = _pending_approval()
    started = _tool_event(
        EventType.TOOL_CALL_STARTED,
        approval_id=approval.approval_id,
    )
    started.payload["tool_call_id"] = "unknown-call"

    with pytest.raises(
        resume_ledger.ToolCallEvidenceConflict,
        match="outside the pending tool round",
    ):
        approval_support.recorded_tool_outcomes(
            events=[started],
            approval=approval,
        )

    with pytest.raises(resume_ledger.ToolCallEvidenceConflict):
        approval_support.validate_recovery_target(
            events=[started],
            approval=approval,
            tool_call_id="call-1",
        )


def test_approval_recovery_fails_closed_when_exact_round_omits_approval_id() -> None:
    approval = _pending_approval()
    started = _tool_event(EventType.TOOL_CALL_STARTED)

    with pytest.raises(approval_support.ToolApprovalManualRecoveryRequired):
        approval_support.recorded_tool_outcomes(
            events=[started],
            approval=approval,
        )


def test_approval_resolution_history_accepts_exact_pending_call_descriptor() -> None:
    approval = _pending_approval()
    approved = Event(
        type=EventType.TOOL_CALL_APPROVED,
        session_id="session-1",
        agent_name=approval.agent_name,
        environment_name=approval.environment_name,
        tool_name=approval.tool_name,
        payload={
            **_identity().payload(),
            "approval_id": approval.approval_id,
            "tool_call_id": approval.tool_call_id,
        },
    )

    history = approval_support.approval_resolution_history(
        events=[approved],
        approval=approval,
    )

    assert history.has_approved_call is True
    assert history.has_granted_activity is True


def test_approval_resolution_history_tracks_pre_grant_resolution_attempt() -> None:
    approval = _pending_approval()
    resumed = Event(
        type=EventType.SESSION_RESUMED,
        session_id="session-1",
        agent_name=approval.agent_name,
        environment_name=approval.environment_name,
        payload={
            **_identity().payload(),
            "approval_id": approval.approval_id,
            "tool_call_id": approval.tool_call_id,
            "decision": ToolApprovalDecision.APPROVE.value,
        },
    )

    history = approval_support.approval_resolution_history(
        events=[resumed],
        approval=approval,
    )

    assert history.has_resolution_attempt is True
    assert history.has_resolution_activity is True
    assert history.has_granted_activity is False


def test_approval_resolution_history_rejects_conflicting_resolution_attempt() -> None:
    approval = _pending_approval()
    resumed = Event(
        type=EventType.SESSION_RESUMED,
        session_id="session-1",
        agent_name=approval.agent_name,
        environment_name=approval.environment_name,
        payload={
            **_identity().payload(),
            "approval_id": approval.approval_id,
            "tool_call_id": "different-call",
            "decision": ToolApprovalDecision.APPROVE.value,
        },
    )

    with pytest.raises(RuntimeError, match="contradictory resolution attempt"):
        approval_support.approval_resolution_history(
            events=[resumed],
            approval=approval,
        )


@pytest.mark.parametrize(
    ("tool_call_id", "tool_name", "agent_name", "environment_name"),
    [
        ("unknown-call", "side_effect", "assistant", None),
        ("call-1", "different_tool", "assistant", None),
        ("call-1", "side_effect", "different_agent", None),
        ("call-1", "side_effect", "assistant", "different_environment"),
    ],
)
def test_approval_resolution_history_rejects_conflicting_pending_call_descriptor(
    tool_call_id: str,
    tool_name: str,
    agent_name: str,
    environment_name: str | None,
) -> None:
    approval = _pending_approval()
    conflicting = Event(
        type=EventType.TOOL_CALL_APPROVED,
        session_id="session-1",
        agent_name=agent_name,
        environment_name=environment_name,
        tool_name=tool_name,
        payload={
            **_identity().payload(),
            "approval_id": approval.approval_id,
            "tool_call_id": tool_call_id,
        },
    )

    with pytest.raises(RuntimeError, match="contradictory tool-call descriptor"):
        approval_support.approval_resolution_history(
            events=[conflicting],
            approval=approval,
        )


@pytest.mark.parametrize(
    "payload_update",
    [
        {"approval_id": "different-approval"},
        {"model_attempt_id": f"matt_{'9' * 32}"},
    ],
)
def test_approval_resolution_history_rejects_partial_lifecycle_identity(
    payload_update: dict[str, str],
) -> None:
    approval = _pending_approval()
    payload = {
        **_identity().payload(),
        "approval_id": approval.approval_id,
        "tool_call_id": approval.tool_call_id,
        **payload_update,
    }
    conflicting = Event(
        type=EventType.TOOL_CALL_APPROVED,
        session_id="session-1",
        agent_name=approval.agent_name,
        tool_name=approval.tool_name,
        payload=payload,
    )

    with pytest.raises(RuntimeError, match="contradictory execution identity"):
        approval_support.approval_resolution_history(
            events=[conflicting],
            approval=approval,
        )


def test_user_input_recovery_fails_closed_on_conflicting_parent_identity() -> None:
    identity = _identity()
    input_id = "input-1"
    pause = Event(
        type=EventType.SESSION_AWAITING_USER_INPUT,
        session_id="session-1",
        payload={
            **identity.payload(),
            "input_id": input_id,
            "tool_call_id": "call-1",
        },
    )
    started = _tool_event(
        EventType.TOOL_CALL_STARTED,
        identity=_conflicting_parent_identity(),
        input_id=input_id,
    )

    with pytest.raises(
        approval_support.RoundToolManualRecoveryRequired,
        match="started without a terminal result",
    ):
        approval_support.recorded_round_tool_outcomes(
            events=[pause, started],
            pending_calls=[_pending_call()],
            input_id=input_id,
            tool_round_identity=identity,
        )

    approval_support.validate_round_recovery_target(
        events=[pause, started],
        pending_calls=[_pending_call()],
        tool_call_id="call-1",
        input_id=input_id,
        tool_round_identity=identity,
    )


def test_user_input_recovery_rejects_unknown_call_in_exact_round() -> None:
    identity = _identity()
    input_id = "input-1"
    pause = Event(
        type=EventType.SESSION_AWAITING_USER_INPUT,
        session_id="session-1",
        payload={
            **identity.payload(),
            "input_id": input_id,
            "tool_call_id": "call-1",
        },
    )
    started = _tool_event(
        EventType.TOOL_CALL_STARTED,
        input_id=input_id,
    )
    started.payload["tool_call_id"] = "unknown-call"

    with pytest.raises(
        resume_ledger.ToolCallEvidenceConflict,
        match="outside the pending tool round",
    ):
        approval_support.recorded_round_tool_outcomes(
            events=[pause, started],
            pending_calls=[_pending_call()],
            input_id=input_id,
            tool_round_identity=identity,
        )

    with pytest.raises(resume_ledger.ToolCallEvidenceConflict):
        approval_support.validate_round_recovery_target(
            events=[pause, started],
            pending_calls=[_pending_call()],
            tool_call_id="call-1",
            input_id=input_id,
            tool_round_identity=identity,
        )


def test_ordinary_round_rejects_unknown_call_in_exact_round() -> None:
    pending_round = _pending_round()
    started = _tool_event(EventType.TOOL_CALL_STARTED)
    started.payload["tool_call_id"] = "unknown-call"

    with pytest.raises(
        resume_ledger.ToolCallEvidenceConflict,
        match="outside the pending tool round",
    ):
        tool_round_recovery.recorded_tool_outcomes(
            events=[started],
            pending_round=pending_round,
        )

    with pytest.raises(resume_ledger.ToolCallEvidenceConflict):
        tool_round_recovery.validate_tool_round_recovery_target(
            events=[started],
            pending_round=pending_round,
            tool_call_id="call-1",
        )


def test_tool_round_recovery_target_accepts_known_sibling_evidence() -> None:
    identity = _identity()
    _, pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="assistant",
        environment_name=None,
        task_id=None,
        tool_calls=[
            runtime_records.ToolCallRequest(
                id=tool_call_id,
                name="side_effect",
                arguments={},
            )
            for tool_call_id in ("call-1", "call-2")
        ],
        policy_outcomes=None,
        structured_output=None,
        tool_round_identity=identity,
    )
    sibling_started = _tool_event(EventType.TOOL_CALL_STARTED)
    sibling_started.payload["tool_call_id"] = "call-2"

    tool_round_recovery.validate_tool_round_recovery_target(
        events=[
            _tool_event(EventType.TOOL_CALL_STARTED),
            sibling_started,
        ],
        pending_round=pending_round,
        tool_call_id="call-1",
    )


def test_ordinary_round_projects_conflicting_identity_as_unknown_started_work() -> None:
    identity = _identity()
    _, pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="assistant",
        environment_name=None,
        task_id=None,
        tool_calls=[
            runtime_records.ToolCallRequest(
                id="call-1",
                name="side_effect",
                arguments={},
            )
        ],
        policy_outcomes=None,
        structured_output=None,
        tool_round_identity=identity,
    )

    outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
        events=[
            _tool_event(
                EventType.TOOL_CALL_COMPLETED,
                identity=_conflicting_parent_identity(),
            )
        ],
        pending_round=pending_round,
    )

    assert outcomes == {}
    assert started_ids == {"call-1"}


def test_ordinary_round_projects_missing_round_id_as_unknown_started_work() -> None:
    identity = _identity()
    _, pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="assistant",
        environment_name=None,
        task_id=None,
        tool_calls=[
            runtime_records.ToolCallRequest(
                id="call-1",
                name="side_effect",
                arguments={},
            )
        ],
        policy_outcomes=None,
        structured_output=None,
        tool_round_identity=identity,
    )
    terminal = _tool_event(EventType.TOOL_CALL_COMPLETED)
    terminal.payload.pop("tool_round_id")

    outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
        events=[terminal],
        pending_round=pending_round,
    )

    assert outcomes == {}
    assert started_ids == {"call-1"}


def test_ordinary_round_rejects_terminal_for_different_tool_descriptor() -> None:
    identity = _identity()
    _, pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="assistant",
        environment_name=None,
        task_id=None,
        tool_calls=[
            runtime_records.ToolCallRequest(
                id="call-1",
                name="side_effect",
                arguments={},
            )
        ],
        policy_outcomes=None,
        structured_output=None,
        tool_round_identity=identity,
    )

    outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
        events=[
            _tool_event(
                EventType.TOOL_CALL_COMPLETED,
                tool_name="different_tool",
            )
        ],
        pending_round=pending_round,
    )

    assert outcomes == {}
    assert started_ids == {"call-1"}


def test_ordinary_round_rejects_started_event_for_different_arguments() -> None:
    identity = _identity()
    _, pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="assistant",
        environment_name=None,
        task_id=None,
        tool_calls=[
            runtime_records.ToolCallRequest(
                id="call-1",
                name="side_effect",
                arguments={"expected": True},
            )
        ],
        policy_outcomes=None,
        structured_output=None,
        tool_round_identity=identity,
    )
    started = _tool_event(EventType.TOOL_CALL_STARTED)
    started.payload["arguments"] = {"expected": False}

    outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
        events=[started],
        pending_round=pending_round,
    )

    assert outcomes == {}
    assert started_ids == {"call-1"}


def test_manual_recovery_cannot_override_a_conflicting_tool_descriptor() -> None:
    approval = _pending_approval()
    events = [
        _tool_event(
            EventType.TOOL_CALL_STARTED,
            approval_id=approval.approval_id,
        ),
        _tool_event(
            EventType.TOOL_CALL_COMPLETED,
            approval_id=approval.approval_id,
            manual_recovery=True,
            tool_name="different_tool",
        ),
    ]

    with pytest.raises(approval_support.ToolApprovalManualRecoveryRequired):
        approval_support.recorded_tool_outcomes(
            events=events,
            approval=approval,
        )


def test_duplicate_approval_terminal_evidence_requires_manual_recovery() -> None:
    approval = _pending_approval()
    terminal_events = [
        _tool_event(
            event_type,
            approval_id=approval.approval_id,
        )
        for event_type in (
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
        )
    ]

    with pytest.raises(approval_support.ToolApprovalManualRecoveryRequired):
        approval_support.recorded_tool_outcomes(
            events=terminal_events,
            approval=approval,
        )


def test_manual_recovery_supersedes_earlier_conflicting_identity() -> None:
    approval = _pending_approval()
    events = [
        _tool_event(
            EventType.TOOL_CALL_STARTED,
            identity=_conflicting_parent_identity(),
            approval_id=approval.approval_id,
        ),
        _tool_event(
            EventType.TOOL_CALL_COMPLETED,
            approval_id=approval.approval_id,
            manual_recovery=True,
        ),
    ]

    outcomes = approval_support.recorded_tool_outcomes(
        events=events,
        approval=approval,
    )

    assert outcomes["call-1"].result.content == str(EventType.TOOL_CALL_COMPLETED)


def test_malformed_terminal_evidence_requires_and_allows_manual_recovery() -> None:
    approval = _pending_approval()
    malformed = _tool_event(
        EventType.TOOL_CALL_COMPLETED,
        approval_id=approval.approval_id,
    )
    malformed.payload["result"] = {
        "content": 7,
        "is_error": False,
    }

    with pytest.raises(approval_support.ToolApprovalManualRecoveryRequired):
        approval_support.recorded_tool_outcomes(
            events=[malformed],
            approval=approval,
        )

    approval_support.validate_recovery_target(
        events=[malformed],
        approval=approval,
        tool_call_id="call-1",
    )

    recovered = _tool_event(
        EventType.TOOL_CALL_COMPLETED,
        approval_id=approval.approval_id,
        manual_recovery=True,
    )
    outcomes = approval_support.recorded_tool_outcomes(
        events=[malformed, recovered],
        approval=approval,
    )

    assert outcomes["call-1"].result.content == str(EventType.TOOL_CALL_COMPLETED)


def test_user_input_malformed_terminal_allows_manual_recovery() -> None:
    identity = _identity()
    input_id = "input-1"
    pause = Event(
        type=EventType.SESSION_AWAITING_USER_INPUT,
        session_id="session-1",
        payload={
            **identity.payload(),
            "input_id": input_id,
            "tool_call_id": "call-1",
        },
    )
    malformed = _tool_event(
        EventType.TOOL_CALL_COMPLETED,
        input_id=input_id,
    )
    malformed.payload["result"] = {"content": 7}

    with pytest.raises(approval_support.RoundToolManualRecoveryRequired):
        approval_support.recorded_round_tool_outcomes(
            events=[pause, malformed],
            pending_calls=[_pending_call()],
            input_id=input_id,
            tool_round_identity=identity,
        )

    approval_support.validate_round_recovery_target(
        events=[pause, malformed],
        pending_calls=[_pending_call()],
        tool_call_id="call-1",
        input_id=input_id,
        tool_round_identity=identity,
    )

    recovered = _tool_event(
        EventType.TOOL_CALL_COMPLETED,
        input_id=input_id,
        manual_recovery=True,
    )
    outcomes = approval_support.recorded_round_tool_outcomes(
        events=[pause, malformed, recovered],
        pending_calls=[_pending_call()],
        input_id=input_id,
        tool_round_identity=identity,
    )

    assert outcomes["call-1"].result.content == str(EventType.TOOL_CALL_COMPLETED)


def test_ordinary_round_malformed_terminal_allows_manual_recovery() -> None:
    identity = _identity()
    _, pending_round = tool_round_recovery.checkpoint_with_pending_tool_round(
        None,
        agent_name="assistant",
        environment_name=None,
        task_id=None,
        tool_calls=[
            runtime_records.ToolCallRequest(
                id="call-1",
                name="side_effect",
                arguments={},
            )
        ],
        policy_outcomes=None,
        structured_output=None,
        tool_round_identity=identity,
    )
    malformed = _tool_event(EventType.TOOL_CALL_COMPLETED)
    malformed.payload["result"] = {"content": 7}

    outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
        events=[malformed],
        pending_round=pending_round,
    )
    assert outcomes == {}
    assert started_ids == {"call-1"}

    tool_round_recovery.validate_tool_round_recovery_target(
        events=[malformed],
        pending_round=pending_round,
        tool_call_id="call-1",
    )

    recovered = _tool_event(
        EventType.TOOL_CALL_COMPLETED,
        manual_recovery=True,
    )
    outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
        events=[malformed, recovered],
        pending_round=pending_round,
    )

    assert outcomes["call-1"].result.content == str(EventType.TOOL_CALL_COMPLETED)
    assert started_ids == set()


def test_duplicate_manual_terminal_evidence_remains_conflicting() -> None:
    approval = _pending_approval()
    manual_events = [
        _tool_event(
            event_type,
            approval_id=approval.approval_id,
            manual_recovery=True,
        )
        for event_type in (
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
        )
    ]

    with pytest.raises(approval_support.ToolApprovalManualRecoveryRequired):
        approval_support.recorded_tool_outcomes(
            events=manual_events,
            approval=approval,
        )

    approval_support.validate_recovery_target(
        events=manual_events,
        approval=approval,
        tool_call_id="call-1",
    )
