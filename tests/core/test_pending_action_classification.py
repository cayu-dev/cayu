"""Headline classification preserves labels without conferring recovery authority."""

from copy import deepcopy
from datetime import UTC, datetime
from itertools import product

import pytest
from tests.core.pending_action_conformance import (
    _TOOL_ROUND_ID,
    _approval_checkpoint,
    _approval_request_event,
    _input_checkpoint,
    _round_checkpoint,
    _tool_round_identity_payload,
)

from cayu.approvals.user_input import user_input_lifecycle_authority_from_checkpoint
from cayu.events import Event, EventType
from cayu.runtime._session_finalization import _pending_interaction_action_kind
from cayu.sessions import _pending_approval_reader as approvals
from cayu.sessions import _pending_tool_round_reader as rounds
from cayu.sessions.pending_actions import (
    _pending_tool_action_kind,
    pending_action_evidence_round_from_checkpoint,
    pending_action_from_records,
    pending_action_source_is_invalid,
    project_pending_action_event_record,
)
from cayu.sessions.records import (
    EventRecord,
    PendingActionKind,
    PendingActionSession,
    SessionStatus,
)


def state(kind):
    session = PendingActionSession(
        id="session",
        instance_id="instance",
        runtime_name="cayu",
        created_at=datetime(2026, 10, 14, tzinfo=UTC),
        updated_at=datetime(2026, 10, 14, tzinfo=UTC),
        agent_name="assistant",
        provider_name="fake",
        model="fake",
        causal_budget_id="session",
        status=SessionStatus.INTERRUPTED,
    )
    if kind == "approval":
        checkpoint = _approval_checkpoint("approval", "call", "deploy")
        event = _approval_request_event(
            event_id="event",
            session_id=session.id,
            approval_id="approval",
            tool_call_id="call",
            tool_name="deploy",
        )
    elif kind == "input":
        checkpoint = _input_checkpoint("input", "call", "Continue?")
        checkpoint["pending_user_input"].update(
            session_id=session.id,
            session_instance_id=session.instance_id,
            source_interaction_id="interaction",
            source_run_epoch=0,
            execution_profile_fingerprint="e" * 64,
        )
        event = Event(
            type=EventType.SESSION_AWAITING_USER_INPUT,
            session_id=session.id,
            tool_name="ask_user",
            payload={
                **_tool_round_identity_payload(),
                "input_id": "input",
                "tool_call_id": "call",
                "question": "Continue?",
                "options": ["yes", "no"],
            },
        )
    else:
        checkpoint = _round_checkpoint(_TOOL_ROUND_ID, "call")
        event = Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id=session.id,
            tool_name="charge",
            payload={**_tool_round_identity_payload(), "tool_call_id": "call", "arguments": {}},
        )
    record = project_pending_action_event_record(EventRecord(sequence=1, event=event))
    return session, checkpoint, [record]


@pytest.mark.parametrize(
    "has_approval, has_input, has_round", list(product([False, True], repeat=3))
)
def test_headline_precedence_is_approval_then_input_then_recovery(
    has_approval, has_input, has_round
):
    approval = approvals.pending_approval_from_checkpoint(state("approval")[1])
    user_input, _ = user_input_lifecycle_authority_from_checkpoint(state("input")[1])
    tool_round = rounds.pending_tool_round_from_checkpoint(state("round")[1])
    kind = _pending_tool_action_kind(
        approval=approval if has_approval else None,
        user_input=user_input if has_input else None,
        tool_round=tool_round if has_round else None,
    )
    expected = (
        PendingActionKind.TOOL_APPROVAL
        if has_approval
        else PendingActionKind.USER_INPUT
        if has_input
        else PendingActionKind.MANUAL_RECOVERY
        if has_round
        else None
    )
    assert kind is expected


@pytest.mark.parametrize(
    "source, label, kind",
    [
        ("approval", "tool_approval", PendingActionKind.TOOL_APPROVAL),
        ("input", "user_input", PendingActionKind.USER_INPUT),
        ("round", "tool_recovery", PendingActionKind.MANUAL_RECOVERY),
    ],
)
def test_lifecycle_label_and_validated_action_agree(source, label, kind):
    session, checkpoint, records = state(source)
    original = deepcopy(checkpoint)
    assert _pending_interaction_action_kind(checkpoint, run_epoch=0) == label
    assert pending_action_evidence_round_from_checkpoint(checkpoint) is not None
    action = pending_action_from_records(session, records, checkpoint)
    assert action is not None and action.kind is kind
    assert not pending_action_source_is_invalid(session, checkpoint, action, records)
    assert checkpoint == original


@pytest.mark.parametrize("source", ["approval", "input", "round"])
def test_headline_without_matching_events_does_not_authorize_an_action(source):
    session, checkpoint, _ = state(source)
    assert _pending_interaction_action_kind(checkpoint, run_epoch=0) is not None
    assert pending_action_from_records(session, [], checkpoint) is None
    # A round with no started calls can replay automatically. Approval and input
    # require matching durable event evidence even when their headlines exist.
    assert pending_action_source_is_invalid(session, checkpoint, None, []) is (source != "round")


@pytest.mark.parametrize("source", ["approval", "input", "round"])
def test_event_labels_without_checkpoint_authority_do_not_create_an_action(source):
    session, _, records = state(source)
    assert _pending_interaction_action_kind(None, run_epoch=0) is None
    assert pending_action_from_records(session, records, None) is None


def test_conflicting_candidates_keep_headline_precedence_but_fail_rich_validation():
    session, checkpoint, records = state("approval")
    checkpoint.update(state("input")[1])
    assert _pending_interaction_action_kind(checkpoint, run_epoch=0) == "tool_approval"
    with pytest.raises(ValueError, match="more than one pending"):
        pending_action_evidence_round_from_checkpoint(checkpoint)
    action = pending_action_from_records(session, records, checkpoint)
    assert pending_action_source_is_invalid(session, checkpoint, action, records)


@pytest.mark.parametrize(
    "source, lower_key, label",
    [
        ("approval", "pending_user_input", "tool_approval"),
        ("input", "pending_tool_round", "user_input"),
    ],
)
def test_lower_priority_malformed_state_does_not_change_summary_read_order(
    source, lower_key, label
):
    session, checkpoint, records = state(source)
    checkpoint[lower_key] = {"malformed": True}
    assert _pending_interaction_action_kind(checkpoint, run_epoch=0) == label
    with pytest.raises(ValueError):
        pending_action_evidence_round_from_checkpoint(checkpoint)
    action = pending_action_from_records(session, records, checkpoint)
    assert pending_action_source_is_invalid(session, checkpoint, action, records)


@pytest.mark.parametrize(
    "source, key",
    [
        ("approval", "pending_tool_approval"),
        ("input", "pending_user_input"),
        ("round", "pending_tool_round"),
    ],
)
def test_malformed_highest_priority_candidate_is_rejected(source, key):
    session, checkpoint, records = state(source)
    checkpoint[key] = {"malformed": True}
    with pytest.raises(ValueError):
        _pending_interaction_action_kind(checkpoint, run_epoch=0)
    assert pending_action_from_records(session, records, checkpoint) is None
    assert pending_action_source_is_invalid(session, checkpoint, None, records)


def test_paired_approval_and_planned_round_keep_approval_headline():
    _, checkpoint, _ = state("approval")
    approval = approvals.pending_approval_from_checkpoint(checkpoint)
    assert approval is not None
    planned = approvals.planned_tool_round_from_pending_approval(approval)
    checkpoint["pending_tool_round"] = planned.model_dump(mode="json")
    assert _pending_interaction_action_kind(checkpoint, run_epoch=0) == "tool_approval"
    assert pending_action_evidence_round_from_checkpoint(checkpoint) == planned
    checkpoint["pending_tool_round"]["tool_round_id"] = "tround_" + "4" * 32
    with pytest.raises(ValueError, match="conflicting paired approval"):
        pending_action_evidence_round_from_checkpoint(checkpoint)
