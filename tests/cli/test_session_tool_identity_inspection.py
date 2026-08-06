from __future__ import annotations

import pytest

from cayu.cli.session import _tool_call_rows, _tool_inspection_record
from cayu.core import Event, EventType, ToolResult
from cayu.runtime import EventRecord


def _identity() -> dict[str, str]:
    return {
        "model_step_id": f"mstep_{'1' * 32}",
        "model_attempt_id": f"matt_{'2' * 32}",
        "tool_round_id": f"tround_{'3' * 32}",
    }


def _record(
    sequence: int,
    event_type: EventType,
    *,
    payload_extra: dict[str, object] | None = None,
    result: ToolResult | None = None,
    tool_name: str = "side_effect",
) -> EventRecord:
    payload: dict[str, object] = {
        **_identity(),
        "tool_call_id": "call-1",
    }
    if payload_extra is not None:
        payload.update(payload_extra)
    if (
        event_type == EventType.TOOL_CALL_STARTED
        and "arguments" not in payload
        and "arguments_state" not in payload
    ):
        payload["arguments"] = {}
    if result is not None:
        payload["result"] = result.model_dump(mode="json")
    return _tool_inspection_record(
        EventRecord(
            sequence=sequence,
            event=Event(
                type=event_type,
                session_id="session-1",
                tool_name=tool_name,
                payload=payload,
            ),
        )
    )


def _approval_request(
    *,
    approval_id: str = "approval-1",
    tool_calls: list[dict[str, object]] | None = None,
) -> EventRecord:
    nested_calls = [] if tool_calls is None else tool_calls
    return _tool_inspection_record(
        EventRecord(
            sequence=1,
            event=Event(
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id="session-1",
                tool_name="side_effect",
                payload={
                    **_identity(),
                    "approval_id": approval_id,
                    "tool_call_id": "call-1",
                    "approval": {
                        **_identity(),
                        "approval_id": approval_id,
                        "tool_call_id": "call-1",
                        "tool_name": "side_effect",
                        "arguments": {},
                        "tool_calls": nested_calls,
                    },
                },
            ),
        )
    )


def test_tool_inspection_marks_conflicting_terminal_outcomes_unavailable() -> None:
    rows = _tool_call_rows(
        [
            _record(1, EventType.TOOL_CALL_STARTED),
            _record(
                2,
                EventType.TOOL_CALL_COMPLETED,
                result=ToolResult(content="completed"),
            ),
            _record(
                3,
                EventType.TOOL_CALL_FAILED,
                result=ToolResult(content="failed", is_error=True),
            ),
        ]
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "unavailable"
    assert row["completed_at"] is None
    assert row["duration_ms"] is None
    assert row["rendered_content_bytes"] is None
    assert row["structured_result_bytes"] is None
    assert row["artifact_bytes"] is None


def test_tool_inspection_marks_terminal_before_start_unavailable() -> None:
    rows = _tool_call_rows(
        [
            _record(
                1,
                EventType.TOOL_CALL_COMPLETED,
                result=ToolResult(content="completed"),
            ),
            _record(2, EventType.TOOL_CALL_STARTED),
        ]
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "unavailable"


@pytest.mark.parametrize(
    ("records", "expected_state", "expected_summary"),
    [
        (
            [
                _record(
                    1,
                    EventType.TOOL_CALL_STARTED,
                    payload_extra={"arguments_state": "quarantined"},
                )
            ],
            "quarantined",
            "[quarantined]",
        ),
        (
            [
                _record(
                    1,
                    EventType.TOOL_CALL_STARTED,
                    payload_extra={"arguments_state": "quarantined"},
                ),
                _record(
                    2,
                    EventType.TOOL_CALL_FAILED,
                    payload_extra={"arguments_state": "unavailable"},
                ),
            ],
            "unavailable",
            "[unavailable]",
        ),
        (
            [
                _record(
                    1,
                    EventType.TOOL_CALL_COMPLETED,
                    payload_extra={"arguments_state": "finalized", "arguments": {}},
                )
            ],
            "finalized",
            "{}",
        ),
    ],
)
def test_tool_inspection_distinguishes_argument_publication_states(
    records: list[EventRecord],
    expected_state: str,
    expected_summary: str,
) -> None:
    rows = _tool_call_rows(records)

    assert len(rows) == 1
    assert rows[0]["arguments_state"] == expected_state
    assert rows[0]["argument_summary"] == expected_summary


def test_tool_inspection_preserves_legacy_argument_summary_without_inventing_state() -> None:
    rows = _tool_call_rows(
        [
            _record(
                1,
                EventType.TOOL_CALL_STARTED,
                payload_extra={"arguments": {"value": "legacy"}},
            )
        ]
    )

    assert rows[0]["arguments_state"] is None
    assert rows[0]["argument_summary"] == '{"value":"legacy"}'


def test_tool_inspection_legacy_terminal_does_not_hide_started_arguments() -> None:
    rows = _tool_call_rows(
        [
            _record(
                1,
                EventType.TOOL_CALL_STARTED,
                payload_extra={"arguments": {"value": "legacy"}},
            ),
            _record(2, EventType.TOOL_CALL_COMPLETED),
        ]
    )

    assert rows[0]["arguments_state"] is None
    assert rows[0]["argument_summary"] == '{"value":"legacy"}'


def test_tool_inspection_preserves_quarantine_state_from_approval_descriptor() -> None:
    approval = _tool_inspection_record(
        EventRecord(
            sequence=1,
            event=Event(
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id="session-1",
                tool_name="side_effect",
                payload={
                    **_identity(),
                    "approval_id": "approval-1",
                    "tool_call_id": "call-1",
                    "approval": {
                        **_identity(),
                        "approval_id": "approval-1",
                        "tool_call_id": "call-1",
                        "tool_name": "side_effect",
                        "arguments_state": "quarantined",
                        "tool_calls": [],
                    },
                },
            ),
        )
    )

    row = _tool_call_rows([approval])[0]
    assert row["arguments_state"] == "quarantined"
    assert row["argument_summary"] == "[quarantined]"


def test_tool_inspection_preserves_unavailable_state_from_user_input_descriptor() -> None:
    awaiting_input = _tool_inspection_record(
        EventRecord(
            sequence=1,
            event=Event(
                type=EventType.SESSION_AWAITING_USER_INPUT,
                session_id="session-1",
                tool_name="ask_user",
                payload={
                    **_identity(),
                    "input_id": "input-1",
                    "tool_call_id": "call-1",
                    "question": "Continue?",
                    "tool_calls": [
                        {
                            "tool_call_id": "call-1",
                            "tool_name": "ask_user",
                            "arguments_state": "unavailable",
                        }
                    ],
                },
            ),
        )
    )

    row = _tool_call_rows([awaiting_input])[0]
    assert row["arguments_state"] == "unavailable"
    assert row["argument_summary"] == "[unavailable]"


def test_tool_inspection_marks_conflicting_tool_descriptor_unavailable() -> None:
    rows = _tool_call_rows(
        [
            _record(1, EventType.TOOL_CALL_STARTED),
            _record(
                2,
                EventType.TOOL_CALL_COMPLETED,
                result=ToolResult(content="completed"),
                tool_name="different_tool",
            ),
        ]
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["completed_at"] is None


def test_tool_inspection_marks_conflicting_approval_decisions_unavailable() -> None:
    rows = _tool_call_rows(
        [
            _record(1, EventType.TOOL_CALL_APPROVED),
            _record(
                2,
                EventType.TOOL_CALL_APPROVAL_DENIED,
                result=ToolResult(content="denied", is_error=True),
            ),
        ]
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["approval_state"] == "unavailable"


def test_tool_inspection_rejects_decision_for_different_approval_identity() -> None:
    rows = _tool_call_rows(
        [
            _approval_request(),
            _record(
                2,
                EventType.TOOL_CALL_APPROVED,
                payload_extra={"approval_id": "approval-2"},
            ),
        ]
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["approval_state"] == "unavailable"


def test_tool_inspection_rejects_missing_approval_identity_after_request() -> None:
    rows = _tool_call_rows(
        [
            _approval_request(),
            _record(2, EventType.TOOL_CALL_APPROVED),
        ]
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["approval_state"] == "unavailable"


def test_tool_inspection_rejects_duplicate_nested_approval_calls() -> None:
    rows = _tool_call_rows(
        [
            _approval_request(
                tool_calls=[
                    {
                        "tool_call_id": "call-1",
                        "tool_name": "side_effect",
                        "arguments": {"value": 1},
                    },
                    {
                        "tool_call_id": "call-1",
                        "tool_name": "different_tool",
                        "arguments": {"value": 2},
                    },
                ]
            )
        ]
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["approval_state"] == "unavailable"


def test_tool_inspection_scopes_approval_state_to_gated_calls_in_mixed_round() -> None:
    approval_id = "approval-1"
    result = ToolResult(content="done")
    approval_request = _approval_request(
        approval_id=approval_id,
        tool_calls=[
            {
                "tool_call_id": "call-denied",
                "tool_name": "denied_tool",
                "arguments": {},
                "policy_decision": "deny",
            },
            {
                "tool_call_id": "call-1",
                "tool_name": "side_effect",
                "arguments": {},
                "policy_decision": "require_approval",
            },
            {
                "tool_call_id": "call-allowed",
                "tool_name": "allowed_tool",
                "arguments": {},
                "policy_decision": "allow",
            },
        ],
    )
    pending_rows = _tool_call_rows([approval_request])
    pending_by_call_id = {row["tool_call_id"]: row for row in pending_rows}

    assert {row["status"] for row in pending_rows} == {"approval_pending"}
    assert pending_by_call_id["call-denied"]["approval_state"] == "none"
    assert pending_by_call_id["call-1"]["approval_state"] == "requested"
    assert pending_by_call_id["call-allowed"]["approval_state"] == "none"

    rows = _tool_call_rows(
        [
            approval_request,
            _record(
                2,
                EventType.TOOL_CALL_BLOCKED,
                payload_extra={
                    "approval_id": approval_id,
                    "tool_call_id": "call-denied",
                },
                result=ToolResult(content="blocked", is_error=True),
                tool_name="denied_tool",
            ),
            _record(
                3,
                EventType.TOOL_CALL_APPROVED,
                payload_extra={"approval_id": approval_id},
            ),
            _record(
                4,
                EventType.TOOL_CALL_STARTED,
                payload_extra={"approval_id": approval_id},
            ),
            _record(
                5,
                EventType.TOOL_CALL_COMPLETED,
                payload_extra={"approval_id": approval_id},
                result=result,
            ),
            _record(
                6,
                EventType.TOOL_CALL_STARTED,
                payload_extra={
                    "approval_id": approval_id,
                    "tool_call_id": "call-allowed",
                },
                tool_name="allowed_tool",
            ),
            _record(
                7,
                EventType.TOOL_CALL_COMPLETED,
                payload_extra={
                    "approval_id": approval_id,
                    "tool_call_id": "call-allowed",
                },
                result=result,
                tool_name="allowed_tool",
            ),
        ]
    )

    assert len(rows) == 3
    rows_by_call_id = {row["tool_call_id"]: row for row in rows}
    assert rows_by_call_id["call-denied"]["status"] == "blocked"
    assert rows_by_call_id["call-denied"]["approval_state"] == "none"
    assert rows_by_call_id["call-1"]["status"] == "success"
    assert rows_by_call_id["call-1"]["approval_state"] == "approved"
    assert rows_by_call_id["call-allowed"]["status"] == "success"
    assert rows_by_call_id["call-allowed"]["approval_state"] == "none"


def test_tool_inspection_projects_ambiguous_acknowledgement_as_blocked() -> None:
    approval_id = "approval-1"
    request = _approval_request(
        approval_id=approval_id,
        tool_calls=[
            {
                "tool_call_id": "call-1",
                "tool_name": "side_effect",
                "arguments": {},
                "policy_evidence": "ambiguous",
                "policy_decision": None,
            },
            {
                "tool_call_id": "call-sibling",
                "tool_name": "sibling_tool",
                "arguments": {},
                "policy_evidence": "ambiguous",
                "policy_decision": None,
            },
        ],
    )
    blocked = _record(
        2,
        EventType.TOOL_CALL_BLOCKED,
        payload_extra={
            "approval_id": approval_id,
            "decision": "ambiguous",
            "blocked_by": "policy_evaluation_ambiguous",
            "requested_decision": "approve",
        },
        result=ToolResult(content="not executed", is_error=True),
    )
    sibling_blocked = _record(
        3,
        EventType.TOOL_CALL_BLOCKED,
        payload_extra={
            "approval_id": approval_id,
            "tool_call_id": "call-sibling",
            "decision": "ambiguous",
            "blocked_by": "policy_evaluation_ambiguous",
            "requested_decision": "approve",
        },
        result=ToolResult(content="not executed", is_error=True),
        tool_name="sibling_tool",
    )

    rows = _tool_call_rows([request, blocked, sibling_blocked])

    rows_by_call_id = {row["tool_call_id"]: row for row in rows}
    gating = rows_by_call_id["call-1"]
    assert gating["status"] == "blocked"
    assert gating["approval_state"] == "blocked"
    assert gating["completed_at"] == blocked.event.timestamp.isoformat()
    sibling = rows_by_call_id["call-sibling"]
    assert sibling["status"] == "blocked"
    assert sibling["approval_state"] == "none"

    bounded_rows = _tool_call_rows([blocked])
    assert len(bounded_rows) == 1
    assert bounded_rows[0]["status"] == "blocked"
    assert bounded_rows[0]["approval_state"] == "none"


def test_tool_inspection_rejects_malformed_ambiguous_resolution_marker() -> None:
    request = _approval_request(
        tool_calls=[
            {
                "tool_call_id": "call-1",
                "tool_name": "side_effect",
                "arguments": {},
                "policy_evidence": "ambiguous",
                "policy_decision": None,
            }
        ]
    )
    malformed = _record(
        2,
        EventType.TOOL_CALL_BLOCKED,
        payload_extra={
            "approval_id": "approval-1",
            "decision": "ambiguous",
            "blocked_by": "different_boundary",
            "requested_decision": "approve",
        },
        result=ToolResult(content="not executed", is_error=True),
    )

    rows = _tool_call_rows([request, malformed])

    assert len(rows) == 1
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["approval_state"] == "unavailable"


def test_tool_inspection_accepts_unregistered_sibling_policy_evidence() -> None:
    approval_id = "approval-1"
    request = _approval_request(
        approval_id=approval_id,
        tool_calls=[
            {
                "tool_call_id": "call-1",
                "tool_name": "side_effect",
                "arguments": {},
                "policy_evidence": "authoritative",
                "policy_decision": "require_approval",
            },
            {
                "tool_call_id": "call-unregistered",
                "tool_name": "missing_tool",
                "arguments": {},
                "policy_evidence": "unregistered",
                "policy_decision": None,
            },
        ],
    )
    approved = _record(
        2,
        EventType.TOOL_CALL_APPROVED,
        payload_extra={"approval_id": approval_id},
    )
    unregistered = _record(
        3,
        EventType.TOOL_CALL_FAILED,
        payload_extra={
            "approval_id": approval_id,
            "tool_call_id": "call-unregistered",
            "registration_state": "unregistered_at_policy_plan",
        },
        result=ToolResult(
            content="Tool was not registered when the policy plan was recorded.",
            structured={"registration_state": "unregistered_at_policy_plan"},
            is_error=True,
        ),
        tool_name="missing_tool",
    )

    rows = _tool_call_rows([request, approved, unregistered])

    rows_by_call_id = {row["tool_call_id"]: row for row in rows}
    missing = rows_by_call_id["call-unregistered"]
    assert missing["status"] == "error"
    assert missing["approval_state"] == "none"
    assert missing["completed_at"] == unregistered.event.timestamp.isoformat()
    assert rows_by_call_id["call-1"]["approval_state"] == "approved"

    denied_gate = _record(
        4,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        payload_extra={
            "approval_id": approval_id,
            "approval_required": True,
        },
        result=ToolResult(content="denied", is_error=True),
    )
    denied_unregistered = _record(
        5,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        payload_extra={
            "approval_id": approval_id,
            "tool_call_id": "call-unregistered",
            "approval_required": False,
        },
        result=ToolResult(content="not executed", is_error=True),
        tool_name="missing_tool",
    )

    denied_rows = _tool_call_rows([request, denied_gate, denied_unregistered])

    denied_by_call_id = {row["tool_call_id"]: row for row in denied_rows}
    missing = denied_by_call_id["call-unregistered"]
    assert missing["status"] == "denied"
    assert missing["approval_state"] == "none"
    assert missing["completed_at"] == denied_unregistered.event.timestamp.isoformat()
    assert denied_by_call_id["call-1"]["approval_state"] == "denied"


@pytest.mark.parametrize("reverse_events", [False, True], ids=["gate-first", "sibling-first"])
def test_tool_inspection_accepts_denied_ambiguous_sibling(
    reverse_events: bool,
) -> None:
    approval_id = "approval-ambiguous"
    request = _approval_request(
        approval_id=approval_id,
        tool_calls=[
            {
                "tool_call_id": "call-1",
                "tool_name": "side_effect",
                "arguments": {},
                "policy_evidence": "ambiguous",
                "policy_decision": None,
            },
            {
                "tool_call_id": "call-sibling",
                "tool_name": "sibling_effect",
                "arguments": {},
                "policy_evidence": "ambiguous",
                "policy_decision": None,
            },
        ],
    )
    gate_denial = _record(
        2,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        payload_extra={
            "approval_id": approval_id,
            "approval_required": True,
        },
        result=ToolResult(content="denied", is_error=True),
    )
    sibling_denial = _record(
        3,
        EventType.TOOL_CALL_APPROVAL_DENIED,
        payload_extra={
            "approval_id": approval_id,
            "tool_call_id": "call-sibling",
            "approval_required": False,
        },
        result=ToolResult(content="not executed", is_error=True),
        tool_name="sibling_effect",
    )
    decisions = [gate_denial, sibling_denial]
    if reverse_events:
        decisions.reverse()

    rows = _tool_call_rows([request, *decisions])

    rows_by_call_id = {row["tool_call_id"]: row for row in rows}
    assert rows_by_call_id["call-1"]["status"] == "denied"
    assert rows_by_call_id["call-1"]["approval_state"] == "denied"
    sibling = rows_by_call_id["call-sibling"]
    assert sibling["status"] == "denied"
    assert sibling["approval_state"] == "none"
    assert sibling["completed_at"] == sibling_denial.event.timestamp.isoformat()


def test_tool_inspection_rejects_executed_unregistered_sibling_in_any_event_order() -> None:
    approval_id = "approval-1"
    request = _approval_request(
        approval_id=approval_id,
        tool_calls=[
            {
                "tool_call_id": "call-1",
                "tool_name": "side_effect",
                "arguments": {},
                "policy_evidence": "authoritative",
                "policy_decision": "require_approval",
            },
            {
                "tool_call_id": "call-unregistered",
                "tool_name": "missing_tool",
                "arguments": {},
                "policy_evidence": "unregistered",
                "policy_decision": None,
            },
        ],
    )
    approved = _record(
        2,
        EventType.TOOL_CALL_APPROVED,
        payload_extra={"approval_id": approval_id},
    )
    started = _record(
        3,
        EventType.TOOL_CALL_STARTED,
        payload_extra={
            "approval_id": approval_id,
            "tool_call_id": "call-unregistered",
        },
        tool_name="missing_tool",
    )
    completed = _record(
        4,
        EventType.TOOL_CALL_COMPLETED,
        payload_extra={
            "approval_id": approval_id,
            "tool_call_id": "call-unregistered",
        },
        result=ToolResult(content="must not execute"),
        tool_name="missing_tool",
    )

    for records in (
        [request, approved, started, completed],
        [completed, started, approved, request],
    ):
        rows = _tool_call_rows(records)
        row = next(item for item in rows if item["tool_call_id"] == "call-unregistered")
        assert row["status"] == "unavailable"
        assert row["approval_state"] == "unavailable"
        assert row["completed_at"] is None


def test_tool_inspection_scopes_denial_state_to_gated_calls_in_mixed_round() -> None:
    approval_id = "approval-1"
    approval_request = _approval_request(
        approval_id=approval_id,
        tool_calls=[
            {
                "tool_call_id": "call-denied",
                "tool_name": "denied_tool",
                "arguments": {},
                "policy_decision": "deny",
            },
            {
                "tool_call_id": "call-1",
                "tool_name": "side_effect",
                "arguments": {},
                "policy_decision": "require_approval",
            },
            {
                "tool_call_id": "call-allowed",
                "tool_name": "allowed_tool",
                "arguments": {},
                "policy_decision": "allow",
            },
        ],
    )

    rows = _tool_call_rows(
        [
            approval_request,
            _record(
                2,
                EventType.TOOL_CALL_BLOCKED,
                payload_extra={
                    "approval_id": approval_id,
                    "tool_call_id": "call-denied",
                },
                result=ToolResult(content="blocked", is_error=True),
                tool_name="denied_tool",
            ),
            _record(
                3,
                EventType.TOOL_CALL_APPROVAL_DENIED,
                payload_extra={
                    "approval_id": approval_id,
                    "approval_required": True,
                },
                result=ToolResult(content="denied", is_error=True),
            ),
            _record(
                4,
                EventType.TOOL_CALL_APPROVAL_DENIED,
                payload_extra={
                    "approval_id": approval_id,
                    "tool_call_id": "call-allowed",
                    "approval_required": False,
                },
                result=ToolResult(content="skipped", is_error=True),
                tool_name="allowed_tool",
            ),
        ]
    )

    rows_by_call_id = {row["tool_call_id"]: row for row in rows}
    assert rows_by_call_id["call-denied"]["status"] == "blocked"
    assert rows_by_call_id["call-denied"]["approval_state"] == "none"
    assert rows_by_call_id["call-1"]["status"] == "denied"
    assert rows_by_call_id["call-1"]["approval_state"] == "denied"
    assert rows_by_call_id["call-allowed"]["status"] == "denied"
    assert rows_by_call_id["call-allowed"]["approval_state"] == "none"


def test_tool_inspection_rejects_unknown_call_approval_as_round_resolution() -> None:
    approval_id = "approval-1"
    rows = _tool_call_rows(
        [
            _approval_request(
                approval_id=approval_id,
                tool_calls=[
                    {
                        "tool_call_id": "call-1",
                        "tool_name": "side_effect",
                        "arguments": {},
                        "policy_decision": "require_approval",
                    },
                    {
                        "tool_call_id": "call-allowed",
                        "tool_name": "allowed_tool",
                        "arguments": {},
                        "policy_decision": "allow",
                    },
                ],
            ),
            _record(
                2,
                EventType.TOOL_CALL_APPROVED,
                payload_extra={
                    "approval_id": approval_id,
                    "tool_call_id": "call-ghost",
                },
                tool_name="ghost_tool",
            ),
        ]
    )

    rows_by_call_id = {row["tool_call_id"]: row for row in rows}
    assert rows_by_call_id["call-1"]["status"] == "approval_pending"
    assert rows_by_call_id["call-1"]["approval_state"] == "requested"
    assert rows_by_call_id["call-allowed"]["status"] == "approval_pending"
    assert rows_by_call_id["call-allowed"]["approval_state"] == "none"
    assert rows_by_call_id["call-ghost"]["status"] == "unavailable"
    assert rows_by_call_id["call-ghost"]["approval_state"] == "unavailable"


def test_tool_inspection_rejects_unknown_nested_policy_decision() -> None:
    rows = _tool_call_rows(
        [
            _approval_request(
                tool_calls=[
                    {
                        "tool_call_id": "call-1",
                        "tool_name": "side_effect",
                        "arguments": {},
                        "policy_decision": "unexpected",
                    }
                ]
            )
        ]
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["approval_state"] == "unavailable"


def test_tool_inspection_folds_expiry_marker_and_denial_as_one_decision() -> None:
    rows = _tool_call_rows(
        [
            _record(
                1,
                EventType.TOOL_CALL_APPROVAL_EXPIRED,
                payload_extra={"approval_id": "approval-1"},
            ),
            _record(
                2,
                EventType.TOOL_CALL_APPROVAL_DENIED,
                payload_extra={
                    "approval_id": "approval-1",
                    "expired": True,
                },
                result=ToolResult(content="expired", is_error=True),
            ),
        ]
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "expired"
    assert rows[0]["approval_state"] == "expired"


def test_tool_inspection_rejects_expiry_and_denial_for_different_approvals() -> None:
    rows = _tool_call_rows(
        [
            _record(
                1,
                EventType.TOOL_CALL_APPROVAL_EXPIRED,
                payload_extra={"approval_id": "approval-1"},
            ),
            _record(
                2,
                EventType.TOOL_CALL_APPROVAL_DENIED,
                payload_extra={
                    "approval_id": "approval-2",
                    "expired": True,
                },
                result=ToolResult(content="expired", is_error=True),
            ),
        ]
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["approval_state"] == "unavailable"


def test_tool_inspection_marks_conflicting_round_parent_identities_unavailable() -> None:
    started = _record(1, EventType.TOOL_CALL_STARTED)
    terminal = _record(
        2,
        EventType.TOOL_CALL_COMPLETED,
        result=ToolResult(content="completed"),
    )
    terminal.event.payload["model_step_id"] = f"mstep_{'9' * 32}"

    rows = _tool_call_rows([started, terminal])

    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"unavailable"}
    assert all(row["completed_at"] is None for row in rows)


def test_tool_inspection_marks_attempt_with_multiple_rounds_unavailable() -> None:
    first = _record(1, EventType.TOOL_CALL_STARTED)
    second = _record(2, EventType.TOOL_CALL_STARTED)
    second.event.payload["tool_round_id"] = f"tround_{'8' * 32}"

    rows = _tool_call_rows([first, second])

    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"unavailable"}


def test_tool_inspection_marks_malformed_identity_evidence_unavailable() -> None:
    terminal = _record(
        1,
        EventType.TOOL_CALL_COMPLETED,
        result=ToolResult(content="completed"),
    )
    terminal.event.payload["tool_round_id"] = "malformed-round-id"

    rows = _tool_call_rows([terminal])

    assert len(rows) == 1
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["completed_at"] is None


def test_tool_inspection_marks_conflicting_nested_approval_identity_unavailable() -> None:
    event_identity = _identity()
    nested_identity = {
        "model_step_id": f"mstep_{'4' * 32}",
        "model_attempt_id": f"matt_{'5' * 32}",
        "tool_round_id": f"tround_{'6' * 32}",
    }
    record = _tool_inspection_record(
        EventRecord(
            sequence=1,
            event=Event(
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id="session-1",
                tool_name="side_effect",
                payload={
                    **event_identity,
                    "approval_id": "approval-1",
                    "tool_call_id": "call-1",
                    "approval": {
                        **nested_identity,
                        "approval_id": "approval-1",
                        "tool_call_id": "call-1",
                        "tool_name": "side_effect",
                        "arguments": {},
                        "tool_calls": [],
                    },
                },
            ),
        )
    )

    rows = _tool_call_rows([record])

    assert len(rows) == 1
    assert rows[0]["status"] == "unavailable"
    assert rows[0]["model_step_id"] is None
    assert rows[0]["model_attempt_id"] is None
    assert rows[0]["tool_round_id"] is None
