"""Runtime evidence tells an expired approval from one a person denied.

When an approval expires, the runtime emits ``tool.call.approval_expired`` and
then runs the ordinary denial path, which emits ``tool.call.approval_denied``
with ``expired: true`` (docs/runtime-contracts.md, "A policy demanding approval
can also bound how long that demand stays valid"). The denial is the last
approval event of an expired call, so the evidence projection must read either
signal as ``expired`` rather than let the denial's event type decide. A denial
with neither signal is a person's decision and stays ``denied``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_tool_support import ExpiringApprovalPolicy, SideEffectTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.evidence import (
    RuntimeEvidenceReport,
    RuntimeEvidenceRequest,
    RuntimeEvidenceWarningCode,
    runtime_evidence,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest

_SESSION_ID = "expired-approval-evidence"
_APPROVAL_EVENT_TYPES = {
    EventType.TOOL_CALL_APPROVED,
    EventType.TOOL_CALL_APPROVAL_DENIED,
    EventType.TOOL_CALL_APPROVAL_EXPIRED,
}


async def _resolve_one_approval(
    *,
    decision: ToolApprovalDecision,
    answered_after: timedelta,
) -> tuple[list[EventType], SideEffectTool, RuntimeEvidenceReport]:
    """Run one gated call whose approval lasts 60 seconds, then resolve it."""

    clock = {"now": datetime(2026, 7, 9, 12, 0, tzinfo=UTC)}
    tool = SideEffectTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("resolution handled"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(clock=lambda: clock["now"], enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=ExpiringApprovalPolicy(expires_in_seconds=60),
    )
    paused = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=_SESSION_ID,
                messages=[Message.text("user", "Use the tool")],
            )
        )
    ]
    requested = next(e for e in paused if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)

    clock["now"] += answered_after
    resolved = [
        event
        async for event in app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=_SESSION_ID,
                approval_id=requested.payload["approval_id"],
                tool_round_id=requested.payload["tool_round_id"],
                tool_call_id=requested.payload["tool_call_id"],
                decision=decision,
            )
        )
    ]
    report = await runtime_evidence(
        app,
        RuntimeEvidenceRequest(root_session_id=_SESSION_ID, max_sessions=10, max_events=200),
    )
    approval_events = [event.type for event in resolved if event.type in _APPROVAL_EVENT_TYPES]
    return approval_events, tool, report


@pytest.mark.parametrize(
    "requested_decision", [ToolApprovalDecision.APPROVE, ToolApprovalDecision.DENY]
)
def test_runtime_evidence_reports_an_expired_approval_as_expired(
    requested_decision: ToolApprovalDecision,
) -> None:
    approval_events, tool, report = asyncio.run(
        _resolve_one_approval(decision=requested_decision, answered_after=timedelta(minutes=2))
    )

    # The contract's order: the expiry marker, then the denial that closes the call.
    # An expired resolution ends the same way whatever the caller asked for.
    assert approval_events == [
        EventType.TOOL_CALL_APPROVAL_EXPIRED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    ]
    assert tool.calls == []
    [session] = report.sessions
    [approval] = session.approvals
    assert approval.decision == "expired"
    assert RuntimeEvidenceWarningCode.MALFORMED_APPROVAL not in {
        warning.code for warning in report.warnings
    }


def test_runtime_evidence_reports_a_denial_by_a_person_as_denied() -> None:
    approval_events, tool, report = asyncio.run(
        _resolve_one_approval(
            decision=ToolApprovalDecision.DENY, answered_after=timedelta(seconds=30)
        )
    )

    assert approval_events == [EventType.TOOL_CALL_APPROVAL_DENIED]
    assert tool.calls == []
    [session] = report.sessions
    [approval] = session.approvals
    assert approval.decision == "denied"


def test_runtime_evidence_reports_an_approval_in_time_as_approved() -> None:
    approval_events, tool, report = asyncio.run(
        _resolve_one_approval(
            decision=ToolApprovalDecision.APPROVE, answered_after=timedelta(seconds=30)
        )
    )

    assert approval_events == [EventType.TOOL_CALL_APPROVED]
    assert tool.calls == [{}]
    [session] = report.sessions
    [approval] = session.approvals
    assert approval.decision == "approved"


def _approval_event(
    event_id: str,
    event_type: EventType,
    *,
    approval_id: str = "approval-1",
    tool_call_id: str = "call-1",
    **payload: object,
) -> Event:
    return Event(
        id=event_id,
        type=event_type,
        session_id="root",
        payload={
            "approval_id": approval_id,
            "tool_round_id": "round-1",
            "tool_call_id": tool_call_id,
            **payload,
        },
    )


async def _project(events: list[Event]) -> RuntimeEvidenceReport:
    store = InMemorySessionStore()
    await store.create(
        RunRequest(
            agent_name="agent",
            session_id="root",
            messages=[Message.text("user", "prompt")],
        ),
        identity=SessionIdentity(provider_name="provider", model="model"),
    )
    for event in events:
        await store.append_event("root", event)
    return await runtime_evidence(
        CayuApp(session_store=store, enable_logging=False),
        RuntimeEvidenceRequest(root_session_id="root", max_sessions=10, max_events=50),
    )


_REQUESTED = EventType.TOOL_CALL_APPROVAL_REQUESTED
_APPROVED = EventType.TOOL_CALL_APPROVED
_DENIED = EventType.TOOL_CALL_APPROVAL_DENIED
_EXPIRED = EventType.TOOL_CALL_APPROVAL_EXPIRED


@pytest.mark.parametrize(
    ("closing_events", "expected"),
    [
        # Either signal alone is enough: stored history may carry only one.
        pytest.param(
            [(_EXPIRED, {}), (_DENIED, {"expired": True})], "expired", id="marker-then-flagged"
        ),
        pytest.param([(_EXPIRED, {}), (_DENIED, {})], "expired", id="marker-then-unflagged"),
        pytest.param([(_DENIED, {"expired": True})], "expired", id="flagged-without-marker"),
        pytest.param([(_EXPIRED, {})], "expired", id="marker-only"),
        pytest.param([(_DENIED, {"expired": False})], "denied", id="denied-by-a-person"),
        pytest.param([(_DENIED, {})], "denied", id="denied-without-flag"),
        # Only the boolean the runtime writes counts as the flag.
        pytest.param([(_DENIED, {"expired": "true"})], "denied", id="denied-with-text-flag"),
        pytest.param([(_APPROVED, {})], "approved", id="approved"),
        # The flag reinterprets a denial only; it never turns a grant into an expiry.
        pytest.param([(_APPROVED, {"expired": True})], "approved", id="approved-ignores-flag"),
        pytest.param([], "pending", id="still-pending"),
    ],
)
def test_runtime_evidence_approval_decision_reads_both_expiry_signals(
    closing_events: list[tuple[EventType, dict[str, object]]],
    expected: str,
) -> None:
    report = asyncio.run(
        _project(
            [
                _approval_event("requested", _REQUESTED),
                *(
                    _approval_event(f"closing-{index}", event_type, **payload)
                    for index, (event_type, payload) in enumerate(closing_events)
                ),
            ]
        )
    )

    [session] = report.sessions
    [approval] = session.approvals
    assert approval.decision == expected
    assert [ref.event_id for ref in approval.source_refs] == [
        "requested",
        *(f"closing-{index}" for index in range(len(closing_events))),
    ]
    assert RuntimeEvidenceWarningCode.MALFORMED_APPROVAL not in {
        warning.code for warning in report.warnings
    }


def test_runtime_evidence_expiry_of_one_approval_does_not_mark_another() -> None:
    report = asyncio.run(
        _project(
            [
                _approval_event("first-requested", _REQUESTED),
                _approval_event("first-expired", _EXPIRED),
                _approval_event("first-denied", _DENIED, expired=True),
                _approval_event(
                    "second-requested", _REQUESTED, approval_id="approval-2", tool_call_id="call-2"
                ),
                _approval_event(
                    "second-denied",
                    _DENIED,
                    approval_id="approval-2",
                    tool_call_id="call-2",
                    expired=False,
                ),
            ]
        )
    )

    [session] = report.sessions
    assert {approval.approval_id: approval.decision for approval in session.approvals} == {
        "approval-1": "expired",
        "approval-2": "denied",
    }
