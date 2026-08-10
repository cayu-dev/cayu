"""Detached-value contracts for public operational and approval evidence."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ConfigDict

from cayu import (
    AggregateAccuracy,
    AggregateAccuracyKind,
    BudgetLimit,
    BusinessApprovalRecord,
    BusinessApprovalResolutionState,
    BusinessApprovalRouting,
    Event,
    EventType,
    ModelPrice,
    PendingToolApprovalEventView,
    PendingToolCallApprovalEventView,
    PriceBook,
    ResolutionActor,
    RetryPolicy,
    RunLimits,
    RunOutcome,
    SessionOperationalSnapshot,
    SessionStatus,
    SessionStatusCounts,
    StructuredOutputResult,
    StructuredOutputSpec,
    TaskOperationalSnapshot,
    TaskStatusCounts,
    ThinkingConfig,
)


def _assert_run_outcome_detaches_event_payloads() -> None:
    event = Event(
        type=EventType.SESSION_COMPLETED,
        session_id="session-1",
        payload={"evidence": {"paths": ["before.txt"]}},
    )

    structured_output = StructuredOutputResult(
        output={"answer": {"citations": ["before.txt"]}},
        name="answer",
        attempt=1,
        max_retries=2,
    )
    outcome = RunOutcome(
        session_id="session-1",
        status=SessionStatus.COMPLETED,
        final_text="done",
        error=None,
        events=(event,),
        structured_output=structured_output,
    )
    event.payload["evidence"]["paths"][0] = "after.txt"
    event.session_id = "mutated"
    structured_output.output["answer"]["citations"][0] = "after.txt"

    assert outcome.events[0].session_id == "session-1"
    assert outcome.events[0].payload == {"evidence": {"paths": ["before.txt"]}}
    assert outcome.structured_output is not None
    assert outcome.structured_output.output == {"answer": {"citations": ["before.txt"]}}


def _assert_task_operational_snapshot_detaches_nested_models() -> None:
    counts = TaskStatusCounts(
        pending=1,
        claimed=0,
        running=0,
        paused=0,
        blocked=0,
        needs_attention=0,
        completed=0,
        failed=0,
        cancelled=0,
    )
    accuracy = AggregateAccuracy(
        kind=AggregateAccuracyKind.SAMPLED,
        reason="bounded sample",
        limit=10,
    )

    snapshot = TaskOperationalSnapshot(
        as_of=datetime(2026, 8, 9, tzinfo=UTC),
        total_count=1,
        counts_by_status=counts,
        claimable_pending_count=1,
        scheduled_pending_count=0,
        accuracy=accuracy,
    )
    counts.pending = 0
    counts.completed = 1
    accuracy.reason = "mutated"

    assert snapshot.counts_by_status.pending == 1
    assert snapshot.counts_by_status.completed == 0
    assert snapshot.accuracy.reason == "bounded sample"


def _assert_session_operational_snapshot_detaches_nested_models() -> None:
    counts = SessionStatusCounts(
        pending=0,
        running=1,
        interrupting=0,
        completed=0,
        failed=0,
        interrupted=0,
    )
    accuracy = AggregateAccuracy(
        kind=AggregateAccuracyKind.TRUNCATED,
        reason="bounded read",
        limit=20,
    )

    snapshot = SessionOperationalSnapshot(
        as_of=datetime(2026, 8, 9, tzinfo=UTC),
        total_count=1,
        counts_by_status=counts,
        accuracy=accuracy,
    )
    counts.running = 0
    counts.completed = 1
    accuracy.reason = "mutated"

    assert snapshot.counts_by_status.running == 1
    assert snapshot.counts_by_status.completed == 0
    assert snapshot.accuracy.reason == "bounded read"


def _assert_business_approval_record_detaches_routing() -> None:
    routing = BusinessApprovalRouting(
        required_tier="national",
        chain=("area", "national"),
        metadata={"package": {"labels": ["fragile"]}},
    )
    resolved_by = ResolutionActor(
        subject="operator-1",
        claims={"roles": ["reviewer"]},
    )

    record = BusinessApprovalRecord(
        approval_id="approval-1",
        session_id="session-1",
        tool_call_id="call-1",
        tool_name="route_package",
        agent_name="router",
        routing=routing,
        resolution_state=BusinessApprovalResolutionState.PENDING,
        decision=None,
        outcome=None,
        condition_text=None,
        approver_id=None,
        approver_tier=None,
        resolved_by=resolved_by,
        reason=None,
        expired=False,
        requested_at=datetime(2026, 8, 9, tzinfo=UTC),
        resolved_at=None,
    )
    routing.metadata["package"]["labels"][0] = "mutated"
    resolved_by.claims["roles"][0] = "mutated"

    assert record.routing is not None
    assert record.routing.metadata == {"package": {"labels": ["fragile"]}}
    assert record.resolved_by is not None
    assert record.resolved_by.claims == {"roles": ["reviewer"]}


def _pending_call(arguments: dict) -> PendingToolCallApprovalEventView:
    return PendingToolCallApprovalEventView(
        tool_call_id="call-1",
        tool_name="send_email",
        arguments_state="finalized",
        arguments=arguments,
        metadata={"audit": {"labels": ["reviewed"]}},
        active_taint_labels=["private"],
    )


def _pending_approval_view(**overrides) -> PendingToolApprovalEventView:
    arguments = {"message": {"recipients": ["before@example.com"]}}
    values = {
        "approval_id": "approval-1",
        "tool_round_id": f"tround_{'1' * 32}",
        "model_step_id": f"mstep_{'2' * 32}",
        "model_attempt_id": f"matt_{'3' * 32}",
        "tool_call_id": "call-1",
        "tool_name": "send_email",
        "arguments_state": "finalized",
        "arguments": arguments,
        "agent_name": "assistant",
        "tool_calls": [_pending_call(arguments)],
    }
    values.update(overrides)
    return PendingToolApprovalEventView(**values)


def test_pending_approval_view_owns_structured_output_snapshot() -> None:
    structured_output = StructuredOutputSpec(
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
        name="answer",
    )
    view = _pending_approval_view(structured_output=structured_output)
    expected = view.model_dump(mode="python")

    structured_output.json_schema["properties"]["answer"]["type"] = "number"
    structured_output.max_retries = 7

    assert view.model_dump(mode="python") == expected


def test_pending_approval_view_owns_run_limits_snapshot() -> None:
    limits = RunLimits(max_total_tokens=100, max_tool_calls=2)
    view = _pending_approval_view(limits=limits)
    expected = view.model_dump(mode="python")

    limits.max_total_tokens = 200
    limits.max_tool_calls = 4

    assert view.model_dump(mode="python") == expected


def test_pending_approval_view_owns_budget_limits_snapshot() -> None:
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
        )
    )
    limit = BudgetLimit(
        max_estimated_cost=Decimal("2"),
        pricing=pricing,
    )
    view = _pending_approval_view(budget_limits=(limit,))
    expected = view.model_dump(mode="python")

    limit.max_estimated_cost = Decimal("9")
    pricing.price_book_version = "mutated"

    assert view.model_dump(mode="python") == expected


def test_pending_approval_view_rejects_mutable_thinking_subclass() -> None:
    class MutableThinkingConfig(ThinkingConfig):
        model_config = ConfigDict(extra="forbid", frozen=False)

    thinking = MutableThinkingConfig(effort="high")

    with pytest.raises(TypeError, match="Thinking config must be a ThinkingConfig instance"):
        _pending_approval_view(thinking=thinking)


def test_pending_approval_view_rejects_mutable_retry_policy_subclass() -> None:
    class MutableRetryPolicy(RetryPolicy):
        model_config = ConfigDict(extra="forbid", frozen=False)

    retry_policy = MutableRetryPolicy(max_attempts=3)

    with pytest.raises(TypeError, match="Retry policy must be a RetryPolicy instance"):
        _pending_approval_view(retry_policy=retry_policy)


def _assert_pending_tool_call_approval_event_view_detaches_nested_payloads() -> None:
    arguments = {"message": {"recipients": ["before@example.com"]}}
    metadata = {"audit": {"labels": ["reviewed"]}}
    labels = ["private"]

    view = PendingToolCallApprovalEventView(
        tool_call_id="call-1",
        tool_name="send_email",
        arguments_state="finalized",
        arguments=arguments,
        metadata=metadata,
        active_taint_labels=labels,
    )
    arguments["message"]["recipients"][0] = "after@example.com"
    metadata["audit"]["labels"][0] = "mutated"
    labels[0] = "mutated"

    assert view.arguments == {"message": {"recipients": ["before@example.com"]}}
    assert view.metadata == {"audit": {"labels": ["reviewed"]}}
    assert view.active_taint_labels == ["private"]


def _assert_pending_tool_approval_event_view_detaches_supplied_call_view() -> None:
    arguments = {"message": {"recipients": ["before@example.com"]}}
    call = _pending_call(arguments)

    view = PendingToolApprovalEventView(
        approval_id="approval-1",
        tool_round_id=f"tround_{'1' * 32}",
        model_step_id=f"mstep_{'2' * 32}",
        model_attempt_id=f"matt_{'3' * 32}",
        tool_call_id="call-1",
        tool_name="send_email",
        arguments_state="finalized",
        arguments=arguments,
        agent_name="assistant",
        tool_calls=[call],
    )
    call.arguments["message"]["recipients"][0] = "after@example.com"
    call.metadata["audit"]["labels"][0] = "mutated"

    assert view.arguments == {"message": {"recipients": ["before@example.com"]}}
    assert view.tool_calls[0].arguments == {"message": {"recipients": ["before@example.com"]}}
    assert view.tool_calls[0].metadata == {"audit": {"labels": ["reviewed"]}}


@pytest.mark.parametrize(
    "assert_detached",
    [
        pytest.param(_assert_run_outcome_detaches_event_payloads, id="run-outcome-events"),
        pytest.param(
            _assert_task_operational_snapshot_detaches_nested_models,
            id="task-operational-snapshot",
        ),
        pytest.param(
            _assert_session_operational_snapshot_detaches_nested_models,
            id="session-operational-snapshot",
        ),
        pytest.param(
            _assert_business_approval_record_detaches_routing,
            id="business-approval-routing",
        ),
        pytest.param(
            _assert_pending_tool_call_approval_event_view_detaches_nested_payloads,
            id="pending-tool-call-view",
        ),
        pytest.param(
            _assert_pending_tool_approval_event_view_detaches_supplied_call_view,
            id="pending-tool-approval-view",
        ),
    ],
)
def test_operational_and_approval_evidence_is_detached(
    assert_detached: Callable[[], None],
) -> None:
    assert_detached()


def _construct_task_snapshot_with_corrupted_counts() -> None:
    counts = TaskStatusCounts(
        pending=1,
        claimed=0,
        running=0,
        paused=0,
        blocked=0,
        needs_attention=0,
        completed=0,
        failed=0,
        cancelled=0,
    )
    counts.claimed = -1
    counts.completed = 1
    TaskOperationalSnapshot(
        as_of=datetime(2026, 8, 9, tzinfo=UTC),
        total_count=1,
        counts_by_status=counts,
        claimable_pending_count=0,
        scheduled_pending_count=0,
        accuracy=AggregateAccuracy(kind=AggregateAccuracyKind.EXACT),
    )


def _construct_task_snapshot_with_corrupted_accuracy() -> None:
    accuracy = AggregateAccuracy(
        kind=AggregateAccuracyKind.SAMPLED,
        reason="bounded sample",
        limit=10,
    )
    accuracy.kind = AggregateAccuracyKind.EXACT
    TaskOperationalSnapshot(
        as_of=datetime(2026, 8, 9, tzinfo=UTC),
        total_count=0,
        counts_by_status=TaskStatusCounts(
            pending=0,
            claimed=0,
            running=0,
            paused=0,
            blocked=0,
            needs_attention=0,
            completed=0,
            failed=0,
            cancelled=0,
        ),
        claimable_pending_count=0,
        scheduled_pending_count=0,
        accuracy=accuracy,
    )


def _construct_session_snapshot_with_corrupted_counts() -> None:
    counts = SessionStatusCounts(
        pending=0,
        running=1,
        interrupting=0,
        completed=0,
        failed=0,
        interrupted=0,
    )
    counts.running = -1
    counts.completed = 2
    SessionOperationalSnapshot(
        as_of=datetime(2026, 8, 9, tzinfo=UTC),
        total_count=1,
        counts_by_status=counts,
        accuracy=AggregateAccuracy(kind=AggregateAccuracyKind.EXACT),
    )


def _construct_session_snapshot_with_corrupted_accuracy() -> None:
    accuracy = AggregateAccuracy(
        kind=AggregateAccuracyKind.TRUNCATED,
        reason="bounded read",
        limit=20,
    )
    accuracy.kind = AggregateAccuracyKind.EXACT
    SessionOperationalSnapshot(
        as_of=datetime(2026, 8, 9, tzinfo=UTC),
        total_count=0,
        counts_by_status=SessionStatusCounts(
            pending=0,
            running=0,
            interrupting=0,
            completed=0,
            failed=0,
            interrupted=0,
        ),
        accuracy=accuracy,
    )


def _construct_business_record_with_corrupted_routing() -> None:
    routing = BusinessApprovalRouting(
        required_tier="national",
        chain=("area", "national"),
        metadata={"package": {"labels": ["fragile"]}},
    )
    routing.metadata["package"]["labels"].append(object())
    BusinessApprovalRecord(
        approval_id="approval-1",
        session_id="session-1",
        tool_call_id="call-1",
        tool_name="route_package",
        agent_name="router",
        routing=routing,
        resolution_state=BusinessApprovalResolutionState.PENDING,
        decision=None,
        outcome=None,
        condition_text=None,
        approver_id=None,
        approver_tier=None,
        resolved_by=None,
        reason=None,
        expired=False,
        requested_at=datetime(2026, 8, 9, tzinfo=UTC),
        resolved_at=None,
    )


def _construct_approval_view_with_corrupted_call() -> None:
    arguments = {"message": {"recipients": ["before@example.com"]}}
    call = _pending_call(arguments)
    call.metadata["audit"]["labels"].append(object())
    PendingToolApprovalEventView(
        approval_id="approval-1",
        tool_round_id=f"tround_{'1' * 32}",
        model_step_id=f"mstep_{'2' * 32}",
        model_attempt_id=f"matt_{'3' * 32}",
        tool_call_id="call-1",
        tool_name="send_email",
        arguments_state="finalized",
        arguments=arguments,
        agent_name="assistant",
        tool_calls=[call],
    )


@pytest.mark.parametrize(
    "construct_with_corrupted_source",
    [
        pytest.param(
            _construct_task_snapshot_with_corrupted_counts,
            id="task-status-counts",
        ),
        pytest.param(
            _construct_task_snapshot_with_corrupted_accuracy,
            id="task-accuracy",
        ),
        pytest.param(
            _construct_session_snapshot_with_corrupted_counts,
            id="session-status-counts",
        ),
        pytest.param(
            _construct_session_snapshot_with_corrupted_accuracy,
            id="session-accuracy",
        ),
        pytest.param(
            _construct_business_record_with_corrupted_routing,
            id="business-routing",
        ),
        pytest.param(
            _construct_approval_view_with_corrupted_call,
            id="pending-call-view",
        ),
    ],
)
def test_operational_evidence_revalidates_previously_accepted_models(
    construct_with_corrupted_source: Callable[[], None],
) -> None:
    with pytest.raises(ValueError):
        construct_with_corrupted_source()
