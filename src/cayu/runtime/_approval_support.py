from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any, NamedTuple

from cayu._validation import copy_json_value
from cayu.core.events import Event, EventType
from cayu.core.tools import ToolResult
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_results as tool_results
from cayu.runtime.approvals import (
    PendingToolApproval,
    PendingToolCallApproval,
    ResolutionActor,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
    resolution_actor_payload,
)
from cayu.runtime.sessions import Session
from cayu.runtime.tool_policy import ToolPolicyDecision, ToolPolicyResult

PENDING_TOOL_APPROVAL_CHECKPOINT_KEY = "pending_tool_approval"


def pending_approval_expired(approval: PendingToolApproval, now: datetime) -> bool:
    """Whether a pending approval's window has closed at ``now``.

    Pure access-time check (no daemon); the resolution winner evaluates it
    after the atomic status claim. A future lifecycle sweep (issue #104) can
    call this on interrupted sessions to proactively deny expired approvals.
    """
    return approval.expires_at is not None and now >= approval.expires_at


class ApprovalResolutionHistory(NamedTuple):
    has_denied_result: bool
    has_approved_call: bool
    has_executed_or_recovered_result: bool

    @property
    def has_granted_activity(self) -> bool:
        """The approval was already granted or produced executed results.

        Expiry gates only the FIRST grant: a retry after a mid-run crash
        re-resolves an approval that was authorized in-window, so coercing it
        to a denial would contradict the recorded grant (and deadlock against
        ``validate_retry_decision``).
        """
        return self.has_approved_call or self.has_executed_or_recovered_result


class ToolApprovalManualRecoveryRequired(RuntimeError):
    def __init__(self, *, tool_call_id: str, tool_name: str) -> None:
        super().__init__(
            "Tool approval cannot be retried automatically because a tool call "
            f"started without a terminal result: {tool_call_id} ({tool_name})."
        )
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name


class RoundToolManualRecoveryRequired(RuntimeError):
    def __init__(self, *, tool_call_id: str, tool_name: str) -> None:
        super().__init__(
            "A paused round cannot be resumed automatically because a tool call started "
            f"without a terminal result: {tool_call_id} ({tool_name})."
        )
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name


def resumed_event(
    *,
    session: Session,
    agent_name: str,
    environment_name: str | None,
    approval: PendingToolApproval,
    decision: ToolApprovalDecision,
    resolved_by: ResolutionActor | None = None,
    expired: bool = False,
) -> Event:
    return Event(
        type=EventType.SESSION_RESUMED,
        session_id=session.id,
        agent_name=agent_name,
        environment_name=environment_name,
        payload={
            "agent_name": agent_name,
            "approval_id": approval.approval_id,
            "tool_call_id": approval.tool_call_id,
            "decision": decision.value,
            "resolved_by": resolution_actor_payload(resolved_by),
            "expired": expired,
        },
    )


def cleared_event(
    *,
    session: Session,
    agent_name: str,
    environment_name: str | None,
    approval_id: str,
) -> Event:
    return Event(
        type=EventType.SESSION_CHECKPOINTED,
        session_id=session.id,
        agent_name=agent_name,
        environment_name=environment_name,
        payload={
            "checkpoint": PENDING_TOOL_APPROVAL_CHECKPOINT_KEY,
            "approval_id": approval_id,
            "cleared": True,
        },
    )


def checkpoint_for_fork(
    *,
    checkpoint: dict[str, Any] | None,
    agent_name: str,
    environment_name: str | None,
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
    pending_approval = pending_approval_from_checkpoint(copied_checkpoint)
    if pending_approval is None:
        return copied_checkpoint
    if pending_approval.agent_name != agent_name:
        raise ValueError(
            "Cannot fork a pending tool approval to a different agent: "
            f"{pending_approval.agent_name} -> {agent_name}"
        )
    if pending_approval.environment_name != environment_name:
        raise ValueError(
            "Cannot fork a pending tool approval to a different environment: "
            f"{pending_approval.environment_name} -> {environment_name}"
        )
    # Dump in JSON mode to match the checkpoint write path: persisted run
    # config (budget limits) carries Decimal values that python-mode dumps
    # would leak into the JSON-only checkpoint.
    copied_checkpoint[PENDING_TOOL_APPROVAL_CHECKPOINT_KEY] = pending_approval.model_copy(
        update={"task_id": None}
    ).model_dump(mode="json")
    return copied_checkpoint


def approval_denied_tool_result(
    request: ToolApprovalRequest,
    *,
    approval: PendingToolApproval,
    tool_call: runtime_records.ToolCallRequest,
    approval_required: bool,
) -> ToolResult:
    if request.reason:
        reason = request.reason
        if approval_required:
            content = f"Tool call denied by approval: {request.reason}"
        else:
            content = (
                "Tool call skipped because approval was denied for the same tool round: "
                f"{request.reason}"
            )
    elif approval_required:
        reason = "Tool call denied by approval."
        content = reason
    else:
        reason = "Tool call skipped because approval was denied for the same tool round."
        content = reason

    return ToolResult(
        content=content,
        structured={
            "decision": request.decision.value,
            "approval_id": approval.approval_id,
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "approval_required": approval_required,
            "denied_by_approval": approval_required,
            "skipped_due_to_approval_denial": not approval_required,
            "denied_tool_call_id": approval.tool_call_id,
            "denied_tool_name": approval.tool_name,
            "reason": reason,
            "metadata": request.metadata,
        },
        is_error=True,
    )


def user_input_resume_events(events: list[Event], input_id: str) -> list[Event]:
    """Return only the events belonging to a user-input pause's resume attempts.

    User-input round terminal events carry no ``approval_id``, and tool-call ids are only unique
    within one assistant message — not per session — so a round's events cannot be identified by
    id alone. The round runs no tools before it pauses, so every ``started``/terminal event for
    the round is emitted AFTER the pause boundary; events before it — prior rounds that may reuse
    the same ids — are excluded.

    The boundary is the FIRST event that marks this pause: either ``session.awaiting_user_input``
    (payload ``input_id``) or a ``session.interrupted`` carrying ``user_input.input_id`` for it.
    Both are needed: ``pending_user_input`` is checkpointed BEFORE the awaiting event is appended,
    so a worker that crashed in between has no awaiting event, but recovery finalizes the pause
    with a durable ``session.interrupted``. Anchoring on that too keeps the retry ledger scoped
    (rather than empty, which would re-run an already-completed sibling).
    """
    for index, event in enumerate(events):
        if _event_marks_user_input_pause(event, input_id):
            return events[index + 1 :]
    return []


def _event_marks_user_input_pause(event: Event, input_id: str) -> bool:
    if event.type == EventType.SESSION_AWAITING_USER_INPUT:
        return event.payload.get("input_id") == input_id
    if event.type == EventType.SESSION_INTERRUPTED:
        user_input = event.payload.get("user_input")
        return isinstance(user_input, dict) and user_input.get("input_id") == input_id
    return False


def recorded_round_tool_outcomes(
    *,
    events: list[Event],
    pending_calls: list[PendingToolCallApproval],
    input_id: str,
) -> dict[str, runtime_records.ToolCallOutcome]:
    """Reconstruct already-recorded terminal outcomes for a paused user-input round, keyed by
    ``tool_call_id``, scoped to the pause's resume window (see ``user_input_resume_events``).

    Lets a retried resume skip re-executing a tool that already completed before a mid-resume
    failure, without colliding with a prior round that reused the same ids.
    """
    pending_by_id = {call.tool_call_id: call for call in pending_calls}
    terminal_event_types = {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
    }
    started_ids: set[str] = set()
    outcomes: dict[str, runtime_records.ToolCallOutcome] = {}
    for event in user_input_resume_events(events, input_id):
        tool_call_id = event.payload.get("tool_call_id")
        if type(tool_call_id) is not str or tool_call_id not in pending_by_id:
            continue
        if event.type == EventType.TOOL_CALL_STARTED:
            started_ids.add(tool_call_id)
            continue
        if event.type in terminal_event_types:
            outcomes[tool_call_id] = _tool_call_outcome_from_terminal_event(
                event=event,
                pending_tool_call=pending_by_id[tool_call_id],
            )
    # A tool that started on a prior resume attempt but has no terminal event (a crash mid-tool)
    # cannot be safely re-run — fail loudly instead of silently double-executing a side effect.
    for tool_call_id in started_ids:
        if tool_call_id not in outcomes:
            pending_call = pending_by_id[tool_call_id]
            raise RoundToolManualRecoveryRequired(
                tool_call_id=tool_call_id,
                tool_name=pending_call.tool_name,
            )
    return outcomes


def recorded_tool_outcomes(
    *,
    events: list[Event],
    approval: PendingToolApproval,
) -> dict[str, runtime_records.ToolCallOutcome]:
    pending_calls = {call.tool_call_id: call for call in pending_round_tool_calls(approval)}
    started_ids: set[str] = set()
    outcomes: dict[str, runtime_records.ToolCallOutcome] = {}
    terminal_event_types = {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }

    for event in events:
        if event.payload.get("approval_id") != approval.approval_id:
            continue

        tool_call_id = event.payload.get("tool_call_id")
        if type(tool_call_id) is not str or tool_call_id not in pending_calls:
            continue

        if event.type == EventType.TOOL_CALL_STARTED:
            started_ids.add(tool_call_id)
            continue

        if event.type in terminal_event_types:
            outcomes[tool_call_id] = _tool_call_outcome_from_terminal_event(
                event=event,
                pending_tool_call=pending_calls[tool_call_id],
            )

    for tool_call_id in started_ids:
        if tool_call_id not in outcomes:
            pending_tool_call = pending_calls[tool_call_id]
            raise ToolApprovalManualRecoveryRequired(
                tool_call_id=tool_call_id,
                tool_name=pending_tool_call.tool_name,
            )

    return outcomes


def approval_resolution_history(
    *,
    events: list[Event],
    approval: PendingToolApproval,
) -> ApprovalResolutionHistory:
    has_denied_result = False
    has_approved_call = False
    has_executed_or_recovered_result = False

    for event in events:
        if event.payload.get("approval_id") != approval.approval_id:
            continue
        if event.type == EventType.TOOL_CALL_APPROVAL_DENIED:
            has_denied_result = True
        elif event.type == EventType.TOOL_CALL_APPROVED:
            has_approved_call = True
        elif event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}:
            has_executed_or_recovered_result = True

    return ApprovalResolutionHistory(
        has_denied_result=has_denied_result,
        has_approved_call=has_approved_call,
        has_executed_or_recovered_result=has_executed_or_recovered_result,
    )


def validate_retry_decision(
    *,
    history: ApprovalResolutionHistory,
    approval: PendingToolApproval,
    decision: ToolApprovalDecision,
) -> None:
    if decision == ToolApprovalDecision.APPROVE and history.has_denied_result:
        raise RuntimeError(
            "Tool approval was already denied and cannot be retried as approved: "
            f"{approval.approval_id}"
        )
    if decision == ToolApprovalDecision.DENY and history.has_granted_activity:
        raise RuntimeError(
            "Tool approval already has approved or executed tool results and "
            f"cannot be retried as denied: {approval.approval_id}"
        )


def pending_tool_call_for_recovery(
    *,
    approval: PendingToolApproval,
    tool_call_id: str,
) -> PendingToolCallApproval:
    for pending_tool_call in pending_round_tool_calls(approval):
        if pending_tool_call.tool_call_id == tool_call_id:
            return pending_tool_call
    raise ValueError(f"Tool call is not part of the pending approval: {tool_call_id}")


def validate_recovery_target(
    *,
    events: list[Event],
    approval: PendingToolApproval,
    tool_call_id: str,
) -> None:
    started = False
    terminal = False
    terminal_event_types = {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
    for event in events:
        if event.payload.get("approval_id") != approval.approval_id:
            continue
        if event.payload.get("tool_call_id") != tool_call_id:
            continue
        if event.type == EventType.TOOL_CALL_STARTED:
            started = True
        elif event.type in terminal_event_types:
            terminal = True

    if terminal:
        raise RuntimeError(
            f"Tool call already has a terminal event and does not need recovery: {tool_call_id}"
        )
    if not started:
        raise RuntimeError(
            f"Tool approval recovery requires a recorded tool.call.started event: {tool_call_id}"
        )


def round_tool_call_for_recovery(
    *,
    pending_calls: list[PendingToolCallApproval],
    tool_call_id: str,
) -> PendingToolCallApproval:
    for pending_tool_call in pending_calls:
        if pending_tool_call.tool_call_id == tool_call_id:
            return PendingToolCallApproval(**pending_tool_call.model_dump())
    raise ValueError(f"Tool call is not part of the paused round: {tool_call_id}")


def validate_round_recovery_target(
    *,
    events: list[Event],
    pending_calls: list[PendingToolCallApproval],
    tool_call_id: str,
    input_id: str,
) -> None:
    # Round terminal events carry no approval_id (user-input rounds) and tool-call ids are not
    # unique across the session, so scope to the pause's resume window (matching
    # recorded_round_tool_outcomes) — a prior round reusing this id must not be seen here.
    if tool_call_id not in {call.tool_call_id for call in pending_calls}:
        raise ValueError(f"Tool call is not part of the paused round: {tool_call_id}")
    started = False
    terminal = False
    terminal_event_types = {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
    }
    for event in user_input_resume_events(events, input_id):
        if event.payload.get("tool_call_id") != tool_call_id:
            continue
        if event.type == EventType.TOOL_CALL_STARTED:
            started = True
        elif event.type in terminal_event_types:
            terminal = True

    if terminal:
        raise RuntimeError(
            f"Tool call already has a terminal event and does not need recovery: {tool_call_id}"
        )
    if not started:
        raise RuntimeError(
            f"User input recovery requires a recorded tool.call.started event: {tool_call_id}"
        )


def recovered_tool_result(
    *,
    request: ToolApprovalRecoveryRequest,
) -> ToolResult:
    if request.outcome not in {
        ToolApprovalRecoveryOutcome.COMPLETED,
        ToolApprovalRecoveryOutcome.FAILED,
    }:
        raise ValueError(f"Unsupported tool approval recovery outcome: {request.outcome}")
    return ToolResult(
        content=request.message,
        structured=request.structured,
        artifacts=request.artifacts,
        is_error=request.outcome == ToolApprovalRecoveryOutcome.FAILED,
    )


def pending_approval_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> PendingToolApproval | None:
    if checkpoint is None:
        return None
    copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
    value = copied_checkpoint.get(PENDING_TOOL_APPROVAL_CHECKPOINT_KEY)
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("Pending tool approval checkpoint must be an object.")
    return PendingToolApproval(**value)


def pending_tool_call_approvals(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] | None,
    active_taint_by_id: Mapping[str, frozenset[str]] = MappingProxyType({}),
) -> list[PendingToolCallApproval]:
    policy_results_by_id: dict[str, ToolPolicyResult | None] = {}
    if policy_outcomes is not None:
        policy_results_by_id = {outcome.call.id: outcome.result for outcome in policy_outcomes}
    pending_approvals: list[PendingToolCallApproval] = []
    for tool_call in tool_calls:
        policy_result = policy_results_by_id.get(tool_call.id)
        pending_approvals.append(
            PendingToolCallApproval(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                arguments=copy_json_value(tool_call.arguments, "arguments"),
                policy_decision=policy_result.decision.value if policy_result is not None else None,
                reason=policy_result.reason if policy_result is not None else None,
                metadata=(
                    copy_json_value(policy_result.metadata, "metadata")
                    if policy_result is not None
                    else {}
                ),
                active_taint_labels=sorted(active_taint_by_id.get(tool_call.id, frozenset())),
            )
        )
    return pending_approvals


def pending_round_tool_calls(
    approval: PendingToolApproval,
) -> list[PendingToolCallApproval]:
    return [PendingToolCallApproval(**call.model_dump()) for call in approval.tool_calls]


def policy_result_from_pending_tool_call(
    pending_tool_call: PendingToolCallApproval,
) -> ToolPolicyResult | None:
    if pending_tool_call.policy_decision is None:
        return None
    return ToolPolicyResult(
        decision=ToolPolicyDecision(pending_tool_call.policy_decision),
        reason=pending_tool_call.reason,
        metadata=copy_json_value(pending_tool_call.metadata, "metadata"),
    )


def taint_labels_from_pending_tool_call(
    pending_tool_call: PendingToolCallApproval,
) -> frozenset[str]:
    """Active taint labels persisted for this call, restored so the resumed tool is gated with the
    same taint the policy used before the pause."""
    return frozenset(pending_tool_call.active_taint_labels)


def _tool_call_outcome_from_terminal_event(
    *,
    event: Event,
    pending_tool_call: PendingToolCallApproval,
) -> runtime_records.ToolCallOutcome:
    result_payload = event.payload.get("result")
    if type(result_payload) is not dict:
        raise ValueError(
            f"Terminal tool event is missing result payload: {pending_tool_call.tool_call_id}"
        )
    result = tool_results.tool_result_from_payload(result_payload)
    return runtime_records.ToolCallOutcome(
        call=runtime_records.ToolCallRequest(
            id=pending_tool_call.tool_call_id,
            name=pending_tool_call.tool_name,
            arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
        ),
        result=result,
    )
