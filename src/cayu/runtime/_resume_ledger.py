from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from cayu._validation import copy_durable_metadata, copy_json_value
from cayu.approvals.tools import PendingToolCallApproval
from cayu.events import Event, EventType
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_results as tool_results
from cayu.sessions import _tool_call_evidence as tool_call_evidence
from cayu.tools import _argument_publication as tool_argument_publication
from cayu.tools.base import _bound_policy_denial_text
from cayu.tools.policy import ToolPolicyDecision, ToolPolicyResult
from cayu.vaults import SecretRedactor


class ToolCallEvidenceConflict(RuntimeError):
    """Current-round tool evidence cannot be attributed to one pending call."""


@dataclass(frozen=True)
class ToolCallLedger:
    outcomes: dict[str, runtime_records.ToolCallOutcome]
    started_ids: set[str]
    conflicting_ids: set[str]
    scope_conflicting: bool

    @property
    def started_without_terminal_ids(self) -> set[str]:
        return self.started_ids - set(self.outcomes)


@dataclass(frozen=True)
class ToolCallRecoveryState:
    started: bool
    terminal: bool
    conflicting: bool


def scan_tool_call_events(
    *,
    events: Iterable[Event],
    pending_calls: Iterable[PendingToolCallApproval],
    in_scope: Callable[[Event], bool],
    candidate_scope: Callable[[Event], bool] | None = None,
    terminal_event_types: frozenset[EventType],
) -> ToolCallLedger:
    scanned = tool_call_evidence._scan_tool_call_events(
        events=events,
        pending_calls=pending_calls,
        in_scope=in_scope,
        candidate_scope=candidate_scope,
        terminal_event_types=terminal_event_types,
        terminal_outcome=lambda event, pending_call: tool_call_outcome_from_terminal_event(
            event=event,
            pending_tool_call=pending_call,
        ),
    )
    return ToolCallLedger(
        outcomes=scanned.outcomes,
        started_ids=scanned.started_ids,
        conflicting_ids=scanned.conflicting_ids,
        scope_conflicting=scanned.scope_conflicting,
    )


def tool_call_recovery_state(
    *,
    events: Iterable[Event],
    pending_calls: Iterable[PendingToolCallApproval],
    tool_call_id: str,
    in_scope: Callable[[Event], bool],
    candidate_scope: Callable[[Event], bool] | None = None,
    terminal_event_types: frozenset[EventType],
) -> ToolCallRecoveryState:
    materialized_calls = tuple(pending_calls)
    if any(type(call) is not PendingToolCallApproval for call in materialized_calls):
        raise TypeError("pending_calls must contain PendingToolCallApproval values.")
    if sum(call.tool_call_id == tool_call_id for call in materialized_calls) != 1:
        raise ValueError("tool_call_id must identify exactly one pending call.")
    ledger = scan_tool_call_events(
        events=events,
        pending_calls=materialized_calls,
        in_scope=in_scope,
        candidate_scope=candidate_scope,
        terminal_event_types=terminal_event_types,
    )
    if ledger.scope_conflicting:
        raise ToolCallEvidenceConflict(
            "Tool recovery evidence contains a call outside the pending tool round."
        )
    return ToolCallRecoveryState(
        started=tool_call_id in ledger.started_ids,
        terminal=tool_call_id in ledger.outcomes,
        conflicting=tool_call_id in ledger.conflicting_ids,
    )


def policy_result_from_pending_tool_call(
    pending_tool_call: PendingToolCallApproval,
) -> ToolPolicyResult | None:
    if pending_tool_call.policy_decision is None:
        return None
    return ToolPolicyResult(
        decision=ToolPolicyDecision(pending_tool_call.policy_decision),
        reason=pending_tool_call.reason,
        command_denial_code=pending_tool_call.command_denial_code,
        metadata=copy_durable_metadata(pending_tool_call.metadata),
    )


def policy_reason_for_pending_tool_call(
    policy_result: ToolPolicyResult | None,
    *,
    redactor: SecretRedactor | None = None,
) -> str | None:
    """Redact durable policy text and bound denial diagnostics."""

    if policy_result is None or policy_result.reason is None:
        return None
    reason = policy_result.reason
    if redactor is not None:
        reason = redactor.redact_text(reason)
    if policy_result.decision is ToolPolicyDecision.DENY:
        bounded = _bound_policy_denial_text(reason)
        return bounded if redactor is None else redactor.redact_text(bounded)
    return reason


def tool_call_outcome_from_terminal_event(
    *,
    event: Event,
    pending_tool_call: PendingToolCallApproval,
) -> runtime_records.ToolCallOutcome:
    if not tool_call_evidence._event_matches_pending_tool_call(event, pending_tool_call):
        raise ValueError(
            "Terminal tool event names a different pending tool call: "
            f"{pending_tool_call.tool_call_id}"
        )
    result_payload = event.payload.get("result")
    if type(result_payload) is not dict:
        raise ValueError(
            f"Terminal tool event is missing result payload: {pending_tool_call.tool_call_id}"
        )
    result = tool_results.tool_result_from_payload(result_payload)
    gateway_outer_call = event.tool_name != pending_tool_call.tool_name
    argument_projection = (
        None
        if gateway_outer_call
        else tool_argument_publication.terminal_argument_projection(
            event.payload,
            legacy_arguments=pending_tool_call.arguments,
        )
    )
    return runtime_records.ToolCallOutcome(
        call=runtime_records.ToolCallRequest(
            id=pending_tool_call.tool_call_id,
            name=pending_tool_call.tool_name,
            arguments=(
                copy_json_value(pending_tool_call.arguments, "arguments")
                if argument_projection is None
                else (
                    {}
                    if argument_projection.arguments is None
                    else copy_json_value(argument_projection.arguments, "arguments")
                )
            ),
            arguments_state=(
                argument_projection.state if argument_projection is not None else "unavailable"
            ),
            targeted_tool_grant_id=pending_tool_call.targeted_tool_grant_id,
            model_tool_name=pending_tool_call.model_tool_name,
            targeted_tool_invocation=pending_tool_call.targeted_tool_invocation,
            targeted_tool_rejection=pending_tool_call.targeted_tool_rejection,
        ),
        result=result,
    )
