from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from cayu._validation import copy_json_value
from cayu.core.events import Event, EventType
from cayu.core.tools import _bound_policy_denial_text
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_argument_publication as tool_argument_publication
from cayu.runtime import _tool_results as tool_results
from cayu.runtime.approvals import PendingToolCallApproval
from cayu.runtime.tool_gateway import gateway_lifecycle_matches_outer_call
from cayu.runtime.tool_policy import ToolPolicyDecision, ToolPolicyResult
from cayu.vaults import SecretRedactor

TOOL_EVIDENCE_CONFLICT_PAYLOAD_KEY = "tool_evidence_conflict"


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
class ToolCallEvidenceLedger:
    terminal_ids: set[str]
    started_ids: set[str]
    conflicting_ids: set[str]
    scope_conflicting: bool

    @property
    def started_without_terminal_ids(self) -> set[str]:
        return self.started_ids - self.terminal_ids


@dataclass(frozen=True)
class ToolCallRecoveryState:
    started: bool
    terminal: bool
    conflicting: bool


_TerminalOutcome = TypeVar("_TerminalOutcome")


def _event_matches_pending_tool_call(
    event: Event,
    pending_call: PendingToolCallApproval,
) -> bool:
    if type(event.tool_name) is not str:
        return False
    if event.tool_name == pending_call.tool_name:
        return True
    if pending_call.model_tool_name is not None:
        return False
    return gateway_lifecycle_matches_outer_call(
        effective_tool_name=event.tool_name,
        event_payload=event.payload,
        outer_tool_name=pending_call.tool_name,
        outer_arguments=pending_call.arguments,
    )


@dataclass(frozen=True)
class _ScannedToolCalls(Generic[_TerminalOutcome]):
    outcomes: dict[str, _TerminalOutcome]
    started_ids: set[str]
    conflicting_ids: set[str]
    scope_conflicting: bool


def _scan_tool_call_events(
    *,
    events: Iterable[Event],
    pending_calls: Iterable[PendingToolCallApproval],
    in_scope: Callable[[Event], bool],
    candidate_scope: Callable[[Event], bool] | None = None,
    terminal_event_types: frozenset[EventType],
    terminal_outcome: Callable[[Event, PendingToolCallApproval], _TerminalOutcome],
) -> _ScannedToolCalls[_TerminalOutcome]:
    pending_by_id = {call.tool_call_id: call for call in pending_calls}
    started_ids: set[str] = set()
    outcomes: dict[str, _TerminalOutcome] = {}
    started_event_ids: set[str] = set()
    terminal_event_ids: set[str] = set()
    last_conflict_index: dict[str, int] = {}
    last_manual_recovery_index: dict[str, int] = {}
    scope_conflicting = False
    relevant_event_types = terminal_event_types | {EventType.TOOL_CALL_STARTED}
    is_candidate = in_scope if candidate_scope is None else candidate_scope

    for index, event in enumerate(events):
        if event.type not in relevant_event_types:
            continue
        if not is_candidate(event):
            continue
        tool_call_id = event.payload.get("tool_call_id")
        if type(tool_call_id) is not str or tool_call_id not in pending_by_id:
            scope_conflicting = True
            continue
        if not in_scope(event):
            started_ids.add(tool_call_id)
            last_conflict_index[tool_call_id] = index
            continue
        pending_call = pending_by_id[tool_call_id]
        if not _event_matches_pending_tool_call(event, pending_call):
            started_ids.add(tool_call_id)
            last_conflict_index[tool_call_id] = index
            outcomes.pop(tool_call_id, None)
            continue
        if event.type == EventType.TOOL_CALL_STARTED:
            gateway_outer_call = event.tool_name != pending_call.tool_name
            if not gateway_outer_call and not (
                tool_argument_publication.started_arguments_match_private_call(
                    event.payload,
                    private_arguments=pending_call.arguments,
                )
            ):
                started_ids.add(tool_call_id)
                last_conflict_index[tool_call_id] = index
                outcomes.pop(tool_call_id, None)
                continue
            if tool_call_id in started_event_ids or tool_call_id in terminal_event_ids:
                last_conflict_index[tool_call_id] = index
            started_event_ids.add(tool_call_id)
            started_ids.add(tool_call_id)
            continue
        if event.type in terminal_event_types:
            duplicate_terminal = tool_call_id in terminal_event_ids
            unresolved_before_event = last_conflict_index.get(
                tool_call_id, -1
            ) > last_manual_recovery_index.get(tool_call_id, -1)
            terminal_event_ids.add(tool_call_id)
            try:
                outcome = terminal_outcome(event, pending_by_id[tool_call_id])
            except Exception:
                # A durable terminal marker whose result cannot be reconstructed
                # proves that the call may have produced an effect, but it does
                # not prove a usable outcome. Keep it recoverable only through
                # an explicit operator-supplied terminal result.
                last_conflict_index[tool_call_id] = index
                outcomes.pop(tool_call_id, None)
                continue
            if event.payload.get("manual_recovery") is True:
                # One valid manual result may resolve preceding ambiguity. A
                # second manual terminal without intervening contradictory
                # evidence is itself contradictory and must not silently win.
                if duplicate_terminal and not unresolved_before_event:
                    last_conflict_index[tool_call_id] = index
                    outcomes.pop(tool_call_id, None)
                    continue
                outcomes[tool_call_id] = outcome
                last_manual_recovery_index[tool_call_id] = index
                continue
            if duplicate_terminal:
                last_conflict_index[tool_call_id] = index
            outcomes[tool_call_id] = outcome

    conflicting_ids = {
        tool_call_id
        for tool_call_id, conflict_index in last_conflict_index.items()
        if conflict_index > last_manual_recovery_index.get(tool_call_id, -1)
    }
    for tool_call_id in conflicting_ids:
        outcomes.pop(tool_call_id, None)
        started_ids.add(tool_call_id)
    return _ScannedToolCalls(
        outcomes=outcomes,
        started_ids=started_ids,
        conflicting_ids=conflicting_ids,
        scope_conflicting=scope_conflicting,
    )


def scan_tool_call_events(
    *,
    events: Iterable[Event],
    pending_calls: Iterable[PendingToolCallApproval],
    in_scope: Callable[[Event], bool],
    candidate_scope: Callable[[Event], bool] | None = None,
    terminal_event_types: frozenset[EventType],
) -> ToolCallLedger:
    scanned = _scan_tool_call_events(
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


def scan_projected_tool_call_evidence(
    *,
    events: Iterable[Event],
    pending_calls: Iterable[PendingToolCallApproval],
    in_scope: Callable[[Event], bool],
    candidate_scope: Callable[[Event], bool] | None = None,
    terminal_event_types: frozenset[EventType],
    terminal_result_is_valid: Callable[[Event], bool],
) -> ToolCallEvidenceLedger:
    """Classify bounded event projections with the runtime recovery state machine."""

    def projected_terminal_outcome(
        event: Event,
        _pending_call: PendingToolCallApproval,
    ) -> None:
        if not terminal_result_is_valid(event):
            raise ValueError("Projected terminal event has no usable result.")

    scanned = _scan_tool_call_events(
        events=events,
        pending_calls=pending_calls,
        in_scope=in_scope,
        candidate_scope=candidate_scope,
        terminal_event_types=terminal_event_types,
        terminal_outcome=projected_terminal_outcome,
    )
    return ToolCallEvidenceLedger(
        terminal_ids=set(scanned.outcomes),
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
        metadata=copy_json_value(pending_tool_call.metadata, "metadata"),
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
    if not _event_matches_pending_tool_call(event, pending_tool_call):
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
            targeted_tool_grant_id=pending_tool_call.targeted_tool_grant_id,
            model_tool_name=pending_tool_call.model_tool_name,
            targeted_tool_invocation=pending_tool_call.targeted_tool_invocation,
            targeted_tool_rejection=pending_tool_call.targeted_tool_rejection,
        ),
        result=result,
    )
