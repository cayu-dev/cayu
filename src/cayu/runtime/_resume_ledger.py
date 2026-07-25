from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from cayu._validation import copy_json_value
from cayu.core.events import Event, EventType
from cayu.core.tools import _bound_policy_denial_text
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_results as tool_results
from cayu.runtime.approvals import PendingToolCallApproval
from cayu.runtime.tool_policy import ToolPolicyDecision, ToolPolicyResult
from cayu.vaults import SecretRedactor


@dataclass(frozen=True)
class ToolCallLedger:
    outcomes: dict[str, runtime_records.ToolCallOutcome]
    started_ids: set[str]
    conflicting_ids: set[str]

    @property
    def started_without_terminal_ids(self) -> set[str]:
        return self.started_ids - set(self.outcomes)


@dataclass(frozen=True)
class ToolCallEvidenceLedger:
    terminal_ids: set[str]
    started_ids: set[str]
    conflicting_ids: set[str]

    @property
    def started_without_terminal_ids(self) -> set[str]:
        return self.started_ids - self.terminal_ids


@dataclass(frozen=True)
class ToolCallRecoveryState:
    started: bool
    terminal: bool
    conflicting: bool


_TerminalOutcome = TypeVar("_TerminalOutcome")


@dataclass(frozen=True)
class _ScannedToolCalls(Generic[_TerminalOutcome]):
    outcomes: dict[str, _TerminalOutcome]
    started_ids: set[str]
    conflicting_ids: set[str]


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
    relevant_event_types = terminal_event_types | {EventType.TOOL_CALL_STARTED}
    is_candidate = in_scope if candidate_scope is None else candidate_scope

    for index, event in enumerate(events):
        if event.type not in relevant_event_types:
            continue
        tool_call_id = event.payload.get("tool_call_id")
        if type(tool_call_id) is not str or tool_call_id not in pending_by_id:
            continue
        if not is_candidate(event):
            continue
        if not in_scope(event):
            started_ids.add(tool_call_id)
            last_conflict_index[tool_call_id] = index
            continue
        if event.type == EventType.TOOL_CALL_STARTED:
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
    )


def tool_call_recovery_state(
    *,
    events: Iterable[Event],
    pending_tool_call: PendingToolCallApproval,
    in_scope: Callable[[Event], bool],
    candidate_scope: Callable[[Event], bool] | None = None,
    terminal_event_types: frozenset[EventType],
) -> ToolCallRecoveryState:
    if type(pending_tool_call) is not PendingToolCallApproval:
        raise TypeError("pending_tool_call must be a PendingToolCallApproval.")
    tool_call_id = pending_tool_call.tool_call_id
    ledger = scan_tool_call_events(
        events=events,
        pending_calls=(pending_tool_call,),
        in_scope=in_scope,
        candidate_scope=candidate_scope,
        terminal_event_types=terminal_event_types,
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
    """Bound only durable denials; approval prompts retain their full policy text."""

    if policy_result is None or policy_result.reason is None:
        return None
    if policy_result.decision is ToolPolicyDecision.DENY:
        reason = policy_result.reason
        if redactor is not None:
            reason = redactor.redact_text(reason)
        return _bound_policy_denial_text(reason)
    return policy_result.reason


def tool_call_outcome_from_terminal_event(
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
