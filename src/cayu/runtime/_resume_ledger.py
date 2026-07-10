from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from cayu._validation import copy_json_value
from cayu.core.events import Event, EventType
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_results as tool_results
from cayu.runtime.approvals import PendingToolCallApproval
from cayu.runtime.tool_policy import ToolPolicyDecision, ToolPolicyResult


@dataclass(frozen=True)
class ToolCallLedger:
    outcomes: dict[str, runtime_records.ToolCallOutcome]
    started_ids: set[str]

    @property
    def started_without_terminal_ids(self) -> set[str]:
        return self.started_ids - set(self.outcomes)


@dataclass(frozen=True)
class ToolCallRecoveryState:
    started: bool
    terminal: bool


def scan_tool_call_events(
    *,
    events: Iterable[Event],
    pending_calls: Iterable[PendingToolCallApproval],
    in_scope: Callable[[Event], bool],
    terminal_event_types: frozenset[EventType],
) -> ToolCallLedger:
    pending_by_id = {call.tool_call_id: call for call in pending_calls}
    started_ids: set[str] = set()
    outcomes: dict[str, runtime_records.ToolCallOutcome] = {}

    for event in events:
        if not in_scope(event):
            continue
        tool_call_id = event.payload.get("tool_call_id")
        if type(tool_call_id) is not str or tool_call_id not in pending_by_id:
            continue
        if event.type == EventType.TOOL_CALL_STARTED:
            started_ids.add(tool_call_id)
            continue
        if event.type in terminal_event_types:
            outcomes[tool_call_id] = tool_call_outcome_from_terminal_event(
                event=event,
                pending_tool_call=pending_by_id[tool_call_id],
            )

    return ToolCallLedger(outcomes=outcomes, started_ids=started_ids)


def tool_call_recovery_state(
    *,
    events: Iterable[Event],
    tool_call_id: str,
    in_scope: Callable[[Event], bool],
    terminal_event_types: frozenset[EventType],
) -> ToolCallRecoveryState:
    started = False
    terminal = False
    for event in events:
        if not in_scope(event):
            continue
        if event.payload.get("tool_call_id") != tool_call_id:
            continue
        if event.type == EventType.TOOL_CALL_STARTED:
            started = True
        elif event.type in terminal_event_types:
            terminal = True
    return ToolCallRecoveryState(started=started, terminal=terminal)


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
