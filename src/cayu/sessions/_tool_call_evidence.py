"""Shared tool-call evidence classification for session queries and recovery."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from cayu.approvals.tools import PendingToolCallApproval
from cayu.events import Event, EventType
from cayu.tools import _argument_publication as tool_argument_publication
from cayu.tools.gateway import gateway_lifecycle_matches_outer_call

TOOL_EVIDENCE_CONFLICT_PAYLOAD_KEY = "tool_evidence_conflict"


@dataclass(frozen=True)
class ToolCallEvidenceLedger:
    terminal_ids: set[str]
    started_ids: set[str]
    conflicting_ids: set[str]
    scope_conflicting: bool

    @property
    def started_without_terminal_ids(self) -> set[str]:
        return self.started_ids - self.terminal_ids


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


def scan_projected_tool_call_evidence(
    *,
    events: Iterable[Event],
    pending_calls: Iterable[PendingToolCallApproval],
    in_scope: Callable[[Event], bool],
    candidate_scope: Callable[[Event], bool] | None = None,
    terminal_event_types: frozenset[EventType],
    terminal_result_is_valid: Callable[[Event], bool],
) -> ToolCallEvidenceLedger:
    """Classify bounded event projections with the shared recovery evidence rules."""

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
