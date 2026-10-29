"""Model-call usage and tool-call views of one session, shared by the CLI and the server.

`cayu session usage` and `cayu session tools` pair durable events by their private
execution identity (model step, attempt, tool round, tool call, reservation). The server's
public event API replaces that identity with record-scoped aliases, so a client can't pair
the events itself. These views run where the private records are: the CLI runs them over a
local store, and an Agent serves them at `GET /api/sessions/{id}/model-calls` and
`GET /api/sessions/{id}/tool-calls`, so a local store and a Cloud Agent report the same rows.

Every view reads only purpose-specific projections of the events it needs, in sequence
pages, and stops at 64 MiB or 100,000 retained records.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from cayu._validation import compact_json_utf8_size
from cayu.budgets.aggregates import summary_usage_metrics_from_event_payload
from cayu.budgets.base import is_complete_budget_reconciliation_pricing
from cayu.budgets.usage import (
    count_model_steps_with_usage,
    session_usage_summary,
    usage_metrics_from_event_payload,
)
from cayu.events import EventType
from cayu.execution_units import ToolRoundIdentity
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.records import EventRecord
from cayu.tools.policy import ToolPolicyDecision


class InspectionTooLarge(ValueError):
    """A view would retain more than its safety limit; narrow the event window."""


class EventReader(Protocol):
    def query_events(self, query: EventQuery) -> Awaitable[list[EventRecord]]: ...


@dataclass(frozen=True)
class ArgumentDisclosure:
    """How tool-call rows show arguments.

    A local operator reads the store itself and sees bounded, redacted summaries of every
    argument the store holds. A server must not show more than its public event API, so it
    sets ``quarantine_pre_execution``: arguments on records written before the call ran (its
    start, approval request or input request) show as ``[quarantined]``, as
    `project_persisted_runtime_event` publishes them, and only a terminal record's arguments
    are summarized. ``redact`` (the application's secret redactor) runs on arguments before
    they are bounded, so a secret is never cut where the redactor can't recognize it.
    """

    quarantine_pre_execution: bool = False
    redact: Callable[[Any], Any] | None = None


# A local operator's view of their own store.
LOCAL_DISCLOSURE = ArgumentDisclosure()


@dataclass(frozen=True)
class InspectionWindow:
    """The events a view reads: a durable sequence window and, optionally, one interaction."""

    after_sequence: int | None = None
    before_sequence: int | None = None
    interaction_id: str | None = None


_MAX_COLLECTED_EVENT_BYTES = 64 * 1024 * 1024


_MAX_COLLECTED_EVENT_RECORDS = 100_000


_EVENT_QUERY_PAGE_SIZE = 200


_USAGE_INSPECTION_PRICING_STATE_KEY = "_cayu_pricing_state"


_TOOL_INSPECTION_IDENTITY_CONFLICT_KEY = "_cayu_tool_identity_conflict"


USAGE_INSPECTION_EVENT_TYPES = (
    EventType.MODEL_COMPLETED,
    EventType.TOOL_CALL_STARTED,
    EventType.BUDGET_RESERVED,
    EventType.BUDGET_RECONCILED,
    EventType.BUDGET_RESERVATION_RELEASED,
)


def model_call_usage(
    records: list[EventRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    unmatched_ledger: list[dict[str, Any]] = []
    reservations: dict[str, dict[str, Any]] = {}
    settled_reservation_ids: set[str] = set()
    for record in records:
        event = record.event
        if event.type == EventType.BUDGET_RESERVED:
            reservation_id = event.payload.get("reservation_id")
            if type(reservation_id) is not str:
                continue
            reservations[reservation_id] = {
                "reservation_id": reservation_id,
                "reserved_amount": _optional_string(event.payload.get("requested")),
                "actual_amount": None,
                "currency": _optional_string(event.payload.get("currency")),
                "pricing_state": "unpriced",
            }
            continue
        if event.type == EventType.MODEL_COMPLETED:
            try:
                metrics = usage_metrics_from_event_payload(event.payload)
            except (TypeError, ValueError):
                metrics = None
            cache = None if metrics is None else metrics.cache
            call = {
                "sequence": record.sequence,
                "timestamp": event.timestamp.isoformat(),
                "provider": None if metrics is None else metrics.provider_name,
                "requested_model": None if metrics is None else metrics.requested_model,
                "resolved_model": None if metrics is None else metrics.model,
                "input_tokens": None if metrics is None else metrics.input_tokens,
                "output_tokens": None if metrics is None else metrics.output_tokens,
                "total_tokens": None if metrics is None else metrics.total_tokens,
                "reasoning_tokens": (None if metrics is None else metrics.reasoning_output_tokens),
                "cache_read_tokens": None if cache is None else cache.read_tokens,
                "cache_write_tokens": None if cache is None else cache.write_tokens,
                "cached_input_tokens": None if cache is None else cache.cached_input_tokens,
                "uncached_input_tokens": None if cache is None else cache.uncached_input_tokens,
                "transcript_cursor": _optional_nonnegative_int(
                    event.payload.get("transcript_cursor")
                ),
                "pricing_state": "unknown",
                "ledger": [],
            }
            calls.append(call)
            continue
        if event.type not in {
            EventType.BUDGET_RECONCILED,
            EventType.BUDGET_RESERVATION_RELEASED,
        }:
            continue
        reservation_id = event.payload.get("reservation_id")
        if type(reservation_id) is not str:
            continue
        ledger = reservations.get(reservation_id)
        if ledger is None:
            ledger = {
                "reservation_id": reservation_id,
                "reserved_amount": _optional_string(event.payload.get("reserved_amount")),
                "actual_amount": None,
                "currency": None,
                "pricing_state": "unpriced",
            }
        if event.type == EventType.BUDGET_RECONCILED:
            ledger["actual_amount"] = _optional_string(event.payload.get("actual_amount"))
            pricing_state = event.payload.get(_USAGE_INSPECTION_PRICING_STATE_KEY)
            ledger["pricing_state"] = (
                pricing_state
                if type(pricing_state) is str and pricing_state in {"priced", "unpriced", "unknown"}
                else "unknown"
            )
        ledger["outcome"] = (
            "reconciled" if event.type == EventType.BUDGET_RECONCILED else "released"
        )
        unmatched_ledger.append(ledger)
        settled_reservation_ids.add(reservation_id)
    for reservation_id, ledger in reservations.items():
        if reservation_id in settled_reservation_ids:
            continue
        unmatched_ledger.append({**ledger, "outcome": "open"})
    return calls, unmatched_ledger


TOOL_INSPECTION_EVENT_TYPES = (
    EventType.TOOL_CALL_STARTED,
    EventType.TOOL_CALL_COMPLETED,
    EventType.TOOL_CALL_FAILED,
    EventType.TOOL_CALL_BLOCKED,
    EventType.TOOL_CALL_APPROVAL_REQUESTED,
    EventType.TOOL_CALL_APPROVED,
    EventType.TOOL_CALL_APPROVAL_DENIED,
    EventType.TOOL_CALL_APPROVAL_EXPIRED,
    EventType.SESSION_AWAITING_USER_INPUT,
)


_TOOL_TERMINAL_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
    }
)


_TOOL_EXECUTION_TERMINAL_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
    }
)


def tool_call_rows(records: list[EventRecord]) -> list[dict[str, Any]]:
    starts: dict[_ToolCallKey, EventRecord] = {}
    terminals: dict[_ToolCallKey, EventRecord] = {}
    approval_requests: dict[_ToolCallKey, EventRecord] = {}
    approval_calls: dict[_ToolCallKey, dict[str, Any]] = {}
    input_requests: dict[_ToolCallKey, EventRecord] = {}
    input_calls: dict[_ToolCallKey, dict[str, Any]] = {}
    decision_records: dict[_ToolCallKey, EventRecord] = {}
    approval_states: dict[_ToolCallKey, str] = {}
    approval_outcomes: dict[_ToolCallKey, str] = {}
    approval_completion_records: dict[_ToolCallKey, EventRecord] = {}
    unregistered_policy_calls: set[_ToolCallKey] = set()
    resolved_approval_keys: set[tuple[str, str, str, str]] = set()
    approval_request_calls: dict[
        tuple[str, str, str, str],
        set[_ToolCallKey],
    ] = {}
    approval_request_rounds: set[tuple[str, str, str]] = set()
    approval_gated_calls: set[_ToolCallKey] = set()
    evidence_conflicts: set[_ToolCallKey] = set()
    approval_decision_conflicts: set[_ToolCallKey] = set()
    execution_terminal_keys: set[_ToolCallKey] = set()
    tool_names: dict[_ToolCallKey, set[str]] = {}
    lifecycle_records: dict[_ToolCallKey, list[EventRecord]] = {}
    approval_scoped_keys: set[_ToolCallKey] = set()

    def retain_tool_name(call_key: _ToolCallKey, value: object) -> None:
        if type(value) is not str or not value or value.strip() != value:
            evidence_conflicts.add(call_key)
            return
        tool_names.setdefault(call_key, set()).add(value)

    def retain_lifecycle_record(call_key: _ToolCallKey, record: EventRecord) -> None:
        lifecycle_records.setdefault(call_key, []).append(record)

    def approval_resolution_key(
        record: EventRecord,
    ) -> tuple[str, str, str, str] | None:
        approval_id = record.event.payload.get("approval_id")
        identity = _tool_event_identity(record)
        if (
            type(approval_id) is not str
            or not approval_id
            or approval_id.strip() != approval_id
            or identity is None
        ):
            return None
        return (
            identity.model_step_id,
            identity.model_attempt_id,
            identity.tool_round_id,
            approval_id,
        )

    def retain_approval_decision_state(
        call_key: _ToolCallKey,
        record: EventRecord,
    ) -> None:
        if record.event.type == EventType.TOOL_CALL_APPROVED:
            approval_states[call_key] = "approved"
            return
        if record.event.type in {
            EventType.TOOL_CALL_APPROVAL_DENIED,
            EventType.TOOL_CALL_APPROVAL_EXPIRED,
        }:
            outcome = (
                "expired"
                if record.event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED
                or record.event.payload.get("expired") is True
                else "denied"
            )
            approval_states[call_key] = outcome
            approval_outcomes[call_key] = outcome
            return
        if _is_ambiguous_approval_block(record):
            approval_states[call_key] = "blocked"

    for record in records:
        event = record.event
        if event.type == EventType.SESSION_AWAITING_USER_INPUT:
            nested_calls = event.payload.get("tool_calls")
            calls = (
                [item for item in nested_calls if type(item) is dict]
                if type(nested_calls) is list and nested_calls
                else [event.payload]
            )
            for call in calls:
                call_id = call.get("tool_call_id")
                if type(call_id) is not str:
                    continue
                call_key = _tool_call_key(record, call_id)
                if call_key in input_requests:
                    evidence_conflicts.add(call_key)
                input_requests.setdefault(call_key, record)
                input_calls.setdefault(call_key, call)
                retain_tool_name(call_key, call.get("tool_name"))
                if call_id == event.payload.get("tool_call_id"):
                    retain_tool_name(call_key, event.tool_name)
            continue
        if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED:
            approval = _tool_approval_payload(record)
            if approval is None:
                continue
            nested_calls = approval.get("tool_calls")
            has_nested_calls = type(nested_calls) is list and bool(nested_calls)
            calls = (
                [item for item in nested_calls if type(item) is dict]
                if has_nested_calls
                else [approval]
            )
            for call in calls:
                call_id = call.get("tool_call_id")
                if type(call_id) is not str:
                    continue
                call_key = _tool_call_key(record, call_id)
                if call_key in approval_requests:
                    evidence_conflicts.add(call_key)
                    approval_decision_conflicts.add(call_key)
                approval_requests.setdefault(call_key, record)
                approval_calls.setdefault(call_key, call)
                resolution_key = approval_resolution_key(record)
                if resolution_key is None:
                    evidence_conflicts.add(call_key)
                    approval_decision_conflicts.add(call_key)
                else:
                    approval_request_calls.setdefault(resolution_key, set()).add(call_key)
                    approval_request_rounds.add(resolution_key[:3])
                policy_decision = call.get("policy_decision")
                policy_evidence = call.get("policy_evidence")
                is_ambiguous_call = policy_evidence == "ambiguous" and policy_decision is None
                is_unregistered_call = policy_evidence == "unregistered" and policy_decision is None
                if is_unregistered_call:
                    unregistered_policy_calls.add(call_key)
                if (
                    not has_nested_calls
                    or policy_decision == ToolPolicyDecision.REQUIRE_APPROVAL.value
                    or (is_ambiguous_call and call_id == approval.get("tool_call_id"))
                ):
                    approval_states[call_key] = "requested"
                    approval_gated_calls.add(call_key)
                elif (
                    policy_decision
                    not in {
                        ToolPolicyDecision.ALLOW.value,
                        ToolPolicyDecision.DENY.value,
                    }
                    and not is_ambiguous_call
                    and not is_unregistered_call
                ):
                    evidence_conflicts.add(call_key)
                    approval_decision_conflicts.add(call_key)
                approval_scoped_keys.add(call_key)
                retain_lifecycle_record(call_key, record)
                retain_tool_name(call_key, call.get("tool_name"))
                if call_id == approval.get("tool_call_id"):
                    retain_tool_name(call_key, event.tool_name)
            continue
        call_id = _tool_event_call_id(record)
        if call_id is None:
            continue
        call_key = _tool_call_key(record, call_id)
        retain_lifecycle_record(call_key, record)
        retain_tool_name(call_key, event.tool_name)
        if event.type == EventType.TOOL_CALL_STARTED:
            if call_key in starts or call_key in terminals:
                evidence_conflicts.add(call_key)
            starts.setdefault(call_key, record)
        elif event.type in _TOOL_TERMINAL_TYPES:
            if event.type in _TOOL_EXECUTION_TERMINAL_TYPES:
                execution_terminal_keys.add(call_key)
            if call_key in terminals:
                evidence_conflicts.add(call_key)
            terminals.setdefault(call_key, record)
            if _is_ambiguous_approval_block_candidate(record):
                approval_scoped_keys.add(call_key)
                if call_key in decision_records:
                    evidence_conflicts.add(call_key)
                    approval_decision_conflicts.add(call_key)
                decision_records.setdefault(call_key, record)
        elif event.type == EventType.TOOL_CALL_APPROVED:
            approval_scoped_keys.add(call_key)
            if call_key in decision_records:
                evidence_conflicts.add(call_key)
                approval_decision_conflicts.add(call_key)
            decision_records.setdefault(call_key, record)
        elif event.type == EventType.TOOL_CALL_APPROVAL_DENIED:
            approval_scoped_keys.add(call_key)
            prior_decision = decision_records.get(call_key)
            if prior_decision is not None and not _approval_denial_completes_expiry(
                prior_decision, record
            ):
                evidence_conflicts.add(call_key)
                approval_decision_conflicts.add(call_key)
            decision_records.setdefault(call_key, record)
        elif event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED:
            approval_scoped_keys.add(call_key)
            if call_key in decision_records:
                evidence_conflicts.add(call_key)
                approval_decision_conflicts.add(call_key)
            decision_records.setdefault(call_key, record)

    for call_key, decision_record in decision_records.items():
        resolution_key = approval_resolution_key(decision_record)
        if resolution_key is None:
            evidence_conflicts.add(call_key)
            approval_decision_conflicts.add(call_key)
            continue
        request_calls = approval_request_calls.get(resolution_key)
        if request_calls is None:
            if resolution_key[:3] in approval_request_rounds:
                evidence_conflicts.add(call_key)
                approval_decision_conflicts.add(call_key)
                continue
            if decision_record.event.type is EventType.TOOL_CALL_BLOCKED:
                # Every ambiguous sibling carries the round-level requested
                # decision. Without the request, the gating call cannot be
                # identified positively, so retain only the terminal block.
                continue
            # A bounded --after-sequence window may intentionally exclude the
            # request. Retain the decision's local state without inventing a
            # call-level join to evidence that is not present.
            retain_approval_decision_state(call_key, decision_record)
            resolved_approval_keys.add(resolution_key)
            continue
        if call_key not in request_calls:
            evidence_conflicts.add(call_key)
            approval_decision_conflicts.add(call_key)
            continue
        if call_key in approval_gated_calls:
            if decision_record.event.type is EventType.TOOL_CALL_BLOCKED:
                approval_call = approval_calls.get(call_key)
                if (
                    not _is_ambiguous_approval_block(decision_record)
                    or approval_call is None
                    or approval_call.get("policy_evidence") != "ambiguous"
                    or approval_call.get("policy_decision") is not None
                ):
                    evidence_conflicts.add(call_key)
                    approval_decision_conflicts.add(call_key)
                    continue
                retain_approval_decision_state(call_key, decision_record)
                resolved_approval_keys.add(resolution_key)
                continue
            if (
                decision_record.event.type == EventType.TOOL_CALL_APPROVAL_DENIED
                and decision_record.event.payload.get("approval_required") is not True
            ):
                evidence_conflicts.add(call_key)
                approval_decision_conflicts.add(call_key)
                continue
            retain_approval_decision_state(call_key, decision_record)
            resolved_approval_keys.add(resolution_key)
            continue

        approval_call = approval_calls.get(call_key)
        policy_decision = None if approval_call is None else approval_call.get("policy_decision")
        policy_evidence = None if approval_call is None else approval_call.get("policy_evidence")
        if (
            _is_ambiguous_approval_block(decision_record)
            and approval_call is not None
            and approval_call.get("policy_evidence") == "ambiguous"
            and policy_decision is None
        ):
            # The round-level acknowledgement belongs only to the gating call.
            # Ambiguous siblings still close as blocked terminal tool calls.
            continue
        if (
            decision_record.event.type == EventType.TOOL_CALL_APPROVAL_DENIED
            and (
                policy_decision == ToolPolicyDecision.ALLOW.value
                or (policy_evidence in {"ambiguous", "unregistered"} and policy_decision is None)
            )
            and decision_record.event.payload.get("approval_required") is False
        ):
            approval_outcomes[call_key] = (
                "expired" if decision_record.event.payload.get("expired") is True else "denied"
            )
            approval_completion_records[call_key] = decision_record
            continue
        evidence_conflicts.add(call_key)
        approval_decision_conflicts.add(call_key)

    for call_key in unregistered_policy_calls:
        started = starts.get(call_key)
        terminal = terminals.get(call_key)
        decision = decision_records.get(call_key)
        valid_failed_closure = (
            terminal is not None
            and terminal.event.type is EventType.TOOL_CALL_FAILED
            and terminal.event.payload.get("registration_state") == "unregistered_at_policy_plan"
        )
        valid_denied_closure = (
            decision is not None
            and decision.event.type is EventType.TOOL_CALL_APPROVAL_DENIED
            and decision.event.payload.get("approval_required") is False
        )
        if (
            started is not None
            or (terminal is not None and not valid_failed_closure)
            or (decision is not None and not valid_denied_closure)
            or (terminal is not None and decision is not None)
        ):
            evidence_conflicts.add(call_key)
            approval_decision_conflicts.add(call_key)

    for call_key in approval_outcomes.keys() & execution_terminal_keys:
        evidence_conflicts.add(call_key)
        approval_decision_conflicts.add(call_key)

    for call_key, names in tool_names.items():
        if len(names) != 1:
            evidence_conflicts.add(call_key)
    for call_key, evidence in lifecycle_records.items():
        if call_key not in approval_scoped_keys and not any(
            "approval_id" in record.event.payload for record in evidence
        ):
            continue
        approval_ids: set[str] = set()
        invalid_approval_identity = False
        for record in evidence:
            raw_approval_id = record.event.payload.get("approval_id")
            if (
                type(raw_approval_id) is not str
                or not raw_approval_id
                or raw_approval_id.strip() != raw_approval_id
            ):
                invalid_approval_identity = True
                continue
            approval_ids.add(raw_approval_id)
        if invalid_approval_identity or len(approval_ids) != 1:
            evidence_conflicts.add(call_key)
            approval_decision_conflicts.add(call_key)

    call_keys = (
        starts.keys()
        | terminals.keys()
        | approval_requests.keys()
        | input_requests.keys()
        | decision_records.keys()
    )
    evidence_conflicts.update(_contradictory_tool_call_keys(call_keys))
    anchors = {
        call_key: (
            starts.get(call_key)
            or approval_requests.get(call_key)
            or input_requests.get(call_key)
            or terminals.get(call_key)
            or decision_records[call_key]
        )
        for call_key in call_keys
    }
    round_widths: dict[tuple[str, str, str], int] = {}
    for model_step_id, model_attempt_id, round_id, _, invalid_event_id in call_keys:
        if round_id is None or invalid_event_id is not None:
            continue
        round_key = (model_step_id or "", model_attempt_id or "", round_id)
        round_widths[round_key] = round_widths.get(round_key, 0) + 1

    rows: list[dict[str, Any]] = []
    ordered_call_keys = sorted(
        call_keys,
        key=lambda call_key: (
            anchors[call_key].sequence,
            call_key[3],
            anchors[call_key].event.id,
        ),
    )
    for call_key in ordered_call_keys:
        model_step_id, model_attempt_id, round_id, call_id, invalid_event_id = call_key
        started = starts.get(call_key)
        approval_request = approval_requests.get(call_key)
        approval_call = approval_calls.get(call_key)
        input_request = input_requests.get(call_key)
        input_call = input_calls.get(call_key)
        anchor = anchors[call_key]
        evidence_conflict = call_key in evidence_conflicts
        terminal = None if evidence_conflict else terminals.get(call_key)
        completion_record = (
            None if evidence_conflict else terminal or approval_completion_records.get(call_key)
        )
        approval_state = (
            "unavailable"
            if call_key in approval_decision_conflicts
            else approval_states.get(call_key, "none")
        )
        approval_outcome = approval_outcomes.get(call_key)
        approval_id = (
            None if approval_request is None else approval_request.event.payload.get("approval_id")
        )
        approval_key = (
            (
                model_step_id,
                model_attempt_id,
                round_id,
                approval_id,
            )
            if invalid_event_id is None
            and type(model_step_id) is str
            and type(model_attempt_id) is str
            and type(round_id) is str
            and type(approval_id) is str
            else None
        )
        awaiting_approval = approval_request is not None and (
            approval_key is None or approval_key not in resolved_approval_keys
        )
        round_key = (
            None
            if round_id is None or invalid_event_id is not None
            else (model_step_id or "", model_attempt_id or "", round_id)
        )
        result = None
        inspection_result = None
        if terminal is not None:
            raw_inspection_result = terminal.event.payload.get("_inspection_result")
            if type(raw_inspection_result) is dict:
                inspection_result = raw_inspection_result
            raw_result = terminal.event.payload.get("result")
            if type(raw_result) is dict:
                result = raw_result
        structured = None if result is None else result.get("structured")
        structured = structured if type(structured) is dict else None
        content = None if result is None else result.get("content")
        artifacts = None if result is None else result.get("artifacts")
        approval = _tool_approval_payload(approval_request)
        argument_summary = None
        arguments_state = None
        for candidate_state in (
            None if terminal is None else terminal.event.payload.get("_arguments_state"),
            None if started is None else started.event.payload.get("_arguments_state"),
            None if approval_call is None else approval_call.get("_arguments_state"),
            None if input_call is None else input_call.get("_arguments_state"),
        ):
            if candidate_state in {"finalized", "quarantined", "unavailable"}:
                arguments_state = candidate_state
                break
        for candidate_summary in (
            None if terminal is None else terminal.event.payload.get("_argument_summary"),
            None if started is None else started.event.payload.get("_argument_summary"),
            None if approval_call is None else approval_call.get("_argument_summary"),
            None if input_call is None else input_call.get("_argument_summary"),
        ):
            if type(candidate_summary) is str:
                argument_summary = candidate_summary
                break
        if type(argument_summary) is not str:
            argument_source = (
                terminal.event.payload
                if terminal is not None
                else started.event.payload
                if started is not None
                else approval_call
                if approval_call is not None
                else input_call
                if input_call is not None
                else approval
                if approval is not None
                else {}
            )
            arguments_state, argument_summary = _tool_argument_inspection_projection(
                argument_source
            )
        tool_name = (
            anchor.event.tool_name
            if approval_call is None and input_call is None
            else (approval_call or input_call or {}).get("tool_name", anchor.event.tool_name)
        )
        rows.append(
            {
                "sequence": anchor.sequence,
                "tool": tool_name,
                "tool_call_id": call_id,
                "model_step_id": model_step_id,
                "model_attempt_id": model_attempt_id,
                "tool_round_id": round_id,
                "parallel_round_width": 1 if round_key is None else round_widths[round_key],
                "arguments_state": arguments_state,
                "argument_summary": argument_summary,
                "started_at": (None if started is None else started.event.timestamp.isoformat()),
                "completed_at": (
                    None
                    if completion_record is None
                    else completion_record.event.timestamp.isoformat()
                ),
                "duration_ms": (None if evidence_conflict else _duration_ms(started, terminal)),
                "status": (
                    "unavailable"
                    if evidence_conflict
                    else _tool_status(
                        started,
                        terminal,
                        approval_state,
                        approval_outcome=approval_outcome,
                        awaiting_approval=awaiting_approval,
                        awaiting_input=input_request is not None,
                    )
                ),
                "approval_state": approval_state,
                "rendered_content_bytes": (
                    None
                    if evidence_conflict
                    else _optional_nonnegative_int(inspection_result.get("rendered_content_bytes"))
                    if inspection_result is not None
                    else len(content.encode("utf-8"))
                    if type(content) is str
                    else 0
                ),
                "structured_result_bytes": (
                    None
                    if evidence_conflict
                    else _optional_nonnegative_int(inspection_result.get("structured_result_bytes"))
                    if inspection_result is not None
                    else compact_json_utf8_size(structured)
                    if structured is not None
                    else 0
                ),
                "artifact_bytes": (
                    None
                    if evidence_conflict
                    else _optional_nonnegative_int(inspection_result.get("artifact_bytes"))
                    if inspection_result is not None
                    else compact_json_utf8_size(artifacts)
                    if type(artifacts) is list
                    else 0
                ),
                "returned": (
                    None
                    if evidence_conflict
                    else _optional_nonnegative_int(inspection_result.get("returned"))
                    if inspection_result is not None
                    else None
                    if structured is None
                    else _optional_nonnegative_int(structured.get("returned"))
                ),
                "truncated": (
                    None
                    if evidence_conflict
                    else inspection_result.get("truncated")
                    if inspection_result is not None
                    and type(inspection_result.get("truncated")) is bool
                    else structured.get("truncated")
                    if structured is not None and type(structured.get("truncated")) is bool
                    else None
                ),
            }
        )
    return rows


_ToolCallKey = tuple[str | None, str | None, str | None, str, str | None]


def _tool_call_key(record: EventRecord, call_id: str) -> _ToolCallKey:
    """Return an exact round/call join key, isolating malformed evidence."""

    if record.event.payload.get(_TOOL_INSPECTION_IDENTITY_CONFLICT_KEY) is True:
        return None, None, None, call_id, record.event.id
    identity = _tool_event_identity(record)
    if identity is None:
        return None, None, None, call_id, record.event.id
    return (
        identity.model_step_id,
        identity.model_attempt_id,
        identity.tool_round_id,
        call_id,
        None,
    )


def _contradictory_tool_call_keys(
    call_keys: set[_ToolCallKey],
) -> set[_ToolCallKey]:
    """Return rows whose execution-unit identity cannot be true simultaneously."""

    parents_by_round_id: dict[str, set[tuple[str, str]]] = {}
    rounds_by_attempt_id: dict[str, set[tuple[str, str]]] = {}
    conflicts = {call_key for call_key in call_keys if call_key[4] is not None}
    for model_step_id, model_attempt_id, round_id, _, invalid_event_id in call_keys:
        if (
            invalid_event_id is not None
            or model_step_id is None
            or model_attempt_id is None
            or round_id is None
        ):
            continue
        parents_by_round_id.setdefault(round_id, set()).add((model_step_id, model_attempt_id))
        rounds_by_attempt_id.setdefault(model_attempt_id, set()).add((model_step_id, round_id))

    conflicting_round_ids = {
        round_id for round_id, parents in parents_by_round_id.items() if len(parents) != 1
    }
    conflicting_attempt_ids = {
        model_attempt_id
        for model_attempt_id, step_rounds in rounds_by_attempt_id.items()
        if len(step_rounds) != 1
    }
    conflicts.update(
        call_key
        for call_key in call_keys
        if call_key[2] in conflicting_round_ids or call_key[1] in conflicting_attempt_ids
    )
    return conflicts


def _tool_event_identity(record: EventRecord) -> ToolRoundIdentity | None:
    payload = record.event.payload
    try:
        return ToolRoundIdentity.model_validate(
            {
                "model_step_id": payload.get("model_step_id"),
                "model_attempt_id": payload.get("model_attempt_id"),
                "tool_round_id": payload.get("tool_round_id"),
            }
        )
    except (TypeError, ValueError):
        return None


def _tool_event_call_id(record: EventRecord) -> str | None:
    call_id = record.event.payload.get("tool_call_id")
    if type(call_id) is str:
        return call_id
    approval = _tool_approval_payload(record)
    if approval is None:
        return None
    nested_call_id = approval.get("tool_call_id")
    return nested_call_id if type(nested_call_id) is str else None


def _tool_approval_payload(record: EventRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    approval = record.event.payload.get("approval")
    return approval if type(approval) is dict else None


def _tool_approval_identity_matches_event(record: EventRecord) -> bool:
    """Validate the duplicated approval identity before bounded projection."""

    approval = _tool_approval_payload(record)
    if approval is None:
        return False
    identity_keys = (
        "approval_id",
        "tool_call_id",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
    )
    for key in identity_keys:
        event_value = record.event.payload.get(key)
        approval_value = approval.get(key)
        if (
            type(event_value) is not str
            or type(approval_value) is not str
            or event_value != approval_value
        ):
            return False
    return True


def _approval_denial_completes_expiry(
    prior_decision: EventRecord,
    denial: EventRecord,
) -> bool:
    """Recognize the runtime's expected expiry marker followed by denial."""

    approval_id = prior_decision.event.payload.get("approval_id")
    return (
        prior_decision.event.type == EventType.TOOL_CALL_APPROVAL_EXPIRED
        and denial.event.type == EventType.TOOL_CALL_APPROVAL_DENIED
        and denial.event.payload.get("expired") is True
        and type(approval_id) is str
        and denial.event.payload.get("approval_id") == approval_id
    )


def _is_ambiguous_approval_block_candidate(record: EventRecord) -> bool:
    """Return whether a block claims to close an ambiguous approval."""

    payload = record.event.payload
    return record.event.type is EventType.TOOL_CALL_BLOCKED and (
        "requested_decision" in payload
        or payload.get("blocked_by") == "policy_evaluation_ambiguous"
        or payload.get("decision") == "ambiguous"
    )


def _is_ambiguous_approval_block(record: EventRecord) -> bool:
    """Recognize the runtime's exact non-authorizing approval resolution."""

    payload = record.event.payload
    return (
        record.event.type is EventType.TOOL_CALL_BLOCKED
        and payload.get("decision") == "ambiguous"
        and payload.get("blocked_by") == "policy_evaluation_ambiguous"
        and payload.get("requested_decision") == "approve"
    )


def _tool_status(
    started: EventRecord | None,
    terminal: EventRecord | None,
    approval_state: str,
    *,
    approval_outcome: str | None,
    awaiting_approval: bool,
    awaiting_input: bool,
) -> str:
    if terminal is None:
        if started is None and awaiting_input:
            return "awaiting_input"
        if started is None and approval_outcome in {"denied", "expired"}:
            return approval_outcome
        if started is None and approval_state in {"denied", "expired"}:
            return approval_state
        if started is None and (awaiting_approval or approval_state == "requested"):
            return "approval_pending"
        return "running"
    if terminal.event.type == EventType.TOOL_CALL_COMPLETED:
        return "success"
    if terminal.event.type == EventType.TOOL_CALL_BLOCKED:
        return "blocked"
    return "error"


def _duration_ms(
    started: EventRecord | None,
    terminal: EventRecord | None,
) -> int | None:
    if started is None or terminal is None:
        return None
    duration = terminal.event.timestamp - started.event.timestamp
    return max(round(duration.total_seconds() * 1000), 0)


def bounded_json_summary(value: object, *, max_bytes: int) -> str:
    safe = redact_inspection_value(value)
    rendered = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return truncate_utf8(rendered, max_bytes=max_bytes)


def truncate_utf8(value: str, *, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    if max_bytes <= 0:
        return ""
    marker = "…"
    if max_bytes < len(marker.encode("utf-8")):
        return "." * max_bytes
    retained = encoded[: max_bytes - len(marker.encode("utf-8"))]
    while retained:
        try:
            return retained.decode("utf-8") + marker
        except UnicodeDecodeError:
            retained = retained[:-1]
    return marker


async def collect_inspection_records(
    reader: EventReader,
    session_id: str,
    *,
    event_types: tuple[EventType, ...] | None = None,
    project_record: Callable[[EventRecord], EventRecord] | None = None,
    after_sequence: int | None = None,
    before_sequence: int | None = None,
    interaction_id: str | None = None,
) -> list[EventRecord]:
    records: list[EventRecord] = []
    retained_bytes = 0
    cursor = 0 if after_sequence is None else after_sequence
    while True:
        page = await reader.query_events(
            EventQuery(
                session_id=session_id,
                interaction_id=interaction_id,
                event_types=() if event_types is None else event_types,
                after_sequence=cursor,
                before_sequence=before_sequence,
                limit=_EVENT_QUERY_PAGE_SIZE,
                order_by=EventOrder.SEQUENCE_ASC,
            )
        )
        if not page:
            return records
        for record in page:
            retained_record = record if project_record is None else project_record(record)
            retained_bytes += compact_json_utf8_size(retained_record.event.model_dump(mode="json"))
            if retained_bytes > _MAX_COLLECTED_EVENT_BYTES:
                raise InspectionTooLarge(
                    "Session inspection exceeds the 64 MiB retained-event safety limit."
                    " Narrow the event window with --after-sequence or --before-sequence."
                )
            records.append(retained_record)
            if len(records) > _MAX_COLLECTED_EVENT_RECORDS:
                raise InspectionTooLarge(
                    "Session inspection exceeds the "
                    f"{_MAX_COLLECTED_EVENT_RECORDS}-event safety limit. Narrow the event "
                    "window with --after-sequence or --before-sequence."
                )
        cursor = page[-1].sequence
        if len(page) < _EVENT_QUERY_PAGE_SIZE:
            return records


def usage_inspection_record(record: EventRecord) -> EventRecord:
    event = record.event
    payload: dict[str, Any] = {}
    if event.type == EventType.MODEL_COMPLETED:
        transcript_cursor = event.payload.get("transcript_cursor")
        if transcript_cursor is not None:
            payload["transcript_cursor"] = transcript_cursor
        try:
            metrics = summary_usage_metrics_from_event_payload(event.payload)
        except (TypeError, ValueError):
            payload["usage_metrics"] = {"_invalid": True}
        else:
            if metrics is not None:
                payload["usage_metrics"] = metrics.model_dump(mode="json")
    elif event.type == EventType.BUDGET_RESERVED:
        for key in ("reservation_id", "currency", "requested"):
            if key in event.payload:
                payload[key] = event.payload[key]
    elif event.type == EventType.BUDGET_RECONCILED:
        for key in ("reservation_id", "reserved_amount", "actual_amount"):
            if key in event.payload:
                payload[key] = event.payload[key]
        pricing = event.payload.get("pricing")
        if pricing is None:
            payload[_USAGE_INSPECTION_PRICING_STATE_KEY] = "unpriced"
        elif is_complete_budget_reconciliation_pricing(pricing):
            payload[_USAGE_INSPECTION_PRICING_STATE_KEY] = "priced"
        else:
            payload[_USAGE_INSPECTION_PRICING_STATE_KEY] = "unknown"
    elif event.type == EventType.BUDGET_RESERVATION_RELEASED:
        if "reservation_id" in event.payload:
            payload["reservation_id"] = event.payload["reservation_id"]
    return EventRecord(
        sequence=record.sequence,
        event=event.model_copy(update={"payload": payload}),
    )


def tool_inspection_record(
    record: EventRecord, disclosure: ArgumentDisclosure = LOCAL_DISCLOSURE
) -> EventRecord:
    event = record.event
    payload: dict[str, Any] = {}
    for key in (
        "tool_call_id",
        "model_step_id",
        "model_attempt_id",
        "tool_round_id",
        "approval_id",
        "input_id",
        "approval_required",
        "registration_state",
    ):
        if key in event.payload:
            payload[key] = event.payload[key]
    if event.type == EventType.TOOL_CALL_STARTED:
        state, summary = _tool_argument_inspection_projection(
            event.payload, disclosure, pre_execution=True
        )
        payload["_arguments_state"] = state
        payload["_argument_summary"] = summary
    if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED:
        approval = _tool_approval_payload(record)
        if approval is not None:
            if not _tool_approval_identity_matches_event(record):
                payload[_TOOL_INSPECTION_IDENTITY_CONFLICT_KEY] = True
            compact_approval = {
                key: approval[key]
                for key in ("approval_id", "tool_call_id", "tool_name")
                if key in approval
            }
            state, summary = _tool_argument_inspection_projection(
                approval, disclosure, pre_execution=True
            )
            compact_approval["_arguments_state"] = state
            compact_approval["_argument_summary"] = summary
            nested_calls = approval.get("tool_calls")
            if type(nested_calls) is list:
                compact_approval["tool_calls"] = [
                    {
                        key: item[key]
                        for key in (
                            "tool_call_id",
                            "tool_name",
                            "policy_decision",
                            "policy_evidence",
                        )
                        if key in item
                    }
                    | _tool_argument_inspection_fields(item, disclosure)
                    for item in nested_calls
                    if type(item) is dict
                ]
            payload["approval"] = compact_approval
    if event.type == EventType.SESSION_AWAITING_USER_INPUT:
        payload["question"] = truncate_utf8(str(event.payload.get("question", "")), max_bytes=512)
        options = event.payload.get("options")
        if type(options) is list:
            payload["options"] = [
                truncate_utf8(str(option), max_bytes=256) for option in options[:100]
            ]
        nested_calls = event.payload.get("tool_calls")
        if type(nested_calls) is list:
            payload["tool_calls"] = [
                {
                    key: item[key]
                    for key in ("tool_call_id", "tool_name", "policy_decision")
                    if key in item
                }
                | _tool_argument_inspection_fields(item, disclosure)
                for item in nested_calls
                if type(item) is dict
            ]
    if (
        event.type == EventType.TOOL_CALL_APPROVAL_DENIED
        and type(event.payload.get("expired")) is bool
    ):
        payload["expired"] = event.payload["expired"]
    if event.type == EventType.TOOL_CALL_BLOCKED:
        for key in ("decision", "blocked_by", "requested_decision"):
            if key in event.payload:
                payload[key] = event.payload[key]
    if event.type in _TOOL_TERMINAL_TYPES:
        if "arguments_state" in event.payload or "arguments" in event.payload:
            state, summary = _tool_argument_inspection_projection(
                event.payload, disclosure, pre_execution=False
            )
            payload["_arguments_state"] = state
            payload["_argument_summary"] = summary
        result = event.payload.get("result")
        result = result if type(result) is dict else {}
        content = result.get("content")
        structured = result.get("structured")
        artifacts = result.get("artifacts")
        payload["_inspection_result"] = {
            "rendered_content_bytes": (len(content.encode("utf-8")) if type(content) is str else 0),
            "structured_result_bytes": (
                compact_json_utf8_size(structured) if type(structured) is dict else 0
            ),
            "artifact_bytes": (compact_json_utf8_size(artifacts) if type(artifacts) is list else 0),
            "returned": (
                _optional_nonnegative_int(structured.get("returned"))
                if type(structured) is dict
                else None
            ),
            "truncated": (
                structured.get("truncated")
                if type(structured) is dict and type(structured.get("truncated")) is bool
                else None
            ),
        }
    return EventRecord(
        sequence=record.sequence,
        event=event.model_copy(update={"payload": payload}),
    )


def _optional_string(value: object) -> str | None:
    return value if type(value) is str else None


def _optional_nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


_SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "aws_secret_access_key",
        "aws_access_key_id",
        "auth_token",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "encrypted_content",
        "password",
        "private_key",
        "provider_state",
        "refresh_token",
        "secret",
        "secret_key",
        "session_token",
        "signature",
        "signing_key",
        "token",
        "id_token",
    }
)


_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b(api[-_ ]?key|auth(?:orization)?|access[-_ ]?token|refresh[-_ ]?token|"
    r"password|secret|credential)\b(\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)


_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")


_SECRET_TOKEN_PATTERN = re.compile(
    r"\b(?:sk|gh[pousr]|github_pat)_[A-Za-z0-9_-]{8,}\b|\bsk-[A-Za-z0-9_-]{8,}\b"
)


_POSTGRES_PASSWORD_PATTERN = re.compile(r"(?i)(postgres(?:ql)?://[^:/\s]+:)([^@\s]+)(@)")


_PEM_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?-----END [^-\r\n]*PRIVATE KEY-----",
    re.DOTALL,
)


def redact_inspection_value(value: Any, *, key: str | None = None) -> Any:
    normalized_key = "" if key is None else _normalize_sensitive_key(key)
    if key is not None and (
        normalized_key in _SENSITIVE_KEYS
        or normalized_key.endswith(("_api_key", "_password", "_secret", "_credential", "_token"))
    ):
        return "[REDACTED]"
    if isinstance(value, dict):
        if value.get("type") == "provider_state":
            return {
                item_key: (
                    "[REDACTED]"
                    if _normalize_sensitive_key(str(item_key)) == "state"
                    else redact_inspection_value(item, key=str(item_key))
                )
                for item_key, item in value.items()
            }
        return {
            item_key: redact_inspection_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact_inspection_value(item) for item in value]
    if isinstance(value, str):
        redacted = _PEM_PRIVATE_KEY_PATTERN.sub("[REDACTED PRIVATE KEY]", value)
        redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", redacted)
        redacted = _SECRET_ASSIGNMENT_PATTERN.sub(
            lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
            redacted,
        )
        redacted = _SECRET_TOKEN_PATTERN.sub("[REDACTED]", redacted)
        return _POSTGRES_PASSWORD_PATTERN.sub(r"\1***\3", redacted)
    return value


def _normalize_sensitive_key(key: str) -> str:
    separated = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", key)
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", separated)
    return re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")


_PRE_EXECUTION_ARGUMENT_KEYS = ("arguments", "effective_arguments", "approved_arguments")


def _tool_argument_inspection_projection(
    value: Any,
    disclosure: ArgumentDisclosure = LOCAL_DISCLOSURE,
    *,
    pre_execution: bool = False,
) -> tuple[str | None, str]:
    if type(value) is not dict:
        return "unavailable", "[unavailable]"
    state = value.get("arguments_state")
    if (
        pre_execution
        and disclosure.quarantine_pre_execution
        and (state == "quarantined" or any(key in value for key in _PRE_EXECUTION_ARGUMENT_KEYS))
    ):
        return "quarantined", "[quarantined]"
    if state == "quarantined":
        return state, "[quarantined]"
    if state == "unavailable":
        return state, "[unavailable]"
    arguments = value.get("arguments")
    if state == "finalized":
        if type(arguments) is dict:
            return state, _bounded_argument_summary(arguments, disclosure)
        return "unavailable", "[unavailable]"
    if state is not None:
        return "unavailable", "[unavailable]"
    if type(arguments) is dict:
        return None, _bounded_argument_summary(arguments, disclosure)
    return "unavailable", "[unavailable]"


def _tool_argument_inspection_fields(
    value: Any, disclosure: ArgumentDisclosure = LOCAL_DISCLOSURE
) -> dict[str, str | None]:
    state, summary = _tool_argument_inspection_projection(value, disclosure, pre_execution=True)
    return {
        "_arguments_state": state,
        "_argument_summary": summary,
    }


def _bounded_argument_summary(value: Any, disclosure: ArgumentDisclosure = LOCAL_DISCLOSURE) -> str:
    """Render a bounded tool-argument summary after structural redaction."""
    if disclosure.redact is not None:
        value = disclosure.redact(value)
    return bounded_json_summary(redact_inspection_value(value), max_bytes=256)


async def inspect_model_calls(
    reader: EventReader,
    session_id: str,
    window: InspectionWindow,
    *,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    """Per-model-call usage, ledger reservations no call claims, and aggregate counters.

    `calls` and `unmatched_ledger` are pages of the same `offset` and `limit`. The aggregate
    usage is an `AggregateUsageMetrics` instance under `aggregate["usage"]`.
    """

    records = await collect_inspection_records(
        reader,
        session_id,
        event_types=USAGE_INSPECTION_EVENT_TYPES,
        project_record=usage_inspection_record,
        after_sequence=window.after_sequence,
        before_sequence=window.before_sequence,
        interaction_id=window.interaction_id,
    )
    usage_events = [
        record.event
        for record in records
        if record.event.type in {EventType.MODEL_COMPLETED, EventType.TOOL_CALL_STARTED}
    ]
    aggregate = session_usage_summary(session_id, usage_events)
    calls, unmatched_ledger = model_call_usage(records)
    page = calls[offset : offset + limit]
    ledger_page = unmatched_ledger[offset : offset + limit]
    next_offset = offset + len(page)
    ledger_next_offset = offset + len(ledger_page)
    has_more = next_offset < len(calls)
    ledger_has_more = ledger_next_offset < len(unmatched_ledger)
    return {
        "calls": page,
        "offset": offset,
        "next_offset": next_offset if has_more else None,
        "has_more": has_more,
        "total_calls": len(calls),
        "unmatched_ledger": ledger_page,
        "unmatched_ledger_total": len(unmatched_ledger),
        "unmatched_ledger_next_offset": ledger_next_offset if ledger_has_more else None,
        "unmatched_ledger_has_more": ledger_has_more,
        "aggregate": {
            "model_calls": aggregate.model_steps,
            "model_calls_with_usage": count_model_steps_with_usage(usage_events),
            "tool_calls": aggregate.tool_calls,
            "usage": aggregate.usage,
        },
    }


async def inspect_tool_calls(
    reader: EventReader,
    session_id: str,
    window: InspectionWindow,
    *,
    offset: int,
    limit: int,
    disclosure: ArgumentDisclosure = LOCAL_DISCLOSURE,
) -> dict[str, Any]:
    """Paired tool calls without result bodies, as one page of `offset` and `limit`."""

    records = await collect_inspection_records(
        reader,
        session_id,
        event_types=TOOL_INSPECTION_EVENT_TYPES,
        project_record=lambda record: tool_inspection_record(record, disclosure),
        after_sequence=window.after_sequence,
        before_sequence=window.before_sequence,
        interaction_id=window.interaction_id,
    )
    calls = tool_call_rows(records)
    page = calls[offset : offset + limit]
    next_offset = offset + len(page)
    has_more = next_offset < len(calls)
    return {
        "calls": page,
        "offset": offset,
        "next_offset": next_offset if has_more else None,
        "has_more": has_more,
        "total_calls": len(calls),
    }
