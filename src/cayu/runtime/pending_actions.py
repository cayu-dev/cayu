from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any

from pydantic import ValidationError

from cayu._validation import (
    EXECUTION_UNIT_ID_MAX_CHARS,
    JsonUtf8SizeCounter,
    copy_durable_json_value,
)
from cayu.core.events import Event, EventType
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _resume_ledger as resume_ledger
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._event_projection import private_event_linkage_value
from cayu.runtime.approvals import (
    _PENDING_TOOL_APPROVAL_EVENT_PROJECTION_KEYS,
    PendingToolApproval,
    ToolPolicyEvidence,
)
from cayu.runtime.execution_units import ToolRoundIdentity
from cayu.runtime.sessions import (
    MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL,
    MAX_PENDING_ACTION_RESULT_BYTES,
    MAX_PENDING_ACTION_TOOL_CALLS,
    PENDING_ACTION_EVENT_TYPE_VALUES,
    EventRecord,
    PendingActionKind,
    PendingActionRecord,
    PendingActionSession,
    SessionStatus,
)
from cayu.runtime.user_input import pending_user_input_from_checkpoint

PENDING_ACTION_SESSION_STATUSES = frozenset(
    {SessionStatus.INTERRUPTED, SessionStatus.FAILED, SessionStatus.COMPLETED}
)
_PENDING_ACTION_TOOL_STATE_KEYS = (
    "pending_tool_approval",
    "pending_user_input",
    "pending_tool_round",
)
PENDING_ACTION_CHECKPOINT_KEYS = frozenset(_PENDING_ACTION_TOOL_STATE_KEYS)
_TOOL_ROUND_IDENTITY_PAYLOAD_KEYS = frozenset(
    {"model_step_id", "model_attempt_id", "tool_round_id"}
)
_APPROVAL_EVENT_PROJECTION_KEYS = frozenset(_PENDING_TOOL_APPROVAL_EVENT_PROJECTION_KEYS)
_TOOL_EVENT_EVIDENCE_PAYLOAD_KEYS = (
    frozenset({"tool_call_id", "manual_recovery"}) | _TOOL_ROUND_IDENTITY_PAYLOAD_KEYS
)

_PENDING_ACTION_EVENT_PAYLOAD_KEYS: dict[str, frozenset[str]] = {
    "tool.call.approval_requested": frozenset({"approval", "approval_id", "tool_call_id"})
    | _TOOL_ROUND_IDENTITY_PAYLOAD_KEYS,
    "session.awaiting_user_input": frozenset({"input_id", "tool_call_id", "question", "options"})
    | _TOOL_ROUND_IDENTITY_PAYLOAD_KEYS,
    "session.interrupted": frozenset(
        {
            "interruption_type",
            "manual_recovery_required",
            "approval_id",
            "tool_call_id",
            "tool_round_id",
            "error",
            "message",
            "tool_name",
            "approval",
            "user_input",
            resume_ledger.TOOL_EVIDENCE_CONFLICT_PAYLOAD_KEY,
        }
    )
    | _TOOL_ROUND_IDENTITY_PAYLOAD_KEYS,
    "tool.call.started": _TOOL_EVENT_EVIDENCE_PAYLOAD_KEYS | frozenset({"arguments"}),
    "tool.call.completed": _TOOL_EVENT_EVIDENCE_PAYLOAD_KEYS,
    "tool.call.failed": _TOOL_EVENT_EVIDENCE_PAYLOAD_KEYS,
    "tool.call.blocked": _TOOL_EVENT_EVIDENCE_PAYLOAD_KEYS,
    "tool.call.approval_denied": _TOOL_EVENT_EVIDENCE_PAYLOAD_KEYS,
    "session.resumed": _TOOL_ROUND_IDENTITY_PAYLOAD_KEYS,
    "session.completed": frozenset(),
    "session.failed": frozenset(
        {
            "tool_call_id",
            resume_ledger.TOOL_EVIDENCE_CONFLICT_PAYLOAD_KEY,
        }
    )
    | _TOOL_ROUND_IDENTITY_PAYLOAD_KEYS,
}
_OVERSIZED_EVENT_PROJECTION_BYTES_KEY = "__cayu_pending_action_projection_bytes__"
_OVERSIZED_EVENT_PROJECTION_REFERENCE = "cayu_oversized_pending_action_projection"
_TERMINAL_RESULT_VALID_KEY = "__cayu_terminal_result_valid__"
_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
)
_LEDGER_EVENT_TYPES = frozenset({EventType.TOOL_CALL_STARTED} | set(_TERMINAL_EVENT_TYPES))
_LEDGER_EVENT_TYPE_VALUES = frozenset(str(event_type) for event_type in _LEDGER_EVENT_TYPES)


def pending_action_event_retains_history(event_type: str) -> bool:
    """Whether every projected event is needed to classify recovery evidence."""
    return event_type in _LEDGER_EVENT_TYPE_VALUES


def pending_action_evidence_round_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> tool_round_recovery.PendingToolRound | None:
    """Return the one tool round represented by any pending control checkpoint."""

    approval = approval_support.pending_approval_from_checkpoint(checkpoint)
    pending_input = pending_user_input_from_checkpoint(checkpoint)
    pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
    if approval is not None and pending_round is not None:
        if (
            pending_input is not None
            or pending_round.policy_state != "planned"
            or pending_round.tool_round_id != approval.tool_round_id
            or pending_round.model_step_id != approval.model_step_id
            or pending_round.model_attempt_id != approval.model_attempt_id
            or [call.model_dump(mode="json") for call in pending_round.tool_calls]
            != [call.model_dump(mode="json") for call in approval.tool_calls]
        ):
            raise ValueError("Checkpoint contains conflicting paired approval state.")
        return pending_round.model_copy(deep=True)
    candidates = [
        candidate for candidate in (approval, pending_input, pending_round) if candidate is not None
    ]
    if len(candidates) > 1:
        raise ValueError("Checkpoint contains more than one pending tool-round state.")
    if not candidates:
        return None
    candidate = candidates[0]
    if type(candidate) is tool_round_recovery.PendingToolRound:
        return candidate.model_copy(deep=True)
    if type(candidate) is PendingToolApproval:
        return approval_support.planned_tool_round_from_pending_approval(candidate)
    return tool_round_recovery.PendingToolRound(
        tool_round_id=candidate.tool_round_id,
        model_step_id=candidate.model_step_id,
        model_attempt_id=candidate.model_attempt_id,
        agent_name=candidate.agent_name,
        environment_name=candidate.environment_name,
        task_id=candidate.task_id,
        tool_calls=candidate.tool_calls,
        structured_output=candidate.structured_output,
    )


def pending_action_event_matches_tool_round(
    event: Event,
    pending_round: tool_round_recovery.PendingToolRound,
) -> bool:
    return event.payload.get("tool_round_id") == pending_round.tool_round_id or (
        event.payload.get("model_step_id") == pending_round.model_step_id
        and event.payload.get("model_attempt_id") == pending_round.model_attempt_id
    )


def pending_action_event_is_from_different_tool_round(
    event: Event,
    pending_round: tool_round_recovery.PendingToolRound,
) -> bool:
    """Whether complete valid evidence belongs to a genuinely different round.

    Missing, malformed, or internally contradictory identity remains in the
    active projection so publication and recovery fail closed. Only a complete
    identity whose round and originating model attempt are both different is
    safely classified as stale evidence for a reused provider tool-call id.
    """

    try:
        identity = ToolRoundIdentity.model_validate(
            {
                "model_step_id": event.payload.get("model_step_id"),
                "model_attempt_id": event.payload.get("model_attempt_id"),
                "tool_round_id": event.payload.get("tool_round_id"),
            }
        )
    except (TypeError, ValueError):
        return False
    return identity.tool_round_id != pending_round.tool_round_id and (
        identity.model_step_id,
        identity.model_attempt_id,
    ) != (
        pending_round.model_step_id,
        pending_round.model_attempt_id,
    )


def pending_action_event_has_unknown_round_call(
    event: Event,
    pending_round: tool_round_recovery.PendingToolRound,
) -> bool:
    """Whether current-round ledger evidence names no known checkpoint call."""

    if event.type not in _LEDGER_EVENT_TYPES or not pending_action_event_matches_tool_round(
        event,
        pending_round,
    ):
        return False
    tool_call_id = event.payload.get("tool_call_id")
    return type(tool_call_id) is not str or all(
        call.tool_call_id != tool_call_id for call in pending_round.tool_calls
    )


def retain_pending_action_index_record(
    records_by_key: dict[str, EventRecord],
    record: EventRecord,
) -> None:
    """Retain one latest action source or a bounded current-ledger history."""
    event_type = str(record.event.type)
    if not pending_action_event_retains_history(event_type):
        records_by_key[event_type] = record
        return

    records_by_key[f"{event_type}:{record.sequence}"] = record
    ledger_records = [
        (key, retained)
        for key, retained in records_by_key.items()
        if retained.event.type in _LEDGER_EVENT_TYPES
    ]
    maximum = MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL + 1
    if len(ledger_records) <= maximum:
        return
    oldest_key, _ = min(ledger_records, key=lambda item: item[1].sequence)
    records_by_key.pop(oldest_key)


def checkpoint_has_pending_action_candidate(checkpoint: dict[str, Any] | None) -> bool:
    return checkpoint is not None and any(
        checkpoint.get(key) is not None for key in PENDING_ACTION_CHECKPOINT_KEYS
    )


def project_pending_action_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    projected = {
        key: copy_durable_json_value(checkpoint[key], key)
        for key in PENDING_ACTION_CHECKPOINT_KEYS
        if checkpoint.get(key) is not None
    }
    return projected or None


def pending_action_checkpoint_tool_call_count(
    checkpoint: dict[str, Any] | None,
) -> int:
    """Return the largest call array that bounded inspection may expand.

    A valid checkpoint contains exactly one pending tool state. Inspect every
    candidate defensively so malformed multi-state checkpoints cannot hide an
    over-complex approval or user-input round from SQL preflight.
    """

    maximum = 0
    if checkpoint is None:
        return maximum
    for key in _PENDING_ACTION_TOOL_STATE_KEYS:
        state = checkpoint.get(key)
        tool_calls = state.get("tool_calls") if type(state) is dict else None
        if type(tool_calls) is list:
            maximum = max(maximum, len(tool_calls))
    return maximum


def pending_action_checkpoint_metrics(
    checkpoint: dict[str, Any] | None,
) -> tuple[int | None, int, int]:
    """Return persisted bounded-query metadata for one checkpoint.

    ``None`` means the checkpoint has no pending-action key. Over-complex tool
    rounds use a zero byte sentinel because the call-count guard rejects them
    first. Otherwise the byte count is computed without serializing or copying
    the projected values; stores persist it beside the checkpoint in the same
    write transaction so read-side guards never have to materialize an oversized
    projection merely to measure it.
    """
    if not checkpoint_has_pending_action_candidate(checkpoint):
        return None, 0, 0
    assert checkpoint is not None
    flags = (
        (1 if checkpoint.get("pending_tool_approval") is not None else 0)
        | (2 if checkpoint.get("pending_user_input") is not None else 0)
        | (4 if checkpoint.get("pending_tool_round") is not None else 0)
    )
    projected = {
        key: checkpoint[key]
        for key in PENDING_ACTION_CHECKPOINT_KEYS
        if checkpoint.get(key) is not None
    }
    tool_call_count = pending_action_checkpoint_tool_call_count(projected)
    if tool_call_count > MAX_PENDING_ACTION_TOOL_CALLS:
        return 0, tool_call_count, flags
    counter = JsonUtf8SizeCounter(2**63 - 1)
    if not counter.value(projected):  # pragma: no cover - in-memory JSON cannot reach int64 bytes.
        raise ValueError("Pending-action checkpoint projection is too large to measure.")
    return (2**63 - 1) - counter.remaining, tool_call_count, flags


def pending_action_checkpoint_lookup_ids(
    checkpoint: dict[str, Any] | None,
) -> frozenset[str]:
    """Return the durable identifiers needed to resolve one current action."""
    try:
        approval = approval_support.pending_approval_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        approval = None
    try:
        pending_input = pending_user_input_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        pending_input = None
    try:
        evidence_round = pending_action_evidence_round_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        evidence_round = None

    identifiers: set[str] = set()
    if approval is not None:
        identifiers.add(approval.approval_id)
    if pending_input is not None:
        identifiers.add(pending_input.input_id)
    if evidence_round is not None:
        identifiers.add(evidence_round.tool_round_id)
        identifiers.update(call.tool_call_id for call in evidence_round.tool_calls)
    return frozenset(identifiers)


def pending_action_event_lookup_id(event: Event) -> str | None:
    """Project the same one event identifier indexed by SQLite and PostgreSQL.

    Tool ledger evidence is owned by the call even when a resumed call also
    carries its approval or user-input pause id. Pause checkpoints are cleared
    before the enclosing tool round, so indexing ledger evidence by the pause
    would make it unreachable from the surviving round checkpoint.
    """
    payload = event.payload
    approval = _object_payload(payload.get("approval"))
    user_input = _object_payload(payload.get("user_input"))
    if event.type in _LEDGER_EVENT_TYPES:
        tool_call_id = _optional_payload_string(payload, "tool_call_id")
        if tool_call_id is not None:
            return tool_call_id
    for value in (
        _optional_payload_string(payload, "approval_id"),
        _optional_payload_string(approval, "approval_id"),
        _optional_payload_string(payload, "input_id"),
        _optional_payload_string(user_input, "input_id"),
        _optional_payload_string(payload, "tool_call_id"),
        _optional_payload_string(payload, "tool_round_id"),
    ):
        if value is not None:
            return value
    return None


def pending_action_lookup_key(value: str) -> str:
    """Return the fixed-size durable lookup key for an action identifier."""
    if type(value) is not str:
        raise TypeError("Pending-action lookup identifiers must be strings.")
    return sha256(value.encode("utf-8")).hexdigest()


def pending_action_event_storage_values(
    event: Event,
) -> tuple[str | None, str | None, int | None]:
    """Return the fixed-size lookup key and compact durable event projection.

    SQL stores persist these values with the event so bounded control-plane
    reads never parse or detoast the original, potentially arbitrary payload.
    Non-pending-action event types have no projection metadata.
    """
    if str(event.type) not in PENDING_ACTION_EVENT_TYPE_VALUES:
        return None, None, None
    lookup_id = pending_action_event_lookup_id(event)
    lookup_key = pending_action_lookup_key(lookup_id) if lookup_id is not None else None
    payload_view = _pending_action_event_payload_view(event)
    projection_view = {
        "type": str(event.type),
        "session_id": event.session_id,
        "interaction_id": event.interaction_id,
        "id": event.id,
        "timestamp": event.timestamp,
        "agent_name": event.agent_name,
        "environment_name": event.environment_name,
        "workflow_name": event.workflow_name,
        "tool_name": event.tool_name,
        "payload": payload_view,
    }
    counter = JsonUtf8SizeCounter(MAX_PENDING_ACTION_RESULT_BYTES)
    if not counter.value(projection_view):
        projected = _oversized_pending_action_event_projection(event)
        projection = projected.model_dump(mode="json")
        projection_json = json.dumps(
            projection,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return (
            lookup_key,
            projection_json,
            MAX_PENDING_ACTION_RESULT_BYTES + 1,
        )
    projected = _project_pending_action_event(event, payload_view=payload_view)
    projection = projected.model_dump(mode="json")
    projection_json = json.dumps(projection, ensure_ascii=False, separators=(",", ":"))
    return (
        lookup_key,
        projection_json,
        len(projection_json.encode("utf-8")),
    )


def _oversized_pending_action_event_projection(event: Event) -> Event:
    """Retain only bounded identity needed to scope an oversized event.

    The byte marker prevents callers from materializing the event as a normal
    action source. Keeping canonical-sized execution identifiers lets every
    store exclude stale evidence for a reused provider tool-call id before
    applying that size failure to the current round.
    """

    identity_payload = {
        key: value
        for key in ("model_step_id", "model_attempt_id", "tool_round_id")
        if type(value := event.payload.get(key)) is str
        and len(value) <= EXECUTION_UNIT_ID_MAX_CHARS
    }
    return Event(
        type=event.type,
        session_id=_OVERSIZED_EVENT_PROJECTION_REFERENCE,
        id=_OVERSIZED_EVENT_PROJECTION_REFERENCE,
        timestamp=event.timestamp,
        payload={
            **identity_payload,
            _OVERSIZED_EVENT_PROJECTION_BYTES_KEY: (MAX_PENDING_ACTION_RESULT_BYTES + 1),
        },
    )


def _pending_action_event_payload_view(event: Event) -> dict[str, Any]:
    keys = _PENDING_ACTION_EVENT_PAYLOAD_KEYS.get(str(event.type), frozenset())
    payload: dict[str, Any] = {}
    for key in keys:
        if key not in event.payload:
            continue
        value = event.payload[key]
        if key == "approval":
            projected = _project_payload_object_view(
                value,
                _APPROVAL_EVENT_PROJECTION_KEYS,
            )
            if projected is not None:
                payload[key] = projected
        elif key == "user_input":
            projected = _project_payload_object_view(
                value,
                frozenset({"input_id", "tool_call_id", "question", "options"}),
            )
            if projected is not None:
                payload[key] = projected
        else:
            payload[key] = value
    if event.type in _TERMINAL_EVENT_TYPES:
        payload[_TERMINAL_RESULT_VALID_KEY] = _terminal_result_payload_is_valid(event.payload)
    return payload


def _terminal_result_payload_is_valid(payload: dict[str, Any]) -> bool:
    """Validate the bounded ToolResult envelope without copying its arbitrary JSON data."""
    result = payload.get("result")
    if type(result) is not dict:
        return False
    if set(result) - {"content", "structured", "artifacts", "is_error"}:
        return False
    if "content" in result and type(result["content"]) is not str:
        return False
    if (
        "structured" in result
        and result["structured"] is not None
        and type(result["structured"]) is not dict
    ):
        return False
    artifacts = result.get("artifacts", [])
    if type(artifacts) is not list or any(type(item) is not dict for item in artifacts):
        return False
    return "is_error" not in result or type(result["is_error"]) is bool


def _project_pending_action_event(
    event: Event,
    *,
    payload_view: dict[str, Any] | None = None,
) -> Event:
    return Event(
        type=event.type,
        session_id=event.session_id,
        interaction_id=event.interaction_id,
        id=event.id,
        timestamp=event.timestamp,
        agent_name=event.agent_name,
        environment_name=event.environment_name,
        workflow_name=event.workflow_name,
        tool_name=event.tool_name,
        payload=(
            _pending_action_event_payload_view(event) if payload_view is None else payload_view
        ),
    )


def project_pending_action_event_record(record: EventRecord) -> EventRecord:
    """Copy only event payload fields used to derive a pending action.

    Tool results and unrelated lifecycle payloads can be arbitrarily large. They
    are deliberately excluded: pending-action recovery only needs their durable
    event type and current round/call identifiers.
    """
    _lookup_key, projection_json, projection_bytes = pending_action_event_storage_values(
        record.event
    )
    if projection_json is None:
        if projection_bytes is None:
            raise ValueError("Pending-action event projection metadata is missing.")
        event = Event(
            type=record.event.type,
            session_id=record.event.session_id,
            interaction_id=record.event.interaction_id,
            id=record.event.id,
            timestamp=record.event.timestamp,
            agent_name=record.event.agent_name,
            environment_name=record.event.environment_name,
            workflow_name=record.event.workflow_name,
            tool_name=record.event.tool_name,
            payload={_OVERSIZED_EVENT_PROJECTION_BYTES_KEY: projection_bytes},
        )
    else:
        event = Event.model_validate_json(projection_json)
    return EventRecord(sequence=record.sequence, event=event)


def _project_payload_object_view(value: Any, keys: frozenset[str]) -> dict[str, Any] | None:
    if type(value) is not dict:
        return None
    return {key: value[key] for key in keys if key in value}


def _approval_event_matches_checkpoint(
    event_pending: PendingToolApproval,
    checkpoint_pending: PendingToolApproval | None,
) -> bool:
    """Compare approval authority while permitting bounded audit metadata."""

    if checkpoint_pending is None:
        return False

    def authority_view(approval: PendingToolApproval) -> dict[str, Any]:
        payload = approval.model_dump(mode="json")
        payload.pop("metadata", None)
        tool_calls = payload.get("tool_calls")
        if type(tool_calls) is not list:
            raise TypeError("Pending approval tool calls must be a list.")
        for pending_call in tool_calls:
            if type(pending_call) is not dict:
                raise TypeError("Pending approval tool calls must be objects.")
            pending_call.pop("metadata", None)
        return payload

    return authority_view(event_pending) == authority_view(checkpoint_pending)


def pending_action_event_projection_bytes(record: EventRecord) -> int | None:
    value = record.event.payload.get(_OVERSIZED_EVENT_PROJECTION_BYTES_KEY)
    return value if type(value) is int and value > 0 else None


def _pending_action_source_reference(record: EventRecord) -> EventRecord:
    """Keep stable source identity without duplicating its arbitrary payload."""
    return EventRecord(
        sequence=record.sequence,
        event=record.event.model_copy(update={"payload": {}}, deep=True),
    )


def _pending_action_source_linkage(record: EventRecord) -> dict[str, str]:
    """Retain only schema-proven linkage before bounding the source event."""

    linkage: dict[str, str] = {}
    for field_name in ("approval_id", "input_id", "tool_round_id", "tool_call_id"):
        value = private_event_linkage_value(record.event, field_name=field_name)
        if value is not None:
            linkage[field_name] = value
    return linkage


def select_pending_action_indexed_records(
    checkpoint: dict[str, Any] | None,
    records_by_lookup_key: Mapping[str, Mapping[str, EventRecord]],
    latest_barrier: EventRecord | None,
) -> tuple[list[EventRecord], bool]:
    """Select bounded current-action records from identifier/event-type indexes."""
    selected: dict[int, EventRecord] = {}
    try:
        evidence_round = pending_action_evidence_round_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        evidence_round = None
    ledger_too_complex = False
    for lookup_id in pending_action_checkpoint_lookup_ids(checkpoint):
        lookup_key = pending_action_lookup_key(lookup_id)
        ledger_count = 0
        for record in records_by_lookup_key.get(lookup_key, {}).values():
            if record.event.type in _LEDGER_EVENT_TYPES:
                if evidence_round is None or not pending_action_event_matches_tool_round(
                    record.event,
                    evidence_round,
                ):
                    continue
                ledger_count += 1
                if ledger_count > MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL:
                    ledger_too_complex = True
                    continue
            selected[record.sequence] = record
    if latest_barrier is not None:
        selected[latest_barrier.sequence] = latest_barrier
    return (
        [selected[sequence] for sequence in sorted(selected, reverse=True)],
        ledger_too_complex,
    )


def _object_payload(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _optional_payload_string(payload: dict[str, Any] | None, key: str) -> str | None:
    if payload is None:
        return None
    value = payload.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _payload_string_list(payload: dict[str, Any] | None, key: str) -> list[str]:
    if payload is None:
        return []
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def pending_action_matches_query(action: PendingActionRecord, q: str | None) -> bool:
    if q is None:
        return True
    needle = q.casefold()
    values = (
        action.id,
        str(action.kind),
        action.title,
        action.detail,
        action.tool_name,
        action.approval_id,
        action.input_id,
        action.round_id,
        action.tool_call_id,
        action.question,
        action.session.id,
        action.session.agent_name,
        action.session.provider_name,
        action.session.model,
        action.session.environment_name,
    )
    return any(value is not None and needle in value.casefold() for value in values)


def _action_from_record(
    *,
    session: PendingActionSession,
    record: EventRecord,
    action_kind: PendingActionKind,
    title: str,
    detail: str | None = None,
    tool_name: str | None = None,
    approval_id: str | None = None,
    input_id: str | None = None,
    round_id: str | None = None,
    tool_call_id: str | None = None,
    policy_evidence: ToolPolicyEvidence | None = None,
    question: str | None = None,
    options: list[str] | None = None,
    arguments: dict[str, Any] | None = None,
) -> PendingActionRecord:
    discriminator = approval_id or input_id or tool_call_id or record.event.id
    return PendingActionRecord(
        id=f"{session.id}:{record.sequence}:{action_kind}:{discriminator}",
        kind=action_kind,
        session=session,
        event=_pending_action_source_reference(record),
        title=title,
        detail=detail,
        tool_name=tool_name,
        approval_id=approval_id,
        input_id=input_id,
        round_id=round_id,
        tool_call_id=tool_call_id,
        source_linkage=_pending_action_source_linkage(record),
        policy_evidence=policy_evidence,
        question=question,
        options=options or [],
        arguments=arguments,
    )


def _pending_approval_checkpoint_call(
    checkpoint: dict[str, Any] | None,
    *,
    approval_id: str,
    tool_call_id: str | None = None,
    identity_payload: Mapping[str, Any] | None = None,
    gating_only: bool = False,
) -> dict[str, Any] | None:
    try:
        pending = approval_support.pending_approval_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        return None
    if pending is None or pending.approval_id != approval_id:
        return None
    if identity_payload is not None and any(
        identity_payload.get(key) != value
        for key, value in (
            ("model_step_id", pending.model_step_id),
            ("model_attempt_id", pending.model_attempt_id),
            ("tool_round_id", pending.tool_round_id),
        )
    ):
        return None
    if tool_call_id is None:
        return {
            "tool_name": pending.tool_name,
            "arguments": pending.arguments,
            "tool_call_id": pending.tool_call_id,
            "tool_round_id": pending.tool_round_id,
            "reason": pending.reason,
            "policy_evidence": approval_support.effective_tool_policy_evidence(
                next(
                    call for call in pending.tool_calls if call.tool_call_id == pending.tool_call_id
                )
            ),
        }
    if gating_only and pending.tool_call_id != tool_call_id:
        return None
    for call in pending.tool_calls:
        if call.tool_call_id == tool_call_id:
            return {
                "tool_name": call.tool_name,
                "arguments": call.arguments,
                "tool_call_id": call.tool_call_id,
                "tool_round_id": pending.tool_round_id,
                "reason": pending.reason if call.tool_call_id == pending.tool_call_id else None,
                "policy_evidence": approval_support.effective_tool_policy_evidence(call),
            }
    return None


def _pending_user_input_checkpoint_call(
    checkpoint: dict[str, Any] | None,
    *,
    input_id: str,
    tool_call_id: str | None = None,
    identity_payload: Mapping[str, Any] | None = None,
    gating_only: bool = False,
) -> dict[str, Any] | None:
    try:
        pending = pending_user_input_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        return None
    if pending is None or pending.input_id != input_id:
        return None
    if identity_payload is not None and any(
        identity_payload.get(key) != value
        for key, value in (
            ("model_step_id", pending.model_step_id),
            ("model_attempt_id", pending.model_attempt_id),
            ("tool_round_id", pending.tool_round_id),
        )
    ):
        return None
    if tool_call_id is None:
        return {
            "tool_name": pending.tool_name,
            "arguments": pending.arguments,
            "tool_call_id": pending.tool_call_id,
            "tool_round_id": pending.tool_round_id,
            "question": pending.question,
            "options": list(pending.options),
        }
    if gating_only and pending.tool_call_id != tool_call_id:
        return None
    for call in pending.tool_calls:
        if call.tool_call_id == tool_call_id:
            return {
                "tool_name": call.tool_name,
                "arguments": call.arguments,
                "tool_call_id": call.tool_call_id,
                "tool_round_id": pending.tool_round_id,
                "question": pending.question if call.tool_call_id == pending.tool_call_id else None,
                "options": (
                    list(pending.options) if call.tool_call_id == pending.tool_call_id else []
                ),
            }
    return None


def _pending_tool_round_checkpoint_call(
    checkpoint: dict[str, Any] | None,
    *,
    round_id: str,
    tool_call_id: str,
    identity_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    try:
        pending = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        return None
    if pending is None or pending.tool_round_id != round_id:
        return None
    if identity_payload is not None and any(
        identity_payload.get(key) != value
        for key, value in (
            ("model_step_id", pending.model_step_id),
            ("model_attempt_id", pending.model_attempt_id),
            ("tool_round_id", pending.tool_round_id),
        )
    ):
        return None
    for call in pending.tool_calls:
        if call.tool_call_id == tool_call_id:
            return {
                "tool_name": call.tool_name,
                "arguments": call.arguments,
            }
    return None


def _pending_tool_round_evidence(
    records_desc: list[EventRecord],
    pending_round: tool_round_recovery.PendingToolRound,
) -> resume_ledger.ToolCallEvidenceLedger:
    identity = tool_round_recovery.pending_tool_round_identity(pending_round)
    return resume_ledger.scan_projected_tool_call_evidence(
        events=(record.event for record in reversed(records_desc)),
        pending_calls=pending_round.tool_calls,
        in_scope=lambda event: identity.matches_payload(event.payload),
        candidate_scope=lambda event: pending_action_event_matches_tool_round(event, pending_round),
        terminal_event_types=_TERMINAL_EVENT_TYPES,
        terminal_result_is_valid=lambda event: (
            event.payload.get(_TERMINAL_RESULT_VALID_KEY) is True
        ),
    )


def _tool_round_manual_recovery_action(
    session: PendingActionSession,
    records_desc: list[EventRecord],
    checkpoint: dict[str, Any] | None,
) -> PendingActionRecord | None:
    try:
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        return None
    if pending_round is None:
        return None

    identity = tool_round_recovery.pending_tool_round_identity(pending_round)
    evidence = _pending_tool_round_evidence(records_desc, pending_round)
    if evidence.scope_conflicting:
        return None
    unresolved_calls = [
        call
        for call in pending_round.tool_calls
        if call.tool_call_id in evidence.started_without_terminal_ids
    ]
    if not unresolved_calls:
        return None

    pending_call = unresolved_calls[0]
    source_record = next(
        (
            record
            for record in records_desc
            if record.event.type in {EventType.SESSION_INTERRUPTED, EventType.SESSION_FAILED}
            and identity.matches_payload(record.event.payload)
        ),
        None,
    )
    if source_record is None:
        source_record = next(
            (
                record
                for record in records_desc
                if record.event.type in _LEDGER_EVENT_TYPES
                and pending_action_event_matches_tool_round(record.event, pending_round)
                and record.event.payload.get("tool_call_id") == pending_call.tool_call_id
            ),
            None,
        )
    if source_record is None:
        return None

    if pending_call.tool_call_id in evidence.conflicting_ids:
        detail = (
            "Tool execution has malformed or contradictory durable evidence; "
            "its outcome must be reconciled manually."
        )
    else:
        detail = (
            "Tool started but no terminal result was recorded before the session failed."
            if session.status == SessionStatus.FAILED
            else "Tool started but no terminal result was recorded."
        )
    return _action_from_record(
        session=session,
        record=source_record,
        action_kind=PendingActionKind.MANUAL_RECOVERY,
        title="Manual recovery required",
        detail=detail,
        tool_name=pending_call.tool_name,
        round_id=pending_round.tool_round_id,
        tool_call_id=pending_call.tool_call_id,
        arguments=pending_call.arguments,
    )


def pending_action_from_records(
    session: PendingActionSession,
    records_desc: list[EventRecord],
    checkpoint: dict[str, Any] | None,
) -> PendingActionRecord | None:
    """Project one current action from bounded action-specific event records."""
    if session.status == SessionStatus.INTERRUPTED:
        for record in records_desc:
            event = record.event
            event_type = str(event.type)
            if event_type in {"session.resumed", "session.completed", "session.failed"}:
                break

            payload = event.payload
            interruption_type = _optional_payload_string(payload, "interruption_type")
            manual_recovery_required = payload.get("manual_recovery_required") is True

            if event_type == "tool.call.approval_requested":
                approval = _object_payload(payload.get("approval"))
                approval_id = _optional_payload_string(payload, "approval_id")
                tool_call_id = _optional_payload_string(payload, "tool_call_id")
                try:
                    event_pending = PendingToolApproval.from_event(event)
                    checkpoint_pending = approval_support.pending_approval_from_checkpoint(
                        checkpoint
                    )
                except (TypeError, ValueError, ValidationError):
                    event_pending = None
                    checkpoint_pending = None
                if (
                    event_pending is not None
                    and _approval_event_matches_checkpoint(
                        event_pending,
                        checkpoint_pending,
                    )
                    and approval_id is not None
                    and tool_call_id is not None
                ):
                    checkpoint_call = _pending_approval_checkpoint_call(
                        checkpoint,
                        approval_id=approval_id,
                        tool_call_id=tool_call_id,
                        identity_payload=payload,
                        gating_only=True,
                    )
                    if checkpoint_call is not None:
                        return _action_from_record(
                            session=session,
                            record=record,
                            action_kind=PendingActionKind.TOOL_APPROVAL,
                            title="Tool approval required",
                            detail=_optional_payload_string(approval, "reason")
                            or _optional_payload_string(checkpoint_call, "reason"),
                            tool_name=_optional_payload_string(checkpoint_call, "tool_name")
                            or event.tool_name,
                            approval_id=approval_id,
                            round_id=_optional_payload_string(checkpoint_call, "tool_round_id"),
                            tool_call_id=tool_call_id,
                            policy_evidence=checkpoint_call.get("policy_evidence"),
                            arguments=_object_payload(checkpoint_call.get("arguments")) or {},
                        )

            if event_type == "session.awaiting_user_input":
                input_id = _optional_payload_string(payload, "input_id")
                tool_call_id = _optional_payload_string(payload, "tool_call_id")
                if input_id is not None and tool_call_id is not None:
                    checkpoint_call = _pending_user_input_checkpoint_call(
                        checkpoint,
                        input_id=input_id,
                        tool_call_id=tool_call_id,
                        identity_payload=payload,
                        gating_only=True,
                    )
                    if checkpoint_call is not None:
                        question = (
                            _optional_payload_string(checkpoint_call, "question")
                            or "Input required"
                        )
                        return _action_from_record(
                            session=session,
                            record=record,
                            action_kind=PendingActionKind.USER_INPUT,
                            title="User input required",
                            detail=question,
                            tool_name=event.tool_name
                            or _optional_payload_string(checkpoint_call, "tool_name"),
                            input_id=input_id,
                            round_id=_optional_payload_string(
                                checkpoint_call,
                                "tool_round_id",
                            ),
                            tool_call_id=tool_call_id,
                            question=question,
                            options=_payload_string_list(checkpoint_call, "options"),
                            arguments=_object_payload(checkpoint_call.get("arguments")),
                        )

            if event_type != "session.interrupted":
                continue

            if manual_recovery_required:
                approval = _object_payload(payload.get("approval"))
                user_input = _object_payload(payload.get("user_input"))
                approval_id = _optional_payload_string(
                    payload, "approval_id"
                ) or _optional_payload_string(approval, "approval_id")
                input_id = _optional_payload_string(user_input, "input_id")
                tool_call_id = _optional_payload_string(payload, "tool_call_id") or (
                    _optional_payload_string(user_input, "tool_call_id")
                )
                round_id = _optional_payload_string(payload, "tool_round_id")
                if tool_call_id is None or (
                    approval_id is None and input_id is None and round_id is None
                ):
                    continue
                if input_id is not None:
                    checkpoint_call = _pending_user_input_checkpoint_call(
                        checkpoint,
                        input_id=input_id,
                        tool_call_id=tool_call_id,
                        identity_payload=payload,
                    )
                elif approval_id is not None:
                    checkpoint_call = _pending_approval_checkpoint_call(
                        checkpoint,
                        approval_id=approval_id,
                        tool_call_id=tool_call_id,
                        identity_payload=payload,
                    )
                else:
                    checkpoint_call = _pending_tool_round_checkpoint_call(
                        checkpoint,
                        round_id=round_id or "",
                        tool_call_id=tool_call_id,
                        identity_payload=payload,
                    )
                if checkpoint_call is None:
                    continue
                arguments = _object_payload(approval.get("arguments")) if approval else None
                if arguments is None:
                    arguments = _object_payload(checkpoint_call.get("arguments"))
                return _action_from_record(
                    session=session,
                    record=record,
                    action_kind=PendingActionKind.MANUAL_RECOVERY,
                    title="Manual recovery required",
                    detail=_optional_payload_string(payload, "error")
                    or _optional_payload_string(payload, "message")
                    or "A previously started tool result must be reconciled before the session can continue.",
                    tool_name=_optional_payload_string(payload, "tool_name")
                    or _optional_payload_string(approval, "tool_name")
                    or event.tool_name
                    or _optional_payload_string(checkpoint_call, "tool_name"),
                    approval_id=approval_id,
                    input_id=input_id,
                    round_id=round_id,
                    tool_call_id=tool_call_id,
                    question=_optional_payload_string(user_input, "question"),
                    options=_payload_string_list(user_input, "options"),
                    arguments=arguments,
                )

            if interruption_type == "tool_approval_required":
                approval = _object_payload(payload.get("approval"))
                approval_id = _optional_payload_string(approval, "approval_id")
                if approval is not None and approval_id is not None:
                    checkpoint_call = _pending_approval_checkpoint_call(
                        checkpoint,
                        approval_id=approval_id,
                        identity_payload=payload,
                    )
                    if checkpoint_call is not None:
                        return _action_from_record(
                            session=session,
                            record=record,
                            action_kind=PendingActionKind.TOOL_APPROVAL,
                            title="Tool approval required",
                            detail=_optional_payload_string(approval, "reason")
                            or _optional_payload_string(checkpoint_call, "reason"),
                            tool_name=_optional_payload_string(checkpoint_call, "tool_name")
                            or event.tool_name,
                            approval_id=approval_id,
                            round_id=_optional_payload_string(checkpoint_call, "tool_round_id"),
                            tool_call_id=_optional_payload_string(checkpoint_call, "tool_call_id"),
                            arguments=_object_payload(checkpoint_call.get("arguments")) or {},
                        )

            if interruption_type == "user_input_required":
                user_input = _object_payload(payload.get("user_input"))
                input_id = _optional_payload_string(user_input, "input_id")
                tool_call_id = _optional_payload_string(user_input, "tool_call_id")
                if user_input is not None and input_id is not None and tool_call_id is not None:
                    checkpoint_call = _pending_user_input_checkpoint_call(
                        checkpoint,
                        input_id=input_id,
                        tool_call_id=tool_call_id,
                        identity_payload=payload,
                        gating_only=True,
                    )
                    if checkpoint_call is not None:
                        question = (
                            _optional_payload_string(checkpoint_call, "question")
                            or "Input required"
                        )
                        return _action_from_record(
                            session=session,
                            record=record,
                            action_kind=PendingActionKind.USER_INPUT,
                            title="User input required",
                            detail=question,
                            tool_name=event.tool_name
                            or _optional_payload_string(checkpoint_call, "tool_name"),
                            input_id=input_id,
                            round_id=_optional_payload_string(
                                checkpoint_call,
                                "tool_round_id",
                            ),
                            tool_call_id=tool_call_id,
                            question=question,
                            options=_payload_string_list(checkpoint_call, "options"),
                            arguments=_object_payload(checkpoint_call.get("arguments")),
                        )

    return _tool_round_manual_recovery_action(session, records_desc, checkpoint)


def pending_action_source_is_invalid(
    session: PendingActionSession,
    checkpoint: dict[str, Any] | None,
    action: PendingActionRecord | None,
    records_desc: list[EventRecord],
) -> bool:
    """Whether a non-null pending checkpoint cannot be resolved safely.

    A valid pending tool round without a manual-recovery action is intentionally
    resumable: all started calls have terminal records, or no call started.
    Approval and user-input state, however, must always have a matching action
    source because generic resume rejects both states.
    """
    if not checkpoint_has_pending_action_candidate(checkpoint):
        return False
    if any(
        record.event.payload.get(resume_ledger.TOOL_EVIDENCE_CONFLICT_PAYLOAD_KEY) is True
        for record in records_desc
    ):
        return True
    if session.status == SessionStatus.COMPLETED:
        return True
    try:
        pending_approval = approval_support.pending_approval_from_checkpoint(checkpoint)
        pending_input = pending_user_input_from_checkpoint(checkpoint)
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
        evidence_round = pending_action_evidence_round_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        return True
    if evidence_round is not None:
        evidence = _pending_tool_round_evidence(records_desc, evidence_round)
        if evidence.scope_conflicting:
            return True
        if evidence.started_without_terminal_ids:
            return action is None or action.kind != PendingActionKind.MANUAL_RECOVERY
    if action is not None:
        return False
    if pending_approval is not None or pending_input is not None:
        return True
    return pending_round is None
