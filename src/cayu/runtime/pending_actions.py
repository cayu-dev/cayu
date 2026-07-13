from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any

from pydantic import ValidationError

from cayu._validation import copy_json_value
from cayu.core.events import Event, EventType
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime.sessions import (
    MAX_PENDING_ACTION_RESULT_BYTES,
    MAX_PENDING_ACTION_TOOL_CALLS,
    PENDING_ACTION_EVENT_TYPE_VALUES,
    EventRecord,
    PendingActionKind,
    PendingActionRecord,
    PendingActionSession,
    SessionStatus,
    _BoundedJsonSize,
)
from cayu.runtime.user_input import pending_user_input_from_checkpoint

PENDING_ACTION_SESSION_STATUSES = frozenset(
    {SessionStatus.INTERRUPTED, SessionStatus.FAILED, SessionStatus.COMPLETED}
)
PENDING_ACTION_CHECKPOINT_KEYS = frozenset(
    {"pending_tool_approval", "pending_user_input", "pending_tool_round"}
)

_PENDING_ACTION_EVENT_PAYLOAD_KEYS: dict[str, frozenset[str]] = {
    "tool.call.approval_requested": frozenset({"approval"}),
    "session.awaiting_user_input": frozenset({"input_id", "tool_call_id", "question", "options"}),
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
        }
    ),
    "tool.call.started": frozenset({"tool_call_id", "tool_round_id"}),
    "tool.call.completed": frozenset({"tool_call_id", "tool_round_id"}),
    "tool.call.failed": frozenset({"tool_call_id", "tool_round_id"}),
    "tool.call.blocked": frozenset({"tool_call_id", "tool_round_id"}),
    "tool.call.approval_denied": frozenset({"tool_call_id", "tool_round_id"}),
    "session.resumed": frozenset(),
    "session.completed": frozenset(),
    "session.failed": frozenset(),
}
_OVERSIZED_EVENT_PROJECTION_BYTES_KEY = "__cayu_pending_action_projection_bytes__"
_TERMINAL_RESULT_VALID_KEY = "__cayu_terminal_result_valid__"
_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
)


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
        key: copy_json_value(checkpoint[key], key)
        for key in PENDING_ACTION_CHECKPOINT_KEYS
        if checkpoint.get(key) is not None
    }
    return projected or None


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
    pending_round = projected.get("pending_tool_round")
    tool_calls = pending_round.get("tool_calls") if type(pending_round) is dict else None
    tool_call_count = len(tool_calls) if type(tool_calls) is list else 0
    if tool_call_count > MAX_PENDING_ACTION_TOOL_CALLS:
        return 0, tool_call_count, flags
    counter = _BoundedJsonSize(2**63 - 1)
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
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        pending_round = None

    identifiers: set[str] = set()
    if approval is not None:
        identifiers.add(approval.approval_id)
    if pending_input is not None:
        identifiers.add(pending_input.input_id)
    if pending_round is not None:
        identifiers.add(pending_round.round_id)
        identifiers.update(call.tool_call_id for call in pending_round.tool_calls)
    return frozenset(identifiers)


def pending_action_event_lookup_id(event: Event) -> str | None:
    """Project the same one event identifier indexed by SQLite and PostgreSQL."""
    payload = event.payload
    approval = _object_payload(payload.get("approval"))
    user_input = _object_payload(payload.get("user_input"))
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
        "id": event.id,
        "timestamp": event.timestamp,
        "agent_name": event.agent_name,
        "environment_name": event.environment_name,
        "workflow_name": event.workflow_name,
        "tool_name": event.tool_name,
        "payload": payload_view,
    }
    counter = _BoundedJsonSize(MAX_PENDING_ACTION_RESULT_BYTES)
    if not counter.value(projection_view):
        return (
            lookup_key,
            None,
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
                frozenset({"approval_id", "reason", "tool_name"}),
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


def pending_action_event_projection_bytes(record: EventRecord) -> int | None:
    value = record.event.payload.get(_OVERSIZED_EVENT_PROJECTION_BYTES_KEY)
    return value if type(value) is int and value > 0 else None


def _pending_action_source_reference(record: EventRecord) -> EventRecord:
    """Keep stable source identity without duplicating its arbitrary payload."""
    return EventRecord(
        sequence=record.sequence,
        event=record.event.model_copy(update={"payload": {}}, deep=True),
    )


def select_pending_action_indexed_records(
    checkpoint: dict[str, Any] | None,
    records_by_lookup_key: Mapping[str, Mapping[str, EventRecord]],
    latest_barrier: EventRecord | None,
) -> list[EventRecord]:
    """Select bounded current-action records from identifier/event-type indexes."""
    selected: dict[int, EventRecord] = {}
    for lookup_id in pending_action_checkpoint_lookup_ids(checkpoint):
        lookup_key = pending_action_lookup_key(lookup_id)
        for record in records_by_lookup_key.get(lookup_key, {}).values():
            selected[record.sequence] = record
    if latest_barrier is not None:
        selected[latest_barrier.sequence] = latest_barrier
    return [selected[sequence] for sequence in sorted(selected, reverse=True)]


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
        question=question,
        options=options or [],
        arguments=arguments,
    )


def _pending_approval_checkpoint_call(
    checkpoint: dict[str, Any] | None,
    *,
    approval_id: str,
    tool_call_id: str | None = None,
) -> dict[str, Any] | None:
    try:
        pending = approval_support.pending_approval_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        return None
    if pending is None or pending.approval_id != approval_id:
        return None
    if tool_call_id is None:
        return {
            "tool_name": pending.tool_name,
            "arguments": pending.arguments,
        }
    for call in pending.tool_calls:
        if call.tool_call_id == tool_call_id:
            return {
                "tool_name": call.tool_name,
                "arguments": call.arguments,
            }
    if pending.tool_call_id == tool_call_id:
        return {
            "tool_name": pending.tool_name,
            "arguments": pending.arguments,
        }
    return None


def _pending_user_input_checkpoint_call(
    checkpoint: dict[str, Any] | None,
    *,
    input_id: str,
    tool_call_id: str | None = None,
) -> dict[str, Any] | None:
    try:
        pending = pending_user_input_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        return None
    if pending is None or pending.input_id != input_id:
        return None
    if tool_call_id is None:
        return {
            "tool_name": pending.tool_name,
            "arguments": pending.arguments,
        }
    for call in pending.tool_calls:
        if call.tool_call_id == tool_call_id:
            return {
                "tool_name": call.tool_name,
                "arguments": call.arguments,
            }
    if pending.tool_call_id == tool_call_id:
        return {
            "tool_name": pending.tool_name,
            "arguments": pending.arguments,
        }
    return None


def _pending_tool_round_checkpoint_call(
    checkpoint: dict[str, Any] | None,
    *,
    round_id: str,
    tool_call_id: str,
) -> dict[str, Any] | None:
    try:
        pending = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        return None
    if pending is None or pending.round_id != round_id:
        return None
    for call in pending.tool_calls:
        if call.tool_call_id == tool_call_id:
            return {
                "tool_name": call.tool_name,
                "arguments": call.arguments,
            }
    return None


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

    started_ids: set[str] = set()
    terminal_ids: set[str] = set()
    terminal_types = {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
    pending_call_ids = {call.tool_call_id for call in pending_round.tool_calls}
    for record in reversed(records_desc):
        event = record.event
        if event.payload.get("tool_round_id") != pending_round.round_id:
            continue
        tool_call_id = event.payload.get("tool_call_id")
        if type(tool_call_id) is not str or tool_call_id not in pending_call_ids:
            continue
        if event.type == EventType.TOOL_CALL_STARTED:
            started_ids.add(tool_call_id)
        elif event.type in terminal_types:
            terminal_ids.add(tool_call_id)
    unresolved_calls = [
        call
        for call in pending_round.tool_calls
        if call.tool_call_id in started_ids and call.tool_call_id not in terminal_ids
    ]
    if not unresolved_calls:
        return None

    pending_call = unresolved_calls[0]
    source_record = next(
        (
            record
            for record in records_desc
            if record.event.type in {EventType.SESSION_INTERRUPTED, EventType.SESSION_FAILED}
            and (
                record.event.payload.get("tool_round_id") in {None, pending_round.round_id}
                or record.event.payload.get("manual_recovery_required") is True
            )
        ),
        None,
    )
    if source_record is None:
        source_record = next(
            (
                record
                for record in records_desc
                if record.event.type == EventType.TOOL_CALL_STARTED
                and record.event.payload.get("tool_round_id") == pending_round.round_id
                and record.event.payload.get("tool_call_id") == pending_call.tool_call_id
            ),
            None,
        )
    if source_record is None:
        return None

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
        round_id=pending_round.round_id,
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
                approval_id = _optional_payload_string(approval, "approval_id")
                if approval is not None and approval_id is not None:
                    checkpoint_call = _pending_approval_checkpoint_call(
                        checkpoint, approval_id=approval_id
                    )
                    if checkpoint_call is not None:
                        return _action_from_record(
                            session=session,
                            record=record,
                            action_kind=PendingActionKind.TOOL_APPROVAL,
                            title="Tool approval required",
                            detail=_optional_payload_string(approval, "reason"),
                            tool_name=_optional_payload_string(approval, "tool_name")
                            or _optional_payload_string(checkpoint_call, "tool_name")
                            or event.tool_name,
                            approval_id=approval_id,
                            arguments=_object_payload(approval.get("arguments"))
                            or _object_payload(checkpoint_call.get("arguments"))
                            or {},
                        )

            if event_type == "session.awaiting_user_input":
                input_id = _optional_payload_string(payload, "input_id")
                if input_id is not None:
                    tool_call_id = _optional_payload_string(payload, "tool_call_id")
                    checkpoint_call = _pending_user_input_checkpoint_call(
                        checkpoint, input_id=input_id, tool_call_id=tool_call_id
                    )
                    if checkpoint_call is not None:
                        question = _optional_payload_string(payload, "question") or "Input required"
                        return _action_from_record(
                            session=session,
                            record=record,
                            action_kind=PendingActionKind.USER_INPUT,
                            title="User input required",
                            detail=question,
                            tool_name=event.tool_name
                            or _optional_payload_string(checkpoint_call, "tool_name"),
                            input_id=input_id,
                            tool_call_id=tool_call_id,
                            question=question,
                            options=_payload_string_list(payload, "options"),
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
                        checkpoint, input_id=input_id, tool_call_id=tool_call_id
                    )
                elif approval_id is not None:
                    checkpoint_call = _pending_approval_checkpoint_call(
                        checkpoint, approval_id=approval_id, tool_call_id=tool_call_id
                    )
                else:
                    checkpoint_call = _pending_tool_round_checkpoint_call(
                        checkpoint, round_id=round_id or "", tool_call_id=tool_call_id
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
                        checkpoint, approval_id=approval_id
                    )
                    if checkpoint_call is not None:
                        return _action_from_record(
                            session=session,
                            record=record,
                            action_kind=PendingActionKind.TOOL_APPROVAL,
                            title="Tool approval required",
                            detail=_optional_payload_string(approval, "reason"),
                            tool_name=_optional_payload_string(approval, "tool_name")
                            or _optional_payload_string(checkpoint_call, "tool_name")
                            or event.tool_name,
                            approval_id=approval_id,
                            arguments=_object_payload(approval.get("arguments"))
                            or _object_payload(checkpoint_call.get("arguments"))
                            or {},
                        )

            if interruption_type == "user_input_required":
                user_input = _object_payload(payload.get("user_input"))
                input_id = _optional_payload_string(user_input, "input_id")
                if user_input is not None and input_id is not None:
                    tool_call_id = _optional_payload_string(user_input, "tool_call_id")
                    checkpoint_call = _pending_user_input_checkpoint_call(
                        checkpoint, input_id=input_id, tool_call_id=tool_call_id
                    )
                    if checkpoint_call is not None:
                        question = (
                            _optional_payload_string(user_input, "question") or "Input required"
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
                            tool_call_id=tool_call_id,
                            question=question,
                            options=_payload_string_list(user_input, "options"),
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
    if session.status == SessionStatus.COMPLETED:
        return True
    try:
        pending_approval = approval_support.pending_approval_from_checkpoint(checkpoint)
        pending_input = pending_user_input_from_checkpoint(checkpoint)
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
    except (TypeError, ValueError, ValidationError):
        return True
    if pending_round is not None:
        pending_call_ids = {call.tool_call_id for call in pending_round.tool_calls}
        checked_terminal_ids: set[str] = set()
        for record in records_desc:
            event = record.event
            if event.type not in _TERMINAL_EVENT_TYPES:
                continue
            if event.payload.get("tool_round_id") != pending_round.round_id:
                continue
            tool_call_id = event.payload.get("tool_call_id")
            if tool_call_id not in pending_call_ids or tool_call_id in checked_terminal_ids:
                continue
            assert isinstance(tool_call_id, str)
            checked_terminal_ids.add(tool_call_id)
            if event.payload.get(_TERMINAL_RESULT_VALID_KEY) is not True:
                return True
    if action is not None:
        return False
    if pending_approval is not None or pending_input is not None:
        return True
    return pending_round is None
