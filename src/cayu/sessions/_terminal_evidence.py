"""Shared classification for bounded terminal-event evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

import cayu.sessions as sessions
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_json_object,
    require_clean_nonblank,
    require_durable_clean_nonblank,
)
from cayu.events import (
    EVENT_ID_MAX_CHARS,
    Event,
    EventType,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.sessions.authority import SessionRunFenced

_INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED = "tool_approval_required"
_INTERRUPTION_TYPE_USER_INPUT_REQUIRED = "user_input_required"
_INTERRUPTION_TYPE_RUNTIME_INTERRUPTED = "runtime_interrupted"
_INTERRUPTION_TYPE_OPERATOR_REQUESTED = "operator_requested"

TERMINAL_EVIDENCE_QUERY_LIMIT = 2
SESSION_RUN_OPERATION_ID_PAYLOAD_KEY = "session_run_operation_id"
TERMINAL_EVENT_TYPES = (
    EventType.SESSION_COMPLETED,
    EventType.SESSION_FAILED,
    EventType.SESSION_INTERRUPTED,
)
TERMINAL_LIFECYCLE_EVENT_TYPES = (
    EventType.SESSION_STARTED,
    EventType.SESSION_RESUMED,
    EventType.SESSION_FORKED,
)
TERMINAL_EVIDENCE_EVENT_TYPES = (
    *TERMINAL_LIFECYCLE_EVENT_TYPES,
    *TERMINAL_EVENT_TYPES,
)


def queued_dispatch_terminal_event_id(operation_id: str) -> str:
    if len(operation_id) != 64 or any(
        character not in "0123456789abcdef" for character in operation_id
    ):
        raise ValueError("Queued dispatch operation_id must be a lowercase SHA-256 digest.")
    return f"cayu-queued-dispatch-terminal-{operation_id}"


@dataclass(frozen=True, slots=True)
class CurrentTerminalEvidence:
    """The bounded events belonging to the latest durable terminal boundary."""

    events: tuple[Event, ...]
    latest_lifecycle_event_type: EventType | str | None
    run_operation_conflict: bool = False


def interruption_request_id_from_payload(payload: dict[str, Any]) -> str | None:
    request_id = payload.get("interruption_request_id")
    if request_id is None:
        return None
    if type(request_id) is not str or not request_id.strip():
        raise ValueError("Interruption request ID must be a non-blank string.")
    return request_id


def require_interruption_event_matches_pending_marker(
    event: Event,
    marker: dict[str, Any],
) -> None:
    """Require every marker-owned field to survive terminal publication exactly.

    Terminal publication may add runtime-owned fields, so the event payload can
    be a strict superset. The pending marker is the durable authority for every
    field it does own, however, and acknowledgement readback must not clear it
    after accepting a conflicting sibling value.
    """

    if event.type is not EventType.SESSION_INTERRUPTED:
        raise RuntimeError("Published interruption evidence does not match the pending marker.")
    expected = copy_durable_json_object(marker, "pending session interrupt marker")
    actual = copy_durable_json_object(event.payload, "published interruption payload")
    if any(key not in actual or actual[key] != value for key, value in expected.items()):
        raise RuntimeError("Published interruption evidence conflicts with the pending marker.")


def classify_current_terminal_evidence(
    *,
    evidence_events: Sequence[Event],
    expected_event_type: EventType,
    run_operation_id: str | None,
    interruption_request_id: str | None,
) -> CurrentTerminalEvidence:
    """Classify newest-first evidence without conflating adjacent interruptions."""

    if len(evidence_events) > TERMINAL_EVIDENCE_QUERY_LIMIT:
        raise ValueError("Terminal evidence exceeds its bounded query.")

    latest_lifecycle_event_type: EventType | str | None = None
    events_after_lifecycle: list[Event] = []
    for event in evidence_events:
        if event.type in TERMINAL_LIFECYCLE_EVENT_TYPES:
            latest_lifecycle_event_type = event.type
            break
        events_after_lifecycle.append(event)

    candidate_events: list[Event] = []
    for event in events_after_lifecycle:
        if event.type != expected_event_type:
            break
        if event.interaction_id is not None:
            continue
        candidate_events.append(event)

    terminal_events: list[Event] = []
    if interruption_request_id is not None:
        for event in candidate_events:
            if interruption_request_id_from_payload(event.payload) != interruption_request_id:
                break
            terminal_events.append(event)
        return CurrentTerminalEvidence(
            events=tuple(terminal_events),
            latest_lifecycle_event_type=latest_lifecycle_event_type,
            run_operation_conflict=(
                run_operation_id is not None
                and any(
                    event.payload.get(SESSION_RUN_OPERATION_ID_PAYLOAD_KEY) != run_operation_id
                    for event in terminal_events
                )
            ),
        )

    if run_operation_id is not None:
        return CurrentTerminalEvidence(
            events=tuple(
                event
                for event in events_after_lifecycle
                if event.type in TERMINAL_EVENT_TYPES
                and event.interaction_id is None
                and event.payload.get(SESSION_RUN_OPERATION_ID_PAYLOAD_KEY) == run_operation_id
            ),
            latest_lifecycle_event_type=latest_lifecycle_event_type,
        )

    current_operation_id: str | None = None
    current_interrupt_request_id: str | None = None
    for event in candidate_events:
        event_operation_id = event.payload.get(SESSION_RUN_OPERATION_ID_PAYLOAD_KEY)
        event_interrupt_request_id = interruption_request_id_from_payload(event.payload)
        if not terminal_events:
            if event_operation_id is not None:
                current_operation_id = require_clean_nonblank(
                    event_operation_id,
                    "terminal event session_run_operation_id",
                )
            current_interrupt_request_id = event_interrupt_request_id
            terminal_events.append(event)
            continue
        if current_operation_id is not None:
            if event_operation_id == current_operation_id:
                terminal_events.append(event)
        elif current_interrupt_request_id is not None:
            if event_interrupt_request_id == current_interrupt_request_id:
                terminal_events.append(event)
        elif event_operation_id is None and event_interrupt_request_id is None:
            terminal_events.append(event)

    return CurrentTerminalEvidence(
        events=tuple(terminal_events),
        latest_lifecycle_event_type=latest_lifecycle_event_type,
    )


_SESSION_RUN_OPERATION_CHECKPOINT_KEY = "session_run_operation"


@dataclass(frozen=True)
class _SessionRunOperation:
    operation_id: str
    run_epoch: int
    terminal_event_id: str | None = None
    queue_task_id: str | None = None


def _session_run_operation_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> _SessionRunOperation | None:
    """Parse the durable identity of the latest resumed/continued session run."""
    if checkpoint is None or _SESSION_RUN_OPERATION_CHECKPOINT_KEY not in checkpoint:
        return None
    marker = checkpoint[_SESSION_RUN_OPERATION_CHECKPOINT_KEY]
    if type(marker) is not dict or marker.get("version") != 1:
        raise ValueError("Session run operation checkpoint is invalid.")
    operation_id = require_durable_clean_nonblank(
        marker.get("operation_id"),
        "session_run_operation.operation_id",
    )
    run_epoch = marker.get("run_epoch")
    if type(run_epoch) is not int or run_epoch < 1 or run_epoch > MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Session run operation checkpoint run_epoch is invalid.")
    terminal_event_id = marker.get("terminal_event_id")
    if terminal_event_id is not None:
        terminal_event_id = require_durable_clean_nonblank(
            terminal_event_id,
            "session_run_operation.terminal_event_id",
        )
        if len(terminal_event_id) > EVENT_ID_MAX_CHARS:
            raise ValueError("Session run operation terminal_event_id is too long.")
    queue_task_id = marker.get("queue_task_id")
    if queue_task_id is not None:
        queue_task_id = require_durable_clean_nonblank(
            queue_task_id,
            "session_run_operation.queue_task_id",
        )
    if queue_task_id is not None and terminal_event_id is None:
        raise ValueError("Session run operation queue_task_id requires a terminal_event_id.")
    return _SessionRunOperation(
        operation_id=operation_id,
        run_epoch=run_epoch,
        terminal_event_id=terminal_event_id,
        queue_task_id=queue_task_id,
    )


class QueuedDispatchTerminalReceipt(BaseModel):
    """Exact session-side handoff awaiting queue-task acknowledgement."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: str
    queue_task_id: str
    operation_id: str
    terminal_event_id: str

    @field_validator(
        "session_id",
        "queue_task_id",
        "operation_id",
        "terminal_event_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        value = require_durable_clean_nonblank(value, info.field_name)
        if info.field_name == "terminal_event_id" and len(value) > EVENT_ID_MAX_CHARS:
            raise ValueError("terminal_event_id is too long.")
        return value


class QueuedDispatchTerminalReceiptQuery(BaseModel):
    """Bounded keyset query for live queued-dispatch handoffs."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    after_session_id: str | None = None
    after_operation_id: str | None = None
    limit: StrictInt = Field(default=100, ge=1, le=1000)

    @field_validator("after_session_id", "after_operation_id")
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def require_complete_cursor(self) -> QueuedDispatchTerminalReceiptQuery:
        if (self.after_session_id is None) != (self.after_operation_id is None):
            raise ValueError("Queued dispatch receipt cursor fields must be supplied together.")
        return self


_SESSION_RUN_OPERATION_ID_PAYLOAD_KEY = SESSION_RUN_OPERATION_ID_PAYLOAD_KEY


_QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY = "queued_dispatch_terminal_receipts"


@dataclass(frozen=True)
class _QueuedDispatchTerminalReceiptIdentity:
    queue_task_id: str
    terminal_event_id: str
    run_epoch: int


def _queued_dispatch_terminal_receipts_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> dict[str, _QueuedDispatchTerminalReceiptIdentity]:
    """Parse terminal evidence retained until its queue task is acknowledged."""

    if checkpoint is None or _QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY not in checkpoint:
        return {}
    marker = checkpoint[_QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY]
    if type(marker) is not dict or set(marker) != {"version", "receipts"}:
        raise ValueError("Queued dispatch terminal receipt checkpoint is invalid.")
    if marker.get("version") != 1 or type(marker.get("receipts")) is not dict:
        raise ValueError("Queued dispatch terminal receipt checkpoint is invalid.")
    receipts: dict[str, _QueuedDispatchTerminalReceiptIdentity] = {}
    for operation_id, receipt in marker["receipts"].items():
        operation_id = require_durable_clean_nonblank(
            operation_id,
            "queued_dispatch_terminal_receipt.operation_id",
        )
        if type(receipt) is not dict or set(receipt) != {
            "queue_task_id",
            "terminal_event_id",
            "run_epoch",
        }:
            raise ValueError("Queued dispatch terminal receipt is invalid.")
        queue_task_id = require_durable_clean_nonblank(
            receipt.get("queue_task_id"),
            "queued_dispatch_terminal_receipt.queue_task_id",
        )
        terminal_event_id = require_durable_clean_nonblank(
            receipt.get("terminal_event_id"),
            "queued_dispatch_terminal_receipt.terminal_event_id",
        )
        if len(terminal_event_id) > EVENT_ID_MAX_CHARS:
            raise ValueError("Queued dispatch terminal receipt event identity is too long.")
        run_epoch = receipt.get("run_epoch")
        if type(run_epoch) is not int or run_epoch < 1 or run_epoch > MAX_DURABLE_JSON_INTEGER:
            raise ValueError("Queued dispatch terminal receipt run_epoch is invalid.")
        receipts[operation_id] = _QueuedDispatchTerminalReceiptIdentity(
            queue_task_id=queue_task_id,
            terminal_event_id=terminal_event_id,
            run_epoch=run_epoch,
        )
    return receipts


def _queued_dispatch_terminal_receipts_for_session(
    session_id: str,
    checkpoint: dict[str, Any] | None,
    *,
    durable_event_ids: set[str],
) -> list[QueuedDispatchTerminalReceipt]:
    """Project all exact live queue/session handoffs from one checkpoint."""

    session_id = require_durable_clean_nonblank(session_id, "session_id")
    projected: dict[str, QueuedDispatchTerminalReceipt] = {}
    run_operation = _session_run_operation_from_checkpoint(checkpoint)
    if (
        run_operation is not None
        and run_operation.queue_task_id is not None
        and run_operation.terminal_event_id is not None
        and run_operation.terminal_event_id in durable_event_ids
    ):
        projected[run_operation.operation_id] = QueuedDispatchTerminalReceipt(
            session_id=session_id,
            queue_task_id=run_operation.queue_task_id,
            operation_id=run_operation.operation_id,
            terminal_event_id=run_operation.terminal_event_id,
        )
    for operation_id, receipt in _queued_dispatch_terminal_receipts_from_checkpoint(
        checkpoint
    ).items():
        candidate = QueuedDispatchTerminalReceipt(
            session_id=session_id,
            queue_task_id=receipt.queue_task_id,
            operation_id=operation_id,
            terminal_event_id=receipt.terminal_event_id,
        )
        existing = projected.get(operation_id)
        if existing is not None and existing != candidate:
            raise RuntimeError("Queued dispatch terminal handoff identity conflicts.")
        projected[operation_id] = candidate
    return list(projected.values())


def _checkpoint_after_session_run_operation_cleanup(
    checkpoint: dict[str, Any] | None,
    *,
    operation: _SessionRunOperation,
    retain_terminal_receipt: bool,
) -> dict[str, Any]:
    """Clear run ownership and optionally transfer terminal retention to the queue."""

    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    updated.pop(_SESSION_RUN_OPERATION_CHECKPOINT_KEY, None)
    if not retain_terminal_receipt or operation.terminal_event_id is None:
        return updated
    receipts = _queued_dispatch_terminal_receipts_from_checkpoint(checkpoint)
    if operation.queue_task_id is None:
        return updated
    receipt = _QueuedDispatchTerminalReceiptIdentity(
        queue_task_id=operation.queue_task_id,
        terminal_event_id=operation.terminal_event_id,
        run_epoch=operation.run_epoch,
    )
    existing = receipts.get(operation.operation_id)
    if existing not in {None, receipt}:
        raise RuntimeError("Queued dispatch terminal receipt identity conflicts.")
    receipts[operation.operation_id] = receipt
    updated[_QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY] = {
        "version": 1,
        "receipts": {
            operation_id: {
                "queue_task_id": stored_receipt.queue_task_id,
                "terminal_event_id": stored_receipt.terminal_event_id,
                "run_epoch": stored_receipt.run_epoch,
            }
            for operation_id, stored_receipt in receipts.items()
        },
    }
    return updated


def _checkpoint_after_queued_dispatch_acknowledgement(
    checkpoint: dict[str, Any] | None,
    *,
    queue_task_id: str,
    operation_id: str,
    terminal_event_id: str,
) -> dict[str, Any] | None:
    """Release exact queue-owned terminal retention after task terminalization."""

    queue_task_id = require_durable_clean_nonblank(queue_task_id, "dispatch queue_task_id")
    operation_id = require_durable_clean_nonblank(operation_id, "dispatch operation_id")
    terminal_event_id = require_durable_clean_nonblank(
        terminal_event_id,
        "dispatch terminal_event_id",
    )
    current_operation = _session_run_operation_from_checkpoint(checkpoint)
    receipts = _queued_dispatch_terminal_receipts_from_checkpoint(checkpoint)
    receipt = receipts.get(operation_id)
    if receipt is not None and (
        receipt.queue_task_id != queue_task_id or receipt.terminal_event_id != terminal_event_id
    ):
        raise RuntimeError("Queued dispatch terminal receipt identity conflicts.")
    if current_operation is not None and current_operation.operation_id == operation_id:
        if (
            current_operation.queue_task_id != queue_task_id
            or current_operation.terminal_event_id != terminal_event_id
        ):
            raise RuntimeError("Queued dispatch run operation identity conflicts.")
    elif receipt is None:
        return None

    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    if current_operation is not None and current_operation.operation_id == operation_id:
        updated.pop(_SESSION_RUN_OPERATION_CHECKPOINT_KEY, None)
    if receipt is not None:
        receipts.pop(operation_id)
        if receipts:
            updated[_QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY] = {
                "version": 1,
                "receipts": {
                    stored_operation_id: {
                        "queue_task_id": stored_receipt.queue_task_id,
                        "terminal_event_id": stored_receipt.terminal_event_id,
                        "run_epoch": stored_receipt.run_epoch,
                    }
                    for stored_operation_id, stored_receipt in receipts.items()
                },
            }
        else:
            updated.pop(_QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY, None)
    # ``transform_checkpoint`` reserves ``None`` for a read-only callback, so an
    # empty object is the required mutation that clears the final receipt.
    return updated


def _checkpoint_with_session_run_operation(
    *,
    checkpoint: dict[str, Any] | None,
    current_session: sessions.Session,
    operation_id: str,
    terminal_event_id: str | None = None,
    queue_task_id: str | None = None,
) -> dict[str, Any]:
    """Record the next running epoch in the same transaction as its status claim."""
    operation_id = require_durable_clean_nonblank(operation_id, "session run operation_id")
    if terminal_event_id is not None:
        terminal_event_id = require_durable_clean_nonblank(
            terminal_event_id,
            "session run terminal_event_id",
        )
        if len(terminal_event_id) > EVENT_ID_MAX_CHARS:
            raise ValueError("Session run terminal_event_id is too long.")
    if queue_task_id is not None:
        queue_task_id = require_durable_clean_nonblank(
            queue_task_id,
            "session run queue_task_id",
        )
        if terminal_event_id is None:
            raise ValueError("Session run queue_task_id requires a terminal_event_id.")
    existing_operation = _session_run_operation_from_checkpoint(checkpoint)
    if existing_operation is not None:
        raise RuntimeError(
            "Session has incomplete terminal evidence for its previous run. "
            "Recover the session before starting another run."
        )
    if queue_task_id is not None:
        receipt = _queued_dispatch_terminal_receipts_from_checkpoint(checkpoint).get(operation_id)
        if receipt is not None:
            if (
                receipt.queue_task_id != queue_task_id
                or receipt.terminal_event_id != terminal_event_id
            ):
                raise RuntimeError("Queued dispatch terminal receipt identity conflicts.")
            raise SessionRunFenced(
                "Queued dispatch terminal evidence is awaiting queue acknowledgement."
            )
    next_run_epoch = current_session.run_epoch + 1
    if next_run_epoch > MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Session run operation run_epoch exceeds the durable integer limit.")
    updated = {} if checkpoint is None else copy_durable_json_object(checkpoint, "checkpoint")
    marker: dict[str, Any] = {
        "version": 1,
        "operation_id": operation_id,
        "run_epoch": next_run_epoch,
    }
    if terminal_event_id is not None:
        marker["terminal_event_id"] = terminal_event_id
    if queue_task_id is not None:
        marker["queue_task_id"] = queue_task_id
    updated[_SESSION_RUN_OPERATION_CHECKPOINT_KEY] = marker
    return updated


def _event_with_session_run_operation(
    event: Event,
    operation: _SessionRunOperation,
) -> Event:
    """Bind terminal evidence to the durable run operation that produced it."""
    payload = copy_durable_json_object(event.payload, "event payload")
    existing_operation_id = payload.get(_SESSION_RUN_OPERATION_ID_PAYLOAD_KEY)
    if existing_operation_id not in {None, operation.operation_id}:
        raise ValueError("Terminal event carries a conflicting session run operation identity.")
    payload[_SESSION_RUN_OPERATION_ID_PAYLOAD_KEY] = operation.operation_id
    updates: dict[str, Any] = {"payload": payload}
    if operation.terminal_event_id is not None:
        updates["id"] = operation.terminal_event_id
    bound = event.model_copy(update=updates, deep=True)
    if operation.terminal_event_id is not None:
        bound = event_with_runtime_generated_id(bound)
    return event_with_runtime_payload_authority(
        bound,
        _SESSION_RUN_OPERATION_ID_PAYLOAD_KEY,
    )
