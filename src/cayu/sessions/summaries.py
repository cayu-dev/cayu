"""Stored-session status and event summaries derived from durable records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_json_object,
    copy_durable_json_value,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.budgets.aggregates import AggregateAccuracy, AggregateCount
from cayu.events import Event, EventType
from cayu.sessions.queries import SessionStatusCounts
from cayu.sessions.records import EventRecord, Session, SessionStatus, copy_session


class SessionOperationalSnapshot(BaseModel):
    """Exact current session counts captured by one store-local read snapshot."""

    model_config = ConfigDict(extra="forbid")

    as_of: datetime
    total_count: AggregateCount = Field(ge=0)
    counts_by_status: SessionStatusCounts
    accuracy: AggregateAccuracy

    @field_validator("counts_by_status")
    @classmethod
    def copy_counts_by_status(cls, value: SessionStatusCounts) -> SessionStatusCounts:
        return SessionStatusCounts.model_validate(value.model_dump(mode="python", warnings=False))

    @field_validator("accuracy")
    @classmethod
    def copy_accuracy(cls, value: AggregateAccuracy) -> AggregateAccuracy:
        return AggregateAccuracy.model_validate(value.model_dump(mode="python", warnings=False))

    @field_validator("as_of")
    @classmethod
    def normalize_as_of(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_total(self) -> SessionOperationalSnapshot:
        if sum(self.counts_by_status.model_dump().values()) != self.total_count:
            raise ValueError("Session status counts must sum to total_count.")
        return self


class EventSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    total_events: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    counts_by_type: dict[str, StrictInt] = Field(default_factory=dict)
    latest_event: EventRecord | None = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "session_id")


class SessionOutcome(BaseModel):
    """Derived reason for the current session state.

    The outcome is computed from durable events. It is intentionally not stored
    as separate state so event replay remains the source of truth.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: SessionStatus
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)
    retry: dict[str, Any] | None = None
    terminal_event: EventRecord | None = None
    latest_retry_event: EventRecord | None = None

    @field_validator("session_id", "reason")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("details", mode="before")
    @classmethod
    def copy_details(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "details")

    @field_validator("retry", mode="before")
    @classmethod
    def copy_retry(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return copy_durable_json_object(value, "retry")


def event_summary_from_records(
    session_id: str,
    records: list[EventRecord],
) -> EventSummary:
    session_id = require_clean_nonblank(session_id, "session_id")
    counts_by_type: dict[str, int] = {}
    latest_record: EventRecord | None = None
    for record in records:
        if record.event.session_id != session_id:
            continue
        event_type = str(record.event.type)
        counts_by_type[event_type] = counts_by_type.get(event_type, 0) + 1
        if latest_record is None or record.sequence > latest_record.sequence:
            latest_record = record
    return EventSummary(
        session_id=session_id,
        total_events=sum(counts_by_type.values()),
        counts_by_type=counts_by_type,
        latest_event=_copy_event_record(latest_record),
    )


def session_outcome_from_records(
    session: Session,
    records: list[EventRecord],
) -> SessionOutcome:
    session = copy_session(session)

    latest_lifecycle_sequence = 0
    for record in reversed(records):
        if record.event.session_id != session.id:
            continue
        if _is_outcome_lifecycle_event(record.event):
            latest_lifecycle_sequence = record.sequence
            break

    terminal_record: EventRecord | None = None
    for record in reversed(records):
        if record.event.session_id != session.id:
            continue
        if record.sequence <= latest_lifecycle_sequence:
            break
        if _is_outcome_terminal_event(record.event):
            terminal_record = record
            break

    retry_record: EventRecord | None = None
    for record in reversed(records):
        if record.event.session_id != session.id:
            continue
        if record.sequence <= latest_lifecycle_sequence:
            break
        if record.event.type == EventType.MODEL_RETRY:
            retry_record = record
            break

    return session_outcome(
        session,
        terminal_event=terminal_record,
        latest_retry_event=retry_record,
    )


def session_outcome(
    session: Session,
    *,
    terminal_event: EventRecord | None,
    latest_retry_event: EventRecord | None,
) -> SessionOutcome:
    session = copy_session(session)
    terminal_event = _copy_event_record(terminal_event)
    latest_retry_event = _copy_event_record(latest_retry_event)
    if not _terminal_event_matches_status(session, terminal_event):
        terminal_event = None
    reason, details = _outcome_reason_and_details(session, terminal_event)
    return SessionOutcome(
        session_id=session.id,
        status=session.status,
        reason=reason,
        details=details,
        retry=_retry_details(latest_retry_event),
        terminal_event=terminal_event,
        latest_retry_event=latest_retry_event,
    )


def _is_outcome_terminal_event(event: Event) -> bool:
    return event.type in {
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
    }


def _is_outcome_lifecycle_event(event: Event) -> bool:
    return event.type in {
        EventType.SESSION_STARTED,
        EventType.SESSION_RESUMED,
    }


def _outcome_reason_and_details(
    session: Session,
    terminal_event: EventRecord | None,
) -> tuple[str, dict[str, Any]]:
    if session.status not in _OUTCOME_TERMINAL_STATUSES:
        return session.status.value, {}
    if terminal_event is None:
        return session.status.value, {}

    event = terminal_event.event
    if event.type != _OUTCOME_EVENT_TYPE_BY_STATUS[session.status]:
        return session.status.value, {}

    payload = event.payload
    if event.type == EventType.SESSION_COMPLETED:
        if payload.get("reason") == "host_rendered_tool":
            return "host_rendered_tool", _copy_payload_fields(payload, ("tool_completion",))
        return "completed", {}
    if event.type == EventType.SESSION_FAILED:
        return "failed", _copy_payload_fields(payload, ("error", "error_type"))
    if event.type == EventType.SESSION_INTERRUPTED:
        reason = _optional_payload_string(payload, "interruption_type") or "interrupted"
        details = _copy_payload_fields(
            payload,
            (
                "interruption_type",
                "reason",
                "limit",
                "maximum",
                "actual",
                "message",
                "error",
                "error_type",
                "manual_recovery_required",
                "tool_call_id",
                "tool_name",
            ),
        )
        return reason, details
    return session.status.value, {}


def _terminal_event_matches_status(
    session: Session,
    terminal_event: EventRecord | None,
) -> bool:
    if terminal_event is None:
        return False
    expected_event_type = _OUTCOME_EVENT_TYPE_BY_STATUS.get(session.status)
    if expected_event_type is None:
        return False
    return terminal_event.event.type == expected_event_type


def _retry_details(latest_retry_event: EventRecord | None) -> dict[str, Any] | None:
    if latest_retry_event is None:
        return None
    return _copy_payload_fields(
        latest_retry_event.event.payload,
        (
            "provider",
            "model",
            "step",
            "attempt",
            "next_attempt",
            "max_attempts",
            "delay_seconds",
            "reason",
            "status_code",
        ),
    )


def _copy_event_record(record: EventRecord | None) -> EventRecord | None:
    if record is None:
        return None
    return EventRecord(sequence=record.sequence, event=record.event)


def _copy_payload_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for field in fields:
        if field in payload and payload[field] is not None:
            copied[field] = copy_durable_json_value(payload[field], field)
    return copied


def _optional_payload_string(payload: dict[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if type(value) is str and value.strip():
        return value
    return None


_OUTCOME_TERMINAL_STATUSES = {
    SessionStatus.COMPLETED,
    SessionStatus.FAILED,
    SessionStatus.INTERRUPTED,
}


_OUTCOME_EVENT_TYPE_BY_STATUS = {
    SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
    SessionStatus.FAILED: EventType.SESSION_FAILED,
    SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}
