"""Durable event claims, delivery records, error bounds and health projections.

Stores own claim mutations; these contracts and projections can be used
independently of store implementations and delivery workers.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import MAX_DURABLE_JSON_INTEGER, copy_durable_json_object
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu._validation import require_durable_nonblank as require_nonblank
from cayu.artifacts.attachments import MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY
from cayu.events import (
    Event,
    EventType,
    copy_event,
    event_payload_authority_is_runtime_generated,
    event_with_runtime_payload_authority,
    validate_event_envelope,
)
from cayu.sessions.transcript_input import SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY

PERSISTED_EVENT_SIDE_EFFECT_MAX_ATTEMPTS = 3
Status = Literal["pending", "leased", "failed", "delivered", "dead_lettered"]


class PersistedEventSideEffectHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observed_at: datetime
    pending: int = 0
    leased_live: int = 0
    leased_expired: int = 0
    failed_retryable: int = 0
    failed_deferred: int = 0
    dead_lettered: int = 0
    delivered: int = 0
    claimable_total: int = 0
    outstanding_total: int = 0
    repeatedly_failing: int = 0
    final_attempt_boundary: int = 0
    max_outstanding_attempts: int = 0
    max_automatic_attempts: int = PERSISTED_EVENT_SIDE_EFFECT_MAX_ATTEMPTS
    oldest_claimable_at: datetime | None = None
    oldest_claimable_age_seconds: float | None = None
    oldest_pending_at: datetime | None = None
    oldest_pending_age_seconds: float | None = None
    oldest_failed_at: datetime | None = None
    oldest_failed_age_seconds: float | None = None
    oldest_dead_letter_at: datetime | None = None
    oldest_dead_letter_age_seconds: float | None = None
    earliest_live_lease_expires_at: datetime | None = None


class PersistedEventSideEffectQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    statuses: set[Status] | None = None
    claimable_only: bool = False
    outstanding_only: bool = True
    limit: StrictInt = Field(default=100, ge=1, le=1000)
    cursor: str | None = Field(default=None, max_length=16384)


class PersistedEventSideEffectInspection(BaseModel):
    session_id: str
    event_id: str
    event_sequence: int
    status: Status
    claimable: bool
    attempts: int
    lease_expires_at: datetime | None
    next_attempt_at: datetime | None
    updated_at: datetime
    last_error: str | None


class PersistedEventSideEffectPage(BaseModel):
    observed_at: datetime
    deliveries: list[PersistedEventSideEffectInspection]
    next_cursor: str | None = None


def safe_error(value: object) -> str | None:
    # Exception text is arbitrary application data. Without the application's
    # sink-boundary redactor it cannot safely be distinguished from a secret.
    return (
        None
        if value is None
        else "Side effect failed; inspect access-controlled application diagnostics."
    )


def cursor_key(query: PersistedEventSideEffectQuery) -> tuple[str, str] | None:
    if query.cursor is None:
        return None
    try:
        data = json.loads(base64.b64decode(query.cursor, altchars=b"-_", validate=True))
        if (
            not isinstance(data, list)
            or len(data) != 4
            or data[0] != 1
            or data[1] != filter_key(query)
            or not all(type(x) is str for x in data[2:])
        ):
            raise ValueError
        return data[2], data[3]
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError("Invalid event side-effect cursor or mismatched filters.") from exc


def filter_key(query: PersistedEventSideEffectQuery) -> list[Any]:
    return [
        None if query.statuses is None else sorted(query.statuses),
        query.claimable_only,
        query.outstanding_only,
    ]


def claimable(row: Any, now: datetime) -> bool:
    return (
        row.status == "pending"
        or (row.status == "failed" and (row.next_attempt_at is None or row.next_attempt_at <= now))
        or (
            row.status == "leased"
            and row.lease_expires_at is not None
            and row.lease_expires_at <= now
        )
    )


def page(
    rows: list[Any], query: PersistedEventSideEffectQuery, now: datetime
) -> PersistedEventSideEffectPage:
    next_cursor = None
    if len(rows) > query.limit:
        last = rows[query.limit - 1]
        next_cursor = base64.urlsafe_b64encode(
            json.dumps(
                [1, filter_key(query), last.session_id, last.event_id],
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).decode()
    return PersistedEventSideEffectPage(
        observed_at=now,
        next_cursor=next_cursor,
        deliveries=[
            PersistedEventSideEffectInspection(
                **row.model_dump(
                    include={
                        "session_id",
                        "event_id",
                        "event_sequence",
                        "status",
                        "attempts",
                        "lease_expires_at",
                        "next_attempt_at",
                        "updated_at",
                    }
                ),
                claimable=claimable(row, now),
                last_error=safe_error(row.last_error),
            )
            for row in rows[: query.limit]
        ],
    )


def finish_health(values: dict[str, Any], now: datetime) -> PersistedEventSideEffectHealth:
    for key, value in list(values.items()):
        if value is None and not key.endswith("_at"):
            values[key] = 0
    for name in ("claimable", "pending", "failed", "dead_letter"):
        key = f"oldest_{name}_at"
        value = values.get(key)
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if value is not None:
            value = value.astimezone(UTC)
            values[key] = value
        values[f"oldest_{name}_age_seconds"] = (
            None if value is None else max(0.0, (now - value).total_seconds())
        )
    return PersistedEventSideEffectHealth(observed_at=now, **values)


# SQL and in-memory implementations share the same public classification.
# A one-row CTE binds one instant for the whole statement.
CLAIMABLE_SQL = "(status = 'pending' OR (status = 'failed' AND (next_attempt_at IS NULL OR next_attempt_at <= observed_at)) OR (status = 'leased' AND lease_expires_at <= observed_at))"
CONDITIONS = {
    "pending": "status = 'pending'",
    "leased_live": "status = 'leased' AND lease_expires_at > observed_at",
    "leased_expired": "status = 'leased' AND lease_expires_at <= observed_at",
    "failed_retryable": "status = 'failed'",
    "failed_deferred": "status = 'failed' AND next_attempt_at > observed_at",
    "dead_lettered": "status = 'dead_lettered'",
    "delivered": "status = 'delivered'",
    "claimable_total": CLAIMABLE_SQL,
    "outstanding_total": "status <> 'delivered'",
    "repeatedly_failing": "status = 'failed' AND attempts > 1",
    "final_attempt_boundary": (
        f"(status IN ('pending', 'failed') OR (status = 'leased' AND lease_expires_at <= observed_at)) AND attempts >= {PERSISTED_EVENT_SIDE_EFFECT_MAX_ATTEMPTS - 1}"
        f" OR (status = 'leased' AND lease_expires_at > observed_at AND attempts >= {PERSISTED_EVENT_SIDE_EFFECT_MAX_ATTEMPTS})"
    ),
}
AGGREGATES = {
    name: f"SUM(CASE WHEN {condition} THEN 1 ELSE 0 END)" for name, condition in CONDITIONS.items()
}
AGGREGATES.update(
    {
        "max_outstanding_attempts": "MAX(CASE WHEN status <> 'delivered' THEN attempts ELSE 0 END)",
        "oldest_claimable_at": f"MIN(CASE WHEN {CLAIMABLE_SQL} THEN CASE WHEN status = 'leased' THEN lease_expires_at WHEN status = 'failed' AND next_attempt_at > updated_at THEN next_attempt_at ELSE updated_at END END)",
        "oldest_pending_at": "MIN(CASE WHEN status = 'pending' THEN updated_at END)",
        "oldest_failed_at": "MIN(CASE WHEN status = 'failed' THEN updated_at END)",
        "oldest_dead_letter_at": "MIN(CASE WHEN status = 'dead_lettered' THEN updated_at END)",
        "earliest_live_lease_expires_at": "MIN(CASE WHEN status = 'leased' AND lease_expires_at > observed_at THEN lease_expires_at END)",
    }
)


def health_sql(placeholder: str) -> str:
    return (
        f"WITH observation AS (SELECT {placeholder} AS observed_at) SELECT "
        + ", ".join(f"{expression} AS {name}" for name, expression in AGGREGATES.items())
        + " FROM cayu_persisted_event_side_effects CROSS JOIN observation"
    )


def page_sql(
    query: PersistedEventSideEffectQuery, now: Any, placeholder: str
) -> tuple[str, list[Any]]:
    key = cursor_key(query)
    # Python and SQLite use code-point/binary order. PostgreSQL must not inherit
    # a deployment locale for the same opaque keyset cursor contract.
    ordered_keys = (
        'session_id COLLATE "C", event_id COLLATE "C"'
        if placeholder == "%s"
        else "session_id, event_id"
    )
    params: list[Any] = [now]
    clauses = []
    if query.outstanding_only:
        clauses.append("status <> 'delivered'")
    if query.claimable_only:
        clauses.append(CLAIMABLE_SQL)
    if query.statuses is not None:
        clauses.append(
            "status IN (" + ", ".join(placeholder for _ in query.statuses) + ")"
            if query.statuses
            else "1 = 0"
        )
        params.extend(sorted(query.statuses))
    if key is not None:
        clauses.append(f"({ordered_keys}) > ({placeholder}, {placeholder})")
        params.extend(key)
    params.append(query.limit + 1)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return (
        f"WITH observation AS (SELECT {placeholder} AS observed_at) SELECT "
        "session_id, event_id, event_sequence, status, attempts, claim_id, "
        "lease_expires_at, next_attempt_at, last_error, updated_at "
        "FROM cayu_persisted_event_side_effects CROSS JOIN observation"
        + where
        + f" ORDER BY {ordered_keys} LIMIT {placeholder}",
        params,
    )


class PersistedEventSideEffectClaimLost(RuntimeError):
    """A side-effect acknowledgement lost ownership to a replacement claim."""


PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES = 4096


_NONPORTABLE_PERSISTED_EVENT_SIDE_EFFECT_ERROR = (
    "Persisted event side effect failed with non-portable error details."
)


_TRUNCATED_PERSISTED_EVENT_SIDE_EFFECT_ERROR_SUFFIX = "... [truncated]"


def validate_persisted_event_side_effect_error(
    value: str,
    field_name: str = "error",
) -> str:
    """Validate a portable, bounded side-effect failure description."""

    value = require_nonblank(value, field_name)
    if len(value.encode("utf-8")) > PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES:
        raise ValueError(
            f"`{field_name}` must not exceed "
            f"{PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES} UTF-8 bytes."
        )
    return value


def portable_persisted_event_side_effect_error(value: object) -> str:
    """Project untrusted exception text into the durable handoff contract."""

    if type(value) is not str:
        return _NONPORTABLE_PERSISTED_EVENT_SIDE_EFFECT_ERROR
    try:
        value = require_nonblank(value, "error")
    except ValueError:
        return _NONPORTABLE_PERSISTED_EVENT_SIDE_EFFECT_ERROR

    encoded = value.encode("utf-8")
    if len(encoded) <= PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES:
        return value

    suffix = _TRUNCATED_PERSISTED_EVENT_SIDE_EFFECT_ERROR_SUFFIX
    prefix = encoded[: PERSISTED_EVENT_SIDE_EFFECT_ERROR_MAX_BYTES - len(suffix.encode("utf-8"))]
    while True:
        try:
            return prefix.decode("utf-8") + suffix
        except UnicodeDecodeError:
            prefix = prefix[:-1]


class PersistedEventSideEffectStatus(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"


class PersistedEventSideEffectClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    event_id: str
    event_sequence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    event: Event
    attempt: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    claim_id: str = Field(default_factory=lambda: str(uuid4()))
    lease_expires_at: datetime

    @field_validator("session_id", "event_id", "claim_id")
    @classmethod
    def validate_clean_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("event")
    @classmethod
    def copy_claim_event(cls, value: Event) -> Event:
        return copy_event(value)

    @field_validator("lease_expires_at")
    @classmethod
    def normalize_lease_expires_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("lease_expires_at must be timezone-aware.")
        return value.astimezone(UTC)


class PersistedEventSideEffectDelivery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str
    event_id: str
    event_sequence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    status: PersistedEventSideEffectStatus
    attempts: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    claim_id: str | None = None
    lease_expires_at: datetime | None = None
    next_attempt_at: datetime | None = None
    last_error: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("session_id", "event_id", "claim_id", "last_error")
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        if info.field_name == "last_error":
            # Stores validate every new write strictly. Normalize legacy rows on
            # reconstruction so an older unbounded diagnostic cannot wedge the
            # side-effect recovery queue after an upgrade.
            return portable_persisted_event_side_effect_error(value)
        return require_clean_nonblank(value, info.field_name)

    @field_validator("lease_expires_at", "next_attempt_at", "updated_at")
    @classmethod
    def normalize_delivery_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)


def _persisted_event_authority_fields(event_type: EventType | str) -> tuple[str, ...]:
    if event_type in {
        EventType.SESSION_EXPORT_PUBLISHED,
        EventType.SESSION_EXPORT_RELEASED,
        EventType.SESSION_EXPORT_RETIRED,
    }:
        return ("export_commitment", "output_commitment")
    if event_type == EventType.SESSION_STARTED:
        return (
            "parent_session_id",
            SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
        )
    if event_type == EventType.SESSION_FORKED:
        return ("parent_session_id", "source_session_id")
    if event_type in {
        EventType.SESSION_RESUMED,
        EventType.SESSION_MESSAGE_QUEUED,
        EventType.SESSION_MESSAGE_DELIVERED,
    }:
        return (SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,)
    if event_type == EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED:
        return (
            "model_attempt_id",
            "model_step_id",
            "operation_id",
            "start_id",
            "stream_protocol",
        )
    if event_type == EventType.PROVIDER_OPERATION_RESOLVED:
        return (
            "model_attempt_id",
            "model_step_id",
            "operation_id",
            "resolution_id",
            "stage_id",
            "stream_protocol",
        )
    if event_type == EventType.TASK_COMPLETION_RESULT_RESOLVED:
        return (
            "application_request_sha256",
            "contract_fingerprint",
            "contract_id",
            "decision_id",
            "resolver_configuration_fingerprint",
            "resolver_id",
            "resolver_version",
            "result_digest",
            "result_kind",
            "result_reference_id",
            "task_id",
        )
    if event_type == EventType.TASK_INTERRUPTED_HANDOFF:
        return ("handoff_id", "task_id")
    if event_type == EventType.REQUEST_FOOTPRINT_RECORDED:
        return ("execution_profile_fingerprint",)
    if event_type == EventType.TOOL_EXPOSURE_RECORDED:
        return (
            "catalogue_revision",
            "execution_profile_fingerprint",
            "exposure_fingerprint",
            "model_step_id",
            "profile_id",
        )
    if event_type == EventType.MODEL_STARTED:
        return (MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY,)
    return ()


def _copy_event_for_session_store(event: Event) -> Event:
    """Strip caller-authored durable authority before persistence."""

    from cayu.collaboration._session_export_store import require_event_publication

    # Check provenance and exact owner bytes before copying can normalize input,
    # then recheck the detached value that is actually sent to persistence.
    # Leave rejection of non-exact events to the existing copy_event contract.
    if type(event) is Event:
        require_event_publication(event)
    copied = copy_event(event)
    validate_event_envelope(copied)
    require_event_publication(copied)
    authority_fields = _persisted_event_authority_fields(copied.type)
    if not authority_fields:
        return copied
    payload = copy_durable_json_object(copied.payload, "event payload")
    changed = False
    for field_name in authority_fields:
        value = payload.get(field_name)
        if type(value) is str and event_payload_authority_is_runtime_generated(
            copied,
            field_name=field_name,
            value=value,
        ):
            continue
        if field_name in payload:
            payload.pop(field_name)
            changed = True
    return copied if not changed else copied.model_copy(update={"payload": payload})


def _event_input_contract_is_runtime_owned(event: Event) -> bool:
    """Return whether one sanitized event carries the runtime-owned input marker."""

    if type(event) is not Event:
        raise TypeError("event must be an exact Event.")
    if event.type not in {
        EventType.SESSION_STARTED,
        EventType.SESSION_RESUMED,
        EventType.SESSION_MESSAGE_QUEUED,
        EventType.SESSION_MESSAGE_DELIVERED,
    }:
        return False
    marker = event.payload.get(SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY)
    return type(marker) is str and event_payload_authority_is_runtime_generated(
        event,
        field_name=SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
        value=marker,
    )


def _event_file_attachment_attestations_are_runtime_owned(event: Event) -> bool:
    """Return whether a model boundary carries exact runtime-resolved file proof."""

    if type(event) is not Event:
        raise TypeError("event must be an exact Event.")
    if event.type != EventType.MODEL_STARTED:
        return False
    marker = event.payload.get(MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY)
    return type(marker) is str and event_payload_authority_is_runtime_generated(
        event,
        field_name=MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY,
        value=marker,
    )


def restore_persisted_event_authority(
    event: Event,
    *,
    input_contract_runtime_owned: bool = False,
    file_attachment_attestations_runtime_owned: bool = False,
) -> Event:
    """Restore private authority proven by the built-in store ingestion boundary.

    Every built-in path capable of persisting an authoritative event field routes through
    :func:`_copy_event_for_session_store`, either as an ordinary event batch or
    through ``RuntimePublicationRequest``. A retained authority field could
    therefore only have crossed a persistence boundary with exact runtime
    authority. SQL serialization does not retain Pydantic private attributes;
    raw-record decoders use this helper to reconstruct that already-proven
    provenance without exposing a marker in the public event payload. Trusted
    Cayu JSONL restore uses the same fixed allowlist at its explicit backup
    boundary; callers must not restore JSONL obtained from an untrusted source.
    """

    copied = copy_event(event)
    if type(input_contract_runtime_owned) is not bool:
        raise TypeError("input_contract_runtime_owned must be a bool.")
    if type(file_attachment_attestations_runtime_owned) is not bool:
        raise TypeError("file_attachment_attestations_runtime_owned must be a bool.")
    fields = tuple(
        field_name
        for field_name in _persisted_event_authority_fields(copied.type)
        if type(copied.payload.get(field_name)) is str
        and (
            field_name != SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY or input_contract_runtime_owned
        )
        and (
            field_name != MODEL_FILE_ATTACHMENT_ATTESTATIONS_PAYLOAD_KEY
            or file_attachment_attestations_runtime_owned
        )
    )
    return event_with_runtime_payload_authority(copied, *fields) if fields else copied


def _copy_session_event_batch(session_id: str, events: list[Event]) -> tuple[str, list[Event]]:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(events) is not list:
        raise TypeError("Session events must be a list.")

    copied_events: list[Event] = []
    seen_event_ids: set[str] = set()
    for event in events:
        if type(event) is not Event:
            raise TypeError("Session events must be Event instances.")
        copied_event = _copy_event_for_session_store(event)
        if copied_event.session_id != session_id:
            raise ValueError("Event session_id does not match target session.")
        if copied_event.id in seen_event_ids:
            raise ValueError(f"Event already exists for session {session_id}: {copied_event.id}")
        seen_event_ids.add(copied_event.id)
        copied_events.append(copied_event)
    return session_id, copied_events
