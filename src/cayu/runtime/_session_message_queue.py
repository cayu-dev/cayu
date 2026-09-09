"""Runtime queue lifecycle rules; stores own the complete storage transaction.

Raw revisions intentionally fingerprint storage values, including undecodable
JSON text. They are opaque backend-local CAS tokens, not portable content hashes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cayu.core.events import Event, EventType
from cayu.runtime.approvals import resolution_actor_payload
from cayu.runtime.session_message_lifecycle import (
    SessionMessageActionRequest,
    SessionMessageConflict,
    SessionMessageCursor,
    SessionMessageQuery,
    SessionMessageQueueStatus,
    SessionMessageSource,
    copy_session_message_cursor,
)


@dataclass(frozen=True)
class OversizedStorageValue:
    """Out-of-band bounded witness, never an ordinary stored JSON object."""

    sha256: str
    byte_length: int | None = None
    storage_type: str | None = None


def _storage_value(value: Any) -> Any:
    if type(value) is OversizedStorageValue:
        return ["oversized", value.sha256, value.byte_length, value.storage_type]
    if isinstance(value, datetime):
        return ["datetime", value.isoformat()]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, Mapping):
        return ["mapping", {key: _storage_value(item) for key, item in value.items()}]
    if isinstance(value, (list, tuple)):
        return ["sequence", [_storage_value(item) for item in value]]
    return ["scalar", value]


def raw_revision(raw: Mapping[str, Any]) -> str:
    """Bind every stored column, including malformed content and terminal proof."""
    return hashlib.sha256(
        json.dumps(
            _storage_value(raw),
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def inspection_priority(delivery_mode: Any) -> int:
    """Existing delivery precedence, with unknown modes inspectable at the end."""
    from cayu.runtime.sessions import SessionMessageDeliveryMode

    if delivery_mode is SessionMessageDeliveryMode.NEXT_TURN or (
        type(delivery_mode) is str and delivery_mode == "next_turn"
    ):
        return 0
    if delivery_mode is SessionMessageDeliveryMode.ON_IDLE or (
        type(delivery_mode) is str and delivery_mode == "on_idle"
    ):
        return 1
    return 2


def copy_inspection_query(query: SessionMessageQuery) -> SessionMessageQuery:
    """Defensively reconstruct even model_copy-created or post-construction cursor values."""
    if type(query) is not SessionMessageQuery:
        raise TypeError("Queue inspection requires a SessionMessageQuery.")
    cursor = query.cursor
    if cursor is not None:
        cursor = copy_session_message_cursor(cursor)
    return SessionMessageQuery(session_id=query.session_id, cursor=cursor, limit=query.limit)


def inspection_boundary(
    session: Any,
    cursor: SessionMessageCursor | None,
    maximum: int,
) -> SessionMessageCursor:
    if cursor is not None:
        if cursor.session_instance_id != session.instance_id:
            raise SessionMessageConflict()
        return cursor
    return SessionMessageCursor(
        session_instance_id=session.instance_id,
        through_ordering_key=maximum,
        after_priority=0,
        after_ordering_key=0,
    )


def require_authorized_session_instance(session: Any, expected: str | None) -> None:
    """Check a private read fence before stores access queue or source content."""
    if expected is not None and (type(expected) is not str or session.instance_id != expected):
        raise SessionMessageConflict()


def inspection_next_cursor(
    boundary: SessionMessageCursor,
    priority: int,
    ordering_key: int,
) -> SessionMessageCursor:
    return SessionMessageCursor(
        session_instance_id=boundary.session_instance_id,
        through_ordering_key=boundary.through_ordering_key,
        after_priority=priority,
        after_ordering_key=ordering_key,
    )


def inspect_record(raw: Mapping[str, Any], decode: Callable[[], Any]) -> Any:
    from cayu.runtime.sessions import SessionMessageInspectionRecord

    try:
        status = SessionMessageQueueStatus(raw["status"])
    except (ValueError, TypeError):
        status = None
    terminal_event_id = None
    terminal = raw.get("terminal_json")
    try:
        terminal = json.loads(terminal) if isinstance(terminal, str) else terminal
        if terminal is not None:
            if type(terminal["version"]) is not int or terminal["version"] != 1:
                raise ValueError("Invalid terminal proof version.")
            event = Event.model_validate(terminal["event"])
            if terminal["status"] != raw["status"] or event.session_id != raw["session_id"]:
                raise ValueError("Invalid terminal proof.")
            if event.payload.get("queue_id") != raw["queue_id"]:
                raise ValueError("Invalid terminal identity.")
            if event.type != f"session.message.{raw['status']}":
                raise ValueError("Invalid terminal event kind.")
            terminal_event_id = event.id
        elif status not in {SessionMessageQueueStatus.QUEUED, SessionMessageQueueStatus.DELIVERED}:
            raise ValueError("Missing terminal proof.")
        message = decode()
        if status is SessionMessageQueueStatus.DELIVERED:
            if any(
                getattr(message, key) is None
                for key in (
                    "delivered_run_epoch",
                    "delivered_transcript_cursor",
                    "delivered_at",
                    "delivered_event_id",
                )
            ):
                raise ValueError("Missing delivery proof.")
            terminal_event_id = message.delivered_event_id
    except Exception:
        message = None
    return SessionMessageInspectionRecord(
        queue_id=raw["queue_id"],
        ordering_key=raw["ordering_key"],
        revision=raw_revision(raw),
        status=status,
        validity="valid" if message is not None else "unreadable",
        message=message,
        terminal_event_id=terminal_event_id,
    )


def action_material(request: SessionMessageActionRequest) -> dict[str, Any]:
    """Complete expected operation, with actor claims excluded from audit identity."""
    return {
        "session_id": request.session_id,
        "session_instance_id": request.session_instance_id,
        "queue_id": request.queue_id,
        "idempotency_key": request.idempotency_key,
        "expected_revision": request.expected_revision,
        "action": request.action,
        "requested_by": resolution_actor_payload(request.requested_by),
    }


def source_event_payload(source: SessionMessageSource | None) -> dict[str, Any]:
    """Only a verified admission or its durable audit may supply this value."""
    return {} if source is None else {"source": source.model_dump(mode="json")}


def source_audit_payload(
    raw: Mapping[str, Any],
    accepted_event: Event | None,
    *,
    quarantine: bool = False,
) -> dict[str, Any]:
    """Read canonical acceptance-owned provenance, never mutable queue conditions."""
    try:
        if (
            accepted_event is None
            or accepted_event.type != EventType.SESSION_MESSAGE_QUEUED
            or accepted_event.session_id != raw["session_id"]
            or (not quarantine and accepted_event.id != raw["accepted_event_id"])
            or accepted_event.payload.get("queue_id") != raw["queue_id"]
        ):
            raise ValueError
        source = accepted_event.payload.get("source")
        return source_event_payload(
            None if source is None else SessionMessageSource.model_validate(source)
        )
    except Exception:
        raise SessionMessageConflict() from None


def terminal_event(
    session: Any,
    raw: Mapping[str, Any],
    status: SessionMessageQueueStatus,
    now: datetime,
    *,
    accepted_event: Event | None,
    actor: Any = None,
    interaction_id: str | None = None,
) -> Event:
    """Never copy rejected row content, mode, actor, or exception text into events."""
    return Event(
        type=EventType(f"session.message.{status}"),
        session_id=session.id,
        interaction_id=interaction_id,
        agent_name=session.agent_name,
        environment_name=session.environment_name,
        timestamp=now,
        payload={
            **source_audit_payload(
                raw,
                accepted_event,
                quarantine=status is SessionMessageQueueStatus.QUARANTINED,
            ),
            "queue_id": raw["queue_id"],
            "ordering_key": raw["ordering_key"],
            "actor": resolution_actor_payload(actor),
            "run_epoch": session.run_epoch,
            "status": str(status),
        },
    )


def terminal_receipt(
    status: SessionMessageQueueStatus,
    event: Event,
    request: SessionMessageActionRequest | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "status": str(status),
        "action": None if request is None else action_material(request),
        "event": event.model_dump(mode="json"),
    }


def replay_action(
    raw: Mapping[str, Any],
    request: SessionMessageActionRequest,
    *,
    accepted_event: Event | None,
) -> Event | None:
    value = raw.get("terminal_json")
    if value is None:
        return None
    try:
        value = json.loads(value) if isinstance(value, str) else value
        if (
            type(value["version"]) is not int
            or value["version"] != 1
            or value["action"] != action_material(request)
        ):
            raise SessionMessageConflict()
        if value["status"] != raw["status"]:
            raise SessionMessageConflict()
        original = dict(raw)
        original["status"] = "queued"
        original["terminal_json"] = None
        if raw_revision(original) != request.expected_revision:
            raise SessionMessageConflict()
        event = Event.model_validate(value["event"])
        expected = "withdrawn" if request.action == "withdraw" else "quarantined"
        if (
            value["status"] != expected
            or event.type != f"session.message.{expected}"
            or event.session_id != request.session_id
            or event.payload.get("queue_id") != request.queue_id
            or type(event.payload.get("ordering_key")) is not int
            or event.payload["ordering_key"] != raw["ordering_key"]
            or event.payload.get("actor") != resolution_actor_payload(request.requested_by)
            or event.payload.get("status") != expected
            or event.payload.get("source")
            != source_audit_payload(
                raw,
                accepted_event,
                quarantine=request.action == "quarantine",
            ).get("source")
        ):
            raise SessionMessageConflict()
        return event
    except Exception:
        raise SessionMessageConflict() from None


class SourceTranscriptHasher:
    """Incremental encoding of fork_source_transcript_sha256's exact document.

    SQL callers page retained records under their existing transaction. Canonical
    per-record encoding preserves numeric normalization and excludes attribution,
    exactly like the public snapshot digest.
    """

    def __init__(self, cursor: int) -> None:
        from cayu._validation import canonical_durable_json_bytes

        empty = canonical_durable_json_bytes(
            {
                "record_type": "cayu.fork-source-transcript",
                "schema_version": 1,
                "cursor": cursor,
                "records": [],
            },
            "fork_source.transcript",
        )
        prefix, self._suffix = empty.split(b'"records":[]', 1)
        self._hash = hashlib.sha256(prefix + b'"records":[')
        self._count = 0

    def add(self, index: int, message: Any) -> None:
        from cayu._validation import canonical_durable_json_bytes

        if self._count:
            self._hash.update(b",")
        self._hash.update(
            canonical_durable_json_bytes(
                {
                    "index": index,
                    "message": message.model_dump(mode="json", warnings=False),
                },
                "fork_source.transcript",
            )
        )
        self._count += 1

    def hexdigest(self) -> str:
        result = self._hash.copy()
        result.update(b"]" + self._suffix)
        return result.hexdigest()


def source_snapshot(
    session: Any,
    cursor: int,
    *,
    transcript: Any = None,
    transcript_sha256: str | None = None,
    checkpoint: Any = None,
    include_checkpoint_digest: bool = False,
) -> SessionMessageSource:
    from cayu.runtime.sessions import fork_source_transcript_sha256

    checkpoint_digest = None
    if include_checkpoint_digest:
        from cayu.runtime.session_message_lifecycle import session_message_checkpoint_sha256

        checkpoint_digest = session_message_checkpoint_sha256(checkpoint)
    return SessionMessageSource(
        session_id=session.id,
        session_instance_id=session.instance_id,
        run_epoch=session.run_epoch,
        transcript_cursor=cursor,
        transcript_sha256=transcript_sha256
        if transcript is None
        else fork_source_transcript_sha256(transcript),
        checkpoint_sha256=checkpoint_digest,
    )
