"""Bounded direct-child session lineage contracts and cursors."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.events import EVENT_ID_MAX_CHARS, EventType
from cayu.sessions.records import MAX_SESSION_ID_BYTES
from cayu.sessions.topology import _bounded_session_topology_text

SESSION_LINEAGE_DEFAULT_CHILD_LIMIT = 25
SESSION_LINEAGE_MAX_CHILD_LIMIT = 100
SESSION_LINEAGE_MAX_IDENTIFIER_BYTES = MAX_SESSION_ID_BYTES
SESSION_LINEAGE_MAX_EVENT_ID_BYTES = EVENT_ID_MAX_CHARS * 4
SESSION_LINEAGE_MAX_TIMESTAMP_BYTES = 64
SESSION_LINEAGE_MAX_CURSOR_BYTES = 8192
SESSION_LINEAGE_MAX_ORIGIN_EVENTS = 2


class SessionLineageOrigin(BaseModel):
    """Payload-free durable identity for one session start or fork event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    event_id: str = Field(max_length=EVENT_ID_MAX_CHARS)
    event_type: EventType

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, value: str) -> str:
        return _bounded_session_topology_text(
            value,
            "event_id",
            max_bytes=SESSION_LINEAGE_MAX_EVENT_ID_BYTES,
        )

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, value: EventType) -> EventType:
        if value not in {EventType.SESSION_STARTED, EventType.SESSION_FORKED}:
            raise ValueError("Session lineage origins must be session start or fork events.")
        return value


class SessionLineageNode(BaseModel):
    """Minimal byte-bounded identity used to enumerate one child branch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    parent_session_id: str
    created_at: datetime
    origin_events: tuple[SessionLineageOrigin, ...] = Field(
        default=(),
        max_length=SESSION_LINEAGE_MAX_ORIGIN_EVENTS,
    )

    @field_validator("id", "parent_session_id")
    @classmethod
    def validate_identifiers(cls, value: str, info) -> str:
        return _bounded_session_topology_text(
            value,
            info.field_name,
            max_bytes=SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
        )

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("origin_events")
    @classmethod
    def copy_origin_events(
        cls,
        value: tuple[SessionLineageOrigin, ...],
    ) -> tuple[SessionLineageOrigin, ...]:
        return tuple(origin.model_copy(deep=True) for origin in value)


class SessionLineageQuery(BaseModel):
    """One independently pageable, payload-free direct-child lineage read."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, frozen=True)

    parent_session_id: str
    cursor: str | None = None
    limit: StrictInt = Field(
        default=SESSION_LINEAGE_DEFAULT_CHILD_LIMIT,
        ge=1,
        le=SESSION_LINEAGE_MAX_CHILD_LIMIT,
    )

    @field_validator("parent_session_id")
    @classmethod
    def validate_parent_session_id(cls, value: str) -> str:
        return _bounded_session_topology_text(
            value,
            "parent_session_id",
            max_bytes=SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
        )

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_session_topology_text(
            value,
            "cursor",
            max_bytes=SESSION_LINEAGE_MAX_CURSOR_BYTES,
        )


class SessionLineageResult(BaseModel):
    """Backend-neutral bounded direct-child lineage page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_session_id: str
    children: tuple[SessionLineageNode, ...] = Field(
        default=(),
        max_length=SESSION_LINEAGE_MAX_CHILD_LIMIT,
    )
    next_cursor: str | None = None
    has_more: StrictBool = False

    @field_validator("parent_session_id")
    @classmethod
    def validate_parent_session_id(cls, value: str) -> str:
        return _bounded_session_topology_text(
            value,
            "parent_session_id",
            max_bytes=SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
        )

    @field_validator("next_cursor")
    @classmethod
    def validate_next_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_session_topology_text(
            value,
            "next_cursor",
            max_bytes=SESSION_LINEAGE_MAX_CURSOR_BYTES,
        )

    @model_validator(mode="after")
    def validate_page(self) -> SessionLineageResult:
        if self.has_more != (self.next_cursor is not None):
            raise ValueError("Lineage continuation state and cursor must agree.")
        child_ids = tuple(child.id for child in self.children)
        if len(child_ids) != len(set(child_ids)):
            raise ValueError("A lineage page must not repeat a child session.")
        if any(child.parent_session_id != self.parent_session_id for child in self.children):
            raise ValueError("A lineage page contains a contradictory parent edge.")
        if tuple(self.children) != tuple(
            sorted(self.children, key=lambda child: (child.created_at, child.id))
        ):
            raise ValueError("Lineage children must use stable creation ordering.")
        if self.next_cursor is not None:
            cursor_created_at, cursor_id = decode_session_lineage_cursor(
                self.next_cursor,
                parent_session_id=self.parent_session_id,
            )
            if not self.children or (cursor_created_at, cursor_id) != (
                self.children[-1].created_at,
                self.children[-1].id,
            ):
                raise ValueError("A lineage cursor must identify the last returned child.")
        return self


def copy_session_lineage_query(query: SessionLineageQuery) -> SessionLineageQuery:
    """Detach and revalidate a bounded lineage query at the store boundary."""
    if type(query) is not SessionLineageQuery:
        raise TypeError("Session lineage queries must be SessionLineageQuery instances.")
    return SessionLineageQuery.model_validate(query.model_dump(mode="python"))


def encode_session_lineage_cursor(
    parent_session_id: str,
    node: SessionLineageNode,
) -> str:
    """Encode one parent-bound minimal-lineage continuation cursor."""

    parent_session_id = _bounded_session_topology_text(
        parent_session_id,
        "parent_session_id",
        max_bytes=SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
    )
    if type(node) is not SessionLineageNode:
        raise TypeError("Session lineage cursors require SessionLineageNode values.")
    raw = json.dumps(
        [
            parent_session_id,
            node.created_at.astimezone(UTC).isoformat(),
            node.id,
        ],
        separators=(",", ":"),
    )
    encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")
    if len(encoded) > SESSION_LINEAGE_MAX_CURSOR_BYTES:
        raise ValueError("Session lineage cursor exceeds its byte limit.")
    return encoded


def decode_session_lineage_cursor(
    cursor: str,
    *,
    parent_session_id: str,
) -> tuple[datetime, str]:
    """Decode a minimal-lineage cursor and enforce its parent binding."""

    parent_session_id = _bounded_session_topology_text(
        parent_session_id,
        "parent_session_id",
        max_bytes=SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
    )
    try:
        cursor = _bounded_session_topology_text(
            cursor,
            "cursor",
            max_bytes=SESSION_LINEAGE_MAX_CURSOR_BYTES,
        )
        encoded = cursor.encode("ascii")
        raw = base64.b64decode(encoded, altchars=b"-_", validate=True)
        if base64.urlsafe_b64encode(raw) != encoded:
            raise ValueError("Non-canonical session lineage cursor.")
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, TypeError) as exc:
        raise ValueError("Invalid session lineage cursor.") from exc
    if (
        type(decoded) is not list
        or len(decoded) != 3
        or type(decoded[0]) is not str
        or type(decoded[1]) is not str
        or type(decoded[2]) is not str
        or decoded[0] != parent_session_id
    ):
        raise ValueError("Invalid session lineage cursor.")
    try:
        child_session_id = _bounded_session_topology_text(
            decoded[2],
            "cursor child_session_id",
            max_bytes=SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
        )
        created_at = datetime.fromisoformat(decoded[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid session lineage cursor.") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("Invalid session lineage cursor.")
    return created_at.astimezone(UTC), child_session_id
