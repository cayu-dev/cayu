"""Read-only retained tool usage and bounded recent calls from durable event evidence."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Callable
from datetime import UTC
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.events import EventType
from cayu.sessions.event_queries import EventOrder, EventQuery, EventQueryResultTooLarge
from cayu.sessions.records import EventRecord
from cayu.sessions.store import SessionStore
from cayu.tool_inventory import TOOL_USAGE_EVENT_TYPES, tool_usage_contribution

_PAGE = 100
_BYTES = 16 * 1024 * 1024


class ToolUsageConflict(ValueError):
    """The retained source of a tool usage read changed; restart the read."""


class InvalidToolUsageQuery(ValueError):
    """A caller supplied an invalid continuation encoding."""


class _Cursor(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    store: str = Field(pattern=r"^[0-9a-f]{32}$")
    through: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    before: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    name_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def encode(self) -> str:
        return base64.urlsafe_b64encode(self.model_dump_json().encode()).decode().rstrip("=")

    @classmethod
    def decode(cls, value: str) -> _Cursor:
        try:
            if len(value) > 8192:
                raise ValueError
            raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
            result = cls.model_validate_json(raw)
            if result.before > result.through:
                raise ValueError
            return result
        except (ValueError, binascii.Error, ValidationError):
            raise InvalidToolUsageQuery("Invalid tool-call cursor.") from None


class ToolUsageReader:
    def __init__(
        self,
        store: SessionStore,
        *,
        names: dict[str, str],
        project_event: Callable[[EventRecord], dict[str, Any]],
    ) -> None:
        self.store = store
        self.names = names
        self.project_event = project_event

    async def _query(self, query: EventQuery) -> list[EventRecord]:
        # A large payload must shrink the batch, not silently truncate totals.
        while True:
            try:
                return await self.store.query_events_bounded(query, max_bytes=_BYTES)
            except EventQueryResultTooLarge:
                if query.limit == 1:
                    raise
                query = query.model_copy(update={"limit": max(1, query.limit // 2)})

    async def totals(self) -> dict[str, Any]:
        rows: dict[str, dict[str, Any]] = {
            name: {
                "name": name,
                "calls": 0,
                "failures": 0,
                "approved": 0,
                "rejected": 0,
                "last_used_at": None,
            }
            for name in sorted(set(self.names.values()))
        }

        def consume(page: list[EventRecord]) -> None:
            for record in page:
                contribution = tool_usage_contribution(record.event)
                if contribution is None or contribution.name not in self.names:
                    continue
                row = rows[self.names[contribution.name]]
                for field in ("calls", "failures", "approved", "rejected"):
                    row[field] += getattr(contribution, field)
                if contribution.calls:
                    at = record.event.timestamp
                    at = at if at.tzinfo is not None else at.replace(tzinfo=UTC)
                    if row["last_used_at"] is None or at > row["last_used_at"]:
                        row["last_used_at"] = at

        through = await self.store.fold_event_snapshot(
            EventQuery(limit=_PAGE, event_types=TOOL_USAGE_EVENT_TYPES),
            consume,
            max_bytes=_BYTES,
        )
        return {"tools": list(rows.values()), "through_sequence": through}

    async def recent(self, *, name: str | None, cursor: str | None, limit: int) -> dict[str, Any]:
        name_sha256 = hashlib.sha256(json.dumps(name, ensure_ascii=False).encode()).hexdigest()
        store_id = await self.store.read_store_identity()
        position = _Cursor.decode(cursor) if cursor is not None else None
        if position is not None and (
            position.store != store_id or position.name_sha256 != name_sha256
        ):
            raise ToolUsageConflict("Tool-call cursor no longer matches this listing.")
        page: list[EventRecord] = []
        newest = await self.store.fold_event_snapshot(
            EventQuery(
                before_sequence=position.before if position is not None else None,
                limit=_PAGE,
                order_by=EventOrder.SEQUENCE_DESC,
                event_types=TOOL_USAGE_EVENT_TYPES,
            ),
            page.extend,
            max_bytes=_BYTES,
            max_pages=1,
        )
        if position is not None and position.through > newest:
            raise ToolUsageConflict("Tool-call cursor no longer matches this listing.")
        through = newest if position is None else position.through
        calls: list[dict[str, Any]] = []
        # Every request owns a stable snapshot. Continuations bound sequence,
        # not cross-request visibility: retained events can change between calls.
        consumed = None
        for record in page:
            consumed = record.sequence
            contribution = tool_usage_contribution(record.event)
            if (
                contribution is None
                or not contribution.calls
                or contribution.name not in self.names
            ):
                continue
            public_name = self.names[contribution.name]
            if name is not None and public_name != name:
                continue
            session = await self.store.load_invocation_snapshot(record.event.session_id)
            if session is None:
                raise ToolUsageConflict("Session changed while reading tool calls.")
            # An old batch may outlive deletion/recreation. Resolve its exact
            # event inside the currently loaded session before assigning a link.
            current_events = await self._query(
                EventQuery(session_id=session.id, event_id=record.event.id, limit=1),
            )
            current = await self.store.load_invocation_snapshot(session.id)
            if (
                current is None
                or current.session_instance_id != session.session_instance_id
                or len(current_events) != 1
                or current_events[0].sequence != record.sequence
            ):
                raise ToolUsageConflict("Session changed while reading tool calls.")
            public = self.project_event(record)
            calls.append(
                {
                    "name": public_name,
                    "session_id": public["session_id"],
                    "session_instance_id": session.session_instance_id,
                    "event_id": public["id"],
                    "sequence": record.sequence,
                    "timestamp": record.event.timestamp,
                    "outcome": "interrupted"
                    if record.event.type
                    in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
                    and record.event.payload.get("interrupted") is True
                    else contribution.outcome,
                }
            )
            if len(calls) == limit:
                break
        more = consumed is not None and consumed > 1
        next_cursor = (
            _Cursor(
                store=store_id, through=through, before=consumed, name_sha256=name_sha256
            ).encode()
            if more
            else None
        )
        return {"calls": calls, "through_sequence": through, "next_cursor": next_cursor}
