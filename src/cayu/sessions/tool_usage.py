"""Read-only retained tool usage and bounded recent calls from durable event evidence."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.events import EventType
from cayu.sessions.event_queries import EventOrder, EventQuery, EventQueryResultTooLarge
from cayu.sessions.records import EventRecord
from cayu.sessions.store import SessionStore
from cayu.tool_inventory import TOOL_USAGE_EVENT_TYPES, tool_usage_contribution

_PAGE = 100
# Pages of call heads an all-tools listing reads to fill one page of registered tools.
_ALL_TOOL_HEAD_PAGES = 5
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


@dataclass(frozen=True, slots=True)
class ToolCallHead:
    """One retained tool event that may be a terminal call, without its payload.

    ``tool_name`` is the raw name the call counts under (``openai_web_search``
    for a hosted web search).
    """

    sequence: int
    session_id: str
    event_id: str
    tool_name: str | None


@dataclass(frozen=True, slots=True)
class ToolCallHeads:
    """Newest-first call heads and the newest event sequence of the same snapshot."""

    heads: tuple[ToolCallHead, ...]
    newest_sequence: int


@dataclass(frozen=True, slots=True)
class ToolUsageTally:
    """Retained usage of one raw tool name: terminal calls and approval decisions.

    ``last_used_at`` is the newest terminal call (UTC), or ``None`` when the
    name has only approval decisions.
    """

    calls: int = 0
    failures: int = 0
    approved: int = 0
    rejected: int = 0
    last_used_at: datetime | None = None


async def fold_tool_usage_totals(store: SessionStore) -> tuple[dict[str, ToolUsageTally], int]:
    """Tally every retained tool usage event by folding one stable event snapshot.

    This is the portable default behind ``SessionStore.tool_usage_totals``. It reads
    every retained tool event's payload, so native stores count in their own query.
    Returns the tallies by raw tool name and the snapshot's newest sequence.
    """

    tallies: dict[str, ToolUsageTally] = {}

    def consume(page: list[EventRecord]) -> None:
        for record in page:
            contribution = tool_usage_contribution(record.event)
            if contribution is None:
                continue
            tally = tallies.get(contribution.name, ToolUsageTally())
            last_used_at = tally.last_used_at
            if contribution.calls:
                at = record.event.timestamp
                at = at.replace(tzinfo=UTC) if at.tzinfo is None else at.astimezone(UTC)
                if last_used_at is None or at > last_used_at:
                    last_used_at = at
            tallies[contribution.name] = ToolUsageTally(
                calls=tally.calls + contribution.calls,
                failures=tally.failures + contribution.failures,
                approved=tally.approved + contribution.approved,
                rejected=tally.rejected + contribution.rejected,
                last_used_at=last_used_at,
            )

    through = await store.fold_event_snapshot(
        EventQuery(limit=_PAGE, event_types=TOOL_USAGE_EVENT_TYPES),
        consume,
        max_bytes=_BYTES,
    )
    return tallies, through


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
        tallies, through = await self.store.tool_usage_totals()
        for raw_name, tally in tallies.items():
            if raw_name not in self.names:
                continue
            row = rows[self.names[raw_name]]
            for field in ("calls", "failures", "approved", "rejected"):
                row[field] += getattr(tally, field)
            at = tally.last_used_at
            if at is not None and (row["last_used_at"] is None or at > row["last_used_at"]):
                row["last_used_at"] = at
        return {"tools": list(rows.values()), "through_sequence": through}

    async def recent(self, *, name: str | None, cursor: str | None, limit: int) -> dict[str, Any]:
        name_sha256 = hashlib.sha256(json.dumps(name, ensure_ascii=False).encode()).hexdigest()
        store_id = await self.store.read_store_identity()
        position = _Cursor.decode(cursor) if cursor is not None else None
        if position is not None and (
            position.store != store_id or position.name_sha256 != name_sha256
        ):
            raise ToolUsageConflict("Tool-call cursor no longer matches this listing.")
        try:
            heads, newest, consumed, more = await self._indexed_heads(name, position, limit)
        except NotImplementedError:
            heads, newest, consumed, more = await self._scanned_heads(name, position, limit)
        if position is not None and position.through > newest:
            raise ToolUsageConflict("Tool-call cursor no longer matches this listing.")
        through = newest if position is None else position.through
        calls = await self._calls(heads)
        next_cursor = (
            _Cursor(
                store=store_id, through=through, before=consumed, name_sha256=name_sha256
            ).encode()
            if more and consumed is not None and consumed > 1
            else None
        )
        return {"calls": calls, "through_sequence": through, "next_cursor": next_cursor}

    def _public_name(self, head: ToolCallHead) -> str | None:
        return None if head.tool_name is None else self.names.get(head.tool_name)

    async def _indexed_heads(
        self, name: str | None, position: _Cursor | None, limit: int
    ) -> tuple[list[ToolCallHead], int, int | None, bool]:
        """Up to ``limit`` call heads newest first, read by tool name from the store's index.

        Returns the heads, the snapshot's newest sequence, the oldest sequence
        read, and whether older calls may follow.
        """

        before = position.before if position is not None else None
        if name is not None:
            raw_names = tuple(sorted(raw for raw, public in self.names.items() if public == name))
            result = await self.store.tool_call_heads(
                raw_names, before_sequence=before, limit=limit
            )
            heads = list(result.heads)
            return (
                heads,
                result.newest_sequence,
                heads[-1].sequence if heads else None,
                len(heads) == limit,
            )
        # Listing every tool can meet calls of unregistered names; read a few
        # pages of heads to fill the page.
        heads = []
        newest = 0
        consumed = None
        more = False
        for _ in range(_ALL_TOOL_HEAD_PAGES):
            result = await self.store.tool_call_heads(None, before_sequence=before, limit=_PAGE)
            newest = max(newest, result.newest_sequence)
            more = len(result.heads) == _PAGE
            for head in result.heads:
                consumed = before = head.sequence
                if self._public_name(head) is None:
                    continue
                heads.append(head)
                if len(heads) == limit:
                    return heads, newest, consumed, True
            if not more:
                break
        return heads, newest, consumed, more

    async def _scanned_heads(
        self, name: str | None, position: _Cursor | None, limit: int
    ) -> tuple[list[ToolCallHead], int, int | None, bool]:
        """One page of retained tool events, filtered, for stores without call heads."""

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
        heads: list[ToolCallHead] = []
        consumed = None
        for record in page:
            consumed = record.sequence
            contribution = tool_usage_contribution(record.event)
            if contribution is None or not contribution.calls:
                continue
            head = ToolCallHead(
                sequence=record.sequence,
                session_id=record.event.session_id,
                event_id=record.event.id,
                tool_name=contribution.name,
            )
            public = self._public_name(head)
            if public is None or (name is not None and public != name):
                continue
            heads.append(head)
            if len(heads) == limit:
                break
        return heads, newest, consumed, consumed is not None and consumed > 1

    async def _calls(self, heads: list[ToolCallHead]) -> list[dict[str, Any]]:
        """Load and project the calls of ``heads``, checking each session before and after.

        An old head may outlive deletion and recreation of its session, so each
        event is read again inside the current incarnation before it gets a link.
        """

        session_ids = list(dict.fromkeys(head.session_id for head in heads))
        before = {
            session_id: await self.store.load_invocation_snapshot(session_id)
            for session_id in session_ids
        }
        records = await self._load(heads) if heads else []
        for session_id, snapshot in before.items():
            after = await self.store.load_invocation_snapshot(session_id)
            if (
                snapshot is None
                or after is None
                or after.session_instance_id != snapshot.session_instance_id
            ):
                raise ToolUsageConflict("Session changed while reading tool calls.")
        calls: list[dict[str, Any]] = []
        for record in records:
            contribution = tool_usage_contribution(record.event)
            if (
                contribution is None
                or not contribution.calls
                or contribution.name not in self.names
            ):
                raise ToolUsageConflict("Session changed while reading tool calls.")
            snapshot = before[record.event.session_id]
            assert snapshot is not None
            public = self.project_event(record)
            calls.append(
                {
                    "name": self.names[contribution.name],
                    "session_id": public["session_id"],
                    "session_instance_id": snapshot.session_instance_id,
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
        return calls

    async def _load(self, heads: list[ToolCallHead]) -> list[EventRecord]:
        """The full events of ``heads``, read in one store call."""

        results = await self.store.query_events_bounded_many(
            [
                EventQuery(session_id=head.session_id, event_id=head.event_id, limit=1)
                for head in heads
            ],
            max_bytes=_BYTES,
        )
        records: list[EventRecord] = []
        for head, result in zip(heads, results, strict=True):
            if isinstance(result, EventQueryResultTooLarge):
                raise result
            if len(result) != 1 or result[0].sequence != head.sequence:
                raise ToolUsageConflict("Session changed while reading tool calls.")
            records.append(result[0])
        return records
