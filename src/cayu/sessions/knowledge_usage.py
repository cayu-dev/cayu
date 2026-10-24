"""Exact retained lookup totals inside one native event-read snapshot."""

from __future__ import annotations

from datetime import UTC, datetime

from cayu.events import EventType
from cayu.knowledge.usage import knowledge_usage_contribution
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import EventRecord
from cayu.sessions.store import SessionStore


async def retained_knowledge_usage(
    store: SessionStore, identities: set[tuple[str, str]]
) -> tuple[dict[tuple[str, str], tuple[int, datetime | None]], int]:
    """Fold bounded pages from one stable retained visibility snapshot.

    The returned sequence describes the snapshot's maximum, not a complete
    globally committed insertion prefix. Accumulated values are published only
    after the native owner has completed successfully and released its snapshot.
    """
    totals: dict[tuple[str, str], tuple[int, datetime | None]] = {
        identity: (0, None) for identity in identities
    }
    after = 0

    def consume(page: list[EventRecord]) -> None:
        nonlocal after
        for record in page:
            if record.sequence <= after:
                raise ValueError("Knowledge usage snapshot did not advance.")
            after = record.sequence
            contribution = knowledge_usage_contribution(record.event)
            if contribution is None:
                continue
            namespace, ids = contribution
            at = record.event.timestamp
            at = at.replace(tzinfo=UTC) if at.tzinfo is None else at.astimezone(UTC)
            for entry_id in ids:
                identity = namespace, entry_id
                if identity not in totals:
                    continue
                count, last = totals[identity]
                totals[identity] = count + 1, at if last is None else max(last, at)

    through = await store.fold_event_snapshot(
        EventQuery(after_sequence=0, limit=100, event_type=EventType.KNOWLEDGE_LOOKUP),
        consume,
        max_bytes=16 * 1024 * 1024,
    )
    return totals, through
