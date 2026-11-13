"""Exact retained lookup totals inside one native event-read snapshot."""

from __future__ import annotations

from collections.abc import Collection
from datetime import UTC, datetime
from typing import NamedTuple

from cayu.events import EventType
from cayu.knowledge.usage import KnowledgeLookupTotal, knowledge_usage_contribution
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import EventRecord
from cayu.sessions.store import SessionStore


async def retained_knowledge_usage(
    store: SessionStore, identities: set[tuple[str, str]]
) -> tuple[dict[tuple[str, str], tuple[int, datetime | None]], int]:
    """Lookup totals for these ``(namespace, entry_id)`` pairs from one stable snapshot.

    Each lookup event counts once per distinct pair it returned. The returned
    sequence describes the snapshot's maximum, not a complete globally committed
    insertion prefix. The store chooses how to count: natively for the requested
    pairs, or by folding every retained lookup (``fold_knowledge_lookup_totals``).
    """
    return await store.knowledge_lookup_totals(identities)


async def fold_knowledge_lookup_totals(
    store: SessionStore, identities: Collection[tuple[str, str]]
) -> tuple[dict[tuple[str, str], tuple[int, datetime | None]], int]:
    """Fold bounded pages from one stable retained visibility snapshot.

    Accumulated values are published only after the native owner has completed
    successfully and released its snapshot.
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


class KnowledgeLookupMismatch(NamedTuple):
    namespace: str
    entry_id: str
    stored: KnowledgeLookupTotal | None
    recomputed: KnowledgeLookupTotal | None


async def recompute_knowledge_lookup_totals(
    store: SessionStore,
) -> tuple[dict[tuple[str, str], KnowledgeLookupTotal], int]:
    """Every entry's lookup totals, folded from the retained ``knowledge.lookup`` events.

    The reference for a store's stored totals: each lookup event counts once per
    distinct entry ID it returned, as ``fold_knowledge_lookup_totals`` counts.
    """

    folded: dict[tuple[str, str], tuple[int, set[str], datetime]] = {}
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
                count, sessions, last = folded.get((namespace, entry_id), (0, set(), at))
                sessions.add(record.event.session_id)
                folded[(namespace, entry_id)] = count + 1, sessions, max(last, at)

    through = await store.fold_event_snapshot(
        EventQuery(after_sequence=0, limit=100, event_type=EventType.KNOWLEDGE_LOOKUP),
        consume,
        max_bytes=16 * 1024 * 1024,
    )
    return {
        identity: KnowledgeLookupTotal(count, len(sessions), last)
        for identity, (count, sessions, last) in folded.items()
    }, through


class KnowledgeLookupCheck(NamedTuple):
    entries_stored: int
    mismatches: list[KnowledgeLookupMismatch]


async def check_knowledge_lookup_totals(store: SessionStore) -> KnowledgeLookupCheck:
    """Compare a store's stored lookup totals with a recompute from its events.

    No mismatches means they agree. Lookups recorded between the two reads show
    up as mismatches, so check a store that is not taking lookups, or check again.
    """

    stored = await store.stored_knowledge_lookup_totals()
    recomputed, _ = await recompute_knowledge_lookup_totals(store)
    return KnowledgeLookupCheck(
        len(stored),
        [
            KnowledgeLookupMismatch(*key, stored.get(key), recomputed.get(key))
            for key in sorted(stored.keys() | recomputed.keys())
            if stored.get(key) != recomputed.get(key)
        ],
    )
