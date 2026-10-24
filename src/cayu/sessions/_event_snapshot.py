"""Traversal for an already-owned stable native event snapshot."""

from collections.abc import Awaitable, Callable

from cayu.sessions.event_queries import (
    EventOrder,
    EventQuery,
    EventQueryResultTooLarge,
    copy_event_query,
)
from cayu.sessions.records import EventRecord

EventSnapshotConsumer = Callable[[list[EventRecord]], None]


class EventSnapshotFold:
    def __init__(self, query: EventQuery, *, max_bytes: int, max_pages: int | None):
        self.query = copy_event_query(query)
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer.")
        if max_pages is not None and (type(max_pages) is not int or max_pages < 1):
            raise ValueError("max_pages must be a positive integer or None.")
        self.max_bytes = max_bytes
        self.max_pages = max_pages
        self.pages = 0

    def smaller(self) -> None:
        if self.query.limit == 1:
            raise EventQueryResultTooLarge(self.max_bytes)
        self.query = self.query.model_copy(update={"limit": max(1, self.query.limit // 2)})

    def advance(self, page: list[EventRecord]) -> bool:
        self.pages += 1
        if not page or (self.max_pages is not None and self.pages >= self.max_pages):
            return False
        field = (
            "after_sequence"
            if self.query.order_by == EventOrder.SEQUENCE_ASC
            else "before_sequence"
        )
        position = page[-1].sequence
        if self.query.order_by == EventOrder.SEQUENCE_ASC:
            if (
                self.query.before_sequence is not None
                and position + 1 >= self.query.before_sequence
            ):
                return False
        elif position <= (self.query.after_sequence or 0) + 1:
            return False
        self.query = self.query.model_copy(update={field: position})
        return True


def fold_snapshot(
    fold: EventSnapshotFold,
    read: Callable[[EventQuery], list[EventRecord]],
    consume: EventSnapshotConsumer,
) -> int:
    newest = read(EventQuery(limit=1, order_by=EventOrder.SEQUENCE_DESC))
    while True:
        try:
            page = read(fold.query)
        except EventQueryResultTooLarge:
            fold.smaller()
            continue
        if page:
            consume(page)
        if not fold.advance(page):
            return newest[0].sequence if newest else 0


async def fold_snapshot_async(
    fold: EventSnapshotFold,
    read: Callable[[EventQuery], Awaitable[list[EventRecord]]],
    consume: EventSnapshotConsumer,
) -> int:
    newest = await read(EventQuery(limit=1, order_by=EventOrder.SEQUENCE_DESC))
    while True:
        try:
            page = await read(fold.query)
        except EventQueryResultTooLarge:
            fold.smaller()
            continue
        if page:
            consume(page)
        if not fold.advance(page):
            return newest[0].sequence if newest else 0
