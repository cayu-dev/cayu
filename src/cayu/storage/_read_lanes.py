"""Read lanes keep bulk reads off the connections that interactive reads use.

A dashboard page asks for small reads (one session, its transcript, its state) and
for aggregate reads that scan many sessions or events (overview counts, usage
rollups, tool usage). When both share one set of store connections, a few aggregate
reads hold every connection and a 0.1 s session read waits for them.

Callers mark aggregate work with :func:`bulk_reads`. Each store decides what the
bulk lane means for it: the SQLite session store gives it separate reader
connections and worker threads, and Postgres stores cap how many pooled
connections it may hold at once. Stores that have nothing to separate ignore it.
"""

from __future__ import annotations

import enum
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class ReadLane(enum.Enum):
    """Which share of a store's read capacity a read draws from."""

    INTERACTIVE = "interactive"
    BULK = "bulk"


_current_read_lane: ContextVar[ReadLane] = ContextVar(
    "cayu_storage_read_lane", default=ReadLane.INTERACTIVE
)


def current_read_lane() -> ReadLane:
    """Return the lane of reads started from the current context."""

    return _current_read_lane.get()


@contextmanager
def bulk_reads() -> Iterator[None]:
    """Run store reads started in this block, and in tasks it creates, on the bulk lane."""

    token = _current_read_lane.set(ReadLane.BULK)
    try:
        yield
    finally:
        _current_read_lane.reset(token)


__all__ = ["ReadLane", "bulk_reads", "current_read_lane"]
