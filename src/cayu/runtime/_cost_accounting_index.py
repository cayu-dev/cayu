"""Store-owned event indexes for bounded in-memory cost snapshot reads."""

from __future__ import annotations

from bisect import bisect_left, insort_right
from collections import OrderedDict
from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import pairwise
from typing import TYPE_CHECKING

from cayu.runtime._cost_accounting import COST_EVENT_TYPES, CostGroupKey, cost_group_key

if TYPE_CHECKING:
    from cayu.runtime._cost_accounting_refresh import CostAccountingRead
    from cayu.runtime.sessions import EventRecord

TimeEntry = tuple[datetime, int, CostGroupKey, "EventRecord"]


def _window_changes(
    read: CostAccountingRead,
) -> tuple[tuple[datetime | None, datetime | None], ...]:
    old, new = read.old_query, read.query
    endpoints = sorted(
        {value for value in (old.since, old.until, new.since, new.until) if value is not None}
    )
    boundaries = [None, *endpoints, None]
    intervals = []

    def included(since, until, lower, upper):
        return (since is None or (lower is not None and lower >= since)) and (
            until is None or (upper is not None and upper <= until)
        )

    for lower, upper in pairwise(boundaries):
        if included(old.since, old.until, lower, upper) != included(
            new.since, new.until, lower, upper
        ):
            intervals.append((lower, upper))
    return tuple(intervals)


def _position(entries: list[TimeEntry], at: datetime | None, *, end: bool = False) -> int:
    if at is None:
        return len(entries) if end else 0
    return bisect_left(entries, (at, 0), key=lambda entry: (entry[0], entry[1]))


class CostEventIndex:
    """References to authoritative records, maintained only at append/delete boundaries."""

    def __init__(self) -> None:
        self._groups: OrderedDict[CostGroupKey, list[EventRecord]] = OrderedDict()
        self._timestamps: list[TimeEntry] = []
        self._group_timestamps: dict[CostGroupKey, list[TimeEntry]] = {}

    @staticmethod
    def validate(record: EventRecord) -> None:
        if record.event.type in COST_EVENT_TYPES:
            record.event.timestamp.astimezone(UTC)

    def append(self, record: EventRecord) -> None:
        if record.event.type not in COST_EVENT_TYPES:
            return
        key = cost_group_key(record.event)
        self._groups.setdefault(key, []).append(record)
        self._groups.move_to_end(key)
        entry = (record.event.timestamp.astimezone(UTC), record.sequence, key, record)
        insort_right(self._timestamps, entry, key=lambda row: (row[0], row[1]))
        insort_right(
            self._group_timestamps.setdefault(key, []), entry, key=lambda row: (row[0], row[1])
        )

    def remove_session(self, session_id: str) -> None:
        self._groups = OrderedDict(
            (key, rows) for key, rows in self._groups.items() if key[0] != session_id
        )
        self._group_timestamps = {
            key: rows for key, rows in self._group_timestamps.items() if key[0] != session_id
        }
        self._timestamps = [row for row in self._timestamps if row[2][0] != session_id]

    def groups(self, read: CostAccountingRead) -> Iterator[tuple[CostGroupKey, list[EventRecord]]]:
        if read.previous is None:
            yield from self._groups.items()
            return
        previous = read.previous.through_sequence
        # Last-touch order makes all-time refresh depend only on appended groups.
        for key, records in reversed(self._groups.items()):
            if records[-1].sequence <= previous:
                break
            yield key, records
        intervals = _window_changes(read)
        for lower, upper in intervals:
            start = _position(self._timestamps, lower)
            end = _position(self._timestamps, upper, end=True)
            for index in range(start, end):
                entry = self._timestamps[index]
                key = entry[2]
                records = self._groups[key]
                if records[-1].sequence > previous:
                    continue  # The appended-group stream already owns this key.
                group_times = self._group_timestamps[key]
                # Select one stable witness across both changed intervals. This
                # avoids retaining an unbounded set of groups seen during expiry.
                first = None
                for group_lower, group_upper in intervals:
                    at = _position(group_times, group_lower)
                    if at < _position(group_times, group_upper, end=True):
                        first = group_times[at]
                        break
                if first is not None and first[1] == entry[1]:
                    yield key, records
        for key in read.remaining_pending_keys:
            records = self._groups.get(key)
            if records:
                yield key, records
