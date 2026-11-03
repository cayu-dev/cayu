"""Summarize the event-loop lag a worker recorded on one session.

A worker that runs a session records ``runtime.event_loop.lagged`` events on it while its
process's event loop stays late (see ``cayu.runtime._event_loop_lag_sessions``).
`cayu session show` and `cayu task show` read them back as one summary: how many reports,
how long the loop was late while the session ran, the worst lag, and when.
"""

from __future__ import annotations

from collections.abc import Awaitable, Iterable
from typing import Any, Protocol, cast

from cayu.events import Event, EventType
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.records import EventRecord

EVENT_LOOP_LAG_EVENT_TYPE = EventType.RUNTIME_EVENT_LOOP_LAGGED
# A worker writes at most one report a minute per session: a day of reports.
MAX_EVENT_LOOP_LAG_REPORTS = 1440
_PAGE = 500


class _EventReader(Protocol):
    def query_events(self, query: EventQuery) -> Awaitable[list[EventRecord]]: ...


def event_loop_lag_summary(events: Iterable[Event], *, truncated: bool = False) -> dict[str, Any]:
    """One session's recorded lag.

    ``reports`` counts the events, ``lagging_ms`` adds up how long the loop was late while
    the session ran, ``max_lag_ms`` is the worst sample, ``first_at`` and ``last_at`` say
    when, and ``processes`` how many worker processes reported. ``truncated`` says only the
    first reports were read.
    """

    reports = 0
    lagging_ms = 0
    max_lag_ms: int | None = None
    first_at = last_at = None
    processes: set[int] = set()
    for event in events:
        if event.type != EVENT_LOOP_LAG_EVENT_TYPE:
            continue
        payload = event.payload
        reports += 1
        lagging_ms += _count(payload.get("lagging_ms")) or 0
        worst = _count(payload.get("max_lag_ms"))
        if worst is not None:
            max_lag_ms = worst if max_lag_ms is None else max(max_lag_ms, worst)
        first_at = event.timestamp if first_at is None else min(first_at, event.timestamp)
        last_at = event.timestamp if last_at is None else max(last_at, event.timestamp)
        pid = _count(payload.get("pid"))
        if pid is not None:
            processes.add(pid)
    return {
        "reports": reports,
        "lagging_ms": lagging_ms,
        "max_lag_ms": max_lag_ms,
        "first_at": None if first_at is None else first_at.isoformat(),
        "last_at": None if last_at is None else last_at.isoformat(),
        "processes": len(processes),
        "truncated": truncated,
    }


async def read_event_loop_lag(reader: _EventReader, session_id: str) -> dict[str, Any]:
    """Read one session's lag reports, a page at a time, and summarize them."""

    events: list[Event] = []
    after = 0
    truncated = False
    while True:
        page = await reader.query_events(
            EventQuery(
                session_id=session_id,
                event_types=(EVENT_LOOP_LAG_EVENT_TYPE,),
                after_sequence=after,
                limit=_PAGE,
                order_by=EventOrder.SEQUENCE_ASC,
            )
        )
        events.extend(record.event for record in page)
        if len(events) > MAX_EVENT_LOOP_LAG_REPORTS:
            truncated = True
            events = events[:MAX_EVENT_LOOP_LAG_REPORTS]
            break
        if len(page) < _PAGE:
            break
        after = page[-1].sequence
    return event_loop_lag_summary(events, truncated=truncated)


def event_loop_lag_text(summary: object) -> str | None:
    """One sentence for people, or ``None`` when there is nothing to say."""

    if not isinstance(summary, dict):
        return None
    data = cast("dict[str, Any]", summary)
    reports = _count(data.get("reports"))
    if not reports:
        return None
    worst = _count(data.get("max_lag_ms"))
    text = (
        "The worker's event loop was late for "
        f"{_duration(_count(data.get('lagging_ms')) or 0)} while this session ran"
        + ("" if worst is None else f", up to {worst / 1000:.2f} s")
        + f" ({reports} report{'s' if reports != 1 else ''}"
        + f", {data.get('first_at')} to {data.get('last_at')})."
    )
    if data.get("truncated") is True:
        text += " Only the first reports were read."
    return text + (
        " Runtime work queued behind other work on the worker's CPU core; run sessions in "
        "more processes."
    )


def _duration(milliseconds: int) -> str:
    seconds = milliseconds / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(round(seconds), 60)
    return f"{minutes}m{rest:02d}s"


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None
