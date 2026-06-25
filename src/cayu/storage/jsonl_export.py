"""Backend-agnostic JSONL export for Cayu storage.

JSONL (one JSON object per line) is Cayu's portable export/replay/backup
format (ADR 0001 Phase 3). These helpers read only through the public
``SessionStore`` / ``TaskStore`` contract methods, so they work identically
across the in-memory, SQLite, and Postgres backends.

Each line is ``json.dumps(obj, ensure_ascii=False)`` plus a newline. Ordering
is deterministic: records are exported oldest-first by creation time.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from cayu.runtime.sessions import SessionOrder, SessionQuery, SessionStore
from cayu.runtime.tasks import TaskOrder, TaskQuery, TaskStore

_EXPORT_PAGE_SIZE = 1000


class _TextStream(Protocol):
    """Minimal text sink: anything with a ``write(str)`` method (e.g. a file)."""

    def write(self, data: str, /) -> Any: ...


def _write_line(stream: _TextStream, obj: dict[str, Any]) -> None:
    stream.write(json.dumps(obj, ensure_ascii=False) + "\n")


async def export_sessions(store: SessionStore, *, stream: _TextStream) -> int:
    """Export every session in ``store`` as JSONL, one session per line.

    Each line is a ``{"type": "session", ...}`` object bundling the session
    record with its events, transcript, and latest checkpoint::

        {"type": "session", "session": {...}, "events": [...],
         "transcript": [...], "checkpoint": {...} | null}

    Sessions are emitted oldest-first by creation time. Returns the number of
    sessions written.
    """
    count = 0
    offset = 0
    while True:
        page = (
            await store.list_sessions(
                SessionQuery(
                    limit=_EXPORT_PAGE_SIZE,
                    offset=offset,
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
        ).sessions
        if not page:
            return count
        for session in page:
            events = await store.load_events(session.id)
            transcript = await store.load_transcript(session.id)
            checkpoint = await store.load_checkpoint(session.id)
            _write_line(
                stream,
                {
                    "type": "session",
                    "session": session.model_dump(mode="json"),
                    "events": [event.model_dump(mode="json") for event in events],
                    "transcript": [message.model_dump(mode="json") for message in transcript],
                    "checkpoint": checkpoint,
                },
            )
            count += 1
        if len(page) < _EXPORT_PAGE_SIZE:
            return count
        offset += _EXPORT_PAGE_SIZE


async def export_tasks(store: TaskStore, *, stream: _TextStream) -> int:
    """Export every task in ``store`` as JSONL, one task per line.

    Each line is a ``{"type": "task", "task": {...}}`` object. Tasks are
    emitted oldest-first by creation time. Returns the number of tasks written.
    """
    count = 0
    offset = 0
    while True:
        page = await store.list_tasks(
            TaskQuery(
                limit=_EXPORT_PAGE_SIZE,
                offset=offset,
                order_by=TaskOrder.CREATED_AT_ASC,
            )
        )
        if not page:
            return count
        for task in page:
            _write_line(
                stream,
                {
                    "type": "task",
                    "task": task.model_dump(mode="json"),
                },
            )
            count += 1
        if len(page) < _EXPORT_PAGE_SIZE:
            return count
        offset += _EXPORT_PAGE_SIZE
