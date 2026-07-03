"""Backend-agnostic JSONL export/import for Cayu storage.

JSONL (one JSON object per line) is Cayu's portable export/replay/backup
format (ADR 0001 Phase 3). These helpers read only through the public
``SessionStore`` / ``TaskStore`` contract methods, so they work identically
across the in-memory, SQLite, and Postgres backends.

Each line is ``json.dumps(obj, ensure_ascii=False)`` plus a newline. Ordering
is deterministic: records are exported oldest-first by creation time.

Session export pages with a **keyset cursor** rather than a live ``OFFSET``.
An offset walk over a store that is being written concurrently silently skips
records: when a session ahead of the cursor is deleted mid-export the offset
window shifts and one live session is never emitted. Keyset paging anchors each
page to the ``(created_at, id)`` of the last row it emitted, so inserts and
deletes elsewhere in the store cannot perturb the page boundaries.

The ``import_*`` helpers are the inverse of the ``export_*`` helpers: they parse
the JSONL back into validated typed records (``Session`` / ``Event`` /
``Message`` / ``Task``) for replay, inspection, or restore into another store.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from cayu.core import Event, Message
from cayu.runtime.sessions import (
    Session,
    SessionOrder,
    SessionQuery,
    SessionStore,
)
from cayu.runtime.tasks import Task, TaskOrder, TaskQuery, TaskStore

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

    Sessions are emitted oldest-first by creation time. Paging uses a keyset
    cursor (see the module docstring), so concurrent inserts and deletes cannot
    make the walk skip or duplicate a session. Returns the number of sessions
    written.
    """
    count = 0
    cursor: str | None = None
    while True:
        result = await store.list_sessions(
            SessionQuery(
                limit=_EXPORT_PAGE_SIZE,
                cursor=cursor,
                order_by=SessionOrder.CREATED_AT_ASC,
            )
        )
        for session in result.sessions:
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
        cursor = result.next_cursor
        if cursor is None:
            return count


async def export_tasks(store: TaskStore, *, stream: _TextStream) -> int:
    """Export every task in ``store`` as JSONL, one task per line.

    Each line is a ``{"type": "task", "task": {...}}`` object. Tasks are
    emitted oldest-first by creation time. Returns the number of tasks written.

    The ``TaskStore`` contract exposes only offset paging (no keyset cursor), so
    this walk pages by offset. It de-duplicates by task id so a concurrent
    insert that shifts the offset window cannot emit the same task on two pages;
    a concurrent delete may still cause a task ahead of the cursor to be missed.
    """
    count = 0
    offset = 0
    seen: set[str] = set()
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
            if task.id in seen:
                continue
            seen.add(task.id)
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


@dataclass(frozen=True)
class ImportedSession:
    """A session record and its nested state, parsed from a JSONL export line.

    This is the inverse of one ``{"type": "session", ...}`` line produced by
    :func:`export_sessions`: the ``Session`` plus its events, transcript, and
    latest checkpoint, all validated back into their typed models.
    """

    session: Session
    events: list[Event]
    transcript: list[Message]
    checkpoint: dict[str, Any] | None


def _iter_json_lines(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Parse non-blank JSONL lines into objects, rejecting non-object lines."""
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        obj = json.loads(stripped)
        if not isinstance(obj, dict):
            raise ValueError("Each JSONL line must be a JSON object.")
        yield obj


def import_sessions(lines: Iterable[str]) -> Iterator[ImportedSession]:
    """Parse ``{"type": "session", ...}`` JSONL lines into typed records.

    ``lines`` is any iterable of text lines (e.g. an open file, which iterates
    by line). Blank lines are skipped. The result is a generator so a large
    export can be streamed without loading the whole file into memory. Raises
    ``ValueError`` on a line whose ``type`` is not ``"session"`` and lets
    ``json.JSONDecodeError`` / pydantic ``ValidationError`` surface for
    malformed content.
    """
    for obj in _iter_json_lines(lines):
        record_type = obj.get("type")
        if record_type != "session":
            raise ValueError(f"Expected a session record, got type={record_type!r}.")
        yield ImportedSession(
            session=Session.model_validate(obj["session"]),
            events=[Event.model_validate(event) for event in obj.get("events", [])],
            transcript=[Message.model_validate(message) for message in obj.get("transcript", [])],
            checkpoint=obj.get("checkpoint"),
        )


def import_tasks(lines: Iterable[str]) -> Iterator[Task]:
    """Parse ``{"type": "task", ...}`` JSONL lines into ``Task`` records.

    ``lines`` is any iterable of text lines (e.g. an open file). Blank lines are
    skipped. Raises ``ValueError`` on a line whose ``type`` is not ``"task"``.
    """
    for obj in _iter_json_lines(lines):
        record_type = obj.get("type")
        if record_type != "task":
            raise ValueError(f"Expected a task record, got type={record_type!r}.")
        yield Task.model_validate(obj["task"])
