"""Backend-agnostic JSONL export/import for Cayu storage.

JSONL (one JSON object per line) is Cayu's portable export/replay/backup
format (ADR 0001 Phase 3). These helpers read only through the public
``SessionStore`` / ``TaskStore`` contract methods, so they work identically
across the in-memory, SQLite, and Postgres backends.

Each record is validated against Cayu's portable durable-value contract before
``json.dumps(..., ensure_ascii=False, allow_nan=False)`` writes the complete
line. Ordering is deterministic: records are exported oldest-first by creation
time.

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
from itertools import pairwise
from typing import Any, Protocol

from cayu._validation import (
    DurableValueError,
    copy_durable_json_object,
    durable_json_object_from_pairs,
    parse_durable_json_integer_literal,
    reject_nonportable_json_constant,
    require_durable_text,
)
from cayu.core import Event, Message
from cayu.runtime.checkpoints import decode_runtime_checkpoint
from cayu.runtime.sessions import (
    DeferredInteractionInput,
    Session,
    SessionOrder,
    SessionQuery,
    SessionStore,
    TranscriptQuery,
    TranscriptRecord,
    restore_persisted_event_authority,
)
from cayu.runtime.tasks import Task, TaskOrder, TaskQuery, TaskStore

_EXPORT_PAGE_SIZE = 1000
_SESSION_RECORD_FIELDS = frozenset(
    {
        "type",
        "session",
        "events",
        "transcript_records",
        "checkpoint",
        "deferred_interaction_input",
    }
)


class _TextStream(Protocol):
    """Minimal text sink: anything with a ``write(str)`` method (e.g. a file)."""

    def write(self, data: str, /) -> Any: ...


def _write_line(stream: _TextStream, obj: dict[str, Any]) -> None:
    portable = copy_durable_json_object(obj, "JSONL record")
    stream.write(json.dumps(portable, ensure_ascii=False, allow_nan=False) + "\n")


async def export_sessions(store: SessionStore, *, stream: _TextStream) -> int:
    """Export every session in ``store`` as JSONL, one session per line.

    Each line is a ``{"type": "session", ...}`` object bundling the session
    record with its events, attributed transcript records, and latest
    checkpoint::

        {"type": "session", "session": {...}, "events": [...],
         "transcript_records": [...],
         "checkpoint": {...} | null,
         "deferred_interaction_input": {...} | null}

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
            transcript_records = await _load_transcript_records(store, session.id)
            checkpoint = decode_runtime_checkpoint(
                await store.load_checkpoint(session.id),
                session_id=session.id,
            )
            deferred_interaction_input = await store.load_deferred_interaction_input(session.id)
            _write_line(
                stream,
                {
                    "type": "session",
                    "session": session.model_dump(mode="json"),
                    "events": [event.model_dump(mode="json") for event in events],
                    "transcript_records": [
                        record.model_dump(mode="json") for record in transcript_records
                    ],
                    "checkpoint": checkpoint,
                    "deferred_interaction_input": (
                        None
                        if deferred_interaction_input is None
                        else deferred_interaction_input.model_dump(mode="json")
                    ),
                },
            )
            count += 1
        cursor = result.next_cursor
        if cursor is None:
            return count


async def _load_transcript_records(
    store: SessionStore,
    session_id: str,
) -> list[TranscriptRecord]:
    records: list[TranscriptRecord] = []
    offset = 0
    while True:
        page = await store.query_transcript(
            TranscriptQuery(
                session_id=session_id,
                offset=offset,
                limit=_EXPORT_PAGE_SIZE,
            )
        )
        if not page.records:
            return records
        records.extend(page.records)
        offset += len(page.records)
        if offset >= page.total_records:
            return records


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
    latest checkpoint, and any private deferred interaction input, all
    validated back into their typed models.
    """

    session: Session
    events: list[Event]
    transcript: list[Message]
    transcript_records: list[TranscriptRecord]
    checkpoint: dict[str, Any] | None
    deferred_interaction_input: DeferredInteractionInput | None


def _iter_json_lines(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Parse non-blank JSONL lines into objects, rejecting non-object lines."""
    for raw in lines:
        if type(raw) is not str:
            raise DurableValueError("invalid_text_type", "JSONL record")
        raw = require_durable_text(raw, "JSONL record")
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(
                stripped,
                parse_constant=_reject_nonportable_json_constant,
                parse_int=_parse_portable_json_integer,
                object_pairs_hook=_reject_duplicate_json_object_keys,
            )
        except RecursionError:
            raise DurableValueError("nesting_too_deep", "JSONL record") from None
        yield copy_durable_json_object(obj, "JSONL record")


def _reject_nonportable_json_constant(value: str) -> None:
    reject_nonportable_json_constant(value, "JSONL record")


def _parse_portable_json_integer(value: str) -> int:
    return parse_durable_json_integer_literal(value, "JSONL record")


def _reject_duplicate_json_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    return durable_json_object_from_pairs(pairs, "JSONL record")


def import_sessions(lines: Iterable[str]) -> Iterator[ImportedSession]:
    """Parse ``{"type": "session", ...}`` JSONL lines into typed records.

    ``lines`` is any iterable of text lines (e.g. an open file, which iterates
    by line). Blank lines are skipped. The result is a generator so a large
    export can be streamed without loading the whole file into memory. Raises
    ``ValueError`` on a line whose ``type`` is not ``"session"`` and lets
    ``json.JSONDecodeError`` / pydantic ``ValidationError`` surface for
    malformed content.

    Import is an explicit trusted-backup boundary. Runtime authority retained
    by a built-in store is restored for Cayu's fixed allowlist of durable event
    fields so appending the imported events to another store preserves resume
    semantics. Do not restore JSONL from an untrusted source.
    """
    for obj in _iter_json_lines(lines):
        record_type = obj.get("type")
        if record_type != "session":
            raise ValueError(f"Expected a session record, got type={record_type!r}.")
        for required_field in _SESSION_RECORD_FIELDS:
            if required_field not in obj:
                raise ValueError(f"Session record is missing {required_field}.")
        if obj.keys() != _SESSION_RECORD_FIELDS:
            raise ValueError("Session record contains unsupported fields.")
        session = Session.model_validate(obj["session"])
        checkpoint = decode_runtime_checkpoint(
            obj["checkpoint"],
            session_id=session.id,
        )
        raw_transcript_records = obj["transcript_records"]
        if type(raw_transcript_records) is not list:
            raise ValueError("Session transcript_records must be a list.")
        transcript_records = [
            TranscriptRecord.model_validate(record) for record in raw_transcript_records
        ]
        transcript_indices = [record.index for record in transcript_records]
        if any(current <= previous for previous, current in pairwise(transcript_indices)):
            raise ValueError("Session transcript record indices must be strictly increasing.")
        transcript = [record.message for record in transcript_records]
        yield ImportedSession(
            session=session,
            events=[
                restore_persisted_event_authority(
                    Event.model_validate(event),
                    input_contract_runtime_owned=True,
                )
                for event in obj["events"]
            ],
            transcript=transcript,
            transcript_records=transcript_records,
            checkpoint=checkpoint,
            deferred_interaction_input=(
                None
                if obj["deferred_interaction_input"] is None
                else DeferredInteractionInput.model_validate(obj["deferred_interaction_input"])
            ),
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
