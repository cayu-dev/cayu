"""Rerun full durable validation over the events a session store holds.

Reads decode stored events through ``event_from_trusted_store_record``, which
skips the payload's node-count, nesting and integer-range checks because the
store's ingestion enforced them. This check reruns the validating decode over
every stored row, so an operator can confirm that no row was edited outside
Cayu. ``cayu storage verify-events`` drives it a page at a time.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from cayu.events import Event
from cayu.sessions.event_delivery import restore_persisted_event_authority

#: Rows one verification page reads, and the most a caller may ask for.
VERIFY_EVENTS_PAGE_LIMIT = 1000
_ERROR_CHARS = 300


@dataclass(frozen=True, slots=True)
class StoredEventFailure:
    """One stored event the validating decode rejects. Holds no payload content."""

    sequence: int
    session_id: str | None
    event_id: str | None
    event_type: str | None
    error_type: str
    error: str


@dataclass(frozen=True, slots=True)
class StoredEventVerificationPage:
    """One page of stored events checked in sequence order.

    ``through_sequence`` is the last sequence checked, or None for an empty
    page; pass it as ``after_sequence`` to continue. ``complete`` is true when
    no stored event follows the page.
    """

    checked: int
    through_sequence: int | None
    complete: bool
    failures: tuple[StoredEventFailure, ...] = field(default=())


def _limit(limit: int) -> int:
    if type(limit) is not int or not 1 <= limit <= VERIFY_EVENTS_PAGE_LIMIT:
        raise ValueError(f"limit must be between 1 and {VERIFY_EVENTS_PAGE_LIMIT}.")
    return limit


def _after(after_sequence: int) -> int:
    if type(after_sequence) is not int or after_sequence < 0:
        raise ValueError("after_sequence must be a non-negative integer.")
    return after_sequence


def _failure(
    sequence: int, session_id: Any, event_id: Any, event_type: Any, exc: Exception
) -> StoredEventFailure:
    def text(value: Any) -> str | None:
        return value if type(value) is str else None

    return StoredEventFailure(
        sequence=sequence,
        session_id=text(session_id),
        event_id=text(event_id),
        event_type=text(event_type),
        error_type=type(exc).__name__,
        error=str(exc)[:_ERROR_CHARS],
    )


def verify_sqlite_events(
    connection: sqlite3.Connection,
    *,
    after_sequence: int = 0,
    limit: int = VERIFY_EVENTS_PAGE_LIMIT,
) -> StoredEventVerificationPage:
    """Validate one page of SQLite ``cayu_events`` rows after ``after_sequence``."""

    from cayu.storage import _sqlite_records as sqlite_records

    after_sequence, limit = _after(after_sequence), _limit(limit)
    columns = ", ".join(sqlite_records.EVENT_COLUMN_NAMES)
    rows = connection.execute(
        f"SELECT sequence, {columns} FROM cayu_events WHERE sequence > ? "
        "ORDER BY sequence ASC LIMIT ?",
        (after_sequence, limit + 1),
    ).fetchall()
    failures: list[StoredEventFailure] = []
    for row in rows[:limit]:
        try:
            sqlite_records.validating_event_from_row(row)
        except (ValueError, TypeError) as exc:
            failures.append(
                _failure(
                    row["sequence"], row["session_id"], row["event_id"], row["event_type"], exc
                )
            )
    checked = rows[:limit]
    return StoredEventVerificationPage(
        checked=len(checked),
        through_sequence=checked[-1]["sequence"] if checked else None,
        complete=len(rows) <= limit,
        failures=tuple(failures),
    )


async def verify_postgres_events(
    cur: Any, *, after_sequence: int = 0, limit: int = VERIFY_EVENTS_PAGE_LIMIT
) -> StoredEventVerificationPage:
    """Validate one page of PostgreSQL ``cayu_events`` documents after ``after_sequence``."""

    from cayu.storage import _postgres_support as pg_support

    after_sequence, limit = _after(after_sequence), _limit(limit)
    await cur.execute(
        "SELECT sequence, event, input_contract_runtime_owned, "
        "file_attachment_attestations_runtime_owned, session_id, event_id, event_type "
        "FROM cayu_events WHERE sequence > %s ORDER BY sequence ASC LIMIT %s",
        (after_sequence, limit + 1),
    )
    rows = await cur.fetchall()
    failures: list[StoredEventFailure] = []
    for row in rows[:limit]:
        try:
            restore_persisted_event_authority(
                Event(**pg_support._json_obj(row[1])),
                input_contract_runtime_owned=row[2],
                file_attachment_attestations_runtime_owned=row[3],
            )
        except (ValueError, TypeError) as exc:
            failures.append(_failure(row[0], row[4], row[5], row[6], exc))
    checked = rows[:limit]
    return StoredEventVerificationPage(
        checked=len(checked),
        through_sequence=checked[-1][0] if checked else None,
        complete=len(rows) <= limit,
        failures=tuple(failures),
    )
