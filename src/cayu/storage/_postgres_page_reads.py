"""PostgreSQL page reads: one round trip per read for a whole page of sessions.

Each function answers exactly what the matching single-session read answers,
for every session of the page. Session IDs travel as one text array parameter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, LiteralString, cast

from cayu.events import Event, EventType
from cayu.sessions._page_reads import count_child_runs_from
from cayu.sessions.event_delivery import restore_persisted_event_authority
from cayu.sessions.records import EventRecord, Session
from cayu.sessions.summaries import SessionOutcome, session_outcome
from cayu.storage import _postgres_support as pg_support
from cayu.storage._postgres_session_queries import load_session_labels_batch

_LIFECYCLE_EVENT_TYPES = [str(EventType.SESSION_STARTED), str(EventType.SESSION_RESUMED)]
_TERMINAL_EVENT_TYPES = [
    str(EventType.SESSION_COMPLETED),
    str(EventType.SESSION_FAILED),
    str(EventType.SESSION_INTERRUPTED),
]


async def load_sessions(cur: Any, session_ids: Sequence[str]) -> dict[str, Session]:
    if not session_ids:
        return {}
    await cur.execute(
        cast(
            "LiteralString",
            f"SELECT {pg_support.SESSION_COLUMNS} FROM cayu_sessions WHERE id = ANY(%s::text[])",
        ),
        (list(session_ids),),
    )
    rows = await cur.fetchall()
    labels = await load_session_labels_batch(cur, [row[0] for row in rows])
    by_id = {
        row[0]: pg_support.session_from_row(row, labels=labels.get(row[0], {})) for row in rows
    }
    return {session_id: by_id[session_id] for session_id in session_ids if session_id in by_id}


async def load_checkpoints(cur: Any, session_ids: Sequence[str]) -> dict[str, Any]:
    """Raw stored checkpoint values; the store copies them as its single read does."""

    if not session_ids:
        return {}
    await cur.execute(
        "SELECT session_id, state FROM cayu_checkpoints WHERE session_id = ANY(%s::text[])",
        (list(session_ids),),
    )
    found = {row[0]: row[1] for row in await cur.fetchall()}
    return {session_id: found[session_id] for session_id in session_ids if session_id in found}


def _latest_after_lifecycle(type_filter: LiteralString) -> LiteralString:
    # The single read scopes terminal and retry events to the latest invocation:
    # only events after the most recent start or resume count.
    return (
        """
        LEFT JOIN LATERAL (
            SELECT sequence, event
            FROM cayu_events
            WHERE session_id = page.session_id
              AND """
        + type_filter
        + """
              AND sequence > COALESCE(
                  (
                      SELECT MAX(sequence)
                      FROM cayu_events
                      WHERE session_id = page.session_id
                        AND event_type = ANY(%(lifecycle)s::text[])
                  ),
                  0
              )
            ORDER BY sequence DESC
            LIMIT 1
        )"""
    )


def _event_record(sequence: Any, event: Any) -> EventRecord | None:
    if sequence is None:
        return None
    return EventRecord(sequence=sequence, event=Event(**pg_support._json_obj(event)))


async def summarize_outcomes(cur: Any, session_ids: Sequence[str]) -> dict[str, SessionOutcome]:
    sessions = await load_sessions(cur, session_ids)
    if not sessions:
        return {}
    await cur.execute(
        "SELECT page.session_id, terminal.sequence, terminal.event, retry.sequence, retry.event "
        "FROM unnest(%(ids)s::text[]) AS page(session_id)"
        + _latest_after_lifecycle("event_type = ANY(%(terminal)s::text[])")
        + " AS terminal ON TRUE"
        + _latest_after_lifecycle("event_type = %(retry)s")
        + " AS retry ON TRUE",
        {
            "ids": list(sessions),
            "terminal": _TERMINAL_EVENT_TYPES,
            "retry": str(EventType.MODEL_RETRY),
            "lifecycle": _LIFECYCLE_EVENT_TYPES,
        },
    )
    events = {row[0]: row[1:] for row in await cur.fetchall()}
    outcomes: dict[str, SessionOutcome] = {}
    for session_id, session in sessions.items():
        terminal_sequence, terminal, retry_sequence, retry = events.get(
            session_id, (None, None, None, None)
        )
        outcomes[session_id] = session_outcome(
            session,
            terminal_event=_event_record(terminal_sequence, terminal),
            latest_retry_event=_event_record(retry_sequence, retry),
        )
    return outcomes


async def query_latest_events(
    cur: Any, session_ids: Sequence[str], event_types: Sequence[EventType]
) -> dict[str, EventRecord]:
    if not session_ids:
        return {}
    await cur.execute(
        """
        SELECT latest.sequence,
               latest.event,
               latest.input_contract_runtime_owned,
               latest.file_attachment_attestations_runtime_owned
        FROM unnest(%s::text[]) AS page(session_id)
        CROSS JOIN LATERAL (
            SELECT cayu_events.sequence,
                   cayu_events.event,
                   cayu_events.input_contract_runtime_owned,
                   cayu_events.file_attachment_attestations_runtime_owned
            FROM cayu_events
            JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
            WHERE cayu_events.session_id = page.session_id
              AND cayu_events.event_type = ANY(%s::text[])
            ORDER BY cayu_events.sequence DESC
            LIMIT 1
        ) AS latest
        """,
        (list(session_ids), [str(event_type) for event_type in event_types]),
    )
    found: dict[str, EventRecord] = {}
    for row in await cur.fetchall():
        record = EventRecord(
            sequence=row[0],
            event=restore_persisted_event_authority(
                Event(**pg_support._json_obj(row[1])),
                input_contract_runtime_owned=row[2],
                file_attachment_attestations_runtime_owned=row[3],
            ),
        )
        found[record.event.session_id] = record
    return {session_id: found[session_id] for session_id in session_ids if session_id in found}


async def load_session_operation_rows(
    cur: Any,
    session_ids: Sequence[str],
    idempotency_key: str,
    checkpoint_root_key: str,
    scalar_max_chars: int,
) -> list[tuple[Any, ...]]:
    if not session_ids:
        return []
    await cur.execute(
        cast(
            "LiteralString",
            f"""
            SELECT
                cayu_sessions.id,
                cayu_session_operations.record,
                jsonb_typeof(cayu_checkpoints.state -> '{checkpoint_root_key}'),
                left(
                    cayu_checkpoints.state ->> '{checkpoint_root_key}',
                    {scalar_max_chars + 1}
                )
            FROM cayu_sessions
            LEFT JOIN cayu_session_operations
                ON cayu_session_operations.session_id = cayu_sessions.id
                AND cayu_session_operations.idempotency_key = %s
            LEFT JOIN cayu_checkpoints
                ON cayu_checkpoints.session_id = cayu_sessions.id
            WHERE cayu_sessions.id = ANY(%s::text[])
            """,
        ),
        (idempotency_key, list(session_ids)),
    )
    return list(await cur.fetchall())


async def count_child_runs(cur: Any, parents: Sequence[Session]) -> dict[str, int]:
    if not parents:
        return {}
    await cur.execute(
        cast(
            "LiteralString",
            f"SELECT {pg_support.SESSION_COLUMNS} FROM cayu_sessions "
            "WHERE parent_session_id = ANY(%s::text[])",
        ),
        ([parent.id for parent in parents],),
    )
    rows = await cur.fetchall()
    # Labels play no part in a child's run lineage, so they are not read.
    return count_child_runs_from(
        parents, (pg_support.session_from_row(row, labels={}) for row in rows)
    )
