"""SQLite page reads: one query per read for a whole page of sessions.

Each function answers exactly what the matching single-session read answers,
for every session of the page. Session IDs travel as one JSON array parameter.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from typing import Any

from cayu.events import EventType
from cayu.messages import MessageRole
from cayu.sessions._page_reads import count_child_runs_from
from cayu.sessions.records import EventRecord, Session, TranscriptRecord
from cayu.sessions.summaries import SessionOutcome, session_outcome
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage import _sqlite_transcript_content as transcript_content

_SESSION_COLUMNS = """
    id, instance_id, agent_name, provider_name, model, parent_session_id,
    causal_budget_id, runtime_name, runtime_version, environment_name,
    status, created_at, updated_at, last_activity_at, run_epoch,
    invocation_json, metadata_json
"""
_PAGE_IDS = "SELECT CAST(value AS TEXT) FROM json_each(?)"
_EVENT_COLUMNS = ", ".join(f"cayu_events.{name}" for name in sqlite_records.EVENT_COLUMN_NAMES)


def _ids(session_ids: Sequence[str]) -> str:
    return sqlite_records.json_dumps(list(session_ids))


def load_sessions(connection: sqlite3.Connection, session_ids: Sequence[str]) -> dict[str, Session]:
    if not session_ids:
        return {}
    rows = connection.execute(
        f"SELECT {_SESSION_COLUMNS} FROM cayu_sessions WHERE id IN ({_PAGE_IDS})",
        (_ids(session_ids),),
    ).fetchall()
    labels = sqlite_records.load_session_labels_batch(connection, [row["id"] for row in rows])
    by_id = {
        row["id"]: sqlite_records.session_from_row(row, labels=labels.get(row["id"], {}))
        for row in rows
    }
    return {session_id: by_id[session_id] for session_id in session_ids if session_id in by_id}


def load_checkpoint_json(
    connection: sqlite3.Connection, session_ids: Sequence[str]
) -> dict[str, str]:
    if not session_ids:
        return {}
    rows = connection.execute(
        f"SELECT session_id, state_json FROM cayu_checkpoints WHERE session_id IN ({_PAGE_IDS})",
        (_ids(session_ids),),
    ).fetchall()
    found = {row["session_id"]: row["state_json"] for row in rows}
    return {session_id: found[session_id] for session_id in session_ids if session_id in found}


def _latest_after_lifecycle(type_filter: str) -> str:
    # The single read scopes terminal and retry events to the latest invocation:
    # only events after the most recent start or resume count.
    return f"""
        SELECT page.value AS page_session_id, cayu_events.sequence, {_EVENT_COLUMNS}
        FROM json_each(?) AS page
        JOIN cayu_events ON cayu_events.sequence = (
            SELECT MAX(latest.sequence)
            FROM cayu_events AS latest
            WHERE latest.session_id = CAST(page.value AS TEXT)
              AND latest.event_type {type_filter}
              AND latest.sequence > COALESCE(
                  (
                      SELECT MAX(lifecycle.sequence)
                      FROM cayu_events AS lifecycle
                      WHERE lifecycle.session_id = CAST(page.value AS TEXT)
                        AND lifecycle.event_type IN ('session.started', 'session.resumed')
                  ),
                  0
              )
        )
    """


def summarize_outcomes(
    connection: sqlite3.Connection, session_ids: Sequence[str]
) -> dict[str, SessionOutcome]:
    sessions = load_sessions(connection, session_ids)
    if not sessions:
        return {}
    page = _ids(list(sessions))
    terminal = {
        row["page_session_id"]: sqlite_records.event_record_from_row(row)
        for row in connection.execute(
            _latest_after_lifecycle(
                "IN ('session.completed', 'session.failed', 'session.interrupted')"
            ),
            (page,),
        ).fetchall()
    }
    retry = {
        row["page_session_id"]: sqlite_records.event_record_from_row(row)
        for row in connection.execute(
            _latest_after_lifecycle("= 'model.retry'"), (page,)
        ).fetchall()
    }
    return {
        session_id: session_outcome(
            session,
            terminal_event=terminal.get(session_id),
            latest_retry_event=retry.get(session_id),
        )
        for session_id, session in sessions.items()
    }


def query_latest_events(
    connection: sqlite3.Connection,
    session_ids: Sequence[str],
    event_types: Sequence[EventType],
) -> dict[str, EventRecord]:
    if not session_ids:
        return {}
    types = sqlite_records.json_dumps([str(event_type) for event_type in event_types])
    rows = connection.execute(
        f"""
        SELECT cayu_events.sequence, {_EVENT_COLUMNS}
        FROM json_each(?) AS page
        JOIN cayu_events ON cayu_events.sequence = (
            SELECT MAX(latest.sequence)
            FROM cayu_events AS latest
            WHERE latest.session_id = CAST(page.value AS TEXT)
              AND latest.event_type IN (SELECT CAST(value AS TEXT) FROM json_each(?))
        )
        JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
        """,
        (_ids(session_ids), types),
    ).fetchall()
    found: dict[str, EventRecord] = {}
    for row in rows:
        record = sqlite_records.event_record_from_row(row)
        assert record is not None
        found[record.event.session_id] = record
    return {session_id: found[session_id] for session_id in session_ids if session_id in found}


def query_first_transcript_records(
    connection: sqlite3.Connection, session_ids: Sequence[str], role: MessageRole
) -> dict[str, TranscriptRecord]:
    if not session_ids:
        return {}

    def read(connection: sqlite3.Connection) -> dict[str, TranscriptRecord]:
        rows = connection.execute(
            """
            SELECT
                transcript.session_id,
                transcript.session_order - 1 AS transcript_index,
                transcript.interaction_id,
                transcript.message_json
            FROM json_each(?) AS page
            JOIN cayu_transcript_messages AS transcript ON transcript.sequence = (
                SELECT earliest.sequence
                FROM cayu_transcript_messages AS earliest
                WHERE earliest.session_id = CAST(page.value AS TEXT)
                  AND earliest.role = ?
                ORDER BY earliest.session_order ASC
                LIMIT 1
            )
            JOIN cayu_sessions ON cayu_sessions.id = transcript.session_id
            """,
            (_ids(session_ids), str(role)),
        ).fetchall()
        # Tool results a message references live in its session's tool events.
        messages = transcript_content.messages_by_session(
            connection, [(row["session_id"], row["message_json"]) for row in rows]
        )
        found = {
            row["session_id"]: TranscriptRecord(
                index=row["transcript_index"],
                interaction_id=row["interaction_id"],
                message=message,
            )
            for row, message in zip(rows, messages, strict=True)
        }
        return {session_id: found[session_id] for session_id in session_ids if session_id in found}

    # The rows and the tool events their results live in come from one snapshot.
    return transcript_content.read_snapshot(connection, read)


def load_session_operation_rows(
    connection: sqlite3.Connection,
    session_ids: Sequence[str],
    idempotency_key: str,
    checkpoint_root_key: str,
    scalar_max_chars: int,
) -> list[sqlite3.Row]:
    if not session_ids:
        return []
    path = f"$.{checkpoint_root_key}"
    return connection.execute(
        f"""
        SELECT
            cayu_sessions.id AS session_id,
            cayu_session_operations.record_json,
            json_type(cayu_checkpoints.state_json, '{path}') AS checkpoint_root_field_type,
            CASE
                WHEN json_type(cayu_checkpoints.state_json, '{path}') = 'integer'
                THEN substr(
                    CAST(json_extract(cayu_checkpoints.state_json, '{path}') AS TEXT),
                    1,
                    {scalar_max_chars + 1}
                )
            END AS checkpoint_root_field_scalar
        FROM cayu_sessions
        LEFT JOIN cayu_session_operations
            ON cayu_session_operations.session_id = cayu_sessions.id
            AND cayu_session_operations.idempotency_key = ?
        LEFT JOIN cayu_checkpoints
            ON cayu_checkpoints.session_id = cayu_sessions.id
        WHERE cayu_sessions.id IN ({_PAGE_IDS})
        """,
        (idempotency_key, _ids(session_ids)),
    ).fetchall()


def operation_record(row: sqlite3.Row) -> dict[str, Any] | None:
    value = row["record_json"]
    return None if value is None else json.loads(value)


def count_child_runs(connection: sqlite3.Connection, parents: Sequence[Session]) -> dict[str, int]:
    if not parents:
        return {}
    rows = connection.execute(
        f"""
        SELECT {_SESSION_COLUMNS}
        FROM cayu_sessions
        WHERE parent_session_id IN ({_PAGE_IDS})
        """,
        (_ids([parent.id for parent in parents]),),
    ).fetchall()
    # Labels play no part in a child's run lineage, so they are not read.
    return count_child_runs_from(
        parents, (sqlite_records.session_from_row(row, labels={}) for row in rows)
    )
