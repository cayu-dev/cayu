"""PostgreSQL page reads: one round trip per read for a whole page of sessions.

Each function answers exactly what the matching single-session read answers,
for every session of the page. Session IDs travel as one text array parameter.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, LiteralString, cast

from cayu._validation import DurableValueError
from cayu.events import Event, EventType
from cayu.messages import MessageRole
from cayu.sessions._page_reads import count_child_runs_from
from cayu.sessions.event_delivery import restore_persisted_event_authority
from cayu.sessions.records import (
    EventRecord,
    Session,
    SessionLineage,
    SessionStatus,
    TranscriptRecord,
    stored_session_lineage,
)
from cayu.sessions.summaries import (
    SessionEventPositions,
    SessionOutcome,
    SessionStepCounts,
    session_status_outcome,
)
from cayu.storage import _postgres_support as pg_support
from cayu.storage import _postgres_transcript_content as transcript_content
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
    # A page's rows are decoded and validated off the event loop.
    by_id = await asyncio.to_thread(_sessions_from_rows, rows, labels)
    return {session_id: by_id[session_id] for session_id in session_ids if session_id in by_id}


def _sessions_from_rows(rows: list[Any], labels: dict[str, dict[str, str]]) -> dict[str, Session]:
    return {row[0]: pg_support.session_from_row(row, labels=labels.get(row[0], {})) for row in rows}


async def load_checkpoints(cur: Any, session_ids: Sequence[str]) -> dict[str, Any]:
    """Raw stored checkpoint values; the store copies them as its single read does."""

    if not session_ids:
        return {}
    await cur.execute(
        "SELECT checkpoint.session_id, checkpoint.state, ledger.record_sha256, ledger.ledger "
        "FROM cayu_checkpoints AS checkpoint "
        "LEFT JOIN cayu_invocation_receipt_ledgers AS ledger "
        "ON ledger.session_id = checkpoint.session_id AND ledger.record_sha256 = "
        "checkpoint.state->'invocation_lifecycle_receipt'->>'record_sha256' "
        "WHERE checkpoint.session_id = ANY(%s::text[])",
        (list(session_ids),),
    )
    found = {row[0]: (row[1], row[2], row[3]) for row in await cur.fetchall()}
    return {session_id: found[session_id] for session_id in session_ids if session_id in found}


async def load_session_instance_ids(cur: Any, session_ids: Sequence[str]) -> dict[str, str]:
    if not session_ids:
        return {}
    await cur.execute(
        "SELECT id, instance_id FROM cayu_sessions WHERE id = ANY(%s::text[])",
        (list(session_ids),),
    )
    found = {row[0]: row[1] for row in await cur.fetchall()}
    return {session_id: found[session_id] for session_id in session_ids if session_id in found}


async def load_session_lineages(cur: Any, session_ids: Sequence[str]) -> dict[str, SessionLineage]:
    if not session_ids:
        return {}
    await cur.execute(
        "SELECT id, instance_id, parent_session_id, invocation->>'source', "
        "metadata ? 'cayu:run_lineage', metadata->'cayu:run_lineage' "
        "FROM cayu_sessions WHERE id = ANY(%s::text[])",
        (list(session_ids),),
    )
    found = {
        row[0]: stored_session_lineage(row[0], row[1], row[2], row[3], row[4], row[5])
        for row in await cur.fetchall()
    }
    return {session_id: found[session_id] for session_id in session_ids if session_id in found}


async def load_checkpoint_fields(
    cur: Any, session_ids: Sequence[str], keys: Sequence[str]
) -> dict[str, Any]:
    """Raw stored values of the given root fields; the store copies them."""

    if not session_ids:
        return {}
    # A receipt ledger field that refers to its own row is joined in the same
    # statement, so both come from one snapshot.
    await cur.execute(
        """
        SELECT checkpoint.session_id,
               jsonb_typeof(checkpoint.state),
               CASE WHEN jsonb_typeof(checkpoint.state) = 'object' THEN
                   (SELECT jsonb_object_agg(field.key, field.value)
                    FROM jsonb_each(checkpoint.state) AS field
                    WHERE field.key = ANY(%(keys)s::text[]))
               END,
               ledger.record_sha256,
               ledger.ledger
        FROM cayu_checkpoints AS checkpoint
        LEFT JOIN cayu_invocation_receipt_ledgers AS ledger
            ON 'invocation_lifecycle_receipt' = ANY(%(keys)s::text[])
            AND ledger.session_id = checkpoint.session_id
            AND ledger.record_sha256 =
                checkpoint.state->'invocation_lifecycle_receipt'->>'record_sha256'
        WHERE checkpoint.session_id = ANY(%(ids)s::text[])
        """,
        {"keys": list(keys), "ids": list(session_ids)},
    )
    found: dict[str, Any] = {}
    for session_id, root_type, fields, ledger_sha256, ledger in await cur.fetchall():
        # A checkpoint is an object, as the whole-checkpoint read requires.
        if root_type != "object":
            raise DurableValueError("invalid_json_object", "checkpoint")
        found[session_id] = ({} if fields is None else fields, ledger_sha256, ledger)
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


async def count_session_steps(cur: Any, session_ids: Sequence[str]) -> dict[str, SessionStepCounts]:
    if not session_ids:
        return {}
    # The event types are literals so the partial step-count index applies.
    await cur.execute(
        """
        SELECT cayu_sessions.id,
               COUNT(cayu_events.session_id)
                   FILTER (WHERE cayu_events.event_type = 'model.completed'),
               COUNT(cayu_events.session_id)
                   FILTER (WHERE cayu_events.event_type = 'tool.call.started')
        FROM cayu_sessions
        LEFT JOIN cayu_events
            ON cayu_events.session_id = cayu_sessions.id
            AND cayu_events.event_type IN ('model.completed', 'tool.call.started')
        WHERE cayu_sessions.id = ANY(%s::text[])
        GROUP BY cayu_sessions.id
        """,
        (list(session_ids),),
    )
    found = {
        row[0]: SessionStepCounts(model_steps=int(row[1]), tool_calls=int(row[2]))
        for row in await cur.fetchall()
    }
    return {session_id: found[session_id] for session_id in session_ids if session_id in found}


async def summarize_event_positions(
    cur: Any, session_ids: Sequence[str], event_types: Sequence[EventType]
) -> dict[str, SessionEventPositions]:
    if not session_ids:
        return {}
    await cur.execute(
        """
        SELECT cayu_sessions.id, COUNT(cayu_events.sequence), COALESCE(MAX(cayu_events.sequence), 0)
        FROM cayu_sessions
        LEFT JOIN cayu_events
            ON cayu_events.session_id = cayu_sessions.id
            AND cayu_events.event_type = ANY(%s::text[])
        WHERE cayu_sessions.id = ANY(%s::text[])
        GROUP BY cayu_sessions.id
        """,
        ([str(event_type) for event_type in event_types], list(session_ids)),
    )
    found = {
        row[0]: SessionEventPositions(count=int(row[1]), last_sequence=int(row[2]))
        for row in await cur.fetchall()
    }
    return {session_id: found[session_id] for session_id in session_ids if session_id in found}


async def summarize_outcomes(cur: Any, session_ids: Sequence[str]) -> dict[str, SessionOutcome]:
    if not session_ids:
        return {}
    # An outcome depends only on each session's status, so whole rows are not read.
    await cur.execute(
        "SELECT id, status FROM cayu_sessions WHERE id = ANY(%s::text[])", (list(session_ids),)
    )
    found = {row[0]: SessionStatus(row[1]) for row in await cur.fetchall()}
    statuses = {session_id: found[session_id] for session_id in session_ids if session_id in found}
    if not statuses:
        return {}
    await cur.execute(
        "SELECT page.session_id, terminal.sequence, terminal.event, retry.sequence, retry.event "
        "FROM unnest(%(ids)s::text[]) AS page(session_id)"
        + _latest_after_lifecycle("event_type = ANY(%(terminal)s::text[])")
        + " AS terminal ON TRUE"
        + _latest_after_lifecycle("event_type = %(retry)s")
        + " AS retry ON TRUE",
        {
            "ids": list(statuses),
            "terminal": _TERMINAL_EVENT_TYPES,
            "retry": str(EventType.MODEL_RETRY),
            "lifecycle": _LIFECYCLE_EVENT_TYPES,
        },
    )
    events = {row[0]: row[1:] for row in await cur.fetchall()}
    outcomes: dict[str, SessionOutcome] = {}
    for session_id, status in statuses.items():
        terminal_sequence, terminal, retry_sequence, retry = events.get(
            session_id, (None, None, None, None)
        )
        outcomes[session_id] = session_status_outcome(
            session_id,
            status,
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


async def query_first_transcript_records(
    cur: Any, session_ids: Sequence[str], role: MessageRole
) -> dict[str, TranscriptRecord]:
    """Each session's first transcript record of ``role``.

    Run it in a repeatable-read transaction, so the rows and the tool events
    their results live in come from the same snapshot.
    """

    if not session_ids:
        return {}
    await cur.execute(
        """
        SELECT page.session_id, earliest.transcript_index, earliest.interaction_id,
               earliest.message
        FROM unnest(%s::text[]) AS page(session_id)
        CROSS JOIN LATERAL (
            SELECT cayu_transcript_messages.session_order - 1 AS transcript_index,
                   cayu_transcript_messages.interaction_id,
                   cayu_transcript_messages.message
            FROM cayu_transcript_messages
            JOIN cayu_sessions ON cayu_sessions.id = cayu_transcript_messages.session_id
            WHERE cayu_transcript_messages.session_id = page.session_id
              AND cayu_transcript_messages.message ->> 'role' = %s
            ORDER BY cayu_transcript_messages.session_order ASC
            LIMIT 1
        ) AS earliest
        """,
        (list(session_ids), str(role)),
    )
    found: dict[str, TranscriptRecord] = {}
    for session_id, index, interaction_id, raw in await cur.fetchall():
        # Tool results a message references live in its session's tool events;
        # the decoder reads them only for a message that references one.
        found[session_id] = TranscriptRecord(
            index=index,
            interaction_id=interaction_id,
            message=await transcript_content.message(cur, session_id, raw),
        )
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
