"""Native PostgreSQL terminal-session evidence reads and export.

Complete reads retain snapshot, limit and cancellation ownership. Composition
supplies native capabilities without a session-store dependency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import LiteralString, cast

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event
from cayu.messages import Message
from cayu.runtime.evidence_spool import EvidenceSpool
from cayu.sessions.base import (
    _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES,
    _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT,
)
from cayu.sessions.event_delivery import restore_persisted_event_authority
from cayu.sessions.records import (
    EventRecord,
    RunnerObservedEventIdentity,
    SessionStatus,
    TranscriptRecord,
)
from cayu.sessions.terminal_evidence import (
    TerminalPublicationMarker,
    TerminalSessionEvidence,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
    TerminalSessionEvidenceLimits,
    _assemble_terminal_session_evidence,
    _classify_terminal_session_evidence_records,
    _copy_runner_owned_interruption_proof,
    _copy_terminal_session_evidence_limits,
    _terminal_session_evidence_expected_event_type,
    _validate_runner_observed_event_identity_snapshot,
)
from cayu.storage import _postgres_support as pg_support
from cayu.storage._postgres_transcript import PostgresConnection, SessionLoader


async def load_terminal_session_evidence(
    connect: PostgresConnection,
    session_id: str,
    *,
    limits: TerminalSessionEvidenceLimits | None = None,
    ensure_ready: Callable[[], Awaitable[None]],
    load_session: SessionLoader,
) -> TerminalSessionEvidence:
    result = await _load_evidence(
        connect,
        session_id,
        ensure_ready=ensure_ready,
        load_session=load_session,
        limits=limits,
        observed_interrupted_events=None,
        expected_interrupted_parent_session_id=None,
        require_interrupted_proof=False,
    )
    assert isinstance(result, TerminalSessionEvidence)
    return result


async def load_runner_owned_interrupted_evidence(
    connect: PostgresConnection,
    session_id: str,
    *,
    observed_events: tuple[RunnerObservedEventIdentity, ...] | None = None,
    expected_parent_session_id: str | None = None,
    limits: TerminalSessionEvidenceLimits | None = None,
    ensure_ready: Callable[[], Awaitable[None]],
    load_session: SessionLoader,
) -> TerminalSessionEvidence:
    result = await _load_evidence(
        connect,
        session_id,
        ensure_ready=ensure_ready,
        load_session=load_session,
        limits=limits,
        observed_interrupted_events=observed_events,
        expected_interrupted_parent_session_id=expected_parent_session_id,
        require_interrupted_proof=True,
    )
    assert isinstance(result, TerminalSessionEvidence)
    return result


async def export_terminal_session_evidence(
    connect: PostgresConnection,
    session_id: str,
    *,
    spool: EvidenceSpool,
    ensure_ready: Callable[[], Awaitable[None]],
    load_session: SessionLoader,
) -> None:
    """Copy one stable terminal snapshot into caller-owned bounded backing."""
    async with asyncio.timeout(spool.limits.max_seconds):
        await _load_evidence(
            connect,
            session_id,
            ensure_ready=ensure_ready,
            load_session=load_session,
            limits=None,
            observed_interrupted_events=None,
            expected_interrupted_parent_session_id=None,
            require_interrupted_proof=False,
            spool=spool,
        )
        await spool.seal_async()


async def _load_evidence(
    connect: PostgresConnection,
    session_id: str,
    *,
    limits: TerminalSessionEvidenceLimits | None,
    observed_interrupted_events: tuple[RunnerObservedEventIdentity, ...] | None,
    expected_interrupted_parent_session_id: str | None,
    require_interrupted_proof: bool,
    spool: EvidenceSpool | None = None,
    ensure_ready: Callable[[], Awaitable[None]],
    load_session: SessionLoader,
) -> TerminalSessionEvidence | None:
    session_id = require_clean_nonblank(session_id, "session_id")
    eager_limits = _copy_terminal_session_evidence_limits(limits)
    selected_limits = eager_limits if spool is None else spool.limits
    observed, expected_parent_session_id = _copy_runner_owned_interruption_proof(
        session_id,
        observed_events=observed_interrupted_events,
        expected_parent_session_id=expected_interrupted_parent_session_id,
        limits=eager_limits,
        required=require_interrupted_proof,
    )
    allow_interrupted = observed is not None or expected_parent_session_id is not None
    evidence_event_types = [
        str(event_type) for event_type in _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES
    ]

    def jsonb_transport_bytes(expression: str) -> str:
        # Bound the complete JSONB text representation that PostgreSQL must
        # transfer and the driver must decode. This intentionally retains
        # serializer whitespace as well as every byte inside string values.
        # The extra byte accounts for JSONB's binary-protocol version prefix
        # when that transfer format is selected.
        return f"octet_length(({expression})::text) + 1"

    def transport_limit(canonical_limit: int) -> int:
        # This is an independent PostgreSQL working-set policy, not an
        # upper bound derived from Cayu's portable JSON representation.
        # Besides separator spaces, JSONB can expand scientific-notation
        # floats into fixed-point decimals by far more than 3:2. Refuse that
        # transport expansion with its own typed error instead of weakening
        # the hydration bound; the shared assembler applies the canonical
        # caller limit to every payload that passes this backend guard.
        return canonical_limit + (canonical_limit + 1) // 2

    event_transport_bytes = f"{jsonb_transport_bytes('event')} + octet_length(sequence::text)"
    transcript_transport_bytes = (
        f"{jsonb_transport_bytes('message')} "
        "+ octet_length(session_order::text) "
        "+ COALESCE(octet_length(interaction_id), 0)"
    )
    session_transport_bytes = " + ".join(
        f"COALESCE(octet_length(session.{column}), 0)"
        for column in (
            "id",
            "instance_id",
            "agent_name",
            "provider_name",
            "model",
            "parent_session_id",
            "causal_budget_id",
            "runtime_name",
            "runtime_version",
            "environment_name",
            "status",
        )
    )
    session_transport_bytes += (
        " + octet_length(session.run_epoch::text)"
        f" + {jsonb_transport_bytes('session.metadata')}"
        f" + {jsonb_transport_bytes('session.invocation')}"
    )
    max_record_transport_bytes = transport_limit(selected_limits.max_record_bytes)
    max_total_transport_bytes = transport_limit(selected_limits.max_total_bytes)

    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                if spool is not None:
                    await cur.execute(
                        "SELECT set_config('statement_timeout', %s, true)",
                        (str(spool.limits.max_seconds * 1000),),
                    )
                await cur.execute(
                    f"""
                            SELECT session.status,
                                   session.run_epoch,
                                   session.parent_session_id,
                                   ({session_transport_bytes})
                                   + COALESCE((
                                       SELECT SUM(
                                           octet_length(label.key)
                                           + octet_length(label.value)
                                       )
                                       FROM cayu_session_labels AS label
                                       WHERE label.session_id = session.id
                                   ), 0) AS transport_bytes
                            FROM cayu_sessions AS session
                            WHERE session.id = %s
                            """,
                    (session_id,),
                )
                session_preflight = await cur.fetchone()
                if session_preflight is None:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.SESSION_NOT_FOUND
                    )
                session_status = SessionStatus(session_preflight[0])
                session_run_epoch = session_preflight[1]
                if type(session_run_epoch) is not int:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                _terminal_session_evidence_expected_event_type(
                    session_status,
                    allow_interrupted=allow_interrupted,
                )
                if allow_interrupted and session_status != SessionStatus.INTERRUPTED:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                if (
                    expected_parent_session_id is not None
                    and session_preflight[2] != expected_parent_session_id
                ):
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                session_transport_size = int(session_preflight[3])
                if session_transport_size > max_record_transport_bytes:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED,
                        limit=max_record_transport_bytes,
                        observed=session_transport_size,
                    )

                if observed is not None:
                    await cur.execute(
                        """
                            WITH bounded_identities AS (
                                SELECT octet_length(event_type)
                                           + octet_length(sequence::text) AS transport_bytes
                                FROM cayu_events
                                WHERE session_id = %s
                                ORDER BY sequence ASC
                                LIMIT %s
                            )
                            SELECT COUNT(*),
                                   COALESCE(MAX(transport_bytes), 0),
                                   COALESCE(SUM(transport_bytes), 0)
                            FROM bounded_identities
                            """,
                        (session_id, selected_limits.max_events + 1),
                    )
                    identity_preflight = await cur.fetchone()
                    if identity_preflight is None:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                        )
                    identity_count = int(identity_preflight[0])
                    if identity_count > selected_limits.max_events:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
                            limit=selected_limits.max_events,
                            observed=identity_count,
                        )
                    if identity_count != len(observed):
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                        )
                    identity_largest_bytes = int(identity_preflight[1])
                    if identity_largest_bytes > max_record_transport_bytes:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED,
                            limit=max_record_transport_bytes,
                            observed=identity_largest_bytes,
                        )
                    identity_total_bytes = int(identity_preflight[2])
                    if identity_total_bytes > max_total_transport_bytes:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED,
                            limit=max_total_transport_bytes,
                            observed=identity_total_bytes,
                        )
                    await cur.execute(
                        """
                            SELECT sequence, event_type
                            FROM cayu_events
                            WHERE session_id = %s
                            ORDER BY sequence ASC
                            LIMIT %s
                            """,
                        (session_id, identity_count),
                    )
                    identity_rows = await cur.fetchall()
                    _validate_runner_observed_event_identity_snapshot(
                        observed,
                        tuple(
                            RunnerObservedEventIdentity(
                                session_id=session_id,
                                sequence=row[0],
                                event_type=row[1],
                            )
                            for row in identity_rows
                        ),
                    )

                await cur.execute(
                    """
                        SELECT
                            CASE
                                WHEN state ? 'session_run_operation'
                                THEN jsonb_typeof(state -> 'session_run_operation')
                            END AS marker_type,
                            jsonb_typeof(
                                state #> '{session_run_operation,version}'
                            ) AS version_type,
                            state #>> '{session_run_operation,version}' AS version_value,
                            jsonb_typeof(
                                state #> '{session_run_operation,operation_id}'
                            ) AS operation_id_type,
                            octet_length(
                                state #>> '{session_run_operation,operation_id}'
                            ) AS operation_id_bytes,
                            length(btrim(COALESCE(
                                state #>> '{session_run_operation,operation_id}',
                                ''
                            ))) > 0 AS operation_id_nonblank,
                            jsonb_typeof(
                                state #> '{session_run_operation,run_epoch}'
                            ) AS run_epoch_type,
                            state #>> '{session_run_operation,run_epoch}' AS run_epoch_value,
                            state ? 'initial_transcript_pending'
                                AS initial_transcript_pending,
                            state ? 'pending_session_interrupt'
                                AS pending_session_interrupt
                        FROM cayu_checkpoints
                        WHERE session_id = %s
                        """,
                    (session_id,),
                )
                checkpoint_projection = await cur.fetchone()
                marker: TerminalPublicationMarker | None = None
                initial_transcript_pending = False
                pending_session_interrupt = False
                marker_stored_bytes = 0
                if checkpoint_projection is not None:
                    initial_transcript_pending = bool(checkpoint_projection[8])
                    pending_session_interrupt = bool(checkpoint_projection[9])
                    marker_type = checkpoint_projection[0]
                    if marker_type is not None:
                        run_epoch_text = checkpoint_projection[7]
                        marker_valid = (
                            marker_type == "object"
                            and checkpoint_projection[1] == "number"
                            and checkpoint_projection[2] == "1"
                            and checkpoint_projection[3] == "string"
                            and bool(checkpoint_projection[5])
                            and checkpoint_projection[6] == "number"
                            and type(run_epoch_text) is str
                            and run_epoch_text.isascii()
                            and run_epoch_text.isdecimal()
                        )
                        if not marker_valid:
                            raise TerminalSessionEvidenceError(
                                TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_INVALID
                            )
                        marker_run_epoch = int(run_epoch_text)
                        if not 1 <= marker_run_epoch <= MAX_DURABLE_JSON_INTEGER:
                            raise TerminalSessionEvidenceError(
                                TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_INVALID
                            )
                        operation_id_bytes = int(checkpoint_projection[4])
                        marker_stored_bytes = operation_id_bytes + len(run_epoch_text)
                        if marker_stored_bytes > selected_limits.max_record_bytes:
                            raise TerminalSessionEvidenceError(
                                TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                                limit=selected_limits.max_record_bytes,
                            )
                        await cur.execute(
                            """
                                SELECT state #>> '{session_run_operation,operation_id}'
                                FROM cayu_checkpoints
                                WHERE session_id = %s
                                """,
                            (session_id,),
                        )
                        operation_row = await cur.fetchone()
                        if operation_row is None:
                            raise TerminalSessionEvidenceError(
                                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                            )
                        try:
                            marker = TerminalPublicationMarker(
                                operation_id=operation_row[0],
                                run_epoch=marker_run_epoch,
                            )
                        except (TypeError, ValueError) as exc:
                            raise TerminalSessionEvidenceError(
                                TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_INVALID
                            ) from exc

                await cur.execute(
                    cast(
                        "LiteralString",
                        f"""
                            SELECT sequence,
                                   event_type,
                                   ({event_transport_bytes}) AS transport_bytes,
                                   jsonb_typeof(
                                       payload -> 'session_run_operation_id'
                                   ) AS operation_id_type,
                                   length(btrim(COALESCE(
                                       payload ->> 'session_run_operation_id',
                                       ''
                                   ))) > 0 AS operation_id_nonblank
                            FROM cayu_events
                            WHERE session_id = %s AND event_type = ANY(%s)
                            ORDER BY sequence DESC
                            LIMIT %s
                            """,
                    ),
                    (
                        session_id,
                        evidence_event_types,
                        _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT,
                    ),
                )
                newest_preflight_rows = await cur.fetchall()
                if any(int(row[2]) > max_record_transport_bytes for row in newest_preflight_rows):
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED,
                        limit=max_record_transport_bytes,
                    )
                if any(
                    row[3] not in {None, "string"} or (row[3] == "string" and not bool(row[4]))
                    for row in newest_preflight_rows
                ):
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                newest_sequences = [row[0] for row in newest_preflight_rows]
                newest_evidence_records: tuple[EventRecord, ...]
                if newest_sequences:
                    await cur.execute(
                        """
                            SELECT sequence,
                                   event_id,
                                   event_type,
                                   payload ->> 'session_run_operation_id'
                            FROM cayu_events
                            WHERE sequence = ANY(%s)
                            ORDER BY sequence DESC
                            """,
                        (newest_sequences,),
                    )
                    newest_rows = await cur.fetchall()
                    newest_evidence_records = tuple(
                        EventRecord(
                            sequence=row[0],
                            event=Event(
                                id=row[1],
                                type=row[2],
                                session_id=session_id,
                                payload=(
                                    {} if row[3] is None else {"session_run_operation_id": row[3]}
                                ),
                            ),
                        )
                        for row in newest_rows
                    )
                else:
                    newest_evidence_records = ()
                terminal_record = _classify_terminal_session_evidence_records(
                    session_id=session_id,
                    status=session_status,
                    run_epoch=session_run_epoch,
                    marker=marker,
                    newest_evidence_records=newest_evidence_records,
                    initial_transcript_pending=initial_transcript_pending,
                    pending_session_interrupt=pending_session_interrupt,
                    allow_interrupted=allow_interrupted,
                )

                await cur.execute(
                    cast(
                        "LiteralString",
                        f"""
                            WITH bounded_events AS (
                                SELECT ({event_transport_bytes}) AS transport_bytes
                                FROM cayu_events
                                WHERE session_id = %s AND sequence <= %s
                                ORDER BY sequence ASC
                                LIMIT %s
                            )
                            SELECT COUNT(*),
                                   COALESCE(MAX(transport_bytes), 0),
                                   COALESCE(SUM(transport_bytes), 0)
                            FROM bounded_events
                            """,
                    ),
                    (session_id, terminal_record.sequence, selected_limits.max_events + 1),
                )
                event_preflight = await cur.fetchone()
                await cur.execute(
                    cast(
                        "LiteralString",
                        f"""
                            WITH bounded_transcript AS (
                                SELECT ({transcript_transport_bytes}) AS transport_bytes
                                FROM cayu_transcript_messages
                                WHERE session_id = %s
                                ORDER BY session_order ASC
                                LIMIT %s
                            )
                            SELECT COUNT(*),
                                   COALESCE(MAX(transport_bytes), 0),
                                   COALESCE(SUM(transport_bytes), 0)
                            FROM bounded_transcript
                            """,
                    ),
                    (session_id, selected_limits.max_transcript_records + 1),
                )
                transcript_preflight = await cur.fetchone()
                if event_preflight is None or transcript_preflight is None:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                event_count = int(event_preflight[0])
                transcript_count = int(transcript_preflight[0])
                if event_count > selected_limits.max_events:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
                        limit=selected_limits.max_events,
                        observed=event_count,
                    )
                if transcript_count > selected_limits.max_transcript_records:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED,
                        limit=selected_limits.max_transcript_records,
                        observed=transcript_count,
                    )
                largest_transport_bytes = max(
                    session_transport_size,
                    int(event_preflight[1]),
                    int(transcript_preflight[1]),
                    marker_stored_bytes,
                )
                if largest_transport_bytes > max_record_transport_bytes:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED,
                        limit=max_record_transport_bytes,
                        observed=largest_transport_bytes,
                    )
                total_transport_bytes = (
                    session_transport_size
                    + int(event_preflight[2])
                    + int(transcript_preflight[2])
                    + marker_stored_bytes
                )
                if total_transport_bytes > max_total_transport_bytes:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED,
                        limit=max_total_transport_bytes,
                        observed=total_transport_bytes,
                    )

                session = await load_session(cur, session_id)
                if session is None:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                if spool is not None:
                    after_sequence = 0
                    while True:
                        spool.check()
                        await cur.execute(
                            "SELECT sequence, event, input_contract_runtime_owned, "
                            f"file_attachment_attestations_runtime_owned, ({event_transport_bytes}), "
                            "((event->>'id') IS NOT DISTINCT FROM event_id AND "
                            "(event->>'session_id') IS NOT DISTINCT FROM session_id AND "
                            "(event->>'type') IS NOT DISTINCT FROM event_type AND "
                            "(event->>'interaction_id') IS NOT DISTINCT FROM interaction_id AND "
                            "(event->>'agent_name') IS NOT DISTINCT FROM agent_name AND "
                            "(event->>'environment_name') IS NOT DISTINCT FROM environment_name AND "
                            "(event->>'workflow_name') IS NOT DISTINCT FROM workflow_name AND "
                            "(event->>'tool_name') IS NOT DISTINCT FROM tool_name AND "
                            "(event->'payload') IS NOT DISTINCT FROM payload) "
                            "FROM cayu_events "
                            "WHERE session_id = %s AND sequence > %s AND sequence <= %s "
                            "ORDER BY sequence ASC LIMIT %s",
                            (
                                session_id,
                                after_sequence,
                                terminal_record.sequence,
                                spool.limits.batch_records,
                            ),
                        )
                        rows = await cur.fetchall()
                        if not rows:
                            break
                        spool.observe_page(
                            records=len(rows), transport_bytes=sum(row[4] for row in rows)
                        )
                        for row in rows:
                            if row[5] is not True:
                                raise TerminalSessionEvidenceError(
                                    TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                                )
                            spool.append(
                                "event",
                                EventRecord(
                                    sequence=row[0],
                                    event=restore_persisted_event_authority(
                                        Event(**pg_support._json_obj(row[1])),
                                        input_contract_runtime_owned=row[2],
                                        file_attachment_attestations_runtime_owned=row[3],
                                    ),
                                ),
                            )
                        after_sequence = rows[-1][0]
                    after_index = 0
                    while True:
                        spool.check()
                        await cur.execute(
                            f"SELECT session_order, interaction_id, message, ({transcript_transport_bytes}) "
                            "FROM cayu_transcript_messages WHERE session_id = %s "
                            "AND session_order > %s ORDER BY session_order ASC LIMIT %s",
                            (session_id, after_index, spool.limits.batch_records),
                        )
                        rows = await cur.fetchall()
                        if not rows:
                            break
                        spool.observe_page(
                            records=len(rows), transport_bytes=sum(row[3] for row in rows)
                        )
                        for row in rows:
                            spool.append(
                                "transcript",
                                TranscriptRecord(
                                    index=row[0] - 1,
                                    interaction_id=row[1],
                                    message=Message(**pg_support._json_obj(row[2])),
                                ),
                            )
                        after_index = rows[-1][0]
                    spool.stage(session, marker, terminal_record, event_count, transcript_count)
                    return None
                await cur.execute(
                    """
                        SELECT sequence, event, input_contract_runtime_owned,
                               file_attachment_attestations_runtime_owned
                        FROM cayu_events
                        WHERE session_id = %s AND sequence <= %s
                        ORDER BY sequence ASC
                        """,
                    (session_id, terminal_record.sequence),
                )
                event_rows = await cur.fetchall()
                await cur.execute(
                    """
                        SELECT session_order - 1, interaction_id, message
                        FROM cayu_transcript_messages
                        WHERE session_id = %s
                        ORDER BY session_order ASC
                        """,
                    (session_id,),
                )
                transcript_rows = await cur.fetchall()
                if len(event_rows) != event_count or len(transcript_rows) != transcript_count:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
                events = tuple(
                    EventRecord(
                        sequence=row[0],
                        event=restore_persisted_event_authority(
                            Event(**pg_support._json_obj(row[1])),
                            input_contract_runtime_owned=row[2],
                            file_attachment_attestations_runtime_owned=row[3],
                        ),
                    )
                    for row in event_rows
                )
                transcript = tuple(
                    TranscriptRecord(
                        index=row[0],
                        interaction_id=row[1],
                        message=Message(**pg_support._json_obj(row[2])),
                    )
                    for row in transcript_rows
                )
                return _assemble_terminal_session_evidence(
                    session=session,
                    marker=marker,
                    terminal_record=terminal_record,
                    events=events,
                    transcript=transcript,
                    limits=eager_limits,
                    allow_interrupted=allow_interrupted,
                )
        except TerminalSessionEvidenceError:
            raise
        except (TypeError, ValueError) as exc:
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
            ) from exc
