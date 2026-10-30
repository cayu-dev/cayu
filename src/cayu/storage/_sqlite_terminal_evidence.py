"""Native SQLite terminal-session evidence reads and export.

Complete reads retain snapshot, limit and cancellation ownership. Composition
supplies native capabilities without a session-store dependency.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event
from cayu.runtime.evidence_spool import EvidenceSpool
from cayu.sessions.records import (
    EventRecord,
    RunnerObservedEventIdentity,
    SessionStatus,
    TranscriptRecord,
)
from cayu.sessions.terminal_evidence import (
    _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES,
    _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES,
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
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage import _sqlite_transcript_content as transcript_content
from cayu.storage._sqlite_connection import SQLiteOperationRunner


async def load_terminal_session_evidence(
    run_read: SQLiteOperationRunner,
    session_id: str,
    *,
    limits: TerminalSessionEvidenceLimits | None = None,
) -> TerminalSessionEvidence:
    result = await _load_evidence(
        run_read,
        session_id,
        limits=limits,
        observed_interrupted_events=None,
        expected_interrupted_parent_session_id=None,
        require_interrupted_proof=False,
    )
    assert isinstance(result, TerminalSessionEvidence)
    return result


async def load_runner_owned_interrupted_evidence(
    run_read: SQLiteOperationRunner,
    session_id: str,
    *,
    observed_events: tuple[RunnerObservedEventIdentity, ...] | None = None,
    expected_parent_session_id: str | None = None,
    limits: TerminalSessionEvidenceLimits | None = None,
) -> TerminalSessionEvidence:
    result = await _load_evidence(
        run_read,
        session_id,
        limits=limits,
        observed_interrupted_events=observed_events,
        expected_interrupted_parent_session_id=expected_parent_session_id,
        require_interrupted_proof=True,
    )
    assert isinstance(result, TerminalSessionEvidence)
    return result


async def export_terminal_session_evidence(
    run_read: SQLiteOperationRunner, session_id: str, *, spool: EvidenceSpool
) -> None:
    """Copy one stable terminal snapshot into caller-owned bounded backing."""
    async with asyncio.timeout(spool.limits.max_seconds):
        await spool.run_owned(
            _load_evidence(
                run_read,
                session_id,
                limits=None,
                observed_interrupted_events=None,
                expected_interrupted_parent_session_id=None,
                require_interrupted_proof=False,
                spool=spool,
            )
        )
        await spool.seal_async()


async def _load_evidence(
    run_read: SQLiteOperationRunner,
    session_id: str,
    *,
    limits: TerminalSessionEvidenceLimits | None,
    observed_interrupted_events: tuple[RunnerObservedEventIdentity, ...] | None,
    expected_interrupted_parent_session_id: str | None,
    require_interrupted_proof: bool,
    spool: EvidenceSpool | None = None,
) -> TerminalSessionEvidence | None:
    session_id = require_clean_nonblank(session_id, "session_id")
    eager_limits = _copy_terminal_session_evidence_limits(limits)
    resolved_limits = eager_limits if spool is None else spool.limits
    observed, expected_parent_session_id = _copy_runner_owned_interruption_proof(
        session_id,
        observed_events=observed_interrupted_events,
        expected_parent_session_id=expected_interrupted_parent_session_id,
        limits=eager_limits,
        required=require_interrupted_proof,
    )
    allow_interrupted = observed is not None or expected_parent_session_id is not None
    evidence_event_types = tuple(
        str(event_type) for event_type in _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES
    )
    evidence_type_placeholders = ", ".join("?" for _ in evidence_event_types)
    event_columns = ", ".join(sqlite_records.EVENT_COLUMN_NAMES)
    event_stored_bytes = " + ".join(
        [
            "length(CAST(sequence AS TEXT))",
            *(
                f"COALESCE(length(CAST({column} AS BLOB)), 0)"
                for column in sqlite_records.EVENT_COLUMN_NAMES
            ),
        ]
    )
    session_stored_bytes = " + ".join(
        f"COALESCE(length(CAST(session.{column} AS BLOB)), 0)"
        for column in (
            "id",
            "instance_id",
            "invocation_json",
            "agent_name",
            "provider_name",
            "model",
            "parent_session_id",
            "causal_budget_id",
            "runtime_name",
            "runtime_version",
            "environment_name",
            "status",
            "created_at",
            "updated_at",
            "last_activity_at",
            "run_epoch",
            "metadata_json",
        )
    )
    transcript_stored_bytes = " + ".join(
        (
            "length(CAST(session_order AS TEXT))",
            "COALESCE(length(CAST(interaction_id AS BLOB)), 0)",
            # The full message size, including a tool result read from its tool event.
            "COALESCE(message_bytes, length(CAST(message_json AS BLOB)))",
        )
    )

    def run_query(connection: sqlite3.Connection) -> TerminalSessionEvidence | None:
        limits = resolved_limits
        if spool is not None:
            connection.set_progress_handler(lambda: int(spool.should_interrupt()), 1000)
        connection.execute("BEGIN")
        try:
            session_preflight = connection.execute(
                f"""
                    SELECT session.status, session.run_epoch, session.parent_session_id,
                           ({session_stored_bytes})
                           + COALESCE((
                               SELECT SUM(
                                   length(CAST(label.key AS BLOB))
                                   + length(CAST(label.value AS BLOB))
                               )
                               FROM cayu_session_labels AS label
                               WHERE label.session_id = session.id
                           ), 0) AS stored_bytes
                    FROM cayu_sessions AS session
                    WHERE session.id = ?
                    """,
                (session_id,),
            ).fetchone()
            if session_preflight is None:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.SESSION_NOT_FOUND
                )
            session_status = SessionStatus(session_preflight["status"])
            session_run_epoch = session_preflight["run_epoch"]
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
                and session_preflight["parent_session_id"] != expected_parent_session_id
            ):
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                )
            if (
                int(session_preflight["stored_bytes"])
                > TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES
            ):
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                    limit=limits.max_record_bytes,
                )

            checkpoint_projection = connection.execute(
                """
                    SELECT
                        json_type(state_json, '$.session_run_operation') AS marker_type,
                        json_type(
                            state_json,
                            '$.session_run_operation.version'
                        ) AS version_type,
                        json_extract(
                            state_json,
                            '$.session_run_operation.version'
                        ) AS version_value,
                        json_type(
                            state_json,
                            '$.session_run_operation.operation_id'
                        ) AS operation_id_type,
                        length(CAST(json_extract(
                            state_json,
                            '$.session_run_operation.operation_id'
                        ) AS BLOB)) AS operation_id_bytes,
                        length(trim(COALESCE(json_extract(
                            state_json,
                            '$.session_run_operation.operation_id'
                        ), ''))) > 0 AS operation_id_nonblank,
                        json_type(
                            state_json,
                            '$.session_run_operation.run_epoch'
                        ) AS run_epoch_type,
                        json_extract(
                            state_json,
                            '$.session_run_operation.run_epoch'
                        ) AS run_epoch_value,
                        json_type(
                            state_json,
                            '$.initial_transcript_pending'
                        ) IS NOT NULL AS initial_transcript_pending,
                        json_type(
                            state_json,
                            '$.pending_session_interrupt'
                        ) IS NOT NULL AS pending_session_interrupt
                    FROM cayu_checkpoints
                    WHERE session_id = ?
                    """,
                (session_id,),
            ).fetchone()
            marker: TerminalPublicationMarker | None = None
            initial_transcript_pending = False
            pending_session_interrupt = False
            marker_stored_bytes = 0
            if checkpoint_projection is not None:
                initial_transcript_pending = bool(
                    checkpoint_projection["initial_transcript_pending"]
                )
                pending_session_interrupt = bool(checkpoint_projection["pending_session_interrupt"])
                marker_type = checkpoint_projection["marker_type"]
                if marker_type is not None:
                    marker_valid = (
                        marker_type == "object"
                        and checkpoint_projection["version_type"] == "integer"
                        and checkpoint_projection["version_value"] == 1
                        and checkpoint_projection["operation_id_type"] == "text"
                        and bool(checkpoint_projection["operation_id_nonblank"])
                        and checkpoint_projection["run_epoch_type"] == "integer"
                        and type(checkpoint_projection["run_epoch_value"]) is int
                        and 1
                        <= checkpoint_projection["run_epoch_value"]
                        <= MAX_DURABLE_JSON_INTEGER
                    )
                    if not marker_valid:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_INVALID
                        )
                    operation_id_bytes = int(checkpoint_projection["operation_id_bytes"])
                    if operation_id_bytes > TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                            limit=limits.max_record_bytes,
                        )
                    marker_stored_bytes = operation_id_bytes + len(
                        str(checkpoint_projection["run_epoch_value"]).encode("utf-8")
                    )
                    if marker_stored_bytes > limits.max_record_bytes:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                            limit=limits.max_record_bytes,
                        )
                    operation_row = connection.execute(
                        """
                            SELECT json_extract(
                                state_json,
                                '$.session_run_operation.operation_id'
                            ) AS operation_id
                            FROM cayu_checkpoints
                            WHERE session_id = ?
                            """,
                        (session_id,),
                    ).fetchone()
                    if operation_row is None:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                        )
                    try:
                        marker = TerminalPublicationMarker(
                            operation_id=operation_row["operation_id"],
                            run_epoch=checkpoint_projection["run_epoch_value"],
                        )
                    except (TypeError, ValueError) as exc:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_INVALID
                        ) from exc

            newest_preflight_rows = connection.execute(
                f"""
                    SELECT sequence, event_type,
                           ({event_stored_bytes}) AS stored_bytes,
                           json_type(
                               payload_json,
                               '$.session_run_operation_id'
                           ) AS operation_id_type,
                           length(trim(COALESCE(json_extract(
                               payload_json,
                               '$.session_run_operation_id'
                           ), ''))) > 0 AS operation_id_nonblank
                    FROM cayu_events
                    WHERE session_id = ?
                      AND event_type IN ({evidence_type_placeholders})
                    ORDER BY sequence DESC
                    LIMIT ?
                    """,
                (
                    session_id,
                    *evidence_event_types,
                    _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT,
                ),
            ).fetchall()
            if any(
                int(row["stored_bytes"]) > limits.max_record_bytes for row in newest_preflight_rows
            ):
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                    limit=limits.max_record_bytes,
                )
            if any(
                row["operation_id_type"] not in {None, "text"}
                or (row["operation_id_type"] == "text" and not bool(row["operation_id_nonblank"]))
                for row in newest_preflight_rows
            ):
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                )
            newest_sequences = tuple(row["sequence"] for row in newest_preflight_rows)
            newest_evidence_records: tuple[EventRecord, ...]
            if newest_sequences:
                sequence_placeholders = ", ".join("?" for _ in newest_sequences)
                newest_rows = connection.execute(
                    f"""
                        SELECT sequence, event_id, event_type,
                               json_extract(
                                   payload_json,
                                   '$.session_run_operation_id'
                               ) AS operation_id
                        FROM cayu_events
                        WHERE sequence IN ({sequence_placeholders})
                        ORDER BY sequence DESC
                        """,
                    newest_sequences,
                ).fetchall()
                newest_evidence_records = tuple(
                    EventRecord(
                        sequence=row["sequence"],
                        event=Event(
                            id=row["event_id"],
                            type=row["event_type"],
                            session_id=session_id,
                            payload=(
                                {}
                                if row["operation_id"] is None
                                else {"session_run_operation_id": row["operation_id"]}
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

            event_preflight = connection.execute(
                f"""
                    WITH bounded_events AS (
                        SELECT ({event_stored_bytes}) AS stored_bytes
                        FROM cayu_events
                        WHERE session_id = ? AND sequence <= ?
                        ORDER BY sequence ASC
                        LIMIT ?
                    )
                    SELECT COUNT(*) AS record_count,
                           COALESCE(MAX(stored_bytes), 0) AS largest_record_bytes,
                           COALESCE(SUM(stored_bytes), 0) AS total_bytes
                    FROM bounded_events
                    """,
                (
                    session_id,
                    terminal_record.sequence,
                    limits.max_events + 1,
                ),
            ).fetchone()
            transcript_preflight = connection.execute(
                f"""
                    WITH bounded_transcript AS (
                        SELECT ({transcript_stored_bytes}) AS stored_bytes
                        FROM cayu_transcript_messages
                        WHERE session_id = ?
                        ORDER BY session_order ASC
                        LIMIT ?
                    )
                    SELECT COUNT(*) AS record_count,
                           COALESCE(MAX(stored_bytes), 0) AS largest_record_bytes,
                           COALESCE(SUM(stored_bytes), 0) AS total_bytes
                    FROM bounded_transcript
                    """,
                (session_id, limits.max_transcript_records + 1),
            ).fetchone()
            if event_preflight is None or transcript_preflight is None:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                )
            event_count = int(event_preflight["record_count"])
            transcript_count = int(transcript_preflight["record_count"])
            if event_count > limits.max_events:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
                    limit=limits.max_events,
                    observed=event_count,
                )
            if transcript_count > limits.max_transcript_records:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED,
                    limit=limits.max_transcript_records,
                    observed=transcript_count,
                )
            session_lower_bytes = int(session_preflight["stored_bytes"])
            largest_lower_bytes = max(
                session_lower_bytes,
                int(event_preflight["largest_record_bytes"]),
                int(transcript_preflight["largest_record_bytes"]),
                marker_stored_bytes,
            )
            if largest_lower_bytes > limits.max_record_bytes:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                    limit=limits.max_record_bytes,
                )
            total_lower_bytes = (
                session_lower_bytes
                + int(event_preflight["total_bytes"])
                + int(transcript_preflight["total_bytes"])
                + marker_stored_bytes
            )
            if total_lower_bytes > limits.max_total_bytes:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
                    limit=limits.max_total_bytes,
                )
            if observed is not None:
                identity_rows = connection.execute(
                    """
                        SELECT sequence, event_type
                        FROM cayu_events
                        WHERE session_id = ? AND sequence <= ?
                        ORDER BY sequence ASC
                        LIMIT ?
                        """,
                    (session_id, terminal_record.sequence, event_count),
                ).fetchall()
                _validate_runner_observed_event_identity_snapshot(
                    observed,
                    tuple(
                        RunnerObservedEventIdentity(
                            session_id=session_id,
                            sequence=row["sequence"],
                            event_type=row["event_type"],
                        )
                        for row in identity_rows
                    ),
                    terminal_sequence=terminal_record.sequence,
                )

            session = sqlite_records.load_session(connection, session_id)
            if session is None:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                )
            if spool is not None:
                cursor = connection.execute(
                    f"SELECT sequence, {event_columns}, ({event_stored_bytes}) AS transport_bytes FROM cayu_events "
                    "WHERE session_id = ? AND sequence <= ? ORDER BY sequence ASC",
                    (session_id, terminal_record.sequence),
                )
                while True:
                    spool.check()
                    rows = cursor.fetchmany(spool.limits.batch_records)
                    if not rows:
                        break
                    spool.observe_page(
                        records=len(rows),
                        transport_bytes=sum(row["transport_bytes"] for row in rows),
                    )
                    for row in rows:
                        spool.append(
                            "event",
                            EventRecord(
                                sequence=row["sequence"],
                                event=sqlite_records.event_from_row(row),
                            ),
                        )
                cursor.close()
                cursor = connection.execute(
                    f"SELECT session_order - 1 AS transcript_index, interaction_id, message_json, ({transcript_stored_bytes}) AS transport_bytes "
                    "FROM cayu_transcript_messages WHERE session_id = ? ORDER BY session_order ASC",
                    (session_id,),
                )
                while True:
                    spool.check()
                    rows = cursor.fetchmany(spool.limits.batch_records)
                    if not rows:
                        break
                    spool.observe_page(
                        records=len(rows),
                        transport_bytes=sum(row["transport_bytes"] for row in rows),
                    )
                    decode = transcript_content.decoder(
                        connection, session_id, [row["message_json"] for row in rows]
                    )
                    for row in rows:
                        spool.append(
                            "transcript",
                            TranscriptRecord(
                                index=row["transcript_index"],
                                interaction_id=row["interaction_id"],
                                message=decode(row["message_json"]),
                            ),
                        )
                cursor.close()
                spool.stage(session, marker, terminal_record, event_count, transcript_count)
                return None
            event_rows = connection.execute(
                f"""
                    SELECT sequence, {event_columns}
                    FROM cayu_events
                    WHERE session_id = ? AND sequence <= ?
                    ORDER BY sequence ASC
                    """,
                (session_id, terminal_record.sequence),
            ).fetchall()
            transcript_rows = connection.execute(
                """
                    SELECT session_order - 1 AS transcript_index,
                           interaction_id,
                           message_json
                    FROM cayu_transcript_messages
                    WHERE session_id = ?
                    ORDER BY session_order ASC
                    """,
                (session_id,),
            ).fetchall()
            if len(event_rows) != event_count or len(transcript_rows) != transcript_count:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                )
            events = tuple(
                EventRecord(sequence=row["sequence"], event=sqlite_records.event_from_row(row))
                for row in event_rows
            )
            decode = transcript_content.decoder(
                connection, session_id, [row["message_json"] for row in transcript_rows]
            )
            transcript = tuple(
                TranscriptRecord(
                    index=row["transcript_index"],
                    interaction_id=row["interaction_id"],
                    message=decode(row["message_json"]),
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
        except (json.JSONDecodeError, sqlite3.DatabaseError, TypeError, ValueError) as exc:
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
            ) from exc
        finally:
            if spool is not None:
                connection.set_progress_handler(None, 0)
            connection.rollback()

    return await run_read(run_query)
