"""Complete SQLite transcript operations with explicit native capabilities."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable
from datetime import datetime
from typing import Any, Protocol

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.messages import Message, MessageRole
from cayu.sessions._checkpoint_preservation import (
    _checkpoint_transform_result_preserving_completion_result_event_publications,
    _copy_checkpoint_for_transform,
)
from cayu.sessions.base import (
    INHERIT_INTERACTION,
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    MODEL_TARGET_PROJECTION_METADATA_KEY,
    RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX,
    CheckpointTransform,
    InteractionAttribution,
    _assert_session_run_epoch,
    _check_closure_lineage_owner,
    _checkpoint_after_initial_transcript_publication,
    resolve_interaction_attribution,
)
from cayu.sessions.records import Session, SessionStatus, TranscriptRecord
from cayu.sessions.transcript_input import (
    DeferredInteractionInput,
    _initial_transcript_prefix_count,
    copy_transcript_messages,
    deferred_interaction_input_from_storage_payload,
    require_deferred_initial_transcript_replacement,
)
from cayu.sessions.transcript_queries import (
    LATEST_TRANSCRIPT_TEXT_MAX_CHARS,
    LATEST_TRANSCRIPT_TEXT_MAX_PARTS,
    LATEST_TRANSCRIPT_TEXT_MAX_SOURCE_BYTES,
    TranscriptPage,
    TranscriptQuery,
    TranscriptSearchHit,
    TranscriptSearchQuery,
    TranscriptSearchResult,
    TranscriptSnapshot,
    TranscriptTextReadLimitExceeded,
    copy_transcript_query,
    copy_transcript_search_query,
    decode_transcript_search_cursor,
    encode_transcript_search_cursor,
    filter_transcript_records,
    transcript_search_document,
    transcript_search_document_score,
    transcript_search_hit_from_message,
    transcript_search_position_after_cursor,
    transcript_search_query_document,
    transcript_search_session_token,
)
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage._sqlite_connection import SQLiteOperationRunner


class ClosureOwners(Protocol):
    def __call__(
        self, targets: Iterable[str], *, connection: sqlite3.Connection | None = None
    ) -> tuple[dict[str, Any], ...]: ...


def transcript_cursor(connection: sqlite3.Connection, session_id: str) -> int:
    """Return the permanent next transcript position, independent of retention."""

    row = connection.execute(
        "SELECT transcript_seq FROM cayu_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Session not found: {session_id}")
    return int(row["transcript_seq"])


def _search_expression(query: TranscriptSearchQuery) -> str:
    session_terms = " OR ".join(
        f'"{transcript_search_session_token(session_id)}"' for session_id in query.session_ids
    )
    text_terms = " OR ".join(
        f'"{token}"' for token in transcript_search_query_document(query.text).split()
    )
    return f"session_token:({session_terms}) AND message_text:({text_terms})"


async def compact_transcript(
    run_write: SQLiteOperationRunner, session_id: str, *, keep_last: int
) -> int:
    """Compact a session's transcript, keeping only its most recent messages.

    Retains the ``keep_last`` newest transcript messages (by insertion order)
    for ``session_id`` and deletes the rest, bounding transcript growth for
    long-lived sessions. Active model stages, pending tool rounds, and
    immutable publication receipts pin their recovery material. Active
    model-target projection runs also pin the transcript; their permanent
    absolute cursor makes retention safe again at a terminal boundary.
    Returns the number of messages deleted.
    """
    if type(keep_last) is not int:
        raise TypeError("compact_transcript 'keep_last' must be an int.")
    if keep_last < 0:
        raise ValueError("compact_transcript 'keep_last' must be >= 0.")
    session_id = require_clean_nonblank(session_id, "session_id")

    def statement(connection: sqlite3.Connection) -> int:
        if not sqlite_records.session_exists(connection, session_id):
            raise KeyError(f"Session not found: {session_id}")
        with connection:
            durability_guard = connection.execute(
                """
                SELECT 1
                FROM cayu_session_operations
                WHERE session_id = ?
                  AND (
                      idempotency_key GLOB ?
                      OR idempotency_key = ?
                  )
                UNION ALL
                SELECT 1
                FROM cayu_checkpoints
                WHERE session_id = ?
                  AND json_type(state_json, '$.pending_tool_round') IS NOT NULL
                UNION ALL
                SELECT 1
                FROM cayu_sessions
                WHERE id = ?
                  AND status IN (?, ?, ?)
                  AND json_type(metadata_json, ?) IS NOT NULL
                LIMIT 1
                """,
                (
                    session_id,
                    RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX + "*",
                    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                    session_id,
                    session_id,
                    str(SessionStatus.PENDING),
                    str(SessionStatus.RUNNING),
                    str(SessionStatus.INTERRUPTING),
                    f'$."{MODEL_TARGET_PROJECTION_METADATA_KEY}"',
                ),
            ).fetchone()
            if durability_guard is not None:
                return 0
            cursor = connection.execute(
                """
                DELETE FROM cayu_transcript_messages
                WHERE session_id = ?
                  AND sequence NOT IN (
                      SELECT sequence
                      FROM cayu_transcript_messages
                      WHERE session_id = ?
                      ORDER BY sequence DESC
                      LIMIT ?
                  )
                """,
                (session_id, session_id, keep_last),
            )
            deleted = cursor.rowcount
        return deleted

    return await run_write(statement)


async def append_transcript_messages(
    run_write: SQLiteOperationRunner,
    session_id: str,
    messages: list[Message],
    *,
    interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ownership_clock: Callable[[], datetime],
    closure_owners: ClosureOwners,
    touch_activity: Callable[[sqlite3.Connection, str, datetime], None],
) -> None:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = resolve_interaction_attribution(session_id, interaction_id)
    copied_messages = copy_transcript_messages(messages)

    def statement(connection: sqlite3.Connection) -> None:
        if not copied_messages:
            if not sqlite_records.session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            return
        try:
            connection.execute("BEGIN IMMEDIATE")
            if not sqlite_records.session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            activity_at = ownership_clock()
            for owner in closure_owners((session_id,)):
                _check_closure_lineage_owner(owner, (session_id,))
            touch_activity(connection, session_id, activity_at)
            connection.executemany(
                """
                INSERT INTO cayu_transcript_messages (
                    session_id,
                    role,
                    interaction_id,
                    message_json,
                    transcript_search_document
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        str(message.role),
                        interaction_id,
                        sqlite_records.json_dumps(message.model_dump(mode="json")),
                        transcript_search_document(message),
                    )
                    for message in copied_messages
                ],
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise

    await run_write(statement)


async def replace_initial_transcript_messages(
    run_write: SQLiteOperationRunner,
    session_id: str,
    expected_messages: list[Message],
    replacement_messages: list[Message],
    *,
    interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    checkpoint_transform: CheckpointTransform | None = None,
    runtime_suffix_count: int = 0,
    ownership_clock: Callable[[], datetime],
    load_session: Callable[[str], Session | None],
    load_checkpoint: Callable[[str], dict[str, Any] | None],
    closure_owners: ClosureOwners,
    touch_activity: Callable[[sqlite3.Connection, str, datetime], None],
) -> None:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = resolve_interaction_attribution(session_id, interaction_id)
    if interaction_id is None:
        raise ValueError("Initial transcript publication requires an interaction identity.")
    expected = copy_transcript_messages(expected_messages)
    replacement = copy_transcript_messages(replacement_messages)

    def statement(connection: sqlite3.Connection) -> None:
        try:
            connection.execute("BEGIN IMMEDIATE")
            updated_at = ownership_clock()
            session = load_session(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            row = connection.execute(
                "SELECT interaction_id, source_messages_json "
                "FROM cayu_deferred_interaction_inputs WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None or row["interaction_id"] != interaction_id:
                raise RuntimeError("Deferred interaction input changed before finalization.")
            for owner in closure_owners((session_id,), connection=connection):
                _check_closure_lineage_owner(owner, (session_id,))
            stored = deferred_interaction_input_from_storage_payload(
                row["interaction_id"],
                json.loads(row["source_messages_json"]),
            )
            require_deferred_initial_transcript_replacement(
                stored,
                expected_messages=expected,
                replacement_messages=replacement,
            )
            existing = connection.execute(
                "SELECT 1 FROM cayu_transcript_messages WHERE session_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
            if existing is not None:
                from cayu.sessions._participant_execution_identity import (
                    require_initial_execution_input,
                )
                from cayu.storage._participant_session_records import reconstruct

                binding_row = connection.execute(
                    "SELECT * FROM cayu_participant_session_bindings WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                current_rows = connection.execute(
                    "SELECT message_json FROM cayu_transcript_messages WHERE session_id = ? ORDER BY session_order",
                    (session_id,),
                ).fetchall()
                require_initial_execution_input(
                    session,
                    None if binding_row is None else reconstruct(dict(binding_row), session),
                    [Message.model_validate_json(row[0]) for row in current_rows],
                    expected,
                )
                connection.execute(
                    "DELETE FROM cayu_transcript_messages WHERE session_id = ?", (session_id,)
                )
                connection.execute(
                    "UPDATE cayu_sessions SET transcript_seq = 0 WHERE id = ?", (session_id,)
                )
            prefix_count = _initial_transcript_prefix_count(
                expected,
                replacement,
                runtime_suffix_count=runtime_suffix_count,
            )
            current_checkpoint = load_checkpoint(session_id)
            if checkpoint_transform is not None:
                transformed = checkpoint_transform(
                    session,
                    _copy_checkpoint_for_transform(
                        current_checkpoint,
                        session_id=session_id,
                    ),
                )
                if transformed is not None:
                    current_checkpoint = _checkpoint_transform_result_preserving_completion_result_event_publications(
                        current_checkpoint,
                        transformed,
                        session_id=session_id,
                    )
            checkpoint = _checkpoint_after_initial_transcript_publication(
                current_checkpoint,
                interaction_id=interaction_id,
            )
            connection.executemany(
                "INSERT INTO cayu_transcript_messages "
                "(session_id, role, interaction_id, message_json, "
                "transcript_search_document) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        session_id,
                        str(message.role),
                        None if index < prefix_count else interaction_id,
                        sqlite_records.json_dumps(message.model_dump(mode="json")),
                        transcript_search_document(message),
                    )
                    for index, message in enumerate(replacement)
                ],
            )
            connection.execute(
                "DELETE FROM cayu_deferred_interaction_inputs WHERE session_id = ?",
                (session_id,),
            )
            if checkpoint is None:
                connection.execute(
                    "DELETE FROM cayu_checkpoints WHERE session_id = ?",
                    (session_id,),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO cayu_checkpoints (
                        session_id, state_json, updated_at,
                        pending_action_source_bytes,
                        pending_action_tool_call_count,
                        pending_action_flags,
                        pending_action_metrics_ready
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        state_json = excluded.state_json,
                        updated_at = excluded.updated_at,
                        pending_action_source_bytes =
                            excluded.pending_action_source_bytes,
                        pending_action_tool_call_count =
                            excluded.pending_action_tool_call_count,
                        pending_action_flags = excluded.pending_action_flags,
                        pending_action_metrics_ready =
                            excluded.pending_action_metrics_ready
                    """,
                    sqlite_records.checkpoint_row_values(
                        session_id,
                        checkpoint,
                        updated_at,
                    ),
                )
            touch_activity(connection, session_id, updated_at)
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    await run_write(statement)


async def materialize_deferred_interaction_input(
    run_write: SQLiteOperationRunner,
    session_id: str,
    *,
    interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ownership_clock: Callable[[], datetime],
    load_session: Callable[[str], Session | None],
    closure_owners: ClosureOwners,
    touch_activity: Callable[[sqlite3.Connection, str, datetime], None],
) -> bool:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = resolve_interaction_attribution(session_id, interaction_id)

    def statement(connection: sqlite3.Connection) -> bool:
        try:
            connection.execute("BEGIN IMMEDIATE")
            session = load_session(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            row = connection.execute(
                "SELECT interaction_id, source_messages_json "
                "FROM cayu_deferred_interaction_inputs WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                connection.commit()
                return False
            if row["interaction_id"] != interaction_id:
                raise RuntimeError("Deferred interaction input belongs to another interaction.")
            for owner in closure_owners((session_id,), connection=connection):
                _check_closure_lineage_owner(owner, (session_id,))
            deferred = deferred_interaction_input_from_storage_payload(
                row["interaction_id"],
                json.loads(row["source_messages_json"]),
            )
            messages = deferred.source_messages
            connection.executemany(
                "INSERT INTO cayu_transcript_messages "
                "(session_id, role, interaction_id, message_json, "
                "transcript_search_document) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        session_id,
                        str(message.role),
                        interaction_id,
                        sqlite_records.json_dumps(message.model_dump(mode="json")),
                        transcript_search_document(message),
                    )
                    for message in messages
                ],
            )
            connection.execute(
                "DELETE FROM cayu_deferred_interaction_inputs WHERE session_id = ?",
                (session_id,),
            )
            touch_activity(connection, session_id, ownership_clock())
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise

    return await run_write(statement)


async def load_deferred_interaction_input(
    run_read: SQLiteOperationRunner, session_id: str
) -> DeferredInteractionInput | None:
    session_id = require_clean_nonblank(session_id, "session_id")

    def query(connection: sqlite3.Connection) -> DeferredInteractionInput | None:
        if (
            connection.execute(
                "SELECT 1 FROM cayu_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            is None
        ):
            raise KeyError(f"Session not found: {session_id}")
        row = connection.execute(
            "SELECT interaction_id, source_messages_json "
            "FROM cayu_deferred_interaction_inputs WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return deferred_interaction_input_from_storage_payload(
            row["interaction_id"],
            json.loads(row["source_messages_json"]),
        )

    return await run_read(query)


async def append_transcript_messages_and_transform_checkpoint(
    run_write: SQLiteOperationRunner,
    session_id: str,
    messages: list[Message],
    checkpoint_transform: CheckpointTransform,
    *,
    interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ownership_clock: Callable[[], datetime],
    load_session: Callable[[str], Session | None],
    load_checkpoint: Callable[[str], dict[str, Any] | None],
    closure_owners: ClosureOwners,
    touch_activity: Callable[[sqlite3.Connection, str, datetime], None],
) -> None:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_id = resolve_interaction_attribution(session_id, interaction_id)
    copied_messages = copy_transcript_messages(messages)
    if checkpoint_transform is None:
        raise TypeError("checkpoint_transform is required.")

    def statement(connection: sqlite3.Connection) -> None:
        try:
            connection.execute("BEGIN IMMEDIATE")
            updated_at = ownership_clock()
            session = load_session(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            for owner in closure_owners((session_id,)):
                _check_closure_lineage_owner(owner, (session_id,))
            current_checkpoint = load_checkpoint(session_id)
            transformed = checkpoint_transform(
                session,
                _copy_checkpoint_for_transform(
                    current_checkpoint,
                    session_id=session_id,
                ),
            )
            if transformed is None:
                raise ValueError("Checkpoint transform must return a checkpoint.")
            transformed = (
                _checkpoint_transform_result_preserving_completion_result_event_publications(
                    current_checkpoint,
                    transformed,
                    session_id=session_id,
                )
            )
            touch_activity(connection, session_id, updated_at)
            if copied_messages:
                connection.executemany(
                    """
                    INSERT INTO cayu_transcript_messages (
                        session_id,
                        role,
                        interaction_id,
                        message_json,
                        transcript_search_document
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            session_id,
                            str(message.role),
                            interaction_id,
                            sqlite_records.json_dumps(message.model_dump(mode="json")),
                            transcript_search_document(message),
                        )
                        for message in copied_messages
                    ],
                )
            connection.execute(
                """
                INSERT INTO cayu_checkpoints (
                    session_id, state_json, updated_at,
                    pending_action_source_bytes,
                    pending_action_tool_call_count,
                    pending_action_flags,
                    pending_action_metrics_ready
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at,
                    pending_action_source_bytes = excluded.pending_action_source_bytes,
                    pending_action_tool_call_count = excluded.pending_action_tool_call_count,
                    pending_action_flags = excluded.pending_action_flags,
                    pending_action_metrics_ready = excluded.pending_action_metrics_ready
                """,
                sqlite_records.checkpoint_row_values(session_id, transformed, updated_at),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    await run_write(statement)


async def load_transcript(run_read: SQLiteOperationRunner, session_id: str) -> list[Message]:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")

    def query(connection: sqlite3.Connection) -> list[Message]:
        if not sqlite_records.session_exists(connection, session_id):
            raise KeyError(f"Session not found: {session_id}")
        rows = connection.execute(
            """
            SELECT message_json
            FROM cayu_transcript_messages
            WHERE session_id = ?
            ORDER BY sequence ASC
            """,
            (session_id,),
        ).fetchall()
        return [Message(**json.loads(row["message_json"])) for row in rows]

    from cayu.storage._session_access_records import sqlite_owner_read

    return await run_read(
        lambda connection: sqlite_owner_read(connection, access_bounds, session_id, query)
    )


async def load_transcript_snapshot(
    run_read: SQLiteOperationRunner, session_id: str
) -> TranscriptSnapshot:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")

    def query(connection: sqlite3.Connection) -> TranscriptSnapshot:
        rows = connection.execute(
            """
            SELECT session.transcript_seq,
                   transcript.session_order - 1 AS transcript_index,
                   transcript.interaction_id,
                   transcript.message_json
            FROM cayu_sessions AS session
            LEFT JOIN cayu_transcript_messages AS transcript
              ON transcript.session_id = session.id
            WHERE session.id = ?
            ORDER BY transcript.session_order ASC
            """,
            (session_id,),
        ).fetchall()
        if not rows:
            raise KeyError(f"Session not found: {session_id}")
        return TranscriptSnapshot(
            records=[
                TranscriptRecord(
                    index=row["transcript_index"],
                    interaction_id=row["interaction_id"],
                    message=Message(**json.loads(row["message_json"])),
                )
                for row in rows
                if row["transcript_index"] is not None
            ],
            cursor=int(rows[0]["transcript_seq"]),
        )

    from cayu.storage._session_access_records import sqlite_owner_read

    return await run_read(
        lambda connection: sqlite_owner_read(connection, access_bounds, session_id, query)
    )


async def load_transcript_cursor(run_read: SQLiteOperationRunner, session_id: str) -> int:
    session_id = require_clean_nonblank(session_id, "session_id")

    def query(connection: sqlite3.Connection) -> int:
        if not sqlite_records.session_exists(connection, session_id):
            raise KeyError(f"Session not found: {session_id}")
        return transcript_cursor(connection, session_id)

    return await run_read(query)


async def load_latest_transcript_message(
    run_read: SQLiteOperationRunner, session_id: str, *, role: MessageRole
) -> TranscriptRecord | None:
    session_id = require_clean_nonblank(session_id, "session_id")
    if not isinstance(role, MessageRole):
        raise TypeError("role must be a MessageRole.")

    def query_latest(connection: sqlite3.Connection) -> TranscriptRecord | None:
        if not sqlite_records.session_exists(connection, session_id):
            raise KeyError(f"Session not found: {session_id}")
        row = connection.execute(
            "SELECT session_order - 1 AS transcript_index, interaction_id, message_json "
            "FROM cayu_transcript_messages "
            "WHERE session_id = ? AND role = ? "
            "ORDER BY session_order DESC LIMIT 1",
            (session_id, str(role)),
        ).fetchone()
        if row is None:
            return None
        return TranscriptRecord(
            index=row["transcript_index"],
            interaction_id=row["interaction_id"],
            message=Message(**json.loads(row["message_json"])),
        )

    return await run_read(query_latest)


async def load_latest_transcript_text(
    run_read: SQLiteOperationRunner, session_id: str, *, role: MessageRole, max_chars: int
) -> tuple[str, bool] | None:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")
    if not isinstance(role, MessageRole):
        raise TypeError("role must be a MessageRole.")
    if type(max_chars) is not int:
        raise TypeError("max_chars must be an integer.")
    if not 1 <= max_chars <= LATEST_TRANSCRIPT_TEXT_MAX_CHARS:
        raise ValueError(f"max_chars must be between 1 and {LATEST_TRANSCRIPT_TEXT_MAX_CHARS}.")

    def query_latest(connection: sqlite3.Connection) -> tuple[str, bool] | None:
        row = connection.execute(
            """
            SELECT session.id,
                   transcript.sequence,
                   length(CAST(transcript.message_json AS BLOB))
            FROM cayu_sessions AS session
            LEFT JOIN cayu_transcript_messages AS transcript
              ON transcript.sequence = (
                  SELECT candidate.sequence
                  FROM cayu_transcript_messages AS candidate
                  WHERE candidate.session_id = session.id
                    AND candidate.role = ?
                  ORDER BY candidate.session_order DESC
                  LIMIT 1
              )
            WHERE session.id = ?
            """,
            (str(role), session_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Session not found: {session_id}")
        sequence = row[1]
        if sequence is None:
            return None
        if int(row[2]) > LATEST_TRANSCRIPT_TEXT_MAX_SOURCE_BYTES:
            raise TranscriptTextReadLimitExceeded(
                "Transcript message exceeds the bounded serialized-source limit."
            )
        projection = connection.execute(
            """
            WITH RECURSIVE
            source(message_json, part_count) AS (
                SELECT
                    message_json,
                    json_array_length(message_json, '$.content')
                FROM cayu_transcript_messages
                WHERE sequence = ?
            ),
            prefix(part_index, text_value, part_count) AS (
                SELECT 0, '', part_count
                FROM source
                UNION ALL
                SELECT
                    prefix.part_index + 1,
                    substr(
                        prefix.text_value ||
                        CASE
                            WHEN json_extract(
                                source.message_json,
                                '$.content[' || prefix.part_index || '].type'
                            ) = 'text'
                            THEN COALESCE(
                                json_extract(
                                    source.message_json,
                                    '$.content[' || prefix.part_index || '].text'
                                ),
                                ''
                            )
                            ELSE ''
                        END,
                        1,
                        ?
                    ),
                    prefix.part_count
                FROM prefix
                CROSS JOIN source
                WHERE prefix.part_index < prefix.part_count
                  AND prefix.part_index < ?
                  AND length(prefix.text_value) <= ?
            )
            SELECT text_value, part_index, part_count
            FROM prefix
            ORDER BY part_index DESC
            LIMIT 1
            """,
            (
                int(sequence),
                max_chars + 1,
                LATEST_TRANSCRIPT_TEXT_MAX_PARTS,
                max_chars,
            ),
        ).fetchone()
        if projection is None:
            raise TranscriptTextReadLimitExceeded(
                "Transcript message changed during its bounded text projection."
            )
        text_value = str(projection[0])
        if int(projection[1]) < int(projection[2]) and len(text_value) <= max_chars:
            raise TranscriptTextReadLimitExceeded(
                "Transcript message exceeds the bounded content-part inspection limit."
            )
        return text_value[:max_chars], len(text_value) > max_chars

    from cayu.storage._session_access_records import sqlite_owner_read

    return await run_read(
        lambda connection: sqlite_owner_read(connection, access_bounds, session_id, query_latest)
    )


async def load_transcript_window(
    run_read: SQLiteOperationRunner, session_id: str, *, start_index: int, limit: int
) -> TranscriptSnapshot:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(start_index) is not int:
        raise TypeError("start_index must be an integer.")
    if not 0 <= start_index <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError("start_index exceeds the durable integer limit.")
    if type(limit) is not int:
        raise TypeError("limit must be an integer.")
    if not 1 <= limit <= 5000:
        raise ValueError("limit must be between 1 and 5000.")

    def query(connection: sqlite3.Connection) -> TranscriptSnapshot:
        rows = connection.execute(
            """
            SELECT session.transcript_seq,
                   transcript.session_order - 1 AS transcript_index,
                   transcript.interaction_id,
                   transcript.message_json
            FROM cayu_sessions AS session
            LEFT JOIN cayu_transcript_messages AS transcript
              ON transcript.session_id = session.id
             AND transcript.session_order > ?
            WHERE session.id = ?
            ORDER BY transcript.session_order ASC
            LIMIT ?
            """,
            (start_index, session_id, limit),
        ).fetchall()
        if not rows:
            raise KeyError(f"Session not found: {session_id}")
        return TranscriptSnapshot(
            records=[
                TranscriptRecord(
                    index=row["transcript_index"],
                    interaction_id=row["interaction_id"],
                    message=Message(**json.loads(row["message_json"])),
                )
                for row in rows
                if row["transcript_index"] is not None
            ],
            cursor=int(rows[0]["transcript_seq"]),
        )

    from cayu.storage._session_access_records import sqlite_owner_read

    return await run_read(
        lambda connection: sqlite_owner_read(connection, access_bounds, session_id, query)
    )


async def query_transcript(
    run_read: SQLiteOperationRunner, query: TranscriptQuery
) -> TranscriptPage:
    from cayu.resource_access import current_data_bounds

    access_bounds = await current_data_bounds()
    query = copy_transcript_query(query)
    filters: list[str] = []
    filter_params: list[object] = []
    if query.role is not None:
        filters.append("role = ?")
        filter_params.append(str(query.role))
    if query.interaction_id is not None:
        filters.append("interaction_id = ?")
        filter_params.append(query.interaction_id)
    filter_clause = " AND " + " AND ".join(filters) if filters else ""

    def run_query(connection: sqlite3.Connection) -> TranscriptPage:
        if not sqlite_records.session_exists(connection, query.session_id):
            raise KeyError(f"Session not found: {query.session_id}")

        total_row = connection.execute(
            f"""
            SELECT COUNT(*) AS total_records
            FROM cayu_transcript_messages
            WHERE session_id = ?
            {filter_clause}
            """,
            [query.session_id, *filter_params],
        ).fetchone()
        total_records = int(total_row["total_records"])

        page_params: list[object] = [
            query.session_id,
            *filter_params,
            query.limit,
            query.offset,
        ]
        rows = connection.execute(
            f"""
            SELECT session_order - 1 AS transcript_index, interaction_id, message_json
            FROM cayu_transcript_messages
            WHERE session_id = ?
            {filter_clause}
            ORDER BY session_order ASC
            LIMIT ? OFFSET ?
            """,
            page_params,
        ).fetchall()
        records = [
            TranscriptRecord(
                index=row["transcript_index"],
                interaction_id=row["interaction_id"],
                message=Message(**json.loads(row["message_json"])),
            )
            for row in rows
        ]
        return TranscriptPage(
            records=filter_transcript_records(records, include_thinking=query.include_thinking),
            total_records=total_records,
        )

    from cayu.storage._session_access_records import sqlite_owner_read

    return await run_read(
        lambda connection: sqlite_owner_read(connection, access_bounds, query.session_id, run_query)
    )


async def search_transcript(
    run_read: SQLiteOperationRunner, query: TranscriptSearchQuery
) -> TranscriptSearchResult:
    query = copy_transcript_search_query(query)
    cursor = decode_transcript_search_cursor(query)
    roles = tuple(str(role) for role in query.roles)
    role_placeholders = ", ".join("?" for _ in roles)
    session_placeholders = ", ".join("?" for _ in query.session_ids)
    before_filter = "".join(
        " AND (transcript.session_id <> ? OR transcript.session_order <= ?)"
        for _ in query.before_transcript_indexes
    )
    before_params = [
        value
        for session_id, before_index in query.before_transcript_indexes.items()
        for value in (session_id, before_index)
    ]
    query_document = transcript_search_query_document(query.text)
    fetch_limit = query.max_records_scanned + 1

    def run_search(connection: sqlite3.Connection) -> TranscriptSearchResult:
        rows = connection.execute(
            f"""
            SELECT
                transcript.session_id,
                transcript.session_order - 1 AS transcript_index,
                transcript.interaction_id,
                transcript.message_json,
                transcript.transcript_search_document
            FROM cayu_transcript_messages_fts
            JOIN cayu_transcript_messages AS transcript
              ON transcript.sequence = cayu_transcript_messages_fts.rowid
            WHERE cayu_transcript_messages_fts MATCH ?
              AND transcript.session_id IN ({session_placeholders})
              AND transcript.role IN ({role_placeholders})
              {before_filter}
            LIMIT ?
            """,
            [
                _search_expression(query),
                *query.session_ids,
                *roles,
                *before_params,
                fetch_limit,
            ],
        ).fetchall()
        if len(rows) > query.max_records_scanned:
            return TranscriptSearchResult(
                query=query,
                matched_records_examined=query.max_records_scanned,
                truncated=True,
                coverage_complete=False,
            )

        candidates: list[tuple[int, sqlite3.Row, Message]] = []
        for row in rows:
            message = Message.model_validate(json.loads(row["message_json"]))
            document = transcript_search_document(message)
            if row["transcript_search_document"] != document:
                raise RuntimeError("SQLite transcript search document is inconsistent.")
            score = transcript_search_document_score(document, query_document)
            if score <= 0:
                raise RuntimeError("SQLite transcript search index is inconsistent.")
            candidates.append((score, row, message))
        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1]["session_id"],
                -item[1]["transcript_index"],
            )
        )
        if cursor is not None:
            candidates = [
                candidate
                for candidate in candidates
                if transcript_search_position_after_cursor(
                    raw_score=candidate[0],
                    session_id=candidate[1]["session_id"],
                    transcript_index=candidate[1]["transcript_index"],
                    cursor=cursor,
                )
            ]

        hits: list[TranscriptSearchHit] = []
        remaining_bytes = query.max_bytes
        truncated = False
        continuation_available = False
        for candidate_index, (score, row, message) in enumerate(candidates):
            if len(hits) >= query.limit:
                truncated = True
                continuation_available = True
                break
            hit = transcript_search_hit_from_message(
                session_id=row["session_id"],
                transcript_index=row["transcript_index"],
                interaction_id=row["interaction_id"],
                message=message,
                max_text_bytes=remaining_bytes,
                raw_score=float(score),
            )
            if hit is None:
                truncated = True
                continuation_available = bool(hits)
                break
            hits.append(hit)
            remaining_bytes -= len(hit.text.encode("utf-8"))
            has_remaining_candidate = candidate_index + 1 < len(candidates)
            if not hit.text_complete:
                truncated = True
                continuation_available = has_remaining_candidate
                break
            if remaining_bytes == 0:
                truncated = has_remaining_candidate
                continuation_available = truncated
                break
        next_cursor = (
            encode_transcript_search_cursor(
                query,
                raw_score=int(hits[-1].raw_score or 0),
                session_id=hits[-1].session_id,
                transcript_index=hits[-1].transcript_index,
            )
            if continuation_available and hits
            else None
        )
        return TranscriptSearchResult(
            query=query,
            hits=tuple(hits),
            matched_records_examined=len(rows),
            truncated=truncated,
            coverage_complete=True,
            next_cursor=next_cursor,
        )

    return await run_read(run_search)
