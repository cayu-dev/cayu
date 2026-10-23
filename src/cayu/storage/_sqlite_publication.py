"""Native publication transactions composed from explicit storage capabilities."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime
from typing import Any

from cayu._validation import MAX_DURABLE_JSON_ENVELOPE_NESTING, copy_durable_json_object
from cayu.events import Event
from cayu.messages import Message
from cayu.sessions._checkpoint_mutations import (
    SessionRuntimePublicationConflict,
    _apply_runtime_publication_checkpoint_mutation,
)
from cayu.sessions._checkpoint_preservation import (
    _checkpoint_transform_result_preserving_completion_result_event_publications,
    _copy_checkpoint_for_transform,
    _replace_checkpoint_preserving_completion_result_event_publications,
)
from cayu.sessions._closure_records import _check_closure_lineage_owner
from cayu.sessions._invocation_scope import _assert_session_run_epoch
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    SessionModelCompletionStageConflict,
)
from cayu.sessions._model_stage_rules import (
    _model_completion_stage_storage_identity,
    _model_completion_stage_winner_record,
    _ModelCompletionStagePromotionContext,
    _reconstruct_active_model_completion_stage,
    _reconstruct_active_model_completion_stage_record,
    _reconstruct_model_completion_stage,
    _replay_promoted_model_completion_stage,
    _validate_model_completion_active_marker_for_promotion,
    _validate_model_completion_promotion_replay_active_marker,
    _validate_model_completion_stage_release,
)
from cayu.sessions._operation_records import (
    SessionOperationPublication,
    _apply_runtime_publication_operation_record_mutations,
    _validate_session_operation_record_keys,
)
from cayu.sessions._runtime_publication import (
    RuntimePublicationReceipt,
    RuntimePublicationResult,
    SessionOperationTransform,
    StoreTimeSessionOperationTransform,
    _runtime_publication_referenced_event_ids,
    _tool_round_lifecycle_event_limit,
    _validate_runtime_publication_event_references,
)
from cayu.sessions._runtime_publication_rules import (
    _build_runtime_publication_receipt,
    _next_runtime_publication_timestamp,
    _PreparedRuntimePublication,
    _reconstruct_runtime_publication_receipt,
    _runtime_publication_receipt_record,
    _validate_runtime_publication_durable_material,
    _validate_runtime_publication_replay_receipt,
)
from cayu.sessions._tool_round_publication_rules import (
    _TOOL_ROUND_LIFECYCLE_EVENT_TYPES,
    _tool_lifecycle_publication_identity,
    _validate_tool_round_checkpoint_mutation,
    _validate_tool_round_publication,
)
from cayu.sessions._user_input_publication import _validate_user_input_checkpoint_mutation
from cayu.sessions.authority import SessionRunFenced, _runtime_publication_json_equal
from cayu.sessions.event_delivery import _copy_session_event_batch
from cayu.sessions.exports import _require_session_export_target
from cayu.sessions.records import (
    CheckpointTransform,
    Session,
    SessionStatus,
    SessionStatusConflict,
    StoreTimeCheckpointTransform,
    TranscriptRecord,
    _validate_status_set,
)
from cayu.sessions.transcript_queries import transcript_search_document
from cayu.storage import _sqlite_event_delivery as event_delivery_ops
from cayu.storage import _sqlite_model_stage as model_stage_ops
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage import _sqlite_session_queries as session_queries
from cayu.storage import _sqlite_transcript as transcript_ops
from cayu.storage._sqlite_connection import SQLiteOperationRunner
from cayu.storage._sqlite_event_publication import EventWriter
from cayu.storage._sqlite_transcript import ClosureOwners


def decode_record(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication receipt is malformed or conflicts with its key."
        ) from exc


async def load_receipt_record(
    session_id: str, storage_key: str, publication_id: str, *, run_read: SQLiteOperationRunner
) -> dict[str, Any] | None:
    def query(connection: sqlite3.Connection) -> dict[str, Any] | None:
        connection.execute("BEGIN")
        try:
            if not sqlite_records.session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            row = connection.execute(
                "SELECT record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key = ?",
                (session_id, storage_key),
            ).fetchone()
            if row is None:
                return None
            record = decode_record(row["record_json"])
            receipt = _reconstruct_runtime_publication_receipt(
                record,
                storage_key=storage_key,
                session_id=session_id,
                publication_id=publication_id,
            )
            validate_material(connection, receipt)
            return record
        finally:
            connection.rollback()

    return await run_read(query)


def validate_material(connection: sqlite3.Connection, receipt: RuntimePublicationReceipt) -> None:
    try:
        transcript_rows = connection.execute(
            "SELECT interaction_id, message_json FROM cayu_transcript_messages "
            "WHERE session_id = ? AND session_order > ? AND session_order <= ? "
            "ORDER BY session_order ASC",
            (
                receipt.session_id,
                receipt.transcript_start_cursor,
                receipt.transcript_end_cursor,
            ),
        ).fetchall()
        transcript = [Message(**json.loads(row["message_json"])) for row in transcript_rows]
        transcript_interaction_ids = [row["interaction_id"] for row in transcript_rows]

        referenced_event_ids = _runtime_publication_referenced_event_ids(receipt.referenced_events)
        requested_event_ids = tuple(
            dict.fromkeys((*receipt.appended_event_ids, *referenced_event_ids))
        )
        events_by_id: dict[str, Event] = {}
        if requested_event_ids:
            placeholders = ", ".join("?" for _ in requested_event_ids)
            rows = connection.execute(
                f"SELECT {', '.join(sqlite_records.EVENT_COLUMN_NAMES)} FROM cayu_events "
                f"WHERE session_id = ? AND event_id IN ({placeholders})",
                (receipt.session_id, *requested_event_ids),
            ).fetchall()
            events_by_id = {row["event_id"]: sqlite_records.event_from_row(row) for row in rows}
        _validate_runtime_publication_durable_material(
            receipt,
            transcript_messages=transcript,
            transcript_interaction_ids=transcript_interaction_ids,
            appended_events=(
                events_by_id[event_id]
                for event_id in receipt.appended_event_ids
                if event_id in events_by_id
            ),
            durable_referenced_events=(
                events_by_id[event_id]
                for event_id in referenced_event_ids
                if event_id in events_by_id
            ),
        )
    except SessionRuntimePublicationConflict:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication material is malformed."
        ) from exc


async def publish_runtime(
    prepared: _PreparedRuntimePublication,
    *,
    _model_completion_stage: _ModelCompletionStagePromotionContext | None = None,
    run_write: SQLiteOperationRunner,
    load_session: Callable[[str], Session | None],
    load_checkpoint: Callable[[str], dict[str, Any] | None],
    closure_owners: ClosureOwners,
    first_existing_event_id: Callable[[sqlite3.Connection, str, list[str]], str | None],
) -> RuntimePublicationResult:
    from cayu.sessions.pending_actions import (
        pending_action_event_storage_values,
        pending_action_lookup_key,
    )

    session_id = prepared.session_id
    request = prepared.request
    checkpoint_codec = prepared.checkpoint_codec
    checkpoint_decode = None if checkpoint_codec is None else checkpoint_codec.decode

    def statement(connection: sqlite3.Connection) -> RuntimePublicationResult:
        try:
            connection.execute("BEGIN IMMEDIATE")
            loaded = load_session(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")

            locked_stage = None
            active_record = None
            winner_record = None
            if _model_completion_stage is not None:
                stage_rows = connection.execute(
                    "SELECT idempotency_key, record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key IN (?, ?, ?, ?)",
                    (
                        session_id,
                        _model_completion_stage.preparation_storage_key,
                        _model_completion_stage.terminal_storage_key,
                        _model_completion_stage.active_storage_key,
                        _model_completion_stage.winner_storage_key,
                    ),
                ).fetchall()
                stage_records = {
                    row["idempotency_key"]: model_stage_ops.decode_stage_record(row["record_json"])
                    for row in stage_rows
                }
                locked_stage = _reconstruct_model_completion_stage(
                    stage_records.get(_model_completion_stage.preparation_storage_key),
                    stage_records.get(_model_completion_stage.terminal_storage_key),
                    session_id=session_id,
                    stage_id=_model_completion_stage.stage_id,
                    preparation_storage_key=(_model_completion_stage.preparation_storage_key),
                    terminal_storage_key=_model_completion_stage.terminal_storage_key,
                )
                if locked_stage is None:
                    raise KeyError(
                        f"Model-completion stage not found: {_model_completion_stage.stage_id}"
                    )
                if (
                    locked_stage.completion_digest != _model_completion_stage.completion_digest
                    or locked_stage.publication is None
                    or not _runtime_publication_json_equal(
                        locked_stage.publication.model_dump(mode="json"),
                        prepared.request.model_dump(mode="json"),
                        max_nesting=MAX_DURABLE_JSON_ENVELOPE_NESTING,
                    )
                ):
                    raise SessionModelCompletionStageConflict(
                        "Model-completion stage changed before atomic promotion."
                    )
                active_record = stage_records.get(_model_completion_stage.active_storage_key)
                winner_record = stage_records.get(_model_completion_stage.winner_storage_key)

            receipt_row = connection.execute(
                "SELECT record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key = ?",
                (session_id, prepared.storage_key),
            ).fetchone()
            if receipt_row is not None:
                receipt_record = decode_record(receipt_row["record_json"])
                if locked_stage is not None:
                    replay_active = None
                    if active_record is not None:
                        active_marker = _reconstruct_active_model_completion_stage_record(
                            active_record,
                            session_id=session_id,
                        )
                        _, _, active_preparation_key, active_terminal_key = (
                            _model_completion_stage_storage_identity(
                                session_id,
                                active_marker.stage_id,
                            )
                        )
                        active_rows = connection.execute(
                            "SELECT idempotency_key, record_json "
                            "FROM cayu_session_operations "
                            "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                            (
                                session_id,
                                active_preparation_key,
                                active_terminal_key,
                            ),
                        ).fetchall()
                        active_records = {
                            row["idempotency_key"]: (
                                model_stage_ops.decode_stage_record(row["record_json"])
                            )
                            for row in active_rows
                        }
                        replay_active = _reconstruct_active_model_completion_stage(
                            active_record,
                            active_records.get(active_preparation_key),
                            active_records.get(active_terminal_key),
                            session_id=session_id,
                        )
                    _validate_model_completion_promotion_replay_active_marker(
                        replay_active,
                        locked_stage,
                    )
                    receipt = _reconstruct_runtime_publication_receipt(
                        receipt_record,
                        storage_key=prepared.storage_key,
                        session_id=session_id,
                        publication_id=request.publication_id,
                    )
                    validate_material(connection, receipt)
                    result = _replay_promoted_model_completion_stage(
                        session=loaded,
                        stage=locked_stage,
                        receipt_record=receipt_record,
                        winner_record=winner_record,
                    )
                    connection.rollback()
                    return result
                receipt = _reconstruct_runtime_publication_receipt(
                    receipt_record,
                    storage_key=prepared.storage_key,
                    session_id=session_id,
                    publication_id=request.publication_id,
                    request_digest=prepared.request_digest,
                )
                _validate_runtime_publication_replay_receipt(receipt, prepared)
                validate_material(connection, receipt)
                connection.rollback()
                return RuntimePublicationResult(
                    session=loaded.model_copy(deep=True),
                    receipt=receipt,
                    replayed=True,
                )

            for owner in closure_owners((session_id,)):
                _check_closure_lineage_owner(owner, (session_id,))

            operation_mutation_records: dict[str, dict[str, Any]] = {}
            if request.operation_record_mutations:
                mutation_keys = tuple(
                    mutation.key for mutation in request.operation_record_mutations
                )
                placeholders = ", ".join("?" for _ in mutation_keys)
                mutation_rows = connection.execute(
                    "SELECT idempotency_key, record_json FROM cayu_session_operations "
                    f"WHERE session_id = ? AND idempotency_key IN ({placeholders})",
                    (session_id, *mutation_keys),
                ).fetchall()
                current_mutation_records = {
                    row["idempotency_key"]: json.loads(row["record_json"]) for row in mutation_rows
                }
                operation_mutation_records = _apply_runtime_publication_operation_record_mutations(
                    request.operation_record_mutations,
                    current_mutation_records,
                )

            if request.argument_continuity is not None:
                from cayu.sessions._argument_continuity import STORAGE_KEY, append_record

                private_row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, STORAGE_KEY),
                ).fetchone()
                operation_mutation_records[STORAGE_KEY] = append_record(
                    None if private_row is None else json.loads(private_row["record_json"]),
                    continuity=request.argument_continuity,
                    request_digest=prepared.request_digest,
                    session=loaded,
                    messages=request.transcript_messages,
                )

            if locked_stage is not None:
                assert _model_completion_stage is not None
                if winner_record is not None:
                    raise SessionModelCompletionStageConflict(
                        "A model-completion winner exists without its runtime publication receipt."
                    )
                if active_record is not None:
                    active_marker = _reconstruct_active_model_completion_stage_record(
                        active_record,
                        session_id=session_id,
                    )
                    if active_marker.stage_id != locked_stage.stage_id:
                        raise SessionModelCompletionStageConflict(
                            "The model-completion stage was superseded before promotion."
                        )
                active = _reconstruct_active_model_completion_stage(
                    active_record,
                    stage_records.get(_model_completion_stage.preparation_storage_key),
                    stage_records.get(_model_completion_stage.terminal_storage_key),
                    session_id=session_id,
                )
                _validate_model_completion_active_marker_for_promotion(
                    active,
                    locked_stage,
                )

            _assert_session_run_epoch(session_id, loaded)
            if (
                prepared.expected_statuses is not None
                and loaded.status not in prepared.expected_statuses
            ):
                raise SessionStatusConflict(
                    f"Session status is not eligible for runtime publication: {loaded.status}"
                )
            if (
                prepared.expected_run_epoch is not None
                and loaded.run_epoch != prepared.expected_run_epoch
            ):
                raise SessionRunFenced(
                    "Session source run epoch is stale: expected "
                    f"{prepared.expected_run_epoch}, current {loaded.run_epoch}."
                )
            transcript_start_cursor = transcript_ops.transcript_cursor(connection, session_id)
            if (
                prepared.expected_transcript_cursor is not None
                and transcript_start_cursor != prepared.expected_transcript_cursor
            ):
                raise ValueError(
                    "Session source transcript cursor is stale: expected "
                    f"{prepared.expected_transcript_cursor}, "
                    f"current {transcript_start_cursor}."
                )

            appended_event_ids = {event.id for event in request.events}
            referenced_event_ids = set(
                _runtime_publication_referenced_event_ids(request.referenced_events)
            )
            if appended_event_ids & referenced_event_ids:
                raise ValueError("Appended and referenced runtime publication events overlap.")
            durable_referenced_events: dict[str, Event] = {}
            if referenced_event_ids:
                placeholders = ", ".join("?" for _ in referenced_event_ids)
                rows = connection.execute(
                    f"SELECT {', '.join(sqlite_records.EVENT_COLUMN_NAMES)} FROM cayu_events "
                    f"WHERE session_id = ? AND event_id IN ({placeholders})",
                    (session_id, *referenced_event_ids),
                ).fetchall()
                durable_referenced_events = {
                    row["event_id"]: sqlite_records.event_from_row(row) for row in rows
                }
            _validate_runtime_publication_event_references(
                request.referenced_events,
                durable_referenced_events,
                interaction_id=request.interaction_id,
            )
            stored_checkpoint = load_checkpoint(session_id)
            current_checkpoint = (
                stored_checkpoint
                if checkpoint_decode is None
                else checkpoint_decode(loaded, stored_checkpoint)
            )
            _validate_user_input_checkpoint_mutation(
                request,
                current_checkpoint,
                session_id=session_id,
                session_instance_id=loaded.instance_id,
                current_run_epoch=loaded.run_epoch,
                durable_events_by_id=durable_referenced_events,
            )
            _validate_tool_round_checkpoint_mutation(
                request,
                current_checkpoint,
            )
            durable_tool_events: list[Event] = []
            tool_round_identity = _tool_lifecycle_publication_identity(request)
            if tool_round_identity is not None:
                execution_identity, tool_call_ids = tool_round_identity
                lookup_keys = tuple(
                    pending_action_lookup_key(tool_call_id) for tool_call_id in tool_call_ids
                )
                lifecycle_event_types = tuple(
                    sorted(str(event_type) for event_type in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES)
                )
                event_type_placeholders = ", ".join("?" for _ in lifecycle_event_types)
                rows = connection.execute(
                    f"SELECT {', '.join(sqlite_records.EVENT_COLUMN_NAMES)} FROM cayu_events "
                    "INDEXED BY idx_cayu_events_pending_action_lookup "
                    f"WHERE session_id = ? AND pending_action_lookup_key IN "
                    "(SELECT value FROM json_each(?)) AND event_type IN "
                    f"({event_type_placeholders}) AND "
                    f"({session_queries.PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
                    "AND (json_extract(payload_json, '$.tool_round_id') = ? "
                    "OR (json_extract(payload_json, '$.model_step_id') = ? "
                    "AND json_extract(payload_json, '$.model_attempt_id') = ?) "
                    "OR cayu_is_execution_unit_id("
                    "json_extract(payload_json, '$.tool_round_id'), 'tool_round_id') = 0 "
                    "OR cayu_is_execution_unit_id("
                    "json_extract(payload_json, '$.model_step_id'), 'model_step_id') = 0 "
                    "OR cayu_is_execution_unit_id("
                    "json_extract(payload_json, '$.model_attempt_id'), 'model_attempt_id') = 0) "
                    "ORDER BY sequence ASC LIMIT ?",
                    (
                        session_id,
                        json.dumps(lookup_keys),
                        *lifecycle_event_types,
                        execution_identity.tool_round_id,
                        execution_identity.model_step_id,
                        execution_identity.model_attempt_id,
                        _tool_round_lifecycle_event_limit(tool_call_ids) + 1,
                    ),
                ).fetchall()
                if len(rows) > _tool_round_lifecycle_event_limit(tool_call_ids):
                    raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
                durable_tool_events = [sqlite_records.event_from_row(row) for row in rows]
            _validate_tool_round_publication(
                request,
                durable_referenced_events,
                durable_tool_events=durable_tool_events,
            )
            existing_event_id = first_existing_event_id(
                connection,
                session_id,
                [event.id for event in request.events],
            )
            if existing_event_id is not None:
                raise ValueError(
                    f"Event already exists for session {session_id}: {existing_event_id}"
                )

            if checkpoint_codec is None:
                checkpoint = _apply_runtime_publication_checkpoint_mutation(
                    request.mutation,
                    current_checkpoint,
                )
                stored_target_checkpoint = checkpoint
            else:
                checkpoint = checkpoint_codec.apply_mutation(
                    loaded,
                    stored_checkpoint,
                    request.mutation,
                )
                stored_target_checkpoint = checkpoint_codec.encode(loaded, checkpoint)

            transcript_rows = [
                (
                    session_id,
                    str(message.role),
                    request.interaction_id,
                    sqlite_records.json_dumps(message_payload),
                    transcript_search_document(message),
                )
                for message, message_payload in zip(
                    request.transcript_messages,
                    prepared.transcript_payloads,
                    strict=True,
                )
            ]
            event_rows = []
            for event, event_payload in zip(
                request.events,
                prepared.event_payloads,
                strict=True,
            ):
                lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                    event
                )
                event_rows.append(
                    (
                        session_id,
                        event.id,
                        event.interaction_id,
                        str(event.type),
                        sqlite_records.format_datetime(event.timestamp),
                        event.agent_name,
                        event.environment_name,
                        event.workflow_name,
                        event.tool_name,
                        sqlite_records.json_dumps(event_payload["payload"]),
                        lookup_key,
                        projection,
                        projection_bytes,
                    )
                )

            published_at = _next_runtime_publication_timestamp(loaded)
            checkpoint_values = (
                None
                if stored_target_checkpoint is None or not request.mutation.operations
                else sqlite_records.checkpoint_row_values(
                    session_id,
                    stored_target_checkpoint,
                    published_at,
                )
            )
            receipt = _build_runtime_publication_receipt(
                prepared,
                source_session=loaded,
                checkpoint=stored_target_checkpoint,
                transcript_start_cursor=transcript_start_cursor,
                published_at=published_at,
            )
            receipt_json = sqlite_records.json_dumps(_runtime_publication_receipt_record(receipt))
            formatted_published_at = sqlite_records.format_datetime(published_at)

            if transcript_rows:
                connection.executemany(
                    """
                        INSERT INTO cayu_transcript_messages (
                            session_id, role, interaction_id, message_json,
                            transcript_search_document
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                    transcript_rows,
                )
            if checkpoint_values is not None:
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
                    checkpoint_values,
                )
            if event_rows:
                connection.executemany(
                    """
                        INSERT INTO cayu_events (
                            session_id, event_id, interaction_id, event_type, timestamp,
                            agent_name, environment_name, workflow_name, tool_name,
                            payload_json, pending_action_lookup_key,
                            pending_action_projection_json, pending_action_projection_bytes
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                    event_rows,
                )
                event_delivery_ops.enqueue_persisted_event_side_effects(
                    connection,
                    session_id,
                    request.events,
                )
            if operation_mutation_records:
                connection.executemany(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                    "record_json = excluded.record_json, updated_at = excluded.updated_at",
                    [
                        (
                            session_id,
                            key,
                            sqlite_records.json_dumps(record),
                            formatted_published_at,
                        )
                        for key, record in operation_mutation_records.items()
                    ],
                )
            connection.execute(
                """
                    INSERT INTO cayu_session_operations (
                        session_id, idempotency_key, record_json, updated_at
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                (
                    session_id,
                    prepared.storage_key,
                    receipt_json,
                    formatted_published_at,
                ),
            )
            if locked_stage is not None and _model_completion_stage is not None:
                winner = _model_completion_stage_winner_record(
                    locked_stage,
                    receipt=receipt,
                )
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        session_id,
                        _model_completion_stage.winner_storage_key,
                        sqlite_records.json_dumps(winner),
                        formatted_published_at,
                    ),
                )
                active_delete = connection.execute(
                    "DELETE FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, _model_completion_stage.active_storage_key),
                )
                if active_delete.rowcount != 1:
                    raise SessionModelCompletionStageConflict(
                        "The active model-completion marker changed before commit."
                    )
            cursor = connection.execute(
                """
                    UPDATE cayu_sessions
                    SET updated_at = ?, last_activity_at = ?
                    WHERE id = ?
                    """,
                (formatted_published_at, formatted_published_at, session_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Session not found: {session_id}")
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            existing_event_id = first_existing_event_id(
                connection,
                session_id,
                [event.id for event in request.events],
            )
            if existing_event_id is not None:
                raise ValueError(
                    f"Event already exists for session {session_id}: {existing_event_id}"
                ) from exc
            receipt_row = connection.execute(
                "SELECT 1 FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key = ?",
                (session_id, prepared.storage_key),
            ).fetchone()
            if receipt_row is not None:
                raise SessionRuntimePublicationConflict(
                    "Runtime publication receipt was inserted concurrently."
                ) from exc
            raise
        except BaseException:
            connection.rollback()
            raise
        updated = loaded.model_copy(
            update={
                "updated_at": published_at,
                "last_activity_at": published_at,
            },
            deep=True,
        )
        return RuntimePublicationResult(
            session=updated,
            receipt=receipt,
            replayed=False,
        )

    return await run_write(statement)


async def publish_checkpoint_events(
    session_id: str,
    *,
    checkpoint_transform: CheckpointTransform | None,
    store_time_checkpoint_transform: StoreTimeCheckpointTransform | None = None,
    operation_idempotency_key: str | None,
    operation_transform: SessionOperationTransform | None,
    store_time_operation_transform: StoreTimeSessionOperationTransform | None,
    operation_commit_guard: Callable[[], None] | None,
    operation_commit_time_guard: Callable[[datetime], None] | None,
    events: list[Event],
    expected_statuses: set[SessionStatus] | None,
    expected_run_epoch: int | None,
    expected_transcript_cursor: int | None,
    context_view_compaction_cursor: int | None = None,
    preserve_completion_result_publications: bool = True,
    run_write: SQLiteOperationRunner,
    load_session: Callable[[str], Session | None],
    load_checkpoint: Callable[[str], dict[str, Any] | None],
    store_now: Callable[[], datetime],
    closure_owners: ClosureOwners,
    first_existing_event_id: Callable[[sqlite3.Connection, str, list[str]], str | None],
    publish_budget_identities: Callable[[sqlite3.Connection, list[Event]], None],
    record_terminal_receipts: EventWriter,
    touch_activity: Callable[[sqlite3.Connection, str, datetime], None],
    validate_compaction: Callable[[str, int], None],
) -> Session:
    from cayu.sessions.pending_actions import pending_action_event_storage_values

    session_id, copied_events = _copy_session_event_batch(session_id, events)
    transform_count = sum(
        transform is not None
        for transform in (
            checkpoint_transform,
            store_time_checkpoint_transform,
            operation_transform,
            store_time_operation_transform,
        )
    )
    if transform_count != 1:
        raise TypeError("Exactly one checkpoint publication transform is required.")
    if (
        operation_transform is not None or store_time_operation_transform is not None
    ) and operation_idempotency_key is None:
        raise TypeError("operation_idempotency_key is required.")
    from cayu.collaboration import _session_export_store as session_exports

    if operation_idempotency_key is not None:
        session_exports.require_operation_key_access(operation_idempotency_key, read=False)
        _require_session_export_target(session_id, operation_idempotency_key)
    allowed_statuses = (
        None
        if expected_statuses is None
        else _validate_status_set(expected_statuses, "expected_statuses")
    )

    def statement(connection: sqlite3.Connection) -> Session:
        try:
            connection.execute("BEGIN IMMEDIATE")
            updated_at = store_now()
            loaded = load_session(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, loaded)
            for owner in closure_owners((session_id,)):
                _check_closure_lineage_owner(owner, (session_id,))
            if allowed_statuses is not None and loaded.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status is not eligible for checkpoint publication: {loaded.status}"
                )
            if expected_run_epoch is not None and loaded.run_epoch != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {loaded.run_epoch}."
                )
            current_cursor = transcript_ops.transcript_cursor(connection, session_id)
            if (
                expected_transcript_cursor is not None
                and current_cursor != expected_transcript_cursor
            ):
                raise ValueError(
                    "Session source transcript cursor is stale: expected "
                    f"{expected_transcript_cursor}, current {current_cursor}."
                )
            if context_view_compaction_cursor is not None:
                validate_compaction(session_id, context_view_compaction_cursor)
            current_checkpoint = load_checkpoint(session_id)
            callback_checkpoint = _copy_checkpoint_for_transform(
                current_checkpoint,
                session_id=session_id,
            )
            operation_records: dict[str, dict[str, Any]] = {}
            model_completion_stage_release = None
            if operation_transform is not None or store_time_operation_transform is not None:
                operation_row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, operation_idempotency_key),
                ).fetchone()
                current_operation = (
                    None if operation_row is None else json.loads(operation_row["record_json"])
                )
                if operation_transform is not None:
                    publication = operation_transform(
                        loaded,
                        callback_checkpoint,
                        current_operation,
                    )
                else:
                    assert store_time_operation_transform is not None
                    publication = store_time_operation_transform(
                        loaded,
                        callback_checkpoint,
                        current_operation,
                        updated_at,
                    )
                if type(publication) is not SessionOperationPublication:
                    raise TypeError(
                        "Session operation transform must return a SessionOperationPublication."
                    )
                transformed = copy_durable_json_object(
                    publication.checkpoint,
                    "checkpoint",
                )
                transformed = (
                    _checkpoint_transform_result_preserving_completion_result_event_publications(
                        current_checkpoint,
                        transformed,
                        session_id=session_id,
                    )
                )
                operation_records = copy_durable_json_object(
                    publication.operation_records,
                    "operation_records",
                )
                model_completion_stage_release = publication.model_completion_stage_release
                _validate_session_operation_record_keys(operation_records)
                indices = session_exports.selected_transcript_indices(session_id=session_id)
                if (
                    indices
                    or session_exports.checkpoint_visible(session_id=session_id)
                    or any(
                        key.startswith(session_exports.OPERATION_PREFIX)
                        for key in operation_records
                    )
                ):
                    selected_rows: tuple[TranscriptRecord, ...] = ()
                    if indices:
                        placeholders = ",".join("?" for _ in indices)
                        rows = connection.execute(
                            "SELECT session_order - 1 AS transcript_index, "
                            "interaction_id, message_json FROM cayu_transcript_messages "
                            f"WHERE session_id = ? AND session_order IN ({placeholders}) "
                            "ORDER BY session_order",
                            (session_id, *(index + 1 for index in indices)),
                        ).fetchall()
                        selected_rows = tuple(
                            TranscriptRecord(
                                index=row["transcript_index"],
                                interaction_id=row["interaction_id"],
                                message=Message(**json.loads(row["message_json"])),
                            )
                            for row in rows
                        )
                    session_exports.validate_publication(
                        session=loaded,
                        current_checkpoint=current_checkpoint,
                        proposed_checkpoint=transformed,
                        operation_records=operation_records,
                        selected_transcript_rows=selected_rows,
                        events=copied_events,
                    )
            else:
                if checkpoint_transform is not None:
                    transformed = checkpoint_transform(loaded, callback_checkpoint)
                else:
                    assert store_time_checkpoint_transform is not None
                    transformed = store_time_checkpoint_transform(
                        loaded,
                        callback_checkpoint,
                        updated_at,
                    )
                if transformed is None:
                    raise ValueError("Checkpoint transform must return a checkpoint.")
                transformed = copy_durable_json_object(transformed, "checkpoint")
                transformed = _replace_checkpoint_preserving_completion_result_event_publications(
                    current_checkpoint,
                    transformed,
                    preserve_completion_result_publications=(
                        preserve_completion_result_publications
                    ),
                    session_id=session_id,
                )
            event_rows = []
            for event in copied_events:
                lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                    event
                )
                event_rows.append(
                    (
                        session_id,
                        event.id,
                        event.interaction_id,
                        str(event.type),
                        sqlite_records.format_datetime(event.timestamp),
                        event.agent_name,
                        event.environment_name,
                        event.workflow_name,
                        event.tool_name,
                        sqlite_records.json_dumps(event.payload),
                        lookup_key,
                        projection,
                        projection_bytes,
                    )
                )
            publish_budget_identities(connection, copied_events)
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
            if operation_records:
                connection.executemany(
                    """
                        INSERT INTO cayu_session_operations (
                            session_id, idempotency_key, record_json, updated_at
                        )
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(session_id, idempotency_key) DO UPDATE SET
                            record_json = excluded.record_json,
                            updated_at = excluded.updated_at
                        """,
                    [
                        (
                            session_id,
                            key,
                            sqlite_records.json_dumps(record),
                            sqlite_records.format_datetime(updated_at),
                        )
                        for key, record in operation_records.items()
                    ],
                )
            if model_completion_stage_release is not None:
                _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
                    session_id,
                    model_completion_stage_release.stage_id,
                )
                stage_rows = connection.execute(
                    "SELECT idempotency_key, record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key IN (?, ?, ?)",
                    (
                        session_id,
                        MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                        preparation_key,
                        terminal_key,
                    ),
                ).fetchall()
                stage_records = {
                    row["idempotency_key"]: model_stage_ops.decode_stage_record(row["record_json"])
                    for row in stage_rows
                }
                _validate_model_completion_stage_release(
                    session=loaded,
                    active_record=stage_records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    preparation_record=stage_records.get(preparation_key),
                    terminal_record=stage_records.get(terminal_key),
                    release=model_completion_stage_release,
                )
                deleted = connection.execute(
                    "DELETE FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                )
                if deleted.rowcount != 1:
                    raise SessionModelCompletionStageConflict(
                        "The active model-completion stage changed during disposition."
                    )
            if event_rows:
                connection.executemany(
                    """
                        INSERT INTO cayu_events (
                            session_id, event_id, interaction_id, event_type, timestamp,
                            agent_name, environment_name, workflow_name, tool_name,
                            payload_json, pending_action_lookup_key,
                            pending_action_projection_json, pending_action_projection_bytes
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                    event_rows,
                )
                record_terminal_receipts(
                    connection, session_id, copied_events, activity_at=updated_at
                )
                event_delivery_ops.enqueue_persisted_event_side_effects(
                    connection,
                    session_id,
                    copied_events,
                )
            if operation_commit_guard is not None:
                operation_commit_guard()
            activity_at = store_now() if operation_commit_guard is not None else updated_at
            if operation_commit_time_guard is not None:
                activity_at = store_now()
                operation_commit_time_guard(activity_at)
            touch_activity(connection, session_id, activity_at)
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            existing_event_id = first_existing_event_id(
                connection,
                session_id,
                [event.id for event in copied_events],
            )
            if existing_event_id is not None:
                raise ValueError(
                    f"Event already exists for session {session_id}: {existing_event_id}"
                ) from exc
            raise
        except BaseException:
            connection.rollback()
            raise
        return loaded.model_copy(update={"updated_at": updated_at, "last_activity_at": activity_at})

    return await run_write(statement)
