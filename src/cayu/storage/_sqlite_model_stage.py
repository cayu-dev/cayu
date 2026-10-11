"""Complete native sqlite model-stage persistence with explicit capabilities.

The owner retains stage records, transaction boundaries, replay and cancellation.
SessionStore protocol hooks compose these operations with native collaborators.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Protocol

from cayu.events import EventType
from cayu.runtime._child_session_notifications import (
    ChildSessionLifecycleOccurrence,
    ChildSessionLifecycleOccurrenceSource,
    child_session_notification_stage_binding,
    child_session_notification_storage_key,
)
from cayu.sessions._child_lifecycle_rules import (
    _child_session_notification_consumption_record,
    _child_session_notification_consumption_replays,
)
from cayu.sessions._invocation_scope import _assert_session_run_epoch
from cayu.sessions._model_failover_rules import (
    _model_failover_predecessor_storage_keys,
    _model_failover_preparation_checkpoint,
    _model_failover_selection_event,
    _validate_model_failover_selection_replay,
)
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    ModelCompletionStage,
    ModelCompletionStageAbandonmentResult,
    ModelCompletionStageDispatch,
    ModelCompletionStageResult,
    SessionModelCompletionStageConflict,
)
from cayu.sessions._model_stage_publication import _validate_model_completion_stage_publication
from cayu.sessions._model_stage_rules import (
    _active_model_completion_stage_record,
    _model_completion_retry_settlement_request,
    _model_completion_stage_abandonment_record,
    _model_completion_stage_dispatch_record,
    _model_completion_stage_dispatch_storage_key,
    _model_completion_stage_preparation_record,
    _model_completion_stage_settlement_record,
    _model_completion_stage_settlement_storage_key,
    _model_completion_stage_storage_identity,
    _model_completion_stage_terminal_record,
    _model_completion_stage_winner_storage_key,
    _model_completion_terminal_advances_last_activity,
    _ModelCompletionStagePromotionContext,
    _prepare_model_completion_stage_promotion,
    _PreparedModelCompletionStage,
    _PreparedModelCompletionStageAbandonment,
    _PreparedModelCompletionStageTerminal,
    _reconstruct_active_model_completion_stage,
    _reconstruct_active_model_completion_stage_record,
    _reconstruct_model_completion_stage,
    _reconstruct_model_completion_stage_abandonment,
    _reconstruct_model_completion_stage_dispatch,
    _reject_settled_model_completion_stage,
    _replay_model_completion_stage_abandonment,
    _validate_model_completion_active_marker_for_preparation,
    _validate_model_completion_preparation_replay_state,
    _validate_model_completion_stage_dispatch,
    _validate_model_completion_stage_for_abandonment,
    _validate_model_completion_stage_for_dispatch,
    _validate_model_completion_stage_for_settlement,
    _validate_model_completion_stage_preparation_replay,
    _validate_model_completion_stage_recovery_fence,
    _validate_model_completion_stage_repreparation,
    _validate_model_completion_stage_terminal_replay,
)
from cayu.sessions._runtime_publication import (
    RuntimePublicationResult,
    _runtime_publication_storage_key,
)
from cayu.sessions._runtime_publication_rules import (
    _next_runtime_publication_timestamp,
    _PreparedRuntimePublication,
)
from cayu.sessions.authority import SessionRunFenced, _runtime_publication_json_equal
from cayu.sessions.records import Session, SessionStatusConflict
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage import _sqlite_transcript as transcript_ops
from cayu.storage._sqlite_connection import SQLiteOperationRunner
from cayu.storage._sqlite_session_messages import EventAppender, SteeringGuard


class StagePublisher(Protocol):
    def __call__(
        self,
        prepared: _PreparedRuntimePublication,
        *,
        _model_completion_stage: _ModelCompletionStagePromotionContext | None = None,
    ) -> Awaitable[RuntimePublicationResult]: ...


def decode_stage_record(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion stage record is malformed."
        ) from exc
    if type(decoded) is not dict:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion stage record is malformed."
        )
    return decoded


async def load_stage_records(
    session_id: str,
    preparation_storage_key: str,
    terminal_storage_key: str,
    *,
    run_read: SQLiteOperationRunner,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    def query(
        connection: sqlite3.Connection,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not sqlite_records.session_exists(connection, session_id):
            raise KeyError(f"Session not found: {session_id}")
        rows = connection.execute(
            "SELECT idempotency_key, record_json FROM cayu_session_operations "
            "WHERE session_id = ? AND idempotency_key IN (?, ?)",
            (session_id, preparation_storage_key, terminal_storage_key),
        ).fetchall()
        records = {row["idempotency_key"]: decode_stage_record(row["record_json"]) for row in rows}
        return records.get(preparation_storage_key), records.get(terminal_storage_key)

    return await run_read(query)


async def load_settlement_record(
    session_id: str, settlement_storage_key: str, *, run_read: SQLiteOperationRunner
) -> dict[str, Any] | None:
    def query(connection: sqlite3.Connection) -> dict[str, Any] | None:
        if not sqlite_records.session_exists(connection, session_id):
            raise KeyError(f"Session not found: {session_id}")
        row = connection.execute(
            "SELECT record_json FROM cayu_session_operations "
            "WHERE session_id = ? AND idempotency_key = ?",
            (session_id, settlement_storage_key),
        ).fetchone()
        return None if row is None else decode_stage_record(row["record_json"])

    return await run_read(query)


async def load_dispatch_record(
    session_id: str, dispatch_storage_key: str, *, run_read: SQLiteOperationRunner
) -> dict[str, Any] | None:
    def query(connection: sqlite3.Connection) -> dict[str, Any] | None:
        if not sqlite_records.session_exists(connection, session_id):
            raise KeyError(f"Session not found: {session_id}")
        row = connection.execute(
            "SELECT record_json FROM cayu_session_operations "
            "WHERE session_id = ? AND idempotency_key = ?",
            (session_id, dispatch_storage_key),
        ).fetchone()
        return None if row is None else decode_stage_record(row["record_json"])

    return await run_read(query)


async def load_active_stage_records(
    session_id: str, *, run_read: SQLiteOperationRunner
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    def query(
        connection: sqlite3.Connection,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        try:
            connection.execute("BEGIN")
            if not sqlite_records.session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            row = connection.execute(
                "SELECT record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key = ?",
                (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
            ).fetchone()
            if row is None:
                connection.rollback()
                return None, None, None
            active_record = decode_stage_record(row["record_json"])
            marker = _reconstruct_active_model_completion_stage_record(
                active_record,
                session_id=session_id,
            )
            _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
                session_id, marker.stage_id
            )
            rows = connection.execute(
                "SELECT idempotency_key, record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                (session_id, preparation_key, terminal_key),
            ).fetchall()
            records = {
                record_row["idempotency_key"]: decode_stage_record(record_row["record_json"])
                for record_row in rows
            }
            connection.rollback()
            return (
                active_record,
                records.get(preparation_key),
                records.get(terminal_key),
            )
        except BaseException:
            connection.rollback()
            raise

    return await run_read(query)


async def mark_dispatched(
    session_id: str,
    *,
    stage: ModelCompletionStage,
    consume_child_session_notifications: bool,
    run_write: SQLiteOperationRunner,
    load_session: Callable[[str], Session | None],
    load_checkpoint: Callable[[str], dict[str, Any] | None],
) -> ModelCompletionStageDispatch:
    _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
        session_id,
        stage.stage_id,
    )
    settlement_key = _model_completion_stage_settlement_storage_key(stage.stage_id)
    dispatch_key = _model_completion_stage_dispatch_storage_key(stage.stage_id)

    def statement(connection: sqlite3.Connection) -> ModelCompletionStageDispatch:
        try:
            connection.execute("BEGIN IMMEDIATE")
            loaded = load_session(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            rows = connection.execute(
                "SELECT idempotency_key, record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key IN (?, ?, ?, ?, ?)",
                (
                    session_id,
                    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                    preparation_key,
                    terminal_key,
                    settlement_key,
                    dispatch_key,
                ),
            ).fetchall()
            records = {
                row["idempotency_key"]: decode_stage_record(row["record_json"]) for row in rows
            }
            _validate_model_completion_stage_for_dispatch(
                session=loaded,
                checkpoint=load_checkpoint(session_id),
                current_transcript_cursor=transcript_ops.transcript_cursor(connection, session_id),
                stage=stage,
                active_record=records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                preparation_record=records.get(preparation_key),
                terminal_record=records.get(terminal_key),
                settlement_record=records.get(settlement_key),
            )
            published_at = _next_runtime_publication_timestamp(loaded)
            dispatch_record = records.get(dispatch_key)
            dispatch_is_new = dispatch_record is None
            if dispatch_is_new:
                dispatch_record = _model_completion_stage_dispatch_record(
                    stage,
                    dispatched_at=published_at,
                )
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        session_id,
                        dispatch_key,
                        sqlite_records.json_dumps(dispatch_record),
                        sqlite_records.format_datetime(published_at),
                    ),
                )
            assert dispatch_record is not None
            dispatch = _reconstruct_model_completion_stage_dispatch(
                dispatch_record,
                session_id=session_id,
                stage_id=stage.stage_id,
                storage_key=dispatch_key,
            )
            _validate_model_completion_stage_dispatch(dispatch, stage)
            binding = child_session_notification_stage_binding(stage.intent)
            if binding is not None:
                for claim in binding.claims:
                    child = sqlite_records.load_session(connection, claim.child_session_id)
                    event_row = (
                        None
                        if child is None
                        else connection.execute(
                            "SELECT * FROM cayu_events "
                            "WHERE session_id = ? "
                            "AND event_type IN (?, ?, ?, ?, ?, ?) "
                            "ORDER BY sequence DESC LIMIT 1",
                            (
                                claim.child_session_id,
                                str(EventType.SESSION_STARTED),
                                str(EventType.SESSION_RESUMED),
                                str(EventType.SESSION_FORKED),
                                str(EventType.SESSION_COMPLETED),
                                str(EventType.SESSION_FAILED),
                                str(EventType.SESSION_INTERRUPTED),
                            ),
                        ).fetchone()
                    )
                    event_record = sqlite_records.event_record_from_row(event_row)
                    if child is None or event_record is None:
                        raise SessionModelCompletionStageConflict(
                            "Child-session notification occurrence is no longer canonical."
                        )
                    occurrence = ChildSessionLifecycleOccurrence(
                        source=ChildSessionLifecycleOccurrenceSource.EVENT,
                        source_id=event_record.event.id,
                        source_sequence=event_record.sequence,
                        source_type=str(event_record.event.type),
                        occurred_at=event_record.event.timestamp,
                    )
                    consumption = _child_session_notification_consumption_record(
                        parent=loaded,
                        child=child,
                        occurrence=occurrence,
                        stage=stage,
                        consumed_at=published_at,
                    )
                    consumption_key = child_session_notification_storage_key(
                        child.instance_id, occurrence.source_id
                    )
                    consumption_row = connection.execute(
                        "SELECT record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, consumption_key),
                    ).fetchone()
                    material = consumption.model_dump(mode="json")
                    if consumption_row is not None:
                        if not _child_session_notification_consumption_replays(
                            json.loads(consumption_row["record_json"]),
                            consumption,
                        ):
                            raise SessionModelCompletionStageConflict(
                                "Child-session terminal notification was consumed by another stage."
                            )
                    elif consume_child_session_notifications:
                        connection.execute(
                            "INSERT INTO cayu_session_operations "
                            "(session_id, idempotency_key, record_json, updated_at) "
                            "VALUES (?, ?, ?, ?)",
                            (
                                session_id,
                                consumption_key,
                                sqlite_records.json_dumps(material),
                                sqlite_records.format_datetime(published_at),
                            ),
                        )
            formatted_at = sqlite_records.format_datetime(published_at)
            cursor = connection.execute(
                "UPDATE cayu_sessions SET updated_at = ?, last_activity_at = ? WHERE id = ?",
                (formatted_at, formatted_at, session_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Session not found: {session_id}")
            connection.commit()
            return dispatch
        except BaseException:
            connection.rollback()
            raise

    return await run_write(statement)


async def prepare(
    prepared: _PreparedModelCompletionStage,
    *,
    run_write: SQLiteOperationRunner,
    load_session: Callable[[str], Session | None],
    load_checkpoint: Callable[[str], dict[str, Any] | None],
    append_events: EventAppender,
    reject_steering: SteeringGuard,
) -> ModelCompletionStageResult:
    session_id = prepared.session_id

    def statement(connection: sqlite3.Connection) -> ModelCompletionStageResult:
        try:
            connection.execute("BEGIN IMMEDIATE")
            loaded = load_session(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            rows = connection.execute(
                "SELECT idempotency_key, record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key IN (?, ?, ?)",
                (
                    session_id,
                    prepared.preparation_storage_key,
                    prepared.terminal_storage_key,
                    prepared.abandonment_storage_key,
                ),
            ).fetchall()
            records = {
                row["idempotency_key"]: decode_stage_record(row["record_json"]) for row in rows
            }
            failover_source_keys = _model_failover_predecessor_storage_keys(prepared)
            if failover_source_keys:
                source_placeholders = ", ".join("?" for _ in failover_source_keys)
                source_rows = connection.execute(
                    "SELECT idempotency_key, record_json FROM cayu_session_operations "
                    f"WHERE session_id = ? AND idempotency_key IN ({source_placeholders})",
                    (session_id, *failover_source_keys),
                ).fetchall()
                records.update(
                    {
                        row["idempotency_key"]: decode_stage_record(row["record_json"])
                        for row in source_rows
                    }
                )
            stage = _reconstruct_model_completion_stage(
                records.get(prepared.preparation_storage_key),
                records.get(prepared.terminal_storage_key),
                session_id=session_id,
                stage_id=prepared.request.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            _validate_model_completion_stage_repreparation(
                records.get(prepared.abandonment_storage_key),
                prepared,
                source_status=loaded.status if stage is None else stage.source_status,
            )
            if stage is not None:
                _validate_model_completion_stage_preparation_replay(stage, prepared)

            active_row = connection.execute(
                "SELECT record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key = ?",
                (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
            ).fetchone()
            active = None
            if active_row is not None:
                active_record = decode_stage_record(active_row["record_json"])
                marker = _reconstruct_active_model_completion_stage_record(
                    active_record,
                    session_id=session_id,
                )
                _, _, active_preparation_key, active_terminal_key = (
                    _model_completion_stage_storage_identity(
                        session_id,
                        marker.stage_id,
                    )
                )
                active_rows = connection.execute(
                    "SELECT idempotency_key, record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                    (session_id, active_preparation_key, active_terminal_key),
                ).fetchall()
                active_records = {
                    row["idempotency_key"]: decode_stage_record(row["record_json"])
                    for row in active_rows
                }
                active = _reconstruct_active_model_completion_stage(
                    active_record,
                    active_records.get(active_preparation_key),
                    active_records.get(active_terminal_key),
                    session_id=session_id,
                )
            publication_rows = connection.execute(
                "SELECT idempotency_key FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                (
                    session_id,
                    prepared.winner_storage_key,
                    prepared.publication_storage_key,
                ),
            ).fetchall()
            publication_keys = {row["idempotency_key"] for row in publication_rows}
            winner_exists = prepared.winner_storage_key in publication_keys
            receipt_exists = prepared.publication_storage_key in publication_keys
            if stage is not None:
                _validate_model_completion_preparation_replay_state(
                    stage,
                    active=active,
                    winner_exists=winner_exists,
                    receipt_exists=receipt_exists,
                )
                _model_failover_preparation_checkpoint(
                    prepared,
                    session=loaded,
                    checkpoint=load_checkpoint(session_id),
                    current_transcript_cursor=transcript_ops.transcript_cursor(
                        connection, session_id
                    ),
                    active=active,
                    records=records,
                    replayed=True,
                )
                expected_selection = _model_failover_selection_event(
                    prepared, session=loaded, prepared_at=stage.prepared_at
                )
                if expected_selection is not None:
                    event_row = connection.execute(
                        "SELECT * FROM cayu_events WHERE session_id = ? AND event_id = ?",
                        (session_id, expected_selection.id),
                    ).fetchone()
                    _validate_model_failover_selection_replay(
                        expected_selection,
                        None if event_row is None else sqlite_records.event_from_row(event_row),
                    )
                connection.rollback()
                return ModelCompletionStageResult(
                    stage=stage,
                    replayed=True,
                    dispatch_authorized=False,
                )
            _validate_model_completion_active_marker_for_preparation(
                active,
                prepared,
                source_status=loaded.status,
            )
            retry_settlement_request = _model_completion_retry_settlement_request(
                active,
                prepared,
            )
            if winner_exists or receipt_exists:
                raise SessionModelCompletionStageConflict(
                    "The logical model step already has durable publication state."
                )

            _assert_session_run_epoch(session_id, loaded)
            if loaded.status not in prepared.expected_statuses:
                raise SessionStatusConflict(
                    "Session status is not eligible for model-completion preparation: "
                    f"{loaded.status}"
                )
            if loaded.run_epoch != prepared.expected_run_epoch:
                raise SessionRunFenced(
                    "Session source run epoch is stale: expected "
                    f"{prepared.expected_run_epoch}, current {loaded.run_epoch}."
                )
            current_cursor = transcript_ops.transcript_cursor(connection, session_id)
            if current_cursor != prepared.expected_transcript_cursor:
                raise ValueError(
                    "Session source transcript cursor is stale: expected "
                    f"{prepared.expected_transcript_cursor}, current {current_cursor}."
                )

            if active is None:
                reject_steering(connection, loaded)
            route_checkpoint = _model_failover_preparation_checkpoint(
                prepared,
                session=loaded,
                checkpoint=load_checkpoint(session_id),
                current_transcript_cursor=current_cursor,
                active=active,
                records=records,
                replayed=False,
            )
            prepared_at = _next_runtime_publication_timestamp(loaded)
            record = _model_completion_stage_preparation_record(
                prepared,
                source_session=loaded,
                prepared_at=prepared_at,
            )
            selection_event = _model_failover_selection_event(
                prepared, session=loaded, prepared_at=prepared_at
            )
            stage = _reconstruct_model_completion_stage(
                record,
                None,
                session_id=session_id,
                stage_id=prepared.request.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            assert stage is not None
            formatted_at = sqlite_records.format_datetime(prepared_at)
            if route_checkpoint is not None:
                connection.execute(
                    "INSERT INTO cayu_checkpoints (session_id, state_json, updated_at, "
                    "pending_action_source_bytes, pending_action_tool_call_count, "
                    "pending_action_flags, pending_action_metrics_ready) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET "
                    "state_json = excluded.state_json, updated_at = excluded.updated_at, "
                    "pending_action_source_bytes = excluded.pending_action_source_bytes, "
                    "pending_action_tool_call_count = excluded.pending_action_tool_call_count, "
                    "pending_action_flags = excluded.pending_action_flags, "
                    "pending_action_metrics_ready = excluded.pending_action_metrics_ready",
                    sqlite_records.checkpoint_row_values(session_id, route_checkpoint, prepared_at),
                )
            connection.execute(
                "INSERT INTO cayu_session_operations "
                "(session_id, idempotency_key, record_json, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    session_id,
                    prepared.preparation_storage_key,
                    sqlite_records.json_dumps(record),
                    formatted_at,
                ),
            )
            active_record = _active_model_completion_stage_record(
                stage,
                activated_at=prepared_at,
            )
            if retry_settlement_request is not None:
                assert active is not None
                retry_settlement_storage_key = _model_completion_stage_settlement_storage_key(
                    active.stage.stage_id
                )
                settlement_row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, retry_settlement_storage_key),
                ).fetchone()
                _validate_model_completion_stage_for_settlement(
                    session=loaded,
                    stage=active.stage,
                    active=active,
                    request=retry_settlement_request,
                    settlement_record=(
                        None
                        if settlement_row is None
                        else decode_stage_record(settlement_row["record_json"])
                    ),
                    winner_exists=winner_exists,
                    receipt_exists=receipt_exists,
                )
                retry_settlement_record = _model_completion_stage_settlement_record(
                    active.stage,
                    request=retry_settlement_request,
                    settled_at=prepared_at,
                )
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        session_id,
                        retry_settlement_storage_key,
                        sqlite_records.json_dumps(retry_settlement_record),
                        formatted_at,
                    ),
                )
            connection.execute(
                "INSERT INTO cayu_session_operations "
                "(session_id, idempotency_key, record_json, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                "record_json = excluded.record_json, updated_at = excluded.updated_at",
                (
                    session_id,
                    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                    sqlite_records.json_dumps(active_record),
                    formatted_at,
                ),
            )
            cursor = connection.execute(
                "UPDATE cayu_sessions SET updated_at = ?, last_activity_at = ? WHERE id = ?",
                (formatted_at, formatted_at, session_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Session not found: {session_id}")
            if selection_event is not None:
                append_events(connection, session_id, (selection_event,), activity_at=prepared_at)
            connection.commit()
            return ModelCompletionStageResult(
                stage=stage,
                replayed=False,
                dispatch_authorized=True,
                prepared_events=() if selection_event is None else (selection_event,),
            )
        except BaseException:
            connection.rollback()
            raise

    return await run_write(statement)


async def complete(
    prepared: _PreparedModelCompletionStageTerminal,
    *,
    run_write: SQLiteOperationRunner,
    load_session: Callable[[str], Session | None],
    load_checkpoint: Callable[[str], dict[str, Any] | None],
    store_now: Callable[[], datetime],
) -> ModelCompletionStageResult:
    session_id = prepared.session_id

    def statement(connection: sqlite3.Connection) -> ModelCompletionStageResult:
        try:
            connection.execute("BEGIN IMMEDIATE")
            loaded = load_session(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            rows = connection.execute(
                "SELECT idempotency_key, record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key IN (?, ?, ?)",
                (
                    session_id,
                    prepared.preparation_storage_key,
                    prepared.terminal_storage_key,
                    prepared.settlement_storage_key,
                ),
            ).fetchall()
            records = {
                row["idempotency_key"]: decode_stage_record(row["record_json"]) for row in rows
            }
            stage = _reconstruct_model_completion_stage(
                records.get(prepared.preparation_storage_key),
                records.get(prepared.terminal_storage_key),
                session_id=session_id,
                stage_id=prepared.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            if stage is None:
                raise KeyError(f"Model-completion stage not found: {prepared.stage_id}")
            _reject_settled_model_completion_stage(
                records.get(prepared.settlement_storage_key),
                session_id=session_id,
                stage_id=prepared.stage_id,
                settlement_storage_key=prepared.settlement_storage_key,
            )
            if stage.state == "completed":
                _validate_model_completion_stage_terminal_replay(stage, prepared)
                connection.rollback()
                return ModelCompletionStageResult(
                    stage=stage,
                    replayed=True,
                    dispatch_authorized=False,
                )
            if prepared.recovery_fence is not None:
                active_row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                ).fetchone()
                dispatch_row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, _model_completion_stage_dispatch_storage_key(stage.stage_id)),
                ).fetchone()
                _validate_model_completion_stage_recovery_fence(
                    prepared.recovery_fence,
                    session=loaded,
                    checkpoint=load_checkpoint(session_id),
                    stage=stage,
                    active_record=(
                        None
                        if active_row is None
                        else decode_stage_record(active_row["record_json"])
                    ),
                    dispatch_record=(
                        None
                        if dispatch_row is None
                        else decode_stage_record(dispatch_row["record_json"])
                    ),
                    now=store_now(),
                )
            _validate_model_completion_stage_publication(
                prepared.publication,
                session_id=session_id,
                stage=stage,
            )
            if not _runtime_publication_json_equal(prepared.publication.intent, stage.intent):
                raise SessionModelCompletionStageConflict(
                    "The terminal model completion intent conflicts with its preparation."
                )
            completed_at = _next_runtime_publication_timestamp(loaded)
            terminal_record = _model_completion_stage_terminal_record(
                prepared,
                stage=stage,
                completed_at=completed_at,
            )
            completed_stage = _reconstruct_model_completion_stage(
                records[prepared.preparation_storage_key],
                terminal_record,
                session_id=session_id,
                stage_id=prepared.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            assert completed_stage is not None
            active_row = connection.execute(
                "SELECT record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key = ?",
                (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
            ).fetchone()
            active_record = (
                None if active_row is None else decode_stage_record(active_row["record_json"])
            )
            advances_last_activity = _model_completion_terminal_advances_last_activity(
                active_record,
                stage=stage,
                current_run_epoch=loaded.run_epoch,
            )
            formatted_at = sqlite_records.format_datetime(completed_at)
            connection.execute(
                "INSERT INTO cayu_session_operations "
                "(session_id, idempotency_key, record_json, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    session_id,
                    prepared.terminal_storage_key,
                    sqlite_records.json_dumps(terminal_record),
                    formatted_at,
                ),
            )
            cursor = connection.execute(
                "UPDATE cayu_sessions SET updated_at = ?, "
                "last_activity_at = CASE WHEN ? THEN ? ELSE last_activity_at END "
                "WHERE id = ?",
                (
                    formatted_at,
                    advances_last_activity,
                    formatted_at,
                    session_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Session not found: {session_id}")
            connection.commit()
            return ModelCompletionStageResult(
                stage=completed_stage,
                replayed=False,
                dispatch_authorized=False,
            )
        except BaseException:
            connection.rollback()
            raise

    return await run_write(statement)


async def abandon(
    prepared: _PreparedModelCompletionStageAbandonment,
    *,
    run_write: SQLiteOperationRunner,
    load_session: Callable[[str], Session | None],
) -> ModelCompletionStageAbandonmentResult:
    session_id = prepared.session_id

    def statement(
        connection: sqlite3.Connection,
    ) -> ModelCompletionStageAbandonmentResult:
        try:
            connection.execute("BEGIN IMMEDIATE")
            loaded = load_session(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            rows = connection.execute(
                "SELECT idempotency_key, record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key IN (?, ?, ?)",
                (
                    session_id,
                    prepared.preparation_storage_key,
                    prepared.terminal_storage_key,
                    prepared.abandonment_storage_key,
                ),
            ).fetchall()
            records = {
                row["idempotency_key"]: decode_stage_record(row["record_json"]) for row in rows
            }
            stage = _reconstruct_model_completion_stage(
                records.get(prepared.preparation_storage_key),
                records.get(prepared.terminal_storage_key),
                session_id=session_id,
                stage_id=prepared.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            active_row = connection.execute(
                "SELECT record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key = ?",
                (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
            ).fetchone()
            active_record = (
                None if active_row is None else decode_stage_record(active_row["record_json"])
            )
            if stage is None:
                if active_record is not None:
                    active_marker = _reconstruct_active_model_completion_stage_record(
                        active_record,
                        session_id=session_id,
                    )
                    if active_marker.stage_id == prepared.stage_id:
                        raise SessionModelCompletionStageConflict(
                            "The active model-completion marker references a missing stage."
                        )
                replayed = _replay_model_completion_stage_abandonment(
                    prepared,
                    records.get(prepared.abandonment_storage_key),
                )
                publication_rows = connection.execute(
                    "SELECT idempotency_key FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                    (
                        session_id,
                        _model_completion_stage_winner_storage_key(
                            replayed.abandonment.logical_step_id
                        ),
                        _runtime_publication_storage_key(replayed.abandonment.logical_step_id),
                    ),
                ).fetchall()
                if publication_rows:
                    raise SessionModelCompletionStageConflict(
                        "An abandoned model-completion stage has durable publication state."
                    )
                connection.rollback()
                return replayed

            active = _reconstruct_active_model_completion_stage(
                active_record,
                records.get(prepared.preparation_storage_key),
                records.get(prepared.terminal_storage_key),
                session_id=session_id,
            )
            winner_storage_key = _model_completion_stage_winner_storage_key(stage.logical_step_id)
            publication_storage_key = _runtime_publication_storage_key(stage.logical_step_id)
            dispatch_storage_key = _model_completion_stage_dispatch_storage_key(stage.stage_id)
            publication_rows = connection.execute(
                "SELECT idempotency_key FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key IN (?, ?, ?)",
                (
                    session_id,
                    winner_storage_key,
                    publication_storage_key,
                    dispatch_storage_key,
                ),
            ).fetchall()
            publication_keys = {row["idempotency_key"] for row in publication_rows}
            _validate_model_completion_stage_for_abandonment(
                session=loaded,
                stage=stage,
                active=active,
                prepared=prepared,
                abandonment_record=records.get(prepared.abandonment_storage_key),
                dispatch_exists=dispatch_storage_key in publication_keys,
                winner_exists=winner_storage_key in publication_keys,
                receipt_exists=publication_storage_key in publication_keys,
            )
            assert active is not None
            abandoned_at = _next_runtime_publication_timestamp(loaded)
            abandonment_record = _model_completion_stage_abandonment_record(
                stage,
                active=active,
                abandoned_at=abandoned_at,
            )
            abandonment = _reconstruct_model_completion_stage_abandonment(
                abandonment_record,
                session_id=session_id,
                stage_id=prepared.stage_id,
                storage_key=prepared.abandonment_storage_key,
            )
            formatted_at = sqlite_records.format_datetime(abandoned_at)
            connection.execute(
                "INSERT INTO cayu_session_operations "
                "(session_id, idempotency_key, record_json, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                "record_json = excluded.record_json, updated_at = excluded.updated_at",
                (
                    session_id,
                    prepared.abandonment_storage_key,
                    sqlite_records.json_dumps(abandonment_record),
                    formatted_at,
                ),
            )
            deleted_preparation = connection.execute(
                "DELETE FROM cayu_session_operations WHERE session_id = ? AND idempotency_key = ?",
                (session_id, prepared.preparation_storage_key),
            )
            if deleted_preparation.rowcount != 1:
                raise SessionModelCompletionStageConflict(
                    "The model-completion preparation changed during abandonment."
                )
            deleted_active = connection.execute(
                "DELETE FROM cayu_session_operations WHERE session_id = ? AND idempotency_key = ?",
                (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
            )
            if deleted_active.rowcount != 1:
                raise SessionModelCompletionStageConflict(
                    "The active model-completion marker changed during abandonment."
                )
            cursor = connection.execute(
                "UPDATE cayu_sessions SET updated_at = ?, last_activity_at = ? WHERE id = ?",
                (formatted_at, formatted_at, session_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Session not found: {session_id}")
            connection.commit()
            return ModelCompletionStageAbandonmentResult(
                abandonment=abandonment,
                replayed=False,
            )
        except BaseException:
            connection.rollback()
            raise

    return await run_write(statement)


async def promote(
    *,
    session_id: str,
    stage_id: str,
    preparation_storage_key: str,
    terminal_storage_key: str,
    expected_run_epoch: int,
    load_stage: Callable[[str, str], Awaitable[ModelCompletionStage | None]],
    publish: StagePublisher,
) -> RuntimePublicationResult:
    stage = await load_stage(session_id, stage_id)
    if stage is None:
        raise KeyError(f"Model-completion stage not found: {stage_id}")
    prepared = _prepare_model_completion_stage_promotion(
        stage,
        expected_run_epoch=expected_run_epoch,
    )
    assert stage.completion_digest is not None
    return await publish(
        prepared,
        _model_completion_stage=_ModelCompletionStagePromotionContext(
            stage_id=stage_id,
            preparation_storage_key=preparation_storage_key,
            terminal_storage_key=terminal_storage_key,
            active_storage_key=MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
            winner_storage_key=_model_completion_stage_winner_storage_key(stage.logical_step_id),
            completion_digest=stage.completion_digest,
        ),
    )
