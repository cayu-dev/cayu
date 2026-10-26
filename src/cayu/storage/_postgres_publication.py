"""Native publication transactions composed from explicit storage capabilities."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import datetime
from typing import Any, Protocol

from cayu._validation import MAX_DURABLE_JSON_ENVELOPE_NESTING, copy_durable_json_object
from cayu.events import Event
from cayu.messages import Message
from cayu.runtime.public_authority import PublicAuthorityAliasCodec
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
from cayu.sessions._retention_attestation import (
    ReceiptMaterial,
    RetentionCompactionAttestation,
    parse_retention_attestation,
    retention_attestation_storage_key,
)
from cayu.sessions._runtime_publication import (
    RuntimePublicationReceipt,
    RuntimePublicationResult,
    SessionOperationTransform,
    StoreTimeSessionOperationTransform,
    _run_session_commit_guard_owned,
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
    _validate_runtime_publication_receipt_material,
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
from cayu.storage import _postgres_event_delivery as event_delivery_ops
from cayu.storage import _postgres_model_stage as model_stage_ops
from cayu.storage import _postgres_public_authority as public_authority_ops
from cayu.storage import _postgres_session_queries as session_queries
from cayu.storage import _postgres_support as pg_support
from cayu.storage import _postgres_transcript as transcript_ops
from cayu.storage._postgres_transcript import PostgresConnection


class TerminalReceiptRecorder(Protocol):
    def __call__(
        self, cur: Any, session_id: str, events: Sequence[Event], *, activity_at: datetime
    ) -> Awaitable[None]: ...


def decode_record(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication receipt is malformed or conflicts with its key."
        )
    return value


async def load_receipt_record(
    session_id: str,
    storage_key: str,
    publication_id: str,
    *,
    connect: PostgresConnection,
    ensure_ready: Callable[[], Awaitable[None]],
) -> dict[str, Any] | None:
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT record FROM cayu_session_operations "
            "WHERE session_id = %s AND idempotency_key = %s FOR SHARE",
            (session_id, storage_key),
        )
        row = await cur.fetchone()
        if row is not None:
            record = decode_record(row[0])
            receipt = _reconstruct_runtime_publication_receipt(
                record,
                storage_key=storage_key,
                session_id=session_id,
                publication_id=publication_id,
            )
            await validate_material(cur, receipt)
            return record
        await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")
        return None


async def load_material(
    cur, receipt: RuntimePublicationReceipt, *, lock_events: bool = True
) -> ReceiptMaterial:
    await cur.execute(
        "SELECT interaction_id, message FROM cayu_transcript_messages "
        "WHERE session_id = %s AND session_order > %s AND session_order <= %s "
        "ORDER BY session_order ASC",
        (
            receipt.session_id,
            receipt.transcript_start_cursor,
            receipt.transcript_end_cursor,
        ),
    )
    transcript_rows = await cur.fetchall()
    transcript = [Message(**pg_support._json_obj(row[1])) for row in transcript_rows]
    transcript_interaction_ids = [row[0] for row in transcript_rows]

    referenced_event_ids = _runtime_publication_referenced_event_ids(receipt.referenced_events)
    requested_event_ids = tuple(dict.fromkeys((*receipt.appended_event_ids, *referenced_event_ids)))
    events_by_id: dict[str, Event] = {}
    if requested_event_ids:
        await cur.execute(
            "SELECT event_id, event FROM cayu_events "
            "WHERE session_id = %s AND event_id = ANY(%s)" + (" FOR SHARE" if lock_events else ""),
            (receipt.session_id, list(requested_event_ids)),
        )
        events_by_id = {
            row[0]: Event(**pg_support._json_obj(row[1])) for row in await cur.fetchall()
        }
    return ReceiptMaterial(
        transcript=transcript,
        transcript_interaction_ids=transcript_interaction_ids,
        appended_events=[
            events_by_id[event_id]
            for event_id in receipt.appended_event_ids
            if event_id in events_by_id
        ],
        referenced_events=[
            events_by_id[event_id] for event_id in referenced_event_ids if event_id in events_by_id
        ],
    )


async def load_attestation(
    cur, receipt: RuntimePublicationReceipt
) -> RetentionCompactionAttestation | None:
    await cur.execute(
        "SELECT record FROM cayu_session_operations WHERE session_id = %s AND idempotency_key = %s",
        (receipt.session_id, retention_attestation_storage_key(receipt.publication_id)),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    try:
        record = pg_support._json_obj(row[0])
    except (TypeError, ValueError):
        return None
    return parse_retention_attestation(record, receipt)


async def validate_material(
    cur, receipt: RuntimePublicationReceipt, *, lock_events: bool = True
) -> None:
    try:
        material = await load_material(cur, receipt, lock_events=lock_events)
        try:
            _validate_runtime_publication_receipt_material(receipt, material)
        except SessionRuntimePublicationConflict:
            # Storage-retention compaction may have replaced tool-output bodies
            # under an attestation bound to this exact receipt.
            attestation = await load_attestation(cur, receipt)
            if attestation is None:
                raise
            _validate_runtime_publication_receipt_material(receipt, material, attestation)
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
    connect: PostgresConnection,
    ensure_ready: Callable[[], Awaitable[None]],
    load_for_update: Callable[[Any, str], Awaitable[Session | None]],
    load_checkpoint: Callable[[Any, str], Awaitable[dict[str, Any] | None]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
    checkpoint_row_values: Callable[[str, dict[str, Any], datetime], tuple[object, ...]],
    unique_violation: type[Exception],
) -> RuntimePublicationResult:
    from cayu.sessions.pending_actions import (
        pending_action_event_storage_values,
        pending_action_lookup_key,
    )

    session_id = prepared.session_id
    request = prepared.request
    checkpoint_codec = prepared.checkpoint_codec
    checkpoint_decode = None if checkpoint_codec is None else checkpoint_codec.decode
    published_at: datetime
    receipt: RuntimePublicationReceipt
    loaded: Session

    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                loaded_result = await load_for_update(cur, session_id)
                if loaded_result is None:
                    raise KeyError(f"Session not found: {session_id}")
                loaded = loaded_result

                locked_stage = None
                active_record = None
                winner_record = None
                if _model_completion_stage is not None:
                    await cur.execute(
                        "SELECT idempotency_key, record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                        (
                            session_id,
                            [
                                _model_completion_stage.preparation_storage_key,
                                _model_completion_stage.terminal_storage_key,
                                _model_completion_stage.active_storage_key,
                                _model_completion_stage.winner_storage_key,
                            ],
                        ),
                    )
                    stage_records = {
                        row[0]: model_stage_ops.decode_stage_record(row[1])
                        for row in await cur.fetchall()
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

                await cur.execute(
                    "SELECT record FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = %s",
                    (session_id, prepared.storage_key),
                )
                receipt_row = await cur.fetchone()
                if receipt_row is not None:
                    receipt_record = decode_record(receipt_row[0])
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
                            await cur.execute(
                                "SELECT idempotency_key, record "
                                "FROM cayu_session_operations "
                                "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                                (
                                    session_id,
                                    [active_preparation_key, active_terminal_key],
                                ),
                            )
                            active_records = {
                                row[0]: model_stage_ops.decode_stage_record(row[1])
                                for row in await cur.fetchall()
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
                        await validate_material(cur, receipt)
                        result = _replay_promoted_model_completion_stage(
                            session=loaded,
                            stage=locked_stage,
                            receipt_record=receipt_record,
                            winner_record=winner_record,
                        )
                        await conn.rollback()
                        return result
                    receipt = _reconstruct_runtime_publication_receipt(
                        receipt_record,
                        storage_key=prepared.storage_key,
                        session_id=session_id,
                        publication_id=request.publication_id,
                        request_digest=prepared.request_digest,
                    )
                    _validate_runtime_publication_replay_receipt(receipt, prepared)
                    await validate_material(cur, receipt)
                    await conn.rollback()
                    return RuntimePublicationResult(
                        session=loaded.model_copy(deep=True),
                        receipt=receipt,
                        replayed=True,
                    )

                for owner in await closure_owners(cur, (session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))

                operation_mutation_records: dict[str, dict[str, Any]] = {}
                if request.operation_record_mutations:
                    mutation_keys = [
                        mutation.key for mutation in request.operation_record_mutations
                    ]
                    await cur.execute(
                        "SELECT idempotency_key, record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = ANY(%s) FOR UPDATE",
                        (session_id, mutation_keys),
                    )
                    current_mutation_records = {
                        row[0]: pg_support._json_obj(row[1]) for row in await cur.fetchall()
                    }
                    operation_mutation_records = (
                        _apply_runtime_publication_operation_record_mutations(
                            request.operation_record_mutations,
                            current_mutation_records,
                        )
                    )

                if request.argument_continuity is not None:
                    from cayu.sessions._argument_continuity import STORAGE_KEY, append_record

                    await cur.execute(
                        "SELECT record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s FOR UPDATE",
                        (session_id, STORAGE_KEY),
                    )
                    private_row = await cur.fetchone()
                    operation_mutation_records[STORAGE_KEY] = append_record(
                        None if private_row is None else pg_support._json_obj(private_row[0]),
                        continuity=request.argument_continuity,
                        request_digest=prepared.request_digest,
                        session=loaded,
                        messages=request.transcript_messages,
                    )

                if locked_stage is not None:
                    assert _model_completion_stage is not None
                    if winner_record is not None:
                        raise SessionModelCompletionStageConflict(
                            "A model-completion winner exists without its runtime "
                            "publication receipt."
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
                transcript_start_cursor = await transcript_ops.transcript_cursor(cur, session_id)
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
                durable_references: dict[str, Event] = {}
                if referenced_event_ids:
                    await cur.execute(
                        "SELECT event_id, event FROM cayu_events "
                        "WHERE session_id = %s AND event_id = ANY(%s) FOR SHARE",
                        (session_id, list(referenced_event_ids)),
                    )
                    durable_references = {
                        row[0]: Event(**pg_support._json_obj(row[1]))
                        for row in await cur.fetchall()
                    }
                _validate_runtime_publication_event_references(
                    request.referenced_events,
                    durable_references,
                    interaction_id=request.interaction_id,
                )
                stored_checkpoint = await load_checkpoint(cur, session_id)
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
                    durable_events_by_id=durable_references,
                )
                _validate_tool_round_checkpoint_mutation(
                    request,
                    current_checkpoint,
                )
                durable_tool_events: list[Event] = []
                tool_round_identity = _tool_lifecycle_publication_identity(request)
                if tool_round_identity is not None:
                    execution_identity, tool_call_ids = tool_round_identity
                    lookup_keys = [
                        pending_action_lookup_key(tool_call_id) for tool_call_id in tool_call_ids
                    ]
                    lifecycle_event_types = [
                        str(event_type)
                        for event_type in sorted(
                            _TOOL_ROUND_LIFECYCLE_EVENT_TYPES,
                            key=str,
                        )
                    ]
                    await cur.execute(
                        "SELECT event_id, event FROM cayu_events "
                        "WHERE session_id = %s "
                        "AND pending_action_lookup_key = ANY(%s) "
                        "AND event_type = ANY(%s) "
                        f"AND ({session_queries.PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
                        "AND (event -> 'payload' ->> 'tool_round_id' = %s "
                        "OR (event -> 'payload' ->> 'model_step_id' = %s "
                        "AND event -> 'payload' ->> 'model_attempt_id' = %s) "
                        "OR ((event -> 'payload' ->> 'tool_round_id') "
                        "~ '^tround_[0-9a-f]{32}$') IS NOT TRUE "
                        "OR ((event -> 'payload' ->> 'model_step_id') "
                        "~ '^mstep_[0-9a-f]{32}$') IS NOT TRUE "
                        "OR ((event -> 'payload' ->> 'model_attempt_id') "
                        "~ '^matt_[0-9a-f]{32}$') IS NOT TRUE) "
                        "ORDER BY session_order ASC LIMIT %s FOR SHARE",
                        (
                            session_id,
                            lookup_keys,
                            lifecycle_event_types,
                            execution_identity.tool_round_id,
                            execution_identity.model_step_id,
                            execution_identity.model_attempt_id,
                            _tool_round_lifecycle_event_limit(tool_call_ids) + 1,
                        ),
                    )
                    rows = await cur.fetchall()
                    if len(rows) > _tool_round_lifecycle_event_limit(tool_call_ids):
                        raise ValueError(
                            "Tool-round lifecycle evidence exceeds the publication limit."
                        )
                    durable_tool_events = [Event(**pg_support._json_obj(row[1])) for row in rows]
                _validate_tool_round_publication(
                    request,
                    durable_references,
                    durable_tool_events=durable_tool_events,
                )
                if request.events:
                    await cur.execute(
                        "SELECT event_id FROM cayu_events "
                        "WHERE session_id = %s AND event_id = ANY(%s)",
                        (session_id, [event.id for event in request.events]),
                    )
                    existing_event_row = await cur.fetchone()
                    if existing_event_row is not None:
                        raise ValueError(
                            f"Event already exists for session {session_id}: "
                            f"{existing_event_row[0]}"
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
                        request.interaction_id,
                        pg_support._dumps(message_payload),
                        transcript_ops.index_document(session_id, message),
                    )
                    for message, message_payload in zip(
                        request.transcript_messages,
                        prepared.transcript_payloads,
                        strict=True,
                    )
                ]
                prepared_event_rows = []
                for event, event_payload in zip(
                    request.events,
                    prepared.event_payloads,
                    strict=True,
                ):
                    lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                        event
                    )
                    prepared_event_rows.append(
                        (
                            session_id,
                            event.id,
                            event.interaction_id,
                            str(event.type),
                            pg_support.to_utc(event.timestamp),
                            event.agent_name,
                            event.environment_name,
                            event.workflow_name,
                            event.tool_name,
                            pg_support._dumps(event_payload["payload"]),
                            pg_support._dumps(event_payload),
                            lookup_key,
                            projection,
                            projection_bytes,
                        )
                    )

                published_at = _next_runtime_publication_timestamp(loaded)
                checkpoint_values = (
                    None
                    if stored_target_checkpoint is None or not request.mutation.operations
                    else checkpoint_row_values(
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
                receipt_record = _runtime_publication_receipt_record(receipt)
                receipt_json = pg_support._dumps(receipt_record)

                await public_authority_ops.register_event_authorities(
                    cur, session_id, request.events, codec=public_authority_alias_codec
                )
                await public_authority_ops.register_authorities(
                    cur,
                    session_id,
                    interaction_ids=()
                    if request.interaction_id is None
                    else (request.interaction_id,),
                    codec=public_authority_alias_codec,
                )
                await cur.execute(
                    """
                        UPDATE cayu_sessions
                        SET event_seq = event_seq + %s,
                            updated_at = %s,
                            last_activity_at = %s
                        WHERE id = %s
                        RETURNING event_seq
                        """,
                    (len(request.events), published_at, published_at, session_id),
                )
                order_row = await cur.fetchone()
                if order_row is None:
                    raise KeyError(f"Session not found: {session_id}")
                if transcript_rows:
                    await cur.executemany(
                        """
                            INSERT INTO cayu_transcript_messages (
                                session_id, interaction_id, message,
                                transcript_search_document
                            )
                            VALUES (%s, %s, %s, %s)
                            """,
                        transcript_rows,
                    )
                if checkpoint_values is not None:
                    await cur.execute(
                        """
                            INSERT INTO cayu_checkpoints (
                                session_id, state, updated_at,
                                pending_action_source_bytes,
                                pending_action_tool_call_count,
                                pending_action_flags,
                                pending_action_metrics_ready
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (session_id) DO UPDATE SET
                                state = EXCLUDED.state,
                                updated_at = EXCLUDED.updated_at,
                                pending_action_source_bytes = EXCLUDED.pending_action_source_bytes,
                                pending_action_tool_call_count = EXCLUDED.pending_action_tool_call_count,
                                pending_action_flags = EXCLUDED.pending_action_flags,
                                pending_action_metrics_ready = EXCLUDED.pending_action_metrics_ready
                            """,
                        checkpoint_values,
                    )

                next_order = order_row[0] - len(prepared_event_rows)
                event_rows = []
                for row in prepared_event_rows:
                    next_order += 1
                    event_rows.append((row[0], next_order, *row[1:]))
                if event_rows:
                    await cur.executemany(
                        """
                            INSERT INTO cayu_events (
                                session_id, session_order, event_id, interaction_id,
                                event_type, timestamp,
                                agent_name, environment_name, workflow_name, tool_name,
                                payload, event, pending_action_lookup_key,
                                pending_action_projection, pending_action_projection_bytes
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s
                            )
                            """,
                        event_rows,
                    )
                    await event_delivery_ops.enqueue_persisted_event_side_effects(
                        cur,
                        session_id,
                        request.events,
                    )
                if operation_mutation_records:
                    await cur.executemany(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record, updated_at) "
                        "VALUES (%s, %s, %s, %s) "
                        "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                        "record = excluded.record, updated_at = excluded.updated_at",
                        [
                            (session_id, key, pg_support._dumps(record), published_at)
                            for key, record in operation_mutation_records.items()
                        ],
                    )
                await cur.execute(
                    """
                        INSERT INTO cayu_session_operations (
                            session_id, idempotency_key, record, updated_at
                        )
                        VALUES (%s, %s, %s, %s)
                        """,
                    (
                        session_id,
                        prepared.storage_key,
                        receipt_json,
                        published_at,
                    ),
                )
                if locked_stage is not None and _model_completion_stage is not None:
                    winner = _model_completion_stage_winner_record(
                        locked_stage,
                        receipt=receipt,
                    )
                    await cur.execute(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record, updated_at) "
                        "VALUES (%s, %s, %s, %s)",
                        (
                            session_id,
                            _model_completion_stage.winner_storage_key,
                            pg_support._dumps(winner),
                            published_at,
                        ),
                    )
                    await cur.execute(
                        "DELETE FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (session_id, _model_completion_stage.active_storage_key),
                    )
                    if cur.rowcount != 1:
                        raise SessionModelCompletionStageConflict(
                            "The active model-completion marker changed before commit."
                        )
            await conn.commit()
        except unique_violation as exc:
            await conn.rollback()
            existing = None
            async with conn.cursor() as cur:
                for event in request.events:
                    await cur.execute(
                        "SELECT 1 FROM cayu_events WHERE session_id = %s AND event_id = %s",
                        (session_id, event.id),
                    )
                    if await cur.fetchone() is not None:
                        existing = event.id
                        break
                await cur.execute(
                    "SELECT 1 FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = %s",
                    (session_id, prepared.storage_key),
                )
                receipt_exists = await cur.fetchone() is not None
            if existing is not None:
                raise ValueError(
                    f"Event already exists for session {session_id}: {existing}"
                ) from exc
            if receipt_exists:
                raise SessionRuntimePublicationConflict(
                    "Runtime publication receipt was inserted concurrently."
                ) from exc
            raise
        except BaseException:
            await conn.rollback()
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
    unique_violation: type[Exception],
    connect: PostgresConnection,
    ensure_ready: Callable[[], Awaitable[None]],
    load_for_update: Callable[[Any, str], Awaitable[Session | None]],
    load_checkpoint: Callable[[Any, str], Awaitable[dict[str, Any] | None]],
    store_now: Callable[[Any], Awaitable[datetime]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
    first_existing_event_id: Callable[[str, list[str]], Awaitable[str | None]],
    publish_budget_identities: Callable[[Any, list[Event]], Awaitable[None]],
    record_terminal_receipts: TerminalReceiptRecorder,
    upsert_checkpoint: Callable[[Any, str, dict[str, Any], datetime], Awaitable[None]],
    validate_compaction: Callable[[Any, str, int], Awaitable[None]],
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
    await ensure_ready()
    async with connect() as conn:
        commit_guard_signal: BaseException | None = None
        try:
            async with conn.cursor() as cur:
                if context_view_compaction_cursor is not None:
                    await cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (f"context-view-session:{session_id}",),
                    )
                loaded = await load_for_update(cur, session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                updated_at = await store_now(cur)
                _assert_session_run_epoch(session_id, loaded)
                for owner in await closure_owners(cur, (session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                if allowed_statuses is not None and loaded.status not in allowed_statuses:
                    raise SessionStatusConflict(
                        "Session status is not eligible for checkpoint publication: "
                        f"{loaded.status}"
                    )
                if expected_run_epoch is not None and loaded.run_epoch != expected_run_epoch:
                    raise SessionRunFenced(
                        f"Session source run epoch is stale: expected {expected_run_epoch}, "
                        f"current {loaded.run_epoch}."
                    )
                current_cursor = await transcript_ops.transcript_cursor(cur, session_id)
                if (
                    expected_transcript_cursor is not None
                    and current_cursor != expected_transcript_cursor
                ):
                    raise ValueError(
                        "Session source transcript cursor is stale: expected "
                        f"{expected_transcript_cursor}, current {current_cursor}."
                    )
                if context_view_compaction_cursor is not None:
                    await validate_compaction(cur, session_id, context_view_compaction_cursor)
                current_checkpoint = await load_checkpoint(cur, session_id)
                callback_checkpoint = _copy_checkpoint_for_transform(
                    current_checkpoint,
                    session_id=session_id,
                )
                operation_records: dict[str, dict[str, Any]] = {}
                model_completion_stage_release = None
                if operation_transform is not None or store_time_operation_transform is not None:
                    await cur.execute(
                        "SELECT record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (session_id, operation_idempotency_key),
                    )
                    operation_row = await cur.fetchone()
                    current_operation = (
                        None if operation_row is None else pg_support._json_obj(operation_row[0])
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
                    transformed = _checkpoint_transform_result_preserving_completion_result_event_publications(
                        current_checkpoint,
                        transformed,
                        session_id=session_id,
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
                            await cur.execute(
                                "SELECT session_order - 1, interaction_id, message "
                                "FROM cayu_transcript_messages "
                                "WHERE session_id = %s AND session_order = ANY(%s) "
                                "ORDER BY session_order",
                                (session_id, [index + 1 for index in indices]),
                            )
                            selected_rows = tuple(
                                TranscriptRecord(
                                    index=row[0],
                                    interaction_id=row[1],
                                    message=Message(**pg_support._json_obj(row[2])),
                                )
                                for row in await cur.fetchall()
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
                    transformed = (
                        _replace_checkpoint_preserving_completion_result_event_publications(
                            current_checkpoint,
                            transformed,
                            preserve_completion_result_publications=(
                                preserve_completion_result_publications
                            ),
                            session_id=session_id,
                        )
                    )

                await public_authority_ops.register_event_authorities(
                    cur, session_id, copied_events, codec=public_authority_alias_codec
                )
                await publish_budget_identities(cur, copied_events)
                await cur.execute(
                    """
                        UPDATE cayu_sessions
                        SET event_seq = event_seq + %s,
                            updated_at = %s,
                            last_activity_at = %s
                        WHERE id = %s
                        RETURNING event_seq
                        """,
                    (len(copied_events), updated_at, updated_at, session_id),
                )
                order_row = await cur.fetchone()
                if order_row is None:
                    raise KeyError(f"Session not found: {session_id}")
                await upsert_checkpoint(cur, session_id, transformed, updated_at)
                if operation_records:
                    await cur.executemany(
                        """
                            INSERT INTO cayu_session_operations (
                                session_id, idempotency_key, record, updated_at
                            )
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT(session_id, idempotency_key) DO UPDATE SET
                                record = excluded.record,
                                updated_at = excluded.updated_at
                            """,
                        [
                            (session_id, key, pg_support._dumps(record), updated_at)
                            for key, record in operation_records.items()
                        ],
                    )
                if model_completion_stage_release is not None:
                    _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
                        session_id,
                        model_completion_stage_release.stage_id,
                    )
                    await cur.execute(
                        "SELECT idempotency_key, record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                        (
                            session_id,
                            [
                                MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                                preparation_key,
                                terminal_key,
                            ],
                        ),
                    )
                    stage_records = {
                        row[0]: model_stage_ops.decode_stage_record(row[1])
                        for row in await cur.fetchall()
                    }
                    marker = _validate_model_completion_stage_release(
                        session=loaded,
                        active_record=stage_records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                        preparation_record=stage_records.get(preparation_key),
                        terminal_record=stage_records.get(terminal_key),
                        release=model_completion_stage_release,
                    )
                    await cur.execute(
                        "DELETE FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s "
                        "AND record->>'record_digest' = %s",
                        (
                            session_id,
                            MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                            marker.record_digest,
                        ),
                    )
                    if cur.rowcount != 1:
                        raise SessionModelCompletionStageConflict(
                            "The active model-completion stage changed during disposition."
                        )

                next_order = order_row[0] - len(copied_events)
                rows = []
                for event in copied_events:
                    next_order += 1
                    lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                        event
                    )
                    rows.append(
                        (
                            session_id,
                            next_order,
                            event.id,
                            event.interaction_id,
                            str(event.type),
                            pg_support.to_utc(event.timestamp),
                            event.agent_name,
                            event.environment_name,
                            event.workflow_name,
                            event.tool_name,
                            pg_support._dumps(event.payload),
                            pg_support._dumps(event.model_dump(mode="json")),
                            lookup_key,
                            projection,
                            projection_bytes,
                        )
                    )
                if rows:
                    await cur.executemany(
                        """
                            INSERT INTO cayu_events (
                                session_id, session_order, event_id, interaction_id,
                                event_type, timestamp,
                                agent_name, environment_name, workflow_name, tool_name,
                                payload, event, pending_action_lookup_key,
                                pending_action_projection, pending_action_projection_bytes
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s
                            )
                            """,
                        rows,
                    )
                    await record_terminal_receipts(
                        cur, session_id, copied_events, activity_at=updated_at
                    )
                    await event_delivery_ops.enqueue_persisted_event_side_effects(
                        cur,
                        session_id,
                        copied_events,
                    )
                if operation_commit_guard is not None:
                    commit_guard_signal = await _run_session_commit_guard_owned(
                        operation_commit_guard
                    )
                activity_at = (
                    await store_now(cur) if operation_commit_guard is not None else updated_at
                )
                if operation_commit_time_guard is not None:
                    activity_at = await store_now(cur)
                    operation_commit_time_guard(activity_at)
                if activity_at != updated_at:
                    await cur.execute(
                        "UPDATE cayu_sessions SET last_activity_at = %s WHERE id = %s",
                        (activity_at, session_id),
                    )
            await conn.commit()
            if commit_guard_signal is not None:
                raise commit_guard_signal
        except unique_violation as exc:
            await conn.rollback()
            if commit_guard_signal is not None:
                raise commit_guard_signal from exc
            existing = await first_existing_event_id(
                session_id,
                [event.id for event in copied_events],
            )
            if existing is not None:
                raise ValueError(
                    f"Event already exists for session {session_id}: {existing}"
                ) from exc
            raise
        except Exception as error:
            await conn.rollback()
            if commit_guard_signal is not None:
                raise commit_guard_signal from error
            raise
        return loaded.model_copy(update={"updated_at": updated_at, "last_activity_at": activity_at})
