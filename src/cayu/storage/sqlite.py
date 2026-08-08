from __future__ import annotations

import asyncio
import contextvars
import hmac
import json
import math
import sqlite3
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Literal, TypeVar, cast
from uuid import uuid4

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    JsonUtf8SizeCounter,
    copy_durable_json_object,
    copy_label_map,
    require_nonblank,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu.core.events import EVENT_ID_MAX_CHARS, Event, EventType
from cayu.core.messages import Message, MessageRole
from cayu.core.workflows import WORKFLOW_ATTEMPT_EVENT_TYPE
from cayu.runtime.aggregates import EXACT_AGGREGATE, UsageRollupStoreResult
from cayu.runtime.approvals import ResolutionActor, resolution_actor_payload
from cayu.runtime.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.runtime.public_authority import PublicAuthorityAliasCodec
from cayu.runtime.service_manifest import RuntimeStoreDurability
from cayu.runtime.sessions import (
    _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES,
    _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT,
    _TOOL_ROUND_LIFECYCLE_EVENT_TYPES,
    CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS,
    DELETE_BLOCKED_SESSION_STATUSES,
    FORK_TRANSCRIPT_VALIDATION_ERROR,
    INHERIT_INTERACTION,
    MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL,
    MAX_PENDING_ACTION_TOOL_CALLS,
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS,
    RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX,
    SESSION_INSPECTION_LABEL_LIMIT,
    SESSION_LINEAGE_MAX_EVENT_ID_BYTES,
    SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
    SESSION_LINEAGE_MAX_ORIGIN_EVENTS,
    SESSION_LINEAGE_MAX_TIMESTAMP_BYTES,
    SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES,
    BudgetReservationIdentityConflict,
    CheckpointRootFieldGuard,
    CheckpointTransform,
    DeferredInteractionInput,
    EnqueueSessionMessageRequest,
    EnqueueSessionMessageResult,
    EventQuery,
    EventQueryResultTooLarge,
    EventRecord,
    EventSummary,
    ForkTranscriptValidator,
    InteractionAttribution,
    InteractionTransitionResult,
    McpManifestBaseline,
    McpManifestBaselineLoadResult,
    McpManifestPublicationResult,
    ModelCompletionStageAbandonmentResult,
    ModelCompletionStageResult,
    PendingActionIssue,
    PendingActionKind,
    PendingActionListResult,
    PendingActionQuery,
    PendingActionSession,
    PersistedEventSideEffectClaim,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectStatus,
    RunnerObservedEventIdentity,
    RunRequest,
    RuntimePublicationReceipt,
    RuntimePublicationResult,
    Session,
    SessionAggregateFilter,
    SessionForkActiveModelStageConflict,
    SessionIdentity,
    SessionInspectionIdentity,
    SessionLineageNode,
    SessionLineageOrigin,
    SessionLineageQuery,
    SessionLineageResult,
    SessionListResult,
    SessionMessageDeliveryBatch,
    SessionMessageQueueStatus,
    SessionModelCompletionStageConflict,
    SessionOperationalSnapshot,
    SessionOperationPublication,
    SessionOperationTransform,
    SessionOrder,
    SessionOutcome,
    SessionQuery,
    SessionQueuedMessage,
    SessionQueuedMessagesPending,
    SessionRunFenced,
    SessionRuntimePublicationConflict,
    SessionStateSnapshot,
    SessionStatus,
    SessionStatusConflict,
    SessionStatusCounts,
    SessionStore,
    SessionTopologyCycle,
    SessionTopologyDepthExceeded,
    SessionTopologyNode,
    SessionTopologyQuery,
    SessionTopologyStoreResult,
    TerminalPublicationMarker,
    TerminalSessionEvidence,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
    TerminalSessionEvidenceLimits,
    TranscriptPage,
    TranscriptQuery,
    TranscriptRecord,
    UsageRollupQuery,
    _activate_session_run_fence,
    _active_model_completion_stage_record,
    _active_unexpired_incomplete_recovery_claim_id,
    _active_unexpired_session_operation_id,
    _apply_runtime_publication_checkpoint_mutation,
    _assemble_terminal_session_evidence,
    _assert_session_run_epoch,
    _assert_session_run_epoch_value,
    _authenticated_public_authority_alias_private_value,
    _build_runtime_publication_receipt,
    _checkpoint_after_initial_transcript_publication,
    _classify_terminal_session_evidence_records,
    _copy_mcp_manifest_publication,
    _copy_optional_interaction_admission,
    _copy_queued_interaction_started_event,
    _copy_runner_owned_interruption_proof,
    _copy_session_event_batch,
    _copy_terminal_session_evidence_limits,
    _copy_transition_interaction_admission,
    _copy_workflow_step_reservation,
    _current_session_run_epoch,
    _deactivate_session_interaction,
    _deactivate_session_run_fence,
    _event_input_contract_is_runtime_owned,
    _initial_transcript_pending_checkpoint,
    _interaction_transition_receipt_record,
    _interaction_transition_storage_key,
    _model_completion_stage_abandonment_record,
    _model_completion_stage_preparation_record,
    _model_completion_stage_storage_identity,
    _model_completion_stage_terminal_record,
    _model_completion_stage_winner_record,
    _model_completion_stage_winner_storage_key,
    _model_completion_terminal_advances_last_activity,
    _ModelCompletionStagePromotionContext,
    _next_runtime_publication_timestamp,
    _prepare_interaction_transition,
    _prepare_model_completion_stage_promotion,
    _prepare_session_fork_request,
    _PreparedModelCompletionStage,
    _PreparedModelCompletionStageAbandonment,
    _PreparedModelCompletionStageTerminal,
    _PreparedRuntimePublication,
    _project_interruption_cascade_marker_fields,
    _public_authority_alias_store_key,
    _queued_session_message_event_payload,
    _reconstruct_active_model_completion_stage,
    _reconstruct_active_model_completion_stage_record,
    _reconstruct_interaction_transition_receipt,
    _reconstruct_model_completion_stage,
    _reconstruct_model_completion_stage_abandonment,
    _reconstruct_runtime_publication_receipt,
    _reject_reserved_runtime_publication_key,
    _replay_model_completion_stage_abandonment,
    _replay_promoted_model_completion_stage,
    _runtime_publication_json_equal,
    _runtime_publication_receipt_record,
    _runtime_publication_referenced_event_ids,
    _runtime_publication_storage_key,
    _session_run_operation_from_checkpoint,
    _stored_mcp_manifest_baseline_json,
    _terminal_publication_delete_block_reason,
    _terminal_session_evidence_expected_event_type,
    _tool_round_publication_identity,
    _validate_equivalent_queued_session_message,
    _validate_interaction_page,
    _validate_mcp_manifest_history_keys,
    _validate_mcp_manifest_publication_state,
    _validate_message_delivery_eligible_through,
    _validate_model_completion_active_marker_for_preparation,
    _validate_model_completion_active_marker_for_promotion,
    _validate_model_completion_preparation_replay_state,
    _validate_model_completion_promotion_replay_active_marker,
    _validate_model_completion_stage_for_abandonment,
    _validate_model_completion_stage_preparation_replay,
    _validate_model_completion_stage_publication,
    _validate_model_completion_stage_repreparation,
    _validate_model_completion_stage_terminal_replay,
    _validate_runner_observed_event_identity_snapshot,
    _validate_runtime_publication_durable_material,
    _validate_runtime_publication_event_references,
    _validate_runtime_publication_replay_receipt,
    _validate_session_fork_source,
    _validate_session_operation_record_keys,
    _validate_status_set,
    _validate_tool_round_call_ids,
    _validate_tool_round_checkpoint_mutation,
    _validate_tool_round_publication,
    build_session_topology_result,
    checkpoint_root_field_projection_from_storage,
    copy_enqueue_session_message_request,
    copy_event_query,
    copy_run_request,
    copy_session_aggregate_filter,
    copy_session_identity,
    copy_session_lineage_query,
    copy_session_query,
    copy_session_user_metadata,
    copy_transcript_messages,
    copy_transcript_query,
    copy_usage_rollup_query,
    decode_session_cursor,
    decode_session_lineage_cursor,
    decode_session_topology_cursor,
    encode_session_cursor,
    encode_session_lineage_cursor,
    enforce_pending_action_result_size,
    filter_transcript_records,
    fork_transcript_is_accepted,
    replace_session_user_metadata,
    resolve_interaction_attribution,
    restore_persisted_event_authority,
    session_next_cursor,
    session_outcome,
    session_query_from_aggregate_filter,
    transform_fork_checkpoint,
    validate_persisted_event_side_effect_error,
)
from cayu.runtime.tasks import (
    TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES,
    Task,
    TaskAggregateFilter,
    TaskCreate,
    TaskOperationalSnapshot,
    TaskOrder,
    TaskQuery,
    TaskStatus,
    TaskStatusCounts,
    TaskStore,
    TaskTopologyInconsistent,
    TaskTopologyNode,
    TaskTopologyQuery,
    TaskTopologyStoreResult,
    _allocate_task_topology_branch_limits,
    _bounded_optional_task_topology_parent_id,
    _copy_optional_status_payload,
    _copy_optional_status_reason,
    _ensure_can_hold_task,
    _ensure_can_resume_task,
    _ensure_can_transition,
    _ensure_claim_query_supported,
    _ensure_owned_active_task_lease,
    _raise_task_claim_attach_error,
    _running_task_from_create,
    _task_from_create,
    _validate_task_topology_ancestry,
    build_task_topology_result,
    copy_task_aggregate_filter,
    copy_task_create,
    copy_task_query,
    decode_task_topology_cursor,
    task_query_from_aggregate_filter,
)
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage import _sqlite_aggregates as sqlite_aggregates
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema

_EVENT_QUERY_SESSION_IDS_BATCH_SIZE = 500
_SQLITE_NON_SESSION_MIN_REQUIRED_REVISION = 18
_SQLITE_SESSION_MIN_REQUIRED_REVISION = 31
_SQLITE_TASK_TOPOLOGY_MIN_REQUIRED_REVISION = 27
_SQL_DIALECT = session_store_sql.SessionStoreSqlDialect(
    placeholder="?",
    contains_style="sqlite_nocase_like",
    datetime_param=sqlite_support.format_datetime,
)
_T = TypeVar("_T")


def _alias_key_fingerprint_matches(value: object, expected: str) -> bool:
    if type(value) is not str:
        return False
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return hmac.compare_digest(encoded, expected.encode("ascii"))


async def _run_off_thread_with_connection_ownership(
    lock: asyncio.Lock,
    connection: sqlite3.Connection,
    operation: Callable[[sqlite3.Connection], _T],
    *,
    executor: Executor | None = None,
) -> _T:
    """Keep a SQLite connection owned until its off-thread operation terminates.

    Cancelling an ``asyncio.to_thread`` await does not stop the worker thread.
    Defer caller cancellation while holding the connection lock so no subsequent
    operation or shutdown can reuse the connection before the worker has left it
    in a terminal transaction state.
    """

    async with lock:

        def capture_outcome() -> tuple[bool, object]:
            try:
                return True, operation(connection)
            except BaseException as worker_failure:
                # The executor future must complete normally even when the
                # operation raises CancelledError. That makes every cancellation
                # from shield() unambiguously caller-owned and keeps ownership
                # tied to the executor's physical completion.
                return False, worker_failure

        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        worker = loop.run_in_executor(executor, context.run, capture_outcome)
        cancellation: asyncio.CancelledError | None = None

        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
            except BaseException:
                if worker.done():
                    break
                raise

        succeeded, outcome = worker.result()
        if not succeeded:
            if not isinstance(outcome, BaseException):
                raise RuntimeError("SQLite worker returned an invalid failure outcome.")
            if cancellation is None:
                raise outcome
            cancellation.add_note(
                "SQLite worker failed while caller cancellation was pending: "
                f"{type(outcome).__name__}: {outcome}"
            )
            raise cancellation from outcome
        if cancellation is not None:
            raise cancellation
        return cast("_T", outcome)


def _like_contains_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _session_exists(connection: sqlite3.Connection, session_id: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM cayu_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return row is not None


def _transcript_cursor(connection: sqlite3.Connection, session_id: str) -> int:
    """Return the permanent next transcript position, independent of retention."""

    row = connection.execute(
        "SELECT transcript_seq FROM cayu_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Session not found: {session_id}")
    return int(row["transcript_seq"])


def _claim_budget_reservation_identity(
    connection: sqlite3.Connection,
    *,
    reservation_id: str,
    publication_session_id: str,
    publication_id: str,
) -> None:
    inserted = connection.execute(
        """
        INSERT OR IGNORE INTO cayu_budget_reservation_identities (
            reservation_id,
            publication_session_id,
            publication_id,
            published
        )
        VALUES (?, ?, ?, 0)
        """,
        (reservation_id, publication_session_id, publication_id),
    )
    if inserted.rowcount == 1:
        return
    existing = connection.execute(
        """
        SELECT publication_session_id, publication_id
        FROM cayu_budget_reservation_identities
        WHERE reservation_id = ?
        """,
        (reservation_id,),
    ).fetchone()
    assert existing is not None
    if (
        existing["publication_session_id"],
        existing["publication_id"],
    ) != (publication_session_id, publication_id):
        raise BudgetReservationIdentityConflict("Budget ledger reused a reservation identity.")


def _publish_budget_reservation_identities(
    connection: sqlite3.Connection,
    events: list[Event],
) -> None:
    for event in events:
        if event.type != EventType.BUDGET_RESERVED:
            continue
        raw_reservation_id = event.payload.get("reservation_id")
        if type(raw_reservation_id) is not str:
            continue
        updated = connection.execute(
            """
            UPDATE cayu_budget_reservation_identities
            SET published = 1
            WHERE reservation_id = ?
              AND publication_session_id = ?
              AND publication_id = ?
              AND published = 0
            """,
            (raw_reservation_id, event.session_id, event.id),
        )
        if updated.rowcount == 1:
            continue
        try:
            connection.execute(
                """
                INSERT INTO cayu_budget_reservation_identities (
                    reservation_id,
                    publication_session_id,
                    publication_id,
                    published
                )
                VALUES (?, ?, ?, 1)
                """,
                (raw_reservation_id, event.session_id, event.id),
            )
        except sqlite3.IntegrityError:
            existing = connection.execute(
                """
                SELECT publication_session_id, publication_id, published
                FROM cayu_budget_reservation_identities
                WHERE reservation_id = ?
                """,
                (raw_reservation_id,),
            ).fetchone()
            if (
                existing is not None
                and (
                    existing["publication_session_id"],
                    existing["publication_id"],
                    existing["published"],
                )
                == (event.session_id, event.id, 1)
                and connection.execute(
                    """
                    SELECT 1
                    FROM cayu_events
                    WHERE session_id = ? AND event_id = ?
                    """,
                    (event.session_id, event.id),
                ).fetchone()
                is not None
            ):
                # The reservation belongs to this exact persisted event. Let the
                # event insert below classify the replay as a duplicate event.
                continue
            raise BudgetReservationIdentityConflict(
                "Budget ledger reused a reservation identity."
            ) from None


def _raise_session_write_conflict(
    connection: sqlite3.Connection,
    session_id: str,
    expected_run_epoch: int,
) -> None:
    row = connection.execute(
        "SELECT run_epoch FROM cayu_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        raise KeyError(f"Session not found: {session_id}")
    raise SessionRunFenced(
        f"Session run epoch no longer owns {session_id}: expected {expected_run_epoch}, "
        f"current {row['run_epoch']}."
    )


def _touch_session_activity(
    connection: sqlite3.Connection,
    session_id: str,
    activity_at: datetime,
) -> None:
    expected_run_epoch = _current_session_run_epoch(session_id)
    if expected_run_epoch is None:
        cursor = connection.execute(
            "UPDATE cayu_sessions SET last_activity_at = ? WHERE id = ?",
            (sqlite_support.format_datetime(activity_at), session_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"Session not found: {session_id}")
        return
    cursor = connection.execute(
        "UPDATE cayu_sessions SET last_activity_at = ? WHERE id = ? AND run_epoch = ?",
        (sqlite_support.format_datetime(activity_at), session_id, expected_run_epoch),
    )
    if cursor.rowcount != 1:
        _raise_session_write_conflict(connection, session_id, expected_run_epoch)


def _load_labels(connection: sqlite3.Connection, session_id: str) -> dict[str, str]:
    rows = connection.execute(
        """
        SELECT key, value
        FROM cayu_session_labels
        WHERE session_id = ?
        ORDER BY key ASC
        """,
        (session_id,),
    ).fetchall()
    return {row["key"]: row["value"] for row in rows}


def _load_session(connection: sqlite3.Connection, session_id: str) -> Session | None:
    row = connection.execute(
        """
        SELECT id, agent_name, provider_name, model, parent_session_id,
               causal_budget_id, runtime_name, runtime_version, environment_name,
               status, created_at, updated_at, last_activity_at, run_epoch,
               metadata_json
        FROM cayu_sessions
        WHERE id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return sqlite_support.session_from_row(
        row,
        labels=_load_labels(connection, session_id),
    )


_SESSION_TOPOLOGY_COLUMNS = """
    id, agent_name, provider_name, model, parent_session_id,
    causal_budget_id, runtime_name, runtime_version, environment_name,
    status, created_at, updated_at, last_activity_at
"""


def _session_topology_node_from_sqlite_row(row: sqlite3.Row) -> SessionTopologyNode:
    return SessionTopologyNode(
        id=row["id"],
        agent_name=row["agent_name"],
        provider_name=row["provider_name"],
        model=row["model"],
        parent_session_id=row["parent_session_id"],
        causal_budget_id=row["causal_budget_id"],
        runtime_name=row["runtime_name"],
        runtime_version=row["runtime_version"],
        environment_name=row["environment_name"],
        status=SessionStatus(row["status"]),
        created_at=sqlite_support.parse_datetime(row["created_at"]),
        updated_at=sqlite_support.parse_datetime(row["updated_at"]),
        last_activity_at=sqlite_support.parse_datetime(row["last_activity_at"]),
    )


def _load_checkpoint_state(
    connection: sqlite3.Connection,
    session_id: str,
) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT state_json
        FROM cayu_checkpoints
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return copy_durable_json_object(json.loads(row["state_json"]), "checkpoint")


def _load_interruption_cascade_marker(
    connection: sqlite3.Connection,
    session_id: str,
    checkpoint_root_guard: CheckpointRootFieldGuard | None,
) -> dict[str, Any] | None:
    checkpoint_root_key = (
        "__cayu_no_checkpoint_root_guard__"
        if checkpoint_root_guard is None
        else checkpoint_root_guard.key
    )
    checkpoint_root_path = f"$.{checkpoint_root_key}"
    row = connection.execute(
        f"""
        SELECT
            json_type(state_json, '{checkpoint_root_path}')
                AS checkpoint_root_field_type,
            CASE
                WHEN json_type(state_json, '{checkpoint_root_path}') = 'integer'
                THEN substr(
                    CAST(json_extract(
                        state_json,
                        '{checkpoint_root_path}'
                    ) AS TEXT),
                    1,
                    {CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS + 1}
                )
            END AS checkpoint_root_field_scalar,
            json_type(state_json, '$.pending_interruption_cascade') AS marker_type,
            json_type(state_json, '$.pending_interruption_cascade.attempt_id') AS attempt_id_type,
            substr(
                CAST(json_extract(
                    state_json,
                    '$.pending_interruption_cascade.attempt_id'
                ) AS TEXT),
                1,
                129
            ) AS attempt_id,
            json_type(
                state_json,
                '$.pending_interruption_cascade.interrupt_payload'
            ) AS interrupt_payload_type,
            json_type(state_json, '$.pending_interruption_cascade.generation') AS generation_type,
            substr(
                CAST(json_extract(
                    state_json,
                    '$.pending_interruption_cascade.generation'
                ) AS TEXT),
                1,
                33
            ) AS generation,
            json_type(
                state_json,
                '$.pending_interruption_cascade.failure_recorded'
            ) AS failure_recorded_type,
            json_extract(
                state_json,
                '$.pending_interruption_cascade.failure_recorded'
            ) AS failure_recorded,
            json_type(state_json, '$.pending_interruption_cascade.claim_id') AS claim_id_type,
            substr(
                CAST(json_extract(
                    state_json,
                    '$.pending_interruption_cascade.claim_id'
                ) AS TEXT),
                1,
                129
            ) AS claim_id,
            json_type(
                state_json,
                '$.pending_interruption_cascade.claim_expires_at'
            ) AS claim_expires_at_type,
            substr(
                CAST(json_extract(
                    state_json,
                    '$.pending_interruption_cascade.claim_expires_at'
                ) AS TEXT),
                1,
                65
            ) AS claim_expires_at,
            json_type(state_json, '$.pending_interruption_cascade.created_at') AS created_at_type,
            substr(
                CAST(json_extract(
                    state_json,
                    '$.pending_interruption_cascade.created_at'
                ) AS TEXT),
                1,
                65
            ) AS created_at
        FROM cayu_checkpoints
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None

    def sqlite_json_type(value: str | None) -> str | None:
        if value == "text":
            return "string"
        if value == "real":
            return "number"
        if value in {"true", "false"}:
            return "boolean"
        return value

    checkpoint_root_field_type = row["checkpoint_root_field_type"]
    scalar_text = row["checkpoint_root_field_scalar"]
    if checkpoint_root_guard is not None:
        checkpoint_root_guard.validate(
            session_id,
            checkpoint_root_field_projection_from_storage(
                json_type=checkpoint_root_field_type,
                scalar_text=scalar_text,
            ),
        )

    field_names = (
        "attempt_id",
        "interrupt_payload",
        "generation",
        "failure_recorded",
        "claim_id",
        "claim_expires_at",
        "created_at",
    )
    field_types = {field: sqlite_json_type(row[f"{field}_type"]) for field in field_names}
    field_values = {
        "attempt_id": row["attempt_id"],
        "generation": row["generation"],
        "failure_recorded": (
            bool(row["failure_recorded"])
            if field_types["failure_recorded"] == "boolean"
            else row["failure_recorded"]
        ),
        "claim_id": row["claim_id"],
        "claim_expires_at": row["claim_expires_at"],
        "created_at": row["created_at"],
    }
    return _project_interruption_cascade_marker_fields(
        sqlite_json_type(row["marker_type"]),
        field_types,
        field_values,
    )


def _first_existing_event_id(
    connection: sqlite3.Connection,
    session_id: str,
    event_ids: list[str],
) -> str | None:
    for event_id in event_ids:
        row = connection.execute(
            "SELECT 1 FROM cayu_events WHERE session_id = ? AND event_id = ?",
            (session_id, event_id),
        ).fetchone()
        if row is not None:
            return event_id
    return None


def _decode_runtime_publication_record(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication receipt is malformed or conflicts with its key."
        ) from exc


def _decode_model_completion_stage_record(value: str) -> dict[str, Any]:
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


def _event_query_session_id_batches(
    session_ids: tuple[str, ...],
) -> list[tuple[str, ...]]:
    return [
        session_ids[index : index + _EVENT_QUERY_SESSION_IDS_BATCH_SIZE]
        for index in range(0, len(session_ids), _EVENT_QUERY_SESSION_IDS_BATCH_SIZE)
    ]


# Columns needed to reconstruct an Event, in a stable order. The formerly-stored
# event_json blob duplicated exactly these (plus payload_json), so the store now
# rebuilds Events from the individual columns instead of parsing a redundant copy.
_EVENT_COLUMN_NAMES: tuple[str, ...] = (
    "session_id",
    "event_id",
    "interaction_id",
    "event_type",
    "timestamp",
    "agent_name",
    "environment_name",
    "workflow_name",
    "tool_name",
    "payload_json",
    "input_contract_runtime_owned",
)

# Keep this predicate text aligned with the revision-17 partial index. SQLite
# can prove a parameterized lifecycle subset is covered by that index only when
# the query also carries the index's literal predicate.
_PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL = """
    event_type IN (
        'tool.call.approval_requested',
        'session.awaiting_user_input',
        'session.interrupted',
        'tool.call.started',
        'tool.call.completed',
        'tool.call.failed',
        'tool.call.blocked',
        'tool.call.approval_denied'
    )
    AND pending_action_lookup_key IS NOT NULL
"""


def _event_from_row(row: sqlite3.Row) -> Event:
    """Reconstruct an :class:`Event` from its individual cayu_events columns."""
    input_contract_runtime_owned = row["input_contract_runtime_owned"]
    if type(input_contract_runtime_owned) is not int or input_contract_runtime_owned not in {
        0,
        1,
    }:
        raise ValueError("Stored input-contract authority proof is malformed.")
    return restore_persisted_event_authority(
        Event(
            type=row["event_type"],
            session_id=row["session_id"],
            interaction_id=row["interaction_id"],
            id=row["event_id"],
            timestamp=row["timestamp"],
            agent_name=row["agent_name"],
            environment_name=row["environment_name"],
            workflow_name=row["workflow_name"],
            tool_name=row["tool_name"],
            payload=json.loads(row["payload_json"]),
        ),
        input_contract_runtime_owned=input_contract_runtime_owned == 1,
    )


def _event_record_from_row(row: sqlite3.Row | None) -> EventRecord | None:
    if row is None:
        return None
    return EventRecord(
        sequence=row["sequence"],
        event=_event_from_row(row),
    )


def _persisted_event_side_effect_delivery_from_row(
    row: sqlite3.Row,
) -> PersistedEventSideEffectDelivery:
    return PersistedEventSideEffectDelivery(
        session_id=row["session_id"],
        event_id=row["event_id"],
        event_sequence=row["event_sequence"],
        status=PersistedEventSideEffectStatus(row["status"]),
        attempts=row["attempts"],
        claim_id=row["claim_id"],
        lease_expires_at=(
            None
            if row["lease_expires_at"] is None
            else sqlite_support.parse_datetime(row["lease_expires_at"])
        ),
        next_attempt_at=(
            None
            if row["next_attempt_at"] is None
            else sqlite_support.parse_datetime(row["next_attempt_at"])
        ),
        last_error=row["last_error"],
        updated_at=sqlite_support.parse_datetime(row["updated_at"]),
    )


def _enqueue_persisted_event_side_effects(
    connection: sqlite3.Connection,
    session_id: str,
    events: Sequence[Event],
) -> None:
    if not events:
        return
    event_ids: list[str] = []
    runtime_owned_input_contract_event_ids: list[str] = []
    for event in events:
        event_ids.append(event.id)
        if _event_input_contract_is_runtime_owned(event):
            runtime_owned_input_contract_event_ids.append(event.id)
    # Rows predating revision 31 may contain caller-authored payload text but
    # cannot carry the proof bit, so that text remains untrusted after migration.
    if runtime_owned_input_contract_event_ids:
        connection.executemany(
            """
            UPDATE cayu_events
            SET input_contract_runtime_owned = 1
            WHERE session_id = ?
              AND event_id = ?
              AND event_type = 'session.started'
              AND json_type(payload_json, '$.input_contract') = 'text'
            """,
            [(session_id, event_id) for event_id in runtime_owned_input_contract_event_ids],
        )
    connection.executemany(
        """
        INSERT INTO cayu_persisted_event_side_effects (
            session_id, event_id, event_sequence, status, attempts, updated_at
        )
        SELECT session_id, event_id, sequence, 'pending', 0, timestamp
        FROM cayu_events
        WHERE session_id = ?
          AND event_id = ?
          AND event_type <> 'runtime.sink.failed'
        """,
        [(session_id, event_id) for event_id in event_ids],
    )


def _queued_session_message_from_row(row: sqlite3.Row) -> SessionQueuedMessage:
    requested_by = row["requested_by_json"]
    return SessionQueuedMessage(
        queue_id=row["queue_id"],
        session_id=row["session_id"],
        idempotency_key=row["idempotency_key"],
        content=row["content"],
        delivery_mode=row["delivery_mode"],
        status=row["status"],
        ordering_key=row["ordering_key"],
        accepted_run_epoch=row["accepted_run_epoch"],
        accepted_transcript_cursor=row["accepted_transcript_cursor"],
        accepted_event_id=row["accepted_event_id"],
        accepted_at=sqlite_support.parse_datetime(row["accepted_at"]),
        requested_by=(
            None
            if requested_by is None
            else ResolutionActor.model_validate(json.loads(requested_by))
        ),
        delivered_run_epoch=row["delivered_run_epoch"],
        delivered_transcript_cursor=row["delivered_transcript_cursor"],
        delivered_event_id=row["delivered_event_id"],
        delivered_at=(
            None
            if row["delivered_at"] is None
            else sqlite_support.parse_datetime(row["delivered_at"])
        ),
    )


class SQLiteSessionStore(SessionStore):
    """SQLite-backed session store for durable local runtime state."""

    supports_usage_aggregates: ClassVar[bool] = True
    supports_mcp_manifest_history: ClassVar[bool] = True
    supports_public_authority_aliases: ClassVar[bool] = True
    supports_session_topology: ClassVar[bool] = True
    supports_session_lineage: ClassVar[bool] = True
    supports_terminal_session_evidence: ClassVar[bool] = True
    supports_runner_owned_interrupted_evidence: ClassVar[bool] = True

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
        read_only: bool = False,
        public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
    ) -> None:
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str:
            db_path = Path(require_nonblank(path, "path"))
        else:
            raise TypeError("SQLiteSessionStore path must be a string or Path.")
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")
        if not isinstance(read_only, bool):
            raise TypeError("read_only must be a bool.")
        self.service_durability = (
            RuntimeStoreDurability.READ_ONLY
            if read_only
            else (
                RuntimeStoreDurability.DEVELOPMENT
                if str(db_path) == ":memory:"
                else RuntimeStoreDurability.DURABLE
            )
        )
        if public_authority_alias_codec is not None and not isinstance(
            public_authority_alias_codec,
            PublicAuthorityAliasCodec,
        ):
            raise TypeError("public_authority_alias_codec must be a PublicAuthorityAliasCodec.")
        if read_only and schema_mode is not schema.SchemaMode.VALIDATE:
            raise ValueError("read_only SQLite stores require schema_mode=validate.")

        self.path = db_path
        self._schema_mode = schema_mode
        self._read_only = read_only
        self._public_authority_alias_codec = public_authority_alias_codec
        self._lock = asyncio.Lock()
        self._connection = self._connect_read_only(db_path) if read_only else self._connect(db_path)
        try:
            self._register_public_authority_alias_sql_function(self._connection)
            self._initialize_schema()
            self._initialize_public_authority_alias_registry()
        except BaseException:
            self._connection.close()
            raise
        # Hot-path queries run on a dedicated read-only connection in worker
        # threads so the event loop never blocks on SQLite I/O and reads never
        # queue behind the writer connection's transactions. In-memory
        # databases are private to their connection, so they fall back to the
        # writer connection (and its lock).
        if read_only or str(db_path) == ":memory:":
            self._read_connection = self._connection
            self._read_lock = self._lock
        else:
            self._read_connection = self._connect_read_only(db_path)
            self._read_lock = asyncio.Lock()

    @property
    def public_authority_alias_codec(self) -> PublicAuthorityAliasCodec | None:
        """Return the immutable codec bound to this store's durable alias registry."""

        return self._public_authority_alias_codec

    def _register_public_authority_alias_sql_function(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        codec = self._public_authority_alias_codec

        def public_authority_alias(
            private_value: object,
            field_name: object,
            scope_session_id: object,
        ) -> str | None:
            if codec is None:
                return None
            if type(private_value) is not str or type(field_name) is not str:
                raise ValueError("Public authority alias source must be text.")
            if scope_session_id is not None and type(scope_session_id) is not str:
                raise ValueError("Public authority alias scope must be text or null.")
            return codec.encode(
                private_value,
                field_name=field_name,
                session_id=scope_session_id,
            )

        connection.create_function(
            "cayu_public_authority_alias",
            3,
            public_authority_alias,
            deterministic=True,
        )

        def public_authority_aliases(
            private_value: object,
            field_name: object,
            scope_session_id: object,
        ) -> str:
            if codec is None:
                return "[]"
            if type(private_value) is not str or type(field_name) is not str:
                raise ValueError("Public authority alias source must be text.")
            if scope_session_id is not None and type(scope_session_id) is not str:
                raise ValueError("Public authority alias scope must be text or null.")
            return json.dumps(
                codec.aliases(
                    private_value,
                    field_name=field_name,
                    session_id=scope_session_id,
                ),
                separators=(",", ":"),
            )

        connection.create_function(
            "cayu_public_authority_aliases",
            3,
            public_authority_aliases,
            deterministic=True,
        )
        connection.create_function(
            "cayu_public_authority_active_key_id",
            0,
            lambda: None if codec is None else codec.keyring.active_key_id,
            deterministic=True,
        )
        connection.create_function(
            "cayu_public_authority_keyring_fingerprint",
            0,
            lambda: None if codec is None else codec.keyring_fingerprint(),
            deterministic=True,
        )

    def _initialize_public_authority_alias_registry(self) -> None:
        """Validate key continuity and backfill each newly configured signing key."""

        codec = self._public_authority_alias_codec
        if codec is None:
            initialized = self._connection.execute(
                "SELECT EXISTS(SELECT 1 FROM cayu_public_authority_alias_config)"
            ).fetchone()[0]
            if initialized:
                raise ValueError(
                    "A public authority alias codec is required for this initialized store."
                )
            return

        configured = tuple(
            (key_id, codec.key_fingerprint(key_id)) for key_id in codec.keyring.key_ids
        )
        if self._read_only:
            rows: dict[str, tuple[object, object]] = {}
            for row in self._connection.execute(
                "SELECT key_id, fingerprint, backfill_completed "
                "FROM cayu_public_authority_alias_keys"
            ):
                if type(row["key_id"]) is str:
                    rows[row["key_id"]] = (row["fingerprint"], row["backfill_completed"])
            for key_id, fingerprint in configured:
                existing = rows.get(key_id)
                if existing is None or existing[1] != 1:
                    raise ValueError(
                        "Read-only store has not completed the configured alias-key backfill."
                    )
                if not _alias_key_fingerprint_matches(existing[0], fingerprint):
                    raise ValueError(
                        "Public authority alias key material conflicts with durable state."
                    )
            config = self._connection.execute(
                "SELECT active_key_id, keyring_fingerprint "
                "FROM cayu_public_authority_alias_config "
                "WHERE singleton = 1"
            ).fetchone()
            if (
                config is None
                or config["active_key_id"] != codec.keyring.active_key_id
                or config["keyring_fingerprint"] != codec.keyring_fingerprint()
            ):
                raise ValueError("Read-only store public authority alias active key is stale.")
            return

        try:
            with self._connection:
                for key_id, fingerprint in configured:
                    self._connection.execute(
                        """
                        INSERT INTO cayu_public_authority_alias_keys (
                            key_id, fingerprint, backfill_completed
                        ) VALUES (?, ?, 0)
                        ON CONFLICT(key_id) DO NOTHING
                        """,
                        (key_id, fingerprint),
                    )
                    row = self._connection.execute(
                        """
                        SELECT fingerprint, backfill_completed
                        FROM cayu_public_authority_alias_keys
                        WHERE key_id = ?
                        """,
                        (key_id,),
                    ).fetchone()
                    if row is None:  # pragma: no cover - guarded by the insert above
                        raise RuntimeError("Public authority alias key state was not persisted.")
                    if not _alias_key_fingerprint_matches(row["fingerprint"], fingerprint):
                        raise ValueError(
                            "Public authority alias key material conflicts with durable state."
                        )

                pending = self._connection.execute(
                    """
                    SELECT EXISTS(
                        SELECT 1
                        FROM cayu_public_authority_alias_keys
                        WHERE key_id IN ({}) AND backfill_completed = 0
                    )
                    """.format(", ".join("?" for _ in configured)),
                    tuple(key_id for key_id, _fingerprint in configured),
                ).fetchone()[0]
                if pending:
                    self._backfill_public_authority_aliases()
                    self._connection.executemany(
                        """
                        UPDATE cayu_public_authority_alias_keys
                        SET backfill_completed = 1
                        WHERE key_id = ?
                        """,
                        ((key_id,) for key_id, _fingerprint in configured),
                    )
                config = self._connection.execute(
                    "SELECT active_key_id, keyring_fingerprint, generation, "
                    "retired_key_ids_json "
                    "FROM cayu_public_authority_alias_config WHERE singleton = 1"
                ).fetchone()
                desired_active = codec.keyring.active_key_id
                desired_keyring_fingerprint = codec.keyring_fingerprint()
                if config is None:
                    self._connection.execute(
                        "INSERT INTO cayu_public_authority_alias_config "
                        "(singleton, active_key_id, keyring_fingerprint, generation, "
                        "retired_key_ids_json) VALUES (1, ?, ?, 1, '[]')",
                        (desired_active, desired_keyring_fingerprint),
                    )
                elif (
                    config["active_key_id"] != desired_active
                    or config["keyring_fingerprint"] != desired_keyring_fingerprint
                ):
                    retired = json.loads(config["retired_key_ids_json"])
                    if type(retired) is not list or not all(
                        type(value) is str for value in retired
                    ):
                        raise ValueError("Public authority alias rotation state is malformed.")
                    if config["active_key_id"] != desired_active and desired_active in retired:
                        raise ValueError(
                            "A retired public authority alias key cannot become active again."
                        )
                    if config["active_key_id"] != desired_active:
                        retired.append(str(config["active_key_id"]))
                    self._connection.execute(
                        "UPDATE cayu_public_authority_alias_config "
                        "SET active_key_id = ?, keyring_fingerprint = ?, generation = ?, "
                        "retired_key_ids_json = ? "
                        "WHERE singleton = 1",
                        (
                            desired_active,
                            desired_keyring_fingerprint,
                            int(config["generation"]) + 1,
                            json.dumps(list(dict.fromkeys(retired)), separators=(",", ":")),
                        ),
                    )
        except sqlite3.IntegrityError:
            raise ValueError(
                "Public authority alias registry conflicts with durable authority."
            ) from None

    def _backfill_public_authority_aliases(self) -> None:
        self._connection.execute(
            """
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT
                'session_id',
                '',
                alias.value,
                session.id
            FROM cayu_sessions AS session,
                 json_each(
                     cayu_public_authority_aliases(session.id, 'session_id', NULL)
                 ) AS alias
            """
        )
        self._connection.execute(
            """
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT DISTINCT
                'interaction_id',
                event.session_id,
                alias.value,
                event.interaction_id
            FROM cayu_events AS event,
                 json_each(cayu_public_authority_aliases(
                     event.interaction_id, 'interaction_id', event.session_id
                 )) AS alias
            WHERE event.interaction_id IS NOT NULL
            """
        )
        self._connection.execute(
            """
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT DISTINCT
                'interaction_id',
                transcript.session_id,
                alias.value,
                transcript.interaction_id
            FROM cayu_transcript_messages AS transcript,
                 json_each(cayu_public_authority_aliases(
                     transcript.interaction_id, 'interaction_id', transcript.session_id
                 )) AS alias
            WHERE transcript.interaction_id IS NOT NULL
            """
        )
        self._connection.execute(
            """
            INSERT OR IGNORE INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            )
            SELECT DISTINCT
                'interaction_id',
                event.session_id,
                alias.value,
                interaction.value
            FROM cayu_events AS event,
                 json_each(event.payload_json, '$.interaction_ids') AS interaction,
                 json_each(cayu_public_authority_aliases(
                     interaction.value, 'interaction_id', event.session_id
                 )) AS alias
            WHERE event.event_type = 'turn.completed'
              AND json_valid(event.payload_json)
              AND json_type(event.payload_json, '$.interaction_ids') = 'array'
              AND interaction.type = 'text'
              AND trim(interaction.value) <> ''
            """
        )

    async def _run_read(self, query: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run a read-only query off the event loop on the read connection."""

        def guarded(connection: sqlite3.Connection) -> _T:
            self._require_current_public_authority_configuration(connection)
            return query(connection)

        return await _run_off_thread_with_connection_ownership(
            self._read_lock,
            self._read_connection,
            guarded,
        )

    async def _run_write(self, statement: Callable[[sqlite3.Connection], _T]) -> _T:
        """Run a write statement off the event loop on the writer connection."""

        def guarded(connection: sqlite3.Connection) -> _T:
            # This is the fail-fast check. The session/event/transcript BEFORE
            # INSERT triggers repeat it after SQLite has acquired the writer
            # transaction, closing the cross-process key-rotation race between
            # this check and an identity-producing statement.
            self._require_current_public_authority_configuration(connection)
            return statement(connection)

        return await _run_off_thread_with_connection_ownership(
            self._lock,
            self._connection,
            guarded,
        )

    def _require_current_public_authority_configuration(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        codec = self.public_authority_alias_codec
        row = connection.execute(
            "SELECT active_key_id, keyring_fingerprint "
            "FROM cayu_public_authority_alias_config WHERE singleton = 1"
        ).fetchone()
        if codec is None:
            if row is not None:
                raise RuntimeError(
                    "SQLite public authority aliases require the deployment keyring."
                )
            return
        if (
            row is None
            or row["active_key_id"] != codec.keyring.active_key_id
            or row["keyring_fingerprint"] != codec.keyring_fingerprint()
        ):
            raise RuntimeError(
                "SQLite public authority alias key configuration is stale; reopen the store."
            )

    async def register_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        private_value: str,
        scope_session_id: str | None = None,
    ) -> None:
        """Atomically register one codec-authenticated public authority alias."""

        field_name, scope_key, public_alias = _public_authority_alias_store_key(
            public_alias,
            field_name=field_name,
            private_value=private_value,
            scope_session_id=scope_session_id,
        )
        codec = self.public_authority_alias_codec
        if codec is None or not codec.matches(
            public_alias,
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        ):
            raise ValueError("Public authority alias lacks valid store-configured provenance.")

        def statement(connection: sqlite3.Connection) -> None:
            with connection:
                try:
                    connection.execute(
                        """
                        INSERT INTO cayu_public_authority_aliases (
                            field_name,
                            scope_session_id,
                            public_alias,
                            private_value
                        ) VALUES (?, ?, ?, ?)
                        ON CONFLICT(field_name, scope_session_id, public_alias) DO NOTHING
                        """,
                        (field_name, scope_key, public_alias, private_value),
                    )
                except sqlite3.IntegrityError:
                    raise ValueError(
                        "Public authority alias conflicts with existing private authority."
                    ) from None
                row = connection.execute(
                    """
                    SELECT private_value
                    FROM cayu_public_authority_aliases
                    WHERE field_name = ?
                      AND scope_session_id = ?
                      AND public_alias = ?
                    """,
                    (field_name, scope_key, public_alias),
                ).fetchone()
                if row is None:  # pragma: no cover - guarded by the insert above
                    raise RuntimeError("Public authority alias registration was not persisted.")
                stored = str(row["private_value"])
                if not hmac.compare_digest(
                    stored.encode("utf-8"),
                    private_value.encode("utf-8"),
                ):
                    raise ValueError(
                        "Public authority alias conflicts with existing private authority."
                    )

        await self._run_write(statement)

    async def resolve_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> str | None:
        """Resolve one exact alias through its indexed authority scope."""

        field_name, scope_key, public_alias = _public_authority_alias_store_key(
            public_alias,
            field_name=field_name,
            private_value=None,
            scope_session_id=scope_session_id,
        )

        def query(connection: sqlite3.Connection) -> str | None:
            row = connection.execute(
                """
                SELECT private_value
                FROM cayu_public_authority_aliases
                WHERE field_name = ?
                  AND scope_session_id = ?
                  AND public_alias = ?
                """,
                (field_name, scope_key, public_alias),
            ).fetchone()
            if row is None:
                return None
            return _authenticated_public_authority_alias_private_value(
                self.public_authority_alias_codec,
                public_alias,
                row["private_value"],
                field_name=field_name,
                scope_session_id=scope_session_id,
            )

        return await self._run_read(query)

    async def public_authority_private_value_exists(
        self,
        private_value: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> bool:
        codec = self.public_authority_alias_codec
        if codec is None:
            raise RuntimeError("Public authority alias codec is unavailable.")
        probe = codec.encode(
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        )
        field_name, scope_key, _probe = _public_authority_alias_store_key(
            probe,
            field_name=field_name,
            private_value=private_value,
            scope_session_id=scope_session_id,
        )

        def query(connection: sqlite3.Connection) -> bool:
            return (
                connection.execute(
                    """
                    SELECT EXISTS(
                        SELECT 1
                        FROM cayu_public_authority_aliases
                        WHERE field_name = ?
                          AND scope_session_id = ?
                          AND private_value = ?
                    )
                    """,
                    (field_name, scope_key, private_value),
                ).fetchone()[0]
                == 1
            )

        return await self._run_read(query)

    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        checkpoint_transform: CheckpointTransform | None = None,
    ) -> Session:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        request = copy_run_request(request)
        identity = copy_session_identity(identity)
        async with self._lock:
            self._require_current_public_authority_configuration(self._connection)
            session = sqlite_support.session_from_request(request, identity=identity)
            admission = _copy_optional_interaction_admission(
                session.id,
                interaction_started_event,
                interaction_source_messages,
                defer_transcript=True,
            )
            if admission is not None:
                session = session.model_copy(
                    update={"status": SessionStatus.RUNNING, "run_epoch": 1}
                )
            if session.parent_session_id == session.id:
                raise ValueError("Session cannot be its own parent.")
            try:
                with self._connection:
                    self._connection.execute(
                        """
                        INSERT INTO cayu_sessions (
                            id,
                            agent_name,
                            provider_name,
                            model,
                            parent_session_id,
                            causal_budget_id,
                            runtime_name,
                            runtime_version,
                            environment_name,
                            status,
                            created_at,
                            updated_at,
                            last_activity_at,
                            run_epoch,
                            metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session.id,
                            session.agent_name,
                            session.provider_name,
                            session.model,
                            session.parent_session_id,
                            session.causal_budget_id,
                            session.runtime_name,
                            session.runtime_version,
                            session.environment_name,
                            str(session.status),
                            sqlite_support.format_datetime(session.created_at),
                            sqlite_support.format_datetime(session.updated_at),
                            sqlite_support.format_datetime(session.last_activity_at),
                            session.run_epoch,
                            sqlite_support.json_dumps(session.metadata),
                        ),
                    )
                    if session.labels:
                        self._connection.executemany(
                            """
                            INSERT INTO cayu_session_labels (session_id, key, value)
                            VALUES (?, ?, ?)
                            """,
                            sqlite_support.session_label_row_values(session),
                        )
                    if admission is not None:
                        started_event, source_messages = admission
                        interaction_id = started_event.interaction_id
                        if interaction_id is None:
                            raise AssertionError("Interaction admission lost its identity.")
                        lookup_key, projection, projection_bytes = (
                            pending_action_event_storage_values(started_event)
                        )
                        self._connection.execute(
                            """
                            INSERT INTO cayu_events (
                                session_id, event_id, interaction_id, event_type,
                                timestamp, agent_name, environment_name, workflow_name,
                                tool_name, payload_json, pending_action_lookup_key,
                                pending_action_projection_json,
                                pending_action_projection_bytes
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                session.id,
                                started_event.id,
                                interaction_id,
                                str(started_event.type),
                                sqlite_support.format_datetime(started_event.timestamp),
                                started_event.agent_name,
                                started_event.environment_name,
                                started_event.workflow_name,
                                started_event.tool_name,
                                sqlite_support.json_dumps(started_event.payload),
                                lookup_key,
                                projection,
                                projection_bytes,
                            ),
                        )
                        _enqueue_persisted_event_side_effects(
                            self._connection,
                            session.id,
                            [started_event],
                        )
                        self._connection.execute(
                            "INSERT INTO cayu_deferred_interaction_inputs "
                            "(session_id, interaction_id, source_messages_json) "
                            "VALUES (?, ?, ?)",
                            (
                                session.id,
                                interaction_id,
                                sqlite_support.json_dumps(
                                    [message.model_dump(mode="json") for message in source_messages]
                                ),
                            ),
                        )
                        self._connection.execute(
                            """
                            INSERT INTO cayu_checkpoints (
                                session_id, state_json, updated_at,
                                pending_action_source_bytes,
                                pending_action_tool_call_count,
                                pending_action_flags,
                                pending_action_metrics_ready
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            sqlite_support.checkpoint_row_values(
                                session.id,
                                _initial_transcript_pending_checkpoint(
                                    session,
                                    interaction_id,
                                    checkpoint_transform=checkpoint_transform,
                                ),
                                session.updated_at,
                            ),
                        )
            except sqlite3.IntegrityError as exc:
                if self._session_exists_unlocked(session.id):
                    raise ValueError(f"Session already exists: {session.id}") from exc
                if session.parent_session_id is not None and not self._session_exists_unlocked(
                    session.parent_session_id
                ):
                    raise ValueError(
                        f"Parent session not found: {session.parent_session_id}"
                    ) from exc
                raise
            if admission is not None:
                _activate_session_run_fence(session)
            return session.model_copy(deep=True)

    async def create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        expected_source_run_epoch: int,
    ) -> Session:
        return await self._create_fork(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=None,
        )

    async def create_fork_with_transcript_validation(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        expected_source_run_epoch: int,
        transcript_validator: ForkTranscriptValidator,
    ) -> Session:
        return await self._create_fork(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=transcript_validator,
        )

    async def _create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        expected_source_run_epoch: int,
        transcript_validator: ForkTranscriptValidator | None,
    ) -> Session:
        source_session_id, fork, allowed_statuses, transcript_cursor = (
            _prepare_session_fork_request(
                source_session_id=source_session_id,
                fork=fork,
                source_statuses=source_statuses,
                transcript_cursor=transcript_cursor,
            )
        )

        async with self._lock:
            self._require_current_public_authority_configuration(self._connection)
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                source_session = _validate_session_fork_source(
                    source_session=self._load_unlocked(source_session_id),
                    source_session_id=source_session_id,
                    fork=fork,
                    allowed_statuses=allowed_statuses,
                    expected_source_run_epoch=expected_source_run_epoch,
                )
                active_stage = self._connection.execute(
                    "SELECT 1 FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (
                        source_session_id,
                        MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                    ),
                ).fetchone()
                if active_stage is not None:
                    raise SessionForkActiveModelStageConflict(
                        "Cannot fork a session while a model-completion stage is active."
                    )
                source_transcript_cursor = _transcript_cursor(
                    self._connection,
                    source_session_id,
                )
                if transcript_cursor is not None and transcript_cursor > source_transcript_cursor:
                    raise ValueError("transcript_cursor is greater than source transcript length.")
                selected_transcript_rows = self._connection.execute(
                    """
                    SELECT message_json, interaction_id
                    FROM cayu_transcript_messages
                    WHERE session_id = ?
                      AND session_order <= ?
                    ORDER BY session_order ASC
                    """,
                    (
                        source_session_id,
                        (
                            source_transcript_cursor
                            if transcript_cursor is None
                            else transcript_cursor
                        ),
                    ),
                ).fetchall()
                copied_messages = [
                    Message(**json.loads(row["message_json"])) for row in selected_transcript_rows
                ]
                copied_interaction_ids = [row["interaction_id"] for row in selected_transcript_rows]
                selected_transcript_rows.clear()
                if not fork_transcript_is_accepted(copied_messages, transcript_validator):
                    copied_messages.clear()
                    copied_messages = []
                    raise ValueError(FORK_TRANSCRIPT_VALIDATION_ERROR) from None
                copied_checkpoint = None
                if checkpoint_transform is not None:
                    checkpoint_input = self._load_checkpoint_unlocked(source_session_id)
                    copied_checkpoint = transform_fork_checkpoint(
                        source_session,
                        checkpoint_input,
                        checkpoint_transform,
                    )
                    checkpoint_input = None
                    if copied_checkpoint is not None:
                        copied_checkpoint = copy_durable_json_object(
                            copied_checkpoint,
                            "checkpoint",
                        )

                self._connection.execute(
                    """
                    INSERT INTO cayu_sessions (
                        id,
                        agent_name,
                        provider_name,
                        model,
                        parent_session_id,
                        causal_budget_id,
                        runtime_name,
                        runtime_version,
                        environment_name,
                        status,
                        created_at,
                        updated_at,
                        last_activity_at,
                        run_epoch,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    sqlite_support.session_to_row_values(fork),
                )
                if fork.labels:
                    self._connection.executemany(
                        """
                        INSERT INTO cayu_session_labels (session_id, key, value)
                        VALUES (?, ?, ?)
                        """,
                        sqlite_support.session_label_row_values(fork),
                    )
                if copied_messages:
                    self._connection.executemany(
                        """
                        INSERT INTO cayu_transcript_messages (
                            session_id,
                            role,
                            interaction_id,
                            message_json
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        [
                            (
                                fork.id,
                                str(message.role),
                                copied_interaction_ids[index],
                                sqlite_support.json_dumps(message.model_dump(mode="json")),
                            )
                            for index, message in enumerate(copied_messages)
                        ],
                    )
                if copied_checkpoint is not None:
                    self._connection.execute(
                        """
                        INSERT INTO cayu_checkpoints (
                            session_id, state_json, updated_at,
                            pending_action_source_bytes,
                            pending_action_tool_call_count,
                            pending_action_flags,
                            pending_action_metrics_ready
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        sqlite_support.checkpoint_row_values(
                            fork.id, copied_checkpoint, fork.updated_at
                        ),
                    )
                self._connection.commit()
            except sqlite3.IntegrityError as exc:
                self._connection.rollback()
                if self._session_exists_unlocked(fork.id):
                    raise ValueError(f"Session already exists: {fork.id}") from exc
                raise
            except Exception:
                self._connection.rollback()
                raise

            loaded = self._load_unlocked(fork.id)
            if loaded is None:
                raise KeyError(f"Session not found: {fork.id}")
            return loaded

    async def load(self, session_id: str) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        return await self._run_read(lambda connection: _load_session(connection, session_id))

    async def load_state(self, session_id: str) -> SessionStateSnapshot | None:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> SessionStateSnapshot | None:
            row = connection.execute(
                """
                SELECT id, status, updated_at, last_activity_at
                FROM cayu_sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return SessionStateSnapshot(
                id=row["id"],
                status=SessionStatus(row["status"]),
                updated_at=sqlite_support.parse_datetime(row["updated_at"]),
                last_activity_at=sqlite_support.parse_datetime(row["last_activity_at"]),
            )

        return await self._run_read(query)

    async def inspect_identity(self, session_id: str) -> SessionInspectionIdentity:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> SessionInspectionIdentity:
            row = connection.execute(
                """
                SELECT id, agent_name, provider_name, model, parent_session_id,
                       causal_budget_id, runtime_name, runtime_version, environment_name,
                       status, created_at, updated_at, last_activity_at, run_epoch
                FROM cayu_sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                raise KeyError(session_id)
            label_rows = connection.execute(
                """
                SELECT key, value,
                       (SELECT COUNT(*)
                        FROM cayu_session_labels
                        WHERE session_id = ?) AS label_count
                FROM cayu_session_labels
                WHERE session_id = ?
                ORDER BY key ASC
                LIMIT ?
                """,
                (session_id, session_id, SESSION_INSPECTION_LABEL_LIMIT),
            ).fetchall()
            label_count = 0 if not label_rows else label_rows[0]["label_count"]
            return SessionInspectionIdentity(
                id=row["id"],
                agent_name=row["agent_name"],
                provider_name=row["provider_name"],
                model=row["model"],
                parent_session_id=row["parent_session_id"],
                causal_budget_id=row["causal_budget_id"],
                runtime_name=row["runtime_name"],
                runtime_version=row["runtime_version"],
                environment_name=row["environment_name"],
                status=SessionStatus(row["status"]),
                created_at=sqlite_support.parse_datetime(row["created_at"]),
                updated_at=sqlite_support.parse_datetime(row["updated_at"]),
                last_activity_at=sqlite_support.parse_datetime(row["last_activity_at"]),
                run_epoch=row["run_epoch"],
                labels={label_row["key"]: label_row["value"] for label_row in label_rows},
                label_count=label_count,
                labels_truncated=label_count > len(label_rows),
            )

        return await self._run_read(query)

    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(status, SessionStatus):
            raise ValueError("Session status must be a SessionStatus.")
        # Route the unconditional setter through the guarded transition machine so
        # both write paths share one atomic UPDATE-and-check. Allowing every source
        # status preserves update_status semantics (any -> status) while inheriting
        # the row-level not-found guard.
        return await self.transition_status(
            session_id,
            from_statuses=set(SessionStatus),
            to_status=status,
        )

    async def update_model(self, session_id: str, model: str) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        model = require_clean_nonblank(model, "model")
        updated_at = datetime.now(UTC)
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._lock:
            with self._connection:
                epoch_clause = "" if expected_run_epoch is None else " AND run_epoch = ?"
                params: list[object] = [
                    model,
                    sqlite_support.format_datetime(updated_at),
                    sqlite_support.format_datetime(updated_at),
                    session_id,
                ]
                if expected_run_epoch is not None:
                    params.append(expected_run_epoch)
                cursor = self._connection.execute(
                    f"""
                    UPDATE cayu_sessions
                    SET model = ?, updated_at = ?, last_activity_at = ?
                    WHERE id = ?{epoch_clause}
                    """,
                    params,
                )
            if cursor.rowcount != 1:
                if expected_run_epoch is not None:
                    _raise_session_write_conflict(self._connection, session_id, expected_run_epoch)
                raise KeyError(f"Session not found: {session_id}")

            loaded = self._load_unlocked(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            return loaded

    async def delete_session(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                session = self._load_unlocked(session_id)
                if session is None:
                    self._connection.rollback()
                    return
                if session.status in DELETE_BLOCKED_SESSION_STATUSES:
                    raise ValueError(
                        f"Cannot delete a session while it is {session.status}; "
                        f"interrupt it first: {session_id}"
                    )
                checkpoint = self._load_checkpoint_unlocked(session_id)
                deletion_now = datetime.now(UTC)
                active_recovery_claim_id = _active_unexpired_incomplete_recovery_claim_id(
                    checkpoint,
                    now=deletion_now,
                )
                if active_recovery_claim_id is not None:
                    raise ValueError(
                        "Cannot delete a session while incomplete-session recovery claim "
                        f"{active_recovery_claim_id} is active: {session_id}"
                    )
                run_operation = _session_run_operation_from_checkpoint(checkpoint)
                if run_operation is not None:
                    raise ValueError(
                        "Cannot delete a session while terminal publication "
                        f"{run_operation.operation_id} is incomplete: {session_id}"
                    )
                terminal_evidence_rows = self._connection.execute(
                    f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                    "WHERE session_id = ? "
                    f"AND event_type IN ({', '.join('?' for _ in _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES)}) "
                    "ORDER BY sequence DESC LIMIT ?",
                    (
                        session_id,
                        *(
                            str(event_type)
                            for event_type in _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES
                        ),
                        _TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT,
                    ),
                ).fetchall()
                terminal_publication_block = _terminal_publication_delete_block_reason(
                    session=session,
                    checkpoint=checkpoint,
                    evidence_events=[_event_from_row(row) for row in terminal_evidence_rows],
                )
                if terminal_publication_block is not None:
                    raise ValueError(
                        f"Cannot delete a session while {terminal_publication_block}: {session_id}"
                    )
                active_operation_id = _active_unexpired_session_operation_id(
                    checkpoint,
                    now=deletion_now,
                )
                if active_operation_id is not None:
                    raise ValueError(
                        "Cannot delete a session while durable operation "
                        f"{active_operation_id} is active: {session_id}"
                    )
                active_stage = self._connection.execute(
                    "SELECT 1 FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (
                        session_id,
                        MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                    ),
                ).fetchone()
                if active_stage is not None:
                    raise ValueError(
                        "Cannot delete a session while a model-completion stage is active: "
                        f"{session_id}"
                    )
                pending_budget_settlement = self._connection.execute(
                    """
                    SELECT identity.reservation_id
                    FROM cayu_budget_reservation_identities AS identity
                    LEFT JOIN cayu_events AS event
                      ON event.session_id = identity.publication_session_id
                     AND event.event_type IN (
                         'budget.reconciled',
                         'budget.reservation_released'
                     )
                     AND json_extract(event.payload_json, '$.reservation_id')
                         = identity.reservation_id
                    LEFT JOIN cayu_persisted_event_side_effects AS delivery
                      ON delivery.session_id = event.session_id
                     AND delivery.event_id = event.event_id
                    WHERE identity.publication_session_id = ?
                    GROUP BY identity.reservation_id
                    HAVING COUNT(event.event_id) <> 1
                        OR COUNT(
                            CASE WHEN delivery.status = 'delivered' THEN 1 END
                        ) <> 1
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
                if pending_budget_settlement is not None:
                    raise ValueError(
                        "Cannot delete a session while a budget settlement audit "
                        f"event is pending: {session_id}"
                    )
                # ON DELETE CASCADE removes events/labels/checkpoint/transcript;
                # the self-FK is ON DELETE SET NULL so children keep loading.
                self._connection.execute(
                    "DELETE FROM cayu_sessions WHERE id = ?",
                    (session_id,),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    async def update_labels(self, session_id: str, labels: dict[str, str]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        new_labels = copy_label_map(labels, "labels", allow_reserved=False)
        updated_at = datetime.now(UTC)
        expected_run_epoch = _current_session_run_epoch(session_id)
        async with self._lock:
            with self._connection:
                epoch_clause = "" if expected_run_epoch is None else " AND run_epoch = ?"
                params: list[object] = [
                    sqlite_support.format_datetime(updated_at),
                    session_id,
                ]
                if expected_run_epoch is not None:
                    params.append(expected_run_epoch)
                cursor = self._connection.execute(
                    f"UPDATE cayu_sessions SET updated_at = ? WHERE id = ?{epoch_clause}",
                    params,
                )
                if cursor.rowcount != 1:
                    if expected_run_epoch is not None:
                        _raise_session_write_conflict(
                            self._connection, session_id, expected_run_epoch
                        )
                    raise KeyError(f"Session not found: {session_id}")
                self._connection.execute(
                    "DELETE FROM cayu_session_labels WHERE session_id = ?",
                    (session_id,),
                )
                if new_labels:
                    self._connection.executemany(
                        """
                        INSERT INTO cayu_session_labels (session_id, key, value)
                        VALUES (?, ?, ?)
                        """,
                        [(session_id, key, value) for key, value in new_labels.items()],
                    )
            loaded = self._load_unlocked(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            return loaded

    async def update_metadata(self, session_id: str, metadata: dict[str, Any]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        user_metadata = copy_session_user_metadata(metadata)
        updated_at = datetime.now(UTC)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT run_epoch, metadata_json FROM cayu_sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch_value(session_id, row["run_epoch"])
                new_metadata = replace_session_user_metadata(
                    json.loads(row["metadata_json"]),
                    user_metadata,
                )
                self._connection.execute(
                    "UPDATE cayu_sessions SET metadata_json = ?, updated_at = ? WHERE id = ?",
                    (
                        sqlite_support.json_dumps(new_metadata),
                        sqlite_support.format_datetime(updated_at),
                        session_id,
                    ),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            loaded = self._load_unlocked(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            return loaded

    async def transition_status(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")

        updated_at = datetime.now(UTC)
        async with self._lock:
            expected_run_epoch = _current_session_run_epoch(session_id)
            placeholders = ", ".join("?" for _ in allowed_statuses)
            params: list[object] = [
                str(to_status),
                sqlite_support.format_datetime(updated_at),
                sqlite_support.format_datetime(updated_at),
                1 if to_status == SessionStatus.RUNNING else 0,
                session_id,
                *[str(status) for status in allowed_statuses],
            ]
            epoch_clause = ""
            if expected_run_epoch is not None:
                epoch_clause = " AND run_epoch = ?"
                params.append(expected_run_epoch)
            with self._connection:
                cursor = self._connection.execute(
                    f"""
                    UPDATE cayu_sessions
                    SET status = ?, updated_at = ?, last_activity_at = ?,
                        run_epoch = run_epoch + ?
                    WHERE id = ? AND status IN ({placeholders}){epoch_clause}
                    """,
                    params,
                )
            if cursor.rowcount != 1:
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                if expected_run_epoch is not None and loaded.run_epoch != expected_run_epoch:
                    raise SessionRunFenced(
                        f"Session run epoch no longer owns {session_id}: expected "
                        f"{expected_run_epoch}, current {loaded.run_epoch}."
                    )
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {loaded.status} -> {to_status}"
                )

            loaded = self._load_unlocked(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            if to_status == SessionStatus.RUNNING:
                _activate_session_run_fence(loaded)
            return loaded

    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform: CheckpointTransform,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        continued_interaction_id: str | None = None,
        defer_interaction_source: bool = False,
    ) -> Session:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        admission = _copy_transition_interaction_admission(
            session_id,
            interaction_started_event,
            interaction_source_messages,
            continued_interaction_id=continued_interaction_id,
            defer_interaction_source=defer_interaction_source,
        )
        if admission is not None and to_status is not SessionStatus.RUNNING:
            raise ValueError("Interaction admission requires a transition to running.")

        updated_at = datetime.now(UTC)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, loaded)
                if loaded.status not in allowed_statuses:
                    raise SessionStatusConflict(
                        f"Session status transition not allowed: {loaded.status} -> {to_status}"
                    )
                transformed_checkpoint = checkpoint_transform(
                    loaded,
                    self._load_checkpoint_unlocked(session_id),
                )
                if transformed_checkpoint is not None:
                    transformed_checkpoint = copy_durable_json_object(
                        transformed_checkpoint,
                        "checkpoint",
                    )

                placeholders = ", ".join("?" for _ in allowed_statuses)
                cursor = self._connection.execute(
                    f"""
                    UPDATE cayu_sessions
                    SET status = ?, updated_at = ?, last_activity_at = ?,
                        run_epoch = run_epoch + ?
                    WHERE id = ? AND status IN ({placeholders})
                    """,
                    (
                        str(to_status),
                        sqlite_support.format_datetime(updated_at),
                        sqlite_support.format_datetime(updated_at),
                        1 if to_status == SessionStatus.RUNNING else 0,
                        session_id,
                        *(str(status) for status in allowed_statuses),
                    ),
                )
                if cursor.rowcount != 1:
                    current = self._load_unlocked(session_id)
                    if current is None:
                        raise KeyError(f"Session not found: {session_id}")
                    raise SessionStatusConflict(
                        f"Session status transition not allowed: {current.status} -> {to_status}"
                    )
                if transformed_checkpoint is not None:
                    self._connection.execute(
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
                        sqlite_support.checkpoint_row_values(
                            session_id, transformed_checkpoint, updated_at
                        ),
                    )
                if admission is not None:
                    started_event, interaction_id, source_messages, defer_source = admission
                    existing_deferred = self._connection.execute(
                        "SELECT interaction_id FROM cayu_deferred_interaction_inputs "
                        "WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                    if existing_deferred is not None and (
                        not defer_source or existing_deferred["interaction_id"] != interaction_id
                    ):
                        raise RuntimeError("Session already has deferred interaction input.")
                    if started_event is not None:
                        lookup_key, projection, projection_bytes = (
                            pending_action_event_storage_values(started_event)
                        )
                        self._connection.execute(
                            """
                            INSERT INTO cayu_events (
                                session_id, event_id, interaction_id, event_type,
                                timestamp, agent_name, environment_name, workflow_name,
                                tool_name, payload_json, pending_action_lookup_key,
                                pending_action_projection_json,
                                pending_action_projection_bytes
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                session_id,
                                started_event.id,
                                interaction_id,
                                str(started_event.type),
                                sqlite_support.format_datetime(started_event.timestamp),
                                started_event.agent_name,
                                started_event.environment_name,
                                started_event.workflow_name,
                                started_event.tool_name,
                                sqlite_support.json_dumps(started_event.payload),
                                lookup_key,
                                projection,
                                projection_bytes,
                            ),
                        )
                        _enqueue_persisted_event_side_effects(
                            self._connection,
                            session_id,
                            [started_event],
                        )
                    if defer_source:
                        self._connection.execute(
                            "INSERT INTO cayu_deferred_interaction_inputs "
                            "(session_id, interaction_id, source_messages_json) "
                            "VALUES (?, ?, ?) "
                            "ON CONFLICT(session_id) DO UPDATE SET "
                            "interaction_id = excluded.interaction_id, "
                            "source_messages_json = excluded.source_messages_json",
                            (
                                session_id,
                                interaction_id,
                                sqlite_support.json_dumps(
                                    [message.model_dump(mode="json") for message in source_messages]
                                ),
                            ),
                        )
                    else:
                        self._connection.executemany(
                            "INSERT INTO cayu_transcript_messages "
                            "(session_id, role, interaction_id, message_json) "
                            "VALUES (?, ?, ?, ?)",
                            [
                                (
                                    session_id,
                                    str(message.role),
                                    interaction_id,
                                    sqlite_support.json_dumps(message.model_dump(mode="json")),
                                )
                                for message in source_messages
                            ],
                        )
                self._connection.commit()
                transitioned = loaded.model_copy(
                    update={
                        "status": to_status,
                        "updated_at": updated_at,
                        "last_activity_at": updated_at,
                        "run_epoch": loaded.run_epoch + (to_status == SessionStatus.RUNNING),
                    }
                )
            except sqlite3.IntegrityError as exc:
                self._connection.rollback()
                existing_event_id = (
                    None
                    if admission is None or admission[0] is None
                    else _first_existing_event_id(
                        self._connection,
                        session_id,
                        [admission[0].id],
                    )
                )
                if existing_event_id is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing_event_id}"
                    ) from exc
                raise
            except Exception:
                self._connection.rollback()
                raise

            if to_status == SessionStatus.RUNNING:
                _activate_session_run_fence(transitioned)
            return transitioned

    async def fence_stalled_run(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_before: datetime,
    ) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(statuses, "statuses")
        if inactive_before.tzinfo is None or inactive_before.utcoffset() is None:
            raise ValueError("inactive_before must be timezone-aware.")
        now = datetime.now(UTC)
        placeholders = ", ".join("?" for _ in allowed_statuses)
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    f"""
                    UPDATE cayu_sessions
                    SET run_epoch = run_epoch + 1, last_activity_at = ?
                    WHERE id = ? AND status IN ({placeholders}) AND last_activity_at <= ?
                    """,
                    (
                        sqlite_support.format_datetime(now),
                        session_id,
                        *(str(status) for status in allowed_statuses),
                        sqlite_support.format_datetime(inactive_before),
                    ),
                )
            if cursor.rowcount != 1:
                if not self._session_exists_unlocked(session_id):
                    raise KeyError(f"Session not found: {session_id}")
                return None
            loaded = self._load_unlocked(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            _activate_session_run_fence(loaded)
            return loaded

    async def fence_run_and_transform_checkpoint(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        checkpoint_transform: CheckpointTransform,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(statuses, "statuses")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        updated_at = datetime.now(UTC)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                if loaded.status not in allowed_statuses:
                    raise SessionStatusConflict(f"Session status cannot be fenced: {loaded.status}")
                transformed = checkpoint_transform(
                    loaded,
                    self._load_checkpoint_unlocked(session_id),
                )
                if transformed is None:
                    raise ValueError("Fenced checkpoint transform must return a checkpoint.")
                transformed = copy_durable_json_object(transformed, "checkpoint")
                self._connection.execute(
                    "UPDATE cayu_sessions SET run_epoch = run_epoch + 1, "
                    "last_activity_at = ? WHERE id = ?",
                    (sqlite_support.format_datetime(updated_at), session_id),
                )
                self._connection.execute(
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
                    sqlite_support.checkpoint_row_values(
                        session_id,
                        transformed,
                        updated_at,
                    ),
                )
                self._connection.commit()
                fenced = loaded.model_copy(
                    update={
                        "run_epoch": loaded.run_epoch + 1,
                        "last_activity_at": updated_at,
                    }
                )
            except BaseException:
                self._connection.rollback()
                raise
            _activate_session_run_fence(fenced)
            return fenced

    async def transition_status_if_no_queued_messages(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        updated_at = datetime.now(UTC)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, loaded)
                if loaded.status not in allowed_statuses:
                    raise SessionStatusConflict(
                        f"Session status transition not allowed: {loaded.status} -> {to_status}"
                    )
                pending = self._connection.execute(
                    "SELECT 1 FROM cayu_session_message_queue "
                    "WHERE session_id = ? AND status = 'queued' LIMIT 1",
                    (session_id,),
                ).fetchone()
                if pending is not None:
                    raise SessionQueuedMessagesPending(
                        f"Session has durable queued messages: {session_id}"
                    )
                cursor = self._connection.execute(
                    "UPDATE cayu_sessions SET status = ?, updated_at = ?, "
                    "last_activity_at = ?, run_epoch = run_epoch + ? WHERE id = ?",
                    (
                        str(to_status),
                        sqlite_support.format_datetime(updated_at),
                        sqlite_support.format_datetime(updated_at),
                        1 if to_status == SessionStatus.RUNNING else 0,
                        session_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"Session not found: {session_id}")
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
            transitioned = loaded.model_copy(
                update={
                    "status": to_status,
                    "updated_at": updated_at,
                    "last_activity_at": updated_at,
                    "run_epoch": loaded.run_epoch + (to_status == SessionStatus.RUNNING),
                }
            )
            if to_status == SessionStatus.RUNNING:
                _activate_session_run_fence(transitioned)
            return transitioned

    async def publish_interaction_transition(
        self,
        session_id: str,
        *,
        event: Event,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        only_if_no_queued_messages: bool = False,
    ) -> InteractionTransitionResult:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        (
            session_id,
            copied_event,
            allowed_statuses,
            target_status,
            conditional,
        ) = _prepare_interaction_transition(
            session_id,
            event=event,
            from_statuses=from_statuses,
            to_status=to_status,
            only_if_no_queued_messages=only_if_no_queued_messages,
        )
        receipt_storage_key = _interaction_transition_storage_key(copied_event.id)

        def statement(connection: sqlite3.Connection) -> InteractionTransitionResult:
            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = _load_session(connection, session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, loaded)
                receipt_row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, receipt_storage_key),
                ).fetchone()
                existing_row = connection.execute(
                    f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                    "WHERE session_id = ? AND event_id = ?",
                    (session_id, copied_event.id),
                ).fetchone()
                if receipt_row is not None:
                    receipt = _reconstruct_interaction_transition_receipt(
                        copy_durable_json_object(
                            json.loads(receipt_row["record_json"]),
                            "interaction transition receipt",
                        ),
                        event=copied_event,
                        from_statuses=allowed_statuses,
                        to_status=target_status,
                        only_if_no_queued_messages=conditional,
                    )
                    if existing_row is not None and _event_from_row(existing_row) != receipt.event:
                        raise RuntimeError(
                            "Interaction transition receipt conflicts with retained event history."
                        )
                    connection.commit()
                    return InteractionTransitionResult(
                        session=receipt.session,
                        event=receipt.event,
                        status_changed=receipt.status_changed,
                        replayed=True,
                    )
                if existing_row is not None:
                    raise RuntimeError(
                        "Interaction transition event exists without its immutable receipt."
                    )
                if loaded.status not in allowed_statuses:
                    raise SessionStatusConflict(
                        f"Session status transition not allowed: {loaded.status} -> {target_status}"
                    )
                queued = False
                if conditional:
                    queued = (
                        connection.execute(
                            "SELECT 1 FROM cayu_session_message_queue "
                            "WHERE session_id = ? AND status = 'queued' LIMIT 1",
                            (session_id,),
                        ).fetchone()
                        is not None
                    )
                updated_at = datetime.now(UTC)
                formatted_updated_at = sqlite_support.format_datetime(updated_at)
                if queued:
                    _touch_session_activity(connection, session_id, updated_at)
                else:
                    connection.execute(
                        "UPDATE cayu_sessions SET status = ?, updated_at = ?, "
                        "last_activity_at = ? WHERE id = ?",
                        (
                            str(target_status),
                            formatted_updated_at,
                            formatted_updated_at,
                            session_id,
                        ),
                    )
                lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                    copied_event
                )
                connection.execute(
                    """
                    INSERT INTO cayu_events (
                        session_id, event_id, interaction_id, event_type, timestamp,
                        agent_name, environment_name, workflow_name, tool_name,
                        payload_json, pending_action_lookup_key,
                        pending_action_projection_json, pending_action_projection_bytes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        copied_event.id,
                        copied_event.interaction_id,
                        str(copied_event.type),
                        sqlite_support.format_datetime(copied_event.timestamp),
                        copied_event.agent_name,
                        copied_event.environment_name,
                        copied_event.workflow_name,
                        copied_event.tool_name,
                        sqlite_support.json_dumps(copied_event.payload),
                        lookup_key,
                        projection,
                        projection_bytes,
                    ),
                )
                _enqueue_persisted_event_side_effects(
                    connection,
                    session_id,
                    [copied_event],
                )
                transitioned = _load_session(connection, session_id)
                if transitioned is None:
                    raise KeyError(f"Session not found: {session_id}")
                receipt_record = _interaction_transition_receipt_record(
                    session=transitioned,
                    event=copied_event,
                    from_statuses=allowed_statuses,
                    to_status=target_status,
                    only_if_no_queued_messages=conditional,
                    status_changed=not queued,
                )
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        session_id,
                        receipt_storage_key,
                        sqlite_support.json_dumps(receipt_record),
                        formatted_updated_at,
                    ),
                )
                connection.commit()
                return InteractionTransitionResult(
                    session=transitioned,
                    event=copied_event,
                    status_changed=not queued,
                )
            except Exception:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def release_run_fence(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        expected_run_epoch = _current_session_run_epoch(session_id)
        if expected_run_epoch is None:
            _deactivate_session_interaction(session_id)
            return
        try:
            async with self._lock:
                with self._connection:
                    self._connection.execute(
                        "UPDATE cayu_sessions SET run_epoch = run_epoch + 1 "
                        "WHERE id = ? AND run_epoch = ?",
                        (session_id, expected_run_epoch),
                    )
        finally:
            _deactivate_session_run_fence(session_id)
            _deactivate_session_interaction(session_id)

    async def append_event(self, session_id: str, event: Event) -> None:
        await self.append_events(session_id, [event])

    async def claim_budget_reservation_identity(
        self,
        *,
        reservation_id: str,
        publication_session_id: str,
        publication_id: str,
    ) -> None:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        publication_session_id = require_clean_nonblank(
            publication_session_id,
            "publication_session_id",
        )
        publication_id = require_clean_nonblank(publication_id, "publication_id")
        expected_run_epoch = _current_session_run_epoch(publication_session_id)

        def statement(connection: sqlite3.Connection) -> None:
            with connection:
                # The conditional no-op acquires SQLite's writer lock while it
                # validates the session epoch, so a takeover cannot interleave
                # between this check and the registry claim below.
                if expected_run_epoch is None:
                    cursor = connection.execute(
                        "UPDATE cayu_sessions SET run_epoch = run_epoch WHERE id = ?",
                        (publication_session_id,),
                    )
                else:
                    cursor = connection.execute(
                        "UPDATE cayu_sessions SET run_epoch = run_epoch "
                        "WHERE id = ? AND run_epoch = ?",
                        (publication_session_id, expected_run_epoch),
                    )
                if cursor.rowcount != 1:
                    if expected_run_epoch is not None:
                        _raise_session_write_conflict(
                            connection,
                            publication_session_id,
                            expected_run_epoch,
                        )
                    raise KeyError(f"Session not found: {publication_session_id}")
                _claim_budget_reservation_identity(
                    connection,
                    reservation_id=reservation_id,
                    publication_session_id=publication_session_id,
                    publication_id=publication_id,
                )

        await self._run_write(statement)

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        session_id, copied_events = _copy_session_event_batch(session_id, events)

        def statement(connection: sqlite3.Connection) -> None:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            if not copied_events:
                return

            try:
                with connection:
                    _touch_session_activity(connection, session_id, datetime.now(UTC))
                    _publish_budget_reservation_identities(connection, copied_events)
                    rows = []
                    for event in copied_events:
                        lookup_key, projection, projection_bytes = (
                            pending_action_event_storage_values(event)
                        )
                        rows.append(
                            (
                                session_id,
                                event.id,
                                event.interaction_id,
                                str(event.type),
                                sqlite_support.format_datetime(event.timestamp),
                                event.agent_name,
                                event.environment_name,
                                event.workflow_name,
                                event.tool_name,
                                sqlite_support.json_dumps(event.payload),
                                lookup_key,
                                projection,
                                projection_bytes,
                            )
                        )
                    connection.executemany(
                        """
                        INSERT INTO cayu_events (
                            session_id,
                            event_id,
                            interaction_id,
                            event_type,
                            timestamp,
                            agent_name,
                            environment_name,
                            workflow_name,
                            tool_name,
                            payload_json,
                            pending_action_lookup_key,
                            pending_action_projection_json,
                            pending_action_projection_bytes
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
                    _enqueue_persisted_event_side_effects(
                        connection,
                        session_id,
                        copied_events,
                    )
            except sqlite3.IntegrityError as exc:
                existing_event_id = _first_existing_event_id(
                    connection,
                    session_id,
                    [event.id for event in copied_events],
                )
                if existing_event_id is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing_event_id}"
                    ) from exc
                if "idx_cayu_events_budget_reservation_identity" in str(exc):
                    raise BudgetReservationIdentityConflict(
                        "Budget ledger reused a reservation identity."
                    ) from exc
                raise

        await self._run_write(statement)

    async def append_workflow_step_started(
        self,
        session_id: str,
        event: Event,
        *,
        workflow_name: str,
        attempt_id: str,
    ) -> bool:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        session_id, copied_event, workflow_name, attempt_id = _copy_workflow_step_reservation(
            session_id,
            event,
            workflow_name=workflow_name,
            attempt_id=attempt_id,
        )

        def statement(connection: sqlite3.Connection) -> bool:
            try:
                connection.execute("BEGIN IMMEDIATE")
                if not _session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                row = connection.execute(
                    """
                    SELECT json_extract(payload_json, '$.attempt_id')
                    FROM cayu_events
                    WHERE session_id = ?
                      AND workflow_name = ?
                      AND event_type = ?
                    ORDER BY sequence DESC
                    LIMIT 1
                    """,
                    (session_id, workflow_name, WORKFLOW_ATTEMPT_EVENT_TYPE),
                ).fetchone()
                if row is None or row[0] != attempt_id:
                    connection.rollback()
                    return False
                if _first_existing_event_id(connection, session_id, [copied_event.id]) is not None:
                    connection.rollback()
                    return False

                _touch_session_activity(connection, session_id, datetime.now(UTC))
                lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                    copied_event
                )
                connection.execute(
                    """
                    INSERT INTO cayu_events (
                        session_id,
                        event_id,
                        interaction_id,
                        event_type,
                        timestamp,
                        agent_name,
                        environment_name,
                        workflow_name,
                        tool_name,
                        payload_json,
                        pending_action_lookup_key,
                        pending_action_projection_json,
                        pending_action_projection_bytes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        copied_event.id,
                        copied_event.interaction_id,
                        str(copied_event.type),
                        sqlite_support.format_datetime(copied_event.timestamp),
                        copied_event.agent_name,
                        copied_event.environment_name,
                        copied_event.workflow_name,
                        copied_event.tool_name,
                        sqlite_support.json_dumps(copied_event.payload),
                        lookup_key,
                        projection,
                        projection_bytes,
                    ),
                )
                _enqueue_persisted_event_side_effects(
                    connection,
                    session_id,
                    [copied_event],
                )
                connection.commit()
                return True
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                existing_event_id = _first_existing_event_id(
                    connection,
                    session_id,
                    [copied_event.id],
                )
                if existing_event_id is not None:
                    return False
                raise exc
            except Exception:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def load_mcp_manifest_baselines(
        self,
        history_keys: tuple[str, ...],
    ) -> McpManifestBaselineLoadResult:
        keys = _validate_mcp_manifest_history_keys(history_keys)

        def query(connection: sqlite3.Connection) -> McpManifestBaselineLoadResult:
            result: dict[str, McpManifestBaseline] = {}
            for key in keys:
                row = connection.execute(
                    "SELECT generation, baseline_json FROM cayu_mcp_manifest_baselines "
                    "WHERE history_key = ?",
                    (key,),
                ).fetchone()
                if row is not None:
                    result[key] = _stored_mcp_manifest_baseline_json(
                        key,
                        row["generation"],
                        row["baseline_json"],
                    )
            return McpManifestBaselineLoadResult(baselines=result)

        return await self._run_read(query)

    async def compare_and_publish_mcp_manifest_checks(
        self,
        session_id: str,
        *,
        expected_generations: dict[str, int | None],
        baseline_updates: dict[str, McpManifestBaseline],
        events: list[Event],
    ) -> McpManifestPublicationResult:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        session_id, expected, updates, copied_events = _copy_mcp_manifest_publication(
            session_id,
            expected_generations=expected_generations,
            baseline_updates=baseline_updates,
            events=events,
        )

        def statement(connection: sqlite3.Connection) -> McpManifestPublicationResult:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = _load_session(connection, session_id)
                if session is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, session)
                current: dict[str, McpManifestBaseline] = {}
                for key in expected:
                    row = connection.execute(
                        "SELECT generation, baseline_json "
                        "FROM cayu_mcp_manifest_baselines "
                        "WHERE history_key = ?",
                        (key,),
                    ).fetchone()
                    if row is not None:
                        current[key] = _stored_mcp_manifest_baseline_json(
                            key,
                            row["generation"],
                            row["baseline_json"],
                        )
                if any(
                    expected_generation
                    != (None if (baseline := current.get(key)) is None else baseline.generation)
                    for key, expected_generation in expected.items()
                ):
                    connection.rollback()
                    return McpManifestPublicationResult(
                        published=False,
                        baselines=current,
                    )

                _validate_mcp_manifest_publication_state(
                    expected_generations=expected,
                    current_baselines=current,
                    baseline_updates=updates,
                    events=copied_events,
                )
                _touch_session_activity(connection, session_id, datetime.now(UTC))
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
                            sqlite_support.format_datetime(event.timestamp),
                            event.agent_name,
                            event.environment_name,
                            event.workflow_name,
                            event.tool_name,
                            sqlite_support.json_dumps(event.payload),
                            lookup_key,
                            projection,
                            projection_bytes,
                        )
                    )
                connection.executemany(
                    """
                    INSERT INTO cayu_events (
                        session_id, event_id, interaction_id, event_type, timestamp, agent_name,
                        environment_name, workflow_name, tool_name, payload_json,
                        pending_action_lookup_key, pending_action_projection_json,
                        pending_action_projection_bytes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    event_rows,
                )
                _enqueue_persisted_event_side_effects(
                    connection,
                    session_id,
                    copied_events,
                )
                updated_at = sqlite_support.format_datetime(datetime.now(UTC))
                for key, baseline in updates.items():
                    connection.execute(
                        """
                        INSERT INTO cayu_mcp_manifest_baselines (
                            history_key, generation, baseline_json, updated_at
                        )
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(history_key) DO UPDATE SET
                            generation = excluded.generation,
                            baseline_json = excluded.baseline_json,
                            updated_at = excluded.updated_at
                        """,
                        (
                            key,
                            baseline.generation,
                            sqlite_support.json_dumps(baseline.model_dump(mode="json")),
                            updated_at,
                        ),
                    )
                    current[key] = baseline.model_copy(deep=True)
                connection.commit()
                return McpManifestPublicationResult(
                    published=True,
                    baselines=current,
                )
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                existing_event_id = _first_existing_event_id(
                    connection,
                    session_id,
                    [event.id for event in copied_events],
                )
                if existing_event_id is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing_event_id}"
                    ) from exc
                raise
            except Exception:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectClaim | None:
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        if event_id is not None:
            event_id = require_clean_nonblank(event_id, "event_id")
        if (session_id is None) != (event_id is None):
            raise ValueError("session_id and event_id must be supplied together.")
        if type(lease_seconds) not in {int, float} or lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0.")

        def statement(connection: sqlite3.Connection) -> PersistedEventSideEffectClaim | None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                lease_expires_at = now + timedelta(seconds=float(lease_seconds))
                formatted_now = sqlite_support.format_datetime(now)
                filters = [
                    "(status = 'pending' "
                    "OR (status = 'failed' AND "
                    "(next_attempt_at IS NULL OR next_attempt_at <= ?)) "
                    "OR (status = 'leased' AND lease_expires_at <= ?))"
                ]
                params: list[object] = [formatted_now, formatted_now]
                if session_id is not None and event_id is not None:
                    filters.extend(["session_id = ?", "event_id = ?"])
                    params.extend([session_id, event_id])
                delivery_row = connection.execute(
                    "SELECT * FROM cayu_persisted_event_side_effects WHERE "
                    + " AND ".join(filters)
                    + " ORDER BY event_sequence ASC LIMIT 1",
                    params,
                ).fetchone()
                if delivery_row is None:
                    connection.commit()
                    return None
                claim_id = str(uuid4())
                attempt = int(delivery_row["attempts"]) + 1
                connection.execute(
                    "UPDATE cayu_persisted_event_side_effects "
                    "SET status = 'leased', attempts = ?, claim_id = ?, "
                    "lease_expires_at = ?, next_attempt_at = NULL, "
                    "last_error = NULL, updated_at = ? "
                    "WHERE session_id = ? AND event_id = ?",
                    (
                        attempt,
                        claim_id,
                        sqlite_support.format_datetime(lease_expires_at),
                        sqlite_support.format_datetime(now),
                        delivery_row["session_id"],
                        delivery_row["event_id"],
                    ),
                )
                event_row = connection.execute(
                    f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                    "WHERE session_id = ? AND event_id = ?",
                    (delivery_row["session_id"], delivery_row["event_id"]),
                ).fetchone()
                if event_row is None:
                    raise RuntimeError("Persisted side-effect delivery lost its source event.")
                connection.commit()
                return PersistedEventSideEffectClaim(
                    session_id=delivery_row["session_id"],
                    event_id=delivery_row["event_id"],
                    event_sequence=delivery_row["event_sequence"],
                    event=_event_from_row(event_row),
                    attempt=attempt,
                    claim_id=claim_id,
                    lease_expires_at=lease_expires_at,
                )
            except Exception:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def get_persisted_event_side_effect_delivery(
        self,
        *,
        session_id: str,
        event_id: str,
    ) -> PersistedEventSideEffectDelivery | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        event_id = require_clean_nonblank(event_id, "event_id")

        def query(connection: sqlite3.Connection) -> PersistedEventSideEffectDelivery | None:
            row = connection.execute(
                "SELECT * FROM cayu_persisted_event_side_effects "
                "WHERE session_id = ? AND event_id = ?",
                (session_id, event_id),
            ).fetchone()
            return None if row is None else _persisted_event_side_effect_delivery_from_row(row)

        return await self._run_read(query)

    async def mark_persisted_event_side_effect_delivered(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        return await self._finish_persisted_event_side_effect_claim(
            claim,
            status=PersistedEventSideEffectStatus.DELIVERED,
            error=None,
            retry_delay_seconds=None,
        )

    async def mark_persisted_event_side_effect_failed(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        error: str,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        error = validate_persisted_event_side_effect_error(error)
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be an integer greater than or equal to 1.")
        if (
            type(retry_delay_seconds) not in {int, float}
            or not math.isfinite(retry_delay_seconds)
            or retry_delay_seconds < 0
        ):
            raise ValueError("retry_delay_seconds must be a finite non-negative number.")
        dead_lettered = claim.attempt >= max_attempts
        return await self._finish_persisted_event_side_effect_claim(
            claim,
            status=(
                PersistedEventSideEffectStatus.DEAD_LETTERED
                if dead_lettered
                else PersistedEventSideEffectStatus.FAILED
            ),
            error=error,
            retry_delay_seconds=(None if dead_lettered else float(retry_delay_seconds)),
        )

    async def _finish_persisted_event_side_effect_claim(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        status: PersistedEventSideEffectStatus,
        error: str | None,
        retry_delay_seconds: float | None,
    ) -> PersistedEventSideEffectDelivery:
        def statement(connection: sqlite3.Connection) -> PersistedEventSideEffectDelivery:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = datetime.now(UTC)
                next_attempt_at = (
                    None
                    if retry_delay_seconds is None
                    else now + timedelta(seconds=retry_delay_seconds)
                )
                cursor = connection.execute(
                    "UPDATE cayu_persisted_event_side_effects "
                    "SET status = ?, claim_id = NULL, lease_expires_at = NULL, "
                    "next_attempt_at = ?, last_error = ?, updated_at = ? "
                    "WHERE session_id = ? AND event_id = ? AND status = 'leased' "
                    "AND claim_id = ? AND attempts = ?",
                    (
                        str(status),
                        (
                            None
                            if next_attempt_at is None
                            else sqlite_support.format_datetime(next_attempt_at)
                        ),
                        error,
                        sqlite_support.format_datetime(now),
                        claim.session_id,
                        claim.event_id,
                        claim.claim_id,
                        claim.attempt,
                    ),
                )
                if cursor.rowcount != 1:
                    existing = connection.execute(
                        "SELECT 1 FROM cayu_persisted_event_side_effects "
                        "WHERE session_id = ? AND event_id = ?",
                        (claim.session_id, claim.event_id),
                    ).fetchone()
                    if existing is None:
                        raise ValueError("Persisted event side-effect delivery was not found.")
                    raise PersistedEventSideEffectClaimLost(
                        "Persisted event side-effect claim is no longer active."
                    )
                row = connection.execute(
                    "SELECT * FROM cayu_persisted_event_side_effects "
                    "WHERE session_id = ? AND event_id = ?",
                    (claim.session_id, claim.event_id),
                ).fetchone()
                if row is None:
                    raise RuntimeError("Persisted event side-effect delivery disappeared.")
                delivery = _persisted_event_side_effect_delivery_from_row(row)
                connection.commit()
                return delivery
            except Exception:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def list_persisted_event_side_effect_deliveries(
        self,
        *,
        statuses: set[PersistedEventSideEffectStatus] | None = None,
        claimable_only: bool = False,
        after_sequence: int | None = None,
        limit: int = 100,
    ) -> list[PersistedEventSideEffectDelivery]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000.")
        if type(claimable_only) is not bool:
            raise TypeError("claimable_only must be a bool.")
        if after_sequence is not None and (type(after_sequence) is not int or after_sequence < 0):
            raise ValueError("after_sequence must be a non-negative integer.")
        selected_statuses = (
            None
            if statuses is None
            else sorted(str(PersistedEventSideEffectStatus(status)) for status in statuses)
        )

        def query(connection: sqlite3.Connection) -> list[PersistedEventSideEffectDelivery]:
            clauses: list[str] = []
            params: list[object] = []
            if after_sequence is not None:
                clauses.append("event_sequence > ?")
                params.append(after_sequence)
            if selected_statuses is not None:
                if not selected_statuses:
                    return []
                placeholders = ", ".join("?" for _ in selected_statuses)
                clauses.append(f"status IN ({placeholders})")
                params.extend(selected_statuses)
            if claimable_only:
                clauses.append(
                    "(status = 'pending' "
                    "OR (status = 'failed' AND "
                    "(next_attempt_at IS NULL OR next_attempt_at <= ?)) "
                    "OR (status = 'leased' AND lease_expires_at <= ?))"
                )
                formatted_now = sqlite_support.format_datetime(datetime.now(UTC))
                params.extend([formatted_now, formatted_now])
            where = "" if not clauses else "WHERE " + " AND ".join(clauses)
            params.append(limit)
            rows = connection.execute(
                "SELECT * FROM cayu_persisted_event_side_effects "
                f"{where} ORDER BY event_sequence ASC LIMIT ?",
                params,
            ).fetchall()
            return [_persisted_event_side_effect_delivery_from_row(row) for row in rows]

        return await self._run_read(query)

    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
    ) -> EnqueueSessionMessageResult:
        request = copy_enqueue_session_message_request(request)

        def statement(connection: sqlite3.Connection) -> EnqueueSessionMessageResult:
            from cayu.runtime.pending_actions import pending_action_event_storage_values

            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(request.session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {request.session_id}")
                existing_row = connection.execute(
                    "SELECT * FROM cayu_session_message_queue "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (request.session_id, request.idempotency_key),
                ).fetchone()
                if existing_row is not None:
                    existing = _queued_session_message_from_row(existing_row)
                    _validate_equivalent_queued_session_message(existing, request)
                    event_row = connection.execute(
                        f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                        "WHERE session_id = ? AND event_id = ?",
                        (request.session_id, existing.accepted_event_id),
                    ).fetchone()
                    if event_row is None:
                        raise RuntimeError(
                            "Queued session message is missing its durable acceptance event."
                        )
                    connection.commit()
                    return EnqueueSessionMessageResult(
                        message=existing,
                        event=_event_from_row(event_row),
                        replayed=True,
                    )
                if loaded.status not in {SessionStatus.PENDING, SessionStatus.RUNNING}:
                    raise SessionStatusConflict(
                        "Session messages may be enqueued only while a session is pending or running."
                    )
                transcript_cursor = _transcript_cursor(connection, request.session_id)
                accepted_at = datetime.now(UTC)
                queue_id = str(uuid4())
                accepted_event_id = str(uuid4())
                cursor = connection.execute(
                    """
                    INSERT INTO cayu_session_message_queue (
                        queue_id, session_id, idempotency_key, content,
                        delivery_mode, status, requested_by_json,
                        accepted_run_epoch, accepted_transcript_cursor,
                        accepted_event_id, accepted_at
                    )
                    VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                    """,
                    (
                        queue_id,
                        request.session_id,
                        request.idempotency_key,
                        request.content,
                        str(request.delivery_mode),
                        (
                            None
                            if request.requested_by is None
                            else sqlite_support.json_dumps(
                                resolution_actor_payload(request.requested_by)
                            )
                        ),
                        loaded.run_epoch,
                        transcript_cursor,
                        accepted_event_id,
                        sqlite_support.format_datetime(accepted_at),
                    ),
                )
                ordering_key = cursor.lastrowid
                if type(ordering_key) is not int:
                    raise RuntimeError("SQLite queue insert did not return an ordering key.")
                accepted_event = Event(
                    id=accepted_event_id,
                    type=EventType.SESSION_MESSAGE_QUEUED,
                    session_id=request.session_id,
                    agent_name=loaded.agent_name,
                    environment_name=loaded.environment_name,
                    timestamp=accepted_at,
                    payload=_queued_session_message_event_payload(
                        queue_id=queue_id,
                        delivery_mode=request.delivery_mode,
                        ordering_key=ordering_key,
                        actor=request.requested_by,
                        run_epoch=loaded.run_epoch,
                        transcript_cursor=transcript_cursor,
                    ),
                )
                lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                    accepted_event
                )
                connection.execute(
                    """
                    INSERT INTO cayu_events (
                        session_id, event_id, interaction_id, event_type, timestamp, agent_name,
                        environment_name, workflow_name, tool_name, payload_json,
                        pending_action_lookup_key, pending_action_projection_json,
                        pending_action_projection_bytes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request.session_id,
                        accepted_event.id,
                        accepted_event.interaction_id,
                        str(accepted_event.type),
                        sqlite_support.format_datetime(accepted_event.timestamp),
                        accepted_event.agent_name,
                        accepted_event.environment_name,
                        accepted_event.workflow_name,
                        accepted_event.tool_name,
                        sqlite_support.json_dumps(accepted_event.payload),
                        lookup_key,
                        projection,
                        projection_bytes,
                    ),
                )
                _enqueue_persisted_event_side_effects(
                    connection,
                    request.session_id,
                    [accepted_event],
                )
                _touch_session_activity(connection, request.session_id, accepted_at)
                connection.commit()
                stored_row = connection.execute(
                    "SELECT * FROM cayu_session_message_queue WHERE queue_id = ?",
                    (queue_id,),
                ).fetchone()
                if stored_row is None:
                    raise RuntimeError("Queued session message disappeared after acceptance.")
                return EnqueueSessionMessageResult(
                    message=_queued_session_message_from_row(stored_row),
                    event=accepted_event,
                )
            except Exception:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def deliver_queued_session_messages(
        self,
        session_id: str,
        *,
        include_on_idle: bool,
        delivery_id: str | None = None,
        eligible_through: int | None = None,
        limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
        interaction_id: str | None = None,
        interaction_started_event: Event | None = None,
    ) -> SessionMessageDeliveryBatch:
        session_id = require_clean_nonblank(session_id, "session_id")
        delivery_id = (
            str(uuid4())
            if delivery_id is None
            else require_clean_nonblank(delivery_id, "delivery_id")
        )
        if interaction_id is not None:
            interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
        interaction_started_event = _copy_queued_interaction_started_event(
            session_id,
            interaction_id,
            interaction_started_event,
        )
        if type(include_on_idle) is not bool:
            raise TypeError("include_on_idle must be a bool.")
        eligible_through = _validate_message_delivery_eligible_through(eligible_through)
        if type(limit) is not int or not 1 <= limit <= SESSION_MESSAGE_DELIVERY_BATCH_LIMIT:
            raise ValueError(f"limit must be between 1 and {SESSION_MESSAGE_DELIVERY_BATCH_LIMIT}.")

        def statement(connection: sqlite3.Connection) -> SessionMessageDeliveryBatch:
            from cayu.runtime.pending_actions import pending_action_event_storage_values

            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, loaded)
                delivery_row = connection.execute(
                    "SELECT * FROM cayu_session_message_deliveries WHERE delivery_id = ?",
                    (delivery_id,),
                ).fetchone()
                if delivery_row is not None:
                    stored_started_event = (
                        None
                        if delivery_row["interaction_started_event_json"] is None
                        else Event.model_validate_json(
                            delivery_row["interaction_started_event_json"]
                        )
                    )
                    if (
                        delivery_row["session_id"] != session_id
                        or bool(delivery_row["include_on_idle"]) != include_on_idle
                        or delivery_row["requested_eligible_through"] != eligible_through
                        or delivery_row["batch_limit"] != limit
                        or delivery_row["interaction_id"] != interaction_id
                        or stored_started_event != interaction_started_event
                    ):
                        raise ValueError(
                            "delivery_id was already used for a different queue delivery."
                        )
                    queue_ids = json.loads(delivery_row["queue_ids_json"])
                    replayed_messages: list[SessionQueuedMessage] = []
                    replayed_events = [
                        Event.model_validate(event)
                        for event in json.loads(delivery_row["events_json"])
                    ]
                    for queue_id in queue_ids:
                        queued_row = connection.execute(
                            "SELECT * FROM cayu_session_message_queue WHERE queue_id = ?",
                            (queue_id,),
                        ).fetchone()
                        if queued_row is None:
                            raise RuntimeError("Queue delivery replay lost a delivered message.")
                        replayed_messages.append(_queued_session_message_from_row(queued_row))
                    connection.commit()
                    return SessionMessageDeliveryBatch(
                        messages=tuple(replayed_messages),
                        events=tuple(replayed_events),
                        delivery_id=delivery_id,
                        interaction_id=interaction_id,
                        eligible_through=delivery_row["eligible_through"],
                        has_more=bool(delivery_row["has_more"]),
                        replayed=True,
                    )
                if loaded.status != SessionStatus.RUNNING:
                    raise SessionStatusConflict(
                        "Queued session messages may be delivered only while running."
                    )
                boundary = eligible_through
                if boundary is None:
                    # ``ordering_key`` is a global AUTOINCREMENT primary key.
                    # Reading its global maximum is an end-of-index lookup and
                    # still fences every message this session could currently
                    # contain; BEGIN IMMEDIATE prevents a same-session enqueue
                    # from crossing the boundary during this transaction.
                    boundary_row = connection.execute(
                        "SELECT COALESCE(MAX(ordering_key), 0) AS boundary "
                        "FROM cayu_session_message_queue"
                    ).fetchone()
                    boundary = boundary_row["boundary"]
                rows = connection.execute(
                    "SELECT * FROM cayu_session_message_queue "
                    "WHERE session_id = ? AND status = 'queued' "
                    "AND delivery_mode = 'next_turn' AND ordering_key <= ? "
                    "ORDER BY ordering_key ASC LIMIT ?",
                    (session_id, boundary, limit),
                ).fetchall()
                if not rows and include_on_idle:
                    rows = connection.execute(
                        "SELECT * FROM cayu_session_message_queue "
                        "WHERE session_id = ? AND status = 'queued' "
                        "AND delivery_mode = 'on_idle' AND ordering_key <= ? "
                        "ORDER BY ordering_key ASC LIMIT ?",
                        (session_id, boundary, limit),
                    ).fetchall()
                if not rows:
                    connection.execute(
                        """
                        INSERT INTO cayu_session_message_deliveries (
                            delivery_id, session_id, interaction_id, include_on_idle,
                            requested_eligible_through, eligible_through, batch_limit,
                            has_more, interaction_started_event_json, queue_ids_json,
                            events_json, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, '[]', '[]', ?)
                        """,
                        (
                            delivery_id,
                            session_id,
                            interaction_id,
                            include_on_idle,
                            eligible_through,
                            boundary,
                            limit,
                            (
                                None
                                if interaction_started_event is None
                                else sqlite_support.json_dumps(
                                    interaction_started_event.model_dump(mode="json")
                                )
                            ),
                            sqlite_support.format_datetime(datetime.now(UTC)),
                        ),
                    )
                    connection.commit()
                    return SessionMessageDeliveryBatch(
                        delivery_id=delivery_id,
                        interaction_id=interaction_id,
                        eligible_through=boundary,
                        has_more=False,
                    )
                transcript_cursor = _transcript_cursor(connection, session_id)
                delivered_at = datetime.now(UTC)
                updated_messages: list[SessionQueuedMessage] = []
                delivery_events: list[Event] = []
                transcript_messages: list[Message] = []
                for offset, row in enumerate(rows, start=1):
                    queued_message = _queued_session_message_from_row(row)
                    delivered_cursor = transcript_cursor + offset
                    delivery_event = Event(
                        type=EventType.SESSION_MESSAGE_DELIVERED,
                        session_id=session_id,
                        interaction_id=interaction_id,
                        agent_name=loaded.agent_name,
                        environment_name=loaded.environment_name,
                        timestamp=delivered_at,
                        payload={
                            **_queued_session_message_event_payload(
                                queue_id=queued_message.queue_id,
                                delivery_mode=queued_message.delivery_mode,
                                ordering_key=queued_message.ordering_key,
                                actor=queued_message.requested_by,
                                run_epoch=loaded.run_epoch,
                                transcript_cursor=delivered_cursor,
                            ),
                            "accepted_run_epoch": queued_message.accepted_run_epoch,
                            "accepted_transcript_cursor": (
                                queued_message.accepted_transcript_cursor
                            ),
                        },
                    )
                    updated = queued_message.model_copy(
                        update={
                            "status": SessionMessageQueueStatus.DELIVERED,
                            "delivered_run_epoch": loaded.run_epoch,
                            "delivered_transcript_cursor": delivered_cursor,
                            "delivered_event_id": delivery_event.id,
                            "delivered_at": delivered_at,
                        },
                        deep=True,
                    )
                    updated_messages.append(updated)
                    delivery_events.append(delivery_event)
                    transcript_messages.append(
                        Message.text(MessageRole.USER, queued_message.content)
                    )
                connection.executemany(
                    "INSERT INTO cayu_transcript_messages "
                    "(session_id, role, interaction_id, message_json) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (
                            session_id,
                            str(message.role),
                            interaction_id,
                            sqlite_support.json_dumps(message.model_dump(mode="json")),
                        )
                        for message in transcript_messages
                    ],
                )
                for updated in updated_messages:
                    connection.execute(
                        "UPDATE cayu_session_message_queue SET status = 'delivered', "
                        "delivered_run_epoch = ?, delivered_transcript_cursor = ?, "
                        "delivered_event_id = ?, delivered_at = ? "
                        "WHERE queue_id = ? AND status = 'queued'",
                        (
                            updated.delivered_run_epoch,
                            updated.delivered_transcript_cursor,
                            updated.delivered_event_id,
                            sqlite_support.format_datetime(delivered_at),
                            updated.queue_id,
                        ),
                    )
                persisted_events = [
                    *([interaction_started_event] if interaction_started_event is not None else []),
                    *delivery_events,
                ]
                event_rows = []
                for event in persisted_events:
                    lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                        event
                    )
                    event_rows.append(
                        (
                            session_id,
                            event.id,
                            event.interaction_id,
                            str(event.type),
                            sqlite_support.format_datetime(event.timestamp),
                            event.agent_name,
                            event.environment_name,
                            event.workflow_name,
                            event.tool_name,
                            sqlite_support.json_dumps(event.payload),
                            lookup_key,
                            projection,
                            projection_bytes,
                        )
                    )
                connection.executemany(
                    "INSERT INTO cayu_events (session_id, event_id, interaction_id, "
                    "event_type, timestamp, "
                    "agent_name, environment_name, workflow_name, tool_name, payload_json, "
                    "pending_action_lookup_key, pending_action_projection_json, "
                    "pending_action_projection_bytes) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    event_rows,
                )
                _enqueue_persisted_event_side_effects(
                    connection,
                    session_id,
                    persisted_events,
                )
                _touch_session_activity(connection, session_id, delivered_at)
                remaining_mode_sql = (
                    "delivery_mode IN ('next_turn', 'on_idle')"
                    if include_on_idle
                    else "delivery_mode = 'next_turn'"
                )
                remaining = connection.execute(
                    "SELECT 1 FROM cayu_session_message_queue WHERE session_id = ? "
                    "AND status = 'queued' AND ordering_key <= ? "
                    f"AND {remaining_mode_sql} LIMIT 1",
                    (session_id, boundary),
                ).fetchone()
                connection.execute(
                    """
                    INSERT INTO cayu_session_message_deliveries (
                        delivery_id, session_id, interaction_id, include_on_idle,
                        requested_eligible_through, eligible_through, batch_limit,
                        has_more, interaction_started_event_json, queue_ids_json,
                        events_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        delivery_id,
                        session_id,
                        interaction_id,
                        include_on_idle,
                        eligible_through,
                        boundary,
                        limit,
                        remaining is not None,
                        (
                            None
                            if interaction_started_event is None
                            else sqlite_support.json_dumps(
                                interaction_started_event.model_dump(mode="json")
                            )
                        ),
                        sqlite_support.json_dumps(
                            [message.queue_id for message in updated_messages]
                        ),
                        sqlite_support.json_dumps(
                            [event.model_dump(mode="json") for event in persisted_events]
                        ),
                        sqlite_support.format_datetime(delivered_at),
                    ),
                )
                connection.commit()
                return SessionMessageDeliveryBatch(
                    messages=tuple(updated_messages),
                    events=tuple(persisted_events),
                    delivery_id=delivery_id,
                    interaction_id=interaction_id,
                    eligible_through=boundary,
                    has_more=remaining is not None,
                )
            except Exception:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def publish_checkpoint_and_events(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        return await self._publish_checkpoint_and_events(
            session_id,
            checkpoint_transform=checkpoint_transform,
            operation_idempotency_key=None,
            operation_transform=None,
            operation_commit_guard=None,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )

    async def load_session_operation(
        self,
        session_id: str,
        idempotency_key: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        idempotency_key = _reject_reserved_runtime_publication_key(
            idempotency_key,
            "idempotency_key",
        )

        checkpoint_root_key = (
            "__cayu_no_checkpoint_root_guard__"
            if checkpoint_root_guard is None
            else checkpoint_root_guard.key
        )
        checkpoint_root_path = f"$.{checkpoint_root_key}"

        def query(connection: sqlite3.Connection) -> dict[str, Any] | None:
            row = connection.execute(
                f"""
                SELECT
                    cayu_session_operations.record_json,
                    json_type(
                        cayu_checkpoints.state_json,
                        '{checkpoint_root_path}'
                    ) AS checkpoint_root_field_type,
                    CASE
                        WHEN json_type(
                            cayu_checkpoints.state_json,
                            '{checkpoint_root_path}'
                        ) = 'integer'
                        THEN substr(
                            CAST(json_extract(
                                cayu_checkpoints.state_json,
                                '{checkpoint_root_path}'
                            ) AS TEXT),
                            1,
                            {CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS + 1}
                        )
                    END AS checkpoint_root_field_scalar
                FROM cayu_sessions
                LEFT JOIN cayu_session_operations
                    ON cayu_session_operations.session_id = cayu_sessions.id
                    AND cayu_session_operations.idempotency_key = ?
                LEFT JOIN cayu_checkpoints
                    ON cayu_checkpoints.session_id = cayu_sessions.id
                WHERE cayu_sessions.id = ?
                """,
                (idempotency_key, session_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"Session not found: {session_id}")
            scalar_text = row["checkpoint_root_field_scalar"]
            if checkpoint_root_guard is not None:
                checkpoint_root_guard.validate(
                    session_id,
                    checkpoint_root_field_projection_from_storage(
                        json_type=row["checkpoint_root_field_type"],
                        scalar_text=scalar_text,
                    ),
                )
            record_json = row["record_json"]
            return None if record_json is None else json.loads(record_json)

        return await self._run_read(query)

    async def _load_runtime_publication_receipt_record(
        self,
        session_id: str,
        storage_key: str,
        publication_id: str,
    ) -> dict[str, Any] | None:
        def query(connection: sqlite3.Connection) -> dict[str, Any] | None:
            connection.execute("BEGIN")
            try:
                if not _session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, storage_key),
                ).fetchone()
                if row is None:
                    return None
                record = _decode_runtime_publication_record(row["record_json"])
                receipt = _reconstruct_runtime_publication_receipt(
                    record,
                    storage_key=storage_key,
                    session_id=session_id,
                    publication_id=publication_id,
                )
                self._validate_runtime_publication_material(connection, receipt)
                return record
            finally:
                connection.rollback()

        return await self._run_read(query)

    def _validate_runtime_publication_material(
        self,
        connection: sqlite3.Connection,
        receipt: RuntimePublicationReceipt,
    ) -> None:
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

            referenced_event_ids = _runtime_publication_referenced_event_ids(
                receipt.referenced_events
            )
            requested_event_ids = tuple(
                dict.fromkeys((*receipt.appended_event_ids, *referenced_event_ids))
            )
            events_by_id: dict[str, Event] = {}
            if requested_event_ids:
                placeholders = ", ".join("?" for _ in requested_event_ids)
                rows = connection.execute(
                    f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                    f"WHERE session_id = ? AND event_id IN ({placeholders})",
                    (receipt.session_id, *requested_event_ids),
                ).fetchall()
                events_by_id = {row["event_id"]: _event_from_row(row) for row in rows}
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

    async def _load_model_completion_stage_records(
        self,
        session_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        def query(
            connection: sqlite3.Connection,
        ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            rows = connection.execute(
                "SELECT idempotency_key, record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                (session_id, preparation_storage_key, terminal_storage_key),
            ).fetchall()
            records = {
                row["idempotency_key"]: _decode_model_completion_stage_record(row["record_json"])
                for row in rows
            }
            return records.get(preparation_storage_key), records.get(terminal_storage_key)

        return await self._run_read(query)

    async def _load_active_model_completion_stage_records(
        self,
        session_id: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        def query(
            connection: sqlite3.Connection,
        ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
            try:
                connection.execute("BEGIN")
                if not _session_exists(connection, session_id):
                    raise KeyError(f"Session not found: {session_id}")
                row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                ).fetchone()
                if row is None:
                    connection.rollback()
                    return None, None, None
                active_record = _decode_model_completion_stage_record(row["record_json"])
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
                    record_row["idempotency_key"]: _decode_model_completion_stage_record(
                        record_row["record_json"]
                    )
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

        return await self._run_read(query)

    async def _prepare_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStage,
    ) -> ModelCompletionStageResult:
        session_id = prepared.session_id

        def statement(connection: sqlite3.Connection) -> ModelCompletionStageResult:
            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
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
                    row["idempotency_key"]: _decode_model_completion_stage_record(
                        row["record_json"]
                    )
                    for row in rows
                }
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
                    active_record = _decode_model_completion_stage_record(active_row["record_json"])
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
                        row["idempotency_key"]: _decode_model_completion_stage_record(
                            row["record_json"]
                        )
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
                current_cursor = _transcript_cursor(connection, session_id)
                if current_cursor != prepared.expected_transcript_cursor:
                    raise ValueError(
                        "Session source transcript cursor is stale: expected "
                        f"{prepared.expected_transcript_cursor}, current {current_cursor}."
                    )

                prepared_at = _next_runtime_publication_timestamp(loaded)
                record = _model_completion_stage_preparation_record(
                    prepared,
                    source_session=loaded,
                    prepared_at=prepared_at,
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
                formatted_at = sqlite_support.format_datetime(prepared_at)
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        session_id,
                        prepared.preparation_storage_key,
                        sqlite_support.json_dumps(record),
                        formatted_at,
                    ),
                )
                active_record = _active_model_completion_stage_record(
                    stage,
                    activated_at=prepared_at,
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
                        sqlite_support.json_dumps(active_record),
                        formatted_at,
                    ),
                )
                cursor = connection.execute(
                    "UPDATE cayu_sessions SET updated_at = ?, last_activity_at = ? WHERE id = ?",
                    (formatted_at, formatted_at, session_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"Session not found: {session_id}")
                connection.commit()
                return ModelCompletionStageResult(
                    stage=stage,
                    replayed=False,
                    dispatch_authorized=True,
                )
            except BaseException:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def _complete_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStageTerminal,
    ) -> ModelCompletionStageResult:
        session_id = prepared.session_id

        def statement(connection: sqlite3.Connection) -> ModelCompletionStageResult:
            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                rows = connection.execute(
                    "SELECT idempotency_key, record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                    (
                        session_id,
                        prepared.preparation_storage_key,
                        prepared.terminal_storage_key,
                    ),
                ).fetchall()
                records = {
                    row["idempotency_key"]: _decode_model_completion_stage_record(
                        row["record_json"]
                    )
                    for row in rows
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
                if stage.state == "completed":
                    _validate_model_completion_stage_terminal_replay(stage, prepared)
                    connection.rollback()
                    return ModelCompletionStageResult(
                        stage=stage,
                        replayed=True,
                        dispatch_authorized=False,
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
                    None
                    if active_row is None
                    else _decode_model_completion_stage_record(active_row["record_json"])
                )
                advances_last_activity = _model_completion_terminal_advances_last_activity(
                    active_record,
                    stage=stage,
                    current_run_epoch=loaded.run_epoch,
                )
                formatted_at = sqlite_support.format_datetime(completed_at)
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        session_id,
                        prepared.terminal_storage_key,
                        sqlite_support.json_dumps(terminal_record),
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

        return await self._run_write(statement)

    async def _abandon_model_completion_stage_atomic(
        self,
        prepared: _PreparedModelCompletionStageAbandonment,
    ) -> ModelCompletionStageAbandonmentResult:
        session_id = prepared.session_id

        def statement(
            connection: sqlite3.Connection,
        ) -> ModelCompletionStageAbandonmentResult:
            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
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
                    row["idempotency_key"]: _decode_model_completion_stage_record(
                        row["record_json"]
                    )
                    for row in rows
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
                    None
                    if active_row is None
                    else _decode_model_completion_stage_record(active_row["record_json"])
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
                winner_storage_key = _model_completion_stage_winner_storage_key(
                    stage.logical_step_id
                )
                publication_storage_key = _runtime_publication_storage_key(stage.logical_step_id)
                publication_rows = connection.execute(
                    "SELECT idempotency_key FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key IN (?, ?)",
                    (session_id, winner_storage_key, publication_storage_key),
                ).fetchall()
                publication_keys = {row["idempotency_key"] for row in publication_rows}
                _validate_model_completion_stage_for_abandonment(
                    session=loaded,
                    stage=stage,
                    active=active,
                    prepared=prepared,
                    abandonment_record=records.get(prepared.abandonment_storage_key),
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
                formatted_at = sqlite_support.format_datetime(abandoned_at)
                connection.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                    "record_json = excluded.record_json, updated_at = excluded.updated_at",
                    (
                        session_id,
                        prepared.abandonment_storage_key,
                        sqlite_support.json_dumps(abandonment_record),
                        formatted_at,
                    ),
                )
                deleted_preparation = connection.execute(
                    "DELETE FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (session_id, prepared.preparation_storage_key),
                )
                if deleted_preparation.rowcount != 1:
                    raise SessionModelCompletionStageConflict(
                        "The model-completion preparation changed during abandonment."
                    )
                deleted_active = connection.execute(
                    "DELETE FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
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

        return await self._run_write(statement)

    async def _promote_model_completion_stage_atomic(
        self,
        *,
        session_id: str,
        stage_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
        expected_run_epoch: int,
    ) -> RuntimePublicationResult:
        stage = await self.load_model_completion_stage(session_id, stage_id)
        if stage is None:
            raise KeyError(f"Model-completion stage not found: {stage_id}")
        prepared = _prepare_model_completion_stage_promotion(
            stage,
            expected_run_epoch=expected_run_epoch,
        )
        assert stage.completion_digest is not None
        return await self._publish_runtime_publication_atomic(
            prepared,
            _model_completion_stage=_ModelCompletionStagePromotionContext(
                stage_id=stage_id,
                preparation_storage_key=preparation_storage_key,
                terminal_storage_key=terminal_storage_key,
                active_storage_key=MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                winner_storage_key=_model_completion_stage_winner_storage_key(
                    stage.logical_step_id
                ),
                completion_digest=stage.completion_digest,
            ),
        )

    async def _publish_runtime_publication_atomic(
        self,
        prepared: _PreparedRuntimePublication,
        *,
        _model_completion_stage: _ModelCompletionStagePromotionContext | None = None,
    ) -> RuntimePublicationResult:
        from cayu.runtime.pending_actions import (
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
                loaded = self._load_unlocked(session_id)
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
                        row["idempotency_key"]: _decode_model_completion_stage_record(
                            row["record_json"]
                        )
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
                    receipt_record = _decode_runtime_publication_record(receipt_row["record_json"])
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
                                    _decode_model_completion_stage_record(row["record_json"])
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
                        self._validate_runtime_publication_material(connection, receipt)
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
                    self._validate_runtime_publication_material(connection, receipt)
                    connection.rollback()
                    return RuntimePublicationResult(
                        session=loaded.model_copy(deep=True),
                        receipt=receipt,
                        replayed=True,
                    )

                if locked_stage is not None:
                    assert _model_completion_stage is not None
                    if winner_record is not None:
                        raise SessionModelCompletionStageConflict(
                            "A model-completion winner exists without its runtime publication "
                            "receipt."
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
                transcript_start_cursor = _transcript_cursor(connection, session_id)
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
                        f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                        f"WHERE session_id = ? AND event_id IN ({placeholders})",
                        (session_id, *referenced_event_ids),
                    ).fetchall()
                    durable_referenced_events = {
                        row["event_id"]: _event_from_row(row) for row in rows
                    }
                _validate_runtime_publication_event_references(
                    request.referenced_events,
                    durable_referenced_events,
                    interaction_id=request.interaction_id,
                )
                stored_checkpoint = self._load_checkpoint_unlocked(session_id)
                current_checkpoint = (
                    stored_checkpoint
                    if checkpoint_decode is None
                    else checkpoint_decode(loaded, stored_checkpoint)
                )
                _validate_tool_round_checkpoint_mutation(
                    request,
                    current_checkpoint,
                )
                durable_tool_events: list[Event] = []
                tool_round_identity = _tool_round_publication_identity(request)
                if tool_round_identity is not None:
                    execution_identity, tool_call_ids = tool_round_identity
                    lookup_keys = tuple(
                        pending_action_lookup_key(tool_call_id) for tool_call_id in tool_call_ids
                    )
                    lifecycle_event_types = tuple(
                        sorted(str(event_type) for event_type in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES)
                    )
                    lookup_placeholders = ", ".join("?" for _ in lookup_keys)
                    event_type_placeholders = ", ".join("?" for _ in lifecycle_event_types)
                    rows = connection.execute(
                        f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                        "INDEXED BY idx_cayu_events_pending_action_lookup "
                        f"WHERE session_id = ? AND pending_action_lookup_key IN "
                        f"({lookup_placeholders}) AND event_type IN "
                        f"({event_type_placeholders}) AND "
                        f"({_PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
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
                            *lookup_keys,
                            *lifecycle_event_types,
                            execution_identity.tool_round_id,
                            execution_identity.model_step_id,
                            execution_identity.model_attempt_id,
                            RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS + 1,
                        ),
                    ).fetchall()
                    if len(rows) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
                        raise ValueError(
                            "Tool-round lifecycle evidence exceeds the publication limit."
                        )
                    durable_tool_events = [_event_from_row(row) for row in rows]
                _validate_tool_round_publication(
                    request,
                    durable_referenced_events,
                    durable_tool_events=durable_tool_events,
                )
                existing_event_id = _first_existing_event_id(
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
                        sqlite_support.json_dumps(message_payload),
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
                            sqlite_support.format_datetime(event.timestamp),
                            event.agent_name,
                            event.environment_name,
                            event.workflow_name,
                            event.tool_name,
                            sqlite_support.json_dumps(event_payload["payload"]),
                            lookup_key,
                            projection,
                            projection_bytes,
                        )
                    )

                published_at = _next_runtime_publication_timestamp(loaded)
                checkpoint_values = (
                    None
                    if stored_target_checkpoint is None or not request.mutation.operations
                    else sqlite_support.checkpoint_row_values(
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
                receipt_json = sqlite_support.json_dumps(
                    _runtime_publication_receipt_record(receipt)
                )
                formatted_published_at = sqlite_support.format_datetime(published_at)

                if transcript_rows:
                    connection.executemany(
                        """
                        INSERT INTO cayu_transcript_messages (
                            session_id, role, interaction_id, message_json
                        )
                        VALUES (?, ?, ?, ?)
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
                    _enqueue_persisted_event_side_effects(
                        connection,
                        session_id,
                        request.events,
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
                            sqlite_support.json_dumps(winner),
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
                existing_event_id = _first_existing_event_id(
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

        return await self._run_write(statement)

    async def publish_session_operation(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        return await self._publish_checkpoint_and_events(
            session_id,
            checkpoint_transform=None,
            operation_idempotency_key=_reject_reserved_runtime_publication_key(
                idempotency_key,
                "idempotency_key",
            ),
            operation_transform=operation_transform,
            operation_commit_guard=None,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )

    async def publish_session_operation_guarded(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        commit_guard: Callable[[], None],
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        return await self._publish_checkpoint_and_events(
            session_id,
            checkpoint_transform=None,
            operation_idempotency_key=require_clean_nonblank(
                idempotency_key,
                "idempotency_key",
            ),
            operation_transform=operation_transform,
            operation_commit_guard=commit_guard,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )

    async def _publish_checkpoint_and_events(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform | None,
        operation_idempotency_key: str | None,
        operation_transform: SessionOperationTransform | None,
        operation_commit_guard: Callable[[], None] | None,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None,
        expected_run_epoch: int | None,
        expected_transcript_cursor: int | None,
    ) -> Session:
        from cayu.runtime.pending_actions import pending_action_event_storage_values

        session_id, copied_events = _copy_session_event_batch(session_id, events)
        if (checkpoint_transform is None) == (operation_transform is None):
            raise TypeError("Exactly one checkpoint publication transform is required.")
        if operation_transform is not None and operation_idempotency_key is None:
            raise TypeError("operation_idempotency_key is required.")
        allowed_statuses = (
            None
            if expected_statuses is None
            else _validate_status_set(expected_statuses, "expected_statuses")
        )
        updated_at = datetime.now(UTC)

        def statement(connection: sqlite3.Connection) -> Session:
            try:
                connection.execute("BEGIN IMMEDIATE")
                loaded = self._load_unlocked(session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, loaded)
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
                current_cursor = _transcript_cursor(connection, session_id)
                if (
                    expected_transcript_cursor is not None
                    and current_cursor != expected_transcript_cursor
                ):
                    raise ValueError(
                        "Session source transcript cursor is stale: expected "
                        f"{expected_transcript_cursor}, current {current_cursor}."
                    )
                current_checkpoint = self._load_checkpoint_unlocked(session_id)
                operation_records: dict[str, dict[str, Any]] = {}
                if operation_transform is not None:
                    operation_row = connection.execute(
                        "SELECT record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, operation_idempotency_key),
                    ).fetchone()
                    current_operation = (
                        None if operation_row is None else json.loads(operation_row["record_json"])
                    )
                    publication = operation_transform(
                        loaded,
                        current_checkpoint,
                        current_operation,
                    )
                    if type(publication) is not SessionOperationPublication:
                        raise TypeError(
                            "Session operation transform must return a SessionOperationPublication."
                        )
                    transformed = copy_durable_json_object(
                        publication.checkpoint,
                        "checkpoint",
                    )
                    operation_records = copy_durable_json_object(
                        publication.operation_records,
                        "operation_records",
                    )
                    _validate_session_operation_record_keys(operation_records)
                else:
                    assert checkpoint_transform is not None
                    transformed = checkpoint_transform(loaded, current_checkpoint)
                    if transformed is None:
                        raise ValueError("Checkpoint transform must return a checkpoint.")
                    transformed = copy_durable_json_object(transformed, "checkpoint")
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
                            sqlite_support.format_datetime(event.timestamp),
                            event.agent_name,
                            event.environment_name,
                            event.workflow_name,
                            event.tool_name,
                            sqlite_support.json_dumps(event.payload),
                            lookup_key,
                            projection,
                            projection_bytes,
                        )
                    )
                _publish_budget_reservation_identities(connection, copied_events)
                _touch_session_activity(connection, session_id, updated_at)
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
                    sqlite_support.checkpoint_row_values(session_id, transformed, updated_at),
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
                                sqlite_support.json_dumps(record),
                                sqlite_support.format_datetime(updated_at),
                            )
                            for key, record in operation_records.items()
                        ],
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
                    _enqueue_persisted_event_side_effects(
                        connection,
                        session_id,
                        copied_events,
                    )
                if operation_commit_guard is not None:
                    operation_commit_guard()
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                existing_event_id = _first_existing_event_id(
                    connection,
                    session_id,
                    [event.id for event in copied_events],
                )
                if existing_event_id is not None:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {existing_event_id}"
                    ) from exc
                raise
            except Exception:
                connection.rollback()
                raise
            return loaded.model_copy(
                update={"updated_at": updated_at, "last_activity_at": updated_at}
            )

        return await self._run_write(statement)

    async def load_events(self, session_id: str) -> list[Event]:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> list[Event]:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            rows = connection.execute(
                f"""
                SELECT {", ".join(_EVENT_COLUMN_NAMES)}
                FROM cayu_events
                WHERE session_id = ?
                ORDER BY sequence ASC
                """,
                (session_id,),
            ).fetchall()
            return [_event_from_row(row) for row in rows]

        return await self._run_read(query)

    async def load_tool_round_lifecycle_events(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
    ) -> list[Event]:
        from cayu.runtime.pending_actions import pending_action_lookup_key

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = _validate_tool_round_call_ids(tool_call_ids, "tool_call_ids")
        lookup_keys = tuple(pending_action_lookup_key(call_id) for call_id in copied_ids)
        lifecycle_event_types = tuple(
            sorted(str(event_type) for event_type in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES)
        )

        def query(connection: sqlite3.Connection) -> list[Event]:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            lookup_placeholders = ", ".join("?" for _ in lookup_keys)
            event_type_placeholders = ", ".join("?" for _ in lifecycle_event_types)
            rows = connection.execute(
                f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                "INDEXED BY idx_cayu_events_pending_action_lookup "
                f"WHERE session_id = ? AND pending_action_lookup_key IN "
                f"({lookup_placeholders}) AND event_type IN "
                f"({event_type_placeholders}) AND "
                f"({_PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
                "ORDER BY sequence ASC LIMIT ?",
                (
                    session_id,
                    *lookup_keys,
                    *lifecycle_event_types,
                    RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS + 1,
                ),
            ).fetchall()
            if len(rows) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
                raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            return [_event_from_row(row) for row in rows]

        return await self._run_read(query)

    async def load_tool_round_lifecycle_events_for_round(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
        *,
        tool_round_identity: ToolRoundIdentity,
    ) -> list[Event]:
        from cayu.runtime.pending_actions import pending_action_lookup_key

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = _validate_tool_round_call_ids(tool_call_ids, "tool_call_ids")
        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        lookup_keys = tuple(pending_action_lookup_key(call_id) for call_id in copied_ids)
        lifecycle_event_types = tuple(
            sorted(str(event_type) for event_type in _TOOL_ROUND_LIFECYCLE_EVENT_TYPES)
        )

        def query(connection: sqlite3.Connection) -> list[Event]:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            lookup_placeholders = ", ".join("?" for _ in lookup_keys)
            event_type_placeholders = ", ".join("?" for _ in lifecycle_event_types)
            rows = connection.execute(
                f"SELECT {', '.join(_EVENT_COLUMN_NAMES)} FROM cayu_events "
                "INDEXED BY idx_cayu_events_pending_action_lookup "
                f"WHERE session_id = ? AND pending_action_lookup_key IN "
                f"({lookup_placeholders}) AND event_type IN "
                f"({event_type_placeholders}) AND "
                f"({_PENDING_ACTION_LOOKUP_INDEX_PREDICATE_SQL}) "
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
                    *lookup_keys,
                    *lifecycle_event_types,
                    tool_round_identity.tool_round_id,
                    tool_round_identity.model_step_id,
                    tool_round_identity.model_attempt_id,
                    RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS + 1,
                ),
            ).fetchall()
            if len(rows) > RUNTIME_PUBLICATION_MAX_EVENT_BINDINGS:
                raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            return [_event_from_row(row) for row in rows]

        return await self._run_read(query)

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        query = copy_event_query(query)
        if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
            return await self._query_events_by_session_id_batches(query)

        plan = session_store_sql.build_event_query_sql(query, dialect=_SQL_DIALECT)
        params = [*plan.params, query.limit]

        def run_query(connection: sqlite3.Connection) -> list[EventRecord]:
            event_columns = ", ".join(f"cayu_events.{name}" for name in _EVENT_COLUMN_NAMES)
            rows = connection.execute(
                f"""
                SELECT cayu_events.sequence, {event_columns}
                FROM cayu_events
                JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                {plan.where_sql}
                ORDER BY cayu_events.sequence {plan.order_direction}
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [
                EventRecord(sequence=row["sequence"], event=_event_from_row(row)) for row in rows
            ]

        return await self._run_read(run_query)

    async def query_events_bounded(
        self,
        query: EventQuery,
        *,
        max_bytes: int,
    ) -> list[EventRecord]:
        query = copy_event_query(query)
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer.")
        if len(query.session_ids) > _EVENT_QUERY_SESSION_IDS_BATCH_SIZE:
            raise ValueError("Byte-bounded event queries require one bounded SQL batch.")
        plan = session_store_sql.build_event_query_sql(query, dialect=_SQL_DIALECT)
        params = [*plan.params, query.limit]

        def run_query(connection: sqlite3.Connection) -> list[EventRecord]:
            event_columns = ", ".join(f"cayu_events.{name}" for name in _EVENT_COLUMN_NAMES)
            serialized_bytes = " + ".join(
                [
                    "256",
                    *(
                        f"COALESCE(length(CAST(cayu_events.{name} AS BLOB)), 0)"
                        for name in _EVENT_COLUMN_NAMES
                    ),
                ]
            )
            connection.execute("BEGIN")
            try:
                size_row = connection.execute(
                    f"""
                    WITH bounded_candidates AS (
                        SELECT {serialized_bytes} AS serialized_bytes
                        FROM cayu_events
                        JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                        {plan.where_sql}
                        ORDER BY cayu_events.sequence {plan.order_direction}
                        LIMIT ?
                    )
                    SELECT COALESCE(SUM(serialized_bytes), 0)
                    FROM bounded_candidates
                    """,
                    params,
                ).fetchone()
                if size_row is None or int(size_row[0]) > max_bytes:
                    raise EventQueryResultTooLarge(max_bytes)
                rows = connection.execute(
                    f"""
                    SELECT cayu_events.sequence, {event_columns}
                    FROM cayu_events
                    JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id
                    {plan.where_sql}
                    ORDER BY cayu_events.sequence {plan.order_direction}
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
                return [
                    EventRecord(sequence=row["sequence"], event=_event_from_row(row))
                    for row in rows
                ]
            finally:
                connection.rollback()

        return await self._run_read(run_query)

    async def load_terminal_session_evidence(
        self,
        session_id: str,
        *,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        return await self._load_terminal_session_evidence(
            session_id,
            limits=limits,
            observed_interrupted_events=None,
            expected_interrupted_parent_session_id=None,
            require_interrupted_proof=False,
        )

    async def load_runner_owned_interrupted_evidence(
        self,
        session_id: str,
        *,
        observed_events: tuple[RunnerObservedEventIdentity, ...] | None = None,
        expected_parent_session_id: str | None = None,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        return await self._load_terminal_session_evidence(
            session_id,
            limits=limits,
            observed_interrupted_events=observed_events,
            expected_interrupted_parent_session_id=expected_parent_session_id,
            require_interrupted_proof=True,
        )

    async def _load_terminal_session_evidence(
        self,
        session_id: str,
        *,
        limits: TerminalSessionEvidenceLimits | None,
        observed_interrupted_events: tuple[RunnerObservedEventIdentity, ...] | None,
        expected_interrupted_parent_session_id: str | None,
        require_interrupted_proof: bool,
    ) -> TerminalSessionEvidence:
        session_id = require_clean_nonblank(session_id, "session_id")
        resolved_limits = _copy_terminal_session_evidence_limits(limits)
        observed, expected_parent_session_id = _copy_runner_owned_interruption_proof(
            session_id,
            observed_events=observed_interrupted_events,
            expected_parent_session_id=expected_interrupted_parent_session_id,
            limits=resolved_limits,
            required=require_interrupted_proof,
        )
        allow_interrupted = observed is not None or expected_parent_session_id is not None
        evidence_event_types = tuple(
            str(event_type) for event_type in _TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES
        )
        evidence_type_placeholders = ", ".join("?" for _ in evidence_event_types)
        event_columns = ", ".join(_EVENT_COLUMN_NAMES)
        event_stored_bytes = " + ".join(
            [
                "length(CAST(sequence AS TEXT))",
                *(f"COALESCE(length(CAST({column} AS BLOB)), 0)" for column in _EVENT_COLUMN_NAMES),
            ]
        )
        session_stored_bytes = " + ".join(
            f"COALESCE(length(CAST(session.{column} AS BLOB)), 0)"
            for column in (
                "id",
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
                "length(CAST(message_json AS BLOB))",
            )
        )

        def run_query(connection: sqlite3.Connection) -> TerminalSessionEvidence:
            limits = resolved_limits
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

                if observed is not None:
                    identity_preflight = connection.execute(
                        """
                        WITH bounded_identities AS (
                            SELECT length(CAST(event_type AS BLOB))
                                       + length(CAST(sequence AS TEXT)) AS stored_bytes
                            FROM cayu_events
                            WHERE session_id = ?
                            ORDER BY sequence ASC
                            LIMIT ?
                        )
                        SELECT COUNT(*) AS record_count,
                               COALESCE(MAX(stored_bytes), 0) AS largest_record_bytes,
                               COALESCE(SUM(stored_bytes), 0) AS total_bytes
                        FROM bounded_identities
                        """,
                        (session_id, limits.max_events + 1),
                    ).fetchone()
                    if identity_preflight is None:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                        )
                    identity_count = int(identity_preflight["record_count"])
                    if identity_count > limits.max_events:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
                            limit=limits.max_events,
                            observed=identity_count,
                        )
                    if identity_count != len(observed):
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                        )
                    identity_largest_bytes = int(identity_preflight["largest_record_bytes"])
                    if identity_largest_bytes > limits.max_record_bytes:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                            limit=limits.max_record_bytes,
                            observed=identity_largest_bytes,
                        )
                    identity_total_bytes = int(identity_preflight["total_bytes"])
                    if identity_total_bytes > limits.max_total_bytes:
                        raise TerminalSessionEvidenceError(
                            TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
                            limit=limits.max_total_bytes,
                            observed=identity_total_bytes,
                        )
                    identity_rows = connection.execute(
                        """
                        SELECT sequence, event_type
                        FROM cayu_events
                        WHERE session_id = ?
                        ORDER BY sequence ASC
                        LIMIT ?
                        """,
                        (session_id, identity_count),
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
                    pending_session_interrupt = bool(
                        checkpoint_projection["pending_session_interrupt"]
                    )
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
                    int(row["stored_bytes"]) > limits.max_record_bytes
                    for row in newest_preflight_rows
                ):
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                        limit=limits.max_record_bytes,
                    )
                if any(
                    row["operation_id_type"] not in {None, "text"}
                    or (
                        row["operation_id_type"] == "text"
                        and not bool(row["operation_id_nonblank"])
                    )
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

                session = _load_session(connection, session_id)
                if session is None:
                    raise TerminalSessionEvidenceError(
                        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                    )
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
                    EventRecord(sequence=row["sequence"], event=_event_from_row(row))
                    for row in event_rows
                )
                transcript = tuple(
                    TranscriptRecord(
                        index=row["transcript_index"],
                        interaction_id=row["interaction_id"],
                        message=Message(**json.loads(row["message_json"])),
                    )
                    for row in transcript_rows
                )
                return _assemble_terminal_session_evidence(
                    session=session,
                    marker=marker,
                    terminal_record=terminal_record,
                    events=events,
                    transcript=transcript,
                    limits=limits,
                    allow_interrupted=allow_interrupted,
                )
            except TerminalSessionEvidenceError:
                raise
            except (json.JSONDecodeError, sqlite3.DatabaseError, TypeError, ValueError) as exc:
                raise TerminalSessionEvidenceError(
                    TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
                ) from exc
            finally:
                connection.rollback()

        return await self._run_read(run_query)

    async def query_latest_interaction_events(
        self,
        session_id: str,
        *,
        before_sequence: int | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        session_id = require_clean_nonblank(session_id, "session_id")
        before_sequence, limit = _validate_interaction_page(before_sequence, limit)
        cursor_clause = "" if before_sequence is None else "AND latest.latest_event_sequence < ?"
        params: list[object] = [session_id]
        if before_sequence is not None:
            params.append(before_sequence)
        params.append(limit)

        def run_query(connection: sqlite3.Connection) -> list[EventRecord]:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            event_columns = ", ".join(f"event.{name}" for name in _EVENT_COLUMN_NAMES)
            rows = connection.execute(
                f"""
                SELECT event.sequence, {event_columns}
                FROM cayu_interaction_latest_events AS latest
                JOIN cayu_events AS event
                  ON event.sequence = latest.latest_event_sequence
                WHERE latest.session_id = ? {cursor_clause}
                ORDER BY latest.latest_event_sequence DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [
                EventRecord(sequence=row["sequence"], event=_event_from_row(row)) for row in rows
            ]

        return await self._run_read(run_query)

    async def _query_events_by_session_id_batches(self, query: EventQuery) -> list[EventRecord]:
        records: list[EventRecord] = []
        for batch in _event_query_session_id_batches(query.session_ids):
            records.extend(
                await self.query_events(
                    session_store_sql.event_query_with_session_ids(
                        query,
                        session_ids=batch,
                    ),
                )
            )
        records.sort(
            key=lambda record: record.sequence,
            reverse=query.order_by.value == "sequence_desc",
        )
        return records[: query.limit]

    async def summarize_events(self, session_id: str) -> EventSummary:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> EventSummary:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")

            total_row = connection.execute(
                """
                SELECT COUNT(*) AS total_events
                FROM cayu_events
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            count_rows = connection.execute(
                """
                SELECT event_type, COUNT(*) AS count
                FROM cayu_events
                WHERE session_id = ?
                GROUP BY event_type
                ORDER BY event_type ASC
                """,
                (session_id,),
            ).fetchall()
            latest_row = connection.execute(
                f"""
                SELECT sequence, {", ".join(_EVENT_COLUMN_NAMES)}
                FROM cayu_events
                WHERE session_id = ?
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

            return EventSummary(
                session_id=session_id,
                total_events=int(total_row["total_events"]),
                counts_by_type={row["event_type"]: int(row["count"]) for row in count_rows},
                latest_event=_event_record_from_row(latest_row),
            )

        return await self._run_read(query)

    async def summarize_outcome(self, session_id: str) -> SessionOutcome:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> SessionOutcome:
            session = _load_session(connection, session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            terminal_row = connection.execute(
                f"""
                SELECT sequence, {", ".join(_EVENT_COLUMN_NAMES)}
                FROM cayu_events
                WHERE session_id = ?
                  AND event_type IN ('session.completed', 'session.failed', 'session.interrupted')
                  AND sequence > COALESCE(
                      (
                          SELECT MAX(sequence)
                          FROM cayu_events
                          WHERE session_id = ?
                            AND event_type IN ('session.started', 'session.resumed')
                      ),
                      0
                  )
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (session_id, session_id),
            ).fetchone()
            retry_row = connection.execute(
                f"""
                SELECT sequence, {", ".join(_EVENT_COLUMN_NAMES)}
                FROM cayu_events
                WHERE session_id = ?
                  AND event_type = 'model.retry'
                  AND sequence > COALESCE(
                      (
                          SELECT MAX(sequence)
                          FROM cayu_events
                          WHERE session_id = ?
                            AND event_type IN ('session.started', 'session.resumed')
                      ),
                      0
                  )
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (session_id, session_id),
            ).fetchone()

            return session_outcome(
                session,
                terminal_event=_event_record_from_row(terminal_row),
                latest_retry_event=_event_record_from_row(retry_row),
            )

        return await self._run_read(query)

    async def prune_events(
        self,
        *,
        before: datetime,
        session_id: str | None = None,
    ) -> int:
        """Delete events older than ``before`` to bound unbounded event growth.

        ``before`` is compared against each event's timestamp (events strictly
        older are removed). When ``session_id`` is given the prune is scoped to
        that session (which must exist); otherwise every session is pruned.
        The latest active or paused interaction lifecycle event is retained
        until a terminal event replaces it. Sessions with an active
        model-completion stage, pending tool round, or immutable
        runtime-publication receipt are retained because deleting their
        evidence would make exact recovery or receipt replay impossible.
        Returns the number of events deleted.
        """
        if not isinstance(before, datetime):
            raise TypeError("prune_events 'before' must be a datetime.")
        cutoff = sqlite_support.format_datetime(before)
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")

        def statement(connection: sqlite3.Connection) -> int:
            if session_id is not None and not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            publication_key_pattern = RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX + "*"
            with connection:
                if session_id is None:
                    cursor = connection.execute(
                        """
                        DELETE FROM cayu_events
                        WHERE timestamp < ?
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_persisted_event_side_effects AS delivery
                              WHERE delivery.session_id = cayu_events.session_id
                                AND delivery.event_id = cayu_events.event_id
                                AND delivery.status <> 'delivered'
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_session_operations AS operation
                              WHERE operation.session_id = cayu_events.session_id
                                AND operation.idempotency_key GLOB ?
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_session_operations AS active_stage
                              WHERE active_stage.session_id = cayu_events.session_id
                                AND active_stage.idempotency_key = ?
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_checkpoints AS checkpoint
                              WHERE checkpoint.session_id = cayu_events.session_id
                                AND json_type(
                                    checkpoint.state_json,
                                    '$.pending_tool_round'
                                ) IS NOT NULL
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_interaction_latest_events AS latest
                              JOIN cayu_events AS latest_event
                                ON latest_event.sequence = latest.latest_event_sequence
                              WHERE latest.session_id = cayu_events.session_id
                                AND latest.latest_event_sequence = cayu_events.sequence
                                AND latest_event.event_type IN (
                                    'interaction.started',
                                    'interaction.resumed',
                                    'interaction.paused'
                                )
                          )
                        """,
                        (
                            cutoff,
                            publication_key_pattern,
                            MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                        ),
                    )
                else:
                    cursor = connection.execute(
                        """
                        DELETE FROM cayu_events
                        WHERE session_id = ? AND timestamp < ?
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_persisted_event_side_effects AS delivery
                              WHERE delivery.session_id = cayu_events.session_id
                                AND delivery.event_id = cayu_events.event_id
                                AND delivery.status <> 'delivered'
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_session_operations AS operation
                              WHERE operation.session_id = cayu_events.session_id
                                AND operation.idempotency_key GLOB ?
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_session_operations AS active_stage
                              WHERE active_stage.session_id = cayu_events.session_id
                                AND active_stage.idempotency_key = ?
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_checkpoints AS checkpoint
                              WHERE checkpoint.session_id = cayu_events.session_id
                                AND json_type(
                                    checkpoint.state_json,
                                    '$.pending_tool_round'
                                ) IS NOT NULL
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM cayu_interaction_latest_events AS latest
                              JOIN cayu_events AS latest_event
                                ON latest_event.sequence = latest.latest_event_sequence
                              WHERE latest.session_id = cayu_events.session_id
                                AND latest.latest_event_sequence = cayu_events.sequence
                                AND latest_event.event_type IN (
                                    'interaction.started',
                                    'interaction.resumed',
                                    'interaction.paused'
                                )
                          )
                        """,
                        (
                            session_id,
                            cutoff,
                            publication_key_pattern,
                            MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                        ),
                    )
            return cursor.rowcount

        return await self._run_write(statement)

    async def compact_transcript(self, session_id: str, *, keep_last: int) -> int:
        """Compact a session's transcript, keeping only its most recent messages.

        Retains the ``keep_last`` newest transcript messages (by insertion order)
        for ``session_id`` and deletes the rest, bounding transcript growth for
        long-lived sessions. Active model stages, pending tool rounds, and
        immutable publication receipts pin the transcript until their recovery
        material is no longer needed. Returns the number of messages deleted.
        """
        if type(keep_last) is not int:
            raise TypeError("compact_transcript 'keep_last' must be an int.")
        if keep_last < 0:
            raise ValueError("compact_transcript 'keep_last' must be >= 0.")
        session_id = require_clean_nonblank(session_id, "session_id")

        def statement(connection: sqlite3.Connection) -> int:
            if not _session_exists(connection, session_id):
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
                    LIMIT 1
                    """,
                    (
                        session_id,
                        RUNTIME_PUBLICATION_OPERATION_KEY_PREFIX + "*",
                        MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                        session_id,
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

        return await self._run_write(statement)

    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        return await self._list_sessions(query, pending_interruption_cascade_only=False)

    async def query_session_topology(
        self,
        query: SessionTopologyQuery,
    ) -> SessionTopologyStoreResult:
        if type(query) is not SessionTopologyQuery:
            raise TypeError("Session topology queries must be SessionTopologyQuery instances.")
        query = query.model_copy(deep=True)

        def read_topology_snapshot(
            connection: sqlite3.Connection,
        ) -> SessionTopologyStoreResult:
            focus_row = connection.execute(
                f"""
                SELECT {_SESSION_TOPOLOGY_COLUMNS}
                FROM cayu_sessions
                WHERE id = ?
                """,
                (query.focus_session_id,),
            ).fetchone()
            if focus_row is None:
                raise KeyError(f"Session not found: {query.focus_session_id}")
            focus = _session_topology_node_from_sqlite_row(focus_row)

            ancestors: list[SessionTopologyNode] = []
            seen_ids = {focus.id}
            parent_session_id = focus.parent_session_id
            while parent_session_id is not None:
                if parent_session_id in seen_ids:
                    raise SessionTopologyCycle(
                        f"Session topology contains a parent cycle at {parent_session_id}."
                    )
                if len(ancestors) >= query.ancestor_depth_limit:
                    raise SessionTopologyDepthExceeded(
                        f"Session topology exceeds the {query.ancestor_depth_limit}-ancestor limit."
                    )
                parent_row = connection.execute(
                    f"""
                    SELECT {_SESSION_TOPOLOGY_COLUMNS}
                    FROM cayu_sessions
                    WHERE id = ?
                    """,
                    (parent_session_id,),
                ).fetchone()
                if parent_row is None:
                    raise ValueError(
                        f"Session topology references missing parent {parent_session_id}."
                    )
                parent = _session_topology_node_from_sqlite_row(parent_row)
                ancestors.append(parent)
                seen_ids.add(parent.id)
                parent_session_id = parent.parent_session_id
            ancestors.reverse()

            expanded_parents: list[SessionTopologyNode] = []
            if query.expanded_parent_ids:
                placeholders = ", ".join("?" for _ in query.expanded_parent_ids)
                parent_rows = connection.execute(
                    f"""
                    SELECT {_SESSION_TOPOLOGY_COLUMNS}
                    FROM cayu_sessions
                    WHERE id IN ({placeholders})
                    """,
                    query.expanded_parent_ids,
                ).fetchall()
                parents_by_id = {
                    row["id"]: _session_topology_node_from_sqlite_row(row) for row in parent_rows
                }
                for parent_id in query.expanded_parent_ids:
                    parent = parents_by_id.get(parent_id)
                    if parent is None:
                        raise KeyError(f"Session not found: {parent_id}")
                    expanded_parents.append(parent)

            candidates_by_parent: dict[str, list[SessionTopologyNode]] = {
                parent.id: [] for parent in expanded_parents
            }
            if expanded_parents:
                branch_queries: list[str] = []
                branch_params: list[object] = []
                for branch_order, parent in enumerate(expanded_parents):
                    cursor = query.child_cursors.get(parent.id)
                    if cursor is None:
                        cursor_clause = ""
                        cursor_params: list[object] = []
                    else:
                        cursor_created_at, cursor_id = decode_session_topology_cursor(
                            cursor,
                            parent_session_id=parent.id,
                        )
                        cursor_clause = "AND (created_at > ? OR (created_at = ? AND id > ?))"
                        formatted_cursor = sqlite_support.format_datetime(cursor_created_at)
                        cursor_params = [formatted_cursor, formatted_cursor, cursor_id]
                    branch_queries.append(
                        f"""
                        SELECT branch_order, {_SESSION_TOPOLOGY_COLUMNS}
                        FROM (
                            SELECT ? AS branch_order, {_SESSION_TOPOLOGY_COLUMNS}
                            FROM cayu_sessions
                            WHERE parent_session_id = ?
                              {cursor_clause}
                            ORDER BY created_at ASC, id ASC
                            LIMIT ?
                        )
                        """
                    )
                    branch_params.extend(
                        [
                            branch_order,
                            parent.id,
                            *cursor_params,
                            query.child_limit + 1,
                        ]
                    )
                candidate_rows = connection.execute(
                    f"""
                    {" UNION ALL ".join(branch_queries)}
                    ORDER BY branch_order ASC, created_at ASC, id ASC
                    """,
                    branch_params,
                ).fetchall()
                for row in candidate_rows:
                    candidates_by_parent[row["parent_session_id"]].append(
                        _session_topology_node_from_sqlite_row(row)
                    )

            result = build_session_topology_result(
                focus=focus,
                ancestors=ancestors,
                expanded_parents=expanded_parents,
                branch_candidates=(candidates_by_parent[parent.id] for parent in expanded_parents),
                child_limit=query.child_limit,
            )
            return result

        def read_topology(connection: sqlite3.Connection) -> SessionTopologyStoreResult:
            # Multiple point reads plus the batched child query must describe one
            # SQLite snapshot. A plain sequence of SELECT statements in Python's
            # legacy transaction mode would otherwise observe commits made
            # between statements.
            connection.execute("BEGIN")
            try:
                return read_topology_snapshot(connection)
            finally:
                connection.rollback()

        return await self._run_read(read_topology)

    async def query_session_lineage(
        self,
        query: SessionLineageQuery,
    ) -> SessionLineageResult:
        query = copy_session_lineage_query(query)

        def read_lineage_snapshot(connection: sqlite3.Connection) -> SessionLineageResult:
            parent_exists = connection.execute(
                "SELECT 1 FROM cayu_sessions WHERE id = ?",
                (query.parent_session_id,),
            ).fetchone()
            if parent_exists is None:
                raise KeyError(f"Session not found: {query.parent_session_id}")

            cursor_clause = ""
            params: list[object] = [
                SESSION_LINEAGE_MAX_IDENTIFIER_BYTES,
                SESSION_LINEAGE_MAX_TIMESTAMP_BYTES,
                query.parent_session_id,
            ]
            if query.cursor is not None:
                cursor_created_at, cursor_id = decode_session_lineage_cursor(
                    query.cursor,
                    parent_session_id=query.parent_session_id,
                )
                formatted_cursor = sqlite_support.format_datetime(cursor_created_at)
                cursor_clause = "AND (created_at > ? OR (created_at = ? AND id > ?))"
                params.extend((formatted_cursor, formatted_cursor, cursor_id))
            params.append(query.limit + 1)
            rows = connection.execute(
                f"""
                SELECT CASE
                           WHEN length(CAST(id AS BLOB)) <= ? THEN id
                       END AS id,
                       CASE
                           WHEN length(CAST(created_at AS BLOB)) <= ? THEN created_at
                       END AS created_at
                FROM cayu_sessions
                WHERE parent_session_id = ?
                  {cursor_clause}
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
            retained_rows = rows[: query.limit]
            children: list[SessionLineageNode] = []
            for row in retained_rows:
                base = SessionLineageNode(
                    id=row["id"],
                    parent_session_id=query.parent_session_id,
                    created_at=sqlite_support.parse_datetime(row["created_at"]),
                )
                origin_rows = connection.execute(
                    """
                    SELECT sequence,
                           CASE
                               WHEN length(event_id) <= ?
                                AND length(CAST(event_id AS BLOB)) <= ?
                               THEN event_id
                           END AS event_id,
                           event_type
                    FROM cayu_events
                    WHERE session_id = ?
                      AND event_type IN (?, ?)
                    ORDER BY sequence ASC
                    LIMIT ?
                    """,
                    (
                        EVENT_ID_MAX_CHARS,
                        SESSION_LINEAGE_MAX_EVENT_ID_BYTES,
                        base.id,
                        str(EventType.SESSION_STARTED),
                        str(EventType.SESSION_FORKED),
                        SESSION_LINEAGE_MAX_ORIGIN_EVENTS,
                    ),
                ).fetchall()
                children.append(
                    SessionLineageNode(
                        id=base.id,
                        parent_session_id=base.parent_session_id,
                        created_at=base.created_at,
                        origin_events=tuple(
                            SessionLineageOrigin(
                                sequence=origin_row["sequence"],
                                event_id=origin_row["event_id"],
                                event_type=EventType(origin_row["event_type"]),
                            )
                            for origin_row in origin_rows
                        ),
                    )
                )
            has_more = len(rows) > len(retained_rows)
            return SessionLineageResult(
                parent_session_id=query.parent_session_id,
                children=tuple(children),
                next_cursor=(
                    encode_session_lineage_cursor(query.parent_session_id, children[-1])
                    if has_more and children
                    else None
                ),
                has_more=has_more,
            )

        def read_lineage(connection: sqlite3.Connection) -> SessionLineageResult:
            connection.execute("BEGIN")
            try:
                return read_lineage_snapshot(connection)
            finally:
                connection.rollback()

        return await self._run_read(read_lineage)

    async def aggregate_operational_snapshot(
        self,
        filters: SessionAggregateFilter | None = None,
    ) -> SessionOperationalSnapshot:
        filters = copy_session_aggregate_filter(filters)
        plan = session_store_sql.build_session_query_sql(
            session_query_from_aggregate_filter(filters),
            dialect=_SQL_DIALECT,
        )

        def query_snapshot(connection: sqlite3.Connection) -> SessionOperationalSnapshot:
            rows = connection.execute(
                f"""
                WITH
                snapshot(as_of) AS (
                    SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                ),
                status_counts AS (
                    SELECT status, COUNT(*) AS status_count
                    FROM cayu_sessions
                    {plan.filter_where_sql}
                    GROUP BY status
                )
                SELECT snapshot.as_of, status_counts.status, status_counts.status_count
                FROM snapshot
                LEFT JOIN status_counts ON TRUE
                """,
                plan.filter_params,
            ).fetchall()
            counts = {status: 0 for status in SessionStatus}
            for row in rows:
                if row["status"] is not None:
                    status = SessionStatus(row["status"])
                    counts[status] = row["status_count"]
            return SessionOperationalSnapshot(
                as_of=sqlite_support.parse_datetime(rows[0]["as_of"]),
                total_count=sum(counts.values()),
                counts_by_status=SessionStatusCounts.model_validate(counts),
                accuracy=EXACT_AGGREGATE.model_copy(),
            )

        return await self._run_read(query_snapshot)

    async def aggregate_usage(self, query: UsageRollupQuery) -> UsageRollupStoreResult:
        query = copy_usage_rollup_query(query)
        plan = session_store_sql.build_session_query_sql(
            session_query_from_aggregate_filter(query.sessions),
            dialect=_SQL_DIALECT,
        )

        def query_aggregate(connection: sqlite3.Connection) -> UsageRollupStoreResult:
            return sqlite_aggregates.aggregate_session_usage(
                connection,
                session_plan=plan,
                query=query,
            )

        return await self._run_read(query_aggregate)

    async def list_sessions_with_pending_interruption_cascade(
        self,
        query: SessionQuery | None = None,
    ) -> SessionListResult:
        return await self._list_sessions(query, pending_interruption_cascade_only=True)

    async def query_pending_actions(
        self,
        query: PendingActionQuery | None = None,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> PendingActionListResult:
        from cayu.runtime.pending_actions import (
            pending_action_from_records,
            pending_action_matches_query,
            pending_action_source_is_invalid,
        )

        if query is None:
            query = PendingActionQuery()
        elif type(query) is not PendingActionQuery:
            raise TypeError("Pending-action queries must be PendingActionQuery instances.")
        else:
            query = query.model_copy(deep=True)

        inspected_candidate_limit = min(query.limit * 4, 800)
        candidate_limit = inspected_candidate_limit + 1
        filters = [
            "cayu_sessions.status IN ('interrupted', 'failed', 'completed')",
            "cayu_checkpoints.pending_action_metrics_ready = 1",
            "cayu_checkpoints.pending_action_flags <> 0",
        ]
        params: list[Any] = []
        if query.session_id is not None:
            filters.append("cayu_sessions.id = ?")
            params.append(query.session_id)
        if query.agent_name is not None:
            filters.append("cayu_sessions.agent_name = ?")
            params.append(query.agent_name)
        if query.environment_name is not None:
            filters.append("cayu_sessions.environment_name = ?")
            params.append(query.environment_name)
        if query.kind == PendingActionKind.TOOL_APPROVAL:
            filters.append("(cayu_checkpoints.pending_action_flags & 1) <> 0")
        elif query.kind == PendingActionKind.USER_INPUT:
            filters.append("(cayu_checkpoints.pending_action_flags & 2) <> 0")
        if query.cursor is not None:
            cursor_dt, cursor_id = decode_session_cursor(query.cursor)
            cursor_value = sqlite_support.format_datetime(cursor_dt)
            filters.append(
                """
                (
                    cayu_sessions.updated_at < ?
                    OR (cayu_sessions.updated_at = ? AND cayu_sessions.id > ?)
                )
                """
            )
            params.extend((cursor_value, cursor_value, cursor_id))

        where_sql = " AND ".join(f"({clause.strip()})" for clause in filters)
        candidate_select_sql = f"""
            SELECT
                cayu_sessions.id,
                cayu_sessions.agent_name,
                cayu_sessions.provider_name,
                cayu_sessions.model,
                cayu_sessions.parent_session_id,
                cayu_sessions.causal_budget_id,
                cayu_sessions.runtime_name,
                cayu_sessions.runtime_version,
                cayu_sessions.environment_name,
                cayu_sessions.status,
                cayu_sessions.created_at,
                cayu_sessions.updated_at
            FROM cayu_checkpoints
                INDEXED BY idx_cayu_checkpoints_pending_control_action
            JOIN cayu_sessions ON cayu_sessions.id = cayu_checkpoints.session_id
            WHERE {where_sql}
            ORDER BY cayu_sessions.updated_at DESC, cayu_sessions.id ASC
            LIMIT ?
        """
        selected_candidate_sql = """
            SELECT
                cayu_checkpoints.session_id AS id,
                json_object(
                    'pending_tool_approval',
                    json_extract(
                        cayu_checkpoints.state_json,
                        '$.pending_tool_approval'
                    ),
                    'pending_user_input',
                    json_extract(
                        cayu_checkpoints.state_json,
                        '$.pending_user_input'
                    ),
                    'pending_tool_round',
                    json_extract(
                        cayu_checkpoints.state_json,
                        '$.pending_tool_round'
                    )
                ) AS pending_state_json
            FROM cayu_checkpoints
            WHERE cayu_checkpoints.session_id IN (
                SELECT CAST(value AS TEXT) FROM json_each(?)
            )
        """
        checkpoint_root_key = (
            "__cayu_no_checkpoint_root_guard__"
            if checkpoint_root_guard is None
            else checkpoint_root_guard.key
        )
        checkpoint_root_path = f"$.{checkpoint_root_key}"
        checkpoint_preflight_sql = f"""
            SELECT
                cayu_checkpoints.session_id,
                cayu_checkpoints.pending_action_source_bytes AS pending_state_bytes,
                cayu_checkpoints.pending_action_tool_call_count AS pending_tool_call_count,
                json_type(
                    cayu_checkpoints.state_json,
                    '{checkpoint_root_path}'
                ) AS checkpoint_root_field_type,
                CASE
                    WHEN json_type(
                        cayu_checkpoints.state_json,
                        '{checkpoint_root_path}'
                    ) = 'integer'
                    THEN substr(
                        CAST(json_extract(
                            cayu_checkpoints.state_json,
                            '{checkpoint_root_path}'
                        ) AS TEXT),
                        1,
                        {CHECKPOINT_ROOT_FIELD_SCALAR_MAX_CHARS + 1}
                    )
                END AS checkpoint_root_field_scalar
            FROM cayu_checkpoints
            WHERE cayu_checkpoints.session_id IN (
                SELECT CAST(value AS TEXT) FROM json_each(?)
            )
        """
        projected_event_sql = "json(source_event.pending_action_projection_json)"
        pending_action_ctes = f"""
            WITH candidates AS ({selected_candidate_sql}),
            candidate_tool_scopes AS (
                SELECT candidates.id AS session_id,
                    CASE
                        WHEN json_type(
                            candidates.pending_state_json,
                            '$.pending_tool_approval'
                        ) = 'object'
                        THEN json_extract(
                            candidates.pending_state_json,
                            '$.pending_tool_approval'
                        )
                        WHEN json_type(
                            candidates.pending_state_json,
                            '$.pending_user_input'
                        ) = 'object'
                        THEN json_extract(
                            candidates.pending_state_json,
                            '$.pending_user_input'
                        )
                        WHEN json_type(
                            candidates.pending_state_json,
                            '$.pending_tool_round'
                        ) = 'object'
                        THEN json_extract(
                            candidates.pending_state_json,
                            '$.pending_tool_round'
                        )
                        ELSE NULL
                    END AS pending_tool_state_json
                FROM candidates
            ),
            candidate_tool_calls AS (
                SELECT
                    tool_scope.session_id,
                    json_extract(pending_call.value, '$.tool_call_id') AS tool_call_id
                FROM candidate_tool_scopes AS tool_scope
                JOIN json_each(
                    CASE
                        WHEN json_type(
                            tool_scope.pending_tool_state_json,
                            '$.tool_calls'
                        ) = 'array'
                        THEN json_extract(
                            tool_scope.pending_tool_state_json,
                            '$.tool_calls'
                        )
                        ELSE json('[]')
                    END
                ) AS pending_call
                WHERE json_type(pending_call.value, '$.tool_call_id') = 'text'
            ),
            candidate_action_keys AS (
                SELECT id AS session_id,
                    cayu_pending_action_lookup_key(json_extract(
                        pending_state_json,
                        '$.pending_tool_approval.approval_id'
                    )) AS action_key
                FROM candidates
                WHERE json_type(
                    pending_state_json,
                    '$.pending_tool_approval.approval_id'
                ) = 'text'
                UNION
                SELECT id,
                    cayu_pending_action_lookup_key(
                        json_extract(pending_state_json, '$.pending_user_input.input_id')
                    )
                FROM candidates
                WHERE json_type(
                    pending_state_json,
                    '$.pending_user_input.input_id'
                ) = 'text'
                UNION
                SELECT tool_scope.session_id,
                    cayu_pending_action_lookup_key(
                        json_extract(
                            tool_scope.pending_tool_state_json,
                            '$.tool_round_id'
                        )
                    )
                FROM candidate_tool_scopes AS tool_scope
                WHERE json_type(
                    tool_scope.pending_tool_state_json,
                    '$.tool_round_id'
                ) = 'text'
                UNION
                SELECT pending_call.session_id,
                    cayu_pending_action_lookup_key(pending_call.tool_call_id)
                FROM candidate_tool_calls AS pending_call
            ),
            pending_action_event_types(event_type) AS (
                VALUES
                    ('tool.call.approval_requested'),
                    ('session.awaiting_user_input'),
                    ('session.interrupted')
            ),
            latest_barriers AS (
                SELECT candidates.id AS session_id,
                    COALESCE((
                        SELECT MAX(event.sequence)
                        FROM cayu_events AS event
                            INDEXED BY idx_cayu_events_pending_action_barrier
                        WHERE event.session_id = candidates.id
                          AND (
                              event.event_type = 'session.resumed'
                              OR event.event_type = 'session.completed'
                              OR event.event_type = 'session.failed'
                          )
                    ), 0) AS sequence
                FROM candidates
            ),
            matched_action_sequences AS (
                SELECT
                    action_keys.session_id AS candidate_session_id,
                    (
                        SELECT MAX(candidate_event.sequence)
                        FROM cayu_events AS candidate_event
                            INDEXED BY idx_cayu_events_pending_action_lookup
                        WHERE candidate_event.session_id = action_keys.session_id
                          AND candidate_event.event_type = action_type.event_type
                          AND candidate_event.event_type IN (
                              'tool.call.approval_requested',
                              'session.awaiting_user_input',
                              'session.interrupted',
                              'tool.call.started',
                              'tool.call.completed',
                              'tool.call.failed',
                              'tool.call.blocked',
                              'tool.call.approval_denied'
                          )
                          AND candidate_event.pending_action_lookup_key IS NOT NULL
                          AND candidate_event.pending_action_lookup_key = action_keys.action_key
                    ) AS sequence
                FROM candidate_action_keys AS action_keys
                CROSS JOIN pending_action_event_types AS action_type
            ),
            matched_ledger_sequences AS (
                SELECT
                    action_keys.session_id AS candidate_session_id,
                    action_keys.action_key,
                    candidate_event.sequence
                FROM candidate_action_keys AS action_keys
                JOIN candidates ON candidates.id = action_keys.session_id
                JOIN candidate_tool_scopes AS tool_scope
                    ON tool_scope.session_id = action_keys.session_id
                JOIN cayu_events AS candidate_event
                    ON candidate_event.sequence IN (
                        SELECT scoped_event.sequence
                        FROM cayu_events AS scoped_event
                            INDEXED BY idx_cayu_events_pending_action_lookup
                        WHERE scoped_event.session_id = action_keys.session_id
                          AND scoped_event.pending_action_lookup_key
                              = action_keys.action_key
                          AND scoped_event.event_type IN (
                              'tool.call.approval_requested',
                              'session.awaiting_user_input',
                              'session.interrupted',
                              'tool.call.started',
                              'tool.call.completed',
                              'tool.call.failed',
                              'tool.call.blocked',
                              'tool.call.approval_denied'
                          )
                          AND scoped_event.event_type IN (
                              'tool.call.started',
                              'tool.call.completed',
                              'tool.call.failed',
                              'tool.call.blocked',
                              'tool.call.approval_denied'
                          )
                          AND scoped_event.pending_action_lookup_key IS NOT NULL
                          AND (
                              json_extract(
                                  scoped_event.pending_action_projection_json,
                                  '$.payload.tool_round_id'
                              ) = json_extract(
                                  tool_scope.pending_tool_state_json,
                                  '$.tool_round_id'
                              )
                              OR (
                                  json_extract(
                                      scoped_event.pending_action_projection_json,
                                      '$.payload.model_step_id'
                                  ) = json_extract(
                                      tool_scope.pending_tool_state_json,
                                      '$.model_step_id'
                                  )
                                  AND json_extract(
                                      scoped_event.pending_action_projection_json,
                                      '$.payload.model_attempt_id'
                                  ) = json_extract(
                                      tool_scope.pending_tool_state_json,
                                      '$.model_attempt_id'
                                  )
                              )
                          )
                        LIMIT {MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL + 1}
                    )
                WHERE json_type(
                    tool_scope.pending_tool_state_json
                ) = 'object'
            ),
            scope_conflict_sequences AS (
                SELECT tool_scope.session_id AS candidate_session_id,
                    COALESCE(
                    (
                        SELECT scoped_event.sequence
                        FROM cayu_events AS scoped_event
                            INDEXED BY idx_cayu_events_pending_action_round_scope
                        WHERE scoped_event.session_id = tool_scope.session_id
                          AND scoped_event.event_type IN (
                              'tool.call.started',
                              'tool.call.completed',
                              'tool.call.failed',
                              'tool.call.blocked',
                              'tool.call.approval_denied'
                          )
                          AND json_type(
                              scoped_event.pending_action_projection_json,
                              '$.payload.tool_round_id'
                          ) = 'text'
                          AND length(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.tool_round_id'
                          )) = 39
                          AND substr(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.tool_round_id'
                          ), 1, 7) = 'tround_'
                          AND substr(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.tool_round_id'
                          ), 8) NOT GLOB '*[^0-9a-f]*'
                          AND json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.tool_round_id'
                          ) = json_extract(
                              tool_scope.pending_tool_state_json,
                              '$.tool_round_id'
                          )
                          AND COALESCE(
                              json_extract(
                                  scoped_event.pending_action_projection_json,
                                  '$.payload.tool_round_id'
                              ) = json_extract(
                                  tool_scope.pending_tool_state_json,
                                  '$.tool_round_id'
                              )
                              AND json_extract(
                                  scoped_event.pending_action_projection_json,
                                  '$.payload.model_step_id'
                              ) = json_extract(
                                  tool_scope.pending_tool_state_json,
                                  '$.model_step_id'
                              )
                              AND json_extract(
                                  scoped_event.pending_action_projection_json,
                                  '$.payload.model_attempt_id'
                              ) = json_extract(
                                  tool_scope.pending_tool_state_json,
                                  '$.model_attempt_id'
                              )
                              AND EXISTS (
                                  SELECT 1
                                  FROM candidate_tool_calls AS pending_call
                                  WHERE pending_call.session_id = tool_scope.session_id
                                    AND pending_call.tool_call_id = json_extract(
                                        scoped_event.pending_action_projection_json,
                                        '$.payload.tool_call_id'
                                    )
                              ),
                              0
                          ) = 0
                        LIMIT 1
                    ),
                    (
                        SELECT scoped_event.sequence
                        FROM cayu_events AS scoped_event
                            INDEXED BY idx_cayu_events_pending_action_attempt_scope
                        WHERE scoped_event.session_id = tool_scope.session_id
                          AND scoped_event.event_type IN (
                              'tool.call.started',
                              'tool.call.completed',
                              'tool.call.failed',
                              'tool.call.blocked',
                              'tool.call.approval_denied'
                          )
                          AND json_type(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_step_id'
                          ) = 'text'
                          AND json_type(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_attempt_id'
                          ) = 'text'
                          AND length(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_step_id'
                          )) = 38
                          AND substr(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_step_id'
                          ), 1, 6) = 'mstep_'
                          AND substr(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_step_id'
                          ), 7) NOT GLOB '*[^0-9a-f]*'
                          AND length(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_attempt_id'
                          )) = 37
                          AND substr(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_attempt_id'
                          ), 1, 5) = 'matt_'
                          AND substr(json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_attempt_id'
                          ), 6) NOT GLOB '*[^0-9a-f]*'
                          AND json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_step_id'
                          ) = json_extract(
                              tool_scope.pending_tool_state_json,
                              '$.model_step_id'
                          )
                          AND json_extract(
                              scoped_event.pending_action_projection_json,
                              '$.payload.model_attempt_id'
                          ) = json_extract(
                              tool_scope.pending_tool_state_json,
                              '$.model_attempt_id'
                          )
                          AND COALESCE(
                              json_extract(
                                  scoped_event.pending_action_projection_json,
                                  '$.payload.tool_round_id'
                              ) = json_extract(
                                  tool_scope.pending_tool_state_json,
                                  '$.tool_round_id'
                              )
                              AND json_extract(
                                  scoped_event.pending_action_projection_json,
                                  '$.payload.model_step_id'
                              ) = json_extract(
                                  tool_scope.pending_tool_state_json,
                                  '$.model_step_id'
                              )
                              AND json_extract(
                                  scoped_event.pending_action_projection_json,
                                  '$.payload.model_attempt_id'
                              ) = json_extract(
                                  tool_scope.pending_tool_state_json,
                                  '$.model_attempt_id'
                              )
                              AND EXISTS (
                                  SELECT 1
                                  FROM candidate_tool_calls AS pending_call
                                  WHERE pending_call.session_id = tool_scope.session_id
                                    AND pending_call.tool_call_id = json_extract(
                                        scoped_event.pending_action_projection_json,
                                        '$.payload.tool_call_id'
                                    )
                              ),
                              0
                          ) = 0
                        LIMIT 1
                    )
                    ) AS sequence
                FROM candidate_tool_scopes AS tool_scope
                WHERE json_type(tool_scope.pending_tool_state_json) = 'object'
            ),
            matched_event_sequences AS (
                SELECT
                    matched_action.candidate_session_id,
                    matched_action.sequence
                FROM matched_action_sequences AS matched_action
                WHERE matched_action.sequence IS NOT NULL
                UNION
                SELECT
                    matched_ledger.candidate_session_id,
                    matched_ledger.sequence
                FROM matched_ledger_sequences AS matched_ledger
                UNION
                SELECT
                    scope_conflict.candidate_session_id,
                    scope_conflict.sequence
                FROM scope_conflict_sequences AS scope_conflict
                WHERE scope_conflict.sequence IS NOT NULL
                UNION
                SELECT
                    candidates.id,
                    event.sequence
                FROM candidates
                JOIN latest_barriers ON latest_barriers.session_id = candidates.id
                JOIN cayu_events AS event ON event.sequence = latest_barriers.sequence
            ),
            matched_events AS (
                SELECT
                    matched_event_sequences.candidate_session_id,
                    source_event.sequence,
                    source_event.pending_action_projection_bytes AS event_bytes,
                    source_event.pending_action_projection_bytes IS NOT NULL
                        AND source_event.pending_action_projection_json IS NOT NULL
                        AS projection_ready
                FROM matched_event_sequences
                JOIN cayu_events AS source_event
                    ON source_event.sequence = matched_event_sequences.sequence
            )
        """
        source_size_sql = f"""
            {pending_action_ctes}
            SELECT candidates.id,
                length(CAST(candidates.pending_state_json AS BLOB))
                + COALESCE((
                    SELECT SUM(length(CAST(json_object(
                        'key', label.key,
                        'value', label.value
                    ) AS BLOB)))
                    FROM cayu_session_labels AS label
                    WHERE label.session_id = candidates.id
                ), 0)
                + COALESCE((
                    SELECT SUM(
                        matched_event.event_bytes
                        + length(CAST(matched_event.sequence AS TEXT))
                        + 22
                    )
                    FROM matched_events AS matched_event
                    WHERE matched_event.candidate_session_id = candidates.id
                ), 0) AS source_bytes,
                COALESCE((
                    SELECT MIN(CASE WHEN matched_event.projection_ready THEN 1 ELSE 0 END)
                    FROM matched_events AS matched_event
                    WHERE matched_event.candidate_session_id = candidates.id
                ), 1) AS projections_ready,
                EXISTS (
                    SELECT 1
                    FROM matched_ledger_sequences AS matched_ledger
                    WHERE matched_ledger.candidate_session_id = candidates.id
                    GROUP BY matched_ledger.action_key
                    HAVING COUNT(*) > {MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL}
                ) AS ledger_too_complex,
                COALESCE((
                    SELECT json_group_array(ordered_sequence.sequence)
                    FROM (
                        SELECT matched_event.sequence
                        FROM matched_events AS matched_event
                        WHERE matched_event.candidate_session_id = candidates.id
                        ORDER BY matched_event.sequence DESC
                    ) AS ordered_sequence
                ), json('[]')) AS matched_event_sequences_json
            FROM candidates
        """
        materialize_sql = f"""
            WITH candidates AS ({selected_candidate_sql}),
            matched_events AS (
                SELECT
                    source_event.session_id AS candidate_session_id,
                    source_event.sequence,
                    {projected_event_sql} AS event_json
                FROM cayu_events AS source_event
                WHERE source_event.sequence IN (
                    SELECT CAST(value AS INTEGER) FROM json_each(?)
                )
            )
            SELECT
                candidates.id,
                candidates.pending_state_json,
                COALESCE((
                    SELECT json_group_array(json_object(
                        'sequence', ordered_event.sequence,
                        'event', json(ordered_event.event_json)
                    ))
                    FROM (
                        SELECT *
                        FROM matched_events
                        WHERE candidate_session_id = candidates.id
                        ORDER BY sequence DESC
                    ) AS ordered_event
                ), json('[]')) AS pending_events_json
            FROM candidates
        """

        def run_query(connection: sqlite3.Connection) -> PendingActionListResult:
            connection.execute("BEGIN")
            try:
                candidate_rows = connection.execute(
                    candidate_select_sql,
                    [*params, candidate_limit],
                ).fetchall()
                has_more_candidates = len(candidate_rows) > inspected_candidate_limit
                inspected_rows = candidate_rows[:inspected_candidate_limit]
                candidate_sessions = {
                    row["id"]: sqlite_support.pending_action_session_from_row(row, labels={})
                    for row in inspected_rows
                }
                inspected_ids = [row["id"] for row in inspected_rows]
                selected_ids_json = sqlite_support.json_dumps(inspected_ids)

                checkpoint_preflight_by_session_id: dict[str, tuple[int, int]] = {}
                if inspected_ids:
                    for row in connection.execute(
                        checkpoint_preflight_sql,
                        (selected_ids_json,),
                    ).fetchall():
                        version_type = row["checkpoint_root_field_type"]
                        scalar_text = row["checkpoint_root_field_scalar"]
                        if checkpoint_root_guard is not None:
                            checkpoint_root_guard.validate(
                                row["session_id"],
                                checkpoint_root_field_projection_from_storage(
                                    json_type=version_type,
                                    scalar_text=scalar_text,
                                ),
                            )
                        if row["pending_state_bytes"] is not None:
                            checkpoint_preflight_by_session_id[row["session_id"]] = (
                                int(row["pending_state_bytes"]),
                                int(row["pending_tool_call_count"]),
                            )

                oversized_ids: set[str] = set()
                overcomplex_ids: set[str] = set()
                preflight_eligible_ids: list[str] = []
                preflight_processable_ids: list[str] = []
                preflight_source_bytes = 0
                preflight_stopped_for_bytes = False
                for session_id in inspected_ids:
                    checkpoint_preflight = checkpoint_preflight_by_session_id.get(session_id)
                    if checkpoint_preflight is None:
                        oversized_ids.add(session_id)
                        preflight_processable_ids.append(session_id)
                        continue
                    pending_state_bytes, pending_tool_call_count = checkpoint_preflight
                    if pending_state_bytes > query.max_result_bytes:
                        oversized_ids.add(session_id)
                        preflight_processable_ids.append(session_id)
                        continue
                    if pending_tool_call_count > MAX_PENDING_ACTION_TOOL_CALLS:
                        overcomplex_ids.add(session_id)
                        preflight_processable_ids.append(session_id)
                        continue
                    if preflight_source_bytes + pending_state_bytes > query.max_result_bytes:
                        preflight_stopped_for_bytes = True
                        break
                    preflight_source_bytes += pending_state_bytes
                    preflight_eligible_ids.append(session_id)
                    preflight_processable_ids.append(session_id)

                source_metadata_by_session_id: dict[str, tuple[int, list[int]]] = {}
                invalid_ids: set[str] = set()
                ledger_overcomplex_ids: set[str] = set()
                if preflight_eligible_ids:
                    for row in connection.execute(
                        source_size_sql,
                        (sqlite_support.json_dumps(preflight_eligible_ids),),
                    ).fetchall():
                        sequence_values = json.loads(row["matched_event_sequences_json"])
                        if type(sequence_values) is not list or any(
                            type(sequence) is not int for sequence in sequence_values
                        ):
                            raise ValueError(
                                "SQLite pending event sequence projection must be an integer array."
                            )
                        source_metadata_by_session_id[row["id"]] = (
                            int(row["source_bytes"]),
                            sequence_values,
                        )
                        if not bool(row["projections_ready"]):
                            invalid_ids.add(row["id"])
                        if bool(row["ledger_too_complex"]):
                            ledger_overcomplex_ids.add(row["id"])

                processable_ids: list[str] = []
                materializable_ids: list[str] = []
                materialized_source_bytes = 0
                stopped_for_bytes = preflight_stopped_for_bytes
                for session_id in preflight_processable_ids:
                    session = candidate_sessions[session_id]
                    if (
                        session_id in oversized_ids
                        or session_id in overcomplex_ids
                        or session_id in ledger_overcomplex_ids
                        or session_id in invalid_ids
                    ):
                        processable_ids.append(session_id)
                        continue
                    session_size = JsonUtf8SizeCounter(query.max_result_bytes)
                    session_fits = session_size.value(session)
                    source_metadata = source_metadata_by_session_id.get(session_id)
                    if not session_fits or source_metadata is None:
                        oversized_ids.add(session_id)
                        processable_ids.append(session_id)
                        continue
                    stored_source_bytes = source_metadata[0]
                    candidate_bytes = (
                        query.max_result_bytes - session_size.remaining + stored_source_bytes
                    )
                    if candidate_bytes > query.max_result_bytes:
                        oversized_ids.add(session_id)
                        processable_ids.append(session_id)
                        continue
                    if materialized_source_bytes + candidate_bytes > query.max_result_bytes:
                        stopped_for_bytes = True
                        break
                    materialized_source_bytes += candidate_bytes
                    materializable_ids.append(session_id)
                    processable_ids.append(session_id)

                grouped: dict[str, tuple[dict[str, Any], list[EventRecord]]] = {}
                if materializable_ids:
                    materializable_sequences = sorted(
                        {
                            sequence
                            for session_id in materializable_ids
                            for sequence in source_metadata_by_session_id[session_id][1]
                        }
                    )
                    rows = connection.execute(
                        materialize_sql,
                        (
                            sqlite_support.json_dumps(materializable_ids),
                            sqlite_support.json_dumps(materializable_sequences),
                        ),
                    ).fetchall()
                    for row in rows:
                        session_id = row["id"]
                        pending_events = json.loads(row["pending_events_json"])
                        if type(pending_events) is not list:
                            raise ValueError("SQLite pending events projection must be an array.")
                        records: list[EventRecord] = []
                        for pending_event in pending_events:
                            if type(pending_event) is not dict:
                                raise ValueError(
                                    "SQLite pending event projections must be objects."
                                )
                            event_value = pending_event.get("event")
                            if type(event_value) is not dict:
                                raise ValueError(
                                    "SQLite pending event values must be event objects."
                                )
                            records.append(
                                EventRecord(
                                    sequence=pending_event.get("sequence"),
                                    event=Event(**event_value),
                                )
                            )
                        grouped[session_id] = (
                            copy_durable_json_object(
                                json.loads(row["pending_state_json"]),
                                "checkpoint",
                            ),
                            records,
                        )

                labels_by_session_id = self._load_labels_for_sessions_unlocked(
                    materializable_ids,
                    connection=connection,
                )
                actions = []
                issues: list[PendingActionIssue] = []
                inspected_count = 0
                more_matching = False
                last_inspected_session: PendingActionSession | None = None
                for session_id in processable_ids:
                    session = candidate_sessions[session_id]
                    if session_id in oversized_ids:
                        if len(actions) + len(issues) == query.limit:
                            more_matching = True
                            break
                        issues.append(
                            PendingActionIssue.source_too_large(
                                session,
                                max_bytes=query.max_result_bytes,
                            )
                        )
                        inspected_count += 1
                        last_inspected_session = session
                        continue
                    if session_id in overcomplex_ids:
                        if len(actions) + len(issues) == query.limit:
                            more_matching = True
                            break
                        issues.append(
                            PendingActionIssue.source_too_complex(
                                session,
                                max_tool_calls=MAX_PENDING_ACTION_TOOL_CALLS,
                            )
                        )
                        inspected_count += 1
                        last_inspected_session = session
                        continue
                    if session_id in ledger_overcomplex_ids:
                        if len(actions) + len(issues) == query.limit:
                            more_matching = True
                            break
                        issues.append(
                            PendingActionIssue.ledger_too_complex(
                                session,
                                max_events_per_call=(MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL),
                            )
                        )
                        inspected_count += 1
                        last_inspected_session = session
                        continue
                    if session_id in invalid_ids:
                        if len(actions) + len(issues) == query.limit:
                            more_matching = True
                            break
                        issues.append(PendingActionIssue.source_invalid(session))
                        inspected_count += 1
                        last_inspected_session = session
                        continue

                    checkpoint, records = grouped[session_id]
                    session = session.model_copy(
                        update={"labels": labels_by_session_id.get(session_id, {})},
                        deep=True,
                    )
                    action = pending_action_from_records(session, records, checkpoint)
                    if pending_action_source_is_invalid(session, checkpoint, action, records):
                        if len(actions) + len(issues) == query.limit:
                            more_matching = True
                            break
                        issues.append(PendingActionIssue.source_invalid(session))
                        inspected_count += 1
                        last_inspected_session = session
                        continue
                    if action is None or (query.kind is not None and action.kind != query.kind):
                        inspected_count += 1
                        last_inspected_session = session
                        continue
                    if not pending_action_matches_query(action, query.q):
                        inspected_count += 1
                        last_inspected_session = session
                        continue
                    if len(actions) + len(issues) == query.limit:
                        more_matching = True
                        break
                    actions.append(action)
                    inspected_count += 1
                    last_inspected_session = session

                has_more = more_matching or has_more_candidates or stopped_for_bytes
                next_cursor = (
                    encode_session_cursor(
                        last_inspected_session,
                        SessionOrder.UPDATED_AT_DESC,
                    )
                    if has_more and last_inspected_session is not None
                    else None
                )
                return enforce_pending_action_result_size(
                    PendingActionListResult(
                        actions=actions,
                        issues=issues,
                        next_cursor=next_cursor,
                        has_more=has_more,
                        total_count=None,
                        inspected_candidate_count=inspected_count,
                    ),
                    max_bytes=query.max_result_bytes,
                )
            finally:
                # End the pinned WAL snapshot on success and on every failure.
                connection.rollback()

        return await self._run_read(run_query)

    async def _list_sessions(
        self,
        query: SessionQuery | None,
        *,
        pending_interruption_cascade_only: bool,
    ) -> SessionListResult:
        query = copy_session_query(query)
        session_source_sql = (
            """
            (
                SELECT session_id
                FROM cayu_checkpoints
                    INDEXED BY idx_cayu_checkpoints_pending_interruption_cascade
                WHERE json_type(
                    state_json,
                    '$.pending_interruption_cascade'
                ) IS NOT NULL
            ) AS pending_interruption_cascades
            CROSS JOIN cayu_sessions
                ON cayu_sessions.id = pending_interruption_cascades.session_id
            """
            if pending_interruption_cascade_only
            else "cayu_sessions"
        )
        plan = session_store_sql.build_session_query_sql(query, dialect=_SQL_DIALECT)

        def run_query(connection: sqlite3.Connection) -> SessionListResult:
            total_count: int | None = None
            if query.include_total_count:
                total_count = connection.execute(
                    f"SELECT COUNT(*) FROM {session_source_sql} {plan.filter_where_sql}",
                    plan.filter_params,
                ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT id, agent_name, provider_name, model, parent_session_id,
                       causal_budget_id, runtime_name, runtime_version, environment_name,
                       status, created_at, updated_at, last_activity_at, run_epoch,
                       metadata_json
                FROM {session_source_sql}
                {plan.page_where_sql}
                ORDER BY {plan.order_sql}, id ASC
                {plan.pagination_sql}
                """,
                plan.page_params,
            ).fetchall()
            has_more = len(rows) > query.limit
            rows = rows[: query.limit]
            labels_by_session_id = self._load_labels_for_sessions_unlocked(
                [row["id"] for row in rows],
                connection=connection,
            )
            sessions = [
                sqlite_support.session_from_row(
                    row,
                    labels=labels_by_session_id.get(row["id"], {}),
                )
                for row in rows
            ]
            next_cursor = session_next_cursor(sessions, has_more, query.order_by)
            return SessionListResult(
                sessions=sessions,
                next_cursor=next_cursor,
                total_count=total_count,
            )

        return await self._run_read(run_query)

    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        copied_messages = copy_transcript_messages(messages)

        def statement(connection: sqlite3.Connection) -> None:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            if not copied_messages:
                return
            with connection:
                _touch_session_activity(connection, session_id, datetime.now(UTC))
                connection.executemany(
                    """
                    INSERT INTO cayu_transcript_messages (
                        session_id,
                        role,
                        interaction_id,
                        message_json
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            session_id,
                            str(message.role),
                            interaction_id,
                            sqlite_support.json_dumps(message.model_dump(mode="json")),
                        )
                        for message in copied_messages
                    ],
                )

        await self._run_write(statement)

    async def replace_initial_transcript_messages(
        self,
        session_id: str,
        expected_messages: list[Message],
        replacement_messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
        checkpoint_transform: CheckpointTransform | None = None,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        if interaction_id is None:
            raise ValueError("Initial transcript publication requires an interaction identity.")
        expected = copy_transcript_messages(expected_messages)
        replacement = copy_transcript_messages(replacement_messages)
        updated_at = datetime.now(UTC)

        def statement(connection: sqlite3.Connection) -> None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = self._load_unlocked(session_id)
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
                stored = [Message(**item) for item in json.loads(row["source_messages_json"])]
                if stored != expected:
                    raise RuntimeError("Deferred interaction input changed before finalization.")
                existing = connection.execute(
                    "SELECT 1 FROM cayu_transcript_messages WHERE session_id = ? LIMIT 1",
                    (session_id,),
                ).fetchone()
                if existing is not None:
                    raise RuntimeError("Initial transcript changed before finalization.")
                if len(replacement) < len(expected) or (
                    expected and replacement[-len(expected) :] != expected
                ):
                    raise RuntimeError(
                        "Initial transcript must preserve the admitted source suffix."
                    )
                prefix_count = len(replacement) - len(expected)
                current_checkpoint = self._load_checkpoint_unlocked(session_id)
                if checkpoint_transform is not None:
                    transformed = checkpoint_transform(
                        session,
                        current_checkpoint,
                    )
                    if transformed is not None:
                        current_checkpoint = copy_durable_json_object(
                            transformed,
                            "checkpoint",
                        )
                checkpoint = _checkpoint_after_initial_transcript_publication(
                    current_checkpoint,
                    interaction_id=interaction_id,
                )
                connection.executemany(
                    "INSERT INTO cayu_transcript_messages "
                    "(session_id, role, interaction_id, message_json) VALUES (?, ?, ?, ?)",
                    [
                        (
                            session_id,
                            str(message.role),
                            None if index < prefix_count else interaction_id,
                            sqlite_support.json_dumps(message.model_dump(mode="json")),
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
                        sqlite_support.checkpoint_row_values(
                            session_id,
                            checkpoint,
                            updated_at,
                        ),
                    )
                _touch_session_activity(connection, session_id, updated_at)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        await self._run_write(statement)

    async def materialize_deferred_interaction_input(
        self,
        session_id: str,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> bool:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)

        def statement(connection: sqlite3.Connection) -> bool:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = self._load_unlocked(session_id)
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
                messages = [Message(**item) for item in json.loads(row["source_messages_json"])]
                connection.executemany(
                    "INSERT INTO cayu_transcript_messages "
                    "(session_id, role, interaction_id, message_json) VALUES (?, ?, ?, ?)",
                    [
                        (
                            session_id,
                            str(message.role),
                            interaction_id,
                            sqlite_support.json_dumps(message.model_dump(mode="json")),
                        )
                        for message in messages
                    ],
                )
                connection.execute(
                    "DELETE FROM cayu_deferred_interaction_inputs WHERE session_id = ?",
                    (session_id,),
                )
                _touch_session_activity(connection, session_id, datetime.now(UTC))
                connection.commit()
                return True
            except Exception:
                connection.rollback()
                raise

        return await self._run_write(statement)

    async def load_deferred_interaction_input(
        self,
        session_id: str,
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
            return DeferredInteractionInput(
                interaction_id=row["interaction_id"],
                source_messages=[
                    Message(**item) for item in json.loads(row["source_messages_json"])
                ],
            )

        return await self._run_read(query)

    async def append_transcript_messages_and_transform_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint_transform: CheckpointTransform,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        copied_messages = copy_transcript_messages(messages)
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        updated_at = datetime.now(UTC)

        def statement(connection: sqlite3.Connection) -> None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = self._load_unlocked(session_id)
                if session is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, session)
                transformed = checkpoint_transform(
                    session,
                    self._load_checkpoint_unlocked(session_id),
                )
                if transformed is None:
                    raise ValueError("Checkpoint transform must return a checkpoint.")
                transformed = copy_durable_json_object(transformed, "checkpoint")
                _touch_session_activity(connection, session_id, updated_at)
                if copied_messages:
                    connection.executemany(
                        """
                        INSERT INTO cayu_transcript_messages (
                            session_id,
                            role,
                            interaction_id,
                            message_json
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        [
                            (
                                session_id,
                                str(message.role),
                                interaction_id,
                                sqlite_support.json_dumps(message.model_dump(mode="json")),
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
                    sqlite_support.checkpoint_row_values(session_id, transformed, updated_at),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        await self._run_write(statement)

    async def load_transcript(self, session_id: str) -> list[Message]:
        session_id = require_clean_nonblank(session_id, "session_id")

        def query(connection: sqlite3.Connection) -> list[Message]:
            if not _session_exists(connection, session_id):
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

        return await self._run_read(query)

    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
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
            if not _session_exists(connection, query.session_id):
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

        return await self._run_read(run_query)

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        checkpoint = copy_durable_json_object(state, "checkpoint")
        updated_at = datetime.now(UTC)

        def statement(connection: sqlite3.Connection) -> None:
            if not _session_exists(connection, session_id):
                raise KeyError(f"Session not found: {session_id}")
            with connection:
                _touch_session_activity(connection, session_id, updated_at)
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
                    sqlite_support.checkpoint_row_values(session_id, checkpoint, updated_at),
                )

        await self._run_write(statement)

    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        updated_at = datetime.now(UTC)

        def statement(connection: sqlite3.Connection) -> None:
            try:
                connection.execute("BEGIN IMMEDIATE")
                session = self._load_unlocked(session_id)
                if session is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, session)
                transformed = checkpoint_transform(
                    session,
                    self._load_checkpoint_unlocked(session_id),
                )
                if transformed is not None:
                    transformed = copy_durable_json_object(transformed, "checkpoint")
                    _touch_session_activity(connection, session_id, updated_at)
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
                        sqlite_support.checkpoint_row_values(session_id, transformed, updated_at),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

        await self._run_write(statement)

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        return await self._run_read(
            lambda connection: _load_checkpoint_state(connection, session_id)
        )

    async def load_interruption_cascade_marker(
        self,
        session_id: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        return await self._run_read(
            lambda connection: _load_interruption_cascade_marker(
                connection,
                session_id,
                checkpoint_root_guard,
            )
        )

    def _load_checkpoint_unlocked(self, session_id: str) -> dict[str, Any] | None:
        return _load_checkpoint_state(self._connection, session_id)

    async def close(self) -> None:
        async with self._lock:
            if self._read_connection is not self._connection:
                async with self._read_lock:
                    self._read_connection.close()
            self._connection.close()

    def _connect(self, path: Path) -> sqlite3.Connection:
        return sqlite_support.connect(path)

    def _connect_read_only(self, path: Path) -> sqlite3.Connection:
        return sqlite_support.connect(path, read_only=True)

    def _initialize_schema(self) -> None:
        sqlite_support.reconcile_schema(
            self._connection,
            self._schema_mode,
            app_min_supported=_SQLITE_SESSION_MIN_REQUIRED_REVISION,
        )
        state = sqlite_support.read_schema_state(self._connection)
        if state.revision < _SQLITE_SESSION_MIN_REQUIRED_REVISION:
            raise schema.SchemaTooOld(
                f"SQLite session schema is at revision {state.revision}; this build requires "
                f">= {_SQLITE_SESSION_MIN_REQUIRED_REVISION}. Run `cayu storage migrate` before "
                "starting."
            )

    def _load_unlocked(self, session_id: str) -> Session | None:
        return _load_session(self._connection, session_id)

    def _load_labels_unlocked(self, session_id: str) -> dict[str, str]:
        return _load_labels(self._connection, session_id)

    def _load_labels_for_sessions_unlocked(
        self,
        session_ids: list[str],
        *,
        connection: sqlite3.Connection | None = None,
    ) -> dict[str, dict[str, str]]:
        if not session_ids:
            return {}
        source = self._connection if connection is None else connection
        placeholders = ", ".join("?" for _ in session_ids)
        rows = source.execute(
            f"""
            SELECT session_id, key, value
            FROM cayu_session_labels
            WHERE session_id IN ({placeholders})
            ORDER BY session_id ASC, key ASC
            """,
            session_ids,
        ).fetchall()
        labels_by_session_id: dict[str, dict[str, str]] = {
            session_id: {} for session_id in session_ids
        }
        for row in rows:
            labels_by_session_id[row["session_id"]][row["key"]] = row["value"]
        return labels_by_session_id

    def _session_exists_unlocked(self, session_id: str) -> bool:
        return _session_exists(self._connection, session_id)

    def _first_existing_event_id_unlocked(
        self,
        session_id: str,
        event_ids: list[str],
    ) -> str | None:
        return _first_existing_event_id(self._connection, session_id, event_ids)


class SQLiteTaskStore(TaskStore):
    """SQLite-backed task store for durable local work items."""

    supports_task_topology: ClassVar[bool] = True

    def __init__(
        self,
        path: str | Path,
        *,
        schema_mode: schema.SchemaMode = schema.SchemaMode.CREATE,
    ) -> None:
        if isinstance(path, Path):
            db_path = path
        elif type(path) is str:
            db_path = Path(require_nonblank(path, "path"))
        else:
            raise TypeError("SQLiteTaskStore path must be a string or Path.")
        self.service_durability = (
            RuntimeStoreDurability.DEVELOPMENT
            if str(db_path) == ":memory:"
            else RuntimeStoreDurability.DURABLE
        )
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")

        self.path = db_path
        self._schema_mode = schema_mode
        self._lock = asyncio.Lock()
        self._connection = self._connect(db_path)
        self._initialize_schema()

    async def create_task(self, request: TaskCreate) -> Task:
        request = copy_task_create(request)
        async with self._lock:
            task = _task_from_create(request)
            self._insert_task_unlocked(task)
            return task.model_copy(deep=True)

    async def create_running_task(self, request: TaskCreate) -> Task:
        request = copy_task_create(request)
        async with self._lock:
            task = _running_task_from_create(request)
            self._insert_task_unlocked(task)
            return task.model_copy(deep=True)

    def _insert_task_unlocked(self, task: Task) -> None:
        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO cayu_tasks (
                        id,
                        type,
                        title,
                        description,
                        status,
                        session_id,
                        parent_task_id,
                        assigned_agent_name,
                        worker_id,
                        lease_expires_at,
                        status_reason,
                        status_payload_json,
                        input_json,
                        result_json,
                        error_json,
                        metadata_json,
                        created_at,
                        updated_at,
                        started_at,
                        completed_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    sqlite_support.task_to_row_values(task),
                )
        except sqlite3.IntegrityError as exc:
            if self._task_exists_unlocked(task.id):
                raise ValueError(f"Task already exists: {task.id}") from exc
            raise

    async def load_task(self, task_id: str) -> Task | None:
        task_id = require_clean_nonblank(task_id, "task_id")
        async with self._lock:
            return self._load_task_unlocked(task_id)

    async def list_tasks(self, query: TaskQuery | None = None) -> list[Task]:
        query = copy_task_query(query)
        clauses: list[str] = []
        params: list[object] = []

        if query.q is not None:
            like = _like_contains_pattern(query.q)
            clauses.append(
                """
                (
                    id COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR type COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR title COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR description COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR status COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR session_id COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR parent_task_id COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR assigned_agent_name COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR worker_id COLLATE NOCASE LIKE ? ESCAPE '\\'
                    OR status_reason COLLATE NOCASE LIKE ? ESCAPE '\\'
                )
                """
            )
            params.extend([like] * 10)
        if query.status is not None:
            clauses.append("status = ?")
            params.append(str(query.status))
        if query.type is not None:
            clauses.append("type = ?")
            params.append(query.type)
        if query.session_id is not None:
            clauses.append("session_id = ?")
            params.append(query.session_id)
        if query.parent_task_id is not None:
            clauses.append("parent_task_id = ?")
            params.append(query.parent_task_id)
        if query.assigned_agent_name is not None:
            clauses.append("assigned_agent_name = ?")
            params.append(query.assigned_agent_name)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order_sql = sqlite_support.task_order_sql(query.order_by)
        params.extend([query.limit, query.offset])

        async with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT *
                FROM cayu_tasks
                {where_sql}
                ORDER BY {order_sql}, id ASC
                LIMIT ? OFFSET ?
                """,
                params,
            ).fetchall()
            return [sqlite_support.task_from_row(row) for row in rows]

    async def query_task_topology(
        self,
        query: TaskTopologyQuery,
    ) -> TaskTopologyStoreResult:
        if type(query) is not TaskTopologyQuery:
            raise TypeError("Task topology queries must be TaskTopologyQuery instances.")
        query = TaskTopologyQuery.model_validate(query.model_dump(mode="python"))
        session_branch_limits, child_branch_limits = _allocate_task_topology_branch_limits(query)

        def read_branch_candidates(
            *,
            branch_ids: tuple[str, ...],
            cursors: dict[str, str],
            scope_kind: Literal["session", "parent_task"],
            scope_column: Literal["session_id", "parent_task_id"],
            branch_limits: tuple[int, ...],
        ) -> list[list[TaskTopologyNode]]:
            candidates: list[list[TaskTopologyNode]] = [[] for _ in branch_ids]
            if not branch_ids:
                return candidates
            branch_queries: list[str] = []
            branch_params: list[object] = []
            for branch_order, (branch_id, branch_limit) in enumerate(
                zip(branch_ids, branch_limits, strict=True)
            ):
                cursor = cursors.get(branch_id)
                if cursor is None:
                    cursor_clause = ""
                    cursor_params: list[object] = []
                else:
                    cursor_created_at, cursor_id = decode_task_topology_cursor(
                        cursor,
                        scope_kind=scope_kind,
                        scope_id=branch_id,
                    )
                    cursor_clause = "AND (created_at > ? OR (created_at = ? AND id > ?))"
                    formatted = sqlite_support.format_datetime(cursor_created_at)
                    cursor_params = [formatted, formatted, cursor_id]
                branch_queries.append(
                    f"""
                    SELECT branch_order, candidate.*
                    FROM (
                        SELECT ? AS branch_order, {sqlite_support.TASK_TOPOLOGY_COLUMNS}
                        FROM cayu_tasks
                        WHERE {scope_column} = ?
                          {cursor_clause}
                        ORDER BY created_at ASC, id ASC
                        LIMIT ?
                    ) AS candidate
                    """
                )
                branch_params.extend(
                    [
                        branch_order,
                        branch_id,
                        *cursor_params,
                        branch_limit + 1,
                    ]
                )
            rows = self._connection.execute(
                f"""
                {" UNION ALL ".join(branch_queries)}
                ORDER BY branch_order ASC, topology_created_at ASC, topology_id ASC
                """,
                branch_params,
            ).fetchall()
            for row in rows:
                candidates[row["branch_order"]].append(
                    sqlite_support.task_topology_node_from_row(row)
                )
            return candidates

        async with self._lock:
            self._connection.execute("BEGIN")
            try:
                observed_row = self._connection.execute(
                    "SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"
                ).fetchone()
                if observed_row is None:
                    raise RuntimeError("SQLite did not return a topology snapshot timestamp.")
                observed_at = sqlite_support.parse_datetime(observed_row[0])

                expanded_parents: list[TaskTopologyNode] = []
                if query.expanded_parent_ids:
                    placeholders = ", ".join("?" for _ in query.expanded_parent_ids)
                    rows = self._connection.execute(
                        f"""
                        SELECT {sqlite_support.TASK_TOPOLOGY_COLUMNS}
                        FROM cayu_tasks
                        WHERE id IN ({placeholders})
                        """,
                        query.expanded_parent_ids,
                    ).fetchall()
                    parents_by_id = {
                        row["topology_id"]: sqlite_support.task_topology_node_from_row(row)
                        for row in rows
                    }
                    for parent_id in query.expanded_parent_ids:
                        parent = parents_by_id.get(parent_id)
                        if parent is None:
                            raise KeyError(f"Task not found: {parent_id}")
                        expanded_parents.append(parent)

                session_candidates = read_branch_candidates(
                    branch_ids=query.linked_session_ids,
                    cursors=query.session_cursors,
                    scope_kind="session",
                    scope_column="session_id",
                    branch_limits=session_branch_limits,
                )
                child_candidates = read_branch_candidates(
                    branch_ids=query.expanded_parent_ids,
                    cursors=query.child_cursors,
                    scope_kind="parent_task",
                    scope_column="parent_task_id",
                    branch_limits=child_branch_limits,
                )

                async def load_parent_links(
                    task_ids: tuple[str, ...],
                ) -> dict[str, str | None]:
                    links: dict[str, str | None] = {}
                    for index in range(0, len(task_ids), 500):
                        batch = task_ids[index : index + 500]
                        placeholders = ", ".join("?" for _ in batch)
                        rows = self._connection.execute(
                            f"""
                            SELECT
                                id,
                                CASE
                                    WHEN parent_task_id IS NULL
                                      OR length(CAST(parent_task_id AS BLOB))
                                         <= {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
                                    THEN parent_task_id
                                END AS topology_parent_task_id,
                                parent_task_id IS NOT NULL
                                  AND length(CAST(parent_task_id AS BLOB))
                                      > {TASK_TOPOLOGY_MAX_IDENTIFIER_BYTES}
                                    AS topology_parent_task_id_oversized
                            FROM cayu_tasks
                            WHERE id IN ({placeholders})
                            """,
                            batch,
                        ).fetchall()
                        for row in rows:
                            if row["topology_parent_task_id_oversized"]:
                                raise TaskTopologyInconsistent(
                                    "A task topology ancestor contains an oversized "
                                    "parent identifier."
                                )
                            links[row["id"]] = _bounded_optional_task_topology_parent_id(
                                row["topology_parent_task_id"]
                            )
                    return links

                await _validate_task_topology_ancestry(
                    (
                        *expanded_parents,
                        *(task for branch in session_candidates for task in branch),
                        *(task for branch in child_candidates for task in branch),
                    ),
                    load_parent_links,
                )
                result = build_task_topology_result(
                    observed_at=observed_at,
                    linked_session_ids=query.linked_session_ids,
                    session_branch_candidates=session_candidates,
                    session_branch_limits=session_branch_limits,
                    expanded_parents=expanded_parents,
                    child_branch_candidates=child_candidates,
                    child_branch_limits=child_branch_limits,
                    session_task_limit=query.session_task_limit,
                    child_limit=query.child_limit,
                )
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    async def aggregate_operational_snapshot(
        self,
        filters: TaskAggregateFilter | None = None,
    ) -> TaskOperationalSnapshot:
        filters = copy_task_aggregate_filter(filters)
        clauses, params = self._task_filter_clauses(task_query_from_aggregate_filter(filters))
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        async with self._lock:
            rows = self._connection.execute(
                f"""
                WITH
                snapshot(as_of) AS (
                    SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                ),
                status_counts AS (
                    SELECT status, COUNT(*) AS status_count
                    FROM cayu_tasks
                    {where_sql}
                    GROUP BY status
                )
                SELECT snapshot.as_of, status_counts.status, status_counts.status_count
                FROM snapshot
                LEFT JOIN status_counts ON TRUE
                """,
                params,
            ).fetchall()
            counts = {status: 0 for status in TaskStatus}
            for row in rows:
                if row["status"] is not None:
                    status = TaskStatus(row["status"])
                    counts[status] = row["status_count"]
            return TaskOperationalSnapshot(
                as_of=sqlite_support.parse_datetime(rows[0]["as_of"]),
                total_count=sum(counts.values()),
                counts_by_status=TaskStatusCounts.model_validate(counts),
                accuracy=EXACT_AGGREGATE.model_copy(),
            )

    async def start_task(
        self,
        task_id: str,
        *,
        session_id: str | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            now = datetime.now(UTC)
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        session_id = COALESCE(?, session_id),
                        started_at = COALESCE(started_at, ?),
                        updated_at = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        str(TaskStatus.RUNNING),
                        session_id,
                        sqlite_support.format_datetime(now),
                        sqlite_support.format_datetime(now),
                        task_id,
                        str(TaskStatus.PENDING),
                    ),
                )
            if cursor.rowcount == 1:
                updated = self._require_task_unlocked(task_id)
                return updated.model_copy(deep=True)
            task = self._require_task_unlocked(task_id)
            _ensure_can_transition(task, TaskStatus.RUNNING)
            raise ValueError(f"Task {task.id} cannot transition to running from {task.status}")

    async def attach_task(
        self,
        task_id: str,
        *,
        session_id: str,
        worker_id: str,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        session_id = require_clean_nonblank(session_id, "session_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        now = datetime.now(UTC)
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        session_id = ?,
                        started_at = COALESCE(started_at, ?),
                        updated_at = ?
                    WHERE id = ?
                      AND status = ?
                      AND worker_id = ?
                      AND session_id IS NULL
                      AND lease_expires_at IS NOT NULL
                      AND lease_expires_at > ?
                    """,
                    (
                        str(TaskStatus.RUNNING),
                        session_id,
                        sqlite_support.format_datetime(now),
                        sqlite_support.format_datetime(now),
                        task_id,
                        str(TaskStatus.CLAIMED),
                        worker_id,
                        sqlite_support.format_datetime(now),
                    ),
                )
            if cursor.rowcount != 1:
                self._raise_task_claim_attach_error(task_id, worker_id)
            updated = self._require_task_unlocked(task_id)
            return updated.model_copy(deep=True)

    async def complete_task(
        self, task_id: str, result: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        result = copy_durable_json_object(result, "result")
        async with self._lock:
            return self._finish_task_unlocked(
                task_id,
                TaskStatus.COMPLETED,
                result=result,
                error=None,
                worker_id=worker_id,
            )

    async def fail_task(
        self, task_id: str, error: dict[str, Any], *, worker_id: str | None = None
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        error = copy_durable_json_object(error, "error")
        async with self._lock:
            return self._finish_task_unlocked(
                task_id,
                TaskStatus.FAILED,
                result=None,
                error=error,
                worker_id=worker_id,
            )

    async def cancel_task(
        self,
        task_id: str,
        error: dict[str, Any] | None = None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        copied_error = None if error is None else copy_durable_json_object(error, "error")
        async with self._lock:
            return self._finish_task_unlocked(
                task_id,
                TaskStatus.CANCELLED,
                result=None,
                error=copied_error,
            )

    async def pause_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        return await self._hold_task(
            task_id,
            TaskStatus.PAUSED,
            reason=reason,
            payload=payload,
        )

    async def block_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        return await self._hold_task(
            task_id,
            TaskStatus.BLOCKED,
            reason=reason,
            payload=payload,
        )

    async def mark_task_needs_attention(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        return await self._hold_task(
            task_id,
            TaskStatus.NEEDS_ATTENTION,
            reason=reason,
            payload=payload,
        )

    async def resume_task(self, task_id: str) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        now = datetime.now(UTC)
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        status_reason = NULL,
                        status_payload_json = NULL,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ?
                      AND status IN (?, ?, ?)
                    """,
                    (
                        str(TaskStatus.PENDING),
                        sqlite_support.format_datetime(now),
                        task_id,
                        str(TaskStatus.PAUSED),
                        str(TaskStatus.BLOCKED),
                        str(TaskStatus.NEEDS_ATTENTION),
                    ),
                )
            if cursor.rowcount != 1:
                task = self._require_task_unlocked(task_id)
                _ensure_can_resume_task(task)
                raise ValueError(f"Task {task.id} cannot resume from {task.status}")
            updated = self._require_task_unlocked(task_id)
            return updated.model_copy(deep=True)

    async def claim_task(
        self,
        worker_id: str,
        query: TaskQuery | None = None,
        *,
        lease_seconds: int = 300,
    ) -> Task | None:
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        lease_seconds = _validate_task_positive_int(lease_seconds, "lease_seconds")
        if query.status is not None and query.status is not TaskStatus.PENDING:
            return None
        clauses, params = self._task_filter_clauses(query)
        where_sql = " AND ".join(["status = ?", "session_id IS NULL", *clauses])
        # Claiming is always FIFO by creation time, independent of the query's
        # display ordering, so the oldest pending task is dispatched first.
        order_sql = sqlite_support.task_order_sql(TaskOrder.CREATED_AT_ASC)
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=lease_seconds)

        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    f"""
                    SELECT id
                    FROM cayu_tasks
                    WHERE {where_sql}
                    ORDER BY {order_sql}, id ASC
                    LIMIT 1
                    """,
                    [str(TaskStatus.PENDING), *params],
                ).fetchone()
                if row is None:
                    self._connection.commit()
                    return None
                task_id = row["id"]
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        worker_id = ?,
                        lease_expires_at = ?,
                        updated_at = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        str(TaskStatus.CLAIMED),
                        worker_id,
                        sqlite_support.format_datetime(lease_expires_at),
                        sqlite_support.format_datetime(now),
                        task_id,
                        str(TaskStatus.PENDING),
                    ),
                )
                if cursor.rowcount != 1:
                    self._connection.rollback()
                    return None
                updated = self._require_task_unlocked(task_id)
                self._connection.commit()
                return updated.model_copy(deep=True)
            except Exception:
                self._connection.rollback()
                raise

    async def heartbeat(
        self,
        task_id: str,
        worker_id: str,
        *,
        extend_seconds: int = 300,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        extend_seconds = _validate_task_positive_int(extend_seconds, "extend_seconds")
        now = datetime.now(UTC)
        lease_expires_at = now + timedelta(seconds=extend_seconds)
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET lease_expires_at = ?,
                        updated_at = ?
                    WHERE id = ? AND worker_id = ? AND status IN (?, ?)
                      AND lease_expires_at IS NOT NULL AND lease_expires_at > ?
                    """,
                    (
                        sqlite_support.format_datetime(lease_expires_at),
                        sqlite_support.format_datetime(now),
                        task_id,
                        worker_id,
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.RUNNING),
                        sqlite_support.format_datetime(now),
                    ),
                )
            if cursor.rowcount != 1:
                self._raise_task_active_lease_error(task_id, worker_id)
            updated = self._require_task_unlocked(task_id)
            return updated.model_copy(deep=True)

    async def release_task(self, task_id: str, worker_id: str) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        now = datetime.now(UTC)
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ? AND worker_id = ? AND status = ?
                      AND session_id IS NULL
                      AND lease_expires_at IS NOT NULL AND lease_expires_at > ?
                    """,
                    (
                        str(TaskStatus.PENDING),
                        sqlite_support.format_datetime(now),
                        task_id,
                        worker_id,
                        str(TaskStatus.CLAIMED),
                        sqlite_support.format_datetime(now),
                    ),
                )
            if cursor.rowcount != 1:
                self._raise_task_release_error(task_id, worker_id)
            updated = self._require_task_unlocked(task_id)
            return updated.model_copy(deep=True)

    async def release_attached_task_worker(self, task_id: str, worker_id: str) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        now = datetime.now(UTC)
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ? AND worker_id = ? AND status = ?
                      AND session_id IS NOT NULL
                      AND lease_expires_at IS NOT NULL AND lease_expires_at > ?
                    """,
                    (
                        sqlite_support.format_datetime(now),
                        task_id,
                        worker_id,
                        str(TaskStatus.RUNNING),
                        sqlite_support.format_datetime(now),
                    ),
                )
            if cursor.rowcount != 1:
                self._raise_attached_task_worker_release_error(task_id, worker_id)
            updated = self._require_task_unlocked(task_id)
            return updated.model_copy(deep=True)

    async def reclaim_expired(
        self,
        *,
        query: TaskQuery | None = None,
        max_reclaims: int = 100,
    ) -> list[Task]:
        query = copy_task_query(query)
        _ensure_claim_query_supported(query)
        max_reclaims = _validate_task_positive_int(max_reclaims, "max_reclaims")
        if query.status is not None and query.status is not TaskStatus.CLAIMED:
            return []
        clauses, params = self._task_filter_clauses(query)
        where_sql = " AND ".join(
            [
                "status = ?",
                "session_id IS NULL",
                "lease_expires_at IS NOT NULL",
                "lease_expires_at <= ?",
                *clauses,
            ]
        )
        now = datetime.now(UTC)
        async with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                rows = self._connection.execute(
                    f"""
                    SELECT id
                    FROM cayu_tasks
                    WHERE {where_sql}
                    ORDER BY lease_expires_at ASC, id ASC
                    LIMIT ?
                    """,
                    [
                        str(TaskStatus.CLAIMED),
                        sqlite_support.format_datetime(now),
                        *params,
                        max_reclaims,
                    ],
                ).fetchall()
                task_ids = [row["id"] for row in rows]
                if task_ids:
                    self._connection.executemany(
                        """
                        UPDATE cayu_tasks
                        SET status = ?,
                            worker_id = NULL,
                            lease_expires_at = NULL,
                            updated_at = ?
                        WHERE id = ? AND status = ? AND session_id IS NULL
                        """,
                        [
                            (
                                str(TaskStatus.PENDING),
                                sqlite_support.format_datetime(now),
                                task_id,
                                str(TaskStatus.CLAIMED),
                            )
                            for task_id in task_ids
                        ],
                    )
                reclaimed = [self._require_task_unlocked(task_id) for task_id in task_ids]
                self._connection.commit()
                return [task.model_copy(deep=True) for task in reclaimed]
            except Exception:
                self._connection.rollback()
                raise

    async def close(self) -> None:
        async with self._lock:
            self._connection.close()

    def _connect(self, path: Path) -> sqlite3.Connection:
        return sqlite_support.connect(path)

    def _initialize_schema(self) -> None:
        sqlite_support.reconcile_schema(
            self._connection,
            self._schema_mode,
            app_min_supported=_SQLITE_TASK_TOPOLOGY_MIN_REQUIRED_REVISION,
        )

    def _load_task_unlocked(self, task_id: str) -> Task | None:
        row = self._connection.execute(
            "SELECT * FROM cayu_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return None
        return sqlite_support.task_from_row(row)

    def _require_task_unlocked(self, task_id: str) -> Task:
        task = self._load_task_unlocked(task_id)
        if task is None:
            raise KeyError(f"Task not found: {task_id}")
        return task

    def _task_exists_unlocked(self, task_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM cayu_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        return row is not None

    def _finish_task_unlocked(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
        worker_id: str | None = None,
    ) -> Task:
        now = datetime.now(UTC)
        # When a worker_id is given, only terminalize if that worker still owns an active
        # lease — a worker that lost its lease must not clobber a task another has reclaimed.
        owner_clause = ""
        owner_params: list[str] = []
        if worker_id is not None:
            owner_clause = (
                "\n                  AND worker_id = ?"
                "\n                  AND lease_expires_at IS NOT NULL AND lease_expires_at > ?"
            )
            owner_params = [worker_id, sqlite_support.format_datetime(now)]
        with self._connection:
            cursor = self._connection.execute(
                f"""
                UPDATE cayu_tasks
                SET status = ?,
                    status_reason = NULL,
                    status_payload_json = NULL,
                    result_json = ?,
                    error_json = ?,
                    worker_id = NULL,
                    lease_expires_at = NULL,
                    started_at = COALESCE(started_at, ?),
                    completed_at = ?,
                    updated_at = ?
                WHERE id = ?
                  AND status NOT IN (?, ?, ?){owner_clause}
                """,
                (
                    str(status),
                    None if result is None else sqlite_support.json_dumps(result),
                    None if error is None else sqlite_support.json_dumps(error),
                    sqlite_support.format_datetime(now),
                    sqlite_support.format_datetime(now),
                    sqlite_support.format_datetime(now),
                    task_id,
                    str(TaskStatus.COMPLETED),
                    str(TaskStatus.FAILED),
                    str(TaskStatus.CANCELLED),
                    *owner_params,
                ),
            )
        if cursor.rowcount != 1:
            if worker_id is not None:
                self._raise_task_active_lease_error(task_id, worker_id)
            task = self._require_task_unlocked(task_id)
            _ensure_can_transition(task, status)
            raise ValueError(f"Task {task.id} cannot transition from {task.status}")
        updated = self._require_task_unlocked(task_id)
        return updated.model_copy(deep=True)

    async def _hold_task(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        reason: str | None,
        payload: dict[str, Any] | None,
    ) -> Task:
        task_id = require_clean_nonblank(task_id, "task_id")
        reason = _copy_optional_status_reason(reason)
        payload = _copy_optional_status_payload(payload)
        now = datetime.now(UTC)
        async with self._lock:
            with self._connection:
                cursor = self._connection.execute(
                    """
                    UPDATE cayu_tasks
                    SET status = ?,
                        status_reason = ?,
                        status_payload_json = ?,
                        worker_id = NULL,
                        lease_expires_at = NULL,
                        updated_at = ?
                    WHERE id = ?
                      AND (
                        status = ?
                        OR status = ?
                        OR status = ?
                        OR status = ?
                        OR status = ?
                        OR (status = ? AND session_id IS NULL)
                      )
                    """,
                    (
                        str(status),
                        reason,
                        None if payload is None else sqlite_support.json_dumps(payload),
                        sqlite_support.format_datetime(now),
                        task_id,
                        str(TaskStatus.PENDING),
                        str(TaskStatus.CLAIMED),
                        str(TaskStatus.PAUSED),
                        str(TaskStatus.BLOCKED),
                        str(TaskStatus.NEEDS_ATTENTION),
                        str(TaskStatus.RUNNING),
                    ),
                )
            if cursor.rowcount != 1:
                task = self._require_task_unlocked(task_id)
                _ensure_can_hold_task(task, status)
                raise ValueError(f"Task {task.id} cannot transition to {status}")
            updated = self._require_task_unlocked(task_id)
            return updated.model_copy(deep=True)

    def _task_filter_clauses(self, query: TaskQuery) -> tuple[list[str], list[object]]:
        clauses: list[str] = []
        params: list[object] = []
        if query.type is not None:
            clauses.append("type = ?")
            params.append(query.type)
        if query.session_id is not None:
            clauses.append("session_id = ?")
            params.append(query.session_id)
        if query.parent_task_id is not None:
            clauses.append("parent_task_id = ?")
            params.append(query.parent_task_id)
        if query.assigned_agent_name is not None:
            clauses.append("assigned_agent_name = ?")
            params.append(query.assigned_agent_name)
        return clauses, params

    def _raise_task_active_lease_error(self, task_id: str, worker_id: str) -> None:
        task = self._require_task_unlocked(task_id)
        _ensure_owned_active_task_lease(task, worker_id)
        raise RuntimeError(f"Task {task.id} active-lease mutation did not update a row.")

    def _raise_task_release_error(self, task_id: str, worker_id: str) -> None:
        task = self._require_task_unlocked(task_id)
        _ensure_owned_active_task_lease(task, worker_id)
        if task.session_id is not None:
            raise ValueError(f"Task {task.id} is already attached to session {task.session_id}.")
        if task.status is not TaskStatus.CLAIMED:
            raise ValueError(f"Task {task.id} is not claimed.")
        raise RuntimeError(f"Task {task.id} active claim could not be released.")

    def _raise_attached_task_worker_release_error(
        self,
        task_id: str,
        worker_id: str,
    ) -> None:
        task = self._require_task_unlocked(task_id)
        _ensure_owned_active_task_lease(task, worker_id)
        if task.status is not TaskStatus.RUNNING:
            raise ValueError(f"Task {task.id} is not running.")
        if task.session_id is None:
            raise ValueError(f"Task {task.id} is not attached to a session.")
        raise RuntimeError(f"Task {task.id} active attached claim could not be released.")

    def _raise_task_claim_attach_error(self, task_id: str, worker_id: str) -> None:
        task = self._require_task_unlocked(task_id)
        _raise_task_claim_attach_error(task, worker_id)


def _validate_task_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return value
