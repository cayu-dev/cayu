"""Complete SQLite queued-message persistence operations.

Native capabilities retain authorization, replay and publication within each
operation's transaction. Queue readers share that same authoritative snapshot.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

from cayu._validation import copy_durable_json_object
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.approvals.tools import ResolutionActor, resolution_actor_payload
from cayu.events import Event, EventType, event_with_runtime_payload_authority
from cayu.messages import Message
from cayu.runtime import _session_message_queue as message_queue
from cayu.sessions._execution_profile_checkpoint import ActiveInvocationExecutionProfile
from cayu.sessions.access import require_resource_session
from cayu.sessions.base import (
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    QueuedInteractionProfileHandoff,
    SessionMessageQueueStatus,
    SessionRunFenced,
    SessionStatusConflict,
    _assert_session_run_epoch,
    _check_closure_lineage_owner,
    _checkpoint_after_queued_interaction_profile_handoff,
    _copy_historical_queued_interaction_profile_handoff,
    _copy_queued_interaction_profile_handoff,
    _copy_queued_interaction_started_event,
    _historical_queued_handoff_stage_from_records,
    _interaction_transition_storage_key,
    _model_completion_stage_dispatch_storage_key,
    _model_completion_stage_storage_identity,
    _reconstruct_active_model_completion_stage,
    _reconstruct_active_model_completion_stage_record,
    _reconstruct_model_completion_stage_dispatch,
    _validate_message_delivery_eligible_through,
)
from cayu.sessions.messaging import (
    SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    SESSION_MESSAGE_QUEUE_STORAGE_VALUE_MAX_BYTES,
    EnqueueSessionMessageRequest,
    EnqueueSessionMessageResult,
    SessionMessageActionRequest,
    SessionMessageActionResult,
    SessionMessageConditions,
    SessionMessageConflict,
    SessionMessageDeliveryBatch,
    SessionMessageInspection,
    SessionMessageQuery,
    SessionMessageSource,
    SessionQueuedMessage,
    _queued_session_message_event_payload,
    _validate_equivalent_queued_session_message,
    copy_enqueue_session_message_request,
    enqueue_session_message_input,
    queued_session_message_input,
    session_message_rejection,
)
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.transcript_input import (
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    session_messages_input_contract_evidence,
)
from cayu.sessions.transcript_queries import transcript_search_document
from cayu.storage import _sqlite_event_delivery as event_delivery_ops
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage import _sqlite_transcript as transcript_ops
from cayu.storage._sqlite_connection import SQLiteOperationRunner
from cayu.storage._sqlite_transcript import ClosureOwners


class EventAppender(Protocol):
    def __call__(
        self,
        connection: sqlite3.Connection,
        session_id: str,
        events: Sequence[Event],
        *,
        activity_at: datetime,
    ) -> None: ...


class SteeringGuard(Protocol):
    def __call__(
        self,
        connection: sqlite3.Connection,
        session: Session,
        *,
        allow_completed_interaction: bool = False,
    ) -> None: ...


def _acceptance_events(
    connection: sqlite3.Connection,
    session_id: str,
    rows: list[Any],
    *,
    quarantine_queue_id: str | None = None,
) -> dict[str, Event]:
    """Batch-read bounded audit projections of canonical events in the queue transaction."""
    ids = [row["accepted_event_id"] for row in rows]
    if quarantine_queue_id is None and any(
        type(event_id) is not str or len(event_id) > 512 for event_id in ids
    ):
        raise SessionMessageConflict()
    if not ids:
        return {}
    projection = (
        "json_object('queue_id', json_extract(payload_json, '$.queue_id'), "
        "'source', json_extract(payload_json, '$.source'))"
    )
    predicate = (
        f"event_id IN ({', '.join('?' for _ in ids)})"
        if quarantine_queue_id is None
        else "json_extract(payload_json, '$.queue_id') = ? LIMIT 2"
    )
    events = connection.execute(
        f"SELECT event_id, CASE WHEN length(CAST({projection} AS BLOB)) <= 32768 "
        f"THEN {projection} END AS audit_json FROM cayu_events WHERE session_id = ? "
        "AND event_type = 'session.message.queued' "
        f"AND {predicate}",
        (session_id, *(ids if quarantine_queue_id is None else [quarantine_queue_id])),
    ).fetchall()
    if quarantine_queue_id is not None and len(events) != 1:
        raise SessionMessageConflict()
    if any(row["audit_json"] is None for row in events):
        raise SessionMessageConflict()
    return {
        row["event_id"]: Event(
            id=row["event_id"],
            type=EventType.SESSION_MESSAGE_QUEUED,
            session_id=session_id,
            payload=json.loads(row["audit_json"]),
        )
        for row in events
    }


def _raw_bounded(
    connection: sqlite3.Connection,
    session_id: str,
    queue_id: str,
) -> dict[str, Any]:
    """Hash oversized cells in chunks without hydrating rejected content."""
    from hashlib import sha256

    columns = (
        "ordering_key",
        "queue_id",
        "session_id",
        "idempotency_key",
        "content",
        "message_json",
        "conditions_json",
        "terminal_json",
        "delivery_mode",
        "status",
        "requested_by_json",
        "accepted_run_epoch",
        "accepted_transcript_cursor",
        "accepted_event_id",
        "accepted_at",
        "delivered_run_epoch",
        "delivered_transcript_cursor",
        "delivered_event_id",
        "delivered_at",
    )
    projection = ", ".join(
        f"CASE WHEN length(CAST({name} AS BLOB)) <= {SESSION_MESSAGE_QUEUE_STORAGE_VALUE_MAX_BYTES} "
        f"THEN {name} END AS {name}"
        for name in columns
    )
    row = connection.execute(
        f"SELECT {projection} FROM cayu_session_message_queue WHERE session_id = ? AND queue_id = ?",
        (session_id, queue_id),
    ).fetchone()
    if row is None:
        raise SessionMessageConflict()
    raw = dict(row)
    sizes = connection.execute(
        "SELECT "
        + ", ".join(f"length(CAST({name} AS BLOB))" for name in columns)
        + " FROM cayu_session_message_queue WHERE session_id = ? AND queue_id = ?",
        (session_id, queue_id),
    ).fetchone()
    for name, size in zip(columns, sizes, strict=True):
        if size is None or size <= SESSION_MESSAGE_QUEUE_STORAGE_VALUE_MAX_BYTES:
            continue
        digest = sha256()
        for offset in range(1, size + 1, 65536):
            chunk = connection.execute(
                f"SELECT substr(CAST({name} AS BLOB), ?, 65536) FROM cayu_session_message_queue "
                "WHERE session_id = ? AND queue_id = ?",
                (offset, session_id, queue_id),
            ).fetchone()[0]
            digest.update(chunk)
        storage_type = connection.execute(
            f"SELECT typeof({name}) FROM cayu_session_message_queue WHERE session_id = ? AND queue_id = ?",
            (session_id, queue_id),
        ).fetchone()[0]
        raw[name] = message_queue.OversizedStorageValue(
            digest.hexdigest(), byte_length=size, storage_type=storage_type
        )
    return raw


def message_from_row(row: sqlite3.Row | dict[str, Any]) -> SessionQueuedMessage:
    requested_by = row["requested_by_json"]
    message_json = row["message_json"]
    return SessionQueuedMessage(
        queue_id=row["queue_id"],
        session_id=row["session_id"],
        idempotency_key=row["idempotency_key"],
        conditions=SessionMessageConditions.model_validate(
            {} if row["conditions_json"] is None else json.loads(row["conditions_json"])
        ),
        content=row["content"],
        message=(
            None if message_json is None else Message.model_validate(json.loads(message_json))
        ),
        delivery_mode=row["delivery_mode"],
        status=row["status"],
        ordering_key=row["ordering_key"],
        accepted_run_epoch=row["accepted_run_epoch"],
        accepted_transcript_cursor=row["accepted_transcript_cursor"],
        accepted_event_id=row["accepted_event_id"],
        accepted_at=sqlite_records.parse_datetime(row["accepted_at"]),
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
            else sqlite_records.parse_datetime(row["delivered_at"])
        ),
    )


def _source_snapshot(
    connection: sqlite3.Connection,
    session: Session,
    *,
    include_transcript_digest: bool,
    include_checkpoint_digest: bool,
    load_checkpoint: Callable[[sqlite3.Connection, str], dict[str, Any] | None],
) -> SessionMessageSource:
    cursor = transcript_ops.transcript_cursor(connection, session.id)
    transcript_digest = None
    if include_transcript_digest:
        hasher = message_queue.SourceTranscriptHasher(cursor)
        for row in connection.execute(
            "SELECT session_order, message_json FROM cayu_transcript_messages "
            "WHERE session_id = ? ORDER BY session_order",
            (session.id,),
        ):
            hasher.add(row[0] - 1, Message.model_validate_json(row[1]))
        transcript_digest = hasher.hexdigest()
    checkpoint = None
    if include_checkpoint_digest:
        checkpoint = load_checkpoint(connection, session.id)
    return message_queue.source_snapshot(
        session,
        cursor,
        transcript_sha256=transcript_digest,
        checkpoint=checkpoint,
        include_checkpoint_digest=include_checkpoint_digest,
    )


async def snapshot_session_message_source(
    run_read: SQLiteOperationRunner,
    session_id: str,
    *,
    include_transcript_digest: bool = False,
    include_checkpoint_digest: bool = False,
    expected_authorized_session_instance_id: str | None = None,
    load_checkpoint: Callable[[sqlite3.Connection, str], dict[str, Any] | None],
) -> SessionMessageSource:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(include_transcript_digest) is not bool or type(include_checkpoint_digest) is not bool:
        raise TypeError("Snapshot digest flags must be bool.")

    def query(connection: sqlite3.Connection) -> SessionMessageSource:
        connection.execute("BEGIN")
        try:
            session = sqlite_records.load_session(connection, session_id)
            require_resource_session(session, "read")
            if session is None:
                raise KeyError("Session not found.")
            message_queue.require_authorized_session_instance(
                session, expected_authorized_session_instance_id
            )
            result = _source_snapshot(
                connection,
                session,
                include_transcript_digest=include_transcript_digest,
                include_checkpoint_digest=include_checkpoint_digest,
                load_checkpoint=load_checkpoint,
            )
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise

    return await run_read(query)


async def inspect_session_messages(
    run_read: SQLiteOperationRunner,
    query: SessionMessageQuery,
    *,
    expected_authorized_session_instance_id: str | None = None,
) -> SessionMessageInspection:
    query = message_queue.copy_inspection_query(query)

    def read(connection: sqlite3.Connection) -> SessionMessageInspection:
        connection.execute("BEGIN")
        try:
            session = sqlite_records.load_session(connection, query.session_id)
            require_resource_session(session, "read")
            if session is None:
                raise KeyError("Session not found.")
            message_queue.require_authorized_session_instance(
                session, expected_authorized_session_instance_id
            )
            maximum = 0
            if query.cursor is None:
                maximum = connection.execute(
                    "SELECT COALESCE(MAX(ordering_key), 0) FROM cayu_session_message_queue "
                    "WHERE session_id = ?",
                    (session.id,),
                ).fetchone()[0]
            boundary = message_queue.inspection_boundary(session, query.cursor, maximum)
            rows = connection.execute(
                "WITH ordered AS (SELECT queue_id, ordering_key, "
                "CASE delivery_mode WHEN 'next_turn' THEN 0 WHEN 'on_idle' THEN 1 ELSE 2 END "
                "AS priority FROM cayu_session_message_queue "
                "WHERE session_id = ? AND ordering_key <= ?) "
                "SELECT queue_id, ordering_key, priority FROM ordered "
                "WHERE (priority, ordering_key) > (?, ?) "
                "ORDER BY priority, ordering_key LIMIT ?",
                (
                    session.id,
                    boundary.through_ordering_key,
                    boundary.after_priority,
                    boundary.after_ordering_key,
                    query.limit + 1,
                ),
            ).fetchall()
            raw_rows = [
                _raw_bounded(connection, query.session_id, row["queue_id"])
                for row in rows[: query.limit]
            ]
            records = tuple(
                message_queue.inspect_record(raw, lambda raw=raw: message_from_row(raw))
                for raw in raw_rows
            )
            result = SessionMessageInspection(
                session_id=session.id,
                session_instance_id=session.instance_id,
                records=records,
                next_cursor=(
                    message_queue.inspection_next_cursor(
                        boundary,
                        rows[query.limit - 1]["priority"],
                        records[-1].ordering_key,
                    )
                    if len(rows) > query.limit
                    else None
                ),
            )
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise

    return await run_read(read)


async def apply_session_message_action(
    run_write: SQLiteOperationRunner,
    request: SessionMessageActionRequest,
    *,
    load_session: Callable[[str], Session | None],
    store_now: Callable[[], datetime],
    closure_owners: ClosureOwners,
    append_events: EventAppender,
) -> SessionMessageActionResult:
    request = SessionMessageActionRequest(**message_queue.action_material(request))

    def statement(connection: sqlite3.Connection) -> SessionMessageActionResult:
        connection.execute("BEGIN IMMEDIATE")
        try:
            session = load_session(request.session_id)
            require_resource_session(session, "modify")
            if session is None or session.instance_id != request.session_instance_id:
                raise SessionMessageConflict()
            raw = _raw_bounded(connection, session.id, request.queue_id)
            accepted_events = _acceptance_events(
                connection,
                session.id,
                [raw],
                quarantine_queue_id=request.queue_id if request.action == "quarantine" else None,
            )
            accepted_event = (
                next(iter(accepted_events.values()))
                if request.action == "quarantine"
                else accepted_events.get(raw["accepted_event_id"])
            )
            replay = message_queue.replay_action(raw, request, accepted_event=accepted_event)
            record = message_queue.inspect_record(raw, lambda: message_from_row(raw))
            if replay is not None:
                connection.commit()
                return SessionMessageActionResult(record=record, event=replay, replayed=True)
            for owner in closure_owners((session.id,), connection=connection):
                _check_closure_lineage_owner(owner, (session.id,))
            if (
                record.revision != request.expected_revision
                or raw["status"] != "queued"
                or any(
                    raw[key] is not None
                    for key in (
                        "delivered_event_id",
                        "delivered_at",
                        "delivered_run_epoch",
                        "delivered_transcript_cursor",
                    )
                )
            ):
                raise SessionMessageConflict()
            # Independent receipt evidence survives event retention and malformed content.
            delivered = connection.execute(
                "SELECT 1 FROM cayu_session_message_deliveries, json_each(queue_ids_json) "
                "WHERE session_id = ? AND json_each.value = ? LIMIT 1",
                (session.id, request.queue_id),
            ).fetchone()
            if delivered is not None or (
                request.action == "withdraw" and record.validity != "valid"
            ):
                raise SessionMessageConflict()
            status = SessionMessageQueueStatus(
                "withdrawn" if request.action == "withdraw" else "quarantined"
            )
            now = store_now()
            event = message_queue.terminal_event(
                session,
                raw,
                status,
                now,
                actor=request.requested_by,
                accepted_event=accepted_event,
            )
            proof = sqlite_records.json_dumps(
                message_queue.terminal_receipt(status, event, request)
            )
            connection.execute(
                "UPDATE cayu_session_message_queue SET status = ?, terminal_json = ? "
                "WHERE session_id = ? AND queue_id = ? AND status = 'queued'",
                (str(status), proof, session.id, request.queue_id),
            )
            append_events(connection, session.id, [event], activity_at=now)
            updated = _raw_bounded(connection, session.id, request.queue_id)
            result = SessionMessageActionResult(
                record=message_queue.inspect_record(updated, lambda: message_from_row(updated)),
                event=event,
            )
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise

    return await run_write(statement)


async def enqueue_session_message(
    run_write: SQLiteOperationRunner,
    request: EnqueueSessionMessageRequest,
    *,
    expected_authorized_target_instance_id: str | None = None,
    load_session: Callable[[str], Session | None],
    store_now: Callable[[], datetime],
    closure_owners: ClosureOwners,
    load_checkpoint: Callable[[sqlite3.Connection, str], dict[str, Any] | None],
    touch_activity: Callable[[sqlite3.Connection, str, datetime], None],
) -> EnqueueSessionMessageResult:
    request = copy_enqueue_session_message_request(request)

    def statement(connection: sqlite3.Connection) -> EnqueueSessionMessageResult:
        from cayu.sessions.pending_actions import pending_action_event_storage_values

        try:
            connection.execute("BEGIN IMMEDIATE")
            loaded = load_session(request.session_id)
            require_resource_session(loaded, "modify")
            if loaded is None:
                raise KeyError(f"Session not found: {request.session_id}")
            if expected_authorized_target_instance_id is not None and (
                type(expected_authorized_target_instance_id) is not str
                or loaded.instance_id != expected_authorized_target_instance_id
            ):
                raise SessionMessageConflict()
            if request.conditions.source is not None:
                source = load_session(request.conditions.source.session_id)
                require_resource_session(source, "read")
                if (
                    source is None
                    or source.instance_id != request.conditions.source.session_instance_id
                ):
                    raise SessionMessageConflict()
            existing_row = connection.execute(
                "SELECT * FROM cayu_session_message_queue "
                "WHERE session_id = ? AND idempotency_key = ?",
                (request.session_id, request.idempotency_key),
            ).fetchone()
            if existing_row is not None:
                existing = message_from_row(existing_row)
                _validate_equivalent_queued_session_message(existing, request)
                event_row = connection.execute(
                    f"SELECT {', '.join(sqlite_records.EVENT_COLUMN_NAMES)} FROM cayu_events "
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
                    event=sqlite_records.event_from_row(event_row),
                    replayed=True,
                )
            for owner in closure_owners((request.session_id,)):
                _check_closure_lineage_owner(owner, (request.session_id,))
            checkpoint = load_checkpoint(connection, request.session_id)
            message_queue.require_open_admission(loaded.status, checkpoint)
            if request.conditions.source is not None:
                expected_source = request.conditions.source
                source = load_session(expected_source.session_id)
                require_resource_session(source, "read")
                if (
                    source is None
                    or _source_snapshot(
                        connection,
                        source,
                        include_transcript_digest=expected_source.transcript_sha256 is not None,
                        include_checkpoint_digest=expected_source.checkpoint_sha256 is not None,
                        load_checkpoint=load_checkpoint,
                    )
                    != expected_source
                ):
                    raise SessionMessageConflict()
            transcript_cursor = transcript_ops.transcript_cursor(connection, request.session_id)
            accepted_at = store_now()
            queue_id = str(uuid4())
            accepted_event_id = str(uuid4())
            cursor = connection.execute(
                """
                    INSERT INTO cayu_session_message_queue (
                        queue_id, session_id, idempotency_key, content, message_json,
                        delivery_mode, status, requested_by_json,
                        accepted_run_epoch, accepted_transcript_cursor,
                        accepted_event_id, accepted_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
                    """,
                (
                    queue_id,
                    request.session_id,
                    request.idempotency_key,
                    request.content,
                    (
                        None
                        if request.message is None
                        else sqlite_records.json_dumps(request.message.model_dump(mode="json"))
                    ),
                    str(request.delivery_mode),
                    (
                        None
                        if request.requested_by is None
                        else sqlite_records.json_dumps(
                            resolution_actor_payload(request.requested_by)
                        )
                    ),
                    loaded.run_epoch,
                    transcript_cursor,
                    accepted_event_id,
                    sqlite_records.format_datetime(accepted_at),
                ),
            )
            ordering_key = cursor.lastrowid
            if type(ordering_key) is not int:
                raise RuntimeError("SQLite queue insert did not return an ordering key.")
            connection.execute(
                "UPDATE cayu_session_message_queue SET conditions_json = ? WHERE queue_id = ?",
                (
                    sqlite_records.json_dumps(request.conditions.model_dump(mode="json")),
                    queue_id,
                ),
            )
            accepted_message = enqueue_session_message_input(request)
            accepted_event = event_with_runtime_payload_authority(
                Event(
                    id=accepted_event_id,
                    type=EventType.SESSION_MESSAGE_QUEUED,
                    session_id=request.session_id,
                    agent_name=loaded.agent_name,
                    environment_name=loaded.environment_name,
                    timestamp=accepted_at,
                    payload={
                        **_queued_session_message_event_payload(
                            queue_id=queue_id,
                            delivery_mode=request.delivery_mode,
                            ordering_key=ordering_key,
                            actor=request.requested_by,
                            run_epoch=loaded.run_epoch,
                            transcript_cursor=transcript_cursor,
                        ),
                        **message_queue.source_event_payload(request.conditions.source),
                        SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: (
                            session_messages_input_contract_evidence(
                                (accepted_message,),
                                message_start_index=transcript_cursor,
                                redactions_applied=request._input_redactions_applied,
                                structured_output_requested=False,
                            )
                        ),
                    },
                ),
                SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
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
                    sqlite_records.format_datetime(accepted_event.timestamp),
                    accepted_event.agent_name,
                    accepted_event.environment_name,
                    accepted_event.workflow_name,
                    accepted_event.tool_name,
                    sqlite_records.json_dumps(accepted_event.payload),
                    lookup_key,
                    projection,
                    projection_bytes,
                ),
            )
            event_delivery_ops.enqueue_persisted_event_side_effects(
                connection,
                request.session_id,
                [accepted_event],
            )
            touch_activity(connection, request.session_id, accepted_at)
            connection.commit()
            stored_row = connection.execute(
                "SELECT * FROM cayu_session_message_queue WHERE queue_id = ?",
                (queue_id,),
            ).fetchone()
            if stored_row is None:
                raise RuntimeError("Queued session message disappeared after acceptance.")
            return EnqueueSessionMessageResult(
                message=message_from_row(stored_row),
                event=accepted_event,
            )
        except Exception:
            connection.rollback()
            raise

    return await run_write(statement)


async def deliver_queued_session_messages(
    run_write: SQLiteOperationRunner,
    session_id: str,
    *,
    include_on_idle: bool,
    reject_only: bool = False,
    delivery_id: str | None = None,
    eligible_through: int | None = None,
    limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    interaction_id: str | None = None,
    interaction_started_event: Event | None = None,
    profile_handoff: QueuedInteractionProfileHandoff | None = None,
    load_session: Callable[[str], Session | None],
    store_now: Callable[[], datetime],
    decode_stage_record: Callable[[str], dict[str, Any]],
    load_checkpoint: Callable[[sqlite3.Connection, str], dict[str, Any] | None],
    touch_activity: Callable[[sqlite3.Connection, str, datetime], None],
    reject_steering: SteeringGuard,
) -> SessionMessageDeliveryBatch:
    session_id = require_clean_nonblank(session_id, "session_id")
    delivery_id = (
        str(uuid4()) if delivery_id is None else require_clean_nonblank(delivery_id, "delivery_id")
    )
    if interaction_id is not None:
        interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
    interaction_started_event = _copy_queued_interaction_started_event(
        session_id,
        interaction_id,
        interaction_started_event,
    )
    profile_handoff = _copy_queued_interaction_profile_handoff(
        session_id,
        delivery_id,
        interaction_id,
        interaction_started_event,
        profile_handoff,
    )
    if type(include_on_idle) is not bool:
        raise TypeError("include_on_idle must be a bool.")
    if type(reject_only) is not bool:
        raise TypeError("reject_only must be a bool.")
    eligible_through = _validate_message_delivery_eligible_through(eligible_through)
    if type(limit) is not int or not 1 <= limit <= SESSION_MESSAGE_DELIVERY_BATCH_LIMIT:
        raise ValueError(f"limit must be between 1 and {SESSION_MESSAGE_DELIVERY_BATCH_LIMIT}.")

    def statement(connection: sqlite3.Connection) -> SessionMessageDeliveryBatch:
        from cayu.sessions.pending_actions import pending_action_event_storage_values

        try:
            connection.execute("BEGIN IMMEDIATE")
            loaded = load_session(session_id)
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
                    else Event.model_validate_json(delivery_row["interaction_started_event_json"])
                )
                if (
                    delivery_row["session_id"] != session_id
                    or bool(delivery_row["reject_only"]) != reject_only
                    or bool(delivery_row["include_on_idle"]) != include_on_idle
                    or delivery_row["requested_eligible_through"] != eligible_through
                    or delivery_row["batch_limit"] != limit
                    or delivery_row["interaction_id"] != interaction_id
                    or stored_started_event != interaction_started_event
                ):
                    raise ValueError("delivery_id was already used for a different queue delivery.")
                queue_ids = json.loads(delivery_row["queue_ids_json"])
                replayed_messages: list[SessionQueuedMessage] = []
                replayed_events = [
                    Event.model_validate(event) for event in json.loads(delivery_row["events_json"])
                ]
                for queue_id in queue_ids:
                    queued_row = connection.execute(
                        "SELECT * FROM cayu_session_message_queue WHERE queue_id = ?",
                        (queue_id,),
                    ).fetchone()
                    if queued_row is None:
                        raise RuntimeError("Queue delivery replay lost a delivered message.")
                    replayed_messages.append(message_from_row(queued_row))
                if replayed_messages and profile_handoff is not None:
                    receipt_row = connection.execute(
                        "SELECT record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (
                            session_id,
                            _interaction_transition_storage_key(
                                profile_handoff.predecessor_settlement_event_id
                            ),
                        ),
                    ).fetchone()
                    if receipt_row is None:
                        raise SessionRunFenced(
                            "Queued interaction handoff lost its predecessor settlement receipt."
                        )
                    active_model_stage = None
                    stage_dispatch = None
                    active_row = connection.execute(
                        "SELECT record_json FROM cayu_session_operations "
                        "WHERE session_id = ? AND idempotency_key = ?",
                        (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    ).fetchone()
                    if active_row is not None:
                        active_record = decode_stage_record(active_row["record_json"])
                        marker = _reconstruct_active_model_completion_stage_record(
                            active_record,
                            session_id=session_id,
                        )
                        _, _, preparation_key, terminal_key = (
                            _model_completion_stage_storage_identity(
                                session_id,
                                marker.stage_id,
                            )
                        )
                        dispatch_key = _model_completion_stage_dispatch_storage_key(marker.stage_id)
                        stage_rows = connection.execute(
                            "SELECT idempotency_key, record_json "
                            "FROM cayu_session_operations WHERE session_id = ? "
                            "AND idempotency_key IN (?, ?, ?)",
                            (
                                session_id,
                                preparation_key,
                                terminal_key,
                                dispatch_key,
                            ),
                        ).fetchall()
                        stage_records = {
                            row["idempotency_key"]: (decode_stage_record(row["record_json"]))
                            for row in stage_rows
                        }
                        active_model_stage = _reconstruct_active_model_completion_stage(
                            active_record,
                            stage_records.get(preparation_key),
                            stage_records.get(terminal_key),
                            session_id=session_id,
                        )
                        dispatch_record = stage_records.get(dispatch_key)
                        if dispatch_record is not None:
                            stage_dispatch = _reconstruct_model_completion_stage_dispatch(
                                dispatch_record,
                                session_id=session_id,
                                stage_id=marker.stage_id,
                                storage_key=dispatch_key,
                            )
                    repaired_checkpoint = _checkpoint_after_queued_interaction_profile_handoff(
                        loaded,
                        load_checkpoint(connection, session_id),
                        profile_handoff,
                        settlement_record=copy_durable_json_object(
                            json.loads(receipt_row["record_json"]),
                            "interaction transition receipt",
                        ),
                        replayed_delivery=True,
                        active_model_stage=active_model_stage,
                        stage_dispatch=stage_dispatch,
                    )
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
                                pending_action_source_bytes = excluded.pending_action_source_bytes,
                                pending_action_tool_call_count =
                                    excluded.pending_action_tool_call_count,
                                pending_action_flags = excluded.pending_action_flags,
                                pending_action_metrics_ready =
                                    excluded.pending_action_metrics_ready
                            """,
                        sqlite_records.checkpoint_row_values(
                            session_id,
                            repaired_checkpoint,
                            store_now(),
                        ),
                    )
                connection.commit()
                return SessionMessageDeliveryBatch(
                    messages=tuple(replayed_messages),
                    events=tuple(replayed_events),
                    delivery_id=delivery_id,
                    interaction_id=interaction_id,
                    eligible_through=delivery_row["eligible_through"],
                    has_more=bool(delivery_row["has_more"]),
                    replayed=True,
                    active_invocation_profile=(
                        None
                        if not replayed_messages or profile_handoff is None
                        else profile_handoff.target_active_profile
                    ),
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
            reject_only_more = False
            if reject_only:
                # Eligible rows remain pending. Scan in bounded pages so they cannot hide
                # an expired record behind the first delivery-sized prefix.
                rows = []
                scan_now = store_now()
                scan_cursor = transcript_ops.transcript_cursor(connection, session_id)
                for mode in ("next_turn", "on_idle") if include_on_idle else ("next_turn",):
                    after = 0
                    while len(rows) < limit + 1:
                        page = connection.execute(
                            "SELECT * FROM cayu_session_message_queue "
                            "WHERE session_id = ? AND status = 'queued' AND delivery_mode = ? "
                            "AND ordering_key > ? AND ordering_key <= ? "
                            "ORDER BY ordering_key LIMIT 100",
                            (session_id, mode, after, boundary),
                        ).fetchall()
                        if not page:
                            break
                        for candidate in page:
                            queued = message_from_row(candidate)
                            if (
                                session_message_rejection(
                                    queued.conditions,
                                    session_instance_id=loaded.instance_id,
                                    run_epoch=loaded.run_epoch,
                                    transcript_cursor=scan_cursor,
                                    now=scan_now,
                                )
                                is not None
                            ):
                                rows.append(candidate)
                                if len(rows) == limit + 1:
                                    reject_only_more = True
                                    break
                        after = page[-1]["ordering_key"]
                    if len(rows) == limit + 1:
                        break
                rows = rows[:limit]
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
                            else sqlite_records.json_dumps(
                                interaction_started_event.model_dump(mode="json")
                            )
                        ),
                        sqlite_records.format_datetime(store_now()),
                    ),
                )
                connection.execute(
                    "UPDATE cayu_session_message_deliveries SET reject_only = ? WHERE delivery_id = ?",
                    (reject_only, delivery_id),
                )
                connection.commit()
                return SessionMessageDeliveryBatch(
                    delivery_id=delivery_id,
                    interaction_id=interaction_id,
                    eligible_through=boundary,
                    has_more=False,
                )
            transcript_cursor = transcript_ops.transcript_cursor(connection, session_id)
            delivered_at = scan_now if reject_only else store_now()
            accepted_events = _acceptance_events(connection, session_id, rows)
            rejection_events: list[Event] = []
            deliverable_rows = []
            for row in rows:
                queued = message_from_row(row)
                rejection = session_message_rejection(
                    queued.conditions,
                    session_instance_id=loaded.instance_id,
                    run_epoch=loaded.run_epoch,
                    transcript_cursor=transcript_cursor + len(deliverable_rows),
                    now=delivered_at,
                )
                if rejection is None:
                    if not reject_only:
                        deliverable_rows.append(row)
                    continue
                event = message_queue.terminal_event(
                    loaded,
                    dict(row),
                    rejection,
                    delivered_at,
                    accepted_event=accepted_events.get(row["accepted_event_id"]),
                    actor=queued.requested_by,
                    interaction_id=interaction_id,
                )
                rejection_events.append(event)
                connection.execute(
                    "UPDATE cayu_session_message_queue SET status = ?, terminal_json = ? "
                    "WHERE queue_id = ? AND status = 'queued'",
                    (
                        str(rejection),
                        sqlite_records.json_dumps(message_queue.terminal_receipt(rejection, event)),
                        queued.queue_id,
                    ),
                )
            rows = deliverable_rows
            rebound_checkpoint: dict[str, Any] | None = None
            if rows and profile_handoff is not None:
                reject_steering(connection, loaded, allow_completed_interaction=True)
                receipt_row = connection.execute(
                    "SELECT record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key = ?",
                    (
                        session_id,
                        _interaction_transition_storage_key(
                            profile_handoff.predecessor_settlement_event_id
                        ),
                    ),
                ).fetchone()
                if receipt_row is None:
                    raise SessionRunFenced(
                        "Queued interaction handoff lost its predecessor settlement receipt."
                    )
                rebound_checkpoint = _checkpoint_after_queued_interaction_profile_handoff(
                    loaded,
                    load_checkpoint(connection, session_id),
                    profile_handoff,
                    settlement_record=copy_durable_json_object(
                        json.loads(receipt_row["record_json"]),
                        "interaction transition receipt",
                    ),
                    replayed_delivery=False,
                )
            updated_messages: list[SessionQueuedMessage] = []
            delivery_events: list[Event] = list(rejection_events)
            transcript_messages: list[Message] = []
            for offset, row in enumerate(rows, start=1):
                queued_message = message_from_row(row)
                delivered_cursor = transcript_cursor + offset
                delivered_message = queued_session_message_input(queued_message)
                delivery_event = event_with_runtime_payload_authority(
                    Event(
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
                            **message_queue.source_audit_payload(
                                dict(row), accepted_events.get(row["accepted_event_id"])
                            ),
                            "accepted_run_epoch": queued_message.accepted_run_epoch,
                            "accepted_transcript_cursor": (
                                queued_message.accepted_transcript_cursor
                            ),
                            SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: (
                                session_messages_input_contract_evidence(
                                    (delivered_message,),
                                    message_start_index=delivered_cursor - 1,
                                    redactions_applied=False,
                                    structured_output_requested=False,
                                )
                            ),
                        },
                    ),
                    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
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
                transcript_messages.append(delivered_message)
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
                        sqlite_records.format_datetime(delivered_at),
                        updated.queue_id,
                    ),
                )
            delivery_events.sort(key=lambda event: event.payload["ordering_key"])
            persisted_events = [
                *(
                    [interaction_started_event]
                    if updated_messages and interaction_started_event is not None
                    else []
                ),
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
            connection.executemany(
                "INSERT INTO cayu_events (session_id, event_id, interaction_id, "
                "event_type, timestamp, "
                "agent_name, environment_name, workflow_name, tool_name, payload_json, "
                "pending_action_lookup_key, pending_action_projection_json, "
                "pending_action_projection_bytes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                event_rows,
            )
            event_delivery_ops.enqueue_persisted_event_side_effects(
                connection,
                session_id,
                persisted_events,
            )
            touch_activity(connection, session_id, delivered_at)
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
            has_more = reject_only_more if reject_only else remaining is not None
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
                    has_more,
                    (
                        None
                        if interaction_started_event is None
                        else sqlite_records.json_dumps(
                            interaction_started_event.model_dump(mode="json")
                        )
                    ),
                    sqlite_records.json_dumps([message.queue_id for message in updated_messages]),
                    sqlite_records.json_dumps(
                        [event.model_dump(mode="json") for event in persisted_events]
                    ),
                    sqlite_records.format_datetime(delivered_at),
                ),
            )
            connection.execute(
                "UPDATE cayu_session_message_deliveries SET reject_only = ? WHERE delivery_id = ?",
                (reject_only, delivery_id),
            )
            if rebound_checkpoint is not None:
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
                            pending_action_source_bytes = excluded.pending_action_source_bytes,
                            pending_action_tool_call_count =
                                excluded.pending_action_tool_call_count,
                            pending_action_flags = excluded.pending_action_flags,
                            pending_action_metrics_ready = excluded.pending_action_metrics_ready
                        """,
                    sqlite_records.checkpoint_row_values(
                        session_id,
                        rebound_checkpoint,
                        delivered_at,
                    ),
                )
            connection.commit()
            return SessionMessageDeliveryBatch(
                messages=tuple(updated_messages),
                events=tuple(persisted_events),
                delivery_id=delivery_id,
                interaction_id=interaction_id,
                eligible_through=boundary,
                has_more=has_more,
                active_invocation_profile=(
                    None
                    if not updated_messages or profile_handoff is None
                    else profile_handoff.target_active_profile
                ),
            )
        except Exception:
            connection.rollback()
            raise

    return await run_write(statement)


async def repair_queued_interaction_profile_handoff(
    run_write: SQLiteOperationRunner,
    session_id: str,
    *,
    interaction_started_event: Event,
    profile_handoff: QueuedInteractionProfileHandoff,
    load_session: Callable[[str], Session | None],
    store_now: Callable[[], datetime],
    decode_stage_record: Callable[[str], dict[str, Any]],
    load_checkpoint: Callable[[sqlite3.Connection, str], dict[str, Any] | None],
) -> ActiveInvocationExecutionProfile:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_started_event, profile_handoff = (
        _copy_historical_queued_interaction_profile_handoff(
            session_id,
            interaction_started_event,
            profile_handoff,
        )
    )
    target = profile_handoff.target_active_profile

    def statement(connection: sqlite3.Connection) -> ActiveInvocationExecutionProfile:
        try:
            connection.execute("BEGIN IMMEDIATE")
            loaded = load_session(session_id)
            if loaded is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, loaded)
            delivery_row = connection.execute(
                "SELECT session_id, interaction_id, interaction_started_event_json, "
                "queue_ids_json FROM cayu_session_message_deliveries "
                "WHERE delivery_id = ?",
                (target.interaction_id,),
            ).fetchone()
            stored_started_event = (
                None
                if delivery_row is None or delivery_row["interaction_started_event_json"] is None
                else Event.model_validate_json(delivery_row["interaction_started_event_json"])
            )
            if (
                delivery_row is None
                or delivery_row["session_id"] != session_id
                or delivery_row["interaction_id"] != target.interaction_id
                or stored_started_event != interaction_started_event
                or not json.loads(delivery_row["queue_ids_json"])
            ):
                raise SessionRunFenced(
                    "Historical queued interaction handoff lacks its exact delivery receipt."
                )
            receipt_row = connection.execute(
                "SELECT record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key = ?",
                (
                    session_id,
                    _interaction_transition_storage_key(
                        profile_handoff.predecessor_settlement_event_id
                    ),
                ),
            ).fetchone()
            if receipt_row is None:
                raise SessionRunFenced(
                    "Historical queued interaction handoff lost its predecessor settlement."
                )
            active_row = connection.execute(
                "SELECT record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key = ?",
                (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
            ).fetchone()
            stage_records: dict[str, Any] = {}
            if active_row is not None:
                active_record = decode_stage_record(active_row["record_json"])
                marker = _reconstruct_active_model_completion_stage_record(
                    active_record,
                    session_id=session_id,
                )
                _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
                    session_id, marker.stage_id
                )
                dispatch_key = _model_completion_stage_dispatch_storage_key(marker.stage_id)
                rows = connection.execute(
                    "SELECT idempotency_key, record_json FROM cayu_session_operations "
                    "WHERE session_id = ? AND idempotency_key IN (?, ?, ?)",
                    (session_id, preparation_key, terminal_key, dispatch_key),
                ).fetchall()
                stage_records = {
                    row["idempotency_key"]: decode_stage_record(row["record_json"]) for row in rows
                }
                stage_records[MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY] = active_record
            active_model_stage, stage_dispatch = _historical_queued_handoff_stage_from_records(
                session_id,
                stage_records,
            )
            repaired_checkpoint = _checkpoint_after_queued_interaction_profile_handoff(
                loaded,
                load_checkpoint(connection, session_id),
                profile_handoff,
                settlement_record=copy_durable_json_object(
                    json.loads(receipt_row["record_json"]),
                    "interaction transition receipt",
                ),
                replayed_delivery=True,
                active_model_stage=active_model_stage,
                stage_dispatch=stage_dispatch,
            )
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
                        pending_action_source_bytes = excluded.pending_action_source_bytes,
                        pending_action_tool_call_count = excluded.pending_action_tool_call_count,
                        pending_action_flags = excluded.pending_action_flags,
                        pending_action_metrics_ready = excluded.pending_action_metrics_ready
                    """,
                sqlite_records.checkpoint_row_values(
                    session_id,
                    repaired_checkpoint,
                    store_now(),
                ),
            )
            connection.commit()
            return target.model_copy(deep=True)
        except Exception:
            connection.rollback()
            raise

    return await run_write(statement)
