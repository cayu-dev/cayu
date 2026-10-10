"""Complete PostgreSQL queued-message persistence operations.

Native capabilities retain authorization, replay and publication within each
operation's transaction. Queue readers share that same authoritative snapshot.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import datetime
from typing import Any, Protocol
from uuid import uuid4

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
from cayu.storage import _postgres_event_delivery as event_delivery_ops
from cayu.storage import _postgres_support as pg_support
from cayu.storage import _postgres_transcript as transcript_ops
from cayu.storage._postgres_transcript import AuthorityRegistrar, PostgresConnection


class EventAppender(Protocol):
    def __call__(
        self, cur: Any, session_id: str, events: Sequence[Event], *, expected_run_epoch: int | None
    ) -> Awaitable[None]: ...


class SteeringGuard(Protocol):
    def __call__(
        self, cur: Any, session: Session, *, allow_completed_interaction: bool = False
    ) -> Awaitable[None]: ...


QUEUE_COLUMNS = (
    "ordering_key, queue_id, session_id, idempotency_key, content, delivery_mode, status, "
    "requested_by, accepted_run_epoch, accepted_transcript_cursor, accepted_event_id, "
    "accepted_at, delivered_run_epoch, delivered_transcript_cursor, delivered_event_id, "
    "delivered_at, message_json, conditions_json, terminal_json"
)


def _raw_row(row: Any) -> dict[str, Any]:
    return dict(zip(QUEUE_COLUMNS.split(", "), row, strict=True))


def message_from_row(row: Any) -> SessionQueuedMessage:
    requested_by = row[7]
    return SessionQueuedMessage(
        ordering_key=row[0],
        queue_id=row[1],
        session_id=row[2],
        idempotency_key=row[3],
        conditions=SessionMessageConditions.model_validate(
            {} if row[17] is None else pg_support._json_obj(row[17])
        ),
        content=row[4],
        message=(
            None if row[16] is None else Message.model_validate(pg_support._json_obj(row[16]))
        ),
        delivery_mode=row[5],
        status=row[6],
        requested_by=(
            None
            if requested_by is None
            else ResolutionActor.model_validate(pg_support._json_obj(requested_by))
        ),
        accepted_run_epoch=row[8],
        accepted_transcript_cursor=row[9],
        accepted_event_id=row[10],
        accepted_at=row[11],
        delivered_run_epoch=row[12],
        delivered_transcript_cursor=row[13],
        delivered_event_id=row[14],
        delivered_at=row[15],
    )


async def _source_snapshot(
    cur: Any,
    session: Session,
    *,
    include_transcript_digest: bool,
    include_checkpoint_digest: bool,
    load_checkpoint: Callable[[Any, str], Awaitable[dict[str, Any] | None]],
) -> SessionMessageSource:
    cursor = await transcript_ops.transcript_cursor(cur, session.id)
    transcript_digest = None
    if include_transcript_digest:
        hasher = message_queue.SourceTranscriptHasher(cursor)
        after = 0
        while True:
            await cur.execute(
                "SELECT session_order, message FROM cayu_transcript_messages "
                "WHERE session_id = %s AND session_order > %s ORDER BY session_order LIMIT 100",
                (session.id, after),
            )
            rows = await cur.fetchall()
            if not rows:
                break
            for row in rows:
                hasher.add(row[0] - 1, Message.model_validate(pg_support._json_obj(row[1])))
            after = rows[-1][0]
        transcript_digest = hasher.hexdigest()
    checkpoint = None
    if include_checkpoint_digest:
        checkpoint = await load_checkpoint(cur, session.id)
    return message_queue.source_snapshot(
        session,
        cursor,
        transcript_sha256=transcript_digest,
        checkpoint=checkpoint,
        include_checkpoint_digest=include_checkpoint_digest,
    )


async def _read_session(
    cur: Any, session_id: str, *, load_labels: Callable[[Any, str], Awaitable[dict[str, str]]]
) -> Session:
    # Inspection must also work on read-only connections. One MVCC snapshot
    # binds session identity, queue pages, transcript and checkpoint reads.
    await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
    await cur.execute(
        f"SELECT {pg_support.SESSION_COLUMNS} FROM cayu_sessions WHERE id = %s",
        (session_id,),
    )
    row = await cur.fetchone()
    if row is None:
        require_resource_session(None)
        raise KeyError("Session not found.")
    return pg_support.session_from_row(row, labels=await load_labels(cur, session_id))


async def snapshot_session_message_source(
    connect: PostgresConnection,
    session_id: str,
    *,
    include_transcript_digest: bool = False,
    include_checkpoint_digest: bool = False,
    expected_authorized_session_instance_id: str | None = None,
    ensure_ready: Callable[[], Awaitable[None]],
    load_checkpoint: Callable[[Any, str], Awaitable[dict[str, Any] | None]],
    load_labels: Callable[[Any, str], Awaitable[dict[str, str]]],
) -> SessionMessageSource:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(include_transcript_digest) is not bool or type(include_checkpoint_digest) is not bool:
        raise TypeError("Snapshot digest flags must be bool.")
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        session = await _read_session(cur, session_id, load_labels=load_labels)
        require_resource_session(session, "read")
        message_queue.require_authorized_session_instance(
            session, expected_authorized_session_instance_id
        )
        return await _source_snapshot(
            cur,
            session,
            include_transcript_digest=include_transcript_digest,
            include_checkpoint_digest=include_checkpoint_digest,
            load_checkpoint=load_checkpoint,
        )


async def _raw_bounded(cur: Any, session_id: str, queue_id: str) -> dict[str, Any]:
    columns = QUEUE_COLUMNS.split(", ")
    projection = ", ".join(
        f"CASE WHEN octet_length({name}::text) <= {SESSION_MESSAGE_QUEUE_STORAGE_VALUE_MAX_BYTES} "
        f"THEN {name} END AS {name}"
        for name in columns
    )
    hashes = ", ".join(
        f"CASE WHEN octet_length({name}::text) > {SESSION_MESSAGE_QUEUE_STORAGE_VALUE_MAX_BYTES} "
        f"THEN encode(sha256(convert_to({name}::text, 'UTF8')), 'hex') END"
        for name in columns
    )
    await cur.execute(
        f"SELECT {projection}, {hashes} FROM cayu_session_message_queue "
        "WHERE session_id = %s AND queue_id = %s",
        (session_id, queue_id),
    )
    row = await cur.fetchone()
    if row is None:
        raise SessionMessageConflict()
    raw: dict[str, Any] = dict(zip(columns, row[: len(columns)], strict=True))
    for name, digest in zip(columns, row[len(columns) :], strict=True):
        if digest is not None:
            raw[name] = message_queue.OversizedStorageValue(digest)
    return raw


async def _acceptance_events(
    cur: Any, session_id: str, rows: list[dict[str, Any]], *, quarantine_queue_id: str | None = None
) -> dict[str, Event]:
    """Batch-read bounded audit projections of canonical events under the session lock."""
    ids = [row["accepted_event_id"] for row in rows]
    if quarantine_queue_id is None and any(
        type(event_id) is not str or len(event_id) > 512 for event_id in ids
    ):
        raise SessionMessageConflict()
    if not ids:
        return {}
    projection = (
        "jsonb_build_object('queue_id', event->'payload'->'queue_id', "
        "'source', event->'payload'->'source')"
    )
    predicate = (
        "event_id = ANY(%s)"
        if quarantine_queue_id is None
        else "event #>> '{payload,queue_id}' = %s LIMIT 2"
    )
    await cur.execute(
        f"SELECT event_id, CASE WHEN octet_length(({projection})::text) <= 32768 "
        f"THEN {projection} END FROM cayu_events WHERE session_id = %s "
        f"AND event_type = 'session.message.queued' AND {predicate}",
        (session_id, ids if quarantine_queue_id is None else quarantine_queue_id),
    )
    events = await cur.fetchall()
    if quarantine_queue_id is not None and len(events) != 1:
        raise SessionMessageConflict()
    if any(row[1] is None for row in events):
        raise SessionMessageConflict()
    return {
        row[0]: Event(
            id=row[0],
            type=EventType.SESSION_MESSAGE_QUEUED,
            session_id=session_id,
            payload=pg_support._json_obj(row[1]),
        )
        for row in events
    }


async def inspect_session_messages(
    connect: PostgresConnection,
    query: SessionMessageQuery,
    *,
    expected_authorized_session_instance_id: str | None = None,
    ensure_ready: Callable[[], Awaitable[None]],
    load_labels: Callable[[Any, str], Awaitable[dict[str, str]]],
) -> SessionMessageInspection:
    query = message_queue.copy_inspection_query(query)
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        session = await _read_session(cur, query.session_id, load_labels=load_labels)
        require_resource_session(session, "read")
        message_queue.require_authorized_session_instance(
            session, expected_authorized_session_instance_id
        )
        maximum = 0
        if query.cursor is None:
            await cur.execute(
                "SELECT COALESCE(MAX(ordering_key), 0) FROM cayu_session_message_queue "
                "WHERE session_id = %s",
                (session.id,),
            )
            maximum = (await cur.fetchone())[0]
        boundary = message_queue.inspection_boundary(session, query.cursor, maximum)
        await cur.execute(
            "WITH ordered AS (SELECT queue_id, ordering_key, "
            "CASE delivery_mode WHEN 'next_turn' THEN 0 WHEN 'on_idle' THEN 1 ELSE 2 END "
            "AS priority FROM cayu_session_message_queue "
            "WHERE session_id = %s AND ordering_key <= %s) "
            "SELECT queue_id, ordering_key, priority FROM ordered "
            "WHERE (priority, ordering_key) > (%s, %s) "
            "ORDER BY priority, ordering_key LIMIT %s",
            (
                session.id,
                boundary.through_ordering_key,
                boundary.after_priority,
                boundary.after_ordering_key,
                query.limit + 1,
            ),
        )
        rows = await cur.fetchall()
        raw_rows = [await _raw_bounded(cur, session.id, row[0]) for row in rows[: query.limit]]
        records = tuple(
            message_queue.inspect_record(raw, lambda raw=raw: message_from_row(tuple(raw.values())))
            for raw in raw_rows
        )
        return SessionMessageInspection(
            session_id=session.id,
            session_instance_id=session.instance_id,
            records=records,
            next_cursor=(
                message_queue.inspection_next_cursor(
                    boundary,
                    rows[query.limit - 1][2],
                    records[-1].ordering_key,
                )
                if len(rows) > query.limit
                else None
            ),
        )


async def apply_session_message_action(
    connect: PostgresConnection,
    request: SessionMessageActionRequest,
    *,
    ensure_ready: Callable[[], Awaitable[None]],
    load_session: Callable[[Any, str], Awaitable[Session | None]],
    store_now: Callable[[Any], Awaitable[datetime]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    append_events: EventAppender,
) -> SessionMessageActionResult:
    request = SessionMessageActionRequest(**message_queue.action_material(request))
    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                session = await load_session(cur, request.session_id)
                require_resource_session(session, "modify")
                if session is None or session.instance_id != request.session_instance_id:
                    raise SessionMessageConflict()
                await cur.execute(
                    "SELECT queue_id FROM cayu_session_message_queue "
                    "WHERE session_id = %s AND queue_id = %s FOR UPDATE",
                    (session.id, request.queue_id),
                )
                row = await cur.fetchone()
                if row is None:
                    raise SessionMessageConflict()
                raw = await _raw_bounded(cur, session.id, request.queue_id)
                accepted_events = await _acceptance_events(
                    cur,
                    session.id,
                    [raw],
                    quarantine_queue_id=request.queue_id
                    if request.action == "quarantine"
                    else None,
                )
                accepted_event = (
                    next(iter(accepted_events.values()))
                    if request.action == "quarantine"
                    else accepted_events.get(raw["accepted_event_id"])
                )
                replay = message_queue.replay_action(raw, request, accepted_event=accepted_event)
                record = message_queue.inspect_record(
                    raw, lambda: message_from_row(tuple(raw.values()))
                )
                if replay is not None:
                    await conn.commit()
                    return SessionMessageActionResult(record=record, event=replay, replayed=True)
                for owner in await closure_owners(cur, (session.id,)):
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
                await cur.execute(
                    "SELECT 1 FROM cayu_session_message_deliveries "
                    "WHERE session_id = %s AND queue_ids @> %s::jsonb LIMIT 1",
                    (session.id, pg_support._dumps([request.queue_id])),
                )
                if await cur.fetchone() is not None or (
                    request.action == "withdraw" and record.validity != "valid"
                ):
                    raise SessionMessageConflict()
                status = SessionMessageQueueStatus(
                    "withdrawn" if request.action == "withdraw" else "quarantined"
                )
                event = message_queue.terminal_event(
                    session,
                    raw,
                    status,
                    await store_now(cur),
                    actor=request.requested_by,
                    accepted_event=accepted_event,
                )
                await cur.execute(
                    "UPDATE cayu_session_message_queue SET status = %s, terminal_json = %s "
                    "WHERE session_id = %s AND queue_id = %s AND status = 'queued'",
                    (
                        str(status),
                        pg_support._dumps(message_queue.terminal_receipt(status, event, request)),
                        session.id,
                        request.queue_id,
                    ),
                )
                await append_events(cur, session.id, [event], expected_run_epoch=None)
                updated = await _raw_bounded(cur, session.id, request.queue_id)
                result = SessionMessageActionResult(
                    record=message_queue.inspect_record(
                        updated,
                        lambda: message_from_row(tuple(updated.values())),
                    ),
                    event=event,
                )
            await conn.commit()
            return result
        except BaseException:
            await conn.rollback()
            raise


async def enqueue_session_message(
    connect: PostgresConnection,
    request: EnqueueSessionMessageRequest,
    *,
    expected_authorized_target_instance_id: str | None = None,
    ensure_ready: Callable[[], Awaitable[None]],
    load_session: Callable[[Any, str], Awaitable[Session | None]],
    store_now: Callable[[Any], Awaitable[datetime]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    load_checkpoint: Callable[[Any, str], Awaitable[dict[str, Any] | None]],
    register_event_authorities: Callable[
        [Any, str, list[Event] | tuple[Event, ...]], Awaitable[None]
    ],
) -> EnqueueSessionMessageResult:
    from cayu.sessions.pending_actions import pending_action_event_storage_values

    request = copy_enqueue_session_message_request(request)
    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                # A->B and B->A provenance admissions acquire the same lock order.
                source_id = (
                    None
                    if request.conditions.source is None
                    else request.conditions.source.session_id
                )
                locked = {
                    sid: await load_session(cur, sid)
                    for sid in sorted({request.session_id} | ({source_id} if source_id else set()))
                }
                loaded = locked[request.session_id]
                require_resource_session(loaded, "modify")
                if loaded is None:
                    raise KeyError(f"Session not found: {request.session_id}")
                if expected_authorized_target_instance_id is not None and (
                    type(expected_authorized_target_instance_id) is not str
                    or loaded.instance_id != expected_authorized_target_instance_id
                ):
                    raise SessionMessageConflict()
                if request.conditions.source is not None:
                    source = locked[request.conditions.source.session_id]
                    require_resource_session(source, "read")
                    if (
                        source is None
                        or source.instance_id != request.conditions.source.session_instance_id
                    ):
                        raise SessionMessageConflict()
                await cur.execute(
                    f"SELECT {QUEUE_COLUMNS} "
                    "FROM cayu_session_message_queue "
                    "WHERE session_id = %s AND idempotency_key = %s",
                    (request.session_id, request.idempotency_key),
                )
                existing_row = await cur.fetchone()
                if existing_row is not None:
                    existing = message_from_row(existing_row)
                    _validate_equivalent_queued_session_message(existing, request)
                    await cur.execute(
                        "SELECT event FROM cayu_events WHERE session_id = %s AND event_id = %s",
                        (request.session_id, existing.accepted_event_id),
                    )
                    event_row = await cur.fetchone()
                    if event_row is None:
                        raise RuntimeError(
                            "Queued session message is missing its durable acceptance event."
                        )
                    await conn.commit()
                    return EnqueueSessionMessageResult(
                        message=existing,
                        event=Event(**pg_support._json_obj(event_row[0])),
                        replayed=True,
                    )
                for owner in await closure_owners(cur, (request.session_id,)):
                    _check_closure_lineage_owner(owner, (request.session_id,))
                checkpoint = await load_checkpoint(cur, request.session_id)
                message_queue.require_open_admission(loaded.status, checkpoint)
                if request.conditions.source is not None:
                    expected_source = request.conditions.source
                    source = locked[expected_source.session_id]
                    require_resource_session(source, "read")
                    if (
                        source is None
                        or await _source_snapshot(
                            cur,
                            source,
                            include_transcript_digest=expected_source.transcript_sha256 is not None,
                            include_checkpoint_digest=expected_source.checkpoint_sha256 is not None,
                            load_checkpoint=load_checkpoint,
                        )
                        != expected_source
                    ):
                        raise SessionMessageConflict()
                transcript_cursor = await transcript_ops.transcript_cursor(cur, request.session_id)
                accepted_at = await store_now(cur)
                queue_id = str(uuid4())
                accepted_event_id = str(uuid4())
                await cur.execute(
                    """
                        INSERT INTO cayu_session_message_queue (
                            queue_id, session_id, idempotency_key, content, message_json,
                            delivery_mode, status, requested_by,
                            accepted_run_epoch, accepted_transcript_cursor,
                            accepted_event_id, accepted_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, 'queued', %s, %s, %s, %s, %s)
                        RETURNING ordering_key
                        """,
                    (
                        queue_id,
                        request.session_id,
                        request.idempotency_key,
                        request.content,
                        (
                            None
                            if request.message is None
                            else pg_support._dumps(request.message.model_dump(mode="json"))
                        ),
                        str(request.delivery_mode),
                        (
                            None
                            if request.requested_by is None
                            else pg_support._dumps(resolution_actor_payload(request.requested_by))
                        ),
                        loaded.run_epoch,
                        transcript_cursor,
                        accepted_event_id,
                        accepted_at,
                    ),
                )
                ordering_row = await cur.fetchone()
                if ordering_row is None:
                    raise RuntimeError("Postgres queue insert did not return an ordering key.")
                ordering_key = ordering_row[0]
                await cur.execute(
                    "UPDATE cayu_session_message_queue SET conditions_json = %s WHERE queue_id = %s",
                    (
                        pg_support._dumps(request.conditions.model_dump(mode="json")),
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
                await cur.execute(
                    "UPDATE cayu_sessions SET event_seq = event_seq + 1, "
                    "last_activity_at = %s WHERE id = %s RETURNING event_seq",
                    (accepted_at, request.session_id),
                )
                event_order_row = await cur.fetchone()
                if event_order_row is None:
                    raise KeyError(f"Session not found: {request.session_id}")
                lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                    accepted_event
                )
                await register_event_authorities(
                    cur,
                    request.session_id,
                    [accepted_event],
                )
                await cur.execute(
                    """
                        INSERT INTO cayu_events (
                            session_id, session_order, event_id, interaction_id,
                            event_type, timestamp,
                            agent_name, environment_name, workflow_name, tool_name,
                            payload, event, pending_action_lookup_key,
                            pending_action_projection, pending_action_projection_bytes
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                    (
                        request.session_id,
                        event_order_row[0],
                        accepted_event.id,
                        accepted_event.interaction_id,
                        str(accepted_event.type),
                        accepted_event.timestamp,
                        accepted_event.agent_name,
                        accepted_event.environment_name,
                        accepted_event.workflow_name,
                        accepted_event.tool_name,
                        pg_support._dumps(accepted_event.payload),
                        pg_support._dumps(accepted_event.model_dump(mode="json")),
                        lookup_key,
                        projection,
                        projection_bytes,
                    ),
                )
                await event_delivery_ops.enqueue_persisted_event_side_effects(
                    cur,
                    request.session_id,
                    [accepted_event],
                )
                await cur.execute(
                    f"SELECT {QUEUE_COLUMNS} FROM cayu_session_message_queue WHERE queue_id = %s",
                    (queue_id,),
                )
                stored_row = await cur.fetchone()
                if stored_row is None:
                    raise RuntimeError("Queued session message disappeared after acceptance.")
            await conn.commit()
            return EnqueueSessionMessageResult(
                message=message_from_row(stored_row),
                event=accepted_event,
            )
        except Exception:
            await conn.rollback()
            raise


async def deliver_queued_session_messages(
    connect: PostgresConnection,
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
    ensure_ready: Callable[[], Awaitable[None]],
    load_session: Callable[[Any, str], Awaitable[Session | None]],
    store_now: Callable[[Any], Awaitable[datetime]],
    load_checkpoint: Callable[[Any, str], Awaitable[dict[str, Any] | None]],
    register_event_authorities: Callable[
        [Any, str, list[Event] | tuple[Event, ...]], Awaitable[None]
    ],
    register_authorities: AuthorityRegistrar,
    reject_steering: SteeringGuard,
    upsert_checkpoint: Callable[[Any, str, dict[str, Any], datetime], Awaitable[None]],
    decode_stage_record: Callable[[Any], dict[str, Any]],
) -> SessionMessageDeliveryBatch:
    from cayu.sessions.pending_actions import pending_action_event_storage_values

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
    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                loaded = await load_session(cur, session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, loaded)
                await cur.execute(
                    """
                        SELECT session_id, interaction_id, include_on_idle,
                               requested_eligible_through, eligible_through,
                               batch_limit, has_more, interaction_started_event,
                               queue_ids, events, reject_only
                        FROM cayu_session_message_deliveries
                        WHERE delivery_id = %s
                        """,
                    (delivery_id,),
                )
                delivery_row = await cur.fetchone()
                if delivery_row is not None:
                    stored_started_event = (
                        None
                        if delivery_row[7] is None
                        else Event(**pg_support._json_obj(delivery_row[7]))
                    )
                    if (
                        delivery_row[0] != session_id
                        or delivery_row[10] != reject_only
                        or delivery_row[1] != interaction_id
                        or delivery_row[2] != include_on_idle
                        or delivery_row[3] != eligible_through
                        or delivery_row[5] != limit
                        or stored_started_event != interaction_started_event
                    ):
                        raise ValueError(
                            "delivery_id was already used for a different queue delivery."
                        )
                    queue_ids = list(delivery_row[8])
                    queued_by_id: dict[str, SessionQueuedMessage] = {}
                    replayed_events = tuple(
                        Event(**pg_support._json_obj(event)) for event in delivery_row[9]
                    )
                    if queue_ids:
                        await cur.execute(
                            f"SELECT {QUEUE_COLUMNS} "
                            "FROM cayu_session_message_queue "
                            "WHERE queue_id = ANY(%s)",
                            (queue_ids,),
                        )
                        queued_by_id = {
                            message.queue_id: message
                            for message in (message_from_row(row) for row in await cur.fetchall())
                        }
                    if len(queued_by_id) != len(queue_ids):
                        raise RuntimeError("Queue delivery replay lost a delivered message.")
                    if queue_ids and profile_handoff is not None:
                        await cur.execute(
                            "SELECT record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s",
                            (
                                session_id,
                                _interaction_transition_storage_key(
                                    profile_handoff.predecessor_settlement_event_id
                                ),
                            ),
                        )
                        receipt_row = await cur.fetchone()
                        if receipt_row is None:
                            raise SessionRunFenced(
                                "Queued interaction handoff lost its predecessor "
                                "settlement receipt."
                            )
                        active_model_stage = None
                        stage_dispatch = None
                        await cur.execute(
                            "SELECT record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s",
                            (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                        )
                        active_row = await cur.fetchone()
                        if active_row is not None:
                            active_record = decode_stage_record(active_row[0])
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
                            dispatch_key = _model_completion_stage_dispatch_storage_key(
                                marker.stage_id
                            )
                            await cur.execute(
                                "SELECT idempotency_key, record "
                                "FROM cayu_session_operations WHERE session_id = %s "
                                "AND idempotency_key = ANY(%s)",
                                (
                                    session_id,
                                    [preparation_key, terminal_key, dispatch_key],
                                ),
                            )
                            stage_records = {
                                row[0]: decode_stage_record(row[1]) for row in await cur.fetchall()
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
                            await load_checkpoint(cur, session_id),
                            profile_handoff,
                            settlement_record=pg_support._json_obj(receipt_row[0]),
                            replayed_delivery=True,
                            active_model_stage=active_model_stage,
                            stage_dispatch=stage_dispatch,
                        )
                        await upsert_checkpoint(
                            cur,
                            session_id,
                            repaired_checkpoint,
                            await store_now(cur),
                        )
                    await conn.commit()
                    return SessionMessageDeliveryBatch(
                        messages=tuple(queued_by_id[queue_id] for queue_id in queue_ids),
                        events=replayed_events,
                        delivery_id=delivery_id,
                        interaction_id=interaction_id,
                        eligible_through=delivery_row[4],
                        has_more=delivery_row[6],
                        replayed=True,
                        active_invocation_profile=(
                            None
                            if not queue_ids or profile_handoff is None
                            else profile_handoff.target_active_profile
                        ),
                    )
                if loaded.status != SessionStatus.RUNNING:
                    raise SessionStatusConflict(
                        "Queued session messages may be delivered only while running."
                    )
                boundary = eligible_through
                if boundary is None:
                    # ``ordering_key`` is a global identity primary key. Its
                    # global maximum is an end-of-index lookup and still
                    # fences every message this session can currently
                    # contain; the locked session row serializes enqueues for
                    # this session until the transaction completes.
                    await cur.execute(
                        "SELECT COALESCE(MAX(ordering_key), 0) FROM cayu_session_message_queue"
                    )
                    boundary_row = await cur.fetchone()
                    boundary = boundary_row[0] if boundary_row is not None else 0
                await cur.execute(
                    f"SELECT {QUEUE_COLUMNS} "
                    "FROM cayu_session_message_queue WHERE session_id = %s "
                    "AND status = 'queued' AND delivery_mode = 'next_turn' "
                    "AND ordering_key <= %s ORDER BY ordering_key ASC LIMIT %s FOR UPDATE",
                    (session_id, boundary, limit),
                )
                rows = await cur.fetchall()
                if not rows and include_on_idle:
                    await cur.execute(
                        f"SELECT {QUEUE_COLUMNS} "
                        "FROM cayu_session_message_queue WHERE session_id = %s "
                        "AND status = 'queued' AND delivery_mode = 'on_idle' "
                        "AND ordering_key <= %s ORDER BY ordering_key ASC LIMIT %s FOR UPDATE",
                        (session_id, boundary, limit),
                    )
                    rows = await cur.fetchall()
                reject_only_more = False
                if reject_only:
                    # Eligible rows remain pending. Scan in bounded pages so they cannot hide
                    # an expired record behind the first delivery-sized prefix.
                    rows = []
                    scan_now = await store_now(cur)
                    scan_cursor = await transcript_ops.transcript_cursor(cur, session_id)
                    for mode in ("next_turn", "on_idle") if include_on_idle else ("next_turn",):
                        after = 0
                        while len(rows) < limit + 1:
                            await cur.execute(
                                f"SELECT {QUEUE_COLUMNS} FROM cayu_session_message_queue "
                                "WHERE session_id = %s AND status = 'queued' AND delivery_mode = %s "
                                "AND ordering_key > %s AND ordering_key <= %s "
                                "ORDER BY ordering_key LIMIT 100 FOR UPDATE",
                                (session_id, mode, after, boundary),
                            )
                            page = await cur.fetchall()
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
                            after = page[-1][0]
                        if len(rows) == limit + 1:
                            break
                    rows = rows[:limit]
                if not rows:
                    await cur.execute(
                        """
                            INSERT INTO cayu_session_message_deliveries (
                                delivery_id, session_id, interaction_id,
                                include_on_idle, requested_eligible_through,
                                eligible_through, batch_limit, has_more,
                                interaction_started_event, queue_ids, events,
                                created_at
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s, %s, FALSE,
                                %s, '[]'::jsonb, '[]'::jsonb, %s
                            )
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
                                else pg_support._dumps(
                                    interaction_started_event.model_dump(mode="json")
                                )
                            ),
                            await store_now(cur),
                        ),
                    )
                    await cur.execute(
                        "UPDATE cayu_session_message_deliveries SET reject_only = %s WHERE delivery_id = %s",
                        (reject_only, delivery_id),
                    )
                    await conn.commit()
                    return SessionMessageDeliveryBatch(
                        delivery_id=delivery_id,
                        interaction_id=interaction_id,
                        eligible_through=boundary,
                        has_more=False,
                    )
                transcript_cursor = await transcript_ops.transcript_cursor(cur, session_id)
                delivered_at = scan_now if reject_only else await store_now(cur)
                accepted_events = await _acceptance_events(
                    cur,
                    session_id,
                    [_raw_row(row) for row in rows],
                )
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
                        _raw_row(row),
                        rejection,
                        delivered_at,
                        accepted_event=accepted_events.get(row[10]),
                        actor=queued.requested_by,
                        interaction_id=interaction_id,
                    )
                    rejection_events.append(event)
                    await cur.execute(
                        "UPDATE cayu_session_message_queue SET status = %s, terminal_json = %s "
                        "WHERE queue_id = %s AND status = 'queued'",
                        (
                            str(rejection),
                            pg_support._dumps(message_queue.terminal_receipt(rejection, event)),
                            queued.queue_id,
                        ),
                    )
                rows = deliverable_rows
                rebound_checkpoint: dict[str, Any] | None = None
                if rows and profile_handoff is not None:
                    await cur.execute(
                        "SELECT record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (
                            session_id,
                            _interaction_transition_storage_key(
                                profile_handoff.predecessor_settlement_event_id
                            ),
                        ),
                    )
                    receipt_row = await cur.fetchone()
                    if receipt_row is None:
                        raise SessionRunFenced(
                            "Queued interaction handoff lost its predecessor settlement receipt."
                        )
                    await reject_steering(cur, loaded, allow_completed_interaction=True)
                    rebound_checkpoint = _checkpoint_after_queued_interaction_profile_handoff(
                        loaded,
                        await load_checkpoint(cur, session_id),
                        profile_handoff,
                        settlement_record=pg_support._json_obj(receipt_row[0]),
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
                                    _raw_row(row), accepted_events.get(row[10])
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
                    updated_messages.append(
                        queued_message.model_copy(
                            update={
                                "status": SessionMessageQueueStatus.DELIVERED,
                                "delivered_run_epoch": loaded.run_epoch,
                                "delivered_transcript_cursor": delivered_cursor,
                                "delivered_event_id": delivery_event.id,
                                "delivered_at": delivered_at,
                            },
                            deep=True,
                        )
                    )
                    delivery_events.append(delivery_event)
                    transcript_messages.append(delivered_message)
                await cur.executemany(
                    "INSERT INTO cayu_transcript_messages "
                    "(session_id, interaction_id, message, "
                    "transcript_search_document) VALUES (%s, %s, %s, %s)",
                    [
                        (
                            session_id,
                            interaction_id,
                            pg_support._dumps(message.model_dump(mode="json")),
                            transcript_ops.index_document(session_id, message),
                        )
                        for message in transcript_messages
                    ],
                )
                await register_event_authorities(
                    cur,
                    session_id,
                    delivery_events,
                )
                await register_authorities(
                    cur,
                    session_id,
                    interaction_ids=(() if interaction_id is None else (interaction_id,)),
                )
                for updated in updated_messages:
                    await cur.execute(
                        "UPDATE cayu_session_message_queue SET status = 'delivered', "
                        "delivered_run_epoch = %s, delivered_transcript_cursor = %s, "
                        "delivered_event_id = %s, delivered_at = %s "
                        "WHERE queue_id = %s AND status = 'queued'",
                        (
                            updated.delivered_run_epoch,
                            updated.delivered_transcript_cursor,
                            updated.delivered_event_id,
                            delivered_at,
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
                await cur.execute(
                    "UPDATE cayu_sessions SET event_seq = event_seq + %s, "
                    "last_activity_at = %s WHERE id = %s RETURNING event_seq",
                    (len(persisted_events), delivered_at, session_id),
                )
                event_order_row = await cur.fetchone()
                if event_order_row is None:
                    raise KeyError(f"Session not found: {session_id}")
                next_order = event_order_row[0] - len(persisted_events)
                event_rows = []
                for event in persisted_events:
                    next_order += 1
                    lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                        event
                    )
                    event_rows.append(
                        (
                            session_id,
                            next_order,
                            event.id,
                            event.interaction_id,
                            str(event.type),
                            event.timestamp,
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
                await cur.executemany(
                    "INSERT INTO cayu_events (session_id, session_order, event_id, "
                    "interaction_id, event_type, timestamp, agent_name, "
                    "environment_name, workflow_name, "
                    "tool_name, payload, event, pending_action_lookup_key, "
                    "pending_action_projection, pending_action_projection_bytes) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    event_rows,
                )
                await event_delivery_ops.enqueue_persisted_event_side_effects(
                    cur,
                    session_id,
                    persisted_events,
                )
                mode_clause = (
                    "delivery_mode IN ('next_turn', 'on_idle')"
                    if include_on_idle
                    else "delivery_mode = 'next_turn'"
                )
                await cur.execute(
                    "SELECT 1 FROM cayu_session_message_queue WHERE session_id = %s "
                    "AND status = 'queued' AND ordering_key <= %s "
                    f"AND {mode_clause} LIMIT 1",
                    (session_id, boundary),
                )
                remaining = await cur.fetchone()
                has_more = reject_only_more if reject_only else remaining is not None
                await cur.execute(
                    """
                        INSERT INTO cayu_session_message_deliveries (
                            delivery_id, session_id, interaction_id,
                            include_on_idle, requested_eligible_through,
                            eligible_through, batch_limit, has_more,
                            interaction_started_event, queue_ids, events,
                            created_at
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s
                        )
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
                            else pg_support._dumps(
                                interaction_started_event.model_dump(mode="json")
                            )
                        ),
                        pg_support._dumps([message.queue_id for message in updated_messages]),
                        pg_support._dumps(
                            [event.model_dump(mode="json") for event in persisted_events]
                        ),
                        delivered_at,
                    ),
                )
                await cur.execute(
                    "UPDATE cayu_session_message_deliveries SET reject_only = %s WHERE delivery_id = %s",
                    (reject_only, delivery_id),
                )
                if rebound_checkpoint is not None:
                    await upsert_checkpoint(
                        cur,
                        session_id,
                        rebound_checkpoint,
                        delivered_at,
                    )
            await conn.commit()
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
            await conn.rollback()
            raise


async def repair_queued_interaction_profile_handoff(
    connect: PostgresConnection,
    session_id: str,
    *,
    interaction_started_event: Event,
    profile_handoff: QueuedInteractionProfileHandoff,
    ensure_ready: Callable[[], Awaitable[None]],
    load_session: Callable[[Any, str], Awaitable[Session | None]],
    store_now: Callable[[Any], Awaitable[datetime]],
    load_checkpoint: Callable[[Any, str], Awaitable[dict[str, Any] | None]],
    upsert_checkpoint: Callable[[Any, str, dict[str, Any], datetime], Awaitable[None]],
    decode_stage_record: Callable[[Any], dict[str, Any]],
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
    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                loaded = await load_session(cur, session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                _assert_session_run_epoch(session_id, loaded)
                await cur.execute(
                    "SELECT session_id, interaction_id, interaction_started_event, "
                    "queue_ids FROM cayu_session_message_deliveries "
                    "WHERE delivery_id = %s",
                    (target.interaction_id,),
                )
                delivery_row = await cur.fetchone()
                stored_started_event = (
                    None
                    if delivery_row is None or delivery_row[2] is None
                    else Event(**pg_support._json_obj(delivery_row[2]))
                )
                if (
                    delivery_row is None
                    or delivery_row[0] != session_id
                    or delivery_row[1] != target.interaction_id
                    or stored_started_event != interaction_started_event
                    or not list(delivery_row[3])
                ):
                    raise SessionRunFenced(
                        "Historical queued interaction handoff lacks its exact delivery receipt."
                    )
                await cur.execute(
                    "SELECT record FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = %s",
                    (
                        session_id,
                        _interaction_transition_storage_key(
                            profile_handoff.predecessor_settlement_event_id
                        ),
                    ),
                )
                receipt_row = await cur.fetchone()
                if receipt_row is None:
                    raise SessionRunFenced(
                        "Historical queued interaction handoff lost its predecessor settlement."
                    )
                await cur.execute(
                    "SELECT record FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = %s",
                    (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                )
                active_row = await cur.fetchone()
                stage_records: dict[str, Any] = {}
                if active_row is not None:
                    active_record = decode_stage_record(active_row[0])
                    marker = _reconstruct_active_model_completion_stage_record(
                        active_record,
                        session_id=session_id,
                    )
                    _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
                        session_id,
                        marker.stage_id,
                    )
                    dispatch_key = _model_completion_stage_dispatch_storage_key(marker.stage_id)
                    await cur.execute(
                        "SELECT idempotency_key, record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                        (
                            session_id,
                            [preparation_key, terminal_key, dispatch_key],
                        ),
                    )
                    stage_records = {
                        row[0]: decode_stage_record(row[1]) for row in await cur.fetchall()
                    }
                    stage_records[MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY] = active_record
                active_model_stage, stage_dispatch = _historical_queued_handoff_stage_from_records(
                    session_id,
                    stage_records,
                )
                repaired_checkpoint = _checkpoint_after_queued_interaction_profile_handoff(
                    loaded,
                    await load_checkpoint(cur, session_id),
                    profile_handoff,
                    settlement_record=pg_support._json_obj(receipt_row[0]),
                    replayed_delivery=True,
                    active_model_stage=active_model_stage,
                    stage_dispatch=stage_dispatch,
                )
                await upsert_checkpoint(
                    cur,
                    session_id,
                    repaired_checkpoint,
                    await store_now(cur),
                )
            await conn.commit()
            return target.model_copy(deep=True)
        except Exception:
            await conn.rollback()
            raise
