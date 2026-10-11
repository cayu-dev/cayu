"""Complete native postgres model-stage persistence with explicit capabilities.

The owner retains stage records, transaction boundaries, replay and cancellation.
SessionStore protocol hooks compose these operations with native collaborators.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, Protocol

from cayu.events import Event, EventType
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
from cayu.storage import _postgres_support as pg_support
from cayu.storage import _postgres_transcript as transcript_ops
from cayu.storage._postgres_session_messages import EventAppender, SteeringGuard
from cayu.storage._postgres_transcript import PostgresConnection


class StagePublisher(Protocol):
    def __call__(
        self,
        prepared: _PreparedRuntimePublication,
        *,
        _model_completion_stage: _ModelCompletionStagePromotionContext | None = None,
    ) -> Awaitable[RuntimePublicationResult]: ...


def decode_stage_record(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise SessionModelCompletionStageConflict(
            "The durable model-completion stage record is malformed."
        )
    return value


async def load_stage_records(
    session_id: str,
    preparation_storage_key: str,
    terminal_storage_key: str,
    *,
    connect: PostgresConnection,
    ensure_ready: Callable[[], Awaitable[None]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT idempotency_key, record FROM cayu_session_operations "
            "WHERE session_id = %s AND idempotency_key = ANY(%s)",
            (session_id, [preparation_storage_key, terminal_storage_key]),
        )
        records = {row[0]: decode_stage_record(row[1]) for row in await cur.fetchall()}
        if records:
            return records.get(preparation_storage_key), records.get(terminal_storage_key)
        await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")
        return None, None


async def load_settlement_record(
    session_id: str,
    settlement_storage_key: str,
    *,
    connect: PostgresConnection,
    ensure_ready: Callable[[], Awaitable[None]],
) -> dict[str, Any] | None:
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT record FROM cayu_session_operations "
            "WHERE session_id = %s AND idempotency_key = %s",
            (session_id, settlement_storage_key),
        )
        row = await cur.fetchone()
        if row is not None:
            return decode_stage_record(row[0])
        await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")
        return None


async def load_dispatch_record(
    session_id: str,
    dispatch_storage_key: str,
    *,
    connect: PostgresConnection,
    ensure_ready: Callable[[], Awaitable[None]],
) -> dict[str, Any] | None:
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT record FROM cayu_session_operations "
            "WHERE session_id = %s AND idempotency_key = %s",
            (session_id, dispatch_storage_key),
        )
        row = await cur.fetchone()
        if row is not None:
            return decode_stage_record(row[0])
        await cur.execute("SELECT 1 FROM cayu_sessions WHERE id = %s", (session_id,))
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")
        return None


async def load_active_stage_records(
    session_id: str, *, connect: PostgresConnection, ensure_ready: Callable[[], Awaitable[None]]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        await cur.execute(
            "SELECT 1 FROM cayu_sessions WHERE id = %s",
            (session_id,),
        )
        if await cur.fetchone() is None:
            raise KeyError(f"Session not found: {session_id}")
        await cur.execute(
            "SELECT record FROM cayu_session_operations "
            "WHERE session_id = %s AND idempotency_key = %s",
            (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
        )
        row = await cur.fetchone()
        if row is None:
            return None, None, None
        active_record = decode_stage_record(row[0])
        marker = _reconstruct_active_model_completion_stage_record(
            active_record,
            session_id=session_id,
        )
        _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
            session_id,
            marker.stage_id,
        )
        await cur.execute(
            "SELECT idempotency_key, record FROM cayu_session_operations "
            "WHERE session_id = %s AND idempotency_key = ANY(%s)",
            (session_id, [preparation_key, terminal_key]),
        )
        records = {
            record_row[0]: decode_stage_record(record_row[1]) for record_row in await cur.fetchall()
        }
        return (
            active_record,
            records.get(preparation_key),
            records.get(terminal_key),
        )


async def mark_dispatched(
    session_id: str,
    *,
    stage: ModelCompletionStage,
    consume_child_session_notifications: bool,
    connect: PostgresConnection,
    ensure_ready: Callable[[], Awaitable[None]],
    load_session: Callable[[Any, str], Awaitable[Session | None]],
    load_for_update: Callable[[Any, str], Awaitable[Session | None]],
    load_checkpoint: Callable[[Any, str], Awaitable[dict[str, Any] | None]],
) -> ModelCompletionStageDispatch:
    _, _, preparation_key, terminal_key = _model_completion_stage_storage_identity(
        session_id,
        stage.stage_id,
    )
    settlement_key = _model_completion_stage_settlement_storage_key(stage.stage_id)
    dispatch_key = _model_completion_stage_dispatch_storage_key(stage.stage_id)
    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                loaded = await load_for_update(cur, session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                await cur.execute(
                    "SELECT idempotency_key, record FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                    (
                        session_id,
                        [
                            MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                            preparation_key,
                            terminal_key,
                            settlement_key,
                            dispatch_key,
                        ],
                    ),
                )
                records = {row[0]: decode_stage_record(row[1]) for row in await cur.fetchall()}
                _validate_model_completion_stage_for_dispatch(
                    session=loaded,
                    checkpoint=await load_checkpoint(cur, session_id),
                    current_transcript_cursor=await transcript_ops.transcript_cursor(
                        cur, session_id
                    ),
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
                    await cur.execute(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record, updated_at) "
                        "VALUES (%s, %s, %s, %s)",
                        (
                            session_id,
                            dispatch_key,
                            pg_support._dumps(dispatch_record),
                            published_at,
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
                        # Serialize canonical occurrence selection with child
                        # status changes and event appends, both of which
                        # update the child session row.
                        await cur.execute(
                            "SELECT 1 FROM cayu_sessions WHERE id = %s FOR SHARE",
                            (claim.child_session_id,),
                        )
                        child_exists = await cur.fetchone()
                        child = await load_session(cur, claim.child_session_id)
                        event_row = None
                        if child is not None:
                            await cur.execute(
                                "SELECT sequence, event FROM cayu_events "
                                "WHERE session_id = %s AND event_type = ANY(%s) "
                                "ORDER BY sequence DESC LIMIT 1 FOR SHARE",
                                (
                                    claim.child_session_id,
                                    [
                                        str(EventType.SESSION_STARTED),
                                        str(EventType.SESSION_RESUMED),
                                        str(EventType.SESSION_FORKED),
                                        str(EventType.SESSION_COMPLETED),
                                        str(EventType.SESSION_FAILED),
                                        str(EventType.SESSION_INTERRUPTED),
                                    ],
                                ),
                            )
                            event_row = await cur.fetchone()
                        if child_exists is None or child is None or event_row is None:
                            raise SessionModelCompletionStageConflict(
                                "Child-session notification occurrence is no longer canonical."
                            )
                        event = Event(**pg_support._json_obj(event_row[1]))
                        occurrence = ChildSessionLifecycleOccurrence(
                            source=ChildSessionLifecycleOccurrenceSource.EVENT,
                            source_id=event.id,
                            source_sequence=event_row[0],
                            source_type=str(event.type),
                            occurred_at=event.timestamp,
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
                        await cur.execute(
                            "SELECT record FROM cayu_session_operations "
                            "WHERE session_id = %s AND idempotency_key = %s FOR UPDATE",
                            (session_id, consumption_key),
                        )
                        consumption_row = await cur.fetchone()
                        material = consumption.model_dump(mode="json")
                        if consumption_row is not None:
                            if not _child_session_notification_consumption_replays(
                                pg_support._json_obj(consumption_row[0]),
                                consumption,
                            ):
                                raise SessionModelCompletionStageConflict(
                                    "Child-session terminal notification was consumed by "
                                    "another stage."
                                )
                        elif consume_child_session_notifications:
                            await cur.execute(
                                "INSERT INTO cayu_session_operations "
                                "(session_id, idempotency_key, record, updated_at) "
                                "VALUES (%s, %s, %s, %s)",
                                (
                                    session_id,
                                    consumption_key,
                                    pg_support._dumps(material),
                                    published_at,
                                ),
                            )
                await cur.execute(
                    "UPDATE cayu_sessions SET updated_at = %s, last_activity_at = %s WHERE id = %s",
                    (published_at, published_at, session_id),
                )
                if cur.rowcount != 1:
                    raise KeyError(f"Session not found: {session_id}")
            await conn.commit()
            return dispatch
        except BaseException:
            await conn.rollback()
            raise


async def prepare(
    prepared: _PreparedModelCompletionStage,
    *,
    connect: PostgresConnection,
    ensure_ready: Callable[[], Awaitable[None]],
    load_for_update: Callable[[Any, str], Awaitable[Session | None]],
    load_checkpoint: Callable[[Any, str], Awaitable[dict[str, Any] | None]],
    append_events: EventAppender,
    reject_steering: SteeringGuard,
    upsert_checkpoint: Callable[[Any, str, dict[str, Any], datetime], Awaitable[None]],
) -> ModelCompletionStageResult:
    session_id = prepared.session_id
    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                loaded = await load_for_update(cur, session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                await cur.execute(
                    "SELECT idempotency_key, record FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                    (
                        session_id,
                        [
                            prepared.preparation_storage_key,
                            prepared.terminal_storage_key,
                            prepared.abandonment_storage_key,
                            *_model_failover_predecessor_storage_keys(prepared),
                        ],
                    ),
                )
                records = {row[0]: decode_stage_record(row[1]) for row in await cur.fetchall()}
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

                await cur.execute(
                    "SELECT record FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = %s",
                    (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                )
                active_row = await cur.fetchone()
                active = None
                if active_row is not None:
                    active_record = decode_stage_record(active_row[0])
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
                    await cur.execute(
                        "SELECT idempotency_key, record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                        (
                            session_id,
                            [active_preparation_key, active_terminal_key],
                        ),
                    )
                    active_records = {
                        row[0]: decode_stage_record(row[1]) for row in await cur.fetchall()
                    }
                    active = _reconstruct_active_model_completion_stage(
                        active_record,
                        active_records.get(active_preparation_key),
                        active_records.get(active_terminal_key),
                        session_id=session_id,
                    )
                await cur.execute(
                    "SELECT idempotency_key FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                    (
                        session_id,
                        [
                            prepared.winner_storage_key,
                            prepared.publication_storage_key,
                        ],
                    ),
                )
                publication_keys = {row[0] for row in await cur.fetchall()}
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
                        checkpoint=await load_checkpoint(cur, session_id),
                        current_transcript_cursor=await transcript_ops.transcript_cursor(
                            cur, session_id
                        ),
                        active=active,
                        records=records,
                        replayed=True,
                    )
                    expected_selection = _model_failover_selection_event(
                        prepared, session=loaded, prepared_at=stage.prepared_at
                    )
                    if expected_selection is not None:
                        await cur.execute(
                            "SELECT event FROM cayu_events WHERE session_id = %s AND event_id = %s",
                            (session_id, expected_selection.id),
                        )
                        event_row = await cur.fetchone()
                        _validate_model_failover_selection_replay(
                            expected_selection,
                            None
                            if event_row is None
                            else Event(**pg_support._json_obj(event_row[0])),
                        )
                    await conn.rollback()
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
                current_cursor = await transcript_ops.transcript_cursor(cur, session_id)
                if current_cursor != prepared.expected_transcript_cursor:
                    raise ValueError(
                        "Session source transcript cursor is stale: expected "
                        f"{prepared.expected_transcript_cursor}, current {current_cursor}."
                    )

                if active is None:
                    await reject_steering(cur, loaded)
                route_checkpoint = _model_failover_preparation_checkpoint(
                    prepared,
                    session=loaded,
                    checkpoint=await load_checkpoint(cur, session_id),
                    current_transcript_cursor=current_cursor,
                    active=active,
                    records=records,
                    replayed=False,
                )
                prepared_at = _next_runtime_publication_timestamp(loaded)
                if route_checkpoint is not None:
                    await upsert_checkpoint(cur, session_id, route_checkpoint, prepared_at)
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
                await cur.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record, updated_at) "
                    "VALUES (%s, %s, %s, %s)",
                    (
                        session_id,
                        prepared.preparation_storage_key,
                        pg_support._dumps(record),
                        prepared_at,
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
                    await cur.execute(
                        "SELECT record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (session_id, retry_settlement_storage_key),
                    )
                    settlement_row = await cur.fetchone()
                    _validate_model_completion_stage_for_settlement(
                        session=loaded,
                        stage=active.stage,
                        active=active,
                        request=retry_settlement_request,
                        settlement_record=(
                            None
                            if settlement_row is None
                            else decode_stage_record(settlement_row[0])
                        ),
                        winner_exists=winner_exists,
                        receipt_exists=receipt_exists,
                    )
                    retry_settlement_record = _model_completion_stage_settlement_record(
                        active.stage,
                        request=retry_settlement_request,
                        settled_at=prepared_at,
                    )
                    await cur.execute(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record, updated_at) "
                        "VALUES (%s, %s, %s, %s)",
                        (
                            session_id,
                            retry_settlement_storage_key,
                            pg_support._dumps(retry_settlement_record),
                            prepared_at,
                        ),
                    )
                await cur.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record, updated_at) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                    "record = EXCLUDED.record, updated_at = EXCLUDED.updated_at",
                    (
                        session_id,
                        MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                        pg_support._dumps(active_record),
                        prepared_at,
                    ),
                )
                await cur.execute(
                    "UPDATE cayu_sessions SET updated_at = %s, last_activity_at = %s WHERE id = %s",
                    (prepared_at, prepared_at, session_id),
                )
                if selection_event is not None:
                    await append_events(
                        cur,
                        session_id,
                        (selection_event,),
                        expected_run_epoch=prepared.expected_run_epoch,
                    )
            await conn.commit()
            return ModelCompletionStageResult(
                stage=stage,
                replayed=False,
                dispatch_authorized=True,
                prepared_events=() if selection_event is None else (selection_event,),
            )
        except BaseException:
            await conn.rollback()
            raise


async def complete(
    prepared: _PreparedModelCompletionStageTerminal,
    *,
    connect: PostgresConnection,
    ensure_ready: Callable[[], Awaitable[None]],
    load_for_update: Callable[[Any, str], Awaitable[Session | None]],
    load_checkpoint: Callable[[Any, str], Awaitable[dict[str, Any] | None]],
    store_now: Callable[[Any], Awaitable[datetime]],
) -> ModelCompletionStageResult:
    session_id = prepared.session_id
    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                loaded = await load_for_update(cur, session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                await cur.execute(
                    "SELECT idempotency_key, record FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                    (
                        session_id,
                        [
                            prepared.preparation_storage_key,
                            prepared.terminal_storage_key,
                            prepared.settlement_storage_key,
                        ],
                    ),
                )
                records = {row[0]: decode_stage_record(row[1]) for row in await cur.fetchall()}
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
                    await conn.rollback()
                    return ModelCompletionStageResult(
                        stage=stage,
                        replayed=True,
                        dispatch_authorized=False,
                    )
                if prepared.recovery_fence is not None:
                    await cur.execute(
                        "SELECT record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    )
                    active_row = await cur.fetchone()
                    await cur.execute(
                        "SELECT record FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = %s",
                        (
                            session_id,
                            _model_completion_stage_dispatch_storage_key(stage.stage_id),
                        ),
                    )
                    dispatch_row = await cur.fetchone()
                    _validate_model_completion_stage_recovery_fence(
                        prepared.recovery_fence,
                        session=loaded,
                        checkpoint=await load_checkpoint(cur, session_id),
                        stage=stage,
                        active_record=(
                            None if active_row is None else decode_stage_record(active_row[0])
                        ),
                        dispatch_record=(
                            None if dispatch_row is None else decode_stage_record(dispatch_row[0])
                        ),
                        now=await store_now(cur),
                    )
                _validate_model_completion_stage_publication(
                    prepared.publication,
                    session_id=session_id,
                    stage=stage,
                )
                if not _runtime_publication_json_equal(
                    prepared.publication.intent,
                    stage.intent,
                ):
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
                await cur.execute(
                    "SELECT record FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = %s",
                    (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                )
                active_row = await cur.fetchone()
                active_record = None if active_row is None else decode_stage_record(active_row[0])
                advances_last_activity = _model_completion_terminal_advances_last_activity(
                    active_record,
                    stage=stage,
                    current_run_epoch=loaded.run_epoch,
                )
                await cur.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record, updated_at) "
                    "VALUES (%s, %s, %s, %s)",
                    (
                        session_id,
                        prepared.terminal_storage_key,
                        pg_support._dumps(terminal_record),
                        completed_at,
                    ),
                )
                await cur.execute(
                    "UPDATE cayu_sessions SET updated_at = %s, "
                    "last_activity_at = CASE WHEN %s THEN %s ELSE last_activity_at END "
                    "WHERE id = %s",
                    (
                        completed_at,
                        advances_last_activity,
                        completed_at,
                        session_id,
                    ),
                )
            await conn.commit()
            return ModelCompletionStageResult(
                stage=completed_stage,
                replayed=False,
                dispatch_authorized=False,
            )
        except BaseException:
            await conn.rollback()
            raise


async def abandon(
    prepared: _PreparedModelCompletionStageAbandonment,
    *,
    connect: PostgresConnection,
    ensure_ready: Callable[[], Awaitable[None]],
    load_for_update: Callable[[Any, str], Awaitable[Session | None]],
) -> ModelCompletionStageAbandonmentResult:
    session_id = prepared.session_id
    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                loaded = await load_for_update(cur, session_id)
                if loaded is None:
                    raise KeyError(f"Session not found: {session_id}")
                await cur.execute(
                    "SELECT idempotency_key, record FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                    (
                        session_id,
                        [
                            prepared.preparation_storage_key,
                            prepared.terminal_storage_key,
                            prepared.abandonment_storage_key,
                        ],
                    ),
                )
                records = {row[0]: decode_stage_record(row[1]) for row in await cur.fetchall()}
                stage = _reconstruct_model_completion_stage(
                    records.get(prepared.preparation_storage_key),
                    records.get(prepared.terminal_storage_key),
                    session_id=session_id,
                    stage_id=prepared.stage_id,
                    preparation_storage_key=prepared.preparation_storage_key,
                    terminal_storage_key=prepared.terminal_storage_key,
                )
                await cur.execute(
                    "SELECT record FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = %s",
                    (session_id, MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                )
                active_row = await cur.fetchone()
                active_record = None if active_row is None else decode_stage_record(active_row[0])
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
                    await cur.execute(
                        "SELECT idempotency_key FROM cayu_session_operations "
                        "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                        (
                            session_id,
                            [
                                _model_completion_stage_winner_storage_key(
                                    replayed.abandonment.logical_step_id
                                ),
                                _runtime_publication_storage_key(
                                    replayed.abandonment.logical_step_id
                                ),
                            ],
                        ),
                    )
                    if await cur.fetchone() is not None:
                        raise SessionModelCompletionStageConflict(
                            "An abandoned model-completion stage has durable publication state."
                        )
                    await conn.rollback()
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
                dispatch_storage_key = _model_completion_stage_dispatch_storage_key(stage.stage_id)
                await cur.execute(
                    "SELECT idempotency_key FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = ANY(%s)",
                    (
                        session_id,
                        [winner_storage_key, publication_storage_key, dispatch_storage_key],
                    ),
                )
                publication_keys = {row[0] for row in await cur.fetchall()}
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
                await cur.execute(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record, updated_at) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT(session_id, idempotency_key) DO UPDATE SET "
                    "record = EXCLUDED.record, updated_at = EXCLUDED.updated_at",
                    (
                        session_id,
                        prepared.abandonment_storage_key,
                        pg_support._dumps(abandonment_record),
                        abandoned_at,
                    ),
                )
                await cur.execute(
                    "DELETE FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = %s "
                    "AND record->>'record_digest' = %s",
                    (
                        session_id,
                        prepared.preparation_storage_key,
                        prepared.preparation_digest,
                    ),
                )
                if cur.rowcount != 1:
                    raise SessionModelCompletionStageConflict(
                        "The model-completion preparation changed during abandonment."
                    )
                await cur.execute(
                    "DELETE FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = %s "
                    "AND record->>'record_digest' = %s "
                    "AND record->>'preparation_digest' = %s",
                    (
                        session_id,
                        MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                        active.marker_digest,
                        prepared.preparation_digest,
                    ),
                )
                if cur.rowcount != 1:
                    raise SessionModelCompletionStageConflict(
                        "The active model-completion marker changed during abandonment."
                    )
                await cur.execute(
                    "UPDATE cayu_sessions SET updated_at = %s, last_activity_at = %s WHERE id = %s",
                    (abandoned_at, abandoned_at, session_id),
                )
                if cur.rowcount != 1:
                    raise KeyError(f"Session not found: {session_id}")
            await conn.commit()
            return ModelCompletionStageAbandonmentResult(
                abandonment=abandonment,
                replayed=False,
            )
        except BaseException:
            await conn.rollback()
            raise


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
