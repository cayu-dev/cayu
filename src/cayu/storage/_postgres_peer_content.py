"""Complete PostgreSQL peer-content persistence operations.

Native capabilities keep admission, replay and writes in one transaction.
Retry discovery releases its read scope before fresh admission.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Any
from uuid import uuid4

from cayu.collaboration.peer_content import (
    PeerAppendKey,
    PeerContentAppendRequest,
    PeerContentConflict,
    PeerContentExposureReceipt,
    PeerContentExposureRequest,
    PeerContentReceipt,
    PeerContentUnavailable,
)
from cayu.events import Event, EventType, event_with_runtime_payload_authority
from cayu.messages import Message, MessageRole
from cayu.runtime import _session_message_queue as message_queue
from cayu.sessions.access import require_resource_session
from cayu.sessions.creation_fence import SessionCreationTarget
from cayu.sessions.messaging import (
    SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    SessionMessageDeliveryMode,
    _queued_session_message_event_payload,
)
from cayu.sessions.records import Session
from cayu.storage import _creation_fence
from cayu.storage import _postgres_support as pg_support

PostgresConnection = Callable[[], AbstractAsyncContextManager[Any]]


async def append_peer_content(
    connect: PostgresConnection,
    request: PeerContentAppendRequest,
    *,
    qualify_target: Callable[[Session], None] | None = None,
    pending_transcript_cursor: int | None = None,
    ensure_ready: Callable[[], Awaitable[None]],
    load_for_update: Callable[[Any, str], Awaitable[Session | None]],
    session_store_now: Callable[[Any], Awaitable[datetime]],
) -> PeerContentReceipt:
    if type(request) is not PeerContentAppendRequest:
        raise TypeError("Peer append requires a PeerContentAppendRequest.")
    request = PeerContentAppendRequest.model_validate(request)
    from cayu.collaboration.peer_content import resolve_peer_target

    key = request.append_key.model_dump_json()
    commitment = request.model_dump(mode="json")
    await ensure_ready()
    async with connect() as conn:
        async with conn.cursor() as cur:
            creation = request.append_key.creation_target
            if creation is not None:
                await _creation_fence.postgres_lock(cur, creation)
            target_id, target_instance, creation_excluded = resolve_peer_target(
                request.append_key,
                None if creation is None else await _creation_fence.postgres_read(cur, creation),
            )
            from cayu.sessions.access import _query_bounds

            if _query_bounds.get() is not None:
                resource_owners = {
                    sid: await load_for_update(cur, sid)
                    for sid in sorted(
                        {request.occurrence.sender_session_id}
                        | ({target_id} if target_id else set())
                    )
                }
                require_resource_session(
                    resource_owners.get(request.occurrence.sender_session_id), "read"
                )
                require_resource_session(resource_owners.get(target_id), "modify")
            for lock_key in sorted(
                (
                    f"peer-append:{key}",
                    f"peer-operation:{request.operation_key}",
                    f"peer-consumer:{request.append_key.consumer_id}:"
                    f"{request.append_key.consumer_participant_incarnation}",
                )
            ):
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (lock_key,)
                )
            await cur.execute(
                "SELECT commitment_json, receipt_json FROM cayu_peer_content_receipts WHERE append_key_json = %s OR operation_key = %s FOR UPDATE",
                (key, request.operation_key),
            )
            row = await cur.fetchone()
            from cayu.storage._peer_attempts import historical_replay, replay_or_advance

            await cur.execute(
                "SELECT request_json, receipt_json FROM cayu_peer_content_attempts WHERE operation_key = %s",
                (request.operation_key,),
            )
            historical = historical_replay(request, await cur.fetchone())
            if historical is not None:
                await conn.commit()
                return historical
            if row is not None:
                replay = replay_or_advance(
                    request,
                    PeerContentAppendRequest.model_validate(row[0]),
                    PeerContentReceipt.model_validate(row[1]),
                )
                if replay is not None:
                    await conn.commit()
                    return replay
                await cur.execute(
                    "DELETE FROM cayu_peer_content_receipts WHERE append_key_json = %s", (key,)
                )
            if row is None and request.replaces_operation_key is not None:
                raise PeerContentConflict()
            from cayu.storage._peer_attempts import qualify, require_capacity

            await cur.execute(
                """SELECT COUNT(*) FROM cayu_peer_content_receipts p
                    LEFT JOIN cayu_session_message_queue q
                    ON q.queue_id = p.receipt_json->>'queue_id'
                    WHERE p.append_key_json != %s AND NOT p.target_deleted
                    AND p.request_json->'append_key'->>'consumer_id' = %s
                    AND p.request_json->'append_key'->>'consumer_participant_incarnation' = %s
                    AND (p.receipt_json->>'status' = 'pending'
                         OR (p.receipt_json->>'status' = 'appended'
                             AND (q.status IS NULL OR q.status = 'queued')))""",
                (
                    key,
                    request.append_key.consumer_id,
                    request.append_key.consumer_participant_incarnation,
                ),
            )
            outstanding = (await cur.fetchone())[0]
            from cayu.storage._peer_attempts import receiving_cursor

            target_cursor = receiving_cursor(
                request,
                None if row is None else PeerContentReceipt.model_validate(row[1]),
                pending_transcript_cursor,
            )
            await cur.execute(
                "SELECT instance_id, run_epoch, status, agent_name, environment_name FROM cayu_sessions WHERE id = %s FOR UPDATE",
                (target_id,),
            )
            session = await cur.fetchone()
            await cur.execute(
                "SELECT participant_id, participant_incarnation, session_instance_id "
                "FROM cayu_participant_session_bindings WHERE session_id = %s",
                (target_id,),
            )
            target_binding = await cur.fetchone()
            target_binding_valid = target_binding is not None and tuple(target_binding) == (
                request.append_key.consumer_id,
                request.append_key.consumer_participant_incarnation,
                target_instance,
            )
            await cur.execute(
                "SELECT state FROM cayu_checkpoints WHERE session_id = %s",
                (target_id,),
            )
            checkpoint_row = await cur.fetchone()
            checkpoint = None if checkpoint_row is None else checkpoint_row[0]
            parked_target = False
            if session is not None and session[2] in {"completed", "failed", "interrupted"}:
                assert target_id is not None
                from cayu.storage._peer_attempts import (
                    parked_delivery_key,
                    permits_parked_delivery_append,
                )

                wait_key = parked_delivery_key(
                    checkpoint, session_id=target_id, instance_id=session[0]
                )
                if wait_key is not None:
                    await cur.execute(
                        "SELECT record FROM cayu_session_operations WHERE session_id = %s AND idempotency_key = %s",
                        (target_id, wait_key),
                    )
                    wait_row = await cur.fetchone()
                    parked_target = permits_parked_delivery_append(
                        request,
                        checkpoint,
                        None if wait_row is None else wait_row[0],
                        session_id=target_id,
                        instance_id=session[0],
                        run_epoch=session[1],
                    )
            if session is not None and session[2] not in {
                "completed",
                "failed",
                "interrupted",
            }:
                message_queue.require_open_admission(session[2], checkpoint)
            await cur.execute(
                "SELECT participant_id, participant_incarnation, session_instance_id "
                "FROM cayu_participant_session_bindings WHERE session_id = %s",
                (request.occurrence.sender_session_id,),
            )
            source_binding = await cur.fetchone()
            source_valid = (
                source_binding is not None
                and source_binding[0] == request.occurrence.sender_participant_id
                and source_binding[1] == request.occurrence.sender_participant_incarnation
                and source_binding[2] == request.occurrence.sender_session_instance_id
            )
            await cur.execute(
                "SELECT COUNT(*) FROM cayu_transcript_messages WHERE session_id = %s",
                (target_id,),
            )
            cursor = (await cur.fetchone())[0]
            now = await session_store_now(cur)
            expired = int(now.timestamp() * 1000) >= request.attempt_key.deadline_at_ms
            if expired:
                result = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="excluded",
                    reason="delivery_deadline_expired",
                )
            elif (
                creation_excluded
                or not source_valid
                or (
                    session is not None
                    and not parked_target
                    and session[2] in {"completed", "failed", "interrupted"}
                )
            ):
                result = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="excluded",
                    reason="source_or_target_unavailable",
                )
            elif session is None:
                result = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="pending",
                    reason="target_not_created",
                )
            elif target_binding is None:
                result = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="pending",
                    reason="target_binding_pending",
                )
            elif not target_binding_valid or session[0] != target_instance:
                result = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="excluded",
                    reason="target_binding_mismatch",
                )
            elif isinstance(checkpoint, dict) and "pending_tool_round" in checkpoint:
                result = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="pending",
                    reason="target_busy",
                )
            elif session[1] != request.attempt_key.target_run_epoch or cursor != target_cursor:
                result = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="pending",
                    reason="target_cursor_changed",
                )
            else:
                require_capacity(outstanding)
                assert target_id is not None
                qualified_session = await load_for_update(cur, target_id)
                qualify(qualify_target, qualified_session)
                message = Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        request.occurrence.to_message_part(
                            append_key=request.append_key,
                            projection_id=request.append_key.projection_id,
                            operation_key=request.operation_key,
                        ),
                    ),
                )
                from cayu.collaboration.peer_content import peer_queue_id

                assert target_id is not None and target_instance is not None
                queue_id = peer_queue_id(request.append_key, target_id, target_instance)
                await cur.execute(
                    "SELECT 1 FROM cayu_session_message_queue WHERE queue_id = %s "
                    "OR (session_id = %s AND idempotency_key = %s)",
                    (queue_id, target_id, request.operation_key),
                )
                if await cur.fetchone() is not None:
                    raise PeerContentConflict("Peer queue identity already has another authority.")
                delivery_mode = (
                    SessionMessageDeliveryMode.ON_IDLE
                    if request.wake_policy == "ordinary_continuation"
                    else SessionMessageDeliveryMode.NEXT_TURN
                )
                accepted_at = await session_store_now(cur)
                accepted_event_id = str(uuid4())
                await cur.execute(
                    "INSERT INTO cayu_session_message_queue (queue_id, session_id, idempotency_key, content, message_json, delivery_mode, status, accepted_run_epoch, accepted_transcript_cursor, accepted_event_id, accepted_at) VALUES (%s, %s, %s, %s, %s, %s, 'queued', %s, %s, %s, %s) RETURNING ordering_key",
                    (
                        queue_id,
                        target_id,
                        request.operation_key,
                        request.occurrence.payload.text,
                        pg_support._dumps(message.model_dump(mode="json")),
                        str(delivery_mode),
                        session[1],
                        cursor,
                        accepted_event_id,
                        accepted_at,
                    ),
                )
                ordering_row = await cur.fetchone()
                if ordering_row is None:
                    raise RuntimeError("Peer queue insert did not return ordering key.")
                ordering_key = ordering_row[0]
                accepted_event = event_with_runtime_payload_authority(
                    Event(
                        id=accepted_event_id,
                        type=EventType.SESSION_MESSAGE_QUEUED,
                        session_id=str(target_id),
                        agent_name=session[3],
                        environment_name=session[4],
                        timestamp=accepted_at,
                        payload={
                            **_queued_session_message_event_payload(
                                queue_id=queue_id,
                                delivery_mode=delivery_mode,
                                ordering_key=ordering_key,
                                actor=None,
                                run_epoch=session[1],
                                transcript_cursor=cursor,
                            ),
                            "peer_occurrence_id": request.occurrence.occurrence_id,
                            "peer_provenance_sha256": request.occurrence.provenance_sha256,
                        },
                    )
                )
                await cur.execute(
                    "UPDATE cayu_sessions SET event_seq = event_seq + 1 "
                    "WHERE id = %s RETURNING event_seq",
                    (target_id,),
                )
                event_order_row = await cur.fetchone()
                if event_order_row is None:
                    raise RuntimeError("Peer event session sequence was not advanced.")
                await cur.execute(
                    "INSERT INTO cayu_events (session_id, session_order, event_id, "
                    "interaction_id, event_type, timestamp, agent_name, environment_name, "
                    "workflow_name, tool_name, payload, event, pending_action_lookup_key, "
                    "pending_action_projection, pending_action_projection_bytes) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        target_id,
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
                        None,
                        None,
                        None,
                    ),
                )
                result = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="appended",
                    target_session_id=target_id,
                    target_session_instance_id=target_instance,
                    occurrence=request.occurrence,
                    queue_id=queue_id,
                )
            if result.status == "pending":
                require_capacity(outstanding)
            await cur.execute(
                "INSERT INTO cayu_peer_content_receipts (append_key_json, operation_key, commitment_json, receipt_json, request_json) VALUES (%s, %s, %s, %s, %s)",
                (
                    key,
                    request.operation_key,
                    pg_support._dumps(commitment),
                    pg_support._dumps(result.model_dump(mode="json")),
                    pg_support._dumps(request.model_dump(mode="json")),
                ),
            )
            await cur.execute(
                "INSERT INTO cayu_peer_content_attempts (operation_key, request_json, receipt_json) "
                "VALUES (%s, %s, %s) ON CONFLICT(operation_key) DO UPDATE SET "
                "request_json=EXCLUDED.request_json, receipt_json=EXCLUDED.receipt_json",
                (
                    request.operation_key,
                    pg_support._dumps(request.model_dump(mode="json")),
                    pg_support._dumps(result.model_dump(mode="json")),
                ),
            )
        await conn.commit()
        return result


async def read_peer_content_attempt(
    connect: PostgresConnection,
    request: PeerContentAppendRequest,
    *,
    ensure_ready: Callable[[], Awaitable[None]],
) -> PeerContentReceipt | None:
    from cayu.storage._peer_attempts import exact_read

    request = PeerContentAppendRequest.model_validate(request)
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT request_json, receipt_json FROM cayu_peer_content_attempts WHERE operation_key = %s",
            (request.operation_key,),
        )
        return exact_read(request, await cur.fetchone())


async def list_pending_peer_content(
    connect: PostgresConnection,
    *,
    after_operation_key=None,
    limit=32,
    ensure_ready: Callable[[], Awaitable[None]],
):
    """Trusted receiving-owner discovery, independent of creation settlement."""
    from cayu.collaboration.peer_content import validate_peer_discovery

    validate_peer_discovery(after_operation_key, limit)
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT request_json FROM cayu_peer_content_receipts "
            "WHERE receipt_json->>'status' = 'pending' "
            "AND operation_key > %s ORDER BY operation_key LIMIT %s",
            (after_operation_key or "", limit),
        )
        return tuple(
            PeerContentAppendRequest.model_validate(row[0]) for row in await cur.fetchall()
        )


async def read_peer_content(
    connect: PostgresConnection,
    append_key: PeerAppendKey,
    *,
    ensure_ready: Callable[[], Awaitable[None]],
) -> PeerContentReceipt | None:
    if type(append_key) is not PeerAppendKey:
        raise TypeError("append_key must be a PeerAppendKey.")
    append_key = PeerAppendKey.model_validate(append_key)
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT receipt_json FROM cayu_peer_content_receipts WHERE append_key_json = %s",
            (append_key.model_dump_json(),),
        )
        row = await cur.fetchone()
        return None if row is None else PeerContentReceipt.model_validate(row[0])


async def record_peer_content_exposure(
    connect: PostgresConnection,
    request: PeerContentExposureRequest,
    *,
    ensure_ready: Callable[[], Awaitable[None]],
) -> PeerContentExposureReceipt:
    await ensure_ready()
    key = request.append_key.model_dump_json()
    commitment = request.model_dump(mode="json")
    async with connect() as conn, conn.cursor() as cur:
        for lock_key in sorted(
            (f"peer-exposure:{request.exposure_id}", f"peer-operation:{request.operation_key}")
        ):
            await cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (lock_key,))
        await cur.execute(
            "SELECT commitment_json, receipt_json FROM cayu_peer_content_exposures WHERE exposure_id = %s OR operation_key = %s FOR UPDATE",
            (request.exposure_id, request.operation_key),
        )
        row = await cur.fetchone()
        if row is not None:
            prior = PeerContentExposureReceipt.model_validate(row[1])
            expected = request.identity_commitment() if prior.outcome == "pending" else commitment
            if row[0] != expected:
                raise PeerContentConflict()
            if prior.outcome == "pending":
                receipt = PeerContentExposureReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    exposure_id=request.exposure_id,
                    model_attempt_id=request.model_attempt_id,
                    outcome=request.outcome,
                    reason=request.reason,
                )
                await cur.execute(
                    "UPDATE cayu_peer_content_exposures SET commitment_json = %s, receipt_json = %s WHERE exposure_id = %s",
                    (
                        pg_support._dumps(commitment),
                        pg_support._dumps(receipt.model_dump(mode="json")),
                        request.exposure_id,
                    ),
                )
                await conn.commit()
                return receipt
            await conn.commit()
            return prior.model_copy(update={"replayed": True})
        await cur.execute(
            "SELECT receipt_json FROM cayu_peer_content_receipts WHERE append_key_json = %s",
            (key,),
        )
        append = await cur.fetchone()
        if append is None or append[0].get("status") != "appended":
            raise PeerContentUnavailable("Peer content was not durably appended.")
        receipt = PeerContentExposureReceipt(
            operation_key=request.operation_key,
            append_key=request.append_key,
            exposure_id=request.exposure_id,
            model_attempt_id=request.model_attempt_id,
            outcome=request.outcome,
            reason=request.reason,
        )
        await cur.execute(
            "INSERT INTO cayu_peer_content_exposures (exposure_id, operation_key, append_key_json, commitment_json, receipt_json) VALUES (%s, %s, %s, %s, %s)",
            (
                request.exposure_id,
                request.operation_key,
                pg_support._dumps(key),
                pg_support._dumps(commitment),
                pg_support._dumps(receipt.model_dump(mode="json")),
            ),
        )
        await conn.commit()
        return receipt


async def begin_peer_content_exposure(
    connect: PostgresConnection,
    request: PeerContentExposureRequest,
    *,
    ensure_ready: Callable[[], Awaitable[None]],
) -> PeerContentExposureReceipt:
    await ensure_ready()
    key = request.append_key.model_dump_json()
    identity = request.identity_commitment()
    async with connect() as conn, conn.cursor() as cur:
        for lock_key in sorted(
            (f"peer-exposure:{request.exposure_id}", f"peer-operation:{request.operation_key}")
        ):
            await cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (lock_key,))
        await cur.execute(
            "SELECT commitment_json, receipt_json FROM cayu_peer_content_exposures WHERE exposure_id = %s OR operation_key = %s FOR UPDATE",
            (request.exposure_id, request.operation_key),
        )
        row = await cur.fetchone()
        if row is not None:
            prior = PeerContentExposureReceipt.model_validate(row[1])
            expected = identity if prior.outcome == "pending" else request.model_dump(mode="json")
            if row[0] != expected:
                raise PeerContentConflict()
            await conn.commit()
            return prior.model_copy(update={"replayed": True})
        await cur.execute(
            "SELECT receipt_json FROM cayu_peer_content_receipts WHERE append_key_json = %s",
            (key,),
        )
        append = await cur.fetchone()
        if append is None or append[0].get("status") != "appended":
            raise PeerContentUnavailable("Peer content was not durably appended.")
        receipt = PeerContentExposureReceipt(
            operation_key=request.operation_key,
            append_key=request.append_key,
            exposure_id=request.exposure_id,
            model_attempt_id=request.model_attempt_id,
            outcome="pending",
        )
        await cur.execute(
            "INSERT INTO cayu_peer_content_exposures (exposure_id, operation_key, append_key_json, commitment_json, receipt_json) VALUES (%s, %s, %s, %s, %s)",
            (
                request.exposure_id,
                request.operation_key,
                pg_support._dumps(key),
                pg_support._dumps(identity),
                pg_support._dumps(receipt.model_dump(mode="json")),
            ),
        )
        await conn.commit()
        return receipt


async def read_peer_content_exposure(
    connect: PostgresConnection,
    append_key: PeerAppendKey,
    exposure_id: str,
    *,
    ensure_ready: Callable[[], Awaitable[None]],
) -> PeerContentExposureReceipt | None:
    if type(append_key) is not PeerAppendKey or type(exposure_id) is not str:
        raise TypeError("Invalid exposure lookup.")
    append_key = PeerAppendKey.model_validate(append_key)
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT append_key_json, receipt_json FROM cayu_peer_content_exposures WHERE exposure_id = %s",
            (exposure_id,),
        )
        row = await cur.fetchone()
        if row is None or row[0] != append_key.model_dump_json():
            return None
        return PeerContentExposureReceipt.model_validate(row[1])


async def exclude_peer_content(
    connect: PostgresConnection,
    request: PeerContentAppendRequest,
    *,
    reason: str,
    ensure_ready: Callable[[], Awaitable[None]],
) -> PeerContentReceipt:
    from cayu._validation import require_durable_clean_nonblank

    if type(request) is not PeerContentAppendRequest:
        raise TypeError("Peer exclusion requires a PeerContentAppendRequest.")
    request = PeerContentAppendRequest.model_validate(request)
    reason = require_durable_clean_nonblank(reason, "reason")
    from cayu.collaboration.peer_content import resolve_peer_target

    key = request.append_key.model_dump_json()
    commitment = request.model_dump(mode="json")
    result = PeerContentReceipt(
        operation_key=request.operation_key,
        append_key=request.append_key,
        attempt_generation=request.attempt_key.attempt_generation,
        status="excluded",
        reason=reason,
    )
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        creation = request.append_key.creation_target
        if creation is not None:
            await _creation_fence.postgres_lock(cur, creation)
        resolve_peer_target(
            request.append_key,
            None if creation is None else await _creation_fence.postgres_read(cur, creation),
        )
        for lock_key in sorted((f"peer-append:{key}", f"peer-operation:{request.operation_key}")):
            await cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (lock_key,))
        await cur.execute(
            "SELECT commitment_json, receipt_json FROM cayu_peer_content_receipts "
            "WHERE append_key_json = %s OR operation_key = %s FOR UPDATE",
            (key, request.operation_key),
        )
        row = await cur.fetchone()
        from cayu.storage._peer_attempts import historical_replay

        await cur.execute(
            "SELECT request_json, receipt_json FROM cayu_peer_content_attempts WHERE operation_key = %s",
            (request.operation_key,),
        )
        historical = historical_replay(request, await cur.fetchone())
        if historical is not None:
            if historical.status == "excluded" and historical.reason != reason:
                raise PeerContentConflict()
            await conn.commit()
            return historical
        if row is not None:
            existing_commitment = row[0]
            if isinstance(existing_commitment, str):
                existing_commitment = json.loads(existing_commitment)
            if existing_commitment != commitment:
                raise PeerContentConflict()
            stored = PeerContentReceipt.model_validate(row[1])
            if stored.status == "excluded" and stored.reason != reason:
                raise PeerContentConflict()
            if stored.status != "pending":
                await conn.commit()
                return stored.model_copy(update={"replayed": True})
            await cur.execute(
                "DELETE FROM cayu_peer_content_receipts WHERE append_key_json = %s", (key,)
            )
        await cur.execute(
            "INSERT INTO cayu_peer_content_receipts (append_key_json, operation_key, commitment_json, receipt_json, request_json) VALUES (%s, %s, %s, %s, %s)",
            (
                key,
                request.operation_key,
                pg_support._dumps(commitment),
                pg_support._dumps(result.model_dump(mode="json")),
                pg_support._dumps(request.model_dump(mode="json")),
            ),
        )
        await cur.execute(
            "INSERT INTO cayu_peer_content_attempts (operation_key, request_json, receipt_json) "
            "VALUES (%s, %s, %s) ON CONFLICT(operation_key) DO UPDATE SET "
            "request_json=EXCLUDED.request_json, receipt_json=EXCLUDED.receipt_json",
            (
                request.operation_key,
                pg_support._dumps(request.model_dump(mode="json")),
                pg_support._dumps(result.model_dump(mode="json")),
            ),
        )
    return result


async def retry_pending_peer_content(
    connect: PostgresConnection,
    session_id: str,
    *,
    expected_session_instance_id: str,
    expected_run_epoch: int,
    expected_transcript_cursor: int,
    admit=None,
    ensure_ready: Callable[[], Awaitable[None]],
    read_creation_decision: Callable[[SessionCreationTarget], Awaitable[Any]],
) -> tuple[PeerContentReceipt, ...]:
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT request_json FROM cayu_peer_content_receipts "
            "WHERE receipt_json->>'status' = 'pending' "
            "AND (request_json->'append_key'->>'target_session_id' = %s OR EXISTS ("
            "SELECT 1 FROM cayu_session_creation_decisions c WHERE "
            "c.decision_json::jsonb->>'session_id' = %s AND "
            "c.decision_json::jsonb->>'session_instance_id' = %s AND "
            "c.decision_json::jsonb->'target' = request_json->'append_key'->'creation_target')) "
            "ORDER BY operation_key LIMIT %s",
            (
                session_id,
                session_id,
                expected_session_instance_id,
                SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
            ),
        )
        rows = await cur.fetchall()
    requests = [
        PeerContentAppendRequest.model_validate(row[0]) for row in rows if row[0] is not None
    ]
    results: list[PeerContentReceipt] = []
    for request in requests:
        if request.append_key.creation_target is not None:
            from cayu.collaboration._contracts import ExactMatch

            decision = await read_creation_decision(request.append_key.creation_target)
            if not isinstance(decision, ExactMatch) or (
                decision.receipt.session_id != session_id
                or decision.receipt.session_instance_id != expected_session_instance_id
            ):
                continue
        if (
            request.append_key.creation_target is None
            and request.append_key.target_session_instance_id != expected_session_instance_id
        ):
            continue
        if request.attempt_key.target_run_epoch != expected_run_epoch:
            continue
        if admit is None:
            raise PeerContentUnavailable("Pending delivery requires fresh export authorization.")
        results.append(await admit(request, pending_transcript_cursor=expected_transcript_cursor))
    return tuple(results)
