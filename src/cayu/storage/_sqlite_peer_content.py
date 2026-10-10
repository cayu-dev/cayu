"""Complete SQLite peer-content persistence operations.

Native capabilities keep admission, replay and writes in one transaction.
Retry discovery releases its read scope before fresh admission.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Awaitable, Callable
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
from cayu.sessions.base import (
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    session_messages_input_contract_evidence,
)
from cayu.sessions.creation_fence import SessionCreationTarget
from cayu.sessions.messaging import (
    SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    SessionMessageConditions,
    SessionMessageDeliveryMode,
    _queued_session_message_event_payload,
)
from cayu.sessions.records import Session
from cayu.storage import _creation_fence
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage._sqlite_connection import SQLiteOperationRunner


async def append_peer_content(
    run_write: SQLiteOperationRunner,
    request: PeerContentAppendRequest,
    *,
    qualify_target: Callable[[Session], None] | None = None,
    pending_transcript_cursor: int | None = None,
    load_unlocked: Callable[[str], Session | None],
    ownership_clock: Callable[[], datetime],
) -> PeerContentReceipt:
    if type(request) is not PeerContentAppendRequest:
        raise TypeError("Peer append requires a PeerContentAppendRequest.")
    request = PeerContentAppendRequest.model_validate(request)
    from cayu.collaboration.peer_content import resolve_peer_target

    key = request.append_key.model_dump_json()
    commitment = request.model_dump_json()

    def statement(connection: sqlite3.Connection) -> PeerContentReceipt:
        connection.execute("BEGIN IMMEDIATE")
        try:
            creation = request.append_key.creation_target
            target_id, target_instance, creation_excluded = resolve_peer_target(
                request.append_key,
                None if creation is None else _creation_fence.sqlite_read(connection, creation),
            )
            from cayu.sessions.access import _query_bounds

            if _query_bounds.get() is not None:
                require_resource_session(
                    load_unlocked(request.occurrence.sender_session_id), "read"
                )
                require_resource_session(
                    None if target_id is None else load_unlocked(target_id), "modify"
                )
            row = connection.execute(
                "SELECT commitment_json, receipt_json FROM cayu_peer_content_receipts "
                "WHERE append_key_json = ? OR operation_key = ?",
                (key, request.operation_key),
            ).fetchone()
            from cayu.storage._peer_attempts import historical_replay, replay_or_advance

            historical = historical_replay(
                request,
                connection.execute(
                    "SELECT request_json, receipt_json FROM cayu_peer_content_attempts WHERE operation_key = ?",
                    (request.operation_key,),
                ).fetchone(),
            )
            if historical is not None:
                connection.commit()
                return historical
            if row is not None:
                replay = replay_or_advance(
                    request,
                    PeerContentAppendRequest.model_validate_json(row[0]),
                    PeerContentReceipt.model_validate_json(row[1]),
                )
                if replay is not None:
                    connection.commit()
                    return replay
                connection.execute(
                    "DELETE FROM cayu_peer_content_receipts WHERE append_key_json = ?", (key,)
                )
            if row is None and request.replaces_operation_key is not None:
                raise PeerContentConflict()
            from cayu.storage._peer_attempts import qualify, require_capacity

            outstanding = connection.execute(
                """SELECT COUNT(*) FROM cayu_peer_content_receipts p
                    LEFT JOIN cayu_session_message_queue q
                    ON q.queue_id = json_extract(p.receipt_json, '$.queue_id')
                    WHERE p.append_key_json != ? AND p.target_deleted = 0
                    AND json_extract(p.request_json, '$.append_key.consumer_id') = ?
                    AND json_extract(p.request_json, '$.append_key.consumer_participant_incarnation') = ?
                    AND (json_extract(p.receipt_json, '$.status') = 'pending'
                         OR (json_extract(p.receipt_json, '$.status') = 'appended'
                             AND (q.status IS NULL OR q.status = 'queued')))""",
                (
                    key,
                    request.append_key.consumer_id,
                    request.append_key.consumer_participant_incarnation,
                ),
            ).fetchone()[0]
            session = None if target_id is None else load_unlocked(target_id)
            from cayu.storage._peer_attempts import receiving_cursor

            target_cursor = receiving_cursor(
                request,
                None if row is None else PeerContentReceipt.model_validate_json(row[1]),
                pending_transcript_cursor,
            )
            target_binding = connection.execute(
                "SELECT participant_id, participant_incarnation, session_instance_id "
                "FROM cayu_participant_session_bindings WHERE session_id = ?",
                (target_id,),
            ).fetchone()
            target_binding_valid = target_binding is not None and tuple(target_binding) == (
                request.append_key.consumer_id,
                request.append_key.consumer_participant_incarnation,
                target_instance,
            )
            checkpoint_row = connection.execute(
                "SELECT state_json FROM cayu_checkpoints WHERE session_id = ?",
                (target_id,),
            ).fetchone()
            checkpoint = None if checkpoint_row is None else json.loads(checkpoint_row[0])
            parked_target = False
            if session is not None and session.status in {"completed", "failed", "interrupted"}:
                from cayu.storage._peer_attempts import (
                    parked_delivery_key,
                    permits_parked_delivery_append,
                )

                wait_key = parked_delivery_key(
                    checkpoint, session_id=session.id, instance_id=session.instance_id
                )
                if wait_key is not None:
                    wait_row = connection.execute(
                        "SELECT record_json FROM cayu_session_operations WHERE session_id = ? AND idempotency_key = ?",
                        (session.id, wait_key),
                    ).fetchone()
                    parked_target = permits_parked_delivery_append(
                        request,
                        checkpoint,
                        None if wait_row is None else json.loads(wait_row[0]),
                        session_id=session.id,
                        instance_id=session.instance_id,
                        run_epoch=session.run_epoch,
                    )
            if session is not None and session.status not in {
                "completed",
                "failed",
                "interrupted",
            }:
                message_queue.require_open_admission(session.status, checkpoint)
            source_binding = connection.execute(
                "SELECT participant_id, participant_incarnation, session_instance_id "
                "FROM cayu_participant_session_bindings WHERE session_id = ?",
                (request.occurrence.sender_session_id,),
            ).fetchone()
            source_valid = (
                source_binding is not None
                and source_binding[0] == request.occurrence.sender_participant_id
                and source_binding[1] == request.occurrence.sender_participant_incarnation
                and source_binding[2] == request.occurrence.sender_session_instance_id
            )
            cursor = connection.execute(
                "SELECT COUNT(*) FROM cayu_transcript_messages WHERE session_id = ?",
                (target_id,),
            ).fetchone()[0]
            expired = (
                int(ownership_clock().timestamp() * 1000) >= request.attempt_key.deadline_at_ms
            )
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
                    and session.status in {"completed", "failed", "interrupted"}
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
            elif not target_binding_valid or session.instance_id != target_instance:
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
            elif (
                session.run_epoch != request.attempt_key.target_run_epoch or cursor != target_cursor
            ):
                result = PeerContentReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    attempt_generation=request.attempt_key.attempt_generation,
                    status="pending",
                    reason="target_cursor_changed",
                )
            else:
                require_capacity(outstanding)
                qualify(qualify_target, session)
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
                if (
                    connection.execute(
                        "SELECT 1 FROM cayu_session_message_queue WHERE queue_id = ? "
                        "OR (session_id = ? AND idempotency_key = ?)",
                        (queue_id, target_id, request.operation_key),
                    ).fetchone()
                    is not None
                ):
                    raise PeerContentConflict("Peer queue identity already has another authority.")
                delivery_mode = (
                    SessionMessageDeliveryMode.ON_IDLE
                    if request.wake_policy == "ordinary_continuation"
                    else SessionMessageDeliveryMode.NEXT_TURN
                )
                accepted_at = ownership_clock()
                accepted_event_id = str(uuid4())
                ordering_key = connection.execute(
                    "INSERT INTO cayu_session_message_queue "
                    "(queue_id, session_id, idempotency_key, content, message_json, "
                    "delivery_mode, status, accepted_run_epoch, accepted_transcript_cursor, "
                    "accepted_event_id, accepted_at) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?)",
                    (
                        queue_id,
                        target_id,
                        request.operation_key,
                        request.occurrence.payload.text,
                        sqlite_records.json_dumps(message.model_dump(mode="json")),
                        str(delivery_mode),
                        session.run_epoch,
                        cursor,
                        accepted_event_id,
                        sqlite_records.format_datetime(accepted_at),
                    ),
                ).lastrowid
                if type(ordering_key) is not int:
                    raise RuntimeError("Peer queue insert did not return an ordering key.")
                accepted_event = event_with_runtime_payload_authority(
                    Event(
                        id=accepted_event_id,
                        type=EventType.SESSION_MESSAGE_QUEUED,
                        session_id=session.id,
                        agent_name=session.agent_name,
                        environment_name=session.environment_name,
                        timestamp=accepted_at,
                        payload={
                            **_queued_session_message_event_payload(
                                queue_id=queue_id,
                                delivery_mode=delivery_mode,
                                ordering_key=ordering_key,
                                actor=None,
                                run_epoch=session.run_epoch,
                                transcript_cursor=cursor,
                            ),
                            "peer_occurrence_id": request.occurrence.occurrence_id,
                            "peer_provenance_sha256": request.occurrence.provenance_sha256,
                            SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: (
                                session_messages_input_contract_evidence(
                                    (message,),
                                    message_start_index=cursor,
                                    redactions_applied=False,
                                    structured_output_requested=False,
                                )
                            ),
                        },
                    ),
                    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
                )
                lookup_key, projection, projection_bytes = (None, None, None)
                connection.execute(
                    "INSERT INTO cayu_events (session_id, event_id, event_type, timestamp, "
                    "agent_name, environment_name, payload_json, pending_action_lookup_key, "
                    "pending_action_projection_json, pending_action_projection_bytes) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        target_id,
                        accepted_event.id,
                        str(accepted_event.type),
                        sqlite_records.format_datetime(accepted_event.timestamp),
                        accepted_event.agent_name,
                        accepted_event.environment_name,
                        sqlite_records.json_dumps(accepted_event.payload),
                        lookup_key,
                        projection,
                        projection_bytes,
                    ),
                )
                connection.execute(
                    "UPDATE cayu_session_message_queue SET conditions_json = ? WHERE queue_id = ?",
                    (
                        sqlite_records.json_dumps(
                            SessionMessageConditions().model_dump(mode="json")
                        ),
                        queue_id,
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
            connection.execute(
                "INSERT INTO cayu_peer_content_receipts "
                "(append_key_json, operation_key, commitment_json, receipt_json, request_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    key,
                    request.operation_key,
                    commitment,
                    sqlite_records.json_dumps(result.model_dump(mode="json")),
                    sqlite_records.json_dumps(request.model_dump(mode="json")),
                ),
            )
            connection.execute(
                "INSERT INTO cayu_peer_content_attempts (operation_key, request_json, receipt_json) "
                "VALUES (?, ?, ?) ON CONFLICT(operation_key) DO UPDATE SET "
                "request_json=excluded.request_json, receipt_json=excluded.receipt_json",
                (request.operation_key, request.model_dump_json(), result.model_dump_json()),
            )
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise

    return await run_write(statement)


async def read_peer_content_attempt(
    run_read: SQLiteOperationRunner, request: PeerContentAppendRequest
) -> PeerContentReceipt | None:
    from cayu.storage._peer_attempts import exact_read

    request = PeerContentAppendRequest.model_validate(request)

    def query(connection):
        row = connection.execute(
            "SELECT request_json, receipt_json FROM cayu_peer_content_attempts WHERE operation_key = ?",
            (request.operation_key,),
        ).fetchone()
        return exact_read(request, row)

    return await run_read(query)


async def list_pending_peer_content(
    run_read: SQLiteOperationRunner, *, after_operation_key=None, limit=32
):
    """Trusted receiving-owner discovery, independent of creation settlement."""
    from cayu.collaboration.peer_content import validate_peer_discovery

    validate_peer_discovery(after_operation_key, limit)

    def query(connection):
        rows = connection.execute(
            "SELECT request_json FROM cayu_peer_content_receipts "
            "WHERE json_extract(receipt_json, '$.status') = 'pending' "
            "AND operation_key > ? ORDER BY operation_key LIMIT ?",
            (after_operation_key or "", limit),
        ).fetchall()
        return tuple(PeerContentAppendRequest.model_validate_json(row[0]) for row in rows)

    return await run_read(query)


async def read_peer_content(
    run_read: SQLiteOperationRunner, append_key: PeerAppendKey
) -> PeerContentReceipt | None:
    if type(append_key) is not PeerAppendKey:
        raise TypeError("append_key must be a PeerAppendKey.")
    append_key = PeerAppendKey.model_validate(append_key)
    key = append_key.model_dump_json()

    def query(connection: sqlite3.Connection) -> PeerContentReceipt | None:
        row = connection.execute(
            "SELECT receipt_json FROM cayu_peer_content_receipts WHERE append_key_json = ?",
            (key,),
        ).fetchone()
        return None if row is None else PeerContentReceipt.model_validate(json.loads(row[0]))

    return await run_read(query)


async def record_peer_content_exposure(
    run_write: SQLiteOperationRunner, request: PeerContentExposureRequest
) -> PeerContentExposureReceipt:
    key = request.append_key.model_dump_json()
    commitment = request.model_dump_json()

    def statement(connection: sqlite3.Connection) -> PeerContentExposureReceipt:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT commitment_json, receipt_json FROM cayu_peer_content_exposures WHERE exposure_id = ? OR operation_key = ?",
                (request.exposure_id, request.operation_key),
            ).fetchone()
            if row is not None:
                prior = PeerContentExposureReceipt.model_validate(json.loads(row[1]))
                expected = (
                    request.identity_commitment() if prior.outcome == "pending" else commitment
                )
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
                    connection.execute(
                        "UPDATE cayu_peer_content_exposures SET commitment_json = ?, receipt_json = ? WHERE exposure_id = ?",
                        (
                            commitment,
                            sqlite_records.json_dumps(receipt.model_dump(mode="json")),
                            request.exposure_id,
                        ),
                    )
                    connection.commit()
                    return receipt
                connection.commit()
                return prior.model_copy(update={"replayed": True})
            append = connection.execute(
                "SELECT receipt_json FROM cayu_peer_content_receipts WHERE append_key_json = ?",
                (key,),
            ).fetchone()
            if append is None or json.loads(append[0]).get("status") != "appended":
                raise PeerContentUnavailable("Peer content was not durably appended.")
            receipt = PeerContentExposureReceipt(
                operation_key=request.operation_key,
                append_key=request.append_key,
                exposure_id=request.exposure_id,
                model_attempt_id=request.model_attempt_id,
                outcome=request.outcome,
                reason=request.reason,
            )
            connection.execute(
                "INSERT INTO cayu_peer_content_exposures (exposure_id, operation_key, append_key_json, commitment_json, receipt_json) VALUES (?, ?, ?, ?, ?)",
                (
                    request.exposure_id,
                    request.operation_key,
                    key,
                    commitment,
                    sqlite_records.json_dumps(receipt.model_dump(mode="json")),
                ),
            )
            connection.commit()
            return receipt
        except BaseException:
            connection.rollback()
            raise

    return await run_write(statement)


async def begin_peer_content_exposure(
    run_write: SQLiteOperationRunner, request: PeerContentExposureRequest
) -> PeerContentExposureReceipt:
    key = request.append_key.model_dump_json()
    identity = request.identity_commitment()

    def statement(connection: sqlite3.Connection) -> PeerContentExposureReceipt:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT commitment_json, receipt_json FROM cayu_peer_content_exposures WHERE exposure_id = ? OR operation_key = ?",
                (request.exposure_id, request.operation_key),
            ).fetchone()
            if row is not None:
                prior = PeerContentExposureReceipt.model_validate(json.loads(row[1]))
                expected = identity if prior.outcome == "pending" else request.model_dump_json()
                if row[0] != expected:
                    raise PeerContentConflict()
                connection.commit()
                return prior.model_copy(update={"replayed": True})
            append = connection.execute(
                "SELECT receipt_json FROM cayu_peer_content_receipts WHERE append_key_json = ?",
                (key,),
            ).fetchone()
            if append is None or json.loads(append[0]).get("status") != "appended":
                raise PeerContentUnavailable("Peer content was not durably appended.")
            receipt = PeerContentExposureReceipt(
                operation_key=request.operation_key,
                append_key=request.append_key,
                exposure_id=request.exposure_id,
                model_attempt_id=request.model_attempt_id,
                outcome="pending",
            )
            connection.execute(
                "INSERT INTO cayu_peer_content_exposures (exposure_id, operation_key, append_key_json, commitment_json, receipt_json) VALUES (?, ?, ?, ?, ?)",
                (
                    request.exposure_id,
                    request.operation_key,
                    key,
                    identity,
                    sqlite_records.json_dumps(receipt.model_dump(mode="json")),
                ),
            )
            connection.commit()
            return receipt
        except BaseException:
            connection.rollback()
            raise

    return await run_write(statement)


async def read_peer_content_exposure(
    run_read: SQLiteOperationRunner, append_key: PeerAppendKey, exposure_id: str
) -> PeerContentExposureReceipt | None:
    if type(append_key) is not PeerAppendKey or type(exposure_id) is not str:
        raise TypeError("Invalid exposure lookup.")
    append_key = PeerAppendKey.model_validate(append_key)
    key = append_key.model_dump_json()

    def query(connection: sqlite3.Connection) -> PeerContentExposureReceipt | None:
        row = connection.execute(
            "SELECT append_key_json, receipt_json FROM cayu_peer_content_exposures WHERE exposure_id = ?",
            (exposure_id,),
        ).fetchone()
        if row is None or row[0] != key:
            return None
        return PeerContentExposureReceipt.model_validate(json.loads(row[1]))

    return await run_read(query)


async def exclude_peer_content(
    run_write: SQLiteOperationRunner, request: PeerContentAppendRequest, *, reason: str
) -> PeerContentReceipt:
    if type(request) is not PeerContentAppendRequest:
        raise TypeError("Peer exclusion requires a PeerContentAppendRequest.")
    request = PeerContentAppendRequest.model_validate(request)
    from cayu._validation import require_durable_clean_nonblank

    reason = require_durable_clean_nonblank(reason, "reason")
    from cayu.collaboration.peer_content import resolve_peer_target

    key = request.append_key.model_dump_json()
    commitment = request.model_dump_json()
    result = PeerContentReceipt(
        operation_key=request.operation_key,
        append_key=request.append_key,
        attempt_generation=request.attempt_key.attempt_generation,
        status="excluded",
        reason=reason,
    )

    def statement(connection: sqlite3.Connection) -> PeerContentReceipt:
        connection.execute("BEGIN IMMEDIATE")
        try:
            creation = request.append_key.creation_target
            resolve_peer_target(
                request.append_key,
                None if creation is None else _creation_fence.sqlite_read(connection, creation),
            )
            row = connection.execute(
                "SELECT commitment_json, receipt_json FROM cayu_peer_content_receipts "
                "WHERE append_key_json = ? OR operation_key = ?",
                (key, request.operation_key),
            ).fetchone()
            from cayu.storage._peer_attempts import historical_replay

            historical = historical_replay(
                request,
                connection.execute(
                    "SELECT request_json, receipt_json FROM cayu_peer_content_attempts WHERE operation_key = ?",
                    (request.operation_key,),
                ).fetchone(),
            )
            if historical is not None:
                if historical.status == "excluded" and historical.reason != reason:
                    raise PeerContentConflict()
                connection.commit()
                return historical
            if row is not None:
                if row[0] != commitment:
                    raise PeerContentConflict()
                value = PeerContentReceipt.model_validate(json.loads(row[1]))
                if value.status == "excluded" and value.reason != reason:
                    raise PeerContentConflict()
                if value.status != "pending":
                    connection.commit()
                    return value.model_copy(update={"replayed": True})
                connection.execute(
                    "DELETE FROM cayu_peer_content_receipts WHERE append_key_json = ?", (key,)
                )
            connection.execute(
                "INSERT INTO cayu_peer_content_receipts "
                "(append_key_json, operation_key, commitment_json, receipt_json, request_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    key,
                    request.operation_key,
                    commitment,
                    sqlite_records.json_dumps(result.model_dump(mode="json")),
                    sqlite_records.json_dumps(request.model_dump(mode="json")),
                ),
            )
            connection.execute(
                "INSERT INTO cayu_peer_content_attempts (operation_key, request_json, receipt_json) "
                "VALUES (?, ?, ?) ON CONFLICT(operation_key) DO UPDATE SET "
                "request_json=excluded.request_json, receipt_json=excluded.receipt_json",
                (request.operation_key, request.model_dump_json(), result.model_dump_json()),
            )
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise

    return await run_write(statement)


async def retry_pending_peer_content(
    run_read: SQLiteOperationRunner,
    session_id: str,
    *,
    expected_session_instance_id: str,
    expected_run_epoch: int,
    expected_transcript_cursor: int,
    admit=None,
    read_creation_decision: Callable[[SessionCreationTarget], Awaitable[Any]],
) -> tuple[PeerContentReceipt, ...]:
    def query(connection: sqlite3.Connection) -> list[PeerContentAppendRequest]:
        try:
            rows = connection.execute(
                "SELECT request_json FROM cayu_peer_content_receipts "
                "WHERE json_extract(receipt_json, '$.status') = 'pending' "
                "AND (json_extract(request_json, '$.append_key.target_session_id') = ? OR EXISTS ("
                "SELECT 1 FROM cayu_session_creation_decisions c WHERE "
                "json_extract(c.decision_json, '$.session_id') = ? AND "
                "json_extract(c.decision_json, '$.session_instance_id') = ? AND "
                "json_extract(c.decision_json, '$.target.creation_key') = "
                "json_extract(request_json, '$.append_key.creation_target.creation_key') AND "
                "json_extract(c.decision_json, '$.target.permit.intent.request.source_operation.application_scope') = "
                "json_extract(request_json, '$.append_key.creation_target.permit.intent.request.source_operation.application_scope') AND "
                "json_extract(c.decision_json, '$.target.permit.intent.request.source_operation.namespace_incarnation') = "
                "json_extract(request_json, '$.append_key.creation_target.permit.intent.request.source_operation.namespace_incarnation') AND "
                "json_extract(c.decision_json, '$.target.permit.intent.request.source_operation.generation') = "
                "json_extract(request_json, '$.append_key.creation_target.permit.intent.request.source_operation.generation') AND "
                "json_extract(c.decision_json, '$.target.permit.intent.request.source_operation.caller_key') = "
                "json_extract(request_json, '$.append_key.creation_target.permit.intent.request.source_operation.caller_key'))) "
                "ORDER BY operation_key LIMIT ?",
                (
                    session_id,
                    session_id,
                    expected_session_instance_id,
                    SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
                ),
            ).fetchall()
        except sqlite3.OperationalError as error:
            if "no such table" not in str(error):
                raise
            return []
        return [
            PeerContentAppendRequest.model_validate(json.loads(row[0]))
            for row in rows
            if row[0] is not None
        ]

    requests = await run_read(query)
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
