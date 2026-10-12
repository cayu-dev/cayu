"""Complete in-memory peer-content operations over explicit locked state."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.collaboration.peer_content import (
    PeerAppendKey,
    PeerContentAppendRequest,
    PeerContentConflict,
    PeerContentExposureReceipt,
    PeerContentExposureRequest,
    PeerContentReceipt,
    PeerContentUnavailable,
)
from cayu.events import Event, EventType
from cayu.messages import Message, MessageRole
from cayu.sessions import _memory_queued_messages as memory_queued_messages
from cayu.sessions import _tool_round_publication_rules as tool_publication_rules
from cayu.sessions import messaging as session_messaging
from cayu.sessions._access_rules import require_resource_session
from cayu.sessions.messaging import (
    SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    SessionMessageConditions,
    SessionMessageDeliveryMode,
    SessionMessageQueueStatus,
    SessionQueuedMessage,
)
from cayu.sessions.records import Session, SessionStatus

if TYPE_CHECKING:
    from cayu.sessions.context_views import ParticipantSessionBinding
    from cayu.sessions.creation_fence import SessionCreationDecision


class MemoryPeerContentState(NamedTuple):
    """Current message indexes and synchronous publication capabilities.

    The caller resolves these bindings after acquiring its lock on every
    operation. Authorization and publication stay inside that lock. Callbacks
    must finish synchronously and must not acquire the lock again. The ordering
    record is shared by ordinary and peer admission; advance it only after
    acceptance publication succeeds.
    """

    append_events: Callable[[Session, list[Event]], Session]
    checkpoints: dict[str, dict[str, Any]]
    ownership_clock: Callable[[], datetime]
    participant_session_bindings: dict[str, ParticipantSessionBinding]
    peer_content_attempts: dict[str, tuple[str, str]]
    peer_content_deleted_targets: set[tuple[str, str]]
    peer_content_exposures: dict[str, tuple[str, PeerContentExposureReceipt]]
    peer_content_operation_commitments: dict[str, str]
    peer_content_operations: dict[str, str]
    peer_content_receipts: dict[str, PeerContentReceipt]
    peer_content_requests: dict[str, PeerContentAppendRequest]
    pending_session_messages: dict[
        tuple[str, SessionMessageDeliveryMode], deque[SessionQueuedMessage]
    ]
    queued_session_messages_by_idempotency: dict[str, dict[str, SessionQueuedMessage]]
    session_creation_decisions: dict[str, SessionCreationDecision]
    ordering: memory_queued_messages.MemoryMessageOrdering
    session_operation_records: dict[str, dict[str, dict[str, Any]]]
    sessions: dict[str, Session]
    transcripts: dict[str, list[Message]]


LockedPeerContent = Callable[[], AbstractAsyncContextManager[MemoryPeerContentState]]


async def append_peer_content(
    locked: LockedPeerContent,
    request: PeerContentAppendRequest,
    *,
    qualify_target: Callable[[Session], None] | None = None,
    pending_transcript_cursor: int | None = None,
) -> PeerContentReceipt:
    from cayu.collaboration.peer_content import PeerContentReceipt

    if type(request) is not PeerContentAppendRequest:
        raise TypeError("Peer append requires a PeerContentAppendRequest.")
    request = PeerContentAppendRequest.model_validate(request)
    from cayu.collaboration.peer_content import resolve_peer_target

    key = request.append_key.model_dump_json()
    commitment = request.model_dump_json()
    async with locked() as state:
        creation = request.append_key.creation_target
        target_id, target_instance, creation_excluded = resolve_peer_target(
            request.append_key,
            None if creation is None else state.session_creation_decisions.get(creation.key),
        )
        from cayu.sessions._access_rules import _query_bounds

        if _query_bounds.get() is not None:
            require_resource_session(
                state.sessions.get(request.occurrence.sender_session_id), "read"
            )
            require_resource_session(
                None if target_id is None else state.sessions.get(target_id), "modify"
            )
        from cayu.storage._peer_attempts import historical_replay, replay_or_advance

        historical = historical_replay(
            request, state.peer_content_attempts.get(request.operation_key)
        )
        if historical is not None:
            return historical
        existing = state.peer_content_receipts.get(key)
        if existing is not None:
            replay = replay_or_advance(request, state.peer_content_requests[key], existing)
            if replay is not None:
                return replay
        elif request.replaces_operation_key is not None:
            raise PeerContentConflict()
        from cayu.storage._peer_attempts import receiving_cursor

        target_cursor = receiving_cursor(request, existing, pending_transcript_cursor)
        target = None if target_id is None else state.sessions.get(target_id)
        parked_target = False
        if target is not None and target.status in {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        }:
            from cayu.storage._peer_attempts import (
                parked_delivery_key,
                permits_parked_delivery_append,
            )

            checkpoint = state.checkpoints.get(target.id)
            wait_key = parked_delivery_key(
                checkpoint, session_id=target.id, instance_id=target.instance_id
            )
            if wait_key is not None:
                parked_target = permits_parked_delivery_append(
                    request,
                    checkpoint,
                    state.session_operation_records.get(target.id, {}).get(wait_key),
                    session_id=target.id,
                    instance_id=target.instance_id,
                    run_epoch=target.run_epoch,
                )
        if target is not None and target.status not in {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        }:
            from cayu.runtime._session_message_queue import require_open_admission

            require_open_admission(target.status, state.checkpoints.get(target.id))
        target_binding = (
            None if target_id is None else state.participant_session_bindings.get(target_id)
        )
        target_binding_valid = target_binding is not None and (
            target_binding.session_instance_id == target_instance
            and target_binding.participant.participant_id == request.append_key.consumer_id
            and target_binding.participant.incarnation
            == request.append_key.consumer_participant_incarnation
        )
        source_binding = state.participant_session_bindings.get(
            request.occurrence.sender_session_id
        )
        source_valid = (
            source_binding is not None
            and source_binding.session_instance_id == request.occurrence.sender_session_instance_id
            and source_binding.participant.participant_id
            == request.occurrence.sender_participant_id
            and source_binding.participant.incarnation
            == request.occurrence.sender_participant_incarnation
        )
        expired = (
            int(state.ownership_clock().timestamp() * 1000) >= request.attempt_key.deadline_at_ms
        )
        from cayu.storage._peer_attempts import qualify, require_capacity

        outstanding = 0
        for other_key, other in state.peer_content_receipts.items():
            if other_key == key or (
                other.append_key.consumer_id != request.append_key.consumer_id
                or other.append_key.consumer_participant_incarnation
                != request.append_key.consumer_participant_incarnation
            ):
                continue
            if other.status == "pending":
                outstanding += 1
            elif other.status == "appended":
                if (other.target_session_id, other.target_session_instance_id) in (
                    state.peer_content_deleted_targets
                ):
                    continue
                # Appended receipts validate that their target identity is present.
                queued = state.queued_session_messages_by_idempotency.get(
                    cast("str", other.target_session_id), {}
                ).get(other.operation_key)
                if queued is None or queued.status == SessionMessageQueueStatus.QUEUED:
                    outstanding += 1
        if expired:
            receipt = PeerContentReceipt(
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
                target is not None
                and not parked_target
                and target.status
                in {
                    SessionStatus.COMPLETED,
                    SessionStatus.FAILED,
                    SessionStatus.INTERRUPTED,
                }
            )
        ):
            receipt = PeerContentReceipt(
                operation_key=request.operation_key,
                append_key=request.append_key,
                attempt_generation=request.attempt_key.attempt_generation,
                status="excluded",
                reason="source_or_target_unavailable",
            )
        elif target is None:
            receipt = PeerContentReceipt(
                operation_key=request.operation_key,
                append_key=request.append_key,
                attempt_generation=request.attempt_key.attempt_generation,
                status="pending",
                reason="target_not_created",
            )
        elif target_binding is None:
            receipt = PeerContentReceipt(
                operation_key=request.operation_key,
                append_key=request.append_key,
                attempt_generation=request.attempt_key.attempt_generation,
                status="pending",
                reason="target_binding_pending",
            )
        elif not target_binding_valid or target.instance_id != target_instance:
            receipt = PeerContentReceipt(
                operation_key=request.operation_key,
                append_key=request.append_key,
                attempt_generation=request.attempt_key.attempt_generation,
                status="excluded",
                reason="target_binding_mismatch",
            )
        elif (
            isinstance(state.checkpoints.get(target.id), dict)
            and tool_publication_rules._PENDING_TOOL_ROUND_CHECKPOINT_KEY
            in state.checkpoints[target.id]
        ):
            receipt = PeerContentReceipt(
                operation_key=request.operation_key,
                append_key=request.append_key,
                attempt_generation=request.attempt_key.attempt_generation,
                status="pending",
                reason="target_busy",
            )
        elif (
            target.run_epoch != request.attempt_key.target_run_epoch
            or len(state.transcripts.get(target.id, ())) != target_cursor
        ):
            receipt = PeerContentReceipt(
                operation_key=request.operation_key,
                append_key=request.append_key,
                attempt_generation=request.attempt_key.attempt_generation,
                status="pending",
                reason="target_cursor_changed",
            )
        else:
            require_capacity(outstanding)
            qualify(qualify_target, target)
            peer_message = Message(
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
            if any(
                item.queue_id == queue_id or item.idempotency_key == request.operation_key
                for item in state.queued_session_messages_by_idempotency.get(target_id, {}).values()
            ):
                raise PeerContentConflict("Peer queue identity already has another authority.")
            delivery_mode = (
                SessionMessageDeliveryMode.ON_IDLE
                if request.wake_policy == "ordinary_continuation"
                else SessionMessageDeliveryMode.NEXT_TURN
            )
            accepted_at = state.ownership_clock()
            ordering_key = state.ordering.next_key
            accepted_event = Event(
                type=EventType.SESSION_MESSAGE_QUEUED,
                session_id=target.id,
                agent_name=target.agent_name,
                environment_name=target.environment_name,
                timestamp=accepted_at,
                payload={
                    **session_messaging._queued_session_message_event_payload(
                        queue_id=queue_id,
                        delivery_mode=delivery_mode,
                        ordering_key=ordering_key,
                        actor=None,
                        run_epoch=target.run_epoch,
                        transcript_cursor=len(state.transcripts.get(target.id, [])),
                    ),
                    "peer_occurrence_id": request.occurrence.occurrence_id,
                    "peer_provenance_sha256": request.occurrence.provenance_sha256,
                },
            )
            queued_message = SessionQueuedMessage(
                queue_id=queue_id,
                session_id=target.id,
                idempotency_key=request.operation_key,
                content=request.occurrence.payload.text,
                message=peer_message,
                delivery_mode=delivery_mode,
                status=SessionMessageQueueStatus.QUEUED,
                ordering_key=ordering_key,
                accepted_run_epoch=target.run_epoch,
                accepted_transcript_cursor=len(state.transcripts.get(target.id, [])),
                accepted_event_id=accepted_event.id,
                accepted_at=accepted_at,
                conditions=SessionMessageConditions(),
            )
            state.sessions[target.id] = state.append_events(target, [accepted_event])
            state.queued_session_messages_by_idempotency.setdefault(target.id, {})[
                queued_message.idempotency_key
            ] = queued_message
            state.pending_session_messages.setdefault((target.id, delivery_mode), deque()).append(
                queued_message
            )
            state.ordering.next_key += 1
            receipt = PeerContentReceipt(
                operation_key=request.operation_key,
                append_key=request.append_key,
                attempt_generation=request.attempt_key.attempt_generation,
                status="appended",
                target_session_id=target_id,
                target_session_instance_id=target_instance,
                occurrence=request.occurrence,
                queue_id=queue_id,
            )
        if receipt.status == "pending":
            require_capacity(outstanding)
        state.peer_content_receipts[key] = receipt
        state.peer_content_operation_commitments[key] = commitment
        state.peer_content_operations[request.operation_key] = commitment
        state.peer_content_requests[key] = request.model_copy(deep=True)
        state.peer_content_attempts[request.operation_key] = (
            request.model_dump_json(),
            receipt.model_dump_json(),
        )
        return receipt.model_copy(deep=True)


async def read_peer_content_attempt(
    locked: LockedPeerContent, request: PeerContentAppendRequest
) -> PeerContentReceipt | None:
    from cayu.storage._peer_attempts import exact_read

    request = PeerContentAppendRequest.model_validate(request)
    async with locked() as state:
        return exact_read(request, state.peer_content_attempts.get(request.operation_key))


async def list_pending_peer_content(
    locked: LockedPeerContent, *, after_operation_key=None, limit=32
):
    """Trusted receiving-owner discovery, independent of creation settlement."""
    from cayu.collaboration.peer_content import validate_peer_discovery

    validate_peer_discovery(after_operation_key, limit)
    async with locked() as state:
        keys = sorted(
            state.peer_content_requests,
            key=lambda key: state.peer_content_requests[key].operation_key,
        )
        return tuple(
            state.peer_content_requests[key].model_copy(deep=True)
            for key in keys
            if state.peer_content_receipts[key].status == "pending"
            and state.peer_content_requests[key].operation_key > (after_operation_key or "")
        )[:limit]


async def read_peer_content(
    locked: LockedPeerContent, append_key: PeerAppendKey
) -> PeerContentReceipt | None:
    if type(append_key) is not PeerAppendKey:
        raise TypeError("append_key must be a PeerAppendKey.")
    append_key = PeerAppendKey.model_validate(append_key)
    async with locked() as state:
        value = state.peer_content_receipts.get(append_key.model_dump_json())
        return None if value is None else value.model_copy(deep=True)


async def record_peer_content_exposure(
    locked: LockedPeerContent, request: PeerContentExposureRequest
) -> PeerContentExposureReceipt:
    if type(request) is not PeerContentExposureRequest:
        raise TypeError("Exposure requires a PeerContentExposureRequest.")
    key = request.exposure_id
    commitment = request.model_dump_json()
    async with locked() as state:
        existing = state.peer_content_exposures.get(key)
        if existing is None:
            existing = next(
                (
                    value
                    for value in state.peer_content_exposures.values()
                    if value[1].operation_key == request.operation_key
                ),
                None,
            )
        if existing is not None:
            expected = (
                request.identity_commitment() if existing[1].outcome == "pending" else commitment
            )
            actual = existing[0]
            if actual != expected:
                raise PeerContentConflict()
            if existing[1].outcome == "pending":
                receipt = PeerContentExposureReceipt(
                    operation_key=request.operation_key,
                    append_key=request.append_key,
                    exposure_id=request.exposure_id,
                    model_attempt_id=request.model_attempt_id,
                    outcome=request.outcome,
                    reason=request.reason,
                )
                state.peer_content_exposures[key] = (commitment, receipt)
                return receipt
            return existing[1].model_copy(update={"replayed": True})
        append = state.peer_content_receipts.get(request.append_key.model_dump_json())
        if append is None or append.status != "appended":
            raise PeerContentUnavailable("Peer content was not durably appended.")
        receipt = PeerContentExposureReceipt(
            operation_key=request.operation_key,
            append_key=request.append_key,
            exposure_id=request.exposure_id,
            model_attempt_id=request.model_attempt_id,
            outcome=request.outcome,
            reason=request.reason,
        )
        state.peer_content_exposures[key] = (commitment, receipt)
        return receipt.model_copy(deep=True)


async def begin_peer_content_exposure(
    locked: LockedPeerContent, request: PeerContentExposureRequest
) -> PeerContentExposureReceipt:
    if type(request) is not PeerContentExposureRequest:
        raise TypeError("Exposure requires a PeerContentExposureRequest.")
    identity = request.identity_commitment()
    key = request.exposure_id
    async with locked() as state:
        existing = state.peer_content_exposures.get(key)
        if existing is None:
            existing = next(
                (
                    value
                    for value in state.peer_content_exposures.values()
                    if value[1].operation_key == request.operation_key
                ),
                None,
            )
        if existing is not None:
            if existing[1].outcome == "pending" and existing[0] == identity:
                return existing[1].model_copy(update={"replayed": True})
            if existing[1].outcome != "pending" and existing[0] == request.model_dump_json():
                return existing[1].model_copy(update={"replayed": True})
            raise PeerContentConflict()
        append = state.peer_content_receipts.get(request.append_key.model_dump_json())
        if append is None or append.status != "appended":
            raise PeerContentUnavailable("Peer content was not durably appended.")
        receipt = PeerContentExposureReceipt(
            operation_key=request.operation_key,
            append_key=request.append_key,
            exposure_id=request.exposure_id,
            model_attempt_id=request.model_attempt_id,
            outcome="pending",
        )
        state.peer_content_exposures[key] = (identity, receipt)
        return receipt.model_copy(deep=True)


async def read_peer_content_exposure(
    locked: LockedPeerContent, append_key: PeerAppendKey, exposure_id: str
) -> PeerContentExposureReceipt | None:
    if type(append_key) is not PeerAppendKey or type(exposure_id) is not str:
        raise TypeError("Invalid exposure lookup.")
    append_key = PeerAppendKey.model_validate(append_key)
    async with locked() as state:
        value = state.peer_content_exposures.get(exposure_id)
        if value is None or value[1].append_key != append_key:
            return None
        return value[1].model_copy(deep=True)


async def exclude_peer_content(
    locked: LockedPeerContent,
    request: PeerContentAppendRequest,
    *,
    reason: str,
) -> PeerContentReceipt:
    from cayu.collaboration.peer_content import PeerContentReceipt

    if type(request) is not PeerContentAppendRequest:
        raise TypeError("Peer exclusion requires a PeerContentAppendRequest.")
    request = PeerContentAppendRequest.model_validate(request)
    reason = require_clean_nonblank(reason, "reason")
    from cayu.collaboration.peer_content import resolve_peer_target

    key = request.append_key.model_dump_json()
    async with locked() as state:
        creation = request.append_key.creation_target
        resolve_peer_target(
            request.append_key,
            None if creation is None else state.session_creation_decisions.get(creation.key),
        )
        commitment = request.model_dump_json()
        from cayu.storage._peer_attempts import historical_replay

        replay = historical_replay(request, state.peer_content_attempts.get(request.operation_key))
        if replay is not None:
            if replay.status == "excluded" and replay.reason != reason:
                raise PeerContentConflict()
            return replay
        prior_operation = state.peer_content_operations.get(request.operation_key)
        if prior_operation is not None and prior_operation != commitment:
            raise PeerContentConflict()
        existing = state.peer_content_receipts.get(key)
        if existing is not None:
            if state.peer_content_operation_commitments[key] != commitment:
                raise PeerContentConflict()
            if existing.status == "excluded" and existing.reason != reason:
                raise PeerContentConflict()
            if existing.status != "pending":
                return existing.model_copy(deep=True).model_copy(update={"replayed": True})
        receipt = PeerContentReceipt(
            operation_key=request.operation_key,
            append_key=request.append_key,
            attempt_generation=request.attempt_key.attempt_generation,
            status="excluded",
            reason=reason,
        )
        state.peer_content_receipts[key] = receipt
        state.peer_content_operation_commitments[key] = commitment
        state.peer_content_operations[request.operation_key] = commitment
        state.peer_content_requests[key] = request.model_copy(deep=True)
        state.peer_content_attempts[request.operation_key] = (
            request.model_dump_json(),
            receipt.model_dump_json(),
        )
        return receipt.model_copy(deep=True)


async def retry_pending_peer_content(
    locked: LockedPeerContent,
    session_id: str,
    *,
    expected_session_instance_id: str,
    expected_run_epoch: int,
    expected_transcript_cursor: int,
    admit=None,
) -> tuple[PeerContentReceipt, ...]:
    if type(session_id) is not str or type(expected_session_instance_id) is not str:
        raise TypeError("Peer retry identity values must be strings.")
    from cayu.collaboration.peer_content import resolve_peer_target

    def target_matches(request):
        creation = request.append_key.creation_target
        target_id, instance, _ = resolve_peer_target(
            request.append_key,
            None if creation is None else state.session_creation_decisions.get(creation.key),
        )
        return target_id == session_id and instance == expected_session_instance_id

    async with locked() as state:
        requests = [
            request.model_copy(deep=True)
            for key, request in state.peer_content_requests.items()
            if target_matches(request)
            and state.peer_content_receipts.get(key) is not None
            and state.peer_content_receipts[key].status == "pending"
        ][:SESSION_MESSAGE_DELIVERY_BATCH_LIMIT]
    results: list[PeerContentReceipt] = []
    for request in requests:
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
