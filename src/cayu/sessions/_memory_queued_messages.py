"""Complete in-memory queued-message operations over explicit locked state."""

from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, NamedTuple, Protocol
from uuid import uuid4

from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.approvals.tools import ResolutionActor
from cayu.events import Event, EventType, event_with_runtime_payload_authority
from cayu.messages import Message, detach_message
from cayu.sessions import _closure_records as closure_records
from cayu.sessions import _interaction_receipts as interaction_receipts
from cayu.sessions import _interaction_transitions as interaction_rules
from cayu.sessions import _invocation_scope as invocation_scope
from cayu.sessions import _memory_transcripts as memory_transcripts
from cayu.sessions import _model_stage_rules as model_stage_rules
from cayu.sessions import messaging as session_messaging
from cayu.sessions._access_rules import require_resource_session
from cayu.sessions._execution_profile_checkpoint import ActiveInvocationExecutionProfile
from cayu.sessions._interaction_transitions import QueuedInteractionProfileHandoff
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    ActiveModelCompletionStage,
    ModelCompletionStageDispatch,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.messaging import (
    SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    EnqueueSessionMessageRequest,
    EnqueueSessionMessageResult,
    SessionMessageActionRequest,
    SessionMessageActionResult,
    SessionMessageConflict,
    SessionMessageDeliveryBatch,
    SessionMessageDeliveryMode,
    SessionMessageInspection,
    SessionMessageInspectionRecord,
    SessionMessageQuery,
    SessionMessageQueueStatus,
    SessionMessageSource,
    SessionQueuedMessage,
    copy_enqueue_session_message_request,
    copy_session_message_action,
    enqueue_session_message_input,
    queued_session_message_input,
    session_message_checkpoint_sha256,
    session_message_rejection,
)
from cayu.sessions.records import (
    EventRecord,
    Session,
    SessionStatus,
    SessionStatusConflict,
    TranscriptRecord,
)
from cayu.sessions.transcript_input import (
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    session_messages_input_contract_evidence,
)
from cayu.sessions.transcript_queries import TranscriptSnapshot, fork_source_transcript_sha256


@dataclass(slots=True)
class MemoryMessageOrdering:
    """Next global queue key, shared by both admission owners under one lock."""

    next_key: int = 1


class RejectNewWorkAfterSteering(Protocol):
    def __call__(self, session: Session, *, allow_completed_interaction: bool = False) -> None: ...


@dataclass(frozen=True)
class MemoryMessageDeliveryRecord:
    session_id: str
    include_on_idle: bool
    requested_eligible_through: int | None
    limit: int
    interaction_id: str | None
    interaction_started_event: Event | None
    profile_handoff: QueuedInteractionProfileHandoff | None
    batch: SessionMessageDeliveryBatch
    reject_only: bool = False


def _audit_resolution_actor(actor: ResolutionActor | None) -> ResolutionActor | None:
    if actor is None:
        return None
    return ResolutionActor(
        subject=actor.subject,
        tenant=actor.tenant,
        source=actor.source,
    )


def _validate_equivalent_message_delivery(
    existing: MemoryMessageDeliveryRecord,
    *,
    session_id: str,
    include_on_idle: bool,
    eligible_through: int | None,
    limit: int,
    interaction_id: str | None,
    interaction_started_event: Event | None,
    profile_handoff: QueuedInteractionProfileHandoff | None,
    reject_only: bool = False,
) -> None:
    if (
        existing.session_id != session_id
        or existing.include_on_idle != include_on_idle
        or existing.requested_eligible_through != eligible_through
        or existing.limit != limit
        or existing.interaction_id != interaction_id
        or existing.interaction_started_event != interaction_started_event
        or existing.profile_handoff != profile_handoff
        or existing.reject_only != reject_only
    ):
        raise ValueError("delivery_id was already used for a different queue delivery.")


class MemoryQueuedMessageState(NamedTuple):
    """Current message indexes and synchronous publication capabilities.

    The caller resolves these bindings after acquiring its lock on every
    operation. Authorization and publication stay inside that lock. Callbacks
    must finish synchronously and must not acquire the lock again. The ordering
    record is shared by ordinary and peer admission; advance it only after
    acceptance publication succeeds.
    """

    append_events: Callable[[Session, list[Event]], Session]
    checkpoints: dict[str, dict[str, Any]]
    event_records_by_id: dict[tuple[str, str], EventRecord]
    ordering: MemoryMessageOrdering
    ownership_clock: Callable[[], datetime]
    pending_session_messages: dict[
        tuple[str, SessionMessageDeliveryMode], deque[SessionQueuedMessage]
    ]
    queued_session_messages_by_idempotency: dict[str, dict[str, SessionQueuedMessage]]
    reject_new_work_after_steering: RejectNewWorkAfterSteering
    session_closure_progress: dict[tuple[str, str], dict[str, Any]]
    session_message_acceptance_event_ids: dict[tuple[str, str], set[str]]
    session_message_delivery_records: dict[str, MemoryMessageDeliveryRecord]
    session_message_terminal_receipts: dict[tuple[str, str], dict[str, Any]]
    session_operation_records: dict[str, dict[str, dict[str, Any]]]
    sessions: dict[str, Session]
    store_checkpoint: Callable[[str, dict[str, Any]], None]
    transcript_indexes: Callable[[], memory_transcripts.MemoryTranscriptIndexes]
    transcript_interaction_ids: dict[str, list[str | None]]
    transcripts: dict[str, list[Message]]


LockedQueuedMessages = Callable[[], AbstractAsyncContextManager[MemoryQueuedMessageState]]


def _session_message_source_unlocked(
    state: MemoryQueuedMessageState,
    session_id: str,
    *,
    include_transcript_digest: bool,
    include_checkpoint_digest: bool,
) -> SessionMessageSource:
    session = state.sessions.get(session_id)
    if session is None:
        raise KeyError("Session-message source not found.")
    messages = state.transcripts.get(session_id, [])
    transcript = None
    if include_transcript_digest:
        transcript = TranscriptSnapshot(
            records=[
                TranscriptRecord(index=index, message=detach_message(message))
                for index, message in enumerate(messages)
            ],
            cursor=len(messages),
        )
    return SessionMessageSource(
        session_id=session.id,
        session_instance_id=session.instance_id,
        run_epoch=session.run_epoch,
        transcript_cursor=len(messages),
        transcript_sha256=(
            None if transcript is None else fork_source_transcript_sha256(transcript)
        ),
        checkpoint_sha256=(
            session_message_checkpoint_sha256(state.checkpoints.get(session_id))
            if include_checkpoint_digest
            else None
        ),
    )


def _session_message_accepted_event_unlocked(
    state: MemoryQueuedMessageState,
    raw: dict[str, Any],
    *,
    quarantine: bool = False,
) -> Event | None:
    if quarantine:
        ids = state.session_message_acceptance_event_ids.get(
            (raw["session_id"], raw["queue_id"]), set()
        )
        if len(ids) != 1:
            raise SessionMessageConflict()
        record = state.event_records_by_id.get((raw["session_id"], next(iter(ids))))
        return None if record is None else record.event
    event_id = raw["accepted_event_id"]
    if type(event_id) is not str or len(event_id) > 512:
        raise SessionMessageConflict()
    record = state.event_records_by_id.get((raw["session_id"], event_id))
    return None if record is None else record.event


def _raw_session_message_unlocked(
    state: MemoryQueuedMessageState, message: SessionQueuedMessage
) -> dict[str, Any]:
    raw = message.model_dump(mode="json", warnings=False)
    raw["terminal_json"] = state.session_message_terminal_receipts.get(
        (message.session_id, message.queue_id)
    )
    return raw


def _inspect_session_message_unlocked(
    state: MemoryQueuedMessageState, message: SessionQueuedMessage
) -> SessionMessageInspectionRecord:
    from cayu.runtime._session_message_queue import inspect_record

    raw = _raw_session_message_unlocked(state, message)
    return inspect_record(
        raw,
        lambda: SessionQueuedMessage.model_validate(
            {key: value for key, value in raw.items() if key != "terminal_json"}
        ),
    )


async def inspect_session_messages(
    locked: LockedQueuedMessages,
    query: SessionMessageQuery,
    *,
    expected_authorized_session_instance_id: str | None = None,
) -> SessionMessageInspection:
    from cayu.runtime._session_message_queue import (
        copy_inspection_query,
        inspection_boundary,
        inspection_next_cursor,
        inspection_priority,
        require_authorized_session_instance,
    )

    query = copy_inspection_query(query)
    async with locked() as state:
        session = state.sessions.get(query.session_id)
        require_resource_session(session, "read")
        if session is None:
            raise KeyError("Session-message target not found.")
        require_authorized_session_instance(session, expected_authorized_session_instance_id)
        candidates = state.queued_session_messages_by_idempotency.get(session.id, {}).values()
        boundary = inspection_boundary(
            session,
            query.cursor,
            max((message.ordering_key for message in candidates), default=0)
            if query.cursor is None
            else 0,
        )
        messages = heapq.nsmallest(
            query.limit + 1,
            (
                message
                for message in candidates
                if message.ordering_key <= boundary.through_ordering_key
                and (inspection_priority(message.delivery_mode), message.ordering_key)
                > (boundary.after_priority, boundary.after_ordering_key)
            ),
            key=lambda message: (
                inspection_priority(message.delivery_mode),
                message.ordering_key,
            ),
        )
        records = tuple(
            _inspect_session_message_unlocked(state, message) for message in messages[: query.limit]
        )
        return SessionMessageInspection(
            session_id=session.id,
            session_instance_id=session.instance_id,
            records=records,
            next_cursor=(
                inspection_next_cursor(
                    boundary,
                    inspection_priority(messages[query.limit - 1].delivery_mode),
                    records[-1].ordering_key,
                )
                if len(messages) > query.limit
                else None
            ),
        )


async def apply_session_message_action(
    locked: LockedQueuedMessages, request: SessionMessageActionRequest
) -> SessionMessageActionResult:
    from cayu.runtime._session_message_queue import (
        raw_revision,
        replay_action,
        terminal_event,
        terminal_receipt,
    )

    request = copy_session_message_action(request)
    async with locked() as state:
        session = state.sessions.get(request.session_id)
        require_resource_session(session, "modify")
        if session is None or session.instance_id != request.session_instance_id:
            raise SessionMessageConflict()
        messages = state.queued_session_messages_by_idempotency.get(session.id, {})
        message = next(
            (item for item in messages.values() if item.queue_id == request.queue_id), None
        )
        if message is None:
            raise SessionMessageConflict()
        raw = _raw_session_message_unlocked(state, message)
        accepted_event = _session_message_accepted_event_unlocked(
            state,
            raw,
            quarantine=request.action == "quarantine",
        )
        replay = replay_action(raw, request, accepted_event=accepted_event)
        if replay is not None:
            return SessionMessageActionResult(
                record=_inspect_session_message_unlocked(state, message),
                event=replay,
                replayed=True,
            )
        for owner in state.session_closure_progress.values():
            closure_records._check_closure_lineage_owner(owner, (session.id,))
        if (
            raw_revision(raw) != request.expected_revision
            or message.status != SessionMessageQueueStatus.QUEUED
            or message.delivered_event_id is not None
            or any(
                delivered.queue_id == message.queue_id
                for delivery in state.session_message_delivery_records.values()
                if delivery.session_id == session.id
                for delivered in delivery.batch.messages
            )
        ):
            raise SessionMessageConflict()
        if (
            request.action == "withdraw"
            and _inspect_session_message_unlocked(state, message).validity != "valid"
        ):
            raise SessionMessageConflict()
        status = (
            SessionMessageQueueStatus.WITHDRAWN
            if request.action == "withdraw"
            else SessionMessageQueueStatus.QUARANTINED
        )
        event = terminal_event(
            session,
            raw,
            status,
            state.ownership_clock(),
            actor=request.requested_by,
            accepted_event=accepted_event,
        )
        updated = message.model_copy(update={"status": status}, deep=True)
        updated_session = state.append_events(session, [event])
        state.session_message_terminal_receipts[(session.id, message.queue_id)] = terminal_receipt(
            status, event, request
        )
        messages[message.idempotency_key] = updated
        for mode in SessionMessageDeliveryMode:
            key = (session.id, mode)
            pending = state.pending_session_messages.get(key)
            if pending is not None:
                remaining = deque(item for item in pending if item.queue_id != message.queue_id)
                if remaining:
                    state.pending_session_messages[key] = remaining
                else:
                    del state.pending_session_messages[key]
        state.sessions[session.id] = updated_session
        return SessionMessageActionResult(
            record=_inspect_session_message_unlocked(state, updated), event=event
        )


async def snapshot_session_message_source(
    locked: LockedQueuedMessages,
    session_id: str,
    *,
    include_transcript_digest: bool = False,
    include_checkpoint_digest: bool = False,
    expected_authorized_session_instance_id: str | None = None,
) -> SessionMessageSource:
    from cayu.runtime._session_message_queue import require_authorized_session_instance

    session_id = require_clean_nonblank(session_id, "session_id")
    if type(include_transcript_digest) is not bool or type(include_checkpoint_digest) is not bool:
        raise TypeError("Source snapshot digest flags must be bools.")
    async with locked() as state:
        session = state.sessions.get(session_id)
        require_resource_session(session, "read")
        if session is None:
            raise KeyError("Session-message source not found.")
        require_authorized_session_instance(session, expected_authorized_session_instance_id)
        return _session_message_source_unlocked(
            state,
            session_id,
            include_transcript_digest=include_transcript_digest,
            include_checkpoint_digest=include_checkpoint_digest,
        )


async def enqueue_session_message(
    locked: LockedQueuedMessages,
    request: EnqueueSessionMessageRequest,
    *,
    expected_authorized_target_instance_id: str | None = None,
) -> EnqueueSessionMessageResult:
    request = copy_enqueue_session_message_request(request)
    async with locked() as state:
        session = state.sessions.get(request.session_id)
        require_resource_session(session, "modify")
        if session is None:
            raise KeyError(f"Session not found: {request.session_id}")
        if expected_authorized_target_instance_id is not None and (
            type(expected_authorized_target_instance_id) is not str
            or session.instance_id != expected_authorized_target_instance_id
        ):
            raise SessionMessageConflict()
        if request.conditions.source is not None:
            source_session = state.sessions.get(request.conditions.source.session_id)
            require_resource_session(source_session, "read")
            if (
                source_session is None
                or source_session.instance_id != request.conditions.source.session_instance_id
            ):
                raise SessionMessageConflict()
        messages_by_idempotency = state.queued_session_messages_by_idempotency.get(
            request.session_id
        )
        existing = (
            None
            if messages_by_idempotency is None
            else messages_by_idempotency.get(request.idempotency_key)
        )
        if existing is not None:
            session_messaging._validate_equivalent_queued_session_message(existing, request)
            record = state.event_records_by_id.get((request.session_id, existing.accepted_event_id))
            if record is None:
                raise RuntimeError(
                    "Queued session message is missing its durable acceptance event."
                )
            return EnqueueSessionMessageResult(
                message=existing,
                event=record.event,
                replayed=True,
            )
        for owner in state.session_closure_progress.values():
            closure_records._check_closure_lineage_owner(owner, (request.session_id,))
        from cayu.runtime._session_message_queue import require_open_admission

        require_open_admission(session.status, state.checkpoints.get(session.id))
        accepted_at = state.ownership_clock()
        queue_id = str(uuid4())
        source = request.conditions.source
        if source is not None and source != _session_message_source_unlocked(
            state,
            source.session_id,
            include_transcript_digest=source.transcript_sha256 is not None,
            include_checkpoint_digest=source.checkpoint_sha256 is not None,
        ):
            raise SessionMessageConflict()
        ordering_key = state.ordering.next_key
        accepted_transcript_cursor = len(state.transcripts.get(session.id, []))
        accepted_message = enqueue_session_message_input(request)
        from cayu.runtime._session_message_queue import source_event_payload

        accepted_event = event_with_runtime_payload_authority(
            Event(
                type=EventType.SESSION_MESSAGE_QUEUED,
                session_id=session.id,
                agent_name=session.agent_name,
                environment_name=session.environment_name,
                timestamp=accepted_at,
                payload={
                    **session_messaging._queued_session_message_event_payload(
                        queue_id=queue_id,
                        delivery_mode=request.delivery_mode,
                        ordering_key=ordering_key,
                        actor=request.requested_by,
                        run_epoch=session.run_epoch,
                        transcript_cursor=accepted_transcript_cursor,
                    ),
                    **source_event_payload(request.conditions.source),
                    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY: (
                        session_messages_input_contract_evidence(
                            (accepted_message,),
                            message_start_index=accepted_transcript_cursor,
                            redactions_applied=request._input_redactions_applied,
                            structured_output_requested=False,
                        )
                    ),
                },
            ),
            SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
        )
        queued_message = SessionQueuedMessage(
            queue_id=queue_id,
            session_id=session.id,
            idempotency_key=request.idempotency_key,
            content=request.content,
            message=request.message,
            delivery_mode=request.delivery_mode,
            status=SessionMessageQueueStatus.QUEUED,
            ordering_key=ordering_key,
            accepted_run_epoch=session.run_epoch,
            accepted_transcript_cursor=accepted_transcript_cursor,
            accepted_event_id=accepted_event.id,
            accepted_at=accepted_at,
            requested_by=_audit_resolution_actor(request.requested_by),
            conditions=request.conditions,
        )
        state.sessions[session.id] = state.append_events(
            session,
            [accepted_event],
        )
        state.queued_session_messages_by_idempotency.setdefault(session.id, {})[
            queued_message.idempotency_key
        ] = queued_message
        pending_key = (session.id, queued_message.delivery_mode)
        state.pending_session_messages.setdefault(pending_key, deque()).append(queued_message)
        state.ordering.next_key += 1
        return EnqueueSessionMessageResult(
            message=queued_message,
            event=accepted_event,
        )


async def deliver_queued_session_messages(
    locked: LockedQueuedMessages,
    session_id: str,
    *,
    include_on_idle: bool,
    delivery_id: str | None = None,
    eligible_through: int | None = None,
    limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    interaction_id: str | None = None,
    interaction_started_event: Event | None = None,
    profile_handoff: QueuedInteractionProfileHandoff | None = None,
    reject_only: bool = False,
) -> SessionMessageDeliveryBatch:
    session_id = require_clean_nonblank(session_id, "session_id")
    delivery_id = (
        str(uuid4()) if delivery_id is None else require_clean_nonblank(delivery_id, "delivery_id")
    )
    if type(include_on_idle) is not bool:
        raise TypeError("include_on_idle must be a bool.")
    if type(reject_only) is not bool:
        raise TypeError("reject_only must be a bool.")
    if interaction_id is not None:
        interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
    interaction_started_event = interaction_rules._copy_queued_interaction_started_event(
        session_id,
        interaction_id,
        interaction_started_event,
    )
    profile_handoff = interaction_rules._copy_queued_interaction_profile_handoff(
        session_id,
        delivery_id,
        interaction_id,
        interaction_started_event,
        profile_handoff,
    )
    eligible_through = session_messaging._validate_message_delivery_eligible_through(
        eligible_through
    )
    if type(limit) is not int or not 1 <= limit <= SESSION_MESSAGE_DELIVERY_BATCH_LIMIT:
        raise ValueError(f"limit must be between 1 and {SESSION_MESSAGE_DELIVERY_BATCH_LIMIT}.")
    async with locked() as state:
        session = state.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        invocation_scope._assert_session_run_epoch(session_id, session)
        existing_delivery = state.session_message_delivery_records.get(delivery_id)
        if existing_delivery is not None:
            _validate_equivalent_message_delivery(
                existing_delivery,
                session_id=session_id,
                include_on_idle=include_on_idle,
                eligible_through=eligible_through,
                limit=limit,
                interaction_id=interaction_id,
                interaction_started_event=interaction_started_event,
                profile_handoff=profile_handoff,
                reject_only=reject_only,
            )
            replayed_batch = existing_delivery.batch.model_copy(
                update={"replayed": True},
                deep=True,
            )
            if replayed_batch.messages and profile_handoff is not None:
                operation_records = state.session_operation_records.get(session_id, {})
                receipt_record = operation_records.get(
                    interaction_receipts._interaction_transition_storage_key(
                        profile_handoff.predecessor_settlement_event_id
                    )
                )
                if receipt_record is None:
                    raise SessionRunFenced(
                        "Queued interaction handoff lost its predecessor settlement receipt."
                    )
                active_model_stage: ActiveModelCompletionStage | None = None
                stage_dispatch: ModelCompletionStageDispatch | None = None
                active_record = operation_records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
                if active_record is not None:
                    marker = model_stage_rules._reconstruct_active_model_completion_stage_record(
                        active_record,
                        session_id=session_id,
                    )
                    _, _, preparation_key, terminal_key = (
                        model_stage_rules._model_completion_stage_storage_identity(
                            session_id,
                            marker.stage_id,
                        )
                    )
                    active_model_stage = (
                        model_stage_rules._reconstruct_active_model_completion_stage(
                            active_record,
                            operation_records.get(preparation_key),
                            operation_records.get(terminal_key),
                            session_id=session_id,
                        )
                    )
                    dispatch_record = operation_records.get(
                        model_stage_rules._model_completion_stage_dispatch_storage_key(
                            marker.stage_id
                        )
                    )
                    if dispatch_record is not None:
                        stage_dispatch = model_stage_rules._reconstruct_model_completion_stage_dispatch(
                            dispatch_record,
                            session_id=session_id,
                            stage_id=marker.stage_id,
                            storage_key=model_stage_rules._model_completion_stage_dispatch_storage_key(
                                marker.stage_id
                            ),
                        )
                repaired_checkpoint = (
                    interaction_receipts._checkpoint_after_queued_interaction_profile_handoff(
                        session,
                        state.checkpoints.get(session_id),
                        profile_handoff,
                        settlement_record=receipt_record,
                        replayed_delivery=True,
                        active_model_stage=active_model_stage,
                        stage_dispatch=stage_dispatch,
                    )
                )
                state.store_checkpoint(session_id, repaired_checkpoint)
            return replayed_batch
        if session.status != SessionStatus.RUNNING:
            raise SessionStatusConflict(
                "Queued session messages may be delivered only while running."
            )
        boundary = state.ordering.next_key - 1 if eligible_through is None else eligible_through

        selected_key: tuple[str, SessionMessageDeliveryMode] | None = None
        selected: list[SessionQueuedMessage] = []
        delivered_at = state.ownership_clock()
        transcript_cursor = len(state.transcripts.get(session_id, []))

        def validated_message(message: SessionQueuedMessage) -> SessionQueuedMessage:
            # Reconstruct the complete record, as persistent stores do. Passing
            # the existing model to model_validate would trust mutated fields;
            # serializing it first could emit rejected values in warnings.
            return SessionQueuedMessage.model_validate(
                {name: getattr(message, name) for name in SessionQueuedMessage.model_fields}
            )

        def rejected_now(message: SessionQueuedMessage) -> bool:
            message = validated_message(message)
            return (
                session_message_rejection(
                    message.conditions,
                    session_instance_id=session.instance_id,
                    run_epoch=session.run_epoch,
                    transcript_cursor=transcript_cursor,
                    now=delivered_at,
                )
                is not None
            )

        delivery_modes: tuple[SessionMessageDeliveryMode, ...] = (
            SessionMessageDeliveryMode.NEXT_TURN,
        )
        if include_on_idle:
            delivery_modes += (SessionMessageDeliveryMode.ON_IDLE,)
        for delivery_mode in delivery_modes:
            pending_key = (session_id, delivery_mode)
            pending = state.pending_session_messages.get(pending_key)
            if pending is None or pending[0].ordering_key > boundary:
                continue
            selected_key = pending_key
            for message in pending:
                if len(selected) >= limit or message.ordering_key > boundary:
                    break
                message = validated_message(message)
                if reject_only and not rejected_now(message):
                    continue
                selected.append(message)
            if selected:
                break
        if not selected:
            batch = SessionMessageDeliveryBatch(
                delivery_id=delivery_id,
                interaction_id=interaction_id,
                eligible_through=boundary,
                has_more=False,
            )
            state.session_message_delivery_records[delivery_id] = MemoryMessageDeliveryRecord(
                session_id=session_id,
                include_on_idle=include_on_idle,
                requested_eligible_through=eligible_through,
                limit=limit,
                interaction_id=interaction_id,
                interaction_started_event=interaction_started_event,
                profile_handoff=profile_handoff,
                batch=batch,
                reject_only=reject_only,
            )
            return batch
        if selected_key is None:
            raise RuntimeError("Queued message selection lost its pending index.")

        from cayu.runtime._session_message_queue import (
            source_audit_payload,
            terminal_event,
            terminal_receipt,
        )

        selected_all = selected
        selected_ids = {message.queue_id for message in selected_all}

        def has_eligible(delivery_mode: SessionMessageDeliveryMode) -> bool:
            pending = state.pending_session_messages.get((session_id, delivery_mode), ())
            return any(
                message.queue_id not in selected_ids
                and message.ordering_key <= boundary
                and (not reject_only or rejected_now(message))
                for message in pending
            )

        # Lookahead can reject malformed rows. Finish it before publishing
        # any events or changing the pending index, including reject-only drains.
        has_more = has_eligible(SessionMessageDeliveryMode.NEXT_TURN) or (
            include_on_idle and has_eligible(SessionMessageDeliveryMode.ON_IDLE)
        )
        accepted_events = {
            message.queue_id: _session_message_accepted_event_unlocked(
                state, _raw_session_message_unlocked(state, message)
            )
            for message in selected_all
        }
        selected = []
        rejected_messages: list[SessionQueuedMessage] = []
        rejection_events: list[Event] = []
        rejected_receipts: dict[tuple[str, str], dict[str, Any]] = {}
        for queued_message in selected_all:
            rejection = session_message_rejection(
                queued_message.conditions,
                session_instance_id=session.instance_id,
                run_epoch=session.run_epoch,
                transcript_cursor=transcript_cursor + len(selected),
                now=delivered_at,
            )
            if rejection is None:
                selected.append(queued_message)
                continue
            event = terminal_event(
                session,
                _raw_session_message_unlocked(state, queued_message),
                rejection,
                delivered_at,
                accepted_event=accepted_events[queued_message.queue_id],
                actor=queued_message.requested_by,
                interaction_id=interaction_id,
            )
            rejection_events.append(event)
            rejected_messages.append(
                queued_message.model_copy(update={"status": rejection}, deep=True)
            )
            rejected_receipts[(session_id, queued_message.queue_id)] = terminal_receipt(
                rejection, event
            )

        rebound_checkpoint: dict[str, Any] | None = None
        if profile_handoff is not None and selected:
            state.reject_new_work_after_steering(session, allow_completed_interaction=True)
            receipt_record = state.session_operation_records.get(session_id, {}).get(
                interaction_receipts._interaction_transition_storage_key(
                    profile_handoff.predecessor_settlement_event_id
                )
            )
            if receipt_record is None:
                raise SessionRunFenced(
                    "Queued interaction handoff lost its predecessor settlement receipt."
                )
            rebound_checkpoint = (
                interaction_receipts._checkpoint_after_queued_interaction_profile_handoff(
                    session,
                    state.checkpoints.get(session_id),
                    profile_handoff,
                    settlement_record=receipt_record,
                    replayed_delivery=False,
                )
            )

        updated_messages: list[SessionQueuedMessage] = []
        delivery_events: list[Event] = []
        transcript_messages: list[Message] = []
        for offset, queued_message in enumerate(selected, start=1):
            delivered_cursor = transcript_cursor + offset
            delivered_message = queued_session_message_input(queued_message)
            delivery_event = event_with_runtime_payload_authority(
                Event(
                    type=EventType.SESSION_MESSAGE_DELIVERED,
                    session_id=session.id,
                    interaction_id=interaction_id,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                    timestamp=delivered_at,
                    payload={
                        **session_messaging._queued_session_message_event_payload(
                            queue_id=queued_message.queue_id,
                            delivery_mode=queued_message.delivery_mode,
                            ordering_key=queued_message.ordering_key,
                            actor=queued_message.requested_by,
                            run_epoch=session.run_epoch,
                            transcript_cursor=delivered_cursor,
                        ),
                        **source_audit_payload(
                            _raw_session_message_unlocked(state, queued_message),
                            accepted_events[queued_message.queue_id],
                        ),
                        "accepted_run_epoch": queued_message.accepted_run_epoch,
                        "accepted_transcript_cursor": (queued_message.accepted_transcript_cursor),
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
                        "delivered_run_epoch": session.run_epoch,
                        "delivered_transcript_cursor": delivered_cursor,
                        "delivered_event_id": delivery_event.id,
                        "delivered_at": delivered_at,
                    },
                    deep=True,
                )
            )
            delivery_events.append(delivery_event)
            transcript_messages.append(delivered_message)

        persisted_events = [
            *(
                [interaction_started_event]
                if interaction_started_event is not None and selected
                else []
            ),
            *sorted(
                [*delivery_events, *rejection_events],
                key=lambda event: event.payload["ordering_key"],
            ),
        ]
        batch = SessionMessageDeliveryBatch(
            messages=tuple(updated_messages),
            events=tuple(persisted_events),
            delivery_id=delivery_id,
            interaction_id=interaction_id,
            eligible_through=boundary,
            has_more=has_more,
            active_invocation_profile=(
                None
                if profile_handoff is None or not selected
                else profile_handoff.target_active_profile
            ),
        )
        delivery_record = MemoryMessageDeliveryRecord(
            session_id=session_id,
            include_on_idle=include_on_idle,
            requested_eligible_through=eligible_through,
            limit=limit,
            interaction_id=interaction_id,
            interaction_started_event=interaction_started_event,
            profile_handoff=profile_handoff,
            batch=batch,
            reject_only=reject_only,
        )
        selected_queue = deque(
            message
            for message in state.pending_session_messages[selected_key]
            if message.queue_id not in selected_ids
        )
        updated_session = state.append_events(session, persisted_events)
        state.transcripts.setdefault(session_id, []).extend(transcript_messages)
        memory_transcripts.extend_search(
            state.transcript_indexes(), session_id, transcript_messages
        )
        state.transcript_interaction_ids.setdefault(session_id, [])
        memory_transcripts.extend_attribution(
            state.transcript_indexes(),
            session_id,
            [interaction_id] * len(transcript_messages),
        )
        state.pending_session_messages[selected_key] = selected_queue
        if not selected_queue:
            del state.pending_session_messages[selected_key]
        messages_by_idempotency = state.queued_session_messages_by_idempotency[session_id]
        for updated_message in [*updated_messages, *rejected_messages]:
            messages_by_idempotency[updated_message.idempotency_key] = updated_message
        state.session_message_terminal_receipts.update(rejected_receipts)
        state.sessions[session_id] = updated_session
        if rebound_checkpoint is not None:
            state.store_checkpoint(session_id, rebound_checkpoint)
        state.session_message_delivery_records[delivery_id] = delivery_record
        return batch


async def repair_queued_interaction_profile_handoff(
    locked: LockedQueuedMessages,
    session_id: str,
    *,
    interaction_started_event: Event,
    profile_handoff: QueuedInteractionProfileHandoff,
) -> ActiveInvocationExecutionProfile:
    session_id = require_clean_nonblank(session_id, "session_id")
    interaction_started_event, profile_handoff = (
        interaction_rules._copy_historical_queued_interaction_profile_handoff(
            session_id,
            interaction_started_event,
            profile_handoff,
        )
    )
    target = profile_handoff.target_active_profile
    async with locked() as state:
        session = state.sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        invocation_scope._assert_session_run_epoch(session_id, session)
        delivery = state.session_message_delivery_records.get(target.interaction_id)
        if (
            delivery is None
            or delivery.session_id != session_id
            or delivery.interaction_id != target.interaction_id
            or delivery.interaction_started_event != interaction_started_event
            or not delivery.batch.messages
            or (
                delivery.profile_handoff is not None and delivery.profile_handoff != profile_handoff
            )
        ):
            raise SessionRunFenced(
                "Historical queued interaction handoff lacks its exact delivery receipt."
            )
        operation_records = state.session_operation_records.get(session_id, {})
        settlement_record = operation_records.get(
            interaction_receipts._interaction_transition_storage_key(
                profile_handoff.predecessor_settlement_event_id
            )
        )
        if settlement_record is None:
            raise SessionRunFenced(
                "Historical queued interaction handoff lost its predecessor settlement."
            )
        active_model_stage, stage_dispatch = (
            interaction_receipts._historical_queued_handoff_stage_from_records(
                session_id,
                operation_records,
            )
        )
        repaired_checkpoint = (
            interaction_receipts._checkpoint_after_queued_interaction_profile_handoff(
                session,
                state.checkpoints.get(session_id),
                profile_handoff,
                settlement_record=settlement_record,
                replayed_delivery=True,
                active_model_stage=active_model_stage,
                stage_dispatch=stage_dispatch,
            )
        )
        state.store_checkpoint(session_id, repaired_checkpoint)
        return target.model_copy(deep=True)
