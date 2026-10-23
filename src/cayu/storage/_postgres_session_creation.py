"""Native PostgreSQL session creation and fork persistence.

Operations retain admission, authority, initial state and event publication
in one native transaction. Store composition supplies explicit capabilities.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime
from typing import Any
from uuid import uuid4

from cayu._validation import copy_durable_json_object
from cayu.events import Event
from cayu.messages import Message
from cayu.session_metadata import inherited_display_metadata
from cayu.sessions import creation_fence
from cayu.sessions._checkpoint_preservation import (
    _checkpoint_transform_result_preserving_completion_result_event_publications,
    _copy_checkpoint_for_transform,
    _replace_checkpoint_preserving_completion_result_event_publications,
)
from cayu.sessions._checkpoint_publication import _initial_transcript_pending_checkpoint
from cayu.sessions._closure_records import _check_closure_lineage_owner
from cayu.sessions._interaction_transitions import _copy_optional_interaction_admission
from cayu.sessions._invocation_scope import _activate_session_run_fence
from cayu.sessions._model_stage import MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY
from cayu.sessions._run_identity import (
    RUN_LINEAGE_METADATA_KEY,
    run_lineage_for_creation,
    run_marks_for_creation,
)
from cayu.sessions._runtime_publication import (
    SessionOperationInitializer,
    _prepare_initial_session_operation_records,
)
from cayu.sessions.creation import session_invocation_for_run_request, session_metadata_for_creation
from cayu.sessions.forks import (
    FORK_TRANSCRIPT_VALIDATION_ERROR,
    ForkCheckpointAuthorityDecoder,
    ForkSystemPromptReplacement,
    SessionForkActiveModelStageConflict,
    SessionForkProfileRelationship,
    _prepare_profiled_fork_checkpoint_result,
    _prepare_session_fork_request,
    _profiled_fork_authority_validation_error,
    _validate_profiled_fork_authority,
    _validate_session_fork_source,
    apply_fork_system_prompt_replacement,
    transform_fork_checkpoint,
)
from cayu.sessions.records import (
    CheckpointTransform,
    Session,
    SessionIdentity,
    SessionStatus,
    TranscriptRecord,
    copy_session_identity,
)
from cayu.sessions.requests import (
    RunRequest,
    copy_run_request,
    deferred_interaction_input_for_run_request,
    session_instance_id_for_run_request,
)
from cayu.sessions.transcript_input import deferred_interaction_input_storage_payload
from cayu.sessions.transcript_queries import (
    ForkTranscriptValidator,
    TranscriptSnapshot,
    fork_transcript_is_accepted,
)
from cayu.storage import _creation_fence
from cayu.storage import _postgres_event_delivery as event_delivery_ops
from cayu.storage import _postgres_support as pg_support
from cayu.storage import _postgres_transcript as transcript_ops
from cayu.storage._postgres_transcript import AuthorityRegistrar, PostgresConnection


async def create(
    connect: PostgresConnection,
    request: RunRequest,
    *,
    identity: SessionIdentity,
    interaction_started_event: Event | None = None,
    interaction_source_messages: list[Message] | None = None,
    checkpoint_transform: CheckpointTransform | None = None,
    result_checkpoint_transform: CheckpointTransform | None = None,
    operation_initializer: SessionOperationInitializer | None = None,
    participant_binding_factory: Callable[[Session], tuple[Any, Any]] | None = None,
    participant_request_commitment: str | None = None,
    participant_provenance=None,
    recipient_selection=None,
    creation_target=None,
    recipient_receipt_validator=None,
    ensure_ready: Callable[[], Awaitable[None]],
    store_now: Callable[[Any], Awaitable[datetime]],
    lock_closure_lineage: Callable[[Any], Awaitable[None]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    require_available_closure_identity: Callable[[Any, str], Awaitable[None]],
    require_external_creation: Callable[[Any, RunRequest], Awaitable[None]],
    load_for_key_share: Callable[[Any, str], Awaitable[Session | None]],
    load_checkpoint: Callable[[Any, str], Awaitable[dict[str, Any] | None]],
    upsert_checkpoint: Callable[[Any, str, dict[str, Any], datetime], Awaitable[None]],
    register_event_authorities: Callable[
        [Any, str, list[Event] | tuple[Event, ...]], Awaitable[None]
    ],
    jsonb: Callable[[Any], Any],
    unique_violation: type[Exception],
    foreign_key_violation: type[Exception],
) -> Session:
    from cayu.sessions.pending_actions import pending_action_event_storage_values

    request = copy_run_request(request)
    identity = copy_session_identity(identity)
    if result_checkpoint_transform is not None and not callable(result_checkpoint_transform):
        raise TypeError("result_checkpoint_transform must be callable.")
    if participant_binding_factory is not None and not callable(participant_binding_factory):
        raise TypeError("participant_binding_factory must be callable.")
    if participant_binding_factory is not None and type(participant_request_commitment) is not str:
        raise TypeError("participant_request_commitment is required for participant creation.")
    await ensure_ready()
    session_id = request.session_id if request.session_id is not None else _new_id()
    if request.parent_session_id == session_id:
        raise ValueError("Session cannot be its own parent.")
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await lock_closure_lineage(cur)
                await require_available_closure_identity(cur, session_id)
                await require_external_creation(conn, request)
                from cayu.sessions._access_rules import (
                    _RECIPIENT_PROVENANCE_CAPABILITY,
                )

                if (
                    participant_provenance is _RECIPIENT_PROVENANCE_CAPABILITY
                    and creation_target is None
                ):
                    raise PermissionError("Recipient creation requires its exact durable target.")
                if creation_target is not None:
                    creation_target = creation_fence.snapshot_target(creation_target)
                    await _creation_fence.postgres_lock(cur, creation_target)
                    creation_fence.require_pending(
                        creation_target,
                        await _creation_fence.postgres_read(cur, creation_target),
                        request_commitment=participant_request_commitment,
                        requested_session_id=request.session_id,
                    )
                parent_session = (
                    None
                    if request.parent_session_id is None
                    else await load_for_key_share(cur, request.parent_session_id)
                )
                if request.parent_session_id is not None and parent_session is None:
                    raise ValueError(f"Parent session not found: {request.parent_session_id}")
                if parent_session is not None:
                    for owner in await closure_owners(cur, (parent_session.id,)):
                        _check_closure_lineage_owner(owner, (parent_session.id,))
                now = await store_now(cur)
                session = Session(
                    id=session_id,
                    instance_id=session_instance_id_for_run_request(
                        request,
                        session_id=session_id,
                    ),
                    agent_name=request.agent_name,
                    provider_name=identity.provider_name,
                    model=identity.model,
                    parent_session_id=request.parent_session_id,
                    causal_budget_id=request.causal_budget_id or request.task_id or session_id,
                    runtime_name=identity.runtime_name,
                    runtime_version=identity.runtime_version,
                    environment_name=request.environment_name,
                    status=SessionStatus.PENDING,
                    created_at=now,
                    updated_at=now,
                    last_activity_at=now,
                    invocation=session_invocation_for_run_request(
                        request,
                        session_id=session_id,
                        parent_session=parent_session,
                    ),
                    metadata=session_metadata_for_creation(
                        request.metadata,
                        identity=identity,
                        tool_capability_ceiling=request.tool_capability_ceiling,
                        execution_deadline=request.execution_deadline,
                        parent_session=parent_session,
                        prepared_request=request,
                    ),
                    labels=request.labels,
                )
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
                initial_operation_records = _prepare_initial_session_operation_records(
                    session,
                    operation_initializer,
                )
                await cur.execute(
                    f"""
                        INSERT INTO cayu_sessions ({pg_support.SESSION_COLUMNS})
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                    pg_support.session_insert_values(session),
                )
                if participant_binding_factory is not None:
                    binding, receipt = participant_binding_factory(session.model_copy(deep=True))
                    if type(binding).__name__ != "ParticipantSessionBinding":
                        raise TypeError("Participant binding factory returned an invalid binding.")
                    if type(receipt).__name__ != "ParticipantSessionCreationReceipt":
                        raise TypeError("Participant binding factory returned an invalid receipt.")
                    if (
                        binding.session_id != session.id
                        or binding.session_instance_id != session.instance_id
                    ):
                        raise ValueError("Participant binding session identity conflicts.")
                    if receipt.binding != binding:
                        raise ValueError("Participant receipt binding conflicts.")
                    await cur.execute(
                        """
                            INSERT INTO cayu_participant_session_bindings (
                                creation_key, request_commitment, session_id, session_instance_id,
                                application_scope, participant_owner_id, participant_owner_incarnation,
                                participant_id, participant_incarnation, lifecycle_revision,
                                configuration_revision, admission_generation, creator_commitment,
                                authorization_commitment, initial_input_commitment,
                                execution_profile_commitment, binding_json, receipt_json
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                        (
                            binding.creation_key,
                            participant_request_commitment,
                            session.id,
                            session.instance_id,
                            binding.application_scope,
                            binding.participant.owner.owner_id,
                            binding.participant.owner.incarnation,
                            binding.participant.participant_id,
                            binding.participant.incarnation,
                            binding.lifecycle_revision,
                            binding.configuration_revision,
                            binding.admission_generation,
                            binding.creator_commitment,
                            binding.authorization_commitment,
                            binding.initial_input_commitment,
                            binding.execution_profile_commitment,
                            jsonb(binding.model_dump(mode="json")),
                            jsonb(receipt.model_dump(mode="json")),
                        ),
                    )
                    if receipt.recipient_metadata_json is not None:
                        from cayu.sessions._access_rules import (
                            _RECIPIENT_PROVENANCE_CAPABILITY,
                        )

                        if participant_provenance is not _RECIPIENT_PROVENANCE_CAPABILITY:
                            raise PermissionError(
                                "Recipient provenance requires the trusted application boundary."
                            )
                        await cur.executemany(
                            "INSERT INTO cayu_transcript_messages "
                            "(session_id, interaction_id, message, transcript_search_document) "
                            "VALUES (%s, %s, %s, %s)",
                            [
                                (
                                    session.id,
                                    None,
                                    pg_support._dumps(message.model_dump(mode="json")),
                                    transcript_ops.index_document(session.id, message),
                                )
                                for message in request.messages
                            ],
                        )
                    if recipient_selection is not None:
                        from cayu.sessions.context_views import ContextViewSelectionReceipt

                        await cur.execute(
                            "SELECT receipt_json FROM cayu_context_view_selections "
                            "WHERE selection_key = %s FOR UPDATE",
                            (recipient_selection.selection_key,),
                        )
                        row = await cur.fetchone()
                        stored_selection = (
                            None
                            if row is None
                            else ContextViewSelectionReceipt.model_validate(row[0])
                        )
                        if (
                            stored_selection is None
                            or stored_selection != recipient_selection
                            or stored_selection.state not in {"adopted", "transferred"}
                        ):
                            raise PermissionError(
                                "Recipient context-view ownership changed before child creation."
                            )
                    if recipient_receipt_validator is not None:
                        recipient_receipt_validator(session, receipt)
                    if creation_target is not None:
                        creation_fence.validate_binding(
                            creation_target,
                            binding,
                            requested_session_id=receipt.requested_session_id,
                        )
                        await _creation_fence.postgres_write(
                            cur, creation_fence.created(creation_target, session, receipt)
                        )
                elif creation_target is not None:
                    raise PermissionError("Creation targets require participant ownership.")
                if initial_operation_records:
                    await cur.executemany(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record, updated_at) "
                        "VALUES (%s, %s, %s, %s)",
                        [
                            (
                                session.id,
                                key,
                                pg_support._dumps(record),
                                session.updated_at,
                            )
                            for key, record in initial_operation_records.items()
                        ],
                    )
                admission_events: list[Event] = []
                if admission is not None:
                    assert admission.started_event is not None
                    admission_events.append(admission.started_event)
                await register_event_authorities(cur, session.id, admission_events)
                if session.labels:
                    await cur.executemany(
                        """
                            INSERT INTO cayu_session_labels (session_id, key, value)
                            VALUES (%s, %s, %s)
                            """,
                        pg_support.session_label_insert_values(session),
                    )
                if admission is not None:
                    started_event = admission.started_event
                    source_messages = list(admission.source_messages)
                    assert started_event is not None
                    interaction_id = admission.interaction_id
                    deferred_input = deferred_interaction_input_for_run_request(
                        request,
                        session_id=session.id,
                        interaction_id=interaction_id,
                        source_messages=source_messages,
                    )
                    lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                        started_event
                    )
                    await cur.execute(
                        "UPDATE cayu_sessions SET event_seq = 1 WHERE id = %s",
                        (session.id,),
                    )
                    await cur.execute(
                        """
                            INSERT INTO cayu_events (
                                session_id, session_order, event_id, interaction_id,
                                event_type, timestamp, agent_name, environment_name,
                                workflow_name, tool_name, payload, event,
                                pending_action_lookup_key, pending_action_projection,
                                pending_action_projection_bytes
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s
                            )
                            """,
                        (
                            session.id,
                            1,
                            started_event.id,
                            interaction_id,
                            str(started_event.type),
                            pg_support.to_utc(started_event.timestamp),
                            started_event.agent_name,
                            started_event.environment_name,
                            started_event.workflow_name,
                            started_event.tool_name,
                            pg_support._dumps(started_event.payload),
                            pg_support._dumps(started_event.model_dump(mode="json")),
                            lookup_key,
                            projection,
                            projection_bytes,
                        ),
                    )
                    await event_delivery_ops.enqueue_persisted_event_side_effects(
                        cur,
                        session.id,
                        [started_event],
                    )
                    await cur.execute(
                        "INSERT INTO cayu_deferred_interaction_inputs "
                        "(session_id, interaction_id, source_messages) "
                        "VALUES (%s, %s, %s)",
                        (
                            session.id,
                            interaction_id,
                            pg_support._dumps(
                                deferred_interaction_input_storage_payload(deferred_input)
                            ),
                        ),
                    )
                    await upsert_checkpoint(
                        cur,
                        session.id,
                        _initial_transcript_pending_checkpoint(
                            session,
                            interaction_id,
                            checkpoint_transform=checkpoint_transform,
                        ),
                        session.updated_at,
                    )
                elif checkpoint_transform is not None:
                    transformed = checkpoint_transform(session.model_copy(deep=True), None)
                    if transformed is not None:
                        transformed = (
                            _replace_checkpoint_preserving_completion_result_event_publications(
                                None,
                                copy_durable_json_object(transformed, "checkpoint"),
                                session_id=session.id,
                            )
                        )
                        await upsert_checkpoint(
                            cur,
                            session.id,
                            transformed,
                            session.updated_at,
                        )
                if result_checkpoint_transform is not None:
                    current_checkpoint = await load_checkpoint(cur, session.id)
                    transformed = result_checkpoint_transform(
                        session.model_copy(deep=True),
                        _copy_checkpoint_for_transform(
                            current_checkpoint,
                            session_id=session.id,
                        ),
                    )
                    if transformed is None:
                        raise ValueError("Result checkpoint transform must return a checkpoint.")
                    await upsert_checkpoint(
                        cur,
                        session.id,
                        _checkpoint_transform_result_preserving_completion_result_event_publications(
                            current_checkpoint,
                            transformed,
                            session_id=session.id,
                        ),
                        session.updated_at,
                    )
            await conn.commit()
        except unique_violation as exc:
            await conn.rollback()
            constraint_name = getattr(getattr(exc, "diag", None), "constraint_name", None)
            if participant_binding_factory is not None and constraint_name in {
                "cayu_participant_session_bindings_pkey",
                "cayu_participant_session_bindings_session_id_key",
            }:
                raise
            raise ValueError(f"Session already exists: {session.id}") from exc
        except foreign_key_violation as exc:
            await conn.rollback()
            if session.parent_session_id is not None:
                raise ValueError(f"Parent session not found: {session.parent_session_id}") from exc
            raise
    if admission is not None:
        _activate_session_run_fence(session)
    return session.model_copy(deep=True)


async def create_fork(
    connect: PostgresConnection,
    *,
    source_session_id: str,
    fork: Session,
    source_statuses: set[SessionStatus],
    transcript_cursor: int | None,
    checkpoint_transform: CheckpointTransform | None,
    system_prompt_replacement: ForkSystemPromptReplacement | None,
    expected_source_run_epoch: int,
    transcript_validator: ForkTranscriptValidator | None,
    profile_relationship: SessionForkProfileRelationship | None = None,
    events: list[Event] | None = None,
    checkpoint_authority_decoder: ForkCheckpointAuthorityDecoder | None = None,
    operation_initializer: SessionOperationInitializer | None = None,
    ensure_ready: Callable[[], Awaitable[None]],
    store_now: Callable[[Any], Awaitable[datetime]],
    lock_closure_lineage: Callable[[Any], Awaitable[None]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
    require_available_closure_identity: Callable[[Any, str], Awaitable[None]],
    load_checkpoint: Callable[[Any, str], Awaitable[dict[str, Any] | None]],
    register_event_authorities: Callable[
        [Any, str, list[Event] | tuple[Event, ...]], Awaitable[None]
    ],
    register_authorities: AuthorityRegistrar,
    load_session: Callable[[Any, str], Awaitable[Session | None]],
    load_for_update: Callable[[Any, str], Awaitable[Session | None]],
    supports_model_failover: Callable[[], bool],
    unique_violation: type[Exception],
    checkpoint_values: Callable[[str, dict[str, Any], datetime], tuple[object, ...]],
) -> Session:
    source_session_id, fork, allowed_statuses, transcript_cursor = _prepare_session_fork_request(
        source_session_id=source_session_id,
        fork=fork,
        source_statuses=source_statuses,
        transcript_cursor=transcript_cursor,
    )
    fork = fork.model_copy(update={"instance_id": str(uuid4())})

    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await lock_closure_lineage(cur)
                for owner in await closure_owners(cur, (source_session_id,)):
                    _check_closure_lineage_owner(owner, (source_session_id,))
                await require_available_closure_identity(cur, fork.id)
                source_session = _validate_session_fork_source(
                    source_session=await load_for_update(cur, source_session_id),
                    source_session_id=source_session_id,
                    fork=fork,
                    allowed_statuses=allowed_statuses,
                    expected_source_run_epoch=expected_source_run_epoch,
                    profile_relationship=profile_relationship,
                )
                fork.metadata.update(inherited_display_metadata(source_session.metadata))
                fork.metadata[RUN_LINEAGE_METADATA_KEY] = run_lineage_for_creation(source_session)
                fork.metadata.update(run_marks_for_creation(source_session))
                await cur.execute(
                    "SELECT 1 FROM cayu_session_operations "
                    "WHERE session_id = %s AND idempotency_key = %s",
                    (
                        source_session_id,
                        MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
                    ),
                )
                if await cur.fetchone() is not None:
                    raise SessionForkActiveModelStageConflict(
                        "Cannot fork a session while a model-completion stage is active."
                    )

                source_checkpoint = await load_checkpoint(cur, source_session_id)
                source_checkpoint_present = source_checkpoint is not None
                if profile_relationship is not None:
                    profile_checkpoint = None
                    profile_failure: ValueError | None = None
                    try:
                        profile_checkpoint = (
                            source_checkpoint
                            if checkpoint_authority_decoder is None
                            else checkpoint_authority_decoder(
                                None
                                if source_checkpoint is None
                                else copy_durable_json_object(
                                    source_checkpoint,
                                    "source checkpoint",
                                )
                            )
                        )
                        _validate_profiled_fork_authority(
                            source_session=source_session,
                            source_checkpoint=profile_checkpoint,
                            fork=fork,
                            relationship=profile_relationship,
                            events=() if events is None else events,
                            transcript_cursor=transcript_cursor,
                            checkpoint_transform=checkpoint_transform,
                            system_prompt_replacement=system_prompt_replacement,
                            transcript_validator=transcript_validator,
                        )
                    except Exception as exc:
                        profile_failure = _profiled_fork_authority_validation_error(exc)
                        source_checkpoint = None
                    finally:
                        profile_checkpoint = None
                    if profile_failure is not None:
                        raise profile_failure from None

                source_transcript_cursor = await transcript_ops.transcript_cursor(
                    cur, source_session_id
                )
                if transcript_cursor is not None and transcript_cursor > source_transcript_cursor:
                    raise ValueError("transcript_cursor is greater than source transcript length.")
                await cur.execute(
                    """
                        SELECT session_order, message, interaction_id
                        FROM cayu_transcript_messages
                        WHERE session_id = %s
                          AND session_order <= %s
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
                )
                selected_transcript_rows = await cur.fetchall()
                copied_messages = [
                    Message(**pg_support._json_obj(row[1])) for row in selected_transcript_rows
                ]
                copied_interaction_ids = [row[2] for row in selected_transcript_rows]
                source_transcript_snapshot = (
                    None
                    if transcript_validator is None
                    else TranscriptSnapshot(
                        records=[
                            TranscriptRecord(
                                index=int(row[0]) - 1,
                                interaction_id=row[2],
                                message=copied_messages[position],
                            )
                            for position, row in enumerate(selected_transcript_rows)
                        ],
                        cursor=source_transcript_cursor,
                    )
                )
                selected_transcript_rows.clear()
                copied_messages, copied_interaction_ids = apply_fork_system_prompt_replacement(
                    copied_messages,
                    copied_interaction_ids,
                    system_prompt_replacement,
                )
                if not fork_transcript_is_accepted(
                    copied_messages,
                    source_transcript_snapshot,
                    transcript_validator,
                ):
                    copied_messages.clear()
                    copied_messages = []
                    source_transcript_snapshot = None
                    raise ValueError(FORK_TRANSCRIPT_VALIDATION_ERROR) from None
                source_transcript_snapshot = None

                copied_checkpoint = None
                if checkpoint_transform is not None:
                    checkpoint_input = source_checkpoint
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
                if profile_relationship is not None:
                    copied_checkpoint = _prepare_profiled_fork_checkpoint_result(
                        supports_model_failover=supports_model_failover(),
                        fork=fork,
                        transcript_cursor=len(copied_messages),
                        relationship=profile_relationship,
                        source_checkpoint_present=source_checkpoint_present,
                        copied_checkpoint=copied_checkpoint,
                    )
                initial_operation_records = _prepare_initial_session_operation_records(
                    fork,
                    operation_initializer,
                )

                await cur.execute(
                    f"""
                        INSERT INTO cayu_sessions ({pg_support.SESSION_COLUMNS})
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                    pg_support.session_insert_values(fork),
                )
                if initial_operation_records:
                    await cur.executemany(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record, updated_at) "
                        "VALUES (%s, %s, %s, %s)",
                        [
                            (
                                fork.id,
                                key,
                                pg_support._dumps(record),
                                fork.updated_at,
                            )
                            for key, record in initial_operation_records.items()
                        ],
                    )
                await register_authorities(
                    cur,
                    fork.id,
                    interaction_ids=tuple(
                        value for value in copied_interaction_ids if value is not None
                    ),
                )
                if fork.labels:
                    await cur.executemany(
                        """
                            INSERT INTO cayu_session_labels (session_id, key, value)
                            VALUES (%s, %s, %s)
                            """,
                        pg_support.session_label_insert_values(fork),
                    )
                if copied_messages:
                    await cur.executemany(
                        """
                            INSERT INTO cayu_transcript_messages
                                (session_id, interaction_id, message,
                                 transcript_search_document)
                            VALUES (%s, %s, %s, %s)
                            """,
                        [
                            (
                                fork.id,
                                copied_interaction_ids[index],
                                pg_support._dumps(message.model_dump(mode="json")),
                                transcript_ops.index_document(fork.id, message),
                            )
                            for index, message in enumerate(copied_messages)
                        ],
                    )
                if copied_checkpoint is not None:
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
                            """,
                        checkpoint_values(fork.id, copied_checkpoint, fork.updated_at),
                    )
                if events:
                    from cayu.sessions.pending_actions import (
                        pending_action_event_storage_values,
                    )

                    activity_at = await store_now(cur)
                    await cur.execute(
                        "UPDATE cayu_sessions SET event_seq = %s, last_activity_at = %s "
                        "WHERE id = %s",
                        (len(events), activity_at, fork.id),
                    )
                    await register_event_authorities(cur, fork.id, events)
                    rows = []
                    for session_order, event in enumerate(events, start=1):
                        lookup_key, projection, projection_bytes = (
                            pending_action_event_storage_values(event)
                        )
                        rows.append(
                            (
                                fork.id,
                                session_order,
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
                    await cur.executemany(
                        """
                            INSERT INTO cayu_events (
                                session_id, session_order, event_id, interaction_id,
                                event_type, timestamp, agent_name, environment_name,
                                workflow_name, tool_name, payload, event,
                                pending_action_lookup_key, pending_action_projection,
                                pending_action_projection_bytes
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s
                            )
                            """,
                        rows,
                    )
                    await event_delivery_ops.enqueue_persisted_event_side_effects(
                        cur,
                        fork.id,
                        events,
                    )
                loaded = await load_session(cur, fork.id)
                if loaded is None:
                    raise KeyError(f"Session not found: {fork.id}")
            await conn.commit()
        except unique_violation as exc:
            await conn.rollback()
            raise ValueError(f"Session already exists: {fork.id}") from exc
        except Exception:
            await conn.rollback()
            raise

        return loaded


def _new_id() -> str:
    from uuid import uuid4

    return str(uuid4())
