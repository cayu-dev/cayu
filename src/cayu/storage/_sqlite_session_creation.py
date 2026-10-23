"""Native SQLite session creation and fork persistence.

Operations retain admission, authority, initial state and event publication
in one native transaction. Store composition supplies explicit capabilities.
"""

from __future__ import annotations

import json
import sqlite3
from asyncio import Lock
from collections.abc import Callable, Iterable
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
)
from cayu.sessions.transcript_input import deferred_interaction_input_storage_payload
from cayu.sessions.transcript_queries import (
    ForkTranscriptValidator,
    TranscriptSnapshot,
    fork_transcript_is_accepted,
    transcript_search_document,
)
from cayu.storage import _creation_fence
from cayu.storage import _sqlite_event_delivery as event_delivery_ops
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage import _sqlite_transcript as transcript_ops


async def create(
    connection: sqlite3.Connection,
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
    lock: Lock,
    store_now: Callable[[], datetime],
    require_authority_configuration: Callable[[sqlite3.Connection], None],
    closure_owners: Callable[[Iterable[str]], tuple[dict[str, Any], ...]],
    require_available_closure_identity: Callable[[str], None],
    require_external_creation: Callable[[RunRequest], None],
    load_checkpoint: Callable[[str], dict[str, Any] | None],
    session_exists: Callable[[str], bool],
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
    async with lock:
        require_authority_configuration(connection)
        if request.session_id is not None and request.parent_session_id == request.session_id:
            raise ValueError("Session cannot be its own parent.")
        try:
            connection.execute("BEGIN IMMEDIATE")
            with connection:
                created_at = store_now()
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
                    creation_fence.require_pending(
                        creation_target,
                        _creation_fence.sqlite_read(connection, creation_target),
                        request_commitment=participant_request_commitment,
                        requested_session_id=request.session_id,
                    )
                parent_session = (
                    None
                    if request.parent_session_id is None
                    else sqlite_records.load_session(connection, request.parent_session_id)
                )
                if request.parent_session_id is not None and parent_session is None:
                    raise ValueError(f"Parent session not found: {request.parent_session_id}")
                if parent_session is not None:
                    for owner in closure_owners((parent_session.id,)):
                        _check_closure_lineage_owner(owner, (parent_session.id,))
                session = sqlite_records.session_from_request(
                    request,
                    identity=identity,
                    parent_session=parent_session,
                    created_at=created_at,
                )
                require_available_closure_identity(session.id)
                require_external_creation(request)
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
                if session.parent_session_id == session.id:
                    raise ValueError("Session cannot be its own parent.")
                connection.execute(
                    """
                        INSERT INTO cayu_sessions (
                            id,
                            instance_id,
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
                            invocation_json,
                            metadata_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                    (
                        session.id,
                        session.instance_id,
                        session.agent_name,
                        session.provider_name,
                        session.model,
                        session.parent_session_id,
                        session.causal_budget_id,
                        session.runtime_name,
                        session.runtime_version,
                        session.environment_name,
                        str(session.status),
                        sqlite_records.format_datetime(session.created_at),
                        sqlite_records.format_datetime(session.updated_at),
                        sqlite_records.format_datetime(session.last_activity_at),
                        session.run_epoch,
                        sqlite_records.json_dumps(session.invocation.model_dump(mode="json")),
                        sqlite_records.json_dumps(session.metadata),
                    ),
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
                    connection.execute(
                        """
                            INSERT INTO cayu_participant_session_bindings (
                                creation_key, request_commitment, session_id, session_instance_id,
                                application_scope, participant_owner_id, participant_owner_incarnation,
                                participant_id, participant_incarnation, lifecycle_revision,
                                configuration_revision, admission_generation, creator_commitment,
                                authorization_commitment, initial_input_commitment,
                                execution_profile_commitment, binding_json, receipt_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                            sqlite_records.json_dumps(binding.model_dump(mode="json")),
                            sqlite_records.json_dumps(receipt.model_dump(mode="json")),
                        ),
                    )
                    if recipient_selection is not None:
                        from cayu.sessions.context_views import ContextViewSelectionReceipt

                        row = connection.execute(
                            "SELECT receipt_json FROM cayu_context_view_selections "
                            "WHERE selection_key = ?",
                            (recipient_selection.selection_key,),
                        ).fetchone()
                        stored_selection = (
                            None
                            if row is None
                            else ContextViewSelectionReceipt.model_validate_json(row[0])
                        )
                        if (
                            stored_selection is None
                            or stored_selection != recipient_selection
                            or stored_selection.state not in {"adopted", "transferred"}
                        ):
                            raise PermissionError(
                                "Recipient context-view ownership changed before child creation."
                            )
                    if receipt.recipient_metadata_json is not None:
                        from cayu.sessions._access_rules import (
                            _RECIPIENT_PROVENANCE_CAPABILITY,
                        )

                        if participant_provenance is not _RECIPIENT_PROVENANCE_CAPABILITY:
                            raise PermissionError(
                                "Recipient provenance requires the trusted application boundary."
                            )
                        connection.executemany(
                            "INSERT INTO cayu_transcript_messages "
                            "(session_id, role, interaction_id, message_json, transcript_search_document) "
                            "VALUES (?, ?, ?, ?, ?)",
                            [
                                (
                                    session.id,
                                    str(message.role),
                                    None,
                                    sqlite_records.json_dumps(message.model_dump(mode="json")),
                                    transcript_search_document(message),
                                )
                                for message in request.messages
                            ],
                        )
                    if recipient_receipt_validator is not None:
                        recipient_receipt_validator(session, receipt)
                    if creation_target is not None:
                        creation_fence.validate_binding(
                            creation_target,
                            binding,
                            requested_session_id=receipt.requested_session_id,
                        )
                        _creation_fence.sqlite_write(
                            connection,
                            creation_fence.created(creation_target, session, receipt),
                        )
                elif creation_target is not None:
                    raise PermissionError("Creation targets require participant ownership.")
                if initial_operation_records:
                    connection.executemany(
                        "INSERT INTO cayu_session_operations "
                        "(session_id, idempotency_key, record_json, updated_at) "
                        "VALUES (?, ?, ?, ?)",
                        [
                            (
                                session.id,
                                key,
                                sqlite_records.json_dumps(record),
                                sqlite_records.format_datetime(session.updated_at),
                            )
                            for key, record in initial_operation_records.items()
                        ],
                    )
                if session.labels:
                    connection.executemany(
                        """
                            INSERT INTO cayu_session_labels (session_id, key, value)
                            VALUES (?, ?, ?)
                            """,
                        sqlite_records.session_label_row_values(session),
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
                    connection.execute(
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
                            sqlite_records.format_datetime(started_event.timestamp),
                            started_event.agent_name,
                            started_event.environment_name,
                            started_event.workflow_name,
                            started_event.tool_name,
                            sqlite_records.json_dumps(started_event.payload),
                            lookup_key,
                            projection,
                            projection_bytes,
                        ),
                    )
                    event_delivery_ops.enqueue_persisted_event_side_effects(
                        connection,
                        session.id,
                        [started_event],
                    )
                    connection.execute(
                        "INSERT INTO cayu_deferred_interaction_inputs "
                        "(session_id, interaction_id, source_messages_json) "
                        "VALUES (?, ?, ?)",
                        (
                            session.id,
                            interaction_id,
                            sqlite_records.json_dumps(
                                deferred_interaction_input_storage_payload(deferred_input)
                            ),
                        ),
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
                            """,
                        sqlite_records.checkpoint_row_values(
                            session.id,
                            _initial_transcript_pending_checkpoint(
                                session,
                                interaction_id,
                                checkpoint_transform=checkpoint_transform,
                            ),
                            session.updated_at,
                        ),
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
                        connection.execute(
                            """
                                INSERT INTO cayu_checkpoints (
                                    session_id, state_json, updated_at,
                                    pending_action_source_bytes,
                                    pending_action_tool_call_count,
                                    pending_action_flags,
                                    pending_action_metrics_ready
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                            sqlite_records.checkpoint_row_values(
                                session.id,
                                transformed,
                                session.updated_at,
                            ),
                        )
                if result_checkpoint_transform is not None:
                    current_checkpoint = load_checkpoint(session.id)
                    transformed = result_checkpoint_transform(
                        session.model_copy(deep=True),
                        _copy_checkpoint_for_transform(
                            current_checkpoint,
                            session_id=session.id,
                        ),
                    )
                    if transformed is None:
                        raise ValueError("Result checkpoint transform must return a checkpoint.")
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
                            session.id,
                            _checkpoint_transform_result_preserving_completion_result_event_publications(
                                current_checkpoint,
                                transformed,
                                session_id=session.id,
                            ),
                            session.updated_at,
                        ),
                    )
        except sqlite3.IntegrityError as exc:
            if session_exists(session.id):
                raise ValueError(f"Session already exists: {session.id}") from exc
            if session.parent_session_id is not None and not session_exists(
                session.parent_session_id
            ):
                raise ValueError(f"Parent session not found: {session.parent_session_id}") from exc
            raise
        if admission is not None:
            _activate_session_run_fence(session)
        return session.model_copy(deep=True)


async def create_fork(
    connection: sqlite3.Connection,
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
    lock: Lock,
    store_now: Callable[[], datetime],
    require_authority_configuration: Callable[[sqlite3.Connection], None],
    closure_owners: Callable[[Iterable[str]], tuple[dict[str, Any], ...]],
    require_available_closure_identity: Callable[[str], None],
    load_checkpoint: Callable[[str], dict[str, Any] | None],
    session_exists: Callable[[str], bool],
    load_session: Callable[[str], Session | None],
    supports_model_failover: Callable[[], bool],
    touch_activity: Callable[[sqlite3.Connection, str, datetime], None],
) -> Session:
    source_session_id, fork, allowed_statuses, transcript_cursor = _prepare_session_fork_request(
        source_session_id=source_session_id,
        fork=fork,
        source_statuses=source_statuses,
        transcript_cursor=transcript_cursor,
    )
    fork = fork.model_copy(update={"instance_id": str(uuid4())})

    async with lock:
        require_authority_configuration(connection)
        try:
            connection.execute("BEGIN IMMEDIATE")
            for owner in closure_owners((source_session_id,)):
                _check_closure_lineage_owner(owner, (source_session_id,))
            require_available_closure_identity(fork.id)
            source_session = _validate_session_fork_source(
                source_session=load_session(source_session_id),
                source_session_id=source_session_id,
                fork=fork,
                allowed_statuses=allowed_statuses,
                expected_source_run_epoch=expected_source_run_epoch,
                profile_relationship=profile_relationship,
            )
            fork.metadata.update(inherited_display_metadata(source_session.metadata))
            fork.metadata[RUN_LINEAGE_METADATA_KEY] = run_lineage_for_creation(source_session)
            fork.metadata.update(run_marks_for_creation(source_session))
            active_stage = connection.execute(
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
            source_checkpoint = load_checkpoint(source_session_id)
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
            source_transcript_cursor = transcript_ops.transcript_cursor(
                connection,
                source_session_id,
            )
            if transcript_cursor is not None and transcript_cursor > source_transcript_cursor:
                raise ValueError("transcript_cursor is greater than source transcript length.")
            selected_transcript_rows = connection.execute(
                """
                    SELECT session_order, message_json, interaction_id
                    FROM cayu_transcript_messages
                    WHERE session_id = ?
                      AND session_order <= ?
                    ORDER BY session_order ASC
                    """,
                (
                    source_session_id,
                    (source_transcript_cursor if transcript_cursor is None else transcript_cursor),
                ),
            ).fetchall()
            copied_messages = [
                Message(**json.loads(row["message_json"])) for row in selected_transcript_rows
            ]
            copied_interaction_ids = [row["interaction_id"] for row in selected_transcript_rows]
            source_transcript_snapshot = (
                None
                if transcript_validator is None
                else TranscriptSnapshot(
                    records=[
                        TranscriptRecord(
                            index=int(row["session_order"]) - 1,
                            interaction_id=row["interaction_id"],
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

            connection.execute(
                """
                    INSERT INTO cayu_sessions (
                        id,
                        instance_id,
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
                        invocation_json,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                sqlite_records.session_to_row_values(fork),
            )
            if initial_operation_records:
                connection.executemany(
                    "INSERT INTO cayu_session_operations "
                    "(session_id, idempotency_key, record_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (
                            fork.id,
                            key,
                            sqlite_records.json_dumps(record),
                            sqlite_records.format_datetime(fork.updated_at),
                        )
                        for key, record in initial_operation_records.items()
                    ],
                )
            if fork.labels:
                connection.executemany(
                    """
                        INSERT INTO cayu_session_labels (session_id, key, value)
                        VALUES (?, ?, ?)
                        """,
                    sqlite_records.session_label_row_values(fork),
                )
            if copied_messages:
                connection.executemany(
                    """
                        INSERT INTO cayu_transcript_messages (
                            session_id,
                            role,
                            interaction_id,
                            message_json,
                            transcript_search_document
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                    [
                        (
                            fork.id,
                            str(message.role),
                            copied_interaction_ids[index],
                            sqlite_records.json_dumps(message.model_dump(mode="json")),
                            transcript_search_document(message),
                        )
                        for index, message in enumerate(copied_messages)
                    ],
                )
            if copied_checkpoint is not None:
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
                        """,
                    sqlite_records.checkpoint_row_values(
                        fork.id, copied_checkpoint, fork.updated_at
                    ),
                )
            if events:
                from cayu.sessions.pending_actions import pending_action_event_storage_values

                touch_activity(connection, fork.id, store_now())
                rows = []
                for event in events:
                    lookup_key, projection, projection_bytes = pending_action_event_storage_values(
                        event
                    )
                    rows.append(
                        (
                            fork.id,
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
                    """
                        INSERT INTO cayu_events (
                            session_id, event_id, interaction_id, event_type, timestamp,
                            agent_name, environment_name, workflow_name, tool_name,
                            payload_json, pending_action_lookup_key,
                            pending_action_projection_json, pending_action_projection_bytes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                    rows,
                )
                event_delivery_ops.enqueue_persisted_event_side_effects(
                    connection,
                    fork.id,
                    events,
                )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            connection.rollback()
            if session_exists(fork.id):
                raise ValueError(f"Session already exists: {fork.id}") from exc
            raise
        except Exception:
            connection.rollback()
            raise

        loaded = load_session(fork.id)
        if loaded is None:
            raise KeyError(f"Session not found: {fork.id}")
        return loaded
