"""Complete in-memory creation, participant binding and fork operations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar
from uuid import uuid4

from cayu._validation import copy_durable_json_object
from cayu.events import Event, EventType
from cayu.messages import Message, MessageRole, copy_message
from cayu.sessions import _checkpoint_preservation as checkpoint_preservation
from cayu.sessions import _checkpoint_publication as checkpoint_publication
from cayu.sessions import _closure_records as closure_records
from cayu.sessions import _interaction_transitions as interaction_rules
from cayu.sessions import _invocation_scope as invocation_scope
from cayu.sessions import _memory_transcripts as memory_transcripts
from cayu.sessions import creation_fence
from cayu.sessions import forks as session_fork_rules
from cayu.sessions._access_rules import _RECIPIENT_PROVENANCE_CAPABILITY
from cayu.sessions._model_stage import MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY
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
    apply_fork_system_prompt_replacement,
    transform_fork_checkpoint,
)
from cayu.sessions.records import (
    CheckpointTransform,
    EventRecord,
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
from cayu.sessions.transcript_input import DeferredInteractionInput, copy_transcript_messages
from cayu.sessions.transcript_queries import (
    ForkTranscriptValidator,
    TranscriptSnapshot,
    fork_transcript_is_accepted,
)
from cayu.storage._creation_fence import validate_replay

if TYPE_CHECKING:
    from cayu.sessions.context_views import (
        ContextViewSelectionReceipt,
        ParticipantSessionBinding,
        ParticipantSessionCreationReceipt,
        ParticipantSessionCreationRequest,
    )
    from cayu.sessions.context_views import (
        ContextViewSelectionReceipt as _ContextViewSelectionReceipt,
    )


class PreparedEvent(Protocol):
    @property
    def record(self) -> EventRecord: ...


class PreparedEvents(Protocol):
    @property
    def events(self) -> tuple[PreparedEvent, ...]: ...


class PreparedCheckpoint(Protocol):
    @property
    def checkpoint(self) -> dict[str, Any]: ...


EventsT = TypeVar("EventsT", bound=PreparedEvents)
CheckpointT = TypeVar("CheckpointT", bound=PreparedCheckpoint)
Checkpoint_co = TypeVar("Checkpoint_co", bound=PreparedCheckpoint, covariant=True)
Events_contra = TypeVar("Events_contra", bound=PreparedEvents, contravariant=True)


class PrepareCheckpoint(Protocol[Checkpoint_co]):
    def __call__(
        self,
        session_id: str,
        checkpoint: dict[str, Any],
        *,
        additional_event_records: tuple[EventRecord, ...] = (),
    ) -> Checkpoint_co: ...


class ApplyEvents(Protocol[Events_contra]):
    def __call__(
        self, session: Session, prepared: Events_contra, *, activity_at: datetime | None = None
    ) -> Session: ...


class RegisterAlias(Protocol):
    def __call__(
        self, private_value: str, *, field_name: str, scope_session_id: str | None = None
    ) -> None: ...


class CreateSession(Protocol):
    async def __call__(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        checkpoint_transform: CheckpointTransform | None = None,
        result_checkpoint_transform: CheckpointTransform | None = None,
        operation_initializer: SessionOperationInitializer | None = None,
        participant_binding_factory: Callable[
            [Session], tuple[ParticipantSessionBinding, ParticipantSessionCreationReceipt]
        ]
        | None = None,
        participant_request_commitment: str | None = None,
        participant_provenance: object | None = None,
        recipient_selection: _ContextViewSelectionReceipt | None = None,
        creation_target=None,
        recipient_receipt_validator: Callable[[Session, ParticipantSessionCreationReceipt], object]
        | None = None,
    ) -> Session: ...


@dataclass(frozen=True)
class MemorySessionCreationState(Generic[EventsT, CheckpointT]):
    """Current creation indexes and synchronous publication capabilities.

    Resolve every binding after acquiring the store lock. Prepared records stay
    owned by the publication implementation: creation only reads the checkpoint
    and event records needed by validation and returns them to the paired apply
    capability. Callbacks must neither yield nor reacquire the lock.
    """

    apply_checkpoint_store: Callable[[str, CheckpointT], None]
    apply_event_append: ApplyEvents[EventsT]
    append_events: Callable[[Session, list[Event]], Session]
    checkpoints: dict[str, dict[str, Any]]
    child_lifecycle_event_records: dict[str, dict[EventType, EventRecord]]
    context_selection_receipts: dict[str, ContextViewSelectionReceipt]
    deferred_interaction_inputs: dict[str, DeferredInteractionInput]
    event_ids: dict[str, set[str]]
    events: dict[str, list[Event]]
    index_session_parent: Callable[[Session], None]
    interaction_event_records: dict[str, dict[str, list[EventRecord]]]
    latest_interaction_event_records: dict[str, dict[str, EventRecord]]
    latest_interaction_event_records_by_sequence: dict[str, list[EventRecord]]
    latest_transcript_indexes_by_role: dict[str, dict[MessageRole, int]]
    ownership_clock: Callable[[], datetime]
    participant_session_bindings: dict[str, ParticipantSessionBinding]
    participant_session_ids_by_key: dict[str, str]
    participant_session_receipts: dict[str, ParticipantSessionCreationReceipt]
    participant_session_request_commitments: dict[str, str | None]
    pending_action_event_records: dict[str, dict[str, dict[str, EventRecord]]]
    prepare_checkpoint_store: PrepareCheckpoint[CheckpointT]
    prepare_event_append: Callable[[Session, list[Event]], EventsT]
    refresh_child_lifecycle_candidate: Callable[[Session], None]
    register_private_authority_alias: RegisterAlias
    require_available_closure_identity: Callable[[str], None]
    require_external_creation: Callable[[RunRequest], None]
    session_closure_progress: dict[tuple[str, str], dict[str, Any]]
    session_creation_decisions: dict[str, creation_fence.SessionCreationDecision]
    session_event_records: dict[str, list[EventRecord]]
    session_operation_records: dict[str, dict[str, dict[str, Any]]]
    sessions: dict[str, Session]
    store_checkpoint: Callable[[str, dict[str, Any]], None]
    supports_model_failover_stage_protocol: Callable[[], bool]
    transcript_indexes: Callable[[], memory_transcripts.MemoryTranscriptIndexes]
    transcript_interaction_ids: dict[str, list[str | None]]
    transcript_search_documents: dict[str, list[tuple[MessageRole, str]]]
    transcripts: dict[str, list[Message]]


@dataclass(frozen=True)
class MemoryParticipantState:
    """Participant records borrowed under the operation's existing lock."""

    sessions: dict[str, Session]
    participant_session_bindings: dict[str, ParticipantSessionBinding]
    participant_session_ids_by_key: dict[str, str]
    participant_session_receipts: dict[str, ParticipantSessionCreationReceipt]
    participant_session_request_commitments: dict[str, str | None]


@dataclass(frozen=True)
class CreationDecisionReader:
    read_session_creation_decision: Callable[
        [creation_fence.SessionCreationTarget], Awaitable[object]
    ]


@dataclass(frozen=True)
class MemoryParticipantCreationState(MemoryParticipantState):
    """Participant serialization also encloses native create and fence readback.

    The create/read callbacks acquire the store lock after the participant lock,
    matching deletion. They must resolve current state at their own lock boundary.
    """

    create: CreateSession
    read_creation_decision: Callable[[creation_fence.SessionCreationTarget], Awaitable[object]]
    created_receipt: Callable[[str], ParticipantSessionCreationReceipt | None]


LockedCreationState = Callable[
    [], AbstractAsyncContextManager[MemorySessionCreationState[EventsT, CheckpointT]]
]
LockedParticipantState = Callable[[], AbstractAsyncContextManager[MemoryParticipantState]]
LockedParticipantCreationState = Callable[
    [], AbstractAsyncContextManager[MemoryParticipantCreationState]
]


async def create(
    locked_state: LockedCreationState[EventsT, CheckpointT],
    request: RunRequest,
    *,
    identity: SessionIdentity,
    interaction_started_event: Event | None = None,
    interaction_source_messages: list[Message] | None = None,
    checkpoint_transform: CheckpointTransform | None = None,
    result_checkpoint_transform: CheckpointTransform | None = None,
    operation_initializer: SessionOperationInitializer | None = None,
    participant_binding_factory: Callable[
        [Session], tuple[ParticipantSessionBinding, ParticipantSessionCreationReceipt]
    ]
    | None = None,
    participant_request_commitment: str | None = None,
    participant_provenance: object | None = None,
    recipient_selection: _ContextViewSelectionReceipt | None = None,
    creation_target=None,
    recipient_receipt_validator: Callable[[Session, ParticipantSessionCreationReceipt], object]
    | None = None,
) -> Session:
    if type(request) is not RunRequest:
        raise TypeError("Session creation requires a RunRequest.")
    if participant_binding_factory is not None and not callable(participant_binding_factory):
        raise TypeError("participant_binding_factory must be callable.")
    if participant_binding_factory is not None:
        from cayu.sessions.context_views import (
            ParticipantSessionBinding,
            ParticipantSessionCreationReceipt,
        )
    request = copy_run_request(request)
    identity = copy_session_identity(identity)
    if result_checkpoint_transform is not None and not callable(result_checkpoint_transform):
        raise TypeError("result_checkpoint_transform must be callable.")
    async with locked_state() as state:
        session_id = request.session_id or str(uuid4())
        state.require_external_creation(request)
        if participant_provenance is _RECIPIENT_PROVENANCE_CAPABILITY and creation_target is None:
            raise PermissionError("Recipient creation requires its exact durable target.")
        if creation_target is not None:
            creation_target = creation_fence.snapshot_target(creation_target)
            creation_fence.require_pending(
                creation_target,
                state.session_creation_decisions.get(creation_target.key),
                request_commitment=participant_request_commitment,
                requested_session_id=request.session_id,
            )
        state.require_available_closure_identity(session_id)
        admission = interaction_rules._copy_optional_interaction_admission(
            session_id,
            interaction_started_event,
            interaction_source_messages,
            defer_transcript=True,
        )
        if session_id in state.sessions:
            raise ValueError(f"Session already exists: {session_id}")
        if request.parent_session_id == session_id:
            raise ValueError("Session cannot be its own parent.")
        if (
            request.parent_session_id is not None
            and request.parent_session_id not in state.sessions
        ):
            raise ValueError(f"Parent session not found: {request.parent_session_id}")
        parent_session = (
            None if request.parent_session_id is None else state.sessions[request.parent_session_id]
        )
        if parent_session is not None:
            for owner in state.session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (parent_session.id,))
        now = state.ownership_clock()
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
            status=(SessionStatus.RUNNING if admission is not None else SessionStatus.PENDING),
            created_at=now,
            updated_at=now,
            last_activity_at=now,
            invocation=session_invocation_for_run_request(
                request,
                session_id=session_id,
                parent_session=parent_session,
            ),
            labels=request.labels,
            metadata=session_metadata_for_creation(
                request.metadata,
                identity=identity,
                tool_capability_ceiling=request.tool_capability_ceiling,
                execution_deadline=request.execution_deadline,
                parent_session=parent_session,
                prepared_request=request,
            ),
            run_epoch=1 if admission is not None else 0,
        )
        initial_operation_records = _prepare_initial_session_operation_records(
            session,
            operation_initializer,
        )
        prepared_admission_events: EventsT | None = None
        prepared_checkpoint: CheckpointT | None = None
        pending_checkpoint: CheckpointT | None = None
        if admission is None and checkpoint_transform is not None:
            transformed = checkpoint_transform(session.model_copy(deep=True), None)
            if transformed is not None:
                copied_checkpoint = checkpoint_preservation._replace_checkpoint_preserving_completion_result_event_publications(
                    None,
                    copy_durable_json_object(transformed, "checkpoint"),
                    session_id=session.id,
                )
                pending_checkpoint = state.prepare_checkpoint_store(
                    session.id,
                    copied_checkpoint,
                )
        if admission is not None:
            started_event, source_messages = admission
            interaction_id = started_event.interaction_id
            if interaction_id is None:
                raise AssertionError("Interaction admission lost its identity.")
            prepared_admission_events = state.prepare_event_append(
                session,
                [started_event],
            )
            session = session.model_copy(update={"last_activity_at": now})
            prepared_checkpoint = state.prepare_checkpoint_store(
                session.id,
                checkpoint_publication._initial_transcript_pending_checkpoint(
                    session,
                    interaction_id,
                    checkpoint_transform=checkpoint_transform,
                ),
                additional_event_records=tuple(
                    event.record for event in prepared_admission_events.events
                ),
            )
        if result_checkpoint_transform is not None:
            current_checkpoint = (
                None if prepared_checkpoint is None else prepared_checkpoint.checkpoint
            )
            transformed = result_checkpoint_transform(
                session.model_copy(deep=True),
                checkpoint_preservation._copy_checkpoint_for_transform(
                    current_checkpoint,
                    session_id=session.id,
                ),
            )
            if transformed is None:
                raise ValueError("Result checkpoint transform must return a checkpoint.")
            prepared_checkpoint = state.prepare_checkpoint_store(
                session.id,
                checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                    current_checkpoint,
                    transformed,
                    session_id=session.id,
                ),
                additional_event_records=(
                    ()
                    if prepared_admission_events is None
                    else tuple(event.record for event in prepared_admission_events.events)
                ),
            )
        elif pending_checkpoint is not None:
            prepared_checkpoint = pending_checkpoint

        participant_binding = None
        participant_receipt = None
        recipient_messages = []
        if participant_binding_factory is not None:
            participant_binding, participant_receipt = participant_binding_factory(
                session.model_copy(deep=True)
            )
            if type(participant_binding) is not ParticipantSessionBinding:
                raise TypeError("Participant binding factory returned an invalid binding.")
            if type(participant_receipt) is not ParticipantSessionCreationReceipt:
                raise TypeError("Participant binding factory returned an invalid receipt.")
            if participant_binding.session_id != session.id:
                raise ValueError("Participant binding session identity conflicts.")
            if participant_binding.session_instance_id != session.instance_id:
                raise ValueError("Participant binding incarnation conflicts.")
            if participant_receipt.binding != participant_binding:
                raise ValueError("Participant receipt binding conflicts.")
            if participant_binding.creation_key in state.participant_session_ids_by_key:
                raise ValueError("Participant creation key was concurrently claimed.")
            if type(participant_request_commitment) is not str:
                raise ValueError("Participant request commitment is required.")
            if (
                participant_receipt.recipient_metadata_json is not None
                and participant_provenance is not _RECIPIENT_PROVENANCE_CAPABILITY
            ):
                raise PermissionError(
                    "Recipient provenance requires the trusted application boundary."
                )
            if participant_receipt.recipient_metadata_json is not None:
                recipient_messages = copy_transcript_messages(request.messages)
            if recipient_selection is not None:
                stored_selection = state.context_selection_receipts.get(
                    recipient_selection.selection_key
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
                recipient_receipt_validator(session, participant_receipt)

        creation_decision = None
        if creation_target is not None:
            if participant_binding is None:
                raise PermissionError("Creation targets require participant ownership.")
            assert participant_receipt is not None
            creation_fence.validate_binding(
                creation_target,
                participant_binding,
                requested_session_id=participant_receipt.requested_session_id,
            )
            creation_decision = creation_fence.created(
                creation_target, session, participant_receipt
            )

        # Every caller-controlled transform and validation step is complete.
        # Publish the prepared state while retaining the in-memory store lock.
        state.register_private_authority_alias(
            session.id,
            field_name="session_id",
        )
        state.sessions[session.id] = session
        if creation_decision is not None:
            state.session_creation_decisions[creation_decision.target.key] = creation_decision
        state.index_session_parent(session)
        state.events[session.id] = []
        state.event_ids[session.id] = set()
        state.session_event_records[session.id] = []
        state.child_lifecycle_event_records[session.id] = {}
        state.interaction_event_records[session.id] = {}
        state.latest_interaction_event_records[session.id] = {}
        state.latest_interaction_event_records_by_sequence[session.id] = []
        state.pending_action_event_records[session.id] = {}
        state.session_operation_records[session.id] = initial_operation_records
        if participant_binding is not None:
            state.participant_session_bindings[session.id] = participant_binding.model_copy(
                deep=True
            )
            assert participant_receipt is not None
            state.participant_session_receipts[session.id] = participant_receipt.model_copy(
                deep=True
            )
            state.participant_session_ids_by_key[participant_binding.creation_key] = session.id
            state.participant_session_request_commitments[participant_binding.creation_key] = (
                participant_request_commitment
            )
        state.transcripts[session.id] = recipient_messages
        state.transcript_interaction_ids[session.id] = [None] * len(recipient_messages)
        state.latest_transcript_indexes_by_role[session.id] = {}
        state.transcript_search_documents[session.id] = []
        if recipient_messages:
            memory_transcripts.replace_latest_role_indexes(state.transcript_indexes(), session.id)
            memory_transcripts.replace_search_session(
                state.transcript_indexes(), session.id, recipient_messages
            )
        state.refresh_child_lifecycle_candidate(session)
        if prepared_admission_events is not None:
            assert admission is not None
            started_event, source_messages = admission
            interaction_id = started_event.interaction_id
            assert interaction_id is not None
            state.sessions[session.id] = state.apply_event_append(
                session,
                prepared_admission_events,
                activity_at=now,
            )
            state.deferred_interaction_inputs[session.id] = (
                deferred_interaction_input_for_run_request(
                    request,
                    session_id=session.id,
                    interaction_id=interaction_id,
                    source_messages=source_messages,
                )
            )
        if prepared_checkpoint is not None:
            state.apply_checkpoint_store(session.id, prepared_checkpoint)
        session = state.sessions[session.id]
        if admission is not None:
            invocation_scope._activate_session_run_fence(session)
        return session.model_copy(deep=True)


async def create_participant_owned_session(
    locked_state: LockedParticipantCreationState,
    creation_request: ParticipantSessionCreationRequest,
    *,
    resolved_request: RunRequest,
    identity: SessionIdentity,
    binding_factory: Callable[
        [Session], tuple[ParticipantSessionBinding, ParticipantSessionCreationReceipt]
    ],
    recipient_provenance: object | None = None,
    recipient_selection: _ContextViewSelectionReceipt | None = None,
    creation_target=None,
    recipient_receipt_validator=None,
) -> tuple[Session, ParticipantSessionCreationReceipt]:
    from cayu.sessions.context_views import (
        ParticipantSessionCreationRequest,
    )

    if type(creation_request) is not ParticipantSessionCreationRequest:
        raise TypeError("Participant creation requires a typed creation request.")
    if (
        creation_request.metadata_json is not None
        and recipient_provenance is not _RECIPIENT_PROVENANCE_CAPABILITY
    ):
        raise PermissionError("Recipient provenance requires the trusted application boundary.")
    if type(resolved_request) is not RunRequest:
        raise TypeError("Participant creation requires a resolved RunRequest.")
    if not callable(binding_factory):
        raise TypeError("binding_factory must be callable.")
    async with locked_state() as state:
        existing_id = state.participant_session_ids_by_key.get(creation_request.creation_key)
        if existing_id is not None:
            session = state.sessions.get(existing_id)
            receipt = state.participant_session_receipts.get(existing_id)
            if session is None or receipt is None:
                raise RuntimeError("Participant creation index is inconsistent.")
            if (
                state.participant_session_request_commitments.get(creation_request.creation_key)
                != creation_request.request_commitment
            ):
                raise ValueError("Participant creation key conflicts with the request.")
            if (session.provider_name, session.model) != (
                identity.provider_name,
                identity.model,
            ):
                raise ValueError("Participant creation key conflicts with the execution target.")
            expected_binding, expected_receipt = binding_factory(session.model_copy(deep=True))
            if expected_binding != state.participant_session_bindings.get(existing_id):
                raise ValueError("Participant creation key conflicts with its binding.")
            if expected_receipt != receipt:
                raise ValueError("Participant creation key conflicts with its receipt.")
            if creation_target is not None:
                await validate_replay(
                    CreationDecisionReader(state.read_creation_decision), creation_target, session
                )
            return session.model_copy(deep=True), receipt.model_copy(deep=True)
        session = await state.create(
            resolved_request,
            identity=identity,
            participant_binding_factory=binding_factory,
            participant_request_commitment=creation_request.request_commitment,
            participant_provenance=recipient_provenance,
            recipient_selection=recipient_selection,
            creation_target=creation_target,
            recipient_receipt_validator=recipient_receipt_validator,
        )
        # Creation may have waited for the store lock. Read the current receipt
        # index after that boundary, still holding participant serialization.
        receipt = state.created_receipt(session.id)
        if receipt is None:
            raise RuntimeError("Participant creation did not persist its receipt.")
        return session, receipt.model_copy(deep=True)


async def lookup_participant_session_creation(
    locked_state: LockedParticipantCreationState,
    creation_request: ParticipantSessionCreationRequest,
) -> tuple[Session, ParticipantSessionCreationReceipt] | None:
    from cayu.sessions.context_views import ParticipantSessionCreationRequest

    if type(creation_request) is not ParticipantSessionCreationRequest:
        raise TypeError("Participant creation requires a typed creation request.")
    async with locked_state() as state:
        session_id = state.participant_session_ids_by_key.get(creation_request.creation_key)
        if session_id is None:
            return None
        session = state.sessions.get(session_id)
        receipt = state.participant_session_receipts.get(session_id)
        if session is None or receipt is None:
            raise RuntimeError("Participant creation index is inconsistent.")
        if state.participant_session_request_commitments.get(creation_request.creation_key) != (
            creation_request.request_commitment
        ):
            raise ValueError("Participant creation key conflicts with the request.")
        return session.model_copy(deep=True), receipt.model_copy(deep=True)


async def load_participant_session_binding(
    locked_state: LockedParticipantState, session_id: str
) -> ParticipantSessionBinding | None:
    async with locked_state() as state:
        binding = state.participant_session_bindings.get(session_id)
        return None if binding is None else binding.model_copy(deep=True)


async def scan_participant_session_bindings(
    locked_state: LockedParticipantState, participant, *, after=None, limit=32
):
    from cayu.sessions._participant_discovery import prepare_scan, reference

    query = prepare_scan(participant, after, limit)
    async with locked_state() as state:
        bindings = sorted(
            (
                item
                for item in state.participant_session_bindings.values()
                if item.participant == query.participant
                and (query.after is None or item.creation_key > query.after)
            ),
            key=lambda item: item.creation_key,
        )[: query.limit]
        result = []
        for binding in bindings:
            session = state.sessions.get(binding.session_id)
            receipt = state.participant_session_receipts.get(binding.session_id)
            if (
                session is None
                or session.instance_id != binding.session_instance_id
                or receipt is None
                or receipt.binding != binding
                or state.participant_session_ids_by_key.get(binding.creation_key) != session.id
            ):
                raise ValueError("Participant inventory lacks coherent native creation.")
            result.append(reference(receipt, query))
        return tuple(result)


async def load_participant_session_creation_receipt(
    locked_state: LockedParticipantState, session_id: str
) -> ParticipantSessionCreationReceipt | None:
    async with locked_state() as state:
        receipt = state.participant_session_receipts.get(session_id)
        return None if receipt is None else receipt.model_copy(deep=True)


async def create_fork(
    locked_state: LockedCreationState[EventsT, CheckpointT],
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
) -> Session:
    source_session_id, fork, allowed_statuses, transcript_cursor = (
        session_fork_rules._prepare_session_fork_request(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
        )
    )
    fork = fork.model_copy(update={"instance_id": str(uuid4())})
    async with locked_state() as state:
        state.require_available_closure_identity(fork.id)
        for owner in state.session_closure_progress.values():
            closure_records._check_closure_lineage_owner(owner, (source_session_id,))
        source_session = session_fork_rules._validate_session_fork_source(
            source_session=state.sessions.get(source_session_id),
            source_session_id=source_session_id,
            fork=fork,
            allowed_statuses=allowed_statuses,
            expected_source_run_epoch=expected_source_run_epoch,
            profile_relationship=profile_relationship,
        )
        if MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY in state.session_operation_records.get(
            source_session_id, {}
        ):
            raise SessionForkActiveModelStageConflict(
                "Cannot fork a session while a model-completion stage is active."
            )
        if fork.id in state.sessions:
            raise ValueError(f"Session already exists: {fork.id}")

        stored_checkpoint = state.checkpoints.get(source_session_id)
        source_checkpoint_present = stored_checkpoint is not None
        if profile_relationship is not None:
            profile_checkpoint = None
            profile_failure: ValueError | None = None
            try:
                profile_checkpoint = (
                    stored_checkpoint
                    if checkpoint_authority_decoder is None
                    else checkpoint_authority_decoder(
                        None if stored_checkpoint is None else deepcopy(stored_checkpoint)
                    )
                )
                session_fork_rules._validate_profiled_fork_authority(
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
                profile_failure = session_fork_rules._profiled_fork_authority_validation_error(exc)
                stored_checkpoint = None
            finally:
                profile_checkpoint = None
            if profile_failure is not None:
                raise profile_failure from None

        source_transcript = state.transcripts.get(source_session_id, [])
        source_transcript_length = len(source_transcript)
        # Fork and source transcripts may share message objects: internal
        # transcript lists never escape except through the detaching
        # load/append/query boundaries, so the cheap share stays invisible.
        if transcript_cursor is None:
            copied_transcript = [copy_message(message) for message in source_transcript]
        else:
            if transcript_cursor > source_transcript_length:
                source_transcript = []
                raise ValueError("transcript_cursor is greater than source transcript length.")
            copied_transcript = [
                copy_message(message) for message in source_transcript[:transcript_cursor]
            ]
        source_interaction_ids = state.transcript_interaction_ids.get(source_session_id, [])
        copied_interaction_ids = list(source_interaction_ids[: len(copied_transcript)])
        source_transcript_snapshot = (
            None
            if transcript_validator is None
            else TranscriptSnapshot(
                records=[
                    TranscriptRecord(
                        index=index,
                        interaction_id=copied_interaction_ids[index],
                        message=message,
                    )
                    for index, message in enumerate(copied_transcript)
                ],
                cursor=source_transcript_length,
            )
        )
        copied_transcript, copied_interaction_ids = apply_fork_system_prompt_replacement(
            copied_transcript,
            copied_interaction_ids,
            system_prompt_replacement,
        )
        # A cursor may intentionally exclude legacy secret-bearing suffix
        # entries. The source store owns those records; this frame does not.
        source_transcript = []
        if not fork_transcript_is_accepted(
            copied_transcript,
            source_transcript_snapshot,
            transcript_validator,
        ):
            copied_transcript.clear()
            copied_transcript = []
            source_transcript_snapshot = None
            raise ValueError(FORK_TRANSCRIPT_VALIDATION_ERROR) from None
        source_transcript_snapshot = None
        copied_checkpoint = None
        if checkpoint_transform is not None:
            checkpoint_input = None if stored_checkpoint is None else deepcopy(stored_checkpoint)
            stored_checkpoint = None
            copied_checkpoint = transform_fork_checkpoint(
                source_session.model_copy(deep=True),
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
            copied_checkpoint = session_fork_rules._prepare_profiled_fork_checkpoint_result(
                supports_model_failover=state.supports_model_failover_stage_protocol(),
                fork=fork,
                transcript_cursor=len(copied_transcript),
                relationship=profile_relationship,
                source_checkpoint_present=source_checkpoint_present,
                copied_checkpoint=copied_checkpoint,
            )
        initial_operation_records = _prepare_initial_session_operation_records(
            fork,
            operation_initializer,
        )

        state.register_private_authority_alias(
            fork.id,
            field_name="session_id",
        )
        state.sessions[fork.id] = fork.model_copy(deep=True)
        state.index_session_parent(fork)
        state.events[fork.id] = []
        state.event_ids[fork.id] = set()
        state.session_event_records[fork.id] = []
        state.child_lifecycle_event_records[fork.id] = {}
        state.interaction_event_records[fork.id] = {}
        state.latest_interaction_event_records[fork.id] = {}
        state.latest_interaction_event_records_by_sequence[fork.id] = []
        state.pending_action_event_records[fork.id] = {}
        state.session_operation_records[fork.id] = initial_operation_records
        state.transcripts[fork.id] = copied_transcript
        memory_transcripts.replace_latest_role_indexes(state.transcript_indexes(), fork.id)
        memory_transcripts.replace_search_session(
            state.transcript_indexes(), fork.id, copied_transcript
        )
        # Historical attribution remains tied to the source session's
        # interactions; new child work receives new child interaction IDs.
        state.transcript_interaction_ids[fork.id] = copied_interaction_ids
        for interaction_id in set(state.transcript_interaction_ids[fork.id]):
            if interaction_id is None:
                continue
            state.register_private_authority_alias(
                interaction_id,
                field_name="interaction_id",
                scope_session_id=fork.id,
            )
        if copied_checkpoint is not None:
            state.store_checkpoint(fork.id, copied_checkpoint)
        if events:
            state.sessions[fork.id] = state.append_events(
                state.sessions[fork.id],
                events,
            )
        else:
            state.refresh_child_lifecycle_candidate(state.sessions[fork.id])
        return state.sessions[fork.id].model_copy(deep=True)
