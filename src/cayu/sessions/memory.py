"""In-process session storage with owned state, locking and atomic mutations."""

from __future__ import annotations

import asyncio
import base64
import heapq
import math
import secrets
from bisect import bisect_left, bisect_right
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from itertools import islice
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from uuid import uuid4

from pydantic import SecretStr

from cayu._clock import utc_clock, utc_duration_cutoff
from cayu._resource_store_surface import model_store_surface
from cayu._validation import (
    canonical_bounded_durable_json_bytes,
    copy_durable_json_object,
    copy_label_map,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.approvals.tools import ResolutionActor
from cayu.budgets.aggregates import (
    UsageRollupStoreResult,
)
from cayu.budgets.base import BudgetReservationIdentityConflict
from cayu.budgets.pricing import PriceBook
from cayu.collaboration.peer_content import (
    PeerAppendKey,
    PeerContentAppendRequest,
    PeerContentConflict,
    PeerContentExposureReceipt,
    PeerContentExposureRequest,
    PeerContentReceipt,
    PeerContentUnavailable,
)
from cayu.events import Event, EventType, copy_event, event_with_runtime_payload_authority
from cayu.execution_profiles import (
    ExecutionProfileDecision,
    ExecutionProfileIdentity,
    ExecutionProfileRejectionResult,
)
from cayu.execution_units import ToolRoundIdentity
from cayu.memory.evidence import (
    MAX_RECALL_RECEIPT_ITEMS,
    ContextExposure,
    ContextExposurePage,
    ContextExposureTransitionRequest,
    RecallEvidenceConflict,
    RecallEvidenceQuery,
    RecallItemExposure,
    RecallReceipt,
    RecallReceiptPage,
    append_context_exposure_transition,
    context_exposure_creation_matches,
    context_exposure_transition_replays,
    copy_context_exposure,
    copy_recall_item_exposure,
    copy_recall_receipt,
    decode_recall_evidence_cursor,
    encode_recall_evidence_cursor,
    memory_evidence_document_bytes,
    recall_item_exposure_matches_receipt_item,
    require_memory_evidence_id,
    require_memory_evidence_session_id,
    validate_context_exposure_carried_receipt_scope,
    validate_context_exposure_receipt_scope,
    validate_new_context_exposure,
)
from cayu.messages import Message, MessageRole, copy_message, detach_message
from cayu.runtime._child_session_notifications import (
    ChildSessionLifecyclePage,
    ChildSessionLifecycleQuery,
    child_session_notification_stage_binding,
    child_session_notification_storage_key,
)
from cayu.runtime._cost_accounting import CostAccountingSnapshot
from cayu.runtime._usage_accounting import UsageAccountingSnapshot
from cayu.runtime.public_authority import (
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    _authenticated_public_authority_alias_private_value,
    _public_authority_alias_store_key,
    parse_public_authority_alias,
)
from cayu.runtime.service_manifest import RuntimeStoreDurability
from cayu.sessions import _checkpoint_mutations as checkpoint_mutation_rules
from cayu.sessions import _checkpoint_preservation as checkpoint_preservation
from cayu.sessions import _checkpoint_publication as checkpoint_publication
from cayu.sessions import _child_lifecycle_rules as child_lifecycle_rules
from cayu.sessions import _closure_records as closure_records
from cayu.sessions import _completion_finalization as completion_finalization
from cayu.sessions import _execution_profile_rules as session_profile_rules
from cayu.sessions import _interaction_receipts as interaction_receipts
from cayu.sessions import _interaction_transitions as interaction_rules
from cayu.sessions import _invocation_scope as invocation_scope
from cayu.sessions import _memory_event_reads as memory_event_reads
from cayu.sessions import _memory_session_queries as memory_session_queries
from cayu.sessions import _memory_transcripts as memory_transcripts
from cayu.sessions import _model_failover_rules as model_failover_rules
from cayu.sessions import _model_stage_publication as model_stage_publication
from cayu.sessions import _model_stage_rules as model_stage_rules
from cayu.sessions import _operation_records as operation_record_rules
from cayu.sessions import _runtime_publication as publication_contracts
from cayu.sessions import _runtime_publication_rules as publication_rules
from cayu.sessions import _terminal_evidence as terminal_event_evidence
from cayu.sessions import _tool_round_publication_rules as tool_publication_rules
from cayu.sessions import _user_input_publication as user_input_publication_rules
from cayu.sessions import authority as session_authority_rules
from cayu.sessions import creation_fence
from cayu.sessions import event_delivery as session_event_rules
from cayu.sessions import event_delivery as side_effect_health
from cayu.sessions import forks as session_fork_rules
from cayu.sessions import inspection as session_inspection_rules
from cayu.sessions import lineage as lineage_rules
from cayu.sessions import messaging as session_messaging
from cayu.sessions import records as session_record_rules
from cayu.sessions import recovery as session_recovery_rules
from cayu.sessions import requests as session_request_contracts
from cayu.sessions import terminal_evidence as terminal_session_evidence
from cayu.sessions import transcript_input as session_transcript_input
from cayu.sessions._access_rules import _RECIPIENT_PROVENANCE_CAPABILITY
from cayu.sessions._checkpoint_mutations import (
    RuntimePublicationMutation,
    SessionRuntimePublicationConflict,
)
from cayu.sessions._execution_profile_checkpoint import ActiveInvocationExecutionProfile
from cayu.sessions._execution_profile_rules import SessionModelTransition
from cayu.sessions._external_wait_memory import MemoryExternalWaitMixin
from cayu.sessions._interaction_transitions import (
    InteractionTransitionReceiptResult,
    InteractionTransitionResult,
    InteractionTransitionSpec,
    QueuedInteractionProfileHandoff,
)
from cayu.sessions._invocation_scope import (
    INHERIT_INTERACTION,
    InteractionAttribution,
    resolve_interaction_attribution,
)
from cayu.sessions._invocation_terminal_decision import InvocationTerminalDecision
from cayu.sessions._model_completion_publication import ModelCompletionStageSettlementRequest
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    ActiveModelCompletionStage,
    ModelCompletionStage,
    ModelCompletionStageAbandonmentResult,
    ModelCompletionStageDispatch,
    ModelCompletionStageResult,
    SessionModelCompletionDispatchAlreadyAuthorized,
    SessionModelCompletionStageConflict,
)
from cayu.sessions._operation_records import SessionOperationPublication
from cayu.sessions._provider_operation_cancellation_claim import (
    active_provider_operation_cancellation_claim_from_checkpoint,
)
from cayu.sessions._runtime_publication import (
    RuntimePublicationReceipt,
    RuntimePublicationResult,
    SessionOperationInitializer,
    SessionOperationTransform,
    StoreTimeSessionOperationTransform,
    _prepare_initial_session_operation_records,
    _run_session_commit_guard_owned,
)
from cayu.sessions._terminal_evidence import (
    QueuedDispatchTerminalReceipt,
    QueuedDispatchTerminalReceiptQuery,
)
from cayu.sessions.authority import (
    CheckpointRootFieldGuard,
    CheckpointValueAuthority,
    SessionRunFenced,
    checkpoint_root_field_projection,
)
from cayu.sessions.creation import session_invocation_for_run_request, session_metadata_for_creation
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectClaim,
    PersistedEventSideEffectClaimLost,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectHealth,
    PersistedEventSideEffectPage,
    PersistedEventSideEffectQuery,
    PersistedEventSideEffectStatus,
    _copy_workflow_step_reservation,
    validate_persisted_event_side_effect_error,
)
from cayu.sessions.event_queries import (
    EventQuery,
)
from cayu.sessions.exports import _require_session_export_target
from cayu.sessions.forks import (
    FORK_TRANSCRIPT_VALIDATION_ERROR,
    ForkCheckpointAuthorityDecoder,
    ForkSystemPromptReplacement,
    ProfiledSessionForkResult,
    SessionForkActiveModelStageConflict,
    SessionForkProfileRelationship,
    apply_fork_system_prompt_replacement,
    transform_fork_checkpoint,
)
from cayu.sessions.inspection import SessionInspectionIdentity
from cayu.sessions.lineage import (
    DELETE_BLOCKED_SESSION_STATUSES,
    SessionLineageQuery,
    SessionLineageResult,
)
from cayu.sessions.mcp_manifest_history import (
    McpManifestBaseline,
    McpManifestBaselineLoadResult,
    McpManifestPublicationResult,
    _copy_mcp_manifest_publication,
    _validate_mcp_manifest_history_keys,
    _validate_mcp_manifest_publication_state,
)
from cayu.sessions.messaging import (
    SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    EnqueueSessionMessageRequest,
    EnqueueSessionMessageResult,
    SessionMessageActionRequest,
    SessionMessageActionResult,
    SessionMessageConditions,
    SessionMessageConflict,
    SessionMessageDeliveryBatch,
    SessionMessageDeliveryMode,
    SessionMessageInspection,
    SessionMessageInspectionRecord,
    SessionMessageQuery,
    SessionMessageQueueStatus,
    SessionMessageSource,
    SessionQueuedMessage,
    SessionQueuedMessagesPending,
    copy_enqueue_session_message_request,
    copy_session_message_action,
    enqueue_session_message_input,
    queued_session_message_input,
    session_message_checkpoint_sha256,
    session_message_rejection,
)
from cayu.sessions.pending_action_contracts import (
    PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES,
    PENDING_ACTION_EVENT_TYPE_VALUES,
    PendingActionListResult,
    PendingActionQuery,
)
from cayu.sessions.queries import (
    SessionAggregateFilter,
    SessionListResult,
    SessionQuery,
    _validate_inactive_for_seconds,
    copy_session_query,
)
from cayu.sessions.records import (
    CheckpointTransform,
    EventRecord,
    RunnerObservedEventIdentity,
    Session,
    SessionIdentity,
    SessionInvocationSnapshot,
    SessionRuntimeIdentity,
    SessionStateSnapshot,
    SessionStatus,
    SessionStatusConflict,
    StoreTimeCheckpointTransform,
    TranscriptRecord,
    copy_session,
    copy_session_identity,
    copy_session_runtime_identity,
    copy_session_user_metadata,
    replace_session_user_metadata,
)
from cayu.sessions.recovery import (
    _INTERRUPTION_CASCADE_MISSING,
    _project_interruption_cascade_marker,
)
from cayu.sessions.requests import (
    RunRequest,
    copy_run_request,
    deferred_interaction_input_for_run_request,
    session_instance_id_for_run_request,
)
from cayu.sessions.store import SessionStore
from cayu.sessions.summaries import (
    EventSummary,
    SessionOperationalSnapshot,
    SessionOutcome,
)
from cayu.sessions.terminal_evidence import (
    TerminalSessionEvidence,
    TerminalSessionEvidenceLimits,
)
from cayu.sessions.topology import (
    SessionTopologyQuery,
    SessionTopologyStoreResult,
)
from cayu.sessions.transcript_input import (
    SESSION_STARTED_INPUT_CONTRACT_PAYLOAD_KEY,
    DeferredInteractionInput,
    copy_transcript_messages,
    require_deferred_initial_transcript_replacement,
    session_messages_input_contract_evidence,
)
from cayu.sessions.transcript_queries import (
    ForkTranscriptValidator,
    TranscriptPage,
    TranscriptQuery,
    TranscriptSearchQuery,
    TranscriptSearchResult,
    TranscriptSnapshot,
    fork_source_transcript_sha256,
    fork_transcript_is_accepted,
)
from cayu.sessions.usage import UsageRollupQuery
from cayu.storage._creation_fence import MemoryCreationFenceMixin
from cayu.storage._session_execution import MemorySessionExecutionMixin
from cayu.tools.exposure import ToolCapabilityCeiling
from cayu.tools.grants import (
    TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    TARGETED_TOOL_GRANT_MAX_REQUESTS,
    TARGETED_TOOL_REFERENCE_FIELD_NAME,
    TargetedToolGrantIssueOutcome,
    TargetedToolGrantIssueResult,
    TargetedToolGrantReconstructionResult,
    TargetedToolGrantRecord,
    TargetedToolGrantStateSnapshot,
    TargetedToolUseBinding,
    TargetedToolUseDisposition,
    TargetedToolUseRejectionReason,
    TargetedToolUseRequest,
    TargetedToolUseResult,
    copy_targeted_tool_grant_record,
    targeted_tool_grant_event,
    targeted_tool_grant_reconstruction_rejection_reason,
    targeted_tool_grant_with_active_reference,
    targeted_tool_unresolved_rejection_event,
    targeted_tool_use_binding,
    targeted_tool_use_rejection_event,
    targeted_tool_use_rejection_reason,
    targeted_tool_use_scope_rejection_reason,
    validate_targeted_tool_grant_batch_evidence,
    validate_targeted_tool_grant_issuance_evidence,
    validate_targeted_tool_grant_lifecycle_event,
    validate_targeted_tool_grant_reference,
    validate_targeted_tool_grant_revocation_evidence,
    validate_targeted_tool_grant_revocation_reason,
    validate_targeted_tool_unresolved_rejection_evidence,
    validate_targeted_tool_use_rejection_evidence,
)
from cayu.workflows.base import WORKFLOW_ATTEMPT_EVENT_TYPE

if TYPE_CHECKING:
    from cayu.collaboration.participants import ParticipantRef as _ParticipantRef
    from cayu.runtime._zero_work_interruption import (
        ZeroWorkInterruptionPublication,
        ZeroWorkInterruptionRequest,
    )
    from cayu.sessions._access_rules import _SessionAccessBounds
    from cayu.sessions._context_view_source import (
        CompletedTurnSnapshot,
        ContextViewPublicationSource,
    )
    from cayu.sessions._temporary_continuation import TemporaryServiceAdmission
    from cayu.sessions.context_views import ContextViewLifecycleEvent as _ContextViewLifecycleEvent
    from cayu.sessions.context_views import ContextViewManifest as _ContextViewManifest
    from cayu.sessions.context_views import (
        ContextViewOwnershipRequest as _ContextViewOwnershipRequest,
    )
    from cayu.sessions.context_views import ContextViewReadback as _ContextViewReadback
    from cayu.sessions.context_views import (
        ContextViewSelectionReceipt as _ContextViewSelectionReceipt,
    )
    from cayu.sessions.context_views import (
        ContextViewSelectionRequest as _ContextViewSelectionRequest,
    )
    from cayu.sessions.context_views import (
        ParticipantSessionBinding,
        ParticipantSessionCreationReceipt,
        ParticipantSessionCreationRequest,
    )
    from cayu.sessions.exports import SessionExportLimits, SessionExportSnapshot


@dataclass(frozen=True)
class _PreparedInMemoryEvent:
    record: EventRecord
    event_type: str
    projected_record: EventRecord | None
    retained_lookup_keys: tuple[str, ...]
    history_lookup_keys: tuple[str, ...]
    delivery: PersistedEventSideEffectDelivery | None
    invocation_terminal_receipt: tuple[str, dict[str, Any]] | None


@dataclass(frozen=True)
class _PreparedInMemoryEventAppend:
    events: tuple[_PreparedInMemoryEvent, ...]
    next_sequence: int
    budget_reservation_publications: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _PreparedInMemoryCheckpointStore:
    checkpoint: dict[str, Any]
    has_pending_action: bool
    rebuilt_records: dict[str, dict[str, EventRecord]] | None = None
    rebuilt_event_history: dict[str, list[EventRecord]] | None = None
    rebuilt_index_scope: tuple[frozenset[str], str | None, str | None, str | None] | None = None


@dataclass(frozen=True)
class _InMemoryMessageDeliveryRecord:
    session_id: str
    include_on_idle: bool
    requested_eligible_through: int | None
    limit: int
    interaction_id: str | None
    interaction_started_event: Event | None
    profile_handoff: QueuedInteractionProfileHandoff | None
    batch: SessionMessageDeliveryBatch
    reject_only: bool = False


class _InMemoryContextViewSelectionTransaction:
    """Rollback selection-side expiry evidence when validation rejects a request."""

    def __init__(self, store: InMemorySessionStore) -> None:
        self._store = store
        self._receipts: dict[str, Any] | None = None
        self._commitments: dict[str, str] | None = None
        self._events: dict[str, Any] | None = None

    async def __aenter__(self) -> _InMemoryContextViewSelectionTransaction:
        await self._store._lock.acquire()
        self._receipts = self._store._context_selection_receipts.copy()
        self._commitments = self._store._context_selection_request_commitments.copy()
        self._events = self._store._context_lifecycle_events.copy()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        if exc_type is not None:
            assert self._receipts is not None
            assert self._commitments is not None
            assert self._events is not None
            self._store._context_selection_receipts = self._receipts
            self._store._context_selection_request_commitments = self._commitments
            self._store._context_lifecycle_events = self._events
        self._store._lock.release()


def runtime_session_mutation(operation):
    from functools import wraps

    @wraps(operation)
    async def guarded(self, *args, **kwargs):
        from cayu.sessions._access_rules import runtime_session_mutation as wrap

        return await wrap(operation)(self, *args, **kwargs)

    return guarded


def require_resource_session(session, action="read"):
    from cayu.sessions._access_rules import require_resource_session as require

    require(session, action)


def runtime_session_query(operation):
    from functools import wraps

    @wraps(operation)
    async def guarded(self, *args, **kwargs):
        from cayu.sessions._access_rules import runtime_session_query as wrap

        return await wrap(operation)(self, *args, **kwargs)

    return guarded


@model_store_surface("sessions")
class InMemorySessionStore(
    MemoryExternalWaitMixin, MemorySessionExecutionMixin, MemoryCreationFenceMixin, SessionStore
):
    """In-process session store for tests, local development, and examples."""

    session_access_version: ClassVar[int | None] = 1

    async def _access_create_session(self, bounds, request, identity):
        from cayu.sessions._access_rules import _creation_bounds

        token = _creation_bounds.set(bounds)
        try:
            return await self.create(request, identity=identity)
        finally:
            _creation_bounds.reset(token)

    async def _access_update_metadata(self, bounds, session_id, metadata):
        return await self.update_metadata(session_id, metadata, _access_bounds=bounds)

    async def _access_delete_session(self, bounds, session_id):
        await self.delete_session(session_id, _access_bounds=bounds)

    async def _access_read_records(self, bounds, session_id, kind, offset, limit, max_bytes):
        from cayu.storage._session_access_records import memory_read

        return await memory_read(self, bounds, session_id, kind, offset, limit, max_bytes)

    async def _access_list_sessions(
        self, bounds: _SessionAccessBounds, query: SessionQuery
    ) -> SessionListResult:
        return await memory_session_queries.list_sessions(
            self._read_session_query_state,
            query,
            pending_interruption_cascade_only=False,
            access_bounds=bounds,
        )

    async def _access_update_labels(
        self, bounds: _SessionAccessBounds, session_id: str, labels: dict[str, str]
    ) -> Session:
        return await self.update_labels(session_id, labels, _access_bounds=bounds)

    async def _access_load_session(self, bounds: _SessionAccessBounds, session_id: str) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            return bounds.require_read(self._sessions.get(session_id)).model_copy(deep=True)

    peer_content_version: ClassVar[int | None] = 1

    session_export_version: ClassVar[int] = 1
    session_continuation_version: ClassVar[int] = 1
    external_wait_version: ClassVar[int] = 1
    _producer_attachment_version: ClassVar[int] = 1
    supports_session_closure_receipts: ClassVar[bool] = True
    supports_session_closure_detachment: ClassVar[bool] = True
    supports_session_closure_recursive_deletion: ClassVar[bool] = True
    supports_session_closure_progress: ClassVar[bool] = True

    async def load_session_closure_records(
        self, session_id: str, *, max_records: int, max_bytes: int
    ) -> dict[str, Any]:
        async with self._lock:
            return self._load_session_closure_records_unlocked(
                session_id, max_records=max_records, max_bytes=max_bytes
            )

    def _load_session_closure_records_unlocked(
        self, session_id: str, *, max_records: int, max_bytes: int
    ) -> dict[str, Any]:
        from cayu.sessions._closure_records import ClosureRecordsBuilder

        session_id = require_clean_nonblank(session_id, "session_id")
        builder = ClosureRecordsBuilder(max_records=max_records, max_bytes=max_bytes)
        session = self._sessions.get(session_id)
        builder.add_class(
            "session",
            ()
            if session is None
            else (
                {
                    name: getattr(session, name)
                    for name in type(session).model_fields
                    if name not in {"labels", "metadata"}
                },
            ),
        )
        builder.add_class(
            "labels",
            ()
            if session is None
            else ({"key": key, "value": value} for key, value in session.labels.items()),
        )
        builder.add_class("metadata", () if session is None else (session.metadata,))
        builder.add_class(
            "recall_receipts",
            (
                self._recall_receipts[receipt_id]
                for _, receipt_id in self._recall_receipt_page_keys_by_session.get(session_id, ())
            ),
        )
        builder.add_class(
            "context_exposures",
            (
                self._context_exposures[exposure_id]
                for _, exposure_id in self._context_exposure_page_keys_by_session.get(
                    session_id, ()
                )
            ),
        )
        builder.add_class(
            "recall_item_exposures",
            (
                item
                for _, exposure_id in self._context_exposure_page_keys_by_session.get(
                    session_id, ()
                )
                for item in self._recall_item_exposures[exposure_id]
            ),
        )
        builder.add_class("events", self._session_event_records.get(session_id, ()))
        builder.add_class(
            "transcript",
            (
                {
                    "transcript_index": index,
                    "message": message,
                    "interaction_id": self._transcript_interaction_ids[session_id][index],
                }
                for index, message in enumerate(self._transcripts.get(session_id, ()))
            ),
        )
        checkpoint = self._checkpoints.get(session_id)
        builder.add_class("checkpoint", () if checkpoint is None else (checkpoint,))
        builder.add_class(
            "queued_messages",
            (
                {
                    "message": message,
                    "terminal": self._session_message_terminal_receipts.get(
                        (session_id, message.queue_id)
                    ),
                }
                for message in self._queued_session_messages_by_idempotency.get(
                    session_id, {}
                ).values()
            ),
        )
        builder.add_class(
            "session_operations",
            (
                {"idempotency_key": key, "record": value}
                for key, value in self._session_operation_records.get(session_id, {}).items()
            ),
        )
        builder.add_class(
            "event_side_effect_deliveries",
            (
                delivery
                for record in self._session_event_records.get(session_id, ())
                if (
                    delivery := self._persisted_event_side_effect_deliveries.get(
                        (session_id, record.event.id)
                    )
                )
                is not None
            ),
        )
        builder.add_class(
            "queue_deliveries",
            (
                {
                    "delivery_id": key,
                    "session_id": value.session_id,
                    "include_on_idle": value.include_on_idle,
                    "requested_eligible_through": value.requested_eligible_through,
                    "eligible_through": value.batch.eligible_through,
                    "batch_limit": value.limit,
                    "has_more": value.batch.has_more,
                    "interaction_id": value.interaction_id,
                    "interaction_started_event": value.interaction_started_event,
                    "queue_ids": [message.queue_id for message in value.batch.messages],
                    "events": value.batch.events,
                    "reject_only": value.reject_only,
                }
                for key, value in self._session_message_delivery_records.items()
                if value.session_id == session_id
            ),
        )
        deferred = self._deferred_interaction_inputs.get(session_id)
        builder.add_class("deferred_interaction_inputs", () if deferred is None else (deferred,))
        builder.add_class(
            "targeted_tool_grants",
            (
                targeted_tool_grant_with_active_reference(
                    self._targeted_tool_grants[grant_id], self.public_authority_alias_codec
                )
                for grant_id in self._targeted_tool_grant_ids_by_session.get(session_id, ())
            ),
        )
        builder.add_class(
            "targeted_tool_grant_uses",
            (
                use
                for grant_id in self._targeted_tool_grant_ids_by_session.get(session_id, ())
                for use in self._targeted_tool_uses.get(grant_id, {}).values()
            ),
        )
        return builder.finish()

    async def load_session_closure_receipt(
        self, session_id: str, plan_id: str
    ) -> dict[str, Any] | None:
        require_clean_nonblank(session_id, "session_id")
        require_clean_nonblank(plan_id, "plan_id")
        async with self._lock:
            receipt = self._session_closure_receipts.get((session_id, plan_id))
            return None if receipt is None else deepcopy(receipt)

    async def load_session_closure_progress(self, session_id: str, plan_id: str):
        async with self._lock:
            value = self._session_closure_progress.get((session_id, plan_id))
            return None if value is None else deepcopy(value)

    async def save_session_closure_progress(self, progress: dict[str, Any]) -> None:
        root_id = progress.get("root_session_id")
        plan_id = progress.get("plan_id")
        if type(root_id) is not str or type(plan_id) is not str:
            raise ValueError("Closure progress is missing identity.")
        async with self._lock:
            existing = self._session_closure_progress.get((root_id, plan_id))
            if existing is not None:
                closure_records._validate_closure_progress_update(existing, progress)
            self._session_closure_progress[(root_id, plan_id)] = deepcopy(progress)

    async def claim_session_closure_progress(self, progress: dict[str, Any]) -> None:
        progress = deepcopy(progress)
        async with self._lock:
            key = (progress["root_session_id"], progress["plan_id"])
            existing = self._session_closure_progress.get(key)
            if existing is not None:
                closure_records._validate_closure_progress_update(existing, progress)
                return
            targets = closure_records._closure_progress_targets(progress)
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, targets)
            if progress["root_session_id"] not in self._sessions:
                raise ValueError("Closure root disappeared before lineage admission.")
            for item in progress["descendants"]:
                child = self._sessions.get(item["session_id"])
                if child is None or child.parent_session_id != item["parent_session_id"]:
                    raise ValueError("Child lineage changed before closure admission.")
            if progress["phase"] in {"recursive", "reject"}:
                expected = {
                    (item["session_id"], item["parent_session_id"])
                    for item in progress["descendants"]
                }
                observed = set()
                for parent_id in targets:
                    for _, child_id in self._child_session_keys_by_parent.get(parent_id, ()):
                        edge = (child_id, parent_id)
                        if edge not in expected:
                            raise ValueError("Child lineage changed before closure admission.")
                        observed.add(edge)
                if observed != expected:
                    raise ValueError("Child lineage changed before closure admission.")
            for target_id in targets:
                self._require_session_erasure_quiescence_unlocked(self._sessions[target_id])
                self._load_session_closure_records_unlocked(
                    target_id, max_records=progress["max_records"], max_bytes=progress["max_bytes"]
                )
            self._session_closure_progress[key] = progress

    def _require_available_closure_identity_unlocked(self, session_id: str) -> None:
        for owner in self._session_closure_progress.values():
            closure_records._check_closure_lineage_owner(owner, (session_id,))
        if any(target == session_id for target, _plan in self._session_closure_receipts):
            raise ValueError("Session identity was retired by closure.")

    async def load_session_closure_tombstones(
        self, root_session_id: str, plan_id: str
    ) -> tuple[dict[str, Any], ...]:
        require_clean_nonblank(root_session_id, "root_session_id")
        require_clean_nonblank(plan_id, "plan_id")
        async with self._lock:
            return tuple(
                deepcopy(value)
                for (root_id, stored_plan, _child_id), value in sorted(
                    self._session_closure_tombstones.items()
                )
                if root_id == root_session_id and stored_plan == plan_id
            )

    async def detach_session_children(
        self,
        parent_session_id: str,
        child_session_ids: tuple[str, ...],
        *,
        closure_receipt: dict[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        parent_session_id = require_clean_nonblank(parent_session_id, "parent_session_id")
        child_session_ids = tuple(
            require_clean_nonblank(child_id, "child_session_id") for child_id in child_session_ids
        )
        if len(set(child_session_ids)) != len(child_session_ids):
            raise ValueError("Detached child session IDs must be unique.")
        root_id = closure_receipt.get("root_session_id")
        plan_id = closure_receipt.get("plan_id")
        if type(root_id) is not str or type(plan_id) is not str:
            raise ValueError("Closure detachment receipt is missing identity.")
        async with self._lock:
            existing = tuple(
                deepcopy(value)
                for (stored_root, stored_plan, _child_id), value in sorted(
                    self._session_closure_tombstones.items()
                )
                if stored_root == root_id and stored_plan == plan_id
            )
            if existing:
                closure_records._validate_session_closure_detach_replay(
                    existing,
                    root_id=root_id,
                    plan_id=plan_id,
                    parent_session_id=parent_session_id,
                    child_session_ids=child_session_ids,
                )
                return existing
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, child_session_ids)
            parent = self._sessions.get(parent_session_id)
            if parent is None:
                raise KeyError(parent_session_id)
            tombstones: list[dict[str, Any]] = []
            prepared = []
            detached_at = self._ownership_clock()
            for child_id in child_session_ids:
                child = self._sessions.get(child_id)
                if child is None or child.parent_session_id != parent_session_id:
                    raise ValueError("Child lineage changed before detachment.")
                tombstone = {
                    "root_session_id": root_id,
                    "plan_id": plan_id,
                    "child_session_id": child_id,
                    "original_parent_session_id": parent_session_id,
                    "detached_at": detached_at.isoformat(),
                }
                detached = child.model_copy(
                    update={"parent_session_id": None, "updated_at": detached_at}
                )
                prepared.append((child_id, child, detached, deepcopy(tombstone)))
                tombstones.append(tombstone)
            for child_id, child, detached, tombstone in prepared:
                self._remove_session_parent_index_unlocked(child)
                self._sessions[child_id] = detached
                self._refresh_child_lifecycle_candidate_unlocked(detached)
                self._session_closure_tombstones[(root_id, plan_id, child_id)] = tombstone
            return tuple(tombstones)

    supports_usage_aggregates: ClassVar[bool] = True
    supports_private_argument_continuity: ClassVar[bool] = True
    supports_mcp_manifest_history: ClassVar[bool] = True
    supports_session_topology: ClassVar[bool] = True
    supports_session_lineage: ClassVar[bool] = True
    child_session_notification_version: ClassVar[int | None] = 1
    supports_public_authority_aliases: ClassVar[bool] = True
    supports_targeted_tool_grants: ClassVar[bool] = True
    supports_terminal_session_evidence: ClassVar[bool] = True
    supports_runner_owned_interrupted_evidence: ClassVar[bool] = True
    supports_execution_profile_admission: ClassVar[bool] = True
    model_failover_stage_version: ClassVar[int] = 1
    supports_active_invocation_execution_profiles: ClassVar[bool] = True
    invocation_lifecycle_command_version: ClassVar[int | None] = 1
    terminal_interaction_publication_version: ClassVar[int | None] = 1
    durable_model_terminalization_version: ClassVar[int | None] = 1
    queued_interaction_profile_handoff_version: ClassVar[int | None] = 1
    session_message_lifecycle_version: ClassVar[int | None] = 1
    supports_pending_session_initial_checkpoint: ClassVar[bool] = True
    supports_profiled_forks: ClassVar[bool] = True
    supports_atomic_session_operation_initialization: ClassVar[bool] = True
    supports_atomic_model_completion_stage_release: ClassVar[bool] = True
    model_completion_recovery_fence_version: ClassVar[int] = 1
    session_steering_version: ClassVar[int | None] = 1
    supports_completion_result_event_publication_reservations: ClassVar[bool] = True
    supports_transcript_search: ClassVar[bool] = True
    supports_recall_evidence: ClassVar[bool] = True
    supports_owned_off_thread_session_commit_guards: ClassVar[bool] = True
    participant_session_binding_version: ClassVar[int | None] = 1
    recipient_continuation_selection_version: ClassVar[int | None] = 1
    context_view_selection_fence_version: ClassVar[int | None] = 1
    context_view_version: ClassVar[int | None] = 1
    service_durability: RuntimeStoreDurability = RuntimeStoreDurability.DEVELOPMENT

    def __init__(
        self,
        *,
        public_authority_alias_codec: PublicAuthorityAliasCodec | None = None,
        ownership_clock: Callable[[], datetime] | None = None,
    ) -> None:
        from cayu.runtime._cost_accounting_refresh import CostAccountingAuthority
        from cayu.runtime._usage_accounting import SessionUsageCache

        self._cost_accounting_authority = CostAccountingAuthority()
        self._session_usage_cache = SessionUsageCache()
        self._accounting_generation = 0
        if public_authority_alias_codec is not None and not isinstance(
            public_authority_alias_codec,
            PublicAuthorityAliasCodec,
        ):
            raise TypeError("public_authority_alias_codec must be a PublicAuthorityAliasCodec.")
        if public_authority_alias_codec is None:
            # In-memory stores have no restart or multi-worker key-distribution
            # boundary. Give each store lifetime a cryptographically random key;
            # persistent stores require explicit deployment provisioning.
            encoded_key = (
                base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
            )
            public_authority_alias_codec = PublicAuthorityAliasCodec(
                PublicAuthorityAliasKeyring(
                    active_key_id="memory",
                    keys={"memory": SecretStr(encoded_key)},
                )
            )
        self._public_authority_alias_codec = public_authority_alias_codec
        self._ownership_clock = utc_clock(ownership_clock)
        self._lock = asyncio.Lock()
        self._participant_creation_lock = asyncio.Lock()
        self._session_creation_decisions = {}
        self._execution_owners = {}
        self._producer_cleanup_receipts = {}
        self._producer_cleanup_index = {}
        self._producer_cleanup_retirements = {}
        self._sessions: dict[str, Session] = {}
        self._participant_session_bindings: dict[str, ParticipantSessionBinding] = {}
        self._participant_session_receipts: dict[str, ParticipantSessionCreationReceipt] = {}
        self._participant_session_ids_by_key: dict[str, str] = {}
        self._participant_session_request_commitments: dict[str, str | None] = {}
        self._context_views: dict[str, Any] = {}
        self._context_publication_keys: dict[str, str] = {}
        self._context_selection_receipts: dict[str, Any] = {}
        self._context_selection_request_commitments: dict[str, str] = {}
        self._context_selection_exclusions: dict[str, Any] = {}
        self._context_ownership_operation_commitments: dict[str, str] = {}
        self._context_ownership_operation_receipts: dict[str, Any] = {}
        self._context_lifecycle_events: dict[str, Any] = {}
        self._peer_content_receipts: dict[str, Any] = {}
        self._peer_content_deleted_targets: set[tuple[str, str]] = set()
        self._peer_content_operation_commitments: dict[str, str] = {}
        self._peer_content_operations: dict[str, str] = {}
        self._peer_content_requests: dict[str, PeerContentAppendRequest] = {}
        self._peer_content_attempts: dict[str, tuple[str, str]] = {}
        self._peer_content_exposures: dict[str, tuple[str, PeerContentExposureReceipt]] = {}
        self._session_closure_receipts: dict[tuple[str, str], dict[str, Any]] = {}
        self._session_closure_progress: dict[tuple[str, str], dict[str, Any]] = {}
        self._session_closure_tombstones: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._public_authority_aliases: dict[tuple[str, str, str], str] = {}
        self._targeted_tool_grants: dict[str, TargetedToolGrantRecord] = {}
        self._targeted_tool_grant_ids_by_session: dict[str, list[str]] = {}
        self._targeted_tool_grant_ids_by_interaction: dict[tuple[str, str], list[str]] = {}
        self._targeted_tool_grant_ids_by_request: dict[tuple[str, str, str], str] = {}
        self._targeted_tool_grant_ids_by_tool: dict[tuple[str, str, str], str] = {}
        self._targeted_tool_grant_refs: dict[str, tuple[str, str]] = {}
        self._targeted_tool_uses: dict[str, dict[str, TargetedToolUseBinding]] = {}
        self._targeted_tool_use_by_invocation: dict[
            tuple[str, str, str], TargetedToolUseBinding
        ] = {}
        self._targeted_tool_use_by_outer_call: dict[
            tuple[str, str, str], TargetedToolUseBinding
        ] = {}
        # Stable direct-child keys maintained with session lifecycle writes. Topology
        # pages can therefore seek one parent branch without scanning the complete
        # in-memory session registry.
        self._child_session_keys_by_parent: dict[str, list[tuple[datetime, str]]] = {}
        # Canonical priority indexes keep child-context inspection bounded. Each
        # child has exactly one key derived from its session status, latest
        # lifecycle occurrence, and exact durable notification consumption.
        self._child_lifecycle_candidate_keys_by_parent: dict[
            str,
            dict[int, list[tuple[datetime, str]]],
        ] = {}
        self._child_lifecycle_candidate_key_by_child: dict[
            str,
            tuple[str, int, tuple[datetime, str]],
        ] = {}
        self._child_lifecycle_event_records: dict[
            str,
            dict[EventType, EventRecord],
        ] = {}
        self._events: dict[str, list[Event]] = {}
        self._event_records: list[EventRecord] = []
        # Secondary indexes over ``_event_records`` (same EventRecord objects, kept in
        # sequence-ascending order) so per-session summaries and type/session-scoped
        # queries — including the per-step budget read path — stop scanning the global
        # event list. Keyed by session id and by ``str(event.type)`` respectively.
        self._session_event_records: dict[str, list[EventRecord]] = {}
        self._interaction_event_records: dict[
            str,
            dict[str, list[EventRecord]],
        ] = {}
        self._latest_interaction_event_records: dict[str, dict[str, EventRecord]] = {}
        self._latest_interaction_event_records_by_sequence: dict[
            str,
            list[EventRecord],
        ] = {}
        self._pending_action_event_records: dict[
            str,
            dict[str, dict[str, EventRecord]],
        ] = {}
        # The identity scope represented by each bounded pending-action index.
        # A checkpoint can reuse a tool-call id for a later execution, so lookup
        # keys alone are insufficient to decide whether rebuilding is required.
        self._pending_action_index_scopes: dict[
            str,
            tuple[frozenset[str], str | None, str | None, str | None],
        ] = {}
        # Unlike the bounded pending-action projection above, runtime publication
        # validation must observe *every* lifecycle event for the active tool
        # round so a caller cannot omit a contradictory second terminal event.
        # This history is identifier-scoped and released with the pending marker.
        self._pending_action_event_history: dict[
            str,
            dict[str, list[EventRecord]],
        ] = {}
        # Supersession lookup remains available after the pause checkpoint and
        # its general pending-action history are released.
        self._user_input_supersession_event_records: dict[
            tuple[str, str],
            list[EventRecord],
        ] = {}
        self._pending_action_latest_barrier_records: dict[str, EventRecord] = {}
        self._event_records_by_id: dict[tuple[str, str], EventRecord] = {}
        self._type_event_records: dict[str, list[EventRecord]] = {}
        from cayu.runtime._cost_accounting_index import CostEventIndex

        self._cost_event_index = CostEventIndex()
        self._workflow_step_event_records: dict[
            tuple[str, str, str, str, str],
            list[EventRecord],
        ] = {}
        self._workflow_fenced_step_event_records: dict[
            tuple[str, str, str, str],
            list[EventRecord],
        ] = {}
        self._workflow_attempt_event_records: dict[
            tuple[str, str],
            list[EventRecord],
        ] = {}
        self._workflow_active_attempt_ids: dict[tuple[str, str], str] = {}
        self._event_ids: dict[str, set[str]] = {}
        # Store-wide ownership registry. It intentionally outlives session
        # deletion: reservation ids are ledger reconciliation keys, not
        # session-local event attributes.
        self._budget_reservation_identities: dict[str, tuple[str, str, bool]] = {}
        self._persisted_event_side_effect_deliveries: dict[
            tuple[str, str], PersistedEventSideEffectDelivery
        ] = {}
        self._next_event_sequence = 1
        self._transcripts: dict[str, list[Message]] = {}
        self._transcript_interaction_ids: dict[str, list[str | None]] = {}
        self._latest_transcript_indexes_by_role: dict[str, dict[MessageRole, int]] = {}
        self._transcript_indices_by_interaction: dict[str, dict[str | None, list[int]]] = {}
        self._transcript_search_documents: dict[str, list[tuple[MessageRole, str]]] = {}
        self._transcript_search_postings: dict[tuple[str, MessageRole], dict[str, list[int]]] = {}
        self._deferred_interaction_inputs: dict[str, DeferredInteractionInput] = {}
        self._checkpoints: dict[str, dict[str, Any]] = {}
        self._recall_receipts: dict[str, RecallReceipt] = {}
        self._recall_receipt_ids_by_session: dict[str, set[str]] = {}
        self._recall_receipt_page_keys_by_session: dict[str, list[tuple[datetime, str]]] = {}
        self._recall_receipt_page_keys_by_interaction: dict[
            tuple[str, str], list[tuple[datetime, str]]
        ] = {}
        self._recall_receipt_page_keys_by_step: dict[
            tuple[str, str], list[tuple[datetime, str]]
        ] = {}
        self._recall_receipt_page_keys_by_interaction_step: dict[
            tuple[str, str, str], list[tuple[datetime, str]]
        ] = {}
        self._context_exposures: dict[str, ContextExposure] = {}
        self._context_exposure_ids_by_session: dict[str, set[str]] = {}
        self._context_exposure_page_keys_by_session: dict[str, list[tuple[datetime, str]]] = {}
        self._context_exposure_page_keys_by_interaction: dict[
            tuple[str, str], list[tuple[datetime, str]]
        ] = {}
        self._context_exposure_page_keys_by_step: dict[
            tuple[str, str], list[tuple[datetime, str]]
        ] = {}
        self._context_exposure_page_keys_by_interaction_step: dict[
            tuple[str, str, str], list[tuple[datetime, str]]
        ] = {}
        self._recall_item_exposures: dict[str, tuple[RecallItemExposure, ...]] = {}
        self._context_exposure_by_model_attempt: dict[tuple[str, str], str] = {}
        self._context_exposure_by_provider_attempt: dict[tuple[str, str], str] = {}
        # Live queued-dispatch handoffs are indexed incrementally so restart
        # reconciliation is bounded by the requested page, not by historical
        # sessions or events.
        self._queued_dispatch_terminal_receipts: dict[
            tuple[str, str], QueuedDispatchTerminalReceipt
        ] = {}
        self._queued_dispatch_terminal_receipt_keys: list[tuple[str, str]] = []
        self._queued_dispatch_terminal_receipt_keys_by_session: dict[
            str, tuple[tuple[str, str], ...]
        ] = {}
        self._session_operation_records: dict[str, dict[str, dict[str, Any]]] = {}
        self._mcp_manifest_baselines: dict[str, McpManifestBaseline] = {}
        self._pending_action_session_ids: set[str] = set()
        # Retain every accepted message for idempotent replay, but keep queued
        # delivery state in per-session/mode deques. Hot delivery and terminal
        # fencing therefore depend on the pending set, not historical deliveries.
        self._queued_session_messages_by_idempotency: dict[
            str, dict[str, SessionQueuedMessage]
        ] = {}
        self._pending_session_messages: dict[
            tuple[str, SessionMessageDeliveryMode], deque[SessionQueuedMessage]
        ] = {}
        self._session_message_delivery_records: dict[str, _InMemoryMessageDeliveryRecord] = {}
        self._session_message_terminal_receipts: dict[tuple[str, str], dict[str, Any]] = {}
        # Event-owned lookup index, not a copy of acceptance provenance. Retain
        # at most two IDs: a second canonical acceptance makes quarantine ambiguous.
        self._session_message_acceptance_event_ids: dict[tuple[str, str], set[str]] = {}
        self._next_session_message_ordering_key = 1

    @asynccontextmanager
    async def _read_event_state(self) -> AsyncIterator[memory_event_reads.MemoryEventReadState]:
        # Resolve current indexes only after acquiring the writer lock.
        async with self._lock:
            yield memory_event_reads.MemoryEventReadState(
                accounting_generation=self._accounting_generation,
                checkpoints=self._checkpoints,
                cost_accounting_authority=self._cost_accounting_authority,
                cost_event_index=self._cost_event_index,
                event_records=self._event_records,
                event_records_by_id=self._event_records_by_id,
                events=self._events,
                interaction_event_records=self._interaction_event_records,
                latest_interaction_event_records_by_sequence=self._latest_interaction_event_records_by_sequence,
                pending_action_event_history=self._pending_action_event_history,
                session_event_records=self._session_event_records,
                session_usage_cache=self._session_usage_cache,
                sessions=self._sessions,
                transcript_interaction_ids=self._transcript_interaction_ids,
                transcripts=self._transcripts,
                type_event_records=self._type_event_records,
                user_input_supersession_event_records=self._user_input_supersession_event_records,
                workflow_attempt_event_records=self._workflow_attempt_event_records,
                workflow_fenced_step_event_records=self._workflow_fenced_step_event_records,
                workflow_step_event_records=self._workflow_step_event_records,
            )

    @asynccontextmanager
    async def _read_session_query_state(
        self,
    ) -> AsyncIterator[memory_session_queries.MemorySessionQueryState]:
        # Resolve current indexes only after acquiring the writer lock.
        async with self._lock:
            yield memory_session_queries.MemorySessionQueryState(
                checkpoints=self._checkpoints,
                child_lifecycle_candidate_keys_by_parent=self._child_lifecycle_candidate_keys_by_parent,
                child_lifecycle_event_records=self._child_lifecycle_event_records,
                child_session_keys_by_parent=self._child_session_keys_by_parent,
                ownership_clock=self._ownership_clock,
                pending_action_event_records=self._pending_action_event_records,
                pending_action_latest_barrier_records=self._pending_action_latest_barrier_records,
                pending_action_session_ids=self._pending_action_session_ids,
                queued_dispatch_terminal_receipt_keys=self._queued_dispatch_terminal_receipt_keys,
                queued_dispatch_terminal_receipts=self._queued_dispatch_terminal_receipts,
                session_event_records=self._session_event_records,
                session_operation_records=self._session_operation_records,
                sessions=self._sessions,
            )

    def _transcript_indexes_unlocked(self) -> memory_transcripts.MemoryTranscriptIndexes:
        # Called only under the store lock; writers may replace whole indexes.
        return memory_transcripts.MemoryTranscriptIndexes(
            messages=self._transcripts,
            interaction_ids=self._transcript_interaction_ids,
            indices_by_interaction=self._transcript_indices_by_interaction,
            latest_indexes_by_role=self._latest_transcript_indexes_by_role,
            search_documents=self._transcript_search_documents,
            search_postings=self._transcript_search_postings,
        )

    @asynccontextmanager
    async def _read_transcript_state(
        self,
    ) -> AsyncIterator[memory_transcripts.MemoryTranscriptReadState]:
        async with self._lock:
            yield memory_transcripts.MemoryTranscriptReadState(
                indexes=self._transcript_indexes_unlocked(),
                sessions=self._sessions,
                deferred_interaction_inputs=self._deferred_interaction_inputs,
            )

    @property
    def public_authority_alias_codec(self) -> PublicAuthorityAliasCodec:
        return self._public_authority_alias_codec

    def _index_session_parent_unlocked(self, session: Session) -> None:
        parent_session_id = session.parent_session_id
        if parent_session_id is None:
            return
        keys = self._child_session_keys_by_parent.setdefault(parent_session_id, [])
        key = (session.created_at, session.id)
        index = bisect_left(keys, key)
        if index < len(keys) and keys[index] == key:
            raise RuntimeError("Duplicate in-memory session topology index entry.")
        keys.insert(index, key)

    def _remove_child_lifecycle_candidate_unlocked(self, child_session_id: str) -> None:
        indexed = self._child_lifecycle_candidate_key_by_child.pop(child_session_id, None)
        if indexed is None:
            return
        parent_session_id, priority, key = indexed
        priorities = self._child_lifecycle_candidate_keys_by_parent.get(parent_session_id)
        if priorities is None:
            raise RuntimeError("Missing in-memory child lifecycle candidate branch.")
        keys = priorities.get(priority)
        if keys is None:
            raise RuntimeError("Missing in-memory child lifecycle candidate priority.")
        index = bisect_left(keys, key)
        if index >= len(keys) or keys[index] != key:
            raise RuntimeError("Missing in-memory child lifecycle candidate key.")
        keys.pop(index)
        if not keys:
            del priorities[priority]
        if not priorities:
            del self._child_lifecycle_candidate_keys_by_parent[parent_session_id]

    def _refresh_child_lifecycle_candidate_unlocked(self, child: Session) -> None:
        self._remove_child_lifecycle_candidate_unlocked(child.id)
        parent_session_id = child.parent_session_id
        if parent_session_id is None:
            return
        occurrence_source = child_lifecycle_rules._child_session_lifecycle_occurrence(
            child,
            self._child_lifecycle_event_records.get(child.id, {}),
        )
        priority = 1
        sort_at = child.created_at
        if occurrence_source is not None:
            _relationship, occurrence = occurrence_source
            sort_at = occurrence.occurred_at
            if child_lifecycle_rules._child_session_lifecycle_state(child.status).terminal:
                consumption_key = child_session_notification_storage_key(
                    child.instance_id,
                    occurrence.source_id,
                )
                parent_operations = self._session_operation_records.get(parent_session_id, {})
                priority = 2 if consumption_key in parent_operations else 0
        key = (sort_at, child.id)
        priorities = self._child_lifecycle_candidate_keys_by_parent.setdefault(
            parent_session_id,
            {},
        )
        keys = priorities.setdefault(priority, [])
        index = bisect_left(keys, key)
        if index < len(keys) and keys[index] == key:
            raise RuntimeError("Duplicate in-memory child lifecycle candidate key.")
        keys.insert(index, key)
        self._child_lifecycle_candidate_key_by_child[child.id] = (
            parent_session_id,
            priority,
            key,
        )

    def _remove_session_parent_index_unlocked(self, session: Session) -> None:
        parent_session_id = session.parent_session_id
        if parent_session_id is None:
            return
        keys = self._child_session_keys_by_parent.get(parent_session_id)
        if keys is None:
            raise RuntimeError("Missing in-memory session topology index branch.")
        key = (session.created_at, session.id)
        index = bisect_left(keys, key)
        if index >= len(keys) or keys[index] != key:
            raise RuntimeError("Missing in-memory session topology index entry.")
        keys.pop(index)
        if not keys:
            del self._child_session_keys_by_parent[parent_session_id]
        self._remove_child_lifecycle_candidate_unlocked(session.id)

    def _prepare_checkpoint_store_unlocked(
        self,
        session_id: str,
        checkpoint: dict[str, Any],
        *,
        additional_event_records: tuple[EventRecord, ...] = (),
    ) -> _PreparedInMemoryCheckpointStore:
        from cayu.sessions.pending_actions import (
            _pending_action_checkpoint_index_state,
            checkpoint_has_pending_action_candidate,
            pending_action_event_has_unknown_round_call,
            pending_action_event_is_from_different_tool_round,
            pending_action_event_lookup_id,
            pending_action_event_retains_history,
            pending_action_lookup_key,
            project_pending_action_event_record,
            retain_pending_action_index_record,
        )

        if not checkpoint_has_pending_action_candidate(checkpoint):
            return _PreparedInMemoryCheckpointStore(
                checkpoint=checkpoint,
                has_pending_action=False,
            )

        lookup_ids, pending_round = _pending_action_checkpoint_index_state(checkpoint)
        lookup_keys = frozenset(pending_action_lookup_key(identifier) for identifier in lookup_ids)
        index_scope = (
            lookup_keys,
            None if pending_round is None else pending_round.model_step_id,
            None if pending_round is None else pending_round.model_attempt_id,
            None if pending_round is None else pending_round.tool_round_id,
        )
        rebuilt: dict[str, dict[str, EventRecord]] | None = None
        rebuilt_history: dict[str, list[EventRecord]] | None = None
        if self._pending_action_index_scopes.get(session_id) != index_scope:
            # A public checkpoint transform may legitimately reintroduce a
            # previously cleared durable action or reuse a provider call id for
            # a later execution. SQL stores retain the source events, so rebuild
            # both identity-scoped projections from complete durable evidence.
            rebuilt = {}
            rebuilt_history = {}
            round_lookup_key = (
                None
                if pending_round is None
                else pending_action_lookup_key(pending_round.tool_round_id)
            )
            records = (
                *self._session_event_records.get(session_id, ()),
                *additional_event_records,
            )
            for record in records:
                event_type = str(record.event.type)
                if (
                    event_type not in PENDING_ACTION_EVENT_TYPE_VALUES
                    or event_type in PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES
                ):
                    continue
                retain_for_scope = (
                    pending_round is not None
                    and pending_action_event_has_unknown_round_call(
                        record.event,
                        pending_round,
                    )
                )
                lookup_id = pending_action_event_lookup_id(record.event)
                if lookup_id is None and not retain_for_scope:
                    continue
                projected_record = project_pending_action_event_record(record)
                if lookup_id is not None:
                    lookup_key = pending_action_lookup_key(lookup_id)
                    if lookup_key in lookup_keys:
                        rebuilt_history.setdefault(lookup_key, []).append(record)
                        if not (
                            pending_action_event_retains_history(event_type)
                            and (
                                pending_round is None
                                or pending_action_event_is_from_different_tool_round(
                                    record.event,
                                    pending_round,
                                )
                            )
                        ):
                            retain_pending_action_index_record(
                                rebuilt.setdefault(lookup_key, {}),
                                projected_record,
                            )
                if retain_for_scope:
                    assert round_lookup_key is not None
                    retain_pending_action_index_record(
                        rebuilt.setdefault(round_lookup_key, {}),
                        projected_record,
                    )
                    rebuilt_history.setdefault(round_lookup_key, []).append(record)
        return _PreparedInMemoryCheckpointStore(
            checkpoint=checkpoint,
            has_pending_action=True,
            rebuilt_records=rebuilt,
            rebuilt_event_history=rebuilt_history,
            rebuilt_index_scope=index_scope if rebuilt is not None else None,
        )

    def _apply_checkpoint_store_unlocked(
        self,
        session_id: str,
        prepared: _PreparedInMemoryCheckpointStore,
    ) -> None:
        self._refresh_queued_dispatch_terminal_receipts_unlocked(
            session_id,
            prepared.checkpoint,
        )
        self._checkpoints[session_id] = prepared.checkpoint
        if prepared.has_pending_action:
            if prepared.rebuilt_records is not None:
                self._pending_action_event_records[session_id] = prepared.rebuilt_records
                assert prepared.rebuilt_event_history is not None
                self._pending_action_event_history[session_id] = prepared.rebuilt_event_history
                assert prepared.rebuilt_index_scope is not None
                self._pending_action_index_scopes[session_id] = prepared.rebuilt_index_scope
            self._pending_action_session_ids.add(session_id)
            return

        self._pending_action_session_ids.discard(session_id)
        # Once the checkpoint no longer names a pending action, identifier-
        # scoped event history cannot contribute to a future action. Keep the
        # latest lifecycle barrier, but release the potentially growing maps.
        removed = self._pending_action_event_records.pop(session_id, None)
        self._pending_action_event_history.pop(session_id, None)
        if removed or session_id in self._pending_action_index_scopes:
            self._pending_action_index_scopes[session_id] = (
                frozenset(),
                None,
                None,
                None,
            )

    def _refresh_queued_dispatch_terminal_receipts_unlocked(
        self,
        session_id: str,
        checkpoint: dict[str, Any] | None,
    ) -> None:
        previous_keys = self._queued_dispatch_terminal_receipt_keys_by_session.get(session_id, ())
        previous_key_set = set(previous_keys)
        receipts = terminal_event_evidence._queued_dispatch_terminal_receipts_for_session(
            session_id,
            checkpoint,
            durable_event_ids=self._event_ids.get(session_id, set()),
        )
        next_receipts: dict[tuple[str, str], QueuedDispatchTerminalReceipt] = {}
        for receipt in receipts:
            key = (receipt.session_id, receipt.operation_id)
            if key in next_receipts:
                raise RuntimeError("Duplicate in-memory queued dispatch receipt identity.")
            if key in self._queued_dispatch_terminal_receipts and key not in previous_key_set:
                raise RuntimeError("Duplicate in-memory queued dispatch receipt identity.")
            next_receipts[key] = receipt.model_copy(deep=True)

        previous_indices: list[tuple[int, tuple[str, str]]] = []
        for key in previous_keys:
            index = bisect_left(self._queued_dispatch_terminal_receipt_keys, key)
            if (
                index >= len(self._queued_dispatch_terminal_receipt_keys)
                or self._queued_dispatch_terminal_receipt_keys[index] != key
                or key not in self._queued_dispatch_terminal_receipts
            ):
                raise RuntimeError("In-memory queued dispatch receipt index is inconsistent.")
            previous_indices.append((index, key))

        self._queued_dispatch_terminal_receipt_keys_by_session.pop(session_id, None)
        for index, key in reversed(previous_indices):
            self._queued_dispatch_terminal_receipt_keys.pop(index)
            del self._queued_dispatch_terminal_receipts[key]

        next_keys: list[tuple[str, str]] = []
        for key in sorted(next_receipts):
            index = bisect_left(self._queued_dispatch_terminal_receipt_keys, key)
            self._queued_dispatch_terminal_receipt_keys.insert(index, key)
            self._queued_dispatch_terminal_receipts[key] = next_receipts[key]
            next_keys.append(key)
        if next_keys:
            self._queued_dispatch_terminal_receipt_keys_by_session[session_id] = tuple(next_keys)

    def _store_checkpoint_unlocked(self, session_id: str, checkpoint: dict[str, Any]) -> None:
        prepared = self._prepare_checkpoint_store_unlocked(session_id, checkpoint)
        self._apply_checkpoint_store_unlocked(session_id, prepared)

    async def _terminalize_zero_work_interruption(
        self,
        request: ZeroWorkInterruptionRequest,
    ) -> ZeroWorkInterruptionPublication | None:
        """Atomically prove and terminalize zero work; unsupported stores decline."""
        from cayu.storage._zero_work_interruption import memory_terminalize

        return await memory_terminalize(self, request)

    async def register_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        private_value: str,
        scope_session_id: str | None = None,
    ) -> None:
        key = _public_authority_alias_store_key(
            public_alias,
            field_name=field_name,
            private_value=private_value,
            scope_session_id=scope_session_id,
        )
        codec = self.public_authority_alias_codec
        if codec is None or not codec.matches(
            public_alias,
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        ):
            raise ValueError("Public authority alias lacks valid store-configured provenance.")
        async with self._lock:
            self._register_public_authority_alias_unlocked(key, private_value)

    def _register_public_authority_alias_unlocked(
        self,
        key: tuple[str, str, str],
        private_value: str,
    ) -> None:
        existing = self._public_authority_aliases.get(key)
        if existing is not None and existing != private_value:
            raise ValueError("Public authority alias conflicts with existing private authority.")
        self._public_authority_aliases[key] = private_value

    def _register_private_authority_alias_unlocked(
        self,
        private_value: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> None:
        codec = self.public_authority_alias_codec
        if codec is None:
            return
        for public_alias in codec.aliases(
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        ):
            key = _public_authority_alias_store_key(
                public_alias,
                field_name=field_name,
                private_value=private_value,
                scope_session_id=scope_session_id,
            )
            self._register_public_authority_alias_unlocked(key, private_value)

    async def resolve_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> str | None:
        key = _public_authority_alias_store_key(
            public_alias,
            field_name=field_name,
            private_value=None,
            scope_session_id=scope_session_id,
        )
        async with self._lock:
            return _authenticated_public_authority_alias_private_value(
                self.public_authority_alias_codec,
                public_alias,
                self._public_authority_aliases.get(key),
                field_name=field_name,
                scope_session_id=scope_session_id,
            )

    async def public_authority_private_value_exists(
        self,
        private_value: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> bool:
        codec = self.public_authority_alias_codec
        if codec is None:
            raise RuntimeError("Public authority alias codec is unavailable.")
        probe = codec.encode(
            private_value,
            field_name=field_name,
            session_id=scope_session_id,
        )
        field_name, scope_key, _probe = _public_authority_alias_store_key(
            probe,
            field_name=field_name,
            private_value=private_value,
            scope_session_id=scope_session_id,
        )
        async with self._lock:
            return any(
                key_field == field_name and key_scope == scope_key and stored == private_value
                for (key_field, key_scope, _alias), stored in self._public_authority_aliases.items()
            )

    async def issue_targeted_tool_grants(
        self,
        session_id: str,
        *,
        expected_run_epoch: int,
        records: tuple[TargetedToolGrantRecord, ...],
        events: tuple[Event, ...],
    ) -> TargetedToolGrantIssueResult:
        session_id = require_clean_nonblank(session_id, "session_id")
        if type(expected_run_epoch) is not int or expected_run_epoch < 0:
            raise ValueError("expected_run_epoch must be a non-negative integer.")
        if type(records) is not tuple or type(events) is not tuple:
            raise TypeError("records and events must be tuples.")
        if len(records) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
            raise ValueError("Targeted grant issuance exceeds the bounded request count.")
        copied_records = tuple(copy_targeted_tool_grant_record(record) for record in records)
        copied_events = tuple(copy_event(event) for event in events)
        if len(copied_records) != len(copied_events):
            raise ValueError("Each targeted grant record requires one issuance event.")
        if len({record.request_id for record in copied_records}) != len(copied_records):
            raise ValueError("Targeted grant records must have unique request identities.")
        if len({record.tool_id for record in copied_records}) != len(copied_records):
            raise ValueError("Targeted grant records must have unique tool identities.")
        interaction_ids = {record.interaction_id for record in copied_records}
        if len(interaction_ids) > 1:
            raise ValueError("Targeted grant records must share one interaction scope.")
        codec = self.public_authority_alias_codec
        if copied_records and codec is None:
            raise RuntimeError("Targeted grants require a public authority alias codec.")
        for record, event in zip(copied_records, copied_events, strict=True):
            if record.session_id != session_id:
                raise ValueError("Targeted grant scope is inconsistent.")
            assert codec is not None
            validate_targeted_tool_grant_reference(record, codec)
            validate_targeted_tool_grant_issuance_evidence(record, event)

        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.run_epoch != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            if session.status is not SessionStatus.RUNNING:
                raise SessionStatusConflict("Targeted grants require a running session.")
            if interaction_ids:
                latest_interactions = self._latest_interaction_event_records_by_sequence.get(
                    session_id,
                    (),
                )
                latest = latest_interactions[-1] if latest_interactions else None
                if (
                    latest is None
                    or latest.event.interaction_id != next(iter(interaction_ids))
                    or latest.event.type
                    in {
                        EventType.INTERACTION_COMPLETED,
                        EventType.INTERACTION_FAILED,
                        EventType.INTERACTION_INTERRUPTED,
                    }
                ):
                    raise ValueError("Targeted grants require the current open interaction.")
                interaction_started = next(
                    (
                        event_record.event
                        for event_record in latest_interactions
                        if event_record.event.type is EventType.INTERACTION_STARTED
                        and event_record.event.interaction_id == next(iter(interaction_ids))
                    ),
                    None,
                )
                if interaction_started is None:
                    raise RuntimeError("Targeted grant issuance lost interaction admission.")
                validate_targeted_tool_grant_batch_evidence(
                    copied_records,
                    interaction_started,
                )
            for record in copied_records:
                if (
                    record.agent_name != session.agent_name
                    or record.environment_name != session.environment_name
                    or record.principal != session.invocation.origin.subject
                    or record.tenant != session.invocation.origin.tenant
                ):
                    raise ValueError("Targeted grant scope conflicts with session authority.")

            resolved_records: list[TargetedToolGrantRecord] = []
            outcomes: list[TargetedToolGrantIssueOutcome] = []
            resolved_events: list[Event] = []
            new_records: list[tuple[TargetedToolGrantRecord, Event]] = []
            events_to_append: list[Event] = []
            for record, event in zip(copied_records, copied_events, strict=True):
                request_key = (session_id, record.interaction_id, record.request_id)
                tool_key = (session_id, record.interaction_id, record.tool_id)
                existing_id = self._targeted_tool_grant_ids_by_request.get(request_key)
                existing = (
                    None if existing_id is None else self._targeted_tool_grants.get(existing_id)
                )
                if existing is not None:
                    if existing.grant_id != record.grant_id:
                        raise ValueError(
                            "Targeted grant request identity conflicts with durable authority."
                        )
                    if len(self._targeted_tool_uses.get(existing.grant_id, ())) != (
                        existing.used_calls
                    ):
                        raise ValueError("Targeted grant call counter conflicts with durable uses.")
                    existing_event_record = self._event_records_by_id.get((session_id, event.id))
                    if existing_event_record is None:
                        raise RuntimeError("Targeted grant lost its durable issuance evidence.")
                    validate_targeted_tool_grant_issuance_evidence(
                        existing,
                        existing_event_record.event,
                    )
                    reused_event = targeted_tool_grant_event(
                        existing,
                        event_type=EventType.TARGETED_TOOL_GRANT_REUSED,
                        timestamp=event.timestamp,
                        outcome=TargetedToolGrantIssueOutcome.REUSED.value,
                        event_id_suffix="reused",
                    )
                    existing_reuse = self._event_records_by_id.get((session_id, reused_event.id))
                    codec = self.public_authority_alias_codec
                    if codec is None:  # pragma: no cover - capability invariant
                        raise RuntimeError("Targeted grants require a public alias codec.")
                    resolved_records.append(
                        targeted_tool_grant_with_active_reference(existing, codec)
                    )
                    outcomes.append(TargetedToolGrantIssueOutcome.REUSED)
                    if existing_reuse is None:
                        events_to_append.append(reused_event)
                        resolved_events.append(reused_event)
                    else:
                        resolved_events.append(copy_event(existing_reuse.event))
                    continue
                conflicting_tool_grant_id = self._targeted_tool_grant_ids_by_tool.get(tool_key)
                if conflicting_tool_grant_id is not None:
                    raise ValueError(
                        "Targeted grant tool identity conflicts with durable authority."
                    )
                if record.grant_id in self._targeted_tool_grants:
                    raise ValueError("Targeted grant identity collides with existing authority.")
                if (session_id, event.id) in self._event_records_by_id:
                    raise ValueError("Targeted grant event identity is already in use.")
                new_records.append((record, event))
                resolved_records.append(record)
                outcomes.append(TargetedToolGrantIssueOutcome.ISSUED)
                resolved_events.append(event)
                events_to_append.append(event)

            if new_records:
                interaction_id = new_records[0][0].interaction_id
                existing_count = len(
                    self._targeted_tool_grant_ids_by_interaction.get(
                        (session_id, interaction_id),
                        (),
                    )
                )
                if existing_count + len(new_records) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
                    raise ValueError("Targeted grant interaction exceeds its bounded count.")
            if events_to_append:
                prepared_append = self._prepare_event_append_unlocked(
                    session,
                    events_to_append,
                )
            alias_plans: dict[
                str,
                tuple[tuple[str, str, str], str, str],
            ] = {}
            for record, _event in new_records:
                assert codec is not None
                for public_alias in codec.aliases(
                    record.grant_id,
                    field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                    session_id=session_id,
                ):
                    alias_key = _public_authority_alias_store_key(
                        public_alias,
                        field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                        private_value=record.grant_id,
                        scope_session_id=session_id,
                    )
                    existing_private = self._public_authority_aliases.get(alias_key)
                    if existing_private is not None and existing_private != record.grant_id:
                        raise ValueError("Targeted tool reference collides with authority.")
                    existing_plan = alias_plans.get(public_alias)
                    if existing_plan is not None and existing_plan[1] != record.grant_id:
                        raise ValueError("Targeted tool reference batch contains a collision.")
                    existing_ref = self._targeted_tool_grant_refs.get(public_alias)
                    if existing_ref is not None and existing_ref != (
                        session_id,
                        record.grant_id,
                    ):
                        raise ValueError("Targeted tool reference collides with authority.")
                    alias_plans[public_alias] = (
                        alias_key,
                        record.grant_id,
                        session_id,
                    )
            if new_records:
                for record, _event in new_records:
                    for public_alias in codec.aliases(
                        record.grant_id,
                        field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
                        session_id=session_id,
                    ):
                        alias_key, private_value, scope_key = alias_plans[public_alias]
                        self._public_authority_aliases[alias_key] = private_value
                        self._targeted_tool_grant_refs[public_alias] = (
                            scope_key,
                            private_value,
                        )
                    self._targeted_tool_grants[record.grant_id] = record
                    self._targeted_tool_grant_ids_by_session.setdefault(session_id, []).append(
                        record.grant_id
                    )
                    self._targeted_tool_grant_ids_by_interaction.setdefault(
                        (session_id, record.interaction_id),
                        [],
                    ).append(record.grant_id)
                    self._targeted_tool_grant_ids_by_request[
                        (session_id, record.interaction_id, record.request_id)
                    ] = record.grant_id
                    self._targeted_tool_grant_ids_by_tool[
                        (session_id, record.interaction_id, record.tool_id)
                    ] = record.grant_id
            if events_to_append:
                self._sessions[session_id] = self._apply_event_append_unlocked(
                    session,
                    prepared_append,
                )
            return TargetedToolGrantIssueResult(
                records=tuple(resolved_records),
                outcomes=tuple(outcomes),
                events=tuple(resolved_events),
            )

    async def list_targeted_tool_grants(
        self,
        session_id: str,
        *,
        interaction_id: str | None = None,
        limit: int = TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    ) -> tuple[TargetedToolGrantRecord, ...]:
        session_id = require_clean_nonblank(session_id, "session_id")
        if interaction_id is not None:
            interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
        if type(limit) is not int or not 1 <= limit <= TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS:
            raise ValueError(
                f"limit must be between 1 and {TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS}."
            )
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            codec = self.public_authority_alias_codec
            if codec is None:  # pragma: no cover - in-memory capability invariant
                raise RuntimeError("Targeted grants require a public authority alias codec.")
            grant_ids = (
                self._targeted_tool_grant_ids_by_session.get(session_id, ())
                if interaction_id is None
                else self._targeted_tool_grant_ids_by_interaction.get(
                    (session_id, interaction_id),
                    (),
                )
            )
            selected = (self._targeted_tool_grants[grant_id] for grant_id in grant_ids)
            bounded = heapq.nsmallest(
                limit + 1,
                selected,
                key=lambda candidate: (candidate.issued_at, candidate.grant_id),
            )
            if len(bounded) > limit:
                raise ValueError("Targeted grant inspection exceeds its bounded result limit.")
            if any(
                len(self._targeted_tool_uses.get(record.grant_id, ())) != record.used_calls
                for record in bounded
            ):
                raise ValueError("Targeted grant call counter conflicts with durable uses.")
            return tuple(
                targeted_tool_grant_with_active_reference(record, codec) for record in bounded
            )

    async def load_targeted_tool_grant_state(
        self,
        session_id: str,
    ) -> TargetedToolGrantStateSnapshot:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            codec = self.public_authority_alias_codec
            if codec is None:  # pragma: no cover - in-memory capability invariant
                raise RuntimeError("Targeted grants require a public authority alias codec.")
            records = tuple(
                targeted_tool_grant_with_active_reference(record, codec)
                for record in sorted(
                    (
                        self._targeted_tool_grants[grant_id]
                        for grant_id in self._targeted_tool_grant_ids_by_session.get(
                            session_id,
                            (),
                        )
                    ),
                    key=lambda record: (record.issued_at, record.grant_id),
                )
            )
            grant_ids = {record.grant_id for record in records}
            uses = tuple(
                TargetedToolUseBinding.model_validate(binding.model_dump(mode="python"))
                for binding in sorted(
                    (
                        binding
                        for grant_id in grant_ids
                        for binding in self._targeted_tool_uses.get(grant_id, {}).values()
                    ),
                    key=lambda binding: (binding.bound_at, binding.use_id),
                )
            )
            return TargetedToolGrantStateSnapshot(records=records, uses=uses)

    def _resolve_targeted_tool_ref_unlocked(
        self,
        tool_ref: str,
    ) -> tuple[str, str] | None:
        try:
            parsed = parse_public_authority_alias(tool_ref)
        except (TypeError, ValueError):
            return None
        if parsed is None or parsed.field_name != TARGETED_TOOL_REFERENCE_FIELD_NAME:
            return None
        resolved = self._targeted_tool_grant_refs.get(tool_ref)
        if resolved is None:
            return None
        scope_session_id, grant_id = resolved
        codec = self.public_authority_alias_codec
        if codec is None or not codec.matches(
            tool_ref,
            grant_id,
            field_name=TARGETED_TOOL_REFERENCE_FIELD_NAME,
            session_id=scope_session_id,
        ):
            return None
        return resolved

    async def bind_targeted_tool_grant_use(
        self,
        request: TargetedToolUseRequest,
        *,
        observed_at: datetime,
    ) -> TargetedToolUseResult:
        if type(request) is not TargetedToolUseRequest:
            raise TypeError("request must be a TargetedToolUseRequest.")
        request = TargetedToolUseRequest.model_validate(request.model_dump(mode="python"))
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware.")
        observed_at = observed_at.astimezone(UTC)
        async with self._lock:
            session = self._sessions.get(request.session_id)
            if session is None:
                raise KeyError(f"Session not found: {request.session_id}")
            if session.run_epoch != request.expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {request.expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            if session.status is not SessionStatus.RUNNING:
                raise SessionStatusConflict("Targeted tool use requires a running session.")

            resolved = self._resolve_targeted_tool_ref_unlocked(request.tool_ref)
            if resolved is None:
                try:
                    parsed = parse_public_authority_alias(request.tool_ref)
                    well_formed = (
                        parsed is not None
                        and parsed.field_name == TARGETED_TOOL_REFERENCE_FIELD_NAME
                    )
                except (TypeError, ValueError):
                    well_formed = False
                reason = (
                    TargetedToolUseRejectionReason.UNKNOWN
                    if well_formed
                    else TargetedToolUseRejectionReason.MALFORMED
                )
                event = targeted_tool_unresolved_rejection_event(
                    request,
                    reason=reason,
                    timestamp=observed_at,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                )
                existing = self._event_records_by_id.get((request.session_id, event.id))
                if existing is None:
                    prepared = self._prepare_event_append_unlocked(session, [event])
                    session = self._apply_event_append_unlocked(session, prepared)
                    self._sessions[request.session_id] = session
                else:
                    event = copy_event(existing.event)
                validate_targeted_tool_unresolved_rejection_evidence(
                    request,
                    reason=reason,
                    event=event,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                )
                return TargetedToolUseResult(
                    disposition=TargetedToolUseDisposition.REJECTED,
                    reason=reason,
                    event=event,
                )
            scope_session_id, grant_id = resolved
            if scope_session_id != request.session_id:
                event = targeted_tool_unresolved_rejection_event(
                    request,
                    reason=TargetedToolUseRejectionReason.CROSS_SESSION,
                    timestamp=observed_at,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                )
                existing = self._event_records_by_id.get((request.session_id, event.id))
                if existing is None:
                    prepared = self._prepare_event_append_unlocked(session, [event])
                    session = self._apply_event_append_unlocked(session, prepared)
                    self._sessions[request.session_id] = session
                else:
                    event = copy_event(existing.event)
                validate_targeted_tool_unresolved_rejection_evidence(
                    request,
                    reason=TargetedToolUseRejectionReason.CROSS_SESSION,
                    event=event,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                )
                return TargetedToolUseResult(
                    disposition=TargetedToolUseDisposition.REJECTED,
                    reason=TargetedToolUseRejectionReason.CROSS_SESSION,
                    event=event,
                )
            record = self._targeted_tool_grants.get(grant_id)
            if record is None:
                raise RuntimeError("Targeted tool reference lost its durable grant record.")
            if len(self._targeted_tool_uses.get(grant_id, ())) != record.used_calls:
                raise ValueError("Targeted grant call counter conflicts with durable uses.")
            current_session: Session = session

            def append_once(event: Event) -> Event:
                nonlocal current_session
                existing = self._event_records_by_id.get((request.session_id, event.id))
                if existing is not None:
                    return copy_event(existing.event)
                prepared = self._prepare_event_append_unlocked(current_session, [event])
                current_session = self._apply_event_append_unlocked(current_session, prepared)
                self._sessions[request.session_id] = current_session
                return event

            def rejected(reason: TargetedToolUseRejectionReason) -> TargetedToolUseResult:
                if reason is TargetedToolUseRejectionReason.EXPIRED:
                    expiry_event = append_once(
                        targeted_tool_grant_event(
                            record,
                            event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                            timestamp=observed_at,
                            outcome="expired",
                            event_id_suffix="expired",
                            rejection_reason=reason,
                        )
                    )
                    validate_targeted_tool_grant_lifecycle_event(
                        record,
                        expiry_event,
                        event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                        outcome="expired",
                        event_id_suffix="expired",
                        rejection_reason=reason,
                        require_current_call_count=False,
                    )
                event = append_once(
                    targeted_tool_use_rejection_event(
                        record,
                        request,
                        reason=reason,
                        timestamp=observed_at,
                    )
                )
                validate_targeted_tool_use_rejection_evidence(
                    record,
                    request,
                    reason=reason,
                    event=event,
                )
                return TargetedToolUseResult(
                    disposition=TargetedToolUseDisposition.REJECTED,
                    reason=reason,
                    grant=record,
                    event=event,
                )

            from cayu.sessions.interactions import INTERACTION_TERMINAL_EVENT_TYPES

            if any(
                event_record.event.type in INTERACTION_TERMINAL_EVENT_TYPES
                for event_record in self._interaction_event_records.get(
                    request.session_id,
                    {},
                ).get(record.interaction_id, ())
            ):
                return rejected(TargetedToolUseRejectionReason.EXPIRED)
            uses = self._targeted_tool_uses.setdefault(grant_id, {})
            related_by_id = {
                binding.use_id: binding
                for binding in (
                    self._targeted_tool_use_by_invocation.get(
                        (request.session_id, request.interaction_id, request.invocation_id)
                    ),
                    self._targeted_tool_use_by_outer_call.get(
                        (request.session_id, request.interaction_id, request.outer_tool_call_id)
                    ),
                )
                if binding is not None
            }
            related = tuple(related_by_id.values())
            if related:
                scope_rejection = targeted_tool_use_scope_rejection_reason(record, request)
                if scope_rejection is not None:
                    return rejected(scope_rejection)
                candidate = targeted_tool_use_binding(
                    grant_id,
                    request,
                    bound_at=related[0].bound_at,
                )
                if len(related) == 1 and related[0].model_dump(mode="json") == candidate.model_dump(
                    mode="json"
                ):
                    event_record = self._event_records_by_id.get(
                        (request.session_id, f"{grant_id}:use:{candidate.use_id}")
                    )
                    if event_record is None:
                        raise RuntimeError("Targeted tool use lost its durable event evidence.")
                    validate_targeted_tool_grant_lifecycle_event(
                        record,
                        event_record.event,
                        event_type=EventType.TARGETED_TOOL_REFERENCE_CONSUMED,
                        outcome=TargetedToolUseDisposition.BOUND.value,
                        event_id_suffix=f"use:{candidate.use_id}",
                        binding=related[0],
                        require_current_call_count=False,
                    )
                    rejoined_event = append_once(
                        targeted_tool_grant_event(
                            record,
                            event_type=EventType.TARGETED_TOOL_REFERENCE_REJOINED,
                            timestamp=observed_at,
                            outcome=TargetedToolUseDisposition.REJOINED.value,
                            event_id_suffix=f"rejoined:{candidate.use_id}",
                            binding=related[0],
                        )
                    )
                    return TargetedToolUseResult(
                        disposition=TargetedToolUseDisposition.REJOINED,
                        grant=record,
                        binding=related[0],
                        event=rejoined_event,
                    )
                return rejected(TargetedToolUseRejectionReason.ALTERED_REPLAY)

            reason = targeted_tool_use_rejection_reason(
                record,
                request,
                observed_at=observed_at,
            )
            if reason is not None:
                return rejected(reason)
            binding = targeted_tool_use_binding(grant_id, request, bound_at=observed_at)
            updated = TargetedToolGrantRecord.model_validate(
                record.model_copy(update={"used_calls": record.used_calls + 1}).model_dump(
                    mode="python"
                )
            )
            event = targeted_tool_grant_event(
                updated,
                event_type=EventType.TARGETED_TOOL_REFERENCE_CONSUMED,
                timestamp=observed_at,
                outcome=TargetedToolUseDisposition.BOUND.value,
                event_id_suffix=f"use:{binding.use_id}",
                binding=binding,
            )
            prepared_append = self._prepare_event_append_unlocked(session, [event])
            uses[binding.use_id] = binding
            self._targeted_tool_use_by_invocation[
                (binding.session_id, binding.interaction_id, binding.invocation_id)
            ] = binding
            self._targeted_tool_use_by_outer_call[
                (binding.session_id, binding.interaction_id, binding.outer_tool_call_id)
            ] = binding
            self._targeted_tool_grants[grant_id] = updated
            self._sessions[request.session_id] = self._apply_event_append_unlocked(
                session,
                prepared_append,
            )
            return TargetedToolUseResult(
                disposition=TargetedToolUseDisposition.BOUND,
                grant=updated,
                binding=binding,
                event=event,
            )

    async def revoke_targeted_tool_grant(
        self,
        tool_ref: str,
        *,
        session_id: str,
        expected_run_epoch: int,
        reason: str,
        revoked_at: datetime,
    ) -> TargetedToolGrantRecord | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if type(expected_run_epoch) is not int or expected_run_epoch < 0:
            raise ValueError("expected_run_epoch must be a non-negative integer.")
        reason = validate_targeted_tool_grant_revocation_reason(reason)
        if revoked_at.tzinfo is None or revoked_at.utcoffset() is None:
            raise ValueError("revoked_at must be timezone-aware.")
        revoked_at = revoked_at.astimezone(UTC)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.run_epoch != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            resolved = self._resolve_targeted_tool_ref_unlocked(tool_ref)
            if resolved is None or resolved[0] != session_id:
                return None
            record = self._targeted_tool_grants.get(resolved[1])
            if record is None:
                raise RuntimeError("Targeted tool reference lost its durable grant record.")
            if len(self._targeted_tool_uses.get(record.grant_id, ())) != record.used_calls:
                raise ValueError("Targeted grant call counter conflicts with durable uses.")
            if record.revoked_at is not None:
                if record.revocation_reason != reason:
                    raise ValueError("Targeted grant was revoked with a different reason.")
                event_record = self._event_records_by_id.get(
                    (session_id, f"{record.grant_id}:revoked")
                )
                if event_record is None:
                    raise RuntimeError("Targeted grant revocation lost its durable event evidence.")
                validate_targeted_tool_grant_revocation_evidence(
                    record,
                    event_record.event,
                )
                return copy_targeted_tool_grant_record(record)
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            uses = self._targeted_tool_uses.get(record.grant_id, {})
            if any(binding.bound_at > revoked_at for binding in uses.values()):
                raise ValueError("revoked_at cannot precede a bound targeted tool use.")
            updated = TargetedToolGrantRecord.model_validate(
                record.model_copy(
                    update={"revoked_at": revoked_at, "revocation_reason": reason}
                ).model_dump(mode="python")
            )
            event = targeted_tool_grant_event(
                updated,
                event_type=EventType.TARGETED_TOOL_GRANT_REVOKED,
                timestamp=revoked_at,
                outcome="revoked",
                event_id_suffix="revoked",
            )
            prepared_append = self._prepare_event_append_unlocked(session, [event])
            self._targeted_tool_grants[record.grant_id] = updated
            self._sessions[session_id] = self._apply_event_append_unlocked(
                session,
                prepared_append,
            )
            return copy_targeted_tool_grant_record(updated)

    async def reconstruct_targeted_tool_grants(
        self,
        session_id: str,
        *,
        expected_run_epoch: int,
        interaction_id: str,
        generation_id: str,
        agent_name: str,
        task_id: str | None,
        environment_name: str | None,
        principal: str | None,
        tenant: str | None,
        catalogue_revision: str,
        descriptors_by_id: Mapping[str, tuple[str, str, str]],
        capability_ceiling_names: frozenset[str],
        observed_at: datetime,
    ) -> TargetedToolGrantReconstructionResult:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = require_clean_nonblank(interaction_id, "interaction_id")
        if type(expected_run_epoch) is not int or expected_run_epoch < 0:
            raise ValueError("expected_run_epoch must be a non-negative integer.")
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware.")
        observed_at = observed_at.astimezone(UTC)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.run_epoch != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            if session.status is not SessionStatus.RUNNING:
                raise SessionStatusConflict("Grant reconstruction requires a running session.")
            records = sorted(
                (
                    self._targeted_tool_grants[grant_id]
                    for grant_id in self._targeted_tool_grant_ids_by_interaction.get(
                        (session_id, interaction_id),
                        (),
                    )
                ),
                key=lambda candidate: (candidate.issued_at, candidate.grant_id),
            )
            if len(records) > TARGETED_TOOL_GRANT_MAX_REQUESTS:
                raise ValueError("Targeted grant interaction exceeds its bounded count.")
            if any(
                len(self._targeted_tool_uses.get(record.grant_id, ())) != record.used_calls
                for record in records
            ):
                raise ValueError("Targeted grant call counter conflicts with durable uses.")

            current_session: Session = session

            def append_once(event: Event) -> Event:
                nonlocal current_session
                existing = self._event_records_by_id.get((session_id, event.id))
                if existing is not None:
                    return copy_event(existing.event)
                prepared = self._prepare_event_append_unlocked(current_session, [event])
                current_session = self._apply_event_append_unlocked(current_session, prepared)
                self._sessions[session_id] = current_session
                return event

            from cayu.sessions.interactions import INTERACTION_TERMINAL_EVENT_TYPES

            interaction_records = self._interaction_event_records.get(session_id, {}).get(
                interaction_id,
                (),
            )
            interaction_started = next(
                (
                    event_record.event
                    for event_record in interaction_records
                    if event_record.event.type is EventType.INTERACTION_STARTED
                ),
                None,
            )
            if interaction_started is None:
                raise RuntimeError("Targeted grant reconstruction lost interaction admission.")
            validate_targeted_tool_grant_batch_evidence(tuple(records), interaction_started)
            interaction_ended = any(
                event_record.event.type in INTERACTION_TERMINAL_EVENT_TYPES
                for event_record in interaction_records
            )
            valid: list[TargetedToolGrantRecord] = []
            rejected: list[tuple[str, TargetedToolUseRejectionReason]] = []
            events: list[Event] = []
            for record in records:
                reason = targeted_tool_grant_reconstruction_rejection_reason(
                    record,
                    generation_id=generation_id,
                    agent_name=agent_name,
                    task_id=task_id,
                    environment_name=environment_name,
                    principal=principal,
                    tenant=tenant,
                    catalogue_revision=catalogue_revision,
                    descriptors_by_id=descriptors_by_id,
                    capability_ceiling_names=capability_ceiling_names,
                    observed_at=observed_at,
                    interaction_ended=interaction_ended,
                )
                if reason is None:
                    valid.append(record)
                    event = targeted_tool_grant_event(
                        record,
                        event_type=EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED,
                        timestamp=observed_at,
                        outcome="reconstructed",
                        event_id_suffix="reconstructed",
                    )
                else:
                    rejected.append((record.grant_id, reason))
                    if reason is TargetedToolUseRejectionReason.EXPIRED:
                        persisted_expiry = append_once(
                            targeted_tool_grant_event(
                                record,
                                event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                                timestamp=observed_at,
                                outcome="expired",
                                event_id_suffix="expired",
                                rejection_reason=reason,
                            )
                        )
                        validate_targeted_tool_grant_lifecycle_event(
                            record,
                            persisted_expiry,
                            event_type=EventType.TARGETED_TOOL_GRANT_EXPIRED,
                            outcome="expired",
                            event_id_suffix="expired",
                            rejection_reason=reason,
                            require_current_call_count=False,
                        )
                    event = targeted_tool_grant_event(
                        record,
                        event_type=EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED,
                        timestamp=observed_at,
                        outcome="rejected",
                        event_id_suffix=f"reconstruction-rejected:{reason.value}",
                        rejection_reason=reason,
                    )
                persisted = append_once(event)
                validate_targeted_tool_grant_lifecycle_event(
                    record,
                    persisted,
                    event_type=EventType.TARGETED_TOOL_GRANT_RECONSTRUCTED,
                    outcome="reconstructed" if reason is None else "rejected",
                    event_id_suffix=(
                        "reconstructed"
                        if reason is None
                        else f"reconstruction-rejected:{reason.value}"
                    ),
                    rejection_reason=reason,
                    require_current_call_count=False,
                )
                events.append(persisted)
            return TargetedToolGrantReconstructionResult(
                valid=tuple(valid),
                rejected=tuple(rejected),
                events=tuple(events),
            )

    async def create(
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
        async with self._lock:
            session_id = request.session_id or str(uuid4())
            self._require_external_creation_unlocked(request)
            if (
                participant_provenance is _RECIPIENT_PROVENANCE_CAPABILITY
                and creation_target is None
            ):
                raise PermissionError("Recipient creation requires its exact durable target.")
            if creation_target is not None:
                creation_target = creation_fence.snapshot_target(creation_target)
                creation_fence.require_pending(
                    creation_target,
                    self._session_creation_decisions.get(creation_target.key),
                    request_commitment=participant_request_commitment,
                    requested_session_id=request.session_id,
                )
            self._require_available_closure_identity_unlocked(session_id)
            admission = interaction_rules._copy_optional_interaction_admission(
                session_id,
                interaction_started_event,
                interaction_source_messages,
                defer_transcript=True,
            )
            if session_id in self._sessions:
                raise ValueError(f"Session already exists: {session_id}")
            if request.parent_session_id == session_id:
                raise ValueError("Session cannot be its own parent.")
            if (
                request.parent_session_id is not None
                and request.parent_session_id not in self._sessions
            ):
                raise ValueError(f"Parent session not found: {request.parent_session_id}")
            parent_session = (
                None
                if request.parent_session_id is None
                else self._sessions[request.parent_session_id]
            )
            if parent_session is not None:
                for owner in self._session_closure_progress.values():
                    closure_records._check_closure_lineage_owner(owner, (parent_session.id,))
            now = self._ownership_clock()
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
            prepared_admission_events: _PreparedInMemoryEventAppend | None = None
            prepared_checkpoint: _PreparedInMemoryCheckpointStore | None = None
            pending_checkpoint: _PreparedInMemoryCheckpointStore | None = None
            if admission is None and checkpoint_transform is not None:
                transformed = checkpoint_transform(session.model_copy(deep=True), None)
                if transformed is not None:
                    copied_checkpoint = checkpoint_preservation._replace_checkpoint_preserving_completion_result_event_publications(
                        None,
                        copy_durable_json_object(transformed, "checkpoint"),
                        session_id=session.id,
                    )
                    pending_checkpoint = self._prepare_checkpoint_store_unlocked(
                        session.id,
                        copied_checkpoint,
                    )
            if admission is not None:
                started_event, source_messages = admission
                interaction_id = started_event.interaction_id
                if interaction_id is None:
                    raise AssertionError("Interaction admission lost its identity.")
                prepared_admission_events = self._prepare_event_append_unlocked(
                    session,
                    [started_event],
                )
                session = session.model_copy(update={"last_activity_at": now})
                prepared_checkpoint = self._prepare_checkpoint_store_unlocked(
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
                prepared_checkpoint = self._prepare_checkpoint_store_unlocked(
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
                if participant_binding.creation_key in self._participant_session_ids_by_key:
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
                    stored_selection = self._context_selection_receipts.get(
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
            self._register_private_authority_alias_unlocked(
                session.id,
                field_name="session_id",
            )
            self._sessions[session.id] = session
            if creation_decision is not None:
                self._session_creation_decisions[creation_decision.target.key] = creation_decision
            self._index_session_parent_unlocked(session)
            self._events[session.id] = []
            self._event_ids[session.id] = set()
            self._session_event_records[session.id] = []
            self._child_lifecycle_event_records[session.id] = {}
            self._interaction_event_records[session.id] = {}
            self._latest_interaction_event_records[session.id] = {}
            self._latest_interaction_event_records_by_sequence[session.id] = []
            self._pending_action_event_records[session.id] = {}
            self._session_operation_records[session.id] = initial_operation_records
            if participant_binding is not None:
                self._participant_session_bindings[session.id] = participant_binding.model_copy(
                    deep=True
                )
                assert participant_receipt is not None
                self._participant_session_receipts[session.id] = participant_receipt.model_copy(
                    deep=True
                )
                self._participant_session_ids_by_key[participant_binding.creation_key] = session.id
                self._participant_session_request_commitments[participant_binding.creation_key] = (
                    participant_request_commitment
                )
            self._transcripts[session.id] = recipient_messages
            self._transcript_interaction_ids[session.id] = [None] * len(recipient_messages)
            self._latest_transcript_indexes_by_role[session.id] = {}
            self._transcript_search_documents[session.id] = []
            if recipient_messages:
                memory_transcripts.replace_latest_role_indexes(
                    self._transcript_indexes_unlocked(), session.id
                )
                memory_transcripts.replace_search_session(
                    self._transcript_indexes_unlocked(), session.id, recipient_messages
                )
            self._refresh_child_lifecycle_candidate_unlocked(session)
            if prepared_admission_events is not None:
                assert admission is not None
                started_event, source_messages = admission
                interaction_id = started_event.interaction_id
                assert interaction_id is not None
                self._sessions[session.id] = self._apply_event_append_unlocked(
                    session,
                    prepared_admission_events,
                    activity_at=now,
                )
                self._deferred_interaction_inputs[session.id] = (
                    deferred_interaction_input_for_run_request(
                        request,
                        session_id=session.id,
                        interaction_id=interaction_id,
                        source_messages=source_messages,
                    )
                )
            if prepared_checkpoint is not None:
                self._apply_checkpoint_store_unlocked(session.id, prepared_checkpoint)
            session = self._sessions[session.id]
            if admission is not None:
                invocation_scope._activate_session_run_fence(session)
            return session.model_copy(deep=True)

    async def create_participant_owned_session(
        self,
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
        async with self._participant_creation_lock:
            existing_id = self._participant_session_ids_by_key.get(creation_request.creation_key)
            if existing_id is not None:
                session = self._sessions.get(existing_id)
                receipt = self._participant_session_receipts.get(existing_id)
                if session is None or receipt is None:
                    raise RuntimeError("Participant creation index is inconsistent.")
                if (
                    self._participant_session_request_commitments.get(creation_request.creation_key)
                    != creation_request.request_commitment
                ):
                    raise ValueError("Participant creation key conflicts with the request.")
                if (session.provider_name, session.model) != (
                    identity.provider_name,
                    identity.model,
                ):
                    raise ValueError(
                        "Participant creation key conflicts with the execution target."
                    )
                expected_binding, expected_receipt = binding_factory(session.model_copy(deep=True))
                if expected_binding != self._participant_session_bindings.get(existing_id):
                    raise ValueError("Participant creation key conflicts with its binding.")
                if expected_receipt != receipt:
                    raise ValueError("Participant creation key conflicts with its receipt.")
                if creation_target is not None:
                    from cayu.storage._creation_fence import validate_replay

                    await validate_replay(self, creation_target, session)
                return session.model_copy(deep=True), receipt.model_copy(deep=True)
            session = await self.create(
                resolved_request,
                identity=identity,
                participant_binding_factory=binding_factory,
                participant_request_commitment=creation_request.request_commitment,
                participant_provenance=recipient_provenance,
                recipient_selection=recipient_selection,
                creation_target=creation_target,
                recipient_receipt_validator=recipient_receipt_validator,
            )
            receipt = self._participant_session_receipts.get(session.id)
            if receipt is None:
                raise RuntimeError("Participant creation did not persist its receipt.")
            return session, receipt.model_copy(deep=True)

    async def lookup_participant_session_creation(
        self, creation_request: ParticipantSessionCreationRequest
    ) -> tuple[Session, ParticipantSessionCreationReceipt] | None:
        from cayu.sessions.context_views import ParticipantSessionCreationRequest

        if type(creation_request) is not ParticipantSessionCreationRequest:
            raise TypeError("Participant creation requires a typed creation request.")
        async with self._participant_creation_lock:
            session_id = self._participant_session_ids_by_key.get(creation_request.creation_key)
            if session_id is None:
                return None
            session = self._sessions.get(session_id)
            receipt = self._participant_session_receipts.get(session_id)
            if session is None or receipt is None:
                raise RuntimeError("Participant creation index is inconsistent.")
            if self._participant_session_request_commitments.get(creation_request.creation_key) != (
                creation_request.request_commitment
            ):
                raise ValueError("Participant creation key conflicts with the request.")
            return session.model_copy(deep=True), receipt.model_copy(deep=True)

    async def load_participant_session_binding(
        self, session_id: str
    ) -> ParticipantSessionBinding | None:
        async with self._lock:
            binding = self._participant_session_bindings.get(session_id)
            return None if binding is None else binding.model_copy(deep=True)

    async def _scan_participant_session_bindings(self, participant, *, after=None, limit=32):
        from cayu.sessions._participant_discovery import prepare_scan, reference

        query = prepare_scan(participant, after, limit)
        async with self._lock:
            bindings = sorted(
                (
                    item
                    for item in self._participant_session_bindings.values()
                    if item.participant == query.participant
                    and (query.after is None or item.creation_key > query.after)
                ),
                key=lambda item: item.creation_key,
            )[: query.limit]
            result = []
            for binding in bindings:
                session = self._sessions.get(binding.session_id)
                receipt = self._participant_session_receipts.get(binding.session_id)
                if (
                    session is None
                    or session.instance_id != binding.session_instance_id
                    or receipt is None
                    or receipt.binding != binding
                    or self._participant_session_ids_by_key.get(binding.creation_key) != session.id
                ):
                    raise ValueError("Participant inventory lacks coherent native creation.")
                result.append(reference(receipt, query))
            return tuple(result)

    async def load_participant_session_creation_receipt(
        self, session_id: str
    ) -> ParticipantSessionCreationReceipt | None:
        async with self._lock:
            receipt = self._participant_session_receipts.get(session_id)
            return None if receipt is None else receipt.model_copy(deep=True)

    async def capture_context_view_publication_source(
        self, session_id: str
    ) -> ContextViewPublicationSource:
        return (await self._capture_completed_turn_snapshot(session_id)).publication

    async def _capture_completed_turn_snapshot(self, session_id: str) -> CompletedTurnSnapshot:
        from cayu.sessions._context_view_source import (
            CompletedTurnSnapshot,
            capture_source,
            closed_round_publication_id,
            completed_boundary,
            publication_frontier,
        )

        async with self._lock:
            session = self._sessions.get(session_id)
            binding = self._participant_session_bindings.get(session_id)
            checkpoint = self._checkpoints.get(session_id)
            pointer = completed_boundary(session, binding, checkpoint)
            assert session is not None and binding is not None
            tool_receipt = None
            if pointer.tool_round_id is not None:
                publication_id = f"tool-round:{pointer.tool_round_id}"
                key = publication_contracts._runtime_publication_storage_key(publication_id)
                record = self._session_operation_records.get(session_id, {}).get(key)
                if record is None:
                    from itertools import islice

                    closures = tuple(
                        islice(
                            (
                                item.event
                                for item in reversed(
                                    self._session_event_records.get(session_id, ())
                                )
                                if item.event.type is EventType.SESSION_CHECKPOINTED
                                and item.event.payload.get("tool_round_id") == pointer.tool_round_id
                                and (
                                    item.event.payload.get("cleared") is True
                                    or item.event.payload.get("transition") == "answered"
                                )
                            ),
                            2,
                        )
                    )
                    publication_id = closed_round_publication_id(pointer, closures)
                    key = publication_contracts._runtime_publication_storage_key(publication_id)
                    record = self._session_operation_records.get(session_id, {}).get(key)
                if record is not None:
                    tool_receipt = publication_rules._reconstruct_runtime_publication_receipt(
                        record,
                        storage_key=key,
                        session_id=session_id,
                        publication_id=publication_id,
                    )
                publication_frontier(pointer, tool_receipt)
                assert tool_receipt is not None
                self._validate_runtime_publication_material_unlocked(tool_receipt)
            end = publication_frontier(pointer, tool_receipt)
            event = self._event_records_by_id.get((session_id, pointer.completion_event_id))
            messages = self._transcripts.get(session_id, [])
            interactions = self._transcript_interaction_ids.get(session_id, [])
            records = tuple(
                TranscriptRecord(
                    index=index,
                    interaction_id=interactions[index],
                    message=detach_message(messages[index]),
                )
                for index in range(
                    pointer.source_transcript_cursor,
                    min(end, len(messages)),
                )
            )
            publication = capture_source(
                session,
                binding,
                checkpoint,
                pointer,
                None if event is None else event.event,
                records,
                tool_receipt,
            )
            return CompletedTurnSnapshot(
                publication=publication,
                current_session=copy_session(session),
                checkpoint=deepcopy(checkpoint),
                has_queued_input=any(
                    (session_id, mode) in self._pending_session_messages
                    for mode in SessionMessageDeliveryMode
                ),
                has_closure_owner=any(
                    session_id in closure_records._closure_progress_targets(owner)
                    for owner in self._session_closure_progress.values()
                ),
                has_active_model_stage=MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY
                in self._session_operation_records.get(session_id, {}),
                current_transcript_cursor=len(messages),
            )

    async def publish_context_view(
        self, manifest: _ContextViewManifest, *, publication_key: str
    ) -> _ContextViewManifest:
        from cayu.sessions.context_views import (
            CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER,
        )
        from cayu.sessions.context_views import (
            ContextViewManifest as ContextViewManifestType,
        )

        if type(manifest) is not ContextViewManifestType or type(publication_key) is not str:
            raise TypeError("Context-view publication requires typed manifest and key.")
        if len(publication_key.encode("utf-8")) > 512:
            raise ValueError("publication_key must be at most 512 UTF-8 bytes.")
        publication_key = require_clean_nonblank(publication_key, "publication_key")
        async with self._lock:
            existing_id = self._context_publication_keys.get(publication_key)
            if existing_id is not None:
                existing = self._context_views[existing_id]
                if existing != manifest:
                    raise ValueError("Context-view publication key conflicts with its manifest.")
                return existing.model_copy(deep=True)
            if manifest.view_id in self._context_views:
                raise ValueError("Context-view ID is already bound to another manifest.")
            source = self._sessions.get(manifest.source_session_id)
            if source is None or source.instance_id != manifest.source_session_instance_id:
                raise LookupError("The source session incarnation is unavailable.")
            publication_count = sum(
                view.source_owner == manifest.source_owner for view in self._context_views.values()
            )
            if publication_count >= CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER:
                raise OverflowError("Context-view publication quota exceeded for the owner.")
            self._context_views[manifest.view_id] = manifest.model_copy(deep=True)
            self._context_publication_keys[publication_key] = manifest.view_id
            return manifest.model_copy(deep=True)

    async def lookup_context_view_publication(
        self, publication_key: str
    ) -> _ContextViewManifest | None:
        from cayu.sessions.context_views import ContextViewManifest as ContextViewManifestType

        if type(publication_key) is not str or not publication_key.strip():
            raise ValueError("Context-view publication keys must be non-blank strings.")
        if len(publication_key.encode("utf-8")) > 512:
            raise ValueError("publication_key must be at most 512 UTF-8 bytes.")
        async with self._lock:
            view_id = self._context_publication_keys.get(publication_key)
            if view_id is None:
                return None
            manifest = self._context_views.get(view_id)
            if manifest is None or type(manifest) is not ContextViewManifestType:
                raise RuntimeError("Context-view publication index is inconsistent.")
            return manifest.model_copy(deep=True)

    def _require_context_view_lifecycle_capacity(
        self, view_id: str, *, additional_slots: int = 0
    ) -> None:
        from cayu.sessions.context_views import validate_context_view_lifecycle_capacity

        events = sum(event.view_id == view_id for event in self._context_lifecycle_events.values())
        unsettled = sum(
            receipt.view.view_id == view_id
            and receipt.state in {"selected", "adopted", "transferred"}
            for receipt in self._context_selection_receipts.values()
        )
        validate_context_view_lifecycle_capacity(
            events, unsettled, additional_slots=additional_slots
        )

    async def select_context_view(
        self, request: _ContextViewSelectionRequest
    ) -> _ContextViewSelectionReceipt:
        return await self._select_context_view(request)

    async def _select_context_view(self, request, *, target=None):
        from cayu.sessions.context_views import (
            CONTEXT_VIEW_EXPIRY_BATCH_SIZE,
            CONTEXT_VIEW_MAX_SELECTIONS_PER_OWNER,
            context_view_manifest_bytes,
        )
        from cayu.sessions.context_views import (
            ContextViewSelectionReceipt as ContextViewSelectionReceiptType,
        )
        from cayu.sessions.context_views import (
            ContextViewSelectionRequest as ContextViewSelectionRequestType,
        )

        if type(request) is not ContextViewSelectionRequestType:
            raise TypeError("Context-view selection requires a typed request.")
        request = ContextViewSelectionRequestType.model_validate(request)
        from cayu.sessions.context_views import json_commitment

        request_commitment = json_commitment(
            canonical_bounded_durable_json_bytes(
                request.model_dump(mode="json"),
                "context view selection request",
                max_bytes=256 * 1024,
                max_nodes=8192,
                max_nesting=64,
            ).decode("utf-8"),
            "context view selection request",
        )
        async with _InMemoryContextViewSelectionTransaction(self):
            from cayu.sessions._context_selection_fence import require_not_excluded

            require_not_excluded(
                request,
                self._context_selection_exclusions.get(request.selection_key),
                target=target,
            )
            now_ms = int(self._ownership_clock().timestamp() * 1000)
            expired_count = 0
            for selection_key, receipt in sorted(
                self._context_selection_receipts.items(),
                key=lambda item: item[0] != request.selection_key,
            ):
                if expired_count >= CONTEXT_VIEW_EXPIRY_BATCH_SIZE:
                    break
                if (
                    receipt.owner == request.source_owner
                    and receipt.state == "selected"
                    and receipt.expires_at_ms <= now_ms
                ):
                    self._require_context_view_lifecycle_capacity(receipt.view.view_id)
                    expired = receipt.model_copy(update={"state": "expired"}, deep=True)
                    self._context_selection_receipts[selection_key] = expired
                    from cayu.sessions.context_views import ContextViewLifecycleEvent

                    operation_key = f"expiry:{selection_key}:{receipt.ownership_revision}"
                    event = ContextViewLifecycleEvent(
                        event_id="sha256:"
                        + sha256(f"context-view-event:{operation_key}".encode()).hexdigest(),
                        operation_key=operation_key,
                        selection_key=selection_key,
                        view_id=receipt.view.view_id,
                        state="expired",
                        owner=expired.owner,
                        owner_participant=expired.owner_participant,
                        pin_commitment=expired.pin_commitment,
                        ownership_revision=expired.ownership_revision,
                    )
                    self._context_lifecycle_events[event.event_id] = event
                    expired_count += 1
            existing = self._context_selection_receipts.get(request.selection_key)
            if existing is not None:
                if self._context_selection_request_commitments.get(request.selection_key) != (
                    request_commitment
                ):
                    raise ValueError("Context-view selection key conflicts with its request.")
                if existing.state == "selected" and existing.expires_at_ms <= now_ms:
                    self._require_context_view_lifecycle_capacity(existing.view.view_id)
                    existing = existing.model_copy(update={"state": "expired"}, deep=True)
                    self._context_selection_receipts[request.selection_key] = existing
                    from cayu.sessions.context_views import ContextViewLifecycleEvent

                    operation_key = f"expiry:{request.selection_key}:{existing.ownership_revision}"
                    event = ContextViewLifecycleEvent(
                        event_id="sha256:"
                        + sha256(f"context-view-event:{operation_key}".encode()).hexdigest(),
                        operation_key=operation_key,
                        selection_key=request.selection_key,
                        view_id=existing.view.view_id,
                        state="expired",
                        owner=existing.owner,
                        owner_participant=existing.owner_participant,
                        pin_commitment=existing.pin_commitment,
                        ownership_revision=existing.ownership_revision,
                    )
                    self._context_lifecycle_events[event.event_id] = event
                return existing.model_copy(deep=True)
            source = self._sessions.get(request.source_session_id)
            if source is None or source.instance_id != request.source_session_instance_id:
                raise LookupError("Context-view source session incarnation is unavailable.")
            from cayu.sessions._context_selection_fence import require_selection_source

            require_selection_source(
                target,
                self._participant_session_bindings.get(request.source_session_id),
                now_ms=now_ms,
            )
            candidates = [
                view
                for view in self._context_views.values()
                if view.source_owner == request.source_owner
                and view.source_session_id == request.source_session_id
                and view.source_session_instance_id == request.source_session_instance_id
                and view.projection_schema == request.projection_schema
                and view.extension_set_commitment == request.extension_set_commitment
                and (request.selector != "exact" or view.view_id == request.exact_view_id)
                and (
                    request.minimum_transcript_cursor is None
                    or view.transcript_cursor >= request.minimum_transcript_cursor
                )
            ]
            if not candidates:
                raise LookupError("No eligible context view is available.")
            if len(candidates) > request.limits.max_views:
                raise OverflowError("Context-view count exceeds the requested limit.")
            selection_count = sum(
                receipt.owner == request.source_owner
                for receipt in self._context_selection_receipts.values()
            )
            if selection_count >= CONTEXT_VIEW_MAX_SELECTIONS_PER_OWNER:
                raise OverflowError("Context-view selection quota exceeded for the owner.")
            active_pins = sum(
                1
                for receipt in self._context_selection_receipts.values()
                if receipt.owner == request.source_owner
                and receipt.state in {"selected", "adopted", "transferred"}
                and (receipt.state != "selected" or receipt.expires_at_ms > now_ms)
            )
            if active_pins >= request.limits.max_pins:
                raise OverflowError("Context-view pin count exceeds the requested limit.")
            selected = max(candidates, key=lambda view: (view.transcript_cursor, view.view_id))
            from cayu.sessions._context_selection_fence import require_selected_participant

            require_selected_participant(target, selected)
            self._require_context_view_lifecycle_capacity(selected.view_id, additional_slots=1)
            selected_bytes = context_view_manifest_bytes(selected)
            if selected_bytes > request.limits.max_view_bytes:
                raise OverflowError("Context-view manifest exceeds the requested byte limit.")
            retained_views = {
                receipt.view.view_id: receipt.view
                for receipt in self._context_selection_receipts.values()
                if receipt.owner == request.source_owner
                and receipt.state in {"selected", "adopted", "transferred"}
                and (receipt.state != "selected" or receipt.expires_at_ms > now_ms)
            }
            retained_views[selected.view_id] = selected
            retained_bytes = sum(
                context_view_manifest_bytes(view) for view in retained_views.values()
            )
            if retained_bytes > request.limits.max_retained_bytes:
                raise OverflowError("Context-view retained bytes exceed the requested limit.")
            pin_commitment = (
                "sha256:"
                + sha256(
                    f"context-pin:{request.selection_key}:{selected.view_id}".encode()
                ).hexdigest()
            )
            receipt = ContextViewSelectionReceiptType(
                selection_key=request.selection_key,
                view=selected,
                owner=request.source_owner,
                state="selected",
                pin_commitment=pin_commitment,
                owner_participant=selected.participant,
                expires_at_ms=int(
                    self._ownership_clock().timestamp() * 1000
                    + request.limits.max_lifetime_seconds * 1000
                ),
            )
            self._context_selection_receipts[request.selection_key] = receipt
            self._context_selection_request_commitments[request.selection_key] = request_commitment
            return receipt.model_copy(deep=True)

    async def _exclude_context_view_selection(self, request, *, authority):
        return await self._context_selection_control(request, authority=authority, exclude=True)

    async def _reserve_context_view_selection_control(self, request, *, authority):
        return await self._context_selection_control(request, authority=authority, exclude=False)

    async def _context_selection_control(
        self, request, *, authority, exclude, target=None, register=False
    ):
        from cayu.sessions._context_selection_fence import (
            CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER,
            ContextViewSelectionDecision,
            control_transition,
            require_authority,
            snapshot_request,
        )

        require_authority(authority)
        request = snapshot_request(request)
        async with self._lock:
            existing = self._context_selection_decision_unlocked(request)
            decision = control_transition(
                request, existing, exclude=exclude, target=target, register=register
            )
            if decision == existing:
                return decision
            if (
                existing is None
                and sum(
                    value.request.source_owner == request.source_owner
                    for value in self._context_selection_exclusions.values()
                )
                >= CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER
            ):
                raise OverflowError("Context-view selection control quota exceeded.")
            self._context_selection_exclusions[request.selection_key] = decision
            return ContextViewSelectionDecision.model_validate(decision)

    def _context_selection_decision_unlocked(self, request):
        from cayu.sessions._context_selection_fence import (
            ContextViewSelectionConflict,
            ContextViewSelectionDecision,
            selected_decision,
        )

        excluded = self._context_selection_exclusions.get(request.selection_key)
        receipt = self._context_selection_receipts.get(request.selection_key)
        if excluded is not None:
            excluded = ContextViewSelectionDecision.model_validate(excluded)
            if (
                excluded.request.selection_key != request.selection_key
                or (receipt is not None and excluded.state != "reserved")
                or excluded.state not in {"reserved", "excluded"}
            ):
                raise ValueError("Context-view exclusion indexes conflict.")
            if excluded.request != request:
                raise ContextViewSelectionConflict("Context-view exclusion evidence conflicts.")
            if receipt is None:
                return excluded
        if receipt is None:
            return None
        return selected_decision(
            request,
            receipt,
            self._context_selection_request_commitments.get(request.selection_key),
            excluded,
        )

    async def _read_context_view_retention(self, target):
        from cayu.collaboration._contracts import (
            ExactConflict,
            ExactMatch,
            ExactNotFound,
            ExactUnavailable,
        )
        from cayu.sessions._context_selection_fence import (
            ContextViewRetentionEvidence,
            ContextViewSelectionConflict,
            ContextViewSelectionTarget,
            retention_evidence,
        )

        target = ContextViewSelectionTarget.model_validate(target)
        async with self._lock:
            try:
                decision = self._context_selection_decision_unlocked(target.request)
                if decision is not None and decision.target != target:
                    return ExactConflict()
                result = retention_evidence(
                    target,
                    decision,
                    self._context_selection_receipts.get(target.request.selection_key),
                )
            except ContextViewSelectionConflict:
                return ExactConflict()
            except (TypeError, ValueError):
                return ExactUnavailable()
            return (
                ExactNotFound()
                if result is None
                else ExactMatch[ContextViewRetentionEvidence](receipt=result)
            )

    async def read_context_view_selection_decision(self, request):
        from cayu.collaboration._contracts import (
            ExactConflict,
            ExactMatch,
            ExactNotFound,
            ExactUnavailable,
        )
        from cayu.sessions._context_selection_fence import (
            ContextViewSelectionConflict,
            ContextViewSelectionDecision,
            snapshot_request,
        )

        request = snapshot_request(request)
        async with self._lock:
            try:
                decision = self._context_selection_decision_unlocked(request)
            except ContextViewSelectionConflict:
                return ExactConflict()
            except (TypeError, ValueError):
                return ExactUnavailable()
            return (
                ExactNotFound()
                if decision is None
                else ExactMatch[ContextViewSelectionDecision](receipt=decision)
            )

    async def lookup_context_view_selection(self, selection_key: str):
        from cayu.sessions.context_views import ContextViewSelectionReceipt

        async with self._lock:
            receipt = self._context_selection_receipts.get(selection_key)
            if receipt is None:
                return None
            return ContextViewSelectionReceipt.model_validate(receipt.model_dump(mode="json"))

    async def transition_context_view_ownership(
        self, request: _ContextViewOwnershipRequest
    ) -> _ContextViewSelectionReceipt:
        return await self._transition_context_view_ownership(request)

    async def _transition_context_view_ownership(self, request, *, target=None):
        from cayu.sessions.context_views import (
            ContextViewOwnershipRequest as ContextViewOwnershipRequestType,
        )
        from cayu.sessions.context_views import (
            json_commitment,
        )

        if type(request) is not ContextViewOwnershipRequestType:
            raise TypeError("Context-view ownership requires a typed request.")
        request = ContextViewOwnershipRequestType.model_validate(request)
        request_commitment = json_commitment(
            canonical_bounded_durable_json_bytes(
                request.model_dump(mode="json"),
                "context view ownership request",
                max_bytes=256 * 1024,
                max_nodes=8192,
                max_nesting=64,
            ).decode("utf-8"),
            "context view ownership request",
        )
        async with self._lock:
            from cayu.sessions._context_selection_fence import require_selection_adoption

            control = self._context_selection_exclusions.get(request.selection_key)
            require_selection_adoption(
                request,
                self._context_selection_decision_unlocked(control.request)
                if control is not None
                else None,
                target=target,
            )
            prior_commitment = self._context_ownership_operation_commitments.get(
                request.operation_key
            )
            if prior_commitment is not None:
                if prior_commitment != request_commitment:
                    raise ValueError("Ownership operation key conflicts with its request.")
                receipt = self._context_ownership_operation_receipts[request.operation_key]
                return receipt.model_copy(deep=True)
            receipt = self._context_selection_receipts.get(request.selection_key)
            if receipt is None or receipt.view.view_id != request.view_id:
                raise LookupError("Context-view selection is unavailable.")
            if receipt.pin_commitment != request.pin_commitment:
                raise ValueError("Context-view pin commitment conflicts with the request.")
            if receipt.state != request.expected_state:
                raise ValueError("Context-view ownership state is stale or already settled.")
            if receipt.ownership_revision != request.expected_revision:
                raise ValueError("Context-view ownership revision is stale.")
            if receipt.owner != request.current_owner:
                raise PermissionError("The current owner does not control this context-view pin.")
            if (
                request.current_participant is not None
                and receipt.owner_participant is not None
                and receipt.owner_participant != request.current_participant
            ):
                raise PermissionError(
                    "The current participant does not control this context-view pin."
                )
            now_ms = int(self._ownership_clock().timestamp() * 1000)
            from cayu.sessions._context_selection_fence import require_adoption_deadline

            require_adoption_deadline(target, now_ms)
            self._require_context_view_lifecycle_capacity(receipt.view.view_id)
            if receipt.state == "selected" and receipt.expires_at_ms <= now_ms:
                expired = receipt.model_copy(update={"state": "expired"}, deep=True)
                self._context_selection_receipts[request.selection_key] = expired
                from cayu.sessions.context_views import ContextViewLifecycleEvent

                operation_key = f"expiry:{request.selection_key}:{receipt.ownership_revision}"
                self._context_lifecycle_events[
                    "sha256:" + sha256(f"context-view-event:{operation_key}".encode()).hexdigest()
                ] = ContextViewLifecycleEvent(
                    event_id="sha256:"
                    + sha256(f"context-view-event:{operation_key}".encode()).hexdigest(),
                    operation_key=operation_key,
                    selection_key=request.selection_key,
                    view_id=receipt.view.view_id,
                    state="expired",
                    owner=expired.owner,
                    owner_participant=expired.owner_participant,
                    pin_commitment=expired.pin_commitment,
                    ownership_revision=expired.ownership_revision,
                )
                raise LookupError("Context-view selection has expired.")
            if request.operation == "adopt":
                assert request.destination_owner is not None
                next_state = "adopted"
                next_owner = request.destination_owner
            elif request.operation == "transfer":
                assert request.destination_owner is not None
                next_state = "transferred"
                next_owner = request.destination_owner
            else:
                next_state = "released"
                next_owner = receipt.owner
            next_participant = receipt.owner_participant
            if request.operation != "release":
                if request.destination_participant is None:
                    if request.destination_owner != receipt.owner:
                        raise PermissionError(
                            "A transfer to another owner requires destination participant evidence."
                        )
                else:
                    next_participant = request.destination_participant
            updated = receipt.model_copy(
                update={
                    "owner": next_owner,
                    "state": next_state,
                    "ownership_revision": receipt.ownership_revision + 1,
                    "owner_participant": next_participant,
                },
                deep=True,
            )
            self._require_context_view_lifecycle_capacity(
                updated.view.view_id, additional_slots=int(next_state != "released")
            )
            self._context_selection_receipts[request.selection_key] = updated
            self._context_ownership_operation_commitments[request.operation_key] = (
                request_commitment
            )
            self._context_ownership_operation_receipts[request.operation_key] = updated
            from cayu.sessions.context_views import ContextViewLifecycleEvent

            event = ContextViewLifecycleEvent(
                event_id="sha256:"
                + sha256(f"context-view-event:{request.operation_key}".encode()).hexdigest(),
                operation_key=request.operation_key,
                selection_key=request.selection_key,
                view_id=request.view_id,
                state=next_state,
                owner=updated.owner,
                owner_participant=updated.owner_participant,
                pin_commitment=updated.pin_commitment,
                ownership_revision=updated.ownership_revision,
            )
            self._context_lifecycle_events[event.event_id] = event
            return updated.model_copy(deep=True)

    async def validate_context_view_source_closure(self, session_id: str) -> None:
        async with self._lock:
            now_ms = int(self._ownership_clock().timestamp() * 1000)
            for receipt in self._context_selection_receipts.values():
                if (
                    receipt.view.source_session_id == session_id
                    and receipt.state in {"selected", "adopted", "transferred"}
                    and (receipt.state != "selected" or receipt.expires_at_ms > now_ms)
                ):
                    raise ValueError("Session has an active context-view retention pin.")

    async def validate_context_view_compaction(
        self, session_id: str, expected_transcript_cursor: int
    ) -> None:
        if type(expected_transcript_cursor) is not int or expected_transcript_cursor < 0:
            raise ValueError("Context-view compaction requires a non-negative transcript cursor.")
        async with self._lock:
            self._validate_context_view_compaction_unlocked(session_id, expected_transcript_cursor)

    def _validate_context_view_compaction_unlocked(
        self, session_id: str, expected_transcript_cursor: int
    ) -> None:
        from cayu.sessions.context_views import require_independent_context_view_material

        now_ms = int(self._ownership_clock().timestamp() * 1000)
        for receipt in self._context_selection_receipts.values():
            if (
                receipt.view.source_session_id == session_id
                and receipt.state in {"selected", "adopted", "transferred"}
                and (receipt.state != "selected" or receipt.expires_at_ms > now_ms)
                and receipt.view.transcript_cursor <= expected_transcript_cursor
            ):
                manifest = self._context_views.get(receipt.view.view_id)
                if manifest is None or manifest != receipt.view:
                    raise ValueError("Pinned context-view material is unavailable for compaction.")
                require_independent_context_view_material(manifest)

    async def read_context_view_lifecycle_events(
        self, view_id: str, *, limit: int = 256, owner_participant: _ParticipantRef | None = None
    ) -> tuple[_ContextViewLifecycleEvent, ...]:
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("Context-view lifecycle event limits must be between 1 and 256.")
        if owner_participant is not None:
            from cayu.collaboration.participants import ParticipantRef

            owner_participant = ParticipantRef.model_validate(owner_participant)
        async with self._lock:
            events = [
                event
                for event in self._context_lifecycle_events.values()
                if event.view_id == view_id
                and (owner_participant is None or event.owner_participant == owner_participant)
            ]
            events.sort(key=lambda event: (event.ownership_revision, event.event_id))
            return tuple(event.model_copy(deep=True) for event in events[:limit])

    async def read_context_view(
        self, view_id: str, *, source_session_id: str
    ) -> _ContextViewReadback:
        from cayu.sessions.context_views import ContextViewReadback as ContextViewReadbackType

        async with self._lock:
            view = self._context_views.get(view_id)
            if view is None or view.source_session_id != source_session_id:
                raise LookupError("Context view is unavailable.")
            return ContextViewReadbackType(view=view.model_copy(deep=True))

    async def create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        system_prompt_replacement: ForkSystemPromptReplacement | None = None,
        expected_source_run_epoch: int,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> Session:
        return await self._create_fork(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            system_prompt_replacement=system_prompt_replacement,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=None,
            operation_initializer=operation_initializer,
        )

    async def create_fork_with_transcript_validation(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        system_prompt_replacement: ForkSystemPromptReplacement | None = None,
        expected_source_run_epoch: int,
        transcript_validator: ForkTranscriptValidator,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> Session:
        return await self._create_fork(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            system_prompt_replacement=system_prompt_replacement,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=transcript_validator,
            operation_initializer=operation_initializer,
        )

    async def create_profiled_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        system_prompt_replacement: ForkSystemPromptReplacement | None = None,
        expected_source_run_epoch: int,
        relationship: SessionForkProfileRelationship,
        events: list[Event],
        transcript_validator: ForkTranscriptValidator | None = None,
        checkpoint_authority_decoder: ForkCheckpointAuthorityDecoder | None = None,
        operation_initializer: SessionOperationInitializer | None = None,
    ) -> ProfiledSessionForkResult:
        relationship, copied_events = session_fork_rules._copy_profiled_fork_authority(
            fork=fork,
            relationship=relationship,
            events=events,
        )
        created = await self._create_fork(
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            system_prompt_replacement=system_prompt_replacement,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=transcript_validator,
            profile_relationship=relationship,
            events=copied_events,
            checkpoint_authority_decoder=checkpoint_authority_decoder,
            operation_initializer=operation_initializer,
        )
        return ProfiledSessionForkResult(session=created, events=tuple(copied_events))

    async def _create_fork(
        self,
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
        async with self._lock:
            self._require_available_closure_identity_unlocked(fork.id)
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (source_session_id,))
            source_session = session_fork_rules._validate_session_fork_source(
                source_session=self._sessions.get(source_session_id),
                source_session_id=source_session_id,
                fork=fork,
                allowed_statuses=allowed_statuses,
                expected_source_run_epoch=expected_source_run_epoch,
                profile_relationship=profile_relationship,
            )
            if MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY in self._session_operation_records.get(
                source_session_id, {}
            ):
                raise SessionForkActiveModelStageConflict(
                    "Cannot fork a session while a model-completion stage is active."
                )
            if fork.id in self._sessions:
                raise ValueError(f"Session already exists: {fork.id}")

            stored_checkpoint = self._checkpoints.get(source_session_id)
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
                    profile_failure = session_fork_rules._profiled_fork_authority_validation_error(
                        exc
                    )
                    stored_checkpoint = None
                finally:
                    profile_checkpoint = None
                if profile_failure is not None:
                    raise profile_failure from None

            source_transcript = self._transcripts.get(source_session_id, [])
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
            source_interaction_ids = self._transcript_interaction_ids.get(source_session_id, [])
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
                checkpoint_input = (
                    None if stored_checkpoint is None else deepcopy(stored_checkpoint)
                )
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
                    supports_model_failover=self._supports_model_failover_stage_protocol(),
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

            self._register_private_authority_alias_unlocked(
                fork.id,
                field_name="session_id",
            )
            self._sessions[fork.id] = fork.model_copy(deep=True)
            self._index_session_parent_unlocked(fork)
            self._events[fork.id] = []
            self._event_ids[fork.id] = set()
            self._session_event_records[fork.id] = []
            self._child_lifecycle_event_records[fork.id] = {}
            self._interaction_event_records[fork.id] = {}
            self._latest_interaction_event_records[fork.id] = {}
            self._latest_interaction_event_records_by_sequence[fork.id] = []
            self._pending_action_event_records[fork.id] = {}
            self._session_operation_records[fork.id] = initial_operation_records
            self._transcripts[fork.id] = copied_transcript
            memory_transcripts.replace_latest_role_indexes(
                self._transcript_indexes_unlocked(), fork.id
            )
            memory_transcripts.replace_search_session(
                self._transcript_indexes_unlocked(), fork.id, copied_transcript
            )
            # Historical attribution remains tied to the source session's
            # interactions; new child work receives new child interaction IDs.
            self._transcript_interaction_ids[fork.id] = copied_interaction_ids
            for interaction_id in set(self._transcript_interaction_ids[fork.id]):
                if interaction_id is None:
                    continue
                self._register_private_authority_alias_unlocked(
                    interaction_id,
                    field_name="interaction_id",
                    scope_session_id=fork.id,
                )
            if copied_checkpoint is not None:
                self._store_checkpoint_unlocked(fork.id, copied_checkpoint)
            if events:
                self._sessions[fork.id] = self._append_events_unlocked(
                    self._sessions[fork.id],
                    events,
                )
            else:
                self._refresh_child_lifecycle_candidate_unlocked(self._sessions[fork.id])
            return self._sessions[fork.id].model_copy(deep=True)

    async def load(self, session_id: str) -> Session | None:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        if access_bounds is not None:
            from cayu._resource_access_errors import ResourceAccessDenied

            try:
                return await self._access_load_session(access_bounds, session_id)
            except ResourceAccessDenied:
                return None
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return session.model_copy(deep=True)

    async def load_state(self, session_id: str) -> SessionStateSnapshot | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return SessionStateSnapshot(
                id=session.id,
                status=session.status,
                updated_at=session.updated_at,
                last_activity_at=session.last_activity_at,
            )

    async def load_invocation_snapshot(
        self,
        session_id: str,
    ) -> SessionInvocationSnapshot | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return SessionInvocationSnapshot(
                id=session.id,
                session_instance_id=session.instance_id,
                status=session.status,
                invocation=session.invocation,
            )

    async def inspect_identity(self, session_id: str) -> SessionInspectionIdentity:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            labels, label_count, labels_truncated = (
                session_inspection_rules._bounded_session_inspection_labels(session.labels)
            )
            return SessionInspectionIdentity(
                id=session.id,
                agent_name=session.agent_name,
                provider_name=session.provider_name,
                model=session.model,
                parent_session_id=session.parent_session_id,
                causal_budget_id=session.causal_budget_id,
                runtime_name=session.runtime_name,
                runtime_version=session.runtime_version,
                runtime_build_provenance=session.runtime_build_provenance,
                environment_name=session.environment_name,
                status=session.status,
                created_at=session.created_at,
                updated_at=session.updated_at,
                last_activity_at=session.last_activity_at,
                run_epoch=session.run_epoch,
                labels=labels,
                label_count=label_count,
                labels_truncated=labels_truncated,
            )

    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(status, SessionStatus):
            raise ValueError("Session status must be a SessionStatus.")
        # Delegate to the guarded transition machine with every source status
        # allowed: preserves the unconditional (any -> status) setter contract
        # while sharing one write path and its not-found guard.
        return await self.transition_status(
            session_id,
            from_statuses=set(SessionStatus),
            to_status=status,
        )

    def _require_session_erasure_quiescence_unlocked(self, session: Session) -> None:
        """Shared admission for closure and final deletion; no mutations."""
        self._require_external_wait_quiescence_unlocked(session)
        from cayu.collaboration import _session_export_store as session_exports
        from cayu.sessions import _session_continuation_store as continuations
        from cayu.sessions._closure_records import require_terminal_protected_effect

        session_id = session.id
        continuations.require_erasure_quiescence(
            session=session,
            checkpoint=self._checkpoints.get(session_id),
            records=self._session_operation_records.get(session_id, {}),
        )
        session_exports.require_erasure_quiescence(
            session=session,
            checkpoint=self._checkpoints.get(session_id),
            export_records={
                key: value
                for key, value in self._session_operation_records.get(session_id, {}).items()
                if key.startswith(session_exports.OPERATION_PREFIX)
            },
        )
        from cayu.runtime._producer_output_store import require_erasure_quiescence
        from cayu.sessions._producer_checkpoint import OPERATION_PREFIX

        require_erasure_quiescence(
            session=session,
            checkpoint=self._checkpoints.get(session_id),
            records={
                key: value
                for key, value in self._session_operation_records.get(session_id, {}).items()
                if key.startswith(OPERATION_PREFIX)
            },
        )
        for key, raw in self._session_operation_records.get(session_id, {}).items():
            if key.startswith("tool-effect:"):
                require_terminal_protected_effect(session_id, session.instance_id, key, raw)
        if session.status in DELETE_BLOCKED_SESSION_STATUSES:
            raise ValueError("Session closure requires a non-running target.")
        if any(
            delivery.session_id == session_id
            and delivery.status is PersistedEventSideEffectStatus.LEASED
            for delivery in self._persisted_event_side_effect_deliveries.values()
        ):
            raise ValueError("Session closure requires settled event side-effect deliveries.")
        checkpoint = self._checkpoints.get(session_id)
        deletion_now = self._ownership_clock()
        active_recovery_claim_id = (
            session_recovery_rules._active_unexpired_incomplete_recovery_claim_id(
                checkpoint,
                now=deletion_now,
            )
        )
        if active_recovery_claim_id is not None:
            raise ValueError(
                "Cannot delete a session while incomplete-session recovery claim "
                f"{active_recovery_claim_id} is active: {session_id}"
            )
        run_operation = terminal_event_evidence._session_run_operation_from_checkpoint(checkpoint)
        if run_operation is not None:
            raise ValueError(
                "Cannot delete a session while terminal publication "
                f"{run_operation.operation_id} is incomplete: {session_id}"
            )
        queued_terminal_receipts = (
            terminal_event_evidence._queued_dispatch_terminal_receipts_from_checkpoint(checkpoint)
        )
        if queued_terminal_receipts:
            raise ValueError(
                "Cannot delete a session while queued dispatch terminal "
                f"acknowledgement is incomplete: {session_id}"
            )
        evidence_events: list[Event] = []
        for record in reversed(self._session_event_records.get(session_id, [])):
            if (
                record.event.type
                not in terminal_session_evidence._TERMINAL_PUBLICATION_EVIDENCE_EVENT_TYPES
            ):
                continue
            evidence_events.append(record.event)
            if (
                len(evidence_events)
                == terminal_session_evidence._TERMINAL_PUBLICATION_EVIDENCE_QUERY_LIMIT
            ):
                break
        terminal_publication_block = (
            terminal_session_evidence._terminal_publication_delete_block_reason(
                session=session,
                checkpoint=checkpoint,
                evidence_events=evidence_events,
            )
        )
        if terminal_publication_block is not None:
            raise ValueError(
                f"Cannot delete a session while {terminal_publication_block}: {session_id}"
            )
        active_operation_id = operation_record_rules._active_unexpired_session_operation_id(
            checkpoint,
            now=deletion_now,
        )
        if active_operation_id is not None:
            raise ValueError(
                "Cannot delete a session while durable operation "
                f"{active_operation_id} is active: {session_id}"
            )
        completion_result_publication_block = (
            checkpoint_preservation._completion_result_event_publication_delete_block_reason(
                checkpoint,
                now=deletion_now,
            )
        )
        if completion_result_publication_block is not None:
            raise ValueError(
                f"Cannot delete a session while {completion_result_publication_block}: {session_id}"
            )
        if MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY in self._session_operation_records.get(
            session_id, {}
        ):
            raise ValueError(
                f"Cannot delete a session while a model-completion stage is active: {session_id}"
            )
        owned_reservation_ids = {
            reservation_id
            for reservation_id, ownership in self._budget_reservation_identities.items()
            if ownership[0] == session_id
        }
        for reservation_id in owned_reservation_ids:
            terminal_events = [
                event
                for event in self._events.get(session_id, [])
                if event.type
                in {
                    EventType.BUDGET_RECONCILED,
                    EventType.BUDGET_RESERVATION_RELEASED,
                }
                and event.payload.get("reservation_id") == reservation_id
            ]
            delivery = (
                None
                if len(terminal_events) != 1
                else self._persisted_event_side_effect_deliveries.get(
                    (session_id, terminal_events[0].id)
                )
            )
            if delivery is None or delivery.status is not PersistedEventSideEffectStatus.DELIVERED:
                raise ValueError(
                    "Cannot delete a session while a budget settlement audit "
                    f"event is pending: {session_id}"
                )

    async def validate_session_closure_admission(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError("Closure target is unavailable.")
            self._require_session_erasure_quiescence_unlocked(session)

    @runtime_session_mutation
    async def delete_session(
        self,
        session_id: str,
        *,
        closure_receipt: dict[str, Any] | None = None,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> None:
        # Participant creation/replay acquires this lock before the store lock.
        # Deletion follows the same order so a committed session cannot lose
        # its receipt between the atomic create and readback.
        await self._participant_creation_lock.acquire()
        try:
            await self._delete_session_unserialized(
                session_id,
                closure_receipt=closure_receipt,
                _access_bounds=_access_bounds,
            )
        finally:
            self._participant_creation_lock.release()

    async def _delete_session_unserialized(
        self,
        session_id: str,
        *,
        closure_receipt: dict[str, Any] | None = None,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if _access_bounds is not None:
                _access_bounds.require_action(session, "delete")
            if session is None:
                return  # idempotent: deleting a missing session is a no-op
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,), closure_receipt)
            if closure_receipt is not None and closure_receipt.get("operation") == "recursive":
                expected_parent = closure_receipt.get("original_parent_session_id")
                if type(expected_parent) is not str or session.parent_session_id != expected_parent:
                    raise ValueError("Recursive closure child parent identity conflict.")
            if session.status in DELETE_BLOCKED_SESSION_STATUSES:
                raise ValueError(
                    f"Cannot delete a session while it is {session.status}; "
                    f"interrupt it first: {session_id}"
                )
            if self.context_view_version is not None:
                now_ms = int(self._ownership_clock().timestamp() * 1000)
                for receipt in self._context_selection_receipts.values():
                    if (
                        receipt.view.source_session_id == session_id
                        and receipt.state in {"selected", "adopted", "transferred"}
                        and (receipt.state != "selected" or receipt.expires_at_ms > now_ms)
                    ):
                        raise ValueError("Session has an active context-view retention pin.")
            for _, child_id in self._child_session_keys_by_parent.get(session_id, []):
                child = self._sessions.get(child_id)
                if child is None or child.parent_session_id != session_id:
                    raise RuntimeError("Inconsistent in-memory session topology index.")
                for owner in self._session_closure_progress.values():
                    closure_records._check_closure_lineage_owner(owner, (child_id,))
                if closure_receipt is not None or _access_bounds is not None:
                    raise ValueError("Closure deletion requires no remaining child edges.")
                if lineage_rules._is_durable_subagent_child(child):
                    raise ValueError(
                        lineage_rules._durable_subagent_parent_delete_block_reason(child.id)
                    )
            self._require_session_erasure_quiescence_unlocked(session)
            self._remove_session_parent_index_unlocked(session)
            targeted_grant_ids = {
                grant_id
                for grant_id, record in self._targeted_tool_grants.items()
                if record.session_id == session_id
            }
            self._targeted_tool_grants = {
                grant_id: record
                for grant_id, record in self._targeted_tool_grants.items()
                if grant_id not in targeted_grant_ids
            }
            self._targeted_tool_grant_ids_by_session.pop(session_id, None)
            self._targeted_tool_grant_ids_by_interaction = {
                key: grant_ids
                for key, grant_ids in self._targeted_tool_grant_ids_by_interaction.items()
                if key[0] != session_id
            }
            self._targeted_tool_grant_ids_by_request = {
                key: grant_id
                for key, grant_id in self._targeted_tool_grant_ids_by_request.items()
                if key[0] != session_id and grant_id not in targeted_grant_ids
            }
            self._targeted_tool_grant_ids_by_tool = {
                key: grant_id
                for key, grant_id in self._targeted_tool_grant_ids_by_tool.items()
                if key[0] != session_id and grant_id not in targeted_grant_ids
            }
            self._targeted_tool_grant_refs = {
                public_alias: target
                for public_alias, target in self._targeted_tool_grant_refs.items()
                if target[0] != session_id and target[1] not in targeted_grant_ids
            }
            for grant_id in targeted_grant_ids:
                self._targeted_tool_uses.pop(grant_id, None)
            self._targeted_tool_use_by_invocation = {
                key: binding
                for key, binding in self._targeted_tool_use_by_invocation.items()
                if key[0] != session_id and binding.grant_id not in targeted_grant_ids
            }
            self._targeted_tool_use_by_outer_call = {
                key: binding
                for key, binding in self._targeted_tool_use_by_outer_call.items()
                if key[0] != session_id and binding.grant_id not in targeted_grant_ids
            }
            self._sessions.pop(session_id, None)
            self._execution_owners.pop(session_id, None)
            if any(
                receipt.status == "appended"
                and receipt.target_session_id == session.id
                and receipt.target_session_instance_id == session.instance_id
                for receipt in self._peer_content_receipts.values()
            ):
                self._peer_content_deleted_targets.add((session.id, session.instance_id))
            binding = self._participant_session_bindings.pop(session_id, None)
            self._participant_session_receipts.pop(session_id, None)
            if binding is not None:
                self._participant_session_ids_by_key.pop(binding.creation_key, None)
                self._participant_session_request_commitments.pop(binding.creation_key, None)
            self._events.pop(session_id, None)
            self._event_ids.pop(session_id, None)
            self._persisted_event_side_effect_deliveries = {
                key: delivery
                for key, delivery in self._persisted_event_side_effect_deliveries.items()
                if key[0] != session_id
            }
            self._session_event_records.pop(session_id, None)
            self._interaction_event_records.pop(session_id, None)
            self._latest_interaction_event_records.pop(session_id, None)
            self._latest_interaction_event_records_by_sequence.pop(session_id, None)
            self._pending_action_event_records.pop(session_id, None)
            self._pending_action_index_scopes.pop(session_id, None)
            self._pending_action_latest_barrier_records.pop(session_id, None)
            self._user_input_supersession_event_records = {
                key: records
                for key, records in self._user_input_supersession_event_records.items()
                if key[0] != session_id
            }
            memory_transcripts.remove_search_session(
                self._transcript_indexes_unlocked(), session_id
            )
            self._transcripts.pop(session_id, None)
            self._transcript_interaction_ids.pop(session_id, None)
            self._latest_transcript_indexes_by_role.pop(session_id, None)
            self._transcript_indices_by_interaction.pop(session_id, None)
            self._child_lifecycle_event_records.pop(session_id, None)
            self._deferred_interaction_inputs.pop(session_id, None)
            self._refresh_queued_dispatch_terminal_receipts_unlocked(session_id, None)
            self._checkpoints.pop(session_id, None)
            receipt_ids = self._recall_receipt_ids_by_session.pop(session_id, set())
            exposure_ids = self._context_exposure_ids_by_session.pop(session_id, set())
            self._recall_receipt_page_keys_by_session.pop(session_id, None)
            self._context_exposure_page_keys_by_session.pop(session_id, None)
            for receipt_id in receipt_ids:
                receipt = self._recall_receipts.pop(receipt_id)
                self._recall_receipt_page_keys_by_interaction.pop(
                    (session_id, receipt.interaction_id),
                    None,
                )
                self._recall_receipt_page_keys_by_step.pop(
                    (session_id, receipt.model_step_id),
                    None,
                )
                self._recall_receipt_page_keys_by_interaction_step.pop(
                    (session_id, receipt.interaction_id, receipt.model_step_id),
                    None,
                )
            for exposure_id in exposure_ids:
                exposure = self._context_exposures.pop(exposure_id)
                self._context_exposure_page_keys_by_interaction.pop(
                    (session_id, exposure.interaction_id),
                    None,
                )
                self._context_exposure_page_keys_by_step.pop(
                    (session_id, exposure.model_step_id),
                    None,
                )
                self._context_exposure_page_keys_by_interaction_step.pop(
                    (session_id, exposure.interaction_id, exposure.model_step_id),
                    None,
                )
                self._recall_item_exposures.pop(exposure_id, None)
                self._context_exposure_by_model_attempt.pop(
                    (session_id, exposure.model_attempt_id),
                    None,
                )
                self._context_exposure_by_provider_attempt.pop(
                    (session_id, exposure.provider_attempt_id),
                    None,
                )
            self._session_operation_records.pop(session_id, None)
            self._pending_action_session_ids.discard(session_id)
            self._queued_session_messages_by_idempotency.pop(session_id, None)
            self._session_message_acceptance_event_ids = {
                key: ids
                for key, ids in self._session_message_acceptance_event_ids.items()
                if key[0] != session_id
            }
            self._session_message_terminal_receipts = {
                key: receipt
                for key, receipt in self._session_message_terminal_receipts.items()
                if key[0] != session_id
            }
            self._session_message_delivery_records = {
                delivery_id: record
                for delivery_id, record in self._session_message_delivery_records.items()
                if record.session_id != session_id
            }
            for delivery_mode in SessionMessageDeliveryMode:
                self._pending_session_messages.pop((session_id, delivery_mode), None)
            self._event_records = [
                record for record in self._event_records if record.event.session_id != session_id
            ]
            self._event_records_by_id = {
                key: record
                for key, record in self._event_records_by_id.items()
                if key[0] != session_id
            }
            self._accounting_generation += 1
            self._cost_event_index.remove_session(session_id)

            self._workflow_step_event_records = {
                key: records
                for key, records in self._workflow_step_event_records.items()
                if key[0] != session_id
            }
            self._workflow_fenced_step_event_records = {
                key: records
                for key, records in self._workflow_fenced_step_event_records.items()
                if key[0] != session_id
            }
            self._workflow_attempt_event_records = {
                key: records
                for key, records in self._workflow_attempt_event_records.items()
                if key[0] != session_id
            }
            self._workflow_active_attempt_ids = {
                key: attempt_id
                for key, attempt_id in self._workflow_active_attempt_ids.items()
                if key[0] != session_id
            }
            for event_type, records in list(self._type_event_records.items()):
                remaining = [record for record in records if record.event.session_id != session_id]
                if remaining:
                    self._type_event_records[event_type] = remaining
                else:
                    del self._type_event_records[event_type]
            # Mirror the SQL FK's ON DELETE SET NULL: children keep loading, parent ref
            # cleared. The branch index makes this proportional to the deleted
            # session's direct children rather than the complete registry.
            child_keys = self._child_session_keys_by_parent.pop(session_id, [])
            candidate_priorities = self._child_lifecycle_candidate_keys_by_parent.pop(
                session_id,
                {},
            )
            indexed_candidate_ids = {
                child_id for keys in candidate_priorities.values() for _sort_at, child_id in keys
            }
            if indexed_candidate_ids != {child_id for _created_at, child_id in child_keys}:
                raise RuntimeError("Inconsistent in-memory child lifecycle candidate branch.")
            for child_id in indexed_candidate_ids:
                self._child_lifecycle_candidate_key_by_child.pop(child_id, None)
            for _, child_id in child_keys:
                child = self._sessions.get(child_id)
                if child is None or child.parent_session_id != session_id:
                    raise RuntimeError("Inconsistent in-memory session topology index.")
                self._sessions[child_id] = child.model_copy(update={"parent_session_id": None})
            if closure_receipt is not None:
                receipt_plan_id = closure_receipt.get("plan_id")
                if type(receipt_plan_id) is not str:
                    raise ValueError("Session closure receipt is missing plan identity.")
                receipt_key = (session_id, receipt_plan_id)
                existing = self._session_closure_receipts.get(receipt_key)
                if existing is not None and existing != closure_receipt:
                    raise ValueError("Session closure receipt identity conflict.")
                self._session_closure_receipts[receipt_key] = deepcopy(closure_receipt)

    @runtime_session_mutation
    async def update_labels(
        self,
        session_id: str,
        labels: dict[str, str],
        *,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        new_labels = copy_label_map(labels, "labels", allow_reserved=False)
        async with self._lock:
            session = self._sessions.get(session_id)
            if _access_bounds is not None:
                _access_bounds.require_label_update(session, new_labels)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            now = self._ownership_clock()
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            updated = session.model_copy(update={"labels": new_labels, "updated_at": now})
            from cayu.sessions._invocation_lifecycle import (
                require_invocation_lifecycle_release_capacity,
            )

            require_invocation_lifecycle_release_capacity(
                self._checkpoints.get(session_id),
                updated,
            )
            if _access_bounds is not None:
                audit = _access_bounds.label_audit(session, new_labels, now)
                if audit is not None:
                    key, record = audit
                    self._session_operation_records.setdefault(session_id, {})[key] = record
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    @runtime_session_mutation
    async def update_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
        *,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        user_metadata = copy_session_user_metadata(metadata)
        async with self._lock:
            session = self._sessions.get(session_id)
            if _access_bounds is not None:
                _access_bounds.require_action(session, "modify")
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            new_metadata = replace_session_user_metadata(session.metadata, user_metadata)
            now = self._ownership_clock()
            updated = session.model_copy(update={"metadata": new_metadata, "updated_at": now})
            from cayu.sessions._invocation_lifecycle import (
                require_invocation_lifecycle_release_capacity,
            )

            require_invocation_lifecycle_release_capacity(
                self._checkpoints.get(session_id),
                updated,
            )
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def transition_status(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = session_record_rules._validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {to_status}"
                )
            if to_status is SessionStatus.RUNNING:
                self._require_external_wait_admission_unlocked(session)

            now = self._ownership_clock()
            if to_status is SessionStatus.RUNNING:
                session_recovery_rules._require_live_incomplete_recovery_claim_for_run_epoch_transfer(
                    self._checkpoints.get(session_id),
                    now=now,
                )
            updated = session.model_copy(
                update={
                    "status": to_status,
                    "updated_at": now,
                    "last_activity_at": now,
                    "run_epoch": session.run_epoch + (to_status == SessionStatus.RUNNING),
                }
            )
            self._sessions[session_id] = updated
            self._refresh_child_lifecycle_candidate_unlocked(updated)
            result = updated.model_copy(deep=True)
            if to_status == SessionStatus.RUNNING:
                invocation_scope._activate_session_run_fence(result)
            return result

    async def reject_execution_profile_resume(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str | None = None,
        expected_statuses: set[SessionStatus],
        expected_run_epoch: int,
        expected_profile: ExecutionProfileIdentity,
        candidate_profile: ExecutionProfileIdentity,
        event: Event,
        decision: ExecutionProfileDecision | None = None,
        expected_active_invocation_profile_authority: CheckpointValueAuthority | None = None,
    ) -> ExecutionProfileRejectionResult:
        (
            session_id,
            statuses,
            expected_run_epoch,
            expected_profile,
            _candidate_profile,
            copied_event,
        ) = session_profile_rules._prepare_execution_profile_rejection(
            session_id,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_profile=expected_profile,
            candidate_profile=candidate_profile,
            event=event,
            decision=decision,
        )
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            session_profile_rules._validate_execution_profile_rejection_session(
                session,
                checkpoint=self._checkpoints.get(session_id),
                expected_session_instance_id=expected_session_instance_id,
                expected_statuses=statuses,
                expected_run_epoch=expected_run_epoch,
                expected_profile=expected_profile,
                event=copied_event,
                expected_active_invocation_profile_authority=(
                    expected_active_invocation_profile_authority
                ),
            )
            for existing in self._events.get(session_id, []):
                if existing.id != copied_event.id:
                    continue
                if not session_profile_rules._execution_profile_rejection_events_equivalent(
                    existing,
                    copied_event,
                ):
                    raise ValueError(
                        f"Execution-profile rejection id was reused: {copied_event.id}"
                    )
                return ExecutionProfileRejectionResult(event=existing, replayed=True)
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            self._sessions[session_id] = self._append_events_unlocked(
                session,
                [copied_event],
            )
            return ExecutionProfileRejectionResult(event=copied_event, replayed=False)

    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform: CheckpointTransform | None = None,
        store_time_checkpoint_transform: StoreTimeCheckpointTransform | None = None,
        result_checkpoint_transform: CheckpointTransform | None = None,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        continued_interaction_id: str | None = None,
        defer_interaction_source: bool = False,
        model_transition: SessionModelTransition | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
        execution_profile_decision: ExecutionProfileDecision | None = None,
        adopted_runtime_identity: SessionRuntimeIdentity | None = None,
        tool_capability_ceiling: ToolCapabilityCeiling | None = None,
        expected_latest_interaction_event_id: str | None = None,
        require_no_active_model_completion_dispatch: bool = False,
        temporary_service_admission: TemporaryServiceAdmission | None = None,
    ) -> Session:
        from cayu.sessions._temporary_continuation_scope import prepare_temporary_transition

        temporary_service_admission = prepare_temporary_transition(temporary_service_admission)
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = session_record_rules._validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        if (checkpoint_transform is None) == (store_time_checkpoint_transform is None):
            raise TypeError("Exactly one checkpoint transform is required.")
        if result_checkpoint_transform is not None and not callable(result_checkpoint_transform):
            raise TypeError("result_checkpoint_transform must be callable.")
        admission = interaction_rules._copy_transition_interaction_admission(
            session_id,
            interaction_started_event,
            interaction_source_messages,
            continued_interaction_id=continued_interaction_id,
            defer_interaction_source=defer_interaction_source,
        )
        prepared_model_transition = session_profile_rules._copy_session_model_transition(
            session_id,
            model_transition,
            interaction_id=(None if admission is None else admission[1]),
            interaction_is_new=(admission is not None and admission[0] is not None),
        )
        prepared_execution_profile = session_profile_rules._copy_optional_execution_profile(
            execution_profile
        )
        prepared_execution_profile_decision = (
            session_profile_rules._copy_optional_execution_profile_decision(
                execution_profile_decision
            )
        )
        prepared_adopted_runtime_identity = (
            None
            if adopted_runtime_identity is None
            else copy_session_runtime_identity(adopted_runtime_identity)
        )
        prepared_tool_capability_ceiling = (
            session_request_contracts._copy_optional_tool_capability_ceiling(
                tool_capability_ceiling
            )
        )
        expected_latest_interaction_event_id = session_event_rules._copy_optional_event_id(
            expected_latest_interaction_event_id,
            "expected_latest_interaction_event_id",
        )
        if type(require_no_active_model_completion_dispatch) is not bool:
            raise TypeError("require_no_active_model_completion_dispatch must be a boolean.")
        if prepared_execution_profile_decision is not None and admission is None:
            raise ValueError("An execution-profile decision requires atomic interaction admission.")
        if admission is not None and to_status is not SessionStatus.RUNNING:
            raise ValueError("Interaction admission requires a transition to running.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {to_status}"
                )
            if to_status is SessionStatus.RUNNING:
                self._require_external_wait_admission_unlocked(session)
            if expected_latest_interaction_event_id is not None:
                latest_interactions = self._latest_interaction_event_records_by_sequence.get(
                    session_id,
                    [],
                )
                latest_interaction = latest_interactions[-1] if latest_interactions else None
                if (
                    latest_interaction is None
                    or latest_interaction.event.id != expected_latest_interaction_event_id
                ):
                    raise SessionRunFenced(
                        "Session latest interaction changed before the status transition."
                    )
            if require_no_active_model_completion_dispatch:
                operation_records = self._session_operation_records.get(session_id, {})
                active_record = operation_records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
                if active_record is not None:
                    active_marker = (
                        model_stage_rules._reconstruct_active_model_completion_stage_record(
                            active_record,
                            session_id=session_id,
                        )
                    )
                    dispatch_key = model_stage_rules._model_completion_stage_dispatch_storage_key(
                        active_marker.stage_id
                    )
                    if operation_records.get(dispatch_key) is not None:
                        raise SessionModelCompletionDispatchAlreadyAuthorized(
                            "Provider dispatch was authorized before terminal decision election."
                        )
            transition_profile_metadata = (
                session_profile_rules._validate_execution_profile_admission(
                    session,
                    candidate_profile=prepared_execution_profile,
                    model_transition=prepared_model_transition,
                    decision=prepared_execution_profile_decision,
                )
            )

            if session.status == SessionStatus.PENDING and admission is not None and admission[3]:
                receipt = self._participant_session_receipts.get(session_id)
                if receipt is not None:
                    from cayu.sessions._participant_execution_identity import (
                        require_initial_execution_input,
                    )

                    require_initial_execution_input(
                        session, receipt, self._transcripts.get(session_id, []), admission[2]
                    )

            if prepared_model_transition is not None:
                session_profile_rules._validate_session_model_transition(
                    session,
                    self._transcripts.get(session_id, []),
                    len(self._transcripts.get(session_id, [])),
                    prepared_model_transition,
                )

            now = self._ownership_clock()
            if to_status is SessionStatus.RUNNING:
                session_recovery_rules._require_live_incomplete_recovery_claim_for_run_epoch_transfer(
                    self._checkpoints.get(session_id),
                    now=now,
                )
            session_updates: dict[str, Any] = {
                "status": to_status,
                "updated_at": now,
                "last_activity_at": now,
                "run_epoch": session.run_epoch + (to_status == SessionStatus.RUNNING),
            }
            transition_metadata = transition_profile_metadata
            if prepared_model_transition is not None:
                transition_metadata = (
                    session_profile_rules._session_metadata_after_model_transition(
                        session,
                        prepared_model_transition,
                        execution_profile_metadata=transition_profile_metadata,
                    )
                )
                session_updates.update(
                    provider_name=prepared_model_transition.target.provider_name,
                    model=prepared_model_transition.target.model,
                )
            transition_metadata = (
                session_profile_rules._session_metadata_after_runtime_identity_adoption(
                    session,
                    prepared_adopted_runtime_identity,
                    model_transition=prepared_model_transition,
                    execution_profile_metadata=transition_metadata,
                )
            )
            if prepared_adopted_runtime_identity is not None:
                session_updates.update(
                    runtime_name=prepared_adopted_runtime_identity.runtime_name,
                    runtime_version=prepared_adopted_runtime_identity.runtime_version,
                )
            transition_metadata = (
                session_profile_rules._session_metadata_after_tool_capability_ceiling_admission(
                    session,
                    prepared_tool_capability_ceiling,
                    transition_metadata=transition_metadata,
                    require_existing_ceiling=prepared_execution_profile is not None,
                )
            )

            current_checkpoint = self._checkpoints.get(session_id)
            checkpoint_copy = checkpoint_preservation._copy_checkpoint_for_transform(
                current_checkpoint,
                session_id=session_id,
            )
            if store_time_checkpoint_transform is not None:
                transformed_checkpoint = store_time_checkpoint_transform(
                    session.model_copy(deep=True),
                    checkpoint_copy,
                    now,
                )
            else:
                assert checkpoint_transform is not None
                transformed_checkpoint = checkpoint_transform(
                    session.model_copy(deep=True),
                    checkpoint_copy,
                )
            if transformed_checkpoint is not None:
                transformed_checkpoint = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                    current_checkpoint,
                    transformed_checkpoint,
                    session_id=session_id,
                )

            if admission is not None:
                _, interaction_id, _, defer_source = admission
                existing_deferred = self._deferred_interaction_inputs.get(session_id)
                if existing_deferred is not None and (
                    not defer_source or existing_deferred.interaction_id != interaction_id
                ):
                    raise RuntimeError("Session already has deferred interaction input.")
            if transition_metadata is not None:
                session_updates["metadata"] = transition_metadata
            updated = session.model_copy(update=session_updates)
            transformed_checkpoint = (
                model_failover_rules._model_failover_checkpoint_after_profile_admission(
                    source_session=session,
                    admitted_session=updated,
                    source_checkpoint=current_checkpoint,
                    admitted_checkpoint=transformed_checkpoint,
                    candidate_profile=prepared_execution_profile,
                    records=self._session_operation_records.get(session_id, {}),
                    transcript_cursor=len(self._transcripts.get(session_id, [])),
                    supports_model_failover=self._supports_model_failover_stage_protocol(),
                )
            )
            if result_checkpoint_transform is not None:
                result_checkpoint = result_checkpoint_transform(
                    updated.model_copy(deep=True),
                    checkpoint_preservation._copy_checkpoint_for_transform(
                        transformed_checkpoint,
                        session_id=session_id,
                    ),
                )
                if result_checkpoint is None:
                    raise ValueError("Result checkpoint transform must return a checkpoint.")
                transformed_checkpoint = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                    transformed_checkpoint,
                    result_checkpoint,
                    session_id=session_id,
                )
            temporary_records: dict[str, dict[str, Any]] = {}
            if temporary_service_admission is not None:
                from cayu.sessions._session_continuation import continuation_operation_key
                from cayu.sessions._temporary_continuation import temporary_service_key
                from cayu.sessions._temporary_continuation_store import (
                    compose_temporary_service_admission,
                )

                parent_key = continuation_operation_key(
                    temporary_service_admission.dispatch.intent.ticket
                )
                from cayu.sessions._temporary_service_target import target_service_key

                intent = temporary_service_admission.dispatch.intent
                child_key = (
                    temporary_service_key(intent.operation)
                    if intent.mode == "same_session"
                    else target_service_key(intent.operation)
                )
                publication = compose_temporary_service_admission(
                    source_session=session,
                    source_checkpoint=current_checkpoint,
                    parent_record=self._session_operation_records.get(session_id, {}).get(
                        parent_key
                    ),
                    child_record=self._session_operation_records.get(session_id, {}).get(child_key),
                    admitted_session=updated,
                    admitted_checkpoint=transformed_checkpoint,
                    admission=temporary_service_admission,
                    now=now,
                )
                transformed_checkpoint = publication.checkpoint
                temporary_records = publication.operation_records
            prepared_events: _PreparedInMemoryEventAppend | None = None
            admission_events = []
            if prepared_execution_profile_decision is not None:
                admission_events.append(prepared_execution_profile_decision.event)
            if prepared_model_transition is not None:
                admission_events.append(prepared_model_transition.event)
            if admission is not None and admission[0] is not None:
                admission_events.append(admission[0])
            if admission_events:
                prepared_events = self._prepare_event_append_unlocked(
                    updated,
                    admission_events,
                )
            prepared_checkpoint = (
                None
                if transformed_checkpoint is None
                else self._prepare_checkpoint_store_unlocked(
                    session_id,
                    transformed_checkpoint,
                    additional_event_records=(
                        ()
                        if prepared_events is None
                        else tuple(event.record for event in prepared_events.events)
                    ),
                )
            )

            self._sessions[session_id] = updated
            self._refresh_child_lifecycle_candidate_unlocked(updated)
            if prepared_checkpoint is not None:
                self._apply_checkpoint_store_unlocked(session_id, prepared_checkpoint)
            if temporary_records:
                self._session_operation_records.setdefault(session_id, {}).update(temporary_records)
            if admission is not None:
                _, interaction_id, source_messages, defer_source = admission
                if prepared_events is not None:
                    self._sessions[session_id] = self._apply_event_append_unlocked(
                        updated,
                        prepared_events,
                        activity_at=now,
                    )
                if defer_source:
                    self._deferred_interaction_inputs[session_id] = DeferredInteractionInput(
                        interaction_id=interaction_id,
                        source_messages=source_messages,
                    )
                else:
                    self._transcripts[session_id].extend(source_messages)
                    memory_transcripts.extend_search(
                        self._transcript_indexes_unlocked(), session_id, source_messages
                    )
                    memory_transcripts.extend_attribution(
                        self._transcript_indexes_unlocked(),
                        session_id,
                        [interaction_id] * len(source_messages),
                    )
                updated = self._sessions[session_id]
            result = updated.model_copy(deep=True)
            if to_status == SessionStatus.RUNNING:
                invocation_scope._activate_session_run_fence(result)
            return result

    async def transition_status_if_no_queued_messages(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_mutation: dict[str, Any] | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = session_record_rules._validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        mutation = completion_finalization._prepare_queue_completion_checkpoint_mutation(
            checkpoint_mutation, to_status
        )
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {to_status}"
                )
            if any(
                (session_id, delivery_mode) in self._pending_session_messages
                for delivery_mode in SessionMessageDeliveryMode
            ):
                raise SessionQueuedMessagesPending(
                    f"Session has durable queued messages: {session_id}"
                )
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            now = self._ownership_clock()
            checkpoint = (
                None
                if mutation is None
                else completion_finalization._apply_queue_completion_checkpoint_mutation(
                    session, mutation, self._checkpoints.get(session_id)
                )
            )
            updated = session.model_copy(
                update={
                    "status": to_status,
                    "updated_at": now,
                    "last_activity_at": now,
                    "run_epoch": session.run_epoch
                    + (to_status == SessionStatus.RUNNING and mutation is None),
                }
            )
            self._sessions[session_id] = updated
            if checkpoint is not None:
                self._store_checkpoint_unlocked(session_id, checkpoint)
            self._refresh_child_lifecycle_candidate_unlocked(updated)
            result = updated.model_copy(deep=True)
            if to_status == SessionStatus.RUNNING:
                invocation_scope._activate_session_run_fence(result)
            return result

    async def publish_interaction_transition(
        self,
        session_id: str,
        *,
        event: Event,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        only_if_no_queued_messages: bool = False,
        model_completion_stage_settlement: ModelCompletionStageSettlementRequest | None = None,
        checkpoint_mutation: dict[str, Any] | None = None,
        terminal_event: Event | None = None,
        terminal_decision: InvocationTerminalDecision | None = None,
        expected_session_instance_id: str | None = None,
        expected_active_invocation_profile: ActiveInvocationExecutionProfile | None = None,
        expected_invocation_authority_state: Literal["active", "released"] = "active",
        expected_recovery_claim_id: str | None = None,
        terminalization_only: bool = False,
        terminalization_plan_ownership: Any = None,
    ) -> InteractionTransitionResult:
        expected_recovery_claim_id = (
            interaction_receipts._validate_interaction_transition_recovery_claim_id(
                expected_recovery_claim_id
            )
        )
        expected_invocation_authority_state = (
            interaction_receipts._validate_interaction_transition_invocation_authority_parameters(
                expected_session_instance_id=expected_session_instance_id,
                expected_active_invocation_profile=expected_active_invocation_profile,
                expected_invocation_authority_state=expected_invocation_authority_state,
            )
        )
        session_id, transition = interaction_rules._prepare_interaction_transition(
            session_id,
            event=event,
            from_statuses=from_statuses,
            to_status=to_status,
            only_if_no_queued_messages=only_if_no_queued_messages,
            model_completion_stage_settlement=model_completion_stage_settlement,
            checkpoint_mutation=checkpoint_mutation,
            terminal_event=terminal_event,
            terminal_decision=terminal_decision,
        )
        copied_event = transition.event
        allowed_statuses = set(transition.from_statuses)
        target_status = transition.to_status
        conditional = transition.only_if_no_queued_messages
        settlement_request = transition.model_completion_stage_settlement
        checkpoint_mutation_request = transition.checkpoint_mutation
        copied_terminal_event = transition.terminal_event
        terminal_decision = transition.terminal_decision
        receipt_storage_key = interaction_receipts._interaction_transition_storage_key(
            copied_event.id
        )
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if expected_active_invocation_profile is None:
                invocation_scope._assert_session_run_epoch(session_id, session)
            operation_records = self._session_operation_records.setdefault(session_id, {})
            receipt_record = operation_records.get(receipt_storage_key)
            existing = self._event_records_by_id.get((session_id, copied_event.id))
            existing_terminal = (
                None
                if copied_terminal_event is None
                else self._event_records_by_id.get((session_id, copied_terminal_event.id))
            )
            if receipt_record is not None:
                receipt = interaction_receipts._reconstruct_interaction_transition_receipt(
                    receipt_record,
                    transition=transition,
                )
                interaction_receipts._validate_interaction_transition_receipt_authority(
                    receipt,
                    current_session=session,
                    current_checkpoint=self._checkpoints.get(session_id),
                    expected_session_instance_id=expected_session_instance_id,
                    expected_active_invocation_profile=expected_active_invocation_profile,
                    expected_invocation_authority_state=(expected_invocation_authority_state),
                    expected_recovery_claim_id=expected_recovery_claim_id,
                )
                if existing is not None and existing.event != receipt.event:
                    raise RuntimeError(
                        "Interaction transition receipt conflicts with retained event history."
                    )
                if copied_terminal_event is not None and (
                    receipt.terminal_event != copied_terminal_event
                    or (
                        existing_terminal is not None
                        and existing_terminal.event != receipt.terminal_event
                    )
                ):
                    raise RuntimeError(
                        "Interaction transition receipt conflicts with its terminal session event."
                    )
                return InteractionTransitionResult(
                    session=receipt.session,
                    event=receipt.event,
                    terminal_event=receipt.terminal_event,
                    status_changed=receipt.status_changed,
                    replayed=True,
                )
            if existing is not None:
                raise RuntimeError(
                    "Interaction transition event exists without its immutable receipt."
                )
            if existing_terminal is not None:
                raise RuntimeError("Terminal session event exists without its interaction receipt.")
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            current_checkpoint = self._checkpoints.get(session_id)
            if terminalization_only:
                from cayu.runtime._durable_model_terminalization import (
                    require_terminalization_checkpoint,
                )
                from cayu.sessions._model_stage_rules import require_terminalization_plan_owner

                require_terminalization_checkpoint(session, current_checkpoint)

                require_terminalization_plan_owner(
                    current_checkpoint, terminalization_plan_ownership, self._ownership_clock()
                )
                if (
                    self._child_session_keys_by_parent.get(session_id)
                    or self._deferred_interaction_inputs.get(session_id)
                    or any(
                        (session_id, mode) in self._pending_session_messages
                        for mode in SessionMessageDeliveryMode
                    )
                ):
                    raise SessionRunFenced("Model terminalization has dependent work.")
            settled_checkpoint = (
                interaction_receipts._checkpoint_after_exact_invocation_terminal_decision(
                    current_checkpoint,
                    session=session,
                    expected=terminal_decision,
                )
            )
            active_recovery_claim_id = (
                session_recovery_rules._active_unexpired_incomplete_recovery_claim_id(
                    current_checkpoint,
                    now=self._ownership_clock(),
                )
            )
            if expected_recovery_claim_id is None:
                if active_recovery_claim_id is not None:
                    raise SessionRunFenced(
                        "Interaction transition is owned by another terminal recovery claim."
                    )
            elif active_recovery_claim_id != expected_recovery_claim_id:
                raise SessionRunFenced(
                    "Interaction transition lost its exact terminal recovery claim."
                )
            if expected_active_invocation_profile is not None:
                from cayu.sessions._invocation_lifecycle import (
                    require_invocation_command_authority,
                    require_released_invocation_command_authority,
                )

                assert expected_session_instance_id is not None
                if expected_invocation_authority_state == "released":
                    require_released_invocation_command_authority(
                        session,
                        current_checkpoint,
                        session_id=session_id,
                        session_instance_id=expected_session_instance_id,
                        active_profile=expected_active_invocation_profile,
                        events=tuple(
                            event
                            for event in (copied_event, copied_terminal_event)
                            if event is not None
                        ),
                    )
                else:
                    require_invocation_command_authority(
                        session,
                        current_checkpoint,
                        session_id=session_id,
                        session_instance_id=expected_session_instance_id,
                        run_epochs=frozenset({expected_active_invocation_profile.run_epoch}),
                        active_profile=expected_active_invocation_profile,
                        events=tuple(
                            event
                            for event in (copied_event, copied_terminal_event)
                            if event is not None
                        ),
                    )
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {target_status}"
                )
            queued = conditional and any(
                (session_id, delivery_mode) in self._pending_session_messages
                for delivery_mode in SessionMessageDeliveryMode
            )
            from cayu.runtime._session_steering import (
                interaction_completion_steering_key,
                prepare_interaction_completion_steering_record,
            )

            steering_key = interaction_completion_steering_key(
                session, current_checkpoint, copied_event
            )
            completion_record = None
            if steering_key is not None:
                completion_record = prepare_interaction_completion_steering_record(
                    session,
                    current_checkpoint,
                    copied_event,
                    operation_records.get(steering_key),
                    keeps_running=queued or target_status is SessionStatus.RUNNING,
                )
            now = self._ownership_clock()
            settlement_record = None
            settlement_storage_key = None
            if settlement_request is not None:
                active = model_stage_rules._reconstruct_active_model_completion_stage_from_records(
                    operation_records,
                    session_id=session_id,
                )
                if active is None:
                    raise SessionModelCompletionStageConflict(
                        "The interaction transition has no active model-completion stage to settle."
                    )
                stage = active.stage
                settlement_storage_key = (
                    model_stage_rules._model_completion_stage_settlement_storage_key(stage.stage_id)
                )
                model_stage_rules._validate_model_completion_stage_for_settlement(
                    session=session,
                    stage=stage,
                    active=active,
                    request=settlement_request,
                    settlement_record=operation_records.get(settlement_storage_key),
                    winner_exists=(
                        operation_records.get(
                            model_stage_rules._model_completion_stage_winner_storage_key(
                                stage.logical_step_id
                            )
                        )
                        is not None
                    ),
                    receipt_exists=(
                        operation_records.get(
                            publication_contracts._runtime_publication_storage_key(
                                stage.logical_step_id
                            )
                        )
                        is not None
                    ),
                )
                settlement_record = model_stage_rules._model_completion_stage_settlement_record(
                    stage,
                    request=settlement_request,
                    settled_at=now,
                )
            prepared_event_append = self._prepare_event_append_unlocked(
                session,
                [event for event in (copied_event, copied_terminal_event) if event is not None],
            )
            updated_checkpoint = self._checkpoints.get(session_id)
            if not queued and checkpoint_mutation_request is not None:
                updated_checkpoint = (
                    checkpoint_mutation_rules._apply_runtime_publication_checkpoint_mutation(
                        RuntimePublicationMutation.model_validate(checkpoint_mutation_request),
                        updated_checkpoint,
                    )
                )
                if updated_checkpoint is None:
                    raise AssertionError("Interaction checkpoint mutation deleted its checkpoint.")
            updated = session.model_copy(update={"last_activity_at": now})
            if not queued:
                updated = updated.model_copy(
                    update={
                        "status": target_status,
                        "updated_at": now,
                        "last_activity_at": now,
                    }
                )
            receipt_record = interaction_receipts._interaction_transition_receipt_record(
                session=updated,
                event=copied_event,
                from_statuses=allowed_statuses,
                to_status=target_status,
                only_if_no_queued_messages=conditional,
                model_completion_stage_settlement=settlement_request,
                checkpoint_mutation=checkpoint_mutation_request,
                terminal_event=copied_terminal_event,
                terminal_decision=terminal_decision,
                status_changed=not queued,
                invocation_session_instance_id=expected_session_instance_id,
                invocation_active_profile=expected_active_invocation_profile,
                invocation_authority_state=expected_invocation_authority_state,
                recovery_claim_id=expected_recovery_claim_id,
            )
            applied = self._apply_event_append_unlocked(
                session,
                prepared_event_append,
                activity_at=now,
            )
            if not queued:
                applied = applied.model_copy(
                    update={
                        "status": target_status,
                        "updated_at": now,
                        "last_activity_at": now,
                    }
                )
            if applied != updated:
                raise RuntimeError("Prepared interaction transition result changed during commit.")
            self._sessions[session_id] = updated
            if not queued and checkpoint_mutation_request is not None:
                assert updated_checkpoint is not None
                self._checkpoints[session_id] = updated_checkpoint
            if terminal_decision is not None:
                assert settled_checkpoint is not None
                self._store_checkpoint_unlocked(session_id, settled_checkpoint)
            self._refresh_child_lifecycle_candidate_unlocked(updated)
            if settlement_record is not None and settlement_storage_key is not None:
                operation_records[settlement_storage_key] = settlement_record
                del operation_records[MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY]
            operation_records[receipt_storage_key] = receipt_record
            if completion_record is not None:
                assert steering_key is not None
                operation_records[steering_key] = completion_record
            return InteractionTransitionResult(
                session=updated,
                event=copied_event,
                terminal_event=copied_terminal_event,
                status_changed=not queued,
            )

    async def settle_session_invocation(self, command: Any) -> InteractionTransitionResult:
        from cayu.sessions._invocation_lifecycle import (
            SettleInvocationCommand,
            copy_invocation_lifecycle_command,
        )

        copied = copy_invocation_lifecycle_command(command)
        if type(copied) is not SettleInvocationCommand:
            raise TypeError("command must be a SettleInvocationCommand.")
        transition = copied.transition
        kwargs: dict[str, Any] = {
            "event": transition.event,
            "from_statuses": set(transition.from_statuses),
            "to_status": transition.to_status,
            "only_if_no_queued_messages": transition.only_if_no_queued_messages,
            "model_completion_stage_settlement": transition.model_completion_stage_settlement,
            "expected_session_instance_id": copied.expected_session_instance_id,
            "expected_active_invocation_profile": copied.expected_active_profile,
            "expected_invocation_authority_state": copied.expected_authority_state,
        }
        if transition.checkpoint_mutation is not None:
            kwargs["checkpoint_mutation"] = transition.checkpoint_mutation
        if transition.terminal_event is not None:
            kwargs["terminal_event"] = transition.terminal_event
            kwargs["terminal_decision"] = transition.terminal_decision
        if copied.recovery_claim_id is not None:
            kwargs["expected_recovery_claim_id"] = copied.recovery_claim_id
        if copied.terminalization_only:
            kwargs["terminalization_only"] = True
            kwargs["terminalization_plan_ownership"] = copied.terminalization_plan_ownership
        return await self.publish_interaction_transition(copied.session_id, **kwargs)

    async def load_interaction_transition_receipt(
        self,
        session_id: str,
        *,
        transition: InteractionTransitionSpec,
        expected_recovery_claim_id: str | None = None,
    ) -> InteractionTransitionReceiptResult | None:
        session_id, copied_transition = (
            interaction_rules._prepare_interaction_transition_receipt_lookup(
                session_id,
                transition=transition,
            )
        )
        copied_event = copied_transition.event
        receipt_storage_key = interaction_receipts._interaction_transition_storage_key(
            copied_event.id
        )
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            receipt_record = self._session_operation_records.get(session_id, {}).get(
                receipt_storage_key
            )
            existing = self._event_records_by_id.get((session_id, copied_event.id))
            if receipt_record is None:
                if existing is not None:
                    raise RuntimeError(
                        "Interaction transition event exists without its immutable receipt."
                    )
                return None
            receipt = interaction_receipts._reconstruct_interaction_transition_receipt(
                receipt_record,
                transition=copied_transition,
            )
            interaction_receipts._validate_interaction_transition_receipt_recovery_authority(
                receipt,
                current_checkpoint=self._checkpoints.get(session_id),
                expected_recovery_claim_id=expected_recovery_claim_id,
            )
            if existing is not None and existing.event != receipt.event:
                raise RuntimeError(
                    "Interaction transition receipt conflicts with retained event history."
                )
            if receipt.terminal_event is not None:
                retained_terminal = self._event_records_by_id.get(
                    (session_id, receipt.terminal_event.id)
                )
                if (
                    retained_terminal is not None
                    and retained_terminal.event != receipt.terminal_event
                ):
                    raise RuntimeError(
                        "Interaction transition receipt conflicts with retained terminal history."
                    )
            return InteractionTransitionReceiptResult(
                session=receipt.session,
                transition=interaction_receipts._interaction_transition_spec_from_receipt(receipt),
                status_changed=receipt.status_changed,
            )

    async def _load_interaction_transition_receipt_by_event_id(
        self,
        session_id: str,
        *,
        event_id: str,
        expected_session_instance_id: str,
        expected_active_invocation_profile: ActiveInvocationExecutionProfile,
    ) -> InteractionTransitionReceiptResult | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        event_id = require_clean_nonblank(event_id, "event_id")
        receipt_storage_key = interaction_receipts._interaction_transition_storage_key(event_id)
        async with self._lock:
            current_session = self._sessions.get(session_id)
            if current_session is None:
                raise KeyError(f"Session not found: {session_id}")
            record = self._session_operation_records.get(session_id, {}).get(receipt_storage_key)
            existing = self._event_records_by_id.get((session_id, event_id))
            if record is None:
                if existing is not None:
                    raise RuntimeError(
                        "Interaction transition event exists without its immutable receipt."
                    )
                return None
            receipt = interaction_receipts._load_interaction_transition_receipt(record)
            interaction_receipts._validate_invocation_release_settlement_receipt_authority(
                receipt,
                current_session=current_session,
                expected_session_instance_id=expected_session_instance_id,
                expected_active_invocation_profile=expected_active_invocation_profile,
            )
            if receipt.event.id != event_id:
                raise RuntimeError("Interaction transition receipt has a conflicting event ID.")
            if existing is not None and existing.event != receipt.event:
                raise RuntimeError(
                    "Interaction transition receipt conflicts with retained event history."
                )
            return InteractionTransitionReceiptResult(
                session=receipt.session,
                transition=interaction_receipts._interaction_transition_spec_from_receipt(receipt),
                status_changed=receipt.status_changed,
            )

    async def fence_stalled_run(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_for_seconds: int,
    ) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = session_record_rules._validate_status_set(statuses, "statuses")
        validated_inactive_for_seconds = _validate_inactive_for_seconds(inactive_for_seconds)
        assert validated_inactive_for_seconds is not None
        async with self._lock:
            now = self._ownership_clock()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if self._has_live_execution_owner_unlocked(session, now):
                return None
            current_checkpoint = self._checkpoints.get(session_id)
            if (
                active_provider_operation_cancellation_claim_from_checkpoint(
                    current_checkpoint,
                    now=now,
                )
                is not None
                or session_recovery_rules._incomplete_recovery_claim_from_checkpoint(
                    current_checkpoint
                )
                is not None
            ):
                return None
            inactive_before = utc_duration_cutoff(now, validated_inactive_for_seconds)
            if (
                inactive_before is None
                or session.status not in allowed_statuses
                or session.last_activity_at > inactive_before
            ):
                return None
            fenced = session.model_copy(
                update={
                    "run_epoch": session.run_epoch + 1,
                    "last_activity_at": now,
                }
            )
            self._sessions[session_id] = fenced
            result = fenced.model_copy(deep=True)
            invocation_scope._activate_session_run_fence(result)
            return result

    async def reserve_stalled_run_recovery(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_for_seconds: int | None,
        checkpoint_transform: StoreTimeCheckpointTransform,
    ) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = session_record_rules._validate_status_set(statuses, "statuses")
        inactive_for_seconds = _validate_inactive_for_seconds(inactive_for_seconds)
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            now = self._ownership_clock()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if inactive_for_seconds is not None and self._has_live_execution_owner_unlocked(
                session, now
            ):
                return None
            current = self._checkpoints.get(session_id)
            inactive_before = (
                None
                if inactive_for_seconds is None
                else utc_duration_cutoff(now, inactive_for_seconds)
            )
            if (
                session.status not in allowed_statuses
                or (
                    inactive_for_seconds is not None
                    and (inactive_before is None or session.last_activity_at > inactive_before)
                )
                or active_provider_operation_cancellation_claim_from_checkpoint(
                    current,
                    now=now,
                )
                is not None
            ):
                return None
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                checkpoint_preservation._copy_checkpoint_for_transform(
                    current, session_id=session_id
                ),
                now,
            )
            if transformed is None:
                return None
            self._store_checkpoint_unlocked(
                session_id,
                checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                    current,
                    transformed,
                    session_id=session_id,
                ),
            )
            return session.model_copy(deep=True)

    async def fence_run_and_transform_checkpoint(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        checkpoint_transform: CheckpointTransform,
        result_checkpoint_transform: CheckpointTransform | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = session_record_rules._validate_status_set(statuses, "statuses")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        if result_checkpoint_transform is not None and not callable(result_checkpoint_transform):
            raise TypeError("result_checkpoint_transform must be callable.")
        async with self._lock:
            now = self._ownership_clock()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(f"Session status cannot be fenced: {session.status}")
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            current = self._checkpoints.get(session_id)
            session_recovery_rules._require_live_incomplete_recovery_claim_for_run_epoch_transfer(
                current,
                now=now,
            )
            if (
                active_provider_operation_cancellation_claim_from_checkpoint(
                    current,
                    now=now,
                )
                is not None
            ):
                raise SessionStatusConflict(
                    "Provider-operation cancellation still owns the session run epoch."
                )
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                checkpoint_preservation._copy_checkpoint_for_transform(
                    current, session_id=session_id
                ),
            )
            if transformed is None:
                raise ValueError("Fenced checkpoint transform must return a checkpoint.")
            transformed = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                current,
                transformed,
                session_id=session_id,
            )
            fenced = session.model_copy(
                update={
                    "run_epoch": session.run_epoch + 1,
                    "last_activity_at": now,
                }
            )
            if result_checkpoint_transform is not None:
                result_checkpoint = result_checkpoint_transform(
                    fenced.model_copy(deep=True),
                    checkpoint_preservation._copy_checkpoint_for_transform(
                        transformed,
                        session_id=session_id,
                    ),
                )
                if result_checkpoint is None:
                    raise ValueError("Result checkpoint transform must return a checkpoint.")
                transformed = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                    transformed,
                    result_checkpoint,
                    session_id=session_id,
                )
            self._store_checkpoint_unlocked(session_id, transformed)
            self._sessions[session_id] = fenced
            result = fenced.model_copy(deep=True)
            invocation_scope._activate_session_run_fence(result)
            return result

    async def release_run_fence(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        expected_run_epoch = invocation_scope._current_session_run_epoch(session_id)
        if expected_run_epoch is None:
            invocation_scope._deactivate_session_interaction(session_id)
            return
        try:
            async with self._lock:
                session = self._sessions.get(session_id)
                if session is not None and session.run_epoch == expected_run_epoch:
                    self._sessions[session_id] = session.model_copy(
                        update={"run_epoch": session.run_epoch + 1}
                    )
        finally:
            invocation_scope._deactivate_session_run_fence(session_id)
            invocation_scope._deactivate_session_interaction(session_id)

    async def release_session_invocation(self, command: Any) -> Any:
        from cayu.runtime._invocation_lifecycle import (
            checkpoint_with_invocation_lifecycle_receipt,
        )
        from cayu.sessions._invocation_lifecycle import (
            InvocationReleaseResult,
            ReleaseInvocationCommand,
            _invocation_lifecycle_receipt_ledger_from_checkpoint,
            copy_invocation_lifecycle_command,
            invocation_release_replay_from_state,
            require_invocation_command_authority,
            require_invocation_release_store_authority,
        )

        copied = copy_invocation_lifecycle_command(command)
        if type(copied) is not ReleaseInvocationCommand:
            raise TypeError("command must be a ReleaseInvocationCommand.")
        require_invocation_release_store_authority(copied)
        released = False
        try:
            async with self._lock:
                session = self._sessions.get(copied.session_id)
                if session is None:
                    raise KeyError(f"Session not found: {copied.session_id}")
                checkpoint = self._checkpoints.get(copied.session_id)
                ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
                replay = invocation_release_replay_from_state(
                    session,
                    checkpoint,
                    copied,
                    _ledger=ledger,
                )
                if replay is not None:
                    released = True
                    return replay
                if copied.terminal_session_event is not None:
                    terminal_receipt = self._session_operation_records.get(
                        copied.session_id,
                        {},
                    ).get(
                        interaction_receipts._invocation_terminal_event_storage_key(
                            copied.terminal_session_event.id
                        )
                    )
                    retained_terminal_event = self._event_records_by_id.get(
                        (copied.session_id, copied.terminal_session_event.id)
                    )
                    interaction_receipts._require_invocation_release_terminal_session_event(
                        terminal_receipt,
                        (
                            None
                            if retained_terminal_event is None
                            else retained_terminal_event.event
                        ),
                        current_session=session,
                        expected_event=copied.terminal_session_event,
                        expected_session_instance_id=copied.expected_session_instance_id,
                        expected_active_invocation_profile=copied.expected_active_profile,
                    )
                elif copied.settlement_transition is None:
                    assert copied.recovery_claim_id is not None
                    interaction_receipts._require_invocation_release_recovery_claim(
                        checkpoint,
                        current_session=session,
                        recovery_claim_id=copied.recovery_claim_id,
                    )
                else:
                    settlement_record = self._session_operation_records.get(
                        copied.session_id,
                        {},
                    ).get(
                        interaction_receipts._interaction_transition_storage_key(
                            copied.settlement_transition.event.id
                        )
                    )
                    if settlement_record is None:
                        raise SessionRunFenced(
                            "Invocation release lacks exact durable terminal settlement."
                        )
                    interaction_receipts._require_invocation_release_settlement_record(
                        settlement_record,
                        current_session=session,
                        transition=copied.settlement_transition,
                        expected_session_instance_id=copied.expected_session_instance_id,
                        expected_active_invocation_profile=copied.expected_active_profile,
                    )
                require_invocation_command_authority(
                    session,
                    checkpoint,
                    session_id=copied.session_id,
                    session_instance_id=copied.expected_session_instance_id,
                    run_epochs=frozenset({copied.expected_run_epoch}),
                    active_profile=copied.expected_active_profile,
                )
                session = session.model_copy(update={"run_epoch": copied.expected_run_epoch + 1})
                updated_checkpoint = checkpoint_with_invocation_lifecycle_receipt(
                    checkpoint,
                    copied,
                    active_profile=copied.expected_active_profile,
                    result_session=session,
                    _ledger=ledger,
                )
                self._store_checkpoint_unlocked(copied.session_id, updated_checkpoint)
                self._sessions[copied.session_id] = session
                result = InvocationReleaseResult(
                    session=session,
                    active_profile=copied.expected_active_profile,
                    replayed=False,
                )
                released = True
                return result
        finally:
            if released and (
                invocation_scope._current_session_run_epoch(copied.session_id)
                == copied.expected_run_epoch
            ):
                invocation_scope._deactivate_session_run_fence(copied.session_id)
                invocation_scope._deactivate_session_interaction(copied.session_id)

    async def append_event(self, session_id: str, event: Event) -> None:
        await self.append_events(session_id, [event])

    async def claim_budget_reservation_identity(
        self,
        *,
        reservation_id: str,
        publication_session_id: str,
        publication_id: str,
    ) -> None:
        reservation_id = require_clean_nonblank(reservation_id, "reservation_id")
        publication_session_id = require_clean_nonblank(
            publication_session_id,
            "publication_session_id",
        )
        publication_id = require_clean_nonblank(publication_id, "publication_id")
        async with self._lock:
            session = self._sessions.get(publication_session_id)
            if session is None:
                raise KeyError(f"Session not found: {publication_session_id}")
            invocation_scope._assert_session_run_epoch(publication_session_id, session)
            existing = self._budget_reservation_identities.get(reservation_id)
            if existing is None:
                for owner in self._session_closure_progress.values():
                    closure_records._check_closure_lineage_owner(owner, (publication_session_id,))
                self._budget_reservation_identities[reservation_id] = (
                    publication_session_id,
                    publication_id,
                    False,
                )
                return
            if existing[:2] != (publication_session_id, publication_id):
                raise BudgetReservationIdentityConflict(
                    "Budget ledger reused a reservation identity."
                )

    def _prepare_event_append_unlocked(
        self,
        session: Session,
        events: Iterable[Event],
        *,
        events_are_detached: bool = False,
    ) -> _PreparedInMemoryEventAppend:
        from cayu.sessions.pending_actions import (
            pending_action_event_has_unknown_round_call,
            pending_action_event_is_from_different_tool_round,
            pending_action_event_lookup_id,
            pending_action_event_retains_history,
            pending_action_evidence_round_from_checkpoint,
            pending_action_lookup_key,
            project_pending_action_event_record,
        )

        session_id = session.id
        existing_ids = self._event_ids.get(session_id, set())
        event_batch = tuple(events)
        batch_ids: set[str] = set()
        budget_reservation_publications: dict[str, str] = {}
        for event in event_batch:
            if event.id in existing_ids or event.id in batch_ids:
                raise ValueError(f"Event already exists for session {session_id}: {event.id}")
            batch_ids.add(event.id)
            if event.type != EventType.BUDGET_RESERVED:
                continue
            raw_reservation_id = event.payload.get("reservation_id")
            if type(raw_reservation_id) is not str:
                continue
            reservation_id = raw_reservation_id
            existing = self._budget_reservation_identities.get(reservation_id)
            if existing is not None and (
                existing[:2] != (event.session_id, event.id) or existing[2]
            ):
                raise BudgetReservationIdentityConflict(
                    "Budget ledger reused a reservation identity."
                )
            previous_publication_id = budget_reservation_publications.setdefault(
                reservation_id,
                event.id,
            )
            if previous_publication_id != event.id:
                raise BudgetReservationIdentityConflict(
                    "Budget ledger reused a reservation identity."
                )

        try:
            pending_round = pending_action_evidence_round_from_checkpoint(
                self._checkpoints.get(session_id)
            )
        except (TypeError, ValueError):
            pending_round = None
        index_scope = self._pending_action_index_scopes.get(session_id)
        active_lookup_keys = frozenset() if index_scope is None else index_scope[0]
        round_lookup_key = (
            None
            if pending_round is None
            else pending_action_lookup_key(pending_round.tool_round_id)
        )
        prepared: list[_PreparedInMemoryEvent] = []
        next_sequence = self._next_event_sequence
        for event in event_batch:
            stored_event = event if events_are_detached else event.model_copy(deep=True)
            record = EventRecord(sequence=next_sequence, event=stored_event)
            self._cost_event_index.validate(record)
            event_type = str(stored_event.type)
            projected_record: EventRecord | None = None
            retention_keys: list[str] = []
            history_keys: list[str] = []
            if event_type in PENDING_ACTION_EVENT_TYPE_VALUES:
                retain_for_scope = (
                    pending_round is not None
                    and pending_action_event_has_unknown_round_call(
                        stored_event,
                        pending_round,
                    )
                )
                lookup_id = pending_action_event_lookup_id(stored_event)
                if lookup_id is not None:
                    lookup_key = pending_action_lookup_key(lookup_id)
                    if lookup_key in active_lookup_keys:
                        history_keys.append(lookup_key)
                        if not (
                            pending_action_event_retains_history(event_type)
                            and (
                                pending_round is None
                                or pending_action_event_is_from_different_tool_round(
                                    stored_event,
                                    pending_round,
                                )
                            )
                        ):
                            retention_keys.append(lookup_key)
                if retain_for_scope:
                    assert round_lookup_key is not None
                    retention_keys.append(round_lookup_key)
                    history_keys.append(round_lookup_key)
                if (
                    event_type in PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES
                    or retention_keys
                    or history_keys
                ):
                    projected_record = project_pending_action_event_record(record)
            delivery = (
                None
                if event_type == str(EventType.RUNTIME_SINK_FAILED)
                else PersistedEventSideEffectDelivery(
                    session_id=session_id,
                    event_id=stored_event.id,
                    event_sequence=record.sequence,
                    status=PersistedEventSideEffectStatus.PENDING,
                )
            )
            prepared.append(
                _PreparedInMemoryEvent(
                    record=record,
                    event_type=event_type,
                    projected_record=projected_record,
                    retained_lookup_keys=tuple(dict.fromkeys(retention_keys)),
                    history_lookup_keys=tuple(dict.fromkeys(history_keys)),
                    delivery=delivery,
                    invocation_terminal_receipt=(
                        interaction_receipts._invocation_terminal_event_receipt_record(
                            session=session,
                            checkpoint=self._checkpoints.get(session_id),
                            event=stored_event,
                        )
                    ),
                )
            )
            next_sequence += 1
        return _PreparedInMemoryEventAppend(
            events=tuple(prepared),
            next_sequence=next_sequence,
            budget_reservation_publications=tuple(budget_reservation_publications.items()),
        )

    def _apply_event_append_unlocked(
        self,
        session: Session,
        prepared: _PreparedInMemoryEventAppend,
        *,
        activity_at: datetime | None = None,
    ) -> Session:
        from cayu.approvals.user_input import USER_INPUT_SUPERSESSION_INTENT_KEY
        from cayu.sessions.pending_actions import (
            pending_action_lookup_key,
            retain_pending_action_index_record,
        )

        session_id = session.id
        existing_ids = self._event_ids[session_id]
        session_records = self._session_event_records.setdefault(session_id, [])
        from cayu.sessions.interactions import INTERACTION_LIFECYCLE_EVENT_TYPES

        lifecycle_types = {str(event_type) for event_type in INTERACTION_LIFECYCLE_EVENT_TYPES}
        child_lifecycle_changed = False
        for prepared_event in prepared.events:
            record = prepared_event.record
            event_type = prepared_event.event_type
            projected_record = prepared_event.projected_record
            retained_lookup_keys = prepared_event.retained_lookup_keys
            history_lookup_keys = prepared_event.history_lookup_keys
            stored_event = record.event
            self._events[session_id].append(stored_event)
            self._event_records.append(record)
            self._event_records_by_id[(session_id, stored_event.id)] = record
            if stored_event.type is EventType.SESSION_MESSAGE_QUEUED:
                queue_id = stored_event.payload.get("queue_id")
                if type(queue_id) is str and len(queue_id) <= 512:
                    ids = self._session_message_acceptance_event_ids.setdefault(
                        (session_id, queue_id), set()
                    )
                    if len(ids) < 2:
                        ids.add(stored_event.id)
            self._cost_event_index.append(record)
            session_records.append(record)
            if session.parent_session_id is not None and stored_event.type in {
                EventType.SESSION_STARTED,
                EventType.SESSION_RESUMED,
                EventType.SESSION_FORKED,
                EventType.SESSION_COMPLETED,
                EventType.SESSION_FAILED,
                EventType.SESSION_INTERRUPTED,
            }:
                lifecycle_event_type = EventType(stored_event.type)
                self._child_lifecycle_event_records.setdefault(session_id, {})[
                    lifecycle_event_type
                ] = record
                child_lifecycle_changed = True
            if stored_event.type is EventType.SESSION_INTERRUPTED:
                supersession = stored_event.payload.get(USER_INPUT_SUPERSESSION_INTENT_KEY)
                superseded_input_id = (
                    None if type(supersession) is not dict else supersession.get("input_id")
                )
                if type(superseded_input_id) is str and superseded_input_id.strip():
                    lookup_key = pending_action_lookup_key(superseded_input_id)
                    indexed_supersessions = self._user_input_supersession_event_records.setdefault(
                        (session_id, lookup_key),
                        [],
                    )
                    if len(indexed_supersessions) < 2:
                        indexed_supersessions.append(record)
            interaction_id = stored_event.interaction_id
            if interaction_id is not None:
                self._register_private_authority_alias_unlocked(
                    interaction_id,
                    field_name="interaction_id",
                    scope_session_id=session_id,
                )
                self._interaction_event_records.setdefault(session_id, {}).setdefault(
                    interaction_id,
                    [],
                ).append(record)
            if stored_event.type == EventType.TURN_COMPLETED:
                nested_interaction_ids = stored_event.payload.get("interaction_ids")
                if type(nested_interaction_ids) is list:
                    for nested_interaction_id in dict.fromkeys(
                        value
                        for value in nested_interaction_ids
                        if type(value) is str and value.strip()
                    ):
                        self._register_private_authority_alias_unlocked(
                            nested_interaction_id,
                            field_name="interaction_id",
                            scope_session_id=session_id,
                        )
            if interaction_id is not None and event_type in lifecycle_types:
                latest_by_id = self._latest_interaction_event_records.setdefault(session_id, {})
                latest_by_sequence = self._latest_interaction_event_records_by_sequence.setdefault(
                    session_id, []
                )
                previous = latest_by_id.get(interaction_id)
                if previous is not None:
                    previous_index = bisect_left(
                        latest_by_sequence,
                        previous.sequence,
                        key=lambda candidate: candidate.sequence,
                    )
                    if (
                        previous_index >= len(latest_by_sequence)
                        or latest_by_sequence[previous_index].sequence != previous.sequence
                    ):
                        raise RuntimeError(
                            "In-memory interaction latest-event indexes are inconsistent."
                        )
                    latest_by_sequence.pop(previous_index)
                latest_by_id[interaction_id] = record
                latest_by_sequence.append(record)
            if projected_record is not None:
                if event_type in PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES:
                    self._pending_action_latest_barrier_records[session_id] = projected_record
                else:
                    by_lookup_id = self._pending_action_event_records.setdefault(session_id, {})
                    event_history = self._pending_action_event_history.setdefault(
                        session_id,
                        {},
                    )
                    for retention_key in retained_lookup_keys:
                        retain_pending_action_index_record(
                            by_lookup_id.setdefault(retention_key, {}),
                            projected_record,
                        )
                    for history_key in history_lookup_keys:
                        event_history.setdefault(history_key, []).append(record)
            self._type_event_records.setdefault(event_type, []).append(record)
            workflow_name = stored_event.workflow_name
            workflow_attempt_id = stored_event.payload.get("attempt_id")
            workflow_step_id = stored_event.payload.get("step_id")
            if (
                stored_event.type == WORKFLOW_ATTEMPT_EVENT_TYPE
                and isinstance(workflow_name, str)
                and workflow_name
                and isinstance(workflow_attempt_id, str)
                and workflow_attempt_id
            ):
                self._workflow_attempt_event_records.setdefault(
                    (session_id, workflow_name),
                    [],
                ).append(record)
                self._workflow_active_attempt_ids[(session_id, workflow_name)] = workflow_attempt_id
            if (
                isinstance(workflow_name, str)
                and workflow_name
                and isinstance(workflow_attempt_id, str)
                and workflow_attempt_id
                and isinstance(workflow_step_id, str)
                and workflow_step_id
            ):
                self._workflow_step_event_records.setdefault(
                    (
                        session_id,
                        workflow_name,
                        workflow_attempt_id,
                        workflow_step_id,
                        event_type,
                    ),
                    [],
                ).append(record)
                if (
                    self._workflow_active_attempt_ids.get((session_id, workflow_name))
                    == workflow_attempt_id
                ):
                    self._workflow_fenced_step_event_records.setdefault(
                        (
                            session_id,
                            workflow_name,
                            workflow_step_id,
                            event_type,
                        ),
                        [],
                    ).append(record)
            existing_ids.add(stored_event.id)
            if prepared_event.invocation_terminal_receipt is not None:
                receipt_key, receipt_record = prepared_event.invocation_terminal_receipt
                self._session_operation_records[session_id][receipt_key] = receipt_record
            if prepared_event.delivery is not None:
                self._persisted_event_side_effect_deliveries[(session_id, stored_event.id)] = (
                    prepared_event.delivery
                )
        self._budget_reservation_identities.update(
            {
                reservation_id: (session_id, publication_id, True)
                for reservation_id, publication_id in (prepared.budget_reservation_publications)
            }
        )
        self._next_event_sequence = prepared.next_sequence
        self._refresh_queued_dispatch_terminal_receipts_unlocked(
            session_id,
            self._checkpoints.get(session_id),
        )
        if not prepared.events:
            return session
        updated = session.model_copy(
            update={"last_activity_at": activity_at or self._ownership_clock()}
        )
        if child_lifecycle_changed:
            self._refresh_child_lifecycle_candidate_unlocked(updated)
        return updated

    def _append_events_unlocked(
        self,
        session: Session,
        events: list[Event],
    ) -> Session:
        prepared = self._prepare_event_append_unlocked(session, events)
        return self._apply_event_append_unlocked(session, prepared)

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        session_id, copied_events = session_event_rules._copy_session_event_batch(
            session_id, events
        )

        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            if copied_events:
                for owner in self._session_closure_progress.values():
                    closure_records._check_closure_lineage_owner(owner, (session_id,))
            self._sessions[session_id] = self._append_events_unlocked(session, copied_events)

    async def append_tool_effect_conflict(self, request: object) -> Event:
        from cayu.runtime._tool_effect_conflicts import (
            copy_tool_effect_conflict_audit,
            reconcile_tool_effect_conflict_event,
        )

        audit = copy_tool_effect_conflict_audit(request)
        session_id = audit.executing.intent.session_id
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("Tool effect audit session is unavailable.")
            current = self._session_operation_records.get(session_id, {}).get(audit.storage_key)
            event = audit.prepare_event(session, current, now=self._ownership_clock())
            existing = self._event_records_by_id.get((session_id, event.id))
            if existing is not None:
                return reconcile_tool_effect_conflict_event(event, existing.event)
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            prepared = self._prepare_event_append_unlocked(session, [event])
            self._sessions[session_id] = self._apply_event_append_unlocked(
                session, prepared, activity_at=session.last_activity_at
            )
            return copy_event(event)

    async def append_workflow_step_started(
        self,
        session_id: str,
        event: Event,
        *,
        workflow_name: str,
        attempt_id: str,
    ) -> bool:
        session_id, copied_event, workflow_name, attempt_id = _copy_workflow_step_reservation(
            session_id,
            event,
            workflow_name=workflow_name,
            attempt_id=attempt_id,
        )
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            if self._workflow_active_attempt_ids.get((session_id, workflow_name)) != attempt_id:
                return False
            if (session_id, copied_event.id) in self._event_records_by_id:
                return False
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            self._sessions[session_id] = self._append_events_unlocked(session, [copied_event])
            return True

    async def load_mcp_manifest_baselines(
        self,
        history_keys: tuple[str, ...],
    ) -> McpManifestBaselineLoadResult:
        keys = _validate_mcp_manifest_history_keys(history_keys)
        async with self._lock:
            return McpManifestBaselineLoadResult(
                baselines={
                    key: self._mcp_manifest_baselines[key].model_copy(deep=True)
                    for key in keys
                    if key in self._mcp_manifest_baselines
                },
            )

    async def compare_and_publish_mcp_manifest_checks(
        self,
        session_id: str,
        *,
        expected_generations: dict[str, int | None],
        baseline_updates: dict[str, McpManifestBaseline],
        events: list[Event],
    ) -> McpManifestPublicationResult:
        (
            session_id,
            expected,
            updates,
            copied_events,
        ) = _copy_mcp_manifest_publication(
            session_id,
            expected_generations=expected_generations,
            baseline_updates=baseline_updates,
            events=events,
        )

        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            current = {
                key: self._mcp_manifest_baselines[key]
                for key in expected
                if key in self._mcp_manifest_baselines
            }
            if any(
                expected_generation
                != (None if (baseline := current.get(key)) is None else baseline.generation)
                for key, expected_generation in expected.items()
            ):
                return McpManifestPublicationResult(
                    published=False,
                    baselines={
                        key: baseline.model_copy(deep=True) for key, baseline in current.items()
                    },
                )
            _validate_mcp_manifest_publication_state(
                expected_generations=expected,
                current_baselines=current,
                baseline_updates=updates,
                events=copied_events,
            )
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            self._sessions[session_id] = self._append_events_unlocked(session, copied_events)
            for key, baseline in updates.items():
                self._mcp_manifest_baselines[key] = baseline.model_copy(deep=True)
            return McpManifestPublicationResult(
                published=True,
                baselines={
                    key: self._mcp_manifest_baselines[key].model_copy(deep=True)
                    for key in expected
                    if key in self._mcp_manifest_baselines
                },
            )

    async def claim_first_persisted_event_side_effect(
        self, expected: PersistedEventSideEffectDelivery
    ) -> PersistedEventSideEffectClaim | None:
        expected = session_event_rules._copy_pending_first_event_delivery(expected)
        return await self._claim_persisted_event_side_effect(
            session_id=expected.session_id, event_id=expected.event_id, expected=expected
        )

    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectClaim | None:
        return await self._claim_persisted_event_side_effect(
            session_id=session_id, event_id=event_id, lease_seconds=lease_seconds
        )

    async def _claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
        expected: PersistedEventSideEffectDelivery | None = None,
    ) -> PersistedEventSideEffectClaim | None:
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        if event_id is not None:
            event_id = require_clean_nonblank(event_id, "event_id")
        if (session_id is None) != (event_id is None):
            raise ValueError("session_id and event_id must be supplied together.")
        if type(lease_seconds) not in {int, float} or lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0.")
        async with self._lock:
            now = self._ownership_clock()
            deliveries = sorted(
                self._persisted_event_side_effect_deliveries.values(),
                key=lambda delivery: delivery.event_sequence,
            )
            closure_targets = {
                target
                for owner in self._session_closure_progress.values()
                for target in closure_records._closure_progress_targets(owner)
            }
            for delivery in deliveries:
                if expected is not None and delivery != expected:
                    continue
                if delivery.session_id in closure_targets:
                    continue
                if session_id is not None and (
                    delivery.session_id != session_id or delivery.event_id != event_id
                ):
                    continue
                if delivery.status in {
                    PersistedEventSideEffectStatus.DELIVERED,
                    PersistedEventSideEffectStatus.DEAD_LETTERED,
                }:
                    continue
                if (
                    delivery.status is PersistedEventSideEffectStatus.LEASED
                    and delivery.lease_expires_at is not None
                    and delivery.lease_expires_at > now
                ):
                    continue
                if (
                    delivery.status is PersistedEventSideEffectStatus.FAILED
                    and delivery.next_attempt_at is not None
                    and delivery.next_attempt_at > now
                ):
                    continue
                record = self._event_records_by_id.get((delivery.session_id, delivery.event_id))
                if record is None:
                    raise RuntimeError("Persisted side-effect delivery lost its source event.")
                claim = PersistedEventSideEffectClaim(
                    session_id=delivery.session_id,
                    event_id=delivery.event_id,
                    event_sequence=delivery.event_sequence,
                    event=record.event,
                    attempt=delivery.attempts + 1,
                    lease_expires_at=now + timedelta(seconds=float(lease_seconds)),
                )
                self._persisted_event_side_effect_deliveries[
                    (delivery.session_id, delivery.event_id)
                ] = delivery.model_copy(
                    update={
                        "status": PersistedEventSideEffectStatus.LEASED,
                        "attempts": claim.attempt,
                        "claim_id": claim.claim_id,
                        "lease_expires_at": claim.lease_expires_at,
                        "next_attempt_at": None,
                        "last_error": None,
                        "updated_at": now,
                    },
                    deep=True,
                )
                return claim
            return None

    async def get_persisted_event_side_effect_delivery(
        self,
        *,
        session_id: str,
        event_id: str,
    ) -> PersistedEventSideEffectDelivery | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        event_id = require_clean_nonblank(event_id, "event_id")
        async with self._lock:
            delivery = self._persisted_event_side_effect_deliveries.get((session_id, event_id))
            return None if delivery is None else delivery.model_copy(deep=True)

    async def retire_failed_first_event_delivery(
        self, expected: PersistedEventSideEffectDelivery
    ) -> PersistedEventSideEffectDelivery | None:
        expected = session_event_rules._copy_failed_first_delivery_retirement(expected)
        key = (expected.session_id, expected.event_id)
        async with self._lock:
            if self._persisted_event_side_effect_deliveries.get(key) != expected:
                return None
            retired = expected.model_copy(
                update={
                    "status": PersistedEventSideEffectStatus.DEAD_LETTERED,
                    "next_attempt_at": None,
                    "updated_at": self._ownership_clock(),
                }
            )
            self._persisted_event_side_effect_deliveries[key] = retired
            return retired.model_copy(deep=True)

    async def mark_persisted_event_side_effect_delivered(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        async with self._lock:
            delivery = self._matching_persisted_event_side_effect_claim_unlocked(claim)
            updated = delivery.model_copy(
                update={
                    "status": PersistedEventSideEffectStatus.DELIVERED,
                    "claim_id": None,
                    "lease_expires_at": None,
                    "next_attempt_at": None,
                    "last_error": None,
                    "updated_at": self._ownership_clock(),
                },
                deep=True,
            )
            self._persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = (
                updated
            )
            return updated.model_copy(deep=True)

    async def mark_persisted_event_side_effect_failed(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        error: str,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        error = validate_persisted_event_side_effect_error(error)
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be an integer greater than or equal to 1.")
        if (
            type(retry_delay_seconds) not in {int, float}
            or not math.isfinite(retry_delay_seconds)
            or retry_delay_seconds < 0
        ):
            raise ValueError("retry_delay_seconds must be a finite non-negative number.")
        dead_lettered = claim.attempt >= max_attempts
        async with self._lock:
            now = self._ownership_clock()
            delivery = self._matching_persisted_event_side_effect_claim_unlocked(claim)
            updated = delivery.model_copy(
                update={
                    "status": (
                        PersistedEventSideEffectStatus.DEAD_LETTERED
                        if dead_lettered
                        else PersistedEventSideEffectStatus.FAILED
                    ),
                    "claim_id": None,
                    "lease_expires_at": None,
                    "next_attempt_at": (
                        None
                        if dead_lettered
                        else now + timedelta(seconds=float(retry_delay_seconds))
                    ),
                    "last_error": error,
                    "updated_at": now,
                },
                deep=True,
            )
            self._persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = (
                updated
            )
            return updated.model_copy(deep=True)

    async def defer_persisted_event_side_effect(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        async with self._lock:
            delivery = self._matching_persisted_event_side_effect_claim_unlocked(claim)
            updated = delivery.model_copy(
                update={
                    "status": PersistedEventSideEffectStatus.PENDING,
                    "attempts": delivery.attempts - 1,
                    "claim_id": None,
                    "lease_expires_at": None,
                    "next_attempt_at": None,
                    "last_error": None,
                    "updated_at": self._ownership_clock(),
                },
                deep=True,
            )
            self._persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = (
                updated
            )
            return updated.model_copy(deep=True)

    async def get_persisted_event_side_effect_health(self) -> PersistedEventSideEffectHealth:
        async with self._lock:
            now = self._ownership_clock().astimezone(UTC)
            rows = list(self._persisted_event_side_effect_deliveries.values())
            counts = dict.fromkeys(side_effect_health.CONDITIONS, 0)
            timestamps: dict[str, datetime | None] = dict.fromkeys(
                [
                    "oldest_claimable_at",
                    "oldest_pending_at",
                    "oldest_failed_at",
                    "oldest_dead_letter_at",
                    "earliest_live_lease_expires_at",
                ]
            )
            max_attempts = 0

            def oldest(key: str, value: datetime) -> None:
                previous = timestamps[key]
                timestamps[key] = value if previous is None else min(previous, value)

            for row in rows:
                if row.status == "leased":
                    counts[
                        "leased_expired"
                        if side_effect_health.claimable(row, now)
                        else "leased_live"
                    ] += 1
                    if row.lease_expires_at is not None and row.lease_expires_at > now:
                        oldest("earliest_live_lease_expires_at", row.lease_expires_at)
                elif row.status == "failed":
                    counts["failed_retryable"] += 1
                    counts["failed_deferred"] += int(
                        row.next_attempt_at is not None and row.next_attempt_at > now
                    )
                    counts["repeatedly_failing"] += int(row.attempts > 1)
                    oldest("oldest_failed_at", row.updated_at)
                else:
                    counts[str(row.status)] += 1
                if row.status == "pending":
                    oldest("oldest_pending_at", row.updated_at)
                if row.status == "dead_lettered":
                    oldest("oldest_dead_letter_at", row.updated_at)
                if row.status != "delivered":
                    counts["outstanding_total"] += 1
                    max_attempts = max(max_attempts, row.attempts)
                if row.status in {"pending", "failed", "leased"}:
                    live = (
                        row.status == "leased"
                        and row.lease_expires_at is not None
                        and row.lease_expires_at > now
                    )
                    threshold = side_effect_health.PERSISTED_EVENT_SIDE_EFFECT_MAX_ATTEMPTS - (
                        0 if live else 1
                    )
                    counts["final_attempt_boundary"] += int(row.attempts >= threshold)
                if side_effect_health.claimable(row, now):
                    counts["claimable_total"] += 1
                    eligible = row.updated_at
                    if row.status == "leased" and row.lease_expires_at is not None:
                        eligible = row.lease_expires_at
                    elif row.status == "failed" and row.next_attempt_at is not None:
                        eligible = max(eligible, row.next_attempt_at)
                    oldest("oldest_claimable_at", eligible)
            return side_effect_health.finish_health(
                {**counts, **timestamps, "max_outstanding_attempts": max_attempts},
                now,
            )

    async def query_persisted_event_side_effect_deliveries(
        self,
        query: PersistedEventSideEffectQuery,
    ) -> PersistedEventSideEffectPage:
        query = PersistedEventSideEffectQuery.model_validate(query)
        key = side_effect_health.cursor_key(query)
        async with self._lock:
            now = self._ownership_clock().astimezone(UTC)
            rows = sorted(
                (
                    row
                    for row in self._persisted_event_side_effect_deliveries.values()
                    if (key is None or (row.session_id, row.event_id) > key)
                    and (query.statuses is None or row.status in query.statuses)
                    and (not query.outstanding_only or row.status != "delivered")
                    and (not query.claimable_only or side_effect_health.claimable(row, now))
                ),
                key=lambda row: (row.session_id, row.event_id),
            )[: query.limit + 1]
            return side_effect_health.page(rows, query, now)

    async def list_persisted_event_side_effect_deliveries(
        self,
        *,
        statuses: set[PersistedEventSideEffectStatus] | None = None,
        claimable_only: bool = False,
        after_sequence: int | None = None,
        limit: int = 100,
    ) -> list[PersistedEventSideEffectDelivery]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000.")
        if type(claimable_only) is not bool:
            raise TypeError("claimable_only must be a bool.")
        if after_sequence is not None and (type(after_sequence) is not int or after_sequence < 0):
            raise ValueError("after_sequence must be a non-negative integer.")
        selected_statuses = (
            None
            if statuses is None
            else {PersistedEventSideEffectStatus(status) for status in statuses}
        )
        async with self._lock:
            now = self._ownership_clock()
            deliveries = sorted(
                self._persisted_event_side_effect_deliveries.values(),
                key=lambda delivery: delivery.event_sequence,
            )
            return [
                delivery.model_copy(deep=True)
                for delivery in deliveries
                if after_sequence is None or delivery.event_sequence > after_sequence
                if selected_statuses is None or delivery.status in selected_statuses
                if not claimable_only
                or (
                    delivery.status
                    in {
                        PersistedEventSideEffectStatus.PENDING,
                        PersistedEventSideEffectStatus.FAILED,
                    }
                    and (
                        delivery.status is not PersistedEventSideEffectStatus.FAILED
                        or delivery.next_attempt_at is None
                        or delivery.next_attempt_at <= now
                    )
                )
                or (
                    delivery.status is PersistedEventSideEffectStatus.LEASED
                    and delivery.lease_expires_at is not None
                    and delivery.lease_expires_at <= now
                )
            ][:limit]

    async def renew_persisted_event_side_effect(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        if type(lease_seconds) not in {int, float} or not 0 < lease_seconds <= 86_400:
            raise ValueError("lease_seconds must be positive and at most 86400.")
        async with self._lock:
            if (
                claim.session_id,
                claim.event_id,
            ) not in self._persisted_event_side_effect_deliveries:
                raise PersistedEventSideEffectClaimLost(
                    "Persisted event side-effect claim is no longer active."
                )
            delivery = self._matching_persisted_event_side_effect_claim_unlocked(claim)
            now = self._ownership_clock()
            if delivery.lease_expires_at is None or delivery.lease_expires_at <= now:
                raise PersistedEventSideEffectClaimLost(
                    "Persisted event side-effect claim is no longer active."
                )
            renewed = delivery.model_copy(
                update={
                    "lease_expires_at": max(
                        delivery.lease_expires_at, now + timedelta(seconds=float(lease_seconds))
                    ),
                    "updated_at": now,
                },
                deep=True,
            )
            self._persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = (
                renewed
            )
            return renewed.model_copy(deep=True)

    def _matching_persisted_event_side_effect_claim_unlocked(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        delivery = self._persisted_event_side_effect_deliveries.get(
            (claim.session_id, claim.event_id)
        )
        if delivery is None:
            raise ValueError("Persisted event side-effect delivery was not found.")
        if (
            delivery.status is not PersistedEventSideEffectStatus.LEASED
            or delivery.claim_id != claim.claim_id
            or delivery.attempts != claim.attempt
        ):
            raise PersistedEventSideEffectClaimLost(
                "Persisted event side-effect claim is no longer active."
            )
        return delivery

    def _session_message_source_unlocked(
        self,
        session_id: str,
        *,
        include_transcript_digest: bool,
        include_checkpoint_digest: bool,
    ) -> SessionMessageSource:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("Session-message source not found.")
        messages = self._transcripts.get(session_id, [])
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
                session_message_checkpoint_sha256(self._checkpoints.get(session_id))
                if include_checkpoint_digest
                else None
            ),
        )

    def _session_message_accepted_event_unlocked(
        self,
        raw: dict[str, Any],
        *,
        quarantine: bool = False,
    ) -> Event | None:
        if quarantine:
            ids = self._session_message_acceptance_event_ids.get(
                (raw["session_id"], raw["queue_id"]), set()
            )
            if len(ids) != 1:
                raise SessionMessageConflict()
            record = self._event_records_by_id.get((raw["session_id"], next(iter(ids))))
            return None if record is None else record.event
        event_id = raw["accepted_event_id"]
        if type(event_id) is not str or len(event_id) > 512:
            raise SessionMessageConflict()
        record = self._event_records_by_id.get((raw["session_id"], event_id))
        return None if record is None else record.event

    def _raw_session_message_unlocked(self, message: SessionQueuedMessage) -> dict[str, Any]:
        raw = message.model_dump(mode="json", warnings=False)
        raw["terminal_json"] = self._session_message_terminal_receipts.get(
            (message.session_id, message.queue_id)
        )
        return raw

    def _inspect_session_message_unlocked(
        self, message: SessionQueuedMessage
    ) -> SessionMessageInspectionRecord:
        from cayu.runtime._session_message_queue import inspect_record

        raw = self._raw_session_message_unlocked(message)
        return inspect_record(
            raw,
            lambda: SessionQueuedMessage.model_validate(
                {key: value for key, value in raw.items() if key != "terminal_json"}
            ),
        )

    @runtime_session_query
    async def inspect_session_messages(
        self,
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
        async with self._lock:
            session = self._sessions.get(query.session_id)
            require_resource_session(session, "read")
            if session is None:
                raise KeyError("Session-message target not found.")
            require_authorized_session_instance(session, expected_authorized_session_instance_id)
            candidates = self._queued_session_messages_by_idempotency.get(session.id, {}).values()
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
                self._inspect_session_message_unlocked(message)
                for message in messages[: query.limit]
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

    @runtime_session_query
    async def apply_session_message_action(
        self, request: SessionMessageActionRequest
    ) -> SessionMessageActionResult:
        from cayu.runtime._session_message_queue import (
            raw_revision,
            replay_action,
            terminal_event,
            terminal_receipt,
        )

        request = copy_session_message_action(request)
        async with self._lock:
            session = self._sessions.get(request.session_id)
            require_resource_session(session, "modify")
            if session is None or session.instance_id != request.session_instance_id:
                raise SessionMessageConflict()
            messages = self._queued_session_messages_by_idempotency.get(session.id, {})
            message = next(
                (item for item in messages.values() if item.queue_id == request.queue_id), None
            )
            if message is None:
                raise SessionMessageConflict()
            raw = self._raw_session_message_unlocked(message)
            accepted_event = self._session_message_accepted_event_unlocked(
                raw,
                quarantine=request.action == "quarantine",
            )
            replay = replay_action(raw, request, accepted_event=accepted_event)
            if replay is not None:
                return SessionMessageActionResult(
                    record=self._inspect_session_message_unlocked(message),
                    event=replay,
                    replayed=True,
                )
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session.id,))
            if (
                raw_revision(raw) != request.expected_revision
                or message.status != SessionMessageQueueStatus.QUEUED
                or message.delivered_event_id is not None
                or any(
                    delivered.queue_id == message.queue_id
                    for delivery in self._session_message_delivery_records.values()
                    if delivery.session_id == session.id
                    for delivered in delivery.batch.messages
                )
            ):
                raise SessionMessageConflict()
            if (
                request.action == "withdraw"
                and self._inspect_session_message_unlocked(message).validity != "valid"
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
                self._ownership_clock(),
                actor=request.requested_by,
                accepted_event=accepted_event,
            )
            updated = message.model_copy(update={"status": status}, deep=True)
            updated_session = self._append_events_unlocked(session, [event])
            self._session_message_terminal_receipts[(session.id, message.queue_id)] = (
                terminal_receipt(status, event, request)
            )
            messages[message.idempotency_key] = updated
            for mode in SessionMessageDeliveryMode:
                key = (session.id, mode)
                pending = self._pending_session_messages.get(key)
                if pending is not None:
                    remaining = deque(item for item in pending if item.queue_id != message.queue_id)
                    if remaining:
                        self._pending_session_messages[key] = remaining
                    else:
                        del self._pending_session_messages[key]
            self._sessions[session.id] = updated_session
            return SessionMessageActionResult(
                record=self._inspect_session_message_unlocked(updated), event=event
            )

    @runtime_session_query
    async def snapshot_session_message_source(
        self,
        session_id: str,
        *,
        include_transcript_digest: bool = False,
        include_checkpoint_digest: bool = False,
        expected_authorized_session_instance_id: str | None = None,
    ) -> SessionMessageSource:
        from cayu.runtime._session_message_queue import require_authorized_session_instance

        session_id = require_clean_nonblank(session_id, "session_id")
        if (
            type(include_transcript_digest) is not bool
            or type(include_checkpoint_digest) is not bool
        ):
            raise TypeError("Source snapshot digest flags must be bools.")
        async with self._lock:
            session = self._sessions.get(session_id)
            require_resource_session(session, "read")
            if session is None:
                raise KeyError("Session-message source not found.")
            require_authorized_session_instance(session, expected_authorized_session_instance_id)
            return self._session_message_source_unlocked(
                session_id,
                include_transcript_digest=include_transcript_digest,
                include_checkpoint_digest=include_checkpoint_digest,
            )

    @runtime_session_query
    async def append_peer_content(
        self,
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
        async with self._lock:
            creation = request.append_key.creation_target
            target_id, target_instance, creation_excluded = resolve_peer_target(
                request.append_key,
                None if creation is None else self._session_creation_decisions.get(creation.key),
            )
            from cayu.sessions._access_rules import _query_bounds

            if _query_bounds.get() is not None:
                require_resource_session(
                    self._sessions.get(request.occurrence.sender_session_id), "read"
                )
                require_resource_session(
                    None if target_id is None else self._sessions.get(target_id), "modify"
                )
            from cayu.storage._peer_attempts import historical_replay, replay_or_advance

            historical = historical_replay(
                request, self._peer_content_attempts.get(request.operation_key)
            )
            if historical is not None:
                return historical
            existing = self._peer_content_receipts.get(key)
            if existing is not None:
                replay = replay_or_advance(request, self._peer_content_requests[key], existing)
                if replay is not None:
                    return replay
            elif request.replaces_operation_key is not None:
                raise PeerContentConflict()
            from cayu.storage._peer_attempts import receiving_cursor

            target_cursor = receiving_cursor(request, existing, pending_transcript_cursor)
            target = None if target_id is None else self._sessions.get(target_id)
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

                checkpoint = self._checkpoints.get(target.id)
                wait_key = parked_delivery_key(
                    checkpoint, session_id=target.id, instance_id=target.instance_id
                )
                if wait_key is not None:
                    parked_target = permits_parked_delivery_append(
                        request,
                        checkpoint,
                        self._session_operation_records.get(target.id, {}).get(wait_key),
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

                require_open_admission(target.status, self._checkpoints.get(target.id))
            target_binding = (
                None if target_id is None else self._participant_session_bindings.get(target_id)
            )
            target_binding_valid = target_binding is not None and (
                target_binding.session_instance_id == target_instance
                and target_binding.participant.participant_id == request.append_key.consumer_id
                and target_binding.participant.incarnation
                == request.append_key.consumer_participant_incarnation
            )
            source_binding = self._participant_session_bindings.get(
                request.occurrence.sender_session_id
            )
            source_valid = (
                source_binding is not None
                and source_binding.session_instance_id
                == request.occurrence.sender_session_instance_id
                and source_binding.participant.participant_id
                == request.occurrence.sender_participant_id
                and source_binding.participant.incarnation
                == request.occurrence.sender_participant_incarnation
            )
            expired = (
                int(self._ownership_clock().timestamp() * 1000)
                >= request.attempt_key.deadline_at_ms
            )
            from cayu.storage._peer_attempts import qualify, require_capacity

            outstanding = 0
            for other_key, other in self._peer_content_receipts.items():
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
                        self._peer_content_deleted_targets
                    ):
                        continue
                    queued = self._queued_session_messages_by_idempotency.get(
                        other.target_session_id, {}
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
                isinstance(self._checkpoints.get(target.id), dict)
                and tool_publication_rules._PENDING_TOOL_ROUND_CHECKPOINT_KEY
                in self._checkpoints[target.id]
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
                or len(self._transcripts.get(target.id, ())) != target_cursor
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
                    for item in self._queued_session_messages_by_idempotency.get(
                        target_id, {}
                    ).values()
                ):
                    raise PeerContentConflict("Peer queue identity already has another authority.")
                delivery_mode = (
                    SessionMessageDeliveryMode.ON_IDLE
                    if request.wake_policy == "ordinary_continuation"
                    else SessionMessageDeliveryMode.NEXT_TURN
                )
                accepted_at = self._ownership_clock()
                ordering_key = self._next_session_message_ordering_key
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
                            transcript_cursor=len(self._transcripts.get(target.id, [])),
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
                    accepted_transcript_cursor=len(self._transcripts.get(target.id, [])),
                    accepted_event_id=accepted_event.id,
                    accepted_at=accepted_at,
                    conditions=SessionMessageConditions(),
                )
                self._sessions[target.id] = self._append_events_unlocked(target, [accepted_event])
                self._queued_session_messages_by_idempotency.setdefault(target.id, {})[
                    queued_message.idempotency_key
                ] = queued_message
                self._pending_session_messages.setdefault(
                    (target.id, delivery_mode), deque()
                ).append(queued_message)
                self._next_session_message_ordering_key += 1
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
            self._peer_content_receipts[key] = receipt
            self._peer_content_operation_commitments[key] = commitment
            self._peer_content_operations[request.operation_key] = commitment
            self._peer_content_requests[key] = request.model_copy(deep=True)
            self._peer_content_attempts[request.operation_key] = (
                request.model_dump_json(),
                receipt.model_dump_json(),
            )
            return receipt.model_copy(deep=True)

    async def read_peer_content_attempt(
        self, request: PeerContentAppendRequest
    ) -> PeerContentReceipt | None:
        from cayu.storage._peer_attempts import exact_read

        request = PeerContentAppendRequest.model_validate(request)
        async with self._lock:
            return exact_read(request, self._peer_content_attempts.get(request.operation_key))

    async def list_pending_peer_content(self, *, after_operation_key=None, limit=32):
        """Trusted receiving-owner discovery, independent of creation settlement."""
        from cayu.collaboration.peer_content import validate_peer_discovery

        validate_peer_discovery(after_operation_key, limit)
        async with self._lock:
            keys = sorted(
                self._peer_content_requests,
                key=lambda key: self._peer_content_requests[key].operation_key,
            )
            return tuple(
                self._peer_content_requests[key].model_copy(deep=True)
                for key in keys
                if self._peer_content_receipts[key].status == "pending"
                and self._peer_content_requests[key].operation_key > (after_operation_key or "")
            )[:limit]

    async def read_peer_content(self, append_key: PeerAppendKey) -> PeerContentReceipt | None:
        if type(append_key) is not PeerAppendKey:
            raise TypeError("append_key must be a PeerAppendKey.")
        append_key = PeerAppendKey.model_validate(append_key)
        async with self._lock:
            value = self._peer_content_receipts.get(append_key.model_dump_json())
            return None if value is None else value.model_copy(deep=True)

    async def record_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        if type(request) is not PeerContentExposureRequest:
            raise TypeError("Exposure requires a PeerContentExposureRequest.")
        key = request.exposure_id
        commitment = request.model_dump_json()
        async with self._lock:
            existing = self._peer_content_exposures.get(key)
            if existing is None:
                existing = next(
                    (
                        value
                        for value in self._peer_content_exposures.values()
                        if value[1].operation_key == request.operation_key
                    ),
                    None,
                )
            if existing is not None:
                expected = (
                    request.identity_commitment()
                    if existing[1].outcome == "pending"
                    else commitment
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
                    self._peer_content_exposures[key] = (commitment, receipt)
                    return receipt
                return existing[1].model_copy(update={"replayed": True})
            append = self._peer_content_receipts.get(request.append_key.model_dump_json())
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
            self._peer_content_exposures[key] = (commitment, receipt)
            return receipt.model_copy(deep=True)

    async def begin_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        if type(request) is not PeerContentExposureRequest:
            raise TypeError("Exposure requires a PeerContentExposureRequest.")
        identity = request.identity_commitment()
        key = request.exposure_id
        async with self._lock:
            existing = self._peer_content_exposures.get(key)
            if existing is None:
                existing = next(
                    (
                        value
                        for value in self._peer_content_exposures.values()
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
            append = self._peer_content_receipts.get(request.append_key.model_dump_json())
            if append is None or append.status != "appended":
                raise PeerContentUnavailable("Peer content was not durably appended.")
            receipt = PeerContentExposureReceipt(
                operation_key=request.operation_key,
                append_key=request.append_key,
                exposure_id=request.exposure_id,
                model_attempt_id=request.model_attempt_id,
                outcome="pending",
            )
            self._peer_content_exposures[key] = (identity, receipt)
            return receipt.model_copy(deep=True)

    async def read_peer_content_exposure(
        self, append_key: PeerAppendKey, exposure_id: str
    ) -> PeerContentExposureReceipt | None:
        if type(append_key) is not PeerAppendKey or type(exposure_id) is not str:
            raise TypeError("Invalid exposure lookup.")
        append_key = PeerAppendKey.model_validate(append_key)
        async with self._lock:
            value = self._peer_content_exposures.get(exposure_id)
            if value is None or value[1].append_key != append_key:
                return None
            return value[1].model_copy(deep=True)

    async def exclude_peer_content(
        self,
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
        async with self._lock:
            creation = request.append_key.creation_target
            resolve_peer_target(
                request.append_key,
                None if creation is None else self._session_creation_decisions.get(creation.key),
            )
            commitment = request.model_dump_json()
            from cayu.storage._peer_attempts import historical_replay

            replay = historical_replay(
                request, self._peer_content_attempts.get(request.operation_key)
            )
            if replay is not None:
                if replay.status == "excluded" and replay.reason != reason:
                    raise PeerContentConflict()
                return replay
            prior_operation = self._peer_content_operations.get(request.operation_key)
            if prior_operation is not None and prior_operation != commitment:
                raise PeerContentConflict()
            existing = self._peer_content_receipts.get(key)
            if existing is not None:
                if self._peer_content_operation_commitments[key] != commitment:
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
            self._peer_content_receipts[key] = receipt
            self._peer_content_operation_commitments[key] = commitment
            self._peer_content_operations[request.operation_key] = commitment
            self._peer_content_requests[key] = request.model_copy(deep=True)
            self._peer_content_attempts[request.operation_key] = (
                request.model_dump_json(),
                receipt.model_dump_json(),
            )
            return receipt.model_copy(deep=True)

    async def retry_pending_peer_content(
        self,
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
                None if creation is None else self._session_creation_decisions.get(creation.key),
            )
            return target_id == session_id and instance == expected_session_instance_id

        async with self._lock:
            requests = [
                request.model_copy(deep=True)
                for key, request in self._peer_content_requests.items()
                if target_matches(request)
                and self._peer_content_receipts.get(key) is not None
                and self._peer_content_receipts[key].status == "pending"
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
                raise PeerContentUnavailable(
                    "Pending delivery requires fresh export authorization."
                )
            results.append(
                await admit(request, pending_transcript_cursor=expected_transcript_cursor)
            )
        return tuple(results)

    @runtime_session_query
    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
        *,
        expected_authorized_target_instance_id: str | None = None,
    ) -> EnqueueSessionMessageResult:
        request = copy_enqueue_session_message_request(request)
        async with self._lock:
            session = self._sessions.get(request.session_id)
            require_resource_session(session, "modify")
            if session is None:
                raise KeyError(f"Session not found: {request.session_id}")
            if expected_authorized_target_instance_id is not None and (
                type(expected_authorized_target_instance_id) is not str
                or session.instance_id != expected_authorized_target_instance_id
            ):
                raise SessionMessageConflict()
            if request.conditions.source is not None:
                source_session = self._sessions.get(request.conditions.source.session_id)
                require_resource_session(source_session, "read")
                if (
                    source_session is None
                    or source_session.instance_id != request.conditions.source.session_instance_id
                ):
                    raise SessionMessageConflict()
            messages_by_idempotency = self._queued_session_messages_by_idempotency.get(
                request.session_id
            )
            existing = (
                None
                if messages_by_idempotency is None
                else messages_by_idempotency.get(request.idempotency_key)
            )
            if existing is not None:
                session_messaging._validate_equivalent_queued_session_message(existing, request)
                record = self._event_records_by_id.get(
                    (request.session_id, existing.accepted_event_id)
                )
                if record is None:
                    raise RuntimeError(
                        "Queued session message is missing its durable acceptance event."
                    )
                return EnqueueSessionMessageResult(
                    message=existing,
                    event=record.event,
                    replayed=True,
                )
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (request.session_id,))
            from cayu.runtime._session_message_queue import require_open_admission

            require_open_admission(session.status, self._checkpoints.get(session.id))
            accepted_at = self._ownership_clock()
            queue_id = str(uuid4())
            source = request.conditions.source
            if source is not None and source != self._session_message_source_unlocked(
                source.session_id,
                include_transcript_digest=source.transcript_sha256 is not None,
                include_checkpoint_digest=source.checkpoint_sha256 is not None,
            ):
                raise SessionMessageConflict()
            ordering_key = self._next_session_message_ordering_key
            accepted_transcript_cursor = len(self._transcripts.get(session.id, []))
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
            self._sessions[session.id] = self._append_events_unlocked(
                session,
                [accepted_event],
            )
            self._queued_session_messages_by_idempotency.setdefault(session.id, {})[
                queued_message.idempotency_key
            ] = queued_message
            pending_key = (session.id, queued_message.delivery_mode)
            self._pending_session_messages.setdefault(pending_key, deque()).append(queued_message)
            self._next_session_message_ordering_key += 1
            return EnqueueSessionMessageResult(
                message=queued_message,
                event=accepted_event,
            )

    async def deliver_queued_session_messages(
        self,
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
            str(uuid4())
            if delivery_id is None
            else require_clean_nonblank(delivery_id, "delivery_id")
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
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            existing_delivery = self._session_message_delivery_records.get(delivery_id)
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
                    operation_records = self._session_operation_records.get(session_id, {})
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
                        marker = (
                            model_stage_rules._reconstruct_active_model_completion_stage_record(
                                active_record,
                                session_id=session_id,
                            )
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
                            self._checkpoints.get(session_id),
                            profile_handoff,
                            settlement_record=receipt_record,
                            replayed_delivery=True,
                            active_model_stage=active_model_stage,
                            stage_dispatch=stage_dispatch,
                        )
                    )
                    self._store_checkpoint_unlocked(session_id, repaired_checkpoint)
                return replayed_batch
            if session.status != SessionStatus.RUNNING:
                raise SessionStatusConflict(
                    "Queued session messages may be delivered only while running."
                )
            boundary = (
                self._next_session_message_ordering_key - 1
                if eligible_through is None
                else eligible_through
            )

            selected_key: tuple[str, SessionMessageDeliveryMode] | None = None
            selected: list[SessionQueuedMessage] = []
            delivered_at = self._ownership_clock()
            transcript_cursor = len(self._transcripts.get(session_id, []))

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
                pending = self._pending_session_messages.get(pending_key)
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
                self._session_message_delivery_records[delivery_id] = (
                    _InMemoryMessageDeliveryRecord(
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
                pending = self._pending_session_messages.get((session_id, delivery_mode), ())
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
                message.queue_id: self._session_message_accepted_event_unlocked(
                    self._raw_session_message_unlocked(message)
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
                    self._raw_session_message_unlocked(queued_message),
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
                self._reject_new_work_after_steering_unlocked(
                    session, allow_completed_interaction=True
                )
                receipt_record = self._session_operation_records.get(session_id, {}).get(
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
                        self._checkpoints.get(session_id),
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
                                self._raw_session_message_unlocked(queued_message),
                                accepted_events[queued_message.queue_id],
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
            delivery_record = _InMemoryMessageDeliveryRecord(
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
                for message in self._pending_session_messages[selected_key]
                if message.queue_id not in selected_ids
            )
            updated_session = self._append_events_unlocked(session, persisted_events)
            self._transcripts.setdefault(session_id, []).extend(transcript_messages)
            memory_transcripts.extend_search(
                self._transcript_indexes_unlocked(), session_id, transcript_messages
            )
            self._transcript_interaction_ids.setdefault(session_id, [])
            memory_transcripts.extend_attribution(
                self._transcript_indexes_unlocked(),
                session_id,
                [interaction_id] * len(transcript_messages),
            )
            self._pending_session_messages[selected_key] = selected_queue
            if not selected_queue:
                del self._pending_session_messages[selected_key]
            messages_by_idempotency = self._queued_session_messages_by_idempotency[session_id]
            for updated_message in [*updated_messages, *rejected_messages]:
                messages_by_idempotency[updated_message.idempotency_key] = updated_message
            self._session_message_terminal_receipts.update(rejected_receipts)
            self._sessions[session_id] = updated_session
            if rebound_checkpoint is not None:
                self._store_checkpoint_unlocked(session_id, rebound_checkpoint)
            self._session_message_delivery_records[delivery_id] = delivery_record
            return batch

    async def repair_queued_interaction_profile_handoff(
        self,
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
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            delivery = self._session_message_delivery_records.get(target.interaction_id)
            if (
                delivery is None
                or delivery.session_id != session_id
                or delivery.interaction_id != target.interaction_id
                or delivery.interaction_started_event != interaction_started_event
                or not delivery.batch.messages
                or (
                    delivery.profile_handoff is not None
                    and delivery.profile_handoff != profile_handoff
                )
            ):
                raise SessionRunFenced(
                    "Historical queued interaction handoff lacks its exact delivery receipt."
                )
            operation_records = self._session_operation_records.get(session_id, {})
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
                    self._checkpoints.get(session_id),
                    profile_handoff,
                    settlement_record=settlement_record,
                    replayed_delivery=True,
                    active_model_stage=active_model_stage,
                    stage_dispatch=stage_dispatch,
                )
            )
            self._store_checkpoint_unlocked(session_id, repaired_checkpoint)
            return target.model_copy(deep=True)

    async def publish_checkpoint_and_events(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        return await self._publish_checkpoint_and_events(
            session_id,
            checkpoint_transform=checkpoint_transform,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
            preserve_completion_result_publications=True,
        )

    async def _publish_completion_result_event_publication(
        self,
        session_id: str,
        *,
        checkpoint_transform: StoreTimeCheckpointTransform,
        events: list[Event],
    ) -> Session:
        return await self._publish_checkpoint_and_events(
            session_id,
            checkpoint_transform=None,
            store_time_checkpoint_transform=checkpoint_transform,
            events=events,
            expected_statuses=None,
            expected_run_epoch=None,
            expected_transcript_cursor=None,
            preserve_completion_result_publications=False,
        )

    async def _publish_checkpoint_and_events(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform | None,
        store_time_checkpoint_transform: StoreTimeCheckpointTransform | None = None,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None,
        expected_run_epoch: int | None,
        expected_transcript_cursor: int | None,
        preserve_completion_result_publications: bool,
    ) -> Session:
        session_id, copied_events = session_event_rules._copy_session_event_batch(
            session_id, events
        )
        if (checkpoint_transform is None) == (store_time_checkpoint_transform is None):
            raise TypeError("Exactly one checkpoint transform is required.")
        allowed_statuses = (
            None
            if expected_statuses is None
            else session_record_rules._validate_status_set(expected_statuses, "expected_statuses")
        )
        async with self._lock:
            store_now = self._ownership_clock()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            if allowed_statuses is not None and session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status is not eligible for checkpoint publication: {session.status}"
                )
            if expected_run_epoch is not None and session.run_epoch != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            current_cursor = len(self._transcripts.get(session_id, []))
            if (
                expected_transcript_cursor is not None
                and current_cursor != expected_transcript_cursor
            ):
                raise ValueError(
                    "Session source transcript cursor is stale: expected "
                    f"{expected_transcript_cursor}, current {current_cursor}."
                )
            current = self._checkpoints.get(session_id)
            callback_session = session.model_copy(deep=True)
            callback_checkpoint = checkpoint_preservation._copy_checkpoint_for_transform(
                current,
                session_id=session_id,
            )
            if checkpoint_transform is not None:
                transformed = checkpoint_transform(
                    callback_session,
                    callback_checkpoint,
                )
            else:
                assert store_time_checkpoint_transform is not None
                transformed = store_time_checkpoint_transform(
                    callback_session,
                    callback_checkpoint,
                    store_now,
                )
            if transformed is None:
                raise ValueError("Checkpoint transform must return a checkpoint.")
            copied_checkpoint = copy_durable_json_object(transformed, "checkpoint")
            copied_checkpoint = checkpoint_preservation._replace_checkpoint_preserving_completion_result_event_publications(
                current,
                copied_checkpoint,
                preserve_completion_result_publications=(preserve_completion_result_publications),
                session_id=session_id,
            )
            updated = self._append_events_unlocked(session, copied_events)
            if not copied_events:
                updated = updated.model_copy(update={"last_activity_at": store_now})
            self._store_checkpoint_unlocked(session_id, copied_checkpoint)
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def load_session_operation(
        self,
        session_id: str,
        idempotency_key: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        idempotency_key = operation_record_rules._reject_reserved_runtime_publication_key(
            idempotency_key,
            "idempotency_key",
            browser_control_read=True,
        )
        _require_session_export_target(session_id, idempotency_key)
        async with self._lock:
            if access_bounds is not None:
                access_bounds.require_read(self._sessions.get(session_id))
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            if checkpoint_root_guard is not None:
                checkpoint_root_guard.validate(
                    session_id,
                    checkpoint_root_field_projection(
                        self._checkpoints.get(session_id),
                        checkpoint_root_guard.key,
                    ),
                )
            record = self._session_operation_records.get(session_id, {}).get(idempotency_key)
            return None if record is None else copy_durable_json_object(record, "session_operation")

    async def _load_historical_interaction_settlement_record(
        self, session_id: str, event_id: str
    ) -> dict[str, Any] | None:
        key = interaction_receipts._interaction_transition_storage_key(event_id)
        async with self._lock:
            record = self._session_operation_records.get(session_id, {}).get(key)
            return None if record is None else copy_durable_json_object(record, "settlement")

    async def _load_runtime_publication_receipt_record(
        self,
        session_id: str,
        storage_key: str,
        publication_id: str,
    ) -> dict[str, Any] | None:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            record = self._session_operation_records.get(session_id, {}).get(storage_key)
            if record is None:
                return None
            copied = deepcopy(record)
            receipt = publication_rules._reconstruct_runtime_publication_receipt(
                copied,
                storage_key=storage_key,
                session_id=session_id,
                publication_id=publication_id,
            )
            self._validate_runtime_publication_material_unlocked(receipt)
            return copied

    def _validate_runtime_publication_material_unlocked(
        self,
        receipt: RuntimePublicationReceipt,
    ) -> None:
        transcript = self._transcripts.get(receipt.session_id, [])
        transcript_segment = transcript[
            receipt.transcript_start_cursor : receipt.transcript_end_cursor
        ]
        transcript_interaction_ids = self._transcript_interaction_ids.get(
            receipt.session_id,
            [],
        )[receipt.transcript_start_cursor : receipt.transcript_end_cursor]
        referenced_event_ids = publication_contracts._runtime_publication_referenced_event_ids(
            receipt.referenced_events
        )
        requested_event_ids = set((*receipt.appended_event_ids, *referenced_event_ids))
        events_by_id = {}
        for event_id in requested_event_ids:
            record = self._event_records_by_id.get((receipt.session_id, event_id))
            if record is not None:
                events_by_id[event_id] = record.event
        publication_rules._validate_runtime_publication_durable_material(
            receipt,
            transcript_messages=transcript_segment,
            transcript_interaction_ids=transcript_interaction_ids,
            appended_events=(
                events_by_id[event_id]
                for event_id in receipt.appended_event_ids
                if event_id in events_by_id
            ),
            durable_referenced_events=(
                events_by_id[event_id]
                for event_id in referenced_event_ids
                if event_id in events_by_id
            ),
        )

    async def publish_session_operation(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        return await self._publish_session_operation(
            session_id,
            idempotency_key=idempotency_key,
            operation_transform=operation_transform,
            store_time_operation_transform=None,
            commit_guard=None,
            commit_time_guard=None,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )

    async def publish_session_operation_guarded(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        commit_guard: Callable[[], None],
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        return await self._publish_session_operation(
            session_id,
            idempotency_key=idempotency_key,
            operation_transform=operation_transform,
            store_time_operation_transform=None,
            commit_guard=commit_guard,
            commit_time_guard=None,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )

    async def publish_session_operation_guarded_with_store_time(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: StoreTimeSessionOperationTransform,
        commit_guard: Callable[[], None],
        commit_time_guard: Callable[[datetime], None],
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
        context_view_compaction_cursor: int | None = None,
    ) -> Session:
        return await self._publish_session_operation(
            session_id,
            idempotency_key=idempotency_key,
            operation_transform=None,
            store_time_operation_transform=operation_transform,
            commit_guard=commit_guard,
            commit_time_guard=commit_time_guard,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
            context_view_compaction_cursor=context_view_compaction_cursor,
        )

    async def _prepare_temporary_side_service(self, preparation):
        from cayu.sessions._side_service_preparation import (
            SidePreparationSnapshot,
            plan_preparation,
            preparation_record_keys,
            prepare_selection,
        )

        prepared = prepare_selection(preparation)
        keys = preparation_record_keys(prepared)
        async with self._lock:
            snapshots = {}
            for session_id in sorted(keys):
                session = self._sessions.get(session_id)
                if session is None:
                    raise KeyError("Side-service session is unavailable.")
                invocation_scope._assert_session_run_epoch(session_id, session)
                for owner in self._session_closure_progress.values():
                    closure_records._check_closure_lineage_owner(owner, (session_id,))
                snapshots[session_id] = SidePreparationSnapshot(
                    session=session,
                    checkpoint=self._checkpoints.get(session_id),
                    records={
                        key: value
                        for key in keys[session_id]
                        if (value := self._session_operation_records.get(session_id, {}).get(key))
                        is not None
                    },
                )
            now = self._ownership_clock()
            plans = plan_preparation(prepared, snapshots, now)
            sessions = {
                session_id: snapshots[session_id].session.model_copy(
                    update={"updated_at": now, "last_activity_at": now}
                )
                for session_id in plans
            }
            operations = {
                session_id: self._session_operation_records.get(session_id, {})
                | plan.operation_records
                for session_id, plan in plans.items()
            }
            # Planning verifies that only the continuation root changes. Other
            # checkpoint owners' material and derived indexes remain unchanged.
            for session_id, plan in plans.items():
                self._checkpoints[session_id] = plan.checkpoint
                self._session_operation_records[session_id] = operations[session_id]
                self._sessions[session_id] = sessions[session_id]

    async def _publish_session_operation(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform | None,
        store_time_operation_transform: StoreTimeSessionOperationTransform | None,
        commit_guard: Callable[[], None] | None,
        commit_time_guard: Callable[[datetime], None] | None,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None,
        expected_run_epoch: int | None,
        expected_transcript_cursor: int | None,
        context_view_compaction_cursor: int | None = None,
    ) -> Session:
        session_id, copied_events = session_event_rules._copy_session_event_batch(
            session_id, events
        )
        idempotency_key = operation_record_rules._reject_reserved_runtime_publication_key(
            idempotency_key,
            "idempotency_key",
        )
        if (operation_transform is None) == (store_time_operation_transform is None):
            raise TypeError("Exactly one session operation transform is required.")
        _require_session_export_target(session_id, idempotency_key)
        allowed_statuses = (
            None
            if expected_statuses is None
            else session_record_rules._validate_status_set(expected_statuses, "expected_statuses")
        )
        async with self._lock:
            store_now = self._ownership_clock()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            if allowed_statuses is not None and session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status is not eligible for checkpoint publication: {session.status}"
                )
            if expected_run_epoch is not None and session.run_epoch != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            current_cursor = len(self._transcripts.get(session_id, []))
            if context_view_compaction_cursor is not None:
                self._validate_context_view_compaction_unlocked(
                    session_id, context_view_compaction_cursor
                )
            if (
                expected_transcript_cursor is not None
                and current_cursor != expected_transcript_cursor
            ):
                raise ValueError(
                    "Session source transcript cursor is stale: expected "
                    f"{expected_transcript_cursor}, current {current_cursor}."
                )
            current_checkpoint = self._checkpoints.get(session_id)
            current_record = self._session_operation_records.get(session_id, {}).get(
                idempotency_key
            )
            from cayu.collaboration import _session_export_store as session_exports
            from cayu.sessions import _producer_checkpoint as producers
            from cayu.sessions import _session_continuation_store as continuations

            callback_session = session.model_copy(deep=True)
            callback_checkpoint = (
                None if current_checkpoint is None else deepcopy(current_checkpoint)
            )
            # Keep the raw operation callback's existing view of sibling state;
            # only the newly private export root changes visibility here.
            if callback_checkpoint is not None and not session_exports.checkpoint_visible(
                session_id=session_id
            ):
                callback_checkpoint.pop(session_exports.ROOT_KEY, None)
            if callback_checkpoint is not None and not continuations.checkpoint_visible():
                callback_checkpoint.pop(continuations.ROOT_KEY, None)
            if callback_checkpoint is not None and not producers.checkpoint_visible(
                session_id=session_id
            ):
                callback_checkpoint.pop(producers.ROOT_KEY, None)
            callback_record = (
                None
                if current_record is None
                else copy_durable_json_object(current_record, "session_operation")
            )
            if operation_transform is not None:
                publication = operation_transform(
                    callback_session,
                    callback_checkpoint,
                    callback_record,
                )
            else:
                assert store_time_operation_transform is not None
                publication = store_time_operation_transform(
                    callback_session,
                    callback_checkpoint,
                    callback_record,
                    store_now,
                )
            if type(publication) is not SessionOperationPublication:
                raise TypeError(
                    "Session operation transform must return a SessionOperationPublication."
                )
            copied_checkpoint = copy_durable_json_object(
                publication.checkpoint,
                "checkpoint",
            )
            copied_checkpoint = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                current_checkpoint,
                copied_checkpoint,
                session_id=session_id,
            )
            copied_records = copy_durable_json_object(
                publication.operation_records,
                "operation_records",
            )
            operation_record_rules._validate_session_operation_record_keys(copied_records)

            indices = session_exports.selected_transcript_indices(session_id=session_id)
            if (
                indices
                or session_exports.checkpoint_visible(session_id=session_id)
                or any(key.startswith(session_exports.OPERATION_PREFIX) for key in copied_records)
            ):
                messages = self._transcripts.get(session_id, [])
                interaction_ids = self._transcript_interaction_ids.get(session_id, [])
                session_exports.validate_publication(
                    session=session,
                    current_checkpoint=current_checkpoint,
                    proposed_checkpoint=copied_checkpoint,
                    operation_records=copied_records,
                    selected_transcript_rows=tuple(
                        TranscriptRecord(
                            index=index,
                            interaction_id=interaction_ids[index],
                            message=detach_message(messages[index]),
                        )
                        for index in sorted(set(indices))
                        if 0 <= index < len(messages)
                    ),
                    events=copied_events,
                )
            release = publication.model_completion_stage_release
            if release is not None:
                operation_records = self._session_operation_records.get(session_id, {})
                _, _, preparation_key, terminal_key = (
                    model_stage_rules._model_completion_stage_storage_identity(
                        session_id, release.stage_id
                    )
                )
                model_stage_rules._validate_model_completion_stage_release(
                    session=session,
                    active_record=operation_records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    preparation_record=operation_records.get(preparation_key),
                    terminal_record=operation_records.get(terminal_key),
                    release=release,
                )
            commit_guard_signal = (
                None
                if commit_guard is None
                else await _run_session_commit_guard_owned(commit_guard)
            )
            commit_at = self._ownership_clock()
            if commit_time_guard is not None:
                commit_time_guard(commit_at)
            updated = self._append_events_unlocked(session, copied_events)
            if not copied_events:
                updated = updated.model_copy(update={"last_activity_at": commit_at})
            self._store_checkpoint_unlocked(session_id, copied_checkpoint)
            operation_records = self._session_operation_records.setdefault(session_id, {})
            operation_records.update(copied_records)
            if release is not None:
                del operation_records[MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY]
            self._sessions[session_id] = updated
            if commit_guard_signal is not None:
                raise commit_guard_signal
            return updated.model_copy(deep=True)

    async def _load_model_completion_stage_records(
        self,
        session_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            preparation = records.get(preparation_storage_key)
            terminal = records.get(terminal_storage_key)
            return (
                None if preparation is None else deepcopy(preparation),
                None if terminal is None else deepcopy(terminal),
            )

    async def _load_model_completion_stage_settlement_record(
        self,
        session_id: str,
        settlement_storage_key: str,
    ) -> dict[str, Any] | None:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            record = self._session_operation_records[session_id].get(settlement_storage_key)
            return None if record is None else deepcopy(record)

    async def _load_model_completion_stage_dispatch_record(
        self,
        session_id: str,
        dispatch_storage_key: str,
    ) -> dict[str, Any] | None:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            record = self._session_operation_records[session_id].get(dispatch_storage_key)
            return None if record is None else deepcopy(record)

    async def _load_active_model_completion_stage_records(
        self,
        session_id: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            active_record = records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
            if active_record is None:
                return None, None, None
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
            preparation = records.get(preparation_key)
            terminal = records.get(terminal_key)
            return (
                deepcopy(active_record),
                None if preparation is None else deepcopy(preparation),
                None if terminal is None else deepcopy(terminal),
            )

    async def _mark_model_completion_stage_dispatched_atomic(
        self,
        session_id: str,
        *,
        stage: ModelCompletionStage,
        consume_child_session_notifications: bool,
    ) -> ModelCompletionStageDispatch:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            _, _, preparation_key, terminal_key = (
                model_stage_rules._model_completion_stage_storage_identity(
                    session_id,
                    stage.stage_id,
                )
            )
            settlement_key = model_stage_rules._model_completion_stage_settlement_storage_key(
                stage.stage_id
            )
            dispatch_key = model_stage_rules._model_completion_stage_dispatch_storage_key(
                stage.stage_id
            )
            model_stage_rules._validate_model_completion_stage_for_dispatch(
                session=session,
                checkpoint=self._checkpoints.get(session_id),
                current_transcript_cursor=len(self._transcripts.get(session_id, [])),
                stage=stage,
                active_record=records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                preparation_record=records.get(preparation_key),
                terminal_record=records.get(terminal_key),
                settlement_record=records.get(settlement_key),
            )
            dispatch_record = records.get(dispatch_key)
            dispatch_is_new = dispatch_record is None
            if dispatch_is_new:
                published_at = publication_rules._next_runtime_publication_timestamp(session)
                dispatch_record = model_stage_rules._model_completion_stage_dispatch_record(
                    stage,
                    dispatched_at=published_at,
                )
            else:
                published_at = publication_rules._next_runtime_publication_timestamp(session)
            assert dispatch_record is not None
            dispatch = model_stage_rules._reconstruct_model_completion_stage_dispatch(
                dispatch_record,
                session_id=session_id,
                stage_id=stage.stage_id,
                storage_key=dispatch_key,
            )
            model_stage_rules._validate_model_completion_stage_dispatch(dispatch, stage)
            binding = child_session_notification_stage_binding(stage.intent)
            prepared_consumptions: list[tuple[str, dict[str, Any], str, str]] = []
            if binding is not None:
                for claim in binding.claims:
                    child = self._sessions.get(claim.child_session_id)
                    if child is None:
                        raise SessionModelCompletionStageConflict(
                            "Child-session notification occurrence is no longer canonical."
                        )
                    occurrence_source = child_lifecycle_rules._child_session_lifecycle_occurrence(
                        child,
                        self._child_lifecycle_event_records.get(child.id, {}),
                    )
                    if occurrence_source is None:
                        raise SessionModelCompletionStageConflict(
                            "Child-session notification occurrence is no longer canonical."
                        )
                    _relationship, occurrence = occurrence_source
                    consumption = (
                        child_lifecycle_rules._child_session_notification_consumption_record(
                            parent=session,
                            child=child,
                            occurrence=occurrence,
                            stage=stage,
                            consumed_at=published_at,
                        )
                    )
                    storage_key = child_session_notification_storage_key(
                        child.instance_id,
                        occurrence.source_id,
                    )
                    durable = records.get(storage_key)
                    material = consumption.model_dump(mode="json")
                    if (
                        durable is not None
                        and not child_lifecycle_rules._child_session_notification_consumption_replays(
                            durable,
                            consumption,
                        )
                    ):
                        raise SessionModelCompletionStageConflict(
                            "Child-session terminal notification was consumed by another stage."
                        )
                    prepared_consumptions.append(
                        (storage_key, material, child.id, occurrence.source_id)
                    )
            if dispatch_is_new:
                records[dispatch_key] = dispatch_record
            if consume_child_session_notifications:
                for storage_key, material, _child_id, _occurrence_id in prepared_consumptions:
                    records.setdefault(storage_key, material)
            for _storage_key, _material, child_id, _occurrence_id in (
                prepared_consumptions if consume_child_session_notifications else ()
            ):
                child = self._sessions.get(child_id)
                if child is None:
                    raise RuntimeError("Consumed child-session notification disappeared.")
                self._refresh_child_lifecycle_candidate_unlocked(child)
            self._sessions[session_id] = session.model_copy(
                update={"updated_at": published_at, "last_activity_at": published_at},
                deep=True,
            )
            return dispatch

    def _reject_new_work_after_steering_unlocked(
        self, session: Session, *, allow_completed_interaction: bool = False
    ) -> None:
        from cayu.runtime._session_steering import (
            reject_new_work_after_steering,
            steering_operation_key_from_checkpoint,
        )

        checkpoint = self._checkpoints.get(session.id)
        key = steering_operation_key_from_checkpoint(session, checkpoint)
        if key is not None:
            reject_new_work_after_steering(
                session,
                checkpoint,
                self._session_operation_records[session.id].get(key),
                allow_completed_interaction=allow_completed_interaction,
            )

    async def _prepare_model_completion_stage_atomic(
        self,
        prepared: model_stage_rules._PreparedModelCompletionStage,
    ) -> ModelCompletionStageResult:
        session_id = prepared.session_id
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            stage = model_stage_rules._reconstruct_model_completion_stage(
                records.get(prepared.preparation_storage_key),
                records.get(prepared.terminal_storage_key),
                session_id=session_id,
                stage_id=prepared.request.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            model_stage_rules._validate_model_completion_stage_repreparation(
                records.get(prepared.abandonment_storage_key),
                prepared,
                source_status=session.status if stage is None else stage.source_status,
            )
            if stage is not None:
                model_stage_rules._validate_model_completion_stage_preparation_replay(
                    stage, prepared
                )
                active = model_stage_rules._reconstruct_active_model_completion_stage_from_records(
                    records,
                    session_id=session_id,
                )
                model_stage_rules._validate_model_completion_preparation_replay_state(
                    stage,
                    active=active,
                    winner_exists=records.get(prepared.winner_storage_key) is not None,
                    receipt_exists=(records.get(prepared.publication_storage_key) is not None),
                )
                model_failover_rules._model_failover_preparation_checkpoint(
                    prepared,
                    session=session,
                    checkpoint=self._checkpoints.get(session_id),
                    current_transcript_cursor=len(self._transcripts.get(session_id, [])),
                    active=active,
                    records=records,
                    replayed=True,
                )
                expected_selection = model_failover_rules._model_failover_selection_event(
                    prepared, session=session, prepared_at=stage.prepared_at
                )
                if expected_selection is not None:
                    event_record = self._event_records_by_id.get(
                        (session_id, expected_selection.id)
                    )
                    model_failover_rules._validate_model_failover_selection_replay(
                        expected_selection, None if event_record is None else event_record.event
                    )
                return ModelCompletionStageResult(
                    stage=stage,
                    replayed=True,
                    dispatch_authorized=False,
                )

            active = model_stage_rules._reconstruct_active_model_completion_stage_from_records(
                records,
                session_id=session_id,
            )
            model_stage_rules._validate_model_completion_active_marker_for_preparation(
                active,
                prepared,
                source_status=session.status,
            )
            retry_settlement_request = model_stage_rules._model_completion_retry_settlement_request(
                active,
                prepared,
            )
            if (
                records.get(prepared.winner_storage_key) is not None
                or records.get(prepared.publication_storage_key) is not None
            ):
                raise SessionModelCompletionStageConflict(
                    "The logical model step already has durable publication state."
                )

            invocation_scope._assert_session_run_epoch(session_id, session)
            if session.status not in prepared.expected_statuses:
                raise SessionStatusConflict(
                    "Session status is not eligible for model-completion preparation: "
                    f"{session.status}"
                )
            if session.run_epoch != prepared.expected_run_epoch:
                raise SessionRunFenced(
                    "Session source run epoch is stale: expected "
                    f"{prepared.expected_run_epoch}, current {session.run_epoch}."
                )
            current_cursor = len(self._transcripts.get(session_id, []))
            if current_cursor != prepared.expected_transcript_cursor:
                raise ValueError(
                    "Session source transcript cursor is stale: expected "
                    f"{prepared.expected_transcript_cursor}, current {current_cursor}."
                )

            if active is None:
                self._reject_new_work_after_steering_unlocked(session)
            route_checkpoint = model_failover_rules._model_failover_preparation_checkpoint(
                prepared,
                session=session,
                checkpoint=self._checkpoints.get(session_id),
                current_transcript_cursor=current_cursor,
                active=active,
                records=records,
                replayed=False,
            )
            prepared_at = publication_rules._next_runtime_publication_timestamp(session)
            preparation_record = model_stage_rules._model_completion_stage_preparation_record(
                prepared,
                source_session=session,
                prepared_at=prepared_at,
            )
            selection_event = model_failover_rules._model_failover_selection_event(
                prepared, session=session, prepared_at=prepared_at
            )
            prepared_selection = (
                None
                if selection_event is None
                else self._prepare_event_append_unlocked(session, (selection_event,))
            )
            stage = model_stage_rules._reconstruct_model_completion_stage(
                preparation_record,
                None,
                session_id=session_id,
                stage_id=prepared.request.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            assert stage is not None
            active_record = model_stage_rules._active_model_completion_stage_record(
                stage,
                activated_at=prepared_at,
            )
            retry_settlement_storage_key = None
            retry_settlement_record = None
            if retry_settlement_request is not None:
                assert active is not None
                retry_settlement_storage_key = (
                    model_stage_rules._model_completion_stage_settlement_storage_key(
                        active.stage.stage_id
                    )
                )
                model_stage_rules._validate_model_completion_stage_for_settlement(
                    session=session,
                    stage=active.stage,
                    active=active,
                    request=retry_settlement_request,
                    settlement_record=records.get(retry_settlement_storage_key),
                    winner_exists=(
                        records.get(
                            model_stage_rules._model_completion_stage_winner_storage_key(
                                active.stage.logical_step_id
                            )
                        )
                        is not None
                    ),
                    receipt_exists=(
                        records.get(
                            publication_contracts._runtime_publication_storage_key(
                                active.stage.logical_step_id
                            )
                        )
                        is not None
                    ),
                )
                retry_settlement_record = (
                    model_stage_rules._model_completion_stage_settlement_record(
                        active.stage,
                        request=retry_settlement_request,
                        settled_at=prepared_at,
                    )
                )
            updated_session = session.model_copy(
                update={"updated_at": prepared_at, "last_activity_at": prepared_at},
                deep=True,
            )
            if retry_settlement_storage_key is not None and retry_settlement_record is not None:
                records[retry_settlement_storage_key] = retry_settlement_record
            records[prepared.preparation_storage_key] = preparation_record
            records[MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY] = active_record
            if route_checkpoint is not None:
                # Only the route root changed; pending-action/queue projections
                # remain identical and need no separate mutation owner.
                self._checkpoints[session_id] = route_checkpoint
            if prepared_selection is not None:
                updated_session = self._apply_event_append_unlocked(
                    updated_session, prepared_selection, activity_at=prepared_at
                )
            self._sessions[session_id] = updated_session
            return ModelCompletionStageResult(
                stage=stage,
                replayed=False,
                dispatch_authorized=True,
                prepared_events=() if selection_event is None else (selection_event,),
            )

    async def _complete_model_completion_stage_atomic(
        self,
        prepared: model_stage_rules._PreparedModelCompletionStageTerminal,
    ) -> ModelCompletionStageResult:
        session_id = prepared.session_id
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            stage = model_stage_rules._reconstruct_model_completion_stage(
                records.get(prepared.preparation_storage_key),
                records.get(prepared.terminal_storage_key),
                session_id=session_id,
                stage_id=prepared.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            if stage is None:
                raise KeyError(f"Model-completion stage not found: {prepared.stage_id}")
            model_stage_rules._reject_settled_model_completion_stage(
                records.get(prepared.settlement_storage_key),
                session_id=session_id,
                stage_id=prepared.stage_id,
                settlement_storage_key=prepared.settlement_storage_key,
            )
            if stage.state == "completed":
                model_stage_rules._validate_model_completion_stage_terminal_replay(stage, prepared)
                return ModelCompletionStageResult(
                    stage=stage,
                    replayed=True,
                    dispatch_authorized=False,
                )
            if prepared.recovery_fence is not None:
                model_stage_rules._validate_model_completion_stage_recovery_fence(
                    prepared.recovery_fence,
                    session=session,
                    checkpoint=self._checkpoints.get(session_id),
                    stage=stage,
                    active_record=records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    dispatch_record=records.get(
                        model_stage_rules._model_completion_stage_dispatch_storage_key(
                            stage.stage_id
                        )
                    ),
                    now=self._ownership_clock(),
                )
            model_stage_publication._validate_model_completion_stage_publication(
                prepared.publication,
                session_id=session_id,
                stage=stage,
            )
            if not session_authority_rules._runtime_publication_json_equal(
                prepared.publication.intent, stage.intent
            ):
                raise SessionModelCompletionStageConflict(
                    "The terminal model completion intent conflicts with its preparation."
                )
            completed_at = publication_rules._next_runtime_publication_timestamp(session)
            terminal_record = model_stage_rules._model_completion_stage_terminal_record(
                prepared,
                stage=stage,
                completed_at=completed_at,
            )
            completed_stage = model_stage_rules._reconstruct_model_completion_stage(
                records[prepared.preparation_storage_key],
                terminal_record,
                session_id=session_id,
                stage_id=prepared.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            assert completed_stage is not None
            advances_last_activity = (
                model_stage_rules._model_completion_terminal_advances_last_activity(
                    records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY),
                    stage=stage,
                    current_run_epoch=session.run_epoch,
                )
            )
            records[prepared.terminal_storage_key] = terminal_record
            self._sessions[session_id] = session.model_copy(
                update={
                    "updated_at": completed_at,
                    "last_activity_at": (
                        completed_at if advances_last_activity else session.last_activity_at
                    ),
                },
                deep=True,
            )
            return ModelCompletionStageResult(
                stage=completed_stage,
                replayed=False,
                dispatch_authorized=False,
            )

    async def _abandon_model_completion_stage_atomic(
        self,
        prepared: model_stage_rules._PreparedModelCompletionStageAbandonment,
    ) -> ModelCompletionStageAbandonmentResult:
        session_id = prepared.session_id
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            stage = model_stage_rules._reconstruct_model_completion_stage(
                records.get(prepared.preparation_storage_key),
                records.get(prepared.terminal_storage_key),
                session_id=session_id,
                stage_id=prepared.stage_id,
                preparation_storage_key=prepared.preparation_storage_key,
                terminal_storage_key=prepared.terminal_storage_key,
            )
            if stage is None:
                active_record = records.get(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
                if active_record is not None:
                    active_marker = (
                        model_stage_rules._reconstruct_active_model_completion_stage_record(
                            active_record,
                            session_id=session_id,
                        )
                    )
                    if active_marker.stage_id == prepared.stage_id:
                        raise SessionModelCompletionStageConflict(
                            "The active model-completion marker references a missing stage."
                        )
                replayed = model_stage_rules._replay_model_completion_stage_abandonment(
                    prepared,
                    records.get(prepared.abandonment_storage_key),
                )
                logical_step_id = replayed.abandonment.logical_step_id
                if (
                    records.get(
                        model_stage_rules._model_completion_stage_winner_storage_key(
                            logical_step_id
                        )
                    )
                    is not None
                    or records.get(
                        publication_contracts._runtime_publication_storage_key(logical_step_id)
                    )
                    is not None
                ):
                    raise SessionModelCompletionStageConflict(
                        "An abandoned model-completion stage has durable publication state."
                    )
                return replayed

            active = model_stage_rules._reconstruct_active_model_completion_stage_from_records(
                records,
                session_id=session_id,
            )
            winner_storage_key = model_stage_rules._model_completion_stage_winner_storage_key(
                stage.logical_step_id
            )
            publication_storage_key = publication_contracts._runtime_publication_storage_key(
                stage.logical_step_id
            )
            model_stage_rules._validate_model_completion_stage_for_abandonment(
                session=session,
                stage=stage,
                active=active,
                prepared=prepared,
                abandonment_record=records.get(prepared.abandonment_storage_key),
                dispatch_exists=(
                    records.get(
                        model_stage_rules._model_completion_stage_dispatch_storage_key(
                            stage.stage_id
                        )
                    )
                    is not None
                ),
                winner_exists=records.get(winner_storage_key) is not None,
                receipt_exists=records.get(publication_storage_key) is not None,
            )
            assert active is not None
            abandoned_at = publication_rules._next_runtime_publication_timestamp(session)
            abandonment_record = model_stage_rules._model_completion_stage_abandonment_record(
                stage,
                active=active,
                abandoned_at=abandoned_at,
            )
            abandonment = model_stage_rules._reconstruct_model_completion_stage_abandonment(
                abandonment_record,
                session_id=session_id,
                stage_id=prepared.stage_id,
                storage_key=prepared.abandonment_storage_key,
            )

            records[prepared.abandonment_storage_key] = abandonment_record
            del records[prepared.preparation_storage_key]
            del records[MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY]
            self._sessions[session_id] = session.model_copy(
                update={"updated_at": abandoned_at, "last_activity_at": abandoned_at},
                deep=True,
            )
            return ModelCompletionStageAbandonmentResult(
                abandonment=abandonment,
                replayed=False,
            )

    async def _promote_model_completion_stage_atomic(
        self,
        *,
        session_id: str,
        stage_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
        expected_run_epoch: int,
    ) -> RuntimePublicationResult:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            records = self._session_operation_records[session_id]
            stage = model_stage_rules._reconstruct_model_completion_stage(
                records.get(preparation_storage_key),
                records.get(terminal_storage_key),
                session_id=session_id,
                stage_id=stage_id,
                preparation_storage_key=preparation_storage_key,
                terminal_storage_key=terminal_storage_key,
            )
            if stage is None:
                raise KeyError(f"Model-completion stage not found: {stage_id}")

            publication_storage_key = publication_contracts._runtime_publication_storage_key(
                stage.logical_step_id
            )
            winner_storage_key = model_stage_rules._model_completion_stage_winner_storage_key(
                stage.logical_step_id
            )
            receipt_record = records.get(publication_storage_key)
            if receipt_record is not None:
                active = model_stage_rules._reconstruct_active_model_completion_stage_from_records(
                    records,
                    session_id=session_id,
                )
                model_stage_rules._validate_model_completion_promotion_replay_active_marker(
                    active, stage
                )
                receipt = publication_rules._reconstruct_runtime_publication_receipt(
                    deepcopy(receipt_record),
                    storage_key=publication_storage_key,
                    session_id=session_id,
                    publication_id=stage.logical_step_id,
                )
                self._validate_runtime_publication_material_unlocked(receipt)
                return model_stage_rules._replay_promoted_model_completion_stage(
                    session=session,
                    stage=stage,
                    receipt_record=deepcopy(receipt_record),
                    winner_record=deepcopy(records.get(winner_storage_key)),
                )
            if records.get(winner_storage_key) is not None:
                raise SessionModelCompletionStageConflict(
                    "A model-completion winner exists without its runtime publication receipt."
                )
            active = model_stage_rules._reconstruct_active_model_completion_stage_from_records(
                records,
                session_id=session_id,
            )
            model_stage_rules._validate_model_completion_active_marker_for_promotion(active, stage)
            prepared = model_stage_rules._prepare_model_completion_stage_promotion(
                stage,
                expected_run_epoch=expected_run_epoch,
            )
            result = self._publish_runtime_publication_unlocked(
                prepared,
                session=session,
                model_completion_stage=stage,
            )
            return result

    async def _publish_runtime_publication_atomic(
        self,
        prepared: publication_rules._PreparedRuntimePublication,
    ) -> RuntimePublicationResult:
        async with self._lock:
            session = self._sessions.get(prepared.session_id)
            if session is None:
                raise KeyError(f"Session not found: {prepared.session_id}")
            return self._publish_runtime_publication_unlocked(
                prepared,
                session=session,
            )

    def _publish_runtime_publication_unlocked(
        self,
        prepared: publication_rules._PreparedRuntimePublication,
        *,
        session: Session,
        model_completion_stage: ModelCompletionStage | None = None,
    ) -> RuntimePublicationResult:
        checkpoint_codec = prepared.checkpoint_codec
        checkpoint_decode = None if checkpoint_codec is None else checkpoint_codec.decode
        session_id = prepared.session_id
        request = prepared.request
        operation_records = self._session_operation_records[session_id]
        existing_record = operation_records.get(prepared.storage_key)
        if existing_record is not None:
            receipt = publication_rules._reconstruct_runtime_publication_receipt(
                deepcopy(existing_record),
                storage_key=prepared.storage_key,
                session_id=session_id,
                publication_id=request.publication_id,
                request_digest=prepared.request_digest,
            )
            publication_rules._validate_runtime_publication_replay_receipt(receipt, prepared)
            self._validate_runtime_publication_material_unlocked(receipt)
            return RuntimePublicationResult(
                session=session.model_copy(deep=True),
                receipt=receipt,
                replayed=True,
            )

        for owner in self._session_closure_progress.values():
            closure_records._check_closure_lineage_owner(owner, (session_id,))

        mutated_operation_records = (
            operation_records
            if not request.operation_record_mutations
            else operation_record_rules._apply_runtime_publication_operation_record_mutations(
                request.operation_record_mutations,
                operation_records,
            )
        )

        private_record = None
        if request.argument_continuity is not None:
            from cayu.sessions._argument_continuity import STORAGE_KEY, append_record

            private_record = append_record(
                operation_records.get(STORAGE_KEY),
                continuity=request.argument_continuity,
                request_digest=prepared.request_digest,
                session=session,
                messages=request.transcript_messages,
            )

        invocation_scope._assert_session_run_epoch(session_id, session)
        if (
            prepared.expected_statuses is not None
            and session.status not in prepared.expected_statuses
        ):
            raise SessionStatusConflict(
                f"Session status is not eligible for runtime publication: {session.status}"
            )
        if (
            prepared.expected_run_epoch is not None
            and session.run_epoch != prepared.expected_run_epoch
        ):
            raise SessionRunFenced(
                "Session source run epoch is stale: expected "
                f"{prepared.expected_run_epoch}, current {session.run_epoch}."
            )
        transcript_start_cursor = len(self._transcripts.get(session_id, []))
        if (
            prepared.expected_transcript_cursor is not None
            and transcript_start_cursor != prepared.expected_transcript_cursor
        ):
            raise ValueError(
                "Session source transcript cursor is stale: expected "
                f"{prepared.expected_transcript_cursor}, "
                f"current {transcript_start_cursor}."
            )

        appended_event_ids = {event.id for event in request.events}
        referenced_event_ids = set(
            publication_contracts._runtime_publication_referenced_event_ids(
                request.referenced_events
            )
        )
        if appended_event_ids & referenced_event_ids:
            raise ValueError("Appended and referenced runtime publication events overlap.")
        durable_events_by_id = {}
        for event_id in referenced_event_ids:
            record = self._event_records_by_id.get((session_id, event_id))
            if record is not None:
                durable_events_by_id[event_id] = record.event
        publication_contracts._validate_runtime_publication_event_references(
            request.referenced_events,
            durable_events_by_id,
            interaction_id=request.interaction_id,
        )
        stored_checkpoint = self._checkpoints.get(session_id)
        current_checkpoint = (
            stored_checkpoint
            if checkpoint_decode is None
            else checkpoint_decode(session, deepcopy(stored_checkpoint))
        )
        user_input_publication_rules._validate_user_input_checkpoint_mutation(
            request,
            current_checkpoint,
            session_id=session_id,
            session_instance_id=session.instance_id,
            current_run_epoch=session.run_epoch,
            durable_events_by_id=durable_events_by_id,
        )
        tool_publication_rules._validate_tool_round_checkpoint_mutation(request, current_checkpoint)
        durable_tool_events: list[Event] = []
        tool_round_identity = tool_publication_rules._tool_lifecycle_publication_identity(request)
        if tool_round_identity is not None:
            from cayu.sessions.pending_actions import pending_action_lookup_key

            execution_identity, tool_call_ids = tool_round_identity
            event_history = self._pending_action_event_history.get(session_id, {})
            for tool_call_id in tool_call_ids:
                lookup_key = pending_action_lookup_key(tool_call_id)
                durable_tool_events.extend(
                    record.event
                    for record in event_history.get(lookup_key, ())
                    if record.event.type in tool_publication_rules._TOOL_ROUND_LIFECYCLE_EVENT_TYPES
                    and tool_publication_rules._tool_round_event_matches_scope_or_is_ambiguous(
                        record.event,
                        execution_identity,
                    )
                )
            if len(durable_tool_events) > publication_contracts._tool_round_lifecycle_event_limit(
                tool_call_ids
            ):
                raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
        tool_publication_rules._validate_tool_round_publication(
            request,
            durable_events_by_id,
            durable_tool_events=durable_tool_events,
        )
        existing_event_ids = appended_event_ids & self._event_ids[session_id]
        if existing_event_ids:
            raise ValueError(
                f"Event already exists for session {session_id}: {min(existing_event_ids)}"
            )

        if checkpoint_codec is None:
            checkpoint = checkpoint_mutation_rules._apply_runtime_publication_checkpoint_mutation(
                request.mutation,
                current_checkpoint,
            )
            stored_target_checkpoint = checkpoint
        else:
            checkpoint = checkpoint_codec.apply_mutation(
                session,
                deepcopy(stored_checkpoint),
                request.mutation,
            )
            stored_target_checkpoint = checkpoint_codec.encode(session, checkpoint)
        if prepared.storage_key in operation_records:
            raise SessionRuntimePublicationConflict(
                "Runtime publication receipt appeared while the session lock was held."
            )
        prepared_events = self._prepare_event_append_unlocked(
            session,
            request.events,
            events_are_detached=True,
        )
        prepared_checkpoint = (
            None
            if stored_target_checkpoint is None or not request.mutation.operations
            else self._prepare_checkpoint_store_unlocked(
                session_id,
                stored_target_checkpoint,
                additional_event_records=tuple(event.record for event in prepared_events.events),
            )
        )

        published_at = publication_rules._next_runtime_publication_timestamp(session)
        receipt = publication_rules._build_runtime_publication_receipt(
            prepared,
            source_session=session,
            checkpoint=stored_target_checkpoint,
            transcript_start_cursor=transcript_start_cursor,
            published_at=published_at,
        )
        receipt_record = publication_rules._runtime_publication_receipt_record(receipt)
        winner_record = (
            None
            if model_completion_stage is None
            else model_stage_rules._model_completion_stage_winner_record(
                model_completion_stage,
                receipt=receipt,
            )
        )

        updated = self._apply_event_append_unlocked(
            session,
            prepared_events,
            activity_at=published_at,
        ).model_copy(
            update={
                "updated_at": published_at,
                "last_activity_at": published_at,
            }
        )
        self._transcripts[session_id].extend(request.transcript_messages)
        memory_transcripts.extend_search(
            self._transcript_indexes_unlocked(), session_id, request.transcript_messages
        )
        memory_transcripts.extend_attribution(
            self._transcript_indexes_unlocked(),
            session_id,
            [request.interaction_id] * len(request.transcript_messages),
        )
        if prepared_checkpoint is not None:
            self._apply_checkpoint_store_unlocked(session_id, prepared_checkpoint)
        if private_record is not None:
            mutated_operation_records[STORAGE_KEY] = private_record
        mutated_operation_records[prepared.storage_key] = receipt_record
        if model_completion_stage is not None:
            assert winner_record is not None
            winner_storage_key = model_stage_rules._model_completion_stage_winner_storage_key(
                model_completion_stage.logical_step_id
            )
            mutated_operation_records[winner_storage_key] = winner_record
            mutated_operation_records.pop(MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY)
        self._session_operation_records[session_id] = mutated_operation_records
        self._sessions[session_id] = updated
        return RuntimePublicationResult(
            session=updated.model_copy(deep=True),
            receipt=receipt,
            replayed=False,
        )

    async def load_session_export_snapshot(
        self,
        session_id: str,
        *,
        limits: SessionExportLimits | None = None,
    ) -> SessionExportSnapshot | None:
        from cayu.sessions.exports import SessionExportBuilder

        session_id = require_clean_nonblank(session_id, "session_id")
        builder = SessionExportBuilder(limits)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            for record in self._session_event_records.get(session_id, ()):
                builder.event(record)
            for index, message in enumerate(self._transcripts.get(session_id, ())):
                builder.message(
                    TranscriptRecord(
                        index=index,
                        message=message,
                        interaction_id=self._transcript_interaction_ids[session_id][index],
                    )
                )
            codec = self.public_authority_alias_codec
            records = []
            uses = []
            for grant_id in self._targeted_tool_grant_ids_by_session.get(session_id, ()):
                if codec is None:
                    raise RuntimeError(
                        "Exporting targeted grants requires an authority alias codec."
                    )
                record = targeted_tool_grant_with_active_reference(
                    self._targeted_tool_grants[grant_id],
                    codec,
                )
                builder.charge(record.model_dump(mode="json"))
                records.append(record)
                for binding in self._targeted_tool_uses.get(grant_id, {}).values():
                    builder.charge(binding.model_dump(mode="json"))
                    uses.append(binding.model_copy(deep=True))
            records.sort(key=lambda record: (record.issued_at, record.grant_id))
            uses.sort(key=lambda binding: (binding.bound_at, binding.use_id))
            deferred = self._deferred_interaction_inputs.get(session_id)
            checkpoint = self._checkpoints.get(session_id)
            return builder.finish(
                session=session,
                transcript_cursor=len(self._transcripts.get(session_id, ())),
                checkpoint=None
                if checkpoint is None
                else copy_durable_json_object(checkpoint, "checkpoint"),
                deferred=None if deferred is None else deferred.model_copy(deep=True),
                grants_charged=True,
                grants=TargetedToolGrantStateSnapshot(records=tuple(records), uses=tuple(uses)),
            )

    async def _complete_native_producer_cleanup(self, registration, *, authority):
        from cayu.storage._producer_cleanup import memory_cleanup

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer cleanup is not qualified.")
        return await memory_cleanup(self, registration, authority=authority, commit=True)

    async def _retire_native_producer_cleanup(self, retirement, *, authority, limit):
        from cayu.storage._producer_retirement import memory_retirement

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer retirement is not qualified.")
        return await memory_retirement(self, retirement, authority=authority, limit=limit)

    async def _read_completed_native_producer_cleanup(self, registration):
        from cayu.storage._producer_cleanup import memory_cleanup

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer cleanup readback is not qualified.")
        return await memory_cleanup(self, registration)

    async def _read_native_producer_release(self, command):
        from cayu.storage._producer_observation import memory_observation

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer release readback is not qualified.")
        return await memory_observation(self, command)

    async def _read_native_producer_attachment(self, command):
        from cayu.storage._producer_observation import memory_observation

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer attachment readback is not qualified.")
        return await memory_observation(self, command, attachment_only=True)

    async def _read_native_producer_progress(self, command, *, kind):
        from cayu.storage._producer_observation import memory_observation

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer progress readback is not qualified.")
        return await memory_observation(self, command, kind=kind)

    async def load_events(self, session_id: str) -> list[Event]:
        return await memory_event_reads.load_events(self._read_event_state, session_id)

    async def load_user_input_supersession_events(
        self,
        session_id: str,
        input_id: str,
    ) -> list[Event]:
        return await memory_event_reads.load_user_input_supersession_events(
            self._read_event_state, session_id, input_id
        )

    async def load_tool_round_lifecycle_events(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
    ) -> list[Event]:
        return await memory_event_reads.load_tool_round_lifecycle_events(
            self._read_event_state, session_id, tool_call_ids
        )

    async def load_tool_round_lifecycle_events_for_round(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
        *,
        tool_round_identity: ToolRoundIdentity,
    ) -> list[Event]:
        return await memory_event_reads.load_tool_round_lifecycle_events_for_round(
            self._read_event_state,
            session_id,
            tool_call_ids,
            tool_round_identity=tool_round_identity,
        )

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        return await memory_event_reads.query_events(self._read_event_state, query)

    async def event_exists(self, query: EventQuery) -> bool:
        return await memory_event_reads.event_exists(self._read_event_state, query)

    async def read_usage_accounting(
        self, query: EventQuery, *, by_session: bool = False, by_identity: bool = False
    ) -> UsageAccountingSnapshot:
        return await memory_event_reads.read_usage_accounting(
            self._read_event_state, query, by_session=by_session, by_identity=by_identity
        )

    async def read_cost_accounting(
        self,
        query: EventQuery,
        pricing: PriceBook,
        *,
        currency: str = "USD",
        details: bool = False,
        by_session: bool = False,
        additional_events: tuple[Event, ...] = (),
        max_detail_bytes: int | None = None,
        previous: CostAccountingSnapshot | None = None,
    ) -> CostAccountingSnapshot:
        return await memory_event_reads.read_cost_accounting(
            self._read_event_state,
            query,
            pricing,
            currency=currency,
            details=details,
            by_session=by_session,
            additional_events=additional_events,
            max_detail_bytes=max_detail_bytes,
            previous=previous,
        )

    async def query_events_bounded(
        self,
        query: EventQuery,
        *,
        max_bytes: int,
    ) -> list[EventRecord]:
        return await memory_event_reads.query_events_bounded(
            self._read_event_state, query, max_bytes=max_bytes
        )

    async def load_terminal_session_evidence(
        self,
        session_id: str,
        *,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        return await memory_event_reads.load_terminal_session_evidence(
            self._read_event_state, session_id, limits=limits
        )

    async def load_runner_owned_interrupted_evidence(
        self,
        session_id: str,
        *,
        observed_events: tuple[RunnerObservedEventIdentity, ...] | None = None,
        expected_parent_session_id: str | None = None,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        return await memory_event_reads.load_runner_owned_interrupted_evidence(
            self._read_event_state,
            session_id,
            observed_events=observed_events,
            expected_parent_session_id=expected_parent_session_id,
            limits=limits,
        )

    async def query_latest_interaction_events(
        self,
        session_id: str,
        *,
        before_sequence: int | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        return await memory_event_reads.query_latest_interaction_events(
            self._read_event_state, session_id, before_sequence=before_sequence, limit=limit
        )

    async def summarize_events(self, session_id: str) -> EventSummary:
        return await memory_event_reads.summarize_events(self._read_event_state, session_id)

    async def summarize_outcome(self, session_id: str) -> SessionOutcome:
        return await memory_event_reads.summarize_outcome(self._read_event_state, session_id)

    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        if access_bounds is not None:
            return await self._access_list_sessions(access_bounds, copy_session_query(query))
        return await memory_session_queries.list_sessions(
            self._read_session_query_state, query, pending_interruption_cascade_only=False
        )

    async def query_session_topology(
        self,
        query: SessionTopologyQuery,
    ) -> SessionTopologyStoreResult:
        return await memory_session_queries.query_session_topology(
            self._read_session_query_state, query
        )

    async def query_session_lineage(
        self,
        query: SessionLineageQuery,
    ) -> SessionLineageResult:
        return await memory_session_queries.query_session_lineage(
            self._read_session_query_state, query
        )

    async def query_child_session_lifecycle(
        self,
        query: ChildSessionLifecycleQuery,
    ) -> ChildSessionLifecyclePage:
        return await memory_session_queries.query_child_session_lifecycle(
            self._read_session_query_state, query
        )

    async def aggregate_operational_snapshot(
        self,
        filters: SessionAggregateFilter | None = None,
    ) -> SessionOperationalSnapshot:
        return await memory_session_queries.aggregate_operational_snapshot(
            self._read_session_query_state, filters
        )

    async def aggregate_usage(self, query: UsageRollupQuery) -> UsageRollupStoreResult:
        return await memory_session_queries.aggregate_usage(self._read_session_query_state, query)

    async def list_sessions_with_pending_interruption_cascade(
        self,
        query: SessionQuery | None = None,
    ) -> SessionListResult:
        return await memory_session_queries.list_sessions(
            self._read_session_query_state, query, pending_interruption_cascade_only=True
        )

    async def list_queued_dispatch_terminal_receipts(
        self,
        query: QueuedDispatchTerminalReceiptQuery | None = None,
    ) -> list[QueuedDispatchTerminalReceipt]:
        return await memory_session_queries.list_queued_dispatch_terminal_receipts(
            self._read_session_query_state, query
        )

    async def query_pending_actions(
        self,
        query: PendingActionQuery | None = None,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> PendingActionListResult:
        return await memory_session_queries.query_pending_actions(
            self._read_session_query_state, query, checkpoint_root_guard=checkpoint_root_guard
        )

    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        copied_messages = copy_transcript_messages(messages)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            if not copied_messages:
                return
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            if interaction_id is not None:
                self._register_private_authority_alias_unlocked(
                    interaction_id,
                    field_name="interaction_id",
                    scope_session_id=session_id,
                )
            self._transcripts[session_id].extend(copied_messages)
            memory_transcripts.extend_search(
                self._transcript_indexes_unlocked(), session_id, copied_messages
            )
            memory_transcripts.extend_attribution(
                self._transcript_indexes_unlocked(),
                session_id,
                [interaction_id] * len(copied_messages),
            )
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": self._ownership_clock()}
            )

    async def transform_checkpoint_with_store_time(
        self,
        session_id: str,
        checkpoint_transform: StoreTimeCheckpointTransform,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            now = self._ownership_clock()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            current = self._checkpoints.get(session_id)
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                checkpoint_preservation._copy_checkpoint_for_transform(
                    current, session_id=session_id
                ),
                now,
            )
            if transformed is None:
                return
            self._store_checkpoint_unlocked(
                session_id,
                checkpoint_preservation._replace_checkpoint_preserving_completion_result_event_publications(
                    current,
                    copy_durable_json_object(transformed, "checkpoint"),
                    session_id=session_id,
                ),
            )
            self._sessions[session_id] = session.model_copy(update={"last_activity_at": now})

    async def replace_initial_transcript_messages(
        self,
        session_id: str,
        expected_messages: list[Message],
        replacement_messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
        checkpoint_transform: CheckpointTransform | None = None,
        runtime_suffix_count: int = 0,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        if interaction_id is None:
            raise ValueError("Initial transcript publication requires an interaction identity.")
        expected = copy_transcript_messages(expected_messages)
        replacement = copy_transcript_messages(replacement_messages)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            deferred = self._deferred_interaction_inputs.get(session_id)
            if deferred is None or deferred.interaction_id != interaction_id:
                raise RuntimeError("Deferred interaction input changed before finalization.")
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            require_deferred_initial_transcript_replacement(
                deferred,
                expected_messages=expected,
                replacement_messages=replacement,
            )
            self._register_private_authority_alias_unlocked(
                interaction_id,
                field_name="interaction_id",
                scope_session_id=session_id,
            )
            if self._transcripts.get(session_id):
                from cayu.sessions._participant_execution_identity import (
                    require_initial_execution_input,
                )

                require_initial_execution_input(
                    session,
                    self._participant_session_receipts.get(session_id),
                    self._transcripts[session_id],
                    expected,
                )
            prefix_count = session_transcript_input._initial_transcript_prefix_count(
                expected,
                replacement,
                runtime_suffix_count=runtime_suffix_count,
            )
            current_checkpoint = self._checkpoints.get(session_id)
            if checkpoint_transform is not None:
                transformed = checkpoint_transform(
                    session.model_copy(deep=True),
                    checkpoint_preservation._copy_checkpoint_for_transform(
                        current_checkpoint,
                        session_id=session_id,
                    ),
                )
                if transformed is not None:
                    current_checkpoint = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                        current_checkpoint,
                        transformed,
                        session_id=session_id,
                    )
            checkpoint = checkpoint_publication._checkpoint_after_initial_transcript_publication(
                current_checkpoint,
                interaction_id=interaction_id,
            )
            self._transcripts[session_id] = replacement
            memory_transcripts.replace_latest_role_indexes(
                self._transcript_indexes_unlocked(), session_id
            )
            memory_transcripts.replace_search_session(
                self._transcript_indexes_unlocked(), session_id, replacement
            )
            self._transcript_interaction_ids[session_id] = [None] * prefix_count + [
                interaction_id
            ] * (len(expected) + runtime_suffix_count)
            self._transcript_indices_by_interaction.pop(session_id, None)
            self._deferred_interaction_inputs.pop(session_id, None)
            if checkpoint is None:
                self._refresh_queued_dispatch_terminal_receipts_unlocked(session_id, None)
                self._checkpoints.pop(session_id, None)
            else:
                self._store_checkpoint_unlocked(session_id, checkpoint)
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": self._ownership_clock()}
            )

    async def materialize_deferred_interaction_input(
        self,
        session_id: str,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> bool:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            deferred = self._deferred_interaction_inputs.get(session_id)
            if deferred is None:
                return False
            if deferred.interaction_id != interaction_id:
                raise RuntimeError("Deferred interaction input belongs to another interaction.")
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            messages = deferred.source_messages
            self._transcripts[session_id].extend(messages)
            memory_transcripts.extend_search(
                self._transcript_indexes_unlocked(), session_id, messages
            )
            memory_transcripts.extend_attribution(
                self._transcript_indexes_unlocked(),
                session_id,
                [interaction_id] * len(messages),
            )
            self._deferred_interaction_inputs.pop(session_id, None)
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": self._ownership_clock()}
            )
            return True

    async def load_deferred_interaction_input(
        self,
        session_id: str,
    ) -> DeferredInteractionInput | None:
        return await memory_transcripts.load_deferred_interaction_input(
            self._read_transcript_state, session_id
        )

    async def append_transcript_messages_and_transform_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint_transform: CheckpointTransform,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        interaction_id = resolve_interaction_attribution(session_id, interaction_id)
        copied_messages = copy_transcript_messages(messages)
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            current = self._checkpoints.get(session_id)
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                checkpoint_preservation._copy_checkpoint_for_transform(
                    current, session_id=session_id
                ),
            )
            if transformed is None:
                raise ValueError("Checkpoint transform must return a checkpoint.")
            copied_checkpoint = checkpoint_preservation._checkpoint_transform_result_preserving_completion_result_event_publications(
                current,
                transformed,
                session_id=session_id,
            )
            if copied_messages:
                if interaction_id is not None:
                    self._register_private_authority_alias_unlocked(
                        interaction_id,
                        field_name="interaction_id",
                        scope_session_id=session_id,
                    )
                self._transcripts[session_id].extend(copied_messages)
                memory_transcripts.extend_search(
                    self._transcript_indexes_unlocked(), session_id, copied_messages
                )
                memory_transcripts.extend_attribution(
                    self._transcript_indexes_unlocked(),
                    session_id,
                    [interaction_id] * len(copied_messages),
                )
            self._store_checkpoint_unlocked(session_id, copied_checkpoint)
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": self._ownership_clock()}
            )

    async def load_transcript(self, session_id: str) -> list[Message]:
        return await memory_transcripts.load_transcript(self._read_transcript_state, session_id)

    async def load_transcript_snapshot(self, session_id: str) -> TranscriptSnapshot:
        return await memory_transcripts.load_transcript_snapshot(
            self._read_transcript_state, session_id
        )

    async def load_transcript_cursor(self, session_id: str) -> int:
        return await memory_transcripts.load_transcript_cursor(
            self._read_transcript_state, session_id
        )

    async def load_latest_transcript_message(
        self,
        session_id: str,
        *,
        role: MessageRole,
    ) -> TranscriptRecord | None:
        return await memory_transcripts.load_latest_transcript_message(
            self._read_transcript_state, session_id, role=role
        )

    async def load_latest_transcript_text(
        self,
        session_id: str,
        *,
        role: MessageRole,
        max_chars: int,
    ) -> tuple[str, bool] | None:
        return await memory_transcripts.load_latest_transcript_text(
            self._read_transcript_state, session_id, role=role, max_chars=max_chars
        )

    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
        return await memory_transcripts.query_transcript(self._read_transcript_state, query)

    async def search_transcript(
        self,
        query: TranscriptSearchQuery,
    ) -> TranscriptSearchResult:
        return await memory_transcripts.search_transcript(self._read_transcript_state, query)

    async def create_recall_receipt(self, receipt: RecallReceipt) -> RecallReceipt:
        copied = copy_recall_receipt(receipt)
        async with self._lock:
            current = self._recall_receipts.get(copied.receipt_id)
            if current is not None:
                if memory_evidence_document_bytes(
                    current, "stored recall receipt"
                ) != memory_evidence_document_bytes(copied, "recall receipt"):
                    raise RecallEvidenceConflict("Recall receipt", copied.receipt_id)
                return copy_recall_receipt(current)
            if copied.session_id not in self._sessions:
                raise KeyError(f"Session not found: {copied.session_id}")
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (copied.session_id,))
            self._recall_receipts[copied.receipt_id] = copied
            self._recall_receipt_ids_by_session.setdefault(copied.session_id, set()).add(
                copied.receipt_id
            )
            page_key = copied.created_at, copied.receipt_id
            for page_keys in (
                self._recall_receipt_page_keys_by_session.setdefault(copied.session_id, []),
                self._recall_receipt_page_keys_by_interaction.setdefault(
                    (copied.session_id, copied.interaction_id), []
                ),
                self._recall_receipt_page_keys_by_step.setdefault(
                    (copied.session_id, copied.model_step_id), []
                ),
                self._recall_receipt_page_keys_by_interaction_step.setdefault(
                    (copied.session_id, copied.interaction_id, copied.model_step_id), []
                ),
            ):
                page_keys.insert(bisect_right(page_keys, page_key), page_key)
            return copy_recall_receipt(copied)

    async def load_recall_receipt(
        self,
        session_id: str,
        receipt_id: str,
    ) -> RecallReceipt | None:
        session_id = require_memory_evidence_session_id(session_id)
        receipt_id = require_memory_evidence_id(receipt_id, "receipt_id")
        async with self._lock:
            receipt = self._recall_receipts.get(receipt_id)
            if receipt is None or receipt.session_id != session_id:
                return None
            return copy_recall_receipt(receipt)

    async def list_recall_receipts(
        self,
        query: RecallEvidenceQuery,
    ) -> RecallReceiptPage:
        copied_query = RecallEvidenceQuery.model_validate(query.model_dump(mode="python"))
        query_fingerprint = copied_query.fingerprint("receipt")
        after = (
            None
            if copied_query.cursor is None
            else decode_recall_evidence_cursor(
                copied_query.cursor,
                record_kind="receipt",
                query_fingerprint=query_fingerprint,
            )
        )
        async with self._lock:
            if copied_query.interaction_id is not None and copied_query.model_step_id is not None:
                page_keys = self._recall_receipt_page_keys_by_interaction_step.get(
                    (
                        copied_query.session_id,
                        copied_query.interaction_id,
                        copied_query.model_step_id,
                    ),
                    (),
                )
            elif copied_query.interaction_id is not None:
                page_keys = self._recall_receipt_page_keys_by_interaction.get(
                    (copied_query.session_id, copied_query.interaction_id),
                    (),
                )
            elif copied_query.model_step_id is not None:
                page_keys = self._recall_receipt_page_keys_by_step.get(
                    (copied_query.session_id, copied_query.model_step_id),
                    (),
                )
            else:
                page_keys = self._recall_receipt_page_keys_by_session.get(
                    copied_query.session_id,
                    (),
                )
            start = 0 if after is None else bisect_right(page_keys, after)
            candidates = tuple(
                self._recall_receipts[receipt_id]
                for _, receipt_id in islice(
                    page_keys,
                    start,
                    start + copied_query.limit + 1,
                )
            )
            retained: list[RecallReceipt] = []
            retained_bytes = 2
            for receipt in candidates:
                document_bytes = len(
                    memory_evidence_document_bytes(receipt, "recall receipt page item")
                )
                separator_bytes = 1 if retained else 0
                if len(retained) >= copied_query.limit or (
                    retained_bytes + separator_bytes + document_bytes > copied_query.max_bytes
                ):
                    break
                retained.append(receipt)
                retained_bytes += separator_bytes + document_bytes
            truncated = len(retained) < len(candidates)
            next_cursor = (
                encode_recall_evidence_cursor(
                    record_kind="receipt",
                    query_fingerprint=query_fingerprint,
                    created_at=retained[-1].created_at,
                    record_id=retained[-1].receipt_id,
                )
                if truncated and retained
                else None
            )
            return RecallReceiptPage(
                items=tuple(retained),
                next_cursor=next_cursor,
                truncated=truncated,
            )

    async def create_context_exposure(
        self,
        exposure: ContextExposure,
        item_exposures: tuple[RecallItemExposure, ...] = (),
    ) -> ContextExposure:
        copied = copy_context_exposure(exposure)
        copied_items = tuple(copy_recall_item_exposure(item) for item in item_exposures)
        validate_new_context_exposure(copied, copied_items)

        async with self._lock:
            current = self._context_exposures.get(copied.exposure_id)
            if current is not None:
                current_items = self._recall_item_exposures[copied.exposure_id]
                if not context_exposure_creation_matches(current, copied) or tuple(
                    memory_evidence_document_bytes(item, "stored recall item exposure")
                    for item in current_items
                ) != tuple(
                    memory_evidence_document_bytes(item, "recall item exposure")
                    for item in copied_items
                ):
                    raise RecallEvidenceConflict("Context exposure", copied.exposure_id)
                return copy_context_exposure(current)
            if copied.session_id not in self._sessions:
                raise KeyError(f"Session not found: {copied.session_id}")
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (copied.session_id,))

            attempt_key = copied.session_id, copied.model_attempt_id
            conflicting_exposure_id = self._context_exposure_by_model_attempt.get(attempt_key)
            if conflicting_exposure_id is not None:
                raise RecallEvidenceConflict("Model-attempt exposure", conflicting_exposure_id)
            provider_attempt_key = copied.session_id, copied.provider_attempt_id
            conflicting_exposure_id = self._context_exposure_by_provider_attempt.get(
                provider_attempt_key
            )
            if conflicting_exposure_id is not None:
                raise RecallEvidenceConflict(
                    "Provider-attempt exposure",
                    conflicting_exposure_id,
                )

            receipts: dict[str, RecallReceipt] = {}
            for receipt_id in copied.receipt_ids:
                receipt = self._recall_receipts.get(receipt_id)
                if receipt is None:
                    raise KeyError(f"Recall receipt not found: {receipt_id}")
                validate_context_exposure_receipt_scope(copied, receipt)
                receipts[receipt_id] = receipt
            for receipt_id in copied.carried_receipt_ids:
                receipt = self._recall_receipts.get(receipt_id)
                if receipt is None:
                    raise KeyError(f"Recall receipt not found: {receipt_id}")
                validate_context_exposure_carried_receipt_scope(copied, receipt)
            for item in copied_items:
                if not recall_item_exposure_matches_receipt_item(
                    item,
                    receipts[item.receipt_id],
                ):
                    raise ValueError(
                        "Recall item exposure differs from its immutable receipt item."
                    )

            self._context_exposures[copied.exposure_id] = copied
            self._context_exposure_ids_by_session.setdefault(copied.session_id, set()).add(
                copied.exposure_id
            )
            self._recall_item_exposures[copied.exposure_id] = copied_items
            self._context_exposure_by_model_attempt[attempt_key] = copied.exposure_id
            self._context_exposure_by_provider_attempt[provider_attempt_key] = copied.exposure_id
            page_key = copied.created_at, copied.exposure_id
            for page_keys in (
                self._context_exposure_page_keys_by_session.setdefault(copied.session_id, []),
                self._context_exposure_page_keys_by_interaction.setdefault(
                    (copied.session_id, copied.interaction_id), []
                ),
                self._context_exposure_page_keys_by_step.setdefault(
                    (copied.session_id, copied.model_step_id), []
                ),
                self._context_exposure_page_keys_by_interaction_step.setdefault(
                    (copied.session_id, copied.interaction_id, copied.model_step_id), []
                ),
            ):
                page_keys.insert(bisect_right(page_keys, page_key), page_key)
            return copy_context_exposure(copied)

    async def load_context_exposure(
        self,
        session_id: str,
        exposure_id: str,
    ) -> ContextExposure | None:
        session_id = require_memory_evidence_session_id(session_id)
        exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
        async with self._lock:
            exposure = self._context_exposures.get(exposure_id)
            if exposure is None or exposure.session_id != session_id:
                return None
            return copy_context_exposure(exposure)

    async def load_recall_item_exposures(
        self,
        session_id: str,
        exposure_id: str,
    ) -> tuple[RecallItemExposure, ...]:
        session_id = require_memory_evidence_session_id(session_id)
        exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
        async with self._lock:
            exposure = self._context_exposures.get(exposure_id)
            if exposure is None or exposure.session_id != session_id:
                return ()
            if len(self._recall_item_exposures[exposure_id]) > MAX_RECALL_RECEIPT_ITEMS:
                raise ValueError("Stored recall item exposures exceed their count bound.")
            return tuple(
                copy_recall_item_exposure(item) for item in self._recall_item_exposures[exposure_id]
            )

    async def list_context_exposures(
        self,
        query: RecallEvidenceQuery,
    ) -> ContextExposurePage:
        copied_query = RecallEvidenceQuery.model_validate(query.model_dump(mode="python"))
        query_fingerprint = copied_query.fingerprint("exposure")
        after = (
            None
            if copied_query.cursor is None
            else decode_recall_evidence_cursor(
                copied_query.cursor,
                record_kind="exposure",
                query_fingerprint=query_fingerprint,
            )
        )
        async with self._lock:
            if copied_query.interaction_id is not None and copied_query.model_step_id is not None:
                page_keys = self._context_exposure_page_keys_by_interaction_step.get(
                    (
                        copied_query.session_id,
                        copied_query.interaction_id,
                        copied_query.model_step_id,
                    ),
                    (),
                )
            elif copied_query.interaction_id is not None:
                page_keys = self._context_exposure_page_keys_by_interaction.get(
                    (copied_query.session_id, copied_query.interaction_id),
                    (),
                )
            elif copied_query.model_step_id is not None:
                page_keys = self._context_exposure_page_keys_by_step.get(
                    (copied_query.session_id, copied_query.model_step_id),
                    (),
                )
            else:
                page_keys = self._context_exposure_page_keys_by_session.get(
                    copied_query.session_id,
                    (),
                )
            start = 0 if after is None else bisect_right(page_keys, after)
            candidates = tuple(
                self._context_exposures[exposure_id]
                for _, exposure_id in islice(
                    page_keys,
                    start,
                    start + copied_query.limit + 1,
                )
            )
            retained: list[ContextExposure] = []
            retained_bytes = 2
            for exposure in candidates:
                document_bytes = len(
                    memory_evidence_document_bytes(exposure, "context exposure page item")
                )
                separator_bytes = 1 if retained else 0
                if len(retained) >= copied_query.limit or (
                    retained_bytes + separator_bytes + document_bytes > copied_query.max_bytes
                ):
                    break
                retained.append(exposure)
                retained_bytes += separator_bytes + document_bytes
            truncated = len(retained) < len(candidates)
            next_cursor = (
                encode_recall_evidence_cursor(
                    record_kind="exposure",
                    query_fingerprint=query_fingerprint,
                    created_at=retained[-1].created_at,
                    record_id=retained[-1].exposure_id,
                )
                if truncated and retained
                else None
            )
            return ContextExposurePage(
                items=tuple(retained),
                next_cursor=next_cursor,
                truncated=truncated,
            )

    async def transition_context_exposure(
        self,
        session_id: str,
        exposure_id: str,
        request: ContextExposureTransitionRequest,
    ) -> ContextExposure:
        session_id = require_memory_evidence_session_id(session_id)
        exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
        copied_request = ContextExposureTransitionRequest.model_validate(
            request.model_dump(mode="python")
        )
        async with self._lock:
            exposure = self._context_exposures.get(exposure_id)
            if exposure is None or exposure.session_id != session_id:
                raise KeyError(f"Context exposure not found: {exposure_id}")
            replay = next(
                (
                    transition
                    for transition in exposure.transitions
                    if transition.transition_id == copied_request.transition_id
                ),
                None,
            )
            if replay is not None:
                if not context_exposure_transition_replays(exposure, copied_request):
                    raise RecallEvidenceConflict(
                        "Context exposure transition",
                        copied_request.transition_id,
                    )
                return copy_context_exposure(exposure)
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            updated = append_context_exposure_transition(exposure, copied_request)
            self._context_exposures[exposure_id] = updated
            return copy_context_exposure(updated)

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            self._store_checkpoint_unlocked(
                session_id,
                checkpoint_preservation._replace_checkpoint_preserving_completion_result_event_publications(
                    self._checkpoints.get(session_id),
                    state,
                    session_id=session_id,
                ),
            )
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": self._ownership_clock()}
            )

    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.values():
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            current = self._checkpoints.get(session_id)
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                checkpoint_preservation._copy_checkpoint_for_transform(
                    current, session_id=session_id
                ),
            )
            if transformed is None:
                return
            self._store_checkpoint_unlocked(
                session_id,
                checkpoint_preservation._replace_checkpoint_preserving_completion_result_event_publications(
                    current,
                    copy_durable_json_object(transformed, "checkpoint"),
                    session_id=session_id,
                ),
            )
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": self._ownership_clock()}
            )

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        from cayu.resource_access import current_data_bounds

        access_bounds = await current_data_bounds()
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if access_bounds is not None:
                access_bounds.require_action(self._sessions.get(session_id), "inspect_state")
            checkpoint = self._checkpoints.get(session_id)
            if checkpoint is None:
                return None
            return deepcopy(checkpoint)

    async def load_execution_snapshot_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            checkpoint = self._checkpoints.get(session_id)
            return (
                None
                if checkpoint is None
                else deepcopy(
                    {
                        key: value
                        for key, value in checkpoint.items()
                        if key in {"checkpoint_schema_version", "execution_snapshots"}
                    }
                )
            )

    async def load_interruption_cascade_marker(
        self,
        session_id: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            checkpoint = self._checkpoints.get(session_id)
            if checkpoint_root_guard is not None:
                checkpoint_root_guard.validate(
                    session_id,
                    checkpoint_root_field_projection(
                        checkpoint,
                        checkpoint_root_guard.key,
                    ),
                )
            marker = (
                _INTERRUPTION_CASCADE_MISSING
                if checkpoint is None or "pending_interruption_cascade" not in checkpoint
                else checkpoint["pending_interruption_cascade"]
            )
            return _project_interruption_cascade_marker(marker)


def _audit_resolution_actor(actor: ResolutionActor | None) -> ResolutionActor | None:
    if actor is None:
        return None
    return ResolutionActor(
        subject=actor.subject,
        tenant=actor.tenant,
        source=actor.source,
    )


def _validate_equivalent_message_delivery(
    existing: _InMemoryMessageDeliveryRecord,
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
