"""In-process session storage with owned state, locking and atomic mutations."""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
from bisect import bisect_left
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import SecretStr

from cayu._clock import utc_clock, utc_duration_cutoff
from cayu._resource_store_surface import model_store_surface
from cayu._validation import (
    copy_durable_json_object,
    copy_label_map,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.budgets.aggregates import (
    UsageRollupStoreResult,
)
from cayu.budgets.base import BudgetReservationIdentityConflict
from cayu.budgets.pricing import PriceBook
from cayu.collaboration.peer_content import (
    PeerAppendKey,
    PeerContentAppendRequest,
    PeerContentExposureReceipt,
    PeerContentExposureRequest,
    PeerContentReceipt,
)
from cayu.events import Event, EventType, copy_event
from cayu.execution_profiles import (
    ExecutionProfileDecision,
    ExecutionProfileIdentity,
    ExecutionProfileRejectionResult,
)
from cayu.execution_units import ToolRoundIdentity
from cayu.memory.evidence import (
    ContextExposure,
    ContextExposurePage,
    ContextExposureTransitionRequest,
    RecallEvidenceQuery,
    RecallItemExposure,
    RecallReceipt,
    RecallReceiptPage,
)
from cayu.messages import Message, MessageRole, detach_message
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
)
from cayu.runtime.service_manifest import RuntimeStoreDurability
from cayu.session_metadata import SummaryTone, summary_metadata
from cayu.sessions import _checkpoint_mutations as checkpoint_mutation_rules
from cayu.sessions import _checkpoint_preservation as checkpoint_preservation
from cayu.sessions import _child_lifecycle_rules as child_lifecycle_rules
from cayu.sessions import _closure_records as closure_records
from cayu.sessions import _completion_finalization as completion_finalization
from cayu.sessions import _execution_profile_rules as session_profile_rules
from cayu.sessions import _interaction_receipts as interaction_receipts
from cayu.sessions import _interaction_transitions as interaction_rules
from cayu.sessions import _invocation_scope as invocation_scope
from cayu.sessions import _memory_context_views as memory_context_views
from cayu.sessions import _memory_event_delivery as memory_event_delivery
from cayu.sessions import _memory_event_reads as memory_event_reads
from cayu.sessions import _memory_evidence as memory_evidence
from cayu.sessions import _memory_peer_content as memory_peer_content
from cayu.sessions import _memory_queued_messages as memory_queued_messages
from cayu.sessions import _memory_session_creation as memory_session_creation
from cayu.sessions import _memory_session_queries as memory_session_queries
from cayu.sessions import _memory_tool_grants as memory_tool_grants
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
from cayu.sessions import event_delivery as session_event_rules
from cayu.sessions import forks as session_fork_rules
from cayu.sessions import inspection as session_inspection_rules
from cayu.sessions import lineage as lineage_rules
from cayu.sessions import records as session_record_rules
from cayu.sessions import recovery as session_recovery_rules
from cayu.sessions import requests as session_request_contracts
from cayu.sessions import terminal_evidence as terminal_session_evidence
from cayu.sessions._checkpoint_mutations import (
    RuntimePublicationMutation,
    SessionRuntimePublicationConflict,
)
from cayu.sessions._event_snapshot import EventSnapshotConsumer
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
)
from cayu.sessions._invocation_terminal_decision import InvocationTerminalDecision
from cayu.sessions._model_completion_publication import ModelCompletionStageSettlementRequest
from cayu.sessions._model_stage import (
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
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
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectClaim,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectHealth,
    PersistedEventSideEffectPage,
    PersistedEventSideEffectQuery,
    PersistedEventSideEffectStatus,
    _copy_workflow_step_reservation,
)
from cayu.sessions.event_queries import (
    EventQuery,
)
from cayu.sessions.exports import _require_session_export_target
from cayu.sessions.forks import (
    ForkCheckpointAuthorityDecoder,
    ForkSystemPromptReplacement,
    ProfiledSessionForkResult,
    SessionForkProfileRelationship,
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
    SessionMessageDeliveryBatch,
    SessionMessageDeliveryMode,
    SessionMessageInspection,
    SessionMessageQuery,
    SessionMessageSource,
    SessionQueuedMessage,
    SessionQueuedMessagesPending,
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
    copy_session_runtime_identity,
    copy_session_user_metadata,
    update_session_annotations,
)
from cayu.sessions.recovery import (
    _INTERRUPTION_CASCADE_MISSING,
    _project_interruption_cascade_marker,
)
from cayu.sessions.requests import (
    RunRequest,
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
    DeferredInteractionInput,
)
from cayu.sessions.transcript_queries import (
    BoundedTranscriptWindow,
    ForkTranscriptValidator,
    TranscriptPage,
    TranscriptQuery,
    TranscriptSearchQuery,
    TranscriptSearchResult,
    TranscriptSnapshot,
)
from cayu.sessions.usage import UsageRollupQuery
from cayu.storage._creation_fence import MemoryCreationFenceMixin
from cayu.storage._session_execution import MemorySessionExecutionMixin
from cayu.tools.exposure import ToolCapabilityCeiling
from cayu.tools.grants import (
    TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    TargetedToolGrantIssueResult,
    TargetedToolGrantReconstructionResult,
    TargetedToolGrantRecord,
    TargetedToolGrantStateSnapshot,
    TargetedToolUseBinding,
    TargetedToolUseRequest,
    TargetedToolUseResult,
    targeted_tool_grant_with_active_reference,
)
from cayu.workflows.base import WORKFLOW_ATTEMPT_EVENT_TYPE

if TYPE_CHECKING:
    from cayu.collaboration.participants import ParticipantRef as _ParticipantRef
    from cayu.runtime._zero_work_interruption import (
        ZeroWorkInterruptionPublication,
        ZeroWorkInterruptionRequest,
    )
    from cayu.sessions._access_rules import SessionReadBounds, _SessionAccessBounds
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

#: Closure receipts an in-memory store keeps before evicting the oldest.
DEFAULT_MEMORY_CLOSURE_RECEIPT_LIMIT = 10_000


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
    """In-process session store for tests, local development, and examples.

    Deleting a session through closure (``CayuApp.delete_session`` or
    ``CayuApp.erase_session_closure``) removes its records and keeps a closure
    receipt that retires the session id and lets the same erase be replayed.
    The store keeps at most ``closure_receipt_limit`` receipts and evicts the
    oldest first, so a long-running process that deletes finished sessions
    stays bounded. Eviction frees this store's identity reservation and drops
    erase replay evidence. Dependent stores may still retire the identity;
    application admission refuses retained task-deletion claims. Use fresh IDs
    for new sessions. Durable stores keep every receipt.
    """

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
        self, bounds: SessionReadBounds, query: SessionQuery
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

    async def _access_load_session(self, bounds: SessionReadBounds, session_id: str) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        from cayu.sessions._access_rules import require_session_read

        async with self._lock:
            sessions = self._sessions
            return require_session_read(bounds, sessions.get(session_id), sessions.get).model_copy(
                deep=True
            )

    peer_content_version: ClassVar[int | None] = 1

    session_export_version: ClassVar[int] = 1
    session_continuation_version: ClassVar[int] = 1
    external_wait_version: ClassVar[int] = 1
    _producer_attachment_version: ClassVar[int] = 1
    supports_session_closure_receipts: ClassVar[bool] = True
    supports_session_closure_detachment: ClassVar[bool] = True
    supports_session_closure_recursive_deletion: ClassVar[bool] = True
    supports_session_closure_finished_only: ClassVar[bool] = True
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
            return None if receipt is None else json.loads(receipt)

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

    async def claim_session_closure_progress(
        self, progress: dict[str, Any], *, finished_only: bool = False
    ) -> None:
        progress = deepcopy(progress)
        async with self._lock:
            if finished_only:
                root = self._sessions.get(progress["root_session_id"])
                if root is None or root.status not in {
                    SessionStatus.COMPLETED,
                    SessionStatus.FAILED,
                }:
                    raise ValueError("Only completed or failed sessions can be deleted.")
            key = (progress["root_session_id"], progress["plan_id"])
            existing = self._session_closure_progress.get(key)
            if existing is not None:
                closure_records._validate_closure_progress_update(existing, progress)
                return
            targets = closure_records._closure_progress_targets(progress)
            for owner in self._session_closure_progress.owners(targets):
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
        for owner in self._session_closure_progress.owners((session_id,)):
            closure_records._check_closure_lineage_owner(owner, (session_id,))
        if session_id in self._session_closure_receipt_counts:
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
            for owner in self._session_closure_progress.owners(child_session_ids):
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
    supports_supervision_events: ClassVar[bool] = True
    supports_fork_controller_reads: ClassVar[bool] = True
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
        closure_receipt_limit: int = DEFAULT_MEMORY_CLOSURE_RECEIPT_LIMIT,
    ) -> None:
        from cayu.runtime._cost_accounting_refresh import CostAccountingAuthority
        from cayu.runtime._usage_accounting import SessionUsageCache

        if type(closure_receipt_limit) is not int or closure_receipt_limit < 1:
            raise ValueError("closure_receipt_limit must be a positive integer.")
        self._closure_receipt_limit = closure_receipt_limit

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
        # Nothing outlives the process, so each store is a store of its own.
        self._store_identity = secrets.token_hex(16)
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
        # Compact JSON text, oldest first; bounded by closure_receipt_limit.
        self._session_closure_receipts: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._session_closure_receipt_counts: dict[str, int] = {}
        self._session_closure_progress = closure_records.ClosureProgressIndex()
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
        self._session_message_delivery_records: dict[
            str, memory_queued_messages.MemoryMessageDeliveryRecord
        ] = {}
        self._session_message_terminal_receipts: dict[tuple[str, str], dict[str, Any]] = {}
        # Event-owned lookup index, not a copy of acceptance provenance. Retain
        # at most two IDs: a second canonical acceptance makes quarantine ambiguous.
        self._session_message_acceptance_event_ids: dict[tuple[str, str], set[str]] = {}
        self._session_message_ordering = memory_queued_messages.MemoryMessageOrdering()

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

    @asynccontextmanager
    async def _event_delivery_state(
        self,
    ) -> AsyncIterator[memory_event_delivery.MemoryEventDeliveryState]:
        async with self._lock:
            yield memory_event_delivery.MemoryEventDeliveryState(
                persisted_event_side_effect_deliveries=self._persisted_event_side_effect_deliveries,
                event_records_by_id=self._event_records_by_id,
                session_closure_progress=self._session_closure_progress,
                ownership_clock=self._ownership_clock,
            )

    @asynccontextmanager
    async def _recall_evidence_state(
        self,
    ) -> AsyncIterator[memory_evidence.MemoryRecallState]:
        async with self._lock:
            yield memory_evidence.MemoryRecallState(
                recall_receipt_ids_by_session=self._recall_receipt_ids_by_session,
                recall_receipt_page_keys_by_interaction=self._recall_receipt_page_keys_by_interaction,
                recall_receipt_page_keys_by_interaction_step=self._recall_receipt_page_keys_by_interaction_step,
                recall_receipt_page_keys_by_session=self._recall_receipt_page_keys_by_session,
                recall_receipt_page_keys_by_step=self._recall_receipt_page_keys_by_step,
                recall_receipts=self._recall_receipts,
                session_closure_progress=self._session_closure_progress,
                sessions=self._sessions,
            )

    @asynccontextmanager
    async def _context_exposure_state(
        self,
    ) -> AsyncIterator[memory_evidence.MemoryContextExposureState]:
        async with self._lock:
            yield memory_evidence.MemoryContextExposureState(
                context_exposure_by_model_attempt=self._context_exposure_by_model_attempt,
                context_exposure_by_provider_attempt=self._context_exposure_by_provider_attempt,
                context_exposure_ids_by_session=self._context_exposure_ids_by_session,
                context_exposure_page_keys_by_interaction=self._context_exposure_page_keys_by_interaction,
                context_exposure_page_keys_by_interaction_step=self._context_exposure_page_keys_by_interaction_step,
                context_exposure_page_keys_by_session=self._context_exposure_page_keys_by_session,
                context_exposure_page_keys_by_step=self._context_exposure_page_keys_by_step,
                context_exposures=self._context_exposures,
                recall_item_exposures=self._recall_item_exposures,
                recall_receipts=self._recall_receipts,
                session_closure_progress=self._session_closure_progress,
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

    @asynccontextmanager
    async def _write_transcript_state(
        self,
    ) -> AsyncIterator[memory_transcripts.MemoryTranscriptWriteState]:
        async with self._lock:
            yield memory_transcripts.MemoryTranscriptWriteState(
                indexes=self._transcript_indexes_unlocked(),
                sessions=self._sessions,
                deferred_interaction_inputs=self._deferred_interaction_inputs,
                checkpoints=self._checkpoints,
                session_closure_progress=self._session_closure_progress,
                participant_session_receipts=self._participant_session_receipts,
                ownership_clock=self._ownership_clock,
                register_private_alias=self._register_private_authority_alias_unlocked,
                store_checkpoint=self._store_checkpoint_unlocked,
                refresh_queued_receipts=self._refresh_queued_dispatch_terminal_receipts_unlocked,
            )

    @asynccontextmanager
    async def _tool_grant_state(
        self,
    ) -> AsyncIterator[memory_tool_grants.MemoryToolGrantState[_PreparedInMemoryEventAppend]]:
        async with self._lock:
            yield memory_tool_grants.MemoryToolGrantState(
                event_records_by_id=self._event_records_by_id,
                interaction_event_records=self._interaction_event_records,
                latest_interaction_event_records_by_sequence=self._latest_interaction_event_records_by_sequence,
                public_authority_aliases=self._public_authority_aliases,
                session_closure_progress=self._session_closure_progress,
                sessions=self._sessions,
                targeted_tool_grant_ids_by_interaction=self._targeted_tool_grant_ids_by_interaction,
                targeted_tool_grant_ids_by_request=self._targeted_tool_grant_ids_by_request,
                targeted_tool_grant_ids_by_session=self._targeted_tool_grant_ids_by_session,
                targeted_tool_grant_ids_by_tool=self._targeted_tool_grant_ids_by_tool,
                targeted_tool_grant_refs=self._targeted_tool_grant_refs,
                targeted_tool_grants=self._targeted_tool_grants,
                targeted_tool_use_by_invocation=self._targeted_tool_use_by_invocation,
                targeted_tool_use_by_outer_call=self._targeted_tool_use_by_outer_call,
                targeted_tool_uses=self._targeted_tool_uses,
                prepare_events=self._prepare_event_append_unlocked,
                apply_events=self._apply_event_append_unlocked,
            )

    @asynccontextmanager
    async def _queued_message_state(
        self,
    ) -> AsyncIterator[memory_queued_messages.MemoryQueuedMessageState]:
        async with self._lock:
            yield memory_queued_messages.MemoryQueuedMessageState(
                append_events=self._append_events_unlocked,
                checkpoints=self._checkpoints,
                event_records_by_id=self._event_records_by_id,
                ordering=self._session_message_ordering,
                ownership_clock=self._ownership_clock,
                pending_session_messages=self._pending_session_messages,
                queued_session_messages_by_idempotency=self._queued_session_messages_by_idempotency,
                reject_new_work_after_steering=self._reject_new_work_after_steering_unlocked,
                session_closure_progress=self._session_closure_progress,
                session_message_acceptance_event_ids=self._session_message_acceptance_event_ids,
                session_message_delivery_records=self._session_message_delivery_records,
                session_message_terminal_receipts=self._session_message_terminal_receipts,
                session_operation_records=self._session_operation_records,
                sessions=self._sessions,
                store_checkpoint=self._store_checkpoint_unlocked,
                transcript_indexes=self._transcript_indexes_unlocked,
                transcript_interaction_ids=self._transcript_interaction_ids,
                transcripts=self._transcripts,
            )

    @asynccontextmanager
    async def _peer_content_state(
        self,
    ) -> AsyncIterator[memory_peer_content.MemoryPeerContentState]:
        async with self._lock:
            yield memory_peer_content.MemoryPeerContentState(
                append_events=self._append_events_unlocked,
                checkpoints=self._checkpoints,
                ownership_clock=self._ownership_clock,
                participant_session_bindings=self._participant_session_bindings,
                peer_content_attempts=self._peer_content_attempts,
                peer_content_deleted_targets=self._peer_content_deleted_targets,
                peer_content_exposures=self._peer_content_exposures,
                peer_content_operation_commitments=self._peer_content_operation_commitments,
                peer_content_operations=self._peer_content_operations,
                peer_content_receipts=self._peer_content_receipts,
                peer_content_requests=self._peer_content_requests,
                pending_session_messages=self._pending_session_messages,
                queued_session_messages_by_idempotency=self._queued_session_messages_by_idempotency,
                session_creation_decisions=self._session_creation_decisions,
                ordering=self._session_message_ordering,
                session_operation_records=self._session_operation_records,
                sessions=self._sessions,
                transcripts=self._transcripts,
            )

    @asynccontextmanager
    async def _context_view_source_state(
        self,
    ) -> AsyncIterator[memory_context_views.MemoryContextViewSourceState]:
        async with self._lock:
            yield memory_context_views.MemoryContextViewSourceState(
                checkpoints=self._checkpoints,
                event_records_by_id=self._event_records_by_id,
                participant_session_bindings=self._participant_session_bindings,
                pending_session_messages=self._pending_session_messages,
                session_closure_progress=self._session_closure_progress,
                session_event_records=self._session_event_records,
                session_operation_records=self._session_operation_records,
                sessions=self._sessions,
                transcript_interaction_ids=self._transcript_interaction_ids,
                transcripts=self._transcripts,
                validate_publication_material=self._validate_runtime_publication_material_unlocked,
            )

    @asynccontextmanager
    async def _context_view_state(
        self,
    ) -> AsyncIterator[memory_context_views.MemoryContextViewState]:
        async with self._lock:
            yield memory_context_views.MemoryContextViewState(
                context_lifecycle_events=self._context_lifecycle_events,
                context_ownership_operation_commitments=self._context_ownership_operation_commitments,
                context_ownership_operation_receipts=self._context_ownership_operation_receipts,
                context_selection_exclusions=self._context_selection_exclusions,
                context_selection_receipts=self._context_selection_receipts,
                context_selection_request_commitments=self._context_selection_request_commitments,
                ownership_clock=self._ownership_clock,
                participant_session_bindings=self._participant_session_bindings,
                restore_selection=self._restore_context_selection_unlocked,
                context_publication_keys=self._context_publication_keys,
                context_views=self._context_views,
                sessions=self._sessions,
            )

    def _restore_context_selection_unlocked(
        self, snapshot: memory_context_views.MemoryContextSelectionSnapshot
    ) -> None:
        self._context_selection_receipts = snapshot.receipts
        self._context_selection_request_commitments = snapshot.commitments
        self._context_lifecycle_events = snapshot.events

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
        return await memory_tool_grants.issue_targeted_tool_grants(
            self._tool_grant_state,
            session_id,
            expected_run_epoch=expected_run_epoch,
            records=records,
            events=events,
            get_codec=lambda: self.public_authority_alias_codec,
        )

    async def list_targeted_tool_grants(
        self,
        session_id: str,
        *,
        interaction_id: str | None = None,
        limit: int = TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    ) -> tuple[TargetedToolGrantRecord, ...]:
        return await memory_tool_grants.list_targeted_tool_grants(
            self._tool_grant_state,
            session_id,
            interaction_id=interaction_id,
            limit=limit,
            get_codec=lambda: self.public_authority_alias_codec,
        )

    async def load_targeted_tool_grant_state(
        self,
        session_id: str,
    ) -> TargetedToolGrantStateSnapshot:
        return await memory_tool_grants.load_targeted_tool_grant_state(
            self._tool_grant_state,
            session_id,
            get_codec=lambda: self.public_authority_alias_codec,
        )

    async def bind_targeted_tool_grant_use(
        self,
        request: TargetedToolUseRequest,
        *,
        observed_at: datetime,
    ) -> TargetedToolUseResult:
        return await memory_tool_grants.bind_targeted_tool_grant_use(
            self._tool_grant_state,
            request,
            observed_at=observed_at,
            get_codec=lambda: self.public_authority_alias_codec,
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
        return await memory_tool_grants.revoke_targeted_tool_grant(
            self._tool_grant_state,
            tool_ref,
            session_id=session_id,
            expected_run_epoch=expected_run_epoch,
            reason=reason,
            revoked_at=revoked_at,
            get_codec=lambda: self.public_authority_alias_codec,
        )

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
        return await memory_tool_grants.reconstruct_targeted_tool_grants(
            self._tool_grant_state,
            session_id,
            expected_run_epoch=expected_run_epoch,
            interaction_id=interaction_id,
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
        )

    @asynccontextmanager
    async def _session_creation_state(
        self,
    ) -> AsyncIterator[
        memory_session_creation.MemorySessionCreationState[
            _PreparedInMemoryEventAppend, _PreparedInMemoryCheckpointStore
        ]
    ]:
        async with self._lock:
            yield memory_session_creation.MemorySessionCreationState(
                apply_checkpoint_store=self._apply_checkpoint_store_unlocked,
                apply_event_append=self._apply_event_append_unlocked,
                append_events=self._append_events_unlocked,
                checkpoints=self._checkpoints,
                child_lifecycle_event_records=self._child_lifecycle_event_records,
                context_selection_receipts=self._context_selection_receipts,
                deferred_interaction_inputs=self._deferred_interaction_inputs,
                event_ids=self._event_ids,
                events=self._events,
                index_session_parent=self._index_session_parent_unlocked,
                interaction_event_records=self._interaction_event_records,
                latest_interaction_event_records=self._latest_interaction_event_records,
                latest_interaction_event_records_by_sequence=self._latest_interaction_event_records_by_sequence,
                latest_transcript_indexes_by_role=self._latest_transcript_indexes_by_role,
                ownership_clock=self._ownership_clock,
                participant_session_bindings=self._participant_session_bindings,
                participant_session_ids_by_key=self._participant_session_ids_by_key,
                participant_session_receipts=self._participant_session_receipts,
                participant_session_request_commitments=self._participant_session_request_commitments,
                pending_action_event_records=self._pending_action_event_records,
                prepare_checkpoint_store=self._prepare_checkpoint_store_unlocked,
                prepare_event_append=self._prepare_event_append_unlocked,
                refresh_child_lifecycle_candidate=self._refresh_child_lifecycle_candidate_unlocked,
                register_private_authority_alias=self._register_private_authority_alias_unlocked,
                require_available_closure_identity=self._require_available_closure_identity_unlocked,
                require_external_creation=self._require_external_creation_unlocked,
                session_closure_progress=self._session_closure_progress,
                session_creation_decisions=self._session_creation_decisions,
                session_event_records=self._session_event_records,
                session_operation_records=self._session_operation_records,
                sessions=self._sessions,
                store_checkpoint=self._store_checkpoint_unlocked,
                supports_model_failover_stage_protocol=self._supports_model_failover_stage_protocol,
                transcript_indexes=self._transcript_indexes_unlocked,
                transcript_interaction_ids=self._transcript_interaction_ids,
                transcript_search_documents=self._transcript_search_documents,
                transcripts=self._transcripts,
            )

    @asynccontextmanager
    async def _participant_read_state(
        self,
    ) -> AsyncIterator[memory_session_creation.MemoryParticipantState]:
        async with self._lock:
            yield memory_session_creation.MemoryParticipantState(
                sessions=self._sessions,
                participant_session_bindings=self._participant_session_bindings,
                participant_session_ids_by_key=self._participant_session_ids_by_key,
                participant_session_receipts=self._participant_session_receipts,
                participant_session_request_commitments=self._participant_session_request_commitments,
            )

    @asynccontextmanager
    async def _participant_creation_state(
        self,
    ) -> AsyncIterator[memory_session_creation.MemoryParticipantCreationState]:
        # Match deletion: participant serialization precedes the store lock.
        async with self._participant_creation_lock:
            yield memory_session_creation.MemoryParticipantCreationState(
                sessions=self._sessions,
                participant_session_bindings=self._participant_session_bindings,
                participant_session_ids_by_key=self._participant_session_ids_by_key,
                participant_session_receipts=self._participant_session_receipts,
                participant_session_request_commitments=self._participant_session_request_commitments,
                create=self.create,
                read_creation_decision=self.read_session_creation_decision,
                created_receipt=lambda session_id: self._participant_session_receipts.get(
                    session_id
                ),
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
        return await memory_session_creation.create(
            self._session_creation_state,
            request,
            identity=identity,
            interaction_started_event=interaction_started_event,
            interaction_source_messages=interaction_source_messages,
            checkpoint_transform=checkpoint_transform,
            result_checkpoint_transform=result_checkpoint_transform,
            operation_initializer=operation_initializer,
            participant_binding_factory=participant_binding_factory,
            participant_request_commitment=participant_request_commitment,
            participant_provenance=participant_provenance,
            recipient_selection=recipient_selection,
            creation_target=creation_target,
            recipient_receipt_validator=recipient_receipt_validator,
        )

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
        return await memory_session_creation.create_participant_owned_session(
            self._participant_creation_state,
            creation_request,
            resolved_request=resolved_request,
            identity=identity,
            binding_factory=binding_factory,
            recipient_provenance=recipient_provenance,
            recipient_selection=recipient_selection,
            creation_target=creation_target,
            recipient_receipt_validator=recipient_receipt_validator,
        )

    async def lookup_participant_session_creation(
        self, creation_request: ParticipantSessionCreationRequest
    ) -> tuple[Session, ParticipantSessionCreationReceipt] | None:
        return await memory_session_creation.lookup_participant_session_creation(
            self._participant_creation_state,
            creation_request,
        )

    async def load_participant_session_binding(
        self, session_id: str
    ) -> ParticipantSessionBinding | None:
        return await memory_session_creation.load_participant_session_binding(
            self._participant_read_state,
            session_id,
        )

    async def _scan_participant_session_bindings(self, participant, *, after=None, limit=32):
        return await memory_session_creation.scan_participant_session_bindings(
            self._participant_read_state,
            participant,
            after=after,
            limit=limit,
        )

    async def load_participant_session_creation_receipt(
        self, session_id: str
    ) -> ParticipantSessionCreationReceipt | None:
        return await memory_session_creation.load_participant_session_creation_receipt(
            self._participant_read_state,
            session_id,
        )

    async def capture_context_view_publication_source(
        self, session_id: str
    ) -> ContextViewPublicationSource:
        return (await self._capture_completed_turn_snapshot(session_id)).publication

    async def _capture_completed_turn_snapshot(self, session_id: str) -> CompletedTurnSnapshot:
        return await memory_context_views.capture_completed_turn_snapshot(
            self._context_view_source_state,
            session_id,
        )

    async def publish_context_view(
        self, manifest: _ContextViewManifest, *, publication_key: str
    ) -> _ContextViewManifest:
        return await memory_context_views.publish_context_view(
            self._context_view_state,
            manifest,
            publication_key=publication_key,
        )

    async def lookup_context_view_publication(
        self, publication_key: str
    ) -> _ContextViewManifest | None:
        return await memory_context_views.lookup_context_view_publication(
            self._context_view_state,
            publication_key,
        )

    async def select_context_view(
        self, request: _ContextViewSelectionRequest
    ) -> _ContextViewSelectionReceipt:
        return await self._select_context_view(request)

    async def _select_context_view(self, request, *, target=None):
        return await memory_context_views.select_context_view(
            self._context_view_state,
            request,
            target=target,
        )

    async def _exclude_context_view_selection(self, request, *, authority):
        return await self._context_selection_control(request, authority=authority, exclude=True)

    async def _reserve_context_view_selection_control(self, request, *, authority):
        return await self._context_selection_control(request, authority=authority, exclude=False)

    async def _context_selection_control(
        self, request, *, authority, exclude, target=None, register=False
    ):
        return await memory_context_views.control_selection(
            self._context_view_state,
            request,
            authority=authority,
            exclude=exclude,
            target=target,
            register=register,
        )

    async def _read_context_view_retention(self, target):
        return await memory_context_views.read_retention(
            self._context_view_state,
            target,
        )

    async def read_context_view_selection_decision(self, request):
        return await memory_context_views.read_context_view_selection_decision(
            self._context_view_state,
            request,
        )

    async def lookup_context_view_selection(self, selection_key: str):
        return await memory_context_views.lookup_context_view_selection(
            self._context_view_state,
            selection_key,
        )

    async def transition_context_view_ownership(
        self, request: _ContextViewOwnershipRequest
    ) -> _ContextViewSelectionReceipt:
        return await self._transition_context_view_ownership(request)

    async def _transition_context_view_ownership(self, request, *, target=None):
        return await memory_context_views.transition_ownership(
            self._context_view_state,
            request,
            target=target,
        )

    async def validate_context_view_source_closure(self, session_id: str) -> None:
        return await memory_context_views.validate_context_view_source_closure(
            self._context_view_state,
            session_id,
        )

    async def validate_context_view_compaction(
        self, session_id: str, expected_transcript_cursor: int
    ) -> None:
        return await memory_context_views.validate_context_view_compaction(
            self._context_view_state,
            session_id,
            expected_transcript_cursor,
        )

    async def read_context_view_lifecycle_events(
        self, view_id: str, *, limit: int = 256, owner_participant: _ParticipantRef | None = None
    ) -> tuple[_ContextViewLifecycleEvent, ...]:
        return await memory_context_views.read_context_view_lifecycle_events(
            self._context_view_state,
            view_id,
            limit=limit,
            owner_participant=owner_participant,
        )

    async def read_context_view(
        self, view_id: str, *, source_session_id: str
    ) -> _ContextViewReadback:
        return await memory_context_views.read_context_view(
            self._context_view_state,
            view_id,
            source_session_id=source_session_id,
        )

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
        return await memory_session_creation.create_fork(
            self._session_creation_state,
            source_session_id=source_session_id,
            fork=fork,
            source_statuses=source_statuses,
            transcript_cursor=transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            system_prompt_replacement=system_prompt_replacement,
            expected_source_run_epoch=expected_source_run_epoch,
            transcript_validator=transcript_validator,
            profile_relationship=profile_relationship,
            events=events,
            checkpoint_authority_decoder=checkpoint_authority_decoder,
            operation_initializer=operation_initializer,
        )

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
            for owner in self._session_closure_progress.owners((session_id,)):
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
                for owner in self._session_closure_progress.owners((child_id,)):
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
                if existing is not None and json.loads(existing) != closure_receipt:
                    raise ValueError("Session closure receipt identity conflict.")
                self._record_session_closure_receipt_unlocked(
                    receipt_key,
                    json.dumps(closure_receipt, ensure_ascii=False, separators=(",", ":")),
                )
                # The receipt now answers replays of this closure; its progress
                # record has nothing left to resume.
                self._session_closure_progress.pop(receipt_key, None)
            if closure_receipt is not None:
                self._remove_session_authority_aliases_unlocked(session)

    def _record_session_closure_receipt_unlocked(self, key: tuple[str, str], receipt: str) -> None:
        if key not in self._session_closure_receipts:
            self._session_closure_receipt_counts[key[0]] = (
                self._session_closure_receipt_counts.get(key[0], 0) + 1
            )
        self._session_closure_receipts[key] = receipt
        self._session_closure_receipts.move_to_end(key)
        while len(self._session_closure_receipts) > self._closure_receipt_limit:
            (evicted_session_id, _plan_id), _receipt = self._session_closure_receipts.popitem(
                last=False
            )
            remaining = self._session_closure_receipt_counts[evicted_session_id] - 1
            if remaining:
                self._session_closure_receipt_counts[evicted_session_id] = remaining
            else:
                del self._session_closure_receipt_counts[evicted_session_id]

    def _remove_session_authority_aliases_unlocked(self, deleted: Session) -> None:
        """Retain source authority only while a live fork can replay it."""
        candidates = {deleted.id}
        relationship = session_fork_rules.session_fork_profile_relationship(deleted)
        if relationship is not None:
            candidates.add(relationship.source_session_id)
        retained_sources = {
            relationship.source_session_id
            for session in self._sessions.values()
            if (relationship := session_fork_rules.session_fork_profile_relationship(session))
            is not None
        }
        stale = [
            key
            for key, private_value in self._public_authority_aliases.items()
            if key[1] == deleted.id
            or (
                key[0] == "session_id"
                and private_value in candidates
                and private_value not in self._sessions
                and private_value not in retained_sources
            )
        ]
        for key in stale:
            del self._public_authority_aliases[key]

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
            for owner in self._session_closure_progress.owners((session_id,)):
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
    async def set_summary(
        self,
        session_id: str,
        summary: str,
        *,
        tone: SummaryTone = "neutral",
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> Session:
        return await self._update_display_or_user_metadata(
            session_id,
            None,
            summary_patch=summary_metadata(summary, tone),
            _access_bounds=_access_bounds,
        )

    async def _access_set_summary(self, bounds, session_id, summary, tone):
        return await self.set_summary(session_id, summary, tone=tone, _access_bounds=bounds)

    @runtime_session_mutation
    async def update_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any],
        *,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> Session:
        return await self._update_display_or_user_metadata(
            session_id,
            metadata,
            _access_bounds=_access_bounds,
        )

    async def _update_display_or_user_metadata(
        self,
        session_id: str,
        metadata: dict[str, Any] | None,
        *,
        summary_patch: dict[str, Any] | None = None,
        _access_bounds: _SessionAccessBounds | None = None,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        user_metadata = None if metadata is None else copy_session_user_metadata(metadata)
        async with self._lock:
            session = self._sessions.get(session_id)
            if _access_bounds is not None:
                _access_bounds.require_action(session, "modify")
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.owners((session_id,)):
                closure_records._check_closure_lineage_owner(owner, (session_id,))
            new_metadata = update_session_annotations(
                session.metadata, user_metadata, summary_patch
            )
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
            for owner in self._session_closure_progress.owners((session_id,)):
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
            for owner in self._session_closure_progress.owners((session_id,)):
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
            interaction_id=(None if admission is None else admission.interaction_id),
            interaction_is_new=(admission is not None and admission.started_event is not None),
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
            for owner in self._session_closure_progress.owners((session_id,)):
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

            if (
                session.status == SessionStatus.PENDING
                and admission is not None
                and admission.defer_source
            ):
                receipt = self._participant_session_receipts.get(session_id)
                if receipt is not None:
                    from cayu.sessions._participant_execution_identity import (
                        require_initial_execution_input,
                    )

                    require_initial_execution_input(
                        session,
                        receipt,
                        self._transcripts.get(session_id, []),
                        list(admission.source_messages),
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
                interaction_id = admission.interaction_id
                defer_source = admission.defer_source
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
            if admission is not None and admission.started_event is not None:
                admission_events.append(admission.started_event)
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
                interaction_id = admission.interaction_id
                source_messages = list(admission.source_messages)
                defer_source = admission.defer_source
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
            for owner in self._session_closure_progress.owners((session_id,)):
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
            for owner in self._session_closure_progress.owners((session_id,)):
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
            for owner in self._session_closure_progress.owners((session_id,)):
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
            for owner in self._session_closure_progress.owners((session_id,)):
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
                for owner in self._session_closure_progress.owners((publication_session_id,)):
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
                for owner in self._session_closure_progress.owners((session_id,)):
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
            for owner in self._session_closure_progress.owners((session_id,)):
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
            for owner in self._session_closure_progress.owners((session_id,)):
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
            for owner in self._session_closure_progress.owners((session_id,)):
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
        return await memory_event_delivery.claim_first_persisted_event_side_effect(
            self._event_delivery_state, expected
        )

    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectClaim | None:
        return await memory_event_delivery.claim_persisted_event_side_effect(
            self._event_delivery_state,
            session_id=session_id,
            event_id=event_id,
            lease_seconds=lease_seconds,
        )

    async def get_persisted_event_side_effect_delivery(
        self,
        *,
        session_id: str,
        event_id: str,
    ) -> PersistedEventSideEffectDelivery | None:
        return await memory_event_delivery.get_persisted_event_side_effect_delivery(
            self._event_delivery_state, session_id=session_id, event_id=event_id
        )

    async def retire_failed_first_event_delivery(
        self, expected: PersistedEventSideEffectDelivery
    ) -> PersistedEventSideEffectDelivery | None:
        return await memory_event_delivery.retire_failed_first_event_delivery(
            self._event_delivery_state, expected
        )

    async def mark_persisted_event_side_effect_delivered(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        return await memory_event_delivery.mark_persisted_event_side_effect_delivered(
            self._event_delivery_state, claim
        )

    async def mark_persisted_event_side_effect_failed(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        error: str,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> PersistedEventSideEffectDelivery:
        return await memory_event_delivery.mark_persisted_event_side_effect_failed(
            self._event_delivery_state,
            claim,
            error=error,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )

    async def defer_persisted_event_side_effect(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        return await memory_event_delivery.defer_persisted_event_side_effect(
            self._event_delivery_state, claim
        )

    async def get_persisted_event_side_effect_health(self) -> PersistedEventSideEffectHealth:
        return await memory_event_delivery.get_persisted_event_side_effect_health(
            self._event_delivery_state
        )

    async def query_persisted_event_side_effect_deliveries(
        self,
        query: PersistedEventSideEffectQuery,
    ) -> PersistedEventSideEffectPage:
        return await memory_event_delivery.query_persisted_event_side_effect_deliveries(
            self._event_delivery_state, query
        )

    async def list_persisted_event_side_effect_deliveries(
        self,
        *,
        statuses: set[PersistedEventSideEffectStatus] | None = None,
        claimable_only: bool = False,
        after_sequence: int | None = None,
        limit: int = 100,
    ) -> list[PersistedEventSideEffectDelivery]:
        return await memory_event_delivery.list_persisted_event_side_effect_deliveries(
            self._event_delivery_state,
            statuses=statuses,
            claimable_only=claimable_only,
            after_sequence=after_sequence,
            limit=limit,
        )

    async def renew_persisted_event_side_effect(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectDelivery:
        return await memory_event_delivery.renew_persisted_event_side_effect(
            self._event_delivery_state, claim, lease_seconds=lease_seconds
        )

    @runtime_session_query
    async def inspect_session_messages(
        self,
        query: SessionMessageQuery,
        *,
        expected_authorized_session_instance_id: str | None = None,
    ) -> SessionMessageInspection:
        return await memory_queued_messages.inspect_session_messages(
            self._queued_message_state,
            query,
            expected_authorized_session_instance_id=expected_authorized_session_instance_id,
        )

    @runtime_session_query
    async def apply_session_message_action(
        self, request: SessionMessageActionRequest
    ) -> SessionMessageActionResult:
        return await memory_queued_messages.apply_session_message_action(
            self._queued_message_state,
            request,
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
        return await memory_queued_messages.snapshot_session_message_source(
            self._queued_message_state,
            session_id,
            include_transcript_digest=include_transcript_digest,
            include_checkpoint_digest=include_checkpoint_digest,
            expected_authorized_session_instance_id=expected_authorized_session_instance_id,
        )

    @runtime_session_query
    async def append_peer_content(
        self,
        request: PeerContentAppendRequest,
        *,
        qualify_target: Callable[[Session], None] | None = None,
        pending_transcript_cursor: int | None = None,
    ) -> PeerContentReceipt:
        return await memory_peer_content.append_peer_content(
            self._peer_content_state,
            request,
            qualify_target=qualify_target,
            pending_transcript_cursor=pending_transcript_cursor,
        )

    async def read_peer_content_attempt(
        self, request: PeerContentAppendRequest
    ) -> PeerContentReceipt | None:
        return await memory_peer_content.read_peer_content_attempt(
            self._peer_content_state,
            request,
        )

    async def list_pending_peer_content(self, *, after_operation_key=None, limit=32):
        """Trusted receiving-owner discovery, independent of creation settlement."""
        return await memory_peer_content.list_pending_peer_content(
            self._peer_content_state,
            after_operation_key=after_operation_key,
            limit=limit,
        )

    async def read_peer_content(self, append_key: PeerAppendKey) -> PeerContentReceipt | None:
        return await memory_peer_content.read_peer_content(
            self._peer_content_state,
            append_key,
        )

    async def record_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        return await memory_peer_content.record_peer_content_exposure(
            self._peer_content_state,
            request,
        )

    async def begin_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        return await memory_peer_content.begin_peer_content_exposure(
            self._peer_content_state,
            request,
        )

    async def read_peer_content_exposure(
        self, append_key: PeerAppendKey, exposure_id: str
    ) -> PeerContentExposureReceipt | None:
        return await memory_peer_content.read_peer_content_exposure(
            self._peer_content_state,
            append_key,
            exposure_id,
        )

    async def exclude_peer_content(
        self,
        request: PeerContentAppendRequest,
        *,
        reason: str,
    ) -> PeerContentReceipt:
        return await memory_peer_content.exclude_peer_content(
            self._peer_content_state,
            request,
            reason=reason,
        )

    async def retry_pending_peer_content(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str,
        expected_run_epoch: int,
        expected_transcript_cursor: int,
        admit=None,
    ) -> tuple[PeerContentReceipt, ...]:
        return await memory_peer_content.retry_pending_peer_content(
            self._peer_content_state,
            session_id,
            expected_session_instance_id=expected_session_instance_id,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
            admit=admit,
        )

    @runtime_session_query
    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
        *,
        expected_authorized_target_instance_id: str | None = None,
    ) -> EnqueueSessionMessageResult:
        return await memory_queued_messages.enqueue_session_message(
            self._queued_message_state,
            request,
            expected_authorized_target_instance_id=expected_authorized_target_instance_id,
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
        return await memory_queued_messages.deliver_queued_session_messages(
            self._queued_message_state,
            session_id,
            include_on_idle=include_on_idle,
            delivery_id=delivery_id,
            eligible_through=eligible_through,
            limit=limit,
            interaction_id=interaction_id,
            interaction_started_event=interaction_started_event,
            profile_handoff=profile_handoff,
            reject_only=reject_only,
        )

    async def repair_queued_interaction_profile_handoff(
        self,
        session_id: str,
        *,
        interaction_started_event: Event,
        profile_handoff: QueuedInteractionProfileHandoff,
    ) -> ActiveInvocationExecutionProfile:
        return await memory_queued_messages.repair_queued_interaction_profile_handoff(
            self._queued_message_state,
            session_id,
            interaction_started_event=interaction_started_event,
            profile_handoff=profile_handoff,
        )

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
            for owner in self._session_closure_progress.owners((session_id,)):
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
                for owner in self._session_closure_progress.owners((session_id,)):
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
            for owner in self._session_closure_progress.owners((session_id,)):
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
                memory_context_views.validate_compaction(
                    session_id,
                    context_view_compaction_cursor,
                    ownership_clock=self._ownership_clock,
                    receipts=self._context_selection_receipts,
                    views=self._context_views,
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

        for owner in self._session_closure_progress.owners((session_id,)):
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

    async def read_store_identity(self) -> str:
        return self._store_identity

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
        fork_lineage_owner: str | None = None,
        without_reservation_for: str | None = None,
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
            fork_lineage_owner=fork_lineage_owner,
            without_reservation_for=without_reservation_for,
        )

    async def fold_event_snapshot(
        self,
        query: EventQuery,
        consume: EventSnapshotConsumer,
        *,
        max_bytes: int,
        max_pages: int | None = None,
    ) -> int:
        return await memory_event_reads.fold_event_snapshot(
            self._read_event_state,
            query,
            consume,
            max_bytes=max_bytes,
            max_pages=max_pages,
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
        return await memory_transcripts.append_transcript_messages(
            self._write_transcript_state, session_id, messages, interaction_id=interaction_id
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
            for owner in self._session_closure_progress.owners((session_id,)):
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
        return await memory_transcripts.replace_initial_transcript_messages(
            self._write_transcript_state,
            session_id,
            expected_messages,
            replacement_messages,
            interaction_id=interaction_id,
            checkpoint_transform=checkpoint_transform,
            runtime_suffix_count=runtime_suffix_count,
        )

    async def materialize_deferred_interaction_input(
        self,
        session_id: str,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> bool:
        return await memory_transcripts.materialize_deferred_interaction_input(
            self._write_transcript_state, session_id, interaction_id=interaction_id
        )

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
        return await memory_transcripts.append_transcript_messages_and_transform_checkpoint(
            self._write_transcript_state,
            session_id,
            messages,
            checkpoint_transform,
            interaction_id=interaction_id,
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

    async def load_bounded_transcript_window(
        self,
        session_id: str,
        *,
        start_index: int,
        limit: int,
        max_record_bytes: int,
        max_total_bytes: int,
    ) -> BoundedTranscriptWindow:
        return await memory_transcripts.load_bounded_transcript_window(
            self._read_transcript_state,
            session_id,
            start_index=start_index,
            limit=limit,
            max_record_bytes=max_record_bytes,
            max_total_bytes=max_total_bytes,
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
        return await memory_evidence.create_recall_receipt(self._recall_evidence_state, receipt)

    async def load_recall_receipt(
        self,
        session_id: str,
        receipt_id: str,
    ) -> RecallReceipt | None:
        return await memory_evidence.load_recall_receipt(
            self._recall_evidence_state, session_id, receipt_id
        )

    async def list_recall_receipts(
        self,
        query: RecallEvidenceQuery,
    ) -> RecallReceiptPage:
        return await memory_evidence.list_recall_receipts(self._recall_evidence_state, query)

    async def create_context_exposure(
        self,
        exposure: ContextExposure,
        item_exposures: tuple[RecallItemExposure, ...] = (),
    ) -> ContextExposure:
        return await memory_evidence.create_context_exposure(
            self._context_exposure_state, exposure, item_exposures
        )

    async def load_context_exposure(
        self,
        session_id: str,
        exposure_id: str,
    ) -> ContextExposure | None:
        return await memory_evidence.load_context_exposure(
            self._context_exposure_state, session_id, exposure_id
        )

    async def load_recall_item_exposures(
        self,
        session_id: str,
        exposure_id: str,
    ) -> tuple[RecallItemExposure, ...]:
        return await memory_evidence.load_recall_item_exposures(
            self._context_exposure_state, session_id, exposure_id
        )

    async def list_context_exposures(
        self,
        query: RecallEvidenceQuery,
    ) -> ContextExposurePage:
        return await memory_evidence.list_context_exposures(self._context_exposure_state, query)

    async def transition_context_exposure(
        self,
        session_id: str,
        exposure_id: str,
        request: ContextExposureTransitionRequest,
    ) -> ContextExposure:
        return await memory_evidence.transition_context_exposure(
            self._context_exposure_state, session_id, exposure_id, request
        )

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            invocation_scope._assert_session_run_epoch(session_id, session)
            for owner in self._session_closure_progress.owners((session_id,)):
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
            for owner in self._session_closure_progress.owners((session_id,)):
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
