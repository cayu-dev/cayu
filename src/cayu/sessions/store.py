"""Shared session-store contract and backend capability checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect_left
from collections.abc import Callable, Collection, Mapping
from datetime import datetime
from pathlib import Path
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast, overload

from cayu.session_metadata import SummaryTone

if TYPE_CHECKING:
    from cayu.deadlines import ExecutionDeadline, ExecutionDeadlineRenewal
    from cayu.sessions._invocation_lifecycle import InvocationLifecycleCommand
from uuid import uuid4

from cayu._validation import MAX_DURABLE_JSON_INTEGER, compact_json_utf8_size
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.budgets.aggregates import UsageRollupStoreResult, project_aggregate_usage_inspection_event
from cayu.budgets.base import (
    is_budget_inspection_event,
    is_budget_model_attempt_terminal_event,
    project_budget_inspection_event,
    project_budget_model_attempt_inspection_event,
    session_budget_inspection,
)
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
from cayu.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.memory.evidence import (
    ContextExposure,
    ContextExposurePage,
    ContextExposureTransitionRequest,
    RecallEvidenceQuery,
    RecallItemExposure,
    RecallReceipt,
    RecallReceiptPage,
)
from cayu.messages import Message, MessageRole
from cayu.runtime._child_session_notifications import (
    ChildSessionLifecyclePage,
    ChildSessionLifecycleQuery,
    child_session_notification_stage_binding,
)
from cayu.runtime._cost_accounting import CostAccountingSnapshot
from cayu.runtime._usage_accounting import UsageAccountingSnapshot
from cayu.runtime.public_authority import PublicAuthorityAliasCodec
from cayu.runtime.service_manifest import RuntimeStoreDurability
from cayu.sessions import _checkpoint_preservation as checkpoint_preservation
from cayu.sessions import _interaction_receipts as interaction_receipts
from cayu.sessions import _invocation_scope as invocation_scope
from cayu.sessions import _model_stage_rules as model_stage_rules
from cayu.sessions import _runtime_publication as publication_contracts
from cayu.sessions import _runtime_publication_rules as publication_rules
from cayu.sessions import _tool_round_publication_rules as tool_publication_rules
from cayu.sessions import authority as session_authority_rules
from cayu.sessions import event_delivery as session_event_rules
from cayu.sessions import inspection as session_inspection_rules
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
    checkpoint_with_active_invocation_execution_profile,
)
from cayu.sessions._execution_profile_rules import SessionModelTransition
from cayu.sessions._interaction_transitions import (
    InteractionTransitionReceiptResult,
    InteractionTransitionResult,
    InteractionTransitionSpec,
    QueuedInteractionProfileHandoff,
    SessionInvocationAdmission,
)
from cayu.sessions._invocation_scope import INHERIT_INTERACTION, InteractionAttribution
from cayu.sessions._invocation_terminal_decision import InvocationTerminalDecision
from cayu.sessions._model_completion_publication import ModelCompletionStageSettlementRequest
from cayu.sessions._model_stage import (
    ActiveModelCompletionStage,
    ModelCompletionStage,
    ModelCompletionStageAbandonment,
    ModelCompletionStageAbandonmentResult,
    ModelCompletionStageDispatch,
    ModelCompletionStageRecoveryFence,
    ModelCompletionStageRequest,
    ModelCompletionStageResult,
    ModelCompletionStageSettlement,
    SessionModelCompletionStageConflict,
)
from cayu.sessions._operation_records import SessionOperationPublication
from cayu.sessions._runtime_publication import (
    RuntimePublicationReceipt,
    RuntimePublicationRequest,
    RuntimePublicationResult,
    SessionOperationInitializer,
    SessionOperationTransform,
    StoreTimeSessionOperationTransform,
)
from cayu.sessions._terminal_evidence import (
    QueuedDispatchTerminalReceipt,
    QueuedDispatchTerminalReceiptQuery,
)
from cayu.sessions.authority import (
    CheckpointRootFieldGuard,
    CheckpointValueAuthority,
    SessionRunFenced,
    checkpoint_value_authority,
)
from cayu.sessions.environment_activity import (
    EnvironmentActivityQuery,
    EnvironmentActivitySnapshot,
)
from cayu.sessions.event_delivery import (
    PersistedEventSideEffectClaim,
    PersistedEventSideEffectDelivery,
    PersistedEventSideEffectHealth,
    PersistedEventSideEffectPage,
    PersistedEventSideEffectQuery,
    PersistedEventSideEffectStatus,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.execution import SessionExecutionState
from cayu.sessions.forks import (
    ForkCheckpointAuthorityDecoder,
    ForkSystemPromptReplacement,
    ProfiledSessionForkResult,
    SessionForkProfileRelationship,
)
from cayu.sessions.inspection import (
    SerializedRecordSummary,
    SessionInspectionIdentity,
    SessionInspectionSummary,
)
from cayu.sessions.invocation import SessionInvocationBinding
from cayu.sessions.lineage import SessionLineageQuery, SessionLineageResult
from cayu.sessions.mcp_manifest_history import (
    McpManifestBaseline,
    McpManifestBaselineLoadResult,
    McpManifestPublicationResult,
)
from cayu.sessions.messaging import (
    SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    EnqueueSessionMessageRequest,
    EnqueueSessionMessageResult,
    SessionMessageActionRequest,
    SessionMessageActionResult,
    SessionMessageDeliveryBatch,
    SessionMessageInspection,
    SessionMessageQuery,
    SessionMessageSource,
)
from cayu.sessions.pending_action_contracts import PendingActionListResult, PendingActionQuery
from cayu.sessions.queries import SessionAggregateFilter, SessionListResult, SessionQuery
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
)
from cayu.sessions.requests import RunRequest
from cayu.sessions.summaries import EventSummary, SessionOperationalSnapshot, SessionOutcome
from cayu.sessions.supervision_events import SupervisionEventPage, SupervisionEventQuery
from cayu.sessions.terminal_evidence import TerminalSessionEvidence, TerminalSessionEvidenceLimits
from cayu.sessions.topology import SessionTopologyQuery, SessionTopologyStoreResult
from cayu.sessions.transcript_input import DeferredInteractionInput
from cayu.sessions.transcript_queries import (
    LATEST_TRANSCRIPT_TEXT_MAX_CHARS,
    BoundedTranscriptWindow,
    ForkTranscriptValidator,
    TranscriptPage,
    TranscriptQuery,
    TranscriptSearchQuery,
    TranscriptSearchResult,
    TranscriptSnapshot,
)
from cayu.sessions.usage import UsageRollupQuery
from cayu.tools.exposure import ToolCapabilityCeiling
from cayu.tools.grants import (
    TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    TargetedToolGrantIssueResult,
    TargetedToolGrantReconstructionResult,
    TargetedToolGrantRecord,
    TargetedToolGrantStateSnapshot,
    TargetedToolUseRequest,
    TargetedToolUseResult,
)

if TYPE_CHECKING:
    from cayu.collaboration._contracts import ExactLookup
    from cayu.collaboration.participants import ParticipantRef as _ParticipantRef
    from cayu.runtime._model_failover_stage import ModelFailoverStageAdmission
    from cayu.runtime._zero_work_interruption import (
        ZeroWorkInterruptionPublication,
        ZeroWorkInterruptionRequest,
    )
    from cayu.runtime.evidence_spool import EvidenceSpool
    from cayu.sessions._access_rules import SessionReadBounds, _SessionAccessBounds
    from cayu.sessions._context_view_source import (
        CompletedTurnSnapshot,
        ContextViewPublicationSource,
    )
    from cayu.sessions._external_wait_records import ExternalWaitScheduleObservation
    from cayu.sessions._external_wait_transition import ExternalWaitMutation
    from cayu.sessions._invocation_lifecycle import (
        AdmitInvocationCommand,
        CreateInvocationCommand,
        InvocationMutationResult,
        InvocationReleaseResult,
        RebindInvocationCommand,
        RejectInvocationCommand,
        ReleaseInvocationCommand,
        SettleInvocationCommand,
    )
    from cayu.sessions._recipient_continuation import RecipientContinuationSelection
    from cayu.sessions._session_continuation import (
        ContinuationConsumption,
        ContinuationLatch,
        ContinuationNamespace,
        ContinuationPreparation,
        ContinuationRecord,
        ContinuationRetirement,
        ContinuationTicket,
    )
    from cayu.sessions._temporary_continuation import (
        TemporaryServiceAdmission,
        TemporaryServicePreparation,
        TemporaryServiceRecord,
    )
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
    from cayu.sessions.external_waits import (
        ExternalWaitPruneResult,
        ExternalWaitRecord,
        ExternalWaitRetirement,
        ExternalWaitRetirementRequest,
        ExternalWaitScope,
    )
    from cayu.storage.retention import (
        RetentionAuditEntry,
        RetentionAuditRecord,
        RetentionMode,
        RetentionProgressCallback,
        RetentionProtection,
        RetentionReport,
        SessionRetentionPolicy,
        StorageGrowthReport,
    )


_SESSION_EXPORT_OWNER_METHODS = (
    "__getattr__",
    "load",
    "load_checkpoint",
    "load_session_operation",
    "load_transcript_window",
    "create",
    "create_fork",
    "create_fork_with_transcript_validation",
    "create_profiled_fork",
    "_create_fork",
    "checkpoint",
    "transform_checkpoint",
    "transform_checkpoint_with_store_time",
    "transition_status_and_checkpoint",
    "admit_session_invocation",
    "admit_execution_profile_resume",
    "fence_run_and_transform_checkpoint",
    "replace_initial_transcript_messages",
    "append_transcript_messages_and_transform_checkpoint",
    "append_event",
    "append_events",
    "publish_checkpoint_and_events",
    "publish_checkpoint_and_events_with_store_time",
    "_publish_checkpoint_and_events",
    "publish_session_operation",
    "publish_session_operation_guarded",
    "publish_session_operation_guarded_with_store_time",
    "_publish_session_operation",
    "publish_runtime_publication",
    "_publish_runtime_publication_atomic",
    "_prepare_checkpoint_store_unlocked",
    "_apply_checkpoint_store_unlocked",
    "_store_checkpoint_unlocked",
    "_upsert_checkpoint",
    "_prepare_event_append_unlocked",
    "_append_events_unlocked",
    "_append_events_with_cursor",
    "_load_checkpoint_unlocked",
    "_load_checkpoint",
    "_load_unlocked",
    "_load_for_update",
    "_publish_completion_result_event_publication",
    "delete_session",
    "validate_session_closure_admission",
    "claim_session_closure_progress",
    "_require_session_erasure_quiescence_unlocked",
    "_require_session_erasure_quiescence",
    "_run_write",
    "_run_read",
    "_connection",
)


def _session_export_methods_owned(store: object) -> bool:
    """An override must explicitly re-attest the complete export owner boundary."""
    return _session_owner_methods_owned(
        store, "session_export_version", _SESSION_EXPORT_OWNER_METHODS
    )


def _session_continuation_methods_owned(store: object) -> bool:
    return _session_owner_methods_owned(
        store,
        "session_continuation_version",
        (
            *_SESSION_EXPORT_OWNER_METHODS,
            "_publish_continuation_operation",
            "_publish_temporary_continuation_service",
            "_publish_temporary_service_target",
            "_prepare_temporary_side_service",
            "_load_temporary_service_target",
            "_load_temporary_continuation_service",
            "_read_temporary_continuation_outcome",
            "_reconcile_temporary_continuation_service",
            "_initialize_continuation_namespace",
            "prepare_continuation_ticket",
            "load_continuation_ticket",
            "lookup_continuation_ticket",
            "mark_continuation_waiting",
            "_recover_continuation_writer",
            "latch_continuation",
            "consume_continuation",
            "_consume_continuation",
            "_claim_continuation_admission",
            "_release_continuation_admission_claim",
            "_finalize_continuation_admission",
            "retire_continuation",
            "apply_invocation_lifecycle_command",
        ),
    )


def _session_owner_methods_owned(
    store: object, version_field: str, methods: tuple[str, ...]
) -> bool:
    mro = type(store).__mro__
    capability_owner = next(
        (index for index, owner in enumerate(mro) if version_field in vars(owner)),
        len(mro),
    )
    if capability_owner == len(mro):
        return False
    instance_fields = vars(store)
    if version_field in instance_fields:
        return False
    for method in methods:
        method_owner = next(
            (index for index, owner in enumerate(mro) if method in vars(owner)), len(mro)
        )
        # SQLite's _connection is an owned resource field, whereas PostgreSQL's
        # is a method. Only the latter can be shadowed as a method override.
        if method in instance_fields and not (method == "_connection" and method_owner == len(mro)):
            return False
        if method_owner < capability_owner:
            return False
    return True


def _continuation_writer_was_released(
    session: Session,
    checkpoint: dict[str, Any] | None,
) -> bool:
    """Allow exactly one post-release epoch without accepting a new writer."""

    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    return active is not None and active_invocation_execution_profile_is_released(
        active,
        session_id=session.id,
        run_epoch=session.run_epoch,
    )


class SessionStore(ABC):
    """Persistent store for sessions and append-only events.

    Implementations own the run-epoch contract. A successful transition to
    ``RUNNING`` claims a new epoch for the current execution context. Runtime
    progress writes made by that context must reject a stale epoch with
    ``SessionRunFenced``. Releasing the claim revokes the durable epoch before
    clearing task-local ownership so inherited child contexts cannot write late.

    Implementations also own the clock used for inactivity, recovery-claim,
    epoch-takeover, and activity-stamp decisions. Callers pass durations. The
    store samples, compares, and stamps time inside one atomic boundary; a
    caller or worker clock is never durable ownership authority.
    """

    # Scoped access is an explicit backend protocol, not a filtering wrapper.
    session_access_version: ClassVar[int | None] = None

    async def _access_create_session(self, bounds, request, identity):
        raise NotImplementedError("Atomic scoped session creation is unsupported.")

    async def _access_set_summary(self, bounds, session_id, summary, tone):
        raise NotImplementedError("Atomic scoped summary mutation is unsupported.")

    async def _access_update_metadata(self, bounds, session_id, metadata):
        raise NotImplementedError("Atomic scoped metadata mutation is unsupported.")

    async def _access_delete_session(self, bounds, session_id):
        raise NotImplementedError("Atomic scoped session deletion is unsupported.")

    async def _access_read_records(self, bounds, session_id, kind, offset, limit, max_bytes):
        raise NotImplementedError("Atomic scoped record reads are unsupported.")

    async def _access_load_session(self, bounds: SessionReadBounds, session_id: str) -> Session:
        raise NotImplementedError("Atomic scoped session access is unsupported.")

    async def _access_list_sessions(
        self, bounds: SessionReadBounds, query: SessionQuery
    ) -> SessionListResult:
        raise NotImplementedError("Atomic scoped session access is unsupported.")

    async def _access_update_labels(
        self, bounds: _SessionAccessBounds, session_id: str, labels: dict[str, str]
    ) -> Session:
        raise NotImplementedError("Atomic scoped session access is unsupported.")

    # Custom stores must opt into optional capabilities explicitly. Conservative
    # defaults keep discovery truthful for inherited methods that fail closed.
    session_export_version: ClassVar[int] = 0
    session_continuation_version: ClassVar[int] = 0
    external_wait_version: ClassVar[int] = 0
    _producer_attachment_version: ClassVar[int] = 0
    supports_environment_activity: ClassVar[bool] = False
    supports_usage_series: ClassVar[bool] = False
    supports_usage_cost_details: ClassVar[bool] = False
    supports_usage_aggregates: ClassVar[bool] = False
    supports_private_argument_continuity: ClassVar[bool] = False
    supports_mcp_manifest_history: ClassVar[bool] = False
    supports_session_topology: ClassVar[bool] = False
    supports_session_lineage: ClassVar[bool] = False
    supports_supervision_events: ClassVar[bool] = False
    # Fork-controller reads evaluate _ForkControllerReadBounds on every read path.
    # Declared on the class itself: a subclass that overrides a read must redeclare it.
    supports_fork_controller_reads: ClassVar[bool] = False
    child_session_notification_version: ClassVar[int | None] = None
    supports_public_authority_aliases: ClassVar[bool] = False
    supports_targeted_tool_grants: ClassVar[bool] = False
    supports_incremental_terminal_evidence: ClassVar[bool] = False
    supports_terminal_session_evidence: ClassVar[bool] = False
    supports_runner_owned_interrupted_evidence: ClassVar[bool] = False
    supports_execution_profile_admission: ClassVar[bool] = False
    supports_active_invocation_execution_profiles: ClassVar[bool] = False
    # Versioned proof for the frozen invocation-lifecycle command boundary.
    # Unlike the historical booleans, an explicit integer lets the runtime
    # reject partial, legacy, and unknown-future store implementations before
    # delivering a command they cannot interpret atomically.
    invocation_lifecycle_command_version: ClassVar[int | None] = None
    terminal_interaction_publication_version: ClassVar[int | None] = None
    durable_model_terminalization_version: ClassVar[int | None] = None
    queued_interaction_profile_handoff_version: ClassVar[int | None] = None
    session_message_lifecycle_version: ClassVar[int | None] = None
    session_steering_version: ClassVar[int | None] = None
    supports_pending_session_initial_checkpoint: ClassVar[bool] = False
    supports_profiled_forks: ClassVar[bool] = False
    supports_atomic_session_operation_initialization: ClassVar[bool] = False
    supports_atomic_model_completion_stage_release: ClassVar[bool] = False
    # Session closure erasure must leave a durable, replayable receipt even
    # after the session row has been removed.  Stores opt in only when they can
    # persist that receipt as part of their deletion/recovery boundary.
    supports_session_closure_receipts: ClassVar[bool] = False
    supports_session_closure_detachment: ClassVar[bool] = False
    supports_session_closure_recursive_deletion: ClassVar[bool] = False
    supports_session_closure_finished_only: ClassVar[bool] = False
    supports_session_closure_progress: ClassVar[bool] = False
    # Storage retention is an explicit operator capability, never implicit.
    supports_storage_retention: ClassVar[bool] = False
    model_completion_recovery_fence_version: ClassVar[int] = 0
    model_failover_stage_version: ClassVar[int] = 0
    supports_completion_result_event_publication_reservations: ClassVar[bool] = False
    supports_transcript_search: ClassVar[bool] = False
    supports_recall_evidence: ClassVar[bool] = False
    supports_owned_off_thread_session_commit_guards: ClassVar[bool] = False
    service_durability: RuntimeStoreDurability = RuntimeStoreDurability.UNVERIFIED

    participant_session_binding_version: ClassVar[int | None] = None
    recipient_continuation_selection_version: ClassVar[int | None] = None
    context_view_selection_fence_version: ClassVar[int | None] = None
    context_view_version: ClassVar[int | None] = None
    # Versioned proof for authenticated peer-content append/readback.  Stores
    # must implement the exact append-key CAS before advertising this value.
    peer_content_version: ClassVar[int | None] = None

    async def _prepare_session_creation_target(self, target, *, authority):
        raise NotImplementedError("This SessionStore does not support creation decisions.")

    async def _register_session_creation_target(self, target, *, authority):
        raise NotImplementedError("This SessionStore does not support creation decisions.")

    async def _exclude_session_creation_target(self, target, *, authority):
        raise NotImplementedError("This SessionStore does not support creation decisions.")

    async def _acknowledge_session_creation_settlement(self, target, *, authority):
        raise NotImplementedError("This SessionStore does not support creation settlement.")

    async def read_session_creation_decision(self, target):
        from cayu.collaboration._contracts import ExactUnavailable

        return ExactUnavailable()

    async def list_pending_session_creations(self, owner, *, cursor=None, limit=32):
        raise NotImplementedError("This SessionStore does not support creation discovery.")

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
        recipient_receipt_validator: Callable[[Session, ParticipantSessionCreationReceipt], object]
        | None = None,
    ) -> tuple[Session, ParticipantSessionCreationReceipt]:
        raise NotImplementedError(
            "This SessionStore does not support participant-owned session creation."
        )

    async def load_participant_session_binding(
        self, session_id: str
    ) -> ParticipantSessionBinding | None:
        raise NotImplementedError("This SessionStore does not support participant bindings.")

    async def _scan_participant_session_bindings(self, participant, *, after=None, limit=32):
        """Owner-internal inventory; callers must authenticate current participant access."""
        raise NotImplementedError("This SessionStore does not qualify participant discovery.")

    async def load_participant_session_creation_receipt(
        self, session_id: str
    ) -> ParticipantSessionCreationReceipt | None:
        raise NotImplementedError("This SessionStore does not support participant receipts.")

    async def lookup_participant_session_creation(
        self, creation_request: ParticipantSessionCreationRequest
    ) -> tuple[Session, ParticipantSessionCreationReceipt] | None:
        raise NotImplementedError("This SessionStore does not support creation-key lookup.")

    async def capture_context_view_publication_source(
        self, session_id: str
    ) -> ContextViewPublicationSource:
        """Capture a bounded completed-turn snapshot under one native read boundary."""
        raise NotImplementedError("This SessionStore cannot capture context-view publication.")

    async def _capture_completed_turn_snapshot(self, session_id: str) -> CompletedTurnSnapshot:
        raise NotImplementedError("This SessionStore cannot capture a current completed turn.")

    async def capture_recipient_continuation(
        self, session_id: str
    ) -> RecipientContinuationSelection:
        """Select exact identity without acquiring a writer or execution authority.

        Wrappers must independently qualify this coherent native read. Inheriting
        an implementation is not evidence that wrapper-specific state is covered.
        """
        from cayu.sessions._recipient_continuation import (
            require_continuation_selection_store,
            select_continuation,
        )

        require_continuation_selection_store(self)
        session_id = require_clean_nonblank(session_id, "session_id")
        return select_continuation(await self._capture_completed_turn_snapshot(session_id))

    async def publish_context_view(
        self, manifest: _ContextViewManifest, *, publication_key: str
    ) -> _ContextViewManifest:
        raise NotImplementedError("This SessionStore does not support context-view publication.")

    async def lookup_context_view_selection(
        self, selection_key: str
    ) -> _ContextViewSelectionReceipt | None:
        raise NotImplementedError(
            "This SessionStore does not support context-view selection lookup."
        )

    async def lookup_context_view_publication(
        self, publication_key: str
    ) -> _ContextViewManifest | None:
        raise NotImplementedError(
            "This SessionStore does not support context-view publication lookup."
        )

    async def select_context_view(
        self, request: _ContextViewSelectionRequest
    ) -> _ContextViewSelectionReceipt:
        raise NotImplementedError("This SessionStore does not support context-view selection.")

    async def _exclude_context_view_selection(self, request, *, authority):
        """Atomically decide exclusion or return positive prior selection evidence."""
        raise NotImplementedError("This SessionStore cannot exclude context-view selection.")

    async def _reserve_context_view_selection_control(self, request, *, authority):
        """Reserve exact mandatory exclusion capacity before acquiring a view."""
        raise NotImplementedError("This SessionStore cannot reserve selection cleanup capacity.")

    async def read_context_view_selection_decision(self, request):
        """Exact receiving readback, not current disclosure or ownership authority."""
        raise NotImplementedError("This SessionStore cannot read selection decisions.")

    async def _prepare_context_view_selection_target(self, target, *, authority):
        return await self._context_selection_target_control(target, authority=authority)

    async def _register_context_view_selection_target(self, target, *, authority):
        return await self._context_selection_target_control(
            target, authority=authority, register=True
        )

    async def _exclude_context_view_selection_target(self, target, *, authority):
        return await self._context_selection_target_control(
            target, authority=authority, exclude=True
        )

    async def _context_selection_target_control(
        self, target, *, authority, register=False, exclude=False
    ):
        from cayu.sessions._context_selection_fence import (
            ContextViewSelectionTarget,
            require_authority,
            require_selection_fence_store,
        )

        require_authority(authority)
        require_selection_fence_store(self)
        target = ContextViewSelectionTarget.model_validate(target)
        return await self._context_selection_control(
            target.request, authority=authority, exclude=exclude, target=target, register=register
        )

    async def _context_selection_control(
        self, request, *, authority, exclude, target=None, register=False
    ):
        raise NotImplementedError("This SessionStore cannot own selection responsibility.")

    async def _select_context_view_target(self, target, *, authority):
        from cayu.sessions._context_selection_fence import (
            ContextViewSelectionTarget,
            require_authority,
            require_selection_fence_store,
        )

        require_authority(authority)
        require_selection_fence_store(self)
        target = ContextViewSelectionTarget.model_validate(target)
        return await self._select_context_view(target.request, target=target)

    async def _select_context_view(self, request, *, target=None):
        raise NotImplementedError("This SessionStore cannot consume selection responsibility.")

    async def _read_context_view_retention(self, target):
        raise NotImplementedError("This SessionStore cannot authenticate retention settlement.")

    async def _adopt_context_view_target(self, target, request, *, authority):
        from cayu.sessions._context_selection_fence import (
            ContextViewSelectionTarget,
            require_authority,
            require_selection_fence_store,
        )

        require_authority(authority)
        require_selection_fence_store(self)
        target = ContextViewSelectionTarget.model_validate(target)
        return await self._transition_context_view_ownership(request, target=target)

    async def _transition_context_view_ownership(self, request, *, target=None):
        raise NotImplementedError("This SessionStore cannot consume retention responsibility.")

    async def read_context_view(
        self, view_id: str, *, source_session_id: str
    ) -> _ContextViewReadback:
        raise NotImplementedError("This SessionStore does not support context-view readback.")

    async def transition_context_view_ownership(
        self, request: _ContextViewOwnershipRequest
    ) -> _ContextViewSelectionReceipt:
        raise NotImplementedError(
            "This SessionStore does not support context-view ownership transitions."
        )

    async def validate_context_view_source_closure(self, session_id: str) -> None:
        """Refuse source deletion while an active context-view pin needs it."""

        raise NotImplementedError(
            "This SessionStore does not support context-view closure admission."
        )

    async def validate_context_view_compaction(
        self, session_id: str, expected_transcript_cursor: int
    ) -> None:
        """Prove retained view material survives a source-session compaction."""

        raise NotImplementedError(
            "This SessionStore does not support context-view compaction admission."
        )

    async def read_context_view_lifecycle_events(
        self, view_id: str, *, limit: int = 256, owner_participant: _ParticipantRef | None = None
    ) -> tuple[_ContextViewLifecycleEvent, ...]:
        """Read bounded evidence; apply an exact participant filter before limiting."""
        raise NotImplementedError(
            "This SessionStore does not support context-view lifecycle events."
        )

    def durable_state_paths(self) -> tuple[Path, ...]:
        """Return local files that own this store's durable state.

        In-memory and remote stores return no paths. Filesystem-confined
        features must fail closed when the store cannot provide this evidence;
        custom local stores can opt in by overriding this method.
        """

        return ()

    @property
    def can_register_public_authority_aliases(self) -> bool:
        """Whether this store instance can persist new public authority aliases."""

        return self.supports_public_authority_aliases

    @property
    def public_authority_alias_codec(self) -> PublicAuthorityAliasCodec | None:
        """Return the immutable codec bound to this store, when supported."""

        return None

    async def _terminalize_zero_work_interruption(
        self,
        request: ZeroWorkInterruptionRequest,
    ) -> ZeroWorkInterruptionPublication | None:
        """Atomically prove and terminalize zero work; unsupported stores decline."""
        return None

    async def register_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        private_value: str,
        scope_session_id: str | None = None,
    ) -> None:
        """Persist one public-to-private authority mapping before exposure.

        This optional capability is deliberately explicit. A runtime must not
        publish a one-way public alias unless its store can resolve that alias
        after restart and from another worker.
        """

        raise NotImplementedError(
            "This SessionStore does not support durable public authority aliases."
        )

    async def resolve_public_authority_alias(
        self,
        public_alias: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> str | None:
        """Resolve one exact indexed alias within its authority scope."""

        raise NotImplementedError(
            "This SessionStore does not support durable public authority aliases."
        )

    async def public_authority_private_value_exists(
        self,
        private_value: str,
        *,
        field_name: str,
        scope_session_id: str | None = None,
    ) -> bool:
        """Return positive indexed evidence that private authority exists."""

        raise NotImplementedError(
            "This SessionStore does not support durable public authority aliases."
        )

    async def issue_targeted_tool_grants(
        self,
        session_id: str,
        *,
        expected_run_epoch: int,
        records: tuple[TargetedToolGrantRecord, ...],
        events: tuple[Event, ...],
    ) -> TargetedToolGrantIssueResult:
        """Atomically issue or rejoin one exact interaction-scoped grant batch."""

        raise NotImplementedError(
            "This SessionStore does not support durable targeted tool grants."
        )

    async def list_targeted_tool_grants(
        self,
        session_id: str,
        *,
        interaction_id: str | None = None,
        limit: int = TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    ) -> tuple[TargetedToolGrantRecord, ...]:
        """Load a bounded session or interaction grant view for inspection."""

        raise NotImplementedError(
            "This SessionStore does not support durable targeted tool grants."
        )

    async def load_targeted_tool_grant_state(
        self,
        session_id: str,
    ) -> TargetedToolGrantStateSnapshot:
        """Load complete typed grant state for a trusted durable export."""

        raise NotImplementedError(
            "This SessionStore does not support durable targeted tool grants."
        )

    async def bind_targeted_tool_grant_use(
        self,
        request: TargetedToolUseRequest,
        *,
        observed_at: datetime,
    ) -> TargetedToolUseResult:
        """Atomically consume one budget unit or rejoin its exact prior binding."""

        raise NotImplementedError(
            "This SessionStore does not support durable targeted tool grants."
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
        """Atomically revoke one exact session-scoped reference, idempotently."""

        raise NotImplementedError(
            "This SessionStore does not support durable targeted tool grants."
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
        """Validate durable grants against current bounded registration evidence."""

        raise NotImplementedError(
            "This SessionStore does not support durable targeted tool grants."
        )

    @abstractmethod
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
        """Create a session, optionally admitting its first interaction atomically.

        Implementations must mint the final session ID, load any requested parent,
        and call ``session_invocation_for_run_request`` inside the same create
        boundary. The returned immutable invocation record is persisted with the
        session and preserved by all later updates.

        With admission, create the session directly in ``RUNNING``, claim its
        first run epoch, and persist the start event plus deferred source batch
        in the same transaction. Both optional arguments are supplied together.
        ``checkpoint_transform`` projects initial authority inside the same
        transaction. Without admission, stores that advertise
        ``supports_pending_session_initial_checkpoint`` invoke it with a
        ``PENDING`` session and no current checkpoint. Stores that do not
        advertise that capability must reject such a request before creating
        the session.

        ``operation_initializer`` receives the final detached session and
        synchronously returns bounded operation records that must commit
        atomically with it. The callback runs inside the creation boundary and
        must be deterministic and side-effect-free. Stores must reject the
        callback or any record before exposing a partial session.
        """

    @abstractmethod
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
        """Create a forked session with copied transcript/checkpoint state.

        A prompt replacement, when supplied by the runtime, atomically removes
        inherited system messages and installs its one authoritative system
        message while preserving non-system message attribution.
        """

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
        """Create a fork after validating the final transcript inside the copy boundary.

        Custom stores must opt into this safety capability explicitly. The
        conservative default prevents the runtime from silently falling back to
        a time-of-check/time-of-use transcript validation. When prompt
        replacement is requested, the validator receives the atomically
        transformed child transcript rather than the source transcript.
        """

        raise NotImplementedError(
            "This SessionStore does not support atomic fork transcript validation."
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
        """Atomically create a child and its immutable profile evidence."""

        raise NotImplementedError(
            "This SessionStore does not support atomic profiled session forks."
        )

    @abstractmethod
    async def load(self, session_id: str) -> Session | None:
        """Load a session by id."""

    supports_session_execution = False

    async def _inspect_session_execution_owner(self, session_id: str) -> SessionExecutionState:
        """Internal liveness admission; custom stores without it remain conservative."""
        return await self.inspect_session_execution(session_id)

    async def inspect_session_execution(self, session_id: str) -> SessionExecutionState:
        """Read observational liveness; custom stores without leases return unknown."""
        session = await self.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return SessionExecutionState(
            session_id=session.id,
            run_epoch=session.run_epoch,
            state="terminal"
            if session.status
            in {SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.INTERRUPTED}
            else "unknown",
        )

    @abstractmethod
    async def load_state(self, session_id: str) -> SessionStateSnapshot | None:
        """Load bounded mutable state without labels or unbounded metadata."""

    @abstractmethod
    async def load_invocation_snapshot(
        self,
        session_id: str,
    ) -> SessionInvocationSnapshot | None:
        """Load bounded immutable provenance without labels or metadata."""

    @abstractmethod
    async def inspect_identity(self, session_id: str) -> SessionInspectionIdentity:
        """Load bounded session identity without materializing session metadata."""

    @abstractmethod
    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        """Update session status and return the updated session."""

    @abstractmethod
    async def transition_status(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        """Atomically transition status, claiming a new epoch when entering RUNNING.

        Entering ``RUNNING`` must reject an incomplete-recovery claim whose
        lease has expired at the store's authoritative transaction time.
        """

    @abstractmethod
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
        """Atomically persist status, checkpoint, interaction, target, and profile admission.

        Entering ``RUNNING`` must reject an incomplete-recovery claim whose
        lease has expired at the store's authoritative transaction time.
        """

    async def admit_execution_profile_resume(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform: CheckpointTransform | None = None,
        store_time_checkpoint_transform: StoreTimeCheckpointTransform | None = None,
        result_checkpoint_transform: CheckpointTransform | None = None,
        execution_profile: ExecutionProfileIdentity,
        tool_capability_ceiling: ToolCapabilityCeiling | None = None,
        execution_profile_decision: ExecutionProfileDecision | None = None,
        adopted_runtime_identity: SessionRuntimeIdentity | None = None,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        continued_interaction_id: str | None = None,
        defer_interaction_source: bool = False,
        model_transition: SessionModelTransition | None = None,
        temporary_service_admission: TemporaryServiceAdmission | None = None,
    ) -> Session:
        """Atomically compare a profile through the normal resume transition."""

        if not self.supports_execution_profile_admission:
            raise NotImplementedError(
                "This SessionStore does not support atomic execution-profile admission."
            )
        transition_kwargs: dict[str, Any] = {
            "from_statuses": from_statuses,
            "to_status": to_status,
            "checkpoint_transform": checkpoint_transform,
            "result_checkpoint_transform": result_checkpoint_transform,
            "interaction_started_event": interaction_started_event,
            "interaction_source_messages": interaction_source_messages,
            "continued_interaction_id": continued_interaction_id,
            "defer_interaction_source": defer_interaction_source,
            "model_transition": model_transition,
            "execution_profile": execution_profile,
            "execution_profile_decision": execution_profile_decision,
            "tool_capability_ceiling": tool_capability_ceiling,
        }
        if adopted_runtime_identity is not None:
            transition_kwargs["adopted_runtime_identity"] = adopted_runtime_identity
        if temporary_service_admission is not None:
            transition_kwargs["temporary_service_admission"] = temporary_service_admission
        if store_time_checkpoint_transform is not None:
            transition_kwargs["store_time_checkpoint_transform"] = store_time_checkpoint_transform
        return await self.transition_status_and_checkpoint(session_id, **transition_kwargs)

    async def admit_session_invocation(
        self,
        session_id: str,
        *,
        admission: SessionInvocationAdmission,
    ) -> Session:
        """Atomically claim one valid profiled invocation and its interaction."""

        if not self._supports_active_invocation_execution_profile_protocol():
            raise NotImplementedError(
                "This SessionStore does not support atomic active-invocation execution profiles."
            )
        if not self.supports_execution_profile_admission:
            raise NotImplementedError(
                "This SessionStore does not support atomic execution-profile admission."
            )
        if type(admission) is not SessionInvocationAdmission:
            raise TypeError("admission must be a SessionInvocationAdmission.")
        if not checkpoint_preservation._INVOCATION_LIFECYCLE_AUTHORITY_MUTATION_ALLOWED.get():
            raise SessionRunFenced(
                "Invocation admission must use the typed lifecycle command boundary."
            )
        session_id = require_clean_nonblank(session_id, "session_id")
        active_profile = admission.expected_active_invocation_profile
        if active_profile is not None and active_profile.session_id != session_id:
            raise ValueError("Active invocation profile belongs to another session.")
        interaction_id = (
            admission.continued_interaction_id
            if admission.interaction_started_event is None
            else admission.interaction_started_event.interaction_id
        )
        if interaction_id is None:
            raise ValueError("Invocation admission has no interaction identity.")

        def bind_active_invocation_profile(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            now: datetime | None = None,
        ) -> dict[str, Any]:
            from cayu.sessions._abandonment_checkpoint import ABANDONED_SESSION_CHECKPOINT_KEY

            if checkpoint is not None and ABANDONED_SESSION_CHECKPOINT_KEY in checkpoint:
                raise SessionStatusConflict("An abandoned session cannot be continued.")
            current_active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            if current_active_profile is not None and not (
                active_invocation_execution_profile_is_released(
                    current_active_profile,
                    session_id=current_session.id,
                    run_epoch=current_session.run_epoch,
                )
            ):
                raise SessionRunFenced(
                    "The previous invocation still owns terminal hooks or trailing cleanup."
                )
            if admission.store_time_checkpoint_transform is not None:
                assert now is not None
                updated = admission.store_time_checkpoint_transform(
                    current_session, checkpoint, now
                )
            else:
                assert admission.checkpoint_transform is not None
                updated = admission.checkpoint_transform(current_session, checkpoint)
            return checkpoint_with_active_invocation_execution_profile(
                updated,
                session_id=current_session.id,
                interaction_id=interaction_id,
                run_epoch=current_session.run_epoch + 1,
                profile=admission.execution_profile,
                expected=active_profile,
            )

        transition_kwargs: dict[str, Any] = {
            "from_statuses": set(admission.from_statuses),
            "to_status": SessionStatus.RUNNING,
            "checkpoint_transform": bind_active_invocation_profile,
            "interaction_started_event": admission.interaction_started_event,
            "interaction_source_messages": list(admission.interaction_source_messages),
            "continued_interaction_id": admission.continued_interaction_id,
            "defer_interaction_source": admission.defer_interaction_source,
            "model_transition": admission.model_transition,
            "execution_profile_decision": admission.execution_profile_decision,
            "tool_capability_ceiling": admission.tool_capability_ceiling,
        }
        if admission.store_time_checkpoint_transform is not None:
            del transition_kwargs["checkpoint_transform"]
            transition_kwargs["store_time_checkpoint_transform"] = bind_active_invocation_profile
        if admission.adopted_runtime_identity is not None:
            transition_kwargs["adopted_runtime_identity"] = admission.adopted_runtime_identity
        if active_profile is not None and (
            admission.interaction_started_event is not None
            or admission.execution_profile_decision is not None
        ):
            # A continued interaction compares its profile only when a decision
            # (pending work adopting a changed profile) is committed with it.
            transition_kwargs["execution_profile"] = admission.execution_profile
        if admission.result_checkpoint_transform is not None:
            transition_kwargs["result_checkpoint_transform"] = admission.result_checkpoint_transform
        if admission.temporary_service_admission is not None:
            transition_kwargs["temporary_service_admission"] = admission.temporary_service_admission
        if active_profile is None:
            return await self.admit_execution_profile_resume(
                session_id,
                execution_profile=admission.execution_profile,
                **transition_kwargs,
            )
        return await self.transition_status_and_checkpoint(
            session_id,
            **transition_kwargs,
        )

    def _supports_invocation_lifecycle_command_protocol(self) -> bool:
        """Authenticate one complete version-1 command implementation."""

        method_names = (
            "apply_invocation_lifecycle_command",
            "load",
            "load_checkpoint",
            "transform_checkpoint",
            "update_labels",
            "update_metadata",
            "set_summary",
            "load_interaction_transition_receipt",
            "load_invocation_settlement_transition",
            "_load_interaction_transition_receipt_by_event_id",
            "create",
            "admit_session_invocation",
            "admit_execution_profile_resume",
            "transition_status_and_checkpoint",
            "fence_run_and_transform_checkpoint",
            "reject_active_invocation_execution_profile",
            "reject_execution_profile_resume",
            "publish_interaction_transition",
            "settle_session_invocation",
            "release_session_invocation",
        )
        store_type = type(self)
        mro = type.__getattribute__(store_type, "__mro__")
        concrete_declarations = type.__getattribute__(store_type, "__dict__")
        if concrete_declarations.get("invocation_lifecycle_command_version") != 1:
            return False

        getattribute_owner = next(
            (
                owner
                for owner in mro
                if "__getattribute__" in type.__getattribute__(owner, "__dict__")
            ),
            None,
        )
        if getattribute_owner is None or (
            type.__getattribute__(getattribute_owner, "__dict__")["__getattribute__"]
            is not object.__getattribute__
        ):
            return False
        try:
            instance_state = object.__getattribute__(self, "__dict__")
        except AttributeError:
            instance_state = None
        protected_names = (*method_names, "invocation_lifecycle_command_version")
        if isinstance(instance_state, dict) and any(
            name in instance_state for name in protected_names
        ):
            return False

        for method_name in method_names:
            owner = next(
                (
                    candidate
                    for candidate in mro
                    if method_name in type.__getattribute__(candidate, "__dict__")
                ),
                None,
            )
            if owner is None:
                return False
            declarations = type.__getattribute__(owner, "__dict__")
            implementation = declarations[method_name]
            if type(implementation) is not FunctionType:
                return False
            if owner is not SessionStore and (
                declarations.get("invocation_lifecycle_command_version") != 1
            ):
                return False
            resolved = object.__getattribute__(self, method_name)
            if (
                type(resolved) is not MethodType
                or resolved.__self__ is not self
                or resolved.__func__ is not implementation
            ):
                return False
        return True

    @overload
    async def apply_invocation_lifecycle_command(
        self,
        command: CreateInvocationCommand | AdmitInvocationCommand | RebindInvocationCommand,
    ) -> InvocationMutationResult: ...

    @overload
    async def apply_invocation_lifecycle_command(
        self,
        command: RejectInvocationCommand,
    ) -> ExecutionProfileRejectionResult: ...

    @overload
    async def apply_invocation_lifecycle_command(
        self,
        command: SettleInvocationCommand,
    ) -> InteractionTransitionResult: ...

    @overload
    async def apply_invocation_lifecycle_command(
        self,
        command: ReleaseInvocationCommand,
    ) -> InvocationReleaseResult: ...

    async def apply_invocation_lifecycle_command(
        self,
        command: object,
    ) -> object:
        """Apply one command only after authenticating the store protocol."""

        if not self._supports_invocation_lifecycle_command_protocol():
            raise NotImplementedError(
                "This SessionStore does not support invocation lifecycle command version 1."
            )
        from cayu.runtime._invocation_lifecycle import (
            apply_invocation_lifecycle_command,
        )

        return await apply_invocation_lifecycle_command(
            self,
            cast("InvocationLifecycleCommand", command),
        )

    async def settle_session_invocation(self, command: Any) -> InteractionTransitionResult:
        """Atomically settle an interaction under exact active authority."""

        del command
        raise NotImplementedError(
            "This SessionStore does not support invocation settlement commands."
        )

    async def release_session_invocation(self, command: Any) -> Any:
        """Atomically revoke one exact active invocation epoch."""

        del command
        raise NotImplementedError("This SessionStore does not support invocation release commands.")

    def _supports_active_invocation_execution_profile_protocol(self) -> bool:
        """Return whether the advertised capability covers the active store methods."""

        capability_owner_index = next(
            index
            for index, owner in enumerate(type(self).__mro__)
            if "supports_active_invocation_execution_profiles" in owner.__dict__
        )
        rejection_owner_index = next(
            index
            for index, owner in enumerate(type(self).__mro__)
            if "reject_execution_profile_resume" in owner.__dict__
        )
        return (
            self.supports_active_invocation_execution_profiles is True
            and capability_owner_index <= rejection_owner_index
        )

    def _supports_terminal_interaction_publication_protocol(self) -> bool:
        """Authenticate the optional paired terminal-publication extension."""

        method_names = (
            "transition_status_and_checkpoint",
            "publish_checkpoint_and_events",
            "publish_interaction_transition",
            "_mark_model_completion_stage_dispatched_atomic",
        )
        store_type = type(self)
        mro = type.__getattribute__(store_type, "__mro__")
        concrete_declarations = type.__getattribute__(store_type, "__dict__")
        if concrete_declarations.get("terminal_interaction_publication_version") != 1:
            return False

        getattribute_owner = next(
            (
                owner
                for owner in mro
                if "__getattribute__" in type.__getattribute__(owner, "__dict__")
            ),
            None,
        )
        if getattribute_owner is None or (
            type.__getattribute__(getattribute_owner, "__dict__")["__getattribute__"]
            is not object.__getattribute__
        ):
            return False
        try:
            instance_state = object.__getattribute__(self, "__dict__")
        except AttributeError:
            instance_state = None
        protected_names = (*method_names, "terminal_interaction_publication_version")
        if isinstance(instance_state, dict) and any(
            name in instance_state for name in protected_names
        ):
            return False

        for method_name in method_names:
            owner = next(
                (
                    candidate
                    for candidate in mro
                    if method_name in type.__getattribute__(candidate, "__dict__")
                ),
                None,
            )
            if owner is None:
                return False
            declarations = type.__getattribute__(owner, "__dict__")
            implementation = declarations[method_name]
            if type(implementation) is not FunctionType:
                return False
            if owner is not SessionStore and (
                declarations.get("terminal_interaction_publication_version") != 1
            ):
                return False
            resolved = object.__getattribute__(self, method_name)
            if (
                type(resolved) is not MethodType
                or resolved.__self__ is not self
                or resolved.__func__ is not implementation
            ):
                return False
        return True

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
        """Atomically reject a changed resume profile without admitting work."""

        raise NotImplementedError(
            "This SessionStore does not support atomic execution-profile rejection."
        )

    async def reject_active_invocation_execution_profile(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str | None = None,
        expected_statuses: set[SessionStatus],
        expected_run_epoch: int,
        expected_active_invocation_profile: ActiveInvocationExecutionProfile,
        candidate_profile: ExecutionProfileIdentity,
        event: Event,
        decision: ExecutionProfileDecision | None = None,
    ) -> ExecutionProfileRejectionResult:
        """Atomically compare checkpoint authority and reject changed continuation."""

        if not self._supports_active_invocation_execution_profile_protocol():
            raise NotImplementedError(
                "This SessionStore does not support atomic active-invocation "
                "execution-profile rejection."
            )
        expected_authority = checkpoint_value_authority(
            expected_active_invocation_profile.model_dump(mode="json"),
            "expected_active_invocation_execution_profile",
        )
        return await self.reject_execution_profile_resume(
            session_id,
            expected_session_instance_id=expected_session_instance_id,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_profile=expected_active_invocation_profile.profile,
            candidate_profile=candidate_profile,
            event=event,
            decision=decision,
            expected_active_invocation_profile_authority=expected_authority,
        )

    @abstractmethod
    async def transition_status_if_no_queued_messages(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_mutation: dict[str, Any] | None = None,
    ) -> Session:
        """Atomically terminalize only when no durable queued input remains.

        A completion-finalization checkpoint mutation keeps RUNNING at the
        current epoch while installing its existing owned finalization marker.
        It does not admit a new invocation or rewrite an interaction receipt.
        """

    @abstractmethod
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
        """Atomically publish one interaction state and its session transition.

        A completion guarded by ``only_if_no_queued_messages`` always publishes
        the interaction event, but leaves the session status unchanged when
        queued input remains. Repeating the exact complete transition
        reconstructs the committed result without re-evaluating that queue
        boundary. Reusing its event identity with different transition data
        fails closed. Implementations validate a runtime-supplied recovery claim
        against store-owned time after acquiring the mutation lock and before
        first publication, and reject an unclaimed first publication while a
        store-time-live recovery claim exists. Exact receipt replay remains
        valid after the lease has settled. When terminal evidence is supplied,
        the interaction event, session terminal event, terminal status, and
        exact winning decision commit in one transaction.
        """

    async def load_interaction_transition_receipt(
        self,
        session_id: str,
        *,
        transition: InteractionTransitionSpec,
        expected_recovery_claim_id: str | None = None,
    ) -> InteractionTransitionReceiptResult | None:
        """Load immutable evidence for one exact interaction transition.

        Stores that support acknowledgement-loss recovery must implement this
        read-only boundary. The default keeps existing custom stores source
        compatible while making unsupported cancellation reconciliation fail
        closed instead of issuing another transition publication. Claim-bound
        receipts require the exact original claim and reject a superseding
        durable claimant, while ordinary lease expiry does not invalidate
        exact acknowledgement-loss readback.
        """

        del session_id, transition, expected_recovery_claim_id
        raise NotImplementedError(
            "This session store does not support interaction-transition receipt lookup."
        )

    async def load_historical_interaction_settlement(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str,
        expected_event: Event,
        expected_profile: ExecutionProfileIdentity,
    ) -> InteractionTransitionReceiptResult:
        """Read exact prior settlement evidence without granting run authority.

        Unlike invocation release lookup, this permits a later current epoch.
        It cannot release a fence or authorize a historical invocation: callers
        must separately acquire their own execution or mutation authority.
        """
        session_id = require_clean_nonblank(session_id, "session_id")
        expected_session_instance_id = SessionInvocationBinding.validate_session_instance_id(
            expected_session_instance_id
        )
        event = copy_event(expected_event)
        profile = ExecutionProfileIdentity.model_validate(expected_profile.model_dump(mode="json"))
        if event.session_id != session_id or event.interaction_id is None:
            raise ValueError("Historical settlement requires an exact session interaction.")
        record = await self._load_historical_interaction_settlement_record(session_id, event.id)
        if record is None:
            raise RuntimeError("Historical interaction settlement has no immutable receipt.")
        receipt = interaction_receipts._load_interaction_transition_receipt(record)
        current = await self.load(session_id)
        active = receipt.invocation_active_profile
        if (
            current is None
            or current.instance_id != expected_session_instance_id
            or receipt.session.id != session_id
            or receipt.session.instance_id != expected_session_instance_id
            or receipt.invocation_session_instance_id != expected_session_instance_id
            or receipt.event != event
            or active is None
            or active.session_id != session_id
            or active.interaction_id != event.interaction_id
            or active.profile != profile
            or active.run_epoch != receipt.session.run_epoch
            or active.run_epoch >= current.run_epoch
        ):
            raise SessionRunFenced("Historical interaction settlement has conflicting lineage.")
        return InteractionTransitionReceiptResult(
            session=receipt.session,
            transition=interaction_receipts._interaction_transition_spec_from_receipt(receipt),
            status_changed=receipt.status_changed,
        )

    async def _load_historical_interaction_settlement_record(
        self, session_id: str, event_id: str
    ) -> dict[str, Any] | None:
        raise NotImplementedError("This store does not support historical settlement lookup.")

    async def load_invocation_settlement_transition(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str,
        expected_active_invocation_profile: ActiveInvocationExecutionProfile,
    ) -> InteractionTransitionSpec | None:
        """Resolve exact terminal settlement for one active invocation.

        Event lookup is bounded and supplies only the receipt identity. The
        backend authenticates the complete stored receipt and its invocation
        tuple before returning the transition. Release compares the same
        receipt again inside its mutation transaction.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        expected_session_instance_id = SessionInvocationBinding.validate_session_instance_id(
            expected_session_instance_id
        )
        if type(expected_active_invocation_profile) is not ActiveInvocationExecutionProfile:
            raise TypeError(
                "expected_active_invocation_profile must be an ActiveInvocationExecutionProfile."
            )
        if expected_active_invocation_profile.session_id != session_id:
            raise ValueError("Active invocation settlement belongs to another session.")
        admission_records = await self.query_events(
            EventQuery(
                session_id=session_id,
                interaction_id=expected_active_invocation_profile.interaction_id,
                event_type=EventType.INTERACTION_STARTED,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not admission_records:
            raise RuntimeError(
                "Active invocation authority has no durable interaction admission event."
            )
        records = await self.query_events(
            EventQuery(
                session_id=session_id,
                event_types=(
                    EventType.INTERACTION_PAUSED,
                    EventType.INTERACTION_COMPLETED,
                    EventType.INTERACTION_FAILED,
                    EventType.INTERACTION_INTERRUPTED,
                ),
                after_sequence=admission_records[0].sequence,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not records:
            return None
        receipt = await self._load_interaction_transition_receipt_by_event_id(
            session_id,
            event_id=records[0].event.id,
            expected_session_instance_id=expected_session_instance_id,
            expected_active_invocation_profile=expected_active_invocation_profile,
        )
        if receipt is None:
            raise RuntimeError(
                "Terminal interaction evidence exists without its immutable receipt."
            )
        return receipt.transition

    async def _load_interaction_transition_receipt_by_event_id(
        self,
        session_id: str,
        *,
        event_id: str,
        expected_session_instance_id: str,
        expected_active_invocation_profile: ActiveInvocationExecutionProfile,
    ) -> InteractionTransitionReceiptResult | None:
        """Load one store-authenticated receipt without caller-supplied fields."""

        del (
            session_id,
            event_id,
            expected_session_instance_id,
            expected_active_invocation_profile,
        )
        raise NotImplementedError(
            "This session store does not support invocation settlement receipt lookup."
        )

    @abstractmethod
    async def fence_stalled_run(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_for_seconds: int,
    ) -> Session | None:
        """Atomically evict a stale run and claim its newly incremented epoch.

        Returns ``None`` when the session is no longer in one of ``statuses`` or
        has not been inactive for ``inactive_for_seconds`` at store time. An
        existing incomplete-recovery claim must be reconciled through the exact
        checkpoint-authorized takeover path instead.
        """
        raise NotImplementedError

    @abstractmethod
    async def reserve_stalled_run_recovery(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_for_seconds: int | None,
        checkpoint_transform: StoreTimeCheckpointTransform,
    ) -> Session | None:
        """Reserve recovery at store time without yet advancing the run epoch.

        The reservation transform and eligibility check are atomic. The method
        intentionally leaves session activity unchanged so a subsequent typed
        invocation rebind can authenticate and fence the exact reserved state.
        """
        raise NotImplementedError

    @abstractmethod
    async def fence_run_and_transform_checkpoint(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        checkpoint_transform: CheckpointTransform,
        result_checkpoint_transform: CheckpointTransform | None = None,
    ) -> Session:
        """Atomically transform a checkpoint and claim a newly incremented epoch.

        The transform must return the replacement checkpoint. An exception or
        ``None`` result aborts both the checkpoint update and the epoch fence.
        This operation is for ownership that must persist a checkpoint lease
        and fence the prior run as one transaction. Recovery that needs only an
        inactivity predicate should continue to use :meth:`fence_stalled_run`.
        An incomplete-recovery claim present in the source checkpoint must
        still be live at the store's authoritative transaction time.
        """
        raise NotImplementedError

    @abstractmethod
    async def release_run_fence(self, session_id: str) -> None:
        """Revoke this task's epoch after all trailing writes finish."""
        invocation_scope._deactivate_session_run_fence(
            require_clean_nonblank(session_id, "session_id")
        )

    @abstractmethod
    async def claim_budget_reservation_identity(
        self,
        *,
        reservation_id: str,
        publication_session_id: str,
        publication_id: str,
    ) -> None:
        """Durably bind one ledger-wide reservation id to an exact session event.

        Claims are idempotent only for the same session/event pair and must not
        be removed when a session is deleted. ``append_events`` atomically marks
        the matching claim published when it stores ``budget.reserved``. The
        claim is a session-scoped mutation: the publication session must exist
        and an active run fence must still own its current epoch.
        """

    async def append_event(self, session_id: str, event: Event) -> None:
        """Append one event to a session."""
        await self.append_events(session_id, [event])

    @abstractmethod
    async def append_events(self, session_id: str, events: list[Event]) -> None:
        """Append events to a session in one durable batch."""

    async def append_workflow_step_started(
        self,
        session_id: str,
        event: Event,
        *,
        workflow_name: str,
        attempt_id: str,
    ) -> bool:
        """Atomically append a step reservation only for the current attempt.

        The attempt-fence check and event insert share one store transaction.
        ``False`` means either a newer attempt won or this exact reservation was
        already published.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not support atomic workflow step reservation."
        )

    async def append_tool_effect_conflict(self, request: object) -> Event:
        """Append exact late-dispatch evidence without granting run authority.

        Implementations compare the retained winner and session incarnation in
        the append transaction. This narrow operation accepts only an internal
        ToolEffectConflictAudit, constructs its own event, preserves liveness
        and all execution state, and returns original evidence on exact replay.
        A stale run context is permitted only for this evidence-only operation.
        """
        raise NotImplementedError("SessionStore does not support tool effect conflict audit.")

    async def load_mcp_manifest_baselines(
        self,
        history_keys: tuple[str, ...],
    ) -> McpManifestBaselineLoadResult:
        """Load authoritative baselines for the requested manifest-history keys."""

        raise NotImplementedError(
            f"{type(self).__name__} must implement durable MCP manifest history."
        )

    async def compare_and_publish_mcp_manifest_checks(
        self,
        session_id: str,
        *,
        expected_generations: dict[str, int | None],
        baseline_updates: dict[str, McpManifestBaseline],
        events: list[Event],
    ) -> McpManifestPublicationResult:
        """Atomically publish decisions and return the exact resulting requested baselines.

        Every accepted ``first_seen`` or ``changed`` event must install the
        matching authoritative baseline in the same publication. Every
        expected history key must have exactly one coherent outcome;
        ``unchanged`` must match its current baseline, and blocked outcomes
        must not update one.
        """

        raise NotImplementedError(
            f"{type(self).__name__} must implement atomic MCP manifest publication."
        )

    @abstractmethod
    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectClaim | None:
        """Claim the oldest eligible persisted event side-effect handoff."""

    @abstractmethod
    async def get_persisted_event_side_effect_delivery(
        self,
        *,
        session_id: str,
        event_id: str,
    ) -> PersistedEventSideEffectDelivery | None:
        """Load one persisted event side-effect handoff by event identity."""

    async def claim_first_persisted_event_side_effect(
        self, expected: PersistedEventSideEffectDelivery
    ) -> PersistedEventSideEffectClaim | None:
        """Claim only the exact unattempted snapshot under the ordinary claim lock.

        Never take over a lease or retry a failed delivery. Return None when
        unsupported or when any expected field changed; callers must reread.
        """
        session_event_rules._copy_pending_first_event_delivery(expected)
        return None

    async def retire_failed_first_event_delivery(
        self, expected: PersistedEventSideEffectDelivery
    ) -> PersistedEventSideEffectDelivery | None:
        """Atomically dead-letter an exact failed first delivery, preserving its error.

        Compare the complete snapshot under the claim lock. A competing retry
        wins or retirement wins, never both. Return None on conflict or when
        unsupported; that absence is not proof of callback settlement.
        """
        session_event_rules._copy_failed_first_delivery_retirement(expected)
        return None

    @abstractmethod
    async def mark_persisted_event_side_effect_delivered(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        """Mark one claimed event's configured side effects delivered."""

    @abstractmethod
    async def mark_persisted_event_side_effect_failed(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        error: str,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> PersistedEventSideEffectDelivery:
        """Release one failed claim for retry or dead-letter it."""

    @abstractmethod
    async def list_persisted_event_side_effect_deliveries(
        self,
        *,
        statuses: set[PersistedEventSideEffectStatus] | None = None,
        claimable_only: bool = False,
        after_sequence: int | None = None,
        limit: int = 100,
    ) -> list[PersistedEventSideEffectDelivery]:
        """Inspect persisted side-effect delivery state in event order.

        ``claimable_only`` excludes terminal deliveries and live leases.
        """

    async def get_persisted_event_side_effect_health(self) -> PersistedEventSideEffectHealth:
        """Read exact aggregate delivery health without loading source events."""
        raise NotImplementedError("This store does not support event side-effect health.")

    async def query_persisted_event_side_effect_deliveries(
        self,
        query: PersistedEventSideEffectQuery,
    ) -> PersistedEventSideEffectPage:
        """Read a bounded operational page; inspection never reserves a claim."""
        raise NotImplementedError("This store does not support event side-effect inspection.")

    async def defer_persisted_event_side_effect(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        """Release an exact claim without delivery or spending a failure attempt.

        Used when an internal continuation cannot yet acquire execution ownership.
        The persisted event remains eligible for bounded recovery; its old claim
        must no longer be capable of acknowledging a replacement delivery.
        """
        raise NotImplementedError("This SessionStore does not support side-effect deferral.")

    async def renew_persisted_event_side_effect(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectDelivery:
        """Extend an exact, still-live claim using the store's ownership clock.

        Expired ownership cannot be revived, even before another worker claims
        the event. Renewal preserves the claim identity and attempt count.
        """
        raise NotImplementedError("This SessionStore does not support side-effect renewal.")

    @abstractmethod
    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
        *,
        expected_authorized_target_instance_id: str | None = None,
    ) -> EnqueueSessionMessageResult:
        """Atomically accept queued input with its causal event, or replay it.

        When supplied by the authorization owner, the target incarnation must
        match under the transaction lock before either replay or admission.
        This fence is not a persisted message freshness condition.
        """

    async def append_peer_content(
        self,
        request: PeerContentAppendRequest,
        *,
        qualify_target: Callable[[Session], None] | None = None,
        pending_transcript_cursor: int | None = None,
    ) -> PeerContentReceipt:
        """Atomically append or replay one authenticated peer occurrence.

        The default is fail-closed: a custom store cannot accidentally expose
        peer content by inheriting a user-message implementation.
        Before queue insertion, native owners call the runtime's synchronous,
        side-effect-free qualifier with a detached transaction-owned target.
        Missing qualification permits only pending/excluded decisions or replay.
        pending_transcript_cursor is a receiving-owner fence for servicing an
        already retained exact pending operation. It never changes caller intent.
        """
        raise NotImplementedError("This SessionStore does not support peer content.")

    async def read_peer_content(
        self,
        append_key: PeerAppendKey,
    ) -> PeerContentReceipt | None:
        """Return the bounded append receipt for an exact key, if present."""
        raise NotImplementedError("This SessionStore does not support peer content.")

    async def exclude_peer_content(
        self,
        request: PeerContentAppendRequest,
        *,
        reason: str,
    ) -> PeerContentReceipt:
        """Record a definitive exclusion without pretending it was appended."""
        raise NotImplementedError("This SessionStore does not support peer content.")

    async def read_peer_content_attempt(
        self, request: PeerContentAppendRequest
    ) -> PeerContentReceipt | None:
        """Reconcile the complete expected operation, including excluded predecessors."""
        raise NotImplementedError("This SessionStore does not support exact peer reconciliation.")

    async def list_pending_peer_content(self, *, after_operation_key=None, limit=32):
        """Discover bounded peer responsibility independently of creation settlement."""
        raise NotImplementedError("This SessionStore does not support peer discovery.")

    async def retry_pending_peer_content(
        self,
        session_id: str,
        *,
        expected_session_instance_id: str,
        expected_run_epoch: int,
        expected_transcript_cursor: int,
        admit=None,
    ) -> tuple[PeerContentReceipt, ...]:
        """Retry retained peer appends only when the existing host reaches a boundary."""
        raise NotImplementedError("This SessionStore does not support peer delivery servicing.")

    async def record_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        raise NotImplementedError("This SessionStore does not support peer exposure receipts.")

    async def begin_peer_content_exposure(
        self, request: PeerContentExposureRequest
    ) -> PeerContentExposureReceipt:
        """Durably reserve an exposure before provider serialization."""
        raise NotImplementedError("This SessionStore does not support peer exposure receipts.")

    async def read_peer_content_exposure(
        self, append_key: PeerAppendKey, exposure_id: str
    ) -> PeerContentExposureReceipt | None:
        raise NotImplementedError("This SessionStore does not support peer exposure receipts.")

    async def inspect_session_messages(
        self,
        query: SessionMessageQuery,
        *,
        expected_authorized_session_instance_id: str | None = None,
    ) -> SessionMessageInspection:
        """Read one bounded delivery-priority page, retaining unreadable row identities.

        Check the optional authorization-owner incarnation before reading queue
        content, including on the first page without a cursor.
        """
        raise NotImplementedError("Session-message inspection is not supported by this store.")

    async def apply_session_message_action(
        self, request: SessionMessageActionRequest
    ) -> SessionMessageActionResult:
        """Atomically withdraw/quarantine an exact undelivered row and publish its event."""
        raise NotImplementedError(
            "Session-message terminal actions are not supported by this store."
        )

    async def snapshot_session_message_source(
        self,
        session_id: str,
        *,
        include_transcript_digest: bool = False,
        include_checkpoint_digest: bool = False,
        expected_authorized_session_instance_id: str | None = None,
    ) -> SessionMessageSource:
        """Capture source identity/cursor and requested digests under one store snapshot.

        Check the optional authorization-owner incarnation before accessing
        transcript or checkpoint content, within that same snapshot.
        """
        raise NotImplementedError(
            "Session-message source snapshots are not supported by this store."
        )

    @abstractmethod
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
        """Atomically append and mark one bounded queue batch delivered.

        Runtime callers supply a stable ``delivery_id`` for acknowledgement-loss
        reconstruction. Omitting it creates a one-shot identity for direct store
        callers and returns that identity in the result.

        ``reject_only`` retires stale/expired records without delivering valid
        input, for a runtime that has exhausted its model-step authority. Valid
        records remain pending; ``has_more`` then describes remaining rejection
        work inside the fixed eligibility boundary. The mode is part of the
        exact delivery identity and cannot change on replay.
        """

    async def repair_queued_interaction_profile_handoff(
        self,
        session_id: str,
        *,
        interaction_started_event: Event,
        profile_handoff: QueuedInteractionProfileHandoff,
    ) -> ActiveInvocationExecutionProfile:
        """Repair one historical split delivery from exact provider-stage evidence.

        This compatibility seam is intentionally fail-closed for custom stores.
        New deliveries must bind the target profile in
        :meth:`deliver_queued_session_messages` instead.
        """

        raise NotImplementedError(
            "This SessionStore does not support historical queued handoff repair."
        )

    @abstractmethod
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
        """Atomically transform a checkpoint and append its causal event batch."""

    async def publish_checkpoint_and_events_with_store_time(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        checkpoint_transform: StoreTimeCheckpointTransform,
        commit_time_guard: Callable[[datetime], None],
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        """Publish checkpoint state using the backend's transactional clock."""

        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        if commit_time_guard is None:
            raise TypeError("commit_time_guard is required.")

        def operation_transform(
            session: Session,
            checkpoint: dict[str, Any] | None,
            _record: dict[str, Any] | None,
            store_now: datetime,
        ) -> SessionOperationPublication:
            transformed = checkpoint_transform(session, checkpoint, store_now)
            if transformed is None:
                raise ValueError("Checkpoint transform must return a checkpoint.")
            return SessionOperationPublication(checkpoint=transformed)

        return await self.publish_session_operation_guarded_with_store_time(
            session_id,
            idempotency_key=idempotency_key,
            operation_transform=operation_transform,
            commit_guard=lambda: None,
            commit_time_guard=commit_time_guard,
            events=events,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )

    async def _publish_completion_result_event_publication(
        self,
        session_id: str,
        *,
        checkpoint_transform: StoreTimeCheckpointTransform,
        events: list[Event],
    ) -> Session:
        """Mutate the private publication root with authoritative store time."""

        raise NotImplementedError(
            "This SessionStore does not own completion-result event publication."
        )

    @abstractmethod
    async def load_session_operation(
        self,
        session_id: str,
        idempotency_key: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        """Load one terminal durable operation record by caller idempotency key.

        A supplied root guard runs in the same read snapshot before the
        checkpoint-coupled operation record is interpreted.
        """

    @abstractmethod
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
        """Atomically publish a checkpoint, events, and terminal operation records."""

    def _supports_session_export_protocol(self) -> bool:
        """Qualify exact owner methods, not an inherited version flag alone."""
        return (
            type(self.session_export_version) is int
            and self.session_export_version == 1
            and _session_export_methods_owned(self)
        )

    def _supports_session_continuation_protocol(self) -> bool:
        return (
            type(self.session_continuation_version) is int
            and self.session_continuation_version == 1
            and _session_continuation_methods_owned(self)
        )

    def supports_external_waits(self) -> bool:
        return (
            type(self.external_wait_version) is int
            and self.external_wait_version == 1
            and _session_owner_methods_owned(
                self,
                "external_wait_version",
                (
                    "_mutate_external_wait",
                    "_read_external_wait",
                    "_list_external_waits",
                    "_retire_external_wait_scope",
                    "_read_external_wait_retirement",
                    "_prune_external_wait_scope",
                    "create",
                    "apply_invocation_lifecycle_command",
                    "_require_external_creation_unlocked",
                    "_require_external_creation",
                    "_require_external_wait_admission_unlocked",
                    "_require_external_wait_admission",
                    "transition_status",
                    "transition_status_and_checkpoint",
                    "_connection",
                    "delete_session",
                    "validate_session_closure_admission",
                    "_require_session_erasure_quiescence_unlocked",
                    "_require_session_erasure_quiescence",
                    "_require_external_wait_quiescence_unlocked",
                ),
            )
        )

    async def _read_external_wait_retirement(
        self, scope: ExternalWaitScope
    ) -> ExternalWaitRetirement | None:
        raise NotImplementedError("Store does not qualify external waits.")

    async def _retire_external_wait_scope(
        self, request: ExternalWaitRetirementRequest
    ) -> ExternalWaitRetirement:
        raise NotImplementedError("Store does not qualify external waits.")

    async def _prune_external_wait_scope(
        self, retirement: ExternalWaitRetirement, *, limit: int
    ) -> ExternalWaitPruneResult:
        raise NotImplementedError("Store does not qualify external waits.")

    async def _mutate_external_wait(self, command: ExternalWaitMutation) -> ExternalWaitRecord:
        raise NotImplementedError("Store does not qualify external waits.")

    async def _read_external_wait(
        self, scope: ExternalWaitScope, correlation_key: str
    ) -> ExternalWaitRecord | None:
        raise NotImplementedError("Store does not qualify external waits.")

    async def _list_external_waits(
        self, scope: ExternalWaitScope, *, source: str, after: str, limit: int
    ) -> tuple[ExternalWaitRecord, ...]:
        raise NotImplementedError("Store does not qualify external waits.")

    async def _list_session_external_wait_handoffs(
        self, session_ids: tuple[str, ...]
    ) -> tuple[ExternalWaitRecord, ...]:
        """Waits bound to one of these sessions whose continuation is not settled yet."""
        raise NotImplementedError("Store does not report external waits by session.")

    async def _observe_external_wait_schedule(self) -> ExternalWaitScheduleObservation:
        """The earliest open deadline and the elected outcomes still awaiting service."""
        raise NotImplementedError("Store does not report external wait schedules.")

    def _supports_producer_attachment_protocol(self) -> bool:
        return (
            type(self._producer_attachment_version) is int
            and self._producer_attachment_version == 1
            and self._supports_session_steering_protocol() is True
            and _session_owner_methods_owned(
                self,
                "_producer_attachment_version",
                (
                    *_SESSION_EXPORT_OWNER_METHODS,
                    "_attach_native_producer",
                    "_read_native_producer_attachment",
                    "_admit_native_producer",
                    "_exclude_native_producer",
                    "_read_native_producer_exclusion",
                    "_acknowledge_native_producer_cleanup",
                    "_retain_native_producer_output",
                    "_read_retained_native_producer_output",
                    "_read_native_producer_release",
                    "_read_native_producer_progress",
                    "_complete_native_producer_cleanup",
                    "_read_completed_native_producer_cleanup",
                    "_retire_native_producer_cleanup",
                    "query_events",
                    "load_historical_interaction_settlement",
                    "_load_historical_interaction_settlement_record",
                    "_load_interaction_transition_receipt_by_event_id",
                    "apply_invocation_lifecycle_command",
                ),
            )
        )

    async def _attach_native_producer(self, registration):
        """Registered runtime handoff, not a caller-provided publication callback."""
        from cayu.runtime._producer_output_store import attach_native_producer

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("This store does not qualify native producer attachment.")
        return await attach_native_producer(self, registration)

    async def _read_native_producer_attachment(self, command):
        raise NotImplementedError("Atomic native producer attachment readback is not qualified.")

    async def _admit_native_producer(self, registration, command):
        from cayu.runtime._producer_output_store import admit_native_producer

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("This store does not qualify native producer admission.")
        return await admit_native_producer(self, registration, command)

    async def _exclude_native_producer(self, registration, control):
        """Cleanup-only registered handoff; never accept a public receipt as authority."""
        from cayu.runtime._producer_output_store import exclude_native_producer

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("This store does not qualify native producer exclusion.")
        return await exclude_native_producer(self, registration, control)

    async def _read_native_producer_exclusion(self, registration, control):
        from cayu.runtime._producer_output_store import read_native_exclusion

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("This store does not qualify native producer exclusion.")
        return await read_native_exclusion(self, registration, control)

    async def _retain_native_producer_output(self, session_id, *, invocation, stage_id):
        from cayu.runtime._producer_output_store import retain_native_output

        return await retain_native_output(
            self, session_id, invocation=invocation, stage_id=stage_id
        )

    async def _read_retained_native_producer_output(self, command):
        from cayu.runtime._producer_output_store import read_retained_native_output

        return await read_retained_native_output(self, command)

    async def _read_native_producer_release(self, command):
        raise NotImplementedError("Atomic native producer release readback is not qualified.")

    async def _read_native_producer_progress(self, command, *, kind):
        raise NotImplementedError("Atomic native producer progress readback is not qualified.")

    async def _complete_native_producer_cleanup(self, registration, *, authority):
        raise NotImplementedError("Durable native producer cleanup is not qualified.")

    async def _read_completed_native_producer_cleanup(self, registration):
        raise NotImplementedError("Durable native producer cleanup readback is not qualified.")

    async def _retire_native_producer_cleanup(self, retirement, *, authority, limit):
        raise NotImplementedError("Native producer cleanup retirement is not qualified.")

    async def _acknowledge_native_producer_cleanup(self, registration, control, cleanup):
        from cayu.runtime._producer_output_store import acknowledge_native_cleanup

        if not self._supports_producer_attachment_protocol():
            raise NotImplementedError("Native producer cleanup is not qualified.")
        return await acknowledge_native_cleanup(self, registration, control, cleanup)

    def _supports_owned_off_thread_session_commit_guard_protocol(self) -> bool:
        """Return whether this exact guarded-publication override owns its capability."""

        capability_owner_index = next(
            index
            for index, owner in enumerate(type(self).__mro__)
            if "supports_owned_off_thread_session_commit_guards" in owner.__dict__
        )
        publication_owner_index = next(
            index
            for index, owner in enumerate(type(self).__mro__)
            if "publish_session_operation_guarded" in owner.__dict__
        )
        return (
            self.supports_owned_off_thread_session_commit_guards is True
            and capability_owner_index <= publication_owner_index
        )

    def _supports_atomic_model_completion_stage_release_protocol(self) -> bool:
        """Return whether this exact publication override owns stage release semantics."""

        capability_owner_index = next(
            index
            for index, owner in enumerate(type(self).__mro__)
            if "supports_atomic_model_completion_stage_release" in owner.__dict__
        )
        publication_owner_index = next(
            index
            for index, owner in enumerate(type(self).__mro__)
            if "publish_session_operation" in owner.__dict__
        )
        return (
            self.supports_atomic_model_completion_stage_release is True
            and capability_owner_index <= publication_owner_index
        )

    def _supports_session_steering_protocol(self) -> bool:
        """Require one owner for acceptance, completion, and new-work exclusion."""

        mro = type(self).__mro__
        capability_owner = next(
            index for index, owner in enumerate(mro) if "session_steering_version" in owner.__dict__
        )
        return (
            type(self.session_steering_version) is int
            and self.session_steering_version == 1
            and all(
                capability_owner
                <= next(
                    (index for index, owner in enumerate(mro) if method in owner.__dict__),
                    len(mro),
                )
                for method in (
                    "publish_session_operation",
                    "_publish_session_operation",
                    "prepare_model_completion_stage",
                    "_prepare_model_completion_stage_atomic",
                    "deliver_queued_session_messages",
                    "publish_interaction_transition",
                    "settle_session_invocation",
                )
            )
        )

    def _supports_completion_result_event_publication_reservation_protocol(self) -> bool:
        """Return whether this exact store owns the complete reservation contract."""

        mro = type(self).__mro__
        capability_owner_index = next(
            index
            for index, owner in enumerate(mro)
            if "supports_completion_result_event_publication_reservations" in owner.__dict__
        )
        required_owner_indexes = tuple(
            next(index for index, owner in enumerate(mro) if method_name in owner.__dict__)
            for method_name in (
                "create",
                "create_fork",
                "create_fork_with_transcript_validation",
                "create_profiled_fork",
                "transition_status_and_checkpoint",
                "admit_execution_profile_resume",
                "admit_session_invocation",
                "fence_run_and_transform_checkpoint",
                "checkpoint",
                "transform_checkpoint",
                "delete_session",
                "publish_checkpoint_and_events",
                "publish_session_operation",
                "publish_session_operation_guarded",
                "replace_initial_transcript_messages",
                "append_transcript_messages_and_transform_checkpoint",
                "_publish_completion_result_event_publication",
            )
        )
        return self.supports_completion_result_event_publication_reservations is True and all(
            capability_owner_index <= index for index in required_owner_indexes
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
        """Publish with a guard evaluated at the backend's commit boundary."""

        raise NotImplementedError(
            f"{type(self).__name__} must implement atomic guarded operation publication."
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
        """Publish an operation using time sampled at the commit transaction."""

        raise NotImplementedError(
            f"{type(self).__name__} must implement store-time guarded operation publication."
        )

    async def _publish_continuation_operation(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        events: list[Event],
    ) -> Session:
        """Enclose only a session-owned transform, never a receiving callback."""
        from cayu.sessions._session_continuation_scope import publication_scope
        from cayu.sessions._session_continuation_store import index_publication

        def indexed_transform(session, checkpoint, current):
            publication = operation_transform(session, checkpoint, current)
            return index_publication(session, checkpoint, current, publication, key=idempotency_key)

        with (
            publication_scope(idempotency_key),
            checkpoint_preservation._invocation_lifecycle_authority_read_scope(),
        ):
            return await self.publish_session_operation(
                session_id,
                idempotency_key=idempotency_key,
                operation_transform=indexed_transform,
                events=events,
            )

    async def _publish_temporary_continuation_service(
        self,
        *,
        previous: TemporaryServiceRecord | None,
        proposed: TemporaryServiceRecord,
    ) -> TemporaryServiceRecord:
        """Publish a receiving owner's exact child under the native session lock.

        This private mutation is not an application execution entrance. The
        registered runtime owner must authenticate permits and receiving evidence
        before calling it; it cannot wrap foreign callbacks in publication scope.
        """
        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            ContinuationUnavailable,
            continuation_operation_key,
        )
        from cayu.sessions._session_continuation_scope import service_publication_scope
        from cayu.sessions._temporary_continuation import (
            TemporaryServiceRecord,
            temporary_service_key,
        )
        from cayu.sessions._temporary_continuation_store import (
            publish_service_record,
            require_service_deadline,
        )
        from cayu.vaults.redaction import SecretRedactor

        proposed = prepare_contract(TemporaryServiceRecord, proposed, redactor=SecretRedactor())
        if previous is not None:
            previous = prepare_contract(TemporaryServiceRecord, previous, redactor=SecretRedactor())
        ticket = proposed.intent.ticket
        key = continuation_operation_key(ticket)

        def transform(session, checkpoint, current, now):
            return publish_service_record(session, checkpoint, current, previous, proposed, now)

        def check_commit_time(now):
            if previous is None:
                require_service_deadline(proposed, now)

        with (
            service_publication_scope(key, temporary_service_key(proposed.intent.operation)),
            checkpoint_preservation._invocation_lifecycle_authority_read_scope(),
        ):
            await self.publish_session_operation_guarded_with_store_time(
                ticket.session_id,
                idempotency_key=key,
                operation_transform=transform,
                commit_guard=lambda: None,
                commit_time_guard=check_commit_time,
                events=[],
            )
            retained = await self.load_session_operation(
                ticket.session_id, temporary_service_key(proposed.intent.operation)
            )
        if retained is None:
            raise ContinuationUnavailable("Temporary service publication is not durably readable.")
        observed = prepare_contract(TemporaryServiceRecord, retained, redactor=SecretRedactor())
        if observed != proposed and not (
            proposed.state in {"returned", "excluded"}
            and observed == proposed.model_copy(update={"settlement_acknowledged": True})
        ):
            raise ContinuationUnavailable("Temporary service readback differs from publication.")
        return observed

    async def _prepare_temporary_side_service(self, preparation):
        """Native stores must reserve both existing-session preparations atomically."""
        from cayu.sessions._session_continuation import ContinuationUnavailable

        raise ContinuationUnavailable("Session store lacks joint side-service preparation.")

    async def _publish_temporary_service_target(self, *, previous, proposed):
        """Private registered-owner compare under the target's native transaction."""
        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            CONTINUATION_NAMESPACE_KEY,
            ContinuationUnavailable,
        )
        from cayu.sessions._session_continuation_scope import service_publication_scope
        from cayu.sessions._temporary_service_target import (
            TemporaryServiceTarget,
            publish_target_record,
            target_service_key,
        )
        from cayu.vaults.redaction import SecretRedactor

        proposed = prepare_contract(TemporaryServiceTarget, proposed, redactor=SecretRedactor())
        if previous is not None:
            previous = prepare_contract(TemporaryServiceTarget, previous, redactor=SecretRedactor())
        if previous == proposed:
            retained = await self._load_temporary_service_target(proposed.service.admission)
            if retained != proposed and not (
                proposed.service.state in {"returned", "excluded"}
                and retained == proposed.model_copy(update={"source_acknowledged": True})
            ):
                raise ContinuationUnavailable("Side-session target replay changed its exact state.")
            return retained
        target = proposed.service.intent.target
        key = target_service_key(proposed.service.intent.operation)

        def transform(session, checkpoint, current, now):
            return publish_target_record(session, checkpoint, current, previous, proposed, now)

        with (
            service_publication_scope(key, CONTINUATION_NAMESPACE_KEY),
            checkpoint_preservation._invocation_lifecycle_authority_read_scope(),
        ):
            await self.publish_session_operation_guarded_with_store_time(
                target.object_id,
                idempotency_key=key,
                operation_transform=transform,
                commit_guard=lambda: None,
                commit_time_guard=lambda now: None,
                events=[],
            )
            raw = await self.load_session_operation(target.object_id, key)
        if raw is None:
            raise ContinuationUnavailable("Side-session target publication is unreadable.")
        retained = prepare_contract(TemporaryServiceTarget, raw, redactor=SecretRedactor())
        if retained != proposed and not (
            proposed.service.state in {"returned", "excluded"}
            and retained == proposed.model_copy(update={"source_acknowledged": True})
        ):
            raise ContinuationUnavailable("Side-session target acknowledgement changed.")
        return retained

    async def _load_temporary_service_target(self, expected):
        """Read an exact indexed target, retaining ambiguity across concurrent changes."""
        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationUnavailable,
        )
        from cayu.sessions._session_continuation_scope import publication_scope
        from cayu.sessions._session_continuation_store import ROOT_KEY, ContinuationRoot
        from cayu.sessions._temporary_continuation import (
            TemporaryServiceAdmission,
            TemporaryServicePreparation,
        )
        from cayu.sessions._temporary_service_target import (
            TemporaryServiceTarget,
            target_reference,
            target_service_key,
        )
        from cayu.vaults.redaction import SecretRedactor

        expected = prepare_contract(
            TemporaryServiceAdmission
            if isinstance(expected, TemporaryServiceAdmission)
            else TemporaryServicePreparation,
            expected,
            redactor=SecretRedactor(),
        )
        intent = expected.dispatch.intent
        key = target_service_key(intent.operation)
        with (
            publication_scope(key),
            checkpoint_preservation._invocation_lifecycle_authority_read_scope(),
        ):
            session = await self.load(intent.target.object_id)
            checkpoint = await self.load_checkpoint(intent.target.object_id)
            raw = await self.load_session_operation(intent.target.object_id, key)
        if session is None or session.instance_id != intent.target.incarnation:
            raise ContinuationUnavailable("Side-session receiving incarnation is unavailable.")
        root_raw = None if checkpoint is None else checkpoint.get(ROOT_KEY)
        root = None if root_raw is None else ContinuationRoot.model_validate(root_raw)
        if root is not None and (
            root.namespace.session_id != session.id
            or root.namespace.session_instance_id != session.instance_id
            or root.namespace.owner != intent.target.owner
        ):
            raise ContinuationConflict("Side-session target index belongs to another owner.")
        reference = (
            None
            if root is None
            else next((item for item in root.target_services if item.key == key), None)
        )
        if raw is None and reference is None:
            return None
        if raw is None or reference is None:
            raise ContinuationUnavailable("Side-session target has incomplete native evidence.")
        retained = TemporaryServiceTarget.model_validate(raw)
        if target_reference(retained) != reference:
            raise ContinuationUnavailable("Side-session target changed during indexed readback.")
        observed = retained.service.admission
        if isinstance(observed, TemporaryServiceAdmission) and not isinstance(
            expected, TemporaryServiceAdmission
        ):
            observed = observed.preparation
        if observed != expected:
            raise ContinuationConflict("Side-session target conflicts with the expected operation.")
        return retained

    async def _load_temporary_continuation_service(
        self, expected: TemporaryServiceAdmission | TemporaryServicePreparation
    ) -> TemporaryServiceRecord | None:
        """Reconstruct one exact child, never treating a missing indexed row as absence."""
        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationUnavailable,
            require_ticket_identity,
        )
        from cayu.sessions._temporary_continuation import (
            TemporaryServiceAdmission,
            TemporaryServicePreparation,
            TemporaryServiceRecord,
            reference_for_service,
            temporary_service_key,
        )
        from cayu.vaults.redaction import SecretRedactor

        expected = (
            prepare_contract(TemporaryServiceAdmission, expected, redactor=SecretRedactor())
            if isinstance(expected, TemporaryServiceAdmission)
            else prepare_contract(TemporaryServicePreparation, expected, redactor=SecretRedactor())
        )
        ticket = expected.dispatch.intent.ticket
        retained = await self.load_continuation_ticket(
            ticket.session_id,
            session_instance_id=ticket.session_instance_id,
            registration_key=ticket.registration_key,
        )
        if retained is None:
            raise ContinuationUnavailable("Temporary service has no retained source ticket.")
        require_ticket_identity(ticket, retained.ticket)
        key = temporary_service_key(expected.dispatch.intent.operation)
        reference = next((item for item in retained.services if item.key == key), None)
        raw = await self.load_session_operation(ticket.session_id, key)
        if reference is None:
            if raw is not None:
                raise ContinuationUnavailable("Temporary service child has lost its source index.")
            return None
        if raw is None:
            raise ContinuationUnavailable("Temporary service index has lost its child evidence.")
        observed = prepare_contract(TemporaryServiceRecord, raw, redactor=SecretRedactor())
        if reference_for_service(observed, retained.services) != reference:
            # A concurrent transition may have advanced between these reads.
            # Neither the stale index nor the new child proves an exact match.
            raise ContinuationUnavailable("Temporary service changed during reconstruction.")
        observed_expected = (
            observed.admission.preparation
            if isinstance(observed.admission, TemporaryServiceAdmission)
            and not isinstance(expected, TemporaryServiceAdmission)
            else observed.admission
        )
        if observed_expected != expected:
            raise ContinuationConflict(
                "Temporary service readback conflicts with expected admission."
            )
        return observed

    async def _read_temporary_continuation_outcome(
        self, expected: TemporaryServiceAdmission
    ) -> TemporaryServiceRecord | None:
        """Read exact native admission/release; absence never proves exclusion."""
        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import ContinuationUnavailable
        from cayu.sessions._temporary_continuation import TemporaryServiceAdmission
        from cayu.sessions._temporary_continuation_store import native_service_outcome
        from cayu.vaults.redaction import SecretRedactor

        expected = prepare_contract(TemporaryServiceAdmission, expected, redactor=SecretRedactor())
        target = expected.dispatch.intent.target
        session = await self.load(target.object_id)
        if session is None or session.instance_id != target.incarnation:
            raise ContinuationUnavailable("Temporary service receiving incarnation is unavailable.")
        with checkpoint_preservation._invocation_lifecycle_authority_read_scope():
            checkpoint = await self.load_checkpoint(session.id)
        return native_service_outcome(expected, checkpoint)

    async def _reconcile_temporary_continuation_service(
        self, expected: TemporaryServiceAdmission
    ) -> TemporaryServiceRecord:
        """Settle the source only from native receiving evidence, retaining ambiguity."""
        from cayu.sessions._session_continuation import ContinuationUnavailable

        previous = await self._load_temporary_continuation_service(expected)
        if previous is None:
            raise ContinuationUnavailable(
                "Temporary service has no source responsibility to reconcile."
            )
        target_record = None
        if expected.dispatch.intent.mode == "side_session":
            if previous.state in {"returned", "excluded"}:
                target = expected.dispatch.intent.target
                target_session = await self.load(target.object_id)
                if target_session is None or target_session.instance_id != target.incarnation:
                    # The exact source record already contains positive receiving
                    # settlement. Target absence grants nothing: it only means
                    # there is no old incarnation left to acknowledge/clean up.
                    # Never make terminal replay depend on an unrelated reuse of
                    # the target's public ID.
                    return previous
            target_record = await self._load_temporary_service_target(expected)
            if target_record is None:
                raise ContinuationUnavailable("Side-session receiving fence is unavailable.")
        if previous.state in {"returned", "excluded"}:
            if target_record is not None:
                from cayu.sessions._temporary_service_target import acknowledge_side_target

                await self._publish_temporary_service_target(
                    previous=target_record,
                    proposed=acknowledge_side_target(target_record, previous),
                )
            return previous
        observed = await self._read_temporary_continuation_outcome(expected)
        if observed is None:
            if previous.state == "admitted":
                raise ContinuationUnavailable(
                    "Temporary service native admission evidence is unavailable."
                )
            return previous
        if target_record is not None and observed.state == "returned":
            from cayu.sessions._temporary_service_target import return_side_target

            target_record = await self._publish_temporary_service_target(
                previous=target_record,
                proposed=return_side_target(target_record, observed),
            )
        if observed == previous:
            return previous
        settled = await self._publish_temporary_continuation_service(
            previous=previous, proposed=observed
        )
        if target_record is not None and settled.state == "returned":
            from cayu.sessions._temporary_service_target import acknowledge_side_target

            await self._publish_temporary_service_target(
                previous=target_record,
                proposed=acknowledge_side_target(target_record, settled),
            )
        return settled

    async def _initialize_continuation_namespace(
        self, namespace: ContinuationNamespace
    ) -> ContinuationNamespace:
        """Retain one immutable application/session owner at the fixed bootstrap key."""
        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            CONTINUATION_NAMESPACE_KEY,
            ContinuationConflict,
            ContinuationNamespace,
            ContinuationUnavailable,
        )
        from cayu.vaults.redaction import SecretRedactor

        namespace = prepare_contract(ContinuationNamespace, namespace, redactor=SecretRedactor())
        from cayu.sessions._session_continuation_scope import (
            require_namespace_preparation,
            require_preparation_writer,
        )

        require_namespace_preparation(namespace)

        def transform(session, checkpoint, current):
            if (
                session.id != namespace.session_id
                or session.instance_id != namespace.session_instance_id
            ):
                raise SessionRunFenced(
                    "Continuation namespace belongs to another session instance."
                )
            if current is not None and ContinuationNamespace.model_validate(current) != namespace:
                raise ContinuationConflict("Session continuation owner binding already exists.")
            if current is None:
                require_preparation_writer(session, checkpoint)
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={CONTINUATION_NAMESPACE_KEY: namespace.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            namespace.session_id,
            idempotency_key=CONTINUATION_NAMESPACE_KEY,
            operation_transform=transform,
            events=[],
        )
        retained = await self.load_session_operation(
            namespace.session_id, CONTINUATION_NAMESPACE_KEY
        )
        if retained is None or ContinuationNamespace.model_validate(retained) != namespace:
            raise ContinuationUnavailable("Continuation namespace is not durably readable.")
        return namespace

    async def prepare_continuation_ticket(
        self, command: ContinuationPreparation
    ) -> ContinuationRecord:
        """Atomically retain one exact continuation ticket in ``ARMING`` state."""

        from cayu.collaboration._contracts import CollaborationConflict
        from cayu.collaboration._preparation import prepare_contract, require_exact_contract
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationPreparation,
            ContinuationRecord,
            ContinuationUnavailable,
            continuation_namespace_id,
            continuation_operation_key,
            require_writer_generation,
        )
        from cayu.vaults.redaction import SecretRedactor

        command = prepare_contract(ContinuationPreparation, command, redactor=SecretRedactor())
        from cayu.sessions._session_continuation_scope import (
            require_preparation,
            require_preparation_writer,
        )

        require_preparation(command)
        from cayu.sessions._session_continuation_store import require_reserved_capacity

        require_reserved_capacity(command)
        ticket = command.intent
        session = await self.load(ticket.session_id)
        if session is None:
            raise KeyError(f"Session not found: {ticket.session_id}")
        if (
            session.instance_id != ticket.session_instance_id
            or ticket.namespace.session_id != session.id
            or ticket.namespace.session_instance_id != session.instance_id
            or ticket.namespace.namespace_id
            != continuation_namespace_id(session.id, session.instance_id, ticket.owner)
            or ticket.state != "ARMING"
            or ticket.revision != 1
        ):
            raise SessionRunFenced("Continuation ticket lacks the current session authority.")
        await self._initialize_continuation_namespace(ticket.namespace)
        key = continuation_operation_key(ticket)

        def transform(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            current: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            if current_session.instance_id != ticket.session_instance_id:
                raise SessionRunFenced("Continuation ticket belongs to a retired session instance.")
            if current is None:
                require_preparation_writer(current_session, checkpoint)
                if current_session.run_epoch != ticket.writer_generation:
                    raise SessionRunFenced(
                        "Continuation ticket lacks the current writer generation."
                    )
                require_writer_generation(ticket, current_session.run_epoch)
                record = ContinuationRecord(
                    namespace=ticket.namespace, preparation=command, ticket=ticket
                )
            else:
                record = ContinuationRecord.model_validate(current)
                try:
                    require_exact_contract(command, record.preparation, redactor=SecretRedactor())
                except CollaborationConflict:
                    raise ContinuationConflict("Continuation preparation conflicts.") from None
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: record.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            ticket.session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
        record = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if record is None:
            raise ContinuationUnavailable("Continuation preparation was not durably readable.")
        return record

    async def lookup_continuation_ticket(
        self, expected: ContinuationPreparation
    ) -> ExactLookup[ContinuationRecord]:
        """Return exact retained preparation evidence, never infer an outcome."""

        from cayu.collaboration._contracts import (
            CollaborationConflict,
            ExactConflict,
            ExactMatch,
            ExactNotFound,
            ExactUnavailable,
        )
        from cayu.collaboration._preparation import prepare_contract, require_exact_contract
        from cayu.sessions._session_continuation import (
            ContinuationPreparation,
            ContinuationRecord,
        )
        from cayu.vaults.redaction import SecretRedactor

        expected = prepare_contract(ContinuationPreparation, expected, redactor=SecretRedactor())
        try:
            record = await self.load_continuation_ticket(
                expected.intent.session_id,
                registration_key=expected.operation.caller_key,
                session_instance_id=expected.intent.session_instance_id,
            )
        except (PermissionError, SessionRunFenced):
            raise
        except Exception:
            # A failed/malformed authoritative read is an explicit lookup
            # outcome, never proof of absence. Cancellation/fatal signals are
            # deliberately outside this ordinary-dependency boundary.
            return ExactUnavailable()
        if record is None:
            return ExactNotFound()
        try:
            require_exact_contract(expected, record.preparation, redactor=SecretRedactor())
        except CollaborationConflict:
            return ExactConflict()
        return ExactMatch[ContinuationRecord](receipt=record)

    async def load_continuation_ticket(
        self,
        session_id: str,
        *,
        registration_key: str,
        session_instance_id: str,
    ) -> ContinuationRecord | None:
        """Read one continuation aggregate without treating absence as proof."""

        from cayu.sessions._session_continuation import (
            CONTINUATION_NAMESPACE_KEY,
            ContinuationNamespace,
            ContinuationRecord,
            continuation_operation_key_for_registration,
        )

        session = await self.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        if session.instance_id != session_instance_id:
            raise SessionRunFenced("Continuation readback belongs to a retired session instance.")
        raw_namespace = await self.load_session_operation(session_id, CONTINUATION_NAMESPACE_KEY)
        if raw_namespace is None:
            return None
        namespace = ContinuationNamespace.model_validate(raw_namespace)
        if (
            namespace.session_id != session.id
            or namespace.session_instance_id != session_instance_id
        ):
            raise SessionRunFenced("Continuation namespace belongs to another session instance.")
        key = continuation_operation_key_for_registration(
            session_id=session.id,
            session_instance_id=session_instance_id,
            namespace_id=namespace.namespace_id,
            registration_key=registration_key,
        )
        raw = await self.load_session_operation(session.id, key)
        if raw is None:
            return None
        record = ContinuationRecord.model_validate(raw)
        from cayu.sessions._session_continuation_store import require_history

        require_history(record)
        if (
            record.namespace != namespace
            or record.ticket.session_id != session.id
            or record.ticket.session_instance_id != session_instance_id
            or record.ticket.registration_key != registration_key
        ):
            raise SessionRunFenced("Continuation readback belongs to another session authority.")
        return record

    async def _recover_continuation_writer(
        self, ticket: ContinuationTicket, *, park: bool = False
    ) -> ContinuationRecord:
        """Retain genuine recovery succession without replacing ticket identity."""
        from cayu.collaboration._preparation import prepare_contract
        from cayu.runtime._continuation_recovery import require_recovery_writer
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationRecord,
            ContinuationTicket,
            ContinuationUnavailable,
            continuation_operation_key,
        )
        from cayu.sessions._session_continuation_scope import publication_scope
        from cayu.sessions._session_continuation_store import index_publication
        from cayu.vaults.redaction import SecretRedactor

        ticket = prepare_contract(ContinuationTicket, ticket, redactor=SecretRedactor())
        if type(park) is not bool:
            raise TypeError("Continuation recovery action must be explicit.")
        key = continuation_operation_key(ticket)

        commit_authority: tuple[Session, dict[str, Any] | None, ContinuationRecord] | None = None

        def transform(current_session, checkpoint, current, now):
            nonlocal commit_authority
            if current is None:
                raise ContinuationConflict("Continuation recovery lacks its durable preparation.")
            record = ContinuationRecord.model_validate(current)
            writer = require_recovery_writer(
                ticket, current_session, checkpoint, record, now=now, require_attached_writer=park
            )
            commit_authority = (current_session, checkpoint, record)
            if park and record.recovery_writer != writer:
                raise ContinuationConflict(
                    "Continuation parking lacks retained recovery succession."
                )
            updated = record.model_copy(update={"recovery_writer": writer})
            if park and record.ticket.state == "ARMING":
                updated = updated.model_copy(
                    update={
                        "ticket": record.ticket.model_copy(
                            update={"state": "WAITING", "revision": record.ticket.revision + 1}
                        )
                    }
                )
            publication = SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: updated.model_dump(mode="json")},
            )
            return index_publication(current_session, checkpoint, current, publication, key=key)

        def guard_time(now):
            if commit_authority is None:
                raise ContinuationConflict("Continuation recovery did not validate its writer.")
            current_session, checkpoint, record = commit_authority
            require_recovery_writer(
                ticket, current_session, checkpoint, record, now=now, require_attached_writer=park
            )

        with (
            publication_scope(key),
            checkpoint_preservation._invocation_lifecycle_authority_read_scope(),
        ):
            await self.publish_session_operation_guarded_with_store_time(
                ticket.session_id,
                idempotency_key=key,
                operation_transform=transform,
                commit_guard=lambda: None,
                commit_time_guard=guard_time,
                events=[],
            )
        result = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if result is None:
            raise ContinuationUnavailable("Continuation recovery succession is not readable.")
        return result

    async def mark_continuation_waiting(self, ticket: ContinuationTicket) -> ContinuationRecord:
        """Publish the ARMING-to-WAITING transition under the ticket's CAS."""

        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationRecord,
            ContinuationTicket,
            ContinuationUnavailable,
            continuation_operation_key,
            require_ticket_identity,
            require_writer_generation,
        )
        from cayu.sessions._session_continuation_scope import require_park
        from cayu.vaults.redaction import SecretRedactor

        ticket = prepare_contract(ContinuationTicket, ticket, redactor=SecretRedactor())
        require_park(ticket)
        key = continuation_operation_key(ticket)

        def transform(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            current: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            if current_session.instance_id != ticket.session_instance_id:
                raise SessionRunFenced("Continuation ticket belongs to a retired session instance.")
            require_writer_generation(
                ticket,
                current_session.run_epoch,
                allow_released_next_generation=_continuation_writer_was_released(
                    current_session, checkpoint
                ),
            )
            if current is None:
                raise ContinuationConflict("Continuation ticket is not durably armed.")
            record = ContinuationRecord.model_validate(current)
            require_ticket_identity(ticket, record.ticket)
            if record.ticket.state == "WAITING":
                updated = record
            elif record.ticket.state == "ARMING" and record.ticket.revision == ticket.revision:
                updated_ticket = record.ticket.model_copy(
                    update={"state": "WAITING", "revision": record.ticket.revision + 1}
                )
                updated = record.model_copy(update={"ticket": updated_ticket})
            else:
                raise ContinuationConflict("Continuation ticket cannot enter WAITING state.")
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: updated.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            ticket.session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
        record = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if record is None:
            raise ContinuationUnavailable("Continuation wait transition was not durably readable.")
        return record

    async def latch_continuation(
        self,
        latch: ContinuationLatch,
    ) -> ContinuationRecord:
        """Publish one immutable readiness latch, including an ARMING latch."""

        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationRecord,
            ContinuationUnavailable,
            continuation_operation_key,
            require_latch_identity,
            require_record_writer_generation,
            require_ticket_identity,
        )
        from cayu.sessions._session_continuation_scope import require_authenticated_latch

        require_authenticated_latch(latch)

        ticket = latch.ticket
        key = continuation_operation_key(ticket)

        # Reconcile an already committed latch before re-entering the receiving
        # owner.  A lost acknowledgement must not turn an exact durable replay
        # into a fresh foreign read (or make replay depend on a source that has
        # since expired).  A missing latch still requires positive receiver
        # authentication below.
        existing = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if existing is not None and existing.latch is not None:
            require_ticket_identity(existing.latch.ticket, ticket)
            require_latch_identity(existing.latch, latch)
            return existing

        def transform(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            current: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            if current_session.instance_id != ticket.session_instance_id:
                raise SessionRunFenced("Continuation latch belongs to a retired session instance.")
            if current is None:
                raise ContinuationConflict("Continuation ticket is not durably armed.")
            record = ContinuationRecord.model_validate(current)
            require_record_writer_generation(
                record,
                current_session.run_epoch,
                allow_released_next_generation=_continuation_writer_was_released(
                    current_session, checkpoint
                ),
            )
            require_ticket_identity(ticket, record.ticket)
            if record.latch is not None:
                require_latch_identity(record.latch, latch)
                updated = record
            elif record.ticket.state in {"CONSUMED", "RETIRED"}:
                raise ContinuationConflict("A terminal continuation cannot accept a latch.")
            else:
                updated = record.model_copy(update={"latch": latch})
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: updated.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            ticket.session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
        record = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if record is None:
            raise ContinuationUnavailable("Continuation latch was not durably readable.")
        return record

    async def consume_continuation(
        self,
        consumption: ContinuationConsumption,
    ) -> ContinuationRecord:
        """Atomically elect one inline or queued continuation and replay it."""

        from cayu.sessions._session_continuation_scope import require_consumption

        require_consumption(consumption)

        return await self._consume_continuation(consumption, admission_boundary=False)

    async def _claim_continuation_admission(
        self, consumption: ContinuationConsumption
    ) -> tuple[ContinuationRecord, bool]:
        """Fence retirement before dispatching the typed admission mutation."""

        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationConsumption,
            ContinuationRecord,
            ContinuationUnavailable,
            continuation_operation_key,
            require_latch_identity,
            require_ticket_identity,
        )
        from cayu.sessions._session_continuation_scope import require_consumption
        from cayu.vaults.redaction import SecretRedactor

        consumption = prepare_contract(
            ContinuationConsumption, consumption, redactor=SecretRedactor()
        )
        require_consumption(consumption)
        if consumption.receipt_stage != "prepared":
            raise ContinuationConflict(
                "Admission claims require a prepared responsibility receipt."
            )
        ticket = consumption.ticket
        key = continuation_operation_key(ticket)
        acquired = False
        claim_id = uuid4().hex

        def transform(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            current: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            nonlocal acquired
            if current_session.instance_id != ticket.session_instance_id:
                raise SessionRunFenced(
                    "Continuation admission claim belongs to a retired session instance."
                )
            if current is None:
                raise ContinuationConflict("Continuation ticket is unavailable.")
            record = ContinuationRecord.model_validate(current)
            if record.consumption is None:
                raise ContinuationConflict("Continuation responsibility is not prepared.")
            existing = record.consumption
            require_ticket_identity(existing.ticket, ticket)
            require_latch_identity(existing.latch, consumption.latch)
            if existing.receipt_stage != "prepared":
                raise ContinuationConflict("Continuation responsibility is no longer prepared.")
            base_update = {
                "admission_claimed": False,
                "admission_claim_id": None,
            }
            comparable_existing = existing.model_copy(update=base_update)
            comparable_consumption = consumption.model_copy(update=base_update)
            if comparable_existing.model_dump(mode="json") != comparable_consumption.model_dump(
                mode="json"
            ):
                raise ContinuationConflict("Continuation admission differs from its receipt.")
            if existing.admission_claimed:
                updated = record
            else:
                acquired = True
                updated = record.model_copy(
                    update={
                        "consumption": existing.model_copy(
                            update={
                                "admission_claimed": True,
                                "admission_claim_id": claim_id,
                            }
                        )
                    }
                )
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: updated.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            ticket.session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
        record = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if record is None or record.consumption is None:
            raise ContinuationUnavailable("Continuation admission claim was not durably readable.")
        return record, acquired

    async def _release_continuation_admission_claim(
        self, consumption: ContinuationConsumption
    ) -> ContinuationRecord:
        """Fence an uncommitted claim, or exclude it from positive supersession evidence."""

        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationConsumption,
            ContinuationRecord,
            ContinuationUnavailable,
            continuation_operation_key,
            require_latch_identity,
            require_ticket_identity,
        )
        from cayu.sessions._session_continuation_scope import require_consumption
        from cayu.vaults.redaction import SecretRedactor

        consumption = prepare_contract(
            ContinuationConsumption, consumption, redactor=SecretRedactor()
        )
        require_consumption(consumption)
        if consumption.receipt_stage != "prepared":
            raise ContinuationConflict("Admission claim release requires a prepared receipt.")
        ticket = consumption.ticket
        key = continuation_operation_key(ticket)

        def transform(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            current: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            if current_session.instance_id != ticket.session_instance_id:
                raise SessionRunFenced(
                    "Continuation admission claim belongs to a retired session instance."
                )
            if current is None:
                raise ContinuationConflict("Continuation ticket is unavailable.")
            record = ContinuationRecord.model_validate(current)
            if record.consumption is None:
                raise ContinuationConflict("Continuation responsibility is not claimed.")
            existing = record.consumption
            require_ticket_identity(existing.ticket, ticket)
            require_latch_identity(existing.latch, consumption.latch)
            if (
                existing.receipt_stage != "prepared"
                or existing.admission_claimed != consumption.admission_claimed
                or existing.admission_claim_id != consumption.admission_claim_id
                or existing.model_copy(
                    update={
                        "admission_claimed": False,
                        "admission_claim_id": None,
                    }
                ).model_dump(mode="json")
                != consumption.model_copy(
                    update={
                        "admission_claimed": False,
                        "admission_claim_id": None,
                    }
                ).model_dump(mode="json")
            ):
                raise ContinuationConflict("Continuation admission claim differs from its receipt.")
            from cayu.sessions._invocation_lifecycle import (
                reconcile_invocation_admission_from_state,
                superseding_invocation_admission_digest_from_state,
            )

            superseding_digest = superseding_invocation_admission_digest_from_state(
                current_session,
                checkpoint,
                session_id=ticket.session_id,
                session_instance_id=ticket.session_instance_id,
                expected_run_epoch=consumption.admission_expected_run_epoch,
                command_sha256=consumption.admission_command_digest,
            )
            if superseding_digest is not None:
                from cayu.sessions._session_continuation import ContinuationRetirement

                retired_ticket = record.ticket.model_copy(
                    update={
                        "state": "RETIRED",
                        "revision": record.ticket.revision + 1,
                    }
                )
                updated = record.model_copy(
                    update={
                        "ticket": retired_ticket,
                        "consumption": existing.model_copy(
                            update={
                                "ticket": retired_ticket,
                                "receipt_stage": "excluded",
                                "admission_claimed": False,
                                "admission_claim_id": None,
                            }
                        ),
                        "retirement": ContinuationRetirement(
                            ticket=retired_ticket,
                            control_id="admission-superseded:" + superseding_digest,
                            reason="superseded",
                            retired_at=current_session.updated_at.isoformat(),
                        ),
                    }
                )
                return SessionOperationPublication(
                    checkpoint={} if checkpoint is None else checkpoint,
                    operation_records={key: updated.model_dump(mode="json")},
                )

            if not existing.admission_claimed:
                raise ContinuationConflict(
                    "Unclaimed continuation exclusion requires positive supersession evidence."
                )
            if (
                reconcile_invocation_admission_from_state(
                    current_session,
                    checkpoint,
                    session_id=ticket.session_id,
                    session_instance_id=ticket.session_instance_id,
                    expected_run_epoch=consumption.admission_expected_run_epoch,
                    command_sha256=consumption.admission_command_digest,
                    profile_sha256=consumption.profile_digest,
                )
                is not None
            ):
                raise ContinuationConflict("Committed continuation admission cannot be released.")
            updated = record.model_copy(
                update={
                    "consumption": existing.model_copy(
                        update={
                            "admission_claimed": False,
                            "admission_claim_id": None,
                        }
                    )
                }
            )
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: updated.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            ticket.session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
        record = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if record is None or record.consumption is None:
            raise ContinuationUnavailable("Continuation claim release was not durably readable.")
        return record

    async def _finalize_continuation_admission(
        self,
        consumption: ContinuationConsumption,
        command: AdmitInvocationCommand | None = None,
    ) -> ContinuationRecord:
        """Commit the admitted stage only from the typed lifecycle boundary."""

        from cayu.sessions._session_continuation_scope import require_consumption

        require_consumption(consumption)

        return await self._consume_continuation(
            consumption, admission_boundary=True, admission_command=command
        )

    async def _consume_continuation(
        self,
        consumption: ContinuationConsumption,
        *,
        admission_boundary: bool,
        admission_command: AdmitInvocationCommand | None = None,
    ) -> ContinuationRecord:
        """Apply the store-side CAS for public preparation or typed admission."""

        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationRecord,
            ContinuationUnavailable,
            continuation_admission_digest,
            continuation_operation_key,
            require_latch_identity,
            require_record_writer_generation,
            require_ticket_identity,
        )

        ticket = consumption.ticket
        key = continuation_operation_key(ticket)
        if consumption.receipt_stage == "admitted" and not admission_boundary:
            raise ContinuationConflict(
                "Admitted continuation receipts require the typed lifecycle boundary."
            )

        def transform(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            current: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            if current_session.instance_id != ticket.session_instance_id:
                raise SessionRunFenced(
                    "Continuation consumption belongs to a retired session instance."
                )
            if current is None:
                raise ContinuationConflict("Continuation ticket is unavailable.")
            if admission_boundary:
                from cayu.sessions._invocation_lifecycle import (
                    AdmitInvocationCommand,
                    reconcile_invocation_admission_from_state,
                    replay_invocation_lifecycle_command_from_state,
                )

                if admission_command is not None and (
                    type(admission_command) is not AdmitInvocationCommand
                    or continuation_admission_digest(admission_command)
                    != consumption.admission_command_digest
                    or admission_command.session_id != ticket.session_id
                    or admission_command.expected_session_instance_id != ticket.session_instance_id
                    or replay_invocation_lifecycle_command_from_state(
                        current_session, checkpoint, admission_command
                    )
                    is None
                ):
                    raise ContinuationConflict(
                        "Continuation lacks exact durable admission evidence."
                    )
                if (
                    reconcile_invocation_admission_from_state(
                        current_session,
                        checkpoint,
                        session_id=ticket.session_id,
                        session_instance_id=ticket.session_instance_id,
                        expected_run_epoch=consumption.admission_expected_run_epoch,
                        command_sha256=consumption.admission_command_digest,
                        profile_sha256=consumption.profile_digest,
                    )
                    is None
                ):
                    raise ContinuationConflict(
                        "Continuation lacks exact durable admission evidence."
                    )
            record = ContinuationRecord.model_validate(current)
            if record.consumption is not None:
                existing = record.consumption
                require_ticket_identity(existing.ticket, consumption.ticket)
                require_latch_identity(existing.latch, consumption.latch)
                replay = existing.model_copy(
                    update={"ticket": consumption.ticket, "latch": consumption.latch}
                )
                if replay == consumption or (
                    existing.receipt_stage in {"prepared", "admitted"}
                    and consumption.receipt_stage == "prepared"
                    and replay.model_copy(
                        update={
                            "receipt_stage": "prepared",
                            "admission_claimed": False,
                            "admission_claim_id": None,
                        }
                    )
                    == consumption.model_copy(
                        update={
                            "admission_claimed": False,
                            "admission_claim_id": None,
                        }
                    )
                ):
                    updated = record
                elif (
                    existing.receipt_stage == "prepared"
                    and consumption.receipt_stage == "admitted"
                    and existing.admission_claimed
                    and existing.admission_claim_id == consumption.admission_claim_id
                    and existing.model_copy(
                        update={
                            "ticket": consumption.ticket,
                            "latch": consumption.latch,
                            "receipt_stage": "admitted",
                            "admission_claimed": True,
                        }
                    ).model_dump(mode="json")
                    == consumption.model_dump(mode="json")
                ):
                    consumed_ticket = record.ticket.model_copy(
                        update={"state": "CONSUMED", "revision": record.ticket.revision + 1}
                    )
                    updated_consumption = consumption.model_copy(
                        update={
                            "ticket": consumed_ticket,
                            "latch": record.latch,
                            "admission_claimed": True,
                            "admission_claim_id": consumption.admission_claim_id,
                        }
                    )
                    updated = record.model_copy(
                        update={"ticket": consumed_ticket, "consumption": updated_consumption}
                    )
                else:
                    raise ContinuationConflict("Continuation was consumed differently.")
            else:
                if current_session.run_epoch != consumption.admission_expected_run_epoch:
                    raise ContinuationConflict(
                        "Continuation admission belongs to a stale writer generation."
                    )
                require_ticket_identity(ticket, record.ticket)
                # The new admission command describes the current writer, not
                # the wait's originating authority. A previously latched result
                # must not let an overtaken wait start a fresh consumption.
                # Existing preparations above reconcile their exact handoff.
                require_record_writer_generation(
                    record,
                    current_session.run_epoch,
                    allow_released_next_generation=_continuation_writer_was_released(
                        current_session, checkpoint
                    ),
                )
                if record.latch is None or record.ticket.state != "WAITING":
                    raise ContinuationConflict("Continuation is not ready for consumption.")
                if consumption.receipt_stage != "prepared":
                    raise ContinuationConflict(
                        "Continuation admission requires a prepared responsibility receipt."
                    )
                if record.ticket.revision != ticket.revision:
                    raise ContinuationConflict("Continuation ticket revision is stale.")
                require_ticket_identity(record.latch.ticket, record.ticket)
                require_latch_identity(record.latch, consumption.latch)
                updated = record.model_copy(
                    update={"consumption": consumption.model_copy(update={"latch": record.latch})}
                )
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: updated.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            ticket.session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
        record = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if record is None:
            raise ContinuationUnavailable("Continuation consumption was not durably readable.")
        return record

    async def retire_continuation(self, retirement: ContinuationRetirement) -> ContinuationRecord:
        """Atomically retire a ticket, losing only to an already committed consume."""

        from cayu.collaboration._preparation import prepare_contract
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            ContinuationRecord,
            ContinuationReleasedRetirement,
            ContinuationRetirement,
            ContinuationUnavailable,
            continuation_operation_key,
            continuation_writer_frontier,
            require_record_writer_generation,
            require_ticket_identity,
        )
        from cayu.sessions._session_continuation_scope import require_retirement
        from cayu.vaults.redaction import SecretRedactor

        retirement = prepare_contract(ContinuationRetirement, retirement, redactor=SecretRedactor())
        retirement_authority = require_retirement(retirement)
        ticket = retirement.ticket
        key = continuation_operation_key(ticket)

        def transform(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            current: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            if current_session.instance_id != ticket.session_instance_id:
                raise SessionRunFenced(
                    "Continuation retirement belongs to a retired session instance."
                )
            if current is None:
                raise ContinuationConflict("Continuation ticket is unavailable.")
            record = ContinuationRecord.model_validate(current)
            if record.ticket.state == "SERVICING":
                raise ContinuationConflict("Temporary service must settle before retirement.")
            released_evidence = record.released_retirement
            if isinstance(retirement_authority, ContinuationReleasedRetirement):
                from cayu.sessions._session_continuation_store import (
                    require_released_wait_invocation,
                )

                if released_evidence is None:
                    if record.retirement is not None:
                        raise ContinuationConflict(
                            "Retirement has no retained released-execution proof."
                        )
                    released_evidence = require_released_wait_invocation(
                        ticket,
                        checkpoint,
                        permit_operation=retirement_authority.permit_operation,
                        permit_commitment=retirement_authority.permit_commitment,
                        record=record,
                    )
                elif (
                    released_evidence.permit_operation != retirement_authority.permit_operation
                    or released_evidence.permit_commitment != retirement_authority.permit_commitment
                ):
                    raise ContinuationConflict("Retirement replay changed its released execution.")
            if record.retirement is not None:
                existing = record.retirement
                require_ticket_identity(existing.ticket, retirement.ticket)
                replay = existing.model_copy(update={"ticket": retirement.ticket})
                if replay == retirement:
                    updated = record
                else:
                    raise ContinuationConflict("Continuation was retired differently.")
            else:
                if isinstance(retirement_authority, ContinuationReleasedRetirement):
                    if (
                        record.ticket.state not in {"ARMING", "WAITING"}
                        or record.consumption is not None
                        or any(
                            item.state not in {"returned", "excluded"} for item in record.services
                        )
                        or retirement_authority.settled_services
                        != tuple((item.key, item.record_sha256) for item in record.services)
                        or not _continuation_writer_was_released(current_session, checkpoint)
                    ):
                        raise ContinuationConflict(
                            "Released cleanup requires a quiescent wait and settled services."
                        )
                    require_record_writer_generation(
                        record, current_session.run_epoch, allow_released_next_generation=True
                    )
                elif retirement_authority:
                    # A release alone advances by one. A later epoch in this
                    # exact incarnation proves the old wait has been overtaken.
                    # Never settle prepared/claimed receiving work by inference.
                    writer, released = continuation_writer_frontier(record)
                    if (
                        retirement.reason != "superseded"
                        or current_session.run_epoch <= writer + (not released)
                        or record.consumption is not None
                    ):
                        raise ContinuationConflict(
                            "Supersession requires an overtaken, unconsumed continuation."
                        )
                else:
                    require_record_writer_generation(
                        record,
                        current_session.run_epoch,
                        allow_released_next_generation=_continuation_writer_was_released(
                            current_session, checkpoint
                        ),
                    )
                require_ticket_identity(ticket, record.ticket)
                if record.ticket.revision != ticket.revision:
                    raise ContinuationConflict("Continuation ticket revision is stale.")
                if record.ticket.state in {"CONSUMED", "RETIRED"}:
                    raise ContinuationConflict("Continuation is already terminal.")
                retired_ticket = record.ticket.model_copy(
                    update={"state": "RETIRED", "revision": record.ticket.revision + 1}
                )
                updated_retirement = retirement.model_copy(update={"ticket": retired_ticket})
                if record.consumption is not None:
                    if (
                        record.consumption.receipt_stage != "prepared"
                        or record.consumption.admission_claimed
                    ):
                        raise ContinuationConflict(
                            "Continuation admission is in progress and must settle before retirement."
                        )
                    excluded_consumption = record.consumption.model_copy(
                        update={"ticket": retired_ticket, "receipt_stage": "excluded"}
                    )
                    updated = record.model_copy(
                        update={
                            "ticket": retired_ticket,
                            "consumption": excluded_consumption,
                            "retirement": updated_retirement,
                        }
                    )
                else:
                    updated = record.model_copy(
                        update={
                            "ticket": retired_ticket,
                            "retirement": updated_retirement,
                            "released_retirement": released_evidence,
                        }
                    )
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: updated.model_dump(mode="json")},
            )

        await self._publish_continuation_operation(
            ticket.session_id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
        )
        record = await self.load_continuation_ticket(
            ticket.session_id,
            registration_key=ticket.registration_key,
            session_instance_id=ticket.session_instance_id,
        )
        if record is None:
            raise ContinuationUnavailable("Continuation retirement was not durably readable.")
        return record

    async def load_runtime_publication_receipt(
        self,
        session_id: str,
        publication_id: str,
    ) -> RuntimePublicationReceipt | None:
        """Load and verify one store-owned runtime publication receipt."""

        session_id = require_clean_nonblank(session_id, "session_id")
        publication_id = require_clean_nonblank(publication_id, "publication_id")
        storage_key = publication_contracts._runtime_publication_storage_key(publication_id)
        record = await self._load_runtime_publication_receipt_record(
            session_id,
            storage_key,
            publication_id,
        )
        if record is None:
            return None
        return publication_rules._reconstruct_runtime_publication_receipt(
            record,
            storage_key=storage_key,
            session_id=session_id,
            publication_id=publication_id,
        )

    async def publish_runtime_publication(
        self,
        session_id: str,
        *,
        request: RuntimePublicationRequest,
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> RuntimePublicationResult:
        """Atomically publish or exactly replay one logical runtime boundary."""

        prepared = publication_rules._prepare_runtime_publication(
            session_id,
            request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )
        return await self._publish_runtime_publication_atomic(prepared)

    async def _load_runtime_publication_receipt_record(
        self,
        session_id: str,
        storage_key: str,
        publication_id: str,
    ) -> dict[str, Any] | None:
        """Backend receipt/material lookup hook; fail closed for custom stores.

        Implementations must verify the committed transcript slice, appended
        event batch, and exact content of every referenced event in the same
        locked or transactional snapshot used to load the receipt.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable runtime publications."
        )

    async def _publish_runtime_publication_atomic(
        self,
        prepared: publication_rules._PreparedRuntimePublication,
    ) -> RuntimePublicationResult:
        """Backend atomic publication hook; intentionally non-abstract for compatibility."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable runtime publications."
        )

    async def load_model_completion_stage(
        self,
        session_id: str,
        stage_id: str,
    ) -> ModelCompletionStage | None:
        """Load and verify append-only pre-dispatch and terminal completion evidence."""

        session_id, stage_id, preparation_key, terminal_key = (
            model_stage_rules._model_completion_stage_storage_identity(session_id, stage_id)
        )
        preparation_record, terminal_record = await self._load_model_completion_stage_records(
            session_id,
            preparation_key,
            terminal_key,
        )
        return model_stage_rules._reconstruct_model_completion_stage(
            preparation_record,
            terminal_record,
            session_id=session_id,
            stage_id=stage_id,
            preparation_storage_key=preparation_key,
            terminal_storage_key=terminal_key,
        )

    async def load_model_completion_stage_settlement(
        self,
        session_id: str,
        stage_id: str,
    ) -> ModelCompletionStageSettlement | None:
        """Load verified terminal non-success evidence for one exact stage."""

        session_id, stage_id, _, _ = model_stage_rules._model_completion_stage_storage_identity(
            session_id,
            stage_id,
        )
        storage_key = model_stage_rules._model_completion_stage_settlement_storage_key(stage_id)
        record = await self._load_model_completion_stage_settlement_record(
            session_id,
            storage_key,
        )
        if record is None:
            return None
        return model_stage_rules._reconstruct_model_completion_stage_settlement(
            record,
            session_id=session_id,
            stage_id=stage_id,
            storage_key=storage_key,
        )

    async def load_model_completion_stage_abandonment(
        self,
        session_id: str,
        stage_id: str,
    ) -> ModelCompletionStageAbandonment | None:
        """Read positive no-dispatch evidence after the preparation was removed.

        This is a query, not permission to dispatch. Re-preparation compares the
        complete expected predecessor and this receipt in its own transaction.
        The shared operation-record lookup keeps all supported backends under
        their existing read owner; no second route cache is needed after restart.
        """

        session_id, stage_id, _, _ = model_stage_rules._model_completion_stage_storage_identity(
            session_id, stage_id
        )
        storage_key = model_stage_rules._model_completion_stage_abandonment_storage_key(stage_id)
        record = await self._load_model_completion_stage_settlement_record(session_id, storage_key)
        if record is None:
            return None
        return model_stage_rules._reconstruct_model_completion_stage_abandonment(
            record, session_id=session_id, stage_id=stage_id, storage_key=storage_key
        )

    async def load_active_model_completion_stage(
        self,
        session_id: str,
    ) -> ActiveModelCompletionStage | None:
        """Load a verified snapshot of the dispatch currently eligible to publish."""

        session_id = require_clean_nonblank(session_id, "session_id")
        (
            active_record,
            preparation_record,
            terminal_record,
        ) = await self._load_active_model_completion_stage_records(session_id)
        return model_stage_rules._reconstruct_active_model_completion_stage(
            active_record,
            preparation_record,
            terminal_record,
            session_id=session_id,
        )

    async def load_model_completion_stage_dispatch(
        self,
        session_id: str,
        stage_id: str,
    ) -> ModelCompletionStageDispatch | None:
        """Load exact evidence that a prepared stage crossed the dispatch fence."""

        session_id = require_clean_nonblank(session_id, "session_id")
        storage_key = model_stage_rules._model_completion_stage_dispatch_storage_key(stage_id)
        record = await self._load_model_completion_stage_dispatch_record(
            session_id,
            storage_key,
        )
        if record is None:
            return None
        return model_stage_rules._reconstruct_model_completion_stage_dispatch(
            record,
            session_id=session_id,
            stage_id=stage_id,
            storage_key=storage_key,
        )

    async def mark_model_completion_stage_dispatched(
        self,
        session_id: str,
        *,
        stage: ModelCompletionStage,
        consume_child_session_notifications: bool = True,
    ) -> ModelCompletionStageDispatch:
        """Durably cross the last local fence before entering provider-controlled code."""

        if type(stage) is not ModelCompletionStage:
            raise TypeError("stage must be a ModelCompletionStage.")
        if type(consume_child_session_notifications) is not bool:
            raise TypeError("consume_child_session_notifications must be a boolean.")
        copied_stage = stage.model_copy(deep=True)
        session_id = require_clean_nonblank(session_id, "session_id")
        if copied_stage.session_id != session_id:
            raise SessionModelCompletionStageConflict(
                "Model-completion dispatch belongs to a different session."
            )
        try:
            return await self._mark_model_completion_stage_dispatched_atomic(
                session_id,
                stage=copied_stage,
                consume_child_session_notifications=consume_child_session_notifications,
            )
        except Exception:
            reconciled = await self.load_model_completion_stage_dispatch(
                session_id,
                copied_stage.stage_id,
            )
            if reconciled is None:
                raise
            active = await self.load_active_model_completion_stage(session_id)
            if active is None or active.stage != copied_stage:
                raise
            model_stage_rules._validate_model_completion_stage_dispatch(reconciled, copied_stage)
            if (
                consume_child_session_notifications
                and child_session_notification_stage_binding(copied_stage.intent) is not None
            ):
                return await self._mark_model_completion_stage_dispatched_atomic(
                    session_id,
                    stage=copied_stage,
                    consume_child_session_notifications=True,
                )
            return reconciled

    async def prepare_model_completion_stage(
        self,
        session_id: str,
        *,
        request: ModelCompletionStageRequest,
        expected_statuses: set[SessionStatus],
        expected_run_epoch: int,
        expected_transcript_cursor: int,
    ) -> ModelCompletionStageResult:
        """Insert or replay a stage; dispatch only when the result explicitly authorizes it."""

        prepared = model_stage_rules._prepare_model_completion_stage(
            session_id,
            request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
        )
        return await self._prepare_model_completion_stage_atomic(prepared)

    def _supports_model_failover_stage_protocol(self) -> bool:
        """Overriding a mutation owner requires renewed atomic-route attestation."""

        mro = type(self).__mro__
        capability_owner = next(
            index
            for index, owner in enumerate(mro)
            if "model_failover_stage_version" in owner.__dict__
        )
        return (
            type(self.model_failover_stage_version) is int
            and self.model_failover_stage_version == 1
            and all(
                capability_owner
                <= next((index for index, owner in enumerate(mro) if hook in owner.__dict__), -1)
                for hook in (
                    "_prepare_model_completion_stage_atomic",
                    "_mark_model_completion_stage_dispatched_atomic",
                    "_promote_model_completion_stage_atomic",
                    "create_profiled_fork",
                    "_create_fork",
                    "admit_session_invocation",
                    "admit_execution_profile_resume",
                    "transition_status_and_checkpoint",
                )
            )
        )

    def _require_model_failover_stage_protocol(self) -> None:
        """Reject routed admission unless every native mutation owner attests it."""

        if not self._supports_model_failover_stage_protocol():
            raise NotImplementedError("Store does not attest atomic model failover stages.")

    async def _prepare_model_completion_stage_with_failover(
        self,
        session_id: str,
        *,
        request: ModelCompletionStageRequest,
        expected_statuses: set[SessionStatus],
        expected_run_epoch: int,
        expected_transcript_cursor: int,
        admission: ModelFailoverStageAdmission,
    ) -> ModelCompletionStageResult:
        """Private live-authority entrance; raw public intents cannot select a route."""

        self._require_model_failover_stage_protocol()
        prepared = model_stage_rules._prepare_model_completion_stage(
            session_id,
            request,
            expected_statuses=expected_statuses,
            expected_run_epoch=expected_run_epoch,
            expected_transcript_cursor=expected_transcript_cursor,
            failover_admission=admission,
        )
        return await self._prepare_model_completion_stage_atomic(prepared)

    def _supports_model_completion_recovery_fence_protocol(self) -> bool:
        """Require an explicit attestation from the concrete atomic-hook owner."""
        mro = type(self).__mro__
        capability_owner = next(
            index
            for index, owner in enumerate(mro)
            if "model_completion_recovery_fence_version" in owner.__dict__
        )
        mutation_owner = next(
            index
            for index, owner in enumerate(mro)
            if "_complete_model_completion_stage_atomic" in owner.__dict__
        )
        return (
            type(self.model_completion_recovery_fence_version) is int
            and self.model_completion_recovery_fence_version == 1
            and capability_owner <= mutation_owner
        )

    async def complete_recovered_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        publication: RuntimePublicationRequest,
        recovery_fence: ModelCompletionStageRecoveryFence,
    ) -> ModelCompletionStageResult:
        """Insert synthetic auxiliary terminal evidence under an atomic recovery fence.

        Backends overriding the atomic hook must explicitly attest this protocol.
        Ordinary provider completion continues through complete_model_completion_stage.
        """
        if not self._supports_model_completion_recovery_fence_protocol():
            raise NotImplementedError("Store does not attest atomic model recovery fences.")
        if type(recovery_fence) is not ModelCompletionStageRecoveryFence:
            raise TypeError("Recovery completion requires a ModelCompletionStageRecoveryFence.")
        fence = ModelCompletionStageRecoveryFence.model_validate(
            {
                name: getattr(recovery_fence, name)
                for name in ModelCompletionStageRecoveryFence.model_fields
            }
        )
        if (
            fence.expected_active_profile.session_id != session_id
            or fence.expected_active_profile.run_epoch != fence.expected_run_epoch
        ):
            raise ValueError("Recovery fence conflicts with its session or epoch.")
        prepared = model_stage_rules._prepare_model_completion_stage_terminal(
            session_id,
            stage_id=stage_id,
            publication=publication,
            recovery_fence=fence,
        )
        if prepared.publication.kind != "auxiliary-inference":
            raise ValueError("Recovery-generated completion requires auxiliary publication.")
        return await self._complete_model_completion_stage_atomic(prepared)

    async def complete_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        publication: RuntimePublicationRequest,
    ) -> ModelCompletionStageResult:
        """Append or exactly replay immutable terminal provider-attempt material.

        For auxiliary inference, terminal material records the explicit attempt
        outcome and accounting; it does not imply a successful model response.
        """

        prepared = model_stage_rules._prepare_model_completion_stage_terminal(
            session_id,
            stage_id=stage_id,
            publication=publication,
        )
        return await self._complete_model_completion_stage_atomic(prepared)

    async def abandon_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        preparation_digest: str,
        expected_run_epoch: int,
        stage_source_run_epoch: int | None = None,
    ) -> ModelCompletionStageAbandonmentResult:
        """Atomically abandon one exact active stage before provider dispatch.

        An exact replay reconstructs the durable abandonment tombstone. A later
        preparation of the same request receives a new preparation digest, so a
        delayed retry of an older abandonment can never clear it.
        """

        prepared = model_stage_rules._prepare_model_completion_stage_abandonment(
            session_id,
            stage_id=stage_id,
            preparation_digest=preparation_digest,
            expected_run_epoch=expected_run_epoch,
            stage_source_run_epoch=stage_source_run_epoch,
        )
        return await self._abandon_model_completion_stage_atomic(prepared)

    async def promote_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        expected_run_epoch: int,
    ) -> RuntimePublicationResult:
        """Atomically verify and publish a terminal stage without a load/publish race."""

        expected_run_epoch = session_authority_rules._validate_required_runtime_fence(
            expected_run_epoch,
            "expected_run_epoch",
        )
        session_id, stage_id, preparation_key, terminal_key = (
            model_stage_rules._model_completion_stage_storage_identity(session_id, stage_id)
        )
        return await self._promote_model_completion_stage_atomic(
            session_id=session_id,
            stage_id=stage_id,
            preparation_storage_key=preparation_key,
            terminal_storage_key=terminal_key,
            expected_run_epoch=expected_run_epoch,
        )

    async def _load_model_completion_stage_records(
        self,
        session_id: str,
        preparation_storage_key: str,
        terminal_storage_key: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Backend stage lookup hook; intentionally fail closed for custom stores."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def _load_model_completion_stage_settlement_record(
        self,
        session_id: str,
        settlement_storage_key: str,
    ) -> dict[str, Any] | None:
        """Backend settlement lookup hook; fail closed for custom stores."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion settlements."
        )

    async def _load_model_completion_stage_dispatch_record(
        self,
        session_id: str,
        dispatch_storage_key: str,
    ) -> dict[str, Any] | None:
        """Backend dispatch lookup hook; fail closed for custom stores."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion dispatches."
        )

    async def _load_active_model_completion_stage_records(
        self,
        session_id: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
        """Backend snapshot hook for the active marker and its immutable stage."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def _mark_model_completion_stage_dispatched_atomic(
        self,
        session_id: str,
        *,
        stage: ModelCompletionStage,
        consume_child_session_notifications: bool,
    ) -> ModelCompletionStageDispatch:
        """Backend hook that validates active authority and inserts one dispatch receipt."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support atomic model-completion dispatch."
        )

    async def _prepare_model_completion_stage_atomic(
        self,
        prepared: model_stage_rules._PreparedModelCompletionStage,
    ) -> ModelCompletionStageResult:
        """Backend hook for an insert-only, fenced preparation record."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def _complete_model_completion_stage_atomic(
        self,
        prepared: model_stage_rules._PreparedModelCompletionStageTerminal,
    ) -> ModelCompletionStageResult:
        """Backend hook for an insert-only terminal provider-response record."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def _abandon_model_completion_stage_atomic(
        self,
        prepared: model_stage_rules._PreparedModelCompletionStageAbandonment,
    ) -> ModelCompletionStageAbandonmentResult:
        """Backend hook for an exact, fenced pre-dispatch abandonment."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
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
        """Backend hook that verifies the stage and publishes under one store lock."""

        raise NotImplementedError(
            f"{type(self).__name__} does not support durable model-completion stages."
        )

    async def load_session_export_snapshot(
        self,
        session_id: str,
        *,
        limits: SessionExportLimits | None = None,
    ) -> SessionExportSnapshot | None:
        """Read all export-owned components at one bounded per-session snapshot.

        Return None if the session has been deleted. No sequential-read fallback
        is safe for custom stores: implement one lock or database snapshot.
        """
        raise NotImplementedError("This SessionStore does not support consistent exports.")

    @abstractmethod
    async def load_events(self, session_id: str) -> list[Event]:
        """Load all events for a session."""

    async def load_user_input_supersession_events(
        self,
        session_id: str,
        input_id: str,
    ) -> list[Event]:
        """Load exact interruption evidence for one user-input pause.

        Built-in persistent stores override this with their fixed-size
        pending-action lookup indexes. The fallback pages complete durable
        history for correctness without retaining an unbounded result.
        """

        from cayu.approvals.user_input import USER_INPUT_SUPERSESSION_INTENT_KEY

        session_id = require_clean_nonblank(session_id, "session_id")
        input_id = require_clean_nonblank(input_id, "input_id")
        selected: list[Event] = []
        after_sequence = 0
        while True:
            records = await self.query_events(
                EventQuery(
                    session_id=session_id,
                    event_type=EventType.SESSION_INTERRUPTED,
                    after_sequence=after_sequence,
                    limit=5000,
                    order_by=EventOrder.SEQUENCE_ASC,
                )
            )
            if not records:
                break
            for record in records:
                marker = record.event.payload.get(USER_INPUT_SUPERSESSION_INTENT_KEY)
                if type(marker) is not dict or marker.get("input_id") != input_id:
                    continue
                selected.append(copy_event(record.event))
                if len(selected) == 2:
                    return selected
            after_sequence = records[-1].sequence
            if len(records) < 5000:
                break
        return selected

    async def load_tool_round_lifecycle_events(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
    ) -> list[Event]:
        """Load exact started/terminal candidates for bounded tool-call identities.

        Built-in stores override this with their fixed-size pending-action
        indexes. The paginated fallback preserves correctness for custom stores
        without silently truncating lifecycle evidence.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = tool_publication_rules._validate_tool_round_call_ids(
            tool_call_ids,
            "tool_call_ids",
        )
        selected_ids = set(copied_ids)
        lifecycle_events: list[Event] = []
        after_sequence = 0
        while True:
            records = await self.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=tuple(tool_publication_rules._TOOL_ROUND_LIFECYCLE_EVENT_TYPES),
                    after_sequence=after_sequence,
                    limit=5000,
                    order_by=EventOrder.SEQUENCE_ASC,
                )
            )
            if not records:
                break
            for record in records:
                if record.event.payload.get("tool_call_id") not in selected_ids:
                    continue
                lifecycle_events.append(copy_event(record.event))
                if len(lifecycle_events) > publication_contracts._tool_round_lifecycle_event_limit(
                    copied_ids
                ):
                    raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            after_sequence = records[-1].sequence
            if len(records) < 5000:
                break
        return lifecycle_events

    async def load_tool_round_lifecycle_events_for_round(
        self,
        session_id: str,
        tool_call_ids: list[str] | tuple[str, ...],
        *,
        tool_round_identity: ToolRoundIdentity,
    ) -> list[Event]:
        """Load bounded lifecycle evidence for one exact logical tool round.

        Built-in stores provide indexed implementations. The default
        implementation pages over durable lifecycle events so reused provider
        call ids cannot hide current-round evidence behind the publication
        binding limit.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        copied_ids = tool_publication_rules._validate_tool_round_call_ids(
            tool_call_ids,
            "tool_call_ids",
        )
        tool_round_identity = copy_tool_round_identity(tool_round_identity)
        selected_ids = set(copied_ids)
        lifecycle_events: list[Event] = []
        after_sequence = 0
        while True:
            records = await self.query_events(
                EventQuery(
                    session_id=session_id,
                    event_types=tuple(tool_publication_rules._TOOL_ROUND_LIFECYCLE_EVENT_TYPES),
                    after_sequence=after_sequence,
                    limit=5000,
                    order_by=EventOrder.SEQUENCE_ASC,
                )
            )
            if not records:
                break
            for record in records:
                event = record.event
                if (
                    event.payload.get("tool_call_id") not in selected_ids
                    or not tool_publication_rules._tool_round_event_matches_scope_or_is_ambiguous(
                        event,
                        tool_round_identity,
                    )
                ):
                    continue
                lifecycle_events.append(copy_event(event))
                if len(lifecycle_events) > publication_contracts._tool_round_lifecycle_event_limit(
                    copied_ids
                ):
                    raise ValueError("Tool-round lifecycle evidence exceeds the publication limit.")
            after_sequence = records[-1].sequence
            if len(records) < 5000:
                break
        return lifecycle_events

    @abstractmethod
    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        """Query stored events with durable sequence cursors."""

    async def read_usage_accounting(
        self, query: EventQuery, *, by_session: bool = False, by_identity: bool = False
    ) -> UsageAccountingSnapshot:
        """Reduce exact usage in one stable store snapshot with 256-row pages.

        The result retains totals, identity lists and optional per-session rows,
        never historical events. ``after_sequence`` is a single shared watermark
        across every usage-bearing event type. Unsupported stores fail closed.
        """
        raise NotImplementedError("This SessionStore does not support bounded usage accounting.")

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
        """Price a stable snapshot with bounded working memory plus requested output.

        Built-ins group hosted evidence and completions by session and attempt;
        totals-only reads retain no historical line items. Unsupported stores fail closed.
        ``fork_lineage_owner`` narrows the read to sessions whose fork lineage
        that controller owns, judged as ``resolve_fork_lineage`` does.
        ``without_reservation_for`` counts only model attempts that have no
        accepted ``budget.reserved`` event for that budget limit id, and takes
        no ``additional_events``. A filtered read ignores ``previous`` and
        returns no refresh cursor.
        """
        raise NotImplementedError("This SessionStore does not support bounded cost accounting.")

    async def event_exists(self, query: EventQuery) -> bool:
        """Test matching durable evidence without hydrating event payloads."""
        raise NotImplementedError("This SessionStore does not support event existence queries.")

    async def read_store_identity(self) -> str:
        """This store's identity: 32 lowercase hex digits, chosen once when its storage was created.

        A durable store keeps it in its database, so a restart reads it again and a
        different database has another. A database restored from a backup keeps the
        identity it had when the backup was taken.
        """
        raise NotImplementedError("This SessionStore has no store identity.")

    async def fold_event_snapshot(
        self,
        query: EventQuery,
        consume: Callable[[list[EventRecord]], None],
        *,
        max_bytes: int,
        max_pages: int | None = None,
    ) -> int:
        """Consume byte-bounded pages from one stable retained-event snapshot.

        Optional native capability. The synchronous consumer must only mutate
        request-local state; it may run on a worker thread. Publish its state
        only after this method succeeds. Exceptions/cancellation invalidate the
        accumulator, and implementations must quiesce owned work before return.
        Return the snapshot's unfiltered maximum sequence (zero when empty),
        not a committed-prefix watermark. Each call takes a fresh snapshot.
        ``query.limit`` bounds each page; ``max_pages`` optionally bounds work.
        """
        raise NotImplementedError("This SessionStore has no exact event snapshot fold.")

    async def query_events_bounded(
        self,
        query: EventQuery,
        *,
        max_bytes: int,
    ) -> list[EventRecord]:
        """Query events only when the backend can enforce bytes before hydration."""

        raise NotImplementedError(
            f"{type(self).__name__} does not implement byte-bounded event queries."
        )

    async def load_bounded(self, session_id: str, *, max_bytes: int) -> Session | None:
        """Load a full session only after bounding transport bytes in one snapshot."""
        raise NotImplementedError("This SessionStore does not support byte-bounded session loads.")

    async def export_terminal_session_evidence(
        self,
        session_id: str,
        *,
        spool: EvidenceSpool,
    ) -> None:
        """Export a stable terminal snapshot to bounded indexed backing.

        Optional capability. Unsupported stores fail before reading payloads;
        callers must explicitly choose the separate bounded eager fallback.
        """
        raise NotImplementedError(
            "This SessionStore does not support incremental terminal evidence export."
        )

    async def load_terminal_session_evidence(
        self,
        session_id: str,
        *,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        """Read one exact bounded snapshot through the current terminal event.

        Implementations must resolve terminal status, publication authority,
        the event cutoff, and the complete attributed transcript in one stable
        read. Unsupported, incomplete, contradictory, or oversized evidence
        fails with ``TerminalSessionEvidenceError``; it is never truncated.
        """

        del session_id, limits
        raise NotImplementedError(
            "This SessionStore does not support terminal-session evidence reads."
        )

    async def load_runner_owned_interrupted_evidence(
        self,
        session_id: str,
        *,
        observed_events: tuple[RunnerObservedEventIdentity, ...] | None = None,
        expected_parent_session_id: str | None = None,
        limits: TerminalSessionEvidenceLimits | None = None,
    ) -> TerminalSessionEvidence:
        """Read one exact interrupted snapshot from a runner-owned execution tree.

        A root read must compare the complete durable event-log identity with
        ``observed_events``. A descendant read instead proves its direct parent
        through ``expected_parent_session_id``; the caller must have established
        that parent as part of the same freshly executed tree. Exactly one proof
        is required. Implementations must apply record/count/byte limits before
        hydrating evidence, all inside one stable store snapshot. This optional
        operation does not broaden ordinary historical terminal-evidence reads.
        """

        del session_id, observed_events, expected_parent_session_id, limits
        raise NotImplementedError(
            "This SessionStore does not support runner-owned interrupted evidence reads."
        )

    async def query_latest_interaction_events(
        self,
        session_id: str,
        *,
        before_sequence: int | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        """Return each interaction's latest lifecycle event in descending order.

        Built-in stores implement this as a native keyset query. The explicit
        capability error prevents endpoints from silently degrading to an
        unbounded event-log scan for out-of-tree stores.
        """

        raise NotImplementedError(
            "This SessionStore does not support native interaction pagination."
        )

    @abstractmethod
    async def summarize_events(self, session_id: str) -> EventSummary:
        """Summarize stored events for one session without loading every event."""

    @abstractmethod
    async def summarize_outcome(self, session_id: str) -> SessionOutcome:
        """Derive the current session outcome from durable events."""

    async def inspect_summary(
        self,
        session_id: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> SessionInspectionSummary:
        """Return one exact, content-free diagnostic summary.

        The default implementation deliberately composes the public paginated
        query contracts so out-of-tree stores gain the same semantics without a
        backend-specific schema dependency. Implementations may replace it with
        an equivalent bounded native aggregate.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        identity = await self.inspect_identity(session_id)

        event_count = 0
        event_total_bytes = 0
        event_largest_bytes = 0
        usage_accumulator = session_inspection_rules._SessionInspectionUsageAccumulator()
        budget_events: list[Event] = []
        budget_model_attempt_terminal_events: list[Event] = []
        retained_event_bytes = 0
        queued_message_count = 0
        delivered_message_count = 0
        rejected_message_count = 0
        operation_event_count = 0
        after_sequence = 0
        while True:
            records = await self.query_events(
                EventQuery(
                    session_id=session_id,
                    after_sequence=after_sequence,
                    limit=session_inspection_rules._SESSION_INSPECTION_PAGE_SIZE,
                    order_by=EventOrder.SEQUENCE_ASC,
                )
            )
            if not records:
                break
            event_count += len(records)
            if event_count > session_inspection_rules._SESSION_INSPECTION_MAX_RECORDS:
                raise ValueError(
                    "Session inspection exceeds the "
                    f"{session_inspection_rules._SESSION_INSPECTION_MAX_RECORDS}-event safety limit."
                )
            for record in records:
                payload_bytes = compact_json_utf8_size(record.event.payload)
                event_total_bytes += payload_bytes
                event_largest_bytes = max(event_largest_bytes, payload_bytes)
                event = record.event
                if is_budget_model_attempt_terminal_event(event):
                    model_attempt_event = project_budget_model_attempt_inspection_event(event)
                    retained_event_bytes = (
                        session_inspection_rules._retain_session_inspection_event(
                            retained_event_bytes,
                            model_attempt_event,
                        )
                    )
                    budget_model_attempt_terminal_events.append(model_attempt_event)
                if event.type in {
                    EventType.MODEL_COMPLETED,
                    EventType.MODEL_HOSTED_TOOL_CALL,
                    EventType.TOOL_CALL_STARTED,
                }:
                    usage_event, usage_metrics = project_aggregate_usage_inspection_event(event)
                    retained_event_bytes = (
                        session_inspection_rules._retain_session_inspection_event(
                            retained_event_bytes,
                            usage_event,
                        )
                    )
                    usage_accumulator.add(event.type, usage_metrics)
                if is_budget_inspection_event(event):
                    budget_event = project_budget_inspection_event(event)
                    retained_event_bytes = (
                        session_inspection_rules._retain_session_inspection_event(
                            retained_event_bytes,
                            budget_event,
                        )
                    )
                    budget_events.append(budget_event)
                queued_message_count += event.type == EventType.SESSION_MESSAGE_QUEUED
                delivered_message_count += event.type == EventType.SESSION_MESSAGE_DELIVERED
                rejected_message_count += event.type in {
                    EventType.SESSION_MESSAGE_WITHDRAWN,
                    EventType.SESSION_MESSAGE_QUARANTINED,
                    EventType.SESSION_MESSAGE_STALE,
                    EventType.SESSION_MESSAGE_EXPIRED,
                }
                operation_event_count += event.type == EventType.SERVER_MUTATION_ACCEPTED
            after_sequence = records[-1].sequence
            if len(records) < session_inspection_rules._SESSION_INSPECTION_PAGE_SIZE:
                break

        transcript_count = 0
        transcript_total_bytes = 0
        transcript_largest_bytes = 0
        offset = 0
        while True:
            page = await self.query_transcript(
                TranscriptQuery(
                    session_id=session_id,
                    offset=offset,
                    limit=session_inspection_rules._SESSION_INSPECTION_PAGE_SIZE,
                )
            )
            if (
                offset == 0
                and page.total_records > session_inspection_rules._SESSION_INSPECTION_MAX_RECORDS
            ):
                raise ValueError(
                    "Session inspection exceeds the "
                    f"{session_inspection_rules._SESSION_INSPECTION_MAX_RECORDS}-message safety limit."
                )
            for record in page.records:
                message_bytes = compact_json_utf8_size(record.message.model_dump(mode="json"))
                transcript_count += 1
                transcript_total_bytes += message_bytes
                transcript_largest_bytes = max(transcript_largest_bytes, message_bytes)
            offset += session_inspection_rules._SESSION_INSPECTION_PAGE_SIZE
            if offset >= page.total_records:
                break

        pending = await self.query_pending_actions(
            PendingActionQuery(session_id=session_id, limit=200),
            checkpoint_root_guard=checkpoint_root_guard,
        )
        usage, model_calls_with_usage = usage_accumulator.result(session_id)
        budget = session_budget_inspection(
            budget_events,
            model_attempt_terminal_events=budget_model_attempt_terminal_events,
        )
        return SessionInspectionSummary(
            session=identity,
            transcript=SerializedRecordSummary(
                record_count=transcript_count,
                total_bytes=transcript_total_bytes,
                largest_record_bytes=transcript_largest_bytes,
            ),
            events=SerializedRecordSummary(
                record_count=event_count,
                total_bytes=event_total_bytes,
                largest_record_bytes=event_largest_bytes,
            ),
            usage=usage,
            model_calls=usage.model_steps,
            model_calls_with_usage=model_calls_with_usage,
            tool_calls=usage.tool_calls,
            pending_action_count=len(pending.actions),
            pending_action_kinds=tuple(
                sorted({action.kind for action in pending.actions}, key=str)
            ),
            pending_action_issue_count=len(pending.issues),
            queued_message_count=queued_message_count,
            delivered_message_count=delivered_message_count,
            outstanding_message_count=max(
                queued_message_count - delivered_message_count - rejected_message_count,
                0,
            ),
            operation_event_count=operation_event_count,
            terminal_failure_state=(
                "failed"
                if identity.status is SessionStatus.FAILED
                else "interrupted"
                if identity.status is SessionStatus.INTERRUPTED
                else "none"
            ),
            budget=budget,
        )

    @abstractmethod
    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        """List sessions (filtered/sorted/paginated) with a keyset cursor and total count."""

    async def query_session_topology(
        self,
        query: SessionTopologyQuery,
    ) -> SessionTopologyStoreResult:
        """Load bounded ancestry and batched direct-child branches.

        This optional store-native read prevents the control plane from issuing
        one list query per rendered node. Custom stores opt in explicitly through
        ``supports_session_topology``.
        """

        raise NotImplementedError(
            "This SessionStore does not support bounded session topology queries."
        )

    async def query_session_lineage(
        self,
        query: SessionLineageQuery,
    ) -> SessionLineageResult:
        """Load one minimal, byte-bounded direct-child lineage page.

        This optional store-native read is the durable admission boundary for
        historical trajectory capture. Implementations must project only the
        bounded structural fields represented by ``SessionLineageNode`` and must
        not hydrate session display fields or event payloads.
        """

        raise NotImplementedError(
            "This SessionStore does not support bounded session lineage queries."
        )

    async def query_child_session_lifecycle(
        self,
        query: ChildSessionLifecycleQuery,
    ) -> ChildSessionLifecyclePage:
        """Read bounded canonical direct-child state and notification freshness.

        Implementations advertising ``child_session_notification_version == 1``
        must resolve session state, its exact lifecycle occurrence, and any
        dispatch-fenced terminal consumption in one stable snapshot.
        """

        del query
        raise NotImplementedError(
            "This SessionStore does not support child-session lifecycle projections."
        )

    async def aggregate_environment_activity(
        self, query: EnvironmentActivityQuery
    ) -> EnvironmentActivitySnapshot:
        """Read exact retained-session counts/activity for a bounded set of environments.

        Optional: stores advertise supports_environment_activity before implementing
        a single snapshot with one ordered row for every requested name, including
        zero rows. No eval/fork/status exclusions or history reads are implied.
        """
        raise NotImplementedError("Environment activity snapshots are unsupported.")

    async def aggregate_operational_snapshot(
        self,
        filters: SessionAggregateFilter | None = None,
    ) -> SessionOperationalSnapshot:
        """Count current session states in one store-local read snapshot.

        Default raises ``NotImplementedError`` so adding this control-plane read
        model does not make existing out-of-tree stores uninstantiable.
        """
        raise NotImplementedError(
            "This SessionStore does not support operational aggregate snapshots."
        )

    async def aggregate_usage(self, query: UsageRollupQuery) -> UsageRollupStoreResult:
        """Aggregate bounded event-time activity and usage without loading histories.

        Default raises ``NotImplementedError`` so adding this control-plane read
        model does not make existing out-of-tree stores uninstantiable.
        """
        raise NotImplementedError("This SessionStore does not support usage aggregates.")

    async def query_supervision_events(
        self,
        query: SupervisionEventQuery,
        *,
        include_content: bool = False,
    ) -> SupervisionEventPage:
        """Return one page of settled human-supervision touches across sessions.

        Stores opt in with ``supports_supervision_events``; the projection then
        reads only this store's cross-session and byte-bounded event queries,
        under any active session access bounds. ``include_content`` returns
        runtime-published content and is reserved for trusted in-process callers.
        """
        if not self.supports_supervision_events:
            raise NotImplementedError("This SessionStore does not support supervision events.")
        if type(include_content) is not bool:
            raise TypeError("include_content must be a bool.")
        from cayu.sessions._supervision_projection import (
            query_supervision_events as project_supervision_events,
        )

        return await project_supervision_events(self, query, include_content=include_content)

    @abstractmethod
    async def list_sessions_with_pending_interruption_cascade(
        self,
        query: SessionQuery | None = None,
    ) -> SessionListResult:
        """List only sessions carrying a pending interruption cascade marker.

        Implementations should filter at the storage layer. This is the
        restart-discovery path, so scanning all historical sessions and loading
        their checkpoints individually is not an acceptable implementation.
        """

    async def list_queued_dispatch_terminal_receipts(
        self,
        query: QueuedDispatchTerminalReceiptQuery | None = None,
    ) -> list[QueuedDispatchTerminalReceipt]:
        """List bounded live queue/session handoffs for restart reconciliation."""

        raise NotImplementedError(
            "This SessionStore does not support queued dispatch receipt discovery."
        )

    @abstractmethod
    async def query_pending_actions(
        self,
        query: PendingActionQuery | None = None,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> PendingActionListResult:
        """List current checkpoint-backed actions without scanning session histories.

        Implementations must bound candidate and event reads, apply filtering at
        the storage layer where possible, and avoid per-session history queries.
        A supplied root guard runs on bounded opaque evidence in the same read
        snapshot before any pending state is interpreted.
        """

    async def validate_session_closure_admission(self, session_id: str) -> None:
        """Check every native active-operation/deletion guard without mutation.

        Read the guards in one store snapshot. This preflight is not an
        ownership claim and does not replace final atomic deletion checks.
        """
        raise NotImplementedError("This SessionStore does not support closure admission.")

    async def load_session_closure_records(
        self, session_id: str, *, max_records: int, max_bytes: int
    ) -> dict[str, Any]:
        """Return a bounded native closure snapshot with per-class counts/bytes.

        Read all included classes under one store snapshot. Exceeding a bound
        must return no partial records. Unsupported stores must fail explicitly.
        """
        raise NotImplementedError("This SessionStore does not support closure record snapshots.")

    async def delete_session(
        self,
        session_id: str,
        *,
        closure_receipt: dict[str, Any] | None = None,
    ) -> None:
        """Delete a session and cascade to its events, transcript, and checkpoint.

        Raises ``ValueError`` if the session is in-flight (``RUNNING`` or
        ``INTERRUPTING`` — interrupt it first), has incomplete current-run terminal
        evidence, still owns a durable subagent child, or if one of its accepted
        budget reservations has no fully delivered terminal settlement audit event.
        Durable subagent children must be settled and deleted before their parent so
        deletion cannot erase queue-admission authority while child work still exists.
        When ``closure_receipt`` is supplied, reject every remaining child edge
        in the deletion transaction, including non-subagent children. Closure
        callers must detach or delete those edges explicitly before finalization.
        Idempotent: deleting a session that does not exist is a no-op.

        Default raises ``NotImplementedError`` so out-of-tree stores keep working.
        """
        raise NotImplementedError("This SessionStore does not support delete_session.")

    async def apply_retention_policy(
        self,
        policy: SessionRetentionPolicy,
        *,
        protected_session_ids: Collection[str] = (),
        references: Mapping[RetentionProtection, Collection[str]] | None = None,
        progress: RetentionProgressCallback | None = None,
    ) -> RetentionReport:
        """Dry-run or apply a session retention policy; an operator-only action.

        Only sessions whose status the policy selects and whose last update is
        older than ``policy.older_than`` are considered. A store never compacts
        or deletes a session it cannot prove unreferenced: live tasks and
        leases, pending actions and clarifications, dependent context views,
        snapshot pins, eval and knowledge references, undelivered events and
        lineage members that must be kept all protect a session, and a lineage
        is pruned whole or not at all. ``protected_session_ids`` and
        ``references`` (session ids other stores reference, keyed by the
        protection they imply) add references the store cannot see. A dry run reports exactly what an apply
        with the same inputs and store state acts on. An apply changes one
        lineage per write transaction, re-checking it there, and releases the
        store between lineages; ``progress`` is awaited after planning and after
        each lineage. Every apply writes a durable audit record.

        Default raises ``NotImplementedError``; retention is off unless a store
        implements it and a caller applies a policy.
        """
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def inspect_session_retention(
        self,
        session_ids: Collection[str],
        *,
        mode: RetentionMode | None = None,
        include_store_guards: bool = True,
    ) -> dict[str, tuple[RetentionProtection, ...]]:
        """Report the direct retention protections of existing sessions.

        ``mode`` defaults to delete. Lineage and age selection are not applied. Other retention targets use
        this to keep resources that belong to protected sessions.
        """
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def retention_artifact_references(self) -> frozenset[str]:
        """Artifact ids that this store's transcripts, operations or evidence name."""
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def inspect_storage_growth(self) -> StorageGrowthReport:
        """Read-only size, age and retention-history summary of this store.

        Pass the report to :func:`cayu.storage.storage_growth_warning` to decide
        whether an operator should be told about retention.
        """
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def list_retention_audits(
        self,
        *,
        limit: int = 20,
        item_id: str | None = None,
        store_kind: str | None = None,
    ) -> tuple[RetentionAuditRecord, ...]:
        """List retention audit records, newest first, optionally filtered."""
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def load_retention_audit(self, audit_id: str) -> RetentionAuditRecord | None:
        """Load one retention audit record with every entry it wrote."""
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def begin_retention_audit(
        self,
        *,
        store_kind: str,
        mode: RetentionMode,
        policy: Mapping[str, Any],
        started_at: datetime,
    ) -> str:
        """Start a durable audit record for another store's retention apply."""
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def record_retention_entry(self, audit_id: str, entry: RetentionAuditEntry) -> None:
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def complete_retention_audit(
        self, audit_id: str, *, completed_at: datetime, summary: Mapping[str, Any]
    ) -> None:
        raise NotImplementedError("This SessionStore does not support storage retention.")

    async def detach_session_children(
        self,
        parent_session_id: str,
        child_session_ids: tuple[str, ...],
        *,
        closure_receipt: dict[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        """Atomically detach children and retain auditable lineage tombstones.

        Replay binds the root/plan, original parent and exact unordered child
        set. A conflicting replay or invalid child must mutate no batch member.
        """
        raise NotImplementedError("This SessionStore does not support child detachment.")

    async def load_session_closure_tombstones(
        self, root_session_id: str, plan_id: str
    ) -> tuple[dict[str, Any], ...]:
        """Load durable lineage tombstones for one closure operation."""
        raise NotImplementedError("This SessionStore does not support closure tombstones.")

    async def load_session_closure_receipt(
        self, session_id: str, plan_id: str
    ) -> dict[str, Any] | None:
        """Load an exact durable closure receipt, if this store supports them."""
        raise NotImplementedError("This SessionStore does not support session closure receipts.")

    async def load_session_closure_progress(
        self, session_id: str, plan_id: str
    ) -> dict[str, Any] | None:
        raise NotImplementedError("This SessionStore does not support closure progress.")

    async def save_session_closure_progress(self, progress: dict[str, Any]) -> None:
        raise NotImplementedError("This SessionStore does not support closure progress.")

    async def claim_session_closure_progress(
        self, progress: dict[str, Any], *, finished_only: bool = False
    ) -> None:
        """Atomically validate and reserve a recursive plan's current lineage.

        When finished_only=True, atomically require the root to be COMPLETED or
        FAILED before publishing ownership. Advertise support only if this check
        shares the claim transaction; refusals must leave no claim behind.

        Verify the complete recursive edge set, not just the listed edges,
        including plans with no descendants. Reject unexplained additions before
        publishing ownership. Creation must serialize with this admission and
        reject destinations retired by retained closure claims or receipts.
        Validate every target's native inventory against progress max_records
        and max_bytes in that transaction, before publishing any claim. Replay
        must preserve these admitted limits as part of the exact plan.
        Retain ownership across failures and restart. Detachment and competing
        deletion must reject claimed sessions; only exact closure deletion may
        consume them. No lease expiry may release a still-live cleanup owner.
        """
        raise NotImplementedError("This SessionStore does not support closure lineage claims.")

    async def update_labels(self, session_id: str, labels: dict[str, str]) -> Session:
        """Replace a session's labels (full replacement, not a merge) and return it.

        Annotation writes update ``updated_at`` but do not refresh the runtime
        recovery signal in ``last_activity_at``.

        Raises ``KeyError`` if the session does not exist. Default raises
        ``NotImplementedError`` so out-of-tree stores keep working.
        """
        raise NotImplementedError("This SessionStore does not support update_labels.")

    async def set_summary(
        self,
        session_id: str,
        summary: str,
        *,
        tone: SummaryTone = "neutral",
    ) -> Session:
        """Atomically replace summary and tone, retaining all other metadata.

        Uses the same run/closure ownership fences as annotation writes. Call
        from agent code when its outcome is known; this does not end the run.
        """
        raise NotImplementedError("This SessionStore does not support set_summary.")

    async def update_metadata(self, session_id: str, metadata: dict[str, Any]) -> Session:
        """Replace a session's user metadata and return the updated session.

        This is a full replacement of user-authored metadata, not a merge.
        Runtime-owned entries are retained atomically and cannot be supplied by
        callers. Annotation writes update ``updated_at`` but do not refresh the
        runtime recovery signal in ``last_activity_at``.

        Raises ``KeyError`` if the session does not exist. Default raises
        ``NotImplementedError`` so out-of-tree stores keep working.
        """
        raise NotImplementedError("This SessionStore does not support update_metadata.")

    # True when replace_execution_deadline also accepts a RUNNING session at the
    # expected epoch from the caller that holds that epoch's run fence (the run
    # a settle was just admitted as). Settles then store their new deadline only
    # after the decision is admitted; otherwise they store it before admission.
    # A subclass that overrides replace_execution_deadline must keep this true
    # only if its override accepts that in-run replacement.
    replaces_execution_deadline_in_admitted_run: ClassVar[bool] = False

    async def replace_execution_deadline(
        self,
        session_id: str,
        deadline: ExecutionDeadline,
        *,
        expected_session_instance_id: str | None,
        expected_run_epoch: int,
        renewal: ExecutionDeadlineRenewal | None = None,
    ) -> Session:
        """Store a new execution deadline (and its origin) for a session between runs.

        The replacement commits only while the session still has
        ``expected_session_instance_id`` and ``expected_run_epoch`` (otherwise
        ``SessionRunFenced``) and no run is pending, running or interrupting
        (otherwise ``SessionStatusConflict``). A store that sets
        ``replaces_execution_deadline_in_admitted_run`` also accepts a RUNNING
        session at that epoch from the caller holding its run fence. An
        unbounded ``deadline`` removes the stored one. ``renewal`` records the pause a renewed
        ``run_default`` deadline was computed for; any other replacement clears
        it. The runtime calls this when a resume or a
        settled approval or question carries ``execution_deadline``, and when a
        settled pause renews an inherited task-handler deadline, after it has
        admitted the new deadline.

        Raises ``KeyError`` if the session does not exist. Default raises
        ``NotImplementedError`` before any mutation when unsupported. The runtime
        then keeps the stored deadline for automatic renewals; an explicitly
        requested replacement still raises.
        """
        raise NotImplementedError("This SessionStore does not support replace_execution_deadline.")

    @abstractmethod
    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        """Append provider-neutral transcript messages to a session.

        Stored state must not alias caller-passed messages: implementations
        must detach nested JSON payloads (serialize, or use
        `cayu.messages.detach_message`) so a producer mutating a message
        after append cannot rewrite history.
        """

    @abstractmethod
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
        """Atomically materialize deferred input as the ordered initial transcript.

        The replacement must contain ``expected_messages`` after any leading
        bootstrap/system messages and before ``runtime_suffix_count`` runtime-owned
        messages. The bootstrap prefix is stored with null interaction attribution;
        the admitted source and runtime suffix are attributed to ``interaction_id``.
        When deferred input retains a runtime-authenticated complete initial
        projection, the replacement must equal that projection exactly. No partial
        or externally visible pre-finalization transcript is permitted.
        When supplied, ``checkpoint_transform`` runs inside the same write
        transaction before the initial-transcript authority marker is read.
        """

    @abstractmethod
    async def load_deferred_interaction_input(
        self,
        session_id: str,
    ) -> DeferredInteractionInput | None:
        """Load the private admitted source batch, if one still exists."""

    @abstractmethod
    async def materialize_deferred_interaction_input(
        self,
        session_id: str,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> bool:
        """Materialize deferred source input without a bootstrap prefix.

        Returns ``True`` when a pending admission was materialized and ``False``
        when no matching deferred input exists.  Recovery and setup-failure paths
        use this before making the owning interaction or session terminal.
        """

    @abstractmethod
    async def append_transcript_messages_and_transform_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint_transform: CheckpointTransform,
        *,
        interaction_id: InteractionAttribution = INHERIT_INTERACTION,
    ) -> None:
        """Append transcript messages and atomically transform the current checkpoint.

        The transform must be synchronous and thread-safe; a store may execute
        it on a worker thread while holding its transactional write boundary.
        """

    @abstractmethod
    async def load_transcript(self, session_id: str) -> list[Message]:
        """Load provider-neutral transcript messages for a session.

        Returned messages must not alias stored state: callers own the result
        and may mutate nested payloads without corrupting the transcript.
        """

    @abstractmethod
    async def load_transcript_snapshot(self, session_id: str) -> TranscriptSnapshot:
        """Load retained records and the permanent cursor in one read snapshot.

        Unlike ``len(load_transcript(...))``, ``cursor`` does not move backwards
        when a backend applies physical transcript retention.
        """

    async def load_transcript_cursor(self, session_id: str) -> int:
        """Load the permanent append cursor without requiring retained rows."""

        return (await self.load_transcript_snapshot(session_id)).cursor

    async def load_transcript_window(
        self,
        session_id: str,
        *,
        start_index: int,
        limit: int,
    ) -> TranscriptSnapshot:
        """Load a bounded absolute-index window and its permanent append cursor.

        ``start_index`` is inclusive. If its message was removed, return the
        first surviving records at or after it; at or beyond the append cursor,
        return no records. The returned cursor is the full logical extent, not
        the end of this window. Continue from the last returned index plus one.
        An empty window does not prove that earlier payloads never existed.

        Stores may override this method to avoid loading the complete retained
        transcript. The default keeps custom stores correct through their atomic
        ``load_transcript_snapshot`` implementation.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        if type(start_index) is not int:
            raise TypeError("start_index must be an integer.")
        if not 0 <= start_index <= MAX_DURABLE_JSON_INTEGER:
            raise ValueError("start_index exceeds the durable integer limit.")
        if type(limit) is not int:
            raise TypeError("limit must be an integer.")
        if not 1 <= limit <= 5000:
            raise ValueError("limit must be between 1 and 5000.")
        snapshot = await self.load_transcript_snapshot(session_id)
        start = bisect_left(
            [record.index for record in snapshot.records],
            start_index,
        )
        return TranscriptSnapshot(
            records=[
                record.model_copy(deep=True) for record in snapshot.records[start : start + limit]
            ],
            cursor=snapshot.cursor,
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
        """Load an absolute-index window whose loaded messages fit byte bounds.

        Stores measure each row's stored size before loading it and follow
        ``select_bounded_transcript_rows``: a row over ``max_record_bytes`` is
        returned without its message, and the window ends before the first row
        that no longer fits ``max_total_bytes``. Window rows, sizes, messages
        and the append cursor come from one read snapshot. There is no
        fallback, because loading rows to measure them defeats the bound.
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not support byte-bounded transcript windows."
        )

    async def load_latest_transcript_message(
        self,
        session_id: str,
        *,
        role: MessageRole,
    ) -> TranscriptRecord | None:
        """Load one latest role-matched record without requiring full history.

        Stores must override this method with a reverse-indexed read. There is
        deliberately no OFFSET-based fallback because its work grows with the
        transcript and would make callers' bounded-read guarantees dishonest.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(role, MessageRole):
            raise TypeError("role must be a MessageRole.")
        raise NotImplementedError(
            f"{type(self).__name__} does not support reverse-indexed transcript reads."
        )

    async def load_latest_transcript_text(
        self,
        session_id: str,
        *,
        role: MessageRole,
        max_chars: int,
    ) -> tuple[str, bool] | None:
        """Load a bounded text projection without hydrating the source message."""

        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(role, MessageRole):
            raise TypeError("role must be a MessageRole.")
        if type(max_chars) is not int:
            raise TypeError("max_chars must be an integer.")
        if not 1 <= max_chars <= LATEST_TRANSCRIPT_TEXT_MAX_CHARS:
            raise ValueError(f"max_chars must be between 1 and {LATEST_TRANSCRIPT_TEXT_MAX_CHARS}.")
        raise NotImplementedError(
            f"{type(self).__name__} does not support bounded transcript-text reads."
        )

    @abstractmethod
    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
        """Query provider-neutral transcript messages with stable message indexes.

        Offset and total_records refer to retained role/interaction-matched
        rows, not logical extent. Thinking exclusion is applied after pagination;
        advance by the raw page span, not the number of projected records.
        Same result-ownership isolation contract as `load_transcript`.
        """

    async def search_transcript(
        self,
        query: TranscriptSearchQuery,
    ) -> TranscriptSearchResult:
        """Search explicitly selected transcript narratives within hard bounds.

        This optional capability fails closed for custom stores until they opt in.
        Implementations must enforce the supplied session set inside the query and
        must never interpret an empty or omitted set as global access.
        """

        raise NotImplementedError("This SessionStore does not support transcript search.")

    async def create_recall_receipt(self, receipt: RecallReceipt) -> RecallReceipt:
        """Idempotently persist one immutable, session-scoped recall receipt."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    async def load_recall_receipt(
        self,
        session_id: str,
        receipt_id: str,
    ) -> RecallReceipt | None:
        """Load one exact receipt without widening its session scope."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    async def list_recall_receipts(
        self,
        query: RecallEvidenceQuery,
    ) -> RecallReceiptPage:
        """List a deterministic, byte-bounded page of scoped recall receipts."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    async def create_context_exposure(
        self,
        exposure: ContextExposure,
        item_exposures: tuple[RecallItemExposure, ...] = (),
    ) -> ContextExposure:
        """Atomically persist one planned composition and its exact recalled items."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    async def load_context_exposure(
        self,
        session_id: str,
        exposure_id: str,
    ) -> ContextExposure | None:
        """Load one exact provider-attempt lifecycle within its session scope."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    async def load_recall_item_exposures(
        self,
        session_id: str,
        exposure_id: str,
    ) -> tuple[RecallItemExposure, ...]:
        """Load the complete bounded item set atomically linked to one exposure."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    async def list_context_exposures(
        self,
        query: RecallEvidenceQuery,
    ) -> ContextExposurePage:
        """List a deterministic, byte-bounded page of scoped exposure lifecycles."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    async def transition_context_exposure(
        self,
        session_id: str,
        exposure_id: str,
        request: ContextExposureTransitionRequest,
    ) -> ContextExposure:
        """Apply one idempotent lifecycle transition behind an exact CAS fence."""

        raise NotImplementedError("This SessionStore does not support recall evidence.")

    @abstractmethod
    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist a checkpoint for resume/replay."""

    @abstractmethod
    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        """Atomically transform the latest checkpoint for a session.

        The transform runs while the store owns its session/checkpoint write
        boundary and must be synchronous and thread-safe because a store may
        execute it on a worker thread. Returning ``None`` leaves the checkpoint
        unchanged.
        """

    @abstractmethod
    async def transform_checkpoint_with_store_time(
        self,
        session_id: str,
        checkpoint_transform: StoreTimeCheckpointTransform,
    ) -> None:
        """Transform a checkpoint with time sampled by the store transaction.

        The same sampled value must govern any lease comparison made by the
        transform and the resulting checkpoint/activity timestamps.
        """

    @abstractmethod
    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        """Load the latest checkpoint for a session."""

    async def load_execution_snapshot_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        """Operator inspection of snapshot metadata only, without other checkpoint roots.

        Custom stores may override this fallback with a native bounded projection.
        """
        checkpoint = await self.load_checkpoint(session_id)
        return (
            None
            if checkpoint is None
            else {
                key: value
                for key, value in checkpoint.items()
                if key in {"checkpoint_schema_version", "execution_snapshots"}
            }
        )

    @abstractmethod
    async def load_interruption_cascade_marker(
        self,
        session_id: str,
        *,
        checkpoint_root_guard: CheckpointRootFieldGuard | None = None,
    ) -> dict[str, Any] | None:
        """Load a bounded structural projection of the durable cascade marker.

        A supplied root guard runs on bounded opaque evidence in the same read
        snapshot before the marker is interpreted.
        """
