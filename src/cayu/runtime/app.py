from __future__ import annotations

import asyncio
import inspect
import mimetypes
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from hashlib import sha256
from itertools import islice
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from cayu._exception_groups import (
    exception_cause,
    exception_tree_contains,
    set_exception_cause,
)
from cayu._knowledge_publication_owner import KnowledgePublicationLifecycle
from cayu._task_wait import capture_awaitable_outcome, unexpected_child_cancellation_error
from cayu._validation import (
    copy_json_value,
    copy_label_map,
    require_clean_nonblank,
    require_durable_clean_nonblank,
    require_unicode_scalar_text,
)
from cayu.artifacts import (
    DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
    DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST,
    DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES,
    ArtifactScope,
    ArtifactStore,
    FileAttachmentKind,
    file_attachment,
    validate_file_attachment_bytes,
    validate_file_attachment_content_type,
)
from cayu.core.agents import AgentSpec
from cayu.core.events import (
    Event,
    EventType,
    event_durable_sequence,
    event_with_durable_sequence,
    validate_public_custom_event_type,
)
from cayu.core.execution_identity import (
    ExecutionProfileBehaviorIdentity,
    copy_execution_profile_behavior_identity,
)
from cayu.core.messages import (
    FilePart,
    Message,
)
from cayu.core.thinking import ThinkingConfig
from cayu.core.tools import (
    DurableToolRecovery,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.environments import (
    Environment,
    EnvironmentFactory,
    EnvironmentSpec,
    ExecutionRequirements,
    copy_bound_workspace,
    copy_environment,
)
from cayu.mcp.tools import McpToolAdapter
from cayu.providers import (
    ModelProvider,
    OpenAIWebSearch,
    ProviderOperationSnapshot,
    copy_usage_dialect,
)
from cayu.providers.hosted import copy_openai_web_search
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._completion_decision_application_coordinator import (
    CompletionDecisionApplicationCoordinator,
)
from cayu.runtime._completion_result_resolver_coordinator import (
    CompletionResultResolverCoordinator,
)
from cayu.runtime._completion_verifier_coordinator import CompletionVerifierCoordinator
from cayu.runtime._diagnostics import ExceptionDiagnostic, exception_diagnostic
from cayu.runtime._durable_subagent_coordinator import (
    DurableSubagentCoordinator,
    DurableSubagentPreparedRun,
)
from cayu.runtime._durable_subagents import (
    DurableSubagentSubmissionIntent,
    durable_subagent_worker_incompatible,
)
from cayu.runtime._environment_lifecycle import (
    DEFAULT_MAX_ENVIRONMENT_LIFECYCLE_OWNERS,
    EnvironmentLifecycle,
    render_initial_system_prompt,
)
from cayu.runtime._event_projection import (
    PUBLIC_EVENT_ID_PREFIX,
    private_event_linkage_value,
    project_persisted_runtime_event,
    project_runtime_event,
    public_event_envelope_alias,
    public_event_id,
    public_event_linkage_id,
    public_event_linkage_sequence,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._execution_profile_identity_validation import (
    copy_secret_free_execution_profile_behavior_identity,
)
from cayu.runtime._interruption_coordinator import (
    BackgroundInterruptionCoordinator,
)
from cayu.runtime._model_step_executor import (
    ModelCompletionPublicationRequest,
    ModelCompletionPublicationResult,
    ModelCompletionRecoveryContext,
    ModelStepBudgetEvaluationRequest,
    ModelStepBudgetReservationFailureRequest,
    ModelStepExecutor,
    ModelStepLimitEvaluationRequest,
    model_completion_recovery_context_from_stage,
)
from cayu.runtime._recovery_coordinator import (
    ProviderOperationFailureRequest,
    RecoveryAbandonedTurnRequest,
    RecoveryCoordinator,
    RecoveryInterruptionRequest,
    RecoveryLimitStopRequest,
    RecoverySessionRunRequest,
    RecoveryTaskEventRequest,
    RecoveryTerminalEventRequest,
)
from cayu.runtime._run_limit_accounting import RunLimitAccountingContext
from cayu.runtime._run_limits import (
    RunLimitController,
    SessionUsageTracker,
)
from cayu.runtime._session_control import (
    ActiveSessionRun,
    SessionControl,
)
from cayu.runtime._session_engine import (
    SessionEngine,
    _checkpoint_with_pending_session_interrupt,
    _environment_name,
    _interaction_transition_replay_failures,
    _replace_checkpoint_preserving_runtime_state,
    _require_native_structured_output_support,
    _task_event,
    _validate_resume_request,
    _validate_run_request,
)
from cayu.runtime._session_queries import query_all_event_records, query_all_sessions
from cayu.runtime._structured_output_tool_round import _has_structured_output_tool_call
from cayu.runtime._task_store_operation_boundary import (
    TaskStoreOperationOutcome,
    capture_sensitive_validation,
    capture_task_store_operation,
    raise_task_store_operation_failure,
)
from cayu.runtime._terminal_evidence import (
    SESSION_RUN_OPERATION_ID_PAYLOAD_KEY,
    TERMINAL_EVENT_TYPES,
)
from cayu.runtime._tool_round_executor import (
    InterruptedToolRoundRequest,
    ToolRoundExecutor,
    ToolRoundLimitRequest,
)
from cayu.runtime._verified_work_authority import (
    invocation_contains_secret_public_identity,
)
from cayu.runtime.approvals import (
    PendingToolApproval,
    ToolApprovalDecision,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
    copy_tool_approval_recovery_request,
    copy_tool_approval_request,
)
from cayu.runtime.budgets import (
    BudgetLedger,
    BudgetLimit,
    BudgetPolicy,
    BudgetStore,
    InMemoryBudgetLedger,
    SessionBudgetStore,
    copy_budget_policy,
)
from cayu.runtime.completion_result_resolvers import (
    CompletionResultResolutionRequest,
    CompletionResultResolver,
)
from cayu.runtime.completion_verifier_profiles import CompletionVerifierProfilePolicy
from cayu.runtime.completion_verifiers import (
    CompletionVerifierExecutionRequest,
    DeterministicCompletionVerifier,
)
from cayu.runtime.context import (
    ContextPolicy,
    DefaultContextPolicy,
)
from cayu.runtime.context_counting import (
    ContextCountingConfig,
    copy_context_counting_config,
)
from cayu.runtime.costs import (
    CausalBudgetCostSummary,
    PriceBook,
    SessionCostSummary,
    estimate_causal_budget_cost,
    estimate_session_cost,
)
from cayu.runtime.dispatch import (
    Dispatcher,
    DispatchHandle,
    DispatchRequest,
    DispatchStatus,
    InlineDispatcher,
    _copy_queued_dispatch_envelope,
    _new_queued_dispatch_envelope,
    _QueuedDispatchAuthorityRejected,
    _QueuedDispatchEnvelope,
    _QueuedDispatchSettlement,
    _QueuedDispatchSettlementState,
    copy_dispatch_handle,
    copy_dispatch_request,
    redact_dispatch_request,
)
from cayu.runtime.egress_authority_transitions import (
    EgressAuthorityAdoptionHandler,
    _drain_parked_egress_authority_allocations,
)
from cayu.runtime.event_sinks import EventSink
from cayu.runtime.event_watchers import (
    EVENT_WATCHER_QUERY_PAGE_LIMIT,
    EventWatcher,
    EventWatcherContext,
    EventWatcherDeliveryStatus,
    EventWatcherRunResult,
    EventWatcherStore,
    InMemoryEventWatcherStore,
    _clock_or_utc_now,
    event_query_after_cursor,
    event_watcher_error_payload,
    run_event_watcher_handler,
)
from cayu.runtime.execution_profiles import (
    ActiveInvocationExecutionProfile,
    ExecutionProfileIdentity,
    ExecutionProfilePolicy,
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
    active_invocation_execution_profile_matches_session_epoch,
    execution_profile_from_session_metadata,
    unavailable_execution_profile_components,
)
from cayu.runtime.execution_units import ModelStepIdentity
from cayu.runtime.hooks import (
    RuntimeHook,
    RuntimeHookPhase,
)
from cayu.runtime.invocation import (
    InvocationOrigin,
    InvocationOriginTrust,
    SessionInvocationBinding,
    TaskExecutionSource,
    copy_session_invocation_binding,
)
from cayu.runtime.loop_policies import (
    LoopPolicy,
    validate_loop_policies,
)
from cayu.runtime.manifest import AppManifest, describe_app
from cayu.runtime.mcp_manifest_policy import (
    McpManifestPolicy,
    copy_mcp_manifest_policy,
)
from cayu.runtime.provider_operations import (
    ProviderOperationRecoveryResult,
    ProviderOperationResolutionRequest,
    RecoverableProviderOperation,
    RecoverableProviderOperationStart,
    copy_provider_operation_resolution_request,
)
from cayu.runtime.public_authority import (
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    parse_public_authority_alias,
    public_authority_alias_is_reserved,
)
from cayu.runtime.request_footprints import (
    RequestFootprintConfig,
    copy_request_footprint_config,
    targeted_tool_grant_footprint,
)
from cayu.runtime.retry_policy import (
    RetryPolicy,
    copy_retry_policy,
)
from cayu.runtime.sessions import (
    CompactSessionRequest,
    EnqueueSessionMessageRequest,
    EnqueueSessionMessageResult,
    EventOrder,
    EventQuery,
    EventRecord,
    ForkSessionRequest,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    IncompleteSessionsRecoveryPage,
    IncompleteSessionsRecoveryRequest,
    InMemorySessionStore,
    InterruptSessionRequest,
    ModelCompletionManualRecoveryRequest,
    ModelCompletionManualRecoveryResult,
    ModelCompletionStage,
    ModelTarget,
    PendingActionQuery,
    PendingActionResultTooLarge,
    QueuedDispatchTerminalReceipt,
    QueuedDispatchTerminalReceiptQuery,
    ResumeRequest,
    RunRequest,
    Session,
    SessionOrder,
    SessionQuery,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
    _checkpoint_after_queued_dispatch_acknowledgement,
    _current_session_interaction_id,
    _queued_dispatch_session_instance_fingerprint,
    _queued_dispatch_terminal_receipts_from_checkpoint,
    _session_run_operation_from_checkpoint,
    _SessionRunFenceContext,
    copy_fork_session_request,
    copy_incomplete_session_recovery_request,
    copy_incomplete_sessions_recovery_request,
    copy_interrupt_session_request,
    copy_model_completion_manual_recovery_request,
    copy_resume_request,
    system_prompt_messages_sha256,
)
from cayu.runtime.stop_policy import (
    RunLimits,
    StopDecision,
)
from cayu.runtime.structured_output import (
    StructuredOutputSpec,
    StructuredOutputStrategy,
)
from cayu.runtime.targeted_tool_projection import (
    TargetedToolMode,
    TargetedToolProjectionKind,
    copy_targeted_tool_mode,
    resolve_targeted_tool_projection,
)
from cayu.runtime.tasks import (
    Task,
    TaskCreate,
    TaskInvocationSnapshot,
    TaskStatus,
    TaskStore,
    copy_task,
    copy_task_create,
    preflight_contract_bound_task_creation,
    require_contract_bound_task_creation_snapshot,
    task_create_with_runtime_invocation,
)
from cayu.runtime.tool_catalogue import (
    ToolCatalogSnapshot,
    ToolDescriptor,
    ToolDescriptorProvenance,
    build_tool_catalog_snapshot,
    build_tool_descriptor,
    mcp_source_tool_fingerprint,
    validate_application_tool_name,
)
from cayu.runtime.tool_exposure import (
    ALL_REGISTERED_TOOLS_PROFILE_ID,
    AllRegisteredToolsExposurePolicy,
    RegisteredToolCapability,
    ResolvedToolExposure,
    ToolExposurePolicy,
    resolved_tool_exposure_from_authority,
    tool_capability_ceiling_from_session_metadata,
)
from cayu.runtime.tool_grants import (
    TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    TargetedToolGrantInspection,
    TargetedToolGrantRecord,
    targeted_tool_grant_inspection,
)
from cayu.runtime.tool_policy import (
    AllowAllToolPolicy,
    ToolPolicy,
)
from cayu.runtime.tool_result_projection import (
    ToolResultProjectionPolicy,
    copy_tool_result_projection_policy,
)
from cayu.runtime.tool_rounds import (
    ToolRoundRecoveryRequest,
    copy_tool_round_recovery_request,
)
from cayu.runtime.usage import (
    USAGE_BEARING_EVENT_TYPES,
    CausalBudgetUsageSummary,
    SessionUsageSummary,
    causal_budget_usage_summary,
    session_usage_summary,
)
from cayu.runtime.user_input import (
    UserInputRecoveryRequest,
    UserInputResponse,
    copy_user_input_recovery_request,
    copy_user_input_response,
)
from cayu.runtime.work_contracts import (
    CompletionDecision,
    CompletionDecisionApplicationRequest,
    CompletionResultResolverRef,
    CompletionVerifierRef,
    TaskCompletionDecisionRequired,
    WorkCompletionConflict,
    WorkContract,
    WorkContractConflict,
    WorkContractDraft,
    WorkContractRef,
    copy_work_contract,
    copy_work_contract_ref,
    work_contract_from_draft,
)
from cayu.runtime.workspace_observation_recovery import (
    retain_workspace_observation_pending_cancellation_requests,
    workspace_observation_pending_cancellation_requests,
)
from cayu.storage.memory import (
    KnowledgeAccessScope,
    KnowledgeStore,
    copy_knowledge_access_scope,
)
from cayu.vaults import (
    SecretRedactionStream,
    SecretRedactor,
)

RegisteredAgent = runtime_records.RegisteredAgent
RegisteredEnvironment = runtime_records.RegisteredEnvironment

if TYPE_CHECKING:
    from cayu.runtime.fork_groups import (
        ForkGroupGate,
        ForkGroupGateSelection,
        ForkGroupReplacementPlanner,
        ForkGroupReplacementPlannerSelection,
        ForkGroupRequest,
        ForkGroupResult,
    )


DEFAULT_MAX_PARALLEL_TOOL_CALLS = 4


@dataclass(frozen=True, slots=True)
class _ArtifactStoreRegistration:
    store_id: str
    store: ArtifactStore
    fingerprint: str


class _RunFenceOwnedEventStream:
    """Advance and close one delegated stream under its captured run fences."""

    def __init__(self, stream: AsyncGenerator[Event, None]) -> None:
        self._stream = stream
        self._run_fences = _SessionRunFenceContext.current_or_new()

    def __aiter__(self) -> _RunFenceOwnedEventStream:
        return self

    async def __anext__(self) -> Event:
        with self._run_fences.activate():
            return await anext(self._stream)

    async def aclose(self) -> None:
        with self._run_fences.activate():
            await self._stream.aclose()


def _attach_delegated_failure_causes(
    authoritative_failure: BaseException,
    failures: Iterable[BaseException | None],
    *,
    message: str,
) -> None:
    evidence: list[BaseException] = []
    for failure in (*failures, exception_cause(authoritative_failure)):
        if failure is None or failure is authoritative_failure:
            continue
        if any(candidate is failure for candidate in evidence):
            continue
        evidence.append(failure)
    if not evidence:
        return
    set_exception_cause(
        authoritative_failure,
        evidence[0] if len(evidence) == 1 else BaseExceptionGroup(message, evidence),
    )


async def _close_owned_event_stream_resisting_cancellation(
    owned_stream: _RunFenceOwnedEventStream,
    *,
    cancellation: asyncio.CancelledError | None = None,
) -> tuple[tuple[asyncio.CancelledError, ...], BaseException | None]:
    """Finish delegated cleanup despite cancellation of the awaiting task."""

    cleanup_task = asyncio.create_task(
        capture_awaitable_outcome(owned_stream.aclose),
        name="cayu-delegated-stream-cleanup",
    )
    cancellations = [] if cancellation is None else [cancellation]
    while not cleanup_task.done():
        try:
            await asyncio.wait(
                (cleanup_task,),
                return_when=asyncio.ALL_COMPLETED,
            )
        except asyncio.CancelledError as exc:
            # asyncio.wait raises only when this caller is cancelled. A cancelled
            # cleanup task completes the wait and is inspected through result() below.
            cancellations.append(exc)
            if cleanup_task.cancelled():
                break
            continue

    cleanup_outcome = cleanup_task.result()
    cleanup_failure = cleanup_outcome.error
    if isinstance(cleanup_failure, asyncio.CancelledError):
        cleanup_failure = unexpected_child_cancellation_error(
            cleanup_failure,
            operation="Delegated runtime stream cleanup",
        )
    return tuple(cancellations), cleanup_failure


@asynccontextmanager
async def _close_delegated_event_stream(
    stream: AsyncGenerator[Event, None],
) -> AsyncIterator[_RunFenceOwnedEventStream]:
    """Close a delegated stream synchronously without hiding its exit signal."""

    owned_stream = _RunFenceOwnedEventStream(stream)
    authoritative_failure: BaseException | None = None
    try:
        yield owned_stream
    except BaseException as exc:
        authoritative_failure = exc
        raise
    finally:
        current_task = asyncio.current_task()
        workspace_cancellation_requests = (
            0
            if authoritative_failure is None
            else workspace_observation_pending_cancellation_requests(authoritative_failure)
        )
        cancellation_requests_at_cleanup_entry = (
            0 if current_task is None else current_task.cancelling()
        )
        cancellation_pending_at_cleanup_entry = bool(
            current_task is not None and getattr(current_task, "_must_cancel", False)
        )
        cancellation_requests_to_preserve_at_entry = (
            0
            if workspace_cancellation_requests == 0
            else max(
                workspace_cancellation_requests,
                cancellation_requests_at_cleanup_entry,
            )
        )
        pending_at_cleanup_entry: asyncio.CancelledError | None = None
        # ``Task.cancelling()`` includes historical requests that were already
        # delivered. A real checkpoint is the only positive evidence that one
        # of the entry-time requests is still pending for this cleanup.
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as exc:
            pending_at_cleanup_entry = exc
        cancellation_requests_after_checkpoint = (
            0 if current_task is None else current_task.cancelling()
        )
        new_checkpoint_cancellation_requests = max(
            cancellation_requests_after_checkpoint - cancellation_requests_at_cleanup_entry,
            0,
        )
        try:
            cancellations, cleanup_failure = await _close_owned_event_stream_resisting_cancellation(
                owned_stream,
                cancellation=pending_at_cleanup_entry,
            )
            new_cancellations = list(cancellations)
            if new_cancellations:
                cancellation_requests_after_cleanup = (
                    0 if current_task is None else current_task.cancelling()
                )
                new_cancellation_requests = max(
                    cancellation_requests_after_cleanup - cancellation_requests_at_cleanup_entry,
                    0,
                )
                authoritative_cancellation_failure = isinstance(
                    authoritative_failure,
                    asyncio.CancelledError,
                ) or (
                    isinstance(authoritative_failure, BaseExceptionGroup)
                    and exception_tree_contains(authoritative_failure, asyncio.CancelledError)
                )
                pending_entry_cancellation_is_historical = (
                    pending_at_cleanup_entry is not None
                    and authoritative_cancellation_failure
                    and workspace_cancellation_requests > 0
                    and cancellation_requests_at_cleanup_entry <= workspace_cancellation_requests
                    and cancellation_pending_at_cleanup_entry
                    and new_checkpoint_cancellation_requests == 0
                )
                historical_group_cancellation = (
                    pending_entry_cancellation_is_historical and new_cancellation_requests == 0
                )
                if pending_entry_cancellation_is_historical:
                    new_cancellations = [
                        candidate
                        for candidate in new_cancellations
                        if candidate is not pending_at_cleanup_entry
                    ]
                if (
                    historical_group_cancellation
                    and authoritative_failure is not None
                    and not new_cancellations
                ):
                    process_control = exception_tree_contains(
                        authoritative_failure,
                        (GeneratorExit, KeyboardInterrupt, SystemExit),
                    )
                    authoritative_failure.add_note(
                        "Delegated runtime stream cleanup observed the already-grouped "
                        "caller cancellation. The authoritative failure group remains "
                        + (
                            "intact with concurrent process control."
                            if process_control
                            else "intact."
                        )
                    )
                    _attach_delegated_failure_causes(
                        authoritative_failure,
                        (cleanup_failure,),
                        message="Delegated runtime stream cleanup and concurrent control causes",
                    )
            if new_cancellations:
                authoritative_process_control = (
                    authoritative_failure is not None
                    and not isinstance(authoritative_failure, GeneratorExit)
                    and (
                        isinstance(authoritative_failure, (KeyboardInterrupt, SystemExit))
                        or (
                            isinstance(authoritative_failure, BaseExceptionGroup)
                            and exception_tree_contains(
                                authoritative_failure,
                                (GeneratorExit, KeyboardInterrupt, SystemExit),
                            )
                        )
                    )
                )
                cleanup_process_control = cleanup_failure is not None and exception_tree_contains(
                    cleanup_failure,
                    (GeneratorExit, KeyboardInterrupt, SystemExit),
                )
                if (
                    workspace_cancellation_requests > 0
                    or authoritative_cancellation_failure
                    or authoritative_process_control
                    or cleanup_process_control
                ):
                    failures: list[BaseException] = []
                    if authoritative_failure is not None:
                        failures.append(authoritative_failure)
                    failures.extend(new_cancellations)
                    if cleanup_failure is not None and all(
                        cleanup_failure is not failure for failure in failures
                    ):
                        failures.append(cleanup_failure)
                    propagated_failure = BaseExceptionGroup(
                        "Delegated runtime stream cleanup received additional failures.",
                        failures,
                    )
                    current_cancellation_requests = (
                        0 if current_task is None else current_task.cancelling()
                    )
                    cancellation_requests_after_entry = max(
                        current_cancellation_requests - cancellation_requests_at_cleanup_entry,
                        0,
                    )
                    pending_entry_request_count = int(
                        pending_at_cleanup_entry is not None
                        and cancellation_pending_at_cleanup_entry
                    )
                    authenticated_cancellation_requests = (
                        cancellation_requests_to_preserve_at_entry
                        + cancellation_requests_after_entry
                        if workspace_cancellation_requests > 0
                        else cancellation_requests_after_entry + pending_entry_request_count
                    )
                    if authenticated_cancellation_requests:
                        retain_workspace_observation_pending_cancellation_requests(
                            propagated_failure,
                            authenticated_cancellation_requests,
                        )
                    raise propagated_failure from None
                if (
                    authoritative_failure is not None
                    and authoritative_failure is not new_cancellations[0]
                ):
                    new_cancellations[0].add_note(
                        "Delegated runtime stream cleanup was cancelled after an earlier "
                        f"{type(authoritative_failure).__name__}."
                    )
                if cleanup_failure is not None and cleanup_failure is not new_cancellations[0]:
                    new_cancellations[0].add_note(
                        "Delegated runtime stream cleanup also failed: "
                        f"{type(cleanup_failure).__name__}."
                    )
                _attach_delegated_failure_causes(
                    new_cancellations[0],
                    (authoritative_failure, cleanup_failure),
                    message="Delegated runtime stream cancellation evidence",
                )
                raise new_cancellations[0]
            if cleanup_failure is not None:
                if authoritative_failure is None or isinstance(
                    authoritative_failure, GeneratorExit
                ):
                    raise cleanup_failure
                authoritative_failure.add_note(
                    "Delegated runtime stream cleanup failed: "
                    f"{type(cleanup_failure).__name__}. "
                    "The original stream failure remains authoritative."
                )
                if cleanup_failure is not authoritative_failure:
                    _attach_delegated_failure_causes(
                        authoritative_failure,
                        (cleanup_failure,),
                        message="Delegated runtime stream cleanup and prior failure causes",
                    )
        finally:
            if current_task is not None:
                cancellation_requests_after_cleanup = current_task.cancelling()
                cancellation_requests_to_preserve = (
                    0
                    if workspace_cancellation_requests == 0
                    else cancellation_requests_to_preserve_at_entry
                    + max(
                        cancellation_requests_after_cleanup
                        - cancellation_requests_at_cleanup_entry,
                        0,
                    )
                )
                for _request in range(
                    max(
                        cancellation_requests_to_preserve - cancellation_requests_after_cleanup,
                        0,
                    )
                ):
                    current_task.cancel()


class CayuApp:
    """Application runtime for registered agents, providers, and session state."""

    def __init__(
        self,
        *,
        session_store: SessionStore | None = None,
        task_store: TaskStore | None = None,
        knowledge_store: KnowledgeStore | None = None,
        knowledge_access_scope: KnowledgeAccessScope | None = None,
        knowledge_review_namespace: str | None = None,
        knowledge_review_labels: dict[str, str] | None = None,
        dispatcher: Dispatcher | None = None,
        budget_policy: BudgetPolicy | None = None,
        budget_store: BudgetStore | None = None,
        budget_ledger: BudgetLedger | None = None,
        event_watcher_store: EventWatcherStore | None = None,
        retry_policy: RetryPolicy | None = None,
        runtime_hooks: Iterable[RuntimeHook] | None = None,
        loop_policies: Iterable[LoopPolicy] | None = None,
        mcp_manifest_policy: McpManifestPolicy | None = None,
        tool_result_projection_policy: ToolResultProjectionPolicy | None = None,
        execution_profile_policy: ExecutionProfilePolicy | None = None,
        completion_verifier_profile_policy: CompletionVerifierProfilePolicy | None = None,
        egress_authority_adoption_handler: EgressAuthorityAdoptionHandler | None = None,
        context_counting: ContextCountingConfig | None = None,
        request_footprint: RequestFootprintConfig | None = None,
        event_sinks: Iterable[EventSink] | None = None,
        enable_logging: bool = True,
        secret_redactor: SecretRedactor | None = None,
        public_authority_alias_keyring: PublicAuthorityAliasKeyring | None = None,
        max_file_attachment_bytes: int = DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
        max_total_file_attachment_bytes: int = DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES,
        max_file_attachments_per_request: int = DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST,
        tool_timeout_seconds: float | None = None,
        max_parallel_tool_calls: int = DEFAULT_MAX_PARALLEL_TOOL_CALLS,
        max_environment_lifecycle_owners: int = DEFAULT_MAX_ENVIRONMENT_LIFECYCLE_OWNERS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if session_store is not None and not isinstance(session_store, SessionStore):
            raise TypeError("session_store must be a SessionStore.")
        if task_store is not None and not isinstance(task_store, TaskStore):
            raise TypeError("task_store must be a TaskStore.")
        if knowledge_store is not None and not isinstance(knowledge_store, KnowledgeStore):
            raise TypeError("knowledge_store must be a KnowledgeStore.")
        bound_knowledge_access_scope = (
            None if knowledge_store is None else knowledge_store.bound_access_scope()
        )
        if knowledge_store is not None and knowledge_access_scope is None:
            knowledge_access_scope = bound_knowledge_access_scope
        if (knowledge_store is None) != (knowledge_access_scope is None):
            raise ValueError(
                "knowledge_store and knowledge_access_scope must be configured together."
            )
        if (
            knowledge_access_scope is not None
            and bound_knowledge_access_scope is not None
            and copy_knowledge_access_scope(knowledge_access_scope) != bound_knowledge_access_scope
        ):
            raise ValueError(
                "knowledge_access_scope must match the scope bound to knowledge_store."
            )
        if dispatcher is not None and not isinstance(dispatcher, Dispatcher):
            raise TypeError("dispatcher must be a Dispatcher.")
        if budget_store is not None and not isinstance(budget_store, BudgetStore):
            raise TypeError("budget_store must be a BudgetStore.")
        if budget_ledger is not None and not isinstance(budget_ledger, BudgetLedger):
            raise TypeError("budget_ledger must be a BudgetLedger.")
        if event_watcher_store is not None and not isinstance(
            event_watcher_store,
            EventWatcherStore,
        ):
            raise TypeError("event_watcher_store must be an EventWatcherStore.")
        if secret_redactor is not None and not isinstance(secret_redactor, SecretRedactor):
            raise TypeError("secret_redactor must be a SecretRedactor.")
        if public_authority_alias_keyring is not None and not isinstance(
            public_authority_alias_keyring,
            PublicAuthorityAliasKeyring,
        ):
            raise TypeError("public_authority_alias_keyring must be a PublicAuthorityAliasKeyring.")
        if execution_profile_policy is not None and not isinstance(
            execution_profile_policy,
            ExecutionProfilePolicy,
        ):
            raise TypeError("execution_profile_policy must be an ExecutionProfilePolicy.")
        if completion_verifier_profile_policy is not None and not isinstance(
            completion_verifier_profile_policy,
            CompletionVerifierProfilePolicy,
        ):
            raise TypeError(
                "completion_verifier_profile_policy must be a CompletionVerifierProfilePolicy."
            )
        if egress_authority_adoption_handler is not None and not isinstance(
            egress_authority_adoption_handler,
            EgressAuthorityAdoptionHandler,
        ):
            raise TypeError(
                "egress_authority_adoption_handler must be an EgressAuthorityAdoptionHandler."
            )
        self._egress_authority_adoption_handler = egress_authority_adoption_handler
        if type(enable_logging) is not bool:
            raise TypeError("enable_logging must be a bool.")
        resolved_secret_redactor = (
            secret_redactor if secret_redactor is not None else SecretRedactor()
        )
        hooks = _validate_runtime_hooks(
            runtime_hooks,
            field_name="runtime_hooks",
            redactor=resolved_secret_redactor,
        )
        policies = validate_loop_policies(loop_policies, field_name="loop_policies")
        policy_execution_profile_identities = tuple(
            copy_secret_free_execution_profile_behavior_identity(
                policy.execution_profile_identity,
                redactor=resolved_secret_redactor,
                field_name=f"loop_policies[{index}].execution_profile_identity",
            )
            for index, policy in enumerate(policies)
        )
        # Opaque registrations are comparable only within this frozen app
        # registration. A new app may carry different behavior behind the same
        # component type and slot, even when it lives in the same OS process.
        self._execution_profile_process_identity = uuid4().hex
        # Wall-clock seam for time-based approval expiry (tests inject a fake).
        self._clock = _clock_or_utc_now(clock)
        manifest_policy = copy_mcp_manifest_policy(mcp_manifest_policy)
        result_projection_policy = copy_tool_result_projection_policy(tool_result_projection_policy)
        context_counting_config = copy_context_counting_config(context_counting)
        request_footprint_config = copy_request_footprint_config(request_footprint)
        execution_profile_policy_identity = None
        if execution_profile_policy is not None:
            execution_profile_policy_identity = require_durable_clean_nonblank(
                execution_profile_policy.identity,
                "execution_profile_policy.identity",
            )
            require_unicode_scalar_text(
                execution_profile_policy_identity,
                "execution_profile_policy.identity",
            )
            if len(execution_profile_policy_identity.encode("utf-8")) > 256:
                raise ValueError(
                    "execution_profile_policy.identity must be at most 256 UTF-8 bytes."
                )
            if (
                resolved_secret_redactor.redact_text(execution_profile_policy_identity)
                != execution_profile_policy_identity
            ):
                raise ValueError(
                    "execution_profile_policy.identity contains a workload secret and cannot "
                    "be used as durable policy authority."
                )
        configured_alias_codec = (
            None
            if public_authority_alias_keyring is None
            else PublicAuthorityAliasCodec(public_authority_alias_keyring)
        )
        if event_sinks is None:
            sinks = []
        else:
            if isinstance(event_sinks, str | bytes):
                raise TypeError("event_sinks must be an iterable of EventSink instances.")
            try:
                sinks = list(event_sinks)
            except TypeError as exc:
                raise TypeError("event_sinks must be an iterable of EventSink instances.") from exc
        for sink in sinks:
            if not isinstance(sink, EventSink):
                raise TypeError("event_sinks must contain EventSink instances.")
        if enable_logging:
            from cayu.observability.logging import LoggingEventSink

            sinks.insert(0, LoggingEventSink(redactor=resolved_secret_redactor))
        self._max_file_attachment_bytes = _validate_positive_int(
            max_file_attachment_bytes,
            "max_file_attachment_bytes",
        )
        self._max_total_file_attachment_bytes = _validate_positive_int(
            max_total_file_attachment_bytes,
            "max_total_file_attachment_bytes",
        )
        self._max_file_attachments_per_request = _validate_positive_int(
            max_file_attachments_per_request,
            "max_file_attachments_per_request",
        )
        self._tool_timeout_seconds = _validate_optional_positive_seconds(
            tool_timeout_seconds,
            "tool_timeout_seconds",
        )
        self._max_parallel_tool_calls = _validate_positive_int(
            max_parallel_tool_calls,
            "max_parallel_tool_calls",
        )
        self._max_environment_lifecycle_owners = _validate_positive_int(
            max_environment_lifecycle_owners,
            "max_environment_lifecycle_owners",
        )
        self.session_store = (
            session_store
            if session_store is not None
            else InMemorySessionStore(
                public_authority_alias_codec=configured_alias_codec,
            )
        )
        store_alias_codec = self.session_store.public_authority_alias_codec
        if configured_alias_codec is not None and store_alias_codec != configured_alias_codec:
            raise ValueError(
                "session_store and CayuApp must use the same public authority alias keyring."
            )
        self._public_authority_alias_codec = store_alias_codec or configured_alias_codec
        if resolved_secret_redactor.has_values and (
            self._public_authority_alias_codec is None
            or not self.session_store.supports_public_authority_aliases
        ):
            raise ValueError(
                "A secret-redacting CayuApp requires a SessionStore configured with "
                "durable public authority aliases and an explicit alias keyring."
            )
        self._runtime_session_store = runtime_checkpoint_session_store(self.session_store)
        self.task_store = task_store
        self.knowledge_store = knowledge_store
        self.knowledge_access_scope = (
            None
            if knowledge_access_scope is None
            else copy_knowledge_access_scope(knowledge_access_scope)
        )
        self.knowledge_review_namespace = (
            require_clean_nonblank(knowledge_review_namespace, "knowledge_review_namespace")
            if knowledge_review_namespace is not None
            else None
        )
        self.knowledge_review_labels = copy_label_map(
            knowledge_review_labels or {},
            "knowledge_review_labels",
        )
        self.dispatcher = dispatcher if dispatcher is not None else InlineDispatcher()
        self.budget_policy = copy_budget_policy(budget_policy)
        self.budget_store = (
            budget_store if budget_store is not None else SessionBudgetStore(self.session_store)
        )
        self.budget_ledger = (
            budget_ledger if budget_ledger is not None else InMemoryBudgetLedger(clock=self._clock)
        )
        self.event_watcher_store = (
            event_watcher_store if event_watcher_store is not None else InMemoryEventWatcherStore()
        )
        self._secret_redactor = resolved_secret_redactor
        self._completion_verifier_coordinator = CompletionVerifierCoordinator(
            task_store=self.task_store,
            secret_redactor=self._secret_redactor,
            profile_policy=completion_verifier_profile_policy,
        )
        self._completion_decision_application_coordinator = (
            CompletionDecisionApplicationCoordinator(
                task_store=self.task_store,
                secret_redactor=self._secret_redactor,
            )
        )
        self._default_retry_policy = copy_retry_policy(retry_policy)
        self._runtime_hooks = tuple(hooks)
        self._loop_policies = tuple(policies)
        self._loop_policy_execution_profile_identities = policy_execution_profile_identities
        self._mcp_manifest_policy = manifest_policy
        self._tool_result_projection_policy = result_projection_policy
        self._context_counting = context_counting_config
        self._request_footprint = request_footprint_config
        self._event_sinks = tuple(sinks)
        self._event_writer = RuntimeEventWriter(
            session_store=self._runtime_session_store,
            budget_store=self.budget_store,
            event_sinks=self._event_sinks,
            secret_redactor=self._secret_redactor,
            public_authority_alias_codec=self._public_authority_alias_codec,
        )
        self._completion_result_resolver_coordinator = CompletionResultResolverCoordinator(
            application_coordinator=self._completion_decision_application_coordinator,
            session_store=self._runtime_session_store,
            event_writer=self._event_writer,
            secret_redactor=self._secret_redactor,
        )
        self._environment_lifecycle = EnvironmentLifecycle(
            session_store=self._runtime_session_store,
            event_writer=self._event_writer,
            checkpoint_transform=_replace_checkpoint_preserving_runtime_state,
            secret_redactor=self._secret_redactor,
            max_environment_lifecycle_owners=self._max_environment_lifecycle_owners,
            egress_authority_adoption_handler=egress_authority_adoption_handler,
        )
        self._run_limit_controller = RunLimitController(
            session_store=self._runtime_session_store,
            budget_store=self.budget_store,
            budget_ledger=self.budget_ledger,
            event_writer=self._event_writer,
            clock=self._clock,
        )
        self._agents: dict[str, runtime_records.RegisteredAgentState] = {}
        self._knowledge_publications_sealed = False
        self._fork_group_evaluator_agents: dict[str, str] = {}
        self._providers: dict[str, runtime_records.RegisteredProvider] = {}
        self._environments: dict[str, runtime_records.RegisteredEnvironment] = {}
        self._artifact_store_registrations_by_id: dict[str, _ArtifactStoreRegistration] = {}
        self._default_provider_name: str | None = None
        self._default_environment_name: str | None = None
        self._session_control = SessionControl[SessionUsageTracker](
            session_store=self._runtime_session_store
        )
        self._model_step_executor = ModelStepExecutor(
            session_store=self._runtime_session_store,
            event_writer=self._event_writer,
            session_control=self._session_control,
            run_limit_controller=self._run_limit_controller,
            context_counting=self._context_counting,
            request_footprint=self._request_footprint,
            max_file_attachment_bytes=self._max_file_attachment_bytes,
            max_total_file_attachment_bytes=self._max_total_file_attachment_bytes,
            max_file_attachments_per_request=self._max_file_attachments_per_request,
            secret_redactor=self._secret_redactor,
            clock=self._clock,
            checkpoint_transform=(
                self._environment_lifecycle.checkpoint_transform_preserving_runtime_state
            ),
            apply_budget_evaluation=self._apply_model_step_budget_evaluation,
            apply_limit_evaluation=self._apply_model_step_limit_evaluation,
            stop_for_budget_reservation_failure=(
                self._stop_for_model_step_budget_reservation_failure
            ),
        )
        self._tool_round_executor = ToolRoundExecutor(
            session_store=self._runtime_session_store,
            event_writer=self._event_writer,
            session_control=self._session_control,
            hook_runtime=self,
            runtime_hooks=self._runtime_hooks,
            mcp_manifest_policy=self._mcp_manifest_policy,
            tool_result_projection_policy=self._tool_result_projection_policy,
            secret_redactor=self._secret_redactor,
            tool_timeout_seconds=self._tool_timeout_seconds,
            max_parallel_tool_calls=self._max_parallel_tool_calls,
            clock=self._clock,
            checkpoint_transform=_replace_checkpoint_preserving_runtime_state,
            apply_limit_evaluation=self._apply_tool_round_limit,
            close_interrupted_round=self._close_tool_round_after_interrupt,
        )
        self._recovery_coordinator = RecoveryCoordinator(
            session_store=self._runtime_session_store,
            task_store=self.task_store,
            event_writer=self._event_writer,
            session_control=self._session_control,
            environment_lifecycle=self._environment_lifecycle,
            run_limit_controller=self._run_limit_controller,
            tool_round_executor=self._tool_round_executor,
            secret_redactor=self._secret_redactor,
            clock=self._clock,
            checkpoint_transform=_replace_checkpoint_preserving_runtime_state,
            effective_retry_policy=self._effective_retry_policy,
            run_session=self._run_recovery_session,
            emit_terminal_event_with_hooks=self._emit_recovery_terminal_event_with_hooks,
            fail_provider_operation=self._fail_provider_operation_resolution,
            stop_session_for_limit_reached=self._stop_recovery_session_for_limit_reached,
            task_event=_recovery_task_event,
            resolve_registered_agent=self._get_registered_agent,
            resolve_registered_provider=self._get_registered_provider,
            resolve_registered_environment=self._get_registered_environment_for_session,
            resolve_budget_policy=lambda: self.budget_policy,
            validate_execution_profile_continuation=(
                self._validate_execution_profile_continuation_for_recovery
            ),
            interrupt_session_for_recovery=self._interrupt_session_for_recovery,
            pending_session_interrupt_checkpoint=(
                self._pending_session_interrupt_checkpoint_for_recovery
            ),
            abandoned_turn_completed=self._complete_abandoned_recovery_turn,
            resume_interaction=self._resume_recovery_interaction,
            recover_provider_operation=self._recover_provider_operation,
            recover_provider_operation_start=self._recover_provider_operation_start,
            cancel_provider_operation=self._cancel_provider_operation,
            interaction_transition_replay_failures=_interaction_transition_replay_failures,
        )
        self._background_interruption_coordinator = BackgroundInterruptionCoordinator(
            session_store=self._runtime_session_store,
            event_writer=self._event_writer,
            clock=self._clock,
            interrupt_session=self.interrupt_session,
            load_pending_session_interrupt_payload=self._load_pending_session_interrupt_payload,
            latest_session_interrupted_event=self._session_control.latest_interrupted_event,
            load_pending_interruption_cascade=self._load_pending_interruption_cascade,
            claim_pending_interruption_cascade=self._claim_pending_interruption_cascade,
            mark_pending_interruption_cascade_failed=(
                self._mark_pending_interruption_cascade_failed
            ),
            complete_pending_interruption_cascade=self._complete_pending_interruption_cascade,
            renew_pending_interruption_cascade_claim=(
                self._renew_pending_interruption_cascade_claim
            ),
            release_pending_interruption_cascade_claim=(
                self._release_pending_interruption_cascade_claim
            ),
            secret_redactor=self._secret_redactor,
        )

        self._session_engine = SessionEngine(
            session_store=self._runtime_session_store,
            task_store=self.task_store,
            get_budget_policy=lambda: self.budget_policy,
            event_writer=self._event_writer,
            environment_lifecycle=self._environment_lifecycle,
            run_limit_controller=self._run_limit_controller,
            session_control=self._session_control,
            model_step_executor=self._model_step_executor,
            request_footprint=self._request_footprint,
            tool_round_executor=self._tool_round_executor,
            recovery_coordinator=self._recovery_coordinator,
            background_interruption_coordinator=(self._background_interruption_coordinator),
            secret_redactor=self._secret_redactor,
            clock=self._clock,
            runtime_hooks=self._runtime_hooks,
            loop_policies=self._loop_policies,
            loop_policy_execution_profile_identities=(
                self._loop_policy_execution_profile_identities
            ),
            hook_runtime=self,
            get_registered_agent=self._get_registered_agent,
            get_registered_provider=self._get_registered_provider,
            route_registered_provider_for_model=(
                lambda model: self._route_registered_provider_for_model(model=model)
            ),
            get_registered_environment=self._get_registered_environment,
            get_registered_environment_for_session=(self._get_registered_environment_for_session),
            effective_retry_policy=self._effective_retry_policy,
            execution_profile_policy=execution_profile_policy,
            execution_profile_policy_identity=execution_profile_policy_identity,
            egress_authority_adoption_handler=egress_authority_adoption_handler,
            execution_profile_process_identity=self._execution_profile_process_identity,
        )
        self._durable_subagent_coordinator = DurableSubagentCoordinator(
            session_store=self.session_store,
            runtime_session_store=self._runtime_session_store,
            task_store=self.task_store,
            dispatcher=self.dispatcher,
            prepare_initial_run=self._prepare_durable_subagent_run,
            resolve_registered_agent=self._get_registered_agent,
            resolve_registered_provider=self._get_registered_provider,
            route_registered_provider_for_model=(
                lambda model: self._route_registered_provider_for_model(model=model)
            ),
            resolve_registered_environment=self._get_registered_environment,
            interrupt_session=self._interrupt_session_private,
            load_session_invocation=self.session_invocation_for_dispatch,
            classify_dispatch_settlement=self._queued_dispatch_settlement_state,
            acknowledge_dispatch=self._acknowledge_queued_dispatch,
        )
        from cayu.runtime.fork_groups import ForkGroupCoordinator

        self._fork_group_coordinator = ForkGroupCoordinator(
            session_store=self._runtime_session_store,
            secret_redactor=self._secret_redactor,
            clock=self._clock,
            resolve_public_session_authority=self._resolve_public_session_authority,
            fork_session=self.fork_session,
            run_session=self.run,
            resume_session=self.resume,
            recover_incomplete_session=self.recover_incomplete_session,
            get_session_usage=self.get_session_usage,
            preflight_evaluator_agent=self._preflight_fork_group_evaluator_agent,
            prepare_evaluator_agent=self._prepare_fork_group_evaluator_agent,
            preflight_fork_source=self._preflight_ordinary_fork_source,
            preflight_fork_source_state=(
                self._session_engine.preflight_fork_source_checkpoint_state
            ),
            admit_fork_source=self._admit_ordinary_fork_source,
            interrupt_session=self._interrupt_session_private,
            dispatcher=self.dispatcher,
            task_store=self.task_store,
            prepare_queued_dispatch=self._prepare_fork_group_queued_dispatch,
            dispatch_session=self.dispatch,
        )

    def redact_json(self, value: Any) -> Any:
        """Return a JSON-compatible value with configured secret values redacted."""
        return self._secret_redactor.redact_json(value)

    def redact_uppercase_text(self, value: str) -> str:
        """Redact text after the same uppercase normalization used by authorities."""

        return self._secret_redactor.redact_uppercase_text(value)

    def stream_redacted_bytes(
        self,
        *,
        max_retained_bytes: int | None = None,
    ) -> SecretRedactionStream:
        """Create a chunk-safe redaction stream for application exposure boundaries."""

        return self._secret_redactor.stream_bytes(
            max_retained_bytes=max_retained_bytes,
        )

    def project_event_record_for_exposure(self, record: EventRecord) -> EventRecord:
        """Project a caller-supplied record without granting persisted authority."""

        if type(record) is not EventRecord:
            raise TypeError("record must be an EventRecord.")
        return EventRecord(
            sequence=record.sequence,
            event=project_runtime_event(
                record.event,
                sequence=record.sequence,
                redactor=self._secret_redactor,
                public_authority_alias_codec=self._public_authority_alias_codec,
            ),
        )

    def _project_persisted_event_record_for_exposure(
        self,
        record: EventRecord,
    ) -> EventRecord:
        """Project a record obtained internally from the durable session store."""

        if type(record) is not EventRecord:
            raise TypeError("record must be an EventRecord.")
        return EventRecord(
            sequence=record.sequence,
            event=project_persisted_runtime_event(
                record.event,
                sequence=record.sequence,
                redactor=self._secret_redactor,
                public_authority_alias_codec=self._public_authority_alias_codec,
            ),
        )

    async def _project_emitted_event_for_public_api(self, event: Event) -> Event:
        """Project one emitted private event at the public application boundary."""

        sequence = event_durable_sequence(event)
        if sequence is None:
            records = await self.session_store.query_events(
                EventQuery(session_id=event.session_id, event_id=event.id, limit=2)
            )
            if len(records) != 1:
                raise RuntimeError(
                    "Runtime event has no unique durable record for public projection."
                )
            sequence = records[0].sequence
            event = records[0].event
        projected = project_persisted_runtime_event(
            event,
            sequence=sequence,
            redactor=self._secret_redactor,
            public_authority_alias_codec=self._public_authority_alias_codec,
        )
        return event_with_durable_sequence(projected, sequence)

    async def _project_incomplete_recovery_result_for_public_api(
        self,
        result: IncompleteSessionRecoveryResult,
    ) -> IncompleteSessionRecoveryResult:
        """Project recovery events and their actionable pending identifiers together."""

        projected_events = tuple(
            [await self._project_emitted_event_for_public_api(event) for event in result.events]
        )
        updates: dict[str, Any] = {
            "events": projected_events,
            "session_id": self.project_session_id_for_exposure(result.session_id),
        }
        unavailable_linkage: list[str] = []
        for result_field, event_field in (
            ("pending_approval_id", "approval_id"),
            ("pending_user_input_id", "input_id"),
        ):
            private_value = getattr(result, result_field)
            if private_value is None:
                continue
            aliases = [
                public_event_linkage_id(sequence, event_field)
                for private_event, public_event in zip(
                    result.events,
                    projected_events,
                    strict=True,
                )
                if private_event_linkage_value(
                    private_event,
                    field_name=event_field,
                )
                == private_value
                and (sequence := event_durable_sequence(public_event)) is not None
            ]
            if not aliases:
                try:
                    records = await self.session_store.query_events(
                        EventQuery(
                            session_id=result.session_id,
                            order_by=EventOrder.SEQUENCE_DESC,
                            limit=5000,
                        )
                    )
                except Exception:
                    records = []
                aliases = [
                    public_event_linkage_id(record.sequence, event_field)
                    for record in reversed(records)
                    if private_event_linkage_value(
                        record.event,
                        field_name=event_field,
                    )
                    == private_value
                ]
            if not aliases:
                # Recovery has already committed before this public projection
                # boundary. A bounded legacy-history lookup may not locate an old
                # linkage record, but that must not turn the committed recovery
                # into a reported failure or expose the private action ID. Return
                # a safe non-actionable representation and an explicit diagnostic
                # in the result message instead.
                updates[result_field] = None
                unavailable_linkage.append(result_field)
                continue
            updates[result_field] = aliases[-1]
        if unavailable_linkage:
            fields = ", ".join(unavailable_linkage)
            updates["message"] = (
                f"{result.message} Public linkage unavailable for: {fields}; "
                "inspect pending session actions before continuing."
            )
        return result.model_copy(update=updates, deep=True)

    def project_session_id_for_exposure(self, value: str) -> str:
        """Project private session authority to one stable public identifier."""

        value = require_clean_nonblank(value, "session_id")
        if self._secret_redactor.redact_text(value) == value:
            return value
        return public_event_envelope_alias(
            value,
            field_name="session_id",
            codec=self._require_public_authority_alias_codec(),
        )

    def project_causal_budget_id_for_exposure(
        self,
        value: str,
        *,
        session_ids: Iterable[str],
    ) -> str:
        """Project session authority or redact an opaque causal-budget label."""

        value = require_clean_nonblank(value, "causal_budget_id")
        if any(
            require_clean_nonblank(session_id, "session_id") == value for session_id in session_ids
        ):
            return self.project_session_id_for_exposure(value)
        return self._secret_redactor.redact_text(value)

    def project_interaction_id_for_exposure(
        self,
        value: str,
        *,
        session_id: str,
    ) -> str:
        """Project private interaction authority to one stable public identifier."""

        value = require_clean_nonblank(value, "interaction_id")
        if self._secret_redactor.redact_text(value) == value:
            return value
        return public_event_envelope_alias(
            value,
            field_name="interaction_id",
            codec=self._require_public_authority_alias_codec(),
            session_id=require_clean_nonblank(session_id, "session_id"),
        )

    def _require_public_authority_alias_codec(self) -> PublicAuthorityAliasCodec:
        codec = self._public_authority_alias_codec
        if codec is None:
            raise RuntimeError(
                "Secret-bearing public authority requires a configured alias keyring."
            )
        return codec

    async def _resolve_public_action_linkage(
        self,
        *,
        session_id: str,
        value: str,
        field_name: str,
    ) -> str:
        """Resolve one public event alias back to private durable authority.

        The alias selects a record and schema field only. The durable event,
        rather than the caller-provided alias, remains the authority used by
        approval, input, and recovery operations.
        """

        if not value.startswith(PUBLIC_EVENT_ID_PREFIX):
            return value
        try:
            pending = await self.session_store.query_pending_actions(
                PendingActionQuery(session_id=session_id, limit=200)
            )
        except PendingActionResultTooLarge as exc:
            raise ValueError(
                f"{field_name} cannot be disambiguated from legacy private "
                "authority because the pending-action evidence is too large."
            ) from exc
        action_field_name = (
            "round_id" if field_name in {"round_id", "tool_round_id"} else field_name
        )
        raw_match = any(
            getattr(action, action_field_name, None) == value for action in pending.actions
        )
        sequence = public_event_linkage_sequence(value, field_name=field_name)
        if sequence is None:
            if raw_match:
                return value
            raise ValueError(f"Public {field_name} alias is malformed or field-mismatched.")
        records = await self.session_store.query_events(
            EventQuery(
                session_id=session_id,
                after_sequence=sequence - 1,
                limit=1,
            )
        )
        if (
            not records
            or records[0].sequence != sequence
            or records[0].event.session_id != session_id
        ):
            raise ValueError(f"Public {field_name} alias was not found in the requested session.")
        private_value = private_event_linkage_value(
            records[0].event,
            field_name=field_name,
        )
        if private_value is None:
            raise ValueError(f"Public {field_name} alias has no private durable authority.")
        if raw_match and private_value != value:
            raise ValueError(
                f"{field_name} is ambiguous between legacy private authority "
                "and a public event alias."
            )
        return private_value

    async def _resolve_public_session_id(self, value: str) -> str:
        """Resolve a stable public session alias to private store authority."""

        private_value, _store_resolved_value = await self._resolve_public_session_authority(value)
        return private_value

    async def _resolve_public_causal_budget_id(self, value: str) -> str:
        """Disambiguate raw causal-budget authority from a public session alias."""

        return await self._resolve_public_session_backed_filter(
            value,
            field_name="causal_budget_id",
        )

    async def _resolve_public_parent_session_id(self, value: str) -> str:
        """Disambiguate raw parent authority from a public session alias."""

        return await self._resolve_public_session_backed_filter(
            value,
            field_name="parent_session_id",
        )

    async def _resolve_public_session_backed_filter(
        self,
        value: str,
        *,
        field_name: str,
    ) -> str:
        """Resolve a public session alias while preserving matching legacy linkage."""

        if field_name not in {"causal_budget_id", "parent_session_id"}:
            raise ValueError("Unsupported session-backed authority filter.")
        value = require_clean_nonblank(value, field_name)
        parsed = parse_public_authority_alias(value)
        if parsed is None or parsed.field_name != "session_id":
            return value

        query = (
            SessionQuery(causal_budget_id=value, limit=1)
            if field_name == "causal_budget_id"
            else SessionQuery(parent_session_id=value, limit=1)
        )
        raw_match = bool((await self.session_store.list_sessions(query)).sessions)
        private_value = await self.session_store.resolve_public_authority_alias(
            value,
            field_name="session_id",
        )
        if private_value is None:
            if raw_match:
                return value
            raise ValueError(f"Public {field_name} alias was not found.")
        if raw_match and private_value != value:
            raise ValueError(
                f"{field_name} is ambiguous between legacy private authority "
                "and a public session alias."
            )
        return value if raw_match else private_value

    async def _resolve_public_session_authority(
        self,
        value: str,
    ) -> tuple[str, str | None]:
        """Return private authority plus positive store-resolution evidence."""

        if not public_authority_alias_is_reserved(value):
            return value, None
        raw_session = await self.session_store.load(value)
        parsed = parse_public_authority_alias(value)
        if parsed is None or parsed.field_name != "session_id":
            if raw_session is not None:
                return value, value
            raise ValueError("Public session_id alias is malformed or field-mismatched.")
        private_value = await self.session_store.resolve_public_authority_alias(
            value,
            field_name="session_id",
        )
        if private_value is None:
            if raw_session is not None:
                return value, value
            raise ValueError("Public session_id alias was not found.")
        if raw_session is not None and private_value != value:
            raise ValueError(
                "session_id is ambiguous between legacy private authority and a public event alias."
            )
        return private_value, private_value

    async def _resolve_public_interaction_id(
        self,
        *,
        session_id: str,
        value: str,
    ) -> str:
        """Resolve a stable interaction alias inside one private session."""

        if not public_authority_alias_is_reserved(value):
            return value
        raw_value_exists = await self.session_store.public_authority_private_value_exists(
            value,
            field_name="interaction_id",
            scope_session_id=session_id,
        )
        parsed = parse_public_authority_alias(value)
        if parsed is None or parsed.field_name != "interaction_id":
            if raw_value_exists:
                return value
            raise ValueError("Public interaction_id alias is malformed or field-mismatched.")
        private_value = await self.session_store.resolve_public_authority_alias(
            value,
            field_name="interaction_id",
            scope_session_id=session_id,
        )
        if private_value is None:
            if raw_value_exists:
                return value
            raise ValueError("Public interaction_id alias was not found in the requested session.")
        if raw_value_exists and private_value != value:
            raise ValueError(
                "interaction_id is ambiguous between legacy private authority "
                "and a public event alias."
            )
        return private_value

    def redact_exception_diagnostic(
        self,
        error: BaseException,
        *,
        empty_message: str,
        nonportable_message: str,
    ) -> ExceptionDiagnostic:
        """Snapshot a dispatch diagnostic with workload secrets removed before bounding."""

        return exception_diagnostic(
            error,
            empty_message=empty_message,
            nonportable_message=nonportable_message,
            redactor=self._secret_redactor,
        )

    def redact_dispatch_request(self, request: DispatchRequest) -> DispatchRequest:
        """Return a durable dispatch request scrubbed with this app's redactor."""

        return redact_dispatch_request(request, redactor=self._secret_redactor)

    def _pending_tool_approval_from_checkpoint(
        self,
        checkpoint: dict[str, Any] | None,
        *,
        consume_on_rejection: bool = False,
    ) -> PendingToolApproval | None:
        """Parse trusted approval state through this app's secret boundary."""

        return approval_support.pending_approval_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=consume_on_rejection,
        )

    def describe(self, *, project_root: str | Path | None = None) -> AppManifest:
        """Return this application's deterministic public manifest.

        Description is structural only: it never invokes providers, tools,
        environment factories, stores, workers, watchers, or recovery paths.
        """

        return describe_app(self, project_root=project_root)

    async def drain_background_interruptions(self, *, timeout_s: float = 10.0) -> bool:
        return await self._session_engine.drain_background_interruptions(timeout_s=timeout_s)

    async def drain_environment_cleanups(self, *, timeout_s: float = 10.0) -> bool:
        """Settle retained environment cleanup without cancelling live mutations."""

        if type(timeout_s) not in {int, float} or not isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be a finite positive number.")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(timeout_s)
        handler = self._egress_authority_adoption_handler
        parked_drained = True
        if handler is not None:
            parked_drained = await _drain_parked_egress_authority_allocations(
                handler,
                timeout_s=float(timeout_s),
            )
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        retained_drained = await self._environment_lifecycle.drain_retained_cleanups(
            timeout_s=remaining,
        )
        return retained_drained and parked_drained

    def seal_knowledge_publications(self) -> None:
        """Reject new retained knowledge writes before application shutdown drains."""

        self._knowledge_publications_sealed = True
        for lifecycle in self._registered_knowledge_publication_lifecycles():
            lifecycle.seal()

    async def drain_knowledge_publications(self, *, timeout_s: float = 10.0) -> bool:
        """Seal and concurrently drain publications owned by registered tools."""

        if type(timeout_s) not in {int, float} or not isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be a finite positive number.")
        self.seal_knowledge_publications()
        lifecycles = self._registered_knowledge_publication_lifecycles()
        if not lifecycles:
            return True
        results = await asyncio.gather(
            *(lifecycle.aclose(timeout_s=float(timeout_s)) for lifecycle in lifecycles)
        )
        return all(results)

    def _registered_knowledge_publication_lifecycles(
        self,
    ) -> tuple[KnowledgePublicationLifecycle, ...]:
        lifecycles: list[KnowledgePublicationLifecycle] = []
        seen: set[int] = set()
        for agent in self._agents.values():
            for registered in agent.tools.values():
                lifecycle = getattr(
                    registered.tool,
                    "_knowledge_publication_lifecycle",
                    None,
                )
                if not isinstance(lifecycle, KnowledgePublicationLifecycle):
                    continue
                identity = id(lifecycle)
                if identity in seen:
                    continue
                seen.add(identity)
                lifecycles.append(lifecycle)
        return tuple(lifecycles)

    async def discard_parked_egress_allocations(
        self,
        session_id: str,
        *,
        timeout_s: float = 10.0,
    ) -> bool:
        """Discard parked external allocations before abandoning one session."""

        handler = self._egress_authority_adoption_handler
        if handler is None:
            return True
        return await _drain_parked_egress_authority_allocations(
            handler,
            timeout_s=timeout_s,
            session_id=require_durable_clean_nonblank(session_id, "session_id"),
        )

    async def resume_pending_interruption_cascades(
        self,
        *,
        interrupting_inactive_before: datetime | None = None,
    ) -> int:
        if interrupting_inactive_before is not None:
            if (
                interrupting_inactive_before.tzinfo is None
                or interrupting_inactive_before.utcoffset() is None
            ):
                raise ValueError("interrupting_inactive_before must be timezone-aware.")
            interrupting_inactive_before = interrupting_inactive_before.astimezone(UTC)
        return await self._session_engine.resume_pending_interruption_cascades(
            interrupting_inactive_before=interrupting_inactive_before
        )

    async def interruption_cascade_status(self, session_id: str) -> str:
        session_id = await self._resolve_public_session_id(
            require_clean_nonblank(session_id, "session_id")
        )
        return await self._session_engine.interruption_cascade_status(session_id=session_id)

    async def inspect_targeted_tool_grants(
        self,
        session_id: str,
        *,
        interaction_id: str | None = None,
        limit: int = TARGETED_TOOL_GRANT_INSPECTION_MAX_RECORDS,
    ) -> tuple[TargetedToolGrantInspection, ...]:
        """Return bounded grant state through authenticated public aliases."""

        if not self.session_store.supports_targeted_tool_grants:
            raise RuntimeError("The configured SessionStore does not support targeted grants.")
        private_session_id = await self._resolve_public_session_id(
            require_clean_nonblank(session_id, "session_id")
        )
        private_interaction_id = (
            None
            if interaction_id is None
            else await self._resolve_public_interaction_id(
                session_id=private_session_id,
                value=require_clean_nonblank(interaction_id, "interaction_id"),
            )
        )
        records = await self.session_store.list_targeted_tool_grants(
            private_session_id,
            interaction_id=private_interaction_id,
            limit=limit,
        )
        codec = self.session_store.public_authority_alias_codec
        if codec is None:
            raise RuntimeError("Targeted grant inspection requires public alias authority.")
        return tuple(targeted_tool_grant_inspection(record, codec) for record in records)

    def register_agent(
        self,
        spec: AgentSpec,
        *,
        tools: Iterable[Tool] | None = None,
        hosted_tools: Iterable[OpenAIWebSearch] | None = None,
        context_policy: ContextPolicy | None = None,
        context_overflow_policy: ContextPolicy | None = None,
        tool_exposure_policy: ToolExposurePolicy | None = None,
        targeted_tool_mode: TargetedToolMode | str | None = None,
        tool_policy: ToolPolicy | None = None,
        runtime_hooks: Iterable[RuntimeHook] | None = None,
        loop_policies: Iterable[LoopPolicy] | None = None,
        execution_requirements: ExecutionRequirements | None = None,
    ) -> AgentSpec:
        if type(spec) is not AgentSpec:
            raise TypeError("Agent registration requires an AgentSpec.")
        stored_spec = _validate_agent_spec(spec)
        if stored_spec.name in self._agents:
            raise ValueError(f"Agent already registered: {stored_spec.name}")
        if context_policy is None:
            stored_context_policy = DefaultContextPolicy()
        elif isinstance(context_policy, ContextPolicy):
            stored_context_policy = context_policy
        else:
            raise TypeError("context_policy must be a ContextPolicy.")
        if context_overflow_policy is None:
            stored_context_overflow_policy = None
        elif isinstance(context_overflow_policy, ContextPolicy):
            stored_context_overflow_policy = context_overflow_policy
        else:
            raise TypeError("context_overflow_policy must be a ContextPolicy.")
        if tool_exposure_policy is None:
            stored_tool_exposure_policy = AllRegisteredToolsExposurePolicy()
        elif isinstance(tool_exposure_policy, ToolExposurePolicy):
            stored_tool_exposure_policy = tool_exposure_policy
        else:
            raise TypeError("tool_exposure_policy must be a ToolExposurePolicy.")
        stored_targeted_tool_mode = (
            None if targeted_tool_mode is None else copy_targeted_tool_mode(targeted_tool_mode)
        )
        if stored_targeted_tool_mode is not None and (
            not self.session_store.supports_targeted_tool_grants
            or not self.session_store.supports_public_authority_aliases
            or self.session_store.public_authority_alias_codec is None
        ):
            raise RuntimeError(
                "targeted_tool_mode requires a SessionStore with durable targeted-grant "
                "state and configured public authority aliases."
            )
        if tool_policy is None:
            stored_tool_policy = AllowAllToolPolicy()
        elif isinstance(tool_policy, ToolPolicy):
            stored_tool_policy = tool_policy
        else:
            raise TypeError("tool_policy must be a ToolPolicy.")
        stored_runtime_hooks = _validate_runtime_hooks(
            runtime_hooks,
            field_name="runtime_hooks",
            redactor=self._secret_redactor,
        )
        stored_loop_policies = validate_loop_policies(
            loop_policies,
            field_name="loop_policies",
        )
        if execution_requirements is None:
            stored_execution_requirements = ExecutionRequirements.trusted()
        elif isinstance(execution_requirements, ExecutionRequirements):
            stored_execution_requirements = ExecutionRequirements.model_validate(
                execution_requirements.model_dump(mode="python")
            )
        else:
            raise TypeError("execution_requirements must be ExecutionRequirements or None.")

        if tools is None:
            agent_tools = []
        else:
            if isinstance(tools, str | bytes):
                raise TypeError("Agent tools must be an iterable of Tool instances.")
            try:
                agent_tools = list(tools)
            except TypeError as exc:
                raise TypeError("Agent tools must be an iterable of Tool instances.") from exc

        tools_by_name: dict[str, runtime_records.RegisteredTool] = {}
        for tool in agent_tools:
            if not isinstance(tool, Tool):
                raise TypeError("Agent tools must be Tool instances.")
            registered_tool = _validate_registered_tool(
                tool,
                redactor=self._secret_redactor,
            )
            if registered_tool.name in tools_by_name:
                raise ValueError(f"Duplicate tool registered for agent: {registered_tool.name}")
            tools_by_name[registered_tool.name] = registered_tool

        if hosted_tools is None:
            stored_hosted_tools: tuple[OpenAIWebSearch, ...] = ()
        else:
            if isinstance(hosted_tools, str | bytes):
                raise TypeError("Agent hosted_tools must be an iterable of hosted tool instances.")
            try:
                copied_hosted_tools = tuple(
                    copy_openai_web_search(hosted_tool) for hosted_tool in hosted_tools
                )
            except TypeError as exc:
                raise TypeError(
                    "Agent hosted_tools must be an iterable of hosted tool instances."
                ) from exc
            if len(copied_hosted_tools) > 1:
                raise ValueError("Duplicate OpenAI web search hosted tool registered for agent.")
            stored_hosted_tools = copied_hosted_tools

        registration_source, registration_symbol = _registration_site()
        descriptors_by_name = {
            tool.name: _registered_tool_descriptor(tool) for tool in tools_by_name.values()
        }
        tool_catalogue = build_tool_catalog_snapshot(descriptors_by_name.values())
        tool_capabilities = tuple(
            RegisteredToolCapability(**descriptors_by_name[name].exposure_capability_material())
            for name in tools_by_name
        )
        all_registered_tool_exposure = ResolvedToolExposure(
            profile_id=ALL_REGISTERED_TOOLS_PROFILE_ID,
            catalogue_revision=tool_catalogue.revision,
            tools=tool_capabilities,
            registered_count=len(tool_capabilities),
            ceiling_count=len(tool_capabilities),
        )
        # Keep one frozen exposure graph for registration/profile admission and
        # the expose-all snapshot; the catalogue remains its canonical source.
        tool_capabilities = all_registered_tool_exposure.tools
        registered_agent = runtime_records.RegisteredAgentState(
            spec=stored_spec,
            tools=MappingProxyType(tools_by_name),
            tool_catalogue=tool_catalogue,
            tool_capabilities=tool_capabilities,
            all_registered_tool_exposure=all_registered_tool_exposure,
            tool_exposure_policy=stored_tool_exposure_policy,
            tool_exposure_policy_execution_profile_identity=(
                copy_secret_free_execution_profile_behavior_identity(
                    stored_tool_exposure_policy.execution_profile_identity,
                    redactor=self._secret_redactor,
                    field_name=("tool_exposure_policy.execution_profile_identity"),
                )
            ),
            targeted_tool_mode=stored_targeted_tool_mode,
            hosted_tools=stored_hosted_tools,
            context_policy=stored_context_policy,
            context_policy_execution_profile_identity=(
                copy_secret_free_execution_profile_behavior_identity(
                    stored_context_policy.execution_profile_identity,
                    redactor=self._secret_redactor,
                    field_name="context_policy.execution_profile_identity",
                )
            ),
            context_overflow_policy=stored_context_overflow_policy,
            context_overflow_policy_execution_profile_identity=(
                None
                if stored_context_overflow_policy is None
                else copy_secret_free_execution_profile_behavior_identity(
                    stored_context_overflow_policy.execution_profile_identity,
                    redactor=self._secret_redactor,
                    field_name="context_overflow_policy.execution_profile_identity",
                )
            ),
            tool_policy=stored_tool_policy,
            tool_policy_execution_profile_identity=(
                copy_secret_free_execution_profile_behavior_identity(
                    stored_tool_policy.execution_profile_identity,
                    redactor=self._secret_redactor,
                    field_name="tool_policy.execution_profile_identity",
                )
            ),
            runtime_hooks=stored_runtime_hooks,
            loop_policies=stored_loop_policies,
            loop_policy_execution_profile_identities=tuple(
                copy_secret_free_execution_profile_behavior_identity(
                    policy.execution_profile_identity,
                    redactor=self._secret_redactor,
                    field_name=f"loop_policies[{index}].execution_profile_identity",
                )
                for index, policy in enumerate(stored_loop_policies)
            ),
            execution_requirements=stored_execution_requirements,
            context_behavior_execution_profile_identities=(
                _snapshot_context_behavior_execution_profile_identities(
                    stored_context_policy,
                    stored_context_overflow_policy,
                    redactor=self._secret_redactor,
                )
            ),
            registration_source=registration_source,
            registration_symbol=registration_symbol,
        )
        if self._knowledge_publications_sealed:
            for registered_tool in tools_by_name.values():
                lifecycle = getattr(
                    registered_tool.tool,
                    "_knowledge_publication_lifecycle",
                    None,
                )
                if isinstance(lifecycle, KnowledgePublicationLifecycle):
                    lifecycle.seal()
        self._agents[stored_spec.name] = registered_agent
        return spec

    def register_completion_verifier(
        self,
        reference: CompletionVerifierRef,
        verifier: DeterministicCompletionVerifier,
    ) -> CompletionVerifierRef:
        """Register one deterministic verifier under its complete durable identity."""

        try:
            registered = self._completion_verifier_coordinator.register(reference, verifier)
        except BaseException:
            del reference, verifier
            raise
        del reference, verifier
        return registered

    def register_completion_result_resolver(
        self,
        reference: CompletionResultResolverRef,
        resolver: CompletionResultResolver,
    ) -> CompletionResultResolverRef:
        """Register one result resolver under its complete durable identity."""

        try:
            registered = self._completion_result_resolver_coordinator.register(
                reference,
                resolver,
            )
        except BaseException:
            del reference, resolver
            raise
        del reference, resolver
        return registered

    def register_fork_group_gate(
        self,
        gate_id: str,
        gate: ForkGroupGate,
    ) -> ForkGroupGateSelection:
        """Register one application-owned deterministic fork-group gate."""

        return self._fork_group_coordinator.register_gate(gate_id, gate)

    def register_fork_group_replacement_planner(
        self,
        planner_id: str,
        planner: ForkGroupReplacementPlanner,
    ) -> ForkGroupReplacementPlannerSelection:
        """Register one application-owned idempotent replacement planner."""

        return self._fork_group_coordinator.register_replacement_planner(
            planner_id,
            planner,
        )

    def _fork_group_evaluator_agent_state(
        self,
        *,
        source_agent_name: str,
        synthetic_agent_name: str,
        request_sha256: str,
    ) -> tuple[runtime_records.RegisteredAgentState, bool]:
        """Build or validate an isolated evaluator registration without publishing it."""

        original = self._get_registered_agent(source_agent_name)
        existing = self._agents.get(synthetic_agent_name)
        if existing is not None:
            if self._fork_group_evaluator_agents.get(synthetic_agent_name) != request_sha256:
                raise RuntimeError(
                    "Fork-group evaluator registration collides with application state."
                )
            if (
                existing.tools
                or existing.tool_catalogue.descriptors
                or existing.tool_capabilities
                or existing.hosted_tools
                or existing.spec.workflow_tool_names
                or existing.runtime_hooks
                or existing.loop_policies
            ):
                raise RuntimeError(
                    "Fork-group evaluator registration is not structurally isolated."
                )
            return existing, True
        evaluator_context_policy = DefaultContextPolicy()
        empty_tool_catalogue = ToolCatalogSnapshot(descriptor_count=0)
        return (
            runtime_records.RegisteredAgentState(
                spec=original.spec.model_copy(
                    update={
                        "name": synthetic_agent_name,
                        "workflow_tool_names": (),
                        "metadata": {},
                        "provider_options": {},
                    },
                    deep=True,
                ),
                tools=MappingProxyType({}),
                tool_catalogue=empty_tool_catalogue,
                tool_capabilities=(),
                all_registered_tool_exposure=ResolvedToolExposure(
                    profile_id=ALL_REGISTERED_TOOLS_PROFILE_ID,
                    catalogue_revision=empty_tool_catalogue.revision,
                    tools=(),
                    registered_count=0,
                    ceiling_count=0,
                ),
                tool_exposure_policy=AllRegisteredToolsExposurePolicy(),
                tool_exposure_policy_execution_profile_identity=None,
                targeted_tool_mode=None,
                context_policy=evaluator_context_policy,
                context_policy_execution_profile_identity=(
                    copy_secret_free_execution_profile_behavior_identity(
                        evaluator_context_policy.execution_profile_identity,
                        redactor=self._secret_redactor,
                        field_name="context_policy.execution_profile_identity",
                    )
                ),
                hosted_tools=(),
                context_overflow_policy=None,
                context_overflow_policy_execution_profile_identity=None,
                tool_policy=AllowAllToolPolicy(),
                tool_policy_execution_profile_identity=None,
                runtime_hooks=(),
                loop_policies=(),
                loop_policy_execution_profile_identities=(),
                execution_requirements=ExecutionRequirements.trusted(),
                context_behavior_execution_profile_identities=(
                    _snapshot_context_behavior_execution_profile_identities(
                        evaluator_context_policy,
                        None,
                        redactor=self._secret_redactor,
                    )
                ),
                registration_source=original.registration_source,
                registration_symbol=original.registration_symbol,
            ),
            False,
        )

    def _register_fork_group_evaluator_agent(
        self,
        *,
        source_agent_name: str,
        synthetic_agent_name: str,
        request_sha256: str,
    ) -> str:
        """Own one runtime-generated, structurally isolated evaluator registration."""

        evaluator_agent, already_registered = self._fork_group_evaluator_agent_state(
            source_agent_name=source_agent_name,
            synthetic_agent_name=synthetic_agent_name,
            request_sha256=request_sha256,
        )
        if already_registered:
            return synthetic_agent_name
        self._agents[synthetic_agent_name] = evaluator_agent
        self._fork_group_evaluator_agents[synthetic_agent_name] = request_sha256
        return synthetic_agent_name

    def _ensure_fork_group_task_evaluator_environment(self, request: RunRequest) -> None:
        """Materialize the provider-only environment shared by task workers."""

        from cayu.runtime.fork_groups import _FORK_GROUP_TASK_EVALUATOR_ENVIRONMENT

        if request.environment_name != _FORK_GROUP_TASK_EVALUATOR_ENVIRONMENT:
            return
        expected_spec = EnvironmentSpec(
            name=_FORK_GROUP_TASK_EVALUATOR_ENVIRONMENT,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="cayu:fork-group-task-evaluator-environment",
                behavior_version="1",
                implementation_version="1",
            ),
        )
        registered = self._environments.get(_FORK_GROUP_TASK_EVALUATOR_ENVIRONMENT)
        if registered is None:
            self.register_environment(Environment(expected_spec))
            return
        environment = registered.environment
        if (
            registered.spec != expected_spec
            or registered.factory is not None
            or environment.workspace is not None
            or environment.runner is not None
            or environment.vault is not None
            or environment.proxy is not None
            or environment.mcp_servers
            or environment.workspace_instructions is not None
        ):
            raise RuntimeError(
                "Fork-group task evaluator environment conflicts with runtime authority."
            )

    async def _preflight_fork_group_evaluator_agent(
        self,
        request: RunRequest,
        source_agent_name: str,
        request_sha256: str,
    ) -> tuple[str, str]:
        """Validate evaluator authority without registration or ordinary admission."""

        self._ensure_fork_group_task_evaluator_environment(request)
        evaluator_agent, _ = self._fork_group_evaluator_agent_state(
            source_agent_name=source_agent_name,
            synthetic_agent_name=request.agent_name,
            request_sha256=request_sha256,
        )
        prepared = await self._session_engine._prepare_initial_run(
            request,
            registered_agent_override=evaluator_agent,
            admit_session=False,
        )
        if prepared is None:
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        return request.agent_name, prepared.execution_profile.fingerprint

    async def _prepare_fork_group_evaluator_agent(
        self,
        request: RunRequest,
        source_agent_name: str,
        request_sha256: str,
        store_resolved_existing_session_id: str | None,
    ) -> tuple[str, str]:
        """Freeze the exact tool-free evaluator profile before durable admission."""

        self._ensure_fork_group_task_evaluator_environment(request)
        synthetic_name = self._register_fork_group_evaluator_agent(
            source_agent_name=source_agent_name,
            synthetic_agent_name=request.agent_name,
            request_sha256=request_sha256,
        )
        prepared = await self._session_engine._prepare_initial_run(
            request,
            store_resolved_existing_session_id=store_resolved_existing_session_id,
        )
        if prepared is None:
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        return synthetic_name, prepared.execution_profile.fingerprint

    def register_provider(
        self,
        provider: ModelProvider,
        *,
        default: bool = False,
        model_patterns: Iterable[str] | None = None,
    ) -> ModelProvider:
        if not isinstance(provider, ModelProvider):
            raise TypeError("Provider registration requires a ModelProvider.")
        if not isinstance(default, bool):
            raise TypeError("Provider default flag must be a bool.")
        stored_model_patterns = _validate_provider_model_patterns(model_patterns)
        provider_name = require_clean_nonblank(provider.name, "provider.name")
        usage_dialect = copy_usage_dialect(provider.usage_dialect, "provider.usage_dialect")
        if provider_name in self._providers:
            raise ValueError(f"Provider already registered: {provider_name}")

        registration_source, registration_symbol = _registration_site()
        self._providers[provider_name] = runtime_records.RegisteredProvider(
            name=provider_name,
            provider=provider,
            execution_profile_identity=(
                copy_secret_free_execution_profile_behavior_identity(
                    provider.execution_profile_identity,
                    redactor=self._secret_redactor,
                    field_name="provider.execution_profile_identity",
                )
            ),
            model_patterns=stored_model_patterns,
            registration_source=registration_source,
            registration_symbol=registration_symbol,
            usage_dialect=usage_dialect,
        )
        if default or self._default_provider_name is None:
            self._default_provider_name = provider_name
        return provider

    def register_environment(
        self,
        environment: Environment,
        *,
        default: bool = False,
    ) -> Environment:
        if not isinstance(environment, Environment):
            raise TypeError("Environment registration requires an Environment.")
        if not isinstance(default, bool):
            raise TypeError("Environment default flag must be a bool.")
        stored_environment = copy_environment(environment)
        stored_spec = _validate_environment_spec(
            stored_environment.spec,
            redactor=self._secret_redactor,
        )
        if stored_spec.name in self._environments:
            raise ValueError(f"Environment already registered: {stored_spec.name}")
        artifact_store = stored_environment.artifact_store
        artifact_store_registration = self._validate_artifact_store_registration(artifact_store)

        registration_source, registration_symbol = _registration_site()
        self._environments[stored_spec.name] = runtime_records.RegisteredEnvironment(
            spec=stored_spec,
            environment=stored_environment,
            runner_execution_profile_identity=copy_secret_free_execution_profile_behavior_identity(
                None
                if stored_environment.runner is None
                else stored_environment.runner.execution_profile_identity,
                redactor=self._secret_redactor,
                field_name="environment.runner.execution_profile_identity",
            ),
            registration_source=registration_source,
            registration_symbol=registration_symbol,
        )
        if artifact_store_registration is not None:
            self._artifact_store_registrations_by_id[artifact_store_registration.store_id] = (
                artifact_store_registration
            )
        self._select_default_environment_if_requested(stored_spec.name, default=default)
        return environment

    def register_environment_factory(
        self,
        spec: EnvironmentSpec,
        factory: EnvironmentFactory,
        *,
        artifact_store: ArtifactStore | None = None,
        default: bool = False,
    ) -> EnvironmentFactory:
        if not isinstance(spec, EnvironmentSpec):
            raise TypeError("Environment factory registration requires an EnvironmentSpec.")
        if not isinstance(factory, EnvironmentFactory):
            raise TypeError("Environment factory registration requires an EnvironmentFactory.")
        if not isinstance(default, bool):
            raise TypeError("Environment factory default flag must be a bool.")
        stored_spec = _validate_environment_spec(
            spec,
            redactor=self._secret_redactor,
        )
        if stored_spec.name in self._environments:
            raise ValueError(f"Environment already registered: {stored_spec.name}")
        stored_environment = Environment(stored_spec, artifact_store=artifact_store)
        artifact_store_registration = self._validate_artifact_store_registration(artifact_store)

        registration_source, registration_symbol = _registration_site()
        self._environments[stored_spec.name] = runtime_records.RegisteredEnvironment(
            spec=stored_spec,
            environment=stored_environment,
            factory=factory,
            factory_backed=True,
            factory_execution_profile_identity=copy_secret_free_execution_profile_behavior_identity(
                factory.execution_profile_identity,
                redactor=self._secret_redactor,
                field_name="environment_factory.execution_profile_identity",
            ),
            registration_source=registration_source,
            registration_symbol=registration_symbol,
        )
        if artifact_store_registration is not None:
            self._artifact_store_registrations_by_id[artifact_store_registration.store_id] = (
                artifact_store_registration
            )
        self._select_default_environment_if_requested(stored_spec.name, default=default)
        return factory

    def _validate_artifact_store_registration(
        self,
        artifact_store: ArtifactStore | None,
    ) -> _ArtifactStoreRegistration | None:
        if artifact_store is None:
            return None
        artifact_store_id = require_clean_nonblank(artifact_store.id, "artifact_store.id")
        artifact_store_id = require_unicode_scalar_text(
            artifact_store_id,
            "artifact_store.id",
        )
        registered = self._artifact_store_registrations_by_id.get(artifact_store_id)
        if registered is not None and registered.store is not artifact_store:
            raise ValueError(
                "Artifact store id already belongs to a different registered store: "
                f"{artifact_store_id}"
            )
        if registered is not None:
            return registered
        return _ArtifactStoreRegistration(
            store_id=artifact_store_id,
            store=artifact_store,
            fingerprint=f"sha256:{sha256(artifact_store_id.encode('utf-8')).hexdigest()}",
        )

    def _select_default_environment_if_requested(
        self,
        environment_name: str,
        *,
        default: bool,
    ) -> None:
        if default:
            self._default_environment_name = environment_name

    def get_agent(self, name: str) -> runtime_records.RegisteredAgent:
        agent_name = require_clean_nonblank(name, "agent.name")
        registered_agent = self._get_registered_agent(agent_name)
        return runtime_records.RegisteredAgent(
            spec=registered_agent.spec.model_copy(deep=True),
            tools={
                name: _copy_registered_tool(tool) for name, tool in registered_agent.tools.items()
            },
            hosted_tools=tuple(
                copy_openai_web_search(tool) for tool in registered_agent.hosted_tools
            ),
        )

    def resolve_run_model_target(self, request: RunRequest) -> ModelTarget:
        """Resolve initial provider/model routing without preparing runtime state."""

        return self._session_engine.resolve_initial_model_target(request)

    async def current_prompt_anatomy_sha256(
        self,
        *,
        agent_name: str,
        environment_name: str | None,
    ) -> str:
        """Return a content-free digest of the prompt the current body would install."""

        registered_agent = self._get_registered_agent(agent_name)
        registered_environment = (
            None if environment_name is None else self._get_registered_environment(environment_name)
        )
        if registered_environment is not None and registered_environment.factory is not None:
            raise RuntimeError(
                "Prompt anatomy cannot be inspected for a factory-backed environment before "
                "session materialization."
            )
        workspace_instructions = await self._environment_lifecycle.load_workspace_instructions(
            registered_environment
        )
        rendered = render_initial_system_prompt(
            agent_system_prompt=registered_agent.spec.system_prompt,
            workspace_instructions=workspace_instructions,
        )
        messages = [] if rendered is None else [Message.text("system", rendered)]
        return system_prompt_messages_sha256(messages)

    def list_agents(self) -> tuple[str, ...]:
        """Return the names of all registered agents, sorted."""
        return tuple(sorted(self._agents))

    def list_providers(self) -> tuple[str, ...]:
        """Return the names of all registered providers, sorted."""
        return tuple(sorted(self._providers))

    def list_environments(self) -> tuple[str, ...]:
        """Return the names of all registered environments (concrete or factory), sorted."""
        return tuple(sorted(self._environments))

    def has_registered_artifact_store(self) -> bool:
        """Return whether any registered environment exposes artifact storage.

        The registration paths maintain this value, so the check is constant-time
        and does not copy registration metadata or materialize environment factories.
        """

        return bool(self._artifact_store_registrations_by_id)

    def artifact_store_registration_fingerprints(
        self,
        *,
        limit: int,
    ) -> tuple[tuple[str, ...], int]:
        """Return a bounded snapshot of opaque store identities and the exact count.

        Fingerprints are fixed-size SHA-256 correlations of the store identities
        accepted at registration. They let protected diagnostics correlate shared
        registrations without returning a local path or application-defined id.
        """

        if type(limit) is not int:
            raise TypeError("Artifact store fingerprint limit must be an integer.")
        if limit < 1:
            raise ValueError("Artifact store fingerprint limit must be positive.")
        registrations = self._artifact_store_registrations_by_id
        fingerprints = tuple(
            registration.fingerprint for registration in islice(registrations.values(), limit)
        )
        return fingerprints, len(registrations)

    def list_environment_registrations(self) -> tuple[runtime_records.RegisteredEnvironment, ...]:
        """Return registered environment metadata without materializing factories."""
        registrations: list[runtime_records.RegisteredEnvironment] = []
        for name in sorted(self._environments):
            registered_environment = self._environments[name]
            registrations.append(
                runtime_records.RegisteredEnvironment(
                    spec=registered_environment.spec.model_copy(deep=True),
                    environment=copy_environment(registered_environment.environment),
                    runner_execution_profile_identity=(
                        copy_execution_profile_behavior_identity(
                            registered_environment.runner_execution_profile_identity
                        )
                    ),
                    factory_execution_profile_identity=(
                        copy_execution_profile_behavior_identity(
                            registered_environment.factory_execution_profile_identity
                        )
                    ),
                    factory=registered_environment.factory,
                    factory_backed=registered_environment.factory_backed,
                    bound_workspace=(
                        copy_bound_workspace(registered_environment.bound_workspace)
                        if registered_environment.bound_workspace is not None
                        else None
                    ),
                    binding_payload=copy_json_value(
                        registered_environment.binding_payload,
                        "binding_payload",
                    )
                    if registered_environment.binding_payload is not None
                    else None,
                    registration_source=registered_environment.registration_source,
                    registration_symbol=registered_environment.registration_symbol,
                    binding_generation_id=registered_environment.binding_generation_id,
                )
            )
        return tuple(registrations)

    def _get_registered_agent(self, name: str) -> runtime_records.RegisteredAgentState:
        agent_name = require_clean_nonblank(name, "agent.name")
        try:
            return self._agents[agent_name]
        except KeyError as exc:
            raise KeyError(f"Agent not registered: {agent_name}") from exc

    def get_provider(self, name: str | None = None) -> ModelProvider:
        return self._get_registered_provider(name).provider

    def get_environment(self, name: str | None = None) -> runtime_records.RegisteredEnvironment:
        registered_environment = self._get_registered_environment(name)
        if registered_environment is None:
            raise RuntimeError("No environment registered.")
        if registered_environment.factory is not None:
            raise RuntimeError(
                "Environment is factory-backed and is only concrete for a session: "
                f"{registered_environment.spec.name}"
            )
        return runtime_records.RegisteredEnvironment(
            spec=registered_environment.spec.model_copy(deep=True),
            environment=copy_environment(registered_environment.environment),
            runner_execution_profile_identity=copy_execution_profile_behavior_identity(
                registered_environment.runner_execution_profile_identity
            ),
            factory_execution_profile_identity=copy_execution_profile_behavior_identity(
                registered_environment.factory_execution_profile_identity
            ),
            binding_generation_id=registered_environment.binding_generation_id,
        )

    def get_environment_factory(self, name: str | None = None) -> EnvironmentFactory:
        registered_environment = self._get_registered_environment(name)
        if registered_environment is None:
            raise RuntimeError("No environment registered.")
        if registered_environment.factory is None:
            raise RuntimeError(
                f"Environment is not factory-backed: {registered_environment.spec.name}"
            )
        return registered_environment.factory

    async def attach_file(
        self,
        content: bytes,
        *,
        filename: str,
        kind: FileAttachmentKind | str,
        content_type: str | None = None,
        environment_name: str | None = None,
        scope: ArtifactScope = ArtifactScope.SESSION,
        session_id: str | None = None,
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FilePart:
        """Save a file to the artifact store and return a user-prompt `FilePart` referencing it.

        Attach the returned part to a user `Message` alongside text; the runtime inlines the file
        into the provider request on the turn it is attached (and re-enforces the per-file/per-request
        limits). `kind` is `"image"` (jpeg/png/gif/webp) or `"document"` (pdf). For a session-scoped
        attachment, pass the same `session_id` you will use in the `RunRequest`.

        The bytes are parsed to confirm they are a valid image/PDF whose detected format matches the
        declared/inferred content type before being stored, which requires the optional file
        dependencies (`cayu[files]`); without them this raises. The
        (default or named) environment registration must expose an artifact store. For a
        factory-backed environment, pass the durable store to
        `register_environment_factory(..., artifact_store=...)`; `attach_file` uses that stable
        handle without materializing a session environment.
        """
        if type(content) is not bytes:
            raise TypeError("attach_file content must be bytes.")
        if not content:
            raise ValueError("attach_file content cannot be empty.")
        if len(content) > self._max_file_attachment_bytes:
            raise ValueError(
                "File exceeds the prompt attachment byte limit: "
                f"{len(content)} > {self._max_file_attachment_bytes}"
            )
        resolved_kind = FileAttachmentKind(kind)
        if content_type is None:
            guessed_type, guessed_encoding = mimetypes.guess_type(filename)
            if guessed_encoding is not None:
                raise ValueError(
                    f"Cannot infer a content type for {filename!r} (encoding {guessed_encoding!r}); "
                    "pass content_type explicitly."
                )
            content_type = guessed_type
        if content_type is None:
            raise ValueError(
                f"Could not infer a content type for {filename!r}; pass content_type explicitly."
            )
        resolved_content_type = require_clean_nonblank(content_type, "content_type")
        validate_file_attachment_content_type(
            kind=resolved_kind,
            content_type=resolved_content_type,
        )
        await asyncio.to_thread(
            validate_file_attachment_bytes,
            kind=resolved_kind,
            content=content,
            content_type=resolved_content_type,
        )
        registered_environment = self._get_registered_environment(environment_name)
        artifact_store = _artifact_store(registered_environment)
        if artifact_store is None:
            raise RuntimeError(
                "attach_file requires an environment registration with an artifact store; "
                "pass artifact_store when registering a factory-backed environment."
            )
        artifact = await artifact_store.put_bytes(
            content,
            filename=filename,
            content_type=resolved_content_type,
            scope=scope,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=_environment_name(registered_environment),
            metadata=metadata,
        )
        return FilePart(
            attachment=file_attachment(
                artifact_id=artifact.id,
                kind=resolved_kind,
                filename=artifact.filename,
                content_type=artifact.content_type,
                size_bytes=artifact.size_bytes,
                metadata=artifact.metadata,
            )
        )

    def _provider_operation_completion_publisher(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        publication_context: ModelCompletionRecoveryContext,
    ) -> Callable[
        [ModelCompletionPublicationRequest],
        Awaitable[ModelCompletionPublicationResult],
    ]:
        async def publish(
            publication: ModelCompletionPublicationRequest,
        ) -> ModelCompletionPublicationResult:
            return await self._session_engine._publish_assistant_model_completion(
                publication,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                task_id=publication_context.task_id,
                request_metadata=publication_context.request_metadata,
                structured_output=publication_context.structured_output,
                thinking=publication_context.thinking,
                max_steps=publication_context.max_steps,
                limits=publication_context.limits,
                budget_limits=publication_context.budget_limits,
                retry_policy=publication_context.retry_policy,
                structured_output_attempt=(
                    publication_context.structured_output_attempt
                    if (
                        publication.assistant_step_result is not None
                        and _has_structured_output_tool_call(
                            publication.assistant_step_result.tool_calls
                        )
                    )
                    else None
                ),
                structured_output_retries=(
                    max(publication_context.structured_output_attempt - 1, 0)
                    if publication_context.structured_output_attempt is not None
                    else 0
                ),
                run_limit_accounting=publication_context.run_limit_accounting,
            )

        return publish

    async def _recover_provider_operation(
        self,
        session: Session,
        stage: ModelCompletionStage,
        operation: RecoverableProviderOperation,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> ProviderOperationRecoveryResult:
        recovery_context = model_completion_recovery_context_from_stage(stage)
        publication_context = recovery_context or ModelCompletionRecoveryContext()
        publish = self._provider_operation_completion_publisher(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            publication_context=publication_context,
        )

        return await self._model_step_executor.recover_provider_operation(
            session=session,
            stage=stage,
            operation=operation,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=_environment_name(registered_environment),
            recovery_context=recovery_context,
            model_completion_publisher=publish,
        )

    async def _recover_provider_operation_start(
        self,
        session: Session,
        stage: ModelCompletionStage,
        start: RecoverableProviderOperationStart,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> ProviderOperationRecoveryResult:
        recovery_context = model_completion_recovery_context_from_stage(stage)
        publication_context = recovery_context or ModelCompletionRecoveryContext()
        publish = self._provider_operation_completion_publisher(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            publication_context=publication_context,
        )

        return await self._model_step_executor.recover_provider_operation_start(
            session=session,
            stage=stage,
            start=start,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=_environment_name(registered_environment),
            model_completion_publisher=publish,
        )

    async def _cancel_provider_operation(
        self,
        session: Session,
        stage: ModelCompletionStage,
        operation: RecoverableProviderOperation,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> ProviderOperationSnapshot | None:
        return await self._model_step_executor.cancel_provider_operation_for_interruption(
            session=session,
            stage=stage,
            operation=operation,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=_environment_name(registered_environment),
        )

    def _get_registered_provider(
        self, name: str | None = None
    ) -> runtime_records.RegisteredProvider:
        if name is not None:
            provider_name = require_clean_nonblank(name, "provider.name")
        else:
            provider_name = self._default_provider_name
        if provider_name is None:
            raise RuntimeError("No model provider registered.")
        try:
            return self._providers[provider_name]
        except KeyError as exc:
            raise KeyError(f"Provider not registered: {provider_name}") from exc

    def _route_registered_provider_for_model(
        self,
        *,
        model: str,
    ) -> runtime_records.RegisteredProvider | None:
        model = require_clean_nonblank(model, "model")
        matches: list[runtime_records.RegisteredProvider] = []
        for registered_provider in self._providers.values():
            if any(fnmatchcase(model, pattern) for pattern in registered_provider.model_patterns):
                matches.append(registered_provider)
        if not matches:
            return None
        if len(matches) > 1:
            match_names = ", ".join(provider.name for provider in matches)
            raise ValueError(
                f"Model matches multiple registered providers: {model} -> {match_names}"
            )
        return matches[0]

    def _get_registered_environment(
        self,
        name: str | None = None,
    ) -> runtime_records.RegisteredEnvironment | None:
        if name is not None:
            environment_name = require_clean_nonblank(name, "environment.name")
        else:
            environment_name = self._default_environment_name
        if environment_name is None:
            return None
        try:
            return self._environments[environment_name]
        except KeyError as exc:
            raise KeyError(f"Environment not registered: {environment_name}") from exc

    def _get_registered_environment_for_session(
        self,
        name: str | None,
    ) -> runtime_records.RegisteredEnvironment | None:
        if name is None:
            return None
        return self._get_registered_environment(name)

    def _effective_retry_policy(self, request_policy: RetryPolicy | None) -> RetryPolicy:
        if request_policy is not None:
            return copy_retry_policy(request_policy)
        return copy_retry_policy(self._default_retry_policy)

    async def run(self, request: RunRequest) -> AsyncIterator[Event]:
        stream = self._run_private(request)
        del request
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield await self._project_emitted_event_for_public_api(event)

    async def _run_private(
        self,
        request: RunRequest,
        *,
        expected_execution_profile: ExecutionProfileIdentity | None = None,
        expected_registered_environment: runtime_records.RegisteredEnvironment | None = None,
        expected_context_policy: object | None = None,
    ) -> AsyncGenerator[Event, None]:
        if type(request) is not RunRequest:
            raise TypeError("Runtime run requires a RunRequest.")
        request = _validate_run_request(request)
        stream = self._session_engine.run(
            request=request,
            expected_execution_profile=expected_execution_profile,
            expected_registered_environment=expected_registered_environment,
            expected_context_policy=expected_context_policy,
        )
        del request
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def resume(self, request: ResumeRequest) -> AsyncIterator[Event]:
        if type(request) is not ResumeRequest:
            raise TypeError("Runtime resume requires a ResumeRequest.")
        request = copy_resume_request(request)
        session_id, store_resolved_session_id = await self._resolve_public_session_authority(
            request.session_id
        )
        request = request.model_copy(update={"session_id": session_id})
        stream = self._resume_private(
            request,
            store_resolved_session_id=store_resolved_session_id,
        )
        del request
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield await self._project_emitted_event_for_public_api(event)

    async def _resume_private(
        self,
        request: ResumeRequest,
        *,
        store_resolved_session_id: str | None = None,
    ) -> AsyncGenerator[Event, None]:
        if type(request) is not ResumeRequest:
            raise TypeError("Runtime resume requires a ResumeRequest.")
        request = _validate_resume_request(request)
        stream = self._session_engine.resume(
            request=request,
            store_resolved_session_id=store_resolved_session_id,
        )
        del request
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def compact_session(
        self,
        request: CompactSessionRequest,
    ) -> AsyncIterator[Event]:
        if type(request) is not CompactSessionRequest:
            raise TypeError("Runtime compaction requires a CompactSessionRequest.")
        session_id, store_resolved_session_id = await self._resolve_public_session_authority(
            request.session_id
        )
        request = request.model_copy(update={"session_id": session_id}, deep=True)
        stream = self._compact_session_private(
            request,
            store_resolved_session_id=store_resolved_session_id,
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield await self._project_emitted_event_for_public_api(event)

    async def _compact_session_private(
        self,
        request: CompactSessionRequest,
        *,
        store_resolved_session_id: str | None = None,
    ) -> AsyncGenerator[Event, None]:
        if type(request) is not CompactSessionRequest:
            raise TypeError("Runtime compaction requires a CompactSessionRequest.")
        stream = self._session_engine.compact_session(
            request=request,
            store_resolved_session_id=store_resolved_session_id,
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
    ) -> EnqueueSessionMessageResult:
        if type(request) is not EnqueueSessionMessageRequest:
            raise TypeError("Runtime queued input requires an EnqueueSessionMessageRequest.")
        session_id, store_resolved_session_id = await self._resolve_public_session_authority(
            request.session_id
        )
        request = request.model_copy(update={"session_id": session_id}, deep=True)
        result = await self._enqueue_session_message_private(
            request,
            store_resolved_session_id=store_resolved_session_id,
        )
        event = await self._project_emitted_event_for_public_api(result.event)
        return result.model_copy(
            update={
                "event": event,
                "message": result.message.model_copy(
                    update={"accepted_event_id": event.id},
                    deep=True,
                ),
            },
            deep=True,
        )

    async def _enqueue_session_message_private(
        self,
        request: EnqueueSessionMessageRequest,
        *,
        store_resolved_session_id: str | None = None,
    ) -> EnqueueSessionMessageResult:
        if type(request) is not EnqueueSessionMessageRequest:
            raise TypeError("Runtime queued input requires an EnqueueSessionMessageRequest.")
        return await self._session_engine.enqueue_session_message(
            request=request,
            store_resolved_session_id=store_resolved_session_id,
        )

    async def interrupt_session(self, request: InterruptSessionRequest) -> AsyncIterator[Event]:
        if type(request) is not InterruptSessionRequest:
            raise TypeError("Runtime interruption requires an InterruptSessionRequest.")
        session_id, store_resolved_session_id = await self._resolve_public_session_authority(
            request.session_id
        )
        request = request.model_copy(update={"session_id": session_id}, deep=True)
        stream = self._interrupt_session_private(
            request,
            store_resolved_session_id=store_resolved_session_id,
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield await self._project_emitted_event_for_public_api(event)

    async def _interrupt_session_private(
        self,
        request: InterruptSessionRequest,
        *,
        store_resolved_session_id: str | None = None,
    ) -> AsyncGenerator[Event, None]:
        if type(request) is not InterruptSessionRequest:
            raise TypeError("Runtime interruption requires an InterruptSessionRequest.")
        request = copy_interrupt_session_request(request)
        stream = self._session_engine.interrupt_session(
            request=request,
            store_resolved_session_id=store_resolved_session_id,
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def recover_incomplete_session(
        self,
        request: IncompleteSessionRecoveryRequest,
    ) -> IncompleteSessionRecoveryResult:
        if type(request) is not IncompleteSessionRecoveryRequest:
            raise TypeError(
                "Runtime incomplete-session recovery requires an IncompleteSessionRecoveryRequest."
            )
        request = request.model_copy(
            update={"session_id": await self._resolve_public_session_id(request.session_id)},
            deep=True,
        )
        recovery = self._recover_incomplete_session_private(request)
        del request
        result = await recovery
        return await self._project_incomplete_recovery_result_for_public_api(result)

    async def recover_model_completion_stage(
        self,
        request: ModelCompletionManualRecoveryRequest,
    ) -> ModelCompletionManualRecoveryResult:
        """Settle ambiguous provider work and linked budgets under runtime ownership."""

        if type(request) is not ModelCompletionManualRecoveryRequest:
            raise TypeError(
                "Runtime model-completion recovery requires a ModelCompletionManualRecoveryRequest."
            )
        session_id = await self._resolve_public_session_id(request.session_id)
        request = copy_model_completion_manual_recovery_request(
            request,
            session_id=session_id,
        )
        (
            requires_completion_decision,
            admission_failure,
        ) = await self._verifier_aware_recovery_execution_outcome(
            session_id=request.session_id,
            admit_session=False,
        )
        if admission_failure is not None:
            del request
            raise_task_store_operation_failure(admission_failure)
        if requires_completion_decision:
            del request
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        return await self._session_engine.recover_model_completion_stage(request)

    async def _recover_incomplete_session_private(
        self,
        request: IncompleteSessionRecoveryRequest,
    ) -> IncompleteSessionRecoveryResult:
        request = copy_incomplete_session_recovery_request(request)
        recovery = self._session_engine.recover_incomplete_session(request)
        del request
        return await recovery

    async def recover_persisted_event_side_effects(self, *, limit: int = 1000) -> list[Event]:
        """Retry committed event fan-out that was not acknowledged before a crash.

        Delivery is at-least-once and returns only events whose configured
        budget and sink side effects completed during this sweep. Failed and
        dead-lettered deliveries remain inspectable through ``session_store``.
        """
        return await self._event_writer.recover_persisted_side_effects(limit=limit)

    async def recover_incomplete_sessions(
        self,
        request: IncompleteSessionsRecoveryRequest,
    ) -> IncompleteSessionsRecoveryPage:
        """Sweep one bounded page of requested states, fault-isolated.

        ``results`` contains one result per repaired or otherwise reportable
        session. ``inspected_session_count`` includes healthy terminal rows
        omitted from those results. When ``next_cursor`` is present, pass it in
        a new request with the same statuses, inactivity boundary, reason, and
        metadata to continue without rescanning earlier candidates.

        A session whose agent is not
        registered in this process is reported as
        ``SKIPPED_UNREGISTERED_AGENT``; an unexpected per-session failure is
        reported as ``FAILED`` with the error in ``message`` — neither aborts
        the sweep, so one bad row cannot strand every healthy session. A
        ``FAILED`` entry's ``previous_status`` comes from the sweep's listing
        snapshot; its ``status`` is the current stored status when the session
        can still be reloaded (a failed recovery may have progressed it),
        falling back to the snapshot when it cannot. Session listing failures
        and cancellation still raise. Every invocation inspects at most
        ``request.inspection_limit`` rows, using at most ten store keyset pages
        of at most 1,000 rows each. Terminal sessions are inspected through
        bounded event queries. They are repaired only when their current run
        lacks matching terminal evidence or retains incomplete recovery state.
        Healthy terminal inspection candidates are omitted from the result and
        do not consume ``request.limit``.
        """
        page = await self._recover_incomplete_sessions_private(request)
        projected: list[IncompleteSessionRecoveryResult] = []
        for result in page.results:
            projected.append(await self._project_incomplete_recovery_result_for_public_api(result))
        return page.model_copy(update={"results": tuple(projected)}, deep=True)

    async def _recover_incomplete_sessions_private(
        self,
        request: IncompleteSessionsRecoveryRequest,
    ) -> IncompleteSessionsRecoveryPage:
        request = copy_incomplete_sessions_recovery_request(request)
        return await self._session_engine.recover_incomplete_sessions(request)

    async def dispatch(self, request: DispatchRequest) -> DispatchHandle:
        if type(request) is not DispatchRequest:
            raise TypeError("Runtime dispatch requires a DispatchRequest.")
        request = copy_dispatch_request(request)
        if request.task_id is not None and self.task_store is None:
            raise RuntimeError("task_store is required when DispatchRequest.task_id is set.")
        # Resolve at the public boundary to reject malformed or unknown aliases. Keep
        # the public request value across dispatcher boundaries so durable queues do
        # not persist private session authority; dispatch_inline resolves it again in
        # the worker that owns execution.
        private_session_id, _ = await self._resolve_public_session_authority(request.session_id)
        (
            contract_rejected,
            admission_failure,
        ) = await self._session_engine._verifier_aware_task_execution_outcome(
            request.task_id,
            session_id=private_session_id,
        )
        if admission_failure is not None:
            del private_session_id, request
            raise_task_store_operation_failure(admission_failure)
        if contract_rejected:
            del private_session_id, request
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        handle = await self.dispatcher.submit(self, request)
        _validate_dispatch_handle_for_request(handle=handle, request=request)
        copied = copy_dispatch_handle(handle)
        return copied.model_copy(
            update={
                "session_id": self.project_session_id_for_exposure(private_session_id),
            },
            deep=True,
        )

    async def session_invocation_for_dispatch(
        self,
        session_id: str,
    ) -> SessionInvocationBinding:
        """Return immutable private provenance for a trusted dispatcher boundary."""

        private_session_id, _ = await self._resolve_public_session_authority(session_id)
        snapshot = await self.session_store.load_invocation_snapshot(private_session_id)
        if snapshot is None:
            raise KeyError(f"Session not found: {private_session_id}")
        return copy_session_invocation_binding(snapshot)

    async def _prepare_durable_subagent_run(
        self,
        request: RunRequest,
    ) -> DurableSubagentPreparedRun:
        preparation = self._session_engine._prepare_initial_run(request)
        del request
        prepared = await preparation
        del preparation
        if prepared is None:
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        try:
            return DurableSubagentPreparedRun(
                request=prepared.request,
                provider_name=prepared.registered_provider.name,
                model=prepared.session_identity.model,
                execution_profile=prepared.execution_profile,
            )
        finally:
            del prepared

    async def _submit_durable_subagent(
        self,
        *,
        context: ToolContext,
        request: RunRequest,
        agent_alias: str,
        tool_name: str,
        spawn_fingerprint: str,
        effective_arguments: dict[str, Any],
    ) -> DispatchHandle:
        return await self._durable_subagent_coordinator.submit(
            context=context,
            request=request,
            agent_alias=agent_alias,
            tool_name=tool_name,
            spawn_fingerprint=spawn_fingerprint,
            effective_arguments=effective_arguments,
        )

    async def _ensure_durable_subagent_submission(
        self,
        intent: DurableSubagentSubmissionIntent,
    ) -> tuple[Session, DispatchHandle]:
        return await self._durable_subagent_coordinator.ensure_submission(intent)

    async def _reconcile_durable_subagent(
        self,
        *,
        parent_session: Session,
        tool_name: str,
        tool_round_id: str,
        tool_call_id: str,
        idempotency_key: str,
        effective_arguments: dict[str, Any],
    ) -> Session | ToolResult | None:
        return await self._durable_subagent_coordinator.reconcile(
            parent_session=parent_session,
            tool_name=tool_name,
            tool_round_id=tool_round_id,
            tool_call_id=tool_call_id,
            idempotency_key=idempotency_key,
            effective_arguments=effective_arguments,
        )

    async def dispatch_inline(self, request: DispatchRequest) -> AsyncIterator[Event]:
        if type(request) is not DispatchRequest:
            raise TypeError("Inline dispatch requires a DispatchRequest.")
        session_id, store_resolved_session_id = await self._resolve_public_session_authority(
            request.session_id
        )
        request = request.model_copy(update={"session_id": session_id}, deep=True)
        stream = self._dispatch_inline_private(
            request,
            store_resolved_session_id=store_resolved_session_id,
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield await self._project_emitted_event_for_public_api(event)

    async def _load_queued_dispatch_session_snapshot(
        self,
        session_id: str,
    ) -> tuple[Session, dict[str, Any] | None]:
        """Load session and checkpoint authority under one store-owned boundary."""

        snapshot: tuple[Session, dict[str, Any] | None] | None = None

        def capture_snapshot(
            session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> None:
            nonlocal snapshot
            snapshot = (
                session.model_copy(deep=True),
                None if checkpoint is None else deepcopy(checkpoint),
            )
            # ``None`` is the documented read-only result for this atomic
            # checkpoint-transform boundary.
            return None

        await self.session_store.transform_checkpoint(session_id, capture_snapshot)
        if snapshot is None:
            raise RuntimeError("Queued dispatch session snapshot was not produced.")
        return snapshot

    async def _load_queued_dispatch_terminal_event(
        self,
        *,
        private_session_id: str,
        envelope: _QueuedDispatchEnvelope,
    ) -> Event | None:
        """Load and validate the exact terminal event bound to an envelope."""

        records = await self.session_store.query_events(
            EventQuery(
                session_id=private_session_id,
                event_id=envelope.terminal_event_id,
                limit=2,
            )
        )
        if not records:
            return None
        if len(records) != 1:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal evidence is duplicated."
            )
        terminal_event = records[0].event
        if (
            terminal_event.type not in TERMINAL_EVENT_TYPES
            or terminal_event.payload.get(SESSION_RUN_OPERATION_ID_PAYLOAD_KEY)
            != envelope.dispatch_operation_id
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal evidence conflicts with its envelope."
            )
        return terminal_event

    async def _prepare_queued_dispatch(
        self,
        request: DispatchRequest,
        *,
        queue_task_id: str,
    ) -> _QueuedDispatchEnvelope:
        """Freeze runtime-owned profile authority before a queue task is published."""

        if type(request) is not DispatchRequest:
            raise TypeError("Queued dispatch preparation requires a DispatchRequest.")
        request = copy_dispatch_request(request)
        private_session_id, _ = await self._resolve_public_session_authority(request.session_id)
        (
            contract_rejected,
            admission_failure,
        ) = await self._session_engine._verifier_aware_task_execution_outcome(
            request.task_id,
            session_id=private_session_id,
            admit_session=False,
        )
        if admission_failure is not None:
            del private_session_id, request
            raise_task_store_operation_failure(admission_failure)
        if contract_rejected:
            del private_session_id, request
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        session, checkpoint = await self._load_queued_dispatch_session_snapshot(private_session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        pending_tool_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        pending_model_completion = await self._recovery_coordinator.load_model_completion_boundary(
            session
        )
        continues_active_invocation = (
            pending_tool_round is not None or pending_model_completion is not None
        )
        if active_profile is not None:
            if not active_invocation_execution_profile_matches_session_epoch(
                active_profile,
                session_id=session.id,
                run_epoch=session.run_epoch,
            ):
                raise RuntimeError(
                    "Active invocation execution profile conflicts with the session epoch."
                )
            if (
                active_invocation_execution_profile_is_released(
                    active_profile,
                    session_id=session.id,
                    run_epoch=session.run_epoch,
                )
                and not continues_active_invocation
            ):
                source_profile = execution_profile_from_session_metadata(session.metadata)
            else:
                source_profile = active_profile.profile
        else:
            if continues_active_invocation:
                raise RuntimeError(
                    "Queued dispatch recovery has no durable active invocation execution profile."
                )
            if session.status in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
                raise RuntimeError(
                    "A live session has no durable active invocation execution profile."
                )
            source_profile = execution_profile_from_session_metadata(session.metadata)
        durable_request = self.redact_dispatch_request(request)
        target_changed = durable_request.target is not None and (
            durable_request.target.provider_name != session.provider_name
            or durable_request.target.model != session.model
        )
        if target_changed:
            if continues_active_invocation:
                raise RuntimeError(
                    "A queued dispatch model target cannot change while model or tool "
                    "recovery is pending."
                )
            source_profile = execution_profile_from_session_metadata(session.metadata)
        required_profile = source_profile
        if not continues_active_invocation:
            required_profile = self._session_engine._queued_dispatch_required_profile(
                session=session,
                source_profile=source_profile,
                request=durable_request,
            )
        unavailable = set(unavailable_execution_profile_components(source_profile))
        unavailable.update(unavailable_execution_profile_components(required_profile))
        if unavailable:
            raise RuntimeError(
                "Queued dispatch requires an execution profile with available components: "
                + ", ".join(
                    component.value
                    for component in sorted(unavailable, key=lambda item: item.value)
                )
            )
        if (
            durable_request.structured_output is not None
            and durable_request.structured_output.strategy is StructuredOutputStrategy.NATIVE
        ):
            registered_provider = self._get_registered_provider(
                durable_request.target.provider_name
                if durable_request.target is not None
                else session.provider_name
            )
            _require_native_structured_output_support(
                durable_request.structured_output,
                registered_provider=registered_provider,
            )
        (
            contract_rejected,
            admission_failure,
        ) = await self._session_engine._verifier_aware_task_execution_outcome(
            request.task_id,
            session_id=private_session_id,
        )
        if admission_failure is not None:
            del durable_request, private_session_id, request
            raise_task_store_operation_failure(admission_failure)
        if contract_rejected:
            del durable_request, private_session_id, request
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        return _new_queued_dispatch_envelope(
            queue_task_id=queue_task_id,
            request=durable_request,
            session_instance_fingerprint=(_queued_dispatch_session_instance_fingerprint(session)),
            source_profile=source_profile,
            required_profile=required_profile,
        )

    async def _prepare_fork_group_queued_dispatch(
        self,
        request: DispatchRequest,
        queue_task_id: str,
    ) -> _QueuedDispatchEnvelope:
        """Bind group linkage before the corresponding queue task is claimable."""

        return await self._prepare_queued_dispatch(
            request,
            queue_task_id=queue_task_id,
        )

    async def _queued_dispatch_requests_match(
        self,
        existing: DispatchRequest,
        candidate: DispatchRequest,
    ) -> bool:
        """Compare retries by private session authority, not rotating public aliases."""

        existing = copy_dispatch_request(existing)
        candidate = copy_dispatch_request(candidate)
        existing_session_id, _ = await self._resolve_public_session_authority(existing.session_id)
        candidate_session_id, _ = await self._resolve_public_session_authority(candidate.session_id)
        if existing_session_id != candidate_session_id:
            return False
        comparison_session_id = "cayu-equivalent-session-authority"
        return existing.model_copy(
            update={"session_id": comparison_session_id},
            deep=True,
        ) == candidate.model_copy(
            update={"session_id": comparison_session_id},
            deep=True,
        )

    async def _acknowledge_queued_dispatch(
        self,
        envelope: _QueuedDispatchEnvelope,
        *,
        dispatch_status: DispatchStatus,
        receipt: QueuedDispatchTerminalReceipt | None = None,
    ) -> None:
        """Release exact terminal retention after the queue outcome is durable."""

        envelope = _copy_queued_dispatch_envelope(envelope)
        if type(dispatch_status) is not DispatchStatus:
            raise TypeError("Queued dispatch acknowledgement status has an invalid type.")
        settlement = await self._queued_dispatch_settlement_state(envelope)
        if settlement.state is _QueuedDispatchSettlementState.NOT_ADMITTED:
            return
        if settlement.state is not _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE:
            raise RuntimeError(
                "Queued dispatch terminal evidence is not durable enough to acknowledge."
            )
        if settlement.terminal_status is not dispatch_status:
            raise RuntimeError(
                "Queued dispatch task status conflicts with its exact terminal event."
            )
        private_session_id, _ = await self._resolve_public_session_authority(
            envelope.request.session_id
        )
        if receipt is not None:
            if type(receipt) is not QueuedDispatchTerminalReceipt:
                raise TypeError("Queued dispatch acknowledgement receipt has an invalid type.")
            receipt = QueuedDispatchTerminalReceipt(
                session_id=receipt.session_id,
                queue_task_id=receipt.queue_task_id,
                operation_id=receipt.operation_id,
                terminal_event_id=receipt.terminal_event_id,
            )
            if (
                receipt.session_id != private_session_id
                or receipt.queue_task_id != envelope.queue_task_id
                or receipt.operation_id != envelope.dispatch_operation_id
                or receipt.terminal_event_id != envelope.terminal_event_id
            ):
                raise RuntimeError(
                    "Queued dispatch acknowledgement receipt conflicts with its envelope."
                )

        def acknowledge(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            return _checkpoint_after_queued_dispatch_acknowledgement(
                checkpoint,
                queue_task_id=envelope.queue_task_id,
                operation_id=envelope.dispatch_operation_id,
                terminal_event_id=envelope.terminal_event_id,
            )

        await self.session_store.transform_checkpoint(private_session_id, acknowledge)

    async def _list_queued_dispatch_terminal_receipts(
        self,
        query: QueuedDispatchTerminalReceiptQuery,
    ) -> list[QueuedDispatchTerminalReceipt]:
        """Delegate bounded restart discovery to the durable session store."""

        return await self.session_store.list_queued_dispatch_terminal_receipts(query)

    @staticmethod
    def _queued_dispatch_terminal_ownership_released(
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        envelope: _QueuedDispatchEnvelope,
    ) -> bool:
        """Validate and classify the exact invocation generation behind a terminal event."""

        try:
            run_operation = _session_run_operation_from_checkpoint(checkpoint)
            receipt = _queued_dispatch_terminal_receipts_from_checkpoint(checkpoint).get(
                envelope.dispatch_operation_id
            )
        except (TypeError, ValueError) as exc:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal ownership evidence is malformed."
            ) from exc
        if receipt is not None and (
            receipt.queue_task_id != envelope.queue_task_id
            or receipt.terminal_event_id != envelope.terminal_event_id
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal receipt identity conflicts."
            )
        terminal_run_epoch = None if receipt is None else receipt.run_epoch
        if run_operation is not None and run_operation.operation_id == (
            envelope.dispatch_operation_id
        ):
            if (
                run_operation.queue_task_id != envelope.queue_task_id
                or run_operation.terminal_event_id != envelope.terminal_event_id
            ):
                raise _QueuedDispatchAuthorityRejected(
                    "Queued dispatch run operation identity conflicts."
                )
            if terminal_run_epoch not in {None, run_operation.run_epoch}:
                raise _QueuedDispatchAuthorityRejected(
                    "Queued dispatch terminal ownership epoch conflicts."
                )
            terminal_run_epoch = run_operation.run_epoch

        # The exact terminal event is supplied by the caller of this helper. If
        # neither live handoff representation remains, acknowledgement already
        # completed: terminal publication first leaves either the run marker or
        # its receipt in the checkpoint, and only an exact durable queue outcome
        # removes the last one. A later invocation's active profile must not
        # become ownership evidence for that already-settled operation.
        if terminal_run_epoch is None:
            return True

        try:
            active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        except (TypeError, ValueError) as exc:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal invocation profile is malformed."
            ) from exc
        if active_profile is None:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal evidence has no durable invocation profile."
            )
        if not active_invocation_execution_profile_matches_session_epoch(
            active_profile,
            session_id=session.id,
            run_epoch=session.run_epoch,
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal ownership conflicts with the session epoch."
            )
        if session.run_epoch < terminal_run_epoch:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal ownership belongs to a future session epoch."
            )
        if active_profile.run_epoch < terminal_run_epoch:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal profile predates its ownership epoch."
            )
        if (
            active_profile.run_epoch == terminal_run_epoch
            and active_profile.profile != envelope.required_profile
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal profile conflicts with its envelope."
            )
        return session.run_epoch > terminal_run_epoch

    async def _require_queued_fork_group_authority(
        self,
        envelope: _QueuedDispatchEnvelope,
    ) -> bool:
        """Translate a durable group contradiction into permanent queue rejection."""

        from cayu.runtime.fork_groups import ForkGroupConflict

        try:
            return await self._fork_group_coordinator.require_dispatch_authority(envelope)
        except ForkGroupConflict as exc:
            raise _QueuedDispatchAuthorityRejected(
                "Queued fork-group dispatch authority was rejected."
            ) from exc

    async def _queued_dispatch_settlement_state(
        self,
        envelope: _QueuedDispatchEnvelope,
    ) -> _QueuedDispatchSettlement:
        """Classify the exact event/ownership evidence for one queued operation."""

        envelope = _copy_queued_dispatch_envelope(envelope)
        if envelope.operation_kind == "prepared_subagent":
            await self._durable_subagent_coordinator.require_prepared_subagent_parent_authority(
                envelope
            )
        await self._require_queued_fork_group_authority(envelope)
        private_session_id, _ = await self._resolve_public_session_authority(
            envelope.request.session_id
        )
        terminal_event = await self._load_queued_dispatch_terminal_event(
            private_session_id=private_session_id,
            envelope=envelope,
        )
        terminal_event_durable = terminal_event is not None

        try:
            session, checkpoint = await self._load_queued_dispatch_session_snapshot(
                private_session_id
            )
        except KeyError as exc:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch target session no longer exists."
            ) from exc
        if (
            _queued_dispatch_session_instance_fingerprint(session)
            != envelope.session_instance_fingerprint
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch target session instance changed."
            )
        try:
            run_operation = _session_run_operation_from_checkpoint(checkpoint)
            receipts = _queued_dispatch_terminal_receipts_from_checkpoint(checkpoint)
        except (TypeError, ValueError) as exc:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal ownership evidence is malformed."
            ) from exc
        receipt = receipts.get(envelope.dispatch_operation_id)
        if receipt is not None and (
            receipt.queue_task_id != envelope.queue_task_id
            or receipt.terminal_event_id != envelope.terminal_event_id
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal receipt identity conflicts."
            )
        terminal_run_epoch = None if receipt is None else receipt.run_epoch
        if run_operation is not None and run_operation.operation_id == (
            envelope.dispatch_operation_id
        ):
            if (
                run_operation.queue_task_id != envelope.queue_task_id
                or run_operation.terminal_event_id != envelope.terminal_event_id
            ):
                raise _QueuedDispatchAuthorityRejected(
                    "Queued dispatch run operation identity conflicts."
                )
            if terminal_run_epoch not in {None, run_operation.run_epoch}:
                raise _QueuedDispatchAuthorityRejected(
                    "Queued dispatch terminal ownership epoch conflicts."
                )
            terminal_run_epoch = run_operation.run_epoch
            if not terminal_event_durable:
                return _QueuedDispatchSettlement(
                    _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_PENDING
                )
        if receipt is not None and not terminal_event_durable:
            # Publication and receipt transfer can commit after the first event
            # query. The exact receipt pins the event, so one read after observing
            # that receipt closes the cross-store classification race.
            terminal_event = await self._load_queued_dispatch_terminal_event(
                private_session_id=private_session_id,
                envelope=envelope,
            )
            if terminal_event is None:
                raise _QueuedDispatchAuthorityRejected(
                    "Queued dispatch receipt has no exact durable terminal event."
                )
            terminal_event_durable = True
        if terminal_event_durable:
            assert terminal_event is not None
            if not self._queued_dispatch_terminal_ownership_released(
                session=session,
                checkpoint=checkpoint,
                envelope=envelope,
            ):
                return _QueuedDispatchSettlement(
                    _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_PENDING
                )
            terminal_status_by_type = {
                str(EventType.SESSION_COMPLETED): DispatchStatus.COMPLETED,
                str(EventType.SESSION_FAILED): DispatchStatus.FAILED,
                str(EventType.SESSION_INTERRUPTED): DispatchStatus.INTERRUPTED,
            }
            try:
                terminal_status = terminal_status_by_type[terminal_event.type]
            except KeyError:
                raise _QueuedDispatchAuthorityRejected(
                    "Queued dispatch terminal event has no dispatch status mapping."
                ) from None
            return _QueuedDispatchSettlement(
                _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE,
                terminal_status=terminal_status,
            )
        return _QueuedDispatchSettlement(_QueuedDispatchSettlementState.NOT_ADMITTED)

    async def _dispatch_queued(
        self,
        envelope: _QueuedDispatchEnvelope,
    ) -> AsyncIterator[Event]:
        """Run or replay one queue-owned dispatch under its frozen profile."""

        envelope = _copy_queued_dispatch_envelope(envelope)
        request = envelope.request
        fork_group_terminal = await self._require_queued_fork_group_authority(envelope)
        (
            private_session_id,
            store_resolved_session_id,
        ) = await self._resolve_public_session_authority(request.session_id)
        try:
            session, checkpoint = await self._load_queued_dispatch_session_snapshot(
                private_session_id
            )
        except KeyError as exc:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch target session no longer exists."
            ) from exc
        if (
            _queued_dispatch_session_instance_fingerprint(session)
            != envelope.session_instance_fingerprint
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch target session instance changed."
            )
        replay_event = await self._load_queued_dispatch_terminal_event(
            private_session_id=private_session_id,
            envelope=envelope,
        )
        if replay_event is not None:
            if not self._queued_dispatch_terminal_ownership_released(
                session=session,
                checkpoint=checkpoint,
                envelope=envelope,
            ):
                raise SessionRunFenced(
                    "Queued dispatch terminal hooks or trailing cleanup still own the "
                    "session run fence."
                )
            yield await self._project_emitted_event_for_public_api(replay_event)
            return
        if fork_group_terminal:
            raise _QueuedDispatchAuthorityRejected(
                "Fork-group terminalization revoked this unfinished dispatch attempt."
            )

        try:
            self._get_registered_agent(session.agent_name)
            self._get_registered_provider(
                request.target.provider_name
                if request.target is not None
                else session.provider_name
            )
            self._get_registered_environment_for_session(session.environment_name)
        except KeyError as exc:
            if envelope.operation_kind == "prepared_subagent":
                raise durable_subagent_worker_incompatible() from exc
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch required runtime component is unavailable."
            ) from exc

        if envelope.operation_kind == "prepared_subagent":
            run_request = self._durable_subagent_coordinator.prepare_queued_child_run(
                envelope=envelope,
                session=session,
                checkpoint=checkpoint,
            )
            stream = self._run_private(run_request)
            async with _close_delegated_event_stream(stream) as owned_stream:
                async for event in owned_stream:
                    yield await self._project_emitted_event_for_public_api(event)
            return

        private_request = request.model_copy(
            update={"session_id": private_session_id},
            deep=True,
        )
        stream = self._dispatch_inline_private(
            private_request,
            store_resolved_session_id=store_resolved_session_id,
            source_execution_profile=envelope.source_profile,
            required_execution_profile=envelope.required_profile,
            required_session_instance_fingerprint=(envelope.session_instance_fingerprint),
            dispatch_operation_id=envelope.dispatch_operation_id,
            dispatch_terminal_event_id=envelope.terminal_event_id,
            queue_task_id=envelope.queue_task_id,
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield await self._project_emitted_event_for_public_api(event)

    async def _dispatch_inline_private(
        self,
        request: DispatchRequest,
        *,
        store_resolved_session_id: str | None = None,
        source_execution_profile: ExecutionProfileIdentity | None = None,
        required_execution_profile: ExecutionProfileIdentity | None = None,
        required_session_instance_fingerprint: str | None = None,
        dispatch_operation_id: str | None = None,
        dispatch_terminal_event_id: str | None = None,
        queue_task_id: str | None = None,
    ) -> AsyncGenerator[Event, None]:
        if type(request) is not DispatchRequest:
            raise TypeError("Inline dispatch requires a DispatchRequest.")
        request = copy_dispatch_request(request)
        if request.task_id is not None and self.task_store is None:
            raise RuntimeError("task_store is required when DispatchRequest.task_id is set.")
        resume_request = ResumeRequest(
            session_id=request.session_id,
            messages=request.messages,
            target=request.target,
            metadata=request.metadata,
            max_steps=request.max_steps,
            limits=request.limits,
            budget_limits=request.budget_limits,
            retry_policy=request.retry_policy,
            structured_output=request.structured_output,
            thinking=request.thinking,
            loop_policies=request.loop_policies,
        )
        resume_request = session_request_boundary.prepare_resume_request(
            resume_request,
            redactor=self._secret_redactor,
            store_resolved_session_id=store_resolved_session_id,
        )
        start_event_payload_extra: dict[str, Any] = {"dispatch_id": request.dispatch_id}
        if request.task_id is not None:
            start_event_payload_extra["task_id"] = request.task_id
        if dispatch_operation_id is not None:
            start_event_payload_extra.update(
                {
                    "dispatch_operation_id": dispatch_operation_id,
                    "queue_task_id": queue_task_id,
                    "source_execution_profile_fingerprint": (
                        None
                        if source_execution_profile is None
                        else source_execution_profile.fingerprint
                    ),
                    "required_execution_profile_fingerprint": (
                        None
                        if required_execution_profile is None
                        else required_execution_profile.fingerprint
                    ),
                }
            )
        session_stream = self._session_engine._resume_session(
            request=resume_request,
            task_id=request.task_id,
            start_event_payload_extra=start_event_payload_extra,
            start_task_on_enter=True,
            source_execution_profile=source_execution_profile,
            required_execution_profile=required_execution_profile,
            required_session_instance_fingerprint=(required_session_instance_fingerprint),
            run_operation_id=dispatch_operation_id,
            terminal_event_id=dispatch_terminal_event_id,
            queue_task_id=queue_task_id,
        )
        async with _close_delegated_event_stream(session_stream) as owned_stream:
            forwarded_stream = self._session_control.stream_with_out_of_band_events(
                request.session_id,
                owned_stream,
            )
            async with _close_delegated_event_stream(forwarded_stream) as owned_forwarded_stream:
                async for event in owned_forwarded_stream:
                    yield event

    async def verify_completion_proposal(
        self,
        request: CompletionVerifierExecutionRequest,
    ) -> CompletionDecision:
        """Run the registered deterministic verifier and persist its decision."""

        operation = self._completion_verifier_coordinator.verify(request)
        del request
        return await operation

    async def apply_completion_decision(
        self,
        request: CompletionDecisionApplicationRequest,
    ) -> Task:
        """Apply or exactly replay one durable verifier decision."""

        operation = self._completion_decision_application_coordinator.apply(request)
        del request
        return await operation

    async def resolve_completion_result(
        self,
        request: CompletionResultResolutionRequest,
    ) -> Task:
        """Resolve and exactly apply the accepted result for one durable decision."""

        operation = self._completion_result_resolver_coordinator.resolve(request)
        del request
        return await operation

    async def create_work_contract(self, request: WorkContractDraft) -> WorkContract:
        if type(request) is not WorkContractDraft:
            del request
            raise TypeError("Work-contract creation requires a WorkContractDraft request.")
        if self.task_store is None:
            del request
            raise RuntimeError("task_store is required to create work contracts.")
        if not self.task_store.supports_verified_work_contracts:
            del request
            raise NotImplementedError(
                f"{type(self.task_store).__name__} does not support verified work contracts."
            )
        validation = _validated_public_work_contract(
            request,
            redactor=self._secret_redactor,
        )
        del request
        validation_failure = validation.failure
        contract = validation.result
        del validation
        if validation_failure is not None:
            raise validation_failure from None
        if contract is None:
            raise ValueError("Work-contract creation request is invalid.") from None
        contains_secret_identity = _work_contract_contains_secret_public_identity(
            contract,
            self._secret_redactor,
        )
        if contains_secret_identity:
            del contract
            raise ValueError(
                "Work-contract public identity contains a workload secret and cannot be published."
            ) from None
        published_value, publication_failure = await _publish_public_work_contract(
            self.task_store,
            contract,
            redactor=self._secret_redactor,
        )
        if publication_failure is not None:
            del contract, published_value
            raise_task_store_operation_failure(publication_failure)
        validation = _copied_public_work_contract(
            published_value,
            redactor=self._secret_redactor,
        )
        del published_value
        validation_failure = validation.failure
        published = validation.result
        del validation
        if validation_failure is not None:
            del contract
            raise validation_failure from None
        if published is None:
            del contract
            raise WorkContractConflict(
                "Task store returned an invalid published work contract."
            ) from None
        if published != contract:
            del contract, published
            raise WorkContractConflict(
                "Task store returned a work contract other than the exact published definition."
            ) from None
        return published

    async def load_work_contract(self, reference: WorkContractRef) -> WorkContract | None:
        if type(reference) is not WorkContractRef:
            del reference
            raise TypeError("Work-contract lookup requires a WorkContractRef.")
        if self.task_store is None:
            del reference
            raise RuntimeError("task_store is required to load work contracts.")
        if not self.task_store.supports_verified_work_contracts:
            del reference
            raise NotImplementedError(
                f"{type(self.task_store).__name__} does not support verified work contracts."
            )
        validation = _copied_public_work_contract_ref(
            reference,
            redactor=self._secret_redactor,
        )
        del reference
        validation_failure = validation.failure
        copied_reference = validation.result
        del validation
        if validation_failure is not None:
            raise validation_failure from None
        if copied_reference is None:
            raise ValueError("Work-contract lookup reference is invalid.") from None
        contains_secret_identity = (
            self._secret_redactor.redact_text(copied_reference.contract_id)
            != copied_reference.contract_id
        )
        if contains_secret_identity:
            del copied_reference
            raise ValueError(
                "Work-contract identity contains a workload secret and cannot be used for lookup."
            ) from None
        loaded_value, lookup_failure = await _load_public_work_contract(
            self.task_store,
            copied_reference,
            redactor=self._secret_redactor,
        )
        if lookup_failure is not None:
            del copied_reference, loaded_value
            raise_task_store_operation_failure(lookup_failure)
        if loaded_value is None:
            return None
        validation = _copied_public_work_contract(
            loaded_value,
            redactor=self._secret_redactor,
        )
        del loaded_value
        validation_failure = validation.failure
        loaded = validation.result
        del validation
        if validation_failure is not None:
            del copied_reference
            raise validation_failure from None
        if loaded is None:
            del copied_reference
            raise WorkContractConflict("Task store returned an invalid work contract.") from None
        if loaded.reference() != copied_reference:
            del copied_reference, loaded
            raise WorkContractConflict(
                "Task store returned a work contract other than the exact requested version."
            ) from None
        contains_secret_identity = _work_contract_contains_secret_public_identity(
            loaded,
            self._secret_redactor,
        )
        if contains_secret_identity:
            del copied_reference, loaded
            raise ValueError(
                "Loaded work contract contains a workload secret in a public identity."
            ) from None
        return loaded

    async def create_task(self, request: TaskCreate) -> Task:
        if type(request) is not TaskCreate:
            del request
            raise TypeError("Task creation requires a TaskCreate request.")
        if request.work_contract is None:
            # Preserve the established ordinary-task validation contract,
            # including actionable Pydantic field diagnostics. Contract-bound
            # requests use the detached boundary below because their durable
            # authority fields may contain workload-sensitive material.
            request = copy_task_create(request)
        else:
            validation = _copied_public_task_create(
                request,
                redactor=self._secret_redactor,
            )
            del request
            validation_failure = validation.failure
            copied_request = validation.result
            del validation
            if validation_failure is not None:
                raise validation_failure from None
            if copied_request is None:
                raise ValueError("Task creation request is invalid.") from None
            request = copied_request
            del copied_request
        if self.task_store is None:
            del request
            raise RuntimeError("task_store is required to create tasks.")
        if (
            request.retry_policy is not None
            and self._secret_redactor.redact_uppercase_text(request.retry_policy.cost_currency)
            != request.retry_policy.cost_currency
        ):
            raise ValueError(
                "Task retry cost currency contains a workload secret and cannot be used "
                "as durable accounting authority."
            ) from None
        for origin in (request.invocation_origin, request._verified_invocation_origin):
            if origin is None:
                continue
            for value in (origin.subject, origin.tenant):
                if value is not None and self._secret_redactor.redact_text(value) != value:
                    del origin, request, value
                    raise ValueError(
                        "Task invocation origin contains a workload secret and cannot be "
                        "used as durable task authority."
                    ) from None
        if request.work_contract is not None:
            if not self.task_store.supports_verified_work_contracts:
                del request
                raise NotImplementedError(
                    f"{type(self.task_store).__name__} does not support verified work contracts."
                )
            if request.task_id is None:
                del request
                raise ValueError(
                    "Contracted task creation requires a caller-stable task_id for "
                    "cancellation reconciliation."
                ) from None
            if self._secret_redactor.redact_text(request.task_id) != request.task_id:
                del request
                raise ValueError(
                    "Task identity contains a workload secret and cannot be exposed "
                    "through durable task projections."
                ) from None
            contains_secret_identity = (
                self._secret_redactor.redact_text(request.work_contract.contract_id)
                != request.work_contract.contract_id
            )
            if contains_secret_identity:
                del request
                raise ValueError(
                    "Work-contract identity contains a workload secret and cannot be exposed "
                    "through durable task projections."
                ) from None
        if (
            request.session_id is not None
            and request._verified_invocation_origin is None
            and request._runtime_session_binding is None
        ):
            snapshot = await self.session_store.load_invocation_snapshot(request.session_id)
            if snapshot is not None:
                request = task_create_with_runtime_invocation(
                    request,
                    source=(request._runtime_invocation_source or TaskExecutionSource.SDK_TASK),
                    session_invocation=snapshot,
                )
            del snapshot
        if request.available_at is not None and not self.task_store.supports_delayed_availability:
            del request
            raise NotImplementedError(
                f"{type(self.task_store).__name__} does not support delayed task availability."
            )
        if request.retry_policy is not None and not self.task_store.supports_task_retry_series:
            del request
            raise NotImplementedError(
                f"{type(self.task_store).__name__} does not support task retry series."
            )
        if request.work_contract is None:
            return await self.task_store.create_task(request)
        parent_invocation_snapshot: TaskInvocationSnapshot | None = None
        if request.parent_task_id is not None:
            (
                parent_snapshot_value,
                parent_lookup_failure,
            ) = await _load_public_task_invocation_snapshot(
                self.task_store,
                request.parent_task_id,
                redactor=self._secret_redactor,
            )
            if parent_lookup_failure is not None:
                del parent_snapshot_value, request
                raise_task_store_operation_failure(parent_lookup_failure)
            parent_validation = _copied_public_task_invocation_snapshot(
                parent_snapshot_value,
                redactor=self._secret_redactor,
            )
            del parent_snapshot_value
            parent_validation_failure = parent_validation.failure
            parent_invocation_snapshot = parent_validation.result
            del parent_validation
            if parent_validation_failure is not None:
                del parent_invocation_snapshot, request
                raise parent_validation_failure from None
            if (
                parent_invocation_snapshot is None
                or parent_invocation_snapshot.id != request.parent_task_id
            ):
                del parent_invocation_snapshot, request
                raise WorkContractConflict(
                    "Task parent invocation authority is unavailable for contracted creation."
                ) from None
        session_invocation_contains_secret = (
            request._runtime_session_binding is not None
            and invocation_contains_secret_public_identity(
                request._runtime_session_binding.invocation,
                self._secret_redactor,
            )
        )
        parent_invocation_contains_secret = (
            parent_invocation_snapshot is not None
            and invocation_contains_secret_public_identity(
                parent_invocation_snapshot.invocation,
                self._secret_redactor,
            )
        )
        direct_root_session_contains_secret = (
            request._runtime_session_binding is None
            and parent_invocation_snapshot is None
            and request.session_id is not None
            and self._secret_redactor.redact_text(request.session_id) != request.session_id
        )
        if (
            session_invocation_contains_secret
            or parent_invocation_contains_secret
            or direct_root_session_contains_secret
        ):
            del parent_invocation_snapshot, request
            raise ValueError(
                "Task invocation identity contains a workload secret and cannot be exposed "
                "through durable task projections."
            ) from None
        preflight_contract_bound_task_creation(
            request,
            parent_task=parent_invocation_snapshot,
        )
        task, creation_failure = await _create_public_contracted_task(
            self.task_store,
            request,
            redactor=self._secret_redactor,
        )
        if creation_failure is not None:
            del parent_invocation_snapshot, request, task
            raise_task_store_operation_failure(creation_failure)
        validation = _copied_public_task(
            task,
            redactor=self._secret_redactor,
        )
        del task
        validation_failure = validation.failure
        copied_task = validation.result
        del validation
        if validation_failure is not None:
            del parent_invocation_snapshot, request
            raise validation_failure from None
        if copied_task is None:
            del parent_invocation_snapshot, request
            raise WorkContractConflict("Task store returned an invalid contracted task.") from None
        task = copied_task
        del copied_task
        try:
            require_contract_bound_task_creation_snapshot(task)
        except (TypeError, ValueError):
            del parent_invocation_snapshot, request, task
            raise WorkContractConflict(
                "Task store returned a contracted task outside the creation-snapshot bounds."
            ) from None
        (
            invocation_snapshot,
            invocation_lookup_failure,
        ) = await _load_public_task_invocation_snapshot(
            self.task_store,
            task.id,
            redactor=self._secret_redactor,
        )
        if invocation_lookup_failure is not None:
            del invocation_snapshot, parent_invocation_snapshot, request, task
            raise_task_store_operation_failure(invocation_lookup_failure)
        validation = _copied_public_task_invocation_snapshot(
            invocation_snapshot,
            redactor=self._secret_redactor,
        )
        del invocation_snapshot
        validation_failure = validation.failure
        copied_invocation_snapshot = validation.result
        del validation
        if validation_failure is not None:
            del parent_invocation_snapshot, request, task
            raise validation_failure from None
        if not _contracted_task_creation_result_matches_request(
            task=task,
            request=request,
            invocation_snapshot=copied_invocation_snapshot,
            parent_invocation_snapshot=parent_invocation_snapshot,
            redactor=self._secret_redactor,
        ):
            del copied_invocation_snapshot, parent_invocation_snapshot, request, task
            raise WorkContractConflict(
                "Task store did not preserve the exact contracted task creation request."
            ) from None
        del copied_invocation_snapshot, parent_invocation_snapshot
        return task

    async def pause_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        if self.task_store is None:
            raise RuntimeError("task_store is required to pause tasks.")
        return await self.task_store.pause_task(task_id, reason=reason, payload=payload)

    async def block_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        if self.task_store is None:
            raise RuntimeError("task_store is required to block tasks.")
        return await self.task_store.block_task(task_id, reason=reason, payload=payload)

    async def mark_task_needs_attention(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Task:
        if self.task_store is None:
            raise RuntimeError("task_store is required to mark tasks needs-attention.")
        return await self.task_store.mark_task_needs_attention(
            task_id,
            reason=reason,
            payload=payload,
        )

    async def resume_task(self, task_id: str) -> Task:
        if self.task_store is None:
            raise RuntimeError("task_store is required to resume tasks.")
        return await self.task_store.resume_task(task_id)

    async def get_session_usage(self, session_id: str) -> SessionUsageSummary:
        session_id = await self._resolve_public_session_id(
            require_clean_nonblank(session_id, "session_id")
        )
        session = await self.session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}") from None
        events = await self._run_limit_controller.session_usage_events(session_id)
        summary = session_usage_summary(session_id, events)
        return summary.model_copy(
            update={"session_id": self.project_session_id_for_exposure(session_id)},
            deep=True,
        )

    async def get_causal_budget_usage(
        self,
        causal_budget_id: str,
    ) -> CausalBudgetUsageSummary:
        causal_budget_id = await self._resolve_public_causal_budget_id(causal_budget_id)
        sessions = await self._list_all_sessions(
            SessionQuery(
                causal_budget_id=causal_budget_id,
                order_by=SessionOrder.CREATED_AT_ASC,
            )
        )
        if not sessions:
            raise KeyError("Causal budget not found") from None
        records = await self._query_all_event_records(
            EventQuery(
                causal_budget_id=causal_budget_id,
                event_types=USAGE_BEARING_EVENT_TYPES,
            )
        )
        events = [record.event for record in records]
        summary = causal_budget_usage_summary(
            causal_budget_id=causal_budget_id,
            session_ids=[session.id for session in sessions],
            events=events,
        )
        public_session_ids = [
            self.project_session_id_for_exposure(session_id) for session_id in summary.session_ids
        ]
        public_causal_budget_id = self.project_causal_budget_id_for_exposure(
            causal_budget_id,
            session_ids=(session.id for session in sessions),
        )
        return summary.model_copy(
            update={
                "causal_budget_id": public_causal_budget_id,
                "session_ids": public_session_ids,
                "session_summaries": tuple(
                    session_summary.model_copy(
                        update={
                            "session_id": self.project_session_id_for_exposure(
                                session_summary.session_id
                            )
                        },
                        deep=True,
                    )
                    for session_summary in summary.session_summaries
                ),
            },
            deep=True,
        )

    async def _list_all_sessions(self, query: SessionQuery) -> list[Session]:
        return await query_all_sessions(self.session_store, query)

    async def _query_all_event_records(self, query: EventQuery) -> list[EventRecord]:
        return await query_all_event_records(self.session_store, query)

    async def run_event_watchers(
        self,
        watchers: Iterable[EventWatcher],
        *,
        limit: int = 100,
    ) -> list[EventWatcherRunResult]:
        """Process durable event watchers once.

        Watchers run over already-persisted events. Delivery is ordered and
        at-least-once: a cursor advances only after the handler succeeds or the
        event reaches the watcher's dead-letter threshold.
        """
        watcher_list = _validate_event_watchers(watchers)
        for watcher in watcher_list:
            if self._secret_redactor.redact_text(watcher.name) != watcher.name:
                raise ValueError(
                    "Event watcher name contains a workload secret and cannot be "
                    "used as durable watcher authority."
                )
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be an integer greater than or equal to 1.")

        remaining = limit
        results: list[EventWatcherRunResult] = []
        for watcher in watcher_list:
            deliveries = []
            blocked_by_active_lease = False
            processed_for_watcher = 0
            while remaining > 0 and processed_for_watcher < watcher.batch_size:
                state = await self.event_watcher_store.load_state(watcher.name)
                if (
                    watcher.query.before_sequence is not None
                    and state.cursor_sequence >= watcher.query.before_sequence
                ):
                    break
                page_limit = min(
                    remaining,
                    watcher.batch_size - processed_for_watcher,
                    EVENT_WATCHER_QUERY_PAGE_LIMIT,
                )
                records = await self.session_store.query_events(
                    event_query_after_cursor(
                        watcher.query,
                        state.cursor_sequence,
                        limit=page_limit,
                    )
                )
                if not records:
                    break

                should_fetch_next_page = True
                for record in records:
                    claim = await self.event_watcher_store.claim_event(
                        watcher_name=watcher.name,
                        record=record,
                        lease_seconds=watcher.lease_seconds,
                    )
                    if claim is None:
                        refreshed_state = await self.event_watcher_store.load_state(watcher.name)
                        if refreshed_state.cursor_sequence >= record.sequence:
                            continue
                        blocked_by_active_lease = True
                        should_fetch_next_page = False
                        break

                    watcher_error: str | None = None
                    try:
                        await run_event_watcher_handler(
                            watcher,
                            EventWatcherContext(
                                watcher_name=watcher.name,
                                record=self._project_persisted_event_record_for_exposure(record),
                                attempt=claim.attempt,
                            ),
                        )
                    except Exception as exc:
                        watcher_error = self._secret_redactor.redact_text_bounded(
                            event_watcher_error_payload(
                                exc,
                                redactor=self._secret_redactor,
                            ),
                            max_bytes=4096,
                        )
                    if watcher_error is not None:
                        delivery = await self.event_watcher_store.mark_failure(
                            claim,
                            error=watcher_error,
                            max_attempts=watcher.max_attempts,
                        )
                        deliveries.append(
                            delivery.model_copy(
                                update={"event_id": public_event_id(delivery.event_sequence)},
                                deep=True,
                            )
                        )
                        remaining -= 1
                        processed_for_watcher += 1
                        if delivery.status is not EventWatcherDeliveryStatus.DEAD_LETTERED:
                            should_fetch_next_page = False
                            break
                        continue

                    delivery = await self.event_watcher_store.mark_success(claim)
                    deliveries.append(
                        delivery.model_copy(
                            update={"event_id": public_event_id(delivery.event_sequence)},
                            deep=True,
                        )
                    )
                    remaining -= 1
                    processed_for_watcher += 1

                    if remaining <= 0 or processed_for_watcher >= watcher.batch_size:
                        should_fetch_next_page = False
                        break

                if len(records) < page_limit:
                    break
                if not should_fetch_next_page:
                    break

            results.append(
                EventWatcherRunResult(
                    watcher_name=watcher.name,
                    deliveries=deliveries,
                    blocked_by_active_lease=blocked_by_active_lease,
                )
            )
            if remaining <= 0:
                break
        return results

    async def get_session_cost(
        self,
        session_id: str,
        pricing: PriceBook,
        *,
        currency: str = "USD",
    ) -> SessionCostSummary:
        session_id = await self._resolve_public_session_id(
            require_clean_nonblank(session_id, "session_id")
        )
        session = await self.session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}") from None
        # Token cost derives from completions; independently terminal hosted
        # calls also carry priceable resource evidence when completion is absent.
        cost_event_records = await self._query_all_event_records(
            EventQuery(
                session_id=session_id,
                event_types=(
                    EventType.MODEL_COMPLETED,
                    EventType.MODEL_HOSTED_TOOL_CALL,
                ),
            )
        )
        summary = estimate_session_cost(
            session_id=session_id,
            events=[record.event for record in cost_event_records],
            pricing=pricing,
            currency=currency,
        )
        return summary.model_copy(
            update={"session_id": self.project_session_id_for_exposure(session_id)},
            deep=True,
        )

    async def get_causal_budget_cost(
        self,
        causal_budget_id: str,
        pricing: PriceBook,
        *,
        currency: str = "USD",
    ) -> CausalBudgetCostSummary:
        causal_budget_id = await self._resolve_public_causal_budget_id(causal_budget_id)
        sessions = await self._list_all_sessions(
            SessionQuery(
                causal_budget_id=causal_budget_id,
                order_by=SessionOrder.CREATED_AT_ASC,
            )
        )
        if not sessions:
            raise KeyError("Causal budget not found") from None
        records = await self._query_all_event_records(
            EventQuery(
                causal_budget_id=causal_budget_id,
                event_types=(
                    EventType.MODEL_COMPLETED,
                    EventType.MODEL_HOSTED_TOOL_CALL,
                ),
            )
        )
        summary = estimate_causal_budget_cost(
            causal_budget_id=causal_budget_id,
            session_ids=[session.id for session in sessions],
            events=[record.event for record in records],
            pricing=pricing,
            currency=currency,
        )
        public_causal_budget_id = self.project_causal_budget_id_for_exposure(
            causal_budget_id,
            session_ids=(session.id for session in sessions),
        )
        return summary.model_copy(
            update={
                "causal_budget_id": public_causal_budget_id,
                "session_ids": [
                    self.project_session_id_for_exposure(session_id)
                    for session_id in summary.session_ids
                ],
                "session_costs": tuple(
                    session_cost.model_copy(
                        update={
                            "session_id": self.project_session_id_for_exposure(
                                session_cost.session_id
                            )
                        },
                        deep=True,
                    )
                    for session_cost in summary.session_costs
                ),
            },
            deep=True,
        )

    async def emit_hook_event(
        self,
        *,
        session_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        event_type = validate_public_custom_event_type(event_type)
        event = Event(
            type=event_type,
            session_id=session_id,
            payload=copy_json_value(payload or {}, "payload"),
        )
        emitted = await self._event_writer.emit(event)
        return await self._project_emitted_event_for_public_api(emitted)

    async def fork_session(self, request: ForkSessionRequest) -> AsyncIterator[Event]:
        if type(request) is not ForkSessionRequest:
            raise TypeError("Runtime fork requires a ForkSessionRequest.")
        source_session_id: str | None = None
        store_resolved_source_session_id: str | None = None
        private_request: ForkSessionRequest | None = None
        events: tuple[Event, ...] | None = None
        try:
            (
                source_session_id,
                store_resolved_source_session_id,
            ) = await self._resolve_public_session_authority(request.source_session_id)
            private_request = request.model_copy(
                update={"source_session_id": source_session_id},
                deep=True,
            )
            events = await self._collect_public_fork_events(
                private_request,
                store_resolved_source_session_id=store_resolved_source_session_id,
            )
        finally:
            del request
            source_session_id = store_resolved_source_session_id = None
            private_request = None
        if events is None:
            raise RuntimeError("Session fork ended without events or a failure.")
        for event in events:
            yield event

    async def run_fork_group(self, request: ForkGroupRequest) -> ForkGroupResult:
        """Run or reconstruct one bounded durable fork group."""

        return await self._fork_group_coordinator.run_group(request)

    async def inspect_fork_group(
        self,
        source_session_id: str,
        group_id: str,
    ) -> ForkGroupResult | None:
        """Inspect a durable fork group without claiming or advancing its work."""

        return await self._fork_group_coordinator.inspect_group(source_session_id, group_id)

    async def _fork_session_from_runtime_context(
        self,
        request: ForkSessionRequest,
        *,
        source_session_id: str,
    ) -> AsyncIterator[Event]:
        """Fork a hook-owned source without granting that trust to public callers."""

        if type(request) is not ForkSessionRequest:
            raise TypeError("Runtime fork requires a ForkSessionRequest.")
        private_request: ForkSessionRequest | None = None
        events: tuple[Event, ...] | None = None
        try:
            source_session_id = require_clean_nonblank(
                source_session_id,
                "runtime hook source_session_id",
            )
            if request.source_session_id != source_session_id:
                raise ValueError(
                    "Runtime hook fork source_session_id does not match its context session."
                )
            private_request = copy_fork_session_request(request)
            events = await self._collect_public_fork_events(
                private_request,
                store_resolved_source_session_id=source_session_id,
            )
        finally:
            del request
            source_session_id = ""
            private_request = None
        if events is None:
            raise RuntimeError("Runtime-hook session fork ended without events or a failure.")
        for event in events:
            yield event

    async def _collect_public_fork_events(
        self,
        request: ForkSessionRequest,
        *,
        store_resolved_source_session_id: str | None,
    ) -> tuple[Event, ...]:
        """Detach fork-authority failures before they cross the public boundary."""

        failure: Exception | None = None
        projected_events: list[Event] = []
        stream: AsyncGenerator[Event, None] | None = None
        owned_stream: _RunFenceOwnedEventStream | None = None
        try:
            try:
                session_request_boundary.require_store_resolved_or_secret_free_session_authority(
                    request.source_session_id,
                    store_resolved_value=store_resolved_source_session_id,
                    field_name="source_session_id",
                    redactor=self._secret_redactor,
                )
            except ValueError as exc:
                failure = ValueError(str(exc))
            if failure is None:
                await self._preflight_ordinary_fork_source(request.source_session_id)
                stream = self._fork_session_private(
                    request,
                    store_resolved_source_session_id=store_resolved_source_session_id,
                )
                async with _close_delegated_event_stream(stream) as owned_stream:
                    async for event in owned_stream:
                        projected_events.append(
                            await self._project_emitted_event_for_public_api(event)
                        )
        except session_request_boundary.ForkSourceNotFoundError:
            failure = KeyError("Fork source session was not found.")
        except session_request_boundary.ForkActiveModelStageError:
            failure = ValueError("Fork source session has an active model-completion stage.")
        except session_request_boundary.ForkAuthorityError as exc:
            failure = ValueError(str(exc))
        finally:
            del request
            store_resolved_source_session_id = None
            stream = owned_stream = None
        if failure is not None:
            projected_events.clear()
            raise failure from None
        return tuple(projected_events)

    async def _preflight_ordinary_fork_source(self, session_id: str) -> None:
        await self._session_engine._require_ordinary_session_execution(
            session_id,
            admit_session=False,
        )

    async def _admit_ordinary_fork_source(self, session_id: str) -> None:
        await self._session_engine._require_ordinary_session_execution(
            session_id,
            admit_session=True,
        )

    async def _fork_session_private(
        self,
        request: ForkSessionRequest,
        *,
        store_resolved_source_session_id: str | None = None,
    ) -> AsyncGenerator[Event, None]:
        if type(request) is not ForkSessionRequest:
            raise TypeError("Runtime fork requires a ForkSessionRequest.")
        prepared = session_request_boundary.prepare_fork_session_request_with_identity(
            request,
            redactor=self._secret_redactor,
            public_authority_alias_codec=self._public_authority_alias_codec,
            store_resolved_source_session_id=store_resolved_source_session_id,
        )
        del request
        stream = self._session_engine.fork_session(
            request=prepared.request,
            request_sha256=prepared.request_sha256,
            accepted_request_sha256s=prepared.accepted_request_sha256s,
            store_resolved_source_session_id=store_resolved_source_session_id,
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def _verifier_aware_recovery_execution_outcome(
        self,
        *,
        session_id: str,
        task_id: str | None = None,
        admit_session: bool = True,
    ) -> tuple[bool, BaseException | None]:
        return await self._session_engine._verifier_aware_task_execution_outcome(
            task_id,
            session_id=session_id,
            admit_session=admit_session,
        )

    async def _require_ordinary_recovery_execution(
        self,
        *,
        session_id: str,
        task_id: str | None = None,
        admit_session: bool,
    ) -> None:
        (
            requires_completion_decision,
            admission_failure,
        ) = await self._verifier_aware_recovery_execution_outcome(
            session_id=session_id,
            task_id=task_id,
            admit_session=admit_session,
        )
        if admission_failure is not None:
            raise_task_store_operation_failure(admission_failure)
        if requires_completion_decision:
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None

    async def _run_recovery_session(
        self,
        request: RecoverySessionRunRequest,
    ) -> AsyncGenerator[Event, None]:
        (
            requires_completion_decision,
            admission_failure,
        ) = await self._verifier_aware_recovery_execution_outcome(
            session_id=request.session.id,
            task_id=request.task_id,
        )
        if admission_failure is not None:
            del request
            raise_task_store_operation_failure(admission_failure)
        if requires_completion_decision:
            del request
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        checkpoint = await self._runtime_session_store.load_checkpoint(request.session.id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        expected_profile = request.active_invocation_profile
        if (
            active_profile != expected_profile
            or expected_profile.session_id != request.session.id
            or expected_profile.run_epoch != request.session.run_epoch
        ):
            raise RuntimeError(
                "Recovery run does not reference the active invocation execution profile."
            )
        execution_profile = expected_profile.profile
        initial_tool_exposure = (
            None
            if request.initial_model_step_tool_exposure is None
            else resolved_tool_exposure_from_authority(
                request.initial_model_step_tool_exposure,
                request.registered_agent.tool_capabilities,
                catalogue_revision=request.registered_agent.tool_catalogue.revision,
            )
        )
        stream = self._run_session(
            session=request.session,
            registered_agent=request.registered_agent,
            registered_provider=request.registered_provider,
            registered_environment=request.registered_environment,
            execution_profile=execution_profile,
            messages=request.messages,
            messages_to_append=request.messages_to_append,
            max_steps=request.max_steps,
            limits=request.limits,
            budget_limits=request.budget_limits,
            budget_policy=copy_budget_policy(request.budget_policy),
            retry_policy=request.retry_policy,
            structured_output=request.structured_output,
            thinking=request.thinking,
            request_loop_policies=request.request_loop_policies,
            request_metadata=request.request_metadata,
            task_id=request.task_id,
            task_worker_id=request.task_worker_id,
            start_event_type=request.start_event_type,
            start_event_payload=request.start_event_payload,
            start_task_on_enter=request.start_task_on_enter,
            release_run_fence_on_exit=request.release_run_fence_on_exit,
            deliver_queued_input_before_first_step=False,
            run_limit_accounting=request.run_limit_accounting,
            initial_model_step_identity=request.initial_model_step_identity,
            initial_model_step_number=request.initial_model_step_number,
            initial_model_step_tool_exposure=initial_tool_exposure,
            previous_tool_exposure_profile_id=(request.previous_tool_exposure_profile_id),
            preserve_failure_until_initial_provider_dispatch=(
                request.preserve_failure_until_initial_provider_dispatch
            ),
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def _validate_execution_profile_continuation_for_recovery(
        self,
        session: Session,
        checkpoint: dict[str, Any] | None,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        request_loop_policies: tuple[LoopPolicy, ...] | None = None,
        frozen_candidate_profile: ExecutionProfileIdentity | None = None,
        *,
        budget_policy: BudgetPolicy | None,
        request_budget_limits: tuple[BudgetLimit, ...] = (),
        structured_output: StructuredOutputSpec | None = None,
        thinking: ThinkingConfig | None = None,
        max_steps: int = 16,
        limits: RunLimits | None = None,
        retry_policy: RetryPolicy | None = None,
        invocation_semantics_available: bool = False,
        require_open_interaction: bool = True,
        additional_profile_fingerprints: tuple[str, ...] = (),
    ) -> ActiveInvocationExecutionProfile:
        return await self._session_engine.validate_execution_profile_continuation(
            session=session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            request_loop_policies=request_loop_policies,
            budget_policy=copy_budget_policy(budget_policy),
            request_budget_limits=request_budget_limits,
            structured_output=structured_output,
            thinking=thinking,
            max_steps=max_steps,
            limits=limits,
            retry_policy=retry_policy,
            invocation_semantics_available=invocation_semantics_available,
            frozen_candidate_profile=frozen_candidate_profile,
            require_open_interaction=require_open_interaction,
            additional_profile_fingerprints=additional_profile_fingerprints,
        )

    def _emit_recovery_terminal_event_with_hooks(
        self,
        request: RecoveryTerminalEventRequest,
    ) -> AsyncIterator[Event]:
        return self._emit_terminal_event_with_hooks(
            event=request.event,
            phase=request.phase,
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            execution_profile=request.execution_profile,
        )

    def _fail_provider_operation_resolution(
        self,
        request: ProviderOperationFailureRequest,
    ) -> AsyncIterator[Event]:
        return self._session_engine.fail_provider_operation_resolution(
            resolution_event=request.resolution_event,
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            execution_profile=request.execution_profile,
            legacy_resolution_without_profile=request.legacy_resolution_without_profile,
        )

    def _stop_recovery_session_for_limit_reached(
        self,
        request: RecoveryLimitStopRequest,
    ) -> AsyncIterator[Event]:
        return self._stop_session_for_limit_reached(
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            decision=request.decision,
            usage_summary=request.usage_summary,
            cost_summary=request.cost_summary,
            messages=request.messages,
            tool_calls=request.tool_calls,
            completed_tool_outcomes=request.completed_tool_outcomes,
            pending_approval_to_clear=request.pending_approval_to_clear,
            deferred_messages=request.deferred_messages,
            requested_approval_decision=request.requested_approval_decision,
            approval_resolution_request_digest=request.approval_resolution_request_digest,
            execution_profile=request.execution_profile,
        )

    def _interrupt_session_for_recovery(
        self,
        request: RecoveryInterruptionRequest,
    ) -> AsyncIterator[Event]:
        return self._handle_session_interrupted(
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            execution_profile=request.execution_profile,
        )

    def _pending_session_interrupt_checkpoint_for_recovery(
        self,
        payload: dict[str, Any],
        cascade_created_at: datetime,
    ):
        return _checkpoint_with_pending_session_interrupt(
            payload,
            cascade_created_at=cascade_created_at,
        )

    async def _complete_abandoned_recovery_turn(
        self,
        request: RecoveryAbandonedTurnRequest,
    ) -> Session:
        finalized, _, _ = await self._session_engine._publish_sibling_interaction_transition(
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            to_status=SessionStatus.INTERRUPTED,
            execution_profile=request.execution_profile,
        )
        if request.run_started_at is not None and request.usage_tracker is not None:
            await self._emit_turn_completed_once(
                session=finalized,
                registered_agent=request.registered_agent,
                environment_name=request.environment_name,
                status=SessionStatus.INTERRUPTED,
                run_started_at=request.run_started_at,
                usage_tracker=request.usage_tracker,
                active_run=request.active_run,
            )
        return finalized

    async def _resume_recovery_interaction(
        self,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> Event | None:
        return await self._session_engine.resume_interaction(
            session,
            registered_agent,
            registered_environment,
        )

    async def resolve_user_input(
        self,
        response: UserInputResponse,
    ) -> AsyncIterator[Event]:
        if type(response) is not UserInputResponse:
            raise TypeError("Runtime user input resolution requires a UserInputResponse.")
        session_id = await self._resolve_public_session_id(response.session_id)
        response = copy_user_input_response(response).model_copy(
            update={
                "session_id": session_id,
                "input_id": await self._resolve_public_action_linkage(
                    session_id=session_id,
                    value=response.input_id,
                    field_name="input_id",
                ),
            },
        )
        stream = self._resolve_user_input_private(response)
        del response
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield await self._project_emitted_event_for_public_api(event)

    async def _resolve_user_input_private(
        self,
        response: UserInputResponse,
    ) -> AsyncGenerator[Event, None]:
        """Resume a session paused by ``ask_user`` with the user's answer.

        The answer becomes the ``ask_user`` tool result; any other tool calls in the same
        round (none ran before the pause) execute now, and the session continues.
        """
        if type(response) is not UserInputResponse:
            raise TypeError("Runtime user input resolution requires a UserInputResponse.")
        response = copy_user_input_response(response)
        (
            requires_completion_decision,
            admission_failure,
        ) = await self._verifier_aware_recovery_execution_outcome(
            session_id=response.session_id,
            admit_session=False,
        )
        if admission_failure is not None:
            del response
            raise_task_store_operation_failure(admission_failure)
        if requires_completion_decision:
            del response
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        session_id = response.session_id
        stream = self._recovery_coordinator.resolve_user_input(
            response=response,
            before_mutation=lambda: self._require_ordinary_recovery_execution(
                session_id=session_id,
                admit_session=True,
            ),
        )
        del response
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield event

    async def recover_user_input(
        self,
        request: UserInputRecoveryRequest,
    ) -> AsyncIterator[Event]:
        if type(request) is not UserInputRecoveryRequest:
            raise TypeError("Runtime user input recovery requires a UserInputRecoveryRequest.")
        session_id = await self._resolve_public_session_id(request.session_id)
        request = copy_user_input_recovery_request(request).model_copy(
            update={
                "session_id": session_id,
                "input_id": await self._resolve_public_action_linkage(
                    session_id=session_id,
                    value=request.input_id,
                    field_name="input_id",
                ),
                "tool_call_id": await self._resolve_public_action_linkage(
                    session_id=session_id,
                    value=request.tool_call_id,
                    field_name="tool_call_id",
                ),
            },
        )
        stream = self._recover_user_input_private(request)
        del request
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield await self._project_emitted_event_for_public_api(event)

    async def _recover_user_input_private(
        self,
        request: UserInputRecoveryRequest,
    ) -> AsyncGenerator[Event, None]:
        """Recover a user-input round stuck on `manual_recovery_required`.

        A tool in the paused round started on a prior resume but recorded no terminal event
        (a crash mid-tool), so it cannot be re-run automatically. The caller supplies the
        externally verified outcome for that `tool_call_id`; Cayu persists it as the tool's
        terminal result and continues the round (re-supplying `answer` in case the `ask_user`
        result was not recorded before the crash). Cayu does not infer the outcome itself.
        """
        if type(request) is not UserInputRecoveryRequest:
            raise TypeError("Runtime user input recovery requires a UserInputRecoveryRequest.")
        request = copy_user_input_recovery_request(request)
        (
            requires_completion_decision,
            admission_failure,
        ) = await self._verifier_aware_recovery_execution_outcome(
            session_id=request.session_id,
            admit_session=False,
        )
        if admission_failure is not None:
            del request
            raise_task_store_operation_failure(admission_failure)
        if requires_completion_decision:
            del request
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        session_id = request.session_id
        stream = self._recovery_coordinator.recover_user_input_request(
            request=request,
            before_mutation=lambda: self._require_ordinary_recovery_execution(
                session_id=session_id,
                admit_session=True,
            ),
        )
        del request
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield event

    async def resolve_tool_approval(
        self,
        request: ToolApprovalRequest,
    ) -> AsyncIterator[Event]:
        if type(request) is not ToolApprovalRequest:
            raise TypeError("Runtime approval resolution requires a ToolApprovalRequest.")
        session_id = await self._resolve_public_session_id(request.session_id)
        request = copy_tool_approval_request(request).model_copy(
            update={
                "session_id": session_id,
                "approval_id": await self._resolve_public_action_linkage(
                    session_id=session_id,
                    value=request.approval_id,
                    field_name="approval_id",
                ),
                "tool_round_id": await self._resolve_public_action_linkage(
                    session_id=session_id,
                    value=request.tool_round_id,
                    field_name="tool_round_id",
                ),
                "tool_call_id": await self._resolve_public_action_linkage(
                    session_id=session_id,
                    value=request.tool_call_id,
                    field_name="tool_call_id",
                ),
            },
        )
        stream = self._resolve_tool_approval_private(request)
        del request
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield await self._project_emitted_event_for_public_api(event)

    async def resolve_provider_operation(
        self,
        request: ProviderOperationResolutionRequest,
    ) -> AsyncIterator[Event]:
        """Resolve unavailable provider work by explicit fallback retry or failure."""

        if type(request) is not ProviderOperationResolutionRequest:
            raise TypeError(
                "Runtime provider-operation resolution requires a "
                "ProviderOperationResolutionRequest."
            )
        session_id = await self._resolve_public_session_id(request.session_id)
        request = copy_provider_operation_resolution_request(
            request,
            session_id=session_id,
        )
        (
            requires_completion_decision,
            admission_failure,
        ) = await self._verifier_aware_recovery_execution_outcome(
            session_id=request.session_id,
            admit_session=False,
        )
        if admission_failure is not None:
            del request
            raise_task_store_operation_failure(admission_failure)
        if requires_completion_decision:
            del request
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        session_id = request.session_id
        stream = self._recovery_coordinator.resolve_provider_operation(
            request,
            before_mutation=lambda: self._require_ordinary_recovery_execution(
                session_id=session_id,
                admit_session=True,
            ),
        )
        del request
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield await self._project_emitted_event_for_public_api(event)

    async def _resolve_tool_approval_private(
        self,
        request: ToolApprovalRequest,
    ) -> AsyncGenerator[Event, None]:
        if type(request) is not ToolApprovalRequest:
            raise TypeError("Runtime approval resolution requires a ToolApprovalRequest.")
        request = _validate_tool_approval_request(request)
        (
            requires_completion_decision,
            admission_failure,
        ) = await self._verifier_aware_recovery_execution_outcome(
            session_id=request.session_id,
            admit_session=False,
        )
        if admission_failure is not None:
            del request
            raise_task_store_operation_failure(admission_failure)
        if requires_completion_decision:
            del request
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        session_id = request.session_id
        stream = self._recovery_coordinator.resolve_tool_approval(
            request=request,
            before_mutation=lambda: self._require_ordinary_recovery_execution(
                session_id=session_id,
                admit_session=True,
            ),
        )
        del request
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield event

    async def recover_tool_approval(
        self,
        request: ToolApprovalRecoveryRequest,
    ) -> AsyncIterator[Event]:
        if type(request) is not ToolApprovalRecoveryRequest:
            raise TypeError("Runtime approval recovery requires a ToolApprovalRecoveryRequest.")
        session_id = await self._resolve_public_session_id(request.session_id)
        request = copy_tool_approval_recovery_request(request).model_copy(
            update={
                "session_id": session_id,
                "approval_id": await self._resolve_public_action_linkage(
                    session_id=session_id,
                    value=request.approval_id,
                    field_name="approval_id",
                ),
                "tool_round_id": await self._resolve_public_action_linkage(
                    session_id=session_id,
                    value=request.tool_round_id,
                    field_name="tool_round_id",
                ),
                "tool_call_id": await self._resolve_public_action_linkage(
                    session_id=session_id,
                    value=request.tool_call_id,
                    field_name="tool_call_id",
                ),
            },
        )
        stream = self._recover_tool_approval_private(request)
        del request
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield await self._project_emitted_event_for_public_api(event)

    async def _recover_tool_approval_private(
        self,
        request: ToolApprovalRecoveryRequest,
    ) -> AsyncGenerator[Event, None]:
        if type(request) is not ToolApprovalRecoveryRequest:
            raise TypeError("Runtime approval recovery requires a ToolApprovalRecoveryRequest.")
        request = _validate_tool_approval_recovery_request(request)
        (
            requires_completion_decision,
            admission_failure,
        ) = await self._verifier_aware_recovery_execution_outcome(
            session_id=request.session_id,
            admit_session=False,
        )
        if admission_failure is not None:
            del request
            raise_task_store_operation_failure(admission_failure)
        if requires_completion_decision:
            del request
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        session_id = request.session_id
        stream = self._recovery_coordinator.recover_tool_approval_request(
            request=request,
            before_mutation=lambda: self._require_ordinary_recovery_execution(
                session_id=session_id,
                admit_session=True,
            ),
        )
        del request
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield event

    async def recover_tool_round(
        self,
        request: ToolRoundRecoveryRequest,
    ) -> AsyncIterator[Event]:
        if type(request) is not ToolRoundRecoveryRequest:
            raise TypeError("Runtime tool round recovery requires a ToolRoundRecoveryRequest.")
        session_id = await self._resolve_public_session_id(request.session_id)
        request = copy_tool_round_recovery_request(request).model_copy(
            update={
                "session_id": session_id,
                "round_id": await self._resolve_public_action_linkage(
                    session_id=session_id,
                    value=request.round_id,
                    field_name="tool_round_id",
                ),
                "tool_call_id": await self._resolve_public_action_linkage(
                    session_id=session_id,
                    value=request.tool_call_id,
                    field_name="tool_call_id",
                ),
            },
        )
        stream = self._recover_tool_round_private(request)
        del request
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield await self._project_emitted_event_for_public_api(event)

    async def _recover_tool_round_private(
        self,
        request: ToolRoundRecoveryRequest,
    ) -> AsyncGenerator[Event, None]:
        """Recover a crashed ordinary tool round with an operator-verified outcome.

        A tool call in a non-approval round started but recorded no terminal event
        (a crash mid-tool), so an automatic resume would close it as an
        unknown-outcome failure. The caller supplies the externally verified outcome
        for that `tool_call_id`; Cayu persists it as the call's terminal result and
        never re-runs the tool. One call per invocation: if other
        started-but-unresolved calls remain, the session returns to INTERRUPTED with
        `manual_recovery_required` naming the next call; otherwise the round closes
        from the recorded outcomes and the model loop continues. A crashed round can
        leave the session FAILED (an in-process persistence error) or in a stale live
        status (a process kill), so FAILED and RUNNING are accepted alongside
        INTERRUPTED. An existing INTERRUPTING transition wins rather than being
        reopened by recovery. The in-process claim registered while this recovery
        streams blocks duplicate work in this process, while a durable recovery
        claim serializes other workers and fences an expired owner. If this call
        fails after claiming a stale live session, the session closes to the
        resumable INTERRUPTED state. When the recovered terminal event is already
        durable, the evidence remains authoritative: do not retry the same
        `tool_call_id` — `resume(...)` finishes the round from the persisted outcome.
        """
        if type(request) is not ToolRoundRecoveryRequest:
            raise TypeError("Runtime tool round recovery requires a ToolRoundRecoveryRequest.")
        request = copy_tool_round_recovery_request(request)
        (
            requires_completion_decision,
            admission_failure,
        ) = await self._verifier_aware_recovery_execution_outcome(
            session_id=request.session_id,
            admit_session=False,
        )
        if admission_failure is not None:
            del request
            raise_task_store_operation_failure(admission_failure)
        if requires_completion_decision:
            del request
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        session_id = request.session_id
        stream = self._recovery_coordinator.recover_tool_round_request(
            request=request,
            before_mutation=lambda: self._require_ordinary_recovery_execution(
                session_id=session_id,
                admit_session=True,
            ),
        )
        del request
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield event

    async def _run_session(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None,
        messages: list[Message],
        messages_to_append: list[Message],
        max_steps: int,
        limits: RunLimits,
        budget_limits: tuple[BudgetLimit, ...],
        budget_policy: BudgetPolicy | None,
        retry_policy: RetryPolicy,
        structured_output: StructuredOutputSpec | None,
        thinking: ThinkingConfig | None,
        request_loop_policies: tuple[LoopPolicy, ...],
        request_metadata: dict[str, Any],
        task_id: str | None,
        task_worker_id: str | None,
        start_event_type: EventType | None,
        start_event_payload: dict[str, Any],
        start_task_on_enter: bool = True,
        release_run_fence_on_exit: bool = True,
        deliver_queued_input_before_first_step: bool = True,
        run_limit_accounting: RunLimitAccountingContext | None = None,
        initial_model_step_identity: ModelStepIdentity | None = None,
        initial_model_step_number: int | None = None,
        initial_model_step_tool_exposure: ResolvedToolExposure | None = None,
        previous_tool_exposure_profile_id: str | None = None,
        preserve_failure_until_initial_provider_dispatch: bool = False,
    ) -> AsyncGenerator[Event, None]:
        interaction_id = _current_session_interaction_id(session.id)
        targeted_tool_projection = resolve_targeted_tool_projection(
            registered_agent.targeted_tool_mode,
            provider=registered_provider.provider,
            model=session.model,
        )
        targeted_tool_grant_records: tuple[TargetedToolGrantRecord, ...] = ()
        targeted_tool_grant_events: tuple[Event, ...] = ()
        if interaction_id is not None:
            (
                targeted_tool_grant_records,
                targeted_tool_grant_events,
            ) = await self._session_engine._reconstruct_targeted_tool_grants(
                session=session,
                interaction_id=interaction_id,
                task_id=task_id,
                registered_agent=registered_agent,
                capability_ceiling=tool_capability_ceiling_from_session_metadata(session.metadata),
                observed_at=self._clock(),
                preserve_native_projection_snapshot=(
                    targeted_tool_projection is TargetedToolProjectionKind.OPENAI_ADDITIONAL_TOOLS
                ),
            )
        stream = self._session_engine._run_session(
            session=session,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            execution_profile=execution_profile,
            messages=messages,
            messages_to_append=messages_to_append,
            max_steps=max_steps,
            limits=limits,
            budget_limits=budget_limits,
            budget_policy=budget_policy,
            retry_policy=retry_policy,
            structured_output=structured_output,
            thinking=thinking,
            request_loop_policies=request_loop_policies,
            request_metadata=request_metadata,
            request_trace_metadata=request_metadata,
            targeted_tool_grants=targeted_tool_grant_footprint(
                targeted_tool_grant_records,
                projection=targeted_tool_projection,
            ),
            task_id=task_id,
            task_worker_id=task_worker_id,
            start_event_type=start_event_type,
            start_event_payload=start_event_payload,
            start_task_on_enter=start_task_on_enter,
            release_run_fence_on_exit=release_run_fence_on_exit,
            deliver_queued_input_before_first_step=(deliver_queued_input_before_first_step),
            run_limit_accounting=run_limit_accounting,
            initial_model_step_identity=initial_model_step_identity,
            initial_model_step_number=initial_model_step_number,
            initial_model_step_tool_exposure=initial_model_step_tool_exposure,
            previous_tool_exposure_profile_id=previous_tool_exposure_profile_id,
            preserve_failure_until_initial_provider_dispatch=(
                preserve_failure_until_initial_provider_dispatch
            ),
        )
        for targeted_tool_grant_event in targeted_tool_grant_events:
            yield targeted_tool_grant_event
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def _emit_turn_completed_once(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        status: SessionStatus,
        run_started_at: float,
        usage_tracker: SessionUsageTracker,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
    ) -> Event:
        events = await self._session_engine._emit_turn_completed_once(
            session=session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            status=status,
            run_started_at=run_started_at,
            usage_tracker=usage_tracker,
            active_run=active_run,
        )
        return events[-1]

    async def _apply_model_step_budget_evaluation(
        self,
        request: ModelStepBudgetEvaluationRequest,
    ) -> AsyncIterator[Event]:
        stream = self._session_engine._apply_model_step_budget_evaluation(request=request)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def _apply_model_step_limit_evaluation(
        self,
        request: ModelStepLimitEvaluationRequest,
    ) -> AsyncIterator[Event]:
        stream = self._session_engine._apply_model_step_limit_evaluation(request=request)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def _stop_for_model_step_budget_reservation_failure(
        self,
        request: ModelStepBudgetReservationFailureRequest,
    ) -> AsyncIterator[Event]:
        stream = self._session_engine._stop_for_model_step_budget_reservation_failure(
            request=request
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def _apply_tool_round_limit(
        self,
        request: ToolRoundLimitRequest,
    ) -> AsyncIterator[Event]:
        stream = self._session_engine._apply_tool_round_limit(request=request)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def _stop_session_for_limit_reached(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        decision: StopDecision,
        usage_summary: SessionUsageSummary,
        cost_summary: SessionCostSummary | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome],
        pending_approval_to_clear: PendingToolApproval | None = None,
        deferred_messages: list[Message] | None = None,
        requested_approval_decision: ToolApprovalDecision | None = None,
        approval_resolution_request_digest: str | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
        execution_profile: ExecutionProfileIdentity | None = None,
    ) -> AsyncIterator[Event]:
        # This adapter is owned by RecoveryCoordinator, not SessionEngine._run_session,
        # so its terminal transition must consume the sibling cancellation handoff.
        stream = self._session_engine._stop_session_for_limit_reached(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            decision=decision,
            usage_summary=usage_summary,
            cost_summary=cost_summary,
            messages=messages,
            tool_calls=tool_calls,
            completed_tool_outcomes=completed_tool_outcomes,
            pending_approval_to_clear=pending_approval_to_clear,
            deferred_messages=deferred_messages,
            requested_approval_decision=requested_approval_decision,
            approval_resolution_request_digest=approval_resolution_request_digest,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
            execution_profile=execution_profile,
            reconcile_transition_cancellation=True,
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def _load_pending_session_interrupt_payload(
        self,
        session_id: str,
        *,
        default: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._session_engine._load_pending_session_interrupt_payload(
            session_id=session_id, default=default
        )

    async def _load_pending_interruption_cascade(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        return await self._session_engine._load_pending_interruption_cascade(session_id=session_id)

    async def _claim_pending_interruption_cascade(
        self,
        session_id: str,
        interrupt_payload: dict[str, Any],
        *,
        create_if_missing: bool = True,
        retry_request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        return await self._session_engine._claim_pending_interruption_cascade(
            session_id=session_id,
            interrupt_payload=interrupt_payload,
            create_if_missing=create_if_missing,
            retry_request=retry_request,
        )

    async def _mark_pending_interruption_cascade_failed(
        self,
        session_id: str,
        attempt_id: str,
        generation: int,
        claim_id: str,
    ) -> bool:
        return await self._session_engine._mark_pending_interruption_cascade_failed(
            session_id=session_id, attempt_id=attempt_id, generation=generation, claim_id=claim_id
        )

    async def _complete_pending_interruption_cascade(
        self,
        session_id: str,
        attempt_id: str,
        generation: int,
        claim_id: str,
    ) -> tuple[bool, bool]:
        return await self._session_engine._complete_pending_interruption_cascade(
            session_id=session_id, attempt_id=attempt_id, generation=generation, claim_id=claim_id
        )

    async def _renew_pending_interruption_cascade_claim(
        self,
        session_id: str,
        attempt_id: str,
        generation: int,
        claim_id: str,
    ) -> bool:
        return await self._session_engine._renew_pending_interruption_cascade_claim(
            session_id=session_id, attempt_id=attempt_id, generation=generation, claim_id=claim_id
        )

    async def _release_pending_interruption_cascade_claim(
        self,
        session_id: str,
        attempt_id: str,
        generation: int,
        claim_id: str,
    ) -> None:
        return await self._session_engine._release_pending_interruption_cascade_claim(
            session_id=session_id, attempt_id=attempt_id, generation=generation, claim_id=claim_id
        )

    async def _handle_session_interrupted(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
    ) -> AsyncIterator[Event]:
        stream = self._session_engine._handle_session_interrupted(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            execution_profile=execution_profile,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def _close_tool_round_after_interrupt(
        self,
        request: InterruptedToolRoundRequest,
    ) -> AsyncIterator[Event]:
        stream = self._session_engine._close_tool_round_after_interrupt(request=request)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    def scoped_event_emitter(
        self,
        *,
        event_types: Iterable[EventType | str],
    ) -> Callable[[Event], Awaitable[Event]]:
        """Return an out-of-band emitter constrained to specific event types."""
        allowed = frozenset(str(event_type) for event_type in event_types)
        if not allowed:
            raise ValueError("scoped_event_emitter requires at least one event type.")

        async def emit(event: Event) -> Event:
            if str(event.type) not in allowed:
                raise ValueError(f"Event type {event.type!r} is not allowed for this emitter.")
            return await self.emit_event(event)

        return emit

    def _workflow_event_emitter(
        self,
        session_id: str,
    ) -> Callable[[list[Event]], Awaitable[list[Event]]]:
        """Return a trusted private emitter for workflow-owned runtime internals.

        Results remain private authority because workflow execution feeds them
        back into runtime control flow. Public callers use ``emit_event`` or the
        server projection boundary instead.
        """

        async def emit(events: list[Event]) -> list[Event]:
            _validate_workflow_event_batch(events, allow_cayu_internal=True)
            return await self._event_writer.emit_many(session_id, events)

        return emit

    def _workflow_step_reserver(
        self,
        session_id: str,
        workflow_name: str,
    ) -> Callable[[Event, str], Awaitable[bool]]:
        """Return the atomic reservation boundary for one workflow journal."""

        async def reserve(event: Event, attempt_id: str) -> bool:
            _validate_workflow_event_batch([event], allow_cayu_internal=True)
            if event.session_id != session_id:
                raise ValueError("Workflow reservation event has the wrong session_id.")
            return await self._event_writer.reserve_workflow_step_started(
                event,
                workflow_name=workflow_name,
                attempt_id=attempt_id,
            )

        return reserve

    async def emit_event(self, event: Event) -> Event:
        """Publish an event to the session store and all sinks.

        Low-level seam for runtime-owned out-of-band session events. Prefer
        ``scoped_event_emitter`` when handing an emitter to a component. Redaction
        is applied by the sinks; callers must not place raw secrets in the payload.
        """
        emitted = await self._emit_event_private(event)
        return await self._project_emitted_event_for_public_api(emitted)

    async def _emit_event_private(self, event: Event) -> Event:
        if not isinstance(event, Event):
            raise TypeError("emit_event requires an Event instance.")
        emitted = await self._event_writer.emit(event)
        self._session_control.queue_out_of_band_event(emitted)
        return emitted

    async def _emit_terminal_event_with_hooks(
        self,
        *,
        event: Event,
        phase: RuntimeHookPhase,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None = None,
    ) -> AsyncIterator[Event]:
        stream = self._session_engine._emit_terminal_event_with_hooks(
            event=event,
            phase=phase,
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            execution_profile=execution_profile,
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def emit_events(self, session_id: str, events: list[Event]) -> list[Event]:
        """Persist events for one session and fan them out to runtime sinks.

        Restricted to the ``workflow.`` and ``custom.`` namespaces: runtime
        event namespaces encode Cayu-owned lifecycle and accounting evidence,
        so application callers must not forge them even though every accepted
        batch uses the same durable budget/sink handoff.
        """
        _validate_workflow_event_batch(events, allow_cayu_internal=False)
        emitted = await self._event_writer.emit_many(session_id, events)
        return [await self._project_emitted_event_for_public_api(event) for event in emitted]


def _validate_workflow_event_batch(
    events: list[Event],
    *,
    allow_cayu_internal: bool,
) -> None:
    if type(events) is not list:
        raise TypeError("Runtime events must be a list.")
    for event in events:
        if type(event) is not Event:
            raise TypeError("Runtime events must be Event instances.")
        event_type = str(event.type)
        if not event_type.startswith(("workflow.", "custom.")):
            raise ValueError(
                "emit_events only accepts workflow. or custom. namespace "
                f"events; got {event_type!r}."
            )
        if not allow_cayu_internal and event_type.startswith("custom."):
            validate_public_custom_event_type(event_type)


def _copy_registered_tool(tool: runtime_records.RegisteredTool) -> runtime_records.RegisteredTool:
    return runtime_records.RegisteredTool(
        name=tool.name,
        description=tool.description,
        schema=deepcopy(tool.schema),
        parallel_safe=tool.parallel_safe,
        effect=tool.effect,
        publish_arguments=tool.publish_arguments,
        workspace_mutation=tool.workspace_mutation,
        execution_profile_identity=copy_execution_profile_behavior_identity(
            tool.execution_profile_identity
        ),
        command_policy_execution_profile_identity=copy_execution_profile_behavior_identity(
            tool.command_policy_execution_profile_identity
        ),
        tool=tool.tool,
        child_session_recovery=tool.child_session_recovery,
        durable_tool_recovery=tool.durable_tool_recovery,
    )


def _registration_site() -> tuple[str | None, str | None]:
    """Capture the public call site without retaining a frame or live object."""

    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame is not None and frame.f_back is not None else None
    try:
        if caller is None:
            return None, None
        module = caller.f_globals.get("__name__")
        symbol = caller.f_code.co_qualname
        qualified = f"{module}:{symbol}" if isinstance(module, str) else symbol
        return caller.f_code.co_filename, qualified
    finally:
        del frame
        del caller


def _validate_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


def _validate_optional_positive_seconds(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    if type(value) not in {int, float}:
        raise TypeError(f"{field_name} must be a number or None.")
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return float(value)


def _snapshot_context_behavior_execution_profile_identities(
    context_policy: ContextPolicy,
    context_overflow_policy: ContextPolicy | None,
    *,
    redactor: SecretRedactor,
) -> Mapping[int, ExecutionProfileBehaviorIdentity | None]:
    """Copy declarations reachable through Cayu-owned context wrappers."""

    from cayu.runtime.context import (
        CheckpointCompactionContextPolicy,
        ModelCompactor,
        PromptCacheCompactor,
        UsageTriggeredContextPolicy,
    )
    from cayu.runtime.memory_context import AutomaticRecallContextPolicy

    snapshots: dict[int, ExecutionProfileBehaviorIdentity | None] = {}

    def visit(policy: ContextPolicy, *, field_name: str) -> None:
        policy_id = id(policy)
        if policy_id in snapshots:
            return
        snapshots[policy_id] = copy_secret_free_execution_profile_behavior_identity(
            policy.execution_profile_identity,
            redactor=redactor,
            field_name=f"{field_name}.execution_profile_identity",
        )
        if type(policy) is AutomaticRecallContextPolicy:
            visit(policy.base_policy, field_name=f"{field_name}.base_policy")
            return
        if type(policy) is UsageTriggeredContextPolicy:
            visit(policy.base_policy, field_name=f"{field_name}.base_policy")
            visit(policy.triggered_policy, field_name=f"{field_name}.triggered_policy")
            return
        if type(policy) is not CheckpointCompactionContextPolicy:
            return

        visit_compactor(
            policy.compactor,
            field_name=f"{field_name}.compactor",
        )

    def visit_compactor(compactor: object, *, field_name: str) -> None:
        compactor_id = id(compactor)
        if compactor_id in snapshots:
            return
        snapshots[compactor_id] = copy_secret_free_execution_profile_behavior_identity(
            getattr(compactor, "execution_profile_identity", None),
            redactor=redactor,
            field_name=f"{field_name}.execution_profile_identity",
        )
        if type(compactor) is ModelCompactor:
            provider = compactor.provider
            snapshots[id(provider)] = copy_secret_free_execution_profile_behavior_identity(
                provider.execution_profile_identity,
                redactor=redactor,
                field_name=f"{field_name}.provider.execution_profile_identity",
            )
            return
        if type(compactor) is PromptCacheCompactor:
            provider = compactor.provider
            snapshots[id(provider)] = copy_secret_free_execution_profile_behavior_identity(
                provider.execution_profile_identity,
                redactor=redactor,
                field_name=f"{field_name}.provider.execution_profile_identity",
            )
            visit_compactor(
                compactor._fallback,
                field_name=f"{field_name}.fallback_compactor",
            )

    visit(context_policy, field_name="context_policy")
    if context_overflow_policy is not None:
        visit(context_overflow_policy, field_name="context_overflow_policy")
    return MappingProxyType(snapshots)


def _validate_provider_model_patterns(value: Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str | bytes):
        raise TypeError("Provider model_patterns must be an iterable of strings.")
    try:
        patterns = tuple(value)
    except TypeError as exc:
        raise TypeError("Provider model_patterns must be an iterable of strings.") from exc
    return tuple(
        require_clean_nonblank(pattern, f"model_patterns[{index}]")
        for index, pattern in enumerate(patterns)
    )


def _validate_registered_tool(
    tool: Tool,
    *,
    redactor: SecretRedactor,
) -> runtime_records.RegisteredTool:
    spec = getattr(tool, "spec", None)
    if type(spec) is not ToolSpec:
        raise TypeError("Agent tools must define ToolSpec instances.")
    name = validate_application_tool_name(require_clean_nonblank(spec.name, "name"))
    if not inspect.iscoroutinefunction(tool.run):
        raise TypeError(
            f"{type(tool).__name__}.run must be declared with `async def` and return a ToolResult."
        )
    schema = copy_json_value(tool.schema, "schema")
    if type(schema) is not dict:
        raise TypeError(f"{type(tool).__name__}.schema must return a JSON Schema object.")
    publish_arguments = tool._publish_arguments
    if type(publish_arguments) is not bool:
        raise TypeError(f"{type(tool).__name__} argument publication policy must be a bool.")
    validated_spec = ToolSpec(
        name=name,
        description=spec.description,
        input_schema=schema,
        parallel_safe=spec.parallel_safe,
        effect=spec.effect,
        workspace_mutation=spec.workspace_mutation,
    )
    command_policy = getattr(tool, "command_policy", None)
    return runtime_records.RegisteredTool(
        name=validated_spec.name,
        description=validated_spec.description,
        schema=validated_spec.input_schema,
        parallel_safe=validated_spec.parallel_safe,
        effect=validated_spec.effect,
        publish_arguments=publish_arguments,
        workspace_mutation=validated_spec.workspace_mutation,
        execution_profile_identity=copy_secret_free_execution_profile_behavior_identity(
            tool.execution_profile_identity,
            redactor=redactor,
            field_name=f"tools[{name!r}].execution_profile_identity",
        ),
        command_policy_execution_profile_identity=(
            copy_secret_free_execution_profile_behavior_identity(
                None
                if command_policy is None
                else getattr(command_policy, "execution_profile_identity", None),
                redactor=redactor,
                field_name=f"tools[{name!r}].command_policy.execution_profile_identity",
            )
        ),
        tool=tool,
        child_session_recovery=(
            tool if isinstance(tool, runtime_records.ChildSessionRecoveryMatcher) else None
        ),
        durable_tool_recovery=(tool if isinstance(tool, DurableToolRecovery) else None),
    )


def _registered_tool_descriptor(
    tool: runtime_records.RegisteredTool,
) -> ToolDescriptor:
    """Derive one canonical callable-free descriptor from admitted registration state."""

    registered_tool = tool.tool
    if isinstance(registered_tool, McpToolAdapter):
        binding = registered_tool._manifest_binding
        provenance = ToolDescriptorProvenance(
            kind="mcp",
            source_id=registered_tool.toolset.manifest_identity,
            source_tool_fingerprint=mcp_source_tool_fingerprint(binding.manifest_mcp_name),
            source_contract_fingerprint=binding.manifest_contract_hash,
        )
    else:
        provenance = ToolDescriptorProvenance()
    return build_tool_descriptor(
        name=tool.name,
        description=tool.description,
        input_schema=tool.schema,
        parallel_safe=tool.parallel_safe,
        effect=tool.effect,
        publishes_arguments=tool.publish_arguments,
        workspace_mutation=tool.workspace_mutation,
        provenance=provenance,
    )


def _validate_agent_spec(spec: AgentSpec) -> AgentSpec:
    if type(spec) is not AgentSpec:
        raise TypeError("Agent registration requires an AgentSpec.")
    return AgentSpec(
        name=spec.name,
        model=spec.model,
        provider_name=spec.provider_name,
        system_prompt=spec.system_prompt,
        workflow_tool_names=spec.workflow_tool_names,
        authoring_state=spec.authoring_state,
        metadata=copy_json_value(spec.metadata, "metadata"),
        provider_options=copy_json_value(spec.provider_options, "provider_options"),
        thinking=spec.thinking,
    )


def _validate_environment_spec(
    spec: EnvironmentSpec,
    *,
    redactor: SecretRedactor,
) -> EnvironmentSpec:
    if type(spec) is not EnvironmentSpec:
        raise TypeError("Environment registration requires an EnvironmentSpec.")
    if type(spec.name) is not str:
        raise ValueError("`name` must be a string.")
    return EnvironmentSpec(
        name=spec.name,
        metadata=copy_json_value(spec.metadata, "metadata"),
        execution_profile_identity=copy_secret_free_execution_profile_behavior_identity(
            spec.execution_profile_identity,
            redactor=redactor,
            field_name="environment_spec.execution_profile_identity",
        ),
    )


def _work_contract_contains_secret_public_identity(
    contract: WorkContract,
    redactor: SecretRedactor,
) -> bool:
    public_identities = (
        contract.contract_id,
        contract.verifier.verifier_id,
        contract.verifier.version,
        contract.verifier.configuration_fingerprint,
        contract.result_resolver.resolver_id,
        contract.result_resolver.version,
        contract.result_resolver.configuration_fingerprint,
        *(criterion.criterion_id for criterion in contract.criteria),
        *(constraint.constraint_id for constraint in contract.constraints),
        *(requirement.requirement_id for requirement in contract.evidence_requirements),
    )
    return any(redactor.redact_text(value) != value for value in public_identities)


async def _publish_public_work_contract(
    task_store: TaskStore,
    contract: WorkContract,
    *,
    redactor: SecretRedactor,
) -> tuple[WorkContract | None, BaseException | None]:
    """Capture a publication conflict without exporting its sensitive store traceback."""

    outcome = await capture_task_store_operation(
        lambda: task_store.publish_work_contract(contract),
        operation_name="Work-contract publication",
        redactor=redactor,
        mutation_store=task_store,
        mutation_method_name="publish_work_contract",
    )
    if type(outcome.failure) is WorkContractConflict:
        return None, WorkContractConflict(
            "Task store rejected the work-contract publication because its durable identity "
            "conflicts with existing state."
        )
    return outcome.result, outcome.failure


async def _load_public_work_contract(
    task_store: TaskStore,
    reference: WorkContractRef,
    *,
    redactor: SecretRedactor,
) -> tuple[WorkContract | None, BaseException | None]:
    """Capture a lookup conflict without exporting the stored contract traceback."""

    outcome = await capture_task_store_operation(
        lambda: task_store.load_work_contract(reference),
        operation_name="Work-contract lookup",
        redactor=redactor,
    )
    if type(outcome.failure) is WorkContractConflict:
        return None, WorkContractConflict(
            "Task store rejected the work-contract lookup because its durable identity "
            "conflicts with the requested reference."
        )
    return outcome.result, outcome.failure


async def _create_public_contracted_task(
    task_store: TaskStore,
    request: TaskCreate,
    *,
    redactor: SecretRedactor,
) -> tuple[Task | None, BaseException | None]:
    """Capture contract-binding conflicts without exporting the task payload traceback."""

    outcome = await capture_task_store_operation(
        lambda: task_store.create_task(request),
        operation_name="Contracted task creation",
        redactor=redactor,
        mutation_store=task_store,
        mutation_method_name="create_task",
    )
    if type(outcome.failure) is WorkContractConflict:
        return None, WorkContractConflict(
            "Task store rejected the contracted task because its work contract conflicts "
            "with durable state."
        )
    if type(outcome.failure) is WorkCompletionConflict:
        return None, WorkCompletionConflict(
            "Task store rejected the contracted task because its session binding conflicts "
            "with durable work authority."
        )
    return outcome.result, outcome.failure


async def _load_public_task_invocation_snapshot(
    task_store: TaskStore,
    task_id: str,
    *,
    redactor: SecretRedactor,
) -> tuple[TaskInvocationSnapshot | None, BaseException | None]:
    """Read back one contracted task's durable provenance without leaking extension state."""

    outcome = await capture_task_store_operation(
        lambda: task_store.load_invocation_snapshot(task_id),
        operation_name="Contracted task invocation lookup",
        redactor=redactor,
    )
    return outcome.result, outcome.failure


def _validated_public_work_contract(
    draft: WorkContractDraft,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[WorkContract]:
    """Validate a caller-owned draft without retaining a rejected model traceback."""

    return capture_sensitive_validation(
        lambda: work_contract_from_draft(draft),
        operation_name="Work-contract request validation",
        redactor=redactor,
    )


def _copied_public_work_contract(
    value: object,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[WorkContract]:
    """Copy one extension-returned contract behind a detached validation boundary."""

    if type(value) is not WorkContract:
        return TaskStoreOperationOutcome()
    return capture_sensitive_validation(
        lambda: copy_work_contract(value),
        operation_name="Work-contract result validation",
        redactor=redactor,
    )


def _copied_public_work_contract_ref(
    value: WorkContractRef,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[WorkContractRef]:
    """Copy one caller-owned reference behind a detached validation boundary."""

    return capture_sensitive_validation(
        lambda: cast("WorkContractRef", copy_work_contract_ref(value)),
        operation_name="Work-contract reference validation",
        redactor=redactor,
    )


def _copied_public_task_create(
    value: TaskCreate,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[TaskCreate]:
    """Copy one caller-owned task request behind a detached validation boundary."""

    return capture_sensitive_validation(
        lambda: copy_task_create(value),
        operation_name="Task request validation",
        redactor=redactor,
    )


def _copied_public_task(
    value: object,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[Task]:
    """Copy one extension-returned task behind a detached validation boundary."""

    if type(value) is not Task:
        return TaskStoreOperationOutcome()
    return capture_sensitive_validation(
        lambda: copy_task(value),
        operation_name="Task result validation",
        redactor=redactor,
    )


def _copied_public_task_invocation_snapshot(
    value: object,
    *,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[TaskInvocationSnapshot]:
    """Copy extension-returned task provenance behind a detached validation boundary."""

    if type(value) is not TaskInvocationSnapshot:
        return TaskStoreOperationOutcome()
    return capture_sensitive_validation(
        lambda: TaskInvocationSnapshot(
            id=value.id,
            session_id=value.session_id,
            session_instance_id=value.session_instance_id,
            invocation=value.invocation,
        ),
        operation_name="Task invocation result validation",
        redactor=redactor,
    )


def _contracted_task_invocation_matches_request(
    *,
    task: Task,
    request: TaskCreate,
    invocation_snapshot: TaskInvocationSnapshot,
    parent_invocation_snapshot: TaskInvocationSnapshot | None,
    redactor: SecretRedactor,
) -> bool:
    """Authenticate durable provenance and every request-owned invocation field."""

    invocation = task.invocation
    if (
        invocation_snapshot.id != task.id
        or invocation_snapshot.session_id != task.session_id
        or invocation_snapshot.session_instance_id != task.session_instance_id
        or invocation_snapshot.invocation != invocation
        or invocation.source
        is not (request._runtime_invocation_source or TaskExecutionSource.SDK_TASK)
    ):
        return False
    if redactor.redact_text(task.id) != task.id or invocation_contains_secret_public_identity(
        invocation,
        redactor,
    ):
        return False
    session_binding = request._runtime_session_binding
    if session_binding is not None:
        matches_session = (
            task.session_instance_id == session_binding.session_instance_id
            and invocation_snapshot.session_instance_id == session_binding.session_instance_id
            and invocation.origin == session_binding.invocation.origin
            and invocation.root_invocation_id == session_binding.invocation.root_invocation_id
            and invocation.root_session_id == session_binding.invocation.root_session_id
        )
        if not matches_session:
            return False
        if request.parent_task_id is None:
            return True
        return (
            parent_invocation_snapshot is not None
            and parent_invocation_snapshot.id == request.parent_task_id
            and parent_invocation_snapshot.invocation.origin == invocation.origin
            and parent_invocation_snapshot.invocation.root_invocation_id
            == invocation.root_invocation_id
        )
    if request.parent_task_id is not None:
        return (
            parent_invocation_snapshot is not None
            and parent_invocation_snapshot.id == request.parent_task_id
            and parent_invocation_snapshot.invocation.origin == invocation.origin
            and parent_invocation_snapshot.invocation.root_invocation_id
            == invocation.root_invocation_id
            and parent_invocation_snapshot.invocation.root_session_id == invocation.root_session_id
        )
    if request._verified_invocation_origin is not None:
        expected_origin = request._verified_invocation_origin
    elif request.invocation_origin is not None:
        expected_origin = InvocationOrigin(
            trust=InvocationOriginTrust.HOST_ASSERTED,
            subject=request.invocation_origin.subject,
            tenant=request.invocation_origin.tenant,
        )
    else:
        expected_origin = InvocationOrigin(trust=InvocationOriginTrust.UNATTRIBUTED)
    return invocation.origin == expected_origin and invocation.root_session_id == request.session_id


def _contracted_task_creation_result_matches_request(
    *,
    task: Task,
    request: TaskCreate,
    invocation_snapshot: TaskInvocationSnapshot | None,
    parent_invocation_snapshot: TaskInvocationSnapshot | None,
    redactor: SecretRedactor,
) -> bool:
    """Authenticate a custom store's contracted-create result before publication."""

    if invocation_snapshot is None or not _contracted_task_invocation_matches_request(
        task=task,
        request=request,
        invocation_snapshot=invocation_snapshot,
        parent_invocation_snapshot=parent_invocation_snapshot,
        redactor=redactor,
    ):
        return False
    if request.task_id is not None and task.id != request.task_id:
        return False
    if (
        task.type != request.type
        or task.title != request.title
        or task.description != request.description
        or task.session_id != request.session_id
        or task.parent_task_id != request.parent_task_id
        or task.assigned_agent_name != request.assigned_agent_name
        or task.available_at != request.available_at
        or task.input != request.input
        or task.metadata != request.metadata
        or task.work_contract != request.work_contract
    ):
        return False
    return (
        task.status is TaskStatus.PENDING
        and task.worker_id is None
        and task.lease_expires_at is None
        and task.status_reason is None
        and task.status_payload is None
        and task.result is None
        and task.error is None
        and task.started_at is None
        and task.completed_at is None
    )


def _validate_tool_approval_request(request: ToolApprovalRequest) -> ToolApprovalRequest:
    return copy_tool_approval_request(request)


def _validate_tool_approval_recovery_request(
    request: ToolApprovalRecoveryRequest,
) -> ToolApprovalRecoveryRequest:
    return copy_tool_approval_recovery_request(request)


def _recovery_task_event(request: RecoveryTaskEventRequest) -> Event:
    return _task_event(
        event_type=request.event_type,
        task=request.task,
        session=request.session,
        registered_agent=request.registered_agent,
        registered_environment=request.registered_environment,
    )


def _artifact_store(registered_environment: runtime_records.RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.artifact_store


def _validate_dispatch_handle_for_request(
    *,
    handle: DispatchHandle,
    request: DispatchRequest,
) -> None:
    if type(handle) is not DispatchHandle:
        raise TypeError("Dispatcher must return a DispatchHandle.")
    mismatches = []
    if handle.dispatch_id != request.dispatch_id:
        mismatches.append("dispatch_id")
    if handle.session_id != request.session_id:
        mismatches.append("session_id")
    if handle.task_id != request.task_id:
        mismatches.append("task_id")
    if mismatches:
        fields = ", ".join(mismatches)
        raise ValueError(f"Dispatcher returned a handle for the wrong request fields: {fields}.")


def _validate_runtime_hooks(
    hooks: Iterable[RuntimeHook] | None,
    *,
    field_name: str,
    redactor: SecretRedactor,
) -> tuple[runtime_records.RegisteredRuntimeHook, ...]:
    if hooks is None:
        return ()
    if isinstance(hooks, str | bytes):
        raise TypeError(f"{field_name} must be an iterable of RuntimeHook instances.")
    try:
        hook_list = list(hooks)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of RuntimeHook instances.") from exc
    registered_hooks: list[runtime_records.RegisteredRuntimeHook] = []
    for index, hook in enumerate(hook_list):
        if not isinstance(hook, RuntimeHook):
            raise TypeError(f"{field_name} must contain RuntimeHook instances.")
        registered_hooks.append(
            runtime_records.RegisteredRuntimeHook(
                name=hook.name,
                execution_profile_identity=copy_secret_free_execution_profile_behavior_identity(
                    hook.execution_profile_identity,
                    redactor=redactor,
                    field_name=(f"{field_name}[{index}].execution_profile_identity"),
                ),
                hook=hook,
            )
        )
    return tuple(registered_hooks)


def _validate_event_watchers(watchers: Iterable[EventWatcher]) -> tuple[EventWatcher, ...]:
    if isinstance(watchers, str | bytes):
        raise TypeError("watchers must be an iterable of EventWatcher instances.")
    try:
        watcher_list = list(watchers)
    except TypeError as exc:
        raise TypeError("watchers must be an iterable of EventWatcher instances.") from exc
    names: set[str] = set()
    for watcher in watcher_list:
        if type(watcher) is not EventWatcher:
            raise TypeError("watchers must contain EventWatcher instances.")
        if watcher.name in names:
            raise ValueError("Duplicate event watcher name.")
        names.add(watcher.name)
    return tuple(watcher_list)
