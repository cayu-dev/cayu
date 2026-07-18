from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import mimetypes
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from fnmatch import fnmatchcase
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from cayu._validation import (
    copy_json_value,
    copy_label_map,
    require_clean_nonblank,
    require_nonblank,
    require_unicode_scalar_text,
)
from cayu.artifacts import (
    DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
    DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST,
    DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES,
    ArtifactScope,
    FileAttachmentKind,
    file_attachment,
    validate_file_attachment_bytes,
    validate_file_attachment_content_type,
)
from cayu.core.agents import AgentSpec
from cayu.core.billing import (
    UNRESOLVED_BILLING_IDENTITY,
    BillingIdentity,
    BillingIdentityState,
    ResolvedBillingIdentity,
    resolved_billing_identity,
)
from cayu.core.events import Event, EventType, copy_event
from cayu.core.messages import (
    FilePart,
    Message,
    MessageRole,
    ToolCallPart,
    ToolResultPart,
)
from cayu.core.thinking import ThinkingConfig
from cayu.core.tools import (
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu.environments import (
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentSpec,
    copy_bound_workspace,
    copy_environment,
)
from cayu.providers import (
    ModelProvider,
)
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
    exception_failure_payload,
    render_initial_system_prompt,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._interruption_coordinator import (
    BackgroundInterruptionCoordinator,
    _copy_interruption_cascade_retry_request,
    _interruption_cascade_marker_datetime,
    _interruption_cascade_retry_event_payload,
    _is_background_subagent_session,
    interruption_cascade_lease_seconds,
    interruption_cascade_suppressed,
    suppress_interruption_cascade,
)
from cayu.runtime._model_step_executor import (
    ModelStepBudgetEvaluationRequest,
    ModelStepBudgetReservationFailureRequest,
    ModelStepExecutor,
    ModelStepFlowOutcome,
    ModelStepLimitEvaluationRequest,
    _session_agent_spec,
)
from cayu.runtime._recovery_coordinator import (
    _DEFAULT_APPROVAL_MAX_STEPS,
    RecoveryCoordinator,
    RecoveryLimitStopRequest,
    RecoverySessionRunRequest,
    RecoveryTaskEventRequest,
    RecoveryTerminalEventRequest,
    _effective_approval_structured_output,
    _effective_user_input_structured_output,
    _run_recovery_cleanup_steps,
)
from cayu.runtime._run_limits import (
    BudgetEvaluation,
    BudgetReservationLeaseLost,
    BudgetReservationLeaseLostBeforeModelDispatch,
    BudgetStepReservation,
    LimitEvaluation,
    RunLimitController,
    RunLimitGate,
    SessionUsageTracker,
    add_budget_failure_note,
    budget_limit_reached_payload,
)
from cayu.runtime._session_control import (
    INTERRUPT_REQUESTED_SESSION_STATUSES,
    ActiveSessionRun,
    SessionControl,
    SessionInterruptedByRequest,
    clear_current_task_cancellation,
    interruption_request_id_from_payload,
)
from cayu.runtime._session_queries import query_all_event_records
from cayu.runtime._tool_round_executor import (
    InterruptedToolRoundRequest,
    ToolApprovalRequired,
    ToolRoundExecutor,
    ToolRoundLimitRequest,
    UserInputRequired,
    ordered_tool_result_messages,
)
from cayu.runtime.approvals import (
    PendingToolApproval,
    PendingToolCallApproval,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
    copy_tool_approval_recovery_request,
    copy_tool_approval_request,
    resolution_actor_payload,
)
from cayu.runtime.budgets import (
    BudgetCheck,
    BudgetLedger,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservationResult,
    BudgetStore,
    InMemoryBudgetLedger,
    SessionBudgetStore,
    budget_check_payload,
    budget_limits_for_session,
    budget_reconciliation_payload,
    budget_reservation_payload,
    copy_budget_policy,
    copy_request_budget_limits,
    has_deferred_contextual_price,
    request_budget_limits_for_session,
)
from cayu.runtime.context import (
    CheckpointCompactionContextPolicy,
    ContextBuildError,
    ContextBuildResult,
    ContextCompactionTelemetry,
    ContextPolicy,
    ContextRequest,
    DefaultContextPolicy,
    _automatic_compaction_dispatch_runner_scope,
    _compaction_model_completed_payload,
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
    InlineDispatcher,
    copy_dispatch_handle,
    copy_dispatch_request,
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
from cayu.runtime.hooks import (
    RuntimeHook,
    RuntimeHookContext,
    RuntimeHookPhase,
    _runtime_hook_supports_phase,
)
from cayu.runtime.hooks import (
    _runtime_hook_event as _build_runtime_hook_event,
)
from cayu.runtime.loop_policies import (
    BeforeStopAction,
    BeforeStopContext,
    BeforeStopDecision,
    LoopPolicy,
    copy_before_stop_decision,
    validate_loop_policies,
)
from cayu.runtime.manifest import AppManifest, describe_app
from cayu.runtime.mcp_manifest_policy import (
    McpManifestPolicy,
    copy_mcp_manifest_policy,
)
from cayu.runtime.model_steps import (
    AssistantStepResult,
    StepClassification,
    classify_assistant_step,
)
from cayu.runtime.retry_policy import (
    RetryPolicy,
    copy_retry_policy,
)
from cayu.runtime.sessions import (
    CompactSessionRequest,
    EnqueueSessionMessageRequest,
    EnqueueSessionMessageResult,
    EventQuery,
    EventRecord,
    ForkSessionRequest,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    IncompleteSessionsRecoveryRequest,
    InMemorySessionStore,
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
    Session,
    SessionIdentity,
    SessionMessageDeliveryBatch,
    SessionOperationPublication,
    SessionOrder,
    SessionQuery,
    SessionQueuedMessagesPending,
    SessionStatus,
    SessionStatusConflict,
    SessionStore,
    _activate_session_run_fence,
    _current_session_run_epoch,
    _deactivate_session_run_fence,
    copy_compact_session_request,
    copy_enqueue_session_message_request,
    copy_fork_session_request,
    copy_incomplete_session_recovery_request,
    copy_incomplete_sessions_recovery_request,
    copy_interrupt_session_request,
    copy_resume_request,
    copy_run_request,
)
from cayu.runtime.stop_policy import (
    RunLimits,
    StopDecision,
    StopLimit,
    copy_run_limits,
    has_run_limits,
)
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    NativeStructuredOutputUnsupported,
    StructuredOutputError,
    StructuredOutputSpec,
    StructuredOutputStrategy,
    StructuredOutputValidation,
    copy_structured_output_spec,
    structured_output_repair_lead,
    structured_output_repair_prompt,
    structured_output_tool_required_validation,
    validate_structured_output_text,
    validate_structured_output_tool_arguments,
)
from cayu.runtime.tasks import (
    Task,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    TaskStore,
    copy_task_create,
)
from cayu.runtime.tool_policy import (
    AllowAllToolPolicy,
    ToolPolicy,
    metadata_with_taint_labels,
    taint_labels_from_metadata,
)
from cayu.runtime.tool_rounds import (
    ToolRoundRecoveryRequest,
    copy_tool_round_recovery_request,
)
from cayu.runtime.usage import (
    USAGE_BEARING_EVENT_TYPES,
    CausalBudgetUsageSummary,
    SessionUsageSummary,
    UsageMetrics,
    causal_budget_usage_summary,
    session_usage_summary,
    usage_metrics_from_event_payload,
)
from cayu.runtime.user_input import (
    UserInputRecoveryRequest,
    UserInputResponse,
    copy_user_input_recovery_request,
    copy_user_input_response,
    pending_user_input_from_checkpoint,
)
from cayu.storage.memory import KnowledgeStore
from cayu.vaults import (
    SecretRedactor,
)

RegisteredAgent = runtime_records.RegisteredAgent
RegisteredEnvironment = runtime_records.RegisteredEnvironment


class _SessionCompactionReplay(Exception):
    def __init__(self, event_ids: Iterable[str]) -> None:
        self.event_ids = tuple(event_ids)
        super().__init__("Replay an existing durable session compaction outcome.")


class SessionCompactionAttemptSuperseded(RuntimeError):
    """A recovered compaction attempt owns the durable operation claim."""


logger = logging.getLogger(__name__)


# A crashed ordinary tool round can leave the session FAILED (in-process
# persistence error) or in a stale live status (process kill), so operator
# reconciliation accepts all of them; INTERRUPTED covers interrupt-adjacent
# shapes that preserved the pending round.
_TOOL_ROUND_RECOVERABLE_SESSION_STATUSES = {
    SessionStatus.RUNNING,
    SessionStatus.INTERRUPTING,
    SessionStatus.INTERRUPTED,
    SessionStatus.FAILED,
}
_RESUMABLE_SESSION_STATUSES = {
    SessionStatus.COMPLETED,
    SessionStatus.FAILED,
    SessionStatus.INTERRUPTED,
}
_FORKABLE_SESSION_STATUSES = _RESUMABLE_SESSION_STATUSES
_INTERRUPTIBLE_SESSION_STATUSES = {
    SessionStatus.PENDING,
    SessionStatus.RUNNING,
}
_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY = "pending_session_interrupt"
_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY = "pending_interruption_cascade"
_SESSION_OPERATIONS_CHECKPOINT_KEY = "session_operations"
_CONTEXT_COMPACTION_OPERATION_KIND = "context_compaction"
_SESSION_OPERATION_CLAIM_LEASE = timedelta(minutes=5)
_INTERRUPTION_TYPE_OPERATOR_REQUESTED = "operator_requested"
_INTERRUPTION_TYPE_RUNTIME_INTERRUPTED = "runtime_interrupted"
_INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED = "tool_approval_required"
_INTERRUPTION_TYPE_USER_INPUT_REQUIRED = "user_input_required"
_INTERRUPTION_TYPE_LIMIT_REACHED = "limit_reached"
_ABANDONED_RUN_REASON = "event_stream_closed"
DEFAULT_MAX_PARALLEL_TOOL_CALLS = 4


class CayuApp:
    """Application runtime for registered agents, providers, and session state."""

    def __init__(
        self,
        *,
        session_store: SessionStore | None = None,
        task_store: TaskStore | None = None,
        knowledge_store: KnowledgeStore | None = None,
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
        context_counting: ContextCountingConfig | None = None,
        event_sinks: Iterable[EventSink] | None = None,
        enable_logging: bool = True,
        secret_redactor: SecretRedactor | None = None,
        max_file_attachment_bytes: int = DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
        max_total_file_attachment_bytes: int = DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES,
        max_file_attachments_per_request: int = DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST,
        tool_timeout_seconds: float | None = None,
        max_parallel_tool_calls: int = DEFAULT_MAX_PARALLEL_TOOL_CALLS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if session_store is not None and not isinstance(session_store, SessionStore):
            raise TypeError("session_store must be a SessionStore.")
        if task_store is not None and not isinstance(task_store, TaskStore):
            raise TypeError("task_store must be a TaskStore.")
        if knowledge_store is not None and not isinstance(knowledge_store, KnowledgeStore):
            raise TypeError("knowledge_store must be a KnowledgeStore.")
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
        if type(enable_logging) is not bool:
            raise TypeError("enable_logging must be a bool.")
        hooks = _validate_runtime_hooks(runtime_hooks, field_name="runtime_hooks")
        policies = validate_loop_policies(loop_policies, field_name="loop_policies")
        # Wall-clock seam for time-based approval expiry (tests inject a fake).
        self._clock = _clock_or_utc_now(clock)
        manifest_policy = copy_mcp_manifest_policy(mcp_manifest_policy)
        context_counting_config = copy_context_counting_config(context_counting)
        resolved_secret_redactor = (
            secret_redactor if secret_redactor is not None else SecretRedactor()
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
        self.session_store = session_store if session_store is not None else InMemorySessionStore()
        self.task_store = task_store
        self.knowledge_store = knowledge_store
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
        self._default_retry_policy = copy_retry_policy(retry_policy)
        self._runtime_hooks = tuple(hooks)
        self._loop_policies = tuple(policies)
        self._mcp_manifest_policy = manifest_policy
        self._context_counting = context_counting_config
        self._event_sinks = tuple(sinks)
        self._event_writer = RuntimeEventWriter(
            session_store=self.session_store,
            budget_store=self.budget_store,
            event_sinks=self._event_sinks,
        )
        self._environment_lifecycle = EnvironmentLifecycle(
            session_store=self.session_store,
            event_writer=self._event_writer,
            checkpoint_transform=_replace_checkpoint_preserving_runtime_state,
        )
        self._run_limit_controller = RunLimitController(
            session_store=self.session_store,
            budget_store=self.budget_store,
            budget_ledger=self.budget_ledger,
            event_writer=self._event_writer,
            clock=self._clock,
        )
        self._agents: dict[str, runtime_records.RegisteredAgentState] = {}
        self._providers: dict[str, runtime_records.RegisteredProvider] = {}
        self._environments: dict[str, runtime_records.RegisteredEnvironment] = {}
        self._default_provider_name: str | None = None
        self._default_environment_name: str | None = None
        self._session_control = SessionControl[SessionUsageTracker](
            session_store=self.session_store
        )
        self._model_step_executor = ModelStepExecutor(
            session_store=self.session_store,
            event_writer=self._event_writer,
            session_control=self._session_control,
            run_limit_controller=self._run_limit_controller,
            context_counting=self._context_counting,
            max_file_attachment_bytes=self._max_file_attachment_bytes,
            max_total_file_attachment_bytes=self._max_total_file_attachment_bytes,
            max_file_attachments_per_request=self._max_file_attachments_per_request,
            checkpoint_session=self._environment_lifecycle.checkpoint_preserving_runtime_state,
            apply_budget_evaluation=self._apply_model_step_budget_evaluation,
            apply_limit_evaluation=self._apply_model_step_limit_evaluation,
            stop_for_budget_reservation_failure=(
                self._stop_for_model_step_budget_reservation_failure
            ),
        )
        self._tool_round_executor = ToolRoundExecutor(
            session_store=self.session_store,
            event_writer=self._event_writer,
            session_control=self._session_control,
            hook_runtime=self,
            runtime_hooks=self._runtime_hooks,
            mcp_manifest_policy=self._mcp_manifest_policy,
            secret_redactor=self._secret_redactor,
            tool_timeout_seconds=self._tool_timeout_seconds,
            max_parallel_tool_calls=self._max_parallel_tool_calls,
            clock=self._clock,
            checkpoint_transform=_replace_checkpoint_preserving_runtime_state,
            apply_limit_evaluation=self._apply_tool_round_limit,
            close_interrupted_round=self._close_tool_round_after_interrupt,
        )
        self._recovery_coordinator = RecoveryCoordinator(
            session_store=self.session_store,
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
            finalize_abandoned_session_by_id=self._finalize_abandoned_session_by_id,
            stop_session_for_limit_reached=self._stop_recovery_session_for_limit_reached,
            task_event=_recovery_task_event,
        )
        self._background_interruption_coordinator = BackgroundInterruptionCoordinator(
            session_store=self.session_store,
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
        )

    def redact_json(self, value: Any) -> Any:
        """Return a JSON-compatible value with configured secret values redacted."""
        return self._secret_redactor.redact_json(value)

    def describe(self, *, project_root: str | Path | None = None) -> AppManifest:
        """Return this application's deterministic public manifest.

        Description is structural only: it never invokes providers, tools,
        environment factories, stores, workers, watchers, or recovery paths.
        """

        return describe_app(self, project_root=project_root)

    async def drain_background_interruptions(self, *, timeout_s: float = 10.0) -> bool:
        """Wait for accepted background interruption cascades to finish.

        Returns ``False`` when the bounded wait expires. In-memory coordinators
        and workers are then cancelled; their durable parent markers remain for
        the next process to recover.
        """
        return await self._background_interruption_coordinator.drain(timeout_s=timeout_s)

    async def resume_pending_interruption_cascades(
        self,
        *,
        interrupting_inactive_before: datetime | None = None,
    ) -> int:
        """Resume durable descendant interruption work left by an earlier process.

        Both ``interrupting`` and ``interrupted`` parents are inspected. An
        ``interrupting`` parent is finalized only when
        ``interrupting_inactive_before`` is supplied and the store can fence that
        inactive run. Work remains checkpointed until traversal succeeds, so
        another restart can retry it safely. Returns the number of roots scheduled.
        """

        if interrupting_inactive_before is not None:
            if (
                interrupting_inactive_before.tzinfo is None
                or interrupting_inactive_before.utcoffset() is None
            ):
                raise ValueError("interrupting_inactive_before must be timezone-aware.")
            interrupting_inactive_before = interrupting_inactive_before.astimezone(UTC)

        scheduled = 0
        admitted_parent_ids: set[str] = set()
        for status in (SessionStatus.INTERRUPTING, SessionStatus.INTERRUPTED):
            if status == SessionStatus.INTERRUPTING and interrupting_inactive_before is None:
                continue
            cursor: str | None = None
            while True:
                result = await self.session_store.list_sessions_with_pending_interruption_cascade(
                    SessionQuery(
                        status=status,
                        last_activity_before=(
                            interrupting_inactive_before
                            if status == SessionStatus.INTERRUPTING
                            else None
                        ),
                        limit=1000,
                        cursor=cursor,
                        order_by=SessionOrder.CREATED_AT_ASC,
                    )
                )
                for session in result.sessions:
                    if session.id in admitted_parent_ids:
                        continue
                    try:
                        marker = await self._load_pending_interruption_cascade(session.id)
                    except (TypeError, ValueError) as exc:
                        logger.warning(
                            "Could not resume invalid interruption cascade checkpoint for %s: %s",
                            session.id,
                            exc,
                        )
                        continue
                    if marker is None:
                        continue
                    already_scheduled = self._background_interruption_coordinator.is_admitted(
                        session.id
                    )
                    if session.status == SessionStatus.INTERRUPTING:
                        if interrupting_inactive_before is None:
                            continue
                        with suppress_interruption_cascade():
                            recovery = await self._recover_incomplete_session_scoped(
                                session=session,
                                inactive_before=interrupting_inactive_before,
                                reason="interruption_cascade_startup_recovery",
                                metadata={"source": "resume_pending_interruption_cascades"},
                            )
                        session = await self._require_session(session.id)
                        if session.status != SessionStatus.INTERRUPTED:
                            logger.warning(
                                "Could not finalize interruption cascade parent %s during "
                                "startup recovery: %s",
                                session.id,
                                recovery.message,
                            )
                            continue
                    admitted_parent_ids.add(session.id)
                    self._schedule_background_interruption_cascade(
                        parent_session_id=session.id,
                        interrupt_payload=marker["interrupt_payload"],
                        create_if_missing=False,
                    )
                    if not already_scheduled:
                        scheduled += 1
                cursor = result.next_cursor
                if cursor is None:
                    break
        return scheduled

    async def interruption_cascade_status(self, session_id: str) -> str:
        """Return the public control-plane state of a session's durable cascade."""

        session_id = require_clean_nonblank(session_id, "session_id")
        marker = await self.session_store.load_interruption_cascade_marker(session_id)
        if marker is None:
            return "none"
        if type(marker) is not dict:
            return "failed"
        attempt_id = marker.get("attempt_id")
        interrupt_payload = marker.get("interrupt_payload")
        generation = marker.get("generation", 0)
        if (
            type(attempt_id) is not str
            or not attempt_id.strip()
            or type(interrupt_payload) is not dict
            or type(generation) is not int
            or generation < 0
        ):
            return "failed"
        failure_recorded = marker.get("failure_recorded", False)
        if type(failure_recorded) is not bool:
            return "failed"
        try:
            claim_id = marker.get("claim_id")
            claim_expires_at = _interruption_cascade_marker_datetime(
                marker,
                "claim_expires_at",
            )
            if claim_id is not None:
                if (
                    type(claim_id) is not str
                    or not claim_id.strip()
                    or claim_expires_at is None
                    or generation < 1
                ):
                    return "failed"
            elif claim_expires_at is not None:
                return "failed"
            created_at = _interruption_cascade_marker_datetime(marker, "created_at")
        except ValueError:
            return "failed"
        if self._background_interruption_coordinator.is_pending(session_id):
            return "pending"
        if claim_id is not None:
            if claim_expires_at is None:
                return "failed"
            return "pending" if claim_expires_at > self._clock() else "failed"
        if failure_recorded:
            return "failed"
        if created_at is None:
            return "failed"
        unclaimed_grace = timedelta(seconds=interruption_cascade_lease_seconds())
        return "pending" if created_at + unclaimed_grace > self._clock() else "failed"

    def _schedule_background_interruption_cascade(
        self,
        *,
        parent_session_id: str,
        interrupt_payload: dict[str, Any],
        create_if_missing: bool,
        retry_request: dict[str, Any] | None = None,
        allow_during_drain: bool = False,
    ) -> asyncio.Task[None] | None:
        return self._background_interruption_coordinator.schedule(
            parent_session_id=parent_session_id,
            interrupt_payload=interrupt_payload,
            create_if_missing=create_if_missing,
            retry_request=retry_request,
            allow_during_drain=allow_during_drain,
        )

    def _defer_background_interruption_cascade(
        self,
        *,
        parent_session_id: str,
        interrupt_payload: dict[str, Any],
        retry_at: datetime,
        drain_required: bool,
        retry_request: dict[str, Any] | None,
    ) -> None:
        self._background_interruption_coordinator.defer(
            parent_session_id=parent_session_id,
            interrupt_payload=interrupt_payload,
            retry_at=retry_at,
            drain_required=drain_required,
            retry_request=retry_request,
        )

    def register_agent(
        self,
        spec: AgentSpec,
        *,
        tools: Iterable[Tool] | None = None,
        context_policy: ContextPolicy | None = None,
        context_overflow_policy: ContextPolicy | None = None,
        tool_policy: ToolPolicy | None = None,
        runtime_hooks: Iterable[RuntimeHook] | None = None,
        loop_policies: Iterable[LoopPolicy] | None = None,
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
        if tool_policy is None:
            stored_tool_policy = AllowAllToolPolicy()
        elif isinstance(tool_policy, ToolPolicy):
            stored_tool_policy = tool_policy
        else:
            raise TypeError("tool_policy must be a ToolPolicy.")
        stored_runtime_hooks = _validate_runtime_hooks(
            runtime_hooks,
            field_name="runtime_hooks",
        )
        stored_loop_policies = validate_loop_policies(
            loop_policies,
            field_name="loop_policies",
        )

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
            registered_tool = _validate_registered_tool(tool)
            if registered_tool.name in tools_by_name:
                raise ValueError(f"Duplicate tool registered for agent: {registered_tool.name}")
            tools_by_name[registered_tool.name] = registered_tool

        registration_source, registration_symbol = _registration_site()
        self._agents[stored_spec.name] = runtime_records.RegisteredAgentState(
            spec=stored_spec,
            tools=MappingProxyType(tools_by_name),
            context_policy=stored_context_policy,
            context_overflow_policy=stored_context_overflow_policy,
            tool_policy=stored_tool_policy,
            runtime_hooks=stored_runtime_hooks,
            loop_policies=stored_loop_policies,
            registration_source=registration_source,
            registration_symbol=registration_symbol,
        )
        return spec

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
        require_clean_nonblank(provider.name, "provider.name")
        if provider.name in self._providers:
            raise ValueError(f"Provider already registered: {provider.name}")

        registration_source, registration_symbol = _registration_site()
        self._providers[provider.name] = runtime_records.RegisteredProvider(
            name=provider.name,
            provider=provider,
            model_patterns=stored_model_patterns,
            registration_source=registration_source,
            registration_symbol=registration_symbol,
        )
        if default or self._default_provider_name is None:
            self._default_provider_name = provider.name
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
        stored_spec = _validate_environment_spec(stored_environment.spec)
        if stored_spec.name in self._environments:
            raise ValueError(f"Environment already registered: {stored_spec.name}")

        registration_source, registration_symbol = _registration_site()
        self._environments[stored_spec.name] = runtime_records.RegisteredEnvironment(
            spec=stored_spec,
            environment=stored_environment,
            registration_source=registration_source,
            registration_symbol=registration_symbol,
        )
        self._select_default_environment_if_requested(stored_spec.name, default=default)
        return environment

    def register_environment_factory(
        self,
        spec: EnvironmentSpec,
        factory: EnvironmentFactory,
        *,
        default: bool = False,
    ) -> EnvironmentFactory:
        if not isinstance(spec, EnvironmentSpec):
            raise TypeError("Environment factory registration requires an EnvironmentSpec.")
        if not isinstance(factory, EnvironmentFactory):
            raise TypeError("Environment factory registration requires an EnvironmentFactory.")
        if not isinstance(default, bool):
            raise TypeError("Environment factory default flag must be a bool.")
        stored_spec = _validate_environment_spec(spec)
        if stored_spec.name in self._environments:
            raise ValueError(f"Environment already registered: {stored_spec.name}")

        registration_source, registration_symbol = _registration_site()
        self._environments[stored_spec.name] = runtime_records.RegisteredEnvironment(
            spec=stored_spec,
            environment=Environment(stored_spec),
            factory=factory,
            registration_source=registration_source,
            registration_symbol=registration_symbol,
        )
        self._select_default_environment_if_requested(stored_spec.name, default=default)
        return factory

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
        )

    def list_agents(self) -> tuple[str, ...]:
        """Return the names of all registered agents, sorted."""
        return tuple(sorted(self._agents))

    def list_providers(self) -> tuple[str, ...]:
        """Return the names of all registered providers, sorted."""
        return tuple(sorted(self._providers))

    def list_environments(self) -> tuple[str, ...]:
        """Return the names of all registered environments (concrete or factory), sorted."""
        return tuple(sorted(self._environments))

    def list_environment_registrations(self) -> tuple[runtime_records.RegisteredEnvironment, ...]:
        """Return registered environment metadata without materializing factories."""
        registrations: list[runtime_records.RegisteredEnvironment] = []
        for name in sorted(self._environments):
            registered_environment = self._environments[name]
            registrations.append(
                runtime_records.RegisteredEnvironment(
                    spec=registered_environment.spec.model_copy(deep=True),
                    environment=copy_environment(registered_environment.environment),
                    factory=registered_environment.factory,
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
        (default or named) environment must expose a statically-registered artifact store.
        Factory-backed environments create their store per session at run time, which does not exist
        yet when you call `attach_file`, so this raises for them — register the artifact store on the
        environment directly if you need to attach prompt files.
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
                "attach_file requires an environment with a statically-registered artifact store; "
                "factory-backed environments create their store per session at run time."
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
        if type(request) is not RunRequest:
            raise TypeError("Runtime run requires a RunRequest.")
        request = _validate_run_request(request)
        registered_agent = self._get_registered_agent(request.agent_name)
        # Provider resolution for new sessions: per-run override, then the
        # agent's pinned provider, then model-pattern routing, then the app
        # default. Resume/fork keep honoring the provider recorded on the
        # session.
        model = request.model or registered_agent.spec.model
        if request.provider_name is not None or registered_agent.spec.provider_name is not None:
            registered_provider = self._get_registered_provider(
                request.provider_name or registered_agent.spec.provider_name
            )
        else:
            registered_provider = (
                self._route_registered_provider_for_model(model=model)
                or self._get_registered_provider()
            )
        # Checked before the session is created so it surfaces to the caller.
        _require_native_structured_output_support(
            request.structured_output, registered_provider=registered_provider
        )
        registered_environment = self._get_registered_environment(request.environment_name)
        if request.environment_name is None and registered_environment is not None:
            request = _with_environment_name(request, registered_environment.spec.name)
        workspace_instructions = None
        if registered_environment is None or registered_environment.factory is None:
            workspace_instructions = await self._environment_lifecycle.load_workspace_instructions(
                registered_environment,
            )
        session = await self.session_store.create(
            request,
            identity=_session_identity(
                provider_name=registered_provider.name,
                model=model,
            ),
        )
        try:
            session = await self.session_store.transition_status(
                session.id,
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.RUNNING,
            )
        except ValueError:
            loaded_session = await self.session_store.load(session.id)
            if (
                loaded_session is not None
                and loaded_session.status in INTERRUPT_REQUESTED_SESSION_STATUSES
            ):
                async for event in self._handle_session_interrupted(
                    session=loaded_session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=_environment_name(registered_environment),
                ):
                    yield event
                return
            raise
        current_task = asyncio.current_task()
        active_factory_run: ActiveSessionRun[SessionUsageTracker] | None = None
        if (
            registered_environment is not None
            and registered_environment.factory is not None
            and current_task is not None
        ):
            active_factory_run = self._session_control.register_active_task(
                session.id,
                current_task,
                task_id=request.task_id,
                task_started=False,
                task_finished=False,
            )
        # The run fence belongs to this pre-run setup until control is handed to
        # _run_session. Every earlier exit must revoke it, including cancellation
        # and failures while recording the original setup failure.
        release_before_run = True
        pre_run_task_started = False
        try:
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            if factory_started_event is not None:
                yield factory_started_event
                async for queued_event in self._session_control.drain_out_of_band_events(
                    session.id
                ):
                    yield queued_event
                if request.task_id is not None:
                    task = await self._start_task(
                        task_id=request.task_id,
                        session=session,
                        worker_id=request.task_worker_id,
                    )
                    pre_run_task_started = True
                    if active_factory_run is not None:
                        active_factory_run.task_started = True
                    yield await self._event_writer.emit(
                        _task_event(
                            event_type=EventType.TASK_STARTED,
                            task=task,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                        )
                    )
            resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=EnvironmentFactoryOperation.CREATE,
            )
            registered_environment = resolution.registered_environment
            for event in resolution.events:
                yield event
                async for queued_event in self._session_control.drain_out_of_band_events(
                    session.id
                ):
                    yield queued_event
            async for queued_event in self._session_control.drain_out_of_band_events(session.id):
                yield queued_event
            if resolution.error is not None:
                task_failure_event, task_failure_error = await self._fail_task_for_run_setup_error(
                    task_id=request.task_id,
                    task_worker_id=request.task_worker_id,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    error=resolution.error,
                )
                if task_failure_event is not None:
                    yield task_failure_event
                session = await self.session_store.update_status(session.id, SessionStatus.FAILED)
                failure_payload = exception_failure_payload(resolution.error)
                if task_failure_error is not None:
                    failure_payload["task_update_error"] = str(task_failure_error)
                    failure_payload["task_update_error_type"] = type(task_failure_error).__name__
                async for event in self._emit_terminal_event_with_hooks(
                    event=Event(
                        type=EventType.SESSION_FAILED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        payload=failure_payload,
                    ),
                    phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                ):
                    yield event
                    async for queued_event in self._session_control.drain_out_of_band_events(
                        session.id
                    ):
                        yield queued_event
                return

            if workspace_instructions is None:
                workspace_instructions = (
                    await self._environment_lifecycle.load_workspace_instructions(
                        registered_environment,
                    )
                )
            release_before_run = False
        except asyncio.CancelledError:
            if await self._session_control.interrupt_requested(session.id):
                async for event in self._handle_session_interrupted(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=_environment_name(registered_environment),
                ):
                    yield event
                return
            raise
        except Exception as exc:
            task_failure_event, task_failure_error = await self._fail_task_for_run_setup_error(
                task_id=request.task_id,
                task_worker_id=request.task_worker_id,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                error=exc,
            )
            if task_failure_event is not None:
                yield task_failure_event
            session = await self.session_store.update_status(session.id, SessionStatus.FAILED)
            failure_payload = exception_failure_payload(exc)
            if task_failure_error is not None:
                failure_payload["task_update_error"] = str(task_failure_error)
                failure_payload["task_update_error_type"] = type(task_failure_error).__name__
            async for event in self._emit_terminal_event_with_hooks(
                event=Event(
                    type=EventType.SESSION_FAILED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=_environment_name(registered_environment),
                    payload=failure_payload,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            ):
                yield event
            return
        except GeneratorExit:
            # A consumer that closes the stream during factory-resolution yields would
            # otherwise strand the session RUNNING (this window is outside _run_session's
            # own finalizer). Finalize before propagating the close.
            await self._finalize_abandoned_session_by_id(session.id)
            raise
        finally:
            try:
                if release_before_run:
                    await self.session_store.release_run_fence(session.id)
            finally:
                if current_task is not None and active_factory_run is not None:
                    self._session_control.unregister_active_task(session.id, current_task)

        try:
            messages = transcript_helpers.initial_messages(
                system_prompt=render_initial_system_prompt(
                    agent_system_prompt=registered_agent.spec.system_prompt,
                    workspace_instructions=workspace_instructions,
                ),
                request_messages=request.messages,
            )
            session_stream = self._run_session(
                session=session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                messages=messages,
                messages_to_append=messages,
                max_steps=request.max_steps,
                limits=request.limits,
                budget_limits=request.budget_limits,
                retry_policy=self._effective_retry_policy(request.retry_policy),
                structured_output=request.structured_output,
                thinking=request.thinking,
                request_loop_policies=request.loop_policies,
                request_metadata=request.metadata,
                task_id=request.task_id,
                task_worker_id=request.task_worker_id,
                start_event_type=EventType.SESSION_STARTED,
                start_event_payload={"agent_name": registered_agent.spec.name},
                start_task_on_enter=not pre_run_task_started,
            )
        except BaseException:
            await self.session_store.release_run_fence(session.id)
            raise
        try:
            async for event in self._session_control.stream_with_out_of_band_events(
                session.id,
                session_stream,
            ):
                yield event
        except asyncio.CancelledError:
            if await self._session_control.interrupt_requested(session.id):
                async for event in self._handle_session_interrupted(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=_environment_name(registered_environment),
                ):
                    yield event
                return
            raise
        except GeneratorExit:
            # Close the inner run stream deterministically so it finalizes (not
            # strands) the RUNNING session before the consumer's close returns.
            await session_stream.aclose()
            raise

    async def resume(self, request: ResumeRequest) -> AsyncIterator[Event]:
        if type(request) is not ResumeRequest:
            raise TypeError("Runtime resume requires a ResumeRequest.")
        request = _validate_resume_request(request)
        task_id = await self._linked_running_task_id(request.session_id)
        session_stream = self._resume_session(
            request=request,
            task_id=task_id,
            start_event_payload_extra={},
            start_task_on_enter=False,
        )
        try:
            async for event in self._session_control.stream_with_out_of_band_events(
                request.session_id,
                session_stream,
            ):
                yield event
        except GeneratorExit:
            await session_stream.aclose()
            raise

    async def compact_session(
        self,
        request: CompactSessionRequest,
    ) -> AsyncIterator[Event]:
        if type(request) is not CompactSessionRequest:
            raise TypeError("Runtime compaction requires a CompactSessionRequest.")
        operation_stream = self._compact_session(request)
        try:
            async for event in self._session_control.stream_with_out_of_band_events(
                request.session_id,
                operation_stream,
            ):
                yield event
        except GeneratorExit:
            await operation_stream.aclose()
            raise

    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
    ) -> EnqueueSessionMessageResult:
        """Durably queue user steering for delivery by the active controller."""

        if type(request) is not EnqueueSessionMessageRequest:
            raise TypeError("Runtime queued input requires an EnqueueSessionMessageRequest.")
        result = await self.session_store.enqueue_session_message(
            copy_enqueue_session_message_request(request)
        )
        if not result.replayed:
            await self._event_writer.fan_out_persisted([result.event])
        return result

    async def _deliver_queued_session_messages(
        self,
        *,
        session_id: str,
        messages: list[Message],
        include_on_idle: bool,
    ) -> list[Event]:
        delivered_events: list[Event] = []
        eligible_through: int | None = None
        while True:
            try:
                batch: SessionMessageDeliveryBatch = (
                    await self.session_store.deliver_queued_session_messages(
                        session_id,
                        include_on_idle=include_on_idle,
                        eligible_through=eligible_through,
                    )
                )
            except SessionStatusConflict:
                # An interrupt can win after the loop's durable status check,
                # between bounded delivery batches, or after completion detects
                # queued work. Preserve that lifecycle result through the normal
                # interruption finalizer instead of treating the store's delivery
                # fence as a runtime failure. Unrelated status conflicts still
                # propagate unchanged.
                await self._session_control.raise_if_interrupted(session_id)
                raise
            if eligible_through is None:
                eligible_through = batch.eligible_through
            messages.extend(
                Message.text(MessageRole.USER, queued_message.content)
                for queued_message in batch.messages
            )
            if batch.events:
                await self._event_writer.fan_out_persisted(list(batch.events))
                delivered_events.extend(batch.events)
            if not batch.has_more:
                return delivered_events

    async def _compact_session(
        self,
        request: CompactSessionRequest,
    ) -> AsyncGenerator[Event, None]:
        """Compact model-facing context without appending a conversation turn."""

        request = copy_compact_session_request(request)
        loaded_session = await self.session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")
        request_digest = _compact_session_request_digest(request)
        checkpoint_before_claim = await self.session_store.load_checkpoint(loaded_session.id)
        persisted_before_claim = await self.session_store.load_session_operation(
            loaded_session.id,
            request.idempotency_key,
        )
        if type(persisted_before_claim) is dict:
            if persisted_before_claim.get("request_digest") != request_digest:
                raise ValueError(
                    "Session compaction idempotency key was already used for a different request."
                )
            replay_event_ids = _session_compaction_replay_event_ids(persisted_before_claim)
            if replay_event_ids is not None:
                for event in await self._load_session_compaction_replay_events(
                    session_id=loaded_session.id,
                    event_ids=replay_event_ids,
                ):
                    yield event
                return
        if (
            checkpoint_before_claim is not None
            and _SESSION_OPERATIONS_CHECKPOINT_KEY in checkpoint_before_claim
        ):
            existing_before_claim = _session_operation_state(checkpoint_before_claim)[
                "records"
            ].get(request.idempotency_key)
            if type(existing_before_claim) is dict:
                if existing_before_claim.get("request_digest") != request_digest:
                    raise ValueError(
                        "Session compaction idempotency key was already used for a "
                        "different request."
                    )
                replay_event_ids = _session_compaction_replay_event_ids(existing_before_claim)
                if replay_event_ids is not None:
                    for event in await self._load_session_compaction_replay_events(
                        session_id=loaded_session.id,
                        event_ids=replay_event_ids,
                    ):
                        yield event
                    return
        if loaded_session.status not in _RESUMABLE_SESSION_STATUSES:
            raise ValueError(
                f"Session compaction requires a resumable session boundary: {loaded_session.status}"
            )
        if loaded_session.run_epoch != request.expected_run_epoch:
            raise ValueError(
                "Session compaction source run epoch is stale: expected "
                f"{request.expected_run_epoch}, current {loaded_session.run_epoch}."
            )

        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        context_policy = registered_agent.context_policy
        if not isinstance(context_policy, CheckpointCompactionContextPolicy):
            raise ValueError(
                "Explicit session compaction requires a configured "
                "CheckpointCompactionContextPolicy."
            )
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )
        environment_name = _environment_name(registered_environment)
        transcript = await self.session_store.load_transcript(loaded_session.id)
        if len(transcript) != request.expected_transcript_cursor:
            raise ValueError(
                "Session compaction source transcript cursor is stale: expected "
                f"{request.expected_transcript_cursor}, current {len(transcript)}."
            )

        candidate_app_policy_budget_limits = budget_limits_for_session(
            policy=self.budget_policy,
            agent_name=registered_agent.spec.name,
            causal_budget_id=loaded_session.causal_budget_id,
        )
        candidate_request_budget_limits = request_budget_limits_for_session(
            limits=request.budget_limits,
            agent_name=registered_agent.spec.name,
            causal_budget_id=loaded_session.causal_budget_id,
        )
        compactor_provider_name: str | None = None
        compactor_model: str | None = None
        if candidate_app_policy_budget_limits or candidate_request_budget_limits:
            try:
                provider_budget_identity = context_policy.compactor.provider_budget_identity(
                    loaded_session
                )
            except NotImplementedError as exc:
                raise RuntimeError(
                    "Explicit compaction with cost budgets requires the ContextCompactor "
                    "to declare provider_budget_identity(session), returning provider/model "
                    "or None for deterministic execution."
                ) from exc
            if provider_budget_identity is not None:
                if (
                    type(provider_budget_identity) is not tuple
                    or len(provider_budget_identity) != 2
                ):
                    raise TypeError(
                        "ContextCompactor.provider_budget_identity must return a "
                        "(provider_name, model) tuple or None."
                    )
                compactor_provider_name = _require_application_compaction_event_text(
                    provider_budget_identity[0],
                    "compactor_provider_name",
                )
                compactor_model = _require_application_compaction_event_text(
                    provider_budget_identity[1],
                    "compactor_model",
                )
        if compactor_provider_name is None:
            app_policy_budget_limits: tuple[BudgetLimit, ...] = ()
            budget_limits: tuple[BudgetLimit, ...] = ()
        else:
            app_policy_budget_limits = candidate_app_policy_budget_limits
            budget_limits = (*app_policy_budget_limits, *candidate_request_budget_limits)
        contextual_dispatch_budget_limits = tuple(
            limit
            for limit in budget_limits
            if has_deferred_contextual_price(
                limit.pricing,
                provider_name=compactor_provider_name,
                model=compactor_model,
            )
        )
        if (
            contextual_dispatch_budget_limits
            and not context_policy.compactor._uses_runtime_provider_dispatch_runner_for_forced_compaction()
        ):
            raise RuntimeError(
                "Explicit compaction with contextual pricing requires an "
                "unmodified built-in provider compactor so Cayu can admit every "
                "resolved provider dispatch before execution."
            )
        operation_started_at = time.monotonic()

        operation_id = str(uuid4())
        claim_probe_now = self._clock()
        attempt_id = str(uuid4())
        if (
            type(persisted_before_claim) is dict
            and persisted_before_claim.get("request_digest") == request_digest
            and persisted_before_claim.get("status") == "abandoned"
        ):
            persisted_operation_id = persisted_before_claim.get("operation_id")
            if type(persisted_operation_id) is not str:
                raise ValueError("Persisted session operation is missing its operation id.")
            operation_id = require_clean_nonblank(
                persisted_operation_id,
                "operation_id",
            )
        if (
            checkpoint_before_claim is not None
            and _SESSION_OPERATIONS_CHECKPOINT_KEY in checkpoint_before_claim
        ):
            existing_before_claim = _session_operation_state(checkpoint_before_claim)[
                "records"
            ].get(request.idempotency_key)
            if (
                type(existing_before_claim) is dict
                and existing_before_claim.get("request_digest") == request_digest
                and existing_before_claim.get("status") == "running"
                and (
                    (expiry := _operation_claim_expiry(existing_before_claim)) is not None
                    and expiry <= claim_probe_now
                )
            ):
                operation_id = require_clean_nonblank(
                    existing_before_claim.get("operation_id"),
                    "operation_id",
                )
        started_event = Event(
            type=EventType.CONTEXT_COMPACTION_STARTED,
            session_id=loaded_session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload=_application_compaction_causal_payload(
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                source_cursor=len(transcript),
                compactor=type(context_policy.compactor).__name__,
            ),
        )
        claimed_checkpoint: dict[str, Any] | None = None

        def claim_operation(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            persisted_record: dict[str, Any] | None,
        ) -> SessionOperationPublication:
            nonlocal operation_id, claimed_checkpoint
            claim_now = self._clock()
            claim_expires_at = claim_now + _SESSION_OPERATION_CLAIM_LEASE
            if current_session.run_epoch != request.expected_run_epoch:
                raise ValueError(
                    "Session compaction source run epoch is stale: expected "
                    f"{request.expected_run_epoch}, current {current_session.run_epoch}."
                )
            _reject_unresumable_session_checkpoint(
                current_session,
                checkpoint,
                allow_active_operation=True,
            )
            updated = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
            operations = _session_operation_state(updated)
            _abandon_expired_session_operation(operations, now=claim_now)
            records = operations["records"]
            existing = records.get(request.idempotency_key)
            if existing is None and persisted_record is not None:
                existing = copy_json_value(persisted_record, "session_operation")
                if existing.get("status") == "abandoned":
                    records[request.idempotency_key] = existing
            if existing is not None:
                if existing.get("request_digest") != request_digest:
                    raise ValueError(
                        "Session compaction idempotency key was already used for a "
                        "different request."
                    )
                status = existing.get("status")
                if status == "running":
                    raise RuntimeError(
                        "Equivalent session compaction operation is already running: "
                        f"{existing.get('operation_id')}"
                    )
                if status == "abandoned":
                    operation_id = require_clean_nonblank(
                        existing.get("operation_id"),
                        "operation_id",
                    )
                    if started_event.payload.get("operation_id") != operation_id:
                        raise RuntimeError(
                            "Abandoned session compaction changed during claim; retry it."
                        )
                    if operations.get("active_operation_id") is not None:
                        raise RuntimeError(
                            "Session already has an active durable operation: "
                            f"{operations.get('active_operation_id')}"
                        )
                    operations["active_operation_id"] = operation_id
                    existing["status"] = "running"
                    existing["attempt_count"] = existing.get("attempt_count", 1) + 1
                    existing["current_attempt_id"] = attempt_id
                    existing["event_ids"] = [
                        *existing.get("event_ids", []),
                        started_event.id,
                    ]
                    existing["claim_expires_at"] = claim_expires_at.isoformat()
                    existing["updated_at"] = claim_now.isoformat()
                    existing.pop("abandoned_at", None)
                    updated[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
                    claimed_checkpoint = copy_json_value(updated, "checkpoint")
                    archived_records = _archive_inactive_session_operation_records(
                        updated,
                        except_idempotency_key=request.idempotency_key,
                    )
                    return SessionOperationPublication(
                        checkpoint=updated,
                        operation_records=archived_records,
                    )
                operation_id = require_clean_nonblank(
                    existing.get("operation_id"),
                    "operation_id",
                )
                stored_event_ids = _session_compaction_replay_event_ids(existing)
                if stored_event_ids is None:
                    raise RuntimeError("Running session compaction changed during claim.")
                raise _SessionCompactionReplay(stored_event_ids)
            active_operation_id = operations.get("active_operation_id")
            if active_operation_id is not None:
                raise RuntimeError(
                    f"Session already has an active durable operation: {active_operation_id}"
                )
            operations["active_operation_id"] = operation_id
            records[request.idempotency_key] = {
                "operation_id": operation_id,
                "kind": _CONTEXT_COMPACTION_OPERATION_KIND,
                "reason": request.reason,
                "request_digest": request_digest,
                "status": "running",
                "source_run_epoch": request.expected_run_epoch,
                "source_transcript_cursor": request.expected_transcript_cursor,
                "attempt_count": 1,
                "current_attempt_id": attempt_id,
                "event_ids": [started_event.id],
                "instruction_present": request.instructions is not None,
                "instruction_digest": _optional_text_digest(request.instructions),
                "claim_expires_at": claim_expires_at.isoformat(),
                "created_at": claim_now.isoformat(),
                "updated_at": claim_now.isoformat(),
            }
            updated[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
            claimed_checkpoint = copy_json_value(updated, "checkpoint")
            archived_records = _archive_inactive_session_operation_records(
                updated,
                except_idempotency_key=request.idempotency_key,
            )
            return SessionOperationPublication(
                checkpoint=updated,
                operation_records=archived_records,
            )

        replay_event_ids: tuple[str, ...] | None = None
        try:
            await self.session_store.publish_session_operation(
                loaded_session.id,
                idempotency_key=request.idempotency_key,
                operation_transform=claim_operation,
                events=[started_event],
                expected_statuses=_RESUMABLE_SESSION_STATUSES,
                expected_run_epoch=request.expected_run_epoch,
                expected_transcript_cursor=request.expected_transcript_cursor,
            )
        except _SessionCompactionReplay as replay:
            replay_event_ids = replay.event_ids
        if replay_event_ids is not None:
            for event in await self._load_session_compaction_replay_events(
                session_id=loaded_session.id,
                event_ids=replay_event_ids,
            ):
                yield event
            return
        if claimed_checkpoint is None:
            raise AssertionError("New session compaction did not persist its operation claim.")

        await self._event_writer.fan_out_persisted([started_event])
        yield started_event
        attempt_events: list[Event] = []
        prepublished_dispatch_events: list[Event] = []
        reached_budget_keys: set[tuple[str, str | None, str, str, Decimal]] = set()
        budget_reservations: list[BudgetStepReservation] = []
        budget_reservations_settled = False
        operation_published = False
        try:
            limit_decision = await self._run_limit_controller.evaluate_operation_run_limit(
                session=loaded_session,
                limits=request.limits,
                operation_events=attempt_events,
                operation_started_at=operation_started_at,
            )
            if limit_decision is not None:
                raise RuntimeError(f"Compaction limit reached: {limit_decision.message}")
            budget_error = await self._enforce_compaction_budget_limits(
                session=loaded_session,
                budget_limits=budget_limits,
                app_policy_budget_limits=app_policy_budget_limits,
                attempt_events=attempt_events,
                reached_budget_keys=reached_budget_keys,
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                registered_agent=registered_agent,
                environment_name=environment_name,
                compactor=type(context_policy.compactor).__name__,
                provider_name=compactor_provider_name,
                model=compactor_model,
            )
            if attempt_events:
                persisted_attempt_events = list(attempt_events)
                await self._persist_compaction_attempt_events(
                    session=loaded_session,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    events=persisted_attempt_events,
                )
                attempt_events.clear()
                await self._event_writer.fan_out_persisted(persisted_attempt_events)
                for event in persisted_attempt_events:
                    yield event
            if budget_error is not None:
                raise budget_error

            reservation_failure = await self._reserve_compaction_budget(
                session=loaded_session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                budget_limits=(() if contextual_dispatch_budget_limits else budget_limits),
                provider_name=compactor_provider_name,
                model=compactor_model,
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                compactor=type(context_policy.compactor).__name__,
                reservations=budget_reservations,
                events=attempt_events,
            )
            reservation_error: RuntimeError | None = None
            if reservation_failure is not None:
                budget_reservations_settled = True
                attempt_events.append(
                    _application_compaction_ledger_event(
                        event_type=EventType.BUDGET_LIMIT_REACHED,
                        payload=budget_reservation_payload(reservation_failure),
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        compactor=type(context_policy.compactor).__name__,
                    )
                )
                reservation_error = RuntimeError(
                    f"Compaction budget reservation failed: {reservation_failure.message}"
                )
            if attempt_events:
                persisted_attempt_events = list(attempt_events)
                await self._persist_compaction_attempt_events(
                    session=loaded_session,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    events=persisted_attempt_events,
                )
                attempt_events.clear()
                await self._event_writer.fan_out_persisted(persisted_attempt_events)
                for event in persisted_attempt_events:
                    yield event
            if reservation_error is not None:
                raise reservation_error

            async def publish_dispatch_budget_events() -> None:
                if not attempt_events:
                    return
                events = list(attempt_events)
                await self._persist_compaction_attempt_events(
                    session=loaded_session,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    events=events,
                )
                attempt_events.clear()
                await self._event_writer.fan_out_persisted(events)
                prepublished_dispatch_events.extend(events)

            def dispatch_completion_event(
                *,
                actual_provider: ModelProvider,
                actual_model: str,
                completed_metadata: dict[str, Any],
            ) -> Event:
                return Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id=loaded_session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=_compaction_model_completed_payload(
                        completed_payload=completed_metadata,
                        provider_name=actual_provider.name,
                        fallback_model=actual_model,
                        compactor=type(context_policy.compactor).__name__,
                        usage_dialect=actual_provider.usage_dialect,
                    ),
                )

            async def settle_contextual_dispatch(
                dispatch_reservations: list[BudgetStepReservation],
                *,
                completed_event: Event | None,
                uncertain_reason: str,
            ) -> None:
                dispatch_reservation_ids = {
                    reservation.record.reservation_id for reservation in dispatch_reservations
                }
                if completed_event is None:
                    settlement = self._reconcile_uncertain_compaction_budget_reservations(
                        dispatch_reservations,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        compactor=type(context_policy.compactor).__name__,
                    )
                else:
                    settlement = self._reconcile_compaction_budget_reservations(
                        dispatch_reservations,
                        model_completed_events=[completed_event],
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        compactor=type(context_policy.compactor).__name__,
                    )
                try:
                    async for event in settlement:
                        attempt_events.append(event)
                except Exception as exc:
                    exc.add_note(uncertain_reason)
                    raise
                finally:
                    unsettled_ids = {
                        reservation.record.reservation_id for reservation in dispatch_reservations
                    }
                    settled_ids = dispatch_reservation_ids - unsettled_ids
                    budget_reservations[:] = [
                        reservation
                        for reservation in budget_reservations
                        if reservation.record.reservation_id not in settled_ids
                    ]

            async def run_contextual_provider_dispatch(
                actual_provider: ModelProvider,
                actual_model: str,
                billing_identity: BillingIdentity | None,
                dispatch: Callable[[], Awaitable[tuple[str, dict[str, Any]]]],
            ) -> tuple[str, dict[str, Any]]:
                budget_error = await self._enforce_compaction_budget_limits(
                    session=loaded_session,
                    budget_limits=budget_limits,
                    app_policy_budget_limits=app_policy_budget_limits,
                    attempt_events=attempt_events,
                    reached_budget_keys=reached_budget_keys,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    compactor=type(context_policy.compactor).__name__,
                    provider_name=actual_provider.name,
                    model=actual_model,
                    billing_identity_state=resolved_billing_identity(billing_identity),
                )
                if budget_error is not None:
                    await publish_dispatch_budget_events()
                    raise budget_error

                reservation_start = len(budget_reservations)
                reservation_failure = await self._reserve_compaction_budget(
                    session=loaded_session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    budget_limits=budget_limits,
                    provider_name=actual_provider.name,
                    model=actual_model,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    compactor=type(context_policy.compactor).__name__,
                    reservations=budget_reservations,
                    events=attempt_events,
                    billing_identity=billing_identity,
                )
                dispatch_reservations = budget_reservations[reservation_start:]
                if reservation_failure is not None:
                    attempt_events.append(
                        _application_compaction_ledger_event(
                            event_type=EventType.BUDGET_LIMIT_REACHED,
                            payload=budget_reservation_payload(reservation_failure),
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            session=loaded_session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            compactor=type(context_policy.compactor).__name__,
                        )
                    )
                    await publish_dispatch_budget_events()
                    raise RuntimeError(
                        f"Compaction budget reservation failed: {reservation_failure.message}"
                    )
                await publish_dispatch_budget_events()

                try:
                    (
                        dispatch_result,
                        lease_failure,
                    ) = await self._run_limit_controller.run_operation_with_reservation_heartbeat(
                        dispatch,
                        reservations=dispatch_reservations,
                        authoritative_failure_types=(ContextBuildError,),
                        lease_lost_before_dispatch_message=(
                            "Compaction budget reservation lease was lost before provider dispatch."
                        ),
                        authoritative_failure_note=(
                            "Budget reservation lease was also lost as compaction failed"
                        ),
                        concurrent_failure_note=(
                            "Compactor also failed while reservation lease loss was handled"
                        ),
                    )
                except asyncio.CancelledError as exc:

                    async def settle_cancelled_dispatch() -> None:
                        await settle_contextual_dispatch(
                            dispatch_reservations,
                            completed_event=None,
                            uncertain_reason=(
                                "Contextual compaction dispatch settlement failed "
                                "during cancellation."
                            ),
                        )
                        await publish_dispatch_budget_events()

                    settlement_task = asyncio.create_task(settle_cancelled_dispatch())
                    settlement_failure: BaseException | None = None
                    while not settlement_task.done():
                        try:
                            await asyncio.shield(settlement_task)
                        except asyncio.CancelledError:
                            continue
                        except BaseException as candidate:
                            settlement_failure = candidate
                            break
                    if settlement_failure is None:
                        try:
                            settlement_task.result()
                        except BaseException as candidate:
                            settlement_failure = candidate
                    if settlement_failure is not None:
                        exc.add_note(
                            "Contextual compaction cancellation settlement also failed: "
                            f"{type(settlement_failure).__name__}: {settlement_failure}"
                        )
                    raise
                except BaseException as exc:
                    raw_completed = getattr(exc, "completed_metadata", None)
                    completed_event = (
                        dispatch_completion_event(
                            actual_provider=actual_provider,
                            actual_model=actual_model,
                            completed_metadata=raw_completed,
                        )
                        if type(raw_completed) is dict
                        else None
                    )
                    try:
                        await settle_contextual_dispatch(
                            dispatch_reservations,
                            completed_event=completed_event,
                            uncertain_reason=(
                                "Contextual compaction dispatch settlement failed after "
                                "provider error."
                            ),
                        )
                        await publish_dispatch_budget_events()
                    except BaseException as settlement_failure:
                        if not isinstance(settlement_failure, Exception):
                            raise settlement_failure from exc
                        add_budget_failure_note(
                            exc,
                            operation="contextual compaction dispatch settlement",
                            accounting_failure=settlement_failure,
                        )
                    raise

                completed_event = dispatch_completion_event(
                    actual_provider=actual_provider,
                    actual_model=actual_model,
                    completed_metadata=dispatch_result[1],
                )
                await settle_contextual_dispatch(
                    dispatch_reservations,
                    completed_event=completed_event,
                    uncertain_reason=(
                        "Contextual compaction dispatch settlement failed after completion."
                    ),
                )
                await publish_dispatch_budget_events()
                if lease_failure is not None:
                    raise lease_failure
                return dispatch_result

            async def execute_compaction() -> ContextBuildResult:
                return await context_policy.build_with_checkpoint(
                    ContextRequest(
                        session=loaded_session,
                        agent=_session_agent_spec(
                            registered_agent=registered_agent,
                            session=loaded_session,
                        ),
                        messages=transcript,
                        step=1,
                        environment_name=environment_name,
                        metadata={
                            "operation_id": operation_id,
                            "reason": request.reason,
                        },
                        force_compaction=True,
                        force_bounded_compaction=True,
                        compaction_instructions=request.instructions,
                    ),
                    checkpoint=claimed_checkpoint,
                )

            if contextual_dispatch_budget_limits:
                with _automatic_compaction_dispatch_runner_scope(run_contextual_provider_dispatch):
                    result = await execute_compaction()
                reservation_lease_failure = None
            else:
                (
                    result,
                    reservation_lease_failure,
                ) = await self._run_limit_controller.run_operation_with_reservation_heartbeat(
                    execute_compaction,
                    reservations=budget_reservations,
                    authoritative_failure_types=(ContextBuildError,),
                    lease_lost_before_dispatch_message=(
                        "Compaction budget reservation lease was lost before provider dispatch."
                    ),
                    authoritative_failure_note=(
                        "Budget reservation lease was also lost as compaction failed"
                    ),
                    concurrent_failure_note=(
                        "Compactor also failed while reservation lease loss was handled"
                    ),
                )
            for event in prepublished_dispatch_events:
                yield event
            prepublished_dispatch_events.clear()
            if result.checkpoint is None or result.checkpoint_event_payload is None:
                raise ValueError("Session has no complete older context to compact.")

            telemetry_events = [
                _application_compaction_event(
                    telemetry=telemetry,
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    session=loaded_session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    compactor=type(context_policy.compactor).__name__,
                )
                for telemetry in result.compaction_telemetry
                if telemetry.event_type != EventType.CONTEXT_COMPACTION_STARTED
            ]
            attempt_events.extend(
                event for event in telemetry_events if event.type == EventType.MODEL_COMPLETED
            )
            async for event in self._reconcile_compaction_budget_reservations(
                budget_reservations,
                model_completed_events=[
                    event for event in attempt_events if event.type == EventType.MODEL_COMPLETED
                ],
                session=loaded_session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                compactor=type(context_policy.compactor).__name__,
            ):
                attempt_events.append(event)
            budget_reservations_settled = True
            if reservation_lease_failure is not None:
                raise reservation_lease_failure
            limit_decision = await self._run_limit_controller.evaluate_operation_run_limit(
                session=loaded_session,
                limits=request.limits,
                operation_events=attempt_events,
                operation_started_at=operation_started_at,
            )
            if limit_decision is not None:
                raise RuntimeError(f"Compaction limit reached: {limit_decision.message}")
            budget_error = await self._enforce_compaction_budget_limits(
                session=loaded_session,
                budget_limits=budget_limits,
                app_policy_budget_limits=app_policy_budget_limits,
                attempt_events=attempt_events,
                reached_budget_keys=reached_budget_keys,
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                registered_agent=registered_agent,
                environment_name=environment_name,
                compactor=type(context_policy.compactor).__name__,
                provider_name=compactor_provider_name,
                model=compactor_model,
            )
            if budget_error is not None:
                raise budget_error
            checkpoint_event = Event(
                type=EventType.SESSION_CHECKPOINTED,
                session_id=loaded_session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    **copy_json_value(
                        result.checkpoint_event_payload,
                        "checkpoint_event_payload",
                    ),
                    **_application_compaction_causal_payload(
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        source_cursor=request.expected_transcript_cursor,
                        result_cursor=result.checkpoint_event_payload.get(
                            "compacted_transcript_cursor"
                        ),
                        compactor=type(context_policy.compactor).__name__,
                    ),
                },
            )
            published_events = [
                *attempt_events,
                *[event for event in telemetry_events if event.type != EventType.MODEL_COMPLETED],
                checkpoint_event,
            ]
            event_ids = [started_event.id, *[event.id for event in published_events]]
            await self.session_store.publish_session_operation(
                loaded_session.id,
                idempotency_key=request.idempotency_key,
                operation_transform=lambda _session, checkpoint, persisted_record: (
                    _complete_session_operation_checkpoint(
                        checkpoint=checkpoint,
                        persisted_record=persisted_record,
                        compacted_checkpoint=result.checkpoint,
                        idempotency_key=request.idempotency_key,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        event_ids=event_ids,
                        result_cursor=result.checkpoint_event_payload.get(
                            "compacted_transcript_cursor"
                        ),
                        completed_at=self._clock(),
                    )
                ),
                events=published_events,
                expected_statuses=_RESUMABLE_SESSION_STATUSES,
                expected_run_epoch=request.expected_run_epoch,
                expected_transcript_cursor=request.expected_transcript_cursor,
            )
            operation_published = True
            await self._event_writer.fan_out_persisted(published_events)
            for event in published_events:
                yield event
        except GeneratorExit:
            if budget_reservations and not budget_reservations_settled:
                release_events: list[Event] = []
                try:
                    async for event in self._release_compaction_budget_reservations(
                        budget_reservations,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        compactor=type(context_policy.compactor).__name__,
                        reason="compaction operation abandoned",
                    ):
                        release_events.append(event)
                finally:
                    if release_events:
                        await self._persist_compaction_attempt_events(
                            session=loaded_session,
                            request=request,
                            operation_id=operation_id,
                            attempt_id=attempt_id,
                            events=release_events,
                        )
                        await self._event_writer.fan_out_persisted(release_events)
                    budget_reservations_settled = not budget_reservations
            raise
        except BaseException as exc:
            if operation_published:
                raise
            if isinstance(exc, ContextBuildError):
                failed_model_events = [
                    _application_compaction_event(
                        telemetry=telemetry,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        compactor=type(context_policy.compactor).__name__,
                    )
                    for telemetry in exc.compaction_telemetry
                    if telemetry.event_type == EventType.MODEL_COMPLETED
                ]
                existing_attempt_event_ids = {event.id for event in attempt_events}
                attempt_events.extend(
                    event
                    for event in failed_model_events
                    if event.id not in existing_attempt_event_ids
                )
            if budget_reservations and not budget_reservations_settled:
                model_completed_events = [
                    event for event in attempt_events if event.type == EventType.MODEL_COMPLETED
                ]
                if model_completed_events:
                    settlement_stream = self._reconcile_compaction_budget_reservations(
                        budget_reservations,
                        model_completed_events=model_completed_events,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        compactor=type(context_policy.compactor).__name__,
                    )
                elif isinstance(exc, BudgetReservationLeaseLostBeforeModelDispatch):
                    settlement_stream = self._release_compaction_budget_reservations(
                        budget_reservations,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        compactor=type(context_policy.compactor).__name__,
                        reason="compaction reservation lease lost before provider dispatch",
                    )
                elif isinstance(exc, BudgetReservationLeaseLost):
                    settlement_stream = self._reconcile_uncertain_compaction_budget_reservations(
                        budget_reservations,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        compactor=type(context_policy.compactor).__name__,
                    )
                else:
                    settlement_stream = self._release_compaction_budget_reservations(
                        budget_reservations,
                        session=loaded_session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        compactor=type(context_policy.compactor).__name__,
                        reason="compaction provider step did not complete",
                    )
                settlement_failure: BaseException | None = None
                try:
                    async for event in settlement_stream:
                        attempt_events.append(event)
                except BaseException as candidate:
                    settlement_failure = candidate
                budget_reservations_settled = not budget_reservations
                if settlement_failure is not None:
                    if not isinstance(settlement_failure, Exception):
                        raise settlement_failure from exc
                    add_budget_failure_note(
                        exc,
                        operation="compaction settlement",
                        accounting_failure=settlement_failure,
                    )
            failed_event = Event(
                type=EventType.CONTEXT_COMPACTION_FAILED,
                session_id=loaded_session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    **_application_compaction_causal_payload(
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        source_cursor=request.expected_transcript_cursor,
                        compactor=type(context_policy.compactor).__name__,
                    ),
                    "error_type": type(exc).__name__,
                },
            )
            await self.session_store.publish_session_operation(
                loaded_session.id,
                idempotency_key=request.idempotency_key,
                operation_transform=_fail_session_operation_checkpoint(
                    idempotency_key=request.idempotency_key,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    failed_event_id=failed_event.id,
                    attempt_event_ids=[event.id for event in attempt_events],
                    error_type=type(exc).__name__,
                    completed_at=self._clock(),
                ),
                events=[*attempt_events, failed_event],
            )
            failed_events = [*attempt_events, failed_event]
            await self._event_writer.fan_out_persisted(failed_events)
            for event in prepublished_dispatch_events:
                yield event
            prepublished_dispatch_events.clear()
            for event in failed_events:
                yield event
            raise

    async def _load_session_compaction_replay_events(
        self,
        *,
        session_id: str,
        event_ids: tuple[str, ...],
    ) -> list[Event]:
        events: list[Event] = []
        for event_id in event_ids:
            records = await self.session_store.query_events(
                EventQuery(
                    session_id=session_id,
                    event_id=event_id,
                    limit=1,
                )
            )
            if len(records) != 1:
                raise RuntimeError(
                    f"Session compaction replay event is missing from durable history: {event_id}"
                )
            events.append(copy_event(records[0].event))
        return events

    async def _complete_session_if_no_queued_messages(self, session_id: str) -> Session:
        try:
            return await self.session_store.transition_status_if_no_queued_messages(
                session_id,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.COMPLETED,
            )
        except SessionStatusConflict:
            # An interrupt can win between the final provider response and the
            # atomic completion transition. Route that race through the normal
            # interrupt finalizer instead of the generic failure handler.
            await self._session_control.raise_if_interrupted(session_id)
            raise

    async def _handle_queued_messages_before_completion(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        messages: list[Message],
        step: int,
        max_steps: int,
        run_started_at: float,
        turn_usage_tracker: SessionUsageTracker,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
    ) -> tuple[bool, list[Event]]:
        if step < max_steps:
            return True, await self._deliver_queued_session_messages(
                session_id=session.id,
                messages=messages,
                include_on_idle=True,
            )
        events = [
            event
            async for event in self._stop_session_for_queued_input_step_limit(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                messages=messages,
                step=step,
                max_steps=max_steps,
                run_started_at=run_started_at,
                turn_usage_tracker=turn_usage_tracker,
                active_run=active_run,
            )
        ]
        return False, events

    async def _enforce_compaction_budget_limits(
        self,
        *,
        session: Session,
        budget_limits: tuple[BudgetLimit, ...],
        app_policy_budget_limits: tuple[BudgetLimit, ...],
        attempt_events: list[Event],
        reached_budget_keys: set[tuple[str, str | None, str, str, Decimal]],
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        compactor: str,
        provider_name: str | None,
        model: str | None,
        billing_identity_state: BillingIdentityState = UNRESOLVED_BILLING_IDENTITY,
    ) -> RuntimeError | None:
        checks = await self._run_limit_controller.evaluate_operation_budgets(
            session=session,
            budget_limits=budget_limits,
            operation_events=attempt_events,
            provider_name=provider_name,
            model=model,
            billing_identity_state=billing_identity_state,
        )
        interrupt_error: RuntimeError | None = None
        for outcome in checks:
            budget_limit, check = outcome.limit, outcome.check
            deferred_contextual_check = (
                not isinstance(billing_identity_state, ResolvedBillingIdentity)
                and not check.limit_reached
                and has_deferred_contextual_price(
                    budget_limit.pricing,
                    provider_name=provider_name,
                    model=model,
                )
            )
            if (
                any(budget_limit is limit for limit in app_policy_budget_limits)
                and not deferred_contextual_check
            ):
                attempt_events.append(
                    _application_compaction_ledger_event(
                        event_type=EventType.BUDGET_CHECKED,
                        payload=budget_check_payload(check),
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        compactor=compactor,
                    )
                )
            if not check.limit_reached:
                continue
            budget_key = _budget_check_identity(check)
            if budget_key not in reached_budget_keys:
                attempt_events.append(
                    _application_compaction_budget_event(
                        check=check,
                        request=request,
                        operation_id=operation_id,
                        attempt_id=attempt_id,
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        compactor=compactor,
                    )
                )
                reached_budget_keys.add(budget_key)
            if budget_limit.action == "interrupt" and interrupt_error is None:
                interrupt_error = RuntimeError(f"Compaction budget limit reached: {check.message}")
        return interrupt_error

    async def _persist_compaction_attempt_events(
        self,
        *,
        session: Session,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        events: list[Event],
    ) -> None:
        if not events:
            return
        await self.session_store.publish_session_operation(
            session.id,
            idempotency_key=request.idempotency_key,
            operation_transform=_append_session_operation_attempt_events(
                idempotency_key=request.idempotency_key,
                operation_id=operation_id,
                attempt_id=attempt_id,
                event_ids=[event.id for event in events],
                updated_at=self._clock(),
            ),
            events=events,
            expected_statuses=_RESUMABLE_SESSION_STATUSES,
            expected_run_epoch=request.expected_run_epoch,
            expected_transcript_cursor=request.expected_transcript_cursor,
        )

    async def _reserve_compaction_budget(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        budget_limits: tuple[BudgetLimit, ...],
        provider_name: str | None,
        model: str | None,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        compactor: str,
        reservations: list[BudgetStepReservation],
        events: list[Event],
        billing_identity: BillingIdentity | None = None,
    ) -> BudgetReservationResult | None:
        setup = await self._run_limit_controller.reserve_operation_budgets(
            budget_limits=budget_limits,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            provider_name=provider_name,
            model=model,
            billing_identity=billing_identity,
            rejection_release_reason="compaction budget reservation failed",
            accepted_record_error="Accepted compaction budget reservation has no record.",
        )
        reservations.extend(setup.reservations)
        for result in setup.results:
            events.append(
                _application_compaction_ledger_event(
                    event_type=(
                        EventType.BUDGET_RESERVED
                        if result.accepted
                        else EventType.BUDGET_RESERVATION_FAILED
                    ),
                    payload=budget_reservation_payload(result),
                    request=request,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    session=session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    compactor=compactor,
                )
            )
        events.extend(
            _application_compaction_ledger_event(
                event_type=EventType.BUDGET_RESERVATION_RELEASED,
                payload=budget_reconciliation_payload(reconciliation),
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                compactor=compactor,
            )
            for reconciliation in setup.releases
        )
        if setup.error is not None:
            raise setup.error
        return setup.failure

    async def _reconcile_compaction_budget_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        model_completed_events: list[Event],
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        compactor: str,
    ) -> AsyncIterator[Event]:
        async for reconciliation in self._run_limit_controller.reconcile_operation_reservations(
            reservations,
            model_completed_events=model_completed_events,
            completed_reason="compaction model completed",
            missing_usage_reason=(
                "compaction completed without priced usage; charged reserved amount"
            ),
        ):
            yield _application_compaction_ledger_event(
                event_type=EventType.BUDGET_RECONCILED,
                payload=budget_reconciliation_payload(reconciliation),
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                compactor=compactor,
            )

    async def _reconcile_uncertain_compaction_budget_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        compactor: str,
    ) -> AsyncIterator[Event]:
        async for (
            reconciliation
        ) in self._run_limit_controller.reconcile_uncertain_operation_reservations(
            reservations,
            reason="compaction reservation lease lost; charged reserved amount",
        ):
            yield _application_compaction_ledger_event(
                event_type=EventType.BUDGET_RECONCILED,
                payload=budget_reconciliation_payload(reconciliation),
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                compactor=compactor,
            )

    async def _release_compaction_budget_reservations(
        self,
        reservations: list[BudgetStepReservation],
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        request: CompactSessionRequest,
        operation_id: str,
        attempt_id: str,
        compactor: str,
        reason: str,
    ) -> AsyncIterator[Event]:
        async for reconciliation in self._run_limit_controller.release_operation_reservations(
            reservations,
            reason=reason,
        ):
            yield _application_compaction_ledger_event(
                event_type=EventType.BUDGET_RESERVATION_RELEASED,
                payload=budget_reconciliation_payload(reconciliation),
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                compactor=compactor,
            )

    async def interrupt_session(self, request: InterruptSessionRequest) -> AsyncIterator[Event]:
        if type(request) is not InterruptSessionRequest:
            raise TypeError("Runtime interruption requires an InterruptSessionRequest.")
        request = copy_interrupt_session_request(request)
        loaded_session = await self.session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")
        if loaded_session.status == SessionStatus.INTERRUPTED:
            existing_interrupt_event = (
                await self._session_control.wait_for_active_interrupted_event(loaded_session.id)
            )
            if existing_interrupt_event is not None:
                retry_event: Event | None = None
                retry_request: dict[str, Any] | None = None
                if not interruption_cascade_suppressed():
                    marker = await self._load_pending_interruption_cascade(loaded_session.id)
                    if (
                        marker is not None
                        and await self.interruption_cascade_status(loaded_session.id) == "failed"
                    ):
                        retry_request = {
                            "retry_request_id": str(uuid4()),
                            "reason": request.reason,
                            "metadata": request.metadata,
                            "requested_by": resolution_actor_payload(request.requested_by),
                        }
                        retry_event = await self._event_writer.emit(
                            Event(
                                type=EventType.SESSION_INTERRUPTION_CASCADE_RETRY_REQUESTED,
                                session_id=loaded_session.id,
                                agent_name=loaded_session.agent_name,
                                environment_name=loaded_session.environment_name,
                                payload={
                                    "interruption_type": (_INTERRUPTION_TYPE_OPERATOR_REQUESTED),
                                    "attempt_id": marker["attempt_id"],
                                    "previous_generation": marker.get("generation", 0),
                                    **_interruption_cascade_retry_event_payload(
                                        _copy_interruption_cascade_retry_request(retry_request)
                                    ),
                                },
                            )
                        )
                    self._schedule_background_interruption_cascade(
                        parent_session_id=loaded_session.id,
                        interrupt_payload=existing_interrupt_event.payload,
                        create_if_missing=False,
                        retry_request=retry_request,
                    )
                yield existing_interrupt_event
                if retry_event is not None:
                    yield retry_event
                return
            raise RuntimeError(
                f"Session is interrupted but has no session.interrupted event: {loaded_session.id}"
            )

        if loaded_session.status == SessionStatus.INTERRUPTING:
            pending_interrupt_payload = await self._load_pending_session_interrupt_payload(
                loaded_session.id,
                default={},
            )
            existing_interrupt_event = (
                await self._session_control.wait_for_active_interrupted_event(
                    loaded_session.id,
                    interruption_request_id=interruption_request_id_from_payload(
                        pending_interrupt_payload
                    ),
                )
            )
            if existing_interrupt_event is not None:
                if not interruption_cascade_suppressed():
                    self._schedule_background_interruption_cascade(
                        parent_session_id=loaded_session.id,
                        interrupt_payload=existing_interrupt_event.payload,
                        create_if_missing=False,
                    )
                yield existing_interrupt_event
                return
            raise TimeoutError(f"Session interruption is still finalizing: {loaded_session.id}")

        if loaded_session.status not in _INTERRUPTIBLE_SESSION_STATUSES:
            raise ValueError(f"Session cannot be interrupted from status: {loaded_session.status}")
        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )

        interrupt_payload = {
            "reason": request.reason,
            "metadata": request.metadata,
            "requested_by": resolution_actor_payload(request.requested_by),
            "interruption_type": _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
            "interruption_request_id": str(uuid4()),
        }
        cascade_suppressed = interruption_cascade_suppressed()
        self._session_control.begin_interruption_request(loaded_session.id)
        request_marker_active = True
        try:
            session = await self.session_store.transition_status_and_checkpoint(
                loaded_session.id,
                from_statuses=_INTERRUPTIBLE_SESSION_STATUSES,
                to_status=SessionStatus.INTERRUPTING,
                checkpoint_transform=_checkpoint_with_pending_session_interrupt(
                    interrupt_payload,
                    include_interruption_cascade=not cascade_suppressed,
                    cascade_created_at=self._clock(),
                ),
            )
            self._session_control.signal_interrupt(session.id)
            active_work_signalled = self._session_control.cancel_active_runs(session.id)
            if active_work_signalled:
                existing_interrupt_event = (
                    await self._session_control.wait_for_active_interrupted_event(
                        session.id,
                        interruption_request_id=interrupt_payload["interruption_request_id"],
                    )
                )
                if existing_interrupt_event is not None:
                    request_marker_active = False
                    self._session_control.end_interruption_request(loaded_session.id)
                    yield existing_interrupt_event
                    return
                raise TimeoutError(f"Session interruption is still finalizing: {session.id}")
            if loaded_session.status == SessionStatus.RUNNING:
                existing_interrupt_event = (
                    await self._session_control.wait_for_active_interrupted_event(
                        session.id,
                        interruption_request_id=interrupt_payload["interruption_request_id"],
                    )
                )
                if existing_interrupt_event is not None:
                    request_marker_active = False
                    self._session_control.end_interruption_request(loaded_session.id)
                    yield existing_interrupt_event
                    return
                raise TimeoutError(f"Session interruption is still finalizing: {session.id}")
        except ValueError:
            reloaded_session = await self.session_store.load(loaded_session.id)
            if reloaded_session is None:
                raise KeyError(f"Session not found: {loaded_session.id}") from None
            if reloaded_session.status in INTERRUPT_REQUESTED_SESSION_STATUSES:
                self._session_control.signal_interrupt(reloaded_session.id)
                pending_interrupt_payload = await self._load_pending_session_interrupt_payload(
                    reloaded_session.id,
                    default={},
                )
                existing_interrupt_event = (
                    await self._session_control.wait_for_active_interrupted_event(
                        reloaded_session.id,
                        interruption_request_id=interruption_request_id_from_payload(
                            pending_interrupt_payload
                        ),
                    )
                )
                if existing_interrupt_event is not None:
                    request_marker_active = False
                    self._session_control.end_interruption_request(loaded_session.id)
                    if not interruption_cascade_suppressed():
                        self._schedule_background_interruption_cascade(
                            parent_session_id=reloaded_session.id,
                            interrupt_payload=existing_interrupt_event.payload,
                            create_if_missing=False,
                        )
                    yield existing_interrupt_event
                    return
                if self._session_control.has_active_tasks(reloaded_session.id):
                    raise TimeoutError(
                        f"Session interruption is still finalizing: {reloaded_session.id}"
                    ) from None
                if reloaded_session.status == SessionStatus.INTERRUPTING:
                    raise TimeoutError(
                        f"Session interruption is still finalizing: {reloaded_session.id}"
                    ) from None
                raise RuntimeError(
                    f"Session is interrupted but has no session.interrupted event: "
                    f"{reloaded_session.id}"
                ) from None
            else:
                raise
        except BaseException:
            if request_marker_active:
                self._session_control.end_interruption_request(loaded_session.id)
            raise

        session = await self.session_store.update_status(session.id, SessionStatus.INTERRUPTED)
        payload = await self._load_pending_session_interrupt_payload(
            session.id,
            default={
                "reason": request.reason,
                "metadata": request.metadata,
                "requested_by": resolution_actor_payload(request.requested_by),
                "interruption_type": _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
                "interruption_request_id": interrupt_payload["interruption_request_id"],
            },
        )
        terminal_event_stream: AsyncIterator[Event] | None = None
        try:
            existing_interrupt_event = await self._session_control.latest_interrupted_event(
                session.id,
                interruption_request_id=interruption_request_id_from_payload(payload),
            )
            if existing_interrupt_event is not None:
                await self._clear_pending_session_interrupt(session.id)
                if not cascade_suppressed:
                    self._schedule_background_interruption_cascade(
                        parent_session_id=session.id,
                        interrupt_payload=existing_interrupt_event.payload,
                        create_if_missing=False,
                    )
                yield existing_interrupt_event
                return
            await self._emit_active_turn_completed_if_needed(
                session=session,
                status=SessionStatus.INTERRUPTED,
            )
            terminal_event_stream = self._emit_terminal_event_with_hooks(
                event=Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=_environment_name(registered_environment),
                    payload=payload,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            terminal_prefix, interrupted_event = await _collect_through_event_type(
                terminal_event_stream,
                EventType.SESSION_INTERRUPTED,
                missing_message="Session interruption produced no terminal event.",
            )

            await self._clear_pending_session_interrupt(session.id)
            if not cascade_suppressed:
                self._schedule_background_interruption_cascade(
                    parent_session_id=session.id,
                    interrupt_payload=interrupted_event.payload,
                    create_if_missing=False,
                )
            for event in terminal_prefix:
                yield event
            async for event in terminal_event_stream:
                yield event
        except Exception:
            if terminal_event_stream is not None:
                with contextlib.suppress(Exception):
                    await _close_async_iterator(terminal_event_stream)
            raise
        finally:
            if request_marker_active:
                self._session_control.end_interruption_request(loaded_session.id)
        return

    async def recover_incomplete_session(
        self,
        request: IncompleteSessionRecoveryRequest,
    ) -> IncompleteSessionRecoveryResult:
        request = copy_incomplete_session_recovery_request(request)
        session = await self.session_store.load(request.session_id)
        if session is None:
            raise KeyError(f"Session not found: {request.session_id}") from None
        return await self._recover_incomplete_session_scoped(
            session=session,
            inactive_before=request.inactive_before,
            reason=request.reason,
            metadata=request.metadata,
        )

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
    ) -> list[IncompleteSessionRecoveryResult]:
        """Sweep non-terminal sessions and repair each one, fault-isolated.

        Returns one result per swept session. A session whose agent is not
        registered in this process is reported as
        ``SKIPPED_UNREGISTERED_AGENT``; an unexpected per-session failure is
        reported as ``FAILED`` with the error in ``message`` — neither aborts
        the sweep, so one bad row cannot strand every healthy session. A
        ``FAILED`` entry's ``previous_status`` comes from the sweep's listing
        snapshot; its ``status`` is the current stored status when the session
        can still be reloaded (a failed recovery may have progressed it),
        falling back to the snapshot when it cannot. Session listing failures
        and cancellation still raise.
        """
        request = copy_incomplete_sessions_recovery_request(request)
        sessions: list[Session] = []
        seen_session_ids: set[str] = set()
        for status in (
            SessionStatus.INTERRUPTING,
            SessionStatus.RUNNING,
            SessionStatus.PENDING,
        ):
            if status not in request.statuses:
                continue
            if len(sessions) >= request.limit:
                break
            candidates = (
                await self.session_store.list_sessions(
                    SessionQuery(
                        status=status,
                        last_activity_before=request.inactive_before,
                        limit=min(1000, request.limit - len(sessions)),
                    )
                )
            ).sessions
            for candidate in candidates:
                if (
                    request.inactive_before is not None
                    and candidate.last_activity_at > request.inactive_before
                ):
                    continue
                if candidate.id in seen_session_ids:
                    continue
                seen_session_ids.add(candidate.id)
                sessions.append(candidate)
                if len(sessions) >= request.limit:
                    break

        results: list[IncompleteSessionRecoveryResult] = []
        for session in sessions:
            # Isolate per-session errors: one broken session must not strand
            # the sweep (see docstring).
            try:
                result = await self._recover_incomplete_session_scoped(
                    session=session,
                    inactive_before=request.inactive_before,
                    reason=request.reason,
                    metadata=request.metadata,
                )
            except Exception as exc:
                logger.warning(
                    "Recovery failed for session %s (agent %s): %s",
                    session.id,
                    session.agent_name,
                    exc,
                )
                # The failed recovery may have progressed the stored status
                # before raising; report the current status when the session
                # can still be reloaded.
                try:
                    reloaded = await self.session_store.load(session.id)
                except Exception:
                    reloaded = None
                result = IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=session.status,
                    status=session.status if reloaded is None else reloaded.status,
                    actions=(IncompleteSessionRecoveryAction.FAILED,),
                    message=f"Recovery failed: {type(exc).__name__}: {exc}",
                )
            results.append(result)
        return results

    async def dispatch(self, request: DispatchRequest) -> DispatchHandle:
        if type(request) is not DispatchRequest:
            raise TypeError("Runtime dispatch requires a DispatchRequest.")
        request = copy_dispatch_request(request)
        handle = await self.dispatcher.submit(self, request)
        _validate_dispatch_handle_for_request(handle=handle, request=request)
        return copy_dispatch_handle(handle)

    async def dispatch_inline(self, request: DispatchRequest) -> AsyncIterator[Event]:
        if type(request) is not DispatchRequest:
            raise TypeError("Inline dispatch requires a DispatchRequest.")
        request = copy_dispatch_request(request)
        if request.task_id is not None and self.task_store is None:
            raise RuntimeError("task_store is required when DispatchRequest.task_id is set.")
        resume_request = ResumeRequest(
            session_id=request.session_id,
            messages=request.messages,
            model=request.model,
            metadata=request.metadata,
            max_steps=request.max_steps,
            limits=request.limits,
            budget_limits=request.budget_limits,
            retry_policy=request.retry_policy,
            structured_output=request.structured_output,
            thinking=request.thinking,
            loop_policies=request.loop_policies,
        )
        start_event_payload_extra = {"dispatch_id": request.dispatch_id}
        if request.task_id is not None:
            start_event_payload_extra["task_id"] = request.task_id
        session_stream = self._resume_session(
            request=resume_request,
            task_id=request.task_id,
            start_event_payload_extra=start_event_payload_extra,
            start_task_on_enter=True,
        )
        try:
            async for event in self._session_control.stream_with_out_of_band_events(
                request.session_id,
                session_stream,
            ):
                yield event
        except GeneratorExit:
            await session_stream.aclose()
            raise

    async def create_task(self, request: TaskCreate) -> Task:
        if type(request) is not TaskCreate:
            raise TypeError("Task creation requires a TaskCreate request.")
        if self.task_store is None:
            raise RuntimeError("task_store is required to create tasks.")
        return await self.task_store.create_task(copy_task_create(request))

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
        session_id = require_clean_nonblank(session_id, "session_id")
        session = await self.session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}") from None
        events = await self._run_limit_controller.session_usage_events(session_id)
        return session_usage_summary(session_id, events)

    async def get_causal_budget_usage(
        self,
        causal_budget_id: str,
    ) -> CausalBudgetUsageSummary:
        causal_budget_id = require_clean_nonblank(causal_budget_id, "causal_budget_id")
        sessions = await self._list_all_sessions(
            SessionQuery(
                causal_budget_id=causal_budget_id,
                order_by=SessionOrder.CREATED_AT_ASC,
            )
        )
        if not sessions:
            raise KeyError(f"Causal budget not found: {causal_budget_id}") from None
        records = await self._query_all_event_records(
            EventQuery(
                causal_budget_id=causal_budget_id,
                event_types=USAGE_BEARING_EVENT_TYPES,
            )
        )
        events = [record.event for record in records]
        return causal_budget_usage_summary(
            causal_budget_id=causal_budget_id,
            session_ids=[session.id for session in sessions],
            events=events,
        )

    async def _list_all_sessions(self, query: SessionQuery) -> list[Session]:
        sessions: list[Session] = []
        offset = query.offset
        while True:
            page = (
                await self.session_store.list_sessions(
                    SessionQuery(
                        status=query.status,
                        agent_name=query.agent_name,
                        environment_name=query.environment_name,
                        parent_session_id=query.parent_session_id,
                        causal_budget_id=query.causal_budget_id,
                        limit=query.limit,
                        offset=offset,
                        order_by=query.order_by,
                    )
                )
            ).sessions
            if not page:
                return sessions
            sessions.extend(page)
            if len(page) < query.limit:
                return sessions
            offset += len(page)

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

                    try:
                        await run_event_watcher_handler(
                            watcher,
                            EventWatcherContext(
                                watcher_name=watcher.name,
                                record=record,
                                attempt=claim.attempt,
                            ),
                        )
                    except Exception as exc:
                        delivery = await self.event_watcher_store.mark_failure(
                            claim,
                            error=event_watcher_error_payload(exc),
                            max_attempts=watcher.max_attempts,
                        )
                        deliveries.append(delivery)
                        remaining -= 1
                        processed_for_watcher += 1
                        if delivery.status is not EventWatcherDeliveryStatus.DEAD_LETTERED:
                            should_fetch_next_page = False
                            break
                        continue

                    delivery = await self.event_watcher_store.mark_success(claim)
                    deliveries.append(delivery)
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
        session_id = require_clean_nonblank(session_id, "session_id")
        session = await self.session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}") from None
        # Cost derives only from model.completed events; skip the rest of the log.
        cost_event_records = await self._query_all_event_records(
            EventQuery(
                session_id=session_id,
                event_type=EventType.MODEL_COMPLETED,
            )
        )
        return estimate_session_cost(
            session_id=session_id,
            events=[record.event for record in cost_event_records],
            pricing=pricing,
            currency=currency,
        )

    async def get_causal_budget_cost(
        self,
        causal_budget_id: str,
        pricing: PriceBook,
        *,
        currency: str = "USD",
    ) -> CausalBudgetCostSummary:
        causal_budget_id = require_clean_nonblank(causal_budget_id, "causal_budget_id")
        sessions = await self._list_all_sessions(
            SessionQuery(
                causal_budget_id=causal_budget_id,
                order_by=SessionOrder.CREATED_AT_ASC,
            )
        )
        if not sessions:
            raise KeyError(f"Causal budget not found: {causal_budget_id}") from None
        records = await self._query_all_event_records(
            EventQuery(
                causal_budget_id=causal_budget_id,
                event_type=EventType.MODEL_COMPLETED,
            )
        )
        return estimate_causal_budget_cost(
            causal_budget_id=causal_budget_id,
            session_ids=[session.id for session in sessions],
            events=[record.event for record in records],
            pricing=pricing,
            currency=currency,
        )

    async def emit_hook_event(
        self,
        *,
        session_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        event_type = require_nonblank(event_type, "event_type")
        if not event_type.startswith("custom."):
            raise ValueError("Hook-emitted custom events must use the custom. namespace.")
        event = Event(
            type=event_type,
            session_id=session_id,
            payload=copy_json_value(payload or {}, "payload"),
        )
        return await self._event_writer.emit(event)

    async def _resume_session(
        self,
        *,
        request: ResumeRequest,
        task_id: str | None,
        start_event_payload_extra: dict[str, Any],
        start_task_on_enter: bool,
    ) -> AsyncGenerator[Event, None]:
        loaded_session = await self.session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")

        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        registered_provider = self._get_registered_provider(loaded_session.provider_name)
        # Checked before the status transition so it surfaces to the caller.
        _require_native_structured_output_support(
            request.structured_output, registered_provider=registered_provider
        )
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )

        def reject_unresumable_checkpoint(
            current_session: Session,
            current_checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            updated_checkpoint = (
                None
                if current_checkpoint is None
                else copy_json_value(current_checkpoint, "checkpoint")
            )
            if (
                updated_checkpoint is not None
                and _SESSION_OPERATIONS_CHECKPOINT_KEY in updated_checkpoint
            ):
                operations = _session_operation_state(updated_checkpoint)
                _abandon_expired_session_operation(operations, now=self._clock())
                updated_checkpoint[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
            active_operation_id = _active_session_operation_id(updated_checkpoint)
            if active_operation_id is not None:
                raise RuntimeError(
                    f"Session has an active durable operation: {active_operation_id}"
                )
            if approval_support.pending_approval_from_checkpoint(updated_checkpoint) is not None:
                raise RuntimeError(
                    "Session has a pending tool approval. Resolve it with "
                    "resolve_tool_approval(...) before resuming with new messages."
                )
            if pending_user_input_from_checkpoint(updated_checkpoint) is not None:
                raise RuntimeError(
                    "Session is awaiting user input. Answer it with "
                    "resolve_user_input(...) before resuming with new messages."
                )
            if (
                current_session.status == SessionStatus.COMPLETED
                and tool_round_recovery.pending_tool_round_from_checkpoint(updated_checkpoint)
                is not None
            ):
                raise RuntimeError(
                    "Completed session has an inconsistent pending tool round. "
                    "Inspect or recover the session state before resuming it."
                )
            if (
                updated_checkpoint is not None
                and _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY in updated_checkpoint
            ):
                raise RuntimeError(
                    "Session has an incomplete background interruption cascade. "
                    "Retry the interruption before resuming with new messages."
                )
            return updated_checkpoint

        # Report deterministic checkpoint conflicts before claiming the session,
        # then repeat the same validation inside the atomic transition below so a
        # concurrent checkpoint update cannot bypass the guard.
        checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
        reject_unresumable_checkpoint(loaded_session, checkpoint)

        session = await self.session_store.transition_status_and_checkpoint(
            loaded_session.id,
            from_statuses=_RESUMABLE_SESSION_STATUSES,
            to_status=SessionStatus.RUNNING,
            checkpoint_transform=reject_unresumable_checkpoint,
        )
        try:
            if request.model is not None:
                session = await self.session_store.update_model(session.id, request.model)
            transcript = await self.session_store.load_transcript(session.id)
        except Exception as exc:
            try:
                await self.session_store.update_status(session.id, SessionStatus.FAILED)
                yield await self._event_writer.emit(
                    Event(
                        type=EventType.SESSION_FAILED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        payload={
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )
                )
            finally:
                await self.session_store.release_run_fence(session.id)
            return
        except BaseException:
            await self.session_store.release_run_fence(session.id)
            raise
        messages = transcript + request.messages

        session_stream = self._run_session(
            session=session,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            messages=messages,
            messages_to_append=request.messages,
            max_steps=request.max_steps,
            limits=request.limits,
            budget_limits=request.budget_limits,
            retry_policy=self._effective_retry_policy(request.retry_policy),
            structured_output=request.structured_output,
            thinking=request.thinking,
            request_loop_policies=request.loop_policies,
            request_metadata=request.metadata,
            task_id=task_id,
            task_worker_id=None,
            start_event_type=EventType.SESSION_RESUMED,
            start_event_payload={
                "agent_name": registered_agent.spec.name,
                "appended_messages": len(request.messages),
                **copy_json_value(start_event_payload_extra, "start_event_payload_extra"),
            },
            start_task_on_enter=start_task_on_enter,
        )
        try:
            async for event in session_stream:
                yield event
        except GeneratorExit:
            await session_stream.aclose()
            raise

    async def fork_session(self, request: ForkSessionRequest) -> AsyncIterator[Event]:
        if type(request) is not ForkSessionRequest:
            raise TypeError("Runtime fork requires a ForkSessionRequest.")
        request = copy_fork_session_request(request)
        source_session = await self.session_store.load(request.source_session_id)
        if source_session is None:
            raise KeyError(f"Session not found: {request.source_session_id}")
        if source_session.status not in _FORKABLE_SESSION_STATUSES:
            raise ValueError(
                "Only completed, failed, or interrupted sessions can be forked: "
                f"{source_session.status}"
            )
        if request.transcript_cursor is not None and request.copy_checkpoint:
            raise ValueError(
                "ForkSessionRequest.copy_checkpoint must be false when transcript_cursor is set."
            )
        if source_session.status == SessionStatus.INTERRUPTED and not request.copy_checkpoint:
            raise ValueError("Interrupted sessions cannot be forked without checkpoint state.")

        source_checkpoint = await self.session_store.load_checkpoint(source_session.id)
        if (
            source_checkpoint is not None
            and _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY in source_checkpoint
        ):
            raise RuntimeError(
                "Session has an incomplete background interruption cascade. "
                "Retry the interruption before forking it."
            )

        registered_provider = self._get_registered_provider(source_session.provider_name)
        try:
            source_registered_agent = self._get_registered_agent(source_session.agent_name)
        except KeyError as exc:
            raise KeyError(
                "Source agent must be registered to derive inherited taint before forking: "
                f"{source_session.agent_name}"
            ) from exc
        agent_name = request.agent_name or source_session.agent_name
        registered_agent = self._get_registered_agent(agent_name)
        if (
            request.agent_name is not None
            and registered_agent.spec.provider_name is not None
            and registered_agent.spec.provider_name != source_session.provider_name
        ):
            raise ValueError(
                "Forking a session to an agent with a different provider is not supported: "
                f"{registered_agent.spec.provider_name} != {source_session.provider_name}"
            )
        model = request.model or (
            registered_agent.spec.model if request.agent_name is not None else source_session.model
        )
        environment_name = (
            request.environment_name
            if request.environment_name is not None
            else source_session.environment_name
        )
        registered_environment = self._get_registered_environment_for_session(environment_name)

        checkpoint_transform = None
        if request.copy_checkpoint:

            def checkpoint_transform(
                current_source: Session,
                source_checkpoint: dict[str, Any] | None,
            ) -> dict[str, Any] | None:
                if (
                    source_checkpoint is not None
                    and _SESSION_OPERATIONS_CHECKPOINT_KEY in source_checkpoint
                ):
                    operations = _session_operation_state(source_checkpoint)
                    _abandon_expired_session_operation(operations, now=self._clock())
                    source_checkpoint[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
                active_operation_id = _active_session_operation_id(source_checkpoint)
                if active_operation_id is not None:
                    raise RuntimeError(
                        f"Session has an active durable operation: {active_operation_id}"
                    )
                if current_source.status == SessionStatus.INTERRUPTED and source_checkpoint is None:
                    raise RuntimeError(
                        "Interrupted session cannot be forked because checkpoint state is missing."
                    )
                if pending_user_input_from_checkpoint(source_checkpoint) is not None:
                    raise RuntimeError(
                        "Session awaiting user input cannot be forked; answer it with "
                        "resolve_user_input(...) first."
                    )
                if (
                    source_checkpoint is not None
                    and _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY in source_checkpoint
                ):
                    raise RuntimeError(
                        "Session has an incomplete background interruption cascade. "
                        "Retry the interruption before forking it."
                    )
                fork_checkpoint = approval_support.checkpoint_for_fork(
                    checkpoint=source_checkpoint,
                    agent_name=agent_name,
                    environment_name=environment_name,
                )
                if fork_checkpoint is not None:
                    fork_checkpoint.pop(_SESSION_OPERATIONS_CHECKPOINT_KEY, None)
                return fork_checkpoint
        else:

            def checkpoint_transform(
                _current_source: Session,
                source_checkpoint: dict[str, Any] | None,
            ) -> None:
                if (
                    source_checkpoint is not None
                    and _SESSION_OPERATIONS_CHECKPOINT_KEY in source_checkpoint
                ):
                    operations = _session_operation_state(source_checkpoint)
                    _abandon_expired_session_operation(operations, now=self._clock())
                    source_checkpoint[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
                active_operation_id = _active_session_operation_id(source_checkpoint)
                if active_operation_id is not None:
                    raise RuntimeError(
                        f"Session has an active durable operation: {active_operation_id}"
                    )
                return None

        inherited_taint_labels = await self._tool_round_executor.prior_taint_labels_for_policy(
            session_id=source_session.id,
            policy=source_registered_agent.tool_policy,
            request_metadata=source_session.metadata,
        )
        fork_metadata = request.metadata
        if inherited_taint_labels:
            fork_metadata = metadata_with_taint_labels(
                request.metadata,
                inherited_taint_labels | taint_labels_from_metadata(request.metadata),
            )

        fork_session = Session(
            id=request.session_id or str(uuid4()),
            agent_name=agent_name,
            provider_name=registered_provider.name,
            model=model,
            parent_session_id=source_session.id,
            causal_budget_id=source_session.causal_budget_id,
            runtime_name=source_session.runtime_name,
            runtime_version=source_session.runtime_version,
            environment_name=environment_name,
            status=source_session.status,
            labels=source_session.labels,
            metadata=copy_json_value(fork_metadata, "metadata"),
        )
        created = await self.session_store.create_fork(
            source_session_id=source_session.id,
            fork=fork_session,
            source_statuses=_FORKABLE_SESSION_STATUSES,
            transcript_cursor=request.transcript_cursor,
            checkpoint_transform=checkpoint_transform,
            expected_source_run_epoch=source_session.run_epoch,
        )
        yield await self._event_writer.emit(
            Event(
                type=EventType.SESSION_FORKED,
                session_id=created.id,
                agent_name=registered_agent.spec.name,
                environment_name=_environment_name(registered_environment),
                payload={
                    "source_session_id": source_session.id,
                    "source_status": source_session.status.value,
                    "parent_session_id": created.parent_session_id,
                    "causal_budget_id": created.causal_budget_id,
                    "transcript_cursor": request.transcript_cursor,
                    "copy_checkpoint": request.copy_checkpoint,
                    "agent_name": created.agent_name,
                    "provider_name": created.provider_name,
                    "model": created.model,
                    "environment_name": created.environment_name,
                    "inherited_taint_labels": sorted(inherited_taint_labels),
                },
            )
        )

    def _run_recovery_session(
        self,
        request: RecoverySessionRunRequest,
    ) -> AsyncGenerator[Event, None]:
        return self._run_session(
            session=request.session,
            registered_agent=request.registered_agent,
            registered_provider=request.registered_provider,
            registered_environment=request.registered_environment,
            messages=request.messages,
            messages_to_append=request.messages_to_append,
            max_steps=request.max_steps,
            limits=request.limits,
            budget_limits=request.budget_limits,
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
        )

    async def _cleanup_recovery_handoff(
        self,
        *,
        stream: AsyncGenerator[Event, None],
        session_id: str,
        authoritative_failure: BaseException | None,
        finalize_abandoned: bool,
        release_run_fence: bool,
    ) -> None:
        cleanup_steps: list[tuple[str, Callable[[], Awaitable[None]]]] = [
            ("nested stream close", stream.aclose)
        ]
        if finalize_abandoned:
            cleanup_steps.append(
                (
                    "abandoned session finalization",
                    lambda: self._finalize_abandoned_session_by_id(session_id),
                )
            )
        if release_run_fence:
            cleanup_steps.append(
                ("run fence release", lambda: self.session_store.release_run_fence(session_id))
            )
        try:
            await _run_recovery_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=tuple(cleanup_steps),
            )
        finally:
            # Cancellation-resistant cleanup may run in a child task with a copied
            # context. Clear the caller's task-local epoch as the handoff ends too.
            _deactivate_session_run_fence(session_id)

    async def _transition_recovery_session_to_running(self, session_id: str) -> Session:
        """Claim a paused session without leaving cancellation outcome-uncertain.

        Session stores activate run fences in task-local context after the durable
        transition commits. Keeping the transition in a shielded child task lets it
        reach a definite result even if the caller is cancelled at that boundary.
        A successful claim is then activated in the caller's context. If cancellation
        arrived, the claim is finalized and released before that cancellation is
        propagated.
        """
        transition_task = asyncio.create_task(
            self.session_store.transition_status(
                session_id,
                from_statuses={SessionStatus.INTERRUPTED},
                to_status=SessionStatus.RUNNING,
            )
        )
        cancellation: asyncio.CancelledError | None = None
        transition_failure: BaseException | None = None
        while not transition_task.done():
            try:
                await asyncio.shield(transition_task)
            except asyncio.CancelledError as exc:
                if transition_task.cancelled():
                    transition_failure = exc
                    break
                if cancellation is None:
                    cancellation = exc
            except BaseException as exc:
                transition_failure = exc
                break

        session: Session | None = None
        if transition_failure is None:
            try:
                session = transition_task.result()
            except BaseException as exc:
                transition_failure = exc

        if session is None:
            if cancellation is not None:
                if transition_failure is not None:
                    cancellation.add_note(
                        "Continuation recovery transition also failed after cancellation: "
                        f"{type(transition_failure).__name__}."
                    )
                raise cancellation from transition_failure
            if transition_failure is None:
                raise RuntimeError("Continuation recovery transition completed without a session.")
            raise transition_failure

        # transition_status() activated this epoch only in the child task's copied
        # context. The caller owns all subsequent writes and cleanup.
        _activate_session_run_fence(session)
        if cancellation is None:
            return session

        try:
            await _run_recovery_cleanup_steps(
                authoritative_failure=cancellation,
                steps=(
                    (
                        "abandoned session finalization",
                        lambda: self._finalize_abandoned_session_by_id(session.id),
                    ),
                    (
                        "run fence release",
                        lambda: self.session_store.release_run_fence(session.id),
                    ),
                ),
            )
        finally:
            # Shielded cleanup runs in a copied context. Never leave the caller's
            # task-local epoch active if it catches and handles the cancellation.
            _deactivate_session_run_fence(session.id)
        raise cancellation

    async def resolve_user_input(
        self,
        response: UserInputResponse,
    ) -> AsyncIterator[Event]:
        """Resume a session paused by ``ask_user`` with the user's answer.

        The answer becomes the ``ask_user`` tool result; any other tool calls in the same
        round (none ran before the pause) execute now, and the session continues.
        """
        if type(response) is not UserInputResponse:
            raise TypeError("Runtime user input resolution requires a UserInputResponse.")
        response = copy_user_input_response(response)
        loaded_session = await self.session_store.load(response.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {response.session_id}")

        checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
        pending = pending_user_input_from_checkpoint(checkpoint)
        if pending is None:
            raise RuntimeError("Session has no pending user input.")
        if pending.input_id != response.input_id:
            raise ValueError(f"User input id does not match pending input: {response.input_id}")
        # The output-schema contract is fixed by the paused run's provider history; a resolver
        # cannot swap it (a spec matching or absent is fine; a differing one is rejected). Checked
        # before the status transition so it surfaces to the caller rather than being caught by the
        # resume's failure handler. (thinking is a safe override.)
        effective_structured_output = _effective_user_input_structured_output(
            structured_output=response.structured_output,
            pending=pending,
        )

        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        registered_provider = self._get_registered_provider(loaded_session.provider_name)
        _require_native_structured_output_support(
            effective_structured_output, registered_provider=registered_provider
        )
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )
        session = await self._transition_recovery_session_to_running(loaded_session.id)

        continuation_stream = self._recovery_coordinator.continue_user_input_resolution(
            response=response,
            session=session,
            pending=pending,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
        )
        authoritative_failure: BaseException | None = None
        abandoned = False
        try:
            async for event in continuation_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            abandoned = isinstance(exc, GeneratorExit | asyncio.CancelledError)
            raise
        finally:
            await self._cleanup_recovery_handoff(
                stream=continuation_stream,
                session_id=session.id,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=abandoned,
                release_run_fence=True,
            )

    async def recover_user_input(
        self,
        request: UserInputRecoveryRequest,
    ) -> AsyncIterator[Event]:
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
        loaded_session = await self.session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")

        checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
        pending = pending_user_input_from_checkpoint(checkpoint)
        if pending is None:
            raise RuntimeError("Session has no pending user input.")
        if pending.input_id != request.input_id:
            raise ValueError(f"User input id does not match pending input: {request.input_id}")
        effective_structured_output = _effective_user_input_structured_output(
            structured_output=request.structured_output,
            pending=pending,
        )

        pending_tool_call = approval_support.round_tool_call_for_recovery(
            pending_calls=pending.tool_calls,
            tool_call_id=request.tool_call_id,
        )
        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        registered_provider = self._get_registered_provider(loaded_session.provider_name)
        _require_native_structured_output_support(
            effective_structured_output, registered_provider=registered_provider
        )
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )
        session = await self._transition_recovery_session_to_running(loaded_session.id)
        recovery_stream = self._recovery_coordinator.recover_user_input(
            request=request,
            loaded_session=loaded_session,
            session=session,
            pending=pending,
            pending_tool_call=pending_tool_call,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
        )
        authoritative_failure: BaseException | None = None
        try:
            async for event in recovery_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await self._cleanup_recovery_handoff(
                stream=recovery_stream,
                session_id=session.id,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=False,
                release_run_fence=False,
            )

    async def resolve_tool_approval(
        self,
        request: ToolApprovalRequest,
    ) -> AsyncIterator[Event]:
        if type(request) is not ToolApprovalRequest:
            raise TypeError("Runtime approval resolution requires a ToolApprovalRequest.")
        request = _validate_tool_approval_request(request)
        loaded_session = await self.session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")

        checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
        pending_approval = approval_support.pending_approval_from_checkpoint(checkpoint)
        if pending_approval is None:
            raise RuntimeError("Session has no pending tool approval.")
        if pending_approval.approval_id != request.approval_id:
            raise ValueError(
                f"Tool approval id does not match pending approval: {request.approval_id}"
            )
        effective_structured_output = _effective_approval_structured_output(
            structured_output=request.structured_output,
            pending_approval=pending_approval,
        )

        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        registered_provider = self._get_registered_provider(loaded_session.provider_name)
        _require_native_structured_output_support(
            effective_structured_output, registered_provider=registered_provider
        )
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )
        session = await self._transition_recovery_session_to_running(loaded_session.id)

        continuation_stream = self._recovery_coordinator.continue_tool_approval_resolution(
            request=request,
            session=session,
            pending_approval=pending_approval,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
        )
        authoritative_failure: BaseException | None = None
        abandoned = False
        try:
            async for event in continuation_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            abandoned = isinstance(exc, GeneratorExit | asyncio.CancelledError)
            raise
        finally:
            await self._cleanup_recovery_handoff(
                stream=continuation_stream,
                session_id=session.id,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=abandoned,
                release_run_fence=True,
            )

    async def recover_tool_approval(
        self,
        request: ToolApprovalRecoveryRequest,
    ) -> AsyncIterator[Event]:
        if type(request) is not ToolApprovalRecoveryRequest:
            raise TypeError("Runtime approval recovery requires a ToolApprovalRecoveryRequest.")
        request = _validate_tool_approval_recovery_request(request)
        loaded_session = await self.session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")

        checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
        pending_approval = approval_support.pending_approval_from_checkpoint(checkpoint)
        if pending_approval is None:
            raise RuntimeError("Session has no pending tool approval.")
        if pending_approval.approval_id != request.approval_id:
            raise ValueError(
                f"Tool approval id does not match pending approval: {request.approval_id}"
            )
        effective_structured_output = _effective_approval_structured_output(
            structured_output=request.structured_output,
            pending_approval=pending_approval,
        )

        pending_tool_call = approval_support.pending_tool_call_for_recovery(
            approval=pending_approval,
            tool_call_id=request.tool_call_id,
        )
        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        registered_provider = self._get_registered_provider(loaded_session.provider_name)
        _require_native_structured_output_support(
            effective_structured_output, registered_provider=registered_provider
        )
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )
        session = await self._transition_recovery_session_to_running(loaded_session.id)
        recovery_stream = self._recovery_coordinator.recover_tool_approval(
            request=request,
            loaded_session=loaded_session,
            session=session,
            pending_approval=pending_approval,
            pending_tool_call=pending_tool_call,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
        )
        authoritative_failure: BaseException | None = None
        try:
            async for event in recovery_stream:
                yield event
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await self._cleanup_recovery_handoff(
                stream=recovery_stream,
                session_id=session.id,
                authoritative_failure=authoritative_failure,
                finalize_abandoned=False,
                release_run_fence=False,
            )

    async def recover_tool_round(
        self,
        request: ToolRoundRecoveryRequest,
    ) -> AsyncIterator[Event]:
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
        status (a process kill), so FAILED, RUNNING, and INTERRUPTING are accepted
        alongside INTERRUPTED; the in-process claim registered while this recovery
        streams blocks concurrent recoveries and the sweep, but — like the sweep —
        it cannot see work active on another worker. If this call fails AFTER the
        recovered terminal event persisted, the session closes to the resumable
        INTERRUPTED state with the failure on the `session.interrupted` event and
        the evidence stays durable: do not retry the same `tool_call_id` (the
        guard rejects it) — `resume(...)` finishes the round from the persisted
        outcome.
        """
        if type(request) is not ToolRoundRecoveryRequest:
            raise TypeError("Runtime tool round recovery requires a ToolRoundRecoveryRequest.")
        request = copy_tool_round_recovery_request(request)
        loaded_session = await self.session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")

        checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
        if pending_round is None:
            raise RuntimeError("Session has no pending tool round.")
        if pending_round.round_id != request.round_id:
            raise ValueError(f"Tool round id does not match pending round: {request.round_id}")
        effective_structured_output = _effective_tool_round_structured_output(
            structured_output=request.structured_output,
            pending_round=pending_round,
        )

        pending_tool_call = approval_support.round_tool_call_for_recovery(
            pending_calls=pending_round.tool_calls,
            tool_call_id=request.tool_call_id,
        )
        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        if pending_round.agent_name != registered_agent.spec.name:
            raise RuntimeError(
                f"Pending tool round belongs to a different agent: {pending_round.agent_name}."
            )
        registered_provider = self._get_registered_provider(loaded_session.provider_name)
        _require_native_structured_output_support(
            effective_structured_output, registered_provider=registered_provider
        )
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )
        if self._session_control.has_active_tasks(loaded_session.id):
            raise RuntimeError(f"Session has active work in this process: {loaded_session.id}")
        # Reserve the in-process slot before awaiting the durable transition. The
        # check and registration are await-free, so another local recovery cannot
        # advance the run epoch while this claimant is waiting on storage.
        current_task = asyncio.current_task()
        if current_task is not None:
            self._session_control.register_active_task(
                loaded_session.id,
                current_task,
                task_id=None,
                task_started=False,
                task_finished=False,
            )
        try:
            session = await self.session_store.transition_status(
                loaded_session.id,
                from_statuses=_TOOL_ROUND_RECOVERABLE_SESSION_STATUSES,
                to_status=SessionStatus.RUNNING,
            )
        except BaseException:
            if current_task is not None:
                self._session_control.unregister_active_task(loaded_session.id, current_task)
            raise
        try:
            async for event in self._recover_tool_round_claimed(
                request=request,
                original_status=loaded_session.status,
                session=session,
                pending_round=pending_round,
                pending_tool_call=pending_tool_call,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                effective_structured_output=effective_structured_output,
            ):
                yield event
        finally:
            try:
                await self.session_store.release_run_fence(session.id)
            finally:
                if current_task is not None:
                    self._session_control.unregister_active_task(session.id, current_task)

    async def _recover_tool_round_claimed(
        self,
        *,
        request: ToolRoundRecoveryRequest,
        original_status: SessionStatus,
        session: Session,
        pending_round: tool_round_recovery.PendingToolRound,
        pending_tool_call: PendingToolCallApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        effective_structured_output: StructuredOutputSpec | None,
    ) -> AsyncIterator[Event]:
        recovered_result = ToolResult(
            content=request.message,
            structured=request.structured,
            artifacts=request.artifacts,
            is_error=request.outcome == ToolApprovalRecoveryOutcome.FAILED,
        )
        event_type = (
            EventType.TOOL_CALL_FAILED
            if recovered_result.is_error
            else EventType.TOOL_CALL_COMPLETED
        )
        environment_name = _environment_name(registered_environment)
        recovery_persisted = False

        try:
            events = await self.session_store.load_events(session.id)
            tool_round_recovery.validate_tool_round_recovery_target(
                events=events,
                pending_round=pending_round,
                tool_call_id=request.tool_call_id,
            )
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            if factory_started_event is not None:
                yield factory_started_event
            factory_resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=EnvironmentFactoryOperation.RECONNECT,
            )
            registered_environment = factory_resolution.registered_environment
            environment_name = _environment_name(registered_environment)
            for event in factory_resolution.events:
                yield event
            if factory_resolution.error is not None:
                session = await self.session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
                async for event in self._emit_terminal_event_with_hooks(
                    event=Event(
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload={
                            "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                            "tool_round_id": pending_round.round_id,
                            "error": str(factory_resolution.error),
                            "error_type": type(factory_resolution.error).__name__,
                        },
                    ),
                    phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                ):
                    yield event
                return
            recovery_tool_event, recovered_result = tool_results.redact_tool_result_event(
                event=Event(
                    type=event_type,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=pending_tool_call.tool_name,
                    payload={
                        "tool_round_id": pending_round.round_id,
                        "tool_call_id": pending_tool_call.tool_call_id,
                        "idempotency_key": tool_execution.tool_idempotency_key(
                            session_id=session.id,
                            tool_round_id=pending_round.round_id,
                            tool_call_id=pending_tool_call.tool_call_id,
                        ),
                        "manual_recovery": True,
                        "reason": request.reason,
                        "metadata": request.metadata,
                        "resolved_by": resolution_actor_payload(request.resolved_by),
                        "result": recovered_result.model_dump(),
                    },
                ),
                result=recovered_result,
                redactor=self._secret_redactor,
            )
            recovery_events = [
                Event(
                    type=EventType.SESSION_RESUMED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                        "tool_round_id": pending_round.round_id,
                        "tool_call_id": pending_tool_call.tool_call_id,
                        "resolved_by": resolution_actor_payload(request.resolved_by),
                    },
                ),
                recovery_tool_event,
            ]
            emitted_recovery_events = await self._event_writer.emit_many(
                session.id, recovery_events
            )
            recovery_persisted = True
            for event in emitted_recovery_events:
                yield event
            tool_call = runtime_records.ToolCallRequest(
                id=pending_tool_call.tool_call_id,
                name=pending_tool_call.tool_name,
                arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
            )
            tool_event = emitted_recovery_events[-1]
            # Manual recovery persists the operator-supplied result before hooks run, so
            # after_tool_call is observe-only here (v1): the threaded modification is ignored.
            async for event, _modified in self._tool_round_executor.run_tool_call_hooks(
                session=session,
                tool_event=tool_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=recovered_result,
                task_id=pending_round.task_id,
                redactor=self._secret_redactor,
                allow_modification=False,
            ):
                yield event

            events = await self.session_store.load_events(session.id)
            recorded_outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
                events=events,
                pending_round=pending_round,
            )
            remaining_ids = started_ids - set(recorded_outcomes)
            if remaining_ids:
                # One call per invocation: another call in this round also started
                # without a terminal event, so it needs its own operator-verified
                # outcome before the round can close. The result persisted above is
                # durable; the next recover_tool_round reuses it through the
                # recorded-outcome ledger.
                next_call = next(
                    call for call in pending_round.tool_calls if call.tool_call_id in remaining_ids
                )
                session = await self.session_store.update_status(
                    session.id, SessionStatus.INTERRUPTED
                )
                async for event in self._emit_terminal_event_with_hooks(
                    event=Event(
                        type=EventType.SESSION_INTERRUPTED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload={
                            "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                            "manual_recovery_required": True,
                            "tool_round_id": pending_round.round_id,
                            "tool_call_id": next_call.tool_call_id,
                            "tool_name": next_call.tool_name,
                        },
                    ),
                    phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                ):
                    yield event
                return
        except GeneratorExit:
            # Abandonment: finalize to INTERRUPTED (do NOT roll back to a live status).
            await self._finalize_abandoned_session_by_id(session.id)
            raise
        except Exception as exc:
            if not recovery_persisted:
                # Nothing durable happened yet — restore the crashed status so the
                # operator can retry the recovery unchanged.
                await self.session_store.update_status(session.id, original_status)
                raise
            # The operator's terminal event is durable. Rolling back to the original
            # status would strand a stale-live original (RUNNING/INTERRUPTING is not
            # resumable), so close to the resumable INTERRUPTED state with a terminal
            # event carrying the failure — resume() then finishes the round from the
            # persisted outcome (mirrors recover_user_input's closure branch).
            session = await self.session_store.update_status(session.id, SessionStatus.INTERRUPTED)
            async for event in self._emit_terminal_event_with_hooks(
                event=Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                        "tool_round_id": pending_round.round_id,
                        "tool_call_id": pending_tool_call.tool_call_id,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "resolved_by": resolution_actor_payload(request.resolved_by),
                    },
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            ):
                yield event
            return

        try:
            # Inside the guarded block for GeneratorExit coherence (aclose at a yield
            # finalizes via the handler below). A task CANCELLATION here is caught by
            # neither handler — as in every recovery entrance — and is finalized by
            # the sweep once the task is done; the claim registry is cleaned by the
            # caller's finally either way.
            transcript = await self.session_store.load_transcript(session.id)
            # _run_session recovers the pending round at entry: it reuses the terminal
            # event persisted above, closes never-started calls as not-executed errors,
            # appends the round's tool results, clears the checkpoint atomically, and
            # continues the model loop.
            session_stream = self._run_session(
                session=session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                messages=transcript,
                messages_to_append=[],
                max_steps=request.max_steps or _DEFAULT_APPROVAL_MAX_STEPS,
                limits=request.limits or RunLimits(),
                budget_limits=request.budget_limits or (),
                retry_policy=self._effective_retry_policy(request.retry_policy),
                structured_output=effective_structured_output,
                thinking=request.thinking,
                request_loop_policies=request.loop_policies,
                request_metadata=request.metadata,
                task_id=pending_round.task_id,
                task_worker_id=None,
                start_event_type=None,
                start_event_payload={},
                start_task_on_enter=False,
                release_run_fence_on_exit=False,
            )
            try:
                async for event in session_stream:
                    yield event
            except GeneratorExit:
                await session_stream.aclose()
                raise
        except GeneratorExit:
            # Abandonment while continuing the round: finalize to INTERRUPTED instead
            # of leaking a RUNNING session (mirrors recover_user_input's guard).
            await self._finalize_abandoned_session_by_id(session.id)
            raise

    async def _run_session(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        messages_to_append: list[Message],
        max_steps: int,
        limits: RunLimits,
        budget_limits: tuple[BudgetLimit, ...],
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
    ) -> AsyncGenerator[Event, None]:
        provider = registered_provider.provider
        # Per-run thinking override (RunRequest/ResumeRequest) wins over the agent's
        # default (AgentSpec.thinking); the agent default applies on every path,
        # including continuations that pass no override.
        effective_thinking = thinking if thinking is not None else registered_agent.spec.thinking
        environment_name = _environment_name(registered_environment)
        task_started = task_id is not None and not start_task_on_enter
        task_start_attempted = task_started
        task_finished = False
        current_task = asyncio.current_task()
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None
        run_started_at = time.monotonic()
        # A fresh run means any earlier interrupt was fully handled before the
        # session transitioned back to RUNNING; drop a stale signal so it does
        # not force per-delta store polling for the whole resumed run.
        self._session_control.discard_interrupt_signal(session.id)
        limits = copy_run_limits(limits)
        budget_limits = request_budget_limits_for_session(
            limits=budget_limits,
            agent_name=registered_agent.spec.name,
            causal_budget_id=session.causal_budget_id,
        )
        retry_policy = copy_retry_policy(retry_policy)
        structured_output = copy_structured_output_spec(structured_output)
        request_loop_policies = validate_loop_policies(
            request_loop_policies,
            field_name="request_loop_policies",
        )
        structured_output_retries = 0
        run_baseline: SessionUsageSummary | None = None
        turn_usage_tracker = self._run_limit_controller.usage_tracker(session.id)
        await turn_usage_tracker.mark_current_position()
        if (limits.scope == "run" and has_run_limits(limits)) or _has_run_budget_limit(
            budget_limits
        ):
            baseline_events = await self._run_limit_controller.session_usage_events(session.id)
        else:
            baseline_events = []
        if limits.scope == "run" and has_run_limits(limits):
            run_baseline = session_usage_summary(session.id, baseline_events)
        request_budget_notify_events: list[Event] = []
        if structured_output is not None and STRUCTURED_OUTPUT_TOOL_NAME in registered_agent.tools:
            raise ValueError(
                f"Tool name is reserved for structured output: {STRUCTURED_OUTPUT_TOOL_NAME}"
            )
        if current_task is not None:
            active_run = self._session_control.register_active_task(
                session.id,
                current_task,
                task_id=task_id,
                task_started=task_started,
                task_finished=task_finished,
                turn_registered_agent=registered_agent,
                turn_environment_name=environment_name,
                turn_started_at=run_started_at,
                turn_usage_tracker=turn_usage_tracker,
            )

        async def start_linked_task_if_needed(*, only_if_exists: bool = False) -> Event | None:
            nonlocal task_start_attempted, task_started
            if task_id is None or task_started:
                return None
            if self.task_store is None:
                raise RuntimeError("task_store is required when RunRequest.task_id is set.")
            if only_if_exists and await self.task_store.load_task(task_id) is None:
                return None
            task_start_attempted = True
            task = await self._start_task(
                task_id=task_id,
                session=session,
                worker_id=task_worker_id,
            )
            task_started = True
            if active_run is not None:
                active_run.task_started = True
            return await self._event_writer.emit(
                _task_event(
                    event_type=EventType.TASK_STARTED,
                    task=task,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                )
            )

        try:
            factory_started_event = await self._environment_lifecycle.emit_factory_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            if factory_started_event is not None:
                yield factory_started_event
                task_start_event = await start_linked_task_if_needed()
                if task_start_event is not None:
                    yield task_start_event
            factory_resolution = await self._environment_lifecycle.resolve_factory(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=factory_started_event,
                operation=(
                    EnvironmentFactoryOperation.CREATE
                    if start_event_type is EventType.SESSION_STARTED
                    else EnvironmentFactoryOperation.RECONNECT
                ),
            )
            registered_environment = factory_resolution.registered_environment
            environment_name = _environment_name(registered_environment)
            if active_run is not None:
                active_run.turn_environment_name = environment_name
            for event in factory_resolution.events:
                yield event
            if factory_resolution.error is not None:
                raise factory_resolution.error
            binding_started_event = await self._environment_lifecycle.emit_binding_started(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            if binding_started_event is not None:
                yield binding_started_event
                task_start_event = await start_linked_task_if_needed()
                if task_start_event is not None:
                    yield task_start_event
            binding_result = await self._environment_lifecycle.bind(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                started_event=binding_started_event,
            )
            registered_environment = binding_result.registered_environment
            environment_name = _environment_name(registered_environment)
            if active_run is not None:
                active_run.turn_environment_name = environment_name
            for event in binding_result.events:
                yield event
            if binding_result.error is not None:
                raise binding_result.error
            async for event in self._tool_round_executor.emit_mcp_manifest_checks(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
            ):
                yield event
            if start_event_type is not None:
                yield await self._event_writer.emit(
                    Event(
                        type=start_event_type,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload={
                            **start_event_payload,
                            **_session_trace_event_fields(session, request_metadata),
                        },
                    )
                )
            async for event in self._recover_pending_tool_round(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=messages,
                tail_message_count=len(messages_to_append),
            ):
                yield event
            # Typed backstop for a missed entrance: every entry point already
            # preflights this before touching persisted state. Here the session
            # is running, so a raise fails it cleanly instead of preventing it.
            _require_native_structured_output_support(
                structured_output, registered_provider=registered_provider
            )
            task_start_event = await start_linked_task_if_needed()
            if task_start_event is not None:
                yield task_start_event
            await self.session_store.append_transcript_messages(
                session.id,
                messages_to_append,
            )
            limit_gate = RunLimitGate(
                self._run_limit_controller,
                session=session,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                limits=limits,
                budget_limits=budget_limits,
                run_started_at=run_started_at,
                run_baseline=run_baseline,
                budget_baseline_events=baseline_events,
                budget_notify_events=request_budget_notify_events,
                pricing_provider_name=(
                    registered_provider.provider.billing_provider_name or registered_provider.name
                ),
            )
            tool_round_runner = self._tool_round_executor.create_run(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                limit_gate=limit_gate,
                request_metadata=request_metadata,
                task_id=task_id,
                structured_output=structured_output,
                thinking=effective_thinking,
                max_steps=max_steps,
                limits=limits,
                budget_limits=budget_limits,
                retry_policy=retry_policy,
                run_started_at=run_started_at,
                turn_usage_tracker=turn_usage_tracker,
                active_run=active_run,
            )
            model_step_run = self._model_step_executor.create_run(
                provider=provider,
                session=session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                environment_name=environment_name,
                structured_output=structured_output,
                thinking=effective_thinking,
                knowledge_store=_knowledge_store(registered_environment),
                request_metadata=request_metadata,
                retry_policy=retry_policy,
                request_budget_limits=budget_limits,
                limit_gate=limit_gate,
                budget_policy=self.budget_policy,
                run_started_at=run_started_at,
                turn_usage_tracker=turn_usage_tracker,
                active_run=active_run,
            )
            for step in range(1, max_steps + 1):
                await self._session_control.raise_if_interrupted(session.id)
                for event in await self._deliver_queued_session_messages(
                    session_id=session.id,
                    messages=messages,
                    include_on_idle=False,
                ):
                    yield event
                budget_evaluation = await limit_gate.evaluate_budget(self.budget_policy)
                async for event in self._apply_budget_evaluation(
                    evaluation=budget_evaluation,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    messages=messages,
                    run_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                ):
                    yield event
                if budget_evaluation.check is not None:
                    return
                limit_evaluation = await limit_gate.evaluate_limits()
                async for event in self._apply_limit_evaluation(
                    evaluation=limit_evaluation,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    messages=messages,
                    run_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                ):
                    yield event
                if limit_evaluation.decision is not None:
                    return
                model_step_flow_outcome: ModelStepFlowOutcome | None = None
                model_step_events = model_step_run.execute(
                    step=step,
                    messages=messages,
                )
                try:
                    async for event, flow_outcome in model_step_events:
                        if event is not None:
                            yield event
                        if flow_outcome is not None:
                            if model_step_flow_outcome is not None:
                                raise RuntimeError(
                                    "Model step produced more than one terminal flow outcome."
                                )
                            model_step_flow_outcome = flow_outcome
                finally:
                    await _close_async_iterator(model_step_events)
                if model_step_flow_outcome is None:
                    raise RuntimeError("Model step finished without a terminal flow outcome.")
                if model_step_flow_outcome.stop_session:
                    return
                assistant_step_result = model_step_flow_outcome.assistant_step_result
                if assistant_step_result is None:
                    raise RuntimeError("Successful model step finished without a result.")
                assistant_message = assistant_step_result.assistant_message
                tool_calls = assistant_step_result.tool_calls

                pending_tool_round: tool_round_recovery.PendingToolRound | None = None
                if assistant_message is not None:
                    messages.append(assistant_message)
                    if tool_calls and not (
                        structured_output is not None
                        and structured_output.strategy == StructuredOutputStrategy.TOOL
                        and _has_structured_output_tool_call(tool_calls)
                    ):
                        (
                            checkpoint,
                            pending_tool_round,
                        ) = await self._tool_round_executor.checkpoint_with_pending_tool_round(
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            tool_calls=tool_calls,
                            policy_outcomes=None,
                            task_id=task_id,
                            structured_output=structured_output,
                        )
                        await (
                            self.session_store.append_transcript_messages_and_transform_checkpoint(
                                session.id,
                                [assistant_message],
                                _replace_checkpoint_preserving_runtime_state(checkpoint),
                            )
                        )
                    else:
                        await self.session_store.append_transcript_messages(
                            session.id,
                            [assistant_message],
                        )
                tool_round_id = (
                    pending_tool_round.round_id if pending_tool_round is not None else None
                )

                limit_evaluation = await limit_gate.evaluate_limits(
                    pending_tool_calls=_user_tool_call_count(tool_calls),
                )
                async for event in self._apply_limit_evaluation(
                    evaluation=limit_evaluation,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_round_id=tool_round_id,
                    run_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                ):
                    yield event
                if limit_evaluation.decision is not None:
                    return

                budget_evaluation = await limit_gate.evaluate_budget(self.budget_policy)
                async for event in self._apply_budget_evaluation(
                    evaluation=budget_evaluation,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_round_id=tool_round_id,
                    run_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                ):
                    yield event
                if budget_evaluation.check is not None:
                    return

                if (
                    structured_output is not None
                    and structured_output.strategy == StructuredOutputStrategy.TOOL
                    and _has_structured_output_tool_call(tool_calls)
                ):
                    yield await self._event_writer.emit(
                        _structured_output_validating_event(
                            session=session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            spec=structured_output,
                            step=step,
                            attempt=structured_output_retries + 1,
                        )
                    )
                    validation = _validate_structured_output_tool_round(
                        tool_calls=tool_calls,
                        spec=structured_output,
                    )
                    structured_tool_outcomes = _structured_output_tool_round_outcomes(
                        tool_calls=tool_calls,
                        spec=structured_output,
                        validation=validation,
                    )
                    structured_tool_outcomes = _redact_tool_call_outcomes(
                        structured_tool_outcomes,
                        self._secret_redactor,
                    )
                    tool_result_messages = transcript_helpers.tool_result_messages(
                        structured_tool_outcomes
                    )
                    messages.extend(tool_result_messages)
                    await self.session_store.append_transcript_messages(
                        session.id,
                        tool_result_messages,
                    )
                    if validation.valid:
                        yield await self._event_writer.emit(
                            _structured_output_event(
                                event_type=EventType.STRUCTURED_OUTPUT_VALIDATED,
                                session=session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                spec=structured_output,
                                validation=validation,
                                step=step,
                                attempt=structured_output_retries + 1,
                                redactor=self._secret_redactor,
                            )
                        )
                        try:
                            session = await self._complete_session_if_no_queued_messages(session.id)
                        except SessionQueuedMessagesPending:
                            (
                                should_continue,
                                queued_events,
                            ) = await self._handle_queued_messages_before_completion(
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                environment_name=environment_name,
                                messages=messages,
                                step=step,
                                max_steps=max_steps,
                                run_started_at=run_started_at,
                                turn_usage_tracker=turn_usage_tracker,
                                active_run=active_run,
                            )
                            for event in queued_events:
                                yield event
                            if not should_continue:
                                return
                            continue
                        break
                    yield await self._event_writer.emit(
                        _structured_output_event(
                            event_type=EventType.STRUCTURED_OUTPUT_FAILED,
                            session=session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            spec=structured_output,
                            validation=validation,
                            step=step,
                            attempt=structured_output_retries + 1,
                            redactor=self._secret_redactor,
                        )
                    )
                    if structured_output_retries >= structured_output.max_retries:
                        raise RuntimeError(
                            "Structured output validation failed after "
                            f"{structured_output_retries + 1} attempt(s)."
                        )
                    if step >= max_steps:
                        raise RuntimeError(
                            "Structured output validation failed after "
                            f"{structured_output_retries + 1} attempt(s): "
                            "maximum model steps reached before repair."
                        )
                    structured_output_retries += 1
                    yield await self._event_writer.emit(
                        _structured_output_event(
                            event_type=EventType.STRUCTURED_OUTPUT_RETRY,
                            session=session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            spec=structured_output,
                            validation=validation,
                            step=step,
                            attempt=structured_output_retries,
                            redactor=self._secret_redactor,
                        )
                    )
                    continue

                if not tool_calls:
                    if structured_output is not None:
                        yield await self._event_writer.emit(
                            _structured_output_validating_event(
                                session=session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                spec=structured_output,
                                step=step,
                                attempt=structured_output_retries + 1,
                            )
                        )
                        if structured_output.strategy == StructuredOutputStrategy.NATIVE:
                            if assistant_step_result is None:
                                raise RuntimeError(
                                    "Native structured output validation requires an "
                                    "assistant step result."
                                )
                            validation = validate_structured_output_text(
                                assistant_step_result.text_content,
                                structured_output,
                            )
                            if validation.valid:
                                yield await self._event_writer.emit(
                                    _structured_output_event(
                                        event_type=EventType.STRUCTURED_OUTPUT_VALIDATED,
                                        session=session,
                                        registered_agent=registered_agent,
                                        environment_name=environment_name,
                                        spec=structured_output,
                                        validation=validation,
                                        step=step,
                                        attempt=structured_output_retries + 1,
                                        redactor=self._secret_redactor,
                                    )
                                )
                                try:
                                    session = await self._complete_session_if_no_queued_messages(
                                        session.id
                                    )
                                except SessionQueuedMessagesPending:
                                    (
                                        should_continue,
                                        queued_events,
                                    ) = await self._handle_queued_messages_before_completion(
                                        session=session,
                                        registered_agent=registered_agent,
                                        registered_environment=registered_environment,
                                        environment_name=environment_name,
                                        messages=messages,
                                        step=step,
                                        max_steps=max_steps,
                                        run_started_at=run_started_at,
                                        turn_usage_tracker=turn_usage_tracker,
                                        active_run=active_run,
                                    )
                                    for event in queued_events:
                                        yield event
                                    if not should_continue:
                                        return
                                    continue
                                break
                        else:
                            validation = structured_output_tool_required_validation()
                        yield await self._event_writer.emit(
                            _structured_output_event(
                                event_type=EventType.STRUCTURED_OUTPUT_FAILED,
                                session=session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                spec=structured_output,
                                validation=validation,
                                step=step,
                                attempt=structured_output_retries + 1,
                                redactor=self._secret_redactor,
                            )
                        )
                        if structured_output_retries >= structured_output.max_retries:
                            raise RuntimeError(
                                "Structured output validation failed after "
                                f"{structured_output_retries + 1} attempt(s)."
                            )
                        if step >= max_steps:
                            raise RuntimeError(
                                "Structured output validation failed after "
                                f"{structured_output_retries + 1} attempt(s): "
                                "maximum model steps reached before repair."
                            )
                        structured_output_retries += 1
                        redacted_validation = _redact_structured_output_validation(
                            validation,
                            self._secret_redactor,
                        )
                        repair_message = Message.text(
                            "user",
                            structured_output_repair_prompt(
                                spec=structured_output,
                                validation=redacted_validation,
                            ),
                        )
                        messages.append(repair_message)
                        await self.session_store.append_transcript_messages(
                            session.id,
                            [repair_message],
                        )
                        yield await self._event_writer.emit(
                            _structured_output_event(
                                event_type=EventType.STRUCTURED_OUTPUT_RETRY,
                                session=session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                spec=structured_output,
                                validation=validation,
                                step=step,
                                attempt=structured_output_retries,
                                redactor=self._secret_redactor,
                            )
                        )
                        continue
                    if assistant_step_result is None:
                        raise RuntimeError("Before-stop policies require an assistant step result.")
                    before_stop_decision: BeforeStopDecision | None = None
                    async for event, policy_decision in self._run_before_stop_policies(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        step_result=assistant_step_result,
                        step=step,
                        max_steps=max_steps,
                        request_metadata=request_metadata,
                        request_loop_policies=request_loop_policies,
                    ):
                        yield event
                        if policy_decision is not None:
                            before_stop_decision = policy_decision
                    if before_stop_decision is not None:
                        if before_stop_decision.action == BeforeStopAction.CONTINUE:
                            if step >= max_steps:
                                raise RuntimeError(
                                    "Before-stop policy requested continue, but maximum "
                                    "model steps were reached."
                                )
                            if before_stop_decision.message is None:
                                raise RuntimeError(
                                    "Before-stop continue decision requires a message."
                                )
                            repair_message = before_stop_decision.message
                            messages.append(repair_message)
                            await self.session_store.append_transcript_messages(
                                session.id,
                                [repair_message],
                            )
                            continue
                        if before_stop_decision.action == BeforeStopAction.INTERRUPT:
                            session = await self.session_store.update_status(
                                session.id,
                                SessionStatus.INTERRUPTED,
                            )
                            yield await self._emit_turn_completed_once(
                                session=session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                status=SessionStatus.INTERRUPTED,
                                run_started_at=run_started_at,
                                usage_tracker=turn_usage_tracker,
                                active_run=active_run,
                            )
                            async for event in self._emit_terminal_event_with_hooks(
                                event=Event(
                                    type=EventType.SESSION_INTERRUPTED,
                                    session_id=session.id,
                                    agent_name=registered_agent.spec.name,
                                    environment_name=environment_name,
                                    payload={
                                        "interruption_type": (
                                            _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED
                                        ),
                                        "reason": before_stop_decision.reason,
                                        "policy_metadata": copy_json_value(
                                            before_stop_decision.metadata,
                                            "policy_metadata",
                                        ),
                                    },
                                ),
                                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                            ):
                                yield event
                            return
                        if before_stop_decision.action == BeforeStopAction.FAIL:
                            raise RuntimeError(
                                f"Before-stop policy failed session: {before_stop_decision.reason}"
                            )
                    try:
                        session = await self._complete_session_if_no_queued_messages(session.id)
                    except SessionQueuedMessagesPending:
                        (
                            should_continue,
                            queued_events,
                        ) = await self._handle_queued_messages_before_completion(
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            environment_name=environment_name,
                            messages=messages,
                            step=step,
                            max_steps=max_steps,
                            run_started_at=run_started_at,
                            turn_usage_tracker=turn_usage_tracker,
                            active_run=active_run,
                        )
                        for event in queued_events:
                            yield event
                        if not should_continue:
                            return
                        continue
                    break

                async for event in tool_round_runner.run(
                    messages=messages,
                    tool_calls=tool_calls,
                    tool_round_id=tool_round_id,
                ):
                    yield event
                if tool_round_runner.stopped_for_limit:
                    return
            else:
                raise RuntimeError(f"Maximum model steps exceeded: {max_steps}")

            if task_id is not None:
                await self._session_control.raise_if_interrupted(session.id)
                task = await self._complete_task(
                    task_id=task_id,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                )
                task_finished = True
                if active_run is not None:
                    active_run.task_finished = True
                yield await self._event_writer.emit(
                    _task_event(
                        event_type=EventType.TASK_COMPLETED,
                        task=task,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                    )
                )
            yield await self._emit_turn_completed_once(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=SessionStatus.COMPLETED,
                run_started_at=run_started_at,
                usage_tracker=turn_usage_tracker,
                active_run=active_run,
            )
            async for event in self._emit_terminal_event_with_hooks(
                event=Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_COMPLETED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            ):
                yield event
        except ToolApprovalRequired as exc:
            session = await self.session_store.update_status(session.id, SessionStatus.INTERRUPTED)
            yield await self._emit_turn_completed_once(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=SessionStatus.INTERRUPTED,
                run_started_at=run_started_at,
                usage_tracker=turn_usage_tracker,
                active_run=active_run,
            )
            async for event in self._emit_terminal_event_with_hooks(
                event=Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                        "approval": exc.approval.model_dump(mode="json"),
                    },
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            ):
                yield event
        except UserInputRequired as exc:
            session = await self.session_store.update_status(session.id, SessionStatus.INTERRUPTED)
            yield await self._emit_turn_completed_once(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=SessionStatus.INTERRUPTED,
                run_started_at=run_started_at,
                usage_tracker=turn_usage_tracker,
                active_run=active_run,
            )
            async for event in self._emit_terminal_event_with_hooks(
                event=Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                        "user_input": exc.pending.model_dump(mode="json"),
                    },
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            ):
                yield event
        except SessionInterruptedByRequest:
            async for event in self._handle_session_interrupted(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                run_started_at=run_started_at,
                turn_usage_tracker=turn_usage_tracker,
                active_run=active_run,
            ):
                yield event
            return
        except asyncio.CancelledError:
            if await self._session_control.interrupt_requested(session.id):
                async for event in self._handle_session_interrupted(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    run_started_at=run_started_at,
                    turn_usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                ):
                    yield event
                return
            raise
        except GeneratorExit:
            # The consumer closed the event stream (client disconnect / abandoned
            # async generator) while the session was still live. Finalize instead of
            # stranding it in RUNNING; an async generator must not yield while
            # handling GeneratorExit, so the terminal emission is drained, not
            # streamed.
            await self._finalize_abandoned_session_run(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                run_started_at=run_started_at,
                turn_usage_tracker=turn_usage_tracker,
                active_run=active_run,
            )
            raise
        except Exception as exc:
            task_failure_error: Exception | None = None
            if (
                not task_started
                and not task_start_attempted
                and task_id is not None
                and self.task_store is not None
            ):
                try:
                    task_start_event = await start_linked_task_if_needed(only_if_exists=True)
                    if task_start_event is not None:
                        yield task_start_event
                except Exception as task_exc:
                    task_failure_error = task_exc
            if (
                task_started
                and not task_finished
                and task_id is not None
                and self.task_store is not None
            ):
                try:
                    task = await self.task_store.fail_task(
                        task_id,
                        {
                            "message": str(exc),
                            "type": type(exc).__name__,
                            "session_id": session.id,
                        },
                    )
                    task_finished = True
                    if active_run is not None:
                        active_run.task_finished = True
                    yield await self._event_writer.emit(
                        _task_event(
                            event_type=EventType.TASK_FAILED,
                            task=task,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                        )
                    )
                except Exception as task_exc:
                    task_failure_error = task_exc
            session = await self.session_store.update_status(session.id, SessionStatus.FAILED)
            payload = exception_failure_payload(exc)
            if task_failure_error is not None:
                payload["task_update_error"] = str(task_failure_error)
                payload["task_update_error_type"] = type(task_failure_error).__name__
            yield await self._emit_turn_completed_once(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=SessionStatus.FAILED,
                run_started_at=run_started_at,
                usage_tracker=turn_usage_tracker,
                active_run=active_run,
            )
            async for event in self._emit_terminal_event_with_hooks(
                event=Event(
                    type=EventType.SESSION_FAILED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=payload,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            ):
                yield event
        finally:
            self._session_control.discard_interrupt_signal(session.id)
            try:
                if release_run_fence_on_exit:
                    await self.session_store.release_run_fence(session.id)
            finally:
                if current_task is not None:
                    self._session_control.unregister_active_task(session.id, current_task)

    async def _emit_turn_completed(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        environment_name: str | None,
        status: SessionStatus,
        run_started_at: float,
        usage_tracker: SessionUsageTracker,
    ) -> Event:
        usage_events = await usage_tracker.usage_events()
        summary = session_usage_summary(session.id, usage_events)
        duration_ms = max(0, int((time.monotonic() - run_started_at) * 1000))
        return await self._event_writer.emit(
            Event(
                type=EventType.TURN_COMPLETED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    "status": status.value,
                    "duration_ms": duration_ms,
                    "step_count": summary.model_steps,
                    "tool_call_count": summary.tool_calls,
                    "token_usage": summary.usage.model_dump(),
                    "provider_names": summary.provider_names,
                    "models": summary.models,
                },
            )
        )

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
        if active_run is None:
            return await self._emit_turn_completed(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=status,
                run_started_at=run_started_at,
                usage_tracker=usage_tracker,
            )
        async with active_run.turn_completed_lock:
            if active_run.turn_completed_event is not None:
                return active_run.turn_completed_event
            event = await self._emit_turn_completed(
                session=session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=status,
                run_started_at=run_started_at,
                usage_tracker=usage_tracker,
            )
            active_run.turn_completed_event = event
            return event

    async def _emit_active_turn_completed_if_needed(
        self,
        *,
        session: Session,
        status: SessionStatus,
    ) -> Event | None:
        for active_run in self._session_control.active_runs(session.id):
            if (
                active_run.turn_registered_agent is None
                or active_run.turn_started_at is None
                or active_run.turn_usage_tracker is None
            ):
                continue
            return await self._emit_turn_completed_once(
                session=session,
                registered_agent=active_run.turn_registered_agent,
                environment_name=active_run.turn_environment_name,
                status=status,
                run_started_at=active_run.turn_started_at,
                usage_tracker=active_run.turn_usage_tracker,
                active_run=active_run,
            )
        return None

    async def _start_task(
        self,
        *,
        task_id: str,
        session: Session,
        worker_id: str | None = None,
    ) -> Task:
        if self.task_store is None:
            raise RuntimeError("task_store is required when RunRequest.task_id is set.")
        existing = await self.task_store.load_task(task_id)
        if (
            worker_id is None
            and existing is not None
            and existing.status is TaskStatus.RUNNING
            and existing.session_id == session.id
            and existing.worker_id is None
        ):
            return existing
        if worker_id is not None:
            return await self.task_store.attach_task(
                task_id,
                session_id=session.id,
                worker_id=worker_id,
            )
        return await self.task_store.start_task(task_id, session_id=session.id)

    async def _fail_task_for_run_setup_error(
        self,
        *,
        task_id: str | None,
        task_worker_id: str | None,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        error: Exception,
    ) -> tuple[Event | None, Exception | None]:
        """Fail only a task this run can prove it owns before `_run_session`."""
        if task_id is None:
            return None, None
        if self.task_store is None:
            return None, RuntimeError("task_store is required when RunRequest.task_id is set.")
        try:
            task = await self.task_store.load_task(task_id)
            if task is None:
                return None, KeyError(f"Task not found: {task_id}")
            worker_matches = task.worker_id == task_worker_id
            attached_to_session = task.session_id == session.id
            owned_claim = (
                task_worker_id is not None
                and task.status is TaskStatus.CLAIMED
                and task.session_id is None
                and worker_matches
            )
            unclaimed_pending = (
                task_worker_id is None
                and task.status is TaskStatus.PENDING
                and task.session_id is None
                and task.worker_id is None
            )
            if not ((attached_to_session and worker_matches) or owned_claim or unclaimed_pending):
                return None, None
            task = await self.task_store.fail_task(
                task_id,
                {
                    "message": str(error),
                    "type": type(error).__name__,
                    "session_id": session.id,
                },
                worker_id=task_worker_id,
            )
            return (
                await self._event_writer.emit(
                    _task_event(
                        event_type=EventType.TASK_FAILED,
                        task=task,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                    )
                ),
                None,
            )
        except Exception as task_error:
            return None, task_error

    async def _linked_running_task_id(self, session_id: str) -> str | None:
        """Find the one running task already attached to a resumed session.

        Task attachment is durable in ``TaskStore`` while ``ResumeRequest`` carries
        no task id. Re-associate that task before entering the resumed loop so a
        post-crash completion or failure terminalizes the original work item.
        """
        if self.task_store is None:
            return None
        tasks = await self.task_store.list_tasks(
            TaskQuery(
                status=TaskStatus.RUNNING,
                session_id=session_id,
                limit=2,
            )
        )
        if len(tasks) > 1:
            raise RuntimeError(f"Session has multiple running tasks attached: {session_id}")
        return tasks[0].id if tasks else None

    async def _complete_task(
        self,
        *,
        task_id: str,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> Task:
        if self.task_store is None:
            raise RuntimeError("task_store is required when RunRequest.task_id is set.")
        return await self.task_store.complete_task(
            task_id,
            {
                "session_id": session.id,
                "agent_name": registered_agent.spec.name,
                "environment_name": _environment_name(registered_environment),
            },
        )

    async def _apply_model_step_budget_evaluation(
        self,
        request: ModelStepBudgetEvaluationRequest,
    ) -> AsyncIterator[Event]:
        events = self._apply_budget_evaluation(
            evaluation=request.evaluation,
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            messages=request.messages,
            run_started_at=request.run_started_at,
            turn_usage_tracker=request.turn_usage_tracker,
            active_run=request.active_run,
        )
        try:
            async for event in events:
                yield event
        finally:
            await _close_async_iterator(events)

    async def _apply_model_step_limit_evaluation(
        self,
        request: ModelStepLimitEvaluationRequest,
    ) -> AsyncIterator[Event]:
        events = self._apply_limit_evaluation(
            evaluation=request.evaluation,
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            messages=request.messages,
            run_started_at=request.run_started_at,
            turn_usage_tracker=request.turn_usage_tracker,
            active_run=request.active_run,
        )
        try:
            async for event in events:
                yield event
        finally:
            await _close_async_iterator(events)

    async def _stop_for_model_step_budget_reservation_failure(
        self,
        request: ModelStepBudgetReservationFailureRequest,
    ) -> AsyncIterator[Event]:
        events = self._stop_session_for_budget_reservation_failed(
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            result=request.result,
            messages=request.messages,
            run_started_at=request.run_started_at,
            turn_usage_tracker=request.turn_usage_tracker,
            active_run=request.active_run,
        )
        try:
            async for event in events:
                yield event
        finally:
            await _close_async_iterator(events)

    async def _apply_tool_round_limit(
        self,
        request: ToolRoundLimitRequest,
    ) -> AsyncIterator[Event]:
        async for event in self._apply_limit_evaluation(
            evaluation=request.evaluation,
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
            messages=request.messages,
            tool_calls=request.tool_calls,
            completed_tool_outcomes=request.completed_tool_outcomes,
            tool_round_id=request.tool_round_id,
            run_started_at=request.run_started_at,
            turn_usage_tracker=request.turn_usage_tracker,
            active_run=request.active_run,
        ):
            yield event

    async def _apply_limit_evaluation(
        self,
        *,
        evaluation: LimitEvaluation,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest] | None = None,
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome] | None = None,
        tool_round_id: str | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
    ) -> AsyncIterator[Event]:
        for event in evaluation.events:
            yield event
        if evaluation.decision is None:
            return
        async for event in self._stop_session_for_limit_reached(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            decision=evaluation.decision,
            usage_summary=evaluation.usage_summary,
            cost_summary=evaluation.cost_summary,
            messages=messages,
            tool_calls=tool_calls if tool_calls is not None else [],
            completed_tool_outcomes=(
                completed_tool_outcomes if completed_tool_outcomes is not None else []
            ),
            tool_round_id=tool_round_id,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
        ):
            yield event

    async def _apply_budget_evaluation(
        self,
        *,
        evaluation: BudgetEvaluation,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest] | None = None,
        tool_round_id: str | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
    ) -> AsyncIterator[Event]:
        for event in evaluation.events:
            yield event
        if evaluation.check is None:
            return
        async for event in self._stop_session_for_budget_limit_reached(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            check=evaluation.check,
            messages=messages,
            tool_calls=tool_calls if tool_calls is not None else [],
            completed_tool_outcomes=[],
            tool_round_id=tool_round_id,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
        ):
            yield event

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
        tool_round_id: str | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
    ) -> AsyncIterator[Event]:
        limit_payload = _limit_reached_payload(
            decision=decision,
            usage_summary=usage_summary,
            cost_summary=cost_summary,
        )
        yield await self._event_writer.emit(
            Event(
                type=EventType.SESSION_LIMIT_REACHED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload=limit_payload,
            )
        )
        if tool_calls or completed_tool_outcomes or pending_approval_to_clear is not None:
            async for event in self._close_limited_tool_round(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=messages,
                tool_calls=tool_calls,
                completed_tool_outcomes=completed_tool_outcomes,
                decision=decision,
                pending_approval_to_clear=pending_approval_to_clear,
                tool_round_id=tool_round_id,
            ):
                yield event

        interrupted_session = await self.session_store.update_status(
            session.id,
            SessionStatus.INTERRUPTED,
        )
        terminal_payload = {
            "interruption_type": _INTERRUPTION_TYPE_LIMIT_REACHED,
            **limit_payload,
        }
        if run_started_at is not None and turn_usage_tracker is not None:
            yield await self._emit_turn_completed_once(
                session=interrupted_session,
                registered_agent=registered_agent,
                environment_name=environment_name,
                status=SessionStatus.INTERRUPTED,
                run_started_at=run_started_at,
                usage_tracker=turn_usage_tracker,
                active_run=active_run,
            )
        async for event in self._emit_terminal_event_with_hooks(
            event=Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id=interrupted_session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload=terminal_payload,
            ),
            phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
            session=interrupted_session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
        ):
            yield event

    async def _stop_session_for_queued_input_step_limit(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        messages: list[Message],
        step: int,
        max_steps: int,
        run_started_at: float,
        turn_usage_tracker: SessionUsageTracker,
        active_run: ActiveSessionRun[SessionUsageTracker] | None,
    ) -> AsyncIterator[Event]:
        usage_summary = session_usage_summary(
            session.id,
            await self._run_limit_controller.session_usage_events(session.id),
        )
        decision = StopDecision(
            limit=StopLimit.MODEL_STEPS,
            maximum=max_steps,
            actual=step,
            message=(
                "Run limit reached: durable queued input arrived after the final "
                f"model step ({step} >= {max_steps})."
            ),
        )
        async for event in self._stop_session_for_limit_reached(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            decision=decision,
            usage_summary=usage_summary,
            cost_summary=None,
            messages=messages,
            tool_calls=[],
            completed_tool_outcomes=[],
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
        ):
            yield event

    async def _stop_session_for_budget_reservation_failed(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        result: BudgetReservationResult,
        messages: list[Message],
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
    ) -> AsyncIterator[Event]:
        payload = budget_reservation_payload(result)
        yield await self._event_writer.emit(
            Event(
                type=EventType.BUDGET_LIMIT_REACHED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload=payload,
            )
        )
        session_events = await self._run_limit_controller.session_usage_events(session.id)
        usage_summary = session_usage_summary(session.id, session_events)
        decision = StopDecision(
            limit=StopLimit.ESTIMATED_COST,
            maximum=result.maximum,
            actual=result.actual,
            message=result.message,
        )
        async for event in self._stop_session_for_limit_reached(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            decision=decision,
            usage_summary=usage_summary,
            cost_summary=None,
            messages=messages,
            tool_calls=[],
            completed_tool_outcomes=[],
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
        ):
            yield event

    async def _stop_session_for_budget_limit_reached(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        check: BudgetCheck,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome],
        tool_round_id: str | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
    ) -> AsyncIterator[Event]:
        payload = budget_limit_reached_payload(check)
        yield await self._event_writer.emit(
            Event(
                type=EventType.BUDGET_LIMIT_REACHED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload=payload,
            )
        )
        decision = StopDecision(
            limit=StopLimit.ESTIMATED_COST,
            maximum=check.maximum,
            actual=check.actual,
            message=check.message,
        )
        session_events = await self._run_limit_controller.session_usage_events(session.id)
        usage_summary = session_usage_summary(session.id, session_events)
        async for event in self._stop_session_for_limit_reached(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            decision=decision,
            usage_summary=usage_summary,
            cost_summary=check.cost_summary,
            messages=messages,
            tool_calls=tool_calls,
            completed_tool_outcomes=completed_tool_outcomes,
            tool_round_id=tool_round_id,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
        ):
            yield event

    async def _close_limited_tool_round(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        completed_tool_outcomes: list[runtime_records.ToolCallOutcome],
        decision: StopDecision,
        pending_approval_to_clear: PendingToolApproval | None = None,
        tool_round_id: str | None = None,
    ) -> AsyncIterator[Event]:
        expected_tool_calls = [*tool_calls, *(outcome.call for outcome in completed_tool_outcomes)]
        if await self._tool_round_has_result_messages(session.id, expected_tool_calls):
            if pending_approval_to_clear is not None:
                cleared_checkpoint = await approval_support.checkpoint_without_pending_approval(
                    self.session_store,
                    session.id,
                )
                await self.session_store.transform_checkpoint(
                    session.id,
                    _replace_checkpoint_preserving_runtime_state(cleared_checkpoint),
                )
                yield await self._event_writer.emit(
                    approval_support.cleared_event(
                        session=session,
                        agent_name=registered_agent.spec.name,
                        environment_name=_environment_name(registered_environment),
                        approval_id=pending_approval_to_clear.approval_id,
                    )
                )
            return
        completed_ids = {outcome.call.id for outcome in completed_tool_outcomes}
        remaining_tool_calls = [
            tool_call for tool_call in tool_calls if tool_call.id not in completed_ids
        ]
        skipped_outcomes = _limit_reached_tool_round_results(
            tool_calls=remaining_tool_calls,
            decision=decision,
            tool_round_id=tool_round_id,
        )
        completed_tool_outcomes = _redact_tool_call_outcomes(
            completed_tool_outcomes,
            self._secret_redactor,
        )
        skipped_outcomes = _redact_tool_call_outcomes(
            skipped_outcomes,
            self._secret_redactor,
        )
        for skipped_outcome in skipped_outcomes:
            yield await self._event_writer.emit(
                _limit_reached_tool_call_event(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call_outcome=skipped_outcome,
                    decision=decision,
                    tool_round_id=tool_round_id,
                )
            )
        tool_result_messages = ordered_tool_result_messages(
            tool_calls,
            [*completed_tool_outcomes, *skipped_outcomes],
            parallel=True,
        )
        messages.extend(tool_result_messages)
        if pending_approval_to_clear is not None:
            cleared_checkpoint = await approval_support.checkpoint_without_pending_approval(
                self.session_store,
                session.id,
            )
            await self.session_store.append_transcript_messages_and_transform_checkpoint(
                session.id,
                tool_result_messages,
                _replace_checkpoint_preserving_runtime_state(cleared_checkpoint),
            )
            yield await self._event_writer.emit(
                approval_support.cleared_event(
                    session=session,
                    agent_name=registered_agent.spec.name,
                    environment_name=_environment_name(registered_environment),
                    approval_id=pending_approval_to_clear.approval_id,
                )
            )
        else:
            cleared_checkpoint = (
                await self._tool_round_executor.checkpoint_without_pending_tool_round(session.id)
            )
            await self.session_store.append_transcript_messages_and_transform_checkpoint(
                session.id,
                tool_result_messages,
                _replace_checkpoint_preserving_runtime_state(cleared_checkpoint),
            )

    async def _clear_pending_tool_round_if_matches(
        self,
        session_id: str,
        pending_round: tool_round_recovery.PendingToolRound,
    ) -> None:
        checkpoint = await self.session_store.load_checkpoint(session_id)
        if checkpoint is None:
            return
        copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
        current = tool_round_recovery.pending_tool_round_from_checkpoint(copied_checkpoint)
        if current is None or current.round_id != pending_round.round_id:
            return
        copied_checkpoint.pop(tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY, None)
        await self.session_store.transform_checkpoint(
            session_id,
            _replace_checkpoint_preserving_runtime_state(copied_checkpoint),
        )

    async def _load_pending_session_interrupt_payload(
        self,
        session_id: str,
        *,
        default: dict[str, Any],
    ) -> dict[str, Any]:
        checkpoint = await self.session_store.load_checkpoint(session_id)
        if checkpoint is None:
            return copy_json_value(default, "interrupt_payload")
        copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
        value = copied_checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
        if value is None:
            return copy_json_value(default, "interrupt_payload")
        if type(value) is not dict:
            raise ValueError("Pending session interrupt checkpoint must be an object.")
        return copy_json_value(value, "interrupt_payload")

    async def _clear_pending_session_interrupt(self, session_id: str) -> None:
        def transform(_session: Session, checkpoint: dict[str, Any] | None) -> dict[str, Any]:
            copied = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
            copied.pop(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY, None)
            return copied

        await self.session_store.transform_checkpoint(session_id, transform)

    async def _load_pending_interruption_cascade(
        self,
        session_id: str,
    ) -> dict[str, Any] | None:
        checkpoint = await self.session_store.load_checkpoint(session_id)
        if checkpoint is None:
            return None
        marker = checkpoint.get(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY)
        if marker is None:
            return None
        if type(marker) is not dict:
            raise ValueError("Pending interruption cascade checkpoint must be an object.")
        copied_marker = copy_json_value(marker, "pending_interruption_cascade")
        attempt_id = copied_marker.get("attempt_id")
        interrupt_payload = copied_marker.get("interrupt_payload")
        if type(attempt_id) is not str or not attempt_id.strip():
            raise ValueError("Pending interruption cascade attempt_id must be a non-blank string.")
        if type(interrupt_payload) is not dict:
            raise ValueError("Pending interruption cascade payload must be an object.")
        failure_recorded = copied_marker.get("failure_recorded", False)
        if type(failure_recorded) is not bool:
            raise ValueError("Pending interruption cascade failure_recorded must be a boolean.")
        generation = copied_marker.get("generation", 0)
        if type(generation) is not int or generation < 0:
            raise ValueError("Pending interruption cascade generation must be non-negative.")
        claim_id = copied_marker.get("claim_id")
        claim_expires_at = _interruption_cascade_marker_datetime(
            copied_marker,
            "claim_expires_at",
        )
        if claim_id is not None:
            if type(claim_id) is not str or not claim_id.strip() or claim_expires_at is None:
                raise ValueError("Pending interruption cascade claim is invalid.")
        elif claim_expires_at is not None:
            raise ValueError("Pending interruption cascade claim is invalid.")
        retry_request = _copy_interruption_cascade_retry_request(copied_marker.get("retry_request"))
        if retry_request is not None:
            copied_marker["retry_request"] = retry_request
        _interruption_cascade_marker_datetime(copied_marker, "created_at")
        return copied_marker

    async def _claim_pending_interruption_cascade(
        self,
        session_id: str,
        interrupt_payload: dict[str, Any],
        *,
        create_if_missing: bool = True,
        retry_request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        retry_request = _copy_interruption_cascade_retry_request(retry_request)
        resolved_marker: dict[str, Any] | None = None

        def transform(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            nonlocal resolved_marker
            copied_checkpoint = (
                {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
            )
            existing = copied_checkpoint.get(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY)
            if existing is None:
                if not create_if_missing:
                    return None
                marker = {
                    "attempt_id": str(uuid4()),
                    "interrupt_payload": copy_json_value(interrupt_payload, "interrupt_payload"),
                    "created_at": self._clock().isoformat(),
                }
            elif type(existing) is not dict:
                raise ValueError("Pending interruption cascade checkpoint must be an object.")
            else:
                marker = copy_json_value(existing, "pending_interruption_cascade")
            attempt_id = marker.get("attempt_id")
            if type(attempt_id) is not str or not attempt_id.strip():
                raise ValueError(
                    "Pending interruption cascade attempt_id must be a non-blank string."
                )
            if type(marker.get("interrupt_payload")) is not dict:
                raise ValueError("Pending interruption cascade payload must be an object.")
            existing_retry_request = _copy_interruption_cascade_retry_request(
                marker.get("retry_request")
            )
            if retry_request is not None:
                marker["retry_request"] = copy_json_value(retry_request, "retry_request")
            elif existing_retry_request is not None:
                marker["retry_request"] = existing_retry_request
            failure_recorded = marker.get("failure_recorded", False)
            if type(failure_recorded) is not bool:
                raise ValueError("Pending interruption cascade failure_recorded must be a boolean.")
            generation = marker.get("generation", 0)
            if type(generation) is not int or generation < 0:
                raise ValueError("Pending interruption cascade generation must be non-negative.")
            now = self._clock()
            claim_id = marker.get("claim_id")
            claim_expires_at = _interruption_cascade_marker_datetime(
                marker,
                "claim_expires_at",
            )
            if claim_id is not None:
                if type(claim_id) is not str or not claim_id.strip() or claim_expires_at is None:
                    raise ValueError("Pending interruption cascade claim is invalid.")
                if claim_expires_at > now:
                    return None
            elif claim_expires_at is not None:
                raise ValueError("Pending interruption cascade claim is invalid.")
            marker.setdefault("created_at", now.isoformat())
            marker["generation"] = generation + 1
            marker["claim_id"] = str(uuid4())
            marker["claim_expires_at"] = (
                now + timedelta(seconds=interruption_cascade_lease_seconds())
            ).isoformat()
            copied_checkpoint[_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY] = marker
            resolved_marker = copy_json_value(marker, "pending_interruption_cascade")
            return copied_checkpoint

        await self.session_store.transform_checkpoint(session_id, transform)
        return resolved_marker

    async def _mark_pending_interruption_cascade_failed(
        self,
        session_id: str,
        attempt_id: str,
        generation: int,
        claim_id: str,
    ) -> bool:
        recorded = False

        def transform(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            nonlocal recorded
            if checkpoint is None:
                return None
            copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
            marker = copied_checkpoint.get(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY)
            if (
                type(marker) is not dict
                or marker.get("attempt_id") != attempt_id
                or marker.get("generation") != generation
                or marker.get("claim_id") != claim_id
            ):
                return None
            marker["failure_recorded"] = True
            marker.pop("claim_id", None)
            marker.pop("claim_expires_at", None)
            recorded = True
            return copied_checkpoint

        await self.session_store.transform_checkpoint(session_id, transform)
        return recorded

    async def _complete_pending_interruption_cascade(
        self,
        session_id: str,
        attempt_id: str,
        generation: int,
        claim_id: str,
    ) -> tuple[bool, bool]:
        cleared = False
        failure_recorded = False

        def transform(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            nonlocal cleared, failure_recorded
            if checkpoint is None:
                return None
            copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
            marker = copied_checkpoint.get(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY)
            if (
                type(marker) is not dict
                or marker.get("attempt_id") != attempt_id
                or marker.get("generation") != generation
                or marker.get("claim_id") != claim_id
            ):
                return None
            current_failure_recorded = marker.get("failure_recorded", False)
            if type(current_failure_recorded) is not bool:
                raise ValueError("Pending interruption cascade failure_recorded must be a boolean.")
            failure_recorded = current_failure_recorded
            copied_checkpoint.pop(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY)
            cleared = True
            return copied_checkpoint

        await self.session_store.transform_checkpoint(session_id, transform)
        return cleared, failure_recorded

    async def _renew_pending_interruption_cascade_claim(
        self,
        session_id: str,
        attempt_id: str,
        generation: int,
        claim_id: str,
    ) -> bool:
        renewed = False

        def transform(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            nonlocal renewed
            if checkpoint is None:
                return None
            copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
            marker = copied_checkpoint.get(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY)
            if (
                type(marker) is not dict
                or marker.get("attempt_id") != attempt_id
                or marker.get("generation") != generation
                or marker.get("claim_id") != claim_id
            ):
                return None
            marker["claim_expires_at"] = (
                self._clock() + timedelta(seconds=interruption_cascade_lease_seconds())
            ).isoformat()
            renewed = True
            return copied_checkpoint

        await self.session_store.transform_checkpoint(session_id, transform)
        return renewed

    async def _release_pending_interruption_cascade_claim(
        self,
        session_id: str,
        attempt_id: str,
        generation: int,
        claim_id: str,
    ) -> None:
        def transform(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            if checkpoint is None:
                return None
            copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
            marker = copied_checkpoint.get(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY)
            if (
                type(marker) is not dict
                or marker.get("attempt_id") != attempt_id
                or marker.get("generation") != generation
                or marker.get("claim_id") != claim_id
            ):
                return None
            marker.pop("claim_id", None)
            marker.pop("claim_expires_at", None)
            return copied_checkpoint

        await self.session_store.transform_checkpoint(session_id, transform)

    async def _clear_pending_interruption_cascade(self, session_id: str) -> str | None:
        cleared_attempt_id: str | None = None

        def transform(_session: Session, checkpoint: dict[str, Any] | None) -> dict[str, Any]:
            nonlocal cleared_attempt_id
            if checkpoint is None:
                return {}
            copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
            marker = copied_checkpoint.pop(_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY, None)
            if marker is None:
                return copied_checkpoint
            if type(marker) is not dict or type(marker.get("attempt_id")) is not str:
                raise ValueError("Pending interruption cascade checkpoint is invalid.")
            cleared_attempt_id = marker["attempt_id"]
            return copied_checkpoint

        await self.session_store.transform_checkpoint(session_id, transform)
        return cleared_attempt_id

    async def _require_session(self, session_id: str) -> Session:
        loaded = await self.session_store.load(session_id)
        if loaded is None:
            raise KeyError(f"Session not found: {session_id}") from None
        return loaded

    async def _finalize_interrupting_for_recovery(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        events: list[Event],
    ) -> Session:
        """Finalize an INTERRUPTING session during recovery: drain its terminal events into
        ``events`` and return the reloaded session (a no-op once past INTERRUPTING)."""
        if session.status == SessionStatus.INTERRUPTING:
            async for event in self._handle_session_interrupted(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
            ):
                events.append(event)
            session = await self._require_session(session.id)
        return session

    async def _recover_incomplete_session_scoped(
        self,
        *,
        session: Session,
        inactive_before: datetime | None,
        reason: str,
        metadata: dict[str, Any],
    ) -> IncompleteSessionRecoveryResult:
        owned_epoch_before = _current_session_run_epoch(session.id)
        try:
            return await self._recover_incomplete_session(
                session=session,
                inactive_before=inactive_before,
                reason=reason,
                metadata=metadata,
            )
        finally:
            owned_epoch_after = _current_session_run_epoch(session.id)
            if (
                inactive_before is not None
                and owned_epoch_after is not None
                and owned_epoch_after != owned_epoch_before
            ):
                await self.session_store.release_run_fence(session.id)

    async def _recover_incomplete_session(
        self,
        *,
        session: Session,
        inactive_before: datetime | None = None,
        reason: str,
        metadata: dict[str, Any],
    ) -> IncompleteSessionRecoveryResult:
        reason = require_clean_nonblank(reason, "reason")
        metadata = copy_json_value(metadata, "metadata")
        previous_status = session.status
        actions: list[IncompleteSessionRecoveryAction] = []
        events: list[Event] = []

        if self._session_control.has_active_tasks(session.id):
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                events=(),
                message="Session has active work in this CayuApp process; recovery skipped.",
            )

        checkpoint = await self.session_store.load_checkpoint(session.id)
        pending_approval = approval_support.pending_approval_from_checkpoint(checkpoint)
        pending_user_input = pending_user_input_from_checkpoint(checkpoint)
        pending_tool_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
        if (
            session.status in _RESUMABLE_SESSION_STATUSES
            and pending_approval is None
            and pending_user_input is None
            and pending_tool_round is None
        ):
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,),
                events=(),
                message="Session is terminal; recovery skipped.",
            )

        try:
            registered_agent = self._get_registered_agent(session.agent_name)
        except KeyError:
            # Expected state, not an error — leave the session untouched and
            # report a typed skip instead of aborting recovery.
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=(IncompleteSessionRecoveryAction.SKIPPED_UNREGISTERED_AGENT,),
                events=(),
                message=(f"Agent not registered: {session.agent_name!r}; session left untouched."),
            )
        registered_environment = self._get_registered_environment_for_session(
            session.environment_name
        )
        environment_name = _environment_name(registered_environment)

        if inactive_before is not None:
            fenced = await self.session_store.fence_stalled_run(
                session.id,
                statuses={session.status},
                inactive_before=inactive_before,
            )
            if fenced is None:
                current = await self._require_session(session.id)
                return IncompleteSessionRecoveryResult(
                    session_id=session.id,
                    previous_status=previous_status,
                    status=current.status,
                    actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                    events=(),
                    message="Session activity changed during recovery; recovery skipped.",
                )
            events.append(
                await self._event_writer.emit(
                    Event(
                        type=EventType.SESSION_RUN_FENCED,
                        session_id=session.id,
                        agent_name=session.agent_name,
                        environment_name=environment_name,
                        payload={
                            "previous_run_epoch": session.run_epoch,
                            "run_epoch": fenced.run_epoch,
                            "inactive_before": inactive_before.isoformat(),
                            "reason": reason,
                            "metadata": metadata,
                        },
                    )
                )
            )
            session = fenced

        if session.status in {SessionStatus.PENDING, SessionStatus.RUNNING}:
            if pending_approval is not None:
                interrupt_payload = {
                    "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                    "approval": pending_approval.model_dump(mode="json"),
                    "recovered": True,
                    "reason": reason,
                    "metadata": metadata,
                }
            elif pending_user_input is not None:
                interrupt_payload = {
                    "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                    "user_input": pending_user_input.model_dump(mode="json"),
                    "recovered": True,
                    "reason": reason,
                    "metadata": metadata,
                }
            else:
                interrupt_payload = {
                    "reason": reason,
                    "metadata": metadata,
                    "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                    "recovered": True,
                }
            interrupt_payload["interruption_request_id"] = str(uuid4())
            try:
                session = await self.session_store.transition_status_and_checkpoint(
                    session.id,
                    from_statuses={SessionStatus.PENDING, SessionStatus.RUNNING},
                    to_status=SessionStatus.INTERRUPTING,
                    checkpoint_transform=_checkpoint_with_pending_session_interrupt(
                        interrupt_payload,
                        cascade_created_at=self._clock(),
                    ),
                )
            except ValueError:
                session = await self._require_session(session.id)
                if session.status in _RESUMABLE_SESSION_STATUSES:
                    return IncompleteSessionRecoveryResult(
                        session_id=session.id,
                        previous_status=previous_status,
                        status=session.status,
                        actions=(IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,),
                        events=(),
                        message="Session changed during recovery; recovery skipped.",
                    )
                raise
            session = await self._require_session(session.id)
            checkpoint = await self.session_store.load_checkpoint(session.id)
            pending_tool_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)

        if pending_tool_round is not None:
            transcript = await self.session_store.load_transcript(session.id)
            async for event in self._recover_pending_tool_round(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=transcript,
            ):
                events.append(event)
            actions.append(IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND)
            session = await self._require_session(session.id)
            checkpoint = await self.session_store.load_checkpoint(session.id)

        pending_approval = approval_support.pending_approval_from_checkpoint(checkpoint)
        if pending_approval is not None:
            session = await self._finalize_interrupting_for_recovery(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                events=events,
            )
            actions.append(IncompleteSessionRecoveryAction.PENDING_APPROVAL)
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=tuple(actions),
                events=tuple(events),
                pending_approval_id=pending_approval.approval_id,
                message="Session has a pending tool approval; resolve it with ToolApprovalRequest.",
            )

        pending_user_input = pending_user_input_from_checkpoint(checkpoint)
        if pending_user_input is not None:
            session = await self._finalize_interrupting_for_recovery(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                events=events,
            )
            actions.append(IncompleteSessionRecoveryAction.PENDING_USER_INPUT)
            return IncompleteSessionRecoveryResult(
                session_id=session.id,
                previous_status=previous_status,
                status=session.status,
                actions=tuple(actions),
                events=tuple(events),
                pending_user_input_id=pending_user_input.input_id,
                message="Session is awaiting user input; answer it with UserInputResponse.",
            )

        if session.status == SessionStatus.INTERRUPTING:
            session = await self._finalize_interrupting_for_recovery(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                events=events,
            )
            if previous_status == SessionStatus.INTERRUPTING:
                actions.append(IncompleteSessionRecoveryAction.FINALIZED_INTERRUPT)
            else:
                actions.append(IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED)
        elif not actions:
            actions.append(IncompleteSessionRecoveryAction.SKIPPED_TERMINAL)

        message = "Recovered incomplete session."
        if actions == [IncompleteSessionRecoveryAction.SKIPPED_TERMINAL]:
            message = "Session is terminal; recovery skipped."
        return IncompleteSessionRecoveryResult(
            session_id=session.id,
            previous_status=previous_status,
            status=session.status,
            actions=tuple(actions),
            events=tuple(events),
            message=message,
        )

    async def _finalize_abandoned_session_run(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
    ) -> None:
        """Finalize a session whose event-stream consumer went away mid-run.

        Called while handling ``GeneratorExit``, so it must not yield: it transitions
        the still-live session to INTERRUPTED and persists the terminal event (hook
        events included) without streaming them. Best effort — a closing consumer
        must never turn into a new exception.
        """
        try:
            finalized = await self.session_store.transition_status(
                session.id,
                from_statuses={
                    SessionStatus.PENDING,
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                },
                to_status=SessionStatus.INTERRUPTED,
            )
        except (KeyError, ValueError):
            # Already terminal (or gone): nothing to finalize.
            return
        if run_started_at is not None and turn_usage_tracker is not None:
            with contextlib.suppress(Exception):
                await self._emit_turn_completed_once(
                    session=finalized,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    status=SessionStatus.INTERRUPTED,
                    run_started_at=run_started_at,
                    usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                )
        with contextlib.suppress(Exception):
            async for _ in self._emit_terminal_event_with_hooks(
                event=Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=finalized.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                        "reason": _ABANDONED_RUN_REASON,
                        "abandoned": True,
                    },
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=finalized,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            ):
                pass

    async def _finalize_abandoned_session_by_id(self, session_id: str) -> None:
        """Finalize a session stranded in a live status by an abandoned stream.

        Invoked from ``except GeneratorExit`` guards at entry points and pre-resolution
        windows where the run-body finalizer (``_finalize_abandoned_session_run``) never
        runs — e.g. a consumer that closes the stream during environment-factory
        resolution or a tool-approval continuation. Idempotent and never raises: a
        session already terminal (or gone) is a no-op. The caller retains run ownership
        and releases its fence in the surrounding ``finally`` after this returns. MUST
        NOT yield.
        """
        try:
            session = await self.session_store.load(session_id)
        except Exception:
            return
        if session is None or session.status not in {
            SessionStatus.PENDING,
            SessionStatus.RUNNING,
            SessionStatus.INTERRUPTING,
        }:
            return
        try:
            registered_agent = self._get_registered_agent(session.agent_name)
        except Exception:
            # Agent no longer registered: still leave the live status rather than strand.
            with contextlib.suppress(KeyError, ValueError):
                await self.session_store.transition_status(
                    session.id,
                    from_statuses={
                        SessionStatus.PENDING,
                        SessionStatus.RUNNING,
                        SessionStatus.INTERRUPTING,
                    },
                    to_status=SessionStatus.INTERRUPTED,
                )
            return
        registered_environment = self._get_registered_environment_for_session(
            session.environment_name
        )
        with contextlib.suppress(Exception):
            await self._finalize_abandoned_session_run(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=_environment_name(registered_environment),
            )

    async def _handle_session_interrupted(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
    ) -> AsyncIterator[Event]:
        clear_current_task_cancellation()
        current_task = asyncio.current_task()
        if current_task is not None:
            self._session_control.unregister_active_task(session.id, current_task)
        self._session_control.begin_emitting_interrupted(session.id)
        try:
            loaded_interrupted = await self.session_store.load(session.id)
            if loaded_interrupted is None:
                raise KeyError(f"Session not found: {session.id}") from None
            if loaded_interrupted.status != SessionStatus.INTERRUPTED:
                loaded_interrupted = await self.session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
            payload = await self._load_pending_session_interrupt_payload(session.id, default={})
            interruption_request_id = interruption_request_id_from_payload(payload)
            existing_interrupt_event = await self._session_control.wait_for_interrupted_event(
                session.id,
                interruption_request_id=interruption_request_id,
            )
            if existing_interrupt_event is not None:
                await self._clear_pending_session_interrupt(session.id)
                if not interruption_cascade_suppressed():
                    self._schedule_background_interruption_cascade(
                        parent_session_id=session.id,
                        interrupt_payload=existing_interrupt_event.payload,
                        create_if_missing=False,
                    )
                turn_completed_event = (
                    active_run.turn_completed_event
                    if active_run is not None and active_run.turn_completed_event is not None
                    else self._session_control.active_turn_completed_event(session.id)
                )
                if turn_completed_event is not None:
                    yield turn_completed_event
                yield existing_interrupt_event
                return
            payload.setdefault("interruption_type", _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED)
            payload.setdefault("interruption_request_id", str(uuid4()))
            if run_started_at is not None and turn_usage_tracker is not None:
                yield await self._emit_turn_completed_once(
                    session=loaded_interrupted,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    status=SessionStatus.INTERRUPTED,
                    run_started_at=run_started_at,
                    usage_tracker=turn_usage_tracker,
                    active_run=active_run,
                )
            terminal_event_stream = self._emit_terminal_event_with_hooks(
                event=Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=loaded_interrupted.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=payload,
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=loaded_interrupted,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
            terminal_prefix, interrupted_event = await _collect_through_event_type(
                terminal_event_stream,
                EventType.SESSION_INTERRUPTED,
                missing_message="Session interruption produced no terminal event.",
            )

            await self._clear_pending_session_interrupt(session.id)
            if not interruption_cascade_suppressed():
                self._schedule_background_interruption_cascade(
                    parent_session_id=session.id,
                    interrupt_payload=interrupted_event.payload,
                    create_if_missing=False,
                )
            for event in terminal_prefix:
                yield event
            async for event in terminal_event_stream:
                yield event
        finally:
            self._session_control.end_emitting_interrupted(session.id)

    async def _close_tool_round_after_interrupt(
        self,
        request: InterruptedToolRoundRequest,
    ) -> AsyncIterator[Event]:
        async for event in self._close_interrupted_tool_round(
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            messages=request.messages,
            tool_calls=request.tool_calls,
            tool_outcomes=request.tool_outcomes,
            tool_round_id=request.tool_round_id,
            cancellation_artifacts=request.cancellation_artifacts,
            cancellation_artifacts_by_id=request.cancellation_artifacts_by_id,
        ):
            yield event

    async def _close_interrupted_tool_round(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        tool_outcomes: list[runtime_records.ToolCallOutcome],
        tool_round_id: str | None = None,
        cancellation_artifacts: list[dict[str, Any]] | None = None,
        cancellation_artifacts_by_id: dict[str, list[dict[str, Any]]] | None = None,
    ) -> AsyncIterator[Event]:
        if await self._tool_round_has_result_messages(session.id, tool_calls):
            return
        terminal_event_exists = (
            await self._session_control.latest_interrupted_event(session.id) is not None
        )
        interrupted_results = _interrupted_tool_round_results(
            tool_calls=tool_calls,
            completed_outcomes=tool_outcomes,
            tool_round_id=tool_round_id,
            cancellation_artifacts=cancellation_artifacts,
            cancellation_artifacts_by_id=cancellation_artifacts_by_id,
        )
        # Re-attach any background subagent child spawned by an interrupted spawn call so the parent
        # transcript keeps the parent->child linkage on the interrupt path too (AGT-02 factor-5 sweep).
        interrupted_results = await self._reattach_subagent_children_in_outcomes(
            session_id=session.id,
            tool_round_id=tool_round_id,
            outcomes=interrupted_results,
        )
        tool_outcomes = _redact_tool_call_outcomes(tool_outcomes, self._secret_redactor)
        interrupted_results = _redact_tool_call_outcomes(
            interrupted_results,
            self._secret_redactor,
        )
        if not interrupted_results and not tool_outcomes:
            return
        if not terminal_event_exists:
            for interrupted_result in interrupted_results:
                yield await self._event_writer.emit(
                    _interrupted_tool_call_event(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call_outcome=interrupted_result,
                        tool_round_id=tool_round_id,
                    )
                )
        tool_outcomes.extend(interrupted_results)
        # Restore the model's tool-call order: a parallel/mixed round's completed outcomes can be
        # in completion order, and the interrupted ones are appended after — sort back to the
        # assistant tool-call order (a no-op for an already-ordered sequential round).
        interrupted_messages = ordered_tool_result_messages(
            tool_calls, tool_outcomes, parallel=True
        )
        messages.extend(interrupted_messages)
        cleared_checkpoint = await self._tool_round_executor.checkpoint_without_pending_tool_round(
            session.id
        )
        await self.session_store.append_transcript_messages_and_transform_checkpoint(
            session.id,
            interrupted_messages,
            _replace_checkpoint_preserving_runtime_state(cleared_checkpoint),
        )

    async def _subagent_children_by_idempotency_key(
        self,
        parent_session_id: str,
    ) -> dict[str, Session]:
        """Map spawning ``idempotency_key`` -> child session for this parent's BACKGROUND subagent children.

        Only background children are re-attached: a recovered foreground child has no supported fetch path
        (``subagent_result`` refuses non-background sessions), so re-attaching it would leave a dangling
        reference. The key encodes (session, tool_round, tool_call), so it binds a child to the exact
        pending spawn call and is immune to a provider reusing a ``tool_call_id`` across rounds.
        """
        children: dict[str, Session] = {}
        for child in await self._list_all_sessions(
            SessionQuery(parent_session_id=parent_session_id, order_by=SessionOrder.CREATED_AT_ASC)
        ):
            if not _is_background_subagent_session(child):
                continue
            idempotency_key = tool_round_recovery.subagent_child_idempotency_key(child)
            if idempotency_key is not None:
                children[idempotency_key] = child
        return children

    def _reattached_subagent_result(
        self,
        children: dict[str, Session],
        idempotency_key: str,
        *,
        tool_call_id: str,
        tool_name: str,
        tool_round_id: str,
    ) -> ToolResult | None:
        """The re-attach ToolResult for a spawn call whose child is in ``children``, or None on a miss.

        Shared by the crash-recovery and live-interrupt paths; each supplies its own fallback for a miss.
        """
        child = children.get(idempotency_key)
        if child is None:
            return None
        return tool_round_recovery.recovered_subagent_tool_result(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_round_id=tool_round_id,
            child=child,
        )

    async def _reattach_subagent_children_in_outcomes(
        self,
        *,
        session_id: str,
        tool_round_id: str | None,
        outcomes: list[runtime_records.ToolCallOutcome],
    ) -> list[runtime_records.ToolCallOutcome]:
        """Replace incomplete spawn outcomes with a subagent re-attach when a background child exists.

        Factor-5 sweep of AGT-02: the live-interrupt close path records the parent->child linkage the same
        way crash recovery does, matching each outcome's call to its child by the round-scoped idempotency
        key. Returns ``outcomes`` unchanged when the round is unknown or the parent has no matching child.
        """
        if tool_round_id is None or not outcomes:
            return outcomes
        children = await self._subagent_children_by_idempotency_key(session_id)
        if not children:
            return outcomes
        reattached: list[runtime_records.ToolCallOutcome] = []
        for outcome in outcomes:
            result = self._reattached_subagent_result(
                children,
                tool_execution.tool_idempotency_key(
                    session_id=session_id,
                    tool_round_id=tool_round_id,
                    tool_call_id=outcome.call.id,
                ),
                tool_call_id=outcome.call.id,
                tool_name=outcome.call.name,
                tool_round_id=tool_round_id,
            )
            if result is None:
                reattached.append(outcome)
            else:
                reattached.append(runtime_records.ToolCallOutcome(call=outcome.call, result=result))
        return reattached

    async def _recover_pending_tool_round(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        tail_message_count: int = 0,
    ) -> AsyncIterator[Event]:
        checkpoint = await self.session_store.load_checkpoint(session.id)
        pending_round = tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint)
        if pending_round is None:
            return
        environment_name = _environment_name(registered_environment)
        if pending_round.agent_name != registered_agent.spec.name:
            raise RuntimeError(
                f"Pending tool round belongs to a different agent: {pending_round.agent_name}."
            )
        if pending_round.environment_name != environment_name:
            raise RuntimeError(
                "Pending tool round belongs to a different environment: "
                f"{pending_round.environment_name}."
            )

        pending_tool_calls = tool_round_recovery.pending_round_tool_calls(pending_round)
        if await self._tool_round_has_result_messages(session.id, pending_tool_calls):
            await self._clear_pending_tool_round_if_matches(session.id, pending_round)
            yield await self._event_writer.emit(
                Event(
                    type=EventType.SESSION_CHECKPOINTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "checkpoint": tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY,
                        "tool_round_id": pending_round.round_id,
                        "cleared": True,
                    },
                )
            )
            return

        events = await self.session_store.load_events(session.id)
        recorded_outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
            events=events,
            pending_round=pending_round,
        )
        # Re-attach any background subagent children spawned during this round: a parent that crashed
        # before its spawn tool call's terminal event still has the child linked via the child row's
        # metadata, so recover the child (id + status) instead of resolving the call as an unknown
        # outcome (AGT-02). Match on the per-call idempotency_key so a child binds to the exact pending
        # spawn call in THIS round, not a same-id child from an earlier round. Skip the child scan when
        # every pending call already has a recorded outcome (nothing to re-attach).
        subagent_children: dict[str, Session] = {}
        if any(
            recorded_outcomes.get(call.tool_call_id) is None for call in pending_round.tool_calls
        ):
            subagent_children = await self._subagent_children_by_idempotency_key(session.id)
        tool_outcomes: list[runtime_records.ToolCallOutcome] = []
        for pending_tool_call in pending_round.tool_calls:
            recorded_outcome = recorded_outcomes.get(pending_tool_call.tool_call_id)
            if recorded_outcome is not None:
                tool_outcomes.append(recorded_outcome)
                continue

            tool_call = runtime_records.ToolCallRequest(
                id=pending_tool_call.tool_call_id,
                name=pending_tool_call.tool_name,
                arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
            )
            expected_idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=pending_round.round_id,
                tool_call_id=pending_tool_call.tool_call_id,
            )
            result = self._reattached_subagent_result(
                subagent_children,
                expected_idempotency_key,
                tool_call_id=pending_tool_call.tool_call_id,
                tool_name=pending_tool_call.tool_name,
                tool_round_id=pending_round.round_id,
            )
            if result is None:
                result = tool_round_recovery.unknown_recovered_tool_result(
                    pending_tool_call=pending_tool_call,
                    pending_round=pending_round,
                    started=pending_tool_call.tool_call_id in started_ids,
                )
            event_type = (
                EventType.TOOL_CALL_FAILED if result.is_error else EventType.TOOL_CALL_COMPLETED
            )
            async for event, outcome in self._tool_round_executor.emit_tool_call_result_with_hooks(
                event=Event(
                    type=event_type,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=tool_call.name,
                    payload={
                        "tool_round_id": pending_round.round_id,
                        "tool_call_id": tool_call.id,
                        "idempotency_key": expected_idempotency_key,
                        "recovered": True,
                        "result": result.model_dump(),
                    },
                ),
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=result,
                task_id=pending_round.task_id,
            ):
                yield event
                if outcome is not None:
                    tool_outcomes.append(outcome)

        tool_result_messages = transcript_helpers.tool_result_messages(tool_outcomes)
        insert_at = len(messages) - tail_message_count
        if insert_at < 0:
            raise RuntimeError("Pending tool round recovery received an invalid tail size.")
        messages[insert_at:insert_at] = tool_result_messages
        cleared_checkpoint = await self._tool_round_executor.checkpoint_without_pending_tool_round(
            session.id
        )
        await self.session_store.append_transcript_messages_and_transform_checkpoint(
            session.id,
            tool_result_messages,
            _replace_checkpoint_preserving_runtime_state(cleared_checkpoint),
        )
        yield await self._event_writer.emit(
            Event(
                type=EventType.SESSION_CHECKPOINTED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    "checkpoint": tool_round_recovery.PENDING_TOOL_ROUND_CHECKPOINT_KEY,
                    "tool_round_id": pending_round.round_id,
                    "cleared": True,
                    "recovered_tool_calls": len(tool_outcomes),
                },
            )
        )

    async def _tool_round_has_result_messages(
        self,
        session_id: str,
        tool_calls: list[runtime_records.ToolCallRequest],
    ) -> bool:
        expected_ids = {tool_call.id for tool_call in tool_calls}
        if not expected_ids:
            return True
        transcript = await self.session_store.load_transcript(session_id)
        for message in reversed(transcript):
            result_ids = {
                part.tool_call_id for part in message.content if type(part) is ToolResultPart
            }
            if expected_ids.issubset(result_ids):
                return True
            call_ids = {part.tool_call_id for part in message.content if type(part) is ToolCallPart}
            if expected_ids & call_ids:
                return False
        return False

    async def _interrupt_background_subagent_children(
        self,
        *,
        parent_session_id: str,
        interrupt_payload: dict[str, Any],
        create_if_missing: bool = True,
        retry_request: dict[str, Any] | None = None,
    ) -> None:
        await self._background_interruption_coordinator.run_cascade(
            parent_session_id=parent_session_id,
            interrupt_payload=interrupt_payload,
            create_if_missing=create_if_missing,
            retry_request=retry_request,
        )

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
        """Return a workflow/custom emitter that permits Cayu-owned markers."""

        async def emit(events: list[Event]) -> list[Event]:
            _validate_workflow_event_batch(events, allow_cayu_internal=True)
            return await self._event_writer.emit_many(session_id, events)

        return emit

    async def emit_event(self, event: Event) -> Event:
        """Publish an event to the session store and all sinks.

        Low-level seam for runtime-owned out-of-band session events. Prefer
        ``scoped_event_emitter`` when handing an emitter to a component. Redaction
        is applied by the sinks; callers must not place raw secrets in the payload.
        """
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
    ) -> AsyncIterator[Event]:
        finalize_result = await self._environment_lifecycle.finalize_terminal_event(
            event=event,
            session=session,
            registered_environment=registered_environment,
        )
        for binding_event in finalize_result.events:
            yield binding_event
        terminal_event = await self._event_writer.emit(finalize_result.event)
        yield terminal_event
        async for hook_event in self._run_runtime_hooks(
            phase=phase,
            session=session,
            terminal_event=terminal_event,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            hooks=self._runtime_hooks,
            scope="app",
        ):
            yield hook_event
        async for hook_event in self._run_runtime_hooks(
            phase=phase,
            session=session,
            terminal_event=terminal_event,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            hooks=registered_agent.runtime_hooks,
            scope="agent",
        ):
            yield hook_event

    async def _run_runtime_hooks(
        self,
        *,
        phase: RuntimeHookPhase,
        session: Session,
        terminal_event: Event,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        hooks: tuple[RuntimeHook, ...],
        scope: str,
    ) -> AsyncIterator[Event]:
        for hook in hooks:
            if not _runtime_hook_supports_phase(
                hook=hook,
                phase=phase,
            ):
                continue
            hook_name = require_clean_nonblank(hook.name, "runtime_hook.name")
            yield await self._event_writer.emit(
                _runtime_hook_event(
                    event_type=EventType.HOOK_STARTED,
                    hook_name=hook_name,
                    scope=scope,
                    phase=phase,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    terminal_event=terminal_event,
                    payload={},
                )
            )
            context = RuntimeHookContext(
                runtime=self,
                hook_name=hook_name,
                phase=phase,
                session=session,
                terminal_event=terminal_event,
            )
            try:
                await _call_runtime_hook(hook=hook, phase=phase, context=context)
            except Exception as exc:
                yield await self._event_writer.emit(
                    _runtime_hook_event(
                        event_type=EventType.HOOK_FAILED,
                        hook_name=hook_name,
                        scope=scope,
                        phase=phase,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        terminal_event=terminal_event,
                        payload={
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "actions": context.actions,
                        },
                    )
                )
                continue
            yield await self._event_writer.emit(
                _runtime_hook_event(
                    event_type=EventType.HOOK_COMPLETED,
                    hook_name=hook_name,
                    scope=scope,
                    phase=phase,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    terminal_event=terminal_event,
                    payload={
                        "actions": context.actions,
                    },
                )
            )

    async def _run_before_stop_policies(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        step_result: AssistantStepResult,
        step: int,
        max_steps: int,
        request_metadata: dict[str, Any],
        request_loop_policies: tuple[LoopPolicy, ...],
    ) -> AsyncIterator[tuple[Event, BeforeStopDecision | None]]:
        classification = classify_assistant_step(step_result)
        policy_groups = (
            ("app", self._loop_policies),
            ("agent", registered_agent.loop_policies),
            (
                "request",
                validate_loop_policies(
                    request_loop_policies,
                    field_name="request_loop_policies",
                ),
            ),
        )
        for scope, policies in policy_groups:
            for policy in policies:
                if not _loop_policy_supports_before_stop(policy):
                    continue
                policy_name = require_clean_nonblank(policy.name, "loop_policy.name")
                yield (
                    await self._event_writer.emit(
                        _before_stop_policy_event(
                            event_type="custom.loop.before_stop.started",
                            policy_name=policy_name,
                            scope=scope,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            step=step,
                            classification=classification,
                            payload={},
                        )
                    ),
                    None,
                )
                context = BeforeStopContext(
                    session=session,
                    step_result=step_result,
                    classification=classification,
                    step=step,
                    max_steps=max_steps,
                    metadata=request_metadata,
                )
                try:
                    decision = copy_before_stop_decision(await policy.before_stop(context))
                except Exception as exc:
                    yield (
                        await self._event_writer.emit(
                            _before_stop_policy_event(
                                event_type="custom.loop.before_stop.failed",
                                policy_name=policy_name,
                                scope=scope,
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                step=step,
                                classification=classification,
                                payload={
                                    "error": str(exc),
                                    "error_type": type(exc).__name__,
                                },
                            )
                        ),
                        None,
                    )
                    raise
                yield (
                    await self._event_writer.emit(
                        _before_stop_policy_event(
                            event_type="custom.loop.before_stop.completed",
                            policy_name=policy_name,
                            scope=scope,
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            step=step,
                            classification=classification,
                            payload={
                                "action": decision.action.value,
                                "reason": decision.reason,
                                "metadata": copy_json_value(decision.metadata, "metadata"),
                            },
                        )
                    ),
                    None,
                )
                if decision.action != BeforeStopAction.COMPLETE:
                    yield (
                        await self._event_writer.emit(
                            _before_stop_policy_event(
                                event_type="custom.loop.before_stop.selected",
                                policy_name=policy_name,
                                scope=scope,
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                step=step,
                                classification=classification,
                                payload={
                                    "action": decision.action.value,
                                    "reason": decision.reason,
                                    "metadata": copy_json_value(
                                        decision.metadata,
                                        "metadata",
                                    ),
                                },
                            )
                        ),
                        decision,
                    )
                    return

    async def emit_events(self, session_id: str, events: list[Event]) -> list[Event]:
        """Persist events for one session and fan them out to runtime sinks.

        Restricted to the ``workflow.`` and ``custom.`` namespaces: runtime
        event namespaces encode Cayu-owned lifecycle and accounting evidence,
        so application callers must not forge them even though every accepted
        batch uses the same durable budget/sink handoff.
        """
        _validate_workflow_event_batch(events, allow_cayu_internal=False)
        return await self._event_writer.emit_many(session_id, events)


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
        if not allow_cayu_internal and event_type.startswith("custom.cayu."):
            raise ValueError("The custom.cayu. namespace is reserved for cayu internals.")


def _copy_registered_tool(tool: runtime_records.RegisteredTool) -> runtime_records.RegisteredTool:
    return runtime_records.RegisteredTool(
        name=tool.name,
        description=tool.description,
        schema=deepcopy(tool.schema),
        parallel_safe=tool.parallel_safe,
        effect=tool.effect,
        tool=tool.tool,
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


def _validate_registered_tool(tool: Tool) -> runtime_records.RegisteredTool:
    spec = getattr(tool, "spec", None)
    if type(spec) is not ToolSpec:
        raise TypeError("Agent tools must define ToolSpec instances.")
    name = require_clean_nonblank(spec.name, "name")
    if name == STRUCTURED_OUTPUT_TOOL_NAME:
        raise ValueError(f"Tool name is reserved for structured output: {name}")
    validated_spec = ToolSpec(
        name=name,
        description=spec.description,
        input_schema=copy_json_value(spec.input_schema, "input_schema"),
        parallel_safe=spec.parallel_safe,
        effect=spec.effect,
    )
    return runtime_records.RegisteredTool(
        name=validated_spec.name,
        description=validated_spec.description,
        schema=validated_spec.input_schema,
        parallel_safe=validated_spec.parallel_safe,
        effect=validated_spec.effect,
        tool=tool,
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


def _validate_environment_spec(spec: EnvironmentSpec) -> EnvironmentSpec:
    if type(spec) is not EnvironmentSpec:
        raise TypeError("Environment registration requires an EnvironmentSpec.")
    if type(spec.name) is not str:
        raise ValueError("`name` must be a string.")
    return EnvironmentSpec(
        name=spec.name,
        metadata=copy_json_value(spec.metadata, "metadata"),
    )


def _compact_session_request_digest(request: CompactSessionRequest) -> str:
    payload = request.model_dump(mode="json")
    # The authenticated caller is audit data for each attempt, not part of the
    # operation's semantic identity. A different operator must be able to
    # recover the same idempotent request after its lease expires.
    payload.pop("requested_by", None)
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _optional_text_digest(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode()).hexdigest()


def _session_operation_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    stored = checkpoint.get(_SESSION_OPERATIONS_CHECKPOINT_KEY)
    if stored is None:
        return {"version": 1, "active_operation_id": None, "records": {}}
    if type(stored) is not dict:
        raise ValueError("Session operation checkpoint must be an object.")
    operations = copy_json_value(stored, _SESSION_OPERATIONS_CHECKPOINT_KEY)
    if operations.get("version") != 1:
        raise ValueError("Unsupported session operation checkpoint version.")
    records = operations.get("records")
    if type(records) is not dict:
        raise ValueError("Session operation checkpoint records must be an object.")
    active_operation_id = operations.get("active_operation_id")
    if active_operation_id is not None:
        require_clean_nonblank(active_operation_id, "active_operation_id")
    return operations


def _active_session_operation_id(checkpoint: dict[str, Any] | None) -> str | None:
    if checkpoint is None or _SESSION_OPERATIONS_CHECKPOINT_KEY not in checkpoint:
        return None
    active_operation_id = _session_operation_state(checkpoint).get("active_operation_id")
    return active_operation_id if type(active_operation_id) is str else None


def _operation_claim_expiry(record: dict[str, Any]) -> datetime | None:
    value = record.get("claim_expires_at")
    if type(value) is not str:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _abandon_expired_session_operation(
    operations: dict[str, Any],
    *,
    now: datetime,
) -> str | None:
    active_operation_id = operations.get("active_operation_id")
    if type(active_operation_id) is not str:
        return None
    records = operations.get("records")
    if type(records) is not dict:
        raise ValueError("Session operation checkpoint records must be an object.")
    active_record = next(
        (
            record
            for record in records.values()
            if type(record) is dict and record.get("operation_id") == active_operation_id
        ),
        None,
    )
    if active_record is None:
        raise RuntimeError("Active durable session operation record is missing.")
    expiry = _operation_claim_expiry(active_record)
    if active_record.get("status") != "running" or expiry is None or expiry > now:
        return None
    active_record["status"] = "abandoned"
    active_record["abandoned_at"] = now.isoformat()
    active_record["updated_at"] = now.isoformat()
    active_record.pop("claim_expires_at", None)
    operations["active_operation_id"] = None
    return active_operation_id


def _session_compaction_replay_event_ids(
    record: dict[str, Any],
) -> tuple[str, ...] | None:
    status = record.get("status")
    if status in {"running", "abandoned"}:
        return None
    if status not in {"completed", "failed"}:
        raise RuntimeError(f"Unsupported session compaction operation status: {status}")
    stored_event_ids = record.get("event_ids")
    if type(stored_event_ids) is not list or not all(
        type(event_id) is str and event_id for event_id in stored_event_ids
    ):
        raise RuntimeError("Completed session compaction operation is missing replay event ids.")
    return tuple(stored_event_ids)


def _store_session_operation_state(
    checkpoint: dict[str, Any],
    operations: dict[str, Any],
) -> None:
    records = operations.get("records")
    if type(records) is not dict:
        raise ValueError("Session operation checkpoint records must be an object.")
    if records or operations.get("active_operation_id") is not None:
        checkpoint[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
    else:
        checkpoint.pop(_SESSION_OPERATIONS_CHECKPOINT_KEY, None)


def _archive_inactive_session_operation_records(
    checkpoint: dict[str, Any],
    *,
    except_idempotency_key: str,
) -> dict[str, dict[str, Any]]:
    """Move inactive operation records out of the live checkpoint."""

    operations = _session_operation_state(checkpoint)
    records = operations["records"]
    archived: dict[str, dict[str, Any]] = {}
    for key, record in list(records.items()):
        if key == except_idempotency_key:
            continue
        if type(record) is not dict:
            raise ValueError("Session operation checkpoint records must be objects.")
        if record.get("status") == "running":
            raise RuntimeError("Checkpoint contains an untracked running session operation.")
        archived[key] = records.pop(key)
    _store_session_operation_state(checkpoint, operations)
    return archived


def _reject_unresumable_session_checkpoint(
    session: Session,
    checkpoint: dict[str, Any] | None,
    *,
    allow_active_operation: bool = False,
) -> None:
    if approval_support.pending_approval_from_checkpoint(checkpoint) is not None:
        raise RuntimeError("Session has a pending tool approval.")
    if pending_user_input_from_checkpoint(checkpoint) is not None:
        raise RuntimeError("Session is awaiting user input.")
    if tool_round_recovery.pending_tool_round_from_checkpoint(checkpoint) is not None:
        raise RuntimeError("Session has a pending tool round.")
    if checkpoint is not None and _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY in checkpoint:
        raise RuntimeError("Session has an incomplete background interruption cascade.")
    if checkpoint is not None and not allow_active_operation:
        operations = _session_operation_state(checkpoint)
        active_operation_id = operations.get("active_operation_id")
        if active_operation_id is not None:
            raise RuntimeError(f"Session has an active durable operation: {active_operation_id}")


def _application_compaction_causal_payload(
    *,
    request: CompactSessionRequest,
    operation_id: str,
    attempt_id: str,
    source_cursor: int,
    compactor: str,
    result_cursor: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "operation_id": operation_id,
        "attempt_id": attempt_id,
        "request_id": request.idempotency_key,
        "reason": request.reason,
        "source_run_epoch": request.expected_run_epoch,
        "source_transcript_cursor": source_cursor,
        "compactor": _application_compaction_event_text(compactor) or "ContextCompactor",
        "mode": "bounded",
        "instruction_present": request.instructions is not None,
        "instruction_digest": _optional_text_digest(request.instructions),
        "actor": resolution_actor_payload(request.requested_by),
    }
    if type(result_cursor) is int and result_cursor >= 0:
        payload["result_transcript_cursor"] = result_cursor
    return payload


_APPLICATION_COMPACTION_EVENT_TEXT_MAX_BYTES = 512
_APPLICATION_COMPACTION_EVENT_INTEGER_MAX = 9_223_372_036_854_775_807


def _application_compaction_event_text(value: Any) -> str | None:
    if type(value) is not str or not value or value != value.strip():
        return None
    if any(
        0xD800 <= ord(char) <= 0xDFFF or ord(char) < 0x20 or ord(char) == 0x7F for char in value
    ):
        return None
    if len(value.encode("utf-8")) > _APPLICATION_COMPACTION_EVENT_TEXT_MAX_BYTES:
        return None
    return value


def _require_application_compaction_event_text(value: Any, field_name: str) -> str:
    value = require_clean_nonblank(value, field_name)
    if _application_compaction_event_text(value) is None:
        raise ValueError(
            f"`{field_name}` must contain valid Unicode without control characters "
            f"and be at most {_APPLICATION_COMPACTION_EVENT_TEXT_MAX_BYTES} UTF-8 bytes."
        )
    return value


def _application_compaction_event_integer(value: Any) -> int | None:
    if type(value) is not int or value < 0 or value > _APPLICATION_COMPACTION_EVENT_INTEGER_MAX:
        return None
    return value


def _application_compaction_raw_usage(value: Any) -> dict[str, Any] | None:
    if type(value) is not dict:
        return None
    raw_usage: dict[str, Any] = {}
    for key in (
        "input_tokens",
        "prompt_tokens",
        "output_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        bounded = _application_compaction_event_integer(value.get(key))
        if bounded is not None:
            raw_usage[key] = bounded
    for key, allowed_keys in (
        ("input_tokens_details", ("cached_tokens",)),
        ("prompt_tokens_details", ("cached_tokens",)),
        ("output_tokens_details", ("reasoning_tokens", "thinking_tokens")),
        ("completion_tokens_details", ("reasoning_tokens", "thinking_tokens")),
    ):
        details = value.get(key)
        if type(details) is not dict:
            continue
        bounded_details = {
            detail_key: bounded
            for detail_key in allowed_keys
            if (bounded := _application_compaction_event_integer(details.get(detail_key)))
            is not None
        }
        if bounded_details:
            raw_usage[key] = bounded_details
    cache_creation = value.get("cache_creation")
    if type(cache_creation) is dict:
        cache_creation_total = 0
        for index, cache_value in enumerate(cache_creation.values()):
            if index >= 16:
                break
            bounded = _application_compaction_event_integer(cache_value)
            if bounded is not None:
                cache_creation_total += bounded
        if cache_creation_total:
            raw_usage["cache_creation"] = {"bounded_total": cache_creation_total}
    return raw_usage or None


def _application_compaction_usage_metrics(payload: dict[str, Any]) -> UsageMetrics | None:
    supplied_metrics = payload.get("usage_metrics")
    identity_source = supplied_metrics if type(supplied_metrics) is dict else payload
    provider_name = _application_compaction_event_text(identity_source.get("provider_name"))
    requested_model = _application_compaction_event_text(identity_source.get("requested_model"))
    model = _application_compaction_event_text(identity_source.get("model"))
    if type(supplied_metrics) is dict:
        return usage_metrics_from_event_payload(
            {
                "billing_identity": payload.get("billing_identity"),
                "usage_metrics": copy_json_value(supplied_metrics, "usage_metrics"),
            }
        )
    raw_usage = _application_compaction_raw_usage(payload.get("usage"))
    if raw_usage is None:
        return None
    return usage_metrics_from_event_payload(
        {
            "provider_name": provider_name,
            "requested_model": requested_model,
            "model": model,
            "usage": raw_usage,
        }
    )


def _application_compaction_event(
    *,
    telemetry: ContextCompactionTelemetry,
    request: CompactSessionRequest,
    operation_id: str,
    attempt_id: str,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    compactor: str,
) -> Event:
    telemetry_payload = telemetry.payload
    payload: dict[str, Any] = {}
    if telemetry.event_type == EventType.MODEL_COMPLETED:
        metrics = _application_compaction_usage_metrics(telemetry_payload)
        raw_billing_identity = telemetry_payload.get("billing_identity")
        billing_identity = (
            BillingIdentity.model_validate(raw_billing_identity)
            if type(raw_billing_identity) is dict
            else None
        )
        if billing_identity is None and metrics is not None:
            billing_identity = metrics.billing_identity
        payload["purpose"] = "context_compaction"
        if billing_identity is not None:
            payload["billing_identity"] = billing_identity.model_dump(mode="json")
        if metrics is not None:
            serialized_metrics = metrics.model_dump()
            serialized_metrics.pop("billing_identity", None)
            payload["usage_metrics"] = serialized_metrics
            for key in ("provider_name", "requested_model", "model"):
                value = getattr(metrics, key)
                if value is not None:
                    payload[key] = value
        else:
            for key in ("provider_name", "requested_model", "model"):
                value = _application_compaction_event_text(telemetry_payload.get(key))
                if value is not None:
                    payload[key] = value
        for key in ("compaction_outcome", "usage_unavailable_reason"):
            value = _application_compaction_event_text(telemetry_payload.get(key))
            if value is not None:
                payload[key] = value
    elif telemetry.event_type == EventType.CONTEXT_COMPACTION_COMPLETED:
        payload["checkpoint"] = "context_compaction"
        for key in (
            "compacted_transcript_cursor",
            "previous_compacted_transcript_cursor",
            "newly_compacted_message_count",
            "recent_message_count",
            "summary_chars",
        ):
            value = _application_compaction_event_integer(telemetry_payload.get(key))
            if value is not None:
                payload[key] = value
    payload.update(
        _application_compaction_causal_payload(
            request=request,
            operation_id=operation_id,
            attempt_id=attempt_id,
            source_cursor=request.expected_transcript_cursor,
            result_cursor=payload.get("compacted_transcript_cursor"),
            compactor=compactor,
        )
    )
    return Event(
        type=telemetry.event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=payload,
    )


def _application_compaction_budget_event(
    *,
    check: BudgetCheck,
    request: CompactSessionRequest,
    operation_id: str,
    attempt_id: str,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    compactor: str,
) -> Event:
    return Event(
        type=EventType.BUDGET_LIMIT_REACHED,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload={
            **budget_check_payload(check),
            **_application_compaction_causal_payload(
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                source_cursor=request.expected_transcript_cursor,
                compactor=compactor,
            ),
        },
    )


def _application_compaction_ledger_event(
    *,
    event_type: EventType,
    payload: dict[str, Any],
    request: CompactSessionRequest,
    operation_id: str,
    attempt_id: str,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    compactor: str,
) -> Event:
    return Event(
        type=event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload={
            **copy_json_value(payload, "compaction_budget_payload"),
            **_application_compaction_causal_payload(
                request=request,
                operation_id=operation_id,
                attempt_id=attempt_id,
                source_cursor=request.expected_transcript_cursor,
                compactor=compactor,
            ),
        },
    )


def _budget_check_identity(
    check: BudgetCheck,
) -> tuple[str, str | None, str, str, Decimal]:
    return (
        check.scope,
        check.key,
        check.window.storage_key,
        check.action,
        check.maximum,
    )


def _complete_session_operation_checkpoint(
    *,
    checkpoint: dict[str, Any] | None,
    persisted_record: dict[str, Any] | None,
    compacted_checkpoint: dict[str, Any],
    idempotency_key: str,
    operation_id: str,
    attempt_id: str,
    event_ids: list[str],
    result_cursor: Any,
    completed_at: datetime,
) -> SessionOperationPublication:
    if checkpoint is None:
        raise SessionCompactionAttemptSuperseded(
            "Session compaction attempt was superseded before publication."
        )
    updated = copy_json_value(checkpoint, "checkpoint")
    operations = _session_operation_state(updated)
    record = operations["records"].get(idempotency_key)
    if type(record) is not dict or record.get("operation_id") != operation_id:
        if persisted_record is not None and persisted_record.get("operation_id") == operation_id:
            raise SessionCompactionAttemptSuperseded(
                "Session compaction attempt was superseded before publication."
            )
        raise RuntimeError("Session compaction operation claim was lost before publication.")
    if (
        record.get("status") != "running"
        or record.get("current_attempt_id") != attempt_id
        or operations.get("active_operation_id") != operation_id
    ):
        raise SessionCompactionAttemptSuperseded(
            "Session compaction attempt was superseded before publication."
        )
    if persisted_record is not None:
        if persisted_record.get("operation_id") != operation_id:
            raise RuntimeError(
                "Session compaction operation replay record does not match its claim."
            )
        if persisted_record.get("status") != "abandoned":
            raise SessionCompactionAttemptSuperseded(
                "Session compaction attempt was superseded before publication."
            )
    compacted_state = copy_json_value(compacted_checkpoint, "compacted_checkpoint")
    compacted_context = compacted_state.get(_CONTEXT_COMPACTION_OPERATION_KIND)
    if type(compacted_context) is not dict:
        raise ValueError("Compacted checkpoint is missing context compaction state.")
    updated[_CONTEXT_COMPACTION_OPERATION_KIND] = compacted_context
    record["status"] = "completed"
    existing_event_ids = record.get("event_ids", [])
    record["event_ids"] = [
        *existing_event_ids,
        *(event_id for event_id in event_ids if event_id not in existing_event_ids),
    ]
    record["result_transcript_cursor"] = result_cursor
    record["completed_at"] = completed_at.isoformat()
    record["updated_at"] = completed_at.isoformat()
    record.pop("claim_expires_at", None)
    operations["active_operation_id"] = None
    terminal_record = operations["records"].pop(idempotency_key)
    _store_session_operation_state(updated, operations)
    return SessionOperationPublication(
        checkpoint=updated,
        operation_records={idempotency_key: terminal_record},
    )


def _append_session_operation_attempt_events(
    *,
    idempotency_key: str,
    operation_id: str,
    attempt_id: str,
    event_ids: list[str],
    updated_at: datetime,
) -> Callable[
    [Session, dict[str, Any] | None, dict[str, Any] | None],
    SessionOperationPublication,
]:
    def append_events(
        _session: Session,
        checkpoint: dict[str, Any] | None,
        persisted_record: dict[str, Any] | None,
    ) -> SessionOperationPublication:
        if checkpoint is None:
            raise SessionCompactionAttemptSuperseded(
                "Session compaction attempt was superseded before event publication."
            )
        updated = copy_json_value(checkpoint, "checkpoint")
        operations = _session_operation_state(updated)
        record = operations["records"].get(idempotency_key)
        if type(record) is not dict or record.get("operation_id") != operation_id:
            if (
                persisted_record is not None
                and persisted_record.get("operation_id") == operation_id
            ):
                raise SessionCompactionAttemptSuperseded(
                    "Session compaction attempt was superseded before event publication."
                )
            raise RuntimeError("Session compaction operation claim was lost before publication.")
        if (
            record.get("status") != "running"
            or record.get("current_attempt_id") != attempt_id
            or operations.get("active_operation_id") != operation_id
        ):
            raise SessionCompactionAttemptSuperseded(
                "Session compaction attempt was superseded before event publication."
            )
        existing_event_ids = record.get("event_ids", [])
        record["event_ids"] = [
            *existing_event_ids,
            *(event_id for event_id in event_ids if event_id not in existing_event_ids),
        ]
        record["updated_at"] = updated_at.isoformat()
        updated[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
        return SessionOperationPublication(checkpoint=updated)

    return append_events


def _fail_session_operation_checkpoint(
    *,
    idempotency_key: str,
    operation_id: str,
    attempt_id: str,
    failed_event_id: str,
    attempt_event_ids: list[str],
    error_type: str,
    completed_at: datetime,
) -> Callable[
    [Session, dict[str, Any] | None, dict[str, Any] | None],
    SessionOperationPublication,
]:
    def fail(
        _session: Session,
        checkpoint: dict[str, Any] | None,
        persisted_record: dict[str, Any] | None,
    ) -> SessionOperationPublication:
        updated = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        operations = _session_operation_state(updated)
        record = operations["records"].get(idempotency_key)
        if type(record) is not dict or record.get("operation_id") != operation_id:
            if (
                persisted_record is not None
                and persisted_record.get("operation_id") == operation_id
            ):
                terminal_record = copy_json_value(persisted_record, "session_operation")
                existing_event_ids = terminal_record.get("event_ids", [])
                terminal_record["event_ids"] = [
                    *existing_event_ids,
                    *(
                        event_id
                        for event_id in [*attempt_event_ids, failed_event_id]
                        if event_id not in existing_event_ids
                    ),
                ]
                return SessionOperationPublication(
                    checkpoint=updated,
                    operation_records={idempotency_key: terminal_record},
                )
            raise RuntimeError("Session compaction operation claim was lost before failure.")
        existing_event_ids = record.get("event_ids", [])
        record["event_ids"] = [
            *existing_event_ids,
            *(
                event_id
                for event_id in [*attempt_event_ids, failed_event_id]
                if event_id not in existing_event_ids
            ),
        ]
        if (
            record.get("status") != "running"
            or record.get("current_attempt_id") != attempt_id
            or operations.get("active_operation_id") != operation_id
        ):
            updated[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
            return SessionOperationPublication(checkpoint=updated)
        record["status"] = "failed"
        record["error_type"] = error_type
        record["completed_at"] = completed_at.isoformat()
        record["updated_at"] = completed_at.isoformat()
        record.pop("claim_expires_at", None)
        operations["active_operation_id"] = None
        terminal_record = operations["records"].pop(idempotency_key)
        _store_session_operation_state(updated, operations)
        return SessionOperationPublication(
            checkpoint=updated,
            operation_records={idempotency_key: terminal_record},
        )

    return fail


def _validate_run_request(request: RunRequest) -> RunRequest:
    return copy_run_request(request)


def _validate_resume_request(request: ResumeRequest) -> ResumeRequest:
    return copy_resume_request(request)


def _validate_tool_approval_request(request: ToolApprovalRequest) -> ToolApprovalRequest:
    return copy_tool_approval_request(request)


def _validate_tool_approval_recovery_request(
    request: ToolApprovalRecoveryRequest,
) -> ToolApprovalRecoveryRequest:
    return copy_tool_approval_recovery_request(request)


def _effective_tool_round_structured_output(
    *,
    structured_output: StructuredOutputSpec | None,
    pending_round: tool_round_recovery.PendingToolRound,
) -> StructuredOutputSpec | None:
    # Mirror _effective_user_input_structured_output: inherit the crashed run's spec when
    # the operator supplies none; adopt the operator's spec when the run had none; a
    # differing spec is a swap of the contract fixed by the provider history and is
    # rejected.
    if type(pending_round) is not tool_round_recovery.PendingToolRound:
        raise TypeError("Pending tool round must be a PendingToolRound.")
    if structured_output is None:
        return copy_structured_output_spec(pending_round.structured_output)
    if pending_round.structured_output is None:
        return copy_structured_output_spec(structured_output)
    if not _structured_output_specs_equal(structured_output, pending_round.structured_output):
        raise ValueError("structured_output does not match the crashed run contract.")
    return copy_structured_output_spec(pending_round.structured_output)


def _require_native_structured_output_support(
    structured_output: StructuredOutputSpec | None,
    *,
    registered_provider: runtime_records.RegisteredProvider,
) -> None:
    """Reject an unusable ``strategy=NATIVE`` spec at an entry point.

    Called by every entrance before it creates a session or transitions its
    status, so the typed error reaches the caller with no persisted state
    changed (model-pattern routing can select the provider by model name
    alone). Raises ``NativeStructuredOutputUnsupported`` when the resolved
    provider has no native mode, then lets the provider's own schema preflight
    reject schemas its native mode would refuse at request time
    (``NativeStructuredOutputSchemaInvalid``). One pre-existing mid-run call
    in ``_run_session`` reuses this helper as a typed backstop for missed
    entrances; a raise there fails the already-running session cleanly.
    """
    if structured_output is None or structured_output.strategy != StructuredOutputStrategy.NATIVE:
        return
    if not registered_provider.provider.supports_native_structured_output:
        raise NativeStructuredOutputUnsupported(
            f"Native structured output is not supported by provider: {registered_provider.name}"
        )
    registered_provider.provider.preflight_native_structured_output_schema(
        copy_json_value(structured_output.json_schema, "json_schema")
    )


def _structured_output_specs_equal(
    left: StructuredOutputSpec,
    right: StructuredOutputSpec,
) -> bool:
    if type(left) is not StructuredOutputSpec or type(right) is not StructuredOutputSpec:
        raise TypeError("Structured output comparison requires StructuredOutputSpec values.")
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def _session_identity(*, provider_name: str, model: str) -> SessionIdentity:
    return SessionIdentity(
        provider_name=provider_name,
        model=model,
        runtime_name="cayu",
        runtime_version=_runtime_version(),
    )


def _runtime_version() -> str | None:
    try:
        return version("cayu")
    except PackageNotFoundError:
        return None


def _with_environment_name(request: RunRequest, environment_name: str) -> RunRequest:
    return RunRequest(
        agent_name=request.agent_name,
        messages=[message.model_copy(deep=True) for message in request.messages],
        session_id=request.session_id,
        parent_session_id=request.parent_session_id,
        causal_budget_id=request.causal_budget_id,
        task_id=request.task_id,
        task_worker_id=request.task_worker_id,
        provider_name=request.provider_name,
        model=request.model,
        environment_name=environment_name,
        labels=copy_label_map(request.labels, "labels"),
        metadata=copy_json_value(request.metadata, "metadata"),
        max_steps=request.max_steps,
        limits=copy_run_limits(request.limits),
        budget_limits=copy_request_budget_limits(request.budget_limits),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
        structured_output=copy_structured_output_spec(request.structured_output),
        thinking=request.thinking,
        loop_policies=validate_loop_policies(request.loop_policies, field_name="loop_policies"),
    )


def _environment_name(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None:
        return None
    return registered_environment.spec.name


def _session_trace_event_fields(
    session: Session,
    request_metadata: dict[str, Any],
) -> dict[str, Any]:
    # Surface the parent linkage and any inbound W3C trace context on the session
    # start event so an OpenTelemetryEventSink can parent the root span. Additive:
    # other sinks ignore these keys.
    fields: dict[str, Any] = {}
    if session.parent_session_id:
        fields["parent_session_id"] = session.parent_session_id
    traceparent = request_metadata.get("traceparent")
    if isinstance(traceparent, str) and traceparent:
        fields["traceparent"] = traceparent
        tracestate = request_metadata.get("tracestate")
        if isinstance(tracestate, str) and tracestate:
            fields["tracestate"] = tracestate
    return fields


async def _call_runtime_hook(
    *,
    hook: RuntimeHook,
    phase: RuntimeHookPhase,
    context: RuntimeHookContext,
) -> None:
    if phase == RuntimeHookPhase.AFTER_SESSION_COMPLETED:
        await hook.after_session_completed(context)
        return
    if phase == RuntimeHookPhase.AFTER_SESSION_FAILED:
        await hook.after_session_failed(context)
        return
    if phase == RuntimeHookPhase.AFTER_SESSION_INTERRUPTED:
        await hook.after_session_interrupted(context)
        return
    raise ValueError(f"Unsupported runtime hook phase: {phase}")


def _loop_policy_supports_before_stop(policy: LoopPolicy) -> bool:
    policy_method = type(policy).before_stop
    default_method = LoopPolicy.before_stop
    return policy_method is not default_method


def _before_stop_policy_event(
    *,
    event_type: str,
    policy_name: str,
    scope: str,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    step: int,
    classification: StepClassification,
    payload: dict[str, Any],
) -> Event:
    event_payload = {
        "policy": policy_name,
        "scope": scope,
        "step": step,
        "classification": classification.payload(),
        **copy_json_value(payload, "payload"),
    }
    return Event(
        type=event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        payload=event_payload,
    )


def _checkpoint_with_pending_session_interrupt(
    payload: dict[str, Any],
    *,
    include_interruption_cascade: bool = True,
    cascade_created_at: datetime | None = None,
):
    copied_payload = copy_json_value(payload, "interrupt_payload")
    if cascade_created_at is not None and (
        cascade_created_at.tzinfo is None or cascade_created_at.utcoffset() is None
    ):
        raise ValueError("cascade_created_at must be timezone-aware.")
    resolved_cascade_created_at = (
        datetime.now(UTC) if cascade_created_at is None else cascade_created_at.astimezone(UTC)
    )

    def transform(_session: Session, checkpoint: dict[str, Any] | None) -> dict[str, Any]:
        copied_checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        copied_checkpoint[_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY] = copy_json_value(
            copied_payload,
            "interrupt_payload",
        )
        if (
            include_interruption_cascade
            and copied_payload.get("interruption_type") == _INTERRUPTION_TYPE_OPERATOR_REQUESTED
        ):
            copied_checkpoint[_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY] = {
                "attempt_id": str(uuid4()),
                "interrupt_payload": copy_json_value(
                    copied_payload,
                    "interrupt_payload",
                ),
                "created_at": resolved_cascade_created_at.isoformat(),
            }
        return copied_checkpoint

    return transform


def _replace_checkpoint_preserving_runtime_state(
    checkpoint: dict[str, Any],
):
    replacement = copy_json_value(checkpoint, "checkpoint")

    def transform(_session: Session, current: dict[str, Any] | None) -> dict[str, Any]:
        updated = copy_json_value(replacement, "checkpoint")
        for key, field_name in (
            (_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY, "pending_session_interrupt"),
            (_PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY, "pending_interruption_cascade"),
            (_SESSION_OPERATIONS_CHECKPOINT_KEY, "session_operations"),
        ):
            updated.pop(key, None)
            if current is not None and key in current:
                updated[key] = copy_json_value(current[key], field_name)
        return updated

    return transform


def _interrupted_tool_round_results(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    completed_outcomes: list[runtime_records.ToolCallOutcome],
    tool_round_id: str | None = None,
    cancellation_artifacts: list[dict[str, Any]] | None = None,
    cancellation_artifacts_by_id: dict[str, list[dict[str, Any]]] | None = None,
) -> list[runtime_records.ToolCallOutcome]:
    completed_ids = {outcome.call.id for outcome in completed_outcomes}
    # Prefer per-call attribution (a parallel round records the producing tool_call_id); otherwise
    # fall back to attaching a bare artifact list to the first unfinished call (sequential: its only
    # in-flight call).
    artifacts_for_interrupted_tool = (
        [] if cancellation_artifacts is None else cancellation_artifacts
    )
    interrupted_outcomes: list[runtime_records.ToolCallOutcome] = []
    for tool_call in tool_calls:
        if tool_call.id in completed_ids:
            continue
        if cancellation_artifacts_by_id is not None:
            result_artifacts = cancellation_artifacts_by_id.get(tool_call.id, [])
        else:
            result_artifacts = artifacts_for_interrupted_tool
            artifacts_for_interrupted_tool = []
        structured = {
            "interrupted": True,
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
        }
        if tool_round_id is not None:
            structured["tool_round_id"] = tool_round_id
        interrupted_outcomes.append(
            runtime_records.ToolCallOutcome(
                call=tool_call,
                result=ToolResult(
                    content="Tool call interrupted before completion.",
                    structured=structured,
                    artifacts=result_artifacts,
                    is_error=True,
                ),
            )
        )
    return interrupted_outcomes


def _limit_reached_payload(
    *,
    decision: StopDecision,
    usage_summary: SessionUsageSummary,
    cost_summary: SessionCostSummary | None,
) -> dict[str, Any]:
    payload = {
        "reason": "limit_reached",
        "limit": decision.limit.value,
        "maximum": _limit_value_for_payload(decision.maximum),
        "actual": _limit_value_for_payload(decision.actual),
        "message": decision.message,
        "usage_summary": usage_summary.model_dump(),
    }
    if cost_summary is not None:
        payload["cost_summary"] = cost_summary.model_dump(mode="json")
    return payload


def _has_run_budget_limit(limits: tuple[BudgetLimit, ...]) -> bool:
    return any(limit.scope == "run" for limit in limits)


def _limit_value_for_payload(value: int | Decimal) -> int | str:
    if type(value) is Decimal:
        return str(value)
    if type(value) is int:
        return value
    raise TypeError("limit payload value must be an int or Decimal.")


def _limit_reached_tool_round_results(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    decision: StopDecision,
    tool_round_id: str | None = None,
) -> list[runtime_records.ToolCallOutcome]:
    outcomes: list[runtime_records.ToolCallOutcome] = []
    for tool_call in tool_calls:
        structured = {
            "skipped": True,
            "reason": "limit_reached",
            "limit": decision.limit.value,
            "maximum": _limit_value_for_payload(decision.maximum),
            "actual": _limit_value_for_payload(decision.actual),
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
        }
        if tool_round_id is not None:
            structured["tool_round_id"] = tool_round_id
        outcomes.append(
            runtime_records.ToolCallOutcome(
                call=tool_call,
                result=ToolResult(
                    content="Tool call skipped because a run limit was reached.",
                    structured=structured,
                    is_error=True,
                ),
            )
        )
    return outcomes


def _interrupted_tool_call_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    tool_call_outcome: runtime_records.ToolCallOutcome,
    tool_round_id: str | None = None,
) -> Event:
    payload = {
        "tool_call_id": tool_call_outcome.call.id,
        "idempotency_key": tool_execution.tool_idempotency_key(
            session_id=session.id,
            tool_round_id=tool_round_id,
            tool_call_id=tool_call_outcome.call.id,
        ),
        "result": tool_call_outcome.result.model_dump(),
    }
    if tool_round_id is not None:
        payload["tool_round_id"] = tool_round_id
    # The event type must match the result: genuine interruptions are is_error=True (FAILED), but a
    # re-attached subagent child that already COMPLETED yields is_error=False and must emit COMPLETED —
    # otherwise the event contradicts its own payload (mirrors the crash-recovery path).
    return Event(
        type=(
            EventType.TOOL_CALL_FAILED
            if tool_call_outcome.result.is_error
            else EventType.TOOL_CALL_COMPLETED
        ),
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        tool_name=tool_call_outcome.call.name,
        payload=payload,
    )


def _limit_reached_tool_call_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    tool_call_outcome: runtime_records.ToolCallOutcome,
    decision: StopDecision,
    tool_round_id: str | None = None,
) -> Event:
    payload = {
        "tool_call_id": tool_call_outcome.call.id,
        "idempotency_key": tool_execution.tool_idempotency_key(
            session_id=session.id,
            tool_round_id=tool_round_id,
            tool_call_id=tool_call_outcome.call.id,
        ),
        "reason": "limit_reached",
        "limit": decision.limit.value,
        "result": tool_call_outcome.result.model_dump(),
    }
    if tool_round_id is not None:
        payload["tool_round_id"] = tool_round_id
    return Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        tool_name=tool_call_outcome.call.name,
        payload=payload,
    )


async def _close_async_iterator(iterator: AsyncIterator[Any]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


async def _collect_through_event_type(
    iterator: AsyncIterator[Event],
    event_type: EventType,
    *,
    missing_message: str,
) -> tuple[list[Event], Event]:
    events: list[Event] = []
    async for event in iterator:
        events.append(event)
        if event.type == event_type:
            return events, event
    raise RuntimeError(missing_message)


def _runtime_hook_event(
    *,
    event_type: EventType,
    hook_name: str,
    scope: str,
    phase: RuntimeHookPhase,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    terminal_event: Event,
    payload: dict[str, Any],
) -> Event:
    return _build_runtime_hook_event(
        event_type=event_type,
        hook_name=hook_name,
        scope=scope,
        phase=phase,
        session=session,
        terminal_event=terminal_event,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        payload=payload,
    )


def _recovery_task_event(request: RecoveryTaskEventRequest) -> Event:
    return _task_event(
        event_type=request.event_type,
        task=request.task,
        session=request.session,
        registered_agent=request.registered_agent,
        registered_environment=request.registered_environment,
    )


def _task_event(
    *,
    event_type: EventType,
    task: Task,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> Event:
    return Event(
        type=event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        payload={
            "task_id": task.id,
            "task_type": task.type,
            "task_status": task.status.value,
            "task_session_id": task.session_id,
            "assigned_agent_name": task.assigned_agent_name,
            "parent_task_id": task.parent_task_id,
        },
    )


def _workspace_id(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None or registered_environment.environment.workspace is None:
        return None
    workspace_id = getattr(registered_environment.environment.workspace, "id", None)
    if workspace_id is None:
        return None
    return require_clean_nonblank(workspace_id, "workspace.id")


def _workspace(registered_environment: runtime_records.RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.workspace


def _artifact_store_id(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None or registered_environment.environment.artifact_store is None:
        return None
    artifact_store_id = getattr(registered_environment.environment.artifact_store, "id", None)
    if artifact_store_id is None:
        return None
    artifact_store_id = require_clean_nonblank(artifact_store_id, "artifact_store.id")
    return require_unicode_scalar_text(artifact_store_id, "artifact_store.id")


def _artifact_store(registered_environment: runtime_records.RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.artifact_store


def _knowledge_store(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.knowledge_store


def _has_structured_output_tool_call(
    tool_calls: list[runtime_records.ToolCallRequest],
) -> bool:
    return any(tool_call.name == STRUCTURED_OUTPUT_TOOL_NAME for tool_call in tool_calls)


def _user_tool_call_count(tool_calls: list[runtime_records.ToolCallRequest]) -> int:
    return sum(1 for tool_call in tool_calls if tool_call.name != STRUCTURED_OUTPUT_TOOL_NAME)


def _validate_structured_output_tool_round(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    spec: StructuredOutputSpec,
) -> StructuredOutputValidation:
    internal_calls = [
        tool_call for tool_call in tool_calls if tool_call.name == STRUCTURED_OUTPUT_TOOL_NAME
    ]
    if len(internal_calls) != 1:
        return _structured_output_tool_error(
            "Call the structured-output tool exactly once when submitting final output."
        )
    if len(tool_calls) != 1:
        return _structured_output_tool_error(
            "Call the structured-output tool by itself, not in the same tool round as other tools."
        )
    return validate_structured_output_tool_arguments(internal_calls[0].arguments, spec)


def _structured_output_tool_round_outcomes(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    spec: StructuredOutputSpec,
    validation: StructuredOutputValidation,
) -> list[runtime_records.ToolCallOutcome]:
    if validation.valid:
        return [
            runtime_records.ToolCallOutcome(
                call=tool_calls[0],
                result=ToolResult(
                    content="Structured output accepted.",
                    structured={"output": copy_json_value(validation.output, "output")},
                ),
            )
        ]

    repair_lead = structured_output_repair_lead(spec)
    error_summary = _structured_output_error_summary(validation)
    outcomes: list[runtime_records.ToolCallOutcome] = []
    for tool_call in tool_calls:
        if tool_call.name == STRUCTURED_OUTPUT_TOOL_NAME:
            content = f"Structured output rejected: {error_summary}\n\n{repair_lead}"
        else:
            content = (
                "Tool was not executed because the structured-output finalizer was "
                "called in the same tool round. Retry the needed work before submitting "
                "final structured output."
            )
        outcomes.append(
            runtime_records.ToolCallOutcome(
                call=tool_call,
                result=ToolResult(
                    content=content,
                    structured={
                        "structured_output_errors": [
                            error.model_dump(mode="json") for error in validation.errors
                        ],
                    },
                    is_error=True,
                ),
            )
        )
    return outcomes


def _structured_output_tool_error(message: str) -> StructuredOutputValidation:
    return StructuredOutputValidation(
        valid=False,
        errors=[
            StructuredOutputError(
                path="$",
                message=message,
                schema_path="$",
            )
        ],
    )


def _structured_output_error_summary(validation: StructuredOutputValidation) -> str:
    if not validation.errors:
        return "unknown validation error."
    return "; ".join(f"{error.path}: {error.message}" for error in validation.errors[:3])


def _structured_output_event(
    *,
    event_type: EventType,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    spec: StructuredOutputSpec,
    validation: StructuredOutputValidation,
    step: int,
    attempt: int,
    redactor: SecretRedactor | None = None,
) -> Event:
    if event_type not in {
        EventType.STRUCTURED_OUTPUT_VALIDATED,
        EventType.STRUCTURED_OUTPUT_FAILED,
        EventType.STRUCTURED_OUTPUT_RETRY,
    }:
        raise ValueError(f"Unsupported structured output event type: {event_type}")
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")
    if type(validation) is not StructuredOutputValidation:
        raise TypeError("Structured output validation must be a StructuredOutputValidation.")
    validation = _redact_structured_output_validation(validation, redactor)
    payload: dict[str, Any] = {
        "name": spec.name,
        "step": step,
        "attempt": attempt,
        "max_retries": spec.max_retries,
        "valid": validation.valid,
        "errors": [error.model_dump(mode="json") for error in validation.errors],
    }
    if validation.valid:
        payload["output"] = copy_json_value(validation.output, "output")
    return Event(
        type=event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=payload,
    )


def _structured_output_validating_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    spec: StructuredOutputSpec,
    step: int,
    attempt: int,
) -> Event:
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")
    return Event(
        type=EventType.STRUCTURED_OUTPUT_VALIDATING,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload={
            "name": spec.name,
            "strategy": spec.strategy.value,
            "step": step,
            "attempt": attempt,
            "max_retries": spec.max_retries,
        },
    )


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
) -> tuple[RuntimeHook, ...]:
    if hooks is None:
        return ()
    if isinstance(hooks, str | bytes):
        raise TypeError(f"{field_name} must be an iterable of RuntimeHook instances.")
    try:
        hook_list = list(hooks)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of RuntimeHook instances.") from exc
    for hook in hook_list:
        if not isinstance(hook, RuntimeHook):
            raise TypeError(f"{field_name} must contain RuntimeHook instances.")
        require_clean_nonblank(hook.name, "runtime_hook.name")
    return tuple(hook_list)


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
            raise ValueError(f"Duplicate event watcher name: {watcher.name}")
        names.add(watcher.name)
    return tuple(watcher_list)


def _redact_tool_call_outcomes(
    outcomes: list[runtime_records.ToolCallOutcome],
    redactor: SecretRedactor,
) -> list[runtime_records.ToolCallOutcome]:
    if not redactor.has_values:
        return outcomes
    return [_redact_tool_call_outcome(outcome, redactor) for outcome in outcomes]


def _redact_tool_call_outcome(
    outcome: runtime_records.ToolCallOutcome,
    redactor: SecretRedactor,
) -> runtime_records.ToolCallOutcome:
    return runtime_records.ToolCallOutcome(
        call=outcome.call,
        result=tool_results.redact_tool_result(outcome.result, redactor),
    )


def _redact_structured_output_validation(
    validation: StructuredOutputValidation,
    redactor: SecretRedactor | None,
) -> StructuredOutputValidation:
    if type(validation) is not StructuredOutputValidation:
        raise TypeError("Structured output validation must be a StructuredOutputValidation.")
    if redactor is None or not redactor.has_values:
        return validation
    return StructuredOutputValidation(
        valid=validation.valid,
        output=redactor.redact_json(validation.output),
        errors=[
            StructuredOutputError(
                path=redactor.redact_text(error.path),
                message=redactor.redact_text(error.message),
                schema_path=redactor.redact_text(error.schema_path),
            )
            for error in validation.errors
        ],
    )
