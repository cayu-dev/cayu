from __future__ import annotations

import asyncio
import inspect
import mimetypes
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cayu._validation import (
    copy_json_value,
    copy_label_map,
    require_clean_nonblank,
    require_nonblank,
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
from cayu.core.events import Event, EventType
from cayu.core.messages import (
    FilePart,
    Message,
)
from cayu.core.thinking import ThinkingConfig
from cayu.core.tools import (
    Tool,
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
from cayu.providers import (
    ModelProvider,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._interruption_coordinator import (
    BackgroundInterruptionCoordinator,
)
from cayu.runtime._model_step_executor import (
    ModelStepBudgetEvaluationRequest,
    ModelStepBudgetReservationFailureRequest,
    ModelStepExecutor,
    ModelStepLimitEvaluationRequest,
)
from cayu.runtime._recovery_coordinator import (
    RecoveryAbandonedTurnRequest,
    RecoveryCoordinator,
    RecoveryInterruptionRequest,
    RecoveryLimitStopRequest,
    RecoverySessionRunRequest,
    RecoveryTaskEventRequest,
    RecoveryTerminalEventRequest,
)
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
    _replace_checkpoint_preserving_runtime_state,
    _task_event,
    _validate_resume_request,
    _validate_run_request,
)
from cayu.runtime._session_queries import query_all_event_records, query_all_sessions
from cayu.runtime._tool_round_executor import (
    InterruptedToolRoundRequest,
    ToolRoundExecutor,
    ToolRoundLimitRequest,
)
from cayu.runtime.approvals import (
    PendingToolApproval,
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
    RuntimeHookPhase,
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
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    IncompleteSessionsRecoveryRequest,
    InMemorySessionStore,
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
    Session,
    SessionOrder,
    SessionQuery,
    SessionStatus,
    SessionStore,
    _SessionRunFenceContext,
    copy_fork_session_request,
    copy_incomplete_session_recovery_request,
    copy_incomplete_sessions_recovery_request,
    copy_interrupt_session_request,
)
from cayu.runtime.stop_policy import (
    RunLimits,
    StopDecision,
)
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
)
from cayu.runtime.tasks import (
    Task,
    TaskCreate,
    TaskStore,
    copy_task_create,
)
from cayu.runtime.tool_policy import (
    AllowAllToolPolicy,
    ToolPolicy,
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
from cayu.storage.memory import KnowledgeStore
from cayu.vaults import (
    SecretRedactor,
)

RegisteredAgent = runtime_records.RegisteredAgent
RegisteredEnvironment = runtime_records.RegisteredEnvironment


DEFAULT_MAX_PARALLEL_TOOL_CALLS = 4


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
    for failure in (*failures, authoritative_failure.__cause__):
        if failure is None or failure is authoritative_failure:
            continue
        if any(candidate is failure for candidate in evidence):
            continue
        evidence.append(failure)
    if not evidence:
        return
    authoritative_failure.__cause__ = (
        evidence[0] if len(evidence) == 1 else BaseExceptionGroup(message, evidence)
    )


async def _close_owned_event_stream_resisting_cancellation(
    owned_stream: _RunFenceOwnedEventStream,
) -> tuple[asyncio.CancelledError | None, BaseException | None]:
    """Finish delegated cleanup despite cancellation of the awaiting task."""

    cleanup_task = asyncio.create_task(owned_stream.aclose())
    cancellation: asyncio.CancelledError | None = None
    while not cleanup_task.done():
        try:
            await asyncio.wait(
                (cleanup_task,),
                return_when=asyncio.ALL_COMPLETED,
            )
        except asyncio.CancelledError as exc:
            # asyncio.wait raises only when this caller is cancelled. A cancelled
            # cleanup task completes the wait and is inspected through result() below.
            if cancellation is None:
                cancellation = exc
            else:
                cancellation.add_note(
                    "Additional cancellation arrived during delegated stream cleanup."
                )
            if cleanup_task.cancelled():
                break
            continue

    cleanup_failure: BaseException | None = None
    try:
        cleanup_task.result()
    except BaseException as exc:
        cleanup_failure = exc
    if cancellation is None and isinstance(cleanup_failure, asyncio.CancelledError):
        cancellation = cleanup_failure
        cleanup_failure = None
    return cancellation, cleanup_failure


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
        cancellation, cleanup_failure = await _close_owned_event_stream_resisting_cancellation(
            owned_stream
        )
        if cancellation is not None:
            if authoritative_failure is not None and authoritative_failure is not cancellation:
                cancellation.add_note(
                    "Delegated runtime stream cleanup was cancelled after an earlier "
                    f"{type(authoritative_failure).__name__}."
                )
            if cleanup_failure is not None and cleanup_failure is not cancellation:
                cancellation.add_note(
                    "Delegated runtime stream cleanup also failed: "
                    f"{type(cleanup_failure).__name__}."
                )
            _attach_delegated_failure_causes(
                cancellation,
                (authoritative_failure, cleanup_failure),
                message="Delegated runtime stream cancellation evidence",
            )
            raise cancellation
        if cleanup_failure is not None:
            if authoritative_failure is None or isinstance(authoritative_failure, GeneratorExit):
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
            secret_redactor=self._secret_redactor,
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
            stop_session_for_limit_reached=self._stop_recovery_session_for_limit_reached,
            task_event=_recovery_task_event,
            resolve_registered_agent=self._get_registered_agent,
            resolve_registered_provider=self._get_registered_provider,
            resolve_registered_environment=self._get_registered_environment_for_session,
            interrupt_session_for_recovery=self._interrupt_session_for_recovery,
            pending_session_interrupt_checkpoint=(
                self._pending_session_interrupt_checkpoint_for_recovery
            ),
            abandoned_turn_completed=self._complete_abandoned_recovery_turn,
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

        self._session_engine = SessionEngine(
            session_store=self.session_store,
            task_store=self.task_store,
            get_budget_policy=lambda: self.budget_policy,
            event_writer=self._event_writer,
            environment_lifecycle=self._environment_lifecycle,
            run_limit_controller=self._run_limit_controller,
            session_control=self._session_control,
            model_step_executor=self._model_step_executor,
            tool_round_executor=self._tool_round_executor,
            recovery_coordinator=self._recovery_coordinator,
            background_interruption_coordinator=(self._background_interruption_coordinator),
            secret_redactor=self._secret_redactor,
            clock=self._clock,
            runtime_hooks=self._runtime_hooks,
            loop_policies=self._loop_policies,
            hook_runtime=self,
            get_registered_agent=self._get_registered_agent,
            get_registered_provider=self._get_registered_provider,
            route_registered_provider_for_model=(
                lambda model: self._route_registered_provider_for_model(model=model)
            ),
            get_registered_environment=self._get_registered_environment,
            get_registered_environment_for_session=(self._get_registered_environment_for_session),
            effective_retry_policy=self._effective_retry_policy,
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
        return await self._session_engine.drain_background_interruptions(timeout_s=timeout_s)

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
        session_id = require_clean_nonblank(session_id, "session_id")
        return await self._session_engine.interruption_cascade_status(session_id=session_id)

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
            execution_requirements=stored_execution_requirements,
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
        artifact_store: ArtifactStore | None = None,
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
            environment=Environment(stored_spec, artifact_store=artifact_store),
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
        stream = self._session_engine.run(request=request)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def resume(self, request: ResumeRequest) -> AsyncIterator[Event]:
        if type(request) is not ResumeRequest:
            raise TypeError("Runtime resume requires a ResumeRequest.")
        request = _validate_resume_request(request)
        stream = self._session_engine.resume(request=request)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def compact_session(
        self,
        request: CompactSessionRequest,
    ) -> AsyncIterator[Event]:
        if type(request) is not CompactSessionRequest:
            raise TypeError("Runtime compaction requires a CompactSessionRequest.")
        stream = self._session_engine.compact_session(request=request)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
    ) -> EnqueueSessionMessageResult:
        if type(request) is not EnqueueSessionMessageRequest:
            raise TypeError("Runtime queued input requires an EnqueueSessionMessageRequest.")
        return await self._session_engine.enqueue_session_message(request=request)

    async def interrupt_session(self, request: InterruptSessionRequest) -> AsyncIterator[Event]:
        if type(request) is not InterruptSessionRequest:
            raise TypeError("Runtime interruption requires an InterruptSessionRequest.")
        request = copy_interrupt_session_request(request)
        stream = self._session_engine.interrupt_session(request=request)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

    async def recover_incomplete_session(
        self,
        request: IncompleteSessionRecoveryRequest,
    ) -> IncompleteSessionRecoveryResult:
        request = copy_incomplete_session_recovery_request(request)
        return await self._recovery_coordinator.recover_incomplete_session(request)

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
        return await self._recovery_coordinator.recover_incomplete_sessions(request)

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
        session_stream = self._session_engine._resume_session(
            request=resume_request,
            task_id=request.task_id,
            start_event_payload_extra=start_event_payload_extra,
            start_task_on_enter=True,
        )
        async with _close_delegated_event_stream(session_stream) as owned_stream:
            forwarded_stream = self._session_control.stream_with_out_of_band_events(
                request.session_id,
                owned_stream,
            )
            async with _close_delegated_event_stream(forwarded_stream) as owned_forwarded_stream:
                async for event in owned_forwarded_stream:
                    yield event

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

    async def fork_session(self, request: ForkSessionRequest) -> AsyncIterator[Event]:
        if type(request) is not ForkSessionRequest:
            raise TypeError("Runtime fork requires a ForkSessionRequest.")
        request = copy_fork_session_request(request)
        stream = self._session_engine.fork_session(request=request)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                yield item

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

    def _interrupt_session_for_recovery(
        self,
        request: RecoveryInterruptionRequest,
    ) -> AsyncIterator[Event]:
        return self._handle_session_interrupted(
            session=request.session,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            environment_name=request.environment_name,
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
    ) -> Event:
        return await self._emit_turn_completed_once(
            session=request.session,
            registered_agent=request.registered_agent,
            environment_name=request.environment_name,
            status=SessionStatus.INTERRUPTED,
            run_started_at=request.run_started_at,
            usage_tracker=request.usage_tracker,
            active_run=request.active_run,
        )

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
        stream = self._recovery_coordinator.resolve_user_input(response=response)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield event

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
        stream = self._recovery_coordinator.recover_user_input_request(request=request)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield event

    async def resolve_tool_approval(
        self,
        request: ToolApprovalRequest,
    ) -> AsyncIterator[Event]:
        if type(request) is not ToolApprovalRequest:
            raise TypeError("Runtime approval resolution requires a ToolApprovalRequest.")
        request = _validate_tool_approval_request(request)
        stream = self._recovery_coordinator.resolve_tool_approval(request=request)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield event

    async def recover_tool_approval(
        self,
        request: ToolApprovalRecoveryRequest,
    ) -> AsyncIterator[Event]:
        if type(request) is not ToolApprovalRecoveryRequest:
            raise TypeError("Runtime approval recovery requires a ToolApprovalRecoveryRequest.")
        request = _validate_tool_approval_recovery_request(request)
        stream = self._recovery_coordinator.recover_tool_approval_request(request=request)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield event

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
        stream = self._recovery_coordinator.recover_tool_round_request(request=request)
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
        stream = self._session_engine._run_session(
            session=session,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            messages=messages,
            messages_to_append=messages_to_append,
            max_steps=max_steps,
            limits=limits,
            budget_limits=budget_limits,
            retry_policy=retry_policy,
            structured_output=structured_output,
            thinking=thinking,
            request_loop_policies=request_loop_policies,
            request_metadata=request_metadata,
            task_id=task_id,
            task_worker_id=task_worker_id,
            start_event_type=start_event_type,
            start_event_payload=start_event_payload,
            start_task_on_enter=start_task_on_enter,
            release_run_fence_on_exit=release_run_fence_on_exit,
        )
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
        return await self._session_engine._emit_turn_completed_once(
            session=session,
            registered_agent=registered_agent,
            environment_name=environment_name,
            status=status,
            run_started_at=run_started_at,
            usage_tracker=usage_tracker,
            active_run=active_run,
        )

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
        tool_round_id: str | None = None,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
    ) -> AsyncIterator[Event]:
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
            tool_round_id=tool_round_id,
            run_started_at=run_started_at,
            turn_usage_tracker=turn_usage_tracker,
            active_run=active_run,
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
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
    ) -> AsyncIterator[Event]:
        stream = self._session_engine._handle_session_interrupted(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
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
        stream = self._session_engine._emit_terminal_event_with_hooks(
            event=event,
            phase=phase,
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
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
    if not inspect.iscoroutinefunction(tool.run):
        raise TypeError(
            f"{type(tool).__name__}.run must be declared with `async def` and return a ToolResult."
        )
    schema = copy_json_value(tool.schema, "schema")
    if type(schema) is not dict:
        raise TypeError(f"{type(tool).__name__}.schema must return a JSON Schema object.")
    validated_spec = ToolSpec(
        name=name,
        description=spec.description,
        input_schema=schema,
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
