from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Iterable
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.artifacts import (
    DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
    DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST,
    DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES,
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    FileAttachment,
    file_attachment_from_payload,
    resolved_file_attachment,
)
from cayu.core.agents import AgentSpec
from cayu.core.events import Event, EventType
from cayu.core.messages import (
    Message,
    MessageRole,
    ProviderStatePart,
    ToolCallPart,
    ToolResultPart,
)
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.environments import Environment, EnvironmentSpec, copy_environment
from cayu.providers import (
    ModelCompletion,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    copy_model_stream_event,
    normalize_model_completion,
)
from cayu.runners import RunnerCancelledError
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime.approvals import (
    PendingToolApproval,
    ToolApprovalDecision,
    ToolApprovalRecoveryRequest,
    ToolApprovalRequest,
    copy_pending_tool_approval,
    copy_tool_approval_recovery_request,
    copy_tool_approval_request,
)
from cayu.runtime.context import (
    ContextBuildError,
    ContextCompactionTelemetry,
    ContextPolicy,
    ContextRequest,
    DefaultContextPolicy,
    RuntimeManagedContextPolicy,
    copy_context_messages,
)
from cayu.runtime.costs import (
    CostBudget,
    PricingCatalog,
    SessionCostSummary,
    copy_cost_budget,
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
from cayu.runtime.hooks import (
    RuntimeHook,
    RuntimeHookContext,
    RuntimeHookPhase,
    ToolCallHookContext,
)
from cayu.runtime.model_steps import (
    AssistantStepResult,
    assistant_text_content,
    classify_assistant_step,
    provider_state_count,
)
from cayu.runtime.retry_policy import (
    RetryDecision,
    RetryPolicy,
    copy_retry_policy,
    retry_decision,
    retry_event_payload,
)
from cayu.runtime.sessions import (
    EventQuery,
    EventRecord,
    ForkSessionRequest,
    InMemorySessionStore,
    InterruptSessionRequest,
    ResumeRequest,
    RunRequest,
    Session,
    SessionIdentity,
    SessionStatus,
    SessionStore,
    copy_fork_session_request,
    copy_interrupt_session_request,
    copy_resume_request,
    copy_run_request,
)
from cayu.runtime.stop_policy import (
    RunLimits,
    StopDecision,
    StopLimit,
    copy_run_limits,
    first_reached_limit,
    has_run_limits,
)
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputError,
    StructuredOutputSpec,
    StructuredOutputValidation,
    copy_structured_output_spec,
    structured_output_repair_lead,
    structured_output_repair_prompt,
    structured_output_spec_payload,
    structured_output_tool_instruction,
    structured_output_tool_required_validation,
    structured_output_tool_spec,
    validate_structured_output_tool_arguments,
)
from cayu.runtime.tasks import Task, TaskCreate, TaskStore, copy_task_create
from cayu.runtime.tool_policy import (
    AllowAllToolPolicy,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.runtime.usage import (
    SessionUsageSummary,
    UsageMetrics,
    normalize_usage_metrics,
    session_usage_summary,
    usage_metrics_payload,
)

RegisteredAgent = runtime_records.RegisteredAgent
RegisteredEnvironment = runtime_records.RegisteredEnvironment


class _SessionInterrupted(Exception):
    def __init__(self, approval: PendingToolApproval) -> None:
        super().__init__(f"Tool call requires approval: {approval.tool_name}")
        self.approval = copy_pending_tool_approval(approval)


class _SessionInterruptedByRequest(Exception):
    def __init__(self, session_id: str) -> None:
        self.session_id = require_clean_nonblank(session_id, "session_id")
        super().__init__(f"Session interrupted: {self.session_id}")


class _ModelAttemptFailed(Exception):
    def __init__(
        self,
        *,
        message: str,
        payload: dict[str, Any],
        emitted_error_event: bool,
        cause: Exception | None = None,
    ) -> None:
        self.message = require_nonblank(message, "message")
        self.payload = copy_json_value(payload, "payload")
        self.emitted_error_event = emitted_error_event
        self.cause = cause
        super().__init__(self.message)


@dataclass
class _ActiveSessionRun:
    runtime_task: asyncio.Task[Any]
    task_id: str | None
    task_started: bool
    task_finished: bool


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
_INTERRUPT_REQUESTED_SESSION_STATUSES = {
    SessionStatus.INTERRUPTING,
    SessionStatus.INTERRUPTED,
}
_INTERRUPTED_EVENT_WAIT_ATTEMPTS = 10
_INTERRUPTED_EVENT_WAIT_INTERVAL_S = 0.01
_ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS = 600
_ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S = 0.01
_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY = "pending_session_interrupt"
_INTERRUPTION_TYPE_OPERATOR_REQUESTED = "operator_requested"
_INTERRUPTION_TYPE_RUNTIME_INTERRUPTED = "runtime_interrupted"
_INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED = "tool_approval_required"
_INTERRUPTION_TYPE_LIMIT_REACHED = "limit_reached"


class CayuApp:
    """Application runtime for registered agents, providers, and session state."""

    def __init__(
        self,
        *,
        session_store: SessionStore | None = None,
        task_store: TaskStore | None = None,
        dispatcher: Dispatcher | None = None,
        retry_policy: RetryPolicy | None = None,
        runtime_hooks: Iterable[RuntimeHook] | None = None,
        event_sinks: Iterable[EventSink] | None = None,
        enable_logging: bool = True,
        max_file_attachment_bytes: int = DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
        max_total_file_attachment_bytes: int = DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES,
        max_file_attachments_per_request: int = DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST,
    ) -> None:
        if session_store is not None and not isinstance(session_store, SessionStore):
            raise TypeError("session_store must be a SessionStore.")
        if task_store is not None and not isinstance(task_store, TaskStore):
            raise TypeError("task_store must be a TaskStore.")
        if dispatcher is not None and not isinstance(dispatcher, Dispatcher):
            raise TypeError("dispatcher must be a Dispatcher.")
        if type(enable_logging) is not bool:
            raise TypeError("enable_logging must be a bool.")
        hooks = _validate_runtime_hooks(runtime_hooks, field_name="runtime_hooks")
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

            sinks.insert(0, LoggingEventSink())
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
        self.session_store = session_store if session_store is not None else InMemorySessionStore()
        self.task_store = task_store
        self.dispatcher = dispatcher if dispatcher is not None else InlineDispatcher()
        self._default_retry_policy = copy_retry_policy(retry_policy)
        self._runtime_hooks = tuple(hooks)
        self._event_sinks = sinks
        self._agents: dict[str, runtime_records.RegisteredAgentState] = {}
        self._providers: dict[str, runtime_records.RegisteredProvider] = {}
        self._environments: dict[str, runtime_records.RegisteredEnvironment] = {}
        self._default_provider_name: str | None = None
        self._default_environment_name: str | None = None
        self._active_session_runs: dict[str, dict[asyncio.Task[Any], _ActiveSessionRun]] = {}
        self._sessions_emitting_interrupted: set[str] = set()
        self._sessions_requesting_interruption: set[str] = set()

    def register_agent(
        self,
        spec: AgentSpec,
        *,
        tools: Iterable[Tool] | None = None,
        context_policy: ContextPolicy | None = None,
        tool_policy: ToolPolicy | None = None,
        runtime_hooks: Iterable[RuntimeHook] | None = None,
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

        self._agents[stored_spec.name] = runtime_records.RegisteredAgentState(
            spec=stored_spec,
            tools=MappingProxyType(tools_by_name),
            context_policy=stored_context_policy,
            tool_policy=stored_tool_policy,
            runtime_hooks=stored_runtime_hooks,
        )
        return spec

    def register_provider(
        self,
        provider: ModelProvider,
        *,
        default: bool = False,
    ) -> ModelProvider:
        if not isinstance(provider, ModelProvider):
            raise TypeError("Provider registration requires a ModelProvider.")
        if not isinstance(default, bool):
            raise TypeError("Provider default flag must be a bool.")
        require_clean_nonblank(provider.name, "provider.name")
        if provider.name in self._providers:
            raise ValueError(f"Provider already registered: {provider.name}")

        self._providers[provider.name] = runtime_records.RegisteredProvider(
            name=provider.name,
            provider=provider,
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
        if type(environment) is not Environment:
            raise TypeError("Environment registration requires an Environment.")
        if not isinstance(default, bool):
            raise TypeError("Environment default flag must be a bool.")
        stored_environment = copy_environment(environment)
        stored_spec = _validate_environment_spec(stored_environment.spec)
        if stored_spec.name in self._environments:
            raise ValueError(f"Environment already registered: {stored_spec.name}")

        self._environments[stored_spec.name] = runtime_records.RegisteredEnvironment(
            spec=stored_spec,
            environment=stored_environment,
        )
        if default or self._default_environment_name is None:
            self._default_environment_name = stored_spec.name
        return environment

    def get_agent(self, name: str) -> runtime_records.RegisteredAgent:
        agent_name = require_clean_nonblank(name, "agent.name")
        registered_agent = self._get_registered_agent(agent_name)
        return runtime_records.RegisteredAgent(
            spec=registered_agent.spec.model_copy(deep=True),
            tools={
                name: _copy_registered_tool(tool) for name, tool in registered_agent.tools.items()
            },
        )

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
        return runtime_records.RegisteredEnvironment(
            spec=registered_environment.spec.model_copy(deep=True),
            environment=copy_environment(registered_environment.environment),
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
        registered_provider = self._get_registered_provider()
        registered_environment = self._get_registered_environment(request.environment_name)
        if request.environment_name is None and registered_environment is not None:
            request = _with_environment_name(request, registered_environment.spec.name)
        session = await self.session_store.create(
            request,
            identity=_session_identity(
                provider_name=registered_provider.name,
                model=registered_agent.spec.model,
            ),
        )
        await self.session_store.update_status(session.id, SessionStatus.RUNNING)
        messages = transcript_helpers.initial_messages(
            system_prompt=registered_agent.spec.system_prompt,
            request_messages=request.messages,
        )

        async for event in self._run_session(
            session=session,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            messages=messages,
            messages_to_append=messages,
            max_steps=request.max_steps,
            limits=request.limits,
            cost_budget=request.cost_budget,
            retry_policy=self._effective_retry_policy(request.retry_policy),
            structured_output=request.structured_output,
            request_metadata=request.metadata,
            task_id=request.task_id,
            start_event_type=EventType.SESSION_STARTED,
            start_event_payload={"agent_name": registered_agent.spec.name},
        ):
            yield event

    async def resume(self, request: ResumeRequest) -> AsyncIterator[Event]:
        if type(request) is not ResumeRequest:
            raise TypeError("Runtime resume requires a ResumeRequest.")
        request = _validate_resume_request(request)
        async for event in self._resume_session(
            request=request,
            task_id=None,
            start_event_payload_extra={},
        ):
            yield event

    async def interrupt_session(self, request: InterruptSessionRequest) -> AsyncIterator[Event]:
        if type(request) is not InterruptSessionRequest:
            raise TypeError("Runtime interruption requires an InterruptSessionRequest.")
        request = copy_interrupt_session_request(request)
        loaded_session = await self.session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")
        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )
        if loaded_session.status == SessionStatus.INTERRUPTED:
            existing_interrupt_event = await self._wait_for_active_session_interrupted_event(
                loaded_session.id
            )
            if existing_interrupt_event is not None:
                yield existing_interrupt_event
                return
            raise RuntimeError(
                f"Session is interrupted but has no session.interrupted event: {loaded_session.id}"
            )

        if loaded_session.status == SessionStatus.INTERRUPTING:
            existing_interrupt_event = await self._wait_for_active_session_interrupted_event(
                loaded_session.id
            )
            if existing_interrupt_event is not None:
                yield existing_interrupt_event
                return
            raise TimeoutError(f"Session interruption is still finalizing: {loaded_session.id}")

        if loaded_session.status not in _INTERRUPTIBLE_SESSION_STATUSES:
            raise ValueError(f"Session cannot be interrupted from status: {loaded_session.status}")

        interrupt_payload = {
            "reason": request.reason,
            "metadata": request.metadata,
            "interruption_type": _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
        }
        self._sessions_requesting_interruption.add(loaded_session.id)
        request_marker_active = True
        try:
            session = await self.session_store.transition_status_and_checkpoint(
                loaded_session.id,
                from_statuses=_INTERRUPTIBLE_SESSION_STATUSES,
                to_status=SessionStatus.INTERRUPTING,
                checkpoint_transform=_checkpoint_with_pending_session_interrupt(interrupt_payload),
            )
            active_work_signalled = self._interrupt_active_session_runs(session.id)
            if active_work_signalled:
                existing_interrupt_event = await self._wait_for_active_session_interrupted_event(
                    session.id
                )
                if existing_interrupt_event is not None:
                    request_marker_active = False
                    self._sessions_requesting_interruption.discard(loaded_session.id)
                    yield existing_interrupt_event
                    return
                raise TimeoutError(f"Session interruption is still finalizing: {session.id}")
            if loaded_session.status == SessionStatus.RUNNING:
                existing_interrupt_event = await self._wait_for_active_session_interrupted_event(
                    session.id
                )
                if existing_interrupt_event is not None:
                    request_marker_active = False
                    self._sessions_requesting_interruption.discard(loaded_session.id)
                    yield existing_interrupt_event
                    return
                raise TimeoutError(f"Session interruption is still finalizing: {session.id}")
        except ValueError:
            reloaded_session = await self.session_store.load(loaded_session.id)
            if reloaded_session is None:
                raise KeyError(f"Session not found: {loaded_session.id}") from None
            if reloaded_session.status in _INTERRUPT_REQUESTED_SESSION_STATUSES:
                existing_interrupt_event = await self._wait_for_active_session_interrupted_event(
                    reloaded_session.id
                )
                if existing_interrupt_event is not None:
                    request_marker_active = False
                    self._sessions_requesting_interruption.discard(loaded_session.id)
                    yield existing_interrupt_event
                    return
                if self._has_active_session_tasks(reloaded_session.id):
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
                self._sessions_requesting_interruption.discard(loaded_session.id)
            raise

        session = await self.session_store.update_status(session.id, SessionStatus.INTERRUPTED)
        payload = await self._load_pending_session_interrupt_payload(
            session.id,
            default={
                "reason": request.reason,
                "metadata": request.metadata,
                "interruption_type": _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
            },
        )
        terminal_event_stream: AsyncIterator[Event] | None = None
        try:
            existing_interrupt_event = await self._latest_session_interrupted_event(session.id)
            if existing_interrupt_event is not None:
                await self._clear_pending_session_interrupt(session.id)
                yield existing_interrupt_event
                return
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
            try:
                first_terminal_event = await anext(terminal_event_stream)
            except StopAsyncIteration as exc:
                raise RuntimeError("Session interruption produced no terminal event.") from exc

            await self._clear_pending_session_interrupt(session.id)
            yield first_terminal_event
            async for event in terminal_event_stream:
                yield event
        except Exception:
            if terminal_event_stream is not None:
                with contextlib.suppress(Exception):
                    await _close_async_iterator(terminal_event_stream)
            raise
        finally:
            if request_marker_active:
                self._sessions_requesting_interruption.discard(loaded_session.id)
        return

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
            cost_budget=request.cost_budget,
            retry_policy=request.retry_policy,
            structured_output=request.structured_output,
        )
        start_event_payload_extra = {"dispatch_id": request.dispatch_id}
        if request.task_id is not None:
            start_event_payload_extra["task_id"] = request.task_id
        async for event in self._resume_session(
            request=resume_request,
            task_id=request.task_id,
            start_event_payload_extra=start_event_payload_extra,
        ):
            yield event

    async def create_task(self, request: TaskCreate) -> Task:
        if type(request) is not TaskCreate:
            raise TypeError("Task creation requires a TaskCreate request.")
        if self.task_store is None:
            raise RuntimeError("task_store is required to create tasks.")
        return await self.task_store.create_task(copy_task_create(request))

    async def get_session_usage(self, session_id: str) -> SessionUsageSummary:
        session_id = require_clean_nonblank(session_id, "session_id")
        session = await self.session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}") from None
        usage_event_records = await self._query_all_event_records(
            EventQuery(
                session_id=session_id,
                event_type=EventType.MODEL_COMPLETED,
            )
        )
        tool_event_records = await self._query_all_event_records(
            EventQuery(
                session_id=session_id,
                event_type=EventType.TOOL_CALL_STARTED,
            )
        )
        events = [
            record.event
            for record in sorted(
                [*usage_event_records, *tool_event_records],
                key=lambda record: record.sequence,
            )
        ]
        return session_usage_summary(session_id, events)

    async def _query_all_event_records(self, query: EventQuery) -> list[EventRecord]:
        records: list[EventRecord] = []
        after_sequence = query.after_sequence
        while True:
            page = await self.session_store.query_events(
                EventQuery(
                    session_id=query.session_id,
                    event_type=query.event_type,
                    agent_name=query.agent_name,
                    environment_name=query.environment_name,
                    workflow_name=query.workflow_name,
                    tool_name=query.tool_name,
                    after_sequence=after_sequence,
                    limit=query.limit,
                )
            )
            if not page:
                return records
            records.extend(page)
            if len(page) < query.limit:
                return records
            after_sequence = page[-1].sequence

    async def get_session_cost(
        self,
        session_id: str,
        pricing: PricingCatalog,
        *,
        currency: str = "USD",
    ) -> SessionCostSummary:
        session_id = require_clean_nonblank(session_id, "session_id")
        session = await self.session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}") from None
        events = await self.session_store.load_events(session_id)
        return estimate_session_cost(
            session_id=session_id,
            events=events,
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
        return await self._emit(event)

    async def _resume_session(
        self,
        *,
        request: ResumeRequest,
        task_id: str | None,
        start_event_payload_extra: dict[str, Any],
    ) -> AsyncIterator[Event]:
        loaded_session = await self.session_store.load(request.session_id)
        if loaded_session is None:
            raise KeyError(f"Session not found: {request.session_id}")

        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        registered_provider = self._get_registered_provider(loaded_session.provider_name)
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )
        checkpoint = await self.session_store.load_checkpoint(loaded_session.id)
        if approval_support.pending_approval_from_checkpoint(checkpoint) is not None:
            raise RuntimeError(
                "Session has a pending tool approval. Resolve it with "
                "resolve_tool_approval(...) before resuming with new messages."
            )
        session = await self.session_store.transition_status(
            loaded_session.id,
            from_statuses=_RESUMABLE_SESSION_STATUSES,
            to_status=SessionStatus.RUNNING,
        )
        try:
            if request.model is not None:
                session = await self.session_store.update_model(session.id, request.model)
            transcript = await self.session_store.load_transcript(session.id)
        except Exception as exc:
            await self.session_store.update_status(session.id, SessionStatus.FAILED)
            yield await self._emit(
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
            return
        messages = transcript + request.messages

        async for event in self._run_session(
            session=session,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            messages=messages,
            messages_to_append=request.messages,
            max_steps=request.max_steps,
            limits=request.limits,
            cost_budget=request.cost_budget,
            retry_policy=self._effective_retry_policy(request.retry_policy),
            structured_output=request.structured_output,
            request_metadata=request.metadata,
            task_id=task_id,
            start_event_type=EventType.SESSION_RESUMED,
            start_event_payload={
                "agent_name": registered_agent.spec.name,
                "appended_messages": len(request.messages),
                **copy_json_value(start_event_payload_extra, "start_event_payload_extra"),
            },
        ):
            yield event

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

        registered_provider = self._get_registered_provider(source_session.provider_name)
        agent_name = request.agent_name or source_session.agent_name
        registered_agent = self._get_registered_agent(agent_name)
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
                if current_source.status == SessionStatus.INTERRUPTED and source_checkpoint is None:
                    raise RuntimeError(
                        "Interrupted session cannot be forked because checkpoint state is missing."
                    )
                return approval_support.checkpoint_for_fork(
                    checkpoint=source_checkpoint,
                    agent_name=agent_name,
                    environment_name=environment_name,
                )

        fork_session = Session(
            id=request.session_id or str(uuid4()),
            agent_name=agent_name,
            provider_name=registered_provider.name,
            model=model,
            parent_session_id=source_session.id,
            runtime_name=source_session.runtime_name,
            runtime_version=source_session.runtime_version,
            environment_name=environment_name,
            status=source_session.status,
            metadata=copy_json_value(request.metadata, "metadata"),
        )
        created = await self.session_store.create_fork(
            source_session_id=source_session.id,
            fork=fork_session,
            source_statuses=_FORKABLE_SESSION_STATUSES,
            transcript_cursor=request.transcript_cursor,
            checkpoint_transform=checkpoint_transform,
        )
        yield await self._emit(
            Event(
                type=EventType.SESSION_FORKED,
                session_id=created.id,
                agent_name=registered_agent.spec.name,
                environment_name=_environment_name(registered_environment),
                payload={
                    "source_session_id": source_session.id,
                    "source_status": source_session.status.value,
                    "parent_session_id": created.parent_session_id,
                    "transcript_cursor": request.transcript_cursor,
                    "copy_checkpoint": request.copy_checkpoint,
                    "agent_name": created.agent_name,
                    "provider_name": created.provider_name,
                    "model": created.model,
                    "environment_name": created.environment_name,
                },
            )
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
        _effective_approval_structured_output(
            structured_output=request.structured_output,
            pending_approval=pending_approval,
        )

        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        registered_provider = self._get_registered_provider(loaded_session.provider_name)
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )
        session = await self.session_store.transition_status(
            loaded_session.id,
            from_statuses={SessionStatus.INTERRUPTED},
            to_status=SessionStatus.RUNNING,
        )

        async for event in self._continue_tool_approval_resolution(
            request=request,
            session=session,
            pending_approval=pending_approval,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
        ):
            yield event

    async def _continue_tool_approval_resolution(
        self,
        *,
        request: ToolApprovalRequest,
        session: Session,
        pending_approval: PendingToolApproval,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        emit_resume_event: bool = True,
    ) -> AsyncIterator[Event]:
        environment_name = _environment_name(registered_environment)
        pending_approval_cleared = False
        tool_outcomes: list[runtime_records.ToolCallOutcome] = []
        try:
            transcript = await self.session_store.load_transcript(session.id)
            approval_events = await self.session_store.load_events(session.id)
            approval_support.validate_retry_decision(
                events=approval_events,
                approval=pending_approval,
                decision=request.decision,
            )
            recorded_outcomes = approval_support.recorded_tool_outcomes(
                events=approval_events,
                approval=pending_approval,
            )
            if emit_resume_event:
                yield await self._emit(
                    approval_support.resumed_event(
                        session=session,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        approval=pending_approval,
                        decision=request.decision,
                    )
                )

            if request.decision not in {
                ToolApprovalDecision.APPROVE,
                ToolApprovalDecision.DENY,
            }:
                raise ValueError(f"Unsupported tool approval decision: {request.decision}")

            if request.decision == ToolApprovalDecision.APPROVE:
                run_started_at = time.monotonic()
                limits = copy_run_limits(request.limits)
                cost_budget = copy_cost_budget(request.cost_budget)
                run_baseline = (
                    session_usage_summary(session.id, approval_events)
                    if limits.scope == "run" and has_run_limits(limits)
                    else None
                )
                cost_baseline = (
                    estimate_session_cost(
                        session_id=session.id,
                        events=approval_events,
                        pricing=cost_budget.pricing,
                        currency=cost_budget.currency,
                    )
                    if cost_budget is not None and cost_budget.scope == "run"
                    else None
                )
                recorded_tool_outcomes = list(recorded_outcomes.values())
                pending_tool_calls: list[runtime_records.ToolCallRequest] = []
                executable_pending_tool_calls = 0
                for pending_tool_call in approval_support.pending_round_tool_calls(
                    pending_approval
                ):
                    if pending_tool_call.tool_call_id in recorded_outcomes:
                        continue
                    tool_call = runtime_records.ToolCallRequest(
                        id=pending_tool_call.tool_call_id,
                        name=pending_tool_call.tool_name,
                        arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
                    )
                    pending_tool_calls.append(tool_call)
                    policy_result = approval_support.policy_result_from_pending_tool_call(
                        pending_tool_call
                    )
                    if (
                        policy_result is not None
                        and policy_result.decision == ToolPolicyDecision.DENY
                    ):
                        continue
                    executable_pending_tool_calls += 1
                decision, usage_summary, cost_summary = await self._first_limit_decision(
                    session=session,
                    limits=limits,
                    cost_budget=cost_budget,
                    run_started_at=run_started_at,
                    run_baseline=run_baseline,
                    cost_baseline=cost_baseline,
                    pending_tool_calls=executable_pending_tool_calls,
                )
                if decision is not None:
                    async for event in self._stop_session_for_limit_reached(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        decision=decision,
                        usage_summary=usage_summary,
                        cost_summary=cost_summary,
                        messages=transcript,
                        tool_calls=pending_tool_calls,
                        completed_tool_outcomes=recorded_tool_outcomes,
                        pending_approval_to_clear=pending_approval,
                    ):
                        yield event
                    pending_approval_cleared = True
                    return

            for pending_tool_call in approval_support.pending_round_tool_calls(pending_approval):
                tool_call = runtime_records.ToolCallRequest(
                    id=pending_tool_call.tool_call_id,
                    name=pending_tool_call.tool_name,
                    arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
                )
                policy_result = approval_support.policy_result_from_pending_tool_call(
                    pending_tool_call
                )
                recorded_outcome = recorded_outcomes.get(tool_call.id)
                if recorded_outcome is not None:
                    tool_outcomes.append(recorded_outcome)
                    continue

                if policy_result is not None and policy_result.decision == ToolPolicyDecision.DENY:
                    reason = tool_execution.policy_denial_reason(policy_result)
                    result = tool_execution.blocked_tool_result(policy_result, reason=reason)
                    async for event, outcome in self._emit_tool_call_result_with_hooks(
                        event=Event(
                            type=EventType.TOOL_CALL_BLOCKED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload={
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": tool_call.id,
                                "decision": policy_result.decision.value,
                                "reason": reason,
                                "metadata": policy_result.metadata,
                                "result": result.model_dump(),
                            },
                        ),
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call=tool_call,
                        result=result,
                        task_id=pending_approval.task_id,
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                if (
                    policy_result is not None
                    and policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
                    and request.decision == ToolApprovalDecision.APPROVE
                ):
                    yield await self._emit(
                        Event(
                            type=EventType.TOOL_CALL_APPROVED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload={
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": tool_call.id,
                                "reason": request.reason,
                                "metadata": request.metadata,
                            },
                        )
                    )

                if request.decision == ToolApprovalDecision.DENY:
                    approval_required = (
                        policy_result is not None
                        and policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
                    )
                    result = approval_support.approval_denied_tool_result(
                        request,
                        approval=pending_approval,
                        tool_call=tool_call,
                        approval_required=approval_required,
                    )
                    async for event, outcome in self._emit_tool_call_result_with_hooks(
                        event=Event(
                            type=EventType.TOOL_CALL_APPROVAL_DENIED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload={
                                "approval_id": pending_approval.approval_id,
                                "tool_call_id": tool_call.id,
                                "approval_required": approval_required,
                                "reason": request.reason,
                                "metadata": request.metadata,
                                "result": result.model_dump(),
                            },
                        ),
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call=tool_call,
                        result=result,
                        task_id=pending_approval.task_id,
                    ):
                        yield event
                        if outcome is not None:
                            tool_outcomes.append(outcome)
                    continue

                async for event, outcome in self._execute_tool_call(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    request_metadata=request.metadata,
                    task_id=pending_approval.task_id,
                    check_policy=False,
                    emit_started=True,
                    approval_id=pending_approval.approval_id,
                ):
                    yield event
                    if outcome is not None:
                        tool_outcomes.append(outcome)

            tool_result_messages = transcript_helpers.tool_result_messages(tool_outcomes)
            transcript.extend(tool_result_messages)
            cleared_checkpoint = await self._checkpoint_without_pending_tool_approval(session.id)
            await self.session_store.append_transcript_messages_and_checkpoint(
                session.id,
                tool_result_messages,
                cleared_checkpoint,
            )
            pending_approval_cleared = True
            yield await self._emit(
                approval_support.cleared_event(
                    session=session,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    approval_id=pending_approval.approval_id,
                )
            )

            async for event in self._run_session(
                session=session,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                registered_environment=registered_environment,
                messages=transcript,
                messages_to_append=[],
                max_steps=request.max_steps,
                limits=request.limits,
                cost_budget=request.cost_budget,
                retry_policy=self._effective_retry_policy(request.retry_policy),
                structured_output=_effective_approval_structured_output(
                    structured_output=request.structured_output,
                    pending_approval=pending_approval,
                ),
                request_metadata=request.metadata,
                task_id=pending_approval.task_id,
                start_event_type=None,
                start_event_payload={},
                start_task_on_enter=False,
            ):
                yield event
        except Exception as exc:
            if isinstance(exc, approval_support.ToolApprovalManualRecoveryRequired):
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
                            "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                            "approval": pending_approval.model_dump(),
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "approval_id": pending_approval.approval_id,
                            "tool_call_id": exc.tool_call_id,
                            "tool_name": exc.tool_name,
                            "manual_recovery_required": True,
                        },
                    ),
                    phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                ):
                    yield event
                return

            if not pending_approval_cleared:
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
                            "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                            "approval": pending_approval.model_dump(),
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    ),
                    phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                ):
                    yield event
                return

            task_failure_error: Exception | None = None
            if pending_approval.task_id is not None and self.task_store is not None:
                try:
                    task = await self.task_store.fail_task(
                        pending_approval.task_id,
                        {
                            "message": str(exc),
                            "type": type(exc).__name__,
                            "session_id": session.id,
                            "approval_id": pending_approval.approval_id,
                        },
                    )
                    yield await self._emit(
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
            payload: dict[str, Any] = {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "approval_id": pending_approval.approval_id,
                "tool_call_id": pending_approval.tool_call_id,
            }
            if task_failure_error is not None:
                payload["task_update_error"] = str(task_failure_error)
                payload["task_update_error_type"] = type(task_failure_error).__name__
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
        _effective_approval_structured_output(
            structured_output=request.structured_output,
            pending_approval=pending_approval,
        )

        pending_tool_call = approval_support.pending_tool_call_for_recovery(
            approval=pending_approval,
            tool_call_id=request.tool_call_id,
        )
        registered_agent = self._get_registered_agent(loaded_session.agent_name)
        registered_provider = self._get_registered_provider(loaded_session.provider_name)
        registered_environment = self._get_registered_environment_for_session(
            loaded_session.environment_name
        )
        session = await self.session_store.transition_status(
            loaded_session.id,
            from_statuses={SessionStatus.INTERRUPTED},
            to_status=SessionStatus.RUNNING,
        )
        recovered_result = approval_support.recovered_tool_result(
            request=request,
        )
        event_type = (
            EventType.TOOL_CALL_FAILED
            if recovered_result.is_error
            else EventType.TOOL_CALL_COMPLETED
        )

        try:
            events = await self.session_store.load_events(session.id)
            approval_support.validate_recovery_target(
                events=events,
                approval=pending_approval,
                tool_call_id=request.tool_call_id,
            )
            recovery_events = [
                approval_support.resumed_event(
                    session=session,
                    agent_name=registered_agent.spec.name,
                    environment_name=_environment_name(registered_environment),
                    approval=pending_approval,
                    decision=ToolApprovalDecision.APPROVE,
                ),
                Event(
                    type=event_type,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=_environment_name(registered_environment),
                    tool_name=pending_tool_call.tool_name,
                    payload={
                        "approval_id": pending_approval.approval_id,
                        "tool_call_id": pending_tool_call.tool_call_id,
                        "manual_recovery": True,
                        "reason": request.reason,
                        "metadata": request.metadata,
                        "result": recovered_result.model_dump(),
                    },
                ),
            ]
            emitted_recovery_events = await self._emit_many(session.id, recovery_events)
            for event in emitted_recovery_events:
                yield event
            tool_call = runtime_records.ToolCallRequest(
                id=pending_tool_call.tool_call_id,
                name=pending_tool_call.tool_name,
                arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
            )
            tool_event = emitted_recovery_events[-1]
            async for event in self._run_tool_call_hooks(
                session=session,
                tool_event=tool_event,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=recovered_result,
                task_id=pending_approval.task_id,
            ):
                yield event
        except Exception:
            await self.session_store.update_status(session.id, loaded_session.status)
            raise

        approval_request = ToolApprovalRequest(
            session_id=request.session_id,
            approval_id=request.approval_id,
            decision=ToolApprovalDecision.APPROVE,
            reason=request.reason,
            metadata=request.metadata,
            max_steps=request.max_steps,
            limits=request.limits,
            cost_budget=request.cost_budget,
            retry_policy=request.retry_policy,
            structured_output=request.structured_output,
        )
        async for event in self._continue_tool_approval_resolution(
            request=approval_request,
            session=session,
            pending_approval=pending_approval,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            emit_resume_event=False,
        ):
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
        cost_budget: CostBudget | None,
        retry_policy: RetryPolicy,
        structured_output: StructuredOutputSpec | None,
        request_metadata: dict[str, Any],
        task_id: str | None,
        start_event_type: EventType | None,
        start_event_payload: dict[str, Any],
        start_task_on_enter: bool = True,
    ) -> AsyncIterator[Event]:
        provider = registered_provider.provider
        environment_name = _environment_name(registered_environment)
        task_started = task_id is not None and not start_task_on_enter
        task_finished = False
        current_task = asyncio.current_task()
        active_run: _ActiveSessionRun | None = None
        run_started_at = time.monotonic()
        limits = copy_run_limits(limits)
        cost_budget = copy_cost_budget(cost_budget)
        retry_policy = copy_retry_policy(retry_policy)
        structured_output = copy_structured_output_spec(structured_output)
        structured_output_retries = 0
        run_baseline: SessionUsageSummary | None = None
        cost_baseline: SessionCostSummary | None = None
        if (limits.scope == "run" and has_run_limits(limits)) or (
            cost_budget is not None and cost_budget.scope == "run"
        ):
            baseline_events = await self.session_store.load_events(session.id)
        else:
            baseline_events = []
        if limits.scope == "run" and has_run_limits(limits):
            run_baseline = session_usage_summary(session.id, baseline_events)
        if cost_budget is not None and cost_budget.scope == "run":
            cost_baseline = estimate_session_cost(
                session_id=session.id,
                events=baseline_events,
                pricing=cost_budget.pricing,
                currency=cost_budget.currency,
            )
        if structured_output is not None and STRUCTURED_OUTPUT_TOOL_NAME in registered_agent.tools:
            raise ValueError(
                f"Tool name is reserved for structured output: {STRUCTURED_OUTPUT_TOOL_NAME}"
            )
        if current_task is not None:
            active_run = self._register_active_session_task(
                session.id,
                current_task,
                task_id=task_id,
                task_started=task_started,
                task_finished=task_finished,
            )
        try:
            if start_event_type is not None:
                yield await self._emit(
                    Event(
                        type=start_event_type,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload=start_event_payload,
                    )
                )
            if task_id is not None and start_task_on_enter:
                task = await self._start_task(task_id=task_id, session=session)
                task_started = True
                if active_run is not None:
                    active_run.task_started = True
                yield await self._emit(
                    _task_event(
                        event_type=EventType.TASK_STARTED,
                        task=task,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                    )
                )
            await self.session_store.append_transcript_messages(
                session.id,
                messages_to_append,
            )
            for step in range(1, max_steps + 1):
                await self._raise_if_session_interrupted(session.id)
                decision, usage_summary, cost_summary = await self._first_limit_decision(
                    session=session,
                    limits=limits,
                    cost_budget=cost_budget,
                    run_started_at=run_started_at,
                    run_baseline=run_baseline,
                    cost_baseline=cost_baseline,
                )
                if decision is not None:
                    async for event in self._stop_session_for_limit_reached(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        decision=decision,
                        usage_summary=usage_summary,
                        cost_summary=cost_summary,
                        messages=messages,
                        tool_calls=[],
                        completed_tool_outcomes=[],
                    ):
                        yield event
                    return
                try:
                    (
                        context_messages,
                        checkpoint_update,
                        checkpoint_event_payload,
                        context_compaction_telemetry,
                    ) = await _build_context(
                        context_policy=registered_agent.context_policy,
                        session_store=self.session_store,
                        session=session,
                        agent_spec=_session_agent_spec(
                            registered_agent=registered_agent,
                            session=session,
                        ),
                        messages=messages,
                        step=step,
                        environment_name=environment_name,
                        request_metadata=request_metadata,
                    )
                except ContextBuildError as exc:
                    for telemetry in exc.compaction_telemetry:
                        yield await self._emit(
                            _context_compaction_telemetry_event(
                                telemetry=telemetry,
                                session=session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                            )
                        )
                    raise exc.cause from exc
                for telemetry in context_compaction_telemetry:
                    yield await self._emit(
                        _context_compaction_telemetry_event(
                            telemetry=telemetry,
                            session=session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                        )
                    )
                if checkpoint_event_payload is not None:
                    if checkpoint_update is None:
                        raise RuntimeError(
                            "Context checkpoint event payload requires checkpoint state."
                        )
                    await self.session_store.checkpoint(session.id, checkpoint_update)
                    yield await self._emit(
                        Event(
                            type=EventType.SESSION_CHECKPOINTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            payload=checkpoint_event_payload,
                        )
                    )
                await self._raise_if_session_interrupted(session.id)

                model_tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": deepcopy(tool.schema),
                    }
                    for tool in registered_agent.tools.values()
                ]
                model_messages = context_messages
                if structured_output is not None:
                    model_tools.append(structured_output_tool_spec(structured_output))
                    model_messages = _with_structured_output_tool_instruction(
                        context_messages,
                        structured_output,
                    )

                model_request = ModelRequest(
                    model=session.model,
                    messages=model_messages,
                    tools=model_tools,
                    options={
                        **copy_json_value(
                            registered_agent.spec.provider_options,
                            "provider_options",
                        ),
                        "agent_metadata": deepcopy(registered_agent.spec.metadata),
                        "environment_metadata": (
                            deepcopy(registered_environment.spec.metadata)
                            if registered_environment is not None
                            else {}
                        ),
                        "step": step,
                        "structured_output": (
                            structured_output_spec_payload(structured_output)
                            if structured_output is not None
                            else None
                        ),
                        RESOLVED_FILE_ATTACHMENTS_OPTION: await _resolved_file_attachments(
                            messages=context_messages,
                            session=session,
                            registered_environment=registered_environment,
                            max_file_attachment_bytes=self._max_file_attachment_bytes,
                            max_total_file_attachment_bytes=(self._max_total_file_attachment_bytes),
                            max_file_attachments_per_request=(
                                self._max_file_attachments_per_request
                            ),
                        ),
                    },
                )

                assistant_message: Message | None = None
                tool_calls: list[runtime_records.ToolCallRequest] = []
                async for event, result in self._run_model_step_with_retries(
                    provider=provider,
                    model_request=model_request,
                    session=session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    environment_name=environment_name,
                    step=step,
                    retry_policy=retry_policy,
                ):
                    if event is not None:
                        yield event
                    if result is not None:
                        assistant_message = result.assistant_message
                        tool_calls = result.tool_calls

                if assistant_message is not None:
                    messages.append(assistant_message)
                    await self.session_store.append_transcript_messages(
                        session.id,
                        [assistant_message],
                    )

                decision, usage_summary, cost_summary = await self._first_limit_decision(
                    session=session,
                    limits=limits,
                    cost_budget=cost_budget,
                    run_started_at=run_started_at,
                    run_baseline=run_baseline,
                    cost_baseline=cost_baseline,
                    pending_tool_calls=_user_tool_call_count(tool_calls),
                )
                if decision is not None:
                    async for event in self._stop_session_for_limit_reached(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        decision=decision,
                        usage_summary=usage_summary,
                        cost_summary=cost_summary,
                        messages=messages,
                        tool_calls=tool_calls,
                        completed_tool_outcomes=[],
                    ):
                        yield event
                    return

                if structured_output is not None and _has_structured_output_tool_call(tool_calls):
                    validation = _validate_structured_output_tool_round(
                        tool_calls=tool_calls,
                        spec=structured_output,
                    )
                    structured_tool_outcomes = _structured_output_tool_round_outcomes(
                        tool_calls=tool_calls,
                        spec=structured_output,
                        validation=validation,
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
                        yield await self._emit(
                            _structured_output_event(
                                event_type=EventType.STRUCTURED_OUTPUT_VALIDATED,
                                session=session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                spec=structured_output,
                                validation=validation,
                                step=step,
                                attempt=structured_output_retries + 1,
                            )
                        )
                        break
                    yield await self._emit(
                        _structured_output_event(
                            event_type=EventType.STRUCTURED_OUTPUT_FAILED,
                            session=session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            spec=structured_output,
                            validation=validation,
                            step=step,
                            attempt=structured_output_retries + 1,
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
                    yield await self._emit(
                        _structured_output_event(
                            event_type=EventType.STRUCTURED_OUTPUT_RETRY,
                            session=session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            spec=structured_output,
                            validation=validation,
                            step=step,
                            attempt=structured_output_retries,
                        )
                    )
                    continue

                if not tool_calls:
                    if structured_output is not None:
                        validation = structured_output_tool_required_validation()
                        yield await self._emit(
                            _structured_output_event(
                                event_type=EventType.STRUCTURED_OUTPUT_FAILED,
                                session=session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                spec=structured_output,
                                validation=validation,
                                step=step,
                                attempt=structured_output_retries + 1,
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
                        repair_message = Message.text(
                            "user",
                            structured_output_repair_prompt(
                                spec=structured_output,
                                validation=validation,
                            ),
                        )
                        messages.append(repair_message)
                        await self.session_store.append_transcript_messages(
                            session.id,
                            [repair_message],
                        )
                        yield await self._emit(
                            _structured_output_event(
                                event_type=EventType.STRUCTURED_OUTPUT_RETRY,
                                session=session,
                                registered_agent=registered_agent,
                                environment_name=environment_name,
                                spec=structured_output,
                                validation=validation,
                                step=step,
                                attempt=structured_output_retries,
                            )
                        )
                        continue
                    break

                tool_outcomes: list[runtime_records.ToolCallOutcome] = []
                try:
                    await self._raise_if_session_interrupted(session.id)
                    policy_plan = await self._policy_plan_for_tool_round(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_calls=tool_calls,
                        request_metadata=request_metadata,
                    )
                    await self._raise_if_session_interrupted(session.id)
                except _SessionInterruptedByRequest:
                    async for event in self._close_interrupted_tool_round(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        messages=messages,
                        tool_calls=tool_calls,
                        tool_outcomes=tool_outcomes,
                    ):
                        yield event
                    raise
                except asyncio.CancelledError as exc:
                    if await self._session_interrupt_requested(session.id):
                        _clear_current_task_cancellation()
                        async for event in self._close_interrupted_tool_round(
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            messages=messages,
                            tool_calls=tool_calls,
                            tool_outcomes=tool_outcomes,
                            cancellation_artifacts=_cancellation_artifacts(exc),
                        ):
                            yield event
                    raise

                decision, usage_summary, cost_summary = await self._first_limit_decision(
                    session=session,
                    limits=limits,
                    cost_budget=cost_budget,
                    run_started_at=run_started_at,
                    run_baseline=run_baseline,
                    cost_baseline=cost_baseline,
                    pending_tool_calls=len(tool_calls),
                )
                if decision is not None:
                    async for event in self._stop_session_for_limit_reached(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        decision=decision,
                        usage_summary=usage_summary,
                        cost_summary=cost_summary,
                        messages=messages,
                        tool_calls=tool_calls,
                        completed_tool_outcomes=[],
                    ):
                        yield event
                    return

                if policy_plan.pending_approval is not None:
                    approval_plan = policy_plan.pending_approval
                    try:
                        approval, checkpoint_event = await self._checkpoint_pending_tool_approval(
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            tool_call=approval_plan.call,
                            tool_calls=approval_plan.calls,
                            policy_outcomes=approval_plan.policy_outcomes,
                            task_id=task_id,
                            policy_result=approval_plan.policy_result,
                            structured_output=structured_output,
                        )
                        yield await self._emit(checkpoint_event)
                        yield await self._emit(
                            Event(
                                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                                session_id=session.id,
                                agent_name=registered_agent.spec.name,
                                environment_name=environment_name,
                                tool_name=approval.tool_name,
                                payload={
                                    "approval": approval.model_dump(),
                                },
                            )
                        )
                    except _SessionInterruptedByRequest:
                        await self._clear_pending_tool_approval_for_tool_round(
                            session.id,
                            tool_calls,
                        )
                        async for event in self._close_interrupted_tool_round(
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            messages=messages,
                            tool_calls=tool_calls,
                            tool_outcomes=tool_outcomes,
                        ):
                            yield event
                        raise
                    except asyncio.CancelledError as exc:
                        if await self._session_interrupt_requested(session.id):
                            _clear_current_task_cancellation()
                            await self._clear_pending_tool_approval_for_tool_round(
                                session.id,
                                tool_calls,
                            )
                            async for event in self._close_interrupted_tool_round(
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                messages=messages,
                                tool_calls=tool_calls,
                                tool_outcomes=tool_outcomes,
                                cancellation_artifacts=_cancellation_artifacts(exc),
                            ):
                                yield event
                        raise
                    raise _SessionInterrupted(approval)

                policy_results_by_id = {
                    outcome.call.id: outcome.result for outcome in policy_plan.outcomes
                }
                try:
                    for tool_call in tool_calls:
                        await self._raise_if_session_interrupted(session.id)
                        decision, usage_summary, cost_summary = await self._first_limit_decision(
                            session=session,
                            limits=limits,
                            cost_budget=cost_budget,
                            run_started_at=run_started_at,
                            run_baseline=run_baseline,
                            cost_baseline=cost_baseline,
                            pending_tool_calls=1,
                        )
                        if decision is not None:
                            async for event in self._stop_session_for_limit_reached(
                                session=session,
                                registered_agent=registered_agent,
                                registered_environment=registered_environment,
                                environment_name=environment_name,
                                decision=decision,
                                usage_summary=usage_summary,
                                cost_summary=cost_summary,
                                messages=messages,
                                tool_calls=tool_calls,
                                completed_tool_outcomes=tool_outcomes,
                            ):
                                yield event
                            return
                        async for event, outcome in self._execute_tool_call(
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            tool_call=tool_call,
                            request_metadata=request_metadata,
                            task_id=task_id,
                            policy_result=policy_results_by_id.get(tool_call.id),
                        ):
                            yield event
                            if outcome is not None:
                                tool_outcomes.append(outcome)
                        await self._raise_if_session_interrupted(session.id)
                except _SessionInterruptedByRequest:
                    async for event in self._close_interrupted_tool_round(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        messages=messages,
                        tool_calls=tool_calls,
                        tool_outcomes=tool_outcomes,
                    ):
                        yield event
                    raise
                except asyncio.CancelledError as exc:
                    if await self._session_interrupt_requested(session.id):
                        _clear_current_task_cancellation()
                        async for event in self._close_interrupted_tool_round(
                            session=session,
                            registered_agent=registered_agent,
                            registered_environment=registered_environment,
                            messages=messages,
                            tool_calls=tool_calls,
                            tool_outcomes=tool_outcomes,
                            cancellation_artifacts=_cancellation_artifacts(exc),
                        ):
                            yield event
                    raise

                tool_result_messages = transcript_helpers.tool_result_messages(tool_outcomes)
                messages.extend(tool_result_messages)
                try:
                    await self.session_store.append_transcript_messages(
                        session.id,
                        tool_result_messages,
                    )
                except asyncio.CancelledError:
                    if await self._session_interrupt_requested(session.id):
                        _clear_current_task_cancellation()
                        await self.session_store.append_transcript_messages(
                            session.id,
                            tool_result_messages,
                        )
                    raise
            else:
                raise RuntimeError(f"Maximum model steps exceeded: {max_steps}")

            if task_id is not None:
                await self._raise_if_session_interrupted(session.id)
                task = await self._complete_task(
                    task_id=task_id,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                )
                task_finished = True
                if active_run is not None:
                    active_run.task_finished = True
                yield await self._emit(
                    _task_event(
                        event_type=EventType.TASK_COMPLETED,
                        task=task,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                    )
                )
            session = await self.session_store.update_status(session.id, SessionStatus.COMPLETED)
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
        except _SessionInterrupted as exc:
            session = await self.session_store.update_status(session.id, SessionStatus.INTERRUPTED)
            async for event in self._emit_terminal_event_with_hooks(
                event=Event(
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                        "approval": exc.approval.model_dump(),
                    },
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            ):
                yield event
        except _SessionInterruptedByRequest:
            async for event in self._handle_session_interrupted(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
            ):
                yield event
            return
        except asyncio.CancelledError:
            if await self._session_interrupt_requested(session.id):
                async for event in self._handle_session_interrupted(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                ):
                    yield event
                return
            raise
        except Exception as exc:
            task_failure_error: Exception | None = None
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
                    yield await self._emit(
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
            payload: dict[str, Any] = {
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            if task_failure_error is not None:
                payload["task_update_error"] = str(task_failure_error)
                payload["task_update_error_type"] = type(task_failure_error).__name__
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
            if current_task is not None:
                self._unregister_active_session_task(session.id, current_task)

    async def _start_task(
        self,
        *,
        task_id: str,
        session: Session,
    ) -> Task:
        if self.task_store is None:
            raise RuntimeError("task_store is required when RunRequest.task_id is set.")
        return await self.task_store.start_task(task_id, session_id=session.id)

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

    async def _run_model_step_with_retries(
        self,
        *,
        provider: ModelProvider,
        model_request: ModelRequest,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        retry_policy: RetryPolicy,
    ) -> AsyncIterator[tuple[Event | None, AssistantStepResult | None]]:
        retry_policy = copy_retry_policy(retry_policy)
        attempt = 1
        while True:
            yield (
                await self._emit(
                    Event(
                        type=EventType.MODEL_STARTED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        payload={
                            "model": session.model,
                            "provider": registered_provider.name,
                            "step": step,
                            "attempt": attempt,
                            "max_attempts": retry_policy.max_attempts,
                        },
                        environment_name=environment_name,
                    )
                ),
                None,
            )
            try:
                result: AssistantStepResult | None = None
                async for event, step_result in self._run_model_step_once(
                    provider=provider,
                    model_request=model_request,
                    session=session,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    environment_name=environment_name,
                    step=step,
                    attempt=attempt,
                    max_attempts=retry_policy.max_attempts,
                ):
                    if event is not None:
                        yield event, None
                    if step_result is not None:
                        result = step_result
                if result is None:
                    raise RuntimeError("Model step finished without a result.")
                yield None, result
                return
            except _ModelAttemptFailed as exc:
                decision = retry_decision(
                    policy=retry_policy,
                    attempt=attempt,
                    error=exc.message,
                )
                if decision.reason is not None and not exc.emitted_error_event:
                    yield (
                        await self._emit(
                            Event(
                                type=EventType.MODEL_ERROR,
                                session_id=session.id,
                                agent_name=registered_agent.spec.name,
                                environment_name=environment_name,
                                payload=_retry_attempt_payload(
                                    exc.payload,
                                    step=step,
                                    attempt=attempt,
                                    max_attempts=retry_policy.max_attempts,
                                ),
                            )
                        ),
                        None,
                    )
                if not decision.retry:
                    if exc.cause is not None:
                        raise exc.cause from exc
                    raise RuntimeError(exc.message) from exc
                yield (
                    await self._emit(
                        _model_retry_event(
                            session=session,
                            registered_agent=registered_agent,
                            environment_name=environment_name,
                            registered_provider=registered_provider,
                            step=step,
                            decision=decision,
                            error=exc.message,
                        )
                    ),
                    None,
                )
                await self._sleep_before_retry(session.id, decision)
                attempt += 1

    async def _run_model_step_once(
        self,
        *,
        provider: ModelProvider,
        model_request: ModelRequest,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
    ) -> AsyncIterator[tuple[Event | None, AssistantStepResult | None]]:
        assistant_parts: list[transcript_helpers.AssistantTextPart | ToolCallPart] = []
        tool_calls: list[runtime_records.ToolCallRequest] = []
        provider_state_parts: list[ProviderStatePart] = []
        completed_stream_event: ModelStreamEvent | None = None
        step_result: AssistantStepResult | None = None
        model_completed = False
        try:
            async for raw_stream_event in provider.stream(model_request):
                stream_event = _validate_stream_event(raw_stream_event)
                await self._raise_if_session_interrupted(session.id)
                if model_completed:
                    raise _ModelAttemptFailed(
                        message=(
                            f"Model provider emitted event after completed: {stream_event.type}"
                        ),
                        payload={
                            "error": (
                                f"Model provider emitted event after completed: {stream_event.type}"
                            ),
                            "error_type": "RuntimeError",
                        },
                        emitted_error_event=False,
                        cause=RuntimeError(
                            f"Model provider emitted event after completed: {stream_event.type}"
                        ),
                    )

                if stream_event.type == ModelStreamEventType.TOOL_CALL:
                    tool_call = transcript_helpers.parse_tool_call(stream_event.payload)
                    tool_calls.append(tool_call)
                    assistant_parts.append(transcript_helpers.tool_call_part(tool_call))
                    continue

                if stream_event.type == ModelStreamEventType.TEXT_DELTA:
                    transcript_helpers.append_assistant_text_delta(
                        assistant_parts, stream_event.delta
                    )
                elif stream_event.type == ModelStreamEventType.COMPLETED:
                    model_completed = True
                    completed_stream_event = stream_event
                    provider_state_parts = transcript_helpers.provider_state_parts(
                        stream_event.payload,
                    )
                    assistant_message = transcript_helpers.assistant_message(
                        content_parts=assistant_parts,
                        provider_state_parts=provider_state_parts,
                    )
                    step_result = _assistant_step_result(
                        session_id=session.id,
                        step=step,
                        assistant_message=assistant_message,
                        tool_calls=tool_calls,
                        completion=_stream_event_completion(completed_stream_event),
                    )
                    classification = classify_assistant_step(step_result)
                    event = _model_stream_event_to_runtime_event(
                        stream_event,
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                        provider_name=registered_provider.name,
                        step=step,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        classification=classification.payload(),
                    )
                    yield await self._emit(event), None
                    continue

                event = _model_stream_event_to_runtime_event(
                    stream_event,
                    session=session,
                    registered_agent=registered_agent,
                    environment_name=environment_name,
                    provider_name=registered_provider.name,
                    step=step,
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                emitted_event = await self._emit(event)
                if stream_event.type == ModelStreamEventType.ERROR:
                    yield emitted_event, None
                    raise _ModelAttemptFailed(
                        message=str(stream_event.payload.get("error") or "Model provider error"),
                        payload=copy_json_value(stream_event.payload, "payload"),
                        emitted_error_event=True,
                        cause=RuntimeError(
                            str(stream_event.payload.get("error") or "Model provider error")
                        ),
                    )
                yield emitted_event, None
        except _SessionInterruptedByRequest:
            raise
        except asyncio.CancelledError:
            raise
        except _ModelAttemptFailed:
            raise
        except Exception as exc:
            raise _ModelAttemptFailed(
                message=str(exc),
                payload={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                emitted_error_event=False,
                cause=exc,
            ) from exc

        if not model_completed:
            message = "Model provider stream ended without a completed event."
            raise _ModelAttemptFailed(
                message=message,
                payload={
                    "error": message,
                    "error_type": "RuntimeError",
                },
                emitted_error_event=False,
                cause=RuntimeError(message),
            )
        await self._raise_if_session_interrupted(session.id)
        if completed_stream_event is None:
            raise RuntimeError("Model provider completed without completion metadata.")
        if step_result is None:
            raise RuntimeError("Model provider completed without an assistant step result.")
        yield None, step_result

    async def _sleep_before_retry(
        self,
        session_id: str,
        decision: RetryDecision,
    ) -> None:
        await self._raise_if_session_interrupted(session_id)
        if decision.delay_seconds > 0:
            await asyncio.sleep(decision.delay_seconds)
        await self._raise_if_session_interrupted(session_id)

    async def _first_limit_decision(
        self,
        *,
        session: Session,
        limits: RunLimits,
        cost_budget: CostBudget | None,
        run_started_at: float,
        run_baseline: SessionUsageSummary | None = None,
        cost_baseline: SessionCostSummary | None = None,
        pending_tool_calls: int = 0,
    ) -> tuple[StopDecision | None, SessionUsageSummary, SessionCostSummary | None]:
        if not has_run_limits(limits) and cost_budget is None:
            return None, SessionUsageSummary(session_id=session.id), None
        events = await self.session_store.load_events(session.id)
        usage_summary = session_usage_summary(session.id, events)
        usage_for_limits = usage_summary
        if limits.scope == "run" and run_baseline is not None:
            cur, base = usage_summary.usage, run_baseline.usage
            usage_for_limits = SessionUsageSummary(
                session_id=session.id,
                tool_calls=max(0, usage_summary.tool_calls - run_baseline.tool_calls),
                usage=UsageMetrics(
                    input_tokens=max(0, cur.input_tokens - base.input_tokens),
                    output_tokens=max(0, cur.output_tokens - base.output_tokens),
                    total_tokens=max(0, cur.total_tokens - base.total_tokens),
                ),
            )
        elapsed_seconds = max(0, int(time.monotonic() - run_started_at))
        decision = first_reached_limit(
            limits=limits,
            usage=usage_for_limits,
            elapsed_seconds=elapsed_seconds,
            pending_tool_calls=pending_tool_calls,
        )
        if decision is not None:
            return decision, usage_summary, None

        cost_summary: SessionCostSummary | None = None
        if cost_budget is not None:
            cost_summary = estimate_session_cost(
                session_id=session.id,
                events=events,
                pricing=cost_budget.pricing,
                currency=cost_budget.currency,
            )
            cost_decision = _first_cost_limit_decision(
                session=session,
                cost_budget=cost_budget,
                cost_summary=cost_summary,
                cost_baseline=cost_baseline,
            )
            if cost_decision is not None:
                return cost_decision, usage_summary, cost_summary
        return None, usage_summary, cost_summary

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
    ) -> AsyncIterator[Event]:
        limit_payload = _limit_reached_payload(
            decision=decision,
            usage_summary=usage_summary,
            cost_summary=cost_summary,
        )
        yield await self._emit(
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
    ) -> AsyncIterator[Event]:
        expected_tool_calls = [*tool_calls, *(outcome.call for outcome in completed_tool_outcomes)]
        if await self._tool_round_has_result_messages(session.id, expected_tool_calls):
            if pending_approval_to_clear is not None:
                cleared_checkpoint = await self._checkpoint_without_pending_tool_approval(
                    session.id
                )
                await self.session_store.checkpoint(session.id, cleared_checkpoint)
                yield await self._emit(
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
        )
        for skipped_outcome in skipped_outcomes:
            yield await self._emit(
                _limit_reached_tool_call_event(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call_outcome=skipped_outcome,
                    decision=decision,
                )
            )
        tool_result_messages = transcript_helpers.tool_result_messages(
            [*completed_tool_outcomes, *skipped_outcomes]
        )
        messages.extend(tool_result_messages)
        if pending_approval_to_clear is not None:
            cleared_checkpoint = await self._checkpoint_without_pending_tool_approval(session.id)
            await self.session_store.append_transcript_messages_and_checkpoint(
                session.id,
                tool_result_messages,
                cleared_checkpoint,
            )
            yield await self._emit(
                approval_support.cleared_event(
                    session=session,
                    agent_name=registered_agent.spec.name,
                    environment_name=_environment_name(registered_environment),
                    approval_id=pending_approval_to_clear.approval_id,
                )
            )
        else:
            await self.session_store.append_transcript_messages(session.id, tool_result_messages)

    async def _policy_plan_for_tool_round(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_calls: list[runtime_records.ToolCallRequest],
        request_metadata: dict[str, Any],
    ) -> runtime_records.ToolRoundPolicyPlan:
        policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] = []
        approval_policy_result: ToolPolicyResult | None = None
        approval_tool_call: runtime_records.ToolCallRequest | None = None
        for tool_call in tool_calls:
            if tool_call.name not in registered_agent.tools:
                policy_outcomes.append(
                    runtime_records.ToolCallPolicyOutcome(call=tool_call, result=None)
                )
                continue

            policy_result = await self._authorize_tool_call(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                request_metadata=request_metadata,
            )
            policy_outcomes.append(
                runtime_records.ToolCallPolicyOutcome(call=tool_call, result=policy_result)
            )
            if (
                approval_policy_result is None
                and policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
            ):
                approval_policy_result = policy_result
                approval_tool_call = tool_call

        if approval_policy_result is None or approval_tool_call is None:
            return runtime_records.ToolRoundPolicyPlan(
                outcomes=policy_outcomes, pending_approval=None
            )

        return runtime_records.ToolRoundPolicyPlan(
            outcomes=policy_outcomes,
            pending_approval=runtime_records.PendingToolApprovalPlan(
                call=approval_tool_call,
                calls=[outcome.call for outcome in policy_outcomes],
                policy_outcomes=policy_outcomes,
                policy_result=approval_policy_result,
            ),
        )

    async def _authorize_tool_call(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        request_metadata: dict[str, Any],
    ) -> ToolPolicyResult:
        policy_result = await registered_agent.tool_policy.authorize(
            ToolPolicyRequest(
                session=session.model_copy(deep=True),
                agent=_validate_agent_spec(registered_agent.spec),
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                arguments=tool_call.arguments,
                environment_name=_environment_name(registered_environment),
                workspace_id=_workspace_id(registered_environment),
                metadata=request_metadata,
            )
        )
        return tool_execution.validate_tool_policy_result(policy_result)

    async def _execute_tool_call(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        request_metadata: dict[str, Any],
        task_id: str | None,
        check_policy: bool = True,
        emit_started: bool = True,
        policy_result: ToolPolicyResult | None = None,
        approval_id: str | None = None,
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        environment_name = _environment_name(registered_environment)
        if emit_started:
            payload: dict[str, Any] = {
                "tool_call_id": tool_call.id,
                "arguments": deepcopy(tool_call.arguments),
            }
            if approval_id is not None:
                payload["approval_id"] = approval_id
            yield (
                await self._emit(
                    Event(
                        type=EventType.TOOL_CALL_STARTED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        tool_name=tool_call.name,
                        payload=payload,
                    )
                ),
                None,
            )

        registered_tool = registered_agent.tools.get(tool_call.name)
        if registered_tool is None:
            result = ToolResult(
                content=f"Tool not registered: {tool_call.name}",
                is_error=True,
            )
            payload = {
                "tool_call_id": tool_call.id,
                "result": result.model_dump(),
            }
            if approval_id is not None:
                payload["approval_id"] = approval_id
            async for event in self._emit_tool_call_result_with_hooks(
                event=Event(
                    type=EventType.TOOL_CALL_FAILED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=tool_call.name,
                    payload=payload,
                ),
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                result=result,
                task_id=task_id,
            ):
                yield event
            return

        if check_policy:
            if policy_result is None:
                resolved_policy_result = await self._authorize_tool_call(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    request_metadata=request_metadata,
                )
            else:
                resolved_policy_result = tool_execution.validate_tool_policy_result(policy_result)
            if resolved_policy_result.decision == ToolPolicyDecision.DENY:
                reason = tool_execution.policy_denial_reason(resolved_policy_result)
                result = tool_execution.blocked_tool_result(resolved_policy_result, reason=reason)
                payload = {
                    "tool_call_id": tool_call.id,
                    "decision": resolved_policy_result.decision.value,
                    "reason": reason,
                    "metadata": resolved_policy_result.metadata,
                    "result": result.model_dump(),
                }
                if approval_id is not None:
                    payload["approval_id"] = approval_id
                async for event in self._emit_tool_call_result_with_hooks(
                    event=Event(
                        type=EventType.TOOL_CALL_BLOCKED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        tool_name=tool_call.name,
                        payload=payload,
                    ),
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    result=result,
                    task_id=task_id,
                ):
                    yield event
                return
            if resolved_policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL:
                approval, checkpoint_event = await self._checkpoint_pending_tool_approval(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    tool_calls=[tool_call],
                    policy_outcomes=None,
                    task_id=task_id,
                    policy_result=resolved_policy_result,
                    structured_output=None,
                )
                yield (await self._emit(checkpoint_event), None)
                yield (
                    await self._emit(
                        Event(
                            type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload={
                                "approval": approval.model_dump(),
                            },
                        )
                    ),
                    None,
                )
                raise _SessionInterrupted(approval)
            if resolved_policy_result.decision != ToolPolicyDecision.ALLOW:
                raise ValueError(
                    f"Unsupported tool policy decision: {resolved_policy_result.decision}"
                )

        result = await tool_execution.run_tool(
            tool=registered_tool.tool,
            ctx=ToolContext(
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                workspace_id=_workspace_id(registered_environment),
                artifact_store_id=_artifact_store_id(registered_environment),
                workspace=_workspace(registered_environment),
                artifact_store=_artifact_store(registered_environment),
                runner=_runner(registered_environment),
                vault=_vault(registered_environment),
                mcp_servers=_mcp_servers(registered_environment),
                metadata=tool_execution.context_metadata(
                    tool_call_id=tool_call.id,
                    approval_id=approval_id,
                ),
            ),
            arguments=deepcopy(tool_call.arguments),
        )
        if await self._session_is_interrupting(session.id):
            raise _SessionInterruptedByRequest(session.id)
        event_type = (
            EventType.TOOL_CALL_FAILED if result.is_error else EventType.TOOL_CALL_COMPLETED
        )
        payload = {
            "tool_call_id": tool_call.id,
            "result": result.model_dump(),
        }
        if approval_id is not None:
            payload["approval_id"] = approval_id
        async for event in self._emit_tool_call_result_with_hooks(
            event=Event(
                type=event_type,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                tool_name=tool_call.name,
                payload=payload,
            ),
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            tool_call=tool_call,
            result=result,
            task_id=task_id,
        ):
            yield event

    async def _checkpoint_pending_tool_approval(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        tool_calls: list[runtime_records.ToolCallRequest],
        policy_outcomes: list[runtime_records.ToolCallPolicyOutcome] | None,
        task_id: str | None,
        policy_result: ToolPolicyResult,
        structured_output: StructuredOutputSpec | None,
    ) -> tuple[PendingToolApproval, Event]:
        checkpoint = await self.session_store.load_checkpoint(session.id)
        checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        if approval_support.pending_approval_from_checkpoint(checkpoint) is not None:
            raise RuntimeError("Session already has a pending tool approval.")

        approval = PendingToolApproval(
            approval_id=str(uuid4()),
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            arguments=copy_json_value(tool_call.arguments, "arguments"),
            agent_name=registered_agent.spec.name,
            environment_name=_environment_name(registered_environment),
            workspace_id=_workspace_id(registered_environment),
            task_id=task_id,
            reason=policy_result.reason,
            metadata=copy_json_value(policy_result.metadata, "metadata"),
            tool_calls=approval_support.pending_tool_call_approvals(
                tool_calls=tool_calls,
                policy_outcomes=policy_outcomes,
            ),
            structured_output=copy_structured_output_spec(structured_output),
        )
        checkpoint[approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY] = approval.model_dump()
        await self.session_store.checkpoint(session.id, checkpoint)
        return (
            approval,
            Event(
                type=EventType.SESSION_CHECKPOINTED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=_environment_name(registered_environment),
                payload={
                    "checkpoint": approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY,
                    "approval_id": approval.approval_id,
                    "tool_call_id": approval.tool_call_id,
                },
            ),
        )

    async def _checkpoint_without_pending_tool_approval(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        checkpoint = await self.session_store.load_checkpoint(session_id)
        checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        checkpoint.pop(approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY, None)
        return checkpoint

    async def _clear_pending_tool_approval_for_tool_round(
        self,
        session_id: str,
        tool_calls: list[runtime_records.ToolCallRequest],
    ) -> None:
        expected_ids = {tool_call.id for tool_call in tool_calls}
        if not expected_ids:
            return
        checkpoint = await self.session_store.load_checkpoint(session_id)
        if checkpoint is None:
            return
        copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
        pending_approval = approval_support.pending_approval_from_checkpoint(copied_checkpoint)
        if pending_approval is None or pending_approval.tool_call_id not in expected_ids:
            return
        copied_checkpoint.pop(approval_support.PENDING_TOOL_APPROVAL_CHECKPOINT_KEY, None)
        await self.session_store.checkpoint(session_id, copied_checkpoint)

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
        checkpoint = await self.session_store.load_checkpoint(session_id)
        if checkpoint is None:
            return
        copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
        copied_checkpoint.pop(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY, None)
        await self.session_store.checkpoint(session_id, copied_checkpoint)

    async def _handle_session_interrupted(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
    ) -> AsyncIterator[Event]:
        _clear_current_task_cancellation()
        current_task = asyncio.current_task()
        if current_task is not None:
            self._unregister_active_session_task(session.id, current_task)
        self._sessions_emitting_interrupted.add(session.id)
        try:
            loaded_interrupted = await self.session_store.load(session.id)
            if loaded_interrupted is None:
                raise KeyError(f"Session not found: {session.id}") from None
            if loaded_interrupted.status != SessionStatus.INTERRUPTED:
                loaded_interrupted = await self.session_store.update_status(
                    session.id,
                    SessionStatus.INTERRUPTED,
                )
            existing_interrupt_event = await self._wait_for_session_interrupted_event(session.id)
            if existing_interrupt_event is not None:
                await self._clear_pending_session_interrupt(session.id)
                yield existing_interrupt_event
                return
            payload = await self._load_pending_session_interrupt_payload(session.id, default={})
            payload.setdefault("interruption_type", _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED)
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
            try:
                first_terminal_event = await anext(terminal_event_stream)
            except StopAsyncIteration as exc:
                raise RuntimeError("Session interruption produced no terminal event.") from exc

            await self._clear_pending_session_interrupt(session.id)
            yield first_terminal_event
            async for event in terminal_event_stream:
                yield event
        finally:
            self._sessions_emitting_interrupted.discard(session.id)

    async def _close_interrupted_tool_round(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        tool_calls: list[runtime_records.ToolCallRequest],
        tool_outcomes: list[runtime_records.ToolCallOutcome],
        cancellation_artifacts: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        if await self._tool_round_has_result_messages(session.id, tool_calls):
            return
        terminal_event_exists = await self._latest_session_interrupted_event(session.id) is not None
        interrupted_results = _interrupted_tool_round_results(
            tool_calls=tool_calls,
            completed_outcomes=tool_outcomes,
            cancellation_artifacts=cancellation_artifacts,
        )
        if not interrupted_results and not tool_outcomes:
            return
        if not terminal_event_exists:
            for interrupted_result in interrupted_results:
                yield await self._emit(
                    _interrupted_tool_call_event(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call_outcome=interrupted_result,
                    )
                )
        tool_outcomes.extend(interrupted_results)
        interrupted_messages = transcript_helpers.tool_result_messages(tool_outcomes)
        messages.extend(interrupted_messages)
        await self.session_store.append_transcript_messages(
            session.id,
            interrupted_messages,
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

    async def _raise_if_session_interrupted(self, session_id: str) -> None:
        session = await self.session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        if session.status in _INTERRUPT_REQUESTED_SESSION_STATUSES:
            raise _SessionInterruptedByRequest(session_id)

    async def _session_interrupt_requested(self, session_id: str) -> bool:
        session = await self.session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return session.status in _INTERRUPT_REQUESTED_SESSION_STATUSES

    async def _session_is_interrupting(self, session_id: str) -> bool:
        session = await self.session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return session.status == SessionStatus.INTERRUPTING

    async def _latest_session_interrupted_event(self, session_id: str) -> Event | None:
        events = await self.session_store.load_events(session_id)
        for event in reversed(events):
            if event.type == EventType.SESSION_INTERRUPTED:
                return event.model_copy(deep=True)
        return None

    async def _wait_for_session_interrupted_event(self, session_id: str) -> Event | None:
        for attempt in range(_INTERRUPTED_EVENT_WAIT_ATTEMPTS):
            existing_event = await self._latest_session_interrupted_event(session_id)
            if existing_event is not None:
                return existing_event

            session = await self.session_store.load(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.status != SessionStatus.INTERRUPTED:
                return None
            if attempt < _INTERRUPTED_EVENT_WAIT_ATTEMPTS - 1:
                await asyncio.sleep(_INTERRUPTED_EVENT_WAIT_INTERVAL_S)

        return None

    async def _wait_for_active_session_interrupted_event(self, session_id: str) -> Event | None:
        for attempt in range(_ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS):
            existing_event = await self._latest_session_interrupted_event(session_id)
            if existing_event is not None:
                return existing_event
            if (
                not self._has_active_session_tasks(session_id)
                and not self._is_session_emitting_interrupted(session_id)
                and not self._is_session_interruption_request_active(session_id)
            ):
                return None
            if attempt < _ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS - 1:
                await asyncio.sleep(_ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S)
        return None

    def _register_active_session_task(
        self,
        session_id: str,
        task: asyncio.Task[Any],
        *,
        task_id: str | None,
        task_started: bool,
        task_finished: bool,
    ) -> _ActiveSessionRun:
        session_id = require_clean_nonblank(session_id, "session_id")
        active_run = _ActiveSessionRun(
            runtime_task=task,
            task_id=task_id,
            task_started=task_started,
            task_finished=task_finished,
        )
        self._active_session_runs.setdefault(session_id, {})[task] = active_run
        return active_run

    def _unregister_active_session_task(
        self,
        session_id: str,
        task: asyncio.Task[Any],
    ) -> None:
        active_runs = self._active_session_runs.get(session_id)
        if active_runs is None:
            return
        active_runs.pop(task, None)
        if not active_runs:
            self._active_session_runs.pop(session_id, None)

    def _has_active_session_tasks(self, session_id: str) -> bool:
        return any(
            not active_run.runtime_task.done()
            for active_run in self._active_session_run_records(session_id)
        )

    def _is_session_emitting_interrupted(self, session_id: str) -> bool:
        return session_id in self._sessions_emitting_interrupted

    def _is_session_interruption_request_active(self, session_id: str) -> bool:
        return session_id in self._sessions_requesting_interruption

    def _interrupt_active_session_runs(self, session_id: str) -> bool:
        current_task = asyncio.current_task()
        signalled = False
        for active_run in self._active_session_run_records(session_id):
            task = active_run.runtime_task
            if task is current_task or task.done():
                continue
            task.cancel()
            signalled = True
        return signalled

    def _active_session_run_records(self, session_id: str) -> tuple[_ActiveSessionRun, ...]:
        return tuple(self._active_session_runs.get(session_id, {}).values())

    async def _emit(self, event: Event) -> Event:
        await self.session_store.append_event(event.session_id, event)
        for sink in self._event_sinks:
            try:
                await sink.emit(event.model_copy(deep=True))
            except Exception as exc:
                await self.session_store.append_event(
                    event.session_id,
                    Event(
                        type=EventType.RUNTIME_SINK_FAILED,
                        session_id=event.session_id,
                        agent_name=event.agent_name,
                        environment_name=event.environment_name,
                        payload={
                            "sink": type(sink).__name__,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "event_id": event.id,
                            "event_type": str(event.type),
                        },
                    ),
                )
        return event

    async def _emit_terminal_event_with_hooks(
        self,
        *,
        event: Event,
        phase: RuntimeHookPhase,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> AsyncIterator[Event]:
        terminal_event = await self._emit(event)
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

    async def _emit_tool_call_result_with_hooks(
        self,
        *,
        event: Event,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        result: ToolResult,
        task_id: str | None,
    ) -> AsyncIterator[tuple[Event, runtime_records.ToolCallOutcome | None]]:
        tool_event = await self._emit(event)
        outcome = runtime_records.ToolCallOutcome(call=tool_call, result=result)
        yield tool_event, outcome
        async for hook_event in self._run_tool_call_hooks(
            session=session,
            tool_event=tool_event,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            tool_call=tool_call,
            result=result,
            task_id=task_id,
        ):
            yield hook_event, None

    async def _run_tool_call_hooks(
        self,
        *,
        session: Session,
        tool_event: Event,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        result: ToolResult,
        task_id: str | None,
    ) -> AsyncIterator[Event]:
        async for hook_event in self._run_scoped_tool_call_hooks(
            session=session,
            tool_event=tool_event,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            tool_call=tool_call,
            result=result,
            task_id=task_id,
            hooks=self._runtime_hooks,
            scope="app",
        ):
            yield hook_event
        async for hook_event in self._run_scoped_tool_call_hooks(
            session=session,
            tool_event=tool_event,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            tool_call=tool_call,
            result=result,
            task_id=task_id,
            hooks=registered_agent.runtime_hooks,
            scope="agent",
        ):
            yield hook_event

    async def _run_scoped_tool_call_hooks(
        self,
        *,
        session: Session,
        tool_event: Event,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        result: ToolResult,
        task_id: str | None,
        hooks: tuple[RuntimeHook, ...],
        scope: str,
    ) -> AsyncIterator[Event]:
        for hook in hooks:
            if not _runtime_hook_supports_phase(
                hook=hook,
                phase=RuntimeHookPhase.AFTER_TOOL_CALL,
            ):
                continue
            hook_name = require_clean_nonblank(hook.name, "runtime_hook.name")
            yield await self._emit(
                _runtime_hook_event(
                    event_type=EventType.HOOK_STARTED,
                    hook_name=hook_name,
                    scope=scope,
                    phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    terminal_event=tool_event,
                    payload={
                        "tool_name": tool_call.name,
                        "tool_call_id": tool_call.id,
                    },
                )
            )
            context = ToolCallHookContext(
                runtime=self,
                hook_name=hook_name,
                phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                session=session,
                tool_event=tool_event,
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                arguments=tool_call.arguments,
                result=result,
                task_id=task_id,
            )
            try:
                await hook.after_tool_call(context)
            except Exception as exc:
                yield await self._emit(
                    _runtime_hook_event(
                        event_type=EventType.HOOK_FAILED,
                        hook_name=hook_name,
                        scope=scope,
                        phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        terminal_event=tool_event,
                        payload={
                            "tool_name": tool_call.name,
                            "tool_call_id": tool_call.id,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "actions": context.actions,
                        },
                    )
                )
                continue
            yield await self._emit(
                _runtime_hook_event(
                    event_type=EventType.HOOK_COMPLETED,
                    hook_name=hook_name,
                    scope=scope,
                    phase=RuntimeHookPhase.AFTER_TOOL_CALL,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    terminal_event=tool_event,
                    payload={
                        "tool_name": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "actions": context.actions,
                    },
                )
            )

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
            yield await self._emit(
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
                yield await self._emit(
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
            yield await self._emit(
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

    async def _emit_many(self, session_id: str, events: list[Event]) -> list[Event]:
        if type(events) is not list:
            raise TypeError("Runtime events must be a list.")
        copied_events: list[Event] = []
        for event in events:
            if type(event) is not Event:
                raise TypeError("Runtime events must be Event instances.")
            if event.session_id != session_id:
                raise ValueError("Event session_id does not match target session.")
            copied_events.append(event.model_copy(deep=True))

        await self.session_store.append_events(session_id, copied_events)
        for event in copied_events:
            for sink in self._event_sinks:
                try:
                    await sink.emit(event.model_copy(deep=True))
                except Exception as exc:
                    await self.session_store.append_event(
                        event.session_id,
                        Event(
                            type=EventType.RUNTIME_SINK_FAILED,
                            session_id=event.session_id,
                            agent_name=event.agent_name,
                            environment_name=event.environment_name,
                            payload={
                                "sink": type(sink).__name__,
                                "error": str(exc),
                                "error_type": type(exc).__name__,
                                "event_id": event.id,
                                "event_type": str(event.type),
                            },
                        ),
                    )
        return copied_events


def _copy_registered_tool(tool: runtime_records.RegisteredTool) -> runtime_records.RegisteredTool:
    return runtime_records.RegisteredTool(
        name=tool.name,
        description=tool.description,
        schema=deepcopy(tool.schema),
        tool=tool.tool,
    )


def _validate_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


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
    )
    return runtime_records.RegisteredTool(
        name=validated_spec.name,
        description=validated_spec.description,
        schema=validated_spec.input_schema,
        tool=tool,
    )


def _validate_agent_spec(spec: AgentSpec) -> AgentSpec:
    if type(spec) is not AgentSpec:
        raise TypeError("Agent registration requires an AgentSpec.")
    return AgentSpec(
        name=spec.name,
        model=spec.model,
        system_prompt=spec.system_prompt,
        metadata=copy_json_value(spec.metadata, "metadata"),
        provider_options=copy_json_value(spec.provider_options, "provider_options"),
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


def _effective_approval_structured_output(
    *,
    structured_output: StructuredOutputSpec | None,
    pending_approval: PendingToolApproval,
) -> StructuredOutputSpec | None:
    if type(pending_approval) is not PendingToolApproval:
        raise TypeError("Pending approval must be a PendingToolApproval.")
    if structured_output is None:
        return copy_structured_output_spec(pending_approval.structured_output)
    if pending_approval.structured_output is None:
        return copy_structured_output_spec(structured_output)
    if not _structured_output_specs_equal(
        structured_output,
        pending_approval.structured_output,
    ):
        raise ValueError("Tool approval structured_output does not match the pending run contract.")
    return copy_structured_output_spec(pending_approval.structured_output)


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


def _session_agent_spec(
    *,
    registered_agent: runtime_records.RegisteredAgentState,
    session: Session,
) -> AgentSpec:
    return AgentSpec(
        name=registered_agent.spec.name,
        model=session.model,
        system_prompt=registered_agent.spec.system_prompt,
        metadata=copy_json_value(registered_agent.spec.metadata, "metadata"),
        provider_options=copy_json_value(
            registered_agent.spec.provider_options,
            "provider_options",
        ),
    )


async def _build_context(
    *,
    context_policy: ContextPolicy,
    session_store: SessionStore,
    session: Session,
    agent_spec: AgentSpec,
    messages: list[Message],
    step: int,
    environment_name: str | None,
    request_metadata: dict[str, Any],
) -> tuple[
    list[Message],
    dict[str, Any] | None,
    dict[str, Any] | None,
    list[ContextCompactionTelemetry],
]:
    request = ContextRequest(
        session=session.model_copy(deep=True),
        agent=agent_spec.model_copy(deep=True),
        messages=[message.model_copy(deep=True) for message in messages],
        step=step,
        environment_name=environment_name,
        metadata=copy_json_value(request_metadata, "metadata"),
    )
    if isinstance(context_policy, RuntimeManagedContextPolicy):
        checkpoint = await session_store.load_checkpoint(session.id)
        result = await context_policy.build_with_checkpoint(
            request,
            checkpoint=checkpoint,
        )
        context_messages = copy_context_messages(result.messages)
        return (
            context_messages,
            copy_json_value(result.checkpoint, "checkpoint"),
            result.checkpoint_event_payload,
            [telemetry.model_copy(deep=True) for telemetry in result.compaction_telemetry],
        )

    result = await context_policy.build(request)
    return copy_context_messages(result), None, None, []


def _with_environment_name(request: RunRequest, environment_name: str) -> RunRequest:
    return RunRequest(
        agent_name=request.agent_name,
        messages=[message.model_copy(deep=True) for message in request.messages],
        session_id=request.session_id,
        task_id=request.task_id,
        environment_name=environment_name,
        metadata=copy_json_value(request.metadata, "metadata"),
        max_steps=request.max_steps,
        limits=copy_run_limits(request.limits),
        cost_budget=copy_cost_budget(request.cost_budget),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
        structured_output=copy_structured_output_spec(request.structured_output),
    )


def _environment_name(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None:
        return None
    return registered_environment.spec.name


def _context_compaction_telemetry_event(
    *,
    telemetry: ContextCompactionTelemetry,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
) -> Event:
    if type(telemetry) is not ContextCompactionTelemetry:
        raise TypeError(
            "Context compaction telemetry must be ContextCompactionTelemetry instances."
        )
    return Event(
        type=telemetry.event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=copy_json_value(telemetry.payload, "payload"),
    )


def _model_retry_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    registered_provider: runtime_records.RegisteredProvider,
    step: int,
    decision: RetryDecision,
    error: str,
) -> Event:
    return Event(
        type=EventType.MODEL_RETRY,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=retry_event_payload(
            decision=decision,
            provider_name=registered_provider.name,
            model=session.model,
            step=step,
            error=error,
        ),
    )


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


def _runtime_hook_supports_phase(
    *,
    hook: RuntimeHook,
    phase: RuntimeHookPhase,
) -> bool:
    method_name = _runtime_hook_method_name(phase)
    hook_method = getattr(type(hook), method_name)
    default_method = getattr(RuntimeHook, method_name)
    return hook_method is not default_method


def _runtime_hook_method_name(phase: RuntimeHookPhase) -> str:
    if phase == RuntimeHookPhase.AFTER_SESSION_COMPLETED:
        return "after_session_completed"
    if phase == RuntimeHookPhase.AFTER_SESSION_FAILED:
        return "after_session_failed"
    if phase == RuntimeHookPhase.AFTER_SESSION_INTERRUPTED:
        return "after_session_interrupted"
    if phase == RuntimeHookPhase.AFTER_TOOL_CALL:
        return "after_tool_call"
    raise ValueError(f"Unsupported runtime hook phase: {phase}")


def _clear_current_task_cancellation() -> None:
    current_task = asyncio.current_task()
    if current_task is None:
        return
    while current_task.cancelling():
        current_task.uncancel()


def _checkpoint_with_pending_session_interrupt(
    payload: dict[str, Any],
):
    copied_payload = copy_json_value(payload, "interrupt_payload")

    def transform(_session: Session, checkpoint: dict[str, Any] | None) -> dict[str, Any]:
        copied_checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        copied_checkpoint[_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY] = copy_json_value(
            copied_payload,
            "interrupt_payload",
        )
        return copied_checkpoint

    return transform


def _interrupted_tool_round_results(
    *,
    tool_calls: list[runtime_records.ToolCallRequest],
    completed_outcomes: list[runtime_records.ToolCallOutcome],
    cancellation_artifacts: list[dict[str, Any]] | None = None,
) -> list[runtime_records.ToolCallOutcome]:
    completed_ids = {outcome.call.id for outcome in completed_outcomes}
    artifacts_for_interrupted_tool = (
        [] if cancellation_artifacts is None else cancellation_artifacts
    )
    interrupted_outcomes: list[runtime_records.ToolCallOutcome] = []
    for tool_call in tool_calls:
        if tool_call.id in completed_ids:
            continue
        result_artifacts = artifacts_for_interrupted_tool
        artifacts_for_interrupted_tool = []
        interrupted_outcomes.append(
            runtime_records.ToolCallOutcome(
                call=tool_call,
                result=ToolResult(
                    content="Tool call interrupted before completion.",
                    structured={
                        "interrupted": True,
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                    },
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


def _first_cost_limit_decision(
    *,
    session: Session,
    cost_budget: CostBudget,
    cost_summary: SessionCostSummary,
    cost_baseline: SessionCostSummary | None,
) -> StopDecision | None:
    if type(session) is not Session:
        raise TypeError("session must be a Session instance.")
    if type(cost_budget) is not CostBudget:
        raise TypeError("cost_budget must be a CostBudget instance.")
    if type(cost_summary) is not SessionCostSummary:
        raise TypeError("cost_summary must be a SessionCostSummary.")
    if cost_baseline is not None and type(cost_baseline) is not SessionCostSummary:
        raise TypeError("cost_baseline must be a SessionCostSummary.")

    actual_cost = cost_summary.total_cost
    unpriced_model_steps = cost_summary.unpriced_model_steps
    if cost_budget.scope == "run" and cost_baseline is not None:
        actual_cost = max(cost_summary.total_cost - cost_baseline.total_cost, Decimal("0"))
        unpriced_model_steps = max(
            cost_summary.unpriced_model_steps - cost_baseline.unpriced_model_steps,
            0,
        )

    if unpriced_model_steps > 0 and not cost_budget.allow_unpriced:
        return StopDecision(
            limit=StopLimit.ESTIMATED_COST,
            maximum=cost_budget.max_estimated_cost,
            actual=actual_cost,
            message=(
                "Estimated cost budget cannot be verified because "
                f"{unpriced_model_steps} model step(s) have no matching pricing."
            ),
        )
    preflight_error = _cost_budget_preflight_error(session=session, cost_budget=cost_budget)
    if preflight_error is not None:
        return StopDecision(
            limit=StopLimit.ESTIMATED_COST,
            maximum=cost_budget.max_estimated_cost,
            actual=actual_cost,
            message=preflight_error,
        )
    if actual_cost >= cost_budget.max_estimated_cost:
        return StopDecision(
            limit=StopLimit.ESTIMATED_COST,
            maximum=cost_budget.max_estimated_cost,
            actual=actual_cost,
            message=(
                "Estimated cost budget reached: "
                f"{actual_cost} >= {cost_budget.max_estimated_cost} {cost_budget.currency}."
            ),
        )
    return None


def _cost_budget_preflight_error(*, session: Session, cost_budget: CostBudget) -> str | None:
    if cost_budget.allow_unpriced:
        return None
    price = cost_budget.pricing.match_price(
        provider_name=session.provider_name,
        model=session.model,
    )
    if price is None:
        return (
            "Estimated cost budget cannot be verified because "
            f"{session.provider_name}/{session.model} has no matching pricing."
        )
    if price.currency.upper() != cost_budget.currency.upper():
        return (
            "Estimated cost budget cannot be verified because "
            f"{session.provider_name}/{session.model} pricing currency {price.currency} "
            f"does not match requested {cost_budget.currency}."
        )
    return None


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
) -> list[runtime_records.ToolCallOutcome]:
    outcomes: list[runtime_records.ToolCallOutcome] = []
    for tool_call in tool_calls:
        outcomes.append(
            runtime_records.ToolCallOutcome(
                call=tool_call,
                result=ToolResult(
                    content="Tool call skipped because a run limit was reached.",
                    structured={
                        "skipped": True,
                        "reason": "limit_reached",
                        "limit": decision.limit.value,
                        "maximum": _limit_value_for_payload(decision.maximum),
                        "actual": _limit_value_for_payload(decision.actual),
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                    },
                    is_error=True,
                ),
            )
        )
    return outcomes


def _cancellation_artifacts(exc: asyncio.CancelledError) -> list[dict[str, Any]]:
    if isinstance(exc, RunnerCancelledError):
        return copy_json_value(exc.artifacts, "artifacts")
    return []


def _interrupted_tool_call_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    tool_call_outcome: runtime_records.ToolCallOutcome,
) -> Event:
    return Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        tool_name=tool_call_outcome.call.name,
        payload={
            "tool_call_id": tool_call_outcome.call.id,
            "result": tool_call_outcome.result.model_dump(),
        },
    )


def _limit_reached_tool_call_event(
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    tool_call_outcome: runtime_records.ToolCallOutcome,
    decision: StopDecision,
) -> Event:
    return Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        tool_name=tool_call_outcome.call.name,
        payload={
            "tool_call_id": tool_call_outcome.call.id,
            "reason": "limit_reached",
            "limit": decision.limit.value,
            "result": tool_call_outcome.result.model_dump(),
        },
    )


async def _close_async_iterator(iterator: AsyncIterator[Any]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


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
    event_payload = {
        "hook_name": hook_name,
        "scope": require_clean_nonblank(scope, "runtime_hook.scope"),
        "phase": phase.value,
        "terminal_event_id": terminal_event.id,
        "terminal_event_type": str(terminal_event.type),
        **copy_json_value(payload, "payload"),
    }
    return Event(
        type=event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        payload=event_payload,
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
    return require_clean_nonblank(artifact_store_id, "artifact_store.id")


def _artifact_store(registered_environment: runtime_records.RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.artifact_store


async def _resolved_file_attachments(
    *,
    messages: list[Message],
    session: Session,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    max_file_attachment_bytes: int,
    max_total_file_attachment_bytes: int,
    max_file_attachments_per_request: int,
) -> dict[str, dict[str, Any]]:
    attachment_refs = _file_attachment_refs(messages)
    if not attachment_refs:
        return {}
    if len(attachment_refs) > max_file_attachments_per_request:
        raise RuntimeError(
            "File attachment count exceeds the runtime attachment limit: "
            f"{len(attachment_refs)} > {max_file_attachments_per_request}"
        )
    artifact_store = _artifact_store(registered_environment)
    if artifact_store is None:
        raise RuntimeError("File attachments require an artifact store.")

    environment_name = _environment_name(registered_environment)
    resolved: dict[str, dict[str, Any]] = {}
    total_attachment_bytes = 0
    for attachment in attachment_refs:
        if attachment.size_bytes > max_file_attachment_bytes:
            raise RuntimeError(
                "File attachment exceeds the runtime attachment byte limit: "
                f"{attachment.artifact_id}"
            )
        total_attachment_bytes += attachment.size_bytes
        if total_attachment_bytes > max_total_file_attachment_bytes:
            raise RuntimeError("File attachments exceed the runtime total attachment byte limit.")
        if attachment.artifact_id in resolved:
            continue
        result = await artifact_store.read_bytes(
            attachment.artifact_id,
            max_bytes=attachment.size_bytes,
        )
        artifact = result.metadata
        if artifact.scope.value == "session" and artifact.session_id != session.id:
            raise RuntimeError("File attachment is not available in this session.")
        if artifact.scope.value == "environment" and artifact.environment_name != environment_name:
            raise RuntimeError("File attachment is not available in this environment.")
        if artifact.content_type != attachment.content_type:
            raise RuntimeError("File attachment content type changed before provider request.")
        if artifact.size_bytes != attachment.size_bytes:
            raise RuntimeError("File attachment size changed before provider request.")
        resolved[attachment.artifact_id] = resolved_file_attachment(attachment, result)
    return resolved


def _file_attachment_refs(messages: list[Message]) -> tuple[FileAttachment, ...]:
    refs: dict[str, FileAttachment] = {}
    ordered_refs: list[FileAttachment] = []
    for message in messages:
        for part in message.content:
            if type(part) is not ToolResultPart:
                continue
            for payload in part.artifacts:
                attachment = file_attachment_from_payload(payload)
                if attachment is None:
                    continue
                existing = refs.get(attachment.artifact_id)
                if existing is not None and not _same_file_attachment_ref(existing, attachment):
                    raise RuntimeError(
                        "Conflicting file attachment references for artifact: "
                        f"{attachment.artifact_id}"
                    )
                refs[attachment.artifact_id] = attachment
                ordered_refs.append(attachment)
    return tuple(ordered_refs)


def _same_file_attachment_ref(left: FileAttachment, right: FileAttachment) -> bool:
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def _runner(registered_environment: runtime_records.RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.runner


def _vault(registered_environment: runtime_records.RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.vault


def _mcp_servers(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> tuple[Any, ...]:
    if registered_environment is None:
        return ()
    return registered_environment.environment.mcp_servers


def _validate_stream_event(value: object) -> ModelStreamEvent:
    if type(value) is not ModelStreamEvent:
        raise TypeError("Model providers must yield ModelStreamEvent instances.")
    return copy_model_stream_event(value)


def _model_stream_event_to_runtime_event(
    stream_event: ModelStreamEvent,
    *,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    provider_name: str | None,
    step: int,
    attempt: int,
    max_attempts: int,
    classification: dict[str, str] | None = None,
) -> Event:
    if type(stream_event) is not ModelStreamEvent:
        raise TypeError("Model stream events must be ModelStreamEvent instances.")
    if stream_event.type == ModelStreamEventType.TEXT_DELTA:
        return Event(
            type=EventType.MODEL_TEXT_DELTA,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload=_retry_attempt_payload(
                {"delta": stream_event.delta},
                step=step,
                attempt=attempt,
                max_attempts=max_attempts,
            ),
        )
    if stream_event.type == ModelStreamEventType.COMPLETED:
        payload = transcript_helpers.model_completed_event_payload(stream_event.payload)
        completion = _stream_event_completion(stream_event)
        payload["completion"] = {
            "finish_reason": completion.finish_reason.value,
            "raw_finish_reason": completion.raw_finish_reason,
            "status": completion.status,
        }
        if classification is not None:
            payload["step_classification"] = classification
        usage_metrics = usage_metrics_payload(
            normalize_usage_metrics(
                provider_name=provider_name,
                model=_payload_model(payload, fallback=session.model),
                raw_usage=payload.get("usage"),
            )
        )
        if usage_metrics is not None:
            payload["usage_metrics"] = usage_metrics
        payload = _retry_attempt_payload(
            payload,
            step=step,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        return Event(
            type=EventType.MODEL_COMPLETED,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload=payload,
        )
    if stream_event.type == ModelStreamEventType.ERROR:
        return Event(
            type=EventType.MODEL_ERROR,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload=_retry_attempt_payload(
                copy_json_value(stream_event.payload, "payload"),
                step=step,
                attempt=attempt,
                max_attempts=max_attempts,
            ),
        )
    raise ValueError(f"Unsupported model stream event type: {stream_event.type}")


def _with_structured_output_tool_instruction(
    messages: list[Message],
    spec: StructuredOutputSpec,
) -> list[Message]:
    copied_messages = copy_context_messages(messages)
    instruction = Message.text(
        MessageRole.SYSTEM,
        structured_output_tool_instruction(spec),
    )
    insert_at = 0
    while (
        insert_at < len(copied_messages) and copied_messages[insert_at].role == MessageRole.SYSTEM
    ):
        insert_at += 1
    copied_messages.insert(insert_at, instruction)
    return copied_messages


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


def _stream_event_completion(stream_event: ModelStreamEvent) -> ModelCompletion:
    if type(stream_event) is not ModelStreamEvent:
        raise TypeError("Model stream events must be ModelStreamEvent instances.")
    if stream_event.type != ModelStreamEventType.COMPLETED:
        raise ValueError("Only completed model stream events have completion metadata.")
    if stream_event.completion is not None:
        return stream_event.completion
    return normalize_model_completion(stream_event.payload)


def _assistant_step_result(
    *,
    session_id: str,
    step: int,
    assistant_message: Message | None,
    tool_calls: list[runtime_records.ToolCallRequest],
    completion: ModelCompletion,
) -> AssistantStepResult:
    text_content = assistant_text_content(assistant_message)
    return AssistantStepResult(
        session_id=session_id,
        step=step,
        assistant_message=assistant_message,
        tool_calls=list(tool_calls),
        completion=completion,
        text_content=text_content,
        has_user_visible_content=bool(text_content.strip()),
        provider_state_count=provider_state_count(assistant_message),
    )


def _retry_attempt_payload(
    payload: dict[str, Any],
    *,
    step: int,
    attempt: int,
    max_attempts: int,
) -> dict[str, Any]:
    if max_attempts <= 1:
        return payload
    enriched = dict(payload)
    enriched["step"] = step
    enriched["attempt"] = attempt
    enriched["max_attempts"] = max_attempts
    return enriched


def _payload_model(payload: dict[str, Any], *, fallback: str) -> str:
    model = payload.get("model")
    if type(model) is str and model.strip():
        return model
    return fallback


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
