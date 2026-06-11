from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from copy import deepcopy
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
    ProviderStatePart,
    ToolCallPart,
    ToolResultPart,
)
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.environments import Environment, EnvironmentSpec, copy_environment
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    copy_model_stream_event,
)
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
from cayu.runtime.sessions import (
    ForkSessionRequest,
    InMemorySessionStore,
    ResumeRequest,
    RunRequest,
    Session,
    SessionIdentity,
    SessionStatus,
    SessionStore,
    copy_fork_session_request,
    copy_resume_request,
    copy_run_request,
)
from cayu.runtime.tasks import Task, TaskCreate, TaskStore, copy_task_create
from cayu.runtime.tool_policy import (
    AllowAllToolPolicy,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)

RegisteredAgent = runtime_records.RegisteredAgent
RegisteredEnvironment = runtime_records.RegisteredEnvironment


class _SessionInterrupted(Exception):
    def __init__(self, approval: PendingToolApproval) -> None:
        super().__init__(f"Tool call requires approval: {approval.tool_name}")
        self.approval = copy_pending_tool_approval(approval)


_RESUMABLE_SESSION_STATUSES = {
    SessionStatus.COMPLETED,
    SessionStatus.FAILED,
    SessionStatus.INTERRUPTED,
}
_FORKABLE_SESSION_STATUSES = _RESUMABLE_SESSION_STATUSES


class CayuApp:
    """Application runtime for registered agents, providers, and session state."""

    def __init__(
        self,
        *,
        session_store: SessionStore | None = None,
        task_store: TaskStore | None = None,
        dispatcher: Dispatcher | None = None,
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
        self._runtime_hooks = tuple(hooks)
        self._event_sinks = sinks
        self._agents: dict[str, runtime_records.RegisteredAgentState] = {}
        self._providers: dict[str, runtime_records.RegisteredProvider] = {}
        self._environments: dict[str, runtime_records.RegisteredEnvironment] = {}
        self._default_provider_name: str | None = None
        self._default_environment_name: str | None = None

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

                model_request = ModelRequest(
                    model=session.model,
                    messages=context_messages,
                    tools=[
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": deepcopy(tool.schema),
                        }
                        for tool in registered_agent.tools.values()
                    ],
                    options={
                        "agent_metadata": deepcopy(registered_agent.spec.metadata),
                        "environment_metadata": (
                            deepcopy(registered_environment.spec.metadata)
                            if registered_environment is not None
                            else {}
                        ),
                        "step": step,
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

                yield await self._emit(
                    Event(
                        type=EventType.MODEL_STARTED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        payload={
                            "model": session.model,
                            "provider": registered_provider.name,
                            "step": step,
                        },
                        environment_name=environment_name,
                    )
                )

                assistant_parts: list[transcript_helpers.AssistantTextPart | ToolCallPart] = []
                tool_calls: list[runtime_records.ToolCallRequest] = []
                provider_state_parts: list[ProviderStatePart] = []
                model_completed = False
                async for raw_stream_event in provider.stream(model_request):
                    stream_event = _validate_stream_event(raw_stream_event)
                    if model_completed:
                        raise RuntimeError(
                            f"Model provider emitted event after completed: {stream_event.type}"
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
                        provider_state_parts = transcript_helpers.provider_state_parts(
                            stream_event.payload,
                        )

                    event = _model_stream_event_to_runtime_event(
                        stream_event,
                        session=session,
                        registered_agent=registered_agent,
                        environment_name=environment_name,
                    )
                    yield await self._emit(event)
                    if stream_event.type == ModelStreamEventType.ERROR:
                        raise RuntimeError(
                            str(stream_event.payload.get("error") or "Model provider error")
                        )

                if not model_completed:
                    raise RuntimeError("Model provider stream ended without a completed event.")

                assistant_message = transcript_helpers.assistant_message(
                    content_parts=assistant_parts,
                    provider_state_parts=provider_state_parts,
                )
                if assistant_message is not None:
                    messages.append(assistant_message)
                    await self.session_store.append_transcript_messages(
                        session.id,
                        [assistant_message],
                    )

                if not tool_calls:
                    break

                policy_plan = await self._policy_plan_for_tool_round(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_calls=tool_calls,
                    request_metadata=request_metadata,
                    task_id=task_id,
                )
                if policy_plan.pending_approval is not None:
                    approval, checkpoint_event = policy_plan.pending_approval
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
                    raise _SessionInterrupted(approval)

                policy_results_by_id = {
                    outcome.call.id: outcome.result for outcome in policy_plan.outcomes
                }
                tool_outcomes: list[runtime_records.ToolCallOutcome] = []
                for tool_call in tool_calls:
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

                tool_result_messages = transcript_helpers.tool_result_messages(tool_outcomes)
                messages.extend(tool_result_messages)
                await self.session_store.append_transcript_messages(
                    session.id,
                    tool_result_messages,
                )
            else:
                raise RuntimeError(f"Maximum model steps exceeded: {max_steps}")

            if task_id is not None:
                task = await self._complete_task(
                    task_id=task_id,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                )
                task_finished = True
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
                        "approval": exc.approval.model_dump(),
                    },
                ),
                phase=RuntimeHookPhase.AFTER_SESSION_INTERRUPTED,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            ):
                yield event
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

    async def _policy_plan_for_tool_round(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_calls: list[runtime_records.ToolCallRequest],
        request_metadata: dict[str, Any],
        task_id: str | None,
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

        pending_approval = await self._checkpoint_pending_tool_approval(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            tool_call=approval_tool_call,
            tool_calls=[outcome.call for outcome in policy_outcomes],
            policy_outcomes=policy_outcomes,
            task_id=task_id,
            policy_result=approval_policy_result,
        )
        return runtime_records.ToolRoundPolicyPlan(
            outcomes=policy_outcomes,
            pending_approval=pending_approval,
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
) -> Event:
    if type(stream_event) is not ModelStreamEvent:
        raise TypeError("Model stream events must be ModelStreamEvent instances.")
    if stream_event.type == ModelStreamEventType.TEXT_DELTA:
        return Event(
            type=EventType.MODEL_TEXT_DELTA,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload={"delta": stream_event.delta},
        )
    if stream_event.type == ModelStreamEventType.COMPLETED:
        return Event(
            type=EventType.MODEL_COMPLETED,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload=transcript_helpers.model_completed_event_payload(stream_event.payload),
        )
    if stream_event.type == ModelStreamEventType.ERROR:
        return Event(
            type=EventType.MODEL_ERROR,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload=copy_json_value(stream_event.payload, "payload"),
        )
    raise ValueError(f"Unsupported model stream event type: {stream_event.type}")


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
