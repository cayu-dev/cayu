from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from cayu._validation import copy_json_value, require_nonblank
from cayu.core.agents import AgentSpec
from cayu.core.events import Event, EventType
from cayu.core.messages import (
    Message,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    copy_message_part,
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
from cayu.runtime.approvals import (
    PendingToolApproval,
    PendingToolCallApproval,
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
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


@dataclass(frozen=True)
class RegisteredAgent:
    spec: AgentSpec
    tools: Mapping[str, RegisteredTool]


@dataclass(frozen=True)
class _RegisteredAgentState:
    spec: AgentSpec
    tools: Mapping[str, RegisteredTool]
    context_policy: ContextPolicy
    tool_policy: ToolPolicy


@dataclass
class _AssistantTextPart:
    text: str


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    schema: dict[str, Any]
    tool: Tool


@dataclass(frozen=True)
class RegisteredProvider:
    name: str
    provider: ModelProvider


@dataclass(frozen=True)
class RegisteredEnvironment:
    spec: EnvironmentSpec
    environment: Environment


@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolCallOutcome:
    call: ToolCallRequest
    result: ToolResult


@dataclass(frozen=True)
class ToolCallPolicyOutcome:
    call: ToolCallRequest
    result: ToolPolicyResult | None


@dataclass(frozen=True)
class ToolRoundPolicyPlan:
    outcomes: list[ToolCallPolicyOutcome]
    pending_approval: tuple[PendingToolApproval, Event] | None


class _SessionInterrupted(Exception):
    def __init__(self, approval: PendingToolApproval) -> None:
        super().__init__(f"Tool call requires approval: {approval.tool_name}")
        self.approval = copy_pending_tool_approval(approval)


class _ToolApprovalManualRecoveryRequired(RuntimeError):
    def __init__(self, *, tool_call_id: str, tool_name: str) -> None:
        super().__init__(
            "Tool approval cannot be retried automatically because a tool call "
            f"started without a terminal result: {tool_call_id} ({tool_name})."
        )
        self.tool_call_id = tool_call_id
        self.tool_name = tool_name


_RESUMABLE_SESSION_STATUSES = {
    SessionStatus.COMPLETED,
    SessionStatus.FAILED,
    SessionStatus.INTERRUPTED,
}
_FORKABLE_SESSION_STATUSES = _RESUMABLE_SESSION_STATUSES

_PENDING_TOOL_APPROVAL_CHECKPOINT_KEY = "pending_tool_approval"


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
    ) -> None:
        if session_store is not None and not isinstance(session_store, SessionStore):
            raise TypeError("session_store must be a SessionStore.")
        if task_store is not None and not isinstance(task_store, TaskStore):
            raise TypeError("task_store must be a TaskStore.")
        if dispatcher is not None and not isinstance(dispatcher, Dispatcher):
            raise TypeError("dispatcher must be a Dispatcher.")
        if runtime_hooks is None:
            hooks = []
        else:
            if isinstance(runtime_hooks, str | bytes):
                raise TypeError("runtime_hooks must be an iterable of RuntimeHook instances.")
            try:
                hooks = list(runtime_hooks)
            except TypeError as exc:
                raise TypeError(
                    "runtime_hooks must be an iterable of RuntimeHook instances."
                ) from exc
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
        for hook in hooks:
            if not isinstance(hook, RuntimeHook):
                raise TypeError("runtime_hooks must contain RuntimeHook instances.")
            require_nonblank(hook.name, "runtime_hook.name")
        self.session_store = session_store if session_store is not None else InMemorySessionStore()
        self.task_store = task_store
        self.dispatcher = dispatcher if dispatcher is not None else InlineDispatcher()
        self._runtime_hooks = tuple(hooks)
        self._event_sinks = sinks
        self._agents: dict[str, _RegisteredAgentState] = {}
        self._providers: dict[str, RegisteredProvider] = {}
        self._environments: dict[str, RegisteredEnvironment] = {}
        self._default_provider_name: str | None = None
        self._default_environment_name: str | None = None

    def register_agent(
        self,
        spec: AgentSpec,
        *,
        tools: Iterable[Tool] | None = None,
        context_policy: ContextPolicy | None = None,
        tool_policy: ToolPolicy | None = None,
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

        if tools is None:
            agent_tools = []
        else:
            if isinstance(tools, str | bytes):
                raise TypeError("Agent tools must be an iterable of Tool instances.")
            try:
                agent_tools = list(tools)
            except TypeError as exc:
                raise TypeError("Agent tools must be an iterable of Tool instances.") from exc

        tools_by_name: dict[str, RegisteredTool] = {}
        for tool in agent_tools:
            if not isinstance(tool, Tool):
                raise TypeError("Agent tools must be Tool instances.")
            registered_tool = _validate_registered_tool(tool)
            if registered_tool.name in tools_by_name:
                raise ValueError(f"Duplicate tool registered for agent: {registered_tool.name}")
            tools_by_name[registered_tool.name] = registered_tool

        self._agents[stored_spec.name] = _RegisteredAgentState(
            spec=stored_spec,
            tools=MappingProxyType(tools_by_name),
            context_policy=stored_context_policy,
            tool_policy=stored_tool_policy,
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
        require_nonblank(provider.name, "provider.name")
        if provider.name in self._providers:
            raise ValueError(f"Provider already registered: {provider.name}")

        self._providers[provider.name] = RegisteredProvider(
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

        self._environments[stored_spec.name] = RegisteredEnvironment(
            spec=stored_spec,
            environment=stored_environment,
        )
        if default or self._default_environment_name is None:
            self._default_environment_name = stored_spec.name
        return environment

    def get_agent(self, name: str) -> RegisteredAgent:
        agent_name = require_nonblank(name, "agent.name")
        registered_agent = self._get_registered_agent(agent_name)
        return RegisteredAgent(
            spec=registered_agent.spec.model_copy(deep=True),
            tools={
                name: _copy_registered_tool(tool) for name, tool in registered_agent.tools.items()
            },
        )

    def _get_registered_agent(self, name: str) -> _RegisteredAgentState:
        agent_name = require_nonblank(name, "agent.name")
        try:
            return self._agents[agent_name]
        except KeyError as exc:
            raise KeyError(f"Agent not registered: {agent_name}") from exc

    def get_provider(self, name: str | None = None) -> ModelProvider:
        return self._get_registered_provider(name).provider

    def get_environment(self, name: str | None = None) -> RegisteredEnvironment:
        registered_environment = self._get_registered_environment(name)
        if registered_environment is None:
            raise RuntimeError("No environment registered.")
        return RegisteredEnvironment(
            spec=registered_environment.spec.model_copy(deep=True),
            environment=copy_environment(registered_environment.environment),
        )

    def _get_registered_provider(self, name: str | None = None) -> RegisteredProvider:
        if name is not None:
            provider_name = require_nonblank(name, "provider.name")
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
    ) -> RegisteredEnvironment | None:
        if name is not None:
            environment_name = require_nonblank(name, "environment.name")
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
    ) -> RegisteredEnvironment | None:
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
        messages = _initial_messages(
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
        if _pending_tool_approval_from_checkpoint(checkpoint) is not None:
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
                return _checkpoint_for_fork(
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
        pending_approval = _pending_tool_approval_from_checkpoint(checkpoint)
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
        registered_agent: _RegisteredAgentState,
        registered_provider: RegisteredProvider,
        registered_environment: RegisteredEnvironment | None,
        emit_resume_event: bool = True,
    ) -> AsyncIterator[Event]:
        environment_name = _environment_name(registered_environment)
        pending_approval_cleared = False
        tool_outcomes: list[ToolCallOutcome] = []
        try:
            transcript = await self.session_store.load_transcript(session.id)
            approval_events = await self.session_store.load_events(session.id)
            _validate_approval_retry_decision(
                events=approval_events,
                approval=pending_approval,
                decision=request.decision,
            )
            recorded_outcomes = _recorded_approval_tool_outcomes(
                events=approval_events,
                approval=pending_approval,
            )
            if emit_resume_event:
                yield await self._emit(
                    _tool_approval_resumed_event(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        approval=pending_approval,
                        decision=request.decision,
                    )
                )

            if request.decision not in {
                ToolApprovalDecision.APPROVE,
                ToolApprovalDecision.DENY,
            }:
                raise ValueError(f"Unsupported tool approval decision: {request.decision}")

            for pending_tool_call in _pending_round_tool_calls(pending_approval):
                tool_call = ToolCallRequest(
                    id=pending_tool_call.tool_call_id,
                    name=pending_tool_call.tool_name,
                    arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
                )
                policy_result = _policy_result_from_pending_tool_call(pending_tool_call)
                recorded_outcome = recorded_outcomes.get(tool_call.id)
                if recorded_outcome is not None:
                    tool_outcomes.append(recorded_outcome)
                    continue

                if policy_result is not None and policy_result.decision == ToolPolicyDecision.DENY:
                    reason = _tool_policy_denial_reason(policy_result)
                    result = _blocked_tool_result(policy_result, reason=reason)
                    yield await self._emit(
                        Event(
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
                        )
                    )
                    tool_outcomes.append(ToolCallOutcome(call=tool_call, result=result))
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
                    result = _approval_denied_tool_result(
                        request,
                        approval=pending_approval,
                        tool_call=tool_call,
                        approval_required=approval_required,
                    )
                    yield await self._emit(
                        Event(
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
                        )
                    )
                    tool_outcomes.append(ToolCallOutcome(call=tool_call, result=result))
                    continue

                outcome = None
                async for event, outcome in self._execute_tool_call(  # noqa: B007
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

            tool_result_messages = _tool_result_messages(tool_outcomes)
            transcript.extend(tool_result_messages)
            cleared_checkpoint = await self._checkpoint_without_pending_tool_approval(session.id)
            await self.session_store.append_transcript_messages_and_checkpoint(
                session.id,
                tool_result_messages,
                cleared_checkpoint,
            )
            pending_approval_cleared = True
            yield await self._emit(
                _pending_tool_approval_cleared_event(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
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
            if isinstance(exc, _ToolApprovalManualRecoveryRequired):
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
        pending_approval = _pending_tool_approval_from_checkpoint(checkpoint)
        if pending_approval is None:
            raise RuntimeError("Session has no pending tool approval.")
        if pending_approval.approval_id != request.approval_id:
            raise ValueError(
                f"Tool approval id does not match pending approval: {request.approval_id}"
            )

        pending_tool_call = _pending_tool_call_for_recovery(
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
        recovered_result = _recovered_tool_result(
            request=request,
        )
        event_type = (
            EventType.TOOL_CALL_FAILED
            if recovered_result.is_error
            else EventType.TOOL_CALL_COMPLETED
        )

        try:
            events = await self.session_store.load_events(session.id)
            _validate_tool_approval_recovery_target(
                events=events,
                approval=pending_approval,
                tool_call_id=request.tool_call_id,
            )
            recovery_events = [
                _tool_approval_resumed_event(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
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
            for event in await self._emit_many(session.id, recovery_events):
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
        registered_agent: _RegisteredAgentState,
        registered_provider: RegisteredProvider,
        registered_environment: RegisteredEnvironment | None,
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

                assistant_parts: list[_AssistantTextPart | ToolCallPart] = []
                tool_calls: list[ToolCallRequest] = []
                provider_state_parts: list[ProviderStatePart] = []
                model_completed = False
                async for raw_stream_event in provider.stream(model_request):
                    stream_event = _validate_stream_event(raw_stream_event)
                    if model_completed:
                        raise RuntimeError(
                            f"Model provider emitted event after completed: {stream_event.type}"
                        )

                    if stream_event.type == ModelStreamEventType.TOOL_CALL:
                        tool_call = _parse_tool_call(stream_event.payload)
                        tool_calls.append(tool_call)
                        assistant_parts.append(_tool_call_part(tool_call))
                        continue

                    if stream_event.type == ModelStreamEventType.TEXT_DELTA:
                        _append_assistant_text_delta(assistant_parts, stream_event.delta)
                    elif stream_event.type == ModelStreamEventType.COMPLETED:
                        model_completed = True
                        provider_state_parts = _provider_state_parts(
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

                assistant_message = _assistant_message(
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
                tool_outcomes: list[ToolCallOutcome] = []
                for tool_call in tool_calls:
                    outcome = None
                    async for event, outcome in self._execute_tool_call(  # noqa: B007
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

                tool_result_messages = _tool_result_messages(tool_outcomes)
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
        registered_agent: _RegisteredAgentState,
        registered_environment: RegisteredEnvironment | None,
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
        registered_agent: _RegisteredAgentState,
        registered_environment: RegisteredEnvironment | None,
        tool_calls: list[ToolCallRequest],
        request_metadata: dict[str, Any],
        task_id: str | None,
    ) -> ToolRoundPolicyPlan:
        policy_outcomes: list[ToolCallPolicyOutcome] = []
        approval_policy_result: ToolPolicyResult | None = None
        approval_tool_call: ToolCallRequest | None = None
        for tool_call in tool_calls:
            if tool_call.name not in registered_agent.tools:
                policy_outcomes.append(ToolCallPolicyOutcome(call=tool_call, result=None))
                continue

            policy_result = await self._authorize_tool_call(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                tool_call=tool_call,
                request_metadata=request_metadata,
            )
            policy_outcomes.append(ToolCallPolicyOutcome(call=tool_call, result=policy_result))
            if (
                approval_policy_result is None
                and policy_result.decision == ToolPolicyDecision.REQUIRE_APPROVAL
            ):
                approval_policy_result = policy_result
                approval_tool_call = tool_call

        if approval_policy_result is None or approval_tool_call is None:
            return ToolRoundPolicyPlan(outcomes=policy_outcomes, pending_approval=None)

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
        return ToolRoundPolicyPlan(
            outcomes=policy_outcomes,
            pending_approval=pending_approval,
        )

    async def _authorize_tool_call(
        self,
        *,
        session: Session,
        registered_agent: _RegisteredAgentState,
        registered_environment: RegisteredEnvironment | None,
        tool_call: ToolCallRequest,
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
        return _validate_tool_policy_result(policy_result)

    async def _execute_tool_call(
        self,
        *,
        session: Session,
        registered_agent: _RegisteredAgentState,
        registered_environment: RegisteredEnvironment | None,
        tool_call: ToolCallRequest,
        request_metadata: dict[str, Any],
        task_id: str | None,
        check_policy: bool = True,
        emit_started: bool = True,
        policy_result: ToolPolicyResult | None = None,
        approval_id: str | None = None,
    ) -> AsyncIterator[tuple[Event, ToolCallOutcome | None]]:
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
            yield (
                await self._emit(
                    Event(
                        type=EventType.TOOL_CALL_FAILED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        tool_name=tool_call.name,
                        payload=payload,
                    )
                ),
                ToolCallOutcome(call=tool_call, result=result),
            )
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
                resolved_policy_result = _validate_tool_policy_result(policy_result)
            if resolved_policy_result.decision == ToolPolicyDecision.DENY:
                reason = _tool_policy_denial_reason(resolved_policy_result)
                result = _blocked_tool_result(resolved_policy_result, reason=reason)
                payload = {
                    "tool_call_id": tool_call.id,
                    "decision": resolved_policy_result.decision.value,
                    "reason": reason,
                    "metadata": resolved_policy_result.metadata,
                    "result": result.model_dump(),
                }
                if approval_id is not None:
                    payload["approval_id"] = approval_id
                yield (
                    await self._emit(
                        Event(
                            type=EventType.TOOL_CALL_BLOCKED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            tool_name=tool_call.name,
                            payload=payload,
                        )
                    ),
                    ToolCallOutcome(call=tool_call, result=result),
                )
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

        result = await _run_tool(
            tool=registered_tool.tool,
            ctx=ToolContext(
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                workspace_id=_workspace_id(registered_environment),
                workspace=_workspace(registered_environment),
                runner=_runner(registered_environment),
                vault=_vault(registered_environment),
                mcp_servers=_mcp_servers(registered_environment),
                metadata=_tool_context_metadata(
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
        yield (
            await self._emit(
                Event(
                    type=event_type,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=tool_call.name,
                    payload=payload,
                )
            ),
            ToolCallOutcome(call=tool_call, result=result),
        )

    async def _checkpoint_pending_tool_approval(
        self,
        *,
        session: Session,
        registered_agent: _RegisteredAgentState,
        registered_environment: RegisteredEnvironment | None,
        tool_call: ToolCallRequest,
        tool_calls: list[ToolCallRequest],
        policy_outcomes: list[ToolCallPolicyOutcome] | None,
        task_id: str | None,
        policy_result: ToolPolicyResult,
    ) -> tuple[PendingToolApproval, Event]:
        checkpoint = await self.session_store.load_checkpoint(session.id)
        checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        if _pending_tool_approval_from_checkpoint(checkpoint) is not None:
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
            tool_calls=_pending_tool_call_approvals(
                tool_calls=tool_calls,
                policy_outcomes=policy_outcomes,
            ),
        )
        checkpoint[_PENDING_TOOL_APPROVAL_CHECKPOINT_KEY] = approval.model_dump()
        await self.session_store.checkpoint(session.id, checkpoint)
        return (
            approval,
            Event(
                type=EventType.SESSION_CHECKPOINTED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=_environment_name(registered_environment),
                payload={
                    "checkpoint": _PENDING_TOOL_APPROVAL_CHECKPOINT_KEY,
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
        checkpoint.pop(_PENDING_TOOL_APPROVAL_CHECKPOINT_KEY, None)
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
        registered_agent: _RegisteredAgentState,
        registered_environment: RegisteredEnvironment | None,
    ) -> AsyncIterator[Event]:
        terminal_event = await self._emit(event)
        yield terminal_event
        async for hook_event in self._run_runtime_hooks(
            phase=phase,
            session=session,
            terminal_event=terminal_event,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
        ):
            yield hook_event

    async def _run_runtime_hooks(
        self,
        *,
        phase: RuntimeHookPhase,
        session: Session,
        terminal_event: Event,
        registered_agent: _RegisteredAgentState,
        registered_environment: RegisteredEnvironment | None,
    ) -> AsyncIterator[Event]:
        for hook in self._runtime_hooks:
            hook_name = require_nonblank(hook.name, "runtime_hook.name")
            yield await self._emit(
                _runtime_hook_event(
                    event_type=EventType.HOOK_STARTED,
                    hook_name=hook_name,
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


def _tool_approval_resumed_event(
    *,
    session: Session,
    registered_agent: _RegisteredAgentState,
    registered_environment: RegisteredEnvironment | None,
    approval: PendingToolApproval,
    decision: ToolApprovalDecision,
) -> Event:
    return Event(
        type=EventType.SESSION_RESUMED,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        payload={
            "agent_name": registered_agent.spec.name,
            "approval_id": approval.approval_id,
            "tool_call_id": approval.tool_call_id,
            "decision": decision.value,
        },
    )


def _pending_tool_approval_cleared_event(
    *,
    session: Session,
    registered_agent: _RegisteredAgentState,
    registered_environment: RegisteredEnvironment | None,
    approval_id: str,
) -> Event:
    return Event(
        type=EventType.SESSION_CHECKPOINTED,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=_environment_name(registered_environment),
        payload={
            "checkpoint": _PENDING_TOOL_APPROVAL_CHECKPOINT_KEY,
            "approval_id": approval_id,
            "cleared": True,
        },
    )


def _checkpoint_for_fork(
    *,
    checkpoint: dict[str, Any] | None,
    agent_name: str,
    environment_name: str | None,
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
    pending_approval = _pending_tool_approval_from_checkpoint(copied_checkpoint)
    if pending_approval is None:
        return copied_checkpoint
    if pending_approval.agent_name != agent_name:
        raise ValueError(
            "Cannot fork a pending tool approval to a different agent: "
            f"{pending_approval.agent_name} -> {agent_name}"
        )
    if pending_approval.environment_name != environment_name:
        raise ValueError(
            "Cannot fork a pending tool approval to a different environment: "
            f"{pending_approval.environment_name} -> {environment_name}"
        )
    copied_checkpoint[_PENDING_TOOL_APPROVAL_CHECKPOINT_KEY] = pending_approval.model_copy(
        update={"task_id": None}
    ).model_dump()
    return copied_checkpoint


async def _run_tool(
    *,
    tool: Tool,
    ctx: ToolContext,
    arguments: dict[str, Any],
) -> ToolResult:
    try:
        result = await tool.run(ctx, arguments)
        if type(result) is not ToolResult:
            return ToolResult(
                content=(
                    "Tool returned invalid result type: "
                    f"{type(result).__name__}. Expected ToolResult."
                ),
                is_error=True,
            )
        return _normalize_tool_result(_validate_tool_result(result))
    except Exception as exc:
        return ToolResult(content=_exception_message(exc), is_error=True)


def _tool_policy_denial_reason(policy_result: ToolPolicyResult) -> str:
    return policy_result.reason or "Tool call denied by policy."


def _blocked_tool_result(policy_result: ToolPolicyResult, *, reason: str) -> ToolResult:
    return ToolResult(
        content=reason,
        structured={
            "decision": policy_result.decision.value,
            "reason": reason,
            "metadata": policy_result.metadata,
        },
        is_error=True,
    )


def _approval_denied_tool_result(
    request: ToolApprovalRequest,
    *,
    approval: PendingToolApproval,
    tool_call: ToolCallRequest,
    approval_required: bool,
) -> ToolResult:
    if request.reason:
        reason = request.reason
        if approval_required:
            content = f"Tool call denied by approval: {request.reason}"
        else:
            content = (
                "Tool call skipped because approval was denied for the same tool round: "
                f"{request.reason}"
            )
    elif approval_required:
        reason = "Tool call denied by approval."
        content = reason
    else:
        reason = "Tool call skipped because approval was denied for the same tool round."
        content = reason

    return ToolResult(
        content=content,
        structured={
            "decision": request.decision.value,
            "approval_id": approval.approval_id,
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "approval_required": approval_required,
            "denied_by_approval": approval_required,
            "skipped_due_to_approval_denial": not approval_required,
            "denied_tool_call_id": approval.tool_call_id,
            "denied_tool_name": approval.tool_name,
            "reason": reason,
            "metadata": request.metadata,
        },
        is_error=True,
    )


def _recorded_approval_tool_outcomes(
    *,
    events: list[Event],
    approval: PendingToolApproval,
) -> dict[str, ToolCallOutcome]:
    pending_calls = {call.tool_call_id: call for call in _pending_round_tool_calls(approval)}
    started_ids: set[str] = set()
    outcomes: dict[str, ToolCallOutcome] = {}
    terminal_event_types = {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }

    for event in events:
        if event.payload.get("approval_id") != approval.approval_id:
            continue

        tool_call_id = event.payload.get("tool_call_id")
        if type(tool_call_id) is not str or tool_call_id not in pending_calls:
            continue

        if event.type == EventType.TOOL_CALL_STARTED:
            started_ids.add(tool_call_id)
            continue

        if event.type in terminal_event_types:
            outcomes[tool_call_id] = _tool_call_outcome_from_terminal_event(
                event=event,
                pending_tool_call=pending_calls[tool_call_id],
            )

    for tool_call_id in started_ids:
        if tool_call_id not in outcomes:
            pending_tool_call = pending_calls[tool_call_id]
            raise _ToolApprovalManualRecoveryRequired(
                tool_call_id=tool_call_id,
                tool_name=pending_tool_call.tool_name,
            )

    return outcomes


def _validate_approval_retry_decision(
    *,
    events: list[Event],
    approval: PendingToolApproval,
    decision: ToolApprovalDecision,
) -> None:
    has_denied_result = False
    has_approved_call = False
    has_executed_or_recovered_result = False

    for event in events:
        if event.payload.get("approval_id") != approval.approval_id:
            continue
        if event.type == EventType.TOOL_CALL_APPROVAL_DENIED:
            has_denied_result = True
        elif event.type == EventType.TOOL_CALL_APPROVED:
            has_approved_call = True
        elif event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}:
            has_executed_or_recovered_result = True

    if decision == ToolApprovalDecision.APPROVE and has_denied_result:
        raise RuntimeError(
            "Tool approval was already denied and cannot be retried as approved: "
            f"{approval.approval_id}"
        )
    if decision == ToolApprovalDecision.DENY and (
        has_approved_call or has_executed_or_recovered_result
    ):
        raise RuntimeError(
            "Tool approval already has approved or executed tool results and "
            f"cannot be retried as denied: {approval.approval_id}"
        )


def _pending_tool_call_for_recovery(
    *,
    approval: PendingToolApproval,
    tool_call_id: str,
) -> PendingToolCallApproval:
    for pending_tool_call in _pending_round_tool_calls(approval):
        if pending_tool_call.tool_call_id == tool_call_id:
            return pending_tool_call
    raise ValueError(f"Tool call is not part of the pending approval: {tool_call_id}")


def _validate_tool_approval_recovery_target(
    *,
    events: list[Event],
    approval: PendingToolApproval,
    tool_call_id: str,
) -> None:
    started = False
    terminal = False
    terminal_event_types = {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
        EventType.TOOL_CALL_APPROVAL_DENIED,
    }
    for event in events:
        if event.payload.get("approval_id") != approval.approval_id:
            continue
        if event.payload.get("tool_call_id") != tool_call_id:
            continue
        if event.type == EventType.TOOL_CALL_STARTED:
            started = True
        elif event.type in terminal_event_types:
            terminal = True

    if terminal:
        raise RuntimeError(
            f"Tool call already has a terminal event and does not need recovery: {tool_call_id}"
        )
    if not started:
        raise RuntimeError(
            f"Tool approval recovery requires a recorded tool.call.started event: {tool_call_id}"
        )


def _recovered_tool_result(
    *,
    request: ToolApprovalRecoveryRequest,
) -> ToolResult:
    if request.outcome not in {
        ToolApprovalRecoveryOutcome.COMPLETED,
        ToolApprovalRecoveryOutcome.FAILED,
    }:
        raise ValueError(f"Unsupported tool approval recovery outcome: {request.outcome}")
    return ToolResult(
        content=request.message,
        structured=request.structured,
        artifacts=request.artifacts,
        is_error=request.outcome == ToolApprovalRecoveryOutcome.FAILED,
    )


def _tool_context_metadata(
    *,
    tool_call_id: str,
    approval_id: str | None,
) -> dict[str, str]:
    metadata = {"tool_call_id": tool_call_id}
    if approval_id is not None:
        metadata["approval_id"] = approval_id
    return metadata


def _tool_call_outcome_from_terminal_event(
    *,
    event: Event,
    pending_tool_call: PendingToolCallApproval,
) -> ToolCallOutcome:
    result_payload = event.payload.get("result")
    if type(result_payload) is not dict:
        raise ValueError(
            f"Terminal tool event is missing result payload: {pending_tool_call.tool_call_id}"
        )
    result = _normalize_tool_result(_validate_tool_result(ToolResult(**result_payload)))
    return ToolCallOutcome(
        call=ToolCallRequest(
            id=pending_tool_call.tool_call_id,
            name=pending_tool_call.tool_name,
            arguments=copy_json_value(pending_tool_call.arguments, "arguments"),
        ),
        result=result,
    )


def _validate_tool_policy_result(result: ToolPolicyResult) -> ToolPolicyResult:
    if type(result) is not ToolPolicyResult:
        raise TypeError(
            "Tool policies must return ToolPolicyResult instances. "
            f"Received {type(result).__name__}."
        )
    return ToolPolicyResult(
        decision=result.decision,
        reason=result.reason,
        metadata=copy_json_value(result.metadata, "metadata"),
    )


def _pending_tool_approval_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> PendingToolApproval | None:
    if checkpoint is None:
        return None
    copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
    value = copied_checkpoint.get(_PENDING_TOOL_APPROVAL_CHECKPOINT_KEY)
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("Pending tool approval checkpoint must be an object.")
    return PendingToolApproval(**value)


def _pending_tool_call_approvals(
    *,
    tool_calls: list[ToolCallRequest],
    policy_outcomes: list[ToolCallPolicyOutcome] | None,
) -> list[PendingToolCallApproval]:
    policy_results_by_id: dict[str, ToolPolicyResult | None] = {}
    if policy_outcomes is not None:
        policy_results_by_id = {outcome.call.id: outcome.result for outcome in policy_outcomes}
    return [
        PendingToolCallApproval(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            arguments=copy_json_value(tool_call.arguments, "arguments"),
            policy_decision=(
                policy_results_by_id[tool_call.id].decision.value
                if tool_call.id in policy_results_by_id
                and policy_results_by_id[tool_call.id] is not None
                else None
            ),
            reason=(
                policy_results_by_id[tool_call.id].reason
                if tool_call.id in policy_results_by_id
                and policy_results_by_id[tool_call.id] is not None
                else None
            ),
            metadata=(
                copy_json_value(policy_results_by_id[tool_call.id].metadata, "metadata")
                if tool_call.id in policy_results_by_id
                and policy_results_by_id[tool_call.id] is not None
                else {}
            ),
        )
        for tool_call in tool_calls
    ]


def _pending_round_tool_calls(
    approval: PendingToolApproval,
) -> list[PendingToolCallApproval]:
    return [PendingToolCallApproval(**call.model_dump()) for call in approval.tool_calls]


def _policy_result_from_pending_tool_call(
    pending_tool_call: PendingToolCallApproval,
) -> ToolPolicyResult | None:
    if pending_tool_call.policy_decision is None:
        return None
    return ToolPolicyResult(
        decision=ToolPolicyDecision(pending_tool_call.policy_decision),
        reason=pending_tool_call.reason,
        metadata=copy_json_value(pending_tool_call.metadata, "metadata"),
    )


def _copy_registered_tool(tool: RegisteredTool) -> RegisteredTool:
    return RegisteredTool(
        name=tool.name,
        description=tool.description,
        schema=deepcopy(tool.schema),
        tool=tool.tool,
    )


def _validate_registered_tool(tool: Tool) -> RegisteredTool:
    spec = getattr(tool, "spec", None)
    if type(spec) is not ToolSpec:
        raise TypeError("Agent tools must define ToolSpec instances.")
    name = require_nonblank(spec.name, "name")
    validated_spec = ToolSpec(
        name=name,
        description=spec.description,
        input_schema=copy_json_value(spec.input_schema, "input_schema"),
    )
    return RegisteredTool(
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
    registered_agent: _RegisteredAgentState,
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
    registered_environment: RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None:
        return None
    return registered_environment.spec.name


def _context_compaction_telemetry_event(
    *,
    telemetry: ContextCompactionTelemetry,
    session: Session,
    registered_agent: _RegisteredAgentState,
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


def _runtime_hook_event(
    *,
    event_type: EventType,
    hook_name: str,
    phase: RuntimeHookPhase,
    session: Session,
    registered_agent: _RegisteredAgentState,
    registered_environment: RegisteredEnvironment | None,
    terminal_event: Event,
    payload: dict[str, Any],
) -> Event:
    event_payload = {
        "hook_name": hook_name,
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
    registered_agent: _RegisteredAgentState,
    registered_environment: RegisteredEnvironment | None,
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


def _workspace_id(registered_environment: RegisteredEnvironment | None) -> str | None:
    if registered_environment is None or registered_environment.environment.workspace is None:
        return None
    workspace_id = getattr(registered_environment.environment.workspace, "id", None)
    if workspace_id is None:
        return None
    return require_nonblank(workspace_id, "workspace.id")


def _workspace(registered_environment: RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.workspace


def _runner(registered_environment: RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.runner


def _vault(registered_environment: RegisteredEnvironment | None) -> Any:
    if registered_environment is None:
        return None
    return registered_environment.environment.vault


def _mcp_servers(
    registered_environment: RegisteredEnvironment | None,
) -> tuple[Any, ...]:
    if registered_environment is None:
        return ()
    return registered_environment.environment.mcp_servers


def _normalize_tool_result(result: ToolResult) -> ToolResult:
    if result.is_error and not result.content.strip():
        return result.model_copy(update={"content": "Tool returned an error without details."})
    return result


def _validate_tool_result(result: ToolResult) -> ToolResult:
    if type(result) is not ToolResult:
        raise TypeError("Tool results must be ToolResult instances.")
    return ToolResult(
        content=result.content,
        structured=copy_json_value(result.structured, "structured"),
        artifacts=copy_json_value(result.artifacts, "artifacts"),
        is_error=result.is_error,
    )


def _exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return message
    return f"{type(exc).__name__}: tool execution failed"


def _validate_stream_event(value: object) -> ModelStreamEvent:
    return copy_model_stream_event(value)


def _model_stream_event_to_runtime_event(
    stream_event: ModelStreamEvent,
    *,
    session: Session,
    registered_agent: _RegisteredAgentState,
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
            payload=_model_completed_event_payload(stream_event.payload),
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


def _require_payload_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Model tool call payload requires non-empty string `{key}`.")
    return value


def _require_payload_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if type(value) is not dict:
        raise ValueError(f"Model tool call payload requires object `{key}`.")
    return value


def _parse_tool_call(payload: dict[str, Any]) -> ToolCallRequest:
    return ToolCallRequest(
        id=_optional_payload_string(payload, "id") or str(uuid4()),
        name=_require_payload_string(payload, "name"),
        arguments=copy_json_value(_require_payload_dict(payload, "arguments"), "arguments"),
    )


def _optional_payload_string(payload: dict[str, Any], key: str) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    return _require_payload_string(payload, key)


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


def _initial_messages(
    *,
    system_prompt: str | None,
    request_messages: list[Message],
) -> list[Message]:
    messages: list[Message] = []
    if system_prompt and system_prompt.strip():
        messages.append(Message.text("system", system_prompt))
    messages.extend(message.model_copy(deep=True) for message in request_messages)
    return messages


def _assistant_message(
    *,
    content_parts: list[_AssistantTextPart | ToolCallPart],
    provider_state_parts: list[ProviderStatePart],
) -> Message | None:
    content: list[TextPart | ToolCallPart | ToolResultPart | ProviderStatePart] = []
    for part in content_parts:
        if type(part) is _AssistantTextPart:
            if part.text.strip():
                content.append(TextPart(text=part.text))
            continue
        if type(part) is ToolCallPart:
            content.append(copy_message_part(part))
            continue
        raise TypeError("Assistant content must contain text buffers or tool calls.")
    content.extend(provider_state_parts)
    if not content:
        return None
    return Message(role=MessageRole.ASSISTANT, content=content)


def _append_assistant_text_delta(
    content_parts: list[_AssistantTextPart | ToolCallPart],
    delta: str,
) -> None:
    if not delta:
        return
    if content_parts and type(content_parts[-1]) is _AssistantTextPart:
        previous = content_parts[-1]
        previous.text = f"{previous.text}{delta}"
        return
    content_parts.append(_AssistantTextPart(text=delta))


def _tool_call_part(tool_call: ToolCallRequest) -> ToolCallPart:
    return ToolCallPart(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        arguments=deepcopy(tool_call.arguments),
    )


def _provider_state_parts(payload: dict[str, Any]) -> list[ProviderStatePart]:
    raw_parts = payload.get("provider_state", [])
    if raw_parts is None:
        return []
    if type(raw_parts) is not list:
        raise ValueError("Model completed payload provider_state must be a list.")
    parts: list[ProviderStatePart] = []
    for index, raw_part in enumerate(raw_parts):
        if type(raw_part) is not dict:
            raise ValueError(f"Model completed payload provider_state[{index}] must be an object.")
        provider = raw_part.get("provider")
        state = raw_part.get("state")
        parts.append(ProviderStatePart(provider=provider, state=state))
    return parts


def _model_completed_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    copied = copy_json_value(payload, "payload")
    if type(copied) is not dict:
        raise ValueError("Model completed payload must be an object.")
    copied.pop("provider_state", None)
    return copied


def _tool_result_messages(outcomes: list[ToolCallOutcome]) -> list[Message]:
    return [
        Message.tool_result(
            results=[
                ToolResultPart(
                    tool_call_id=outcome.call.id,
                    tool_name=outcome.call.name,
                    content=outcome.result.content,
                    structured=deepcopy(outcome.result.structured),
                    artifacts=deepcopy(outcome.result.artifacts),
                    is_error=outcome.result.is_error,
                )
                for outcome in outcomes
            ],
        )
    ]
