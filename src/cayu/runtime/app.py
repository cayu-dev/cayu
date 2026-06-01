from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from cayu._validation import copy_json_value, require_nonblank
from cayu.core.agents import AgentSpec
from cayu.core.events import Event, EventType, copy_event
from cayu.core.messages import Message, TextPart, ToolCallPart, ToolResultPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.environments import Environment, EnvironmentSpec, copy_environment
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    copy_model_stream_event,
)
from cayu.runtime.context import (
    ContextPolicy,
    ContextRequest,
    DefaultContextPolicy,
    copy_context_messages,
)
from cayu.runtime.event_sinks import EventSink
from cayu.runtime.sessions import (
    InMemorySessionStore,
    RunRequest,
    Session,
    SessionStatus,
    SessionStore,
    copy_run_request,
)
from cayu.runtime.tasks import Task, TaskStore


@dataclass(frozen=True)
class RegisteredAgent:
    spec: AgentSpec
    tools: Mapping[str, "RegisteredTool"]


@dataclass(frozen=True)
class _RegisteredAgentState:
    spec: AgentSpec
    tools: Mapping[str, "RegisteredTool"]
    context_policy: ContextPolicy


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


class CayuApp:
    """Application runtime for registered agents, providers, and session state."""

    def __init__(
        self,
        *,
        session_store: SessionStore | None = None,
        task_store: TaskStore | None = None,
        event_sinks: Iterable[EventSink] | None = None,
    ) -> None:
        if session_store is not None and not isinstance(session_store, SessionStore):
            raise TypeError("session_store must be a SessionStore.")
        if task_store is not None and not isinstance(task_store, TaskStore):
            raise TypeError("task_store must be a TaskStore.")
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
        self.session_store = (
            session_store if session_store is not None else InMemorySessionStore()
        )
        self.task_store = task_store
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
                raise ValueError(
                    f"Duplicate tool registered for agent: {registered_tool.name}"
                )
            tools_by_name[registered_tool.name] = registered_tool

        self._agents[stored_spec.name] = _RegisteredAgentState(
            spec=stored_spec,
            tools=MappingProxyType(tools_by_name),
            context_policy=stored_context_policy,
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
                name: _copy_registered_tool(tool)
                for name, tool in registered_agent.tools.items()
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

    async def run(self, request: RunRequest) -> AsyncIterator[Event]:
        if type(request) is not RunRequest:
            raise TypeError("Runtime run requires a RunRequest.")
        request = _validate_run_request(request)
        registered_agent = self._get_registered_agent(request.agent_name)
        registered_provider = self._get_registered_provider()
        registered_environment = self._get_registered_environment(
            request.environment_name
        )
        if request.environment_name is None and registered_environment is not None:
            request = _with_environment_name(request, registered_environment.spec.name)
        session = await self.session_store.create(request)
        await self.session_store.update_status(session.id, SessionStatus.RUNNING)

        async for event in self._run_session(
            session=session,
            request=request,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
        ):
            yield event

    async def _run_session(
        self,
        *,
        session: Session,
        request: RunRequest,
        registered_agent: _RegisteredAgentState,
        registered_provider: RegisteredProvider,
        registered_environment: RegisteredEnvironment | None,
    ) -> AsyncIterator[Event]:
        provider = registered_provider.provider
        environment_name = _environment_name(registered_environment)
        task_started = False
        task_finished = False
        try:
            yield await self._emit(
                Event(
                    type=EventType.SESSION_STARTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={"agent_name": registered_agent.spec.name},
                )
            )
            if request.task_id is not None:
                task = await self._start_task(task_id=request.task_id, session=session)
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
            messages = _initial_messages(
                system_prompt=registered_agent.spec.system_prompt,
                request_messages=request.messages,
            )
            await self.session_store.append_transcript_messages(session.id, messages)
            for step in range(1, request.max_steps + 1):
                context_messages = await _build_context_messages(
                    context_policy=registered_agent.context_policy,
                    session=session,
                    registered_agent=registered_agent,
                    messages=messages,
                    step=step,
                    environment_name=environment_name,
                    request_metadata=request.metadata,
                )

                model_request = ModelRequest(
                    model=registered_agent.spec.model,
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
                            "model": registered_agent.spec.model,
                            "provider": registered_provider.name,
                            "step": step,
                        },
                        environment_name=environment_name,
                    )
                )

                assistant_text: list[str] = []
                tool_calls: list[ToolCallRequest] = []
                model_completed = False
                async for raw_stream_event in provider.stream(model_request):
                    stream_event = _validate_stream_event(raw_stream_event)
                    if model_completed:
                        raise RuntimeError(
                            "Model provider emitted event after completed: "
                            f"{stream_event.type}"
                        )

                    if stream_event.type == ModelStreamEventType.TOOL_CALL:
                        tool_calls.append(_parse_tool_call(stream_event.payload))
                        continue

                    if stream_event.type == ModelStreamEventType.TEXT_DELTA:
                        assistant_text.append(stream_event.delta)
                    elif stream_event.type == ModelStreamEventType.COMPLETED:
                        model_completed = True

                    event = _validate_runtime_event(
                        provider.to_event(
                            stream_event,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                        ),
                        session_id=session.id,
                    )
                    event = _with_event_environment(event, environment_name)
                    yield await self._emit(event)
                    if stream_event.type == ModelStreamEventType.ERROR:
                        raise RuntimeError(
                            str(
                                stream_event.payload.get("error")
                                or "Model provider error"
                            )
                        )

                if not model_completed:
                    raise RuntimeError(
                        "Model provider stream ended without a completed event."
                    )

                assistant_message = _assistant_message(
                    text="".join(assistant_text),
                    tool_calls=tool_calls,
                )
                if assistant_message is not None:
                    messages.append(assistant_message)
                    await self.session_store.append_transcript_messages(
                        session.id,
                        [assistant_message],
                    )

                if not tool_calls:
                    break

                tool_outcomes: list[ToolCallOutcome] = []
                for tool_call in tool_calls:
                    outcome = None
                    async for event, outcome in self._execute_tool_call(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call=tool_call,
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
                raise RuntimeError(f"Maximum model steps exceeded: {request.max_steps}")

            if request.task_id is not None:
                task = await self._complete_task(
                    task_id=request.task_id,
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
            await self.session_store.update_status(session.id, SessionStatus.COMPLETED)
            yield await self._emit(
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                )
            )
        except Exception as exc:
            task_failure_error: Exception | None = None
            if (
                task_started
                and not task_finished
                and request.task_id is not None
                and self.task_store is not None
            ):
                try:
                    task = await self.task_store.fail_task(
                        request.task_id,
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
            await self.session_store.update_status(session.id, SessionStatus.FAILED)
            payload: dict[str, Any] = {
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            if task_failure_error is not None:
                payload["task_update_error"] = str(task_failure_error)
                payload["task_update_error_type"] = type(task_failure_error).__name__
            yield await self._emit(
                Event(
                    type=EventType.SESSION_FAILED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload=payload,
                )
            )

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

    async def _execute_tool_call(
        self,
        *,
        session: Session,
        registered_agent: _RegisteredAgentState,
        registered_environment: RegisteredEnvironment | None,
        tool_call: ToolCallRequest,
    ) -> AsyncIterator[tuple[Event, ToolCallOutcome | None]]:
        environment_name = _environment_name(registered_environment)
        yield (
            await self._emit(
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=tool_call.name,
                    payload={
                        "tool_call_id": tool_call.id,
                        "arguments": deepcopy(tool_call.arguments),
                    },
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
            yield (
                await self._emit(
                    Event(
                        type=EventType.TOOL_CALL_FAILED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        tool_name=tool_call.name,
                        payload={
                            "tool_call_id": tool_call.id,
                            "result": result.model_dump(),
                        },
                    )
                ),
                ToolCallOutcome(call=tool_call, result=result),
            )
            return

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
                metadata={"tool_call_id": tool_call.id},
            ),
            arguments=deepcopy(tool_call.arguments),
        )
        event_type = (
            EventType.TOOL_CALL_FAILED
            if result.is_error
            else EventType.TOOL_CALL_COMPLETED
        )
        yield (
            await self._emit(
                Event(
                    type=event_type,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    tool_name=tool_call.name,
                    payload={
                        "tool_call_id": tool_call.id,
                        "result": result.model_dump(),
                    },
                )
            ),
            ToolCallOutcome(call=tool_call, result=result),
        )

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


async def _build_context_messages(
    *,
    context_policy: ContextPolicy,
    session: Session,
    registered_agent: _RegisteredAgentState,
    messages: list[Message],
    step: int,
    environment_name: str | None,
    request_metadata: dict[str, Any],
) -> list[Message]:
    request = ContextRequest(
        session=session.model_copy(deep=True),
        agent=registered_agent.spec.model_copy(deep=True),
        messages=[message.model_copy(deep=True) for message in messages],
        step=step,
        environment_name=environment_name,
        metadata=copy_json_value(request_metadata, "metadata"),
    )
    result = await context_policy.build(request)
    return copy_context_messages(result)


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


def _with_event_environment(event: Event, environment_name: str | None) -> Event:
    if type(event) is not Event:
        raise TypeError("Runtime events must be Event instances.")
    return Event(
        type=event.type,
        session_id=event.session_id,
        id=event.id,
        timestamp=event.timestamp,
        agent_name=event.agent_name,
        environment_name=environment_name,
        workflow_name=event.workflow_name,
        tool_name=event.tool_name,
        payload=copy_json_value(event.payload, "payload"),
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
    if (
        registered_environment is None
        or registered_environment.environment.workspace is None
    ):
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
        return result.model_copy(
            update={"content": "Tool returned an error without details."}
        )
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


def _validate_runtime_event(value: object, *, session_id: str) -> Event:
    if type(value) is not Event:
        raise TypeError("Model providers must convert stream events to Event instances.")
    event = copy_event(value)
    if event.session_id != session_id:
        raise ValueError("Provider event session_id does not match current session.")
    return event


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
    text: str,
    tool_calls: list[ToolCallRequest],
) -> Message | None:
    content: list[TextPart | ToolCallPart] = []
    if text.strip():
        content.append(TextPart(text=text))
    content.extend(
        ToolCallPart(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            arguments=deepcopy(tool_call.arguments),
        )
        for tool_call in tool_calls
    )
    if not content:
        return None
    return Message(role="assistant", content=content)


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
