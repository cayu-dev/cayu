from __future__ import annotations

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, model_validator

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.core.events import Event, EventType, copy_event
from cayu.core.tools import ToolResult
from cayu.runtime.dispatch import DispatchHandle, DispatchRequest, copy_dispatch_handle
from cayu.runtime.sessions import ForkSessionRequest, Session, copy_fork_session_request
from cayu.runtime.tasks import Task, TaskCreate, copy_task


class RuntimeHookPhase(StrEnum):
    AFTER_SESSION_COMPLETED = "after_session_completed"
    AFTER_SESSION_FAILED = "after_session_failed"
    AFTER_SESSION_INTERRUPTED = "after_session_interrupted"
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"


class RuntimeHookRuntime(Protocol):
    def fork_session(self, request: ForkSessionRequest) -> AsyncIterator[Event]:
        """Create a session fork and stream fork events."""

    async def dispatch(self, request: DispatchRequest) -> DispatchHandle:
        """Submit work for an existing session."""

    def dispatch_inline(self, request: DispatchRequest) -> AsyncIterator[Event]:
        """Run dispatched work inline and stream events."""

    async def create_task(self, request: TaskCreate) -> Task:
        """Create a durable task."""

    async def emit_hook_event(
        self,
        *,
        session_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        """Emit a custom event from a hook."""


class _HookActionContext:
    def __init__(
        self,
        *,
        runtime: RuntimeHookRuntime,
        hook_name: str,
        phase: RuntimeHookPhase,
        session: Session,
    ) -> None:
        self._runtime = runtime
        self._hook_name = require_clean_nonblank(hook_name, "hook_name")
        self._phase = phase
        self._session = session.model_copy(deep=True)
        self._actions: list[dict[str, Any]] = []

    @property
    def hook_name(self) -> str:
        return self._hook_name

    @property
    def phase(self) -> RuntimeHookPhase:
        return self._phase

    @property
    def session(self) -> Session:
        return self._session.model_copy(deep=True)

    @property
    def actions(self) -> list[dict[str, Any]]:
        return copy_json_value(self._actions, "actions")

    async def fork_session(self, request: ForkSessionRequest) -> list[Event]:
        request = copy_fork_session_request(request)
        events = [event async for event in self._runtime.fork_session(request)]
        child_session_id = events[-1].session_id if events else request.session_id
        self._record_action(
            "fork_session",
            {
                "source_session_id": request.source_session_id,
                "session_id": child_session_id,
                "events": len(events),
            },
        )
        return [copy_event(event) for event in events]

    async def dispatch(self, request: DispatchRequest) -> DispatchHandle:
        handle = await self._runtime.dispatch(request)
        self._record_action(
            "dispatch",
            {
                "dispatch_id": handle.dispatch_id,
                "session_id": handle.session_id,
                "task_id": handle.task_id,
                "backend": handle.backend,
                "status": handle.status.value,
            },
        )
        return copy_dispatch_handle(handle)

    async def dispatch_inline(self, request: DispatchRequest) -> list[Event]:
        events = [event async for event in self._runtime.dispatch_inline(request)]
        self._record_action(
            "dispatch_inline",
            {
                "dispatch_id": request.dispatch_id,
                "session_id": request.session_id,
                "task_id": request.task_id,
                "events": len(events),
            },
        )
        return [copy_event(event) for event in events]

    async def create_task(self, request: TaskCreate) -> Task:
        task = await self._runtime.create_task(request)
        self._record_action(
            "create_task",
            {
                "task_id": task.id,
                "type": task.type,
                "session_id": task.session_id,
                "assigned_agent_name": task.assigned_agent_name,
            },
        )
        return copy_task(task)

    async def emit_custom_event(
        self,
        event_type: str,
        *,
        session_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        emitted = await self._runtime.emit_hook_event(
            session_id=session_id or self._session.id,
            event_type=event_type,
            payload=payload,
        )
        self._record_action(
            "emit_custom_event",
            {
                "event_id": emitted.id,
                "event_type": str(emitted.type),
                "session_id": emitted.session_id,
            },
        )
        return copy_event(emitted)

    def _record_action(self, action_type: str, payload: dict[str, Any]) -> None:
        self._actions.append(
            {
                "type": action_type,
                "payload": copy_json_value(payload, "payload"),
            }
        )


class RuntimeHookContext(_HookActionContext):
    def __init__(
        self,
        *,
        runtime: RuntimeHookRuntime,
        hook_name: str,
        phase: RuntimeHookPhase,
        session: Session,
        terminal_event: Event,
    ) -> None:
        super().__init__(
            runtime=runtime,
            hook_name=hook_name,
            phase=phase,
            session=session,
        )
        self._terminal_event = copy_event(terminal_event)

    @property
    def terminal_event(self) -> Event:
        return copy_event(self._terminal_event)


class ToolCallHookContext(_HookActionContext):
    def __init__(
        self,
        *,
        runtime: RuntimeHookRuntime,
        hook_name: str,
        phase: RuntimeHookPhase,
        session: Session,
        tool_event: Event,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        result: ToolResult,
        task_id: str | None,
    ) -> None:
        super().__init__(
            runtime=runtime,
            hook_name=hook_name,
            phase=phase,
            session=session,
        )
        self._tool_event = copy_event(tool_event)
        self._tool_name = require_clean_nonblank(tool_name, "tool_name")
        self._tool_call_id = require_clean_nonblank(tool_call_id, "tool_call_id")
        self._arguments = copy_json_value(arguments, "arguments")
        self._result = _copy_tool_result(result)
        self._task_id = require_clean_nonblank(task_id, "task_id") if task_id is not None else None

    @property
    def tool_event(self) -> Event:
        return copy_event(self._tool_event)

    @property
    def tool_name(self) -> str:
        return self._tool_name

    @property
    def tool_call_id(self) -> str:
        return self._tool_call_id

    @property
    def arguments(self) -> dict[str, Any]:
        return copy_json_value(self._arguments, "arguments")

    @property
    def result(self) -> ToolResult:
        return _copy_tool_result(self._result)

    @property
    def task_id(self) -> str | None:
        return self._task_id


class BeforeToolCallHookContext(_HookActionContext):
    """The `before_tool_call` view of a call: identity + arguments, no result yet.

    Distinct from `ToolCallHookContext` because before execution there is no result or result
    event. `arguments` is a read-only copy; mutating it is a no-op — a before-hook changes the
    call only by returning a `BeforeToolCallDecision`.
    """

    def __init__(
        self,
        *,
        runtime: RuntimeHookRuntime,
        hook_name: str,
        phase: RuntimeHookPhase,
        session: Session,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        task_id: str | None,
    ) -> None:
        super().__init__(
            runtime=runtime,
            hook_name=hook_name,
            phase=phase,
            session=session,
        )
        self._tool_name = require_clean_nonblank(tool_name, "tool_name")
        self._tool_call_id = require_clean_nonblank(tool_call_id, "tool_call_id")
        self._arguments = copy_json_value(arguments, "arguments")
        self._task_id = require_clean_nonblank(task_id, "task_id") if task_id is not None else None

    @property
    def tool_name(self) -> str:
        return self._tool_name

    @property
    def tool_call_id(self) -> str:
        return self._tool_call_id

    @property
    def arguments(self) -> dict[str, Any]:
        return copy_json_value(self._arguments, "arguments")

    @property
    def task_id(self) -> str | None:
        return self._task_id


class BeforeToolCallDecision(BaseModel):
    """A `before_tool_call` hook's instruction for the call it just inspected.

    ``proceed`` runs the tool unchanged; ``proceed_modified`` runs it with ``modified_arguments``;
    ``short_circuit`` skips the tool and uses ``synthetic_result``; ``block`` skips the tool and
    returns an error result carrying ``block_reason``. Returning ``None`` from the hook equals
    ``proceed``.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["proceed", "proceed_modified", "short_circuit", "block"]
    modified_arguments: dict[str, Any] | None = None
    synthetic_result: ToolResult | None = None
    block_reason: str | None = None

    @model_validator(mode="after")
    def _check_payload(self) -> BeforeToolCallDecision:
        if self.action == "proceed_modified" and self.modified_arguments is None:
            raise ValueError("proceed_modified requires modified_arguments.")
        if self.action == "short_circuit" and self.synthetic_result is None:
            raise ValueError("short_circuit requires synthetic_result.")
        if self.action == "block" and not (self.block_reason and self.block_reason.strip()):
            raise ValueError("block requires a non-blank block_reason.")
        if self.action != "proceed_modified" and self.modified_arguments is not None:
            raise ValueError("modified_arguments is only valid with proceed_modified.")
        if self.action != "short_circuit" and self.synthetic_result is not None:
            raise ValueError("synthetic_result is only valid with short_circuit.")
        if self.action != "block" and self.block_reason is not None:
            raise ValueError("block_reason is only valid with block.")
        return self


class AfterToolCallDecision(BaseModel):
    """An `after_tool_call` hook's instruction for the result it just inspected.

    ``pass_through`` leaves the result unchanged; ``modify`` replaces it with ``modified_result``
    before the result enters the transcript. Returning ``None`` from the hook equals ``pass_through``.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["pass_through", "modify"]
    modified_result: ToolResult | None = None

    @model_validator(mode="after")
    def _check_payload(self) -> AfterToolCallDecision:
        if self.action == "modify" and self.modified_result is None:
            raise ValueError("modify requires modified_result.")
        if self.action == "pass_through" and self.modified_result is not None:
            raise ValueError("pass_through must not carry modified_result.")
        return self


class RuntimeHook:
    @property
    def name(self) -> str:
        return type(self).__name__

    async def after_session_completed(self, context: RuntimeHookContext) -> None:
        """Run after a session reaches completed state."""

    async def after_session_failed(self, context: RuntimeHookContext) -> None:
        """Run after a session reaches failed state."""

    async def after_session_interrupted(self, context: RuntimeHookContext) -> None:
        """Run after a session reaches interrupted state."""

    async def before_tool_call(
        self, context: BeforeToolCallHookContext
    ) -> BeforeToolCallDecision | None:
        """Run after policy authorization, before the tool executes.

        Return ``None`` to proceed unchanged, or a `BeforeToolCallDecision` to modify the
        arguments, short-circuit with a synthetic result, or block the call.
        """
        return None

    async def after_tool_call(self, context: ToolCallHookContext) -> AfterToolCallDecision | None:
        """Run after the tool executes, before its result enters the transcript.

        Return ``None`` to pass the result through unchanged, or an `AfterToolCallDecision` to
        replace it.
        """
        return None


def _runtime_hook_supports_phase(
    *,
    hook: RuntimeHook,
    phase: RuntimeHookPhase,
) -> bool:
    """Return whether a hook overrides the method for ``phase``."""

    method_name = _runtime_hook_method_name(phase)
    hook_method = getattr(type(hook), method_name)
    default_method = getattr(RuntimeHook, method_name)
    return hook_method is not default_method


def _runtime_hook_event(
    *,
    event_type: EventType,
    hook_name: str,
    scope: str,
    phase: RuntimeHookPhase,
    session: Session,
    terminal_event: Event,
    agent_name: str,
    environment_name: str | None,
    payload: dict[str, Any],
) -> Event:
    """Build the canonical lifecycle event for one runtime-hook invocation."""

    event_payload = {
        "hook_name": require_clean_nonblank(hook_name, "runtime_hook.name"),
        "scope": require_clean_nonblank(scope, "runtime_hook.scope"),
        "phase": phase.value,
        "terminal_event_id": terminal_event.id,
        "terminal_event_type": str(terminal_event.type),
        **copy_json_value(payload, "payload"),
    }
    return Event(
        type=event_type,
        session_id=session.id,
        agent_name=require_clean_nonblank(agent_name, "agent_name"),
        environment_name=environment_name,
        payload=event_payload,
    )


def _runtime_hook_method_name(phase: RuntimeHookPhase) -> str:
    if phase == RuntimeHookPhase.AFTER_SESSION_COMPLETED:
        return "after_session_completed"
    if phase == RuntimeHookPhase.AFTER_SESSION_FAILED:
        return "after_session_failed"
    if phase == RuntimeHookPhase.AFTER_SESSION_INTERRUPTED:
        return "after_session_interrupted"
    if phase == RuntimeHookPhase.BEFORE_TOOL_CALL:
        return "before_tool_call"
    if phase == RuntimeHookPhase.AFTER_TOOL_CALL:
        return "after_tool_call"
    raise ValueError(f"Unsupported runtime hook phase: {phase}")


def _copy_tool_result(result: ToolResult) -> ToolResult:
    if type(result) is not ToolResult:
        raise TypeError("Tool hook result requires a ToolResult.")
    return ToolResult(
        content=result.content,
        structured=copy_json_value(result.structured, "structured"),
        artifacts=copy_json_value(result.artifacts, "artifacts"),
        is_error=result.is_error,
    )
