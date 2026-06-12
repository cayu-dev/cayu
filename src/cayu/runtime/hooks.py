from __future__ import annotations

from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, Protocol

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.core.events import Event, copy_event
from cayu.core.tools import ToolResult
from cayu.runtime.dispatch import DispatchHandle, DispatchRequest, copy_dispatch_handle
from cayu.runtime.sessions import ForkSessionRequest, Session, copy_fork_session_request
from cayu.runtime.tasks import Task, TaskCreate, copy_task


class RuntimeHookPhase(StrEnum):
    AFTER_SESSION_COMPLETED = "after_session_completed"
    AFTER_SESSION_FAILED = "after_session_failed"
    AFTER_SESSION_INTERRUPTED = "after_session_interrupted"
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

    async def after_tool_call(self, context: ToolCallHookContext) -> None:
        """Run after a tool call result event has been persisted."""


def _copy_tool_result(result: ToolResult) -> ToolResult:
    if type(result) is not ToolResult:
        raise TypeError("Tool hook result requires a ToolResult.")
    return ToolResult(
        content=result.content,
        structured=copy_json_value(result.structured, "structured"),
        artifacts=copy_json_value(result.artifacts, "artifacts"),
        is_error=result.is_error,
    )
