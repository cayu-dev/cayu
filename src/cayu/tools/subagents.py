from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping
from enum import StrEnum
from functools import partial
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, MessageRole, TextPart
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.runtime.sessions import (
    InterruptSessionRequest,
    RunRequest,
    Session,
    SessionOrder,
    SessionQuery,
    SessionStatus,
    SessionStore,
    TranscriptQuery,
)
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.tools._errors import structured_invalid_arguments

logger = logging.getLogger(__name__)

DEFAULT_SUBAGENT_RESULT_MAX_CHARS = 12_000
MAX_SUBAGENT_RESULT_MAX_CHARS = 200_000
DEFAULT_SUBAGENT_RESULT_WAIT_TIMEOUT_S = 30.0
MAX_SUBAGENT_RESULT_WAIT_TIMEOUT_S = 600.0
SUBAGENT_CANCEL_CLEANUP_TIMEOUT_S = 10.0
# Poll for terminal status with exponential backoff instead of a fixed tight
# interval: a background subagent may run for many seconds, and reloading every
# child every 50ms wastes store round-trips. Start responsive, then back off.
SUBAGENT_RESULT_POLL_MIN_INTERVAL_S = 0.05
SUBAGENT_RESULT_POLL_MAX_INTERVAL_S = 1.0
# Retained for backward compatibility with callers importing the old name.
SUBAGENT_RESULT_POLL_INTERVAL_S = SUBAGENT_RESULT_POLL_MIN_INTERVAL_S
# Page size for enumerating a parent's background children. ``SessionQuery.limit``
# caps at 1000; we keyset-paginate rather than truncate at a single page.
_SUBAGENT_CHILD_LIST_PAGE_SIZE = 1000
_SUBAGENT_TERMINAL_STATUSES = {
    SessionStatus.COMPLETED,
    SessionStatus.FAILED,
    SessionStatus.INTERRUPTED,
}
BACKGROUND_SUBAGENT_FAILURE_ARTIFACT_TYPE = "cayu.subagent_background_failure.v1"
_MAX_BACKGROUND_FAILURE_RECORDS = 1000


class BackgroundSubagentTaskRegistry:
    """Strong-reference registry for background subagent drain tasks.

    ``asyncio`` only keeps weak references to running tasks, so background
    subagent drains must be pinned here until they finish or the parent is
    torn down (``cancel_parent``). Drain failures are logged and recorded so
    ``subagent_result`` can report them instead of the error being silently
    swallowed.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._children_by_parent: dict[str, set[str]] = {}
        self._failures: dict[str, dict[str, Any]] = {}

    def register(
        self,
        task: asyncio.Task[None],
        *,
        parent_session_id: str,
        child_session_id: str,
    ) -> None:
        parent_session_id = require_clean_nonblank(parent_session_id, "parent_session_id")
        child_session_id = require_clean_nonblank(child_session_id, "child_session_id")
        if not isinstance(task, asyncio.Task):
            raise TypeError("Background subagent registry requires an asyncio.Task.")
        if child_session_id in self._tasks:
            raise ValueError(f"Background subagent task already registered: {child_session_id}")
        self._tasks[child_session_id] = task
        self._children_by_parent.setdefault(parent_session_id, set()).add(child_session_id)
        task.add_done_callback(
            partial(
                self._finalize_task,
                parent_session_id=parent_session_id,
                child_session_id=child_session_id,
            )
        )

    def active_tasks(self, parent_session_id: str) -> tuple[asyncio.Task[None], ...]:
        child_ids = self._children_by_parent.get(parent_session_id, set())
        return tuple(
            self._tasks[child_id] for child_id in sorted(child_ids) if child_id in self._tasks
        )

    def failure(self, child_session_id: str) -> dict[str, Any] | None:
        record = self._failures.get(child_session_id)
        if record is None:
            return None
        return copy_json_value(record, "background_failure")

    async def cancel_parent(self, parent_session_id: str) -> None:
        """Cancel and drain every background subagent task for a parent session."""
        tasks = self.active_tasks(parent_session_id)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _finalize_task(
        self,
        task: asyncio.Task[None],
        *,
        parent_session_id: str,
        child_session_id: str,
    ) -> None:
        self._tasks.pop(child_session_id, None)
        children = self._children_by_parent.get(parent_session_id)
        if children is not None:
            children.discard(child_session_id)
            if not children:
                self._children_by_parent.pop(parent_session_id, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        self._failures[child_session_id] = {
            "type": BACKGROUND_SUBAGENT_FAILURE_ARTIFACT_TYPE,
            "parent_session_id": parent_session_id,
            "child_session_id": child_session_id,
            "error": str(error),
            "error_type": type(error).__name__,
        }
        while len(self._failures) > _MAX_BACKGROUND_FAILURE_RECORDS:
            del self._failures[next(iter(self._failures))]
        logger.error(
            "Background subagent %s (parent %s) failed while draining runtime events.",
            child_session_id,
            parent_session_id,
            exc_info=error,
        )


_default_background_registry = BackgroundSubagentTaskRegistry()


def default_background_subagent_registry() -> BackgroundSubagentTaskRegistry:
    """Shared registry used when subagent tools are not given an explicit one."""
    return _default_background_registry


class _SubagentCancelledError(asyncio.CancelledError):
    """Cancelled subagent execution with optional cleanup diagnostics."""

    def __init__(
        self,
        message: str = "Subagent execution was cancelled.",
        *,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.artifacts = copy_json_value([] if artifacts is None else artifacts, "artifacts")


class SubagentContextMode(StrEnum):
    TASK_ONLY = "task_only"


class SubagentExecutionMode(StrEnum):
    FOREGROUND = "foreground"
    BACKGROUND = "background"


class SubagentSpec(BaseModel):
    """Model-facing subagent target backed by a Cayu agent registration."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    description: str = ""
    context_mode: SubagentContextMode = SubagentContextMode.TASK_ONLY
    mode: SubagentExecutionMode = SubagentExecutionMode.FOREGROUND
    max_steps: StrictInt = Field(default=16, ge=1, le=256)
    result_max_chars: StrictInt = Field(
        default=DEFAULT_SUBAGENT_RESULT_MAX_CHARS,
        ge=1,
        le=MAX_SUBAGENT_RESULT_MAX_CHARS,
    )
    limits: RunLimits = Field(default_factory=RunLimits)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_name")
    @classmethod
    def validate_agent_name(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str, info) -> str:
        if value == "":
            return value
        return require_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("limits")
    @classmethod
    def copy_limits(cls, value: RunLimits) -> RunLimits:
        return copy_run_limits(value)


class SubagentRuntime(Protocol):
    def run(self, request: RunRequest) -> AsyncIterator[Event]:
        """Run a child Cayu session and stream its events."""

    def interrupt_session(self, request: InterruptSessionRequest) -> AsyncIterator[Event]:
        """Interrupt a child Cayu session and stream interruption events."""


class SubagentTool(Tool):
    """Model-facing delegation tool backed by normal Cayu child sessions."""

    def __init__(
        self,
        runtime: SubagentRuntime,
        *,
        agents: Mapping[str, SubagentSpec | str],
        name: str = "subagent",
        description: str | None = None,
        background_registry: BackgroundSubagentTaskRegistry | None = None,
    ) -> None:
        if background_registry is not None and not isinstance(
            background_registry, BackgroundSubagentTaskRegistry
        ):
            raise TypeError("background_registry must be a BackgroundSubagentTaskRegistry.")
        self._runtime = runtime
        self._background_registry = (
            background_registry
            if background_registry is not None
            else default_background_subagent_registry()
        )
        self._agents = _copy_subagent_specs(agents)
        aliases = sorted(self._agents)
        if not aliases:
            raise ValueError("SubagentTool requires at least one subagent.")
        tool_description = description or _subagent_tool_description(self._agents)
        super().__init__(
            ToolSpec(
                name=require_clean_nonblank(name, "name"),
                description=tool_description,
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "enum": aliases,
                            "description": "Subagent to run.",
                        },
                        "task": {
                            "type": "string",
                            "description": "Delegated task for the subagent.",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional JSON metadata for the child session.",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["agent", "task"],
                    "additionalProperties": False,
                },
            )
        )

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        agent_alias = _string_argument(args, "agent", clean=True)
        task = _string_argument(args, "task", clean=False)
        raw_metadata = args.get("metadata", {})
        metadata = copy_json_value({} if raw_metadata is None else raw_metadata, "metadata")
        spec = self._agents.get(agent_alias)
        if spec is None:
            return ToolResult(
                content=f"Unknown subagent: {agent_alias}",
                structured={
                    "agent": agent_alias,
                    "available_agents": sorted(self._agents),
                },
                is_error=True,
            )
        if spec.context_mode != SubagentContextMode.TASK_ONLY:
            return ToolResult(
                content=f"Unsupported subagent context mode: {spec.context_mode.value}",
                structured={
                    "agent": agent_alias,
                    "context_mode": spec.context_mode.value,
                },
                is_error=True,
            )

        child_session_id = f"{ctx.session_id}_subagent_{uuid4().hex[:8]}"
        causal_budget_id = ctx.causal_budget_id or ctx.session_id
        request = RunRequest(
            agent_name=spec.agent_name,
            session_id=child_session_id,
            parent_session_id=ctx.session_id,
            causal_budget_id=causal_budget_id,
            environment_name=ctx.environment_name,
            messages=[Message.text("user", task)],
            metadata={
                **copy_json_value(spec.metadata, "metadata"),
                **metadata,
                "subagent": {
                    "agent": agent_alias,
                    "agent_name": spec.agent_name,
                    "context_mode": spec.context_mode.value,
                    "mode": spec.mode.value,
                    "parent_session_id": ctx.session_id,
                },
            },
            max_steps=spec.max_steps,
            limits=spec.limits,
        )
        structured = _subagent_result_payload(
            agent_alias=agent_alias,
            spec=spec,
            parent_session_id=ctx.session_id,
            child_session_id=child_session_id,
            causal_budget_id=causal_budget_id,
        )
        if spec.mode == SubagentExecutionMode.BACKGROUND:
            try:
                first_event = await _start_background_subagent(
                    self._runtime.run(request),
                    registry=self._background_registry,
                    parent_session_id=ctx.session_id,
                    child_session_id=child_session_id,
                )
            except Exception as exc:
                return ToolResult(
                    content=f"Subagent {agent_alias} could not be started: {exc}",
                    structured={
                        **structured,
                        "status": "start_failed",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                    is_error=True,
                )
            return ToolResult(
                content=(f"Subagent {agent_alias} started in background as {child_session_id}."),
                structured={
                    **structured,
                    "status": "started",
                    "first_event_type": str(first_event.type),
                    "events": 1,
                },
            )

        child_task = asyncio.create_task(
            _collect_subagent_result(
                self._runtime.run(request),
                max_chars=spec.result_max_chars,
            )
        )
        try:
            result = await asyncio.shield(child_task)
        except asyncio.CancelledError:
            _uncancel_current_task()
            cleanup_error: Exception | None = None
            try:
                async with asyncio.timeout(SUBAGENT_CANCEL_CLEANUP_TIMEOUT_S):
                    await _interrupt_child_session(
                        runtime=self._runtime,
                        child_session_id=child_session_id,
                        child_task=child_task,
                    )
            except Exception as exc:
                cleanup_error = exc
            finally:
                # Backstop: the child collector must never outlive the tool
                # call, even when cleanup itself timed out or was cancelled.
                if not child_task.done():
                    child_task.cancel()
            if cleanup_error is not None:
                raise _SubagentCancelledError(
                    artifacts=[
                        {
                            "type": "cayu.subagent_cleanup_error.v1",
                            "child_session_id": child_session_id,
                            "error": str(cleanup_error),
                            "error_type": type(cleanup_error).__name__,
                        }
                    ]
                ) from cleanup_error
            raise
        structured = {
            **structured,
            "status": None if result.terminal is None else str(result.terminal.type),
            "events": result.event_count,
            "result_truncated": result.text_truncated,
        }
        if result.terminal is None:
            return ToolResult(
                content="Subagent finished without a terminal session event.",
                structured=structured,
                is_error=True,
            )
        if result.terminal.type == EventType.SESSION_COMPLETED:
            return ToolResult(
                content=result.text or f"Subagent {agent_alias} completed.",
                structured=structured,
            )
        error = (
            result.terminal.payload.get("error")
            if isinstance(result.terminal.payload, dict)
            else None
        )
        return ToolResult(
            content=str(
                error or f"Subagent {agent_alias} did not complete: {result.terminal.type}"
            ),
            structured={
                **structured,
                "terminal_payload": copy_json_value(
                    result.terminal.payload,
                    "terminal_payload",
                ),
            },
            is_error=True,
        )


class SubagentResultTool(Tool):
    """Fetch or wait for background subagent results from durable child sessions."""

    def __init__(
        self,
        session_store: SessionStore,
        *,
        name: str = "subagent_result",
        description: str | None = None,
        default_timeout_s: float = DEFAULT_SUBAGENT_RESULT_WAIT_TIMEOUT_S,
        background_registry: BackgroundSubagentTaskRegistry | None = None,
    ) -> None:
        if background_registry is not None and not isinstance(
            background_registry, BackgroundSubagentTaskRegistry
        ):
            raise TypeError("background_registry must be a BackgroundSubagentTaskRegistry.")
        if not isinstance(default_timeout_s, int | float) or isinstance(default_timeout_s, bool):
            raise TypeError("default_timeout_s must be a number.")
        if default_timeout_s < 0 or default_timeout_s > MAX_SUBAGENT_RESULT_WAIT_TIMEOUT_S:
            raise ValueError(
                "default_timeout_s must be between 0 and "
                f"{MAX_SUBAGENT_RESULT_WAIT_TIMEOUT_S:g} seconds."
            )
        self._session_store = session_store
        self._background_registry = (
            background_registry
            if background_registry is not None
            else default_background_subagent_registry()
        )
        self._default_timeout_s = float(default_timeout_s)
        super().__init__(
            ToolSpec(
                name=require_clean_nonblank(name, "name"),
                description=description
                or (
                    "Fetch results from background Cayu subagents. Use child_session_id "
                    "for one child, or all=true to wait for every background subagent "
                    "started by the current session."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "child_session_id": {
                            "type": "string",
                            "description": "Background child session id returned by subagent.",
                        },
                        "all": {
                            "type": "boolean",
                            "description": "When true, fetch all background subagents for this session.",
                            "default": False,
                        },
                        "wait": {
                            "type": "boolean",
                            "description": "Wait until requested subagent work reaches a terminal status.",
                            "default": True,
                        },
                        "timeout_s": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": MAX_SUBAGENT_RESULT_WAIT_TIMEOUT_S,
                            "description": "Maximum wait time in seconds.",
                            "default": self._default_timeout_s,
                        },
                        "max_chars": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_SUBAGENT_RESULT_MAX_CHARS,
                            "description": "Maximum assistant-result characters per child.",
                            "default": DEFAULT_SUBAGENT_RESULT_MAX_CHARS,
                        },
                    },
                    "additionalProperties": False,
                },
            )
        )

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        all_children = _bool_argument(args, "all", default=False)
        wait = _bool_argument(args, "wait", default=True)
        timeout_s = _timeout_argument(
            args,
            "timeout_s",
            default=self._default_timeout_s,
        )
        max_chars = _int_argument(
            args,
            "max_chars",
            default=DEFAULT_SUBAGENT_RESULT_MAX_CHARS,
            minimum=1,
            maximum=MAX_SUBAGENT_RESULT_MAX_CHARS,
        )
        child_session_id = args.get("child_session_id")
        if all_children:
            if child_session_id is not None:
                return ToolResult(
                    content="Use either child_session_id or all=true, not both.",
                    structured={"all": all_children},
                    is_error=True,
                )
            return await self._run_all(
                ctx=ctx,
                wait=wait,
                timeout_s=timeout_s,
                max_chars=max_chars,
            )
        if not isinstance(child_session_id, str):
            return ToolResult(
                content="subagent_result requires child_session_id unless all=true.",
                structured={"all": all_children},
                is_error=True,
            )
        return await self._run_one(
            ctx=ctx,
            child_session_id=require_clean_nonblank(child_session_id, "child_session_id"),
            wait=wait,
            timeout_s=timeout_s,
            max_chars=max_chars,
        )

    async def _run_one(
        self,
        *,
        ctx: ToolContext,
        child_session_id: str,
        wait: bool,
        timeout_s: float,
        max_chars: int,
    ) -> ToolResult:
        loaded_child = await self._load_authorized_subagent_child(ctx, child_session_id)
        if isinstance(loaded_child, ToolResult):
            return loaded_child
        child = await _wait_for_subagent_terminal(
            self._session_store,
            loaded_child,
            wait=wait,
            timeout_s=timeout_s,
        )
        summary = await _summarize_child_session(
            self._session_store,
            child,
            max_chars=max_chars,
            background_failure=self._background_registry.failure(child.id),
        )
        return _tool_result_from_child_summary(summary)

    async def _run_all(
        self,
        *,
        ctx: ToolContext,
        wait: bool,
        timeout_s: float,
        max_chars: int,
    ) -> ToolResult:
        children = await _list_background_subagent_children(
            self._session_store,
            parent_session_id=ctx.session_id,
        )
        if not children:
            return ToolResult(
                content="No background subagents were started by this session.",
                structured={
                    "parent_session_id": ctx.session_id,
                    "children": [],
                    "retrieval_status": "empty",
                },
            )
        if wait:
            children = await _wait_for_all_subagents_terminal(
                self._session_store,
                children,
                timeout_s=timeout_s,
            )
        summaries = [
            await _summarize_child_session(
                self._session_store,
                child,
                max_chars=max_chars,
                background_failure=self._background_registry.failure(child.id),
            )
            for child in children
        ]
        lines = ["Background subagent results:"]
        has_error = False
        for summary in summaries:
            status = summary["status"]
            child_id = summary["child_session_id"]
            agent = summary.get("agent") or summary.get("agent_name") or "subagent"
            if summary["retrieval_status"] == "ready":
                text = summary["result_text"] or f"Subagent ended with status {status}."
                lines.append(f"- {agent} ({child_id}, {status}): {text}")
            elif summary.get("background_failure") is not None:
                failure = summary["background_failure"]
                lines.append(
                    f"- {agent} ({child_id}, {status}): background failure: {failure['error']}"
                )
                has_error = True
            else:
                lines.append(f"- {agent} ({child_id}, {status}): still running")
            if summary.get("is_error") is True:
                has_error = True
        return ToolResult(
            content="\n".join(lines),
            structured={
                "parent_session_id": ctx.session_id,
                "retrieval_status": "ready"
                if all(summary["retrieval_status"] == "ready" for summary in summaries)
                else "not_ready",
                "children": summaries,
            },
            is_error=has_error,
        )

    async def _load_authorized_subagent_child(
        self,
        ctx: ToolContext,
        child_session_id: str,
    ) -> Session | ToolResult:
        child = await self._session_store.load(child_session_id)
        if child is None:
            return ToolResult(
                content=f"Subagent session not found: {child_session_id}",
                structured={"child_session_id": child_session_id},
                is_error=True,
            )
        if child.parent_session_id != ctx.session_id:
            return ToolResult(
                content="Subagent result is not available to this parent session.",
                structured={
                    "child_session_id": child_session_id,
                    "parent_session_id": ctx.session_id,
                },
                is_error=True,
            )
        if not _is_background_subagent_session(child):
            return ToolResult(
                content=f"Session is not a background subagent child: {child_session_id}",
                structured={"child_session_id": child_session_id},
                is_error=True,
            )
        return child


def _copy_subagent_specs(
    agents: Mapping[str, SubagentSpec | str],
) -> dict[str, SubagentSpec]:
    if not isinstance(agents, Mapping):
        raise TypeError("SubagentTool agents must be a mapping.")
    copied: dict[str, SubagentSpec] = {}
    for alias, spec in agents.items():
        clean_alias = require_clean_nonblank(alias, "agents.alias")
        if type(spec) is str:
            copied[clean_alias] = SubagentSpec(agent_name=spec)
        elif type(spec) is SubagentSpec:
            copied[clean_alias] = spec.model_copy(deep=True)
        else:
            raise TypeError("SubagentTool agents must map aliases to SubagentSpec or agent names.")
    return copied


def _string_argument(args: dict[str, Any], field_name: str, *, clean: bool) -> str:
    value = args.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"SubagentTool argument {field_name!r} must be a string.")
    if clean:
        return require_clean_nonblank(value, field_name)
    return require_nonblank(value, field_name)


def _bool_argument(args: dict[str, Any], field_name: str, *, default: bool) -> bool:
    value = args.get(field_name, default)
    if not isinstance(value, bool):
        raise ValueError(f"SubagentResultTool argument {field_name!r} must be a bool.")
    return value


def _timeout_argument(args: dict[str, Any], field_name: str, *, default: float) -> float:
    value = args.get(field_name, default)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"SubagentResultTool argument {field_name!r} must be a number.")
    timeout_s = float(value)
    if timeout_s < 0 or timeout_s > MAX_SUBAGENT_RESULT_WAIT_TIMEOUT_S:
        raise ValueError(
            f"SubagentResultTool argument {field_name!r} must be between 0 and "
            f"{MAX_SUBAGENT_RESULT_WAIT_TIMEOUT_S:g}."
        )
    return timeout_s


def _int_argument(
    args: dict[str, Any],
    field_name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = args.get(field_name, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"SubagentResultTool argument {field_name!r} must be an integer.")
    if value < minimum or value > maximum:
        raise ValueError(
            f"SubagentResultTool argument {field_name!r} must be between {minimum} and {maximum}."
        )
    return value


def _subagent_tool_description(agents: Mapping[str, SubagentSpec]) -> str:
    lines = ["Delegate bounded work to a configured Cayu subagent."]
    for alias, spec in sorted(agents.items()):
        detail = f"{alias}: {spec.agent_name} ({spec.mode.value})"
        if spec.description:
            detail = f"{detail} - {spec.description}"
        lines.append(detail)
    return "\n".join(lines)


def _subagent_result_payload(
    *,
    agent_alias: str,
    spec: SubagentSpec,
    parent_session_id: str,
    child_session_id: str,
    causal_budget_id: str,
) -> dict[str, Any]:
    return {
        "agent": agent_alias,
        "agent_name": spec.agent_name,
        "context_mode": spec.context_mode.value,
        "mode": spec.mode.value,
        "parent_session_id": parent_session_id,
        "child_session_id": child_session_id,
        "causal_budget_id": causal_budget_id,
        "result_max_chars": spec.result_max_chars,
    }


class _SubagentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    text_truncated: bool
    event_count: int
    terminal: Event | None


async def _list_background_subagent_children(
    session_store: SessionStore,
    *,
    parent_session_id: str,
) -> list[Session]:
    children: list[Session] = []
    cursor: str | None = None
    while True:
        result = await session_store.list_sessions(
            SessionQuery(
                parent_session_id=parent_session_id,
                limit=_SUBAGENT_CHILD_LIST_PAGE_SIZE,
                cursor=cursor,
                order_by=SessionOrder.CREATED_AT_ASC,
            )
        )
        children.extend(
            session for session in result.sessions if _is_background_subagent_session(session)
        )
        cursor = result.next_cursor
        if cursor is None:
            break
    return children


def _is_background_subagent_session(session: Session) -> bool:
    subagent = session.metadata.get("subagent")
    return (
        isinstance(subagent, dict)
        and subagent.get("mode") == SubagentExecutionMode.BACKGROUND.value
    )


async def _wait_for_subagent_terminal(
    session_store: SessionStore,
    child: Session,
    *,
    wait: bool,
    timeout_s: float,
) -> Session:
    if not wait or child.status in _SUBAGENT_TERMINAL_STATUSES:
        return child
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    loaded = child
    delay = SUBAGENT_RESULT_POLL_MIN_INTERVAL_S
    while loaded.status not in _SUBAGENT_TERMINAL_STATUSES:
        now = loop.time()
        if now >= deadline:
            return loaded
        await asyncio.sleep(min(delay, deadline - now))
        refreshed = await session_store.load(loaded.id)
        if refreshed is None:
            raise RuntimeError(f"Subagent session disappeared: {loaded.id}")
        loaded = refreshed
        delay = min(delay * 2, SUBAGENT_RESULT_POLL_MAX_INTERVAL_S)
    return loaded


async def _wait_for_all_subagents_terminal(
    session_store: SessionStore,
    children: list[Session],
    *,
    timeout_s: float,
) -> list[Session]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    loaded_by_id = {child.id: child for child in children}
    # Only children that have not yet reached a terminal status need reloading;
    # terminal children are done and are never polled again.
    pending = {
        child_id
        for child_id, child in loaded_by_id.items()
        if child.status not in _SUBAGENT_TERMINAL_STATUSES
    }
    delay = SUBAGENT_RESULT_POLL_MIN_INTERVAL_S
    while pending:
        now = loop.time()
        if now >= deadline:
            break
        await asyncio.sleep(min(delay, deadline - now))
        for child_id in list(pending):
            refreshed = await session_store.load(child_id)
            if refreshed is None:
                raise RuntimeError(f"Subagent session disappeared: {child_id}")
            loaded_by_id[child_id] = refreshed
            if refreshed.status in _SUBAGENT_TERMINAL_STATUSES:
                pending.discard(child_id)
        delay = min(delay * 2, SUBAGENT_RESULT_POLL_MAX_INTERVAL_S)
    return list(loaded_by_id.values())


async def _summarize_child_session(
    session_store: SessionStore,
    child: Session,
    *,
    max_chars: int,
    background_failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Tail-limited retrieval: fetch only the last assistant message and an event
    # summary/outcome instead of reloading the child's full transcript and every
    # event, which scales badly for long-running subagents.
    result_text, result_truncated = await _load_last_assistant_text(
        session_store, child.id, max_chars=max_chars
    )
    event_summary = await session_store.summarize_events(child.id)
    outcome = await session_store.summarize_outcome(child.id)
    terminal_record = outcome.terminal_event
    terminal_event = terminal_record.event if terminal_record is not None else None
    subagent = child.metadata.get("subagent")
    subagent_metadata = subagent if isinstance(subagent, dict) else {}
    terminal_payload = (
        copy_json_value(terminal_event.payload, "terminal_payload")
        if terminal_event is not None
        else None
    )
    ready = child.status in _SUBAGENT_TERMINAL_STATUSES
    return {
        "agent": subagent_metadata.get("agent"),
        "agent_name": child.agent_name,
        "child_session_id": child.id,
        "parent_session_id": child.parent_session_id,
        "causal_budget_id": child.causal_budget_id,
        "status": child.status.value,
        "retrieval_status": "ready" if ready else "not_ready",
        "result_text": result_text,
        "result_truncated": result_truncated,
        "events": event_summary.total_events,
        "terminal_event_type": None if terminal_event is None else str(terminal_event.type),
        "terminal_payload": terminal_payload,
        "background_failure": (
            None
            if background_failure is None
            else copy_json_value(background_failure, "background_failure")
        ),
        "is_error": child.status in {SessionStatus.FAILED, SessionStatus.INTERRUPTED},
    }


async def _load_last_assistant_text(
    session_store: SessionStore,
    session_id: str,
    *,
    max_chars: int,
) -> tuple[str, bool]:
    """Return the last assistant message's text, tail-truncated to ``max_chars``.

    Uses a role-filtered, offset-based transcript query so only the final
    assistant message is materialized instead of the whole transcript.
    """
    head = await session_store.query_transcript(
        TranscriptQuery(session_id=session_id, role=MessageRole.ASSISTANT, offset=0, limit=1)
    )
    total = head.total_records
    if total == 0:
        return "", False
    if total == 1:
        records = head.records
    else:
        tail = await session_store.query_transcript(
            TranscriptQuery(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                offset=total - 1,
                limit=1,
            )
        )
        records = tail.records
    if not records:
        return "", False
    message = records[-1].message
    text = "".join(part.text for part in message.content if type(part) is TextPart).strip()
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _tool_result_from_child_summary(summary: dict[str, Any]) -> ToolResult:
    status = summary["status"]
    if summary["retrieval_status"] != "ready":
        background_failure = summary.get("background_failure")
        if background_failure is not None:
            return ToolResult(
                content=(
                    f"Subagent {summary['child_session_id']} background execution failed: "
                    f"{background_failure['error']}"
                ),
                structured=summary,
                is_error=True,
            )
        return ToolResult(
            content=(
                f"Subagent {summary['child_session_id']} is still running with status {status}."
            ),
            structured=summary,
        )
    if summary["is_error"]:
        terminal_payload = summary.get("terminal_payload")
        error = terminal_payload.get("error") if isinstance(terminal_payload, dict) else None
        return ToolResult(
            content=str(error or f"Subagent ended with status {status}."),
            structured=summary,
            is_error=True,
        )
    result_text = summary["result_text"] or f"Subagent completed with status {status}."
    return ToolResult(content=result_text, structured=summary)


async def _collect_subagent_result(
    events: AsyncIterator[Event],
    *,
    max_chars: int,
) -> _SubagentResult:
    # Mirror the retrieval path (`_load_last_assistant_text`): the result is the LAST
    # assistant message's text, tail-truncated to `max_chars`. Accumulate deltas
    # per model turn and reset on each `MODEL_STARTED` so we keep only the final
    # message instead of the first `max_chars` of every turn concatenated.
    message_chunks: list[str] = []
    event_count = 0
    terminal: Event | None = None
    async for event in events:
        event_count += 1
        if event.type == EventType.MODEL_STARTED:
            message_chunks = []
        elif event.type == EventType.MODEL_TEXT_DELTA:
            delta = event.payload.get("delta")
            if isinstance(delta, str):
                message_chunks.append(delta)
        if event.type in {
            EventType.SESSION_COMPLETED,
            EventType.SESSION_FAILED,
            EventType.SESSION_INTERRUPTED,
        }:
            terminal = event
    text = "".join(message_chunks).strip()
    text_truncated = len(text) > max_chars
    if text_truncated:
        text = text[:max_chars]
    return _SubagentResult(
        text=text,
        text_truncated=text_truncated,
        event_count=event_count,
        terminal=terminal,
    )


async def _interrupt_child_session(
    *,
    runtime: SubagentRuntime,
    child_session_id: str,
    child_task: asyncio.Task[_SubagentResult],
) -> None:
    try:
        async for _event in runtime.interrupt_session(
            InterruptSessionRequest(
                session_id=child_session_id,
                reason="Parent session interrupted during subagent call.",
                metadata={"source": "subagent_tool"},
            )
        ):
            pass
    finally:
        if not child_task.done():
            child_task.cancel()
        # ``asyncio.wait`` (unlike ``await child_task``) never raises the
        # child's own exception and never swallows an outer cancellation, so
        # the bounded-cleanup timeout and parent cancellation stay effective
        # while we drain the collector.
        await asyncio.wait([child_task])
        if not child_task.cancelled():
            # Retrieve a failed collector's exception so asyncio does not log
            # "Task exception was never retrieved" during teardown.
            child_task.exception()


async def _start_background_subagent(
    events: AsyncIterator[Event],
    *,
    registry: BackgroundSubagentTaskRegistry,
    parent_session_id: str,
    child_session_id: str,
) -> Event:
    first_event: asyncio.Future[Event] = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(_drain_background_subagent(events, first_event))
    registry.register(
        task,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
    )
    return await first_event


async def _drain_background_subagent(
    events: AsyncIterator[Event],
    first_event: asyncio.Future[Event],
) -> None:
    try:
        async for _event in events:
            if not first_event.done():
                first_event.set_result(_event)
        if not first_event.done():
            first_event.set_exception(RuntimeError("Subagent produced no runtime events."))
    except Exception as exc:
        if not first_event.done():
            first_event.set_exception(exc)
        raise


def _uncancel_current_task() -> None:
    """Consume exactly the cancellation request this handler caught.

    Draining every pending request (``while cancelling(): uncancel()``) would
    strip cancellations owned by enclosing scopes such as ``asyncio.timeout``
    or ``TaskGroup`` and corrupt their bookkeeping, so at most one guarded
    ``uncancel`` is performed.
    """
    current_task = asyncio.current_task()
    if current_task is not None and current_task.cancelling() > 0:
        current_task.uncancel()
