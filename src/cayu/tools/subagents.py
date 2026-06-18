from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping
from enum import StrEnum
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.core.events import Event, EventType
from cayu.core.messages import Message
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.runtime.sessions import InterruptSessionRequest, RunRequest
from cayu.runtime.stop_policy import RunLimits, copy_run_limits

DEFAULT_SUBAGENT_RESULT_MAX_CHARS = 12_000
MAX_SUBAGENT_RESULT_MAX_CHARS = 200_000


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


class SubagentSpec(BaseModel):
    """Model-facing subagent target backed by a Cayu agent registration."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    description: str = ""
    context_mode: SubagentContextMode = SubagentContextMode.TASK_ONLY
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
    ) -> None:
        self._runtime = runtime
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
        request = RunRequest(
            agent_name=spec.agent_name,
            session_id=child_session_id,
            parent_session_id=ctx.session_id,
            causal_budget_id=ctx.causal_budget_id or ctx.session_id,
            environment_name=ctx.environment_name,
            messages=[Message.text("user", task)],
            metadata={
                **copy_json_value(spec.metadata, "metadata"),
                **metadata,
                "subagent": {
                    "agent": agent_alias,
                    "agent_name": spec.agent_name,
                    "context_mode": spec.context_mode.value,
                    "parent_session_id": ctx.session_id,
                },
            },
            max_steps=spec.max_steps,
            limits=spec.limits,
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
            _clear_current_task_cancellation()
            cleanup_error: Exception | None = None
            try:
                await _interrupt_child_session(
                    runtime=self._runtime,
                    child_session_id=child_session_id,
                    child_task=child_task,
                )
            except Exception as exc:
                cleanup_error = exc
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
            "agent": agent_alias,
            "agent_name": spec.agent_name,
            "context_mode": spec.context_mode.value,
            "parent_session_id": ctx.session_id,
            "child_session_id": child_session_id,
            "causal_budget_id": request.causal_budget_id,
            "status": None if result.terminal is None else str(result.terminal.type),
            "events": result.event_count,
            "result_truncated": result.text_truncated,
            "result_max_chars": spec.result_max_chars,
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


def _subagent_tool_description(agents: Mapping[str, SubagentSpec]) -> str:
    lines = ["Delegate bounded work to a configured Cayu subagent."]
    for alias, spec in sorted(agents.items()):
        detail = f"{alias}: {spec.agent_name}"
        if spec.description:
            detail = f"{detail} - {spec.description}"
        lines.append(detail)
    return "\n".join(lines)


class _SubagentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    text_truncated: bool
    event_count: int
    terminal: Event | None


async def _collect_subagent_result(
    events: AsyncIterator[Event],
    *,
    max_chars: int,
) -> _SubagentResult:
    chunks: list[str] = []
    remaining = max_chars
    text_truncated = False
    event_count = 0
    terminal: Event | None = None
    async for event in events:
        event_count += 1
        if event.type == EventType.MODEL_TEXT_DELTA:
            delta = event.payload.get("delta")
            if isinstance(delta, str):
                if remaining > 0:
                    chunk = delta[:remaining]
                    chunks.append(chunk)
                    remaining -= len(chunk)
                    if len(chunk) < len(delta):
                        text_truncated = True
                elif delta:
                    text_truncated = True
        if event.type in {
            EventType.SESSION_COMPLETED,
            EventType.SESSION_FAILED,
            EventType.SESSION_INTERRUPTED,
        }:
            terminal = event
    return _SubagentResult(
        text="".join(chunks).strip(),
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
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await child_task


def _clear_current_task_cancellation() -> None:
    current_task = asyncio.current_task()
    if current_task is None:
        return
    while current_task.cancelling():
        current_task.uncancel()
