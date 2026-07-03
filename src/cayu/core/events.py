from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank

_CUSTOM_EVENT_TYPE_RE = re.compile(r"^custom\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*$")


class EventType(StrEnum):
    SESSION_STARTED = "session.started"
    SESSION_RESUMED = "session.resumed"
    SESSION_COMPLETED = "session.completed"
    SESSION_FAILED = "session.failed"
    SESSION_INTERRUPTED = "session.interrupted"
    SESSION_CHECKPOINTED = "session.checkpointed"
    SESSION_FORKED = "session.forked"
    SESSION_LIMIT_REACHED = "session.limit_reached"

    BUDGET_CHECKED = "budget.checked"
    BUDGET_LIMIT_REACHED = "budget.limit_reached"
    BUDGET_RESERVED = "budget.reserved"
    BUDGET_RECONCILED = "budget.reconciled"
    BUDGET_RESERVATION_FAILED = "budget.reservation_failed"
    BUDGET_RESERVATION_RELEASED = "budget.reservation_released"

    CREDENTIAL_PROXY_CHECKED = "credential.proxy.checked"

    MCP_MANIFEST_CHECKED = "mcp.manifest.checked"
    MCP_MANIFEST_BLOCKED = "mcp.manifest.blocked"

    TASK_CREATED = "task.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_CANCELLED = "task.cancelled"

    MODEL_STARTED = "model.started"
    MODEL_TEXT_DELTA = "model.text.delta"
    MODEL_THINKING_DELTA = "model.thinking.delta"
    MODEL_COMPLETED = "model.completed"
    MODEL_ERROR = "model.error"
    MODEL_RETRY = "model.retry"
    MODEL_ATTEMPT_DISCARDED = "model.attempt_discarded"

    STRUCTURED_OUTPUT_VALIDATED = "structured_output.validated"
    STRUCTURED_OUTPUT_FAILED = "structured_output.failed"
    STRUCTURED_OUTPUT_RETRY = "structured_output.retry"

    CONTEXT_COMPACTION_STARTED = "context.compaction.started"
    CONTEXT_COMPACTION_COMPLETED = "context.compaction.completed"
    CONTEXT_COMPACTION_FAILED = "context.compaction.failed"
    CONTEXT_COUNTED = "context.counted"
    CONTEXT_COUNT_FAILED = "context.count.failed"
    CONTEXT_COUNT_RECONCILED = "context.count.reconciled"
    CONTEXT_PRESSURE_ESTIMATED = "context.pressure.estimated"
    CONTEXT_PRESSURE_RECONCILED = "context.pressure.reconciled"
    CONTEXT_OVERFLOW_DETECTED = "context.overflow.detected"
    CONTEXT_OVERFLOW_RECOVERING = "context.overflow.recovering"
    CONTEXT_OVERFLOW_FAILED = "context.overflow.failed"

    KNOWLEDGE_SEARCH_STARTED = "knowledge.search.started"
    KNOWLEDGE_SEARCH_COMPLETED = "knowledge.search.completed"
    KNOWLEDGE_SEARCH_FAILED = "knowledge.search.failed"
    KNOWLEDGE_INJECTED = "knowledge.injected"

    ENVIRONMENT_BINDING_STARTED = "environment.binding.started"
    ENVIRONMENT_BINDING_COMPLETED = "environment.binding.completed"
    ENVIRONMENT_BINDING_FAILED = "environment.binding.failed"
    ENVIRONMENT_BINDING_FINALIZE_STARTED = "environment.binding.finalize_started"
    ENVIRONMENT_BINDING_FINALIZE_COMPLETED = "environment.binding.finalize_completed"
    ENVIRONMENT_BINDING_FINALIZE_FAILED = "environment.binding.finalize_failed"
    ENVIRONMENT_FACTORY_STARTED = "environment.factory.started"
    ENVIRONMENT_FACTORY_COMPLETED = "environment.factory.completed"
    ENVIRONMENT_FACTORY_FAILED = "environment.factory.failed"

    HOOK_STARTED = "hook.started"
    HOOK_COMPLETED = "hook.completed"
    HOOK_FAILED = "hook.failed"

    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"
    TOOL_CALL_BLOCKED = "tool.call.blocked"
    TOOL_CALL_APPROVAL_REQUESTED = "tool.call.approval_requested"
    TOOL_CALL_APPROVED = "tool.call.approved"
    TOOL_CALL_APPROVAL_DENIED = "tool.call.approval_denied"

    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_STEP_STARTED = "workflow.step.started"
    WORKFLOW_STEP_COMPLETED = "workflow.step.completed"
    WORKFLOW_COMPLETED = "workflow.completed"

    MEMORY_SEARCH = "memory.search"
    RUNNER_EXEC_STARTED = "runner.exec.started"
    RUNNER_EXEC_COMPLETED = "runner.exec.completed"

    RUNTIME_SINK_FAILED = "runtime.sink.failed"


class Event(BaseModel):
    """Append-only runtime event.

    Events are the common language between terminal output, dashboard views,
    persistent sessions, webhooks, and hosted-platform adapters.
    """

    model_config = ConfigDict(extra="forbid")

    type: EventType | str
    session_id: str
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    agent_name: str | None = None
    environment_name: str | None = None
    workflow_name: str | None = None
    tool_name: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload", mode="before")
    @classmethod
    def copy_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "payload")

    @field_validator("session_id", "id")
    @classmethod
    def validate_nonblank_ids(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("agent_name", "environment_name", "workflow_name", "tool_name")
    @classmethod
    def validate_optional_nonblank_names(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: EventType | str) -> EventType | str:
        if isinstance(value, EventType):
            return value

        try:
            return EventType(value)
        except ValueError:
            pass

        if not _CUSTOM_EVENT_TYPE_RE.fullmatch(value):
            raise ValueError(
                "Custom event types must use non-empty dot-separated segments "
                "in the 'custom.' namespace."
            )
        return value


def copy_event(event: Event) -> Event:
    if type(event) is not Event:
        raise TypeError("Events must be Event instances.")
    return Event(
        type=event.type,
        session_id=event.session_id,
        id=event.id,
        timestamp=event.timestamp,
        agent_name=event.agent_name,
        environment_name=event.environment_name,
        workflow_name=event.workflow_name,
        tool_name=event.tool_name,
        payload=copy_json_value(event.payload, "payload"),
    )
