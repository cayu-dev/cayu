from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventType(StrEnum):
    SESSION_STARTED = "session.started"
    SESSION_COMPLETED = "session.completed"
    SESSION_FAILED = "session.failed"
    SESSION_CHECKPOINTED = "session.checkpointed"

    MODEL_STARTED = "model.started"
    MODEL_TEXT_DELTA = "model.text.delta"
    MODEL_COMPLETED = "model.completed"
    MODEL_ERROR = "model.error"

    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_FAILED = "tool.call.failed"
    TOOL_CALL_BLOCKED = "tool.call.blocked"

    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_STEP_STARTED = "workflow.step.started"
    WORKFLOW_STEP_COMPLETED = "workflow.step.completed"
    WORKFLOW_COMPLETED = "workflow.completed"

    MEMORY_SEARCH = "memory.search"
    RUNNER_EXEC_STARTED = "runner.exec.started"
    RUNNER_EXEC_COMPLETED = "runner.exec.completed"


class Event(BaseModel):
    """Append-only runtime event.

    Events are the common language between terminal output, dashboard views,
    persistent sessions, webhooks, and hosted-platform adapters.
    """

    model_config = ConfigDict(extra="forbid")

    type: EventType | str
    session_id: str
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: str | None = None
    workflow_name: str | None = None
    tool_name: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: EventType | str) -> EventType | str:
        if isinstance(value, EventType):
            return value

        try:
            return EventType(value)
        except ValueError:
            pass

        if not value.startswith("custom.") or value == "custom.":
            raise ValueError(
                "Custom event types must use the 'custom.' namespace."
            )
        return value
