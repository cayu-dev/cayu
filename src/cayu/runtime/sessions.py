from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from cayu.core.events import Event
from cayu.core.messages import Message


class SessionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str
    messages: list[Message]
    # Optional caller-provided id for resume/idempotency. SessionStore
    # implementations should use this as the Session.id when present.
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # SessionStore implementations may set this from RunRequest.session_id.
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_name: str
    status: SessionStatus = SessionStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionStore(ABC):
    """Persistent store for sessions and append-only events."""

    @abstractmethod
    async def create(self, request: RunRequest) -> Session:
        """Create a session for a run request."""

    @abstractmethod
    async def append_event(self, session_id: str, event: Event) -> None:
        """Append one event to a session."""

    @abstractmethod
    async def load_events(self, session_id: str) -> list[Event]:
        """Load all events for a session."""

    @abstractmethod
    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist a checkpoint for resume/replay."""

    @abstractmethod
    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        """Load the latest checkpoint for a session."""
