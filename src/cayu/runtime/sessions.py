from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import copy_json_value, require_nonblank
from cayu.core.events import Event, copy_event
from cayu.core.messages import Message, copy_message


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
    # Optional caller-provided id for a new session. It must be unique.
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: StrictInt = Field(default=16, ge=1, le=256)

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        return [copy_message(message) for message in value]

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("agent_name")
    @classmethod
    def validate_nonblank_agent_name(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)

    @field_validator("session_id")
    @classmethod
    def validate_nonblank_session_id(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # SessionStore implementations may set this from RunRequest.session_id.
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_name: str
    status: SessionStatus = SessionStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("id", "agent_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)


class SessionStore(ABC):
    """Persistent store for sessions and append-only events."""

    @abstractmethod
    async def create(self, request: RunRequest) -> Session:
        """Create a session for a run request."""

    @abstractmethod
    async def load(self, session_id: str) -> Session | None:
        """Load a session by id."""

    @abstractmethod
    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        """Update session status and return the updated session."""

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


class InMemorySessionStore(SessionStore):
    """In-process session store for tests, local development, and examples."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, Session] = {}
        self._events: dict[str, list[Event]] = {}
        self._checkpoints: dict[str, dict[str, Any]] = {}

    async def create(self, request: RunRequest) -> Session:
        if type(request) is not RunRequest:
            raise TypeError("Session creation requires a RunRequest.")
        request = copy_run_request(request)
        async with self._lock:
            session_id = request.session_id or str(uuid4())
            if session_id in self._sessions:
                raise ValueError(f"Session already exists: {session_id}")

            now = datetime.now(timezone.utc)
            session = Session(
                id=session_id,
                agent_name=request.agent_name,
                status=SessionStatus.PENDING,
                created_at=now,
                updated_at=now,
                metadata=deepcopy(request.metadata),
            )
            self._sessions[session.id] = session
            self._events[session.id] = []
            return session.model_copy(deep=True)

    async def load(self, session_id: str) -> Session | None:
        session_id = require_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return session.model_copy(deep=True)

    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        session_id = require_nonblank(session_id, "session_id")
        if not isinstance(status, SessionStatus):
            raise ValueError("Session status must be a SessionStatus.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            updated = session.model_copy(
                update={
                    "status": status,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def append_event(self, session_id: str, event: Event) -> None:
        session_id = require_nonblank(session_id, "session_id")
        if type(event) is not Event:
            raise TypeError("Session events must be Event instances.")
        event = _validate_event(event)
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            if event.session_id != session_id:
                raise ValueError("Event session_id does not match target session.")
            self._events[session_id].append(event.model_copy(deep=True))

    async def load_events(self, session_id: str) -> list[Event]:
        session_id = require_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            return [
                event.model_copy(deep=True)
                for event in self._events.get(session_id, [])
            ]

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_nonblank(session_id, "session_id")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            self._checkpoints[session_id] = copy_json_value(state, "checkpoint")

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_nonblank(session_id, "session_id")
        async with self._lock:
            checkpoint = self._checkpoints.get(session_id)
            if checkpoint is None:
                return None
            return deepcopy(checkpoint)


def _validate_event(event: Event) -> Event:
    return copy_event(event)


def copy_run_request(request: RunRequest) -> RunRequest:
    if type(request) is not RunRequest:
        raise TypeError("Session creation requires a RunRequest.")
    messages = getattr(request, "messages", None)
    if type(messages) is not list:
        raise ValueError("RunRequest messages must be a list.")
    return RunRequest(
        agent_name=request.agent_name,
        messages=[copy_message(message) for message in messages],
        session_id=request.session_id,
        metadata=copy_json_value(request.metadata, "metadata"),
        max_steps=request.max_steps,
    )
