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
from cayu.core.events import Event, EventType, copy_event
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
    task_id: str | None = None
    environment_name: str | None = None
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

    @field_validator("session_id", "task_id", "environment_name")
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # SessionStore implementations may set this from RunRequest.session_id.
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_name: str
    environment_name: str | None = None
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

    @field_validator("environment_name")
    @classmethod
    def validate_nonblank_environment_name(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)


class SessionOrder(StrEnum):
    CREATED_AT_ASC = "created_at_asc"
    CREATED_AT_DESC = "created_at_desc"
    UPDATED_AT_ASC = "updated_at_asc"
    UPDATED_AT_DESC = "updated_at_desc"


class SessionQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: SessionStatus | None = None
    agent_name: str | None = None
    environment_name: str | None = None
    limit: StrictInt = Field(default=100, ge=1, le=1000)
    offset: StrictInt = Field(default=0, ge=0)
    order_by: SessionOrder = SessionOrder.UPDATED_AT_DESC

    @field_validator("agent_name", "environment_name")
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: StrictInt = Field(ge=1)
    event: Event

    @field_validator("event")
    @classmethod
    def copy_event(cls, value: Event) -> Event:
        return copy_event(value)


class EventQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    event_type: EventType | str | None = None
    agent_name: str | None = None
    environment_name: str | None = None
    workflow_name: str | None = None
    tool_name: str | None = None
    after_sequence: StrictInt | None = Field(default=None, ge=0)
    limit: StrictInt = Field(default=100, ge=1, le=5000)

    @field_validator(
        "session_id",
        "agent_name",
        "environment_name",
        "workflow_name",
        "tool_name",
    )
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, value: EventType | str | None) -> EventType | str | None:
        if value is None:
            return None
        if isinstance(value, EventType):
            return value
        return Event(type=value, session_id="query").type


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

    async def append_event(self, session_id: str, event: Event) -> None:
        """Append one event to a session."""
        await self.append_events(session_id, [event])

    @abstractmethod
    async def append_events(self, session_id: str, events: list[Event]) -> None:
        """Append events to a session in one durable batch."""

    @abstractmethod
    async def load_events(self, session_id: str) -> list[Event]:
        """Load all events for a session."""

    @abstractmethod
    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        """Query stored events with durable sequence cursors."""

    @abstractmethod
    async def list_sessions(self, query: SessionQuery | None = None) -> list[Session]:
        """List sessions for dashboard, replay, and orchestration views."""

    @abstractmethod
    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
    ) -> None:
        """Append provider-neutral transcript messages to a session."""

    @abstractmethod
    async def load_transcript(self, session_id: str) -> list[Message]:
        """Load provider-neutral transcript messages for a session."""

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
        self._event_records: list[EventRecord] = []
        self._event_ids: dict[str, set[str]] = {}
        self._next_event_sequence = 1
        self._transcripts: dict[str, list[Message]] = {}
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
                environment_name=request.environment_name,
                status=SessionStatus.PENDING,
                created_at=now,
                updated_at=now,
                metadata=deepcopy(request.metadata),
            )
            self._sessions[session.id] = session
            self._events[session.id] = []
            self._event_ids[session.id] = set()
            self._transcripts[session.id] = []
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
        await self.append_events(session_id, [event])

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        session_id = require_nonblank(session_id, "session_id")
        if type(events) is not list:
            raise TypeError("Session events must be a list.")
        copied_events: list[Event] = []
        seen_event_ids: set[str] = set()
        for event in events:
            if type(event) is not Event:
                raise TypeError("Session events must be Event instances.")
            copied_event = _validate_event(event)
            if copied_event.session_id != session_id:
                raise ValueError("Event session_id does not match target session.")
            if copied_event.id in seen_event_ids:
                raise ValueError(
                    f"Event already exists for session {session_id}: {copied_event.id}"
                )
            seen_event_ids.add(copied_event.id)
            copied_events.append(copied_event)

        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            existing_ids = self._event_ids[session_id]
            for event in copied_events:
                if event.id in existing_ids:
                    raise ValueError(
                        f"Event already exists for session {session_id}: {event.id}"
                    )

            for event in copied_events:
                stored_event = event.model_copy(deep=True)
                self._events[session_id].append(stored_event)
                self._event_records.append(
                    EventRecord(
                        sequence=self._next_event_sequence,
                        event=stored_event,
                    )
                )
                existing_ids.add(stored_event.id)
                self._next_event_sequence += 1

    async def load_events(self, session_id: str) -> list[Event]:
        session_id = require_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            return [
                event.model_copy(deep=True)
                for event in self._events.get(session_id, [])
            ]

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        query = copy_event_query(query)
        event_type = str(query.event_type) if query.event_type is not None else None
        async with self._lock:
            records = [
                record
                for record in self._event_records
                if _event_record_matches(record, query, event_type)
            ]
            return [
                EventRecord(
                    sequence=record.sequence,
                    event=record.event,
                )
                for record in records[: query.limit]
            ]

    async def list_sessions(self, query: SessionQuery | None = None) -> list[Session]:
        query = copy_session_query(query)
        async with self._lock:
            sessions = [
                session
                for session in self._sessions.values()
                if _session_matches(session, query)
            ]
            sessions = _sort_sessions(sessions, query.order_by)
            page = sessions[query.offset : query.offset + query.limit]
            return [session.model_copy(deep=True) for session in page]

    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
    ) -> None:
        session_id = require_nonblank(session_id, "session_id")
        copied_messages = copy_transcript_messages(messages)
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            if not copied_messages:
                return
            self._transcripts[session_id].extend(copied_messages)

    async def load_transcript(self, session_id: str) -> list[Message]:
        session_id = require_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            return [
                copy_message(message)
                for message in self._transcripts.get(session_id, [])
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


def copy_transcript_messages(messages: list[Message]) -> list[Message]:
    if type(messages) is not list:
        raise TypeError("Transcript messages must be a list.")
    return [copy_message(message) for message in messages]


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
        task_id=request.task_id,
        environment_name=request.environment_name,
        metadata=copy_json_value(request.metadata, "metadata"),
        max_steps=request.max_steps,
    )


def copy_session_query(query: SessionQuery | None) -> SessionQuery:
    if query is None:
        return SessionQuery()
    if type(query) is not SessionQuery:
        raise TypeError("Session queries must be SessionQuery instances.")
    return SessionQuery(
        status=query.status,
        agent_name=query.agent_name,
        environment_name=query.environment_name,
        limit=query.limit,
        offset=query.offset,
        order_by=query.order_by,
    )


def copy_event_query(query: EventQuery | None) -> EventQuery:
    if query is None:
        return EventQuery()
    if type(query) is not EventQuery:
        raise TypeError("Event queries must be EventQuery instances.")
    return EventQuery(
        session_id=query.session_id,
        event_type=query.event_type,
        agent_name=query.agent_name,
        environment_name=query.environment_name,
        workflow_name=query.workflow_name,
        tool_name=query.tool_name,
        after_sequence=query.after_sequence,
        limit=query.limit,
    )


def _session_matches(session: Session, query: SessionQuery) -> bool:
    if query.status is not None and session.status != query.status:
        return False
    if query.agent_name is not None and session.agent_name != query.agent_name:
        return False
    if (
        query.environment_name is not None
        and session.environment_name != query.environment_name
    ):
        return False
    return True


def _sort_sessions(sessions: list[Session], order_by: SessionOrder) -> list[Session]:
    if order_by == SessionOrder.CREATED_AT_ASC:
        return sorted(sessions, key=lambda session: (session.created_at, session.id))
    if order_by == SessionOrder.CREATED_AT_DESC:
        return sorted(
            sorted(sessions, key=lambda session: session.id),
            key=lambda session: session.created_at,
            reverse=True,
        )
    if order_by == SessionOrder.UPDATED_AT_ASC:
        return sorted(sessions, key=lambda session: (session.updated_at, session.id))
    return sorted(
        sorted(sessions, key=lambda session: session.id),
        key=lambda session: session.updated_at,
        reverse=True,
    )


def _event_record_matches(
    record: EventRecord,
    query: EventQuery,
    event_type: str | None,
) -> bool:
    event = record.event
    if query.after_sequence is not None and record.sequence <= query.after_sequence:
        return False
    if query.session_id is not None and event.session_id != query.session_id:
        return False
    if event_type is not None and str(event.type) != event_type:
        return False
    if query.agent_name is not None and event.agent_name != query.agent_name:
        return False
    if (
        query.environment_name is not None
        and event.environment_name != query.environment_name
    ):
        return False
    if query.workflow_name is not None and event.workflow_name != query.workflow_name:
        return False
    if query.tool_name is not None and event.tool_name != query.tool_name:
        return False
    return True
