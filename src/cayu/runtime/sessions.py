from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.core.events import Event, EventType, copy_event
from cayu.core.messages import Message, MessageRole, copy_message
from cayu.runtime.costs import CostBudget, copy_cost_budget
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.stop_policy import RunLimits, copy_run_limits


class SessionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTING = "interrupting"
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
    limits: RunLimits = Field(default_factory=RunLimits)
    cost_budget: CostBudget | None = None
    retry_policy: RetryPolicy | None = None

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
        return require_clean_nonblank(value, info.field_name)

    @field_validator("session_id", "task_id", "environment_name")
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    messages: list[Message]
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: StrictInt = Field(default=16, ge=1, le=256)
    limits: RunLimits = Field(default_factory=RunLimits)
    cost_budget: CostBudget | None = None
    retry_policy: RetryPolicy | None = None

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        copied_messages = [copy_message(message) for message in value]
        if not copied_messages:
            raise ValueError("ResumeRequest messages cannot be empty.")
        return copied_messages

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("session_id", "model")
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class InterruptSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_optional_reason(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class ForkSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_session_id: str
    session_id: str | None = None
    agent_name: str | None = None
    model: str | None = None
    environment_name: str | None = None
    transcript_cursor: StrictInt | None = Field(default=None, ge=0)
    copy_checkpoint: StrictBool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "source_session_id",
        "session_id",
        "agent_name",
        "model",
        "environment_name",
    )
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")


class SessionIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_name: str
    model: str
    runtime_name: str = "cayu"
    runtime_version: str | None = None

    @field_validator("provider_name", "model", "runtime_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("runtime_version")
    @classmethod
    def validate_optional_runtime_version(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # SessionStore implementations may set this from RunRequest.session_id.
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_name: str
    provider_name: str
    model: str
    parent_session_id: str | None = None
    runtime_name: str = "cayu"
    runtime_version: str | None = None
    environment_name: str | None = None
    status: SessionStatus = SessionStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("id", "agent_name", "provider_name", "model", "runtime_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("parent_session_id", "environment_name", "runtime_version")
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


CheckpointTransform = Callable[
    [Session, dict[str, Any] | None],
    dict[str, Any] | None,
]


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
    parent_session_id: str | None = None
    limit: StrictInt = Field(default=100, ge=1, le=1000)
    offset: StrictInt = Field(default=0, ge=0)
    order_by: SessionOrder = SessionOrder.UPDATED_AT_DESC

    @field_validator("agent_name", "environment_name", "parent_session_id")
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: StrictInt = Field(ge=1)
    event: Event

    @field_validator("event")
    @classmethod
    def copy_event(cls, value: Event) -> Event:
        return copy_event(value)


class EventSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    total_events: StrictInt = Field(ge=0)
    counts_by_type: dict[str, StrictInt] = Field(default_factory=dict)
    latest_event: EventRecord | None = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "session_id")


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
        return require_clean_nonblank(value, info.field_name)

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, value: EventType | str | None) -> EventType | str | None:
        if value is None:
            return None
        if isinstance(value, EventType):
            return value
        return Event(type=value, session_id="query").type


class TranscriptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: StrictInt = Field(ge=0)
    message: Message

    @field_validator("message")
    @classmethod
    def copy_message(cls, value: Message) -> Message:
        return copy_message(value)


class TranscriptPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[TranscriptRecord] = Field(default_factory=list)
    total_records: StrictInt = Field(ge=0)


class TranscriptQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    role: MessageRole | str | None = None
    offset: StrictInt = Field(default=0, ge=0)
    limit: StrictInt = Field(default=100, ge=1, le=5000)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "session_id")

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: MessageRole | str | None) -> MessageRole | None:
        if value is None:
            return None
        return MessageRole(value)


class SessionStore(ABC):
    """Persistent store for sessions and append-only events."""

    @abstractmethod
    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
    ) -> Session:
        """Create a session for a run request."""

    @abstractmethod
    async def create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
    ) -> Session:
        """Create a forked session with copied transcript/checkpoint state."""

    @abstractmethod
    async def load(self, session_id: str) -> Session | None:
        """Load a session by id."""

    @abstractmethod
    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        """Update session status and return the updated session."""

    @abstractmethod
    async def update_model(self, session_id: str, model: str) -> Session:
        """Update the active model for a session and return the updated session."""

    @abstractmethod
    async def transition_status(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        """Atomically transition a session status when its current status is allowed."""

    @abstractmethod
    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform: CheckpointTransform,
    ) -> Session:
        """Atomically transition status and persist transformed checkpoint state."""

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
    async def summarize_events(self, session_id: str) -> EventSummary:
        """Summarize stored events for one session without loading every event."""

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
    async def append_transcript_messages_and_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint: dict[str, Any],
    ) -> None:
        """Append transcript messages and persist a checkpoint atomically."""

    @abstractmethod
    async def load_transcript(self, session_id: str) -> list[Message]:
        """Load provider-neutral transcript messages for a session."""

    @abstractmethod
    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
        """Query provider-neutral transcript messages with stable message indexes."""

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

    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
    ) -> Session:
        if type(request) is not RunRequest:
            raise TypeError("Session creation requires a RunRequest.")
        request = copy_run_request(request)
        identity = copy_session_identity(identity)
        async with self._lock:
            session_id = request.session_id or str(uuid4())
            if session_id in self._sessions:
                raise ValueError(f"Session already exists: {session_id}")

            now = datetime.now(UTC)
            session = Session(
                id=session_id,
                agent_name=request.agent_name,
                provider_name=identity.provider_name,
                model=identity.model,
                runtime_name=identity.runtime_name,
                runtime_version=identity.runtime_version,
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

    async def create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
    ) -> Session:
        source_session_id = require_clean_nonblank(source_session_id, "source_session_id")
        fork = copy_session(fork)
        allowed_statuses = _validate_status_set(source_statuses, "source_statuses")
        if fork.parent_session_id != source_session_id:
            raise ValueError("Fork parent_session_id must match source_session_id.")
        if transcript_cursor is not None and transcript_cursor < 0:
            raise ValueError("transcript_cursor must be greater than or equal to 0.")
        async with self._lock:
            source_session = self._sessions.get(source_session_id)
            if source_session is None:
                raise KeyError(f"Session not found: {source_session_id}")
            if source_session.status not in allowed_statuses:
                raise ValueError(f"Source session status is not forkable: {source_session.status}")
            if fork.status != source_session.status:
                raise ValueError(
                    "Fork status must match source session status: "
                    f"{fork.status} != {source_session.status}"
                )
            if fork.provider_name != source_session.provider_name:
                raise ValueError(
                    "Fork provider_name must match source session provider_name: "
                    f"{fork.provider_name} != {source_session.provider_name}"
                )
            if fork.id in self._sessions:
                raise ValueError(f"Session already exists: {fork.id}")

            source_transcript = self._transcripts.get(source_session_id, [])
            if transcript_cursor is None:
                copied_transcript = [copy_message(message) for message in source_transcript]
            else:
                if transcript_cursor > len(source_transcript):
                    raise ValueError("transcript_cursor is greater than source transcript length.")
                copied_transcript = [
                    copy_message(message) for message in source_transcript[:transcript_cursor]
                ]
            copied_checkpoint = None
            if checkpoint_transform is not None:
                source_checkpoint = self._checkpoints.get(source_session_id)
                copied_checkpoint = checkpoint_transform(
                    source_session.model_copy(deep=True),
                    None if source_checkpoint is None else deepcopy(source_checkpoint),
                )
                if copied_checkpoint is not None:
                    copied_checkpoint = copy_json_value(copied_checkpoint, "checkpoint")

            self._sessions[fork.id] = fork.model_copy(deep=True)
            self._events[fork.id] = []
            self._event_ids[fork.id] = set()
            self._transcripts[fork.id] = copied_transcript
            if copied_checkpoint is not None:
                self._checkpoints[fork.id] = copied_checkpoint
            return fork.model_copy(deep=True)

    async def load(self, session_id: str) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return session.model_copy(deep=True)

    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(status, SessionStatus):
            raise ValueError("Session status must be a SessionStatus.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            updated = session.model_copy(
                update={
                    "status": status,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def update_model(self, session_id: str, model: str) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        model = require_clean_nonblank(model, "model")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            updated = session.model_copy(
                update={
                    "model": model,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def transition_status(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.status not in allowed_statuses:
                raise ValueError(
                    f"Session status transition not allowed: {session.status} -> {to_status}"
                )

            updated = session.model_copy(
                update={
                    "status": to_status,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform: CheckpointTransform,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.status not in allowed_statuses:
                raise ValueError(
                    f"Session status transition not allowed: {session.status} -> {to_status}"
                )

            current_checkpoint = self._checkpoints.get(session_id)
            transformed_checkpoint = checkpoint_transform(
                session.model_copy(deep=True),
                None if current_checkpoint is None else deepcopy(current_checkpoint),
            )
            if transformed_checkpoint is not None:
                transformed_checkpoint = copy_json_value(
                    transformed_checkpoint,
                    "checkpoint",
                )

            updated = session.model_copy(
                update={
                    "status": to_status,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._sessions[session_id] = updated
            if transformed_checkpoint is not None:
                self._checkpoints[session_id] = transformed_checkpoint
            return updated.model_copy(deep=True)

    async def append_event(self, session_id: str, event: Event) -> None:
        await self.append_events(session_id, [event])

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
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
                    raise ValueError(f"Event already exists for session {session_id}: {event.id}")

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
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            return [event.model_copy(deep=True) for event in self._events.get(session_id, [])]

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

    async def summarize_events(self, session_id: str) -> EventSummary:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")

            records = [
                record for record in self._event_records if record.event.session_id == session_id
            ]
            counts_by_type: dict[str, int] = {}
            latest_record: EventRecord | None = None
            for record in records:
                event_type = str(record.event.type)
                counts_by_type[event_type] = counts_by_type.get(event_type, 0) + 1
                if latest_record is None or record.sequence > latest_record.sequence:
                    latest_record = record

            return EventSummary(
                session_id=session_id,
                total_events=len(records),
                counts_by_type=counts_by_type,
                latest_event=(
                    None
                    if latest_record is None
                    else EventRecord(
                        sequence=latest_record.sequence,
                        event=latest_record.event,
                    )
                ),
            )

    async def list_sessions(self, query: SessionQuery | None = None) -> list[Session]:
        query = copy_session_query(query)
        async with self._lock:
            sessions = [
                session for session in self._sessions.values() if _session_matches(session, query)
            ]
            sessions = _sort_sessions(sessions, query.order_by)
            page = sessions[query.offset : query.offset + query.limit]
            return [session.model_copy(deep=True) for session in page]

    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        copied_messages = copy_transcript_messages(messages)
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            if not copied_messages:
                return
            self._transcripts[session_id].extend(copied_messages)

    async def append_transcript_messages_and_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint: dict[str, Any],
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        copied_messages = copy_transcript_messages(messages)
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            if copied_messages:
                self._transcripts[session_id].extend(copied_messages)
            self._checkpoints[session_id] = copied_checkpoint

    async def load_transcript(self, session_id: str) -> list[Message]:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            return [copy_message(message) for message in self._transcripts.get(session_id, [])]

    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
        query = copy_transcript_query(query)
        async with self._lock:
            if query.session_id not in self._sessions:
                raise KeyError(f"Session not found: {query.session_id}")

            indexed_messages = list(enumerate(self._transcripts.get(query.session_id, [])))
            if query.role is not None:
                indexed_messages = [
                    (index, message)
                    for index, message in indexed_messages
                    if message.role == query.role
                ]

            page = indexed_messages[query.offset : query.offset + query.limit]
            return TranscriptPage(
                records=[TranscriptRecord(index=index, message=message) for index, message in page],
                total_records=len(indexed_messages),
            )

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            self._checkpoints[session_id] = copy_json_value(state, "checkpoint")

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
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
        limits=copy_run_limits(request.limits),
        cost_budget=copy_cost_budget(request.cost_budget),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
    )


def copy_resume_request(request: ResumeRequest) -> ResumeRequest:
    if type(request) is not ResumeRequest:
        raise TypeError("Session resume requires a ResumeRequest.")
    messages = getattr(request, "messages", None)
    if type(messages) is not list:
        raise ValueError("ResumeRequest messages must be a list.")
    return ResumeRequest(
        session_id=request.session_id,
        messages=[copy_message(message) for message in messages],
        model=request.model,
        metadata=copy_json_value(request.metadata, "metadata"),
        max_steps=request.max_steps,
        limits=copy_run_limits(request.limits),
        cost_budget=copy_cost_budget(request.cost_budget),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
    )


def copy_interrupt_session_request(request: InterruptSessionRequest) -> InterruptSessionRequest:
    if type(request) is not InterruptSessionRequest:
        raise TypeError("Session interruption requires an InterruptSessionRequest.")
    return InterruptSessionRequest(
        session_id=request.session_id,
        reason=request.reason,
        metadata=copy_json_value(request.metadata, "metadata"),
    )


def copy_fork_session_request(request: ForkSessionRequest) -> ForkSessionRequest:
    if type(request) is not ForkSessionRequest:
        raise TypeError("Session fork requires a ForkSessionRequest.")
    return ForkSessionRequest(
        source_session_id=request.source_session_id,
        session_id=request.session_id,
        agent_name=request.agent_name,
        model=request.model,
        environment_name=request.environment_name,
        transcript_cursor=request.transcript_cursor,
        copy_checkpoint=request.copy_checkpoint,
        metadata=copy_json_value(request.metadata, "metadata"),
    )


def copy_session(session: Session) -> Session:
    if type(session) is not Session:
        raise TypeError("Session copy requires a Session.")
    return Session(
        id=session.id,
        agent_name=session.agent_name,
        provider_name=session.provider_name,
        model=session.model,
        parent_session_id=session.parent_session_id,
        runtime_name=session.runtime_name,
        runtime_version=session.runtime_version,
        environment_name=session.environment_name,
        status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
        metadata=copy_json_value(session.metadata, "metadata"),
    )


def copy_session_identity(identity: SessionIdentity) -> SessionIdentity:
    if type(identity) is not SessionIdentity:
        raise TypeError("Session creation requires a SessionIdentity.")
    return SessionIdentity(
        provider_name=identity.provider_name,
        model=identity.model,
        runtime_name=identity.runtime_name,
        runtime_version=identity.runtime_version,
    )


def _validate_status_set(
    statuses: set[SessionStatus],
    field_name: str,
) -> set[SessionStatus]:
    if type(statuses) is not set:
        raise TypeError(f"{field_name} must be a set of SessionStatus values.")
    if not statuses:
        raise ValueError(f"{field_name} cannot be empty.")
    for status in statuses:
        if not isinstance(status, SessionStatus):
            raise ValueError(f"{field_name} must contain SessionStatus values.")
    return set(statuses)


def copy_session_query(query: SessionQuery | None) -> SessionQuery:
    if query is None:
        return SessionQuery()
    if type(query) is not SessionQuery:
        raise TypeError("Session queries must be SessionQuery instances.")
    return SessionQuery(
        status=query.status,
        agent_name=query.agent_name,
        environment_name=query.environment_name,
        parent_session_id=query.parent_session_id,
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


def copy_transcript_query(query: TranscriptQuery) -> TranscriptQuery:
    if type(query) is not TranscriptQuery:
        raise TypeError("Transcript queries must be TranscriptQuery instances.")
    return TranscriptQuery(
        session_id=query.session_id,
        role=query.role,
        offset=query.offset,
        limit=query.limit,
    )


def _session_matches(session: Session, query: SessionQuery) -> bool:
    if query.status is not None and session.status != query.status:
        return False
    if query.agent_name is not None and session.agent_name != query.agent_name:
        return False
    if query.parent_session_id is not None and session.parent_session_id != query.parent_session_id:
        return False
    return not (
        query.environment_name is not None and session.environment_name != query.environment_name
    )


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
    if query.environment_name is not None and event.environment_name != query.environment_name:
        return False
    if query.workflow_name is not None and event.workflow_name != query.workflow_name:
        return False
    return not (query.tool_name is not None and event.tool_name != query.tool_name)
