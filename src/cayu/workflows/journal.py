"""Journal extension point for workflow resume and attempt fencing.

``WorkflowJournal`` is the public customization seam. The default
``EventStoreJournal`` stores workflow events under the workflow run id and is the
source of truth for resume, step replay, and takeover detection.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid5

from cayu._validation import require_clean_nonblank
from cayu.core.events import Event, EventType
from cayu.runtime import (
    EventQuery,
    EventRecord,
    RunRequest,
    SessionIdentity,
    SessionStatus,
    SessionStore,
)

# Identity stamped on the synthetic session that holds a workflow run's journal.
# The session is never executed by ``app.run``; it only anchors the append-only
# journal so ``append_events`` / ``load_events`` have a home to write to.
WORKFLOW_JOURNAL_PROVIDER = "cayu.workflow"
WORKFLOW_JOURNAL_MODEL = "cayu.workflow"
# Journaled once per WorkflowContext, lazily before its first step. The newest
# marker fences out older in-flight attempts on the same run id (see
# ``WorkflowContext._check_fence``).
WORKFLOW_ATTEMPT_EVENT_TYPE = "custom.cayu.workflow.attempt"
_EVENT_QUERY_PAGE_LIMIT = 5000  # EventQuery.limit hard cap
_WORKFLOW_STEP_STARTED_EVENT_NAMESPACE = UUID("f6c3b09d-f866-42fd-8508-cf50894c4b97")
EventEmitter = Callable[[list[Event]], Awaitable[list[Event]]]


def validate_workflow_journal_event_type(event_type: EventType | str) -> None:
    event_type_value = str(event_type)
    if event_type_value.startswith(("workflow.", "custom.")):
        return
    raise ValueError("Workflow journal events must use the workflow. or custom. namespace.")


def _event_attempt_id(event: Event) -> str:
    attempt_id = event.payload.get("attempt_id")
    if isinstance(attempt_id, str) and attempt_id:
        return attempt_id
    raise ValueError("Workflow journal events require a non-empty attempt_id payload.")


@dataclass(frozen=True, slots=True)
class WorkflowJournalContext:
    """Factory input for journals bound to one workflow run."""

    session_store: SessionStore
    session_id: str
    workflow_name: str
    emit_events: EventEmitter

    def __post_init__(self) -> None:
        if not isinstance(self.session_store, SessionStore):
            raise TypeError("WorkflowJournalContext.session_store must be a SessionStore.")
        object.__setattr__(
            self, "session_id", require_clean_nonblank(self.session_id, "session_id")
        )
        object.__setattr__(
            self,
            "workflow_name",
            require_clean_nonblank(self.workflow_name, "workflow_name"),
        )
        if not callable(self.emit_events):
            raise TypeError("WorkflowJournalContext.emit_events must be callable.")


@runtime_checkable
class WorkflowJournal(Protocol):
    """Durable record of one workflow run's progress."""

    async def append(self, event: Event) -> None:
        """Durably record one workflow/custom event for this run."""
        ...

    async def append_current_attempt(self, event: Event, *, attempt_id: str) -> bool:
        """Append only when ``attempt_id`` is still the latest attempt."""
        ...

    async def append_step_started(self, event: Event, *, attempt_id: str) -> bool:
        """Reserve one ``workflow.step.started`` event before child execution."""
        ...

    async def completed_step_ids(self, *, attempt_id: str) -> set[str]:
        """Step ids with recorded ``workflow.step.completed`` events."""
        ...

    async def step_replay_ids(
        self,
        *,
        step_id: str,
        attempt_id: str,
    ) -> tuple[str | None, str | None]:
        """Latest ``(completed, started)`` child session ids for a step."""
        ...

    async def latest_attempt_id(self) -> str | None:
        """Newest journaled workflow-attempt id, if any."""
        ...


class EventStoreJournal:
    """Default ``WorkflowJournal`` backed by cayu's event store."""

    def __init__(
        self,
        session_store: SessionStore,
        session_id: str,
        workflow_name: str,
        *,
        event_emitter: EventEmitter | None = None,
    ) -> None:
        if not isinstance(session_store, SessionStore):
            raise TypeError("EventStoreJournal requires a SessionStore.")
        self._store = session_store
        self._session_id = require_clean_nonblank(session_id, "session_id")
        self._workflow_name = require_clean_nonblank(workflow_name, "workflow_name")
        self._event_emitter = event_emitter
        self._lock = asyncio.Lock()
        self._ensured = False
        self._attempt_cursor = 0
        self._latest_attempt_id: str | None = None
        self._latest_attempt_sequence = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    async def append(self, event: Event) -> None:
        if type(event) is not Event:
            raise TypeError("WorkflowJournal events must be Event instances.")
        if event.session_id != self._session_id:
            raise ValueError("Journal event session_id must match the workflow run id.")
        if event.workflow_name != self._workflow_name:
            raise ValueError("Journal event workflow_name must match the workflow name.")
        validate_workflow_journal_event_type(event.type)
        _event_attempt_id(event)
        await self._ensure_session()
        await self._append_events([event])

    async def append_current_attempt(self, event: Event, *, attempt_id: str) -> bool:
        attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
        self._validate_event(event)
        if _event_attempt_id(event) != attempt_id:
            raise ValueError("Journal event attempt_id must match the current attempt.")
        async with self._lock:
            await self._ensure_session_unlocked()
            latest_attempt_id, _sequence = await self._latest_attempt_unlocked()
            if latest_attempt_id != attempt_id:
                return False
            await self._append_events([event])
            return True

    async def append_step_started(self, event: Event, *, attempt_id: str) -> bool:
        attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
        self._validate_step_started_event(event)
        if _event_attempt_id(event) != attempt_id:
            raise ValueError("Journal event attempt_id must match the current attempt.")
        step_id = event.payload["step_id"]
        async with self._lock:
            await self._ensure_session_unlocked()
            latest_attempt_id, _sequence = await self._latest_attempt_unlocked()
            if latest_attempt_id != attempt_id:
                return False
            attempt = 1
            async for existing in self._iter_workflow_events(EventType.WORKFLOW_STEP_STARTED):
                if existing.payload.get("step_id") == step_id:
                    attempt += 1
            reserved = event.model_copy(
                update={
                    "id": _step_started_event_id(
                        session_id=self._session_id,
                        workflow_name=self._workflow_name,
                        step_id=step_id,
                        attempt=attempt,
                    )
                }
            )
            try:
                await self._append_events([reserved])
            except ValueError as exc:
                if "Event already exists" in str(exc):
                    return False
                raise
            return True

    async def completed_step_ids(self, *, attempt_id: str) -> set[str]:
        attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
        latest_attempt_id, _sequence = await self._latest_attempt()
        if latest_attempt_id != attempt_id:
            return set()
        completed: set[str] = set()
        active_attempt_id: str | None = None
        async for record in self._iter_workflow_records(None):
            event = record.event
            if event.type == WORKFLOW_ATTEMPT_EVENT_TYPE:
                active_attempt_id = _event_attempt_id(event)
                continue
            if event.type != EventType.WORKFLOW_STEP_COMPLETED:
                continue
            if _event_attempt_id(event) != active_attempt_id:
                continue
            step_id = event.payload.get("step_id")
            if isinstance(step_id, str) and step_id:
                completed.add(step_id)
        return completed

    async def latest_step_child_session_id(
        self,
        *,
        step_id: str,
        event_type: EventType,
    ) -> str | None:
        step_id = require_clean_nonblank(step_id, "step_id")
        latest: str | None = None
        async for event in self._iter_workflow_events(event_type):
            if event.payload.get("step_id") == step_id:
                child_session_id = event.payload.get("child_session_id")
                if isinstance(child_session_id, str) and child_session_id:
                    latest = child_session_id
        return latest

    async def step_replay_ids(
        self,
        *,
        step_id: str,
        attempt_id: str,
    ) -> tuple[str | None, str | None]:
        step_id = require_clean_nonblank(step_id, "step_id")
        attempt_id = require_clean_nonblank(attempt_id, "attempt_id")
        latest_attempt_id, _sequence = await self._latest_attempt()
        if latest_attempt_id != attempt_id:
            return None, None
        completed: str | None = None
        started: str | None = None
        active_attempt_id: str | None = None
        async for record in self._iter_workflow_records(None):
            event = record.event
            if event.type == WORKFLOW_ATTEMPT_EVENT_TYPE:
                active_attempt_id = _event_attempt_id(event)
                continue
            if event.payload.get("step_id") != step_id:
                continue
            if _event_attempt_id(event) != active_attempt_id:
                continue
            child_session_id = event.payload.get("child_session_id")
            if not (isinstance(child_session_id, str) and child_session_id):
                continue
            if event.type == EventType.WORKFLOW_STEP_COMPLETED:
                completed = child_session_id
            elif event.type == EventType.WORKFLOW_STEP_STARTED:
                started = child_session_id
        return completed, started

    async def latest_attempt_id(self) -> str | None:
        attempt_id, _sequence = await self._latest_attempt()
        return attempt_id

    async def _latest_attempt(self) -> tuple[str | None, int]:
        # Appends are sequenced per session; scan past the cursor so any newer
        # attempt marker from any process wins.
        async with self._lock:
            return await self._latest_attempt_unlocked()

    async def _latest_attempt_unlocked(self) -> tuple[str | None, int]:
        # Appends are sequenced per session; scan past the cursor so any newer
        # attempt marker from any process wins.
        after_sequence = self._attempt_cursor
        page_size = _EVENT_QUERY_PAGE_LIMIT
        while True:
            records = await self._store.query_events(
                EventQuery(
                    session_id=self._session_id,
                    workflow_name=self._workflow_name,
                    event_type=WORKFLOW_ATTEMPT_EVENT_TYPE,
                    after_sequence=after_sequence,
                    limit=page_size,
                )
            )
            if not records:
                break
            for record in records:
                attempt_id = record.event.payload.get("attempt_id")
                if isinstance(attempt_id, str) and attempt_id:
                    self._latest_attempt_id = attempt_id
                    self._latest_attempt_sequence = record.sequence
            after_sequence = records[-1].sequence
            if len(records) < page_size:
                break
        self._attempt_cursor = after_sequence
        return self._latest_attempt_id, self._latest_attempt_sequence

    async def _iter_workflow_events(self, event_type: EventType | None) -> AsyncIterator[Event]:
        async for record in self._iter_workflow_records(event_type):
            yield record.event

    async def _iter_workflow_records(
        self,
        event_type: EventType | None,
    ) -> AsyncIterator[EventRecord]:
        # Page forward with ``after_sequence`` so a run with more than one
        # ``EventQuery`` page (the hard limit is 5000) recovers all matching
        # events instead of silently ignoring records past the first page.
        after_sequence = 0
        page_size = _EVENT_QUERY_PAGE_LIMIT
        while True:
            records = await self._store.query_events(
                EventQuery(
                    session_id=self._session_id,
                    workflow_name=self._workflow_name,
                    event_type=event_type,
                    after_sequence=after_sequence,
                    limit=page_size,
                )
            )
            if not records:
                break
            for record in records:
                yield record
            if len(records) < page_size:
                break
            after_sequence = records[-1].sequence

    async def _append_events(self, events: list[Event]) -> None:
        if self._event_emitter is not None:
            await self._event_emitter(events)
            return
        await self._store.append_events(self._session_id, events)

    async def _ensure_session(self) -> None:
        if self._ensured:
            return
        async with self._lock:
            await self._ensure_session_unlocked()

    async def _ensure_session_unlocked(self) -> None:
        if self._ensured:
            return
        existing = await self._store.load(self._session_id)
        if existing is None:
            try:
                await self._store.create(
                    RunRequest(
                        agent_name=self._workflow_name,
                        session_id=self._session_id,
                        messages=[],
                        metadata={
                            "cayu.workflow": self._workflow_name,
                            "cayu.workflow_journal": True,
                        },
                    ),
                    identity=SessionIdentity(
                        provider_name=WORKFLOW_JOURNAL_PROVIDER,
                        model=WORKFLOW_JOURNAL_MODEL,
                    ),
                )
            except ValueError:
                # A concurrent branch may have created the anchor first. If it
                # still does not exist after reload, surface the create failure.
                existing = await self._store.load(self._session_id)
                if existing is None:
                    raise
            else:
                existing = await self._store.load(self._session_id)
        if existing is None:
            raise KeyError(f"Session not found after create: {self._session_id}")

        # Never adopt a foreign session: appending workflow events into a
        # real agent session's log, or force-completing it below, corrupts it.
        if existing.provider_name != WORKFLOW_JOURNAL_PROVIDER:
            raise ValueError(
                f"Session {self._session_id!r} exists but is not a workflow "
                f"journal anchor (provider {existing.provider_name!r}); "
                "refusing to adopt it. Use a distinct workflow run id."
            )
        existing_workflow = existing.metadata.get("cayu.workflow")
        if existing_workflow != self._workflow_name:
            raise ValueError(
                f"Session {self._session_id!r} exists as a different workflow "
                f"journal ({existing_workflow!r}); refusing to adopt it as "
                f"{self._workflow_name!r}."
            )

        # The anchor is storage for events, not an agent run. Keeping it terminal
        # prevents incomplete-session recovery from treating it as work to resume.
        if existing.status != SessionStatus.COMPLETED:
            await self._store.update_status(self._session_id, SessionStatus.COMPLETED)
        self._ensured = True

    def _validate_step_started_event(self, event: Event) -> None:
        self._validate_event(event)
        if event.type != EventType.WORKFLOW_STEP_STARTED:
            raise ValueError("append_step_started requires a workflow.step.started event.")

    def _validate_event(self, event: Event) -> None:
        if type(event) is not Event:
            raise TypeError("WorkflowJournal events must be Event instances.")
        if event.session_id != self._session_id:
            raise ValueError("Journal event session_id must match the workflow run id.")
        if event.workflow_name != self._workflow_name:
            raise ValueError("Journal event workflow_name must match the workflow name.")
        validate_workflow_journal_event_type(event.type)
        _event_attempt_id(event)
        if event.type == EventType.WORKFLOW_STEP_STARTED:
            step_id = event.payload.get("step_id")
            if not isinstance(step_id, str) or not step_id:
                raise ValueError("workflow.step.started payload requires a non-empty step_id.")


def _step_started_event_id(
    *,
    session_id: str,
    workflow_name: str,
    step_id: str,
    attempt: int,
) -> str:
    return str(
        uuid5(
            _WORKFLOW_STEP_STARTED_EVENT_NAMESPACE,
            f"{session_id}\0{workflow_name}\0{step_id}\0{attempt}",
        )
    )
