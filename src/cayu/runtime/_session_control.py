from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from cayu._validation import require_clean_nonblank
from cayu.core.events import Event, EventType, copy_event
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime.sessions import EventOrder, EventQuery, SessionStatus, SessionStore

INTERRUPT_REQUESTED_SESSION_STATUSES = {
    SessionStatus.INTERRUPTING,
    SessionStatus.INTERRUPTED,
}
INTERRUPTED_EVENT_WAIT_ATTEMPTS = 10
INTERRUPTED_EVENT_WAIT_INTERVAL_S = 0.01
ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS = 600
ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S = 0.01
STREAM_INTERRUPT_POLL_INTERVAL_S = 0.05

UsageTrackerT = TypeVar("UsageTrackerT")


class SessionInterruptedByRequest(Exception):
    """Cooperative control-flow signal for a durable session interruption."""

    def __init__(self, session_id: str) -> None:
        self.session_id = require_clean_nonblank(session_id, "session_id")
        super().__init__(f"Session interrupted: {self.session_id}")


@dataclass
class ActiveSessionRun(Generic[UsageTrackerT]):
    """The process-local owner and turn state for one active session run."""

    runtime_task: asyncio.Task[Any]
    task_id: str | None
    task_started: bool
    task_finished: bool
    turn_registered_agent: runtime_records.RegisteredAgentState | None = None
    turn_environment_name: str | None = None
    turn_started_at: float | None = None
    turn_usage_tracker: UsageTrackerT | None = None
    turn_completed_event: Event | None = None
    turn_completed_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    out_of_band_events: asyncio.Queue[Event] = field(default_factory=asyncio.Queue)


def clear_current_task_cancellation() -> None:
    """Consume cancellation after durable interruption ownership is proven."""

    current_task = asyncio.current_task()
    if current_task is None:
        return
    while current_task.cancelling():
        current_task.uncancel()


def interruption_request_id_from_payload(payload: dict[str, Any]) -> str | None:
    request_id = payload.get("interruption_request_id")
    if request_id is None:
        return None
    if type(request_id) is not str or not request_id.strip():
        raise ValueError("Interruption request ID must be a non-blank string.")
    return request_id


async def _close_async_iterator(iterator: AsyncIterator[Any]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


class StreamInterruptPoll:
    """Bound durable status reads while a provider response streams deltas."""

    def __init__(
        self,
        control: SessionControl[Any],
        *,
        session_id: str,
    ) -> None:
        self._control = control
        self._session_id = session_id
        self._last_poll = time.monotonic()

    async def raise_if_interrupted(self) -> None:
        now = time.monotonic()
        if (
            not self._control.interrupt_signalled(self._session_id)
            and now - self._last_poll < STREAM_INTERRUPT_POLL_INTERVAL_S
        ):
            return
        self._last_poll = now
        await self._control.raise_if_interrupted(self._session_id)


class SessionControl(Generic[UsageTrackerT]):
    """Own process-local session runs and durable interruption observation.

    Durable lifecycle state remains in ``SessionStore``. This component owns
    only the process-local coordination needed to cancel a live owner, bound
    provider-stream polling, route out-of-band events, and observe the durable
    terminal interruption event.
    """

    def __init__(self, *, session_store: SessionStore) -> None:
        self._session_store = session_store
        self._active_runs: dict[str, dict[asyncio.Task[Any], ActiveSessionRun[UsageTrackerT]]] = {}
        self._active_control_tasks: dict[str, set[asyncio.Task[Any]]] = {}
        self._sessions_emitting_interrupted: set[str] = set()
        self._sessions_requesting_interruption: set[str] = set()
        self._interrupt_signals: dict[str, asyncio.Event] = {}

    def stream_interrupt_poll(self, session_id: str) -> StreamInterruptPoll:
        return StreamInterruptPoll(self, session_id=session_id)

    async def raise_if_interrupted(self, session_id: str) -> None:
        session = await self._session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        if session.status in INTERRUPT_REQUESTED_SESSION_STATUSES:
            raise SessionInterruptedByRequest(session_id)

    async def interrupt_requested(self, session_id: str) -> bool:
        session = await self._session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return session.status in INTERRUPT_REQUESTED_SESSION_STATUSES

    async def is_interrupting(self, session_id: str) -> bool:
        session = await self._session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return session.status == SessionStatus.INTERRUPTING

    def signal_interrupt(self, session_id: str) -> None:
        """Wake throttled polling after the durable interrupt is persisted."""

        self._interrupt_signals.setdefault(session_id, asyncio.Event()).set()

    def interrupt_signalled(self, session_id: str) -> bool:
        signal = self._interrupt_signals.get(session_id)
        return signal is not None and signal.is_set()

    def discard_interrupt_signal(self, session_id: str) -> None:
        self._interrupt_signals.pop(session_id, None)

    async def latest_interrupted_event(
        self,
        session_id: str,
        *,
        interruption_request_id: str | None = None,
    ) -> Event | None:
        records = await self._session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SESSION_INTERRUPTED,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if records:
            event = records[0].event
            if (
                interruption_request_id is None
                or interruption_request_id_from_payload(event.payload) == interruption_request_id
            ):
                return event.model_copy(deep=True)
        if await self._session_store.load(session_id) is None:
            raise KeyError(f"Session not found: {session_id}")
        return None

    async def wait_for_interrupted_event(
        self,
        session_id: str,
        *,
        interruption_request_id: str | None = None,
    ) -> Event | None:
        for attempt in range(INTERRUPTED_EVENT_WAIT_ATTEMPTS):
            existing_event = await self.latest_interrupted_event(
                session_id,
                interruption_request_id=interruption_request_id,
            )
            if existing_event is not None:
                return existing_event

            session = await self._session_store.load(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.status != SessionStatus.INTERRUPTED:
                return None
            if attempt < INTERRUPTED_EVENT_WAIT_ATTEMPTS - 1:
                await asyncio.sleep(INTERRUPTED_EVENT_WAIT_INTERVAL_S)
        return None

    async def wait_for_active_interrupted_event(
        self,
        session_id: str,
        *,
        interruption_request_id: str | None = None,
    ) -> Event | None:
        for attempt in range(ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS):
            existing_event = await self.latest_interrupted_event(
                session_id,
                interruption_request_id=interruption_request_id,
            )
            if existing_event is not None:
                return existing_event
            if (
                not self.has_active_tasks(session_id)
                and not self.is_emitting_interrupted(session_id)
                and not self.is_interruption_request_active(session_id)
            ):
                return None
            if attempt < ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS - 1:
                await asyncio.sleep(ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S)
        return None

    def register_active_task(
        self,
        session_id: str,
        task: asyncio.Task[Any],
        *,
        task_id: str | None,
        task_started: bool,
        task_finished: bool,
        turn_registered_agent: runtime_records.RegisteredAgentState | None = None,
        turn_environment_name: str | None = None,
        turn_started_at: float | None = None,
        turn_usage_tracker: UsageTrackerT | None = None,
    ) -> ActiveSessionRun[UsageTrackerT]:
        session_id = require_clean_nonblank(session_id, "session_id")
        active_run = ActiveSessionRun(
            runtime_task=task,
            task_id=task_id,
            task_started=task_started,
            task_finished=task_finished,
            turn_registered_agent=turn_registered_agent,
            turn_environment_name=turn_environment_name,
            turn_started_at=turn_started_at,
            turn_usage_tracker=turn_usage_tracker,
        )
        self._active_runs.setdefault(session_id, {})[task] = active_run
        return active_run

    def unregister_active_task(self, session_id: str, task: asyncio.Task[Any]) -> None:
        active_runs = self._active_runs.get(session_id)
        if active_runs is None:
            return
        active_runs.pop(task, None)
        if not active_runs:
            self._active_runs.pop(session_id, None)

    def active_runs(self, session_id: str) -> tuple[ActiveSessionRun[UsageTrackerT], ...]:
        return tuple(self._active_runs.get(session_id, {}).values())

    def register_active_control_task(self, session_id: str, task: asyncio.Task[Any]) -> None:
        """Register cancellable ownership that carries no run or event-delivery state."""
        session_id = require_clean_nonblank(session_id, "session_id")
        self._active_control_tasks.setdefault(session_id, set()).add(task)

    def unregister_active_control_task(self, session_id: str, task: asyncio.Task[Any]) -> None:
        control_tasks = self._active_control_tasks.get(session_id)
        if control_tasks is None:
            return
        control_tasks.discard(task)
        if not control_tasks:
            self._active_control_tasks.pop(session_id, None)

    def has_active_tasks(self, session_id: str) -> bool:
        active_run_exists = any(
            not active_run.runtime_task.done() for active_run in self.active_runs(session_id)
        )
        return active_run_exists or any(
            not task.done() for task in self._active_control_tasks.get(session_id, ())
        )

    def cancel_active_runs(self, session_id: str) -> bool:
        current_task = asyncio.current_task()
        signalled = False
        run_tasks = {
            active_run.runtime_task
            for active_run in self.active_runs(session_id)
            if active_run.runtime_task is not current_task and not active_run.runtime_task.done()
        }
        # Control tasks supervise phases that have no active run worker. Once a
        # run worker exists, cancel it exactly once and leave the supervisor
        # alive to coordinate terminal persistence and owned cleanup.
        tasks = run_tasks or {
            task
            for task in self._active_control_tasks.get(session_id, ())
            if task is not current_task and not task.done()
        }
        for task in tasks:
            task.cancel()
            signalled = True
        return signalled

    def begin_emitting_interrupted(self, session_id: str) -> None:
        self._sessions_emitting_interrupted.add(session_id)

    def end_emitting_interrupted(self, session_id: str) -> None:
        self._sessions_emitting_interrupted.discard(session_id)

    def is_emitting_interrupted(self, session_id: str) -> bool:
        return session_id in self._sessions_emitting_interrupted

    def begin_interruption_request(self, session_id: str) -> None:
        self._sessions_requesting_interruption.add(session_id)

    def end_interruption_request(self, session_id: str) -> None:
        self._sessions_requesting_interruption.discard(session_id)

    def is_interruption_request_active(self, session_id: str) -> bool:
        return session_id in self._sessions_requesting_interruption

    async def stream_with_out_of_band_events(
        self,
        session_id: str,
        stream: AsyncIterator[Event],
    ) -> AsyncGenerator[Event, None]:
        try:
            async for event in stream:
                yield event
                async for queued_event in self.drain_out_of_band_events(session_id):
                    yield queued_event
            async for queued_event in self.drain_out_of_band_events(session_id):
                yield queued_event
        except GeneratorExit:
            await _close_async_iterator(stream)
            raise

    async def drain_out_of_band_events(self, session_id: str) -> AsyncIterator[Event]:
        for active_run in self.active_runs(session_id):
            while not active_run.out_of_band_events.empty():
                yield active_run.out_of_band_events.get_nowait()

    def queue_out_of_band_event(self, event: Event) -> None:
        for active_run in self.active_runs(event.session_id):
            if active_run.runtime_task.done():
                continue
            active_run.out_of_band_events.put_nowait(copy_event(event))

    def active_turn_completed_event(self, session_id: str) -> Event | None:
        for active_run in self.active_runs(session_id):
            if active_run.turn_completed_event is not None:
                return active_run.turn_completed_event
        return None
