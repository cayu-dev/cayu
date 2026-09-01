from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.core.events import Event, EventType, copy_event
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._terminal_evidence import interruption_request_id_from_payload
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


@dataclass
class TerminalFinalizationClaimHandoff:
    """Process-local authority joining one interrupt owner to its live run."""

    session_instance_id: str
    run_epoch: int
    interruption_request_id: str
    expected_interrupt_payload: dict[str, Any]
    claim_id: str
    eligible_tasks: frozenset[asyncio.Task[Any]]
    heartbeat_stop: asyncio.Event
    heartbeat_task: asyncio.Task[None]
    claimed_by: asyncio.Task[Any] | None = None
    heartbeat_failure: BaseException | None = None


def clear_current_task_cancellation() -> None:
    """Consume cancellation after durable interruption ownership is proven."""

    current_task = asyncio.current_task()
    if current_task is None:
        return
    while current_task.cancelling():
        current_task.uncancel()


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
        self._terminal_finalization_claim_handoffs: dict[str, TerminalFinalizationClaimHandoff] = {}

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

    def _interrupt_targets(self, session_id: str) -> frozenset[asyncio.Task[Any]]:
        current_task = asyncio.current_task()
        run_tasks = {
            active_run.runtime_task
            for active_run in self.active_runs(session_id)
            if active_run.runtime_task is not current_task and not active_run.runtime_task.done()
        }
        if run_tasks:
            return frozenset(run_tasks)
        return frozenset(
            task
            for task in self._active_control_tasks.get(session_id, ())
            if task is not current_task and not task.done()
        )

    def register_terminal_finalization_claim_handoff(
        self,
        session_id: str,
        *,
        session_instance_id: str,
        run_epoch: int,
        interruption_request_id: str,
        expected_interrupt_payload: dict[str, Any],
        claim_id: str,
        heartbeat_stop: asyncio.Event,
        heartbeat_task: asyncio.Task[None],
    ) -> bool:
        """Bind an exact durable claim to the tasks about to receive interruption."""

        session_id = require_clean_nonblank(session_id, "session_id")
        targets = self._interrupt_targets(session_id)
        if not targets:
            return False
        existing = self._terminal_finalization_claim_handoffs.get(session_id)
        if existing is not None and existing.claim_id != claim_id:
            raise RuntimeError("A different terminal finalization handoff is already active.")
        handoff = TerminalFinalizationClaimHandoff(
            session_instance_id=require_clean_nonblank(
                session_instance_id,
                "session_instance_id",
            ),
            run_epoch=run_epoch,
            interruption_request_id=require_clean_nonblank(
                interruption_request_id,
                "interruption_request_id",
            ),
            expected_interrupt_payload=copy_json_value(
                expected_interrupt_payload,
                "expected_interrupt_payload",
            ),
            claim_id=require_clean_nonblank(claim_id, "claim_id"),
            eligible_tasks=targets,
            heartbeat_stop=heartbeat_stop,
            heartbeat_task=heartbeat_task,
        )
        self._terminal_finalization_claim_handoffs[session_id] = handoff

        def observe_heartbeat(completed: asyncio.Task[None]) -> None:
            try:
                completed.result()
            except asyncio.CancelledError:
                return
            except BaseException as failure:
                handoff.heartbeat_failure = failure

        heartbeat_task.add_done_callback(observe_heartbeat)

        def stop_unaccepted_handoff(_completed: asyncio.Task[Any]) -> None:
            current = self._terminal_finalization_claim_handoffs.get(session_id)
            if (
                current is handoff
                and current.claimed_by is None
                and all(target.done() for target in current.eligible_tasks)
            ):
                # The durable claim remains available for expiry-based recovery,
                # but a process-local handoff with no receiving task must never
                # renew it forever.
                self._terminal_finalization_claim_handoffs.pop(session_id, None)
                current.heartbeat_stop.set()

        for target in targets:
            target.add_done_callback(stop_unaccepted_handoff)
        return True

    def take_terminal_finalization_claim_handoff(
        self,
        session_id: str,
        *,
        task: asyncio.Task[Any],
        session_instance_id: str,
        run_epoch: int,
        transferred_from: asyncio.Task[Any] | None = None,
    ) -> TerminalFinalizationClaimHandoff | None:
        """Return only the exact claim handed to this interrupted live task."""

        handoff = self._terminal_finalization_claim_handoffs.get(session_id)
        if handoff is None:
            return None
        if (
            (task not in handoff.eligible_tasks and transferred_from not in handoff.eligible_tasks)
            or (transferred_from is not None and transferred_from.done())
            or handoff.session_instance_id != session_instance_id
            or handoff.run_epoch != run_epoch
            or handoff.claimed_by not in {None, task}
        ):
            return None
        # Even a failed keeper must transfer its cleanup owner. The receiving
        # task observes the completed heartbeat before its first store access,
        # then releases only this exact claim in its finalizer.
        handoff.claimed_by = task
        return handoff

    def reclaim_unaccepted_terminal_finalization_claim_handoff(
        self,
        session_id: str,
        *,
        claim_id: str,
    ) -> asyncio.Task[None] | None:
        """Stop one handoff after every eligible receiver has exited."""

        handoff = self._terminal_finalization_claim_handoffs.get(session_id)
        if handoff is None or handoff.claim_id != claim_id or handoff.claimed_by is not None:
            return None
        self._terminal_finalization_claim_handoffs.pop(session_id, None)
        handoff.heartbeat_stop.set()
        return handoff.heartbeat_task

    def end_terminal_finalization_claim_handoff(
        self,
        session_id: str,
        *,
        claim_id: str,
        task: asyncio.Task[Any] | None = None,
    ) -> asyncio.Task[None] | None:
        """Stop and detach one exact process-local claim heartbeat owner."""

        handoff = self._terminal_finalization_claim_handoffs.get(session_id)
        if (
            handoff is None
            or handoff.claim_id != claim_id
            or (task is not None and handoff.claimed_by not in {None, task})
        ):
            return None
        self._terminal_finalization_claim_handoffs.pop(session_id, None)
        handoff.heartbeat_stop.set()
        return handoff.heartbeat_task

    def cancel_active_runs(self, session_id: str) -> bool:
        signalled = False
        for task in self._interrupt_targets(session_id):
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
        except asyncio.CancelledError:
            with contextlib.suppress(BaseException):
                await _close_async_iterator(stream)
            raise
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
