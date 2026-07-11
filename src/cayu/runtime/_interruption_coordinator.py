from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any, Protocol

from cayu._validation import copy_json_value
from cayu.core.events import Event, EventType
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime.approvals import ResolutionActor, resolution_actor_payload
from cayu.runtime.sessions import (
    InterruptSessionRequest,
    Session,
    SessionOrder,
    SessionQuery,
    SessionStatus,
    SessionStore,
    _deactivate_session_run_fence,
)

logger = logging.getLogger(__name__)

_INTERRUPTIBLE_SESSION_STATUSES = {SessionStatus.PENDING, SessionStatus.RUNNING}
_ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS = 600
_ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S = 0.01
_BACKGROUND_INTERRUPTION_CONCURRENCY = 32
_BACKGROUND_INTERRUPTION_FAILURE_DETAIL_LIMIT = 100
_BACKGROUND_INTERRUPTION_LEASE_SECONDS = 30.0
_BACKGROUND_INTERRUPTION_HEARTBEAT_SECONDS = 10.0
_BACKGROUND_INTERRUPTION_HEARTBEAT_RETRY_SECONDS = 1.0
_INTERRUPTION_TYPE_OPERATOR_REQUESTED = "operator_requested"

_SUPPRESS_BACKGROUND_INTERRUPTION_CASCADE: ContextVar[bool] = ContextVar(
    "cayu_suppress_background_interruption_cascade", default=False
)
_BACKGROUND_INTERRUPTION_COORDINATOR_STOP: ContextVar[asyncio.Event | None] = ContextVar(
    "cayu_background_interruption_coordinator_stop", default=None
)


def interruption_cascade_suppressed() -> bool:
    return _SUPPRESS_BACKGROUND_INTERRUPTION_CASCADE.get()


def interruption_cascade_lease_seconds() -> float:
    return _BACKGROUND_INTERRUPTION_LEASE_SECONDS


@contextmanager
def suppress_interruption_cascade() -> Iterator[None]:
    token = _SUPPRESS_BACKGROUND_INTERRUPTION_CASCADE.set(True)
    try:
        yield
    finally:
        # Async-generator finalization can run in a different Context. A token
        # cannot be reset there, and that closer's suppression state must not be
        # overwritten with the creator's previous value.
        with contextlib.suppress(ValueError):
            _SUPPRESS_BACKGROUND_INTERRUPTION_CASCADE.reset(token)


def _clear_current_task_cancellation() -> None:
    current_task = asyncio.current_task()
    if current_task is None:
        return
    while current_task.cancelling():
        current_task.uncancel()


def _interruption_request_id_from_payload(payload: dict[str, Any]) -> str | None:
    request_id = payload.get("interruption_request_id")
    if request_id is None:
        return None
    if type(request_id) is not str or not request_id.strip():
        raise ValueError("Interruption request ID must be a non-blank string.")
    return request_id


def _is_background_subagent_session(session: Session) -> bool:
    subagent = session.metadata.get("subagent")
    return isinstance(subagent, dict) and subagent.get("mode") == "background"


def _is_subagent_session(session: Session) -> bool:
    subagent = session.metadata.get("subagent")
    return isinstance(subagent, dict) and subagent.get("mode") in {"foreground", "background"}


def _interruption_cascade_marker_datetime(marker: dict[str, Any], key: str) -> datetime | None:
    value = marker.get(key)
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"Pending interruption cascade {key} must be an ISO datetime.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Pending interruption cascade {key} must be an ISO datetime.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Pending interruption cascade {key} must be timezone-aware.")
    return parsed.astimezone(UTC)


def _copy_interruption_cascade_retry_request(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("Interruption cascade retry request must be an object.")
    copied = copy_json_value(value, "retry_request")
    retry_request_id = copied.get("retry_request_id")
    if type(retry_request_id) is not str or not retry_request_id.strip():
        raise ValueError("Interruption cascade retry request ID must be a non-blank string.")
    reason = copied.get("reason")
    if reason is not None and type(reason) is not str:
        raise ValueError("Interruption cascade retry reason must be a string or null.")
    metadata = copied.get("metadata", {})
    if type(metadata) is not dict:
        raise ValueError("Interruption cascade retry metadata must be an object.")
    requested_by_payload = copied.get("requested_by")
    requested_by = (
        None
        if requested_by_payload is None
        else ResolutionActor.model_validate(requested_by_payload)
    )
    return {
        "retry_request_id": retry_request_id,
        "reason": reason,
        "metadata": copy_json_value(metadata, "metadata"),
        "requested_by": resolution_actor_payload(requested_by),
    }


def _interruption_cascade_retry_event_payload(
    retry_request: dict[str, Any] | None,
) -> dict[str, Any]:
    if retry_request is None:
        return {}
    return {
        "retry_request_id": retry_request["retry_request_id"],
        "retry_reason": retry_request.get("reason"),
        "retry_metadata": retry_request.get("metadata", {}),
        "retry_requested_by": retry_request.get("requested_by"),
    }


@dataclass
class _BackgroundInterruptionCascadeState:
    parent_session_id: str
    attempt_id: str
    generation: int
    claim_id: str
    reason: str | None
    metadata: dict[str, Any]
    requested_by: ResolutionActor | None
    retry_request: dict[str, Any] | None = None
    outstanding: int = 0
    failure_count: int = 0
    failure_details: list[dict[str, Any]] = field(default_factory=list)
    seen_child_ids: set[str] = field(default_factory=set)
    cascade_session_ids: set[str] = field(default_factory=set)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    claim_lost: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(frozen=True)
class _BackgroundInterruptionNode:
    state: _BackgroundInterruptionCascadeState
    session_id: str
    session: Session | None


@dataclass(frozen=True)
class _DeferredBackgroundInterruption:
    interrupt_payload: dict[str, Any]
    retry_at: datetime
    drain_required: bool
    retry_request: dict[str, Any] | None = None


class _LoadPendingSessionInterruptPayload(Protocol):
    def __call__(
        self,
        session_id: str,
        *,
        default: dict[str, Any],
    ) -> Awaitable[dict[str, Any]]: ...


class _LatestSessionInterruptedEvent(Protocol):
    def __call__(
        self,
        session_id: str,
        *,
        interruption_request_id: str | None = None,
    ) -> Awaitable[Event | None]: ...


class _ClaimPendingInterruptionCascade(Protocol):
    def __call__(
        self,
        session_id: str,
        interrupt_payload: dict[str, Any],
        *,
        create_if_missing: bool = True,
        retry_request: dict[str, Any] | None = None,
    ) -> Awaitable[dict[str, Any] | None]: ...


class BackgroundInterruptionCoordinator:
    """Owns durable background-subagent interruption cascade orchestration."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        clock: Callable[[], datetime],
        interrupt_session: Callable[[InterruptSessionRequest], AsyncIterator[Event]],
        load_pending_session_interrupt_payload: _LoadPendingSessionInterruptPayload,
        latest_session_interrupted_event: _LatestSessionInterruptedEvent,
        load_pending_interruption_cascade: Callable[[str], Awaitable[dict[str, Any] | None]],
        claim_pending_interruption_cascade: _ClaimPendingInterruptionCascade,
        mark_pending_interruption_cascade_failed: Callable[[str, str, int, str], Awaitable[bool]],
        complete_pending_interruption_cascade: Callable[
            [str, str, int, str], Awaitable[tuple[bool, bool]]
        ],
        renew_pending_interruption_cascade_claim: Callable[[str, str, int, str], Awaitable[bool]],
        release_pending_interruption_cascade_claim: Callable[[str, str, int, str], Awaitable[None]],
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._clock = clock
        self._interrupt_session = interrupt_session
        self._load_pending_session_interrupt_payload = load_pending_session_interrupt_payload
        self._latest_session_interrupted_event = latest_session_interrupted_event
        self._load_pending_interruption_cascade = load_pending_interruption_cascade
        self._claim_pending_interruption_cascade = claim_pending_interruption_cascade
        self._mark_pending_interruption_cascade_failed = mark_pending_interruption_cascade_failed
        self._complete_pending_interruption_cascade = complete_pending_interruption_cascade
        self._renew_pending_interruption_cascade_claim = renew_pending_interruption_cascade_claim
        self._release_pending_interruption_cascade_claim = (
            release_pending_interruption_cascade_claim
        )
        self._tasks: set[asyncio.Task[None]] = set()
        self._tasks_by_parent: dict[str, asyncio.Task[None]] = {}
        self._stop = asyncio.Event()
        self._deferred: dict[str, _DeferredBackgroundInterruption] = {}
        self._deferred_wakeup = asyncio.Event()
        self._deferred_task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[_BackgroundInterruptionNode] = asyncio.Queue()
        self._workers: set[asyncio.Task[None]] = set()
        self._worker_stop = asyncio.Event()
        self._states: dict[str, _BackgroundInterruptionCascadeState] = {}
        self._draining = False
        self._shutdown_active = False
        self._workers_stopped = asyncio.Event()

    def is_admitted(self, parent_session_id: str) -> bool:
        task = self._tasks_by_parent.get(parent_session_id)
        return (task is not None and not task.done()) or parent_session_id in self._deferred

    def is_pending(self, parent_session_id: str) -> bool:
        task = self._tasks_by_parent.get(parent_session_id)
        if task is not None and not task.done():
            return True
        deferred = self._deferred.get(parent_session_id)
        return deferred is not None and deferred.drain_required

    async def drain(self, *, timeout_s: float = 10.0) -> bool:
        """Wait for accepted background interruption cascades to finish.

        Returns ``False`` when the bounded wait expires. In-memory coordinators
        and workers are then cancelled; their durable parent markers remain for
        the next process to recover.
        """

        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a finite positive number.")
        self._draining = True
        try:
            return await self._drain_background_interruptions_started(float(timeout_s))
        except asyncio.CancelledError:
            _clear_current_task_cancellation()
            await self._cancel_background_interruption_work()
            raise
        except BaseException:
            await self._cancel_background_interruption_work()
            raise
        finally:
            self._deferred.clear()
            self._deferred_task = None
            self._draining = False

    async def _drain_background_interruptions_started(self, timeout_s: float) -> bool:
        self._deferred = {
            parent_session_id: deferred
            for parent_session_id, deferred in self._deferred.items()
            if deferred.drain_required
        }
        self._deferred_wakeup.set()
        if self._deferred and (self._deferred_task is None or self._deferred_task.done()):
            self._deferred_task = asyncio.create_task(
                self._run_deferred_background_interruption_cascades()
            )
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        drained = True
        while self._tasks or self._has_drain_required_background_interruptions():
            remaining = deadline - loop.time()
            if remaining <= 0:
                drained = False
                break
            tasks: tuple[asyncio.Task[None], ...] = tuple(self._tasks)
            deferred_task = self._deferred_task
            if deferred_task is not None and not deferred_task.done():
                tasks = (*tasks, deferred_task)
            if not tasks:
                await asyncio.sleep(0)
                continue
            await asyncio.wait(
                tasks,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if loop.time() >= deadline and (
                self._tasks or self._has_drain_required_background_interruptions()
            ):
                drained = False
                break
        if not drained:
            await self._cancel_background_interruption_work()
        else:
            await self._finish_background_interruption_work()
        return drained

    async def _cancel_background_interruption_work(self) -> None:
        pending_tasks = tuple(self._tasks)
        workers = tuple(self._workers)
        deferred_task = self._deferred_task
        self._deferred_task = None
        self._deferred.clear()
        self._stop.set()
        for state in self._states.values():
            state.claim_lost.set()
        self._worker_stop.set()
        if deferred_task is not None:
            deferred_task.cancel()
        self._shutdown_active = True
        self._workers_stopped.clear()
        try:
            for task in (*pending_tasks, *workers):
                task.cancel()
            self._discard_background_interruption_queue()
            self._workers.clear()
            self._workers_stopped.set()
            # Give cooperative tasks one event-loop turn to run their finally
            # blocks. Anything that suppresses cancellation is detached below;
            # its captured stop event remains set and its durable claim is left
            # to expire rather than extending the configured shutdown grace.
            await asyncio.sleep(0)
        finally:
            self._tasks.clear()
            self._tasks_by_parent.clear()
            self._workers.clear()
            self._states.clear()
            self._stop = asyncio.Event()
            self._worker_stop = asyncio.Event()
            self._shutdown_active = False
            self._workers_stopped.clear()

    async def _finish_background_interruption_work(self) -> None:
        deferred_task = self._deferred_task
        self._deferred_task = None
        if deferred_task is not None and not deferred_task.done():
            deferred_task.cancel()
            await asyncio.gather(deferred_task, return_exceptions=True)
        await self._stop_background_interruption_workers()

    def _has_drain_required_background_interruptions(self) -> bool:
        return any(deferred.drain_required for deferred in self._deferred.values())

    def schedule(
        self,
        *,
        parent_session_id: str,
        interrupt_payload: dict[str, Any],
        create_if_missing: bool,
        retry_request: dict[str, Any] | None = None,
        allow_during_drain: bool = False,
    ) -> asyncio.Task[None] | None:
        retry_request = _copy_interruption_cascade_retry_request(retry_request)
        existing = self._tasks_by_parent.get(parent_session_id)
        if existing is not None and not existing.done():
            return existing
        deferred = self._deferred.get(parent_session_id)
        if deferred is not None:
            if retry_request is not None:
                self._deferred[parent_session_id] = replace(
                    deferred,
                    retry_request=retry_request,
                )
            return None
        if self._draining and not allow_during_drain:
            return None
        if len(self._tasks) >= _BACKGROUND_INTERRUPTION_CONCURRENCY:
            self.defer(
                parent_session_id=parent_session_id,
                interrupt_payload=interrupt_payload,
                retry_at=self._clock(),
                drain_required=True,
                retry_request=retry_request,
            )
            return None
        # The cascade outlives its run/request task. Preserve tracing and other
        # context, but detach the copied parent run fence so a later epoch change
        # cannot reject durable marker updates from this background task.
        task_context = copy_context()
        task_context.run(_deactivate_session_run_fence, parent_session_id)
        task_context.run(
            _BACKGROUND_INTERRUPTION_COORDINATOR_STOP.set,
            self._stop,
        )
        if create_if_missing and retry_request is None:
            cascade = self.run_cascade(
                parent_session_id=parent_session_id,
                interrupt_payload=interrupt_payload,
            )
        elif retry_request is None:
            cascade = self.run_cascade(
                parent_session_id=parent_session_id,
                interrupt_payload=interrupt_payload,
                create_if_missing=False,
            )
        else:
            cascade = self.run_cascade(
                parent_session_id=parent_session_id,
                interrupt_payload=interrupt_payload,
                create_if_missing=create_if_missing,
                retry_request=retry_request,
            )
        task = asyncio.create_task(cascade, context=task_context)
        self._tasks.add(task)
        self._tasks_by_parent[parent_session_id] = task

        def discard(completed: asyncio.Task[None]) -> None:
            self._tasks.discard(completed)
            if self._tasks_by_parent.get(parent_session_id) is completed:
                self._tasks_by_parent.pop(parent_session_id, None)
            self._deferred_wakeup.set()
            if completed.cancelled():
                return
            exception = completed.exception()
            if exception is not None:
                logger.error(
                    "Background interruption cascade for parent %s failed unexpectedly.",
                    parent_session_id,
                    exc_info=(type(exception), exception, exception.__traceback__),
                )

        task.add_done_callback(discard)
        return task

    def defer(
        self,
        *,
        parent_session_id: str,
        interrupt_payload: dict[str, Any],
        retry_at: datetime,
        drain_required: bool,
        retry_request: dict[str, Any] | None,
    ) -> None:
        if self._draining and not drain_required:
            return
        existing = self._deferred.get(parent_session_id)
        if existing is None:
            self._deferred[parent_session_id] = _DeferredBackgroundInterruption(
                interrupt_payload=copy_json_value(
                    interrupt_payload,
                    "interrupt_payload",
                ),
                retry_at=retry_at,
                drain_required=drain_required,
                retry_request=_copy_interruption_cascade_retry_request(retry_request),
            )
        else:
            self._deferred[parent_session_id] = _DeferredBackgroundInterruption(
                interrupt_payload=existing.interrupt_payload,
                retry_at=min(existing.retry_at, retry_at),
                drain_required=existing.drain_required or drain_required,
                retry_request=(
                    _copy_interruption_cascade_retry_request(retry_request)
                    if retry_request is not None
                    else existing.retry_request
                ),
            )
        self._deferred_wakeup.set()
        task = self._deferred_task
        if task is None or task.done():
            self._deferred_task = asyncio.create_task(
                self._run_deferred_background_interruption_cascades()
            )

    async def _run_deferred_background_interruption_cascades(self) -> None:
        try:
            while self._deferred:
                self._deferred_wakeup.clear()
                now = self._clock()
                available = max(
                    0,
                    _BACKGROUND_INTERRUPTION_CONCURRENCY - len(self._tasks),
                )
                due = sorted(
                    (
                        (parent_session_id, deferred)
                        for parent_session_id, deferred in (self._deferred.items())
                        if deferred.retry_at <= now
                    ),
                    key=lambda item: item[1].retry_at,
                )[:available]
                for parent_session_id, deferred in due:
                    self._deferred.pop(parent_session_id, None)
                    self.schedule(
                        parent_session_id=parent_session_id,
                        interrupt_payload=deferred.interrupt_payload,
                        create_if_missing=False,
                        retry_request=deferred.retry_request,
                        allow_during_drain=True,
                    )
                if not self._deferred:
                    return
                if due:
                    continue
                if available == 0:
                    await self._deferred_wakeup.wait()
                    continue
                next_retry_at = min(deferred.retry_at for deferred in self._deferred.values())
                delay = max(0.0, (next_retry_at - self._clock()).total_seconds())
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._deferred_wakeup.wait(),
                        timeout=delay,
                    )
        finally:
            if asyncio.current_task() is self._deferred_task:
                self._deferred_task = None

    def _ensure_background_interruption_workers(self) -> None:
        self._workers = {task for task in self._workers if not task.done()}
        if not self._workers and self._worker_stop.is_set():
            self._worker_stop = asyncio.Event()
        stop = self._worker_stop
        missing = _BACKGROUND_INTERRUPTION_CONCURRENCY - len(self._workers)
        for _ in range(missing):
            worker = asyncio.create_task(self._background_interruption_worker(stop))
            self._workers.add(worker)
            worker.add_done_callback(self._workers.discard)

    async def _stop_background_interruption_workers(self) -> None:
        workers = tuple(self._workers)
        self._worker_stop.set()
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        self._workers.clear()
        self._worker_stop = asyncio.Event()
        self._discard_background_interruption_queue()

    def _discard_background_interruption_queue(self) -> None:
        while True:
            try:
                node = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            node.state.outstanding -= 1
            if node.state.outstanding == 0:
                node.state.done.set()
            self._queue.task_done()

    async def run_cascade(
        self,
        *,
        parent_session_id: str,
        interrupt_payload: dict[str, Any],
        create_if_missing: bool = True,
        retry_request: dict[str, Any] | None = None,
    ) -> None:
        if _SUPPRESS_BACKGROUND_INTERRUPTION_CASCADE.get():
            return
        coordinator_stop = _BACKGROUND_INTERRUPTION_COORDINATOR_STOP.get()
        if coordinator_stop is None:
            # Direct internal calls do not run in a scheduled coordinator's
            # copied context, so bind them to the application's current
            # generation.
            coordinator_stop = self._stop
        if coordinator_stop.is_set():
            return
        if interrupt_payload.get("interruption_type") != _INTERRUPTION_TYPE_OPERATOR_REQUESTED:
            return

        try:
            reason = interrupt_payload.get("reason")
            if reason is not None and type(reason) is not str:
                raise ValueError("Operator interruption reason must be a string or null.")
            metadata = interrupt_payload.get("metadata", {})
            if type(metadata) is not dict:
                raise ValueError("Operator interruption metadata must be an object.")
            requested_by_payload = interrupt_payload.get("requested_by")
            requested_by = (
                None
                if requested_by_payload is None
                else ResolutionActor.model_validate(requested_by_payload)
            )
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Background interruption cascade for parent %s has an invalid durable payload: %s",
                parent_session_id,
                exc,
            )
            return

        if retry_request is None:
            marker = await self._claim_pending_interruption_cascade(
                parent_session_id,
                interrupt_payload,
                create_if_missing=create_if_missing,
            )
        else:
            marker = await self._claim_pending_interruption_cascade(
                parent_session_id,
                interrupt_payload,
                create_if_missing=create_if_missing,
                retry_request=retry_request,
            )
        # A storage implementation may delay or suppress task cancellation.
        # The captured generation event remains set after hard cleanup even
        # after the application has installed a fresh event for future work.
        if coordinator_stop.is_set():
            return
        if marker is None:
            current_marker = await self._load_pending_interruption_cascade(parent_session_id)
            if coordinator_stop.is_set():
                return
            if current_marker is None:
                return
            claim_expires_at = _interruption_cascade_marker_datetime(
                current_marker,
                "claim_expires_at",
            )
            retry_at = claim_expires_at if claim_expires_at is not None else self._clock()
            self.defer(
                parent_session_id=parent_session_id,
                interrupt_payload=current_marker["interrupt_payload"],
                retry_at=retry_at,
                drain_required=False,
                retry_request=retry_request,
            )
            return
        state = _BackgroundInterruptionCascadeState(
            parent_session_id=parent_session_id,
            attempt_id=marker["attempt_id"],
            generation=marker["generation"],
            claim_id=marker["claim_id"],
            reason=reason,
            metadata=copy_json_value(metadata, "metadata"),
            requested_by=requested_by,
            retry_request=_copy_interruption_cascade_retry_request(marker.get("retry_request")),
            cascade_session_ids={parent_session_id},
        )
        self._states[parent_session_id] = state
        heartbeat = asyncio.create_task(self._heartbeat_background_interruption_claim(state))
        try:
            await self._run_claimed_background_interruption_cascade(state)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            if self._shutdown_active:
                await self._workers_stopped.wait()
            with contextlib.suppress(Exception):
                await self._release_pending_interruption_cascade_claim(
                    state.parent_session_id,
                    state.attempt_id,
                    state.generation,
                    state.claim_id,
                )
            if self._states.get(parent_session_id) is state:
                self._states.pop(parent_session_id, None)

    async def _run_claimed_background_interruption_cascade(
        self,
        state: _BackgroundInterruptionCascadeState,
    ) -> None:
        parent_session_id = state.parent_session_id
        self._ensure_background_interruption_workers()
        self._enqueue_background_interruption_node(
            state,
            session_id=parent_session_id,
            session=None,
        )
        await state.done.wait()
        if state.claim_lost.is_set():
            return

        if state.failure_count:
            try:
                recorded = await self._mark_pending_interruption_cascade_failed(
                    parent_session_id,
                    state.attempt_id,
                    state.generation,
                    state.claim_id,
                )
                if not recorded:
                    return
                parent = await self._session_store.load(parent_session_id)
                if parent is None:
                    return
                await self._event_writer.emit(
                    Event(
                        type=EventType.SESSION_INTERRUPTION_CASCADE_FAILED,
                        session_id=parent.id,
                        agent_name=parent.agent_name,
                        environment_name=parent.environment_name,
                        payload={
                            "interruption_type": _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
                            "attempt_id": state.attempt_id,
                            "generation": state.generation,
                            "failure_count": state.failure_count,
                            "failures": state.failure_details,
                            "failures_truncated": state.failure_count > len(state.failure_details),
                            **_interruption_cascade_retry_event_payload(state.retry_request),
                        },
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Could not persist background interruption cascade failure for parent %s: %s",
                    parent_session_id,
                    exc,
                )
            return

        try:
            parent = await self._session_store.load(parent_session_id)
            if parent is None:
                return
            marker = await self._load_pending_interruption_cascade(parent_session_id)
            if (
                marker is None
                or marker.get("attempt_id") != state.attempt_id
                or marker.get("generation") != state.generation
                or marker.get("claim_id") != state.claim_id
            ):
                return
            failure_recorded = marker.get("failure_recorded", False)
            if failure_recorded:
                await self._event_writer.emit(
                    Event(
                        type=EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED,
                        session_id=parent.id,
                        agent_name=parent.agent_name,
                        environment_name=parent.environment_name,
                        payload={
                            "interruption_type": _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
                            "attempt_id": state.attempt_id,
                            "generation": state.generation,
                            "descendant_count": len(state.cascade_session_ids) - 1,
                            **_interruption_cascade_retry_event_payload(state.retry_request),
                        },
                    )
                )
            try:
                await self._complete_pending_interruption_cascade(
                    parent_session_id,
                    state.attempt_id,
                    state.generation,
                    state.claim_id,
                )
            except Exception as exc:
                await self._record_background_interruption_completion_failure(
                    state,
                    parent,
                    exc,
                )
        except Exception as exc:
            logger.warning(
                "Could not persist background interruption cascade completion for parent %s: %s",
                parent_session_id,
                exc,
            )

    async def _record_background_interruption_completion_failure(
        self,
        state: _BackgroundInterruptionCascadeState,
        parent: Session,
        exc: Exception,
    ) -> None:
        try:
            recorded = await self._mark_pending_interruption_cascade_failed(
                state.parent_session_id,
                state.attempt_id,
                state.generation,
                state.claim_id,
            )
            if not recorded:
                return
            await self._event_writer.emit(
                Event(
                    type=EventType.SESSION_INTERRUPTION_CASCADE_FAILED,
                    session_id=parent.id,
                    agent_name=parent.agent_name,
                    environment_name=parent.environment_name,
                    payload={
                        "interruption_type": _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
                        "attempt_id": state.attempt_id,
                        "generation": state.generation,
                        "failure_count": 1,
                        "failures": [
                            {
                                "scope": "parent",
                                "session_id": parent.id,
                                "reason": "completion_checkpoint_clear_failed",
                                "error_type": type(exc).__name__,
                            }
                        ],
                        "failures_truncated": False,
                        **_interruption_cascade_retry_event_payload(state.retry_request),
                    },
                )
            )
        except Exception as record_exc:
            logger.warning(
                "Could not persist background interruption completion failure for parent %s: %s",
                state.parent_session_id,
                record_exc,
            )

    async def _heartbeat_background_interruption_claim(
        self,
        state: _BackgroundInterruptionCascadeState,
    ) -> None:
        claim_expires_at = self._clock() + timedelta(seconds=_BACKGROUND_INTERRUPTION_LEASE_SECONDS)
        sleep_seconds = _BACKGROUND_INTERRUPTION_HEARTBEAT_SECONDS
        while True:
            await asyncio.sleep(sleep_seconds)
            try:
                renewed = await self._renew_pending_interruption_cascade_claim(
                    state.parent_session_id,
                    state.attempt_id,
                    state.generation,
                    state.claim_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Could not renew background interruption claim for parent %s: %s",
                    state.parent_session_id,
                    exc,
                )
                if self._clock() >= claim_expires_at:
                    state.claim_lost.set()
                    return
                remaining = max(0.0, (claim_expires_at - self._clock()).total_seconds())
                sleep_seconds = min(
                    _BACKGROUND_INTERRUPTION_HEARTBEAT_RETRY_SECONDS,
                    remaining,
                )
                continue
            if not renewed:
                state.claim_lost.set()
                return
            claim_expires_at = self._clock() + timedelta(
                seconds=_BACKGROUND_INTERRUPTION_LEASE_SECONDS
            )
            sleep_seconds = _BACKGROUND_INTERRUPTION_HEARTBEAT_SECONDS

    def _enqueue_background_interruption_node(
        self,
        state: _BackgroundInterruptionCascadeState,
        *,
        session_id: str,
        session: Session | None,
    ) -> None:
        state.outstanding += 1
        self._queue.put_nowait(
            _BackgroundInterruptionNode(
                state=state,
                session_id=session_id,
                session=session,
            )
        )

    async def _background_interruption_worker(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            node = await self._queue.get()
            try:
                await self._process_background_interruption_node(node)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._record_background_interruption_failure(
                    node.state,
                    {
                        "scope": "worker",
                        "session_id": node.session_id,
                        "reason": "unexpected_worker_failure",
                        "error_type": type(exc).__name__,
                    },
                )
                logger.exception(
                    "Unexpected background interruption worker failure for %s.",
                    node.session_id,
                )
            finally:
                node.state.outstanding -= 1
                if node.state.outstanding == 0:
                    node.state.done.set()
                self._queue.task_done()
            if stop.is_set():
                return

    async def _process_background_interruption_node(
        self,
        node: _BackgroundInterruptionNode,
    ) -> None:
        state = node.state
        if state.claim_lost.is_set():
            return
        session = node.session
        if (
            session is not None
            and _is_background_subagent_session(session)
            and session.status
            in {
                *_INTERRUPTIBLE_SESSION_STATUSES,
                SessionStatus.INTERRUPTING,
            }
        ):
            await self._interrupt_background_session(state, session)
        if state.claim_lost.is_set():
            return
        await self._enqueue_background_descendants(state, node.session_id)

    async def _interrupt_background_session(
        self,
        state: _BackgroundInterruptionCascadeState,
        session: Session,
    ) -> None:
        if session.status == SessionStatus.INTERRUPTING:
            reconciled = await self._wait_for_background_session_interruption(session.id)
            if reconciled:
                return
            await self._record_background_interruption_error(
                state,
                session,
                TimeoutError(f"Session interruption is still finalizing: {session.id}"),
            )
            return
        try:
            with suppress_interruption_cascade():
                async for _event in self._interrupt_session(
                    InterruptSessionRequest(
                        session_id=session.id,
                        reason=state.reason or "Parent session interrupted.",
                        metadata={
                            "source": "background_subagent_parent_interrupt",
                            "parent_session_id": state.parent_session_id,
                            "parent_metadata": copy_json_value(state.metadata, "metadata"),
                        },
                        requested_by=state.requested_by,
                    )
                ):
                    pass
        except TimeoutError as exc:
            await self._record_background_interruption_error(state, session, exc)
        except (KeyError, ValueError) as exc:
            await self._record_background_interruption_error(state, session, exc)
        except Exception as exc:
            await self._record_background_interruption_error(state, session, exc)

    async def _wait_for_background_session_interruption(self, session_id: str) -> bool:
        pending_interrupt_payload = await self._load_pending_session_interrupt_payload(
            session_id,
            default={},
        )
        interruption_request_id = _interruption_request_id_from_payload(pending_interrupt_payload)
        for attempt in range(_ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS):
            if (
                await self._latest_session_interrupted_event(
                    session_id,
                    interruption_request_id=interruption_request_id,
                )
                is not None
            ):
                return True
            session = await self._session_store.load(session_id)
            if session is None or session.status in {
                SessionStatus.COMPLETED,
                SessionStatus.FAILED,
            }:
                return True
            if session.status != SessionStatus.INTERRUPTING:
                return False
            if attempt < _ACTIVE_INTERRUPTED_EVENT_WAIT_ATTEMPTS - 1:
                await asyncio.sleep(_ACTIVE_INTERRUPTED_EVENT_WAIT_INTERVAL_S)
        return False

    async def _record_background_interruption_error(
        self,
        state: _BackgroundInterruptionCascadeState,
        session: Session,
        exc: Exception,
    ) -> None:
        try:
            reloaded = await self._session_store.load(session.id)
        except Exception as reload_exc:
            self._record_background_interruption_failure(
                state,
                {
                    "scope": "child",
                    "session_id": session.id,
                    "reason": "status_unknown_after_interruption_error",
                    "error_type": type(exc).__name__,
                    "reload_error_type": type(reload_exc).__name__,
                },
            )
            return
        if reloaded is None or reloaded.status not in {
            *_INTERRUPTIBLE_SESSION_STATUSES,
            SessionStatus.INTERRUPTING,
        }:
            return
        self._record_background_interruption_failure(
            state,
            {
                "scope": "child",
                "session_id": session.id,
                "status": reloaded.status.value,
                "reason": (
                    "interruption_still_finalizing"
                    if reloaded.status == SessionStatus.INTERRUPTING
                    else "interruption_request_failed"
                ),
                "error_type": type(exc).__name__,
            },
        )

    async def _enqueue_background_descendants(
        self,
        state: _BackgroundInterruptionCascadeState,
        parent_session_id: str,
    ) -> None:
        offset = 0
        try:
            while True:
                if state.claim_lost.is_set():
                    return
                page = (
                    await self._session_store.list_sessions(
                        SessionQuery(
                            parent_session_id=parent_session_id,
                            limit=1000,
                            offset=offset,
                            order_by=SessionOrder.CREATED_AT_ASC,
                        )
                    )
                ).sessions
                if not page:
                    break
                for child in page:
                    if state.claim_lost.is_set():
                        return
                    if not _is_subagent_session(child):
                        continue
                    if child.id in state.seen_child_ids:
                        continue
                    state.seen_child_ids.add(child.id)
                    if _is_background_subagent_session(child):
                        state.cascade_session_ids.add(child.id)
                    self._enqueue_background_interruption_node(
                        state,
                        session_id=child.id,
                        session=child,
                    )
                if len(page) < 1000:
                    break
                offset += len(page)
        except Exception as exc:
            self._record_background_interruption_failure(
                state,
                {
                    "scope": "listing",
                    "parent_session_id": parent_session_id,
                    "reason": "child_listing_failed",
                    "error_type": type(exc).__name__,
                },
            )

    @staticmethod
    def _record_background_interruption_failure(
        state: _BackgroundInterruptionCascadeState,
        detail: dict[str, Any],
    ) -> None:
        state.failure_count += 1
        if len(state.failure_details) < _BACKGROUND_INTERRUPTION_FAILURE_DETAIL_LIMIT:
            state.failure_details.append(detail)
